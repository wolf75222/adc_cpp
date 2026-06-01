#pragma once

#include <adc/core/coupled_system.hpp>
#include <adc/core/types.hpp>
#include <adc/coupling/amr_coupler_mp.hpp>  // detail::coupler_inject_aux_mb
#include <adc/coupling/coupled_source.hpp>  // CoupledSourceFor
#include <adc/coupling/elliptic_rhs.hpp>
#include <adc/elliptic/elliptic_problem.hpp>  // field_postprocess, FieldPostProcess
#include <adc/elliptic/elliptic_solver.hpp>
#include <adc/elliptic/geometric_mg.hpp>
#include <adc/integrator/amr_reflux_mf.hpp>  // AmrLevelMP, advance_amr, mf_average_down_mb
#include <adc/integrator/implicit_stepper.hpp>  // backward_euler_source
#include <adc/integrator/scheduler.hpp>  // block_substeps_v, block_time_treatment_v
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/parallel/comm.hpp>  // all_reduce_sum

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

// Coupleur de SYSTEME sur AMR (TODO 2.3).
//
// SystemCoupler porte un CoupledSystem mono-niveau. AmrSystemCoupler le porte sur une
// hierarchie AMR : chaque bloc a SA propre hierarchie de niveaux (un MultiFab par
// niveau), toutes les especes PARTAGENT la meme grille AMR, le meme champ aux
// (phi, grad phi) et le meme Poisson grossier. L'orchestration reutilise le moteur
// AMR existant :
//   sync_down (fin -> grossier, par bloc)
//   -> Poisson grossier f = Sum_s q_s n_s (RHS de systeme, lit tous les blocs)
//   -> aux = grad phi (grossier) + injection vers les fins (aux partage)
//   -> chaque bloc avance par advance_amr<Disc_bloc> (Berger-Oliger + reflux +
//      average_down conservatifs), avec SON schema spatial et SES sous-pas ; les blocs
//      implicites / IMEX sont delegues a un callback (meme contrat que SystemCoupler).
//
// Hypothese du squelette : tous les blocs partagent le BoxArray par niveau (grille AMR
// commune a toutes les especes) -> un seul aux, un seul Poisson. Mono-rang / mono-box
// par niveau valide (comme AmrCoupler) ; le multi-box reutilise les primitives _mb.
//
// CADENCE POISSON (TODO 2.2.3) : quand une espece fait plus de sous-pas qu'une autre, a
// quelle frequence re-resoudre phi ? Le choix est explicite (PoissonCadence) plutot que
// cable en dur :
//   OncePerStep (defaut) : phi resolu une fois par macro-pas, GELE pendant l'avance des
//     blocs. Le moins cher ; coherent quand le pas macro est petit devant l'echelle de phi.
//   PerSubstep : phi re-resolu avant chaque sous-pas d'espece (la charge a bouge). Plus
//     fidele pour un transport pilote par le champ (derive E x B), plus cher. Approximation
//     dans tous les cas (les blocs avancent l'un apres l'autre, pas en lock-step multirate
//     vrai ; ca, c'est un scheduler a part). Le SystemCoupler mono-niveau, lui, re-resout
//     deja phi a CHAQUE etage RK (recompute_aux=true) : cadence maximale par construction.

namespace adc {

enum class PoissonCadence { OncePerStep, PerSubstep };

namespace detail {
template <class>
inline constexpr bool amr_always_false_v = false;
}  // namespace detail

template <CoupledSystemLike System, class RhsAssembler,
          class Elliptic = GeometricMG>
class AmrSystemCoupler {
  static_assert(EllipticSolver<Elliptic>,
                "le backend elliptique doit modeler EllipticSolver");

 public:
  // block_levels[b] = hierarchie du bloc b (niveau 0 = grossier sur ba_coarse, niveaux
  // > 0 = patchs fins). Les AmrLevelMP portent U + dx/dy par niveau ; leur pointeur aux
  // est (re)cable ici vers l'aux PARTAGE. Le ctor re-pointe aussi block.state vers le
  // niveau grossier de sa hierarchie, pour que le RHS de systeme (ChargeDensityRhs) lise
  // bien les densites grossieres.
  AmrSystemCoupler(System system, const Geometry& geom, const BoxArray& ba_coarse,
                   const BCRec& bcPhi, RhsAssembler rhs_assembler,
                   std::vector<std::vector<AmrLevelMP>> block_levels,
                   Periodicity base_per = Periodicity{true, true},
                   bool replicated_coarse = true,
                   PoissonCadence cadence = PoissonCadence::OncePerStep,
                   std::function<bool(Real, Real)> active = {})
      : system_(std::move(system)),
        rhs_assembler_(std::move(rhs_assembler)),
        geom_(geom),
        dom_(geom.domain),
        base_per_(base_per),
        bcPhi_(bcPhi),
        aux_bc_(derive_aux_bc(bcPhi)),
        replicated_coarse_(replicated_coarse),
        cadence_(cadence),
        mg_(geom, ba_coarse, bcPhi, std::move(active), replicated_coarse),
        block_levels_(std::move(block_levels)) {
    // Verifications de construction (revue Codex) : sans elles, une hierarchie mal
    // formee provoque un acces hors borne silencieux dans le cablage / l'avance.
    if (block_levels_.size() != System::n_blocks)
      throw std::runtime_error(
          "AmrSystemCoupler : block_levels doit avoir un vecteur de niveaux par bloc "
          "(taille != n_blocks)");
    nlev_ = block_levels_.empty() ? 0
                                  : static_cast<int>(block_levels_[0].size());
    if (nlev_ == 0)
      throw std::runtime_error("AmrSystemCoupler : au moins un niveau (grossier) requis");
    for (std::size_t b = 0; b < block_levels_.size(); ++b) {
      if (static_cast<int>(block_levels_[b].size()) != nlev_)
        throw std::runtime_error(
            "AmrSystemCoupler : tous les blocs doivent avoir le meme nombre de niveaux");
      // grille partagee par niveau (aux unique) : meme nombre de boites que le bloc 0.
      for (int k = 0; k < nlev_; ++k)
        if (block_levels_[b][k].U.box_array().size() !=
            block_levels_[0][k].U.box_array().size())
          throw std::runtime_error(
              "AmrSystemCoupler : grilles par niveau incoherentes entre blocs "
              "(aux partage exige la meme BoxArray par niveau)");
    }
    // aux PARTAGE : un MultiFab (phi, grad phi) par niveau, sur la grille commune.
    // Dimensionne une seule fois -> adresses stables pour les pointeurs aux des blocs.
    aux_.resize(nlev_);
    for (int k = 0; k < nlev_; ++k)
      aux_[k] = MultiFab(block_levels_[0][k].U.box_array(),
                         block_levels_[0][k].U.dmap(), 3, 1);
    for (auto& levels : block_levels_)
      for (int k = 0; k < nlev_; ++k) levels[k].aux = &aux_[k];

    // re-pointe chaque bloc vers SON niveau grossier (block.U() = grossier du bloc).
    std::size_t b = 0;
    system_.for_each_block([&](auto& block) {
      block.state = &block_levels_[b][0].U;
      ++b;
    });
  }

  System& system() { return system_; }
  const System& system() const { return system_; }
  MultiFab& phi() { return mg_.phi(); }
  int nlev() const { return nlev_; }
  const MultiFab& aux(int k) const { return aux_[k]; }
  std::vector<AmrLevelMP>& levels(std::size_t b) { return block_levels_[b]; }
  MultiFab& coarse(std::size_t b) { return block_levels_[b][0].U; }
  const MultiFab& coarse(std::size_t b) const { return block_levels_[b][0].U; }
  // nombre de resolutions Poisson du dernier step() : diagnostic de la cadence.
  int solve_count() const { return solve_count_; }

  // sync_down (par bloc) + Poisson grossier de systeme + aux grossier + injection fine.
  void solve_fields() {
    ++solve_count_;
    for (auto& levels : block_levels_)
      for (int k = nlev_ - 1; k >= 1; --k)
        mf_average_down_mb(levels[k].U, levels[k - 1].U);

    rhs_assembler_(system_, mg_.rhs());  // f = Sum_s q_s n_s sur le grossier
    mg_.solve();

    // aux grossier = (phi, grad phi) via le MEME chemin propre que le SystemCoupler
    // mono-niveau (revue Codex 9.4) : remplir les ghosts de phi selon bcPhi_, puis
    // field_postprocess, puis remplir les ghosts d'aux selon aux_bc_ (derive de bcPhi_).
    // Gere le non-periodique (Foextrap) au lieu d'un fill_boundary periodique en dur.
    fill_ghosts(mg_.phi(), dom_, bcPhi_);
    const Real cx = Real(1) / (2 * geom_.dx()), cy = Real(1) / (2 * geom_.dy());
    field_postprocess(mg_.phi(), aux_[0], cx, cy,
                      FieldPostProcess{FieldPostProcess::GradSign::Plus, true});
    fill_ghosts(aux_[0], dom_, aux_bc_);
    for (int k = 1; k < nlev_; ++k)
      detail::coupler_inject_aux_mb(aux_[k - 1], aux_[k],
                                    /*replicated_parent=*/(k == 1) && replicated_coarse_);
  }

  // Avance le systeme d'un pas. Blocs explicites : advance_amr avec leur Disc et leurs
  // sous-pas d'espece. Blocs implicites / IMEX : delegues au callback (coupler, block,
  // levels, dt), point de branchement Newton / IMEX (defaut AmrImplicitSourceStepper).
  template <class ImplicitAdvance>
  void step(Real dt, ImplicitAdvance&& implicit_advance) {
    solve_count_ = 0;
    solve_fields();
    std::size_t b = 0;
    system_.for_each_block([&](auto& block) {
      using Block = std::decay_t<decltype(block)>;
      using Disc = typename Block::Spatial;
      constexpr TimeTreatment treatment = block_time_treatment_v<Block>;
      constexpr int n = block_substeps_v<Block>;
      auto& levels = block_levels_[b];
      if constexpr (treatment == TimeTreatment::Explicit) {
        const Real h = dt / static_cast<Real>(n);
        for (int s = 0; s < n; ++s) {
          // PerSubstep : re-resout phi avant chaque sous-pas suivant (la charge a
          // bouge) ; le premier reutilise le solve de tete. OncePerStep : phi gele.
          if (cadence_ == PoissonCadence::PerSubstep && s > 0) solve_fields();
          advance_amr<typename Disc::Limiter, typename Disc::NumericalFlux>(
              block.model, levels, dom_, h, base_per_, replicated_coarse_);
        }
      } else if constexpr (treatment == TimeTreatment::Implicit ||
                           treatment == TimeTreatment::IMEX) {
        implicit_advance(*this, block, levels, dt);
      }
      ++b;
    });
  }

  // Surcharge pour un systeme entierement explicite.
  void step(Real dt) {
    step(dt, [](auto&, auto& block, auto&, Real) {
      using Block = std::decay_t<decltype(block)>;
      static_assert(detail::amr_always_false_v<Block>,
                    "AmrSystemCoupler::step(dt) ne peut pas avancer un bloc "
                    "implicite/IMEX sans callback");
    });
  }

  // Source de couplage inter-especes sur AMR (parite avec SystemCoupler, revue Codex
  // 9.5) : splitting forward-Euler applique PAR NIVEAU. On rafraichit phi (aux par
  // niveau) puis, a chaque niveau k, on repointe temporairement chaque bloc vers son
  // niveau k et on laisse la source lire tous les blocs + aux[k]. NoCoupledSource => no-op.
  template <class CoupledSource>
  void coupled_source_step(CoupledSource&& src, Real dt) {
    static_assert(CoupledSourceFor<std::decay_t<CoupledSource>, System>,
                  "coupled_source_step attend une CoupledSource : apply(system, aux, dt)");
    solve_fields();
    for (int k = 0; k < nlev_; ++k) {
      std::size_t b = 0;
      MultiFab* saved[System::n_blocks == 0 ? 1 : System::n_blocks];
      system_.for_each_block([&](auto& block) {
        saved[b] = block.state;
        block.state = &block_levels_[b][k].U;
        ++b;
      });
      src.apply(system_, aux_[k], dt);
      b = 0;
      system_.for_each_block([&](auto& block) { block.state = saved[b++]; });
    }
  }

  // masse de la composante 0 du grossier du bloc b (somme u*dV sur fabs locaux ;
  // grossier replique -> somme locale = totale, sinon all_reduce).
  Real mass(std::size_t b) const {
    const MultiFab& U = block_levels_[b][0].U;
    const Real dV = geom_.dx() * geom_.dy();
    Real M = 0;
    for (int li = 0; li < U.local_size(); ++li) {
      const ConstArray4 u = U.fab(li).const_array();
      M += for_each_cell_reduce_sum(
          U.box(li), [u, dV] ADC_HD(int i, int j) { return u(i, j, 0) * dV; });
    }
    return replicated_coarse_ ? M : all_reduce_sum(M);
  }

 private:
  System system_;
  RhsAssembler rhs_assembler_;
  static BCRec derive_aux_bc(const BCRec& b) {
    auto t = [](BCType x) {
      return x == BCType::Periodic ? BCType::Periodic : BCType::Foextrap;
    };
    BCRec a;
    a.xlo = t(b.xlo); a.xhi = t(b.xhi);
    a.ylo = t(b.ylo); a.yhi = t(b.yhi);
    return a;
  }

  Geometry geom_;
  Box2D dom_;
  Periodicity base_per_;
  BCRec bcPhi_, aux_bc_;
  bool replicated_coarse_;
  PoissonCadence cadence_;
  mutable int solve_count_ = 0;
  Elliptic mg_;
  std::vector<std::vector<AmrLevelMP>> block_levels_;  // [bloc][niveau]
  std::vector<MultiFab> aux_;                          // [niveau], partage
  int nlev_ = 0;
};

// Defaut implicite sur AMR : backward-Euler (Newton) sur la source du modele, applique
// a CHAQUE niveau de la hierarchie du bloc. Pendant AMR d'ImplicitSourceStepper ; meme
// stabilite (inconditionnelle pour une relaxation lineaire). Aucun solveur cote user.
struct AmrImplicitSourceStepper {
  int iters = 2;

  template <class Coupler, class Block, class Levels>
  void operator()(Coupler& coupler, Block& block, Levels& levels, Real dt) const {
    for (int k = 0; k < static_cast<int>(levels.size()); ++k)
      backward_euler_source(block.model, coupler.aux(k), levels[k].U, dt, iters);
  }
};

}  // namespace adc
