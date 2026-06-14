#pragma once

#include <adc/core/coupled_system.hpp>
#include <adc/core/types.hpp>
#include <adc/coupling/aux_fill.hpp>  // detail::derive_aux_bc + detail::fill_bz_box (partages)
#include <adc/coupling/coupled_source.hpp>
#include <adc/coupling/elliptic_rhs.hpp>
#include <adc/numerics/elliptic/elliptic_problem.hpp>
#include <adc/numerics/elliptic/elliptic_solver.hpp>
#include <adc/numerics/elliptic/geometric_mg.hpp>
#include <adc/numerics/time/implicit_stepper.hpp>
#include <adc/numerics/time/scheduler.hpp>
#include <adc/numerics/time/time_steppers.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/spatial_operator.hpp>
#include <adc/parallel/comm.hpp>

#include <algorithm>
#include <functional>
#include <type_traits>
#include <utility>

/// @file
/// @brief Systeme couple multi-especes MONO-niveau : SystemAssembler (assemble) + SystemDriver (avance).
///
/// DEUX responsabilites, DEUX classes (retour tuteur 8.2 B). SystemAssembler ASSEMBLE : RHS de systeme
/// (f = Sum_s q_s n_s), Poisson, aux = (phi, grad phi), et l'evaluateur de residu d'un bloc
/// R = -div F + S ; il ne fait AUCUN pas de temps. SystemDriver AVANCE : porte le schedule
/// (sous-cyclage par espece, multirate adaptatif, implicite/IMEX delegue), POSSEDE un Assembleur et
/// appelle un TimeStepper. SystemCoupler reste un ALIAS de SystemDriver (compat tests / facade
/// adc_cases). Le canal aux est PARTAGE par tous les blocs, alloue a la largeur MAXIMALE demandee
/// (aux_comps) : un bloc lisant B_z (n_aux=4) le voit, un bloc de base (3) ignore la composante extra.
/// Aucune des deux ne remplace PhysicalModel / assemble_rhs / GeometricMG : elles les CONNECTENT.

// Systeme couple mono-niveau : DEUX responsabilites, DEUX classes (retour tuteur sec.8.2 B).
//
//   SystemAssembler : ASSEMBLE. Couple l'hyperbolique et l'elliptique : RHS de systeme
//     (f = Sum_s q_s n_s), Poisson, aux = (phi, grad phi), et l'evaluateur de residu d'un
//     bloc R = -div F + S. Il ne fait AUCUN pas de temps.
//   SystemDriver    : AVANCE. Porte le schedule (sous-cyclage par espece, implicite/IMEX
//     delegue) et appelle un TimeStepper (take_step). "Avancer un coupleur" etait bancal :
//     ici un Assembleur assemble, un Driver avance. Le Driver POSSEDE un Assembleur.
//
// SystemCoupler reste comme ALIAS de SystemDriver (compat : tests, facades adc_cases).
//
// Aucune des deux ne remplace PhysicalModel (formules locales), assemble_rhs (operateur
// spatial), ni GeometricMG (elliptique) : elles les connectent.

namespace adc {

namespace detail {
template <class>
inline constexpr bool always_false_v = false;

template <class Block>
struct ScopedBlockState {
  Block& block;
  MultiFab* old_state;

  ScopedBlockState(Block& b, MultiFab& stage_state)
      : block(b), old_state(b.state) {
    block.state = &stage_state;
  }

  // REGLE DES CINQ (C.21) : scope-guard a effet de bord dans le dtor (restaure block.state). Copie/move
  // PAR DEFAUT -> double restauration ou restauration depuis une copie morte. Jamais copie ni move
  // (toujours une variable locale de portee) : on supprime les quatre operations.
  ScopedBlockState(const ScopedBlockState&) = delete;
  ScopedBlockState& operator=(const ScopedBlockState&) = delete;
  ScopedBlockState(ScopedBlockState&&) = delete;
  ScopedBlockState& operator=(ScopedBlockState&&) = delete;

  ~ScopedBlockState() { block.state = old_state; }
};
}  // namespace detail

// === ASSEMBLEUR : champs (Poisson de systeme + aux) + residu de bloc. Aucun pas. ======
/// ASSEMBLE les champs (Poisson de systeme + aux partage) et l'evaluateur de residu d'un bloc. Aucun
/// pas de temps. @tparam System : CoupledSystem. @tparam RhsAssembler : assembleur du RHS Poisson.
/// @tparam Elliptic : backend elliptique (concept EllipticSolver, defaut GeometricMG).
template <CoupledSystemLike System, class RhsAssembler,
          class Elliptic = GeometricMG>
class SystemAssembler {
  static_assert(EllipticSolver<Elliptic>,
                "le backend elliptique doit modeler EllipticSolver");

 public:
  // bz : champ magnetique hors-plan B_z(x, y) fourni par l'utilisateur (constante ou champ),
  // partage par TOUS les blocs. Le canal aux PARTAGE est alloue a la largeur MAXIMALE demandee
  // par les blocs (aux_comps) : un bloc qui lit B_z (n_aux=4) le voit, un bloc de base (3)
  // ignore la composante. Sans bloc a champ extra, la largeur reste 3 -> allocation et numerique
  // strictement bit-identiques a l'historique.
  SystemAssembler(System system, const Geometry& geom, const BoxArray& ba,
                  const BCRec& bcPhi, RhsAssembler rhs_assembler,
                  std::function<bool(Real, Real)> active = {},
                  std::function<Real(Real, Real)> bz = {})
      : system_(std::move(system)),
        rhs_assembler_(std::move(rhs_assembler)),
        geom_(geom),
        ba_(ba),
        dm_(ba.size(), n_ranks()),
        bcPhi_(bcPhi),
        aux_bc_(detail::derive_aux_bc(bcPhi)),
        mg_(geom, ba, bcPhi, std::move(active)),
        aux_ncomp_(system_aux_comps(system_)),
        aux_(ba, dm_, aux_ncomp_, 1),
        bz_(std::move(bz)) {
    fill_bz();  // peuple B_z (no-op si aucun bloc ne le demande ou si bz vide)
  }

  System& system() { return system_; }
  const System& system() const { return system_; }
  MultiFab& phi() { return mg_.phi(); }
  MultiFab& aux() { return aux_; }
  const MultiFab& aux() const { return aux_; }
  const Geometry& geom() const { return geom_; }
  const BoxArray& ba() const { return ba_; }
  const DistributionMapping& dm() const { return dm_; }

  // Resout le RHS de systeme (Sum_s q_s n_s), Poisson, puis aux = (phi, grad phi).
  /// Resout le RHS de systeme (Sum_s q_s n_s), le Poisson, puis derive aux = (phi, grad phi). aux()
  /// est a jour au retour.
  void solve_fields() {
    rhs_assembler_(system_, mg_.rhs());
    mg_.solve();
    derive_aux();
  }

  // Residu d'un bloc a un etage : R = -div F + S (+ aux re-resolu si recompute_aux). C'est
  // l'evaluateur (la fleche methode-des-lignes) que le Driver passe au TimeStepper.
  /// Residu R = -div F + S d'un bloc a un etage (avec re-resolution des champs si @p recompute_aux).
  /// C'est la fleche methode-des-lignes que le Driver passe au TimeStepper. Remplit les ghosts de
  /// @p state selon block.bc avant l'assemblage.
  template <class Limiter, class NumericalFlux, class Block>
  void block_residual(Block& block, MultiFab& state, MultiFab& R,
                      bool recompute_aux) {
    if (recompute_aux) {
      detail::ScopedBlockState<Block> swap(block, state);
      solve_fields();
    }
    fill_ghosts(state, geom_.domain, block.bc);
    assemble_rhs<Limiter, NumericalFlux>(block.model, state, aux_, geom_, R);
  }

 private:
  void derive_aux() {
    fill_ghosts(mg_.phi(), geom_.domain, bcPhi_);
    const Real cx = Real(1) / (2 * geom_.dx());
    const Real cy = Real(1) / (2 * geom_.dy());
    field_postprocess(mg_.phi(), aux_, cx, cy,
                      FieldPostProcess{FieldPostProcess::GradSign::Plus, true});
    fill_ghosts(aux_, geom_.domain, aux_bc_);
  }

  // Largeur du canal aux PARTAGE : maximum des aux_comps<Model> sur tous les blocs (au moins
  // kAuxBaseComps). Le canal partage doit etre au moins aussi large que le bloc le plus exigeant
  // pour que load_aux<aux_comps<Model>> n'y lise jamais hors borne ; un bloc moins exigeant
  // ignore simplement les composantes extra.
  static int system_aux_comps(const System& sys) {
    int w = kAuxBaseComps;
    sys.for_each_block([&](const auto& b) {
      using Model = std::decay_t<decltype(b.model)>;
      const int c = aux_comps<Model>();
      if (c > w) w = c;
    });
    return w;
  }

  // Peuple la composante aux B_z (indice kAuxBaseComps) du canal partage depuis bz_(x, y), une
  // seule fois (B_z statique). No-op si aucun bloc ne declare B_z (largeur 3) ou si bz_ vide :
  // garde RUNTIME sur aux_ncomp_ (la largeur n'est connue qu'a la construction). Halos ensuite
  // maintenus par derive_aux (aux_bc_) ; field_postprocess n'ecrit que phi/grad (comp 0..2).
  void fill_bz() {
    if (!bz_ || aux_ncomp_ <= kAuxBaseComps) return;
    for (int li = 0; li < aux_.local_size(); ++li)
      detail::fill_bz_box(aux_.fab(li), aux_.box(li), geom_, bz_);  // boite valide
    fill_ghosts(aux_, geom_.domain, aux_bc_);  // halos de B_z avant le 1er solve
  }

  System system_;
  RhsAssembler rhs_assembler_;
  Geometry geom_;
  BoxArray ba_;
  DistributionMapping dm_;
  BCRec bcPhi_, aux_bc_;
  Elliptic mg_;
  int aux_ncomp_;  // largeur du canal aux partage (max des blocs) ; init avant aux_
  MultiFab aux_;
  std::function<Real(Real, Real)> bz_;  // B_z(x, y) externe (vide si non fourni)
};

// === DRIVER : avance le systeme. Possede un Assembleur, lui delegue les champs. =========
/// AVANCE le systeme : porte le schedule (sous-cyclage par espece, multirate adaptatif,
/// implicite/IMEX delegue) et appelle un TimeStepper. POSSEDE un SystemAssembler et lui delegue les
/// champs. Memes parametres de template que SystemAssembler.
template <CoupledSystemLike System, class RhsAssembler,
          class Elliptic = GeometricMG>
class SystemDriver {
 public:
  /// Construit le driver (qui construit l'assembleur sous-jacent). @p active : predicat de paroi
  /// optionnel passe au MG ; @p bz : champ B_z(x, y) optionnel partage par les blocs.
  SystemDriver(System system, const Geometry& geom, const BoxArray& ba,
               const BCRec& bcPhi, RhsAssembler rhs_assembler,
               std::function<bool(Real, Real)> active = {},
               std::function<Real(Real, Real)> bz = {})
      : asm_(std::move(system), geom, ba, bcPhi, std::move(rhs_assembler),
             std::move(active), std::move(bz)) {}

  // Acces delegues a l'assembleur (compat avec l'ancienne API SystemCoupler).
  System& system() { return asm_.system(); }
  const System& system() const { return asm_.system(); }
  MultiFab& phi() { return asm_.phi(); }
  const MultiFab& aux() const { return asm_.aux(); }
  void solve_fields() { asm_.solve_fields(); }
  SystemAssembler<System, RhsAssembler, Elliptic>& assembler() { return asm_; }

  // Avance les blocs selon leur TimePolicy. Explicites SSPRK2/3 (ou TimeStepper utilisateur)
  // executes via take_step ; implicites/IMEX delegues au callback (Newton, collisions, ...).
  // IMEX = vrai forward-backward : transport explicite par le coeur, source au callback.
  /// Avance les blocs d'un macro-pas dt selon leur TimePolicy : explicites via TimeStepper,
  /// implicites/IMEX delegues a @p implicit_advance (block, h, s, n). Cadence stride par bloc.
  template <class ImplicitAdvance>
  void step(Real dt, ImplicitAdvance&& implicit_advance) {
    ImplicitAdvance& advance_implicit = implicit_advance;
    // macro_step_ : un bloc de cadence `stride` n'avance qu'1 macro-pas sur stride
    // (alors d'un pas effectif stride*dt). stride=1 -> chaque pas (historique).
    advance_subcycled(asm_.system(), dt, macro_step_, [&](auto& block, Real h, int s, int n) {
      advance_block_dispatch(block, h, s, n, advance_implicit);
    });
    ++macro_step_;
  }

  // Multirate PLEINEMENT ADAPTATIF (sec.8.2 C) : le pas macro est fixe par l'espece la plus
  // rapide (CFL), et le `stride` de CHAQUE espece est derive AU RUNTIME du ratio des
  // vitesses d'onde, stride_s = max(1, floor(w_max / w_s)). Une espece lente (gaz) avance
  // donc automatiquement moins souvent, par un pas plus grand (stride_s * macro_dt ~ son dt
  // stable). Retourne le pas macro. (vs `Stride` fixe a la compilation + `step_cfl`.)
  /// Multirate PLEINEMENT ADAPTATIF : pas macro fixe par l'espece la plus rapide (CFL @p cfl), stride
  /// de chaque espece derive au RUNTIME du ratio des vitesses d'onde (espece lente avancee moins
  /// souvent, pas plus grand). Retourne le pas macro. @p implicit_advance traite les blocs implicites/IMEX.
  template <class ImplicitAdvance>
  Real step_adaptive(Real cfl, ImplicitAdvance&& implicit_advance) {
    ImplicitAdvance& advance_implicit = implicit_advance;
    asm_.solve_fields();  // aux a jour pour les vitesses d'onde
    const Real h = std::min(asm_.geom().dx(), asm_.geom().dy());
    const Real wmax = system_max_wave_speed();
    const Real macro_dt = cfl * h / std::max(wmax, Real(1e-30));
    asm_.system().for_each_block([&](auto& block) {
      using Block = std::decay_t<decltype(block)>;
      if constexpr (block_time_treatment_v<Block> != TimeTreatment::Prescribed) {
        const Real w_s = max_wave_speed_mf(block.model, block.U(), asm_.aux());
        const int stride = (w_s <= Real(0))
                               ? 1
                               : std::max(1, static_cast<int>(wmax / w_s));
        if (macro_step_ % stride == 0) {
          constexpr int n = block_substeps_v<Block>;
          const Real hh = (macro_dt * static_cast<Real>(stride)) / static_cast<Real>(n);
          for (int s = 0; s < n; ++s)
            advance_block_dispatch(block, hh, s, n, advance_implicit);
        }
      }
    });
    ++macro_step_;
    return macro_dt;
  }
  Real step_adaptive(Real cfl) {
    return step_adaptive(cfl, [](auto&, auto& block, Real, int, int) {
      using Block = std::decay_t<decltype(block)>;
      static_assert(detail::always_false_v<Block>,
                    "SystemDriver::step_adaptive(cfl) ne peut pas avancer un bloc "
                    "implicite/IMEX sans callback");
    });
  }

  // Surcharge pratique pour un systeme entierement explicite.
  void step(Real dt) {
    step(dt, [](auto&, auto& block, Real, int, int) {
      using Block = std::decay_t<decltype(block)>;
      static_assert(detail::always_false_v<Block>,
                    "SystemDriver::step(dt) ne peut pas avancer un bloc "
                    "implicite/IMEX sans callback");
    });
  }

  // Pas macro choisi par CFL multi-especes (sec.8.2 C) : dt = cfl * min(dx,dy) / w_max, ou
  // w_max est la PLUS GRANDE vitesse d'onde sur TOUTES les especes (l'espece la plus rapide
  // contraint le pas). Combine au `Stride` d'une espece lente (gaz), cela donne le multirate
  // pratique : pas macro fixe par les rapides, lente avancee 1 fois sur stride. Retourne le
  // dt utilise. aux est rafraichi avant la mesure (vitesses d'onde dependantes de phi).
  /// Pas macro choisi par CFL multi-especes : dt = cfl * min(dx, dy) / w_max (w_max = plus grande
  /// vitesse d'onde sur TOUTES les especes). Rafraichit aux avant la mesure.
  Real cfl_dt(Real cfl) {
    asm_.solve_fields();
    const Real h = std::min(asm_.geom().dx(), asm_.geom().dy());
    return cfl * h / std::max(system_max_wave_speed(), Real(1e-30));
  }
  template <class ImplicitAdvance>
  Real step_cfl(Real cfl, ImplicitAdvance&& implicit_advance) {
    const Real dt = cfl_dt(cfl);
    step(dt, std::forward<ImplicitAdvance>(implicit_advance));
    return dt;
  }
  Real step_cfl(Real cfl) {
    const Real dt = cfl_dt(cfl);
    step(dt);
    return dt;
  }

  // Source de couplage inter-especes (splitting forward-Euler) : rafraichit phi (aux) puis
  // laisse la source lire tous les blocs + aux. Distinct de model.source (local au bloc).
  /// Applique une source de COUPLAGE inter-especes (splitting forward-Euler) : rafraichit phi (aux)
  /// puis appelle src.apply(system, aux, dt). Distinct de model.source (local au bloc).
  template <class CoupledSource>
  void coupled_source_step(CoupledSource&& src, Real dt) {
    static_assert(CoupledSourceFor<std::decay_t<CoupledSource>, System>,
                  "coupled_source_step attend une CoupledSource : "
                  "apply(system, aux, dt)");
    asm_.solve_fields();
    src.apply(asm_.system(), asm_.aux(), dt);
  }

 private:
  // Plus grande vitesse d'onde sur TOUTES les especes (aux suppose a jour). Fixe le pas CFL.
  Real system_max_wave_speed() {
    Real wmax = 0;
    asm_.system().for_each_block([&](auto& block) {
      wmax = std::max(wmax, max_wave_speed_mf(block.model, block.U(), asm_.aux()));
    });
    return wmax;
  }

  // Dispatch d'un (sous-)pas pour UN bloc, selon son traitement. Partage par step (cadence
  // compile-time) et step_adaptive (cadence CFL runtime) : pas de duplication.
  //   Explicite : avance via TimeStepper. Implicite/IMEX : re-resout les champs, (IMEX)
  //   transport explicite, puis source implicite par le callback.
  template <class Block, class ImplicitAdvance>
  void advance_block_dispatch(Block& block, Real h, int s, int n,
                              ImplicitAdvance& advance_implicit) {
    constexpr TimeTreatment treatment = block_time_treatment_v<Block>;
    if constexpr (treatment == TimeTreatment::Explicit) {
      advance_explicit_block(block, h);
    } else if constexpr (treatment == TimeTreatment::Implicit ||
                         treatment == TimeTreatment::IMEX) {
      asm_.solve_fields();
      if constexpr (treatment == TimeTreatment::IMEX) explicit_transport(block, h);
      advance_implicit(*this, block, h, s, n);
    }
  }

  // Avance explicite d'un bloc : DELEGUE le schema a un objet TimeStepper (SSPRK2/3 du coeur
  // ou integrateur utilisateur), en lui passant l'evaluateur de residu de l'assembleur.
  template <class Block>
  void advance_explicit_block(Block& block, Real dt) {
    using Time = TimePolicyTraits<typename Block::Time>;
    using Method = typename Time::Method;
    using Limiter = typename Block::Spatial::Limiter;
    using NumericalFlux = typename Block::Spatial::NumericalFlux;
    static_assert(Time::treatment == TimeTreatment::Explicit,
                  "advance_explicit_block attend un bloc explicite");

    auto rhs_eval = [&](MultiFab& stage, MultiFab& R) {
      asm_.template block_residual<Limiter, NumericalFlux>(block, stage, R,
                                                           /*recompute_aux=*/true);
    };
    if constexpr (std::is_same_v<Method, SSPRK3>)
      SSPRK3Step{}.take_step(rhs_eval, block.U(), dt);
    else if constexpr (std::is_same_v<Method, SSPRK2>)
      SSPRK2Step{}.take_step(rhs_eval, block.U(), dt);
    else if constexpr (TimeStepper<Method>)
      Method{}.take_step(rhs_eval, block.U(), dt);
    else
      static_assert(detail::always_false_v<Method>,
                    "Method explicite doit etre SSPRK2, SSPRK3, ou un TimeStepper "
                    "(objet a take_step(rhs_eval, U, dt)) fourni par l'utilisateur");
  }

  // Demi-pas EXPLICITE d'un bloc IMEX : transport seul (-div F, source-free), Euler avant.
  // La source raide est traitee separement en implicite par le callback. aux suppose a jour.
  template <class Block>
  void explicit_transport(Block& block, Real dt) {
    using Model = typename Block::Model;
    using Limiter = typename Block::Spatial::Limiter;
    using NumericalFlux = typename Block::Spatial::NumericalFlux;
    const SourceFreeModel<Model> sf{block.model};
    MultiFab R(asm_.ba(), asm_.dm(), Model::n_vars, 0);
    fill_ghosts(block.U(), asm_.geom().domain, block.bc);
    assemble_rhs<Limiter, NumericalFlux>(sf, block.U(), asm_.aux(), asm_.geom(), R);
    saxpy(block.U(), dt, R);
  }

  SystemAssembler<System, RhsAssembler, Elliptic> asm_;
  int macro_step_ = 0;  // compteur de macro-pas (cadence stride par bloc)
};

// Compat / nommage historique : SystemCoupler == le Driver (qui avance). On garde l'alias
// SystemCoupler (tests, facade MultiSpeciesSolver) ET SystemDriver (le nom "qui avance").
template <CoupledSystemLike System, class RhsAssembler, class Elliptic = GeometricMG>
using SystemCoupler = SystemDriver<System, RhsAssembler, Elliptic>;

}  // namespace adc
