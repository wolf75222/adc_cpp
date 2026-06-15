#pragma once

#include <adc/core/coupled_system.hpp>
#include <adc/core/types.hpp>
#include <adc/coupling/amr_coupler_mp.hpp>  // detail::coupler_inject_aux_mb
#include <adc/coupling/aux_fill.hpp>        // detail::derive_aux_bc + detail::fill_bz_box (partages)
#include <adc/coupling/coupled_source.hpp>  // CoupledSourceFor
#include <adc/coupling/elliptic_rhs.hpp>
#include <adc/numerics/elliptic/elliptic_problem.hpp>  // field_postprocess, FieldPostProcess
#include <adc/numerics/elliptic/elliptic_solver.hpp>
#include <adc/numerics/elliptic/geometric_mg.hpp>
#include <adc/numerics/time/amr_reflux_mf.hpp>  // AmrLevelMP, advance_amr, mf_average_down_mb
#include <adc/numerics/time/implicit_stepper.hpp>  // backward_euler_source
#include <adc/numerics/time/scheduler.hpp>  // block_substeps_v, block_time_treatment_v
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

/// @file
/// @brief AmrSystemCoupler : coupleur de SYSTEME multi-especes sur AMR (jalon 2.3).
///
/// Porte un CoupledSystem sur une hierarchie AMR : chaque bloc a SA hierarchie de niveaux, toutes les
/// especes PARTAGENT la meme grille AMR, le meme champ aux (phi, grad phi [, B_z, ...]) et le meme
/// Poisson grossier. Orchestration : sync_down (par bloc) -> Poisson grossier f = Sum_s q_s n_s ->
/// aux grossier + injection vers les fins -> chaque bloc avance par advance_amr<Disc_bloc> (avec son
/// schema et ses sous-pas ; blocs implicites/IMEX delegues a un callback). INVARIANT FORT : tous les
/// blocs vivent sur EXACTEMENT la meme grille par niveau (l'aux est partage) ; same_layout_or_throw
/// le verifie au ctor. PoissonCadence choisit la frequence de re-solve de phi (OncePerStep gele vs
/// PerSubstep). Mono-bloc = chemin bit-identique a l'historique (boucles sur les autres blocs vides).

// Coupleur de SYSTEME sur AMR (jalon 2.3).
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
// CADENCE POISSON (jalon 2.2.3) : quand une espece fait plus de sous-pas qu'une autre, a
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

/// Frequence de re-resolution du Poisson sur AMR : OncePerStep (phi resolu une fois par macro-pas,
/// gele pendant l'avance ; le moins cher) ; PerSubstep (phi re-resolu avant chaque sous-pas d'espece,
/// plus fidele pour un transport pilote par le champ, plus cher).
enum class PoissonCadence { OncePerStep, PerSubstep };

// Layout EXPLICITE d'une hierarchie AMR partagee (point 2 du capstone multi-blocs, premier pas
// MINIMAL). Source unique de verite sur la GRILLE que tous les blocs partagent : par niveau le
// BoxArray (les boites ET leur ordre), le DistributionMapping (rang par boite), dx/dy, et le
// nombre de niveaux (= ba.size()). Aujourd'hui cette information est implicite, eparpillee dans
// chaque AmrLevelMP (U.box_array() / U.dmap() / dx,dy). Ce type ne fait que l'EXTRAIRE pour le
// garde-fou same_layout_or_throw : il NE remplace PAS EquationBlock / AmrLevelMP et n'introduit
// AUCUNE abstraction de bloc (l'AmrBlock large du design est un pas ULTERIEUR, et seulement si
// necessaire). On lit le layout d'une pile de niveaux via from_levels.
/// Source unique de verite sur la GRILLE partagee par tous les blocs : par niveau le BoxArray (boites
/// ET ordre), le DistributionMapping (rang par boite) et dx/dy. EXTRAIT seulement (ne remplace ni
/// EquationBlock ni AmrLevelMP) ; sert au garde-fou same_layout_or_throw.
struct AmrHierarchyLayout {
  std::vector<BoxArray> ba;             // [niveau] : boites du niveau (ensemble ET ordre)
  std::vector<DistributionMapping> dm;  // [niveau], parallele a ba : rang MPI par boite
  std::vector<Real> dx, dy;             // [niveau] : pas d'espace (= dx_coarse / 2^k)

  /// Nombre de niveaux (= ba.size()).
  int nlev() const { return static_cast<int>(ba.size()); }

  // Lit le layout porte par la pile de niveaux d'UN bloc (chaque AmrLevelMP porte
  // U.box_array() / U.dmap() / dx,dy). Aucune copie de donnees de champ : seulement la grille.
  /// Extrait le layout (BoxArray + DistributionMapping + dx/dy par niveau) de la pile de niveaux d'UN
  /// bloc. Aucune copie de donnees de champ, seulement la grille.
  static AmrHierarchyLayout from_levels(const std::vector<AmrLevelMP>& levels) {
    AmrHierarchyLayout L;
    const int n = static_cast<int>(levels.size());
    L.ba.reserve(n);
    L.dm.reserve(n);
    L.dx.reserve(n);
    L.dy.reserve(n);
    for (const auto& lv : levels) {
      L.ba.push_back(lv.U.box_array());
      L.dm.push_back(lv.U.dmap());
      L.dx.push_back(lv.dx);
      L.dy.push_back(lv.dy);
    }
    return L;
  }
};

namespace detail {
template <class>
inline constexpr bool amr_always_false_v = false;

// Comparaison EXACTE des grilles de deux niveaux (point 1) : meme BoxArray (boites ET ordre),
// meme DistributionMapping (rang par boite), meme dx/dy (au bit pres). Renvoie true si tout
// concorde. dx/dy sont les pas du niveau, identiques par construction si les boites le sont ;
// on les compare quand meme pour attraper une geometrie mal cablee.
inline bool same_level_layout(const BoxArray& a_ba, const DistributionMapping& a_dm, Real a_dx,
                              Real a_dy, const BoxArray& b_ba, const DistributionMapping& b_dm,
                              Real b_dx, Real b_dy) {
  return a_ba.boxes() == b_ba.boxes() && a_dm.ranks() == b_dm.ranks() && a_dx == b_dx &&
         a_dy == b_dy;
}

// Garde-fou de COHERENCE DE LAYOUT entre blocs (point 1 du capstone). L'aux est PARTAGE par
// niveau : tous les blocs DOIVENT vivre sur EXACTEMENT la meme grille a chaque niveau, sinon le
// recablage levels[k].aux = &aux_[k] et l'avance lisent une grille incoherente (acces hors borne
// silencieux). L'ancien controle ne comparait que le NOMBRE de boites (.size()) ; ici on compare
// EXACTEMENT : nombre de niveaux, puis par niveau BoxArray (boites ET ordre), DistributionMapping
// et dx/dy. Jette une erreur claire au PREMIER ecart (bloc et niveau localises). Un seul bloc
// concorde trivialement avec lui-meme -> chemin mono-bloc strictement bit-identique (la boucle
// sur les autres blocs est vide).
inline void same_layout_or_throw(const std::vector<std::vector<AmrLevelMP>>& block_levels) {
  if (block_levels.empty()) return;
  const auto& ref = block_levels[0];
  const int nlev = static_cast<int>(ref.size());
  for (std::size_t b = 1; b < block_levels.size(); ++b) {
    const auto& cur = block_levels[b];
    if (static_cast<int>(cur.size()) != nlev)
      throw std::runtime_error(
          "AmrSystemCoupler : tous les blocs doivent avoir le meme nombre de niveaux "
          "(layout AMR partage)");
    for (int k = 0; k < nlev; ++k) {
      if (!same_level_layout(cur[k].U.box_array(), cur[k].U.dmap(), cur[k].dx, cur[k].dy,
                             ref[k].U.box_array(), ref[k].U.dmap(), ref[k].dx, ref[k].dy))
        throw std::runtime_error(
            "AmrSystemCoupler : layout AMR incoherent entre blocs (l'aux partage exige le MEME "
            "BoxArray [boites et ordre], le MEME DistributionMapping et le MEME dx/dy par niveau)");
    }
  }
}
}  // namespace detail

/// Coupleur de systeme multi-especes sur AMR. @tparam System : CoupledSystem (blocs/especes).
/// @tparam RhsAssembler : assembleur du RHS de Poisson (f = Sum_s q_s n_s, p.ex. ChargeDensityRhs).
/// @tparam Elliptic : backend elliptique (concept EllipticSolver, defaut GeometricMG). PRECONDITION :
/// tous les blocs partagent EXACTEMENT le meme layout AMR par niveau (verifie au ctor).
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
  // bz : champ magnetique hors-plan B_z(x, y) fourni par l'utilisateur (constante ou champ),
  // partage par TOUS les blocs. Pose sur la composante B_z (indice kAuxBaseComps) du canal aux
  // PARTAGE de CHAQUE niveau, depuis les centres de cellule DE CE NIVEAU (chaque niveau a sa
  // propre geometrie / dx). Calque AMR du bz_ de SystemAssembler (chemin non-AMR). Un bloc qui
  // lit B_z (n_aux=4) le voit a tous les niveaux, un bloc de base (3) ignore la composante. Sans
  // bloc a champ extra (largeur 3) ou si bz vide : no-op -> bit-identique a l'historique.
  AmrSystemCoupler(System system, const Geometry& geom, const BoxArray& ba_coarse,
                   const BCRec& bcPhi, RhsAssembler rhs_assembler,
                   std::vector<std::vector<AmrLevelMP>> block_levels,
                   Periodicity base_per = Periodicity{true, true},
                   bool replicated_coarse = true,
                   PoissonCadence cadence = PoissonCadence::OncePerStep,
                   std::function<bool(Real, Real)> active = {},
                   std::function<Real(Real, Real)> bz = {})
      : system_(std::move(system)),
        rhs_assembler_(std::move(rhs_assembler)),
        geom_(geom),
        dom_(geom.domain),
        base_per_(base_per),
        bcPhi_(bcPhi),
        aux_bc_(detail::derive_aux_bc(bcPhi)),
        replicated_coarse_(replicated_coarse),
        cadence_(cadence),
        mg_(geom, ba_coarse, bcPhi, std::move(active), replicated_coarse),
        block_levels_(std::move(block_levels)),
        bz_(std::move(bz)) {
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
    // Coherence de layout EXACTE entre blocs (l'aux est partage par niveau) : meme nombre de
    // niveaux, et par niveau meme BoxArray (boites ET ordre), meme DistributionMapping, meme
    // dx/dy. Remplace l'ancien controle qui ne comparait que le NOMBRE de boites (.size()).
    // Mono-bloc : la verification est triviale (un seul bloc) -> bit-identique a l'historique.
    detail::same_layout_or_throw(block_levels_);
    // aux PARTAGE : un MultiFab (phi, grad phi [, B_z, ...]) par niveau, sur la grille commune.
    // Dimensionne une seule fois -> adresses stables pour les pointeurs aux des blocs. Largeur =
    // max des aux_comps<Model> sur les blocs (au moins 3) : un bloc lisant B_z (n_aux > 3) dispose
    // de la place a CHAQUE niveau, un bloc de base ignore les composantes extra. Sans bloc a champ
    // extra -> largeur 3 -> allocation strictement bit-identique a l'historique.
    aux_ncomp_ = system_aux_comps(system_);
    aux_.resize(nlev_);
    for (int k = 0; k < nlev_; ++k)
      aux_[k] = MultiFab(block_levels_[0][k].U.box_array(),
                         block_levels_[0][k].U.dmap(), aux_ncomp_, 1);
    for (auto& levels : block_levels_)
      for (int k = 0; k < nlev_; ++k) levels[k].aux = &aux_[k];

    // re-pointe chaque bloc vers SON niveau grossier (block.U() = grossier du bloc).
    std::size_t b = 0;
    system_.for_each_block([&](auto& block) {
      block.state = &block_levels_[b][0].U;
      ++b;
    });

    fill_bz();  // peuple B_z par niveau (no-op si aucun bloc ne le demande ou si bz vide)
  }

  // Setter (parite avec le ctor : alternative pour poser B_z apres construction). Re-peuple
  // immediatement le canal aux de chaque niveau. No-op effectif si la largeur aux <= base.
  void set_bz(std::function<Real(Real, Real)> bz) {
    bz_ = std::move(bz);
    fill_bz();
  }

  System& system() { return system_; }
  const System& system() const { return system_; }
  MultiFab& phi() { return mg_.phi(); }
  int nlev() const { return nlev_; }
  const MultiFab& aux(int k) const { return aux_[k]; }
  // Acces ECRITURE au canal aux partage du niveau k (parite avec SystemAssembler::aux()) :
  // permet de peupler une composante extra (B_z, ...) que field_postprocess ne touche pas
  // (il n'ecrit que phi/grad, comp 0..2). La largeur est aux_ncomp_ (max aux_comps des blocs).
  MultiFab& aux(int k) { return aux_[k]; }
  int aux_ncomp() const { return aux_ncomp_; }
  std::vector<AmrLevelMP>& levels(std::size_t b) { return block_levels_[b]; }
  MultiFab& coarse(std::size_t b) { return block_levels_[b][0].U; }
  const MultiFab& coarse(std::size_t b) const { return block_levels_[b][0].U; }
  // nombre de resolutions Poisson du dernier step() : diagnostic de la cadence.
  int solve_count() const { return solve_count_; }

  // sync_down (par bloc) + Poisson grossier de systeme + aux grossier + injection fine.
  /// Resout les champs : average_down par bloc, Poisson grossier de systeme (RHS = Sum_s q_s n_s),
  /// aux grossier (phi, grad phi) puis injection vers les fins + re-pose B_z par niveau. Incremente
  /// solve_count().
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

    // B_z PAR NIVEAU (pas seulement propage) : coupler_inject_aux_mb recopie TOUTES les
    // composantes du parent (dont B_z) vers les fins, ce qui ecraserait le B_z fin par un B_z
    // grossier injecte (constant par cellule grossiere). On re-pose donc B_z depuis bz_ aux
    // centres FINS apres l'injection, pour qu'un B_z spatialement variable soit echantillonne a
    // la resolution du niveau. Statique et bon marche ; no-op si la largeur aux <= base ou bz vide
    // (B_z constant : ce re-fill est idempotent, l'injection aurait suffi).
    fill_bz();
  }

  // Avance le systeme d'un pas. Blocs explicites : advance_amr avec leur Disc et leurs
  // sous-pas d'espece. Blocs implicites / IMEX : delegues au callback (coupler, block,
  // levels, dt), point de branchement Newton / IMEX (defaut AmrImplicitSourceStepper).
  /// Avance le systeme d'un macro-pas dt. Blocs explicites via advance_amr (leur Disc + sous-pas) ;
  /// blocs implicites/IMEX delegues a @p implicit_advance (coupler, block, levels, dt). La cadence
  /// par bloc (stride) tient un bloc lent puis le rattrape (hold-then-catch-up).
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
      constexpr int stride = block_stride_v<Block>;
      const std::size_t bi = b++;
      // cadence HOLD-THEN-CATCH-UP (doc add_block, sec.8.2 C) : le bloc est TENU aux
      // macro-pas 0..stride-2 et rattrape au macro-pas stride-1 (quand
      // (macro_step_+1) % stride == 0). Evite qu'un bloc lent avance en avance
      // au premier macro-pas (macro_step_=0, ancienne condition 0%stride==0 vraie),
      // ce qui mettait le bloc lent DANS LE FUTUR par rapport aux blocs rapides.
      // stride=1 : (macro_step_+1)%1==0 toujours vrai -> chaque pas, bit-identique.
      if ((macro_step_ + 1) % stride != 0) return;
      const Real bdt = dt * static_cast<Real>(stride);
      auto& levels = block_levels_[bi];
      if constexpr (treatment == TimeTreatment::Explicit) {
        const Real h = bdt / static_cast<Real>(n);
        for (int s = 0; s < n; ++s) {
          // PerSubstep : re-resout phi avant chaque sous-pas suivant (la charge a
          // bouge) ; le premier reutilise le solve de tete. OncePerStep : phi gele.
          if (cadence_ == PoissonCadence::PerSubstep && s > 0) solve_fields();
          advance_amr<typename Disc::Limiter, typename Disc::NumericalFlux>(
              block.model, levels, dom_, h, base_per_, replicated_coarse_);
        }
      } else if constexpr (treatment == TimeTreatment::Implicit ||
                           treatment == TimeTreatment::IMEX) {
        // IMEX = vrai forward-backward (revue Codex 9.1) : transport explicite par le
        // moteur AMR sur un modele SOURCE-FREE (-div F seul), puis source implicite par
        // le callback. Implicite pur : tout au callback (pas de transport).
        if constexpr (treatment == TimeTreatment::IMEX)
          advance_amr<typename Disc::Limiter, typename Disc::NumericalFlux>(
              SourceFreeModel<typename Block::Model>{block.model}, levels, dom_, bdt,
              base_per_, replicated_coarse_);
        implicit_advance(*this, block, levels, bdt);
      }
    });
    ++macro_step_;
  }

  // Surcharge pour un systeme entierement explicite.
  /// Surcharge pour un systeme ENTIEREMENT explicite (static_assert si un bloc implicite/IMEX y passe).
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
    // INVARIANT DE COUVERTURE : la source a ete appliquee independamment sur CHAQUE
    // niveau, donc une cellule grossiere COUVERTE par un patch fin porte maintenant sa
    // propre source grossiere, sans rapport avec la source vue par ses enfants fins. Une
    // cellule grossiere couverte doit, par definition, etre la moyenne 2x2 de ses enfants
    // (elle ne represente pas de matiere a elle seule, elle est une vue grossiere du fin).
    // On restaure cette coherence par une cascade fin -> grossier identique a celle de
    // solve_fields et du chemin transport-IMEX (subcycle_level_mp). Sans elle, le
    // diagnostic amr_mass (qui somme le seul niveau grossier) compte une source grossiere
    // fantome sous le patch. Hierarchie mono-niveau : aucune cellule couverte, la boucle
    // ne s'execute pas -> strictement bit-identique a l'historique.
    for (auto& levels : block_levels_)
      for (int k = nlev_ - 1; k >= 1; --k)
        mf_average_down_mb(levels[k].U, levels[k - 1].U);
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
  // Largeur du canal aux PARTAGE : maximum des aux_comps<Model> sur tous les blocs (au moins
  // kAuxBaseComps). Le canal partage par niveau doit etre au moins aussi large que le bloc le
  // plus exigeant pour que load_aux<aux_comps<Model>> n'y lise jamais hors borne dans les chemins
  // AMR (compute_face_fluxes, mf_apply_source, ...) ; un bloc moins exigeant ignore simplement les
  // composantes extra. Calque exact de SystemAssembler::system_aux_comps (chemin non-AMR). Sans
  // bloc a champ extra, la largeur reste 3 -> allocation strictement bit-identique a l'historique.
  static int system_aux_comps(const System& sys) {
    int w = kAuxBaseComps;
    sys.for_each_block([&](const auto& b) {
      using Model = std::decay_t<decltype(b.model)>;
      const int c = aux_comps<Model>();
      if (c > w) w = c;
    });
    return w;
  }
  // Peuple la composante aux B_z (indice kAuxBaseComps) du canal partage de CHAQUE niveau depuis
  // bz_(x, y). B_z est statique (externe a l'elliptique) : pose une fois (au ctor / set_bz),
  // preserve par solve_fields (field_postprocess n'ecrit que phi/grad, comp 0..2 ; on re-pose
  // apres l'injection coarse->fine qui, elle, recopierait un B_z grossier) et par l'avance (le
  // moteur AMR ne touche pas l'aux). Chaque niveau a SA geometrie : niveau k = geom_.refine(1 << k),
  // memes extents physiques mais domaine d'indices raffine, donc x_cell/y_cell pointent au centre
  // physique de la cellule FINE. On remplit la GROWN box (valides + halos) directement depuis
  // bz_(x, y) : bz_ etant fonction pure de la position physique, son evaluation aux centres ghost
  // donne le B_z physiquement correct la aussi (independant des BC du patch fin, sans ambiguite de
  // periodicite sur un domaine de patch). No-op si la largeur aux <= kAuxBaseComps (aucun bloc ne
  // lit B_z) ou si bz_ vide : garde RUNTIME (la largeur n'est connue qu'a la construction) ->
  // modele de base strictement bit-identique a l'historique.
  void fill_bz() {
    if (!bz_ || aux_ncomp_ <= kAuxBaseComps) return;
    for (int k = 0; k < nlev_; ++k) {
      const Geometry gk = geom_.refine(1 << k);  // geometrie du niveau k (dx = dx_coarse / 2^k)
      MultiFab& A = aux_[k];
      for (int li = 0; li < A.local_size(); ++li) {
        Fab2D& f = A.fab(li);
        // grown box (valides + halos) : B_z(x,y) correct partout, geometrie du niveau k.
        detail::fill_bz_box(f, f.grown_box(), gk, bz_);
      }
    }
  }

  Geometry geom_;
  Box2D dom_;
  Periodicity base_per_;
  BCRec bcPhi_, aux_bc_;
  bool replicated_coarse_;
  PoissonCadence cadence_;
  mutable int solve_count_ = 0;
  int macro_step_ = 0;  // compteur de macro-pas (cadence stride par bloc)
  Elliptic mg_;
  std::vector<std::vector<AmrLevelMP>> block_levels_;  // [bloc][niveau]
  std::vector<MultiFab> aux_;                          // [niveau], partage
  int aux_ncomp_ = kAuxBaseComps;  // largeur du canal aux partage (max aux_comps sur les blocs)
  int nlev_ = 0;
  std::function<Real(Real, Real)> bz_;  // B_z(x, y) externe (vide si non fourni)
};

// Defaut implicite sur AMR : backward-Euler (Newton) sur la source du modele, applique
// a CHAQUE niveau de la hierarchie du bloc. Pendant AMR d'ImplicitSourceStepper ; meme
// stabilite (inconditionnelle pour une relaxation lineaire). Aucun solveur cote user.
/// Callback implicite par defaut pour AmrSystemCoupler::step : backward-Euler (Newton) sur la source
/// du modele, applique a CHAQUE niveau de la hierarchie, suivi d'une cascade fin -> grossier
/// (coherence de couverture, cf. coupled_source_step). @p iters : iterations de Newton par etage.
struct AmrImplicitSourceStepper {
  int iters = 2;

  template <class Coupler, class Block, class Levels>
  void operator()(Coupler& coupler, Block& block, Levels& levels, Real dt) const {
    const int nlev = static_cast<int>(levels.size());
    for (int k = 0; k < nlev; ++k)
      backward_euler_source(block.model, coupler.aux(k), levels[k].U, dt, iters);
    // INVARIANT DE COUVERTURE (cf. coupled_source_step) : la source implicite a ete
    // resolue independamment niveau par niveau, donc les cellules grossieres COUVERTES
    // portent une source grossiere fantome au lieu de la moyenne 2x2 de leurs enfants
    // fins. On retablit la coherence par la meme cascade fin -> grossier que le chemin
    // transport-IMEX, pour qu'une cellule grossiere couverte reste la vue grossiere du fin
    // (sinon amr_mass, somme du seul grossier, compte la source du patch en double).
    // Mono-niveau : aucune cellule couverte, boucle vide -> bit-identique a l'historique.
    for (int k = nlev - 1; k >= 1; --k)
      mf_average_down_mb(levels[k].U, levels[k - 1].U);
  }
};

// Alias "qui avance" (retour tuteur sec.8.2 B, sec.9.6) : AmrSystemCoupler assemble (Poisson de
// systeme + aux par niveau) ET avance (step, reflux, sous-cyclage). Scission en deux classes
// cosmetique et reportee (classe unifiee validee bit-identique).
template <CoupledSystemLike System, class RhsAssembler, class Elliptic = GeometricMG>
using AmrSystemDriver = AmrSystemCoupler<System, RhsAssembler, Elliptic>;

}  // namespace adc
