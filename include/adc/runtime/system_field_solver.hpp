#pragma once

#include <adc/core/state.hpp>       // kAuxBaseComps (composante de base du canal aux)
#include <adc/core/types.hpp>       // Real
#include <adc/mesh/multifab.hpp>    // MultiFab, Array4, ConstArray4
#include <adc/mesh/box2d.hpp>       // Box2D
#include <adc/mesh/for_each.hpp>    // device_fence
#include <adc/mesh/physical_bc.hpp>  // BCRec, fill_ghosts, fill_boundary
#include <adc/numerics/elliptic/geometric_mg.hpp>
#include <adc/numerics/elliptic/poisson_fft_solver.hpp>
#include <adc/numerics/elliptic/polar_poisson_solver.hpp>  // PolarPoissonSolver (Poisson polaire direct)
#include <adc/parallel/comm.hpp>  // n_ranks() (garde-fou FFT MPI)
#include <adc/runtime/block_builder_polar.hpp>  // derive_aux_polar (aux polaire en base locale)
#include <adc/runtime/wall_predicate.hpp>       // detail::wall_predicate

#include <cstdio>   // ADC_TRACE_SOLVE_FIELDS : trace de diagnostic device (env-gate, inerte par defaut)
#include <cstdlib>  // getenv
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

/// @file
/// @brief SystemFieldSolver : la responsabilite RESOLUTION ELLIPTIQUE + DERIVATION DE CHAMP extraite
///        du god-class System::Impl (audit Lot B, cf. docs/SYSTEM_CPP_EXTRACTION_PLAN.md section 2).
///        Extrait VERBATIM de python/system.cpp : aucune modification de numerique, d'ordre des
///        operations, de fill_ghosts/fill_boundary, de device_fence ni de tolerance. STRICTEMENT
///        bit-identique -- le code est deplace tel quel, seul l'acces aux membres PARTAGES de Impl
///        (aux, sp, cfg, geom, pgeom_, ba, dm, bc_, dom, per_, periodic_, polar_) passe par le
///        back-pointer owner_->.
///
/// CONTRAT / INVARIANTS
/// - POSSEDE : les solveurs elliptiques (ell_ cartesien GeometricMG/PoissonFFTSolver ; pell_ polaire
///   PolarPoissonSolver), les jetons de configuration Poisson (p_rhs/p_solver/p_bc/p_wall/
///   p_wall_radius/p_eps_), les champs et drapeaux de coefficient (eps(x), eps_x/eps_y, kappa) ainsi
///   que les tampons d'APPLICATION de champ aux (bz_field_ et te_src_) avec leurs methodes apply_bz /
///   apply_te.
/// - LIT (sans posseder) l'aux PARTAGE et la liste de blocs via owner_-> : l'aux est peuplee par
///   solve_fields (phi, grad phi, B_z, T_e) puis ses halos sont remplis ; la liste de blocs fournit le
///   second membre du Poisson (somme des briques elliptiques par bloc) et la source fluide de T_e.
/// - DISPATCH cartesien vs polaire : solve_fields() route vers solve_fields_polar() quand owner_->polar_,
///   sinon le chemin cartesien. Les deux chemins sont ind+ependants (ell_ jamais touche en polaire et
///   reciproquement). ensure_elliptic / ensure_elliptic_polar construisent paresseusement le solveur.
/// - INVARIANT device CRITIQUE : le device_fence() entre ell_solve() et la derivation de grad phi DOIT
///   rester atomique (sans lui, le V-cycle GPU n'est pas termine quand on lit phi). Idem en polaire
///   apres pell_->solve(). NE PAS reordonner.
/// - INVARIANT MPI : les boucles de derivation / peuplement (B_z, T_e, eps, kappa) iterent sur les fabs
///   LOCAUX (local_size()), jamais fab(0) en dur : no-op sur un rang sans box, bit-identique au
///   proprietaire. Cette garde est PRESERVEE par l'extraction.
///
/// Comme System::Impl reste PRIVE a python/system.cpp, ce helper est un TEMPLATE parametre sur le type
/// Impl reel (meme technique que native_loader) : python/system.cpp l'instancie avec System::Impl apres
/// avoir defini Impl. owner_ est un Impl* (la duree de vie du helper est sous-jacente a celle de Impl).

namespace adc {
namespace field_solver {

/// Vrai si la trace de DIAGNOSTIC du chemin solve_fields est active (variable d'environnement
/// ADC_TRACE_SOLVE_FIELDS posee). Jalon #93 (crash device GH200) : ecrit sur stderr avec flush immediat
/// pour localiser le dernier marqueur avant un crash device. INERTE par defaut : aucun effet sur les
/// sorties ni sur la numerique. Diagnostic CONSERVE (env-gate) : utile pour un futur crash device.
inline bool adc_trace_sf() {
  static const bool on = std::getenv("ADC_TRACE_SOLVE_FIELDS") != nullptr;
  return on;
}
/// Ecrit le marqueur @p w sur stderr (avec flush) SEULEMENT si adc_trace_sf() ; no-op sinon.
inline void adc_sf_mark(const char* w) {
  if (adc_trace_sf()) { std::fprintf(stderr, "[sf] %s\n", w); std::fflush(stderr); }
}

/// SystemFieldSolver<Impl> : voir contrat ci-dessus. Toutes les methodes sont des MEMBRES (et non des
/// fonctions libres) car elles partagent l'etat elliptique possede par cette classe ; les acces a l'etat
/// PARTAGE de Impl passent par owner_-> verbatim. Template sur Impl pour rester sans dependance sur la
/// definition (privee) de System::Impl.
template <class Impl>
class SystemFieldSolver {
 public:
  /// @param owner back-pointer vers System::Impl (duree de vie sous-jacente a celle de Impl).
  explicit SystemFieldSolver(Impl* owner) : owner_(owner) {}

  /// Composante canonique de T_e (apres phi/grad/B_z) ; cf. adc::Aux et AUX_CANONICAL cote DSL.
  static constexpr int kTeComp = kAuxBaseComps + 1;  // = 4

  // --- etat POSSEDE (resolution elliptique + champs de coefficient + tampons d'application) --------
  // Configuration Poisson (solveur elliptique construit paresseusement).
  std::string p_rhs = "charge_density";
  std::string p_solver = "geometric_mg";
  std::string p_bc = "auto";
  std::string p_wall = "none";
  double p_wall_radius = 0.0;
  Real p_eps_ = 1;  // permittivite CONSTANTE : div(eps grad phi) = f <=> lap phi = f/eps
  bool has_eps_field_ = false;          // permittivite VARIABLE eps(x) fournie (porte par l'operateur)
  std::vector<double> p_eps_field_;     // champ eps(x), n*n row-major (si has_eps_field_)
  bool has_eps_xy_field_ = false;       // permittivite ANISOTROPE eps_x(x), eps_y(x) (operateur div(diag(eps_x,eps_y) grad phi))
  std::vector<double> p_eps_x_field_;   // champ eps_x(x), n*n row-major (faces normales a x ; si has_eps_xy_field_)
  std::vector<double> p_eps_y_field_;   // champ eps_y(x), n*n row-major (faces normales a y ; si has_eps_xy_field_)
  bool has_kappa_field_ = false;        // terme de REACTION kappa(x) fourni : div(eps grad phi) - kappa phi
  std::vector<double> p_kappa_field_;   // champ kappa(x), n*n row-major (si has_kappa_field_)
  std::optional<std::variant<GeometricMG, PoissonFFTSolver>> ell_;
  // Solveur de Poisson POLAIRE direct (FFT-en-theta + tridiag-en-r), construit paresseusement quand
  // polar_ (cf. ensure_elliptic_polar). SEPARE de ell_ (geom() rend une PolarGeometry, pas une
  // Geometry) : le chemin cartesien n'est jamais touche. INERTE (nullopt) en cartesien.
  std::optional<PolarPoissonSolver> pell_;
  // Tampon phi de l'ETAGE SOURCE condense POLAIRE (Voie A etape 2c). Le PolarPoissonSolver direct
  // (pell_->phi()) est SANS ghost (box valide = allocation) ; or le PolarCondensedSchurSourceStepper
  // a besoin d'un phi AVEC 1 ghost (fill_ghosts + apply_polar_tensor + grad centre + ecriture de
  // phi^{n+1}). On lui passe donc ce tampon dedie (1 ghost), alimente par phi^n (= aux[0] apres
  // solve_fields_polar) avant le source stage, et qui porte phi^{n+1} en sortie (warm start du pas
  // suivant). En CARTESIEN ce tampon est INERTE (nullopt) : ell_phi() route vers ell_->phi() comme
  // avant, BIT-IDENTIQUE.
  std::optional<MultiFab> phi_src_polar_;
  std::vector<Real> bz_field_;        // champ B_z(x) n*n row-major (vide si non fourni)
  int te_src_ = -1;                   // indice du bloc fluide source de T_e (-1 = aucune)

  /// Peuple la composante B_z (indice kAuxBaseComps) du canal partage depuis bz_field_, sur les
  /// cellules valides. No-op si B_z non fourni ou si aucun bloc ne le lit (largeur de base). Les
  /// halos de B_z sont remplis par solve_fields (comme grad) ; field_postprocess n'ecrit que comp 0..2.
  void apply_bz() {
    if (bz_field_.empty() || owner_->aux_ncomp_ <= kAuxBaseComps) return;
    // LARGEUR DE LIGNE (axe rapide i) du tableau row-major bz_field_ : n en cartesien (carre n x n,
    // BIT-IDENTIQUE), nr en POLAIRE (anneau nr x ntheta, i = r de taille nr, cf. set_magnetic_field).
    // L'index reste flat[j * row + i] : en cartesien row == n (inchange) ; en polaire row == nr.
    const int row = owner_->polar_ ? owner_->aux.box(0).nx() : owner_->cfg.n;
    // Peuplement LOCAL au rang proprietaire (cf. solve_fields) : iteration sur les fabs locaux du
    // canal aux au lieu de fab(0) en dur (no-op sur un rang sans box locale a np>1, bit-identique au
    // proprietaire).
    for (int li = 0; li < owner_->aux.local_size(); ++li) {
      Array4 a = owner_->aux.fab(li).array();
      const Box2D v = owner_->aux.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          a(i, j, kAuxBaseComps) = bz_field_[static_cast<std::size_t>(j) * row + i];
    }
  }

  /// Peuple la composante T_e (temperature electronique) = p/rho du bloc fluide source te_src_.
  /// RECALCULEE a chaque solve_fields (T_e varie avec le fluide, contrairement a B_z statique).
  /// No-op si aucune source ou si aucun bloc ne lit T_e (largeur insuffisante). Le bloc source est
  /// compressible (4 var) ; p = (gamma-1)(E - 0.5 rho|v|^2), T = p/rho.
  void apply_te() {
    if (te_src_ < 0 || owner_->aux_ncomp_ <= kTeComp) return;
    const auto& s = owner_->sp[static_cast<std::size_t>(te_src_)];
    const Real gm1 = Real(s.gamma) - Real(1);
    // Peuplement LOCAL au rang proprietaire (cf. solve_fields) : on itere sur les fabs locaux du
    // canal aux au lieu de fab(0) en dur (no-op sur un rang sans box locale a np>1, bit-identique au
    // proprietaire). s.U et aux partagent la meme DistributionMapping -> meme indexation locale.
    for (int li = 0; li < owner_->aux.local_size(); ++li) {
      const ConstArray4 us = s.U.fab(li).const_array();
      Array4 a = owner_->aux.fab(li).array();
      const Box2D v = owner_->aux.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
          const Real rho = us(i, j, 0), mx = us(i, j, 1), my = us(i, j, 2), E = us(i, j, 3);
          const Real p = gm1 * (E - Real(0.5) * (mx * mx + my * my) / rho);
          a(i, j, kTeComp) = p / rho;  // T = p / rho
        }
    }
  }

  // --- solveur elliptique (Poisson de systeme) -----------------------------
  /// Resout le mode de CL en BCRec : "auto" -> dirichlet si paroi/non-periodique, sinon periodic ;
  /// "periodic"|"dirichlet"|"neumann" (Foextrap). @throws std::runtime_error sur un mode inconnu.
  BCRec poisson_bc() {
    std::string mode = p_bc;
    if (mode == "auto") mode = (p_wall == "circle" || !owner_->cfg.periodic) ? "dirichlet" : "periodic";
    BCRec b;
    if (mode == "periodic") return b;
    if (mode == "dirichlet") {
      b.xlo = b.xhi = b.ylo = b.yhi = BCType::Dirichlet;
      return b;
    }
    if (mode == "neumann") {
      b.xlo = b.xhi = b.ylo = b.yhi = BCType::Foextrap;
      return b;
    }
    throw std::runtime_error("System::set_poisson : bc inconnu '" + mode + "'");
  }
  /// Predicat "interieur du conducteur" depuis p_wall / p_wall_radius / cfg.L (cf. wall_predicate) ;
  /// vide si pas de paroi.
  std::function<bool(Real, Real)> wall_active() {
    return detail::wall_predicate(p_wall, p_wall_radius, owner_->cfg.L, "System::set_poisson");
  }
  /// Construit PARESSEUSEMENT le solveur elliptique cartesien (ell_) selon p_solver : GeometricMG
  /// (porte eps(x)/aniso/kappa s'ils sont fournis) ou PoissonFFTSolver (coefficient constant, mono-rang,
  /// sans paroi). No-op si ell_ existe deja. @throws std::runtime_error sur rhs/solver inconnu ou
  /// combinaison non supportee (fft + MPI/paroi/eps variable/kappa ; kappa + eps constante != 1).
  void ensure_elliptic() {
    if (ell_) return;
    // Le second membre de systeme est TOUJOURS f = Sum_s elliptic_rhs_s(u_s), assemble par
    // solve_fields a partir de la brique elliptique de CHAQUE bloc (charge q n, fond alpha (n-n0),
    // couplage gravite 4piG (rho-rho0)). Le token n'est donc PAS un mode de calcul mais une ETIQUETTE
    // de ce second membre compose. "composite" nomme honnetement ce comportement ; "charge_density"
    // reste l'alias historique (defaut, bit-identique) car le cas usuel est un bloc de charge.
    if (p_rhs != "charge_density" && p_rhs != "composite")
      throw std::runtime_error("System::set_poisson : rhs '" + p_rhs +
                               "' inconnu (charge_density|composite ; le second membre = somme des "
                               "briques elliptiques par bloc)");
    const BCRec pbc = poisson_bc();
    std::function<bool(Real, Real)> active = wall_active();
    if (p_solver == "fft") {
      // FFT directe mono-rang : sous MPI (n_ranks>1) System repartit UNE box en round-robin, donc
      // des rangs ont local_size()==0 et PoissonFFTSolver::solve() dereferencerait fab(0) inexistant
      // (SIGSEGV, l'ancien assert disparaissait en Release). On REFUSE explicitement ici, sur TOUS
      // les rangs (ensure_elliptic est appele collectivement par solve_fields) -> pas d'interblocage.
      // Le periodique distribue passe par DistributedFFTSolver (bandes), non cable dans System (sa
      // decomposition par bandes est incompatible avec la box unique de System -> assemblage du rhs /
      // relecture de phi). PoissonFFTSolver garde aussi un garde-fou dur dans son constructeur.
      if (n_ranks() > 1)
        throw std::runtime_error(
            "solveur fft non supporte en MPI (n_ranks>1) : utiliser geometric_mg ou le solveur fft "
            "distribue");
      if (active)
        throw std::runtime_error("System : solver 'fft' incompatible avec une paroi -> 'geometric_mg'");
      if (has_eps_field_)
        throw std::runtime_error("System : solver 'fft' a coefficient CONSTANT, incompatible avec un "
                                 "champ eps(x) variable -> utiliser solver='geometric_mg'");
      if (has_eps_xy_field_)
        throw std::runtime_error("System : solver 'fft' a coefficient CONSTANT, incompatible avec une "
                                 "permittivite ANISOTROPE eps_x(x), eps_y(x) -> utiliser solver='geometric_mg'");
      if (has_kappa_field_)
        throw std::runtime_error("System : solver 'fft' (Poisson pur) incompatible avec un terme de "
                                 "reaction kappa(x) -> utiliser solver='geometric_mg'");
      ell_.emplace(std::in_place_type<PoissonFFTSolver>, owner_->geom, owner_->ba, pbc, active);
    } else if (p_solver == "geometric_mg") {
      ell_.emplace(std::in_place_type<GeometricMG>, owner_->geom, owner_->ba, pbc, std::move(active));
      if (has_eps_field_) apply_epsilon_field();    // operateur div(eps grad phi) a eps(x) variable
      if (has_eps_xy_field_) apply_epsilon_anisotropic_field();  // div(diag(eps_x, eps_y) grad phi)
      if (has_kappa_field_) apply_reaction_field();  // terme - kappa phi (Poisson ecrante / Helmholtz)
      // Garde-fou : avec kappa et une permittivite CONSTANTE eps != 1 (sans champ eps(x)), le rhs
      // serait mis a l'echelle 1/eps (raccourci lap phi = f/eps) -- incoherent avec le terme -kappa phi.
      // On exige alors eps = 1 ou un champ eps(x) (porte par l'operateur, sans mise a l'echelle).
      if (has_kappa_field_ && !has_eps_field_ && !has_eps_xy_field_ && p_eps_ != Real(1))
        throw std::runtime_error("System : terme de reaction kappa(x) + permittivite CONSTANTE eps != 1 "
                                 "non supporte ; utiliser eps = 1 ou un champ eps(x) (set_epsilon_field)");
    } else {
      throw std::runtime_error("System::set_poisson : solver '" + p_solver +
                               "' inconnu (geometric_mg|fft)");
    }
  }
  /// Installe le champ eps(x) (n*n row-major) sur le GeometricMG : l'operateur passe a
  /// div(eps grad phi) = f, eps PORTE PAR L'OPERATEUR (coefficient de face harmonique, ordre 2),
  /// sans mise a l'echelle 1/eps du second membre. Seul GeometricMG supporte ce coefficient variable.
  void apply_epsilon_field() {
    GeometricMG& mg = std::get<GeometricMG>(*ell_);
    MultiFab eps_fine(owner_->ba, owner_->dm, 1, 0);
    const int n = owner_->cfg.n;
    // Remplissage du champ source LOCAL au rang proprietaire (iteration sur les fabs locaux, jamais
    // fab(0) en dur) : no-op sur un rang sans box locale a np>1, identique a avant sur le
    // proprietaire. mg.set_epsilon est ensuite COLLECTIF (copie locale + restriction MPI-safe).
    for (int li = 0; li < eps_fine.local_size(); ++li) {
      Array4 e = eps_fine.fab(li).array();
      const Box2D v = eps_fine.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          e(i, j, 0) = static_cast<Real>(p_eps_field_[static_cast<std::size_t>(j) * n + i]);
    }
    mg.set_epsilon(eps_fine);  // copie sur le niveau fin + restriction (average_down) aux grossiers
  }
  /// Installe les champs eps_x(x), eps_y(x) (n*n row-major chacun) sur le GeometricMG : l'operateur
  /// passe a div(diag(eps_x, eps_y) grad phi) = f. Les faces normales a x lisent eps_x, celles
  /// normales a y lisent eps_y (coefficients de face harmoniques, ordre 2), PORTES PAR L'OPERATEUR
  /// sans mise a l'echelle 1/eps du second membre. GeometricMG seul (coefficient tensoriel variable).
  void apply_epsilon_anisotropic_field() {
    GeometricMG& mg = std::get<GeometricMG>(*ell_);
    MultiFab eps_x_fine(owner_->ba, owner_->dm, 1, 0), eps_y_fine(owner_->ba, owner_->dm, 1, 0);
    const int n = owner_->cfg.n;
    // Remplissage LOCAL au rang proprietaire (cf. apply_epsilon_field) : iteration sur les fabs
    // locaux (no-op sur un rang vide a np>1). eps_x_fine et eps_y_fine partagent ba/dm -> meme
    // indexation locale. mg.set_epsilon_anisotropic est ensuite COLLECTIF (copie + restriction).
    for (int li = 0; li < eps_x_fine.local_size(); ++li) {
      Array4 ex = eps_x_fine.fab(li).array();
      Array4 ey = eps_y_fine.fab(li).array();
      const Box2D v = eps_x_fine.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
          const std::size_t k = static_cast<std::size_t>(j) * n + i;
          ex(i, j, 0) = static_cast<Real>(p_eps_x_field_[k]);
          ey(i, j, 0) = static_cast<Real>(p_eps_y_field_[k]);
        }
    }
    mg.set_epsilon_anisotropic(eps_x_fine, eps_y_fine);  // faces x <- eps_x, faces y <- eps_y (+ restriction)
  }
  /// Installe le terme de reaction kappa(x) (n*n row-major) sur le GeometricMG : l'operateur passe a
  /// div(eps grad phi) - kappa phi = f (Poisson ecrante / Helmholtz ; kappa = 1/lambda_D^2 pour Debye).
  /// kappa est DIAGONAL (lu en cellule), restreint par moyenne aux niveaux grossiers. GeometricMG seul.
  void apply_reaction_field() {
    GeometricMG& mg = std::get<GeometricMG>(*ell_);
    MultiFab kappa_fine(owner_->ba, owner_->dm, 1, 0);
    const int n = owner_->cfg.n;
    // Remplissage LOCAL au rang proprietaire (cf. apply_epsilon_field) : iteration sur les fabs
    // locaux (no-op sur un rang vide a np>1). mg.set_reaction est ensuite COLLECTIF.
    for (int li = 0; li < kappa_fine.local_size(); ++li) {
      Array4 k = kappa_fine.fab(li).array();
      const Box2D v = kappa_fine.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          k(i, j, 0) = static_cast<Real>(p_kappa_field_[static_cast<std::size_t>(j) * n + i]);
    }
    mg.set_reaction(kappa_fine);
  }
  /// Second membre f du solveur elliptique cartesien actif (GeometricMG ou FFT), par std::visit.
  MultiFab& ell_rhs() {
    return std::visit([](auto& e) -> MultiFab& { return e.rhs(); }, *ell_);
  }
  /// Potentiel phi lu (et reecrit) par l'etage source condense. CARTESIEN : le phi du solveur
  /// elliptique actif (GeometricMG/FFT, AVEC ghosts), BIT-IDENTIQUE. POLAIRE : un tampon dedie 1 ghost
  /// (phi_src_polar_), alimente avec phi^n (= aux[0], pose par solve_fields_polar) au moment de l'appel
  /// -- le PolarPoissonSolver direct n'a pas de ghosts, donc on ne peut pas exposer pell_->phi()
  /// directement a un stepper qui fait fill_ghosts/apply_polar_tensor. Le stepper y ecrit phi^{n+1}
  /// (warm start du pas suivant ; l'aux[0] sera de toute facon reecrit par le prochain solve_fields).
  MultiFab& ell_phi() {
    if (owner_->polar_) {
      // Alloue paresseusement (1 ghost) sur le layout du System, puis copie phi^n depuis aux[0].
      if (!phi_src_polar_) phi_src_polar_.emplace(owner_->ba, owner_->dm, 1, 1);
      for (int li = 0; li < phi_src_polar_->local_size(); ++li) {
        const ConstArray4 a = owner_->aux.fab(li).const_array();
        Array4 p = phi_src_polar_->fab(li).array();
        const Box2D v = phi_src_polar_->box(li);
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i) p(i, j, 0) = a(i, j, 0);  // aux[0] = phi^n
      }
      return *phi_src_polar_;
    }
    return std::visit([](auto& e) -> MultiFab& { return e.phi(); }, *ell_);
  }
  /// Resout le Poisson cartesien actif (GeometricMG V-cycle ou FFT directe). Pose les marqueurs de
  /// trace ; le device_fence apres ell_solve est porte par l'APPELANT (solve_fields), pas ici.
  void ell_solve() {
    adc_sf_mark("ell_solve: avant std::visit");
    std::visit(
        [](auto& e) {
          using T = std::decay_t<decltype(e)>;
          if (adc_trace_sf())
            adc_sf_mark(std::is_same_v<T, GeometricMG> ? "ell_solve: GeometricMG::solve() debut"
                                                       : "ell_solve: PoissonFFTSolver::solve() debut");
          e.solve();
          adc_sf_mark("ell_solve: solve() retour");
        },
        *ell_);
    adc_sf_mark("ell_solve: apres std::visit");
  }

  // --- Poisson POLAIRE direct (PolarPoissonSolver) -------------------------
  /// Construit PARESSEUSEMENT le Poisson POLAIRE direct (PolarPoissonSolver, mono-rang, box unique
  /// couvrant l'anneau). La BC radiale vient de poisson_bc() (Foextrap -> Neumann homogene, paroi ; le
  /// 'wall' cartesien circulaire n'a pas de sens sur un anneau global et n'est pas applique). theta est
  /// PERIODIQUE (gere par la FFT-en-theta, aucune BC azimutale). ADDITIF : ne touche jamais ell_.
  /// @throws std::runtime_error sur rhs/solver inconnu ou permittivite variable/aniso/reaction (non
  /// supportee par le Poisson polaire direct).
  void ensure_elliptic_polar() {
    if (pell_) return;
    if (p_rhs != "charge_density" && p_rhs != "composite")
      throw std::runtime_error("System::set_poisson (polaire) : rhs '" + p_rhs +
                               "' inconnu (charge_density|composite)");
    if (p_solver != "geometric_mg" && p_solver != "polar")
      throw std::runtime_error(
          "System::set_poisson (polaire) : solver '" + p_solver +
          "' non supporte sur un anneau ; le Poisson polaire est direct (FFT-en-theta + tridiag-en-r). "
          "Laisser le defaut ('geometric_mg') ou demander 'polar'.");
    if (has_eps_field_ || has_eps_xy_field_ || has_kappa_field_)
      throw std::runtime_error(
          "System::set_poisson (polaire) : permittivite variable / anisotrope / reaction non supportee "
          "par le Poisson polaire direct (Phase 2b ; operateur (1/r) d_r(r d_r) + (1/r^2) d_theta^2)");
    // BC radiale : Dirichlet/Neumann depuis poisson_bc() (xlo/xhi). theta toujours periodique.
    const BCRec pbc = poisson_bc();
    pell_.emplace(owner_->pgeom_, owner_->ba, pbc);
  }

  /// solve_fields POLAIRE : assemble f = Sum_s elliptic_rhs_s(U_s) (boucle hote par cellule), resout le
  /// Poisson polaire, puis DERIVE l'aux en base locale (e_r, e_theta) :
  ///   aux[0] = phi ;  aux[1] = grad_r = d phi/dr ;  aux[2] = grad_theta = (1/r) d phi/d theta.
  /// C'est la disposition attendue par ExBVelocityPolar (v_r = -grad_theta/B, v_theta = grad_r/B).
  void solve_fields_polar() {
    ensure_elliptic_polar();
    MultiFab& rhs = pell_->rhs();
    rhs.set_val(Real(0));
    for (auto& s : owner_->sp) s.add_poisson_rhs(s.U, rhs);
    // Permittivite CONSTANTE eps != 1 : lap phi = f/eps (mise a l'echelle 1/eps du rhs), comme le
    // cartesien. (eps(x) variable/aniso est refuse par ensure_elliptic_polar.)
    if (p_eps_ != Real(1)) {
      const Real inv = Real(1) / p_eps_;
      for (int li = 0; li < rhs.local_size(); ++li) {
        Array4 r = rhs.fab(li).array();
        const Box2D v = rhs.box(li);
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i) r(i, j, 0) *= inv;
      }
    }
    pell_->solve();
    device_fence();
    // Derivation (phi, grad_r, grad_theta) en base locale (e_r, e_theta) via le MEME helper que le test
    // C++ (derive_aux_polar de block_builder_polar.hpp). phi est SANS ghost (solveur direct mono-box) :
    // le helper n'en lit donc jamais d'index hors domaine (radial DECENTRE aux parois, theta ENROULE en
    // periodique) -- c'etait le bug : la difference centree lisait phi(lo-1)/phi(hi+1)/phi(.,jlo-1) hors
    // allocation -> gradient parasite -> vitesse divergente -> nan.
    derive_aux_polar(pell_->phi(), owner_->aux, owner_->pgeom_);
    apply_te();  // inerte en polaire ExB (aucun bloc fluide source de T_e), conserve par symetrie
    // Ghosts de l'aux : theta PERIODIQUE (joint 0/2pi), r PHYSIQUE (extrapolation au bord). fill_ghosts
    // route deja par bc_ (xlo/xhi Foextrap, ylo/yhi Periodic) -> halo azimutal periodique correct.
    fill_ghosts(owner_->aux, owner_->dom, owner_->bc_);
  }

  /// Resout le Poisson de systeme puis DERIVE l'aux = (phi, grad phi[, B_z, T_e]). Route vers
  /// solve_fields_polar() en geometrie polaire. INVARIANT device : le device_fence() entre ell_solve()
  /// et la derivation de grad phi DOIT rester atomique (sans lui le V-cycle GPU n'est pas termine quand
  /// on lit phi) ; les boucles de derivation / peuplement iterent sur les fabs LOCAUX (MPI-safe).
  void solve_fields() {
    if (owner_->polar_) return solve_fields_polar();  // anneau : Poisson polaire + aux en base locale (e_r, e_theta)
    adc_sf_mark("solve_fields: debut");
    ensure_elliptic();
    adc_sf_mark("solve_fields: apres ensure_elliptic");
    MultiFab& rhs = ell_rhs();
    rhs.set_val(Real(0));
    adc_sf_mark("solve_fields: apres rhs.set_val(0)");
    // f = Sum_s elliptic_rhs_s(U_s) sur l'etat COURANT de chaque bloc. STRIDE : un bloc de cadence M
    // est tenu (hold-then-catch-up) entre deux rattrapages, donc U_s y reste FIGE a sa derniere avance ;
    // sa densite / charge entre dans cette somme avec un etat PERIME jusqu'a son prochain rattrapage
    // (couplage Poisson lache du bloc lent, assume par le choix du stride).
    for (auto& s : owner_->sp) s.add_poisson_rhs(s.U, rhs);
    adc_sf_mark("solve_fields: apres add_poisson_rhs");
    // Permittivite CONSTANTE : div(eps grad phi) = f <=> lap phi = f/eps, donc on met le rhs a
    // l'echelle 1/eps. Avec un champ eps(x) VARIABLE ou ANISOTROPE on NE le fait PAS : l'operateur
    // GeometricMG porte eps directement (apply_epsilon_field / apply_epsilon_anisotropic_field), le
    // rhs reste f tel quel.
    if (!has_eps_field_ && !has_eps_xy_field_ && p_eps_ != Real(1)) {
      const Real inv = Real(1) / p_eps_;
      for (int li = 0; li < rhs.local_size(); ++li) {
        Array4 r = rhs.fab(li).array();
        const Box2D v = rhs.box(li);
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i) r(i, j, 0) *= inv;
      }
    }
    adc_sf_mark("solve_fields: avant ell_solve");
    ell_solve();
    adc_sf_mark("solve_fields: apres ell_solve, avant device_fence");
    device_fence();
    adc_sf_mark("solve_fields: apres device_fence (derivation aux)");
    const Real dx = owner_->geom.dx(), dy = owner_->geom.dy();
    // Derivation par cellule (phi, grad phi) -> canal aux : LOCALE au rang proprietaire. System
    // repartit UNE box (round-robin DistributionMapping(1, n_ranks())), donc a np>1 un seul rang la
    // possede ; les autres ont local_size()==0 et N'ONT PAS de fab a deriver. On itere sur les fabs
    // LOCAUX (jamais fab(0) en dur) : no-op sur un rang vide, identique a avant sur le proprietaire
    // (boucle executee une fois, bit-identique a np=1). ell_phi() et aux partagent la meme
    // DistributionMapping -> meme indexation locale.
    MultiFab& phi_mf = ell_phi();
    for (int li = 0; li < owner_->aux.local_size(); ++li) {
      const ConstArray4 p = phi_mf.fab(li).const_array();
      Array4 a = owner_->aux.fab(li).array();
      const Box2D v = owner_->aux.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
          a(i, j, 0) = p(i, j);
          a(i, j, 1) = (p(i + 1, j) - p(i - 1, j)) / (2 * dx);
          a(i, j, 2) = (p(i, j + 1) - p(i, j - 1)) / (2 * dy);
        }
    }
    adc_sf_mark("solve_fields: apres derivation aux (phi, grad phi)");
    apply_te();  // T_e = p/rho du bloc fluide source, recalculee a chaque solve (B_z, comp 3, preservee)
    adc_sf_mark("solve_fields: apres apply_te");
    if (owner_->periodic_)
      fill_boundary(owner_->aux, owner_->dom, owner_->per_);
    else
      fill_ghosts(owner_->aux, owner_->dom, owner_->bc_);  // extrapolation au bord (paroi / sortie libre)
    adc_sf_mark("solve_fields: fin (fill ghosts aux)");
  }

 private:
  Impl* owner_;
};

}  // namespace field_solver
}  // namespace adc
