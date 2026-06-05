#include <adc/runtime/system.hpp>

#include <adc/core/variables.hpp>  // VariableSet + VariableRole : descripteur a roles porte par chaque bloc
#include <adc/runtime/abi_key.hpp>  // adc::abi_key + detail::abi_key_string (frontiere ABI du loader natif)
#include <adc/runtime/block_builder.hpp>  // GridContext + make_block/make_max_speed (fermetures compilees)
#include <adc/runtime/model_factory.hpp>  // detail::dispatch_model + briques compilees
#include <adc/coupling/condensed_schur_source_stepper.hpp>  // etage source condense par Schur (adc.Split / CondensedSchur, #126)
#include <adc/coupling/coupled_source_program.hpp>  // CoupledSourceKernel : source couplee generique (DSL P5, bytecode)
#include <adc/numerics/elliptic/geometric_mg.hpp>
#include <adc/numerics/elliptic/poisson_fft_solver.hpp>
#include <adc/numerics/time/implicit_stepper.hpp>   // backward_euler_source
#include <adc/numerics/time/time_steppers.hpp>      // ForwardEuler, SSPRK2Step (math RK du coeur)
#include <adc/numerics/spatial_operator.hpp>     // assemble_rhs, SourceFreeModel, max_wave_speed_mf, load_state

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>  // sum
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>  // fill_ghosts, fill_boundary
#include <adc/runtime/dynamic_model.hpp>  // IModel : modele charge a l'execution (bloc dynamique)
#include <adc/runtime/native_loader.hpp>  // chargement .so (JIT/AOT/natif) + garde-fou ABI : VERBATIM, inclus apres la def de Impl ci-dessous (templates instancies plus bas)
#include <adc/runtime/wall_predicate.hpp>  // detail::wall_predicate (paroi partagee System/AmrSystem)

#include <algorithm>
#include <cmath>
#include <cstdio>   // ADC_TRACE_SOLVE_FIELDS : trace de diagnostic device (env-gate, inerte par defaut)
#include <cstdlib>  // getenv
#include <dlfcn.h>  // dlopen/dlsym : chargement d'une brique generee (.so)
#include <functional>
#include <limits>  // std::numeric_limits (CFL par bloc : dt = min sur les blocs)
#include <memory>
#include <optional>
#include <stdexcept>
#include <variant>

namespace adc {

// Trace de DIAGNOSTIC du chemin solve_fields (jalon #93, crash device GH200). Active SEULEMENT si la
// variable d'environnement ADC_TRACE_SOLVE_FIELDS est posee ; ecrit sur stderr avec flush immediat pour
// localiser le dernier marqueur avant un crash device. INERTE par defaut : aucun effet sur les sorties
// ni sur la numerique. Diagnostic CONSERVE (env-gate, inerte par defaut) : utile pour un futur
// crash device dans solve_fields.
namespace {
inline bool adc_trace_sf() {
  static const bool on = std::getenv("ADC_TRACE_SOLVE_FIELDS") != nullptr;
  return on;
}
inline void adc_sf_mark(const char* w) {
  if (adc_trace_sf()) { std::fprintf(stderr, "[sf] %s\n", w); std::fflush(stderr); }
}

// Resout le MASQUE IMPLICITE d'un bloc (cf. add_block : implicit_vars / implicit_roles) en une liste
// d'indices de composantes conservees, contre le descripteur du bloc @p cons. Le masque vit cote BLOC /
// politique temporelle (et NON le modele) : meme modele, traitements implicites distincts par bloc. Un
// nom ou un role absent du bloc leve une erreur EXPLICITE (pas d'ignore silencieux). Renvoie les indices
// UNIQUES, tries (l'ordre est sans importance pour le masque). VIDE en entree -> vide -> masque inactif.
inline std::vector<int> resolve_implicit_components(const std::string& block,
                                                    const VariableSet& cons,
                                                    const std::vector<std::string>& names,
                                                    const std::vector<std::string>& roles) {
  std::vector<int> out;
  auto push_unique = [&out](int c) {
    if (std::find(out.begin(), out.end(), c) == out.end()) out.push_back(c);
  };
  for (const std::string& nm : names) {
    int idx = -1;
    for (int i = 0; i < static_cast<int>(cons.names.size()); ++i)
      if (cons.names[i] == nm) { idx = i; break; }
    if (idx < 0) {
      std::string have;
      for (std::size_t i = 0; i < cons.names.size(); ++i) {
        if (i) have += ", ";
        have += cons.names[i];
      }
      throw std::runtime_error("System::add_block : implicit_vars : variable '" + nm +
                               "' absente du bloc '" + block + "' (variables conservees : " + have + ")");
    }
    push_unique(idx);
  }
  for (const std::string& rn : roles) {
    const VariableRole role = role_from_name(rn);
    const int idx = cons.index_of(role);
    if (role == VariableRole::Custom || idx < 0) {
      std::string have = roles_csv(cons);
      throw std::runtime_error("System::add_block : implicit_roles : role '" + rn +
                               "' absent du bloc '" + block + "' (roles : " +
                               (have.empty() ? std::string("<non renseignes>") : have) + ")");
    }
    push_unique(idx);
  }
  std::sort(out.begin(), out.end());
  return out;
}
}  // namespace

// Cle d'ABI du MODULE (figee a la compilation de cette TU). Definie ici pour que le module _adc
// l'exporte (ADC_EXPORT) : add_native_block la compare a la cle baked dans le loader .so.
std::string abi_key() { return detail::abi_key_string(); }

// Methode statique pratique (binding Python + add_native_block) : delegue a la cle libre du module.
std::string System::abi_key() { return adc::abi_key(); }

namespace {
// Indice de la composante portant @p role dans @p vs, ou @p fallback si le bloc ne renseigne
// pas ce role (bloc dynamique / compile : descripteur sans roles). Permet aux couplages de viser
// une composante par son SENS sans coder l'indice en dur, tout en restant retro-compatible.
int role_index(const VariableSet& vs, VariableRole role, int fallback) {
  const int c = vs.index_of(role);
  return c >= 0 ? c : fallback;
}
}  // namespace

struct System::Impl {
  // Fermetures compilees figees a l'ajout du bloc (modele compose + schema spatial + temps).
  // Type-erased SEULEMENT au niveau de la liste de blocs ; le noyau reste compile.
  struct Species {
    std::string name;
    MultiFab U;
    int ncomp;
    int substeps;                                             // sous-pas statiques (add_block)
    bool evolve;                                              // false = espece gelee (fond fixe non avance)
    int stride = 1;                                           // cadence : avance 1 fois tous les stride macro-pas
    double gamma;                                             // pour l'energie au repos (4 var)
    std::function<void(MultiFab&, Real, int)> advance;        // (U, dt, n) : n sous-pas de dt/n
    std::function<void(MultiFab&, MultiFab&)> rhs_into;        // R <- -div F + S (Poisson fige)
    std::function<Real(const MultiFab&)> max_speed;           // max |vitesse d'onde| du bloc
    std::function<void(const MultiFab&, MultiFab&)> add_poisson_rhs;  // += elliptic_rhs(U)
    // Descripteur des variables conservatives / primitives (noms + ROLES physiques) du bloc.
    // Les roles (fournis par M::conservative_vars()) permettent aux couplages inter-especes de
    // viser une composante par son SENS (qte de mvt, energie) au lieu d'un indice u[1]/u[3] en dur.
    VariableSet cons_vars, prim_vars;
    // Conversions PONCTUELLES cons <-> prim DU MODELE du bloc (une cellule, ncomp doubles in/out).
    // Posees a l'ajout (install_block / push_dynamic) depuis le modele reel ; vides -> identite (le
    // modele n'expose pas de conversion, p.ex. scalaire pur ou .so genere avant ce chantier).
    // Consommees par set_primitive_state / get_primitive_state (init/diagnostic en primitif).
    System::CellConvert prim_to_cons, cons_to_prim;
    // ETAGE SOURCE condense par Schur (OPT-IN, adc.Split(source=CondensedSchur), cf. set_source_stage).
    // nullptr (defaut) = pas d'etage source condense : le bloc avance EXACTEMENT comme avant
    // (bit-identique). Non nul = apres le transport hyperbolique, le pas joue l'etage source autonome
    // (CondensedSchurSourceStepper, #126) en lieu et place de la source explicite / IMEX. shared_ptr :
    // garde Species MOVABLE (le stepper porte un GeometricMG, ni copiable ni movable simplement).
    std::shared_ptr<CondensedSchurSourceStepper> schur;
    double schur_theta = 0.5;  // theta-schema de l'etage source (0.5 = Crank-Nicolson)
  };

  SystemConfig cfg;
  Geometry geom;
  BoxArray ba;
  DistributionMapping dm;
  BCRec bc_;        // CL transport (periodique ou Foextrap selon cfg.periodic)
  Box2D dom;
  Periodicity per_;
  bool periodic_;
  MultiFab aux;
  int aux_ncomp_ = kAuxBaseComps;     // largeur du canal aux PARTAGE (max des blocs ; >= 3)
  std::vector<Real> bz_field_;        // champ B_z(x) n*n row-major (vide si non fourni)
  int te_src_ = -1;                   // indice du bloc fluide source de T_e (-1 = aucune)
  std::vector<Species> sp;
  double t = 0;
  int macro_step_ = 0;  // compteur de macro-pas (0-indexe) : sert au filtre stride par bloc
  std::vector<std::function<void(Real)>> couplings;  // sources couplees inter-especes (splitting)

  // SEMANTIQUE STRIDE = HOLD-THEN-CATCH-UP (rattrapage en FIN de fenetre). Un bloc de cadence M est
  // TENU (non avance) sur les macro-pas ou (macro_step + 1) % M != 0, puis avance d'un pas effectif
  // M*dt au macro-pas ou (macro_step + 1) % M == 0, i.e. a la FIN de sa fenetre de M macro-pas. Au
  // macro-pas k, le temps du systeme est (k+1)*dt et le bloc qui RATTRAPE a alors avance du meme
  // (k+1)*dt cumule : il est temporellement COHERENT avec les blocs rapides, jamais "dans le futur".
  // (L'ancienne semantique avancait au DEBUT de fenetre, macro_step % M == 0 : a k=0 le bloc avancait
  // deja M*dt alors que le systeme n'avancait que dt -> bloc anticipe, couplage Poisson/source faux.)
  static bool stride_due(int macro_step, int stride) { return (macro_step + 1) % stride == 0; }

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

  explicit Impl(const SystemConfig& c)
      : cfg(c),
        geom{Box2D::from_extents(c.n, c.n), 0.0, c.L, 0.0, c.L},
        ba(std::vector<Box2D>{Box2D::from_extents(c.n, c.n)}),
        dm(1, n_ranks()),
        bc_(make_bc(c)),
        dom(Box2D::from_extents(c.n, c.n)),
        per_{c.periodic, c.periodic},
        periodic_(c.periodic),
        aux(ba, dm, kAuxBaseComps, 1) {}

  // Garantit une largeur aux >= ncomp (canal PARTAGE). Reallouer l'aux GARDE son adresse (membre :
  // les fermetures de bloc capturent &aux via grid_ctx) et re-applique B_z. No-op si deja assez large.
  void ensure_aux_width(int ncomp) {
    if (ncomp <= aux_ncomp_) return;
    aux_ncomp_ = ncomp;
    aux = MultiFab(ba, dm, aux_ncomp_, 1);
    apply_bz();
    apply_te();
  }

  // Peuple la composante B_z (indice kAuxBaseComps) du canal partage depuis bz_field_, sur les
  // cellules valides. No-op si B_z non fourni ou si aucun bloc ne le lit (largeur de base). Les
  // halos de B_z sont remplis par solve_fields (comme grad) ; field_postprocess n'ecrit que comp 0..2.
  void apply_bz() {
    if (bz_field_.empty() || aux_ncomp_ <= kAuxBaseComps) return;
    const int n = cfg.n;
    // Peuplement LOCAL au rang proprietaire (cf. solve_fields) : iteration sur les fabs locaux du
    // canal aux au lieu de fab(0) en dur (no-op sur un rang sans box locale a np>1, bit-identique au
    // proprietaire).
    for (int li = 0; li < aux.local_size(); ++li) {
      Array4 a = aux.fab(li).array();
      const Box2D v = aux.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          a(i, j, kAuxBaseComps) = bz_field_[static_cast<std::size_t>(j) * n + i];
    }
  }

  // Garantit que l'etat U du bloc @p name porte au moins @p ng ghosts (stencil du schema spatial).
  // WENO5 lit 3 ghosts, > les 2 alloues par defaut dans install_block ; sans cette largeur,
  // fill_ghosts + assemble_rhs liraient hors bornes (cf. AmrSystem qui alloue avec Limiter::n_ghost,
  // PR #22). Reallue le MultiFab et RECOPIE les cellules valides (set_density peut avoir precede) ;
  // no-op si U a deja assez de ghosts -> allocation et donnees bit-identiques a avant pour MUSCL.
  void set_block_ghosts(const std::string& name, int ng) {
    Species& s = find(name);
    if (s.U.n_grow() >= ng) return;
    MultiFab nu(s.U.box_array(), s.U.dmap(), s.ncomp, ng);
    nu.set_val(Real(0));
    for (int li = 0; li < s.U.local_size(); ++li) {
      const ConstArray4 old = s.U.fab(li).const_array();
      Array4 dst = nu.fab(li).array();
      const Box2D v = s.U.box(li);  // cellules valides (hors ghost) : copiees telles quelles
      for (int c = 0; c < s.ncomp; ++c)
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i) dst(i, j, c) = old(i, j, c);
    }
    s.U = std::move(nu);
  }

  // Composante canonique de T_e (apres phi/grad/B_z) ; cf. adc::Aux et AUX_CANONICAL cote DSL.
  static constexpr int kTeComp = kAuxBaseComps + 1;  // = 4

  // Peuple la composante T_e (temperature electronique) = p/rho du bloc fluide source te_src_.
  // RECALCULEE a chaque solve_fields (T_e varie avec le fluide, contrairement a B_z statique).
  // No-op si aucune source ou si aucun bloc ne lit T_e (largeur insuffisante). Le bloc source est
  // compressible (4 var) ; p = (gamma-1)(E - 0.5 rho|v|^2), T = p/rho.
  void apply_te() {
    if (te_src_ < 0 || aux_ncomp_ <= kTeComp) return;
    const Species& s = sp[static_cast<std::size_t>(te_src_)];
    const Real gm1 = Real(s.gamma) - Real(1);
    // Peuplement LOCAL au rang proprietaire (cf. solve_fields) : on itere sur les fabs locaux du
    // canal aux au lieu de fab(0) en dur (no-op sur un rang sans box locale a np>1, bit-identique au
    // proprietaire). s.U et aux partagent la meme DistributionMapping -> meme indexation locale.
    for (int li = 0; li < aux.local_size(); ++li) {
      const ConstArray4 us = s.U.fab(li).const_array();
      Array4 a = aux.fab(li).array();
      const Box2D v = aux.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
          const Real rho = us(i, j, 0), mx = us(i, j, 1), my = us(i, j, 2), E = us(i, j, 3);
          const Real p = gm1 * (E - Real(0.5) * (mx * mx + my * my) / rho);
          a(i, j, kTeComp) = p / rho;  // T = p / rho
        }
    }
  }

  static BCRec make_bc(const SystemConfig& c) {
    BCRec b;  // periodique par defaut
    if (!c.periodic) b.xlo = b.xhi = b.ylo = b.yhi = BCType::Foextrap;
    return b;
  }

  Species& find(const std::string& name) {
    for (auto& s : sp)
      if (s.name == name) return s;
    throw std::runtime_error("System : bloc inconnu '" + name + "'");
  }
  const Species& find(const std::string& name) const {
    for (auto& s : sp)
      if (s.name == name) return s;
    throw std::runtime_error("System : bloc inconnu '" + name + "'");
  }
  int index(const std::string& name) const {
    for (std::size_t k = 0; k < sp.size(); ++k)
      if (sp[k].name == name) return static_cast<int>(k);
    throw std::runtime_error("System : bloc inconnu '" + name + "'");
  }

  // Sources de COUPLAGE inter-especes : appliquees par SPLITTING (un pas additif explicite de dt)
  // APRES le transport de chaque bloc. Chaque couplage est un for_each_cell (kernel DEVICE) lisant /
  // mettant a jour plusieurs blocs au meme point ; ils s'ordonnent apres le transport sur le meme
  // espace d'execution, donc plus de device_fence prealable (plus d'acces hote).
  void apply_couplings(Real dt) {
    if (couplings.empty()) return;
    for (auto& c : couplings) c(dt);
  }

  // --- solveur elliptique (Poisson de systeme) -----------------------------
  BCRec poisson_bc() {
    std::string mode = p_bc;
    if (mode == "auto") mode = (p_wall == "circle" || !cfg.periodic) ? "dirichlet" : "periodic";
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
  std::function<bool(Real, Real)> wall_active() {
    return detail::wall_predicate(p_wall, p_wall_radius, cfg.L, "System::set_poisson");
  }
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
      ell_.emplace(std::in_place_type<PoissonFFTSolver>, geom, ba, pbc, active);
    } else if (p_solver == "geometric_mg") {
      ell_.emplace(std::in_place_type<GeometricMG>, geom, ba, pbc, std::move(active));
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
  // Installe le champ eps(x) (n*n row-major) sur le GeometricMG : l'operateur passe a
  // div(eps grad phi) = f, eps PORTE PAR L'OPERATEUR (coefficient de face harmonique, ordre 2),
  // sans mise a l'echelle 1/eps du second membre. Seul GeometricMG supporte ce coefficient variable.
  void apply_epsilon_field() {
    GeometricMG& mg = std::get<GeometricMG>(*ell_);
    MultiFab eps_fine(ba, dm, 1, 0);
    const int n = cfg.n;
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
  // Installe les champs eps_x(x), eps_y(x) (n*n row-major chacun) sur le GeometricMG : l'operateur
  // passe a div(diag(eps_x, eps_y) grad phi) = f. Les faces normales a x lisent eps_x, celles
  // normales a y lisent eps_y (coefficients de face harmoniques, ordre 2), PORTES PAR L'OPERATEUR
  // sans mise a l'echelle 1/eps du second membre. GeometricMG seul (coefficient tensoriel variable).
  void apply_epsilon_anisotropic_field() {
    GeometricMG& mg = std::get<GeometricMG>(*ell_);
    MultiFab eps_x_fine(ba, dm, 1, 0), eps_y_fine(ba, dm, 1, 0);
    const int n = cfg.n;
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
  // Installe le terme de reaction kappa(x) (n*n row-major) sur le GeometricMG : l'operateur passe a
  // div(eps grad phi) - kappa phi = f (Poisson ecrante / Helmholtz ; kappa = 1/lambda_D^2 pour Debye).
  // kappa est DIAGONAL (lu en cellule), restreint par moyenne aux niveaux grossiers. GeometricMG seul.
  void apply_reaction_field() {
    GeometricMG& mg = std::get<GeometricMG>(*ell_);
    MultiFab kappa_fine(ba, dm, 1, 0);
    const int n = cfg.n;
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
  MultiFab& ell_rhs() {
    return std::visit([](auto& e) -> MultiFab& { return e.rhs(); }, *ell_);
  }
  MultiFab& ell_phi() {
    return std::visit([](auto& e) -> MultiFab& { return e.phi(); }, *ell_);
  }
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

  // ETAGE SOURCE condense par Schur (OPT-IN, cf. set_source_stage). No-op si le bloc n'a pas d'etage
  // source (s.schur == nullptr) : le chemin par defaut reste BIT-IDENTIQUE. Sinon, APRES le transport
  // hyperbolique du bloc (deja joue par s.advance), on joue l'etage source AUTONOME
  // (CondensedSchurSourceStepper, #126) sur l'etat post-transport :
  //   - state = s.U (rho gelee dans la source, mom/E mis a jour) ;
  //   - phi    = le potentiel du Poisson de systeme (ell_phi(), warm start phi^n issu de solve_fields
  //              en tete de step) -- l'etage resout son PROPRE operateur condense et ECRIT phi^{n+1}
  //              dedans, il NE rappelle PAS solve_fields (pas de duplication) ;
  //   - B_z    = canal aux a l'indice kAuxBaseComps (peuple + ghosts remplis par solve_fields).
  // theta/dt du theta-schema ; dt = eff_dt (facteur stride deja inclus par l'appelant, comme s.advance).
  void run_source_stage(Species& s, Real eff_dt) {
    if (!s.schur) return;
    s.schur->step(s.U, ell_phi(), aux, kAuxBaseComps, static_cast<Real>(s.schur_theta), eff_dt);
  }

  // --- schemas spatiaux compiles -------------------------------------------
  // Evaluateur methode-des-lignes d'un bloc (L/F/Model figes) : ghosts puis R = -div F + S.
  // Construction des fermetures de bloc (avance + residu + Poisson) deplacee en en-tete
  // (adc/runtime/block_builder.hpp : make_block / make_max_speed / make_poisson_rhs) afin que le
  // chemin template de production soit instanciable hors de cette unite (compilation AOT d'un
  // modele genere). Ici on ne fournit que le contexte de grille a leur passer.
  GridContext grid_ctx() { return GridContext{dom, bc_, geom, &aux}; }

  void solve_fields() {
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
    for (auto& s : sp) s.add_poisson_rhs(s.U, rhs);
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
    const Real dx = geom.dx(), dy = geom.dy();
    // Derivation par cellule (phi, grad phi) -> canal aux : LOCALE au rang proprietaire. System
    // repartit UNE box (round-robin DistributionMapping(1, n_ranks())), donc a np>1 un seul rang la
    // possede ; les autres ont local_size()==0 et N'ONT PAS de fab a deriver. On itere sur les fabs
    // LOCAUX (jamais fab(0) en dur) : no-op sur un rang vide, identique a avant sur le proprietaire
    // (boucle executee une fois, bit-identique a np=1). ell_phi() et aux partagent la meme
    // DistributionMapping -> meme indexation locale.
    MultiFab& phi_mf = ell_phi();
    for (int li = 0; li < aux.local_size(); ++li) {
      const ConstArray4 p = phi_mf.fab(li).const_array();
      Array4 a = aux.fab(li).array();
      const Box2D v = aux.box(li);
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
    if (periodic_)
      fill_boundary(aux, dom, per_);
    else
      fill_ghosts(aux, dom, bc_);  // extrapolation au bord (paroi / sortie libre)
    adc_sf_mark("solve_fields: fin (fill ghosts aux)");
  }

  std::vector<double> copy_comp0(const MultiFab& mf) const {
    device_fence();
    const ConstArray4 u = mf.fab(0).const_array();
    const Box2D v = mf.box(0);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(u(i, j, 0));
    return out;
  }
  std::vector<double> copy_state(const MultiFab& mf, int ncomp) const {
    device_fence();
    const ConstArray4 u = mf.fab(0).const_array();
    const Box2D v = mf.box(0);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(ncomp) * v.nx() * v.ny());
    for (int c = 0; c < ncomp; ++c)
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(u(i, j, c));
    return out;
  }
  void write_state(MultiFab& mf, int ncomp, const std::vector<double>& in) {
    const Box2D v = mf.box(0);
    const std::size_t need = static_cast<std::size_t>(ncomp) * v.nx() * v.ny();
    if (in.size() != need)
      throw std::runtime_error("System::set_state : taille != ncomp*n*n");
    Array4 u = mf.fab(0).array();
    std::size_t k = 0;
    for (int c = 0; c < ncomp; ++c)
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) u(i, j, c) = in[k++];
  }

  // push_dynamic<NV> (bloc DYNAMIQUE IModel<NV> charge depuis un .so) a ete EXTRAIT VERBATIM vers
  // adc::native_loader::push_dynamic (include/adc/runtime/native_loader.hpp, template sur Impl) ;
  // add_dynamic_block plus bas l'instancie avec System::Impl. Voir SYSTEM_CPP_EXTRACTION_PLAN.md.
};

namespace {
// Garde-fou geometrie (chantier "grille polaire", Phase 1). Le CHOIX de geometrie est porte par la
// config (adc.CartesianMesh / adc.PolarMesh). "cartesian" : chemin historique, bit-identique. "polar" :
// la geometrie annulaire + l'operateur de transport polaire (assemble_rhs_polar) + sa validation MMS
// sont livres par cette phase, MAIS le transport polaire A TRAVERS System::step n'est PAS encore cable
// (il demanderait aussi le Poisson polaire, hors scope Phase 1). On REFUSE donc explicitement un System
// polaire au lieu de faire tourner SILENCIEUSEMENT la numerique cartesienne sur une config polaire
// (ce qui serait un piege). Tout autre token est une erreur. Cartesien : aucun effet (chemin inchange).
void check_geometry(const SystemConfig& c) {
  if (c.geometry == "cartesian") return;
  if (c.geometry == "polar")
    throw std::runtime_error(
        "System : geometry='polar' (adc.PolarMesh) n'est pas encore branche dans System.step (Phase 1 "
        "livre la grille annulaire PolarGeometry, l'operateur de transport polaire assemble_rhs_polar "
        "et sa validation MMS ; le transport polaire via System, qui demande aussi le Poisson polaire, "
        "est une phase ulterieure). Utiliser adc.CartesianMesh (defaut) ou l'operateur polaire en C++.");
  throw std::runtime_error("System : geometry '" + c.geometry +
                           "' inconnu (cartesian | polar) ; cf. adc.CartesianMesh / adc.PolarMesh");
}
}  // namespace

System::System(const SystemConfig& c) : p_(std::make_unique<Impl>(c)) { check_geometry(c); }
System::~System() = default;
System::System(System&&) noexcept = default;
System& System::operator=(System&&) noexcept = default;

void System::add_block(const std::string& name, const ModelSpec& model,
                       const std::string& limiter, const std::string& riemann,
                       const std::string& recon, const std::string& time, int substeps,
                       bool evolve, int stride, const std::vector<std::string>& implicit_vars,
                       const std::vector<std::string>& implicit_roles) {
  Impl* P = p_.get();
  if (substeps < 1) throw std::runtime_error("System::add_block : substeps >= 1");
  if (stride < 1) throw std::runtime_error("System::add_block : stride >= 1");
  // @p time porte le TRAITEMENT et, en explicite, le SCHEMA RK : "explicit"/"ssprk2" = SSPRK2
  // (defaut historique), "ssprk3" = SSPRK3 (ordre 3), "imex" = transport explicite + source raide
  // implicite. La math RK reste un FONCTEUR du coeur (build_block).
  if (time != "explicit" && time != "ssprk2" && time != "ssprk3" && time != "imex")
    throw std::runtime_error("System::add_block : time 'explicit'|'ssprk2'|'ssprk3'|'imex' (recu '" +
                             time + "')");
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error("System::add_block : recon 'conservative' | 'primitive' (recu '" +
                             recon + "')");
  const bool imex = (time == "imex");
  const bool recon_prim = (recon == "primitive");
  const std::string method = (time == "ssprk3") ? "ssprk3" : "ssprk2";
  // Le masque implicite (implicit_vars / implicit_roles) ne s'applique qu'au pas de source IMEX. Le
  // demander en explicite est une ERREUR (pas d'ignore silencieux) : l'explicite n'a pas de pas implicite.
  if (!imex && (!implicit_vars.empty() || !implicit_roles.empty()))
    throw std::runtime_error("System::add_block : implicit_vars / implicit_roles exigent time='imex' "
                             "(le masque implicite ne s'applique qu'au pas de source IMEX ; recu time='" +
                             time + "')");

  int ncomp = 1;
  BlockClosures clo;
  std::function<Real(const MultiFab&)> max_speed;
  std::function<void(const MultiFab&, MultiFab&)> add_poisson_rhs;
  CellConvert prim_to_cons, cons_to_prim;  // conversions ponctuelles du modele (set/get_primitive_state)
  VariableSet cons_vs, prim_vs;
  const GridContext ctx = P->grid_ctx();
  // Le modele est compose a partir des briques designees par la spec ; le visiteur cable les
  // fermetures (constructeurs en en-tete, instanciables AOT). ncomp = n_vars du modele compose ;
  // set_density s'y adapte. Les noms de variables viennent du descripteur Variables porte par le
  // modele (brique Vars), source unique de verite.
  detail::dispatch_model(model, [&](auto m) {
    using M = decltype(m);
    ncomp = M::n_vars;
    cons_vs = M::conservative_vars();  // noms + ROLES physiques (source unique de verite)
    prim_vs = M::primitive_vars();
    // Masque implicite PORTE PAR LE BLOC : resout noms/roles -> indices contre le descripteur du bloc
    // (erreur explicite sur un nom/role absent). Vide -> make_implicit_mask inactif -> defaut modele
    // (bit-identique). Ne joue qu'en IMEX (garde ci-dessus pour l'explicite).
    const std::vector<int> impl_components =
        resolve_implicit_components(name, cons_vs, implicit_vars, implicit_roles);
    clo = make_block(m, limiter, riemann, ctx, imex, recon_prim, method, impl_components);
    max_speed = make_max_speed(m, ctx);
    add_poisson_rhs = make_poisson_rhs(m);
    // Conversions cons <-> prim DU MODELE (set/get_primitive_state) : memes formules que le flux/CFL.
    auto conv = make_cell_convert(m);
    prim_to_cons = std::move(conv.first);
    cons_to_prim = std::move(conv.second);
  });
  // Installation commune (meme chemin que add_compiled_model pour un modele genere par le DSL) :
  // les fermetures tournent sur les MultiFab REELS du System (halos MPI via fill_boundary, device
  // via Kokkos), sans recopie.
  install_block(name, ncomp, cons_vs, prim_vs, model.gamma, std::move(clo), std::move(max_speed),
                std::move(add_poisson_rhs), substeps, evolve, stride);
  set_block_conversion(name, std::move(prim_to_cons), std::move(cons_to_prim));
  // GHOSTS du schema : WENO5 lit un stencil 5 points (3 ghosts) > les 2 alloues par defaut dans
  // install_block. On reallue l'etat du bloc avec block_n_ghost(limiter) si besoin (cf. AmrSystem qui
  // alloue avec Limiter::n_ghost, PR #22) pour que fill_ghosts + assemble_rhs ne lisent pas hors
  // bornes. minmod/vanleer (2 ghosts) : no-op, allocation et resultat bit-identiques a avant.
  P->set_block_ghosts(name, block_n_ghost(limiter));
}

// Contexte de grille reel (maillage + CL + aux) : sert au gabarit add_compiled_model pour fabriquer
// les fermetures d'un modele compile AOT sur les vrais champs du System (parite native, sans marshaling).
GridContext System::grid_context() { return p_->grid_ctx(); }

// Installe un bloc a partir de fermetures deja fabriquees (par dispatch_model cote add_block, ou par
// block_builder cote add_compiled_model). Centralise la creation de l'espece (U, noms, schema).
void System::install_block(const std::string& name, int ncomp,
                           const VariableSet& cons_vars, const VariableSet& prim_vars, double gamma,
                           BlockClosures closures, std::function<Real(const MultiFab&)> max_speed,
                           std::function<void(const MultiFab&, MultiFab&)> poisson_rhs,
                           int substeps, bool evolve, int stride) {
  if (stride < 1) throw std::runtime_error("System::install_block : stride >= 1");
  Impl* P = p_.get();
  P->sp.push_back(Impl::Species{name, MultiFab(P->ba, P->dm, ncomp, 2), ncomp, substeps, evolve,
                                stride, gamma, std::move(closures.advance),
                                std::move(closures.rhs_into), std::move(max_speed),
                                std::move(poisson_rhs)});
  P->sp.back().U.set_val(Real(0));
  P->sp.back().cons_vars = cons_vars;
  P->sp.back().prim_vars = prim_vars;
}

// Reallocation width-aware de l'etat d'un bloc (delegue a Impl::set_block_ghosts). Exposee
// (ADC_EXPORT) pour que le gabarit en-tete add_compiled_model (chemin natif, loader .so) puisse
// elargir le bloc compile a block_n_ghost(limiter) -- 3 pour weno5 -- comme le fait add_block.
void System::set_block_ghosts(const std::string& name, int n_ghost) {
  p_->set_block_ghosts(name, n_ghost);
}

// Corps EXTRAIT VERBATIM vers adc::native_loader::add_dynamic_block (native_loader.hpp) ; instancie
// ici avec System::Impl (defini ci-dessus, prive a cette TU). Bit-identique : delegation pure.
void System::add_dynamic_block(const std::string& name, const std::string& so_path, int substeps,
                               const std::vector<std::string>& names, const std::string& recon) {
  native_loader::add_dynamic_block(this, p_.get(), name, so_path, substeps, names, recon);
}

// Corps EXTRAIT VERBATIM vers adc::native_loader::add_compiled_block (native_loader.hpp) ; instancie
// ici avec System::Impl. Bit-identique : delegation pure.
void System::add_compiled_block(const std::string& name, const std::string& so_path,
                                const std::string& limiter, const std::string& riemann,
                                const std::string& recon, const std::string& time, int substeps,
                                const std::vector<std::string>& names) {
  native_loader::add_compiled_block(this, p_.get(), name, so_path, limiter, riemann, recon, time,
                                    substeps, names);
}

// Corps EXTRAIT VERBATIM vers adc::native_loader::add_native_block (native_loader.hpp) ; instancie
// ici avec System::Impl. Bit-identique : delegation pure (this marshale au loader natif inchange).
void System::add_native_block(const std::string& name, const std::string& so_path,
                              const std::string& limiter, const std::string& riemann,
                              const std::string& recon, const std::string& time, double gamma,
                              int substeps, bool evolve, int stride) {
  native_loader::add_native_block(this, p_.get(), name, so_path, limiter, riemann, recon, time,
                                  gamma, substeps, evolve, stride);
}

void System::set_poisson(const std::string& rhs, const std::string& solver,
                         const std::string& bc, const std::string& wall, double wall_radius,
                         double epsilon) {
  if (epsilon == 0.0) throw std::runtime_error("System::set_poisson : epsilon != 0 requis");
  p_->p_rhs = rhs;
  p_->p_solver = solver;
  p_->p_bc = bc;
  p_->p_wall = wall;
  p_->p_wall_radius = wall_radius;
  p_->p_eps_ = static_cast<Real>(epsilon);
  p_->ell_.reset();
}

void System::set_epsilon_field(const std::vector<double>& eps) {
  const int n = p_->cfg.n;
  if (static_cast<int>(eps.size()) != n * n)
    throw std::runtime_error("System::set_epsilon_field : taille != n*n");
  for (double e : eps)
    if (!(e > 0.0))
      throw std::runtime_error("System::set_epsilon_field : permittivite eps(x) > 0 requise");
  p_->p_eps_field_ = eps;
  p_->has_eps_field_ = true;
  p_->ell_.reset();  // l'operateur sera reconstruit avec le champ eps au prochain solve_fields
}

void System::set_epsilon_anisotropic_field(const std::vector<double>& eps_x,
                                           const std::vector<double>& eps_y) {
  const int n = p_->cfg.n;
  if (static_cast<int>(eps_x.size()) != n * n || static_cast<int>(eps_y.size()) != n * n)
    throw std::runtime_error("System::set_epsilon_anisotropic_field : taille != n*n (eps_x et eps_y)");
  for (double e : eps_x)
    if (!(e > 0.0))
      throw std::runtime_error("System::set_epsilon_anisotropic_field : permittivite eps_x(x) > 0 requise");
  for (double e : eps_y)
    if (!(e > 0.0))
      throw std::runtime_error("System::set_epsilon_anisotropic_field : permittivite eps_y(x) > 0 requise");
  p_->p_eps_x_field_ = eps_x;
  p_->p_eps_y_field_ = eps_y;
  p_->has_eps_xy_field_ = true;
  p_->ell_.reset();  // operateur reconstruit en div(diag(eps_x, eps_y) grad phi) au prochain solve_fields
}

void System::set_reaction_field(const std::vector<double>& kappa) {
  const int n = p_->cfg.n;
  if (static_cast<int>(kappa.size()) != n * n)
    throw std::runtime_error("System::set_reaction_field : taille != n*n");
  for (double k : kappa)
    if (!(k >= 0.0))
      throw std::runtime_error("System::set_reaction_field : terme de reaction kappa(x) >= 0 requis "
                               "(operateur elliptique bien pose et multigrille convergente)");
  p_->p_kappa_field_ = kappa;
  p_->has_kappa_field_ = true;
  p_->ell_.reset();  // operateur reconstruit avec - kappa phi au prochain solve_fields
}

void System::ensure_aux_width(int ncomp) { p_->ensure_aux_width(ncomp); }

void System::set_magnetic_field(const std::vector<double>& bz) {
  const int n = p_->cfg.n;
  if (static_cast<int>(bz.size()) != n * n)
    throw std::runtime_error("System::set_magnetic_field : taille != n*n");
  p_->bz_field_.assign(bz.begin(), bz.end());
  p_->apply_bz();  // applique tout de suite si un bloc lit deja B_z ; sinon conserve pour ensure_aux_width
}

void System::set_electron_temperature_from(const std::string& name) {
  const int idx = p_->index(name);  // leve si bloc inconnu
  if (p_->sp[static_cast<std::size_t>(idx)].ncomp != 4)
    throw std::runtime_error("System::set_electron_temperature_from : le bloc '" + name +
                             "' doit etre compressible (4 var : rho, rho u, rho v, E) pour T = p/rho");
  p_->te_src_ = idx;
  // T_e (comp canonique 4) DERIVE : recalcule a chaque solve_fields. Inerte tant qu'aucun bloc ne
  // lit T_e (n_aux=5 -> ensure_aux_width(5)), comme set_magnetic_field pour B_z.
  p_->apply_te();
}

void System::add_ionization(const std::string& electron, const std::string& ion,
                            const std::string& neutral, double rate) {
  Impl* P = p_.get();
  const int ie = P->index(electron), ii = P->index(ion), ig = P->index(neutral);
  const Real k = static_cast<Real>(rate);
  // Densite resolue par ROLE (comme add_collision / add_thermal_exchange), fallback comp 0 si le
  // bloc ne renseigne pas ses roles (bloc dynamique / compile). Un bloc rangeant sa densite ailleurs
  // que l'indice 0 reste correctement couple.
  const int de = role_index(P->sp[ie].cons_vars, VariableRole::Density, 0);
  const int di = role_index(P->sp[ii].cons_vars, VariableRole::Density, 0);
  const int dg = role_index(P->sp[ig].cons_vars, VariableRole::Density, 0);
  // Ionisation (operator-split, sur la densite) : taux r = k n_e n_g. Un neutre disparait, un ion et
  // un electron apparaissent : n_g -= dt r, n_i += dt r, n_e += dt r. La masse est transferee du
  // neutre vers l'ion (n_i + n_g conserve). Premiere brique de couplage ; le transfert de quantite
  // de mouvement / energie (especes fluides) est un raffinement ulterieur.
  P->couplings.push_back([P, ie, ii, ig, k, de, di, dg](Real dt) {
    Array4 ue = P->sp[ie].U.fab(0).array();
    Array4 ui = P->sp[ii].U.fab(0).array();
    Array4 ug = P->sp[ig].U.fab(0).array();
    for_each_cell(P->sp[ie].U.box(0), [=] ADC_HD(int i, int j) {  // sur device (lit n_e, n_g)
      const Real dn = dt * k * ue(i, j, de) * ug(i, j, dg);
      ug(i, j, dg) -= dn;
      ui(i, j, di) += dn;
      ue(i, j, de) += dn;
    });
  });
}

void System::add_collision(const std::string& a, const std::string& b, double rate) {
  Impl* P = p_.get();
  const int ia = P->index(a), ib = P->index(b);
  if (P->sp[ia].ncomp < 3 || P->sp[ib].ncomp < 3)
    throw std::runtime_error("System::add_collision : les deux blocs doivent porter une quantite "
                             "de mouvement (transport fluide >= 3 variables)");
  const Real k = static_cast<Real>(rate);
  // Composantes resolues par ROLE (qte de mvt x/y, densite) plutot que par indice litteral : un
  // bloc qui range ses variables autrement reste correctement couple. Fallback aux indices
  // historiques (1, 2, 0) si le bloc ne renseigne pas ses roles (bloc dynamique / compile).
  const VariableSet& va_set = P->sp[ia].cons_vars;
  const VariableSet& vb_set = P->sp[ib].cons_vars;
  const int mxa = role_index(va_set, VariableRole::MomentumX, 1);
  const int mya = role_index(va_set, VariableRole::MomentumY, 2);
  const int da = role_index(va_set, VariableRole::Density, 0);
  const int mxb = role_index(vb_set, VariableRole::MomentumX, 1);
  const int myb = role_index(vb_set, VariableRole::MomentumY, 2);
  const int db = role_index(vb_set, VariableRole::Density, 0);
  // Friction inter-especes (operator-split) : force F = k (u_a - u_b) sur la quantite de
  // mouvement, opposee sur chaque espece (qte de mvt totale conservee) ; les vitesses relaxent
  // l'une vers l'autre. L'echauffement par friction (energie) est un raffinement ulterieur
  // (neglige : convient aux especes isothermes, sans eq. d'energie).
  P->couplings.push_back([P, ia, ib, k, mxa, mya, da, mxb, myb, db](Real dt) {
    Array4 ua = P->sp[ia].U.fab(0).array();
    Array4 ub = P->sp[ib].U.fab(0).array();
    for_each_cell(P->sp[ia].U.box(0), [=] ADC_HD(int i, int j) {  // sur device
      const Real fx = dt * k * (ua(i, j, mxa) / ua(i, j, da) - ub(i, j, mxb) / ub(i, j, db));
      ua(i, j, mxa) -= fx; ub(i, j, mxb) += fx;
      const Real fy = dt * k * (ua(i, j, mya) / ua(i, j, da) - ub(i, j, myb) / ub(i, j, db));
      ua(i, j, mya) -= fy; ub(i, j, myb) += fy;
    });
  });
}

void System::add_thermal_exchange(const std::string& a, const std::string& b, double rate) {
  Impl* P = p_.get();
  const int ia = P->index(a), ib = P->index(b);
  if (P->sp[ia].ncomp != 4 || P->sp[ib].ncomp != 4)
    throw std::runtime_error("System::add_thermal_exchange : les deux blocs doivent porter une "
                             "energie (Euler compressible, 4 variables)");
  const Real k = static_cast<Real>(rate);
  const Real ga = static_cast<Real>(P->sp[ia].gamma), gb = static_cast<Real>(P->sp[ib].gamma);
  // Composantes resolues par ROLE (energie, qte de mvt x/y, densite) plutot que par indice litteral.
  // Fallback aux indices historiques (3, 1, 2, 0) si le bloc ne renseigne pas ses roles.
  const VariableSet& va_set = P->sp[ia].cons_vars;
  const VariableSet& vb_set = P->sp[ib].cons_vars;
  const int ea = role_index(va_set, VariableRole::Energy, 3);
  const int mxa = role_index(va_set, VariableRole::MomentumX, 1);
  const int mya = role_index(va_set, VariableRole::MomentumY, 2);
  const int da = role_index(va_set, VariableRole::Density, 0);
  const int eb = role_index(vb_set, VariableRole::Energy, 3);
  const int mxb = role_index(vb_set, VariableRole::MomentumX, 1);
  const int myb = role_index(vb_set, VariableRole::MomentumY, 2);
  const int db = role_index(vb_set, VariableRole::Density, 0);
  // Echange thermique (operator-split) : flux de chaleur q = k (T_a - T_b) sur l'energie, oppose
  // sur chaque espece (energie totale conservee) ; les temperatures relaxent. T = p/rho (a une
  // constante pres), p = (gamma-1)(E - 1/2 rho |u|^2). Transfere l'energie INTERNE (u inchange).
  P->couplings.push_back([P, ia, ib, k, ga, gb, ea, mxa, mya, da, eb, mxb, myb, db](Real dt) {
    Array4 ua = P->sp[ia].U.fab(0).array();
    Array4 ub = P->sp[ib].U.fab(0).array();
    for_each_cell(P->sp[ia].U.box(0), [=] ADC_HD(int i, int j) {  // sur device
      const Real ra = ua(i, j, da), rb = ub(i, j, db);
      const Real pa = (ga - Real(1)) * (ua(i, j, ea) -
          Real(0.5) * (ua(i, j, mxa) * ua(i, j, mxa) + ua(i, j, mya) * ua(i, j, mya)) / ra);
      const Real pb = (gb - Real(1)) * (ub(i, j, eb) -
          Real(0.5) * (ub(i, j, mxb) * ub(i, j, mxb) + ub(i, j, myb) * ub(i, j, myb)) / rb);
      const Real q = dt * k * (pa / ra - pb / rb);  // k (T_a - T_b), T = p/rho
      ua(i, j, ea) -= q;
      ub(i, j, eb) += q;
    });
  });
}

void System::add_coupled_source(const std::vector<std::string>& in_blocks,
                                const std::vector<std::string>& in_roles,
                                const std::vector<double>& consts,
                                const std::vector<std::string>& out_blocks,
                                const std::vector<std::string>& out_roles,
                                const std::vector<int>& prog_ops,
                                const std::vector<int>& prog_args,
                                const std::vector<int>& prog_lens) {
  Impl* P = p_.get();
  const int n_in = static_cast<int>(in_blocks.size());
  const int n_const = static_cast<int>(consts.size());
  const int n_terms = static_cast<int>(out_blocks.size());
  // --- validation de forme (avant tout pas, erreurs EXPLICITES) ------------------------------------
  if (n_terms == 0)
    throw std::runtime_error("System::add_coupled_source : aucun terme de source (out_blocks vide)");
  if (static_cast<int>(in_roles.size()) != n_in)
    throw std::runtime_error("System::add_coupled_source : in_blocks / in_roles de tailles differentes");
  if (static_cast<int>(out_roles.size()) != n_terms || static_cast<int>(prog_lens.size()) != n_terms)
    throw std::runtime_error("System::add_coupled_source : out_blocks / out_roles / prog_lens de tailles "
                             "differentes");
  if (prog_ops.size() != prog_args.size())
    throw std::runtime_error("System::add_coupled_source : prog_ops / prog_args de tailles differentes");
  if (n_in + n_const > kCsMaxReg)
    throw std::runtime_error("System::add_coupled_source : trop de registres (entrees + constantes > " +
                             std::to_string(kCsMaxReg) + ")");
  if (n_terms > kCsMaxTerms)
    throw std::runtime_error("System::add_coupled_source : trop de termes de source (> " +
                             std::to_string(kCsMaxTerms) + ")");
  // Resout role -> composante par le descripteur CONSERVATIF du bloc (comme add_collision) ; fallback
  // comp 0 si le bloc ne renseigne pas le role. Un bloc inconnu leve via P->index().
  auto resolve = [&](const std::string& block, const std::string& role) -> std::pair<int, int> {
    const int sidx = P->index(block);  // leve si bloc inconnu
    const VariableRole r = role_from_name(role);
    if (r == VariableRole::Custom)
      throw std::runtime_error("System::add_coupled_source : role '" + role + "' inconnu (bloc '" +
                               block + "')");
    const int comp = role_index(P->sp[static_cast<std::size_t>(sidx)].cons_vars, r, 0);
    return {sidx, comp};
  };
  // Entrees : (espece, composante) lues par cellule. Capturees par INDICE (les fabs peuvent etre
  // reallouees entre l'enregistrement et l'application : on reconstruit les Array4 a CHAQUE pas).
  struct InRef { int sidx, comp; };
  std::vector<InRef> ins(static_cast<std::size_t>(n_in));
  for (int c = 0; c < n_in; ++c) {
    auto [s, comp] = resolve(in_blocks[static_cast<std::size_t>(c)], in_roles[static_cast<std::size_t>(c)]);
    ins[static_cast<std::size_t>(c)] = {s, comp};
  }
  struct OutRef { int sidx, comp; CsProgram prog; };
  std::vector<OutRef> outs(static_cast<std::size_t>(n_terms));
  int off = 0;
  for (int t = 0; t < n_terms; ++t) {
    auto [s, comp] = resolve(out_blocks[static_cast<std::size_t>(t)], out_roles[static_cast<std::size_t>(t)]);
    const int len = prog_lens[static_cast<std::size_t>(t)];
    if (len < 0 || len > kCsMaxProg)
      throw std::runtime_error("System::add_coupled_source : programme du terme " + std::to_string(t) +
                               " trop long (> " + std::to_string(kCsMaxProg) + ")");
    if (off + len > static_cast<int>(prog_ops.size()))
      throw std::runtime_error("System::add_coupled_source : prog_lens incoherent avec prog_ops");
    CsProgram pg;
    pg.len = len;
    for (int k = 0; k < len; ++k) {
      const int opc = prog_ops[static_cast<std::size_t>(off + k)];
      const int a = prog_args[static_cast<std::size_t>(off + k)];
      if (opc < 0 || opc > static_cast<int>(CsOp::Sqrt))
        throw std::runtime_error("System::add_coupled_source : opcode invalide");
      if (opc == static_cast<int>(CsOp::PushReg) && (a < 0 || a >= n_in + n_const))
        throw std::runtime_error("System::add_coupled_source : registre hors bornes dans le programme");
      pg.op[k] = opc;
      pg.arg[k] = a;
    }
    outs[static_cast<std::size_t>(t)] = {s, comp, pg};
    off += len;
  }
  // Toutes les especes touchees (entrees + sorties) partagent la DistributionMapping du System (une box
  // repartie en round-robin), donc meme local_size() et meme indexation locale -> on iterait en parallele
  // sur les fabs locaux. Conversion en valeurs CAPTUREES (pas de reference a 'this' du lambda C++).
  std::vector<Real> kconsts(consts.begin(), consts.end());
  P->couplings.push_back(
      [P, ins, outs, kconsts, n_in, n_const, n_terms](Real dt) {
        // MPI-safe : iteration sur les fabs LOCAUX du premier bloc d'entree (ou de sortie si aucune
        // entree). local_size()==0 sur un rang sans box -> boucle vide, no-op (pas de fab(0) en dur).
        const int sref = n_in > 0 ? ins[0].sidx : outs[0].sidx;
        MultiFab& Uref = P->sp[static_cast<std::size_t>(sref)].U;
        for (int li = 0; li < Uref.local_size(); ++li) {
          CoupledSourceKernel kern;
          kern.dt = dt;
          kern.n_in = n_in;
          kern.n_const = n_const;
          kern.n_terms = n_terms;
          for (int c = 0; c < n_in; ++c) {
            kern.in[c] = P->sp[static_cast<std::size_t>(ins[static_cast<std::size_t>(c)].sidx)].U.fab(li).array();
            kern.in_comp[c] = ins[static_cast<std::size_t>(c)].comp;
          }
          for (int c = 0; c < n_const; ++c) kern.consts[c] = kconsts[static_cast<std::size_t>(c)];
          for (int t = 0; t < n_terms; ++t) {
            kern.out[t] = P->sp[static_cast<std::size_t>(outs[static_cast<std::size_t>(t)].sidx)].U.fab(li).array();
            kern.out_comp[t] = outs[static_cast<std::size_t>(t)].comp;
            kern.prog[t] = outs[static_cast<std::size_t>(t)].prog;
          }
          for_each_cell(Uref.box(li), kern);  // foncteur NOMME (device-clean), additif forward-Euler
        }
      });
}

void System::set_source_stage(const std::string& name, const std::string& kind, double theta,
                              double alpha) {
  Impl* P = p_.get();
  Impl::Species& s = P->find(name);  // leve si bloc inconnu
  // SEUL kind cable pour l'instant : ElectrostaticLorentzCondensation (cf. CondensedSchurSourceStepper).
  // D'autres kind pourront s'ajouter sans toucher la facade (rejet explicite, pas d'ignore silencieux).
  if (kind != "electrostatic_lorentz")
    throw std::runtime_error("System::set_source_stage : kind '" + kind +
                             "' inconnu (seul 'electrostatic_lorentz' est supporte)");
  if (!(theta > 0.0 && theta <= 1.0))
    throw std::runtime_error("System::set_source_stage : theta doit etre dans (0, 1] (recu " +
                             std::to_string(theta) + ")");
  // Geometrie cartesienne uniquement (le branchement polaire de l'etage condense est un chantier
  // ulterieur, cf. docs/SCHUR_CONDENSATION_DESIGN.md section 7) ; System lui-meme rejette deja polar.
  if (P->cfg.geometry != "cartesian")
    throw std::runtime_error("System::set_source_stage : etage source condense supporte uniquement la "
                             "geometrie cartesienne (recu '" + P->cfg.geometry + "')");
  // CONTRAT roles : le bloc doit exposer Density / MomentumX / MomentumY (Energy optionnel). On lit le
  // descripteur CONSERVATIF du bloc (peuple par add_block / les .so a roles, dont le DSL compile qui
  // declare les electrons en roles). Un role requis absent leve une erreur EXPLICITE ICI (avant le pas)
  // -- le constructeur du stepper la leverait aussi, mais on diagnostique cote bloc nomme.
  const VariableSet& vs = s.cons_vars;
  auto require_role = [&](VariableRole role, const char* label) {
    if (vs.index_of(role) < 0)
      throw std::runtime_error(
          "System::set_source_stage : le bloc '" + name + "' n'expose pas le role " + label +
          " requis par adc.CondensedSchur (le modele doit declarer Density / MomentumX / MomentumY ; "
          "Energy optionnel). Verifier les roles du modele (variable_roles).");
  };
  require_role(VariableRole::Density, "Density");
  require_role(VariableRole::MomentumX, "MomentumX");
  require_role(VariableRole::MomentumY, "MomentumY");
  // B_z OBLIGATOIRE : l'etage de Lorentz lit Omega = B_z. On exige set_magnetic_field appele
  // (bz_field_ renseigne) et on elargit le canal aux au canal B_z (kAuxBaseComps) pour que apply_bz le
  // peuple et que solve_fields en remplisse les ghosts. Un B_z absent leve une erreur EXPLICITE.
  if (P->bz_field_.empty())
    throw std::runtime_error(
        "System::set_source_stage : le bloc '" + name + "' n'a pas de champ B_z (aux Omega) ; "
        "adc.CondensedSchur exige set_magnetic_field(B_z) (le terme de Lorentz lit Omega = B_z).");
  P->ensure_aux_width(kAuxBaseComps + 1);  // garantit le canal B_z dans l'aux partage + re-applique B_z
  // Construit l'etage source condense sur le layout REEL du System (ba/dm/geom) avec la CL du Poisson.
  // Le stepper alloue ses tampons UNE fois ; step() les reutilise (cf. son cycle de vie). alpha =
  // constante de couplage electrostatique du sous-systeme source. n_precond_vcycles = defaut (1).
  s.schur = std::make_shared<CondensedSchurSourceStepper>(vs, P->geom, P->ba, P->poisson_bc(),
                                                          static_cast<Real>(alpha));
  s.schur_theta = theta;
}

void System::set_density(const std::string& name, const std::vector<double>& rho) {
  Impl::Species& s = p_->find(name);
  const int n = p_->cfg.n;
  if (static_cast<int>(rho.size()) != n * n)
    throw std::runtime_error("System::set_density : taille != n*n");
  const Real gm1 = Real(s.gamma) - Real(1);
  Array4 u = s.U.fab(0).array();
  const Box2D v = s.U.box(0);
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
      const Real r = rho[static_cast<std::size_t>(j) * n + i];
      u(i, j, 0) = r;
      if (s.ncomp >= 3) { u(i, j, 1) = 0; u(i, j, 2) = 0; }  // qte de mvt au repos
      if (s.ncomp == 4) u(i, j, 3) = r / gm1;                // E = p/(g-1), p = rho
    }
}

void System::set_block_conversion(const std::string& name, CellConvert prim_to_cons,
                                  CellConvert cons_to_prim) {
  Impl::Species& s = p_->find(name);
  s.prim_to_cons = std::move(prim_to_cons);
  s.cons_to_prim = std::move(cons_to_prim);
}

void System::set_primitive_state(const std::string& name, const std::vector<double>& prim) {
  Impl::Species& s = p_->find(name);
  const int nc = s.ncomp;
  const std::size_t nn = static_cast<std::size_t>(p_->cfg.n) * p_->cfg.n;
  if (prim.size() != static_cast<std::size_t>(nc) * nn)
    throw std::runtime_error("System::set_primitive_state : taille != ncomp*n*n (bloc '" + name +
                             "' a " + std::to_string(nc) + " variables)");
  if (!s.prim_to_cons)
    throw std::runtime_error("System::set_primitive_state : le modele du bloc '" + name +
                             "' n'expose pas de conversion primitif -> conservatif (.so genere avant "
                             "ce chantier ?) ; utiliser set_state (etat conservatif direct)");
  // Conversion CELLULE PAR CELLULE via le modele du bloc : on lit les nc primitives composante-majeur
  // (prim[c*nn + k]) dans un petit tampon contigu, on convertit, et on ecrit les conservatives au
  // meme emplacement d'un tampon de sortie. Puis write_state pousse tout vers le MultiFab (chemin de
  // set_state, marshaling identique). Reutilise donc le marshaling existant (copy/write_state).
  std::vector<double> cons(prim.size());
  std::vector<double> cell_in(static_cast<std::size_t>(nc)), cell_out(static_cast<std::size_t>(nc));
  for (std::size_t k = 0; k < nn; ++k) {
    for (int c = 0; c < nc; ++c) cell_in[c] = prim[static_cast<std::size_t>(c) * nn + k];
    s.prim_to_cons(cell_in.data(), cell_out.data());
    for (int c = 0; c < nc; ++c) cons[static_cast<std::size_t>(c) * nn + k] = cell_out[c];
  }
  p_->write_state(s.U, nc, cons);
}

std::vector<double> System::get_primitive_state(const std::string& name) {
  Impl::Species& s = p_->find(name);
  const int nc = s.ncomp;
  const std::size_t nn = static_cast<std::size_t>(p_->cfg.n) * p_->cfg.n;
  if (!s.cons_to_prim)
    throw std::runtime_error("System::get_primitive_state : le modele du bloc '" + name +
                             "' n'expose pas de conversion conservatif -> primitif (.so genere avant "
                             "ce chantier ?) ; utiliser get_state (etat conservatif direct)");
  const std::vector<double> cons = p_->copy_state(s.U, nc);  // chemin de get_state (meme marshaling)
  std::vector<double> prim(cons.size());
  std::vector<double> cell_in(static_cast<std::size_t>(nc)), cell_out(static_cast<std::size_t>(nc));
  for (std::size_t k = 0; k < nn; ++k) {
    for (int c = 0; c < nc; ++c) cell_in[c] = cons[static_cast<std::size_t>(c) * nn + k];
    s.cons_to_prim(cell_in.data(), cell_out.data());
    for (int c = 0; c < nc; ++c) prim[static_cast<std::size_t>(c) * nn + k] = cell_out[c];
  }
  return prim;
}

void System::solve_fields() { p_->solve_fields(); }

void System::step(double dt) {
  Impl* P = p_.get();
  // COUPLAGE / POISSON : solve_fields assemble f = Sum_s elliptic_rhs_s(U_s) sur l'etat COURANT de
  // chaque bloc. Un bloc TENU (cadence M, hors fin de fenetre) y contribue avec son etat PERIME (sa
  // derniere avance, donc figee jusqu'a son prochain rattrapage) : densite / charge stale dans la
  // somme du Poisson tant qu'il n'a pas rattrape. Choix assume du stride (couplage lache du bloc lent).
  P->solve_fields();
  for (auto& s : P->sp) {
    if (!s.evolve) continue;  // bloc gele : non avance
    if (!Impl::stride_due(P->macro_step_, s.stride)) continue;  // hold : pas en fin de fenetre stride
    const Real eff_dt = Real(dt) * Real(s.stride);  // catch-up : pas effectif s.stride * dt
    s.advance(s.U, eff_dt, s.substeps);
    P->run_source_stage(s, eff_dt);  // OPT-IN : etage source condense par Schur (no-op sinon)
  }
  P->apply_couplings(Real(dt));  // sources couplees inter-especes (splitting), apres transport
  P->t += dt;
  P->macro_step_++;
}
void System::advance(double dt, int nsteps) {
  for (int s = 0; s < nsteps; ++s) step(dt);
}
double System::step_cfl(double cfl) {
  Impl* P = p_.get();
  P->solve_fields();
  const Real h = std::min(P->geom.dx(), P->geom.dy());
  // CFL PAR BLOC, FACTEUR STRIDE ET SUBSTEPS INCLUS. Un bloc de cadence M avance d'un pas effectif
  // M*dt en substeps_b sous-pas, donc chaque sous-pas vaut stride_b * dt / substeps_b : la condition
  // stable par sous-pas est stride_b * dt / substeps_b <= cfl * h / w_b, soit
  //   dt <= cfl * h * substeps_b / (stride_b * w_b).
  // Le dt GLOBAL est le min sur les blocs evolutifs (le plus contraignant). Sans cela, le pas calcule
  // sur w_max seul puis multiplie par M violerait la CFL d'un facteur M sur le bloc a stride.
  //
  // RETRO-COMPATIBILITE (post-#121). La formule est SUBSTEPS-AWARE : avec substeps_b > 1, le dt
  // retourne est substeps_b fois plus grand que l'ancienne formule dt = cfl*h/(stride*w).
  // bit-identique seulement pour substeps=1 (a tout stride) ; step_cfl est desormais substeps-aware
  // (dt = cfl*h*substeps/(stride*w)), donc un run step_cfl avec substeps>1 avance un dt plus grand
  // qu'avant #121 (pas CFL-maximal, chaque sous-pas est a la limite de stabilite).
  // Pour reproduire un run calibre avec l'ancienne formule, utiliser step(dt) avec le dt historique
  // explicite, PAS step_cfl.
  double dt = std::numeric_limits<double>::infinity();
  for (auto& s : P->sp) {
    if (!s.evolve) continue;  // bloc gele : ne contraint pas le pas
    const Real w = std::max(s.max_speed(s.U), Real(1e-30));
    const double dt_b = cfl * static_cast<double>(h) * static_cast<double>(s.substeps) /
                        (static_cast<double>(s.stride) * static_cast<double>(w));
    dt = std::min(dt, dt_b);
  }
  if (!std::isfinite(dt)) dt = cfl * static_cast<double>(h) / 1e-30;  // tous geles : pas degenere
  for (auto& s : P->sp) {
    if (!s.evolve) continue;
    if (!Impl::stride_due(P->macro_step_, s.stride)) continue;  // hold : pas en fin de fenetre stride
    const Real eff_dt = Real(dt) * Real(s.stride);  // catch-up : pas effectif s.stride * dt
    s.advance(s.U, eff_dt, s.substeps);
    P->run_source_stage(s, eff_dt);  // OPT-IN : etage source condense par Schur (no-op sinon)
  }
  P->apply_couplings(Real(dt));
  P->t += dt;
  P->macro_step_++;
  return dt;
}
double System::step_adaptive(double cfl) {
  Impl* P = p_.get();
  P->solve_fields();
  // Multirate : macro-pas = pas stable du bloc le plus LENT ; chaque bloc plus rapide est
  // sous-cycle n_b. aux fige sur le macro-pas (couplage once-per-step). SEMANTIQUE STRIDE =
  // hold-then-catch-up : un bloc de cadence M est TENU tant que (macro_step + 1) % M != 0, puis
  // avance d'un pas effectif M*macro_dt en fin de fenetre (cf. stride_due).
  Real wmin = Real(1e30);
  std::vector<Real> wb;
  wb.reserve(P->sp.size());
  for (auto& s : P->sp) {
    const Real w = s.evolve ? s.max_speed(s.U) : Real(0);  // bloc gele : hors cadence
    wb.push_back(w);
    if (s.evolve) wmin = std::min(wmin, w);
  }
  if (wmin >= Real(1e30)) wmin = Real(1e-30);  // aucun bloc evolutif (tous geles)
  const Real h = std::min(P->geom.dx(), P->geom.dy());
  const double macro_dt = cfl * static_cast<double>(h) / static_cast<double>(wmin);
  for (std::size_t b = 0; b < P->sp.size(); ++b) {
    auto& s = P->sp[b];
    if (!s.evolve) continue;  // bloc gele : non avance
    if (!Impl::stride_due(P->macro_step_, s.stride)) continue;  // hold : pas en fin de fenetre stride
    // Sous-cyclage stable du pas EFFECTIF M*macro_dt : chaque sous-pas doit verifier
    // M*macro_dt / n <= cfl*h / w_b, i.e. n >= ceil(M * w_b / w_min). Le facteur stride M est donc
    // porte par le nombre de sous-pas (sans lui, n sur w_b/w_min seul violerait la CFL d'un facteur M).
    int n = static_cast<int>(std::ceil(static_cast<double>(s.stride) *
                                       static_cast<double>(wb[b] / wmin)));
    if (n < 1) n = 1;
    const Real eff_dt = Real(macro_dt) * Real(s.stride);  // catch-up : pas effectif M*macro_dt
    s.advance(s.U, eff_dt, n);
    P->run_source_stage(s, eff_dt);  // OPT-IN : etage source condense par Schur (no-op sinon)
  }
  P->apply_couplings(Real(macro_dt));
  P->t += macro_dt;
  P->macro_step_++;
  return macro_dt;
}

std::vector<double> System::eval_rhs(const std::string& name) {
  Impl::Species& s = p_->find(name);
  MultiFab R(p_->ba, p_->dm, s.ncomp, 0);
  s.rhs_into(s.U, R);
  return p_->copy_state(R, s.ncomp);
}
std::vector<double> System::get_state(const std::string& name) {
  Impl::Species& s = p_->find(name);
  return p_->copy_state(s.U, s.ncomp);
}
void System::set_state(const std::string& name, const std::vector<double>& u) {
  Impl::Species& s = p_->find(name);
  p_->write_state(s.U, s.ncomp, u);
}
int System::n_vars(const std::string& name) const { return p_->find(name).ncomp; }
std::vector<std::string> System::variable_names(const std::string& name,
                                               const std::string& kind) const {
  const Impl::Species& s = p_->find(name);
  if (kind == "conservative") return s.cons_vars.names;
  if (kind == "primitive") return s.prim_vars.names;
  throw std::runtime_error("System::variable_names : kind 'conservative' | 'primitive' (recu '" +
                           kind + "')");
}
std::vector<std::string> System::variable_roles(const std::string& name,
                                               const std::string& kind) const {
  const Impl::Species& s = p_->find(name);
  const VariableSet* vs = nullptr;
  if (kind == "conservative") vs = &s.cons_vars;
  else if (kind == "primitive") vs = &s.prim_vars;
  else throw std::runtime_error("System::variable_roles : kind 'conservative' | 'primitive' (recu '" +
                                kind + "')");
  std::vector<std::string> out;
  out.reserve(static_cast<std::size_t>(vs->size));
  for (int i = 0; i < vs->size; ++i) out.push_back(role_name(vs->at(i).role));  // 'custom' si absent
  return out;
}
double System::block_gamma(const std::string& name) const { return p_->find(name).gamma; }

int System::nx() const { return p_->cfg.n; }
double System::time() const { return p_->t; }
int System::n_species() const { return static_cast<int>(p_->sp.size()); }
std::vector<std::string> System::block_names() const {
  // Lit le registre de blocs UNIQUE (p_->sp), peuple par tous les chemins d'ajout : un bloc charge
  // via add_dynamic_block / add_compiled_block (.so) y figure au meme titre qu'un add_block.
  std::vector<std::string> out;
  out.reserve(p_->sp.size());
  for (const auto& s : p_->sp) out.push_back(s.name);
  return out;
}
double System::mass(const std::string& name) const { return sum(p_->find(name).U, 0); }
std::vector<double> System::density(const std::string& name) const {
  return p_->copy_comp0(p_->find(name).U);
}
std::vector<double> System::potential() {
  p_->ensure_elliptic();
  device_fence();
  const ConstArray4 ph = p_->ell_phi().fab(0).const_array();
  const Box2D v = p_->aux.box(0);
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(ph(i, j));
  return out;
}

}  // namespace adc
