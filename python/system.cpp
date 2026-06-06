#include <adc/runtime/system.hpp>

#include <adc/core/variables.hpp>  // VariableSet + VariableRole : descripteur a roles porte par chaque bloc
#include <adc/runtime/abi_key.hpp>  // adc::abi_key + detail::abi_key_string (frontiere ABI du loader natif)
#include <adc/runtime/block_builder.hpp>  // GridContext + make_block/make_max_speed (fermetures compilees)
#include <adc/runtime/model_factory.hpp>  // detail::dispatch_model + briques compilees
#include <adc/coupling/condensed_schur_source_stepper.hpp>  // etage source condense par Schur (adc.Split / CondensedSchur, #126)
#include <adc/coupling/coupled_source_program.hpp>  // CoupledSourceKernel : source couplee generique (DSL P5, bytecode)
#include <adc/numerics/elliptic/geometric_mg.hpp>
#include <adc/numerics/elliptic/poisson_fft_solver.hpp>
#include <adc/numerics/elliptic/polar_poisson_solver.hpp>  // PolarPoissonSolver (Poisson polaire direct, REUTILISE)
#include <adc/runtime/system_field_solver.hpp>  // SystemFieldSolver : resolution elliptique + derivation de champ (Lot B)
#include <adc/runtime/system_stepper.hpp>  // SystemStepper : avance en temps (step/advance/step_cfl/step_adaptive) (Lot B)
#include <adc/runtime/block_builder_polar.hpp>  // fermetures de bloc POLAIRE (assemble_rhs_polar, REUTILISE)
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

// La trace de DIAGNOSTIC du chemin solve_fields (adc_trace_sf / adc_sf_mark, jalon #93) a ete extraite
// avec SystemFieldSolver vers include/adc/runtime/system_field_solver.hpp (namespace field_solver) ;
// elle reste env-gatee (ADC_TRACE_SOLVE_FIELDS) et inerte par defaut.
namespace {
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
  // GEOMETRIE POLAIRE (chantier "grille polaire diocotron", Phase 2b). polar_ == true quand
  // cfg.geometry == "polar" : le System tourne alors sur un anneau global (r, theta), avec le transport
  // polaire (assemble_rhs_polar) et le Poisson polaire (PolarPoissonSolver) au lieu du chemin cartesien.
  // pgeom_ est l'anneau (r_min, r_max, nr, ntheta) ; INERTE (jamais lu) en cartesien -> chemin
  // bit-identique. dom/ba/dm couvrent toujours l'espace d'INDICES (nx() x ny()), commun aux deux
  // geometries : seule la correspondance indices -> espace physique (geom vs pgeom_) change.
  bool polar_;
  PolarGeometry pgeom_;
  BoxArray ba;
  DistributionMapping dm;
  BCRec bc_;        // CL transport (periodique ou Foextrap selon cfg.periodic ; polaire : r physique, theta periodique)
  Box2D dom;
  Periodicity per_;
  bool periodic_;
  MultiFab aux;
  int aux_ncomp_ = kAuxBaseComps;     // largeur du canal aux PARTAGE (max des blocs ; >= 3)
  // Champs d'APPLICATION aux (bz_field_, te_src_) et tampons apply_bz/apply_te EXTRAITS vers
  // fields_ (SystemFieldSolver, Lot B) ; l'aux PARTAGE et sa largeur restent ici (canal commun).
  std::vector<Species> sp;
  double t = 0;
  int macro_step_ = 0;  // compteur de macro-pas (0-indexe) : sert au filtre stride par bloc
  std::vector<std::function<void(Real)>> couplings;  // sources couplees inter-especes (splitting)

  // stride_due (filtre de cadence hold-then-catch-up) EXTRAIT vers stepper_ (SystemStepper, Lot B) :
  // il sert exclusivement a l'avance en temps. macro_step_ (ci-dessus) reste un membre PARTAGE de Impl
  // (lu par time() indirectement via t, incremente par stepper_ via owner_->macro_step_).

  // Resolution elliptique + derivation de champ EXTRAITES vers fields_ (SystemFieldSolver, Lot B,
  // cf. docs/SYSTEM_CPP_EXTRACTION_PLAN.md section 2) : la configuration Poisson (p_rhs/p_solver/
  // p_bc/p_wall/p_wall_radius/p_eps_), les champs de coefficient (eps(x), eps_x/eps_y, kappa), les
  // solveurs (ell_ cartesien, pell_ polaire) et les tampons d'application aux (B_z, T_e) y vivent
  // desormais. fields_ lit l'aux/sp/cfg/geom/pgeom_/ba/dm/bc_/dom/per_ PARTAGES de Impl via son
  // back-pointer. Declare apres les membres partages qu'il capture (initialise dans le constructeur).

  // Nombre de cellules radiales / azimutales en POLAIRE (0 => repli sur cfg.n, cf. SystemConfig).
  static int polar_nr(const SystemConfig& c) { return c.nr > 0 ? c.nr : c.n; }
  static int polar_ntheta(const SystemConfig& c) { return c.ntheta > 0 ? c.ntheta : c.n; }
  // Domaine d'INDICES : carre n x n en cartesien ; nr x ntheta en polaire (i = r, j = theta).
  static Box2D index_domain(const SystemConfig& c) {
    if (c.geometry == "polar") return Box2D::from_extents(polar_nr(c), polar_ntheta(c));
    return Box2D::from_extents(c.n, c.n);
  }

  explicit Impl(const SystemConfig& c)
      : cfg(c),
        geom{Box2D::from_extents(c.n, c.n), 0.0, c.L, 0.0, c.L},
        polar_(c.geometry == "polar"),
        pgeom_{index_domain(c), Real(c.r_min), Real(c.r_max)},
        ba(std::vector<Box2D>{index_domain(c)}),
        dm(1, n_ranks()),
        bc_(make_bc(c)),
        dom(index_domain(c)),
        per_{!polar_ && c.periodic, !polar_ && c.periodic},
        periodic_(!polar_ && c.periodic),
        aux(ba, dm, kAuxBaseComps, 1),
        fields_(this),
        stepper_(this) {}

  // Resolution elliptique + derivation de champ (Lot B). POSSEDE les solveurs (ell_/pell_), la config
  // Poisson, les champs de coefficient et les tampons d'application aux (B_z, T_e). owner_ = this : le
  // helper lit l'aux/sp/cfg/geom/pgeom_/ba/dm/bc_/dom/per_/periodic_/polar_ PARTAGES de Impl. Aucun de
  // ces acces ne dereference Impl a la CONSTRUCTION (back-pointer pur) -> init en fin de liste sans
  // dependance d'ordre. Voir include/adc/runtime/system_field_solver.hpp.
  field_solver::SystemFieldSolver<Impl> fields_;

  // Avance en temps (Lot B). ORCHESTRE step / advance / step_cfl / step_adaptive, le filtre de cadence
  // (stride_due), l'etage source condense (run_source_stage) et les couplages (apply_couplings). owner_
  // = this : le stepper lit sp / fields_ / aux / couplings / t / macro_step_ / geom / pgeom_ / polar_
  // PARTAGES de Impl via son back-pointer. Back-pointer pur a la construction (aucun dereferencement) ->
  // init en fin de liste sans dependance d'ordre. Voir include/adc/runtime/system_stepper.hpp.
  stepper::SystemStepper<Impl> stepper_;

  // Garantit une largeur aux >= ncomp (canal PARTAGE). Reallouer l'aux GARDE son adresse (membre :
  // les fermetures de bloc capturent &aux via grid_ctx) et re-applique B_z. No-op si deja assez large.
  void ensure_aux_width(int ncomp) {
    if (ncomp <= aux_ncomp_) return;
    aux_ncomp_ = ncomp;
    aux = MultiFab(ba, dm, aux_ncomp_, 1);
    fields_.apply_bz();
    fields_.apply_te();
  }

  // apply_bz (peuplement de la composante B_z du canal aux) EXTRAIT vers fields_ (SystemFieldSolver).

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

  // kTeComp (composante canonique de T_e) et apply_te (peuplement de T_e = p/rho du bloc source)
  // EXTRAITS vers fields_ (SystemFieldSolver) : T_e fait partie de l'application de champ aux.

  static BCRec make_bc(const SystemConfig& c) {
    BCRec b;  // periodique par defaut
    if (c.geometry == "polar") {
      // POLAIRE : r (dir 0, xlo/xhi) porte une CL PHYSIQUE (paroi / sortie libre, Foextrap) ; theta
      // (dir 1, ylo/yhi) est PERIODIQUE (l'anneau couvre [0, 2pi)). C'est la convention de
      // test_polar_transport_mms et de assemble_rhs_polar (theta periodique, r physique).
      b.xlo = b.xhi = BCType::Foextrap;
      b.ylo = b.yhi = BCType::Periodic;
      return b;
    }
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

  // apply_couplings (sources de couplage inter-especes par splitting, APRES le transport) et
  // run_source_stage (etage source condense par Schur, OPT-IN) EXTRAITS vers stepper_ (SystemStepper,
  // Lot B) : ce sont des etapes de l'avance en temps, invoquees par step / step_cfl / step_adaptive.
  // Ils lisent l'etat PARTAGE via owner_-> (couplings, fields_.ell_phi(), aux, kAuxBaseComps). La liste
  // couplings (ci-dessus) reste un membre de Impl (peuplee par add_ionization / add_collision / ...).

  // --- solveur elliptique (Poisson de systeme) -----------------------------
  // poisson_bc / wall_active / ensure_elliptic / apply_epsilon_field / apply_epsilon_anisotropic_field
  // / apply_reaction_field / ell_rhs / ell_phi / ell_solve / ensure_elliptic_polar / solve_fields_polar
  // / solve_fields EXTRAITS vers fields_ (SystemFieldSolver, Lot B). Voir le header.

  // --- schemas spatiaux compiles -------------------------------------------
  // Evaluateur methode-des-lignes d'un bloc (L/F/Model figes) : ghosts puis R = -div F + S.
  // Construction des fermetures de bloc (avance + residu + Poisson) deplacee en en-tete
  // (adc/runtime/block_builder.hpp : make_block / make_max_speed / make_poisson_rhs) afin que le
  // chemin template de production soit instanciable hors de cette unite (compilation AOT d'un
  // modele genere). Ici on ne fournit que le contexte de grille a leur passer.
  GridContext grid_ctx() { return GridContext{dom, bc_, geom, &aux}; }

  // Contexte de grille POLAIRE (anneau pgeom_ + CL r/theta + aux) pour les fermetures de bloc polaires
  // (block_builder_polar.hpp). Pendant de grid_ctx() ; jamais appele en cartesien.
  PolarGridContext grid_ctx_polar() { return PolarGridContext{dom, bc_, pgeom_, &aux}; }

  // ensure_elliptic_polar / solve_fields_polar / solve_fields (corps) EXTRAITS vers fields_
  // (SystemFieldSolver, Lot B). Delegation pure : le dispatch cartesien/polaire, le device_fence et
  // l'ordre des fill_ghosts/fill_boundary vivent maintenant dans le header (bit-identique).
  void solve_fields() { fields_.solve_fields(); }

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
// Garde-fou geometrie (chantier "grille polaire"). Le CHOIX de geometrie est porte par la config
// (adc.CartesianMesh / adc.PolarMesh). "cartesian" : chemin historique, bit-identique. "polar" : anneau
// global (r, theta) branche dans System.step (Phase 2b) : transport polaire (assemble_rhs_polar) +
// Poisson polaire (PolarPoissonSolver) + aux en base locale (e_r, e_theta). On valide ICI les bornes
// radiales de l'anneau (r_max > r_min >= 0) ; le Python (PolarMesh) les valide deja, mais un appelant
// qui construit le SystemConfig a la main doit aussi etre protege. Tout autre token est une erreur.
void check_geometry(const SystemConfig& c) {
  if (c.geometry == "cartesian") return;
  if (c.geometry == "polar") {
    if (!(c.r_max > c.r_min && c.r_min >= 0.0))
      throw std::runtime_error(
          "System : geometry='polar' exige un anneau r_max > r_min >= 0 (r_min > 0 evite la "
          "singularite de coordonnee r=0) ; cf. adc.PolarMesh");
    // nr >= 3 IMPOSE : la derive radiale de l'aux (derive_aux_polar) utilise un stencil DECENTRE
    // d'ordre 2 aux deux parois (lit phi(i+1),phi(i+2) a r_min et phi(i-1),phi(i-2) a r_max). phi est
    // alloue SANS ghost par le solveur direct (sa box valide EST son allocation) : nr < 3 ferait lire
    // phi hors bornes (UB). On le refuse ICI (meme calcul de repli que Impl::polar_nr : nr ou n).
    const int nr = c.nr > 0 ? c.nr : c.n;
    if (nr < 3)
      throw std::runtime_error(
          "System : geometry='polar' exige nr >= 3 (stencil radial decentre d'ordre 2 aux parois ; "
          "phi sans ghost) ; cf. adc.PolarMesh");
    return;
  }
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
  if (P->polar_) {
    // CHEMIN POLAIRE (anneau) : fermetures bati par block_builder_polar.hpp (assemble_rhs_polar +
    // ExBVelocityPolar + Poisson polaire). IMEX n'a pas de sens ici (transport ExB scalaire, pas de
    // source raide) : on le refuse explicitement plutot que de jouer le seul transport en silence.
    if (imex)
      throw std::runtime_error(
          "System::add_block (polaire) : time='imex' non supporte (transport ExB scalaire sur un "
          "anneau : pas de source raide a traiter en implicite). Utiliser 'explicit'/'ssprk2'/'ssprk3'.");
    const PolarGridContext pctx = P->grid_ctx_polar();
    detail::dispatch_model_polar(model, [&](auto m) {
      using M = decltype(m);
      ncomp = M::n_vars;
      cons_vs = M::conservative_vars();
      prim_vs = M::primitive_vars();
      // wall_radial = true : paroi solide aux deux bords radiaux (no-penetration) -> flux radial nul
      // a r_min / r_max -> masse Sum n r dr dtheta conservee A LA MACHINE (l'anneau diocotron est borne
      // par deux parois conductrices). C'est la BC qui rend le pas couple conservatif.
      clo = make_block_polar(m, limiter, riemann, pctx, recon_prim, method, /*wall_radial=*/true);
      max_speed = make_max_speed_polar(m, &P->aux);
      add_poisson_rhs = make_poisson_rhs_polar(m);
      auto conv = make_cell_convert(m);
      prim_to_cons = std::move(conv.first);
      cons_to_prim = std::move(conv.second);
    });
  } else {
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
  }
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
  p_->fields_.p_rhs = rhs;
  p_->fields_.p_solver = solver;
  p_->fields_.p_bc = bc;
  p_->fields_.p_wall = wall;
  p_->fields_.p_wall_radius = wall_radius;
  p_->fields_.p_eps_ = static_cast<Real>(epsilon);
  p_->fields_.ell_.reset();
}

void System::set_epsilon_field(const std::vector<double>& eps) {
  const int n = p_->cfg.n;
  if (static_cast<int>(eps.size()) != n * n)
    throw std::runtime_error("System::set_epsilon_field : taille != n*n");
  for (double e : eps)
    if (!(e > 0.0))
      throw std::runtime_error("System::set_epsilon_field : permittivite eps(x) > 0 requise");
  p_->fields_.p_eps_field_ = eps;
  p_->fields_.has_eps_field_ = true;
  p_->fields_.ell_.reset();  // l'operateur sera reconstruit avec le champ eps au prochain solve_fields
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
  p_->fields_.p_eps_x_field_ = eps_x;
  p_->fields_.p_eps_y_field_ = eps_y;
  p_->fields_.has_eps_xy_field_ = true;
  p_->fields_.ell_.reset();  // operateur reconstruit en div(diag(eps_x, eps_y) grad phi) au prochain solve_fields
}

void System::set_reaction_field(const std::vector<double>& kappa) {
  const int n = p_->cfg.n;
  if (static_cast<int>(kappa.size()) != n * n)
    throw std::runtime_error("System::set_reaction_field : taille != n*n");
  for (double k : kappa)
    if (!(k >= 0.0))
      throw std::runtime_error("System::set_reaction_field : terme de reaction kappa(x) >= 0 requis "
                               "(operateur elliptique bien pose et multigrille convergente)");
  p_->fields_.p_kappa_field_ = kappa;
  p_->fields_.has_kappa_field_ = true;
  p_->fields_.ell_.reset();  // operateur reconstruit avec - kappa phi au prochain solve_fields
}

void System::ensure_aux_width(int ncomp) { p_->ensure_aux_width(ncomp); }

void System::set_magnetic_field(const std::vector<double>& bz) {
  const int n = p_->cfg.n;
  if (static_cast<int>(bz.size()) != n * n)
    throw std::runtime_error("System::set_magnetic_field : taille != n*n");
  p_->fields_.bz_field_.assign(bz.begin(), bz.end());
  p_->fields_.apply_bz();  // applique tout de suite si un bloc lit deja B_z ; sinon conserve pour ensure_aux_width
}

void System::set_electron_temperature_from(const std::string& name) {
  const int idx = p_->index(name);  // leve si bloc inconnu
  if (p_->sp[static_cast<std::size_t>(idx)].ncomp != 4)
    throw std::runtime_error("System::set_electron_temperature_from : le bloc '" + name +
                             "' doit etre compressible (4 var : rho, rho u, rho v, E) pour T = p/rho");
  p_->fields_.te_src_ = idx;
  // T_e (comp canonique 4) DERIVE : recalcule a chaque solve_fields. Inerte tant qu'aucun bloc ne
  // lit T_e (n_aux=5 -> ensure_aux_width(5)), comme set_magnetic_field pour B_z.
  p_->fields_.apply_te();
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
    // STRICT (pas de repli silencieux) : une source couplee DSL vise un (bloc, role) EXPLICITEMENT
    // demande par l'utilisateur. Si le bloc n'expose PAS ce role, c'est une erreur : un repli sur la
    // composante 0 appliquerait la source au mauvais champ EN SILENCE (le faux-positif identifie a la
    // revue). On leve. Distinct des couplages NOMMES (add_collision/add_pair) qui assument volontairement
    // la disposition canonique via role_index(..., fallback) et restent inchanges.
    const int comp = P->sp[static_cast<std::size_t>(sidx)].cons_vars.index_of(r);
    if (comp < 0)
      throw std::runtime_error("System::add_coupled_source : le bloc '" + block +
                               "' n'expose pas le role '" + role +
                               "' (pas de repli silencieux sur la composante 0)");
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
  if (P->fields_.bz_field_.empty())
    throw std::runtime_error(
        "System::set_source_stage : le bloc '" + name + "' n'a pas de champ B_z (aux Omega) ; "
        "adc.CondensedSchur exige set_magnetic_field(B_z) (le terme de Lorentz lit Omega = B_z).");
  P->ensure_aux_width(kAuxBaseComps + 1);  // garantit le canal B_z dans l'aux partage + re-applique B_z
  // Construit l'etage source condense sur le layout REEL du System (ba/dm/geom) avec la CL du Poisson.
  // Le stepper alloue ses tampons UNE fois ; step() les reutilise (cf. son cycle de vie). alpha =
  // constante de couplage electrostatique du sous-systeme source. n_precond_vcycles = defaut (1).
  s.schur = std::make_shared<CondensedSchurSourceStepper>(vs, P->geom, P->ba, P->fields_.poisson_bc(),
                                                          static_cast<Real>(alpha));
  s.schur_theta = theta;
}

void System::set_density(const std::string& name, const std::vector<double>& rho) {
  Impl::Species& s = p_->find(name);
  // Layout row-major du tableau d'entree : (ni x nj) = extents de la box de l'etat. En cartesien
  // ni = nj = cfg.n (indexation et taille bit-identiques a avant). En polaire ni = nr, nj = ntheta :
  // on indexe par les extents reels de la box (et non n*n), donc nr != ntheta est correctement gere.
  const Box2D v = s.U.box(0);
  const int ni = v.nx(), nj = v.ny();
  if (static_cast<int>(rho.size()) != ni * nj)
    throw std::runtime_error("System::set_density : taille != nr*ntheta (ou n*n en cartesien)");
  const Real gm1 = Real(s.gamma) - Real(1);
  Array4 u = s.U.fab(0).array();
  // CONVENTION DE LAYOUT (inchangee vs l'historique) : axe lent = 2nd indice de box (j), axe rapide =
  // 1er (i), i.e. flat[(j-lo) * ni + (i-lo)]. En cartesien ni = n, lo = 0 -> flat[j*n+i] (bit-identique
  // a avant). En polaire le tableau est donc (nr, ntheta) ligne-par-ligne radiale (i = r lent par
  // rapport a... non : j = theta lent, i = r rapide), MEME ordre que density()/copy_comp0 -> coherent.
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
      const Real r = rho[static_cast<std::size_t>(j - v.lo[1]) * ni + (i - v.lo[0])];
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
  // Nombre de cellules = EXTENTS REELS du domaine d'indices (n*n cartesien, nr*ntheta polaire), PAS
  // cfg.n*cfg.n : en polaire cfg.n = nr, donc cfg.n^2 != nr*ntheta -> debordement de tas (ntheta<nr) ou
  // contenu partiel/faux (ntheta>nr). Cartesien bit-identique (dom.nx()==dom.ny()==n).
  const std::size_t nn = static_cast<std::size_t>(p_->dom.nx()) * static_cast<std::size_t>(p_->dom.ny());
  if (prim.size() != static_cast<std::size_t>(nc) * nn)
    throw std::runtime_error("System::set_primitive_state : taille != ncomp*nr*ntheta (n*n cartesien) (bloc '" + name +
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
  // Nombre de cellules = EXTENTS REELS du domaine d'indices (n*n cartesien, nr*ntheta polaire), PAS
  // cfg.n*cfg.n : en polaire cfg.n = nr, donc cfg.n^2 != nr*ntheta -> debordement de tas (ntheta<nr) ou
  // contenu partiel/faux (ntheta>nr). Cartesien bit-identique (dom.nx()==dom.ny()==n).
  const std::size_t nn = static_cast<std::size_t>(p_->dom.nx()) * static_cast<std::size_t>(p_->dom.ny());
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

// Avance en temps EXTRAITE vers stepper_ (SystemStepper, Lot B). Delegation pure : le dispatch
// cartesien/polaire du pas physique h, la formule CFL par bloc (substeps/stride), la semantique
// hold-then-catch-up du compteur de macro-pas, l'etage source condense et les couplages vivent
// maintenant dans le header (bit-identique). L'API publique reste inchangee.
void System::step(double dt) { p_->stepper_.step(dt); }
void System::advance(double dt, int nsteps) { p_->stepper_.advance(dt, nsteps); }
double System::step_cfl(double cfl) { return p_->stepper_.step_cfl(cfl); }
double System::step_adaptive(double cfl) { return p_->stepper_.step_adaptive(cfl); }

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
// Axe LENT du champ (lignes du tableau (ny, nx)). On le lit du domaine d'INDICES (dom = nx() x ny()),
// SOURCE UNIQUE des extents pour les deux geometries : cartesien dom = n x n -> ny() == nx() == n (carre,
// INCHANGE) ; polaire dom = nr x ntheta -> nx() == nr (rapide, i), ny() == ntheta (lent, j). C'est cette
// dimension qui dimensionne le tableau numpy cote bindings : un champ polaire fait nx()*ny() = nr*ntheta
// valeurs, et avec nr != ntheta le remodelage carre (nx, nx) deborde le tampon (bug de teardown).
int System::ny() const { return p_->dom.ny(); }
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
double System::mass(const std::string& name) const {
  const Impl::Species& s = p_->find(name);
  if (!p_->polar_) return sum(s.U, 0);  // cartesien : somme nue des cellules (bit-identique)
  // POLAIRE : masse FV = Sum_ij n_ij r_i dr dtheta (volume de cellule annulaire r dr dtheta). C'est la
  // quantite CONSERVEE par assemble_rhs_polar (cf. test_polar_transport_mms). Boucle hote sur les
  // cellules valides (mono-rang : un seul fab local), reduite sur les rangs par symetrie (n_ranks==1).
  device_fence();
  const PolarGeometry& g = p_->pgeom_;
  const Real dr = g.dr(), dth = g.dtheta();
  double m = 0.0;
  for (int li = 0; li < s.U.local_size(); ++li) {
    const ConstArray4 u = s.U.fab(li).const_array();
    const Box2D v = s.U.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        m += static_cast<double>(u(i, j, 0)) * static_cast<double>(g.r_cell(i) * dr * dth);
  }
  return all_reduce_sum(m);
}
std::vector<double> System::density(const std::string& name) const {
  return p_->copy_comp0(p_->find(name).U);
}
std::vector<double> System::potential() {
  device_fence();
  // POLAIRE : phi vient du Poisson polaire (pell_), pas du solveur cartesien (ell_). On le construit
  // paresseusement si besoin (un appel avant tout step) et on lit phi() de PolarPoissonSolver.
  if (p_->polar_) {
    p_->fields_.ensure_elliptic_polar();
    const ConstArray4 ph = p_->fields_.pell_->phi().fab(0).const_array();
    const Box2D v = p_->aux.box(0);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(ph(i, j));
    return out;
  }
  p_->fields_.ensure_elliptic();
  const ConstArray4 ph = p_->fields_.ell_phi().fab(0).const_array();
  const Box2D v = p_->aux.box(0);
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(ph(i, j));
  return out;
}

}  // namespace adc
