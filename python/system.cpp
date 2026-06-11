#include <adc/runtime/system.hpp>

#include <adc/core/variables.hpp>  // VariableSet + VariableRole : descripteur a roles porte par chaque bloc
#include <adc/runtime/abi_key.hpp>  // adc::abi_key + detail::abi_key_string (frontiere ABI du loader natif)
#include <adc/runtime/block_builder.hpp>  // GridContext + make_block/make_max_speed (fermetures compilees)
#include <adc/runtime/model_factory.hpp>  // detail::dispatch_model + briques compilees
#include <adc/coupling/condensed_schur_source_stepper.hpp>  // etage source condense par Schur (adc.Split / CondensedSchur, #126)
#include <adc/coupling/polar_condensed_schur_source_stepper.hpp>  // pendant POLAIRE de l'etage source condense (Voie A etape 2c, #212)
#include <adc/coupling/coupled_source_program.hpp>  // CoupledSourceKernel : source couplee generique (DSL P5, bytecode)
#include <adc/numerics/elliptic/geometric_mg.hpp>
#include <adc/numerics/elliptic/poisson_fft_solver.hpp>
#include <adc/numerics/elliptic/polar_poisson_solver.hpp>  // PolarPoissonSolver (Poisson polaire direct, REUTILISE)
#include <adc/runtime/system_field_solver.hpp>  // SystemFieldSolver : resolution elliptique + derivation de champ (Lot B)
#include <adc/runtime/system_stepper.hpp>  // SystemStepper : avance en temps (step/advance/step_cfl/step_adaptive) (Lot B)
#include <adc/runtime/system_block_store.hpp>  // SystemBlockStore : gestion de blocs (BlockState + registre + index/copy/write) (Lot B.3)
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
#include <map>     // std::map (registre des params runtime par bloc, P7-b)
#include <memory>
#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>

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
ADC_EXPORT std::string abi_key() { return detail::abi_key_string(); }

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
  // GESTION DE BLOCS extraite vers SystemBlockStore (Lot B.3, derniere extraction P0 du god-class) :
  // la struct des blocs (ex-Species, renommee BlockState), le registre ordonne (blocks_.blocks), les
  // acces par nom (index / find) et le marshaling d'etat (copy_comp0 / copy_state / write_state) y
  // vivent desormais. Voir include/adc/runtime/system_block_store.hpp.
  //
  // ALIAS DE COMPATIBILITE. Les gabarits en-tete deja extraits (SystemFieldSolver, SystemStepper,
  // native_loader) iterent `owner_->sp` / `P->sp` et nomment `Impl::Species` ; on conserve ces deux
  // points d'acces a l'identique (zero churn hors de ce fichier) :
  //  - `Species` = le type des blocs porte par le store (init par agregat positionnel inchange) ;
  //  - `sp` = une REFERENCE sur le registre du store (meme objet, meme iteration, meme indexation).
  using Species = SystemBlockStore::BlockState;

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

  // MASQUE DE DOMAINE DISQUE (chantier T2, CONTRAT inerte par defaut). disc_set_ == false : aucun
  // disque fixe -> le masque est "tout actif" et le chemin de transport reste BIT-IDENTIQUE. Quand
  // set_disc_domain est appele, disc_ porte le descripteur (centre + rayon, SOURCE UNIQUE reutilisant
  // le level set du mur conducteur) et disc_mask_ materialise le champ 0/1 cellule-centre (1 ghost,
  // pour que le transport mask-aware lise les voisins). Le masque est CONSTRUIT mais PAS encore branche
  // dans step() (scaffolding) ; il est consultable (disc_mask()) et consomme par assemble_rhs_masked.
  detail::DiscDomain disc_;
  bool disc_set_ = false;
  MultiFab disc_mask_;  // 0/1 cellule-centre, meme layout que les blocs (ba/dm), 1 ghost ; vide tant que !disc_set_
  // MODE DE GEOMETRIE DE TRANSPORT (chantier T5-PR3, cablage du disque dans step()). None (defaut) :
  // transport plein cartesien (assemble_rhs) -> BIT-IDENTIQUE. Staircase : assemble_rhs_masked (masque
  // 0/1). CutCell : assemble_rhs_eb (cut-cell EB). Le stepper lit ce mode pour AIGUILLER l'avance de
  // transport de chaque bloc (advance vs advance_masked vs advance_eb). Pose par set_disc_domain(mode=)
  // / set_geometry_mode ; n'a d'effet que si un disque est fixe (disc_set_) ET le bloc porte l'avance
  // disque correspondante. None tant qu'aucun mode disque n'est demande.
  GeometryMode geometry_mode_ = GeometryMode::None;
  // Champs d'APPLICATION aux (bz_field_, te_src_) et tampons apply_bz/apply_te EXTRAITS vers
  // fields_ (SystemFieldSolver, Lot B) ; l'aux PARTAGE et sa largeur restent ici (canal commun).
  // Registre de blocs POSSEDE par le store (Lot B.3). `sp` est une REFERENCE sur blocks_.blocks : meme
  // objet (aucune copie), donc owner_->sp / P->sp dans les gabarits en-tete restent bit-identiques.
  SystemBlockStore blocks_;
  std::vector<Species>& sp = blocks_.blocks;
  // P7-b : valeurs des parametres RUNTIME par bloc AOT (nom du bloc -> vecteur des valeurs courantes).
  // Le vecteur est PARTAGE (shared_ptr) avec les fermetures du bloc compile : ecrire dedans
  // (set_block_params) change le comportement du bloc au prochain pas SANS recompiler. Absent pour un
  // bloc sans param runtime ou pour les autres chemins (natif / dynamique). Pose par add_compiled_block.
  std::map<std::string, std::shared_ptr<std::vector<double>>> block_params_;
  // Diagnostics Newton (IMEX, OPT-IN) : rapport par bloc, possede ICI en shared_ptr (adresse STABLE
  // meme quand sp realloue) ; les fermetures AdvanceImex* du bloc ecrivent dedans par pointeur brut.
  // Absent (cle manquante) pour un bloc sans newton_diagnostics -> newton_report leve une erreur claire.
  std::map<std::string, std::shared_ptr<NewtonReport>> newton_reports_;
  double t = 0;
  int macro_step_ = 0;  // compteur de macro-pas (0-indexe) : sert au filtre stride par bloc
  std::vector<std::function<void(Real)>> couplings;  // sources couplees inter-especes (splitting)
  // Bornes GLOBALES de pas de temps (System::add_dt_bound) : evaluees UNE fois par pas (hote) par
  // step_cfl / step_adaptive. Crochet des contraintes non locales-cellule (couplage multi-blocs,
  // Schur/Poisson, scheduler). Vide (defaut) -> politique de pas historique, bit-identique.
  struct GlobalDtBound {
    std::string label;
    std::function<double()> fn;
  };
  std::vector<GlobalDtBound> dt_bounds_;
  // Frequences DECLAREES des sources couplees (CoupledSource.frequency, audit vague 3) : les
  // couplages s'appliquent UNE fois par MACRO-pas (apply_couplings(dt)), la borne porte donc sur
  // le macro-dt : dt <= cfl / mu, SANS facteur substeps/stride. Vide (defaut) -> aucune borne.
  struct CoupledFreq {
    std::string label;
    double mu;
  };
  std::vector<CoupledFreq> coupled_freqs_;
  // Frequences PAR CELLULE des sources couplees (CoupledSource.frequency avec une Expr, raffinement
  // de la frequence CONSTANTE ci-dessus) : un programme bytecode mu(U) evalue par cellule a CHAQUE
  // pas (reduction MAX, all_reduce_max global), borne dt <= cfl / max(mu). Les entrees REUTILISENT la
  // resolution resolve() des registres d'entree (sidx, comp) ; les constantes sont les memes que la
  // source. Vide (defaut) -> aucune borne par cellule (chemin historique). Stocke APRES la validation
  // complete (meme regle anti-borne-fantome que le scalaire).
  struct CoupledFreqExpr {
    std::string label;
    CsProgram prog;
    struct In { int sidx, comp; };
    std::vector<In> ins;  // (espece, composante) des entrees (memes que la source ; resolues une fois)
    int n_in = 0;
    std::vector<Real> kconsts;  // constantes chargees dans r[n_in ..] (memes que la source)
  };
  std::vector<CoupledFreqExpr> coupled_freq_exprs_;

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
  // BoxArray du domaine d'INDICES. Cartesien (et polaire mono-box, theta_boxes <= 1) : UNE box couvrant
  // tout le domaine -> STRICTEMENT bit-identique a l'historique (ba = {index_domain}). Polaire avec
  // theta_boxes > 1 : decoupage en BANDES theta -- chaque box couvre tout le rayon [0, nr-1] et une
  // bande azimutale contigue (memes bornes que theta_split de test_polar_schur_multibox). Les bandes
  // pavent EXACTEMENT [0, ntheta-1] (longueurs base + reste, mais theta_boxes divise ntheta -> bandes
  // egales). check_geometry valide deja 1 <= theta_boxes <= ntheta et la divisibilite.
  static BoxArray index_boxarray(const SystemConfig& c) {
    if (c.geometry != "polar" || c.theta_boxes <= 1)
      return BoxArray(std::vector<Box2D>{index_domain(c)});
    const int nr = polar_nr(c), nth = polar_ntheta(c), nseg = c.theta_boxes;
    std::vector<Box2D> boxes;
    boxes.reserve(static_cast<std::size_t>(nseg));
    int base = nth / nseg, rem = nth % nseg, cur = 0;
    for (int k = 0; k < nseg; ++k) {
      const int len = base + (k < rem ? 1 : 0);
      boxes.push_back(Box2D{{0, cur}, {nr - 1, cur + len - 1}});
      cur += len;
    }
    return BoxArray(std::move(boxes));
  }

  explicit Impl(const SystemConfig& c)
      : cfg(c),
        geom{Box2D::from_extents(c.n, c.n), 0.0, c.L, 0.0, c.L},
        polar_(c.geometry == "polar"),
        pgeom_{index_domain(c), Real(c.r_min), Real(c.r_max)},
        // ba : UNE box (cartesien ou polaire mono-box) ; bandes theta si polaire theta_boxes > 1
        // (decoupage du TRANSPORT). dm : round-robin (nboxes, nranks) -- 1 box -> dm(1, n_ranks())
        // bit-identique a l'historique.
        ba(index_boxarray(c)),
        dm(ba.size(), n_ranks()),
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

  // Acces par nom DELEGUES au store (Lot B.3) : meme recherche lineaire, meme indexation par ordre
  // d'insertion, meme message d'erreur ("System : bloc inconnu '...'").
  Species& find(const std::string& name) { return blocks_.find(name); }
  const Species& find(const std::string& name) const { return blocks_.find(name); }
  int index(const std::string& name) const { return blocks_.index(name); }

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
  // GridContext : maillage + CL + aux + geometrie DISQUE (chantier T5-PR3). disc_mask_ / disc_ sont
  // des MEMBRES a adresse STABLE -> les fermetures de bloc (build_block) les lisent par pointeur au
  // pas, donc l'ordre add_block / set_disc_domain est indifferent (le masque est materialise / le
  // rayon pose avant le 1er step ; tant que !disc_set_ le stepper ne selectionne pas l'avance disque).
  GridContext grid_ctx() { return GridContext{dom, bc_, geom, &aux, &disc_mask_, &disc_}; }

  // Contexte de grille POLAIRE (anneau pgeom_ + CL r/theta + aux) pour les fermetures de bloc polaires
  // (block_builder_polar.hpp). Pendant de grid_ctx() ; jamais appele en cartesien.
  PolarGridContext grid_ctx_polar() { return PolarGridContext{dom, bc_, pgeom_, &aux}; }

  // ensure_elliptic_polar / solve_fields_polar / solve_fields (corps) EXTRAITS vers fields_
  // (SystemFieldSolver, Lot B). Delegation pure : le dispatch cartesien/polaire, le device_fence et
  // l'ordre des fill_ghosts/fill_boundary vivent maintenant dans le header (bit-identique).
  void solve_fields() { fields_.solve_fields(); }

  // Marshaling d'etat DELEGUE au store (Lot B.3) : copy_comp0 / copy_state / write_state portent le
  // device_fence, le layout (composante-majeur) et l'erreur de taille a l'identique. Conserves comme
  // helpers de Impl car native_loader et les methodes de facade les appellent via P->copy_state /
  // P->write_state / P->copy_comp0 (point d'acces inchange, zero churn hors de ce fichier).
  //
  // MULTI-BOX (decoupage theta du transport polaire, ADC-67). Le store marshale via fab(0) -- valable
  // pour l'unique box locale du cartesien et du polaire mono-box (local_size() <= 1, y compris MPI mono-
  // box ou un rang sans box rend {}). Avec theta_boxes > 1, un rang porte PLUSIEURS boites locales
  // (local_size() > 1) : on reconstruit alors le champ GLOBAL (taille dom.nx() x dom.ny()) en placant
  // chaque box a ses indices GLOBAUX, exactement comme density_global / state_global (gather collectif,
  // all_reduce_sum ; identite mono-rang). On NE TOUCHE PAS le store (extraction VERBATIM bit-identique) :
  // le branchement local_size() <= 1 delegue tel quel -> cartesien et polaire mono-box INCHANGES.
  std::vector<double> copy_comp0(const MultiFab& mf) const {
    if (mf.local_size() <= 1) return blocks_.copy_comp0(mf);
    device_fence();
    const int gnx = dom.nx(), gny = dom.ny();
    std::vector<double> out(static_cast<std::size_t>(gnx) * gny, 0.0);
    for (int li = 0; li < mf.local_size(); ++li) {
      const ConstArray4 u = mf.fab(li).const_array();
      const Box2D v = mf.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          out[static_cast<std::size_t>(j) * gnx + i] = static_cast<double>(u(i, j, 0));
    }
    all_reduce_sum_inplace(out.data(), static_cast<int>(out.size()));
    return out;
  }
  std::vector<double> copy_state(const MultiFab& mf, int ncomp) const {
    if (mf.local_size() <= 1) return blocks_.copy_state(mf, ncomp);
    device_fence();
    const int gnx = dom.nx(), gny = dom.ny();
    std::vector<double> out(static_cast<std::size_t>(ncomp) * gnx * gny, 0.0);
    for (int li = 0; li < mf.local_size(); ++li) {
      const ConstArray4 u = mf.fab(li).const_array();
      const Box2D v = mf.box(li);
      for (int c = 0; c < ncomp; ++c)
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i)
            out[(static_cast<std::size_t>(c) * gny + j) * gnx + i] = static_cast<double>(u(i, j, c));
    }
    all_reduce_sum_inplace(out.data(), static_cast<int>(out.size()));
    return out;
  }
  void write_state(MultiFab& mf, int ncomp, const std::vector<double>& in) {
    if (mf.local_size() <= 1) { blocks_.write_state(mf, ncomp, in); return; }
    // SCATTER multi-box : @p in est le champ GLOBAL (composante-majeur (c*gny + j)*gnx + i, meme layout
    // que copy_state). Chaque rang ecrit UNIQUEMENT les cellules de ses boites locales (lecture aux
    // indices globaux) -- aucune communication. Mono-rang : ecrit toutes les bandes.
    const int gnx = dom.nx(), gny = dom.ny();
    const std::size_t need = static_cast<std::size_t>(ncomp) * gnx * gny;
    if (in.size() != need)
      throw std::runtime_error("System::set_state : taille != ncomp*nr*ntheta (multi-box theta)");
    for (int li = 0; li < mf.local_size(); ++li) {
      Array4 u = mf.fab(li).array();
      const Box2D v = mf.box(li);
      for (int c = 0; c < ncomp; ++c)
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i)
            u(i, j, c) = in[(static_cast<std::size_t>(c) * gny + j) * gnx + i];
    }
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
    // DECOUPAGE THETA du transport (theta_boxes, ADC-67). 1 (defaut) = mono-box, bit-identique. > 1 :
    // bandes theta -- on exige 1 <= theta_boxes <= ntheta (au moins une cellule azimutale par bande) ET
    // theta_boxes DIVISE ntheta (bandes EGALES : le decoupage par boite ne doit pas dependre du reste,
    // et l'anneau periodique se recolle proprement). PolarMesh valide deja cote Python ; un appelant qui
    // construit le SystemConfig a la main est protege ici.
    const int nth = c.ntheta > 0 ? c.ntheta : c.n;
    if (c.theta_boxes < 1)
      throw std::runtime_error("System : geometry='polar' exige theta_boxes >= 1 (cf. adc.PolarMesh)");
    if (c.theta_boxes > nth)
      throw std::runtime_error(
          "System : geometry='polar' exige theta_boxes <= ntheta (au moins une cellule azimutale par "
          "bande) ; cf. adc.PolarMesh");
    if (nth % c.theta_boxes != 0)
      throw std::runtime_error(
          "System : geometry='polar' exige que theta_boxes DIVISE ntheta (bandes azimutales egales) ; "
          "cf. adc.PolarMesh");
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
                       const std::vector<std::string>& implicit_roles,
                       int newton_max_iters, double newton_rel_tol, double newton_abs_tol,
                       double newton_fd_eps, bool newton_diagnostics, double newton_damping,
                       const std::string& newton_fail_policy) {
  Impl* P = p_.get();
  if (substeps < 1) throw std::runtime_error("System::add_block : substeps >= 1");
  if (stride < 1) throw std::runtime_error("System::add_block : stride >= 1");
  if (newton_max_iters < 1)
    throw std::runtime_error("System::add_block : newton_max_iters >= 1");
  if (newton_rel_tol < 0.0 || newton_abs_tol < 0.0 || newton_fd_eps <= 0.0)
    throw std::runtime_error("System::add_block : newton_rel_tol/abs_tol >= 0 et newton_fd_eps > 0");
  if (!(newton_damping > 0.0 && newton_damping <= 1.0))
    throw std::runtime_error("System::add_block : newton_damping dans (0, 1]");
  int fail_policy = NewtonOptions::kFailNone;
  if (newton_fail_policy == "warn") fail_policy = NewtonOptions::kFailWarn;
  else if (newton_fail_policy == "throw") fail_policy = NewtonOptions::kFailThrow;
  else if (newton_fail_policy != "none")
    throw std::runtime_error("System::add_block : newton_fail_policy 'none'|'warn'|'throw' (recu '" +
                             newton_fail_policy + "')");
  // @p time porte le TRAITEMENT et, en explicite, le SCHEMA RK : "explicit"/"ssprk2" = SSPRK2
  // (defaut historique), "ssprk3" = SSPRK3 (ordre 3), "imex" = transport explicite + source raide
  // implicite backward-Euler local (ordre 1), "imexrk_ars222" = famille IMEX-RK schema ARS(2,2,2)
  // (ordre 2, avance PARALLELE distincte, cartesien seul). La math RK reste un FONCTEUR du coeur
  // (build_block). "imex" et "imexrk_ars222" partagent le drapeau @c imex ; @c method les distingue.
  if (time != "explicit" && time != "ssprk2" && time != "ssprk3" && time != "imex" &&
      time != "imexrk_ars222")
    throw std::runtime_error(
        "System::add_block : time 'explicit'|'ssprk2'|'ssprk3'|'imex'|'imexrk_ars222' (recu '" + time +
        "')");
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error("System::add_block : recon 'conservative' | 'primitive' (recu '" +
                             recon + "')");
  const bool imexrk = (time == "imexrk_ars222");
  const bool imex = (time == "imex" || imexrk);  // les deux passent par le pas implicite de source
  const bool recon_prim = (recon == "primitive");
  const std::string method = imexrk ? std::string("imexrk_ars222")
                                     : ((time == "ssprk3") ? std::string("ssprk3")
                                                           : std::string("ssprk2"));
  // Le masque implicite (implicit_vars / implicit_roles) ne s'applique qu'au pas de source IMEX. Le
  // demander en explicite est une ERREUR (pas d'ignore silencieux) : l'explicite n'a pas de pas implicite.
  if (!imex && (!implicit_vars.empty() || !implicit_roles.empty()))
    throw std::runtime_error("System::add_block : implicit_vars / implicit_roles exigent time='imex' "
                             "(le masque implicite ne s'applique qu'au pas de source IMEX ; recu time='" +
                             time + "')");
  // IMEX-RK ARS(2,2,2) : source PLEINEMENT implicite (la relation de coherence d'etage suppose un solve
  // homogene). Un masque partiel y serait SILENCIEUSEMENT ignore -> on le rejette explicitement. Le
  // masque partiel reste disponible sur time='imex' (backward-Euler local).
  if (imexrk && (!implicit_vars.empty() || !implicit_roles.empty()))
    throw std::runtime_error(
        "System::add_block : implicit_vars / implicit_roles (masque IMEX partiel) non supportes par "
        "time='imexrk_ars222' (sa source est PLEINEMENT implicite). Utiliser time='imex' pour un "
        "masque partiel, ou retirer implicit_vars / implicit_roles.");
  // Memes regles pour les options/diagnostics Newton : elles ne pilotent que le pas de source IMEX.
  // Des valeurs non-defaut en explicite seraient ignorees EN SILENCE -> erreur explicite.
  const bool newton_non_default = newton_max_iters != 2 || newton_rel_tol != 0.0 ||
                                  newton_abs_tol != 0.0 || newton_fd_eps != 1e-7 ||
                                  newton_diagnostics || newton_damping != 1.0 ||
                                  fail_policy != NewtonOptions::kFailNone;
  if (!imex && newton_non_default)
    throw std::runtime_error("System::add_block : les options Newton (newton_max_iters/rel_tol/"
                             "abs_tol/fd_eps/diagnostics) exigent time='imex' (recu time='" +
                             time + "')");

  int ncomp = 1;
  BlockClosures clo;
  std::function<Real(const MultiFab&)> max_speed;
  std::function<void(const MultiFab&, MultiFab&)> add_poisson_rhs;
  std::function<Real(const MultiFab&)> src_freq, stab_dt;  // bornes de pas optionnelles (traits modele)
  CellConvert prim_to_cons, cons_to_prim;  // conversions ponctuelles du modele (set/get_primitive_state)
  VariableSet cons_vs, prim_vs;
  if (P->polar_) {
    // CHEMIN POLAIRE (anneau) : fermetures bati par block_builder_polar.hpp (assemble_rhs_polar +
    // transport polaire scalaire ExBVelocityPolar OU fluide IsothermalFluxPolar + Poisson polaire
    // scalaire). IMEX n'est pas supporte sur l'anneau a cette etape : le couplage electrostatique
    // passe par une source LOCALE explicite (regime non raide, Voie A etape 1) ; on le refuse
    // explicitement plutot que de jouer le seul transport en silence.
    if (imex)
      throw std::runtime_error(
          "System::add_block (polaire) : time='" + time + "' (IMEX / IMEX-RK ARS(2,2,2)) non supporte "
          "(anneau : couplage par source locale explicite, pas de source raide a traiter en implicite "
          "a cette etape). Utiliser 'explicit'/'ssprk2'/'ssprk3'.");
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
      // StabilityPolicy POLAIRE (audit vague 3) : meme politique que le cartesien -- lambda* de
      // stabilite (trait) sinon max_wave_speed ; bornes source/pas admissible si declarees,
      // fermetures VIDES sinon (politique de pas historique, bit-identique).
      max_speed = make_cfl_speed_polar(m, &P->aux);
      src_freq = make_source_frequency_polar(m, &P->aux);
      stab_dt = make_stability_dt_polar(m, &P->aux);
      add_poisson_rhs = make_poisson_rhs_polar(m);
      auto conv = make_cell_convert(m);
      prim_to_cons = std::move(conv.first);
      cons_to_prim = std::move(conv.second);
    });
  } else {
  const GridContext ctx = P->grid_ctx();
  // Options du Newton de la source implicite IMEX (defauts = constantes historiques, bit-identique).
  // Le rapport (diagnostics OPT-IN) vit dans Impl::newton_reports_ en shared_ptr -> adresse STABLE
  // capturee par les fermetures meme quand sp realloue a un add_block ulterieur.
  NewtonOptions nopts;
  nopts.max_iters = newton_max_iters;
  nopts.rel_tol = static_cast<Real>(newton_rel_tol);
  nopts.abs_tol = static_cast<Real>(newton_abs_tol);
  nopts.fd_eps = static_cast<Real>(newton_fd_eps);
  nopts.damping = static_cast<Real>(newton_damping);
  nopts.fail_policy = fail_policy;
  NewtonReport* nreport = nullptr;
  if (newton_diagnostics) {
    auto rep = std::make_shared<NewtonReport>();
    P->newton_reports_[name] = rep;
    nreport = rep.get();
  }
  // Le modele est compose a partir des briques designees par la spec ; le visiteur cable les
  // fermetures (constructeurs en en-tete, instanciables AOT). ncomp = n_vars du modele compose ;
  // set_density s'y adapte. Les noms de variables viennent du descripteur Variables porte par le
  // modele (brique Vars), source unique de verite.
  detail::dispatch_model(model, [&](auto m) {
    using M = decltype(m);
    ncomp = M::n_vars;
    cons_vs = M::conservative_vars();  // noms + ROLES physiques (source unique de verite)
    prim_vs = M::primitive_vars();
    // LARGEUR AUX du modele compose (n_aux > 3 pour une brique magnetisee lisant B_z) : on elargit
    // le canal PARTAGE comme le fait add_compiled_model (dsl_block.hpp). Sans cet appel, load_aux<4>
    // lisait la composante 3 HORS BORNES d'un aux a 3 composantes (B_z silencieusement faux) -- le
    // chemin natif add_block + source 'magnetic' n'avait jamais ete exerce de bout en bout (audit
    // 2026-06, exposition Python des briques magnetiques). ensure_aux_width preserve l'ADRESSE de
    // l'aux (capturee par les fermetures via grid_ctx) et re-applique B_z deja fourni.
    P->ensure_aux_width(aux_comps<M>());
    // Masque implicite PORTE PAR LE BLOC : resout noms/roles -> indices contre le descripteur du bloc
    // (erreur explicite sur un nom/role absent). Vide -> make_implicit_mask inactif -> defaut modele
    // (bit-identique). Ne joue qu'en IMEX (garde ci-dessus pour l'explicite).
    const std::vector<int> impl_components =
        resolve_implicit_components(name, cons_vs, implicit_vars, implicit_roles);
    // Routage disque (chantier T5-PR3) : make_block fabrique AUSSI les avances disque (advance_masked /
    // advance_eb) car ctx (grid_ctx()) porte desormais ctx.disc_mask / ctx.disc (adresses stables des
    // membres de Impl). Elles restent INERTES tant que le System n'est pas mis en mode Staircase /
    // CutCell (cf. step()) : construites a l'ajout, selectionnees seulement sur opt-in.
    clo = make_block(m, limiter, riemann, ctx, imex, recon_prim, method, impl_components, nopts,
                     nreport);
    max_speed = make_max_speed(m, ctx);  // stability_speed (trait) ou max_wave_speed (fallback)
    add_poisson_rhs = make_poisson_rhs(m);
    // Bornes de pas optionnelles (traits HasSourceFrequency / HasStabilityDt) : fonctions VIDES si
    // le modele ne les declare pas -> step_cfl garde la politique historique (bit-identique).
    src_freq = make_source_frequency(m, ctx);
    stab_dt = make_stability_dt(m, ctx);
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
  set_block_dt_bounds(name, std::move(src_freq), std::move(stab_dt));
  // GHOSTS du schema : WENO5 lit un stencil 5 points (3 ghosts) > les 2 alloues par defaut dans
  // install_block. On reallue l'etat du bloc avec block_n_ghost(limiter) si besoin (cf. AmrSystem qui
  // alloue avec Limiter::n_ghost, PR #22) pour que fill_ghosts + assemble_rhs ne lisent pas hors
  // bornes. minmod/vanleer (2 ghosts) : no-op, allocation et resultat bit-identiques a avant.
  P->set_block_ghosts(name, block_n_ghost(limiter));
}

// Contexte de grille reel (maillage + CL + aux) : sert au gabarit add_compiled_model pour fabriquer
// les fermetures d'un modele compile AOT sur les vrais champs du System (parite native, sans marshaling).
ADC_EXPORT GridContext System::grid_context() { return p_->grid_ctx(); }

// Installe un bloc a partir de fermetures deja fabriquees (par dispatch_model cote add_block, ou par
// block_builder cote add_compiled_model). Centralise la creation de l'espece (U, noms, schema).
ADC_EXPORT void System::install_block(const std::string& name, int ncomp,
                                      const VariableSet& cons_vars,
                                      const VariableSet& prim_vars, double gamma,
                                      BlockClosures closures,
                                      std::function<Real(const MultiFab&)> max_speed,
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
  // Avances de transport DISQUE (chantier T5-PR3) : vides sauf si build_block les a fabriquees (bloc
  // cartesien avec disc_mask_/disc_ fournis). Vides -> le stepper retombe sur advance (bit-identique).
  P->sp.back().advance_masked = std::move(closures.advance_masked);
  P->sp.back().advance_eb = std::move(closures.advance_eb);
}

// Reallocation width-aware de l'etat d'un bloc (delegue a Impl::set_block_ghosts). Exposee
// (ADC_EXPORT) pour que le gabarit en-tete add_compiled_model (chemin natif, loader .so) puisse
// elargir le bloc compile a block_n_ghost(limiter) -- 3 pour weno5 -- comme le fait add_block.
ADC_EXPORT void System::set_block_ghosts(const std::string& name, int n_ghost) {
  p_->set_block_ghosts(name, n_ghost);
}

// Bornes de pas OPTIONNELLES d'un bloc (traits modele) : posees apres install_block, lues par
// step_cfl / step_adaptive. Fonctions vides = le bloc n'impose pas la borne (historique).
void System::set_block_dt_bounds(const std::string& name,
                                 std::function<Real(const MultiFab&)> source_frequency,
                                 std::function<Real(const MultiFab&)> stability_dt) {
  Impl::Species& s = p_->find(name);  // leve si bloc inconnu
  s.source_frequency = std::move(source_frequency);
  s.stability_dt = std::move(stability_dt);
}

// Borne GLOBALE de pas (hote, une evaluation par pas) : couplage multi-blocs, Schur/Poisson,
// scheduler, politique utilisateur. cf. SystemStepper::step_cfl pour l'agregation.
void System::add_dt_bound(const std::string& label, std::function<double()> fn) {
  if (!fn) throw std::runtime_error("System::add_dt_bound : fonction de borne vide");
  p_->dt_bounds_.push_back(Impl::GlobalDtBound{label, std::move(fn)});
}

// Borne ACTIVE du dernier step_cfl (diagnostic de la politique de pas). "" avant le premier pas.
std::string System::last_dt_bound() const { return p_->stepper_.last_dt_reason(); }

// Rapport Newton (diagnostics IMEX OPT-IN) du bloc : copie a plat du NewtonReport agrege par la
// DERNIERE avance du bloc (reset en tete d'avance par AdvanceImex*). Erreur claire si le bloc n'a
// pas active newton_diagnostics (pas de rapport silencieusement vide).
System::SourceNewtonReport System::newton_report(const std::string& name) const {
  p_->index(name);  // leve si bloc inconnu
  const auto it = p_->newton_reports_.find(name);
  if (it == p_->newton_reports_.end())
    throw std::runtime_error(
        "System::newton_report : diagnostics Newton non actives pour le bloc '" + name +
        "' ; ajouter le bloc avec newton_diagnostics=true (adc.IMEX(newton_diagnostics=True) / "
        "adc.SourceImplicit(newton_diagnostics=True))");
  const NewtonReport& r = *it->second;
  return SourceNewtonReport{r.enabled,
                            r.converged,
                            static_cast<double>(r.max_residual),
                            static_cast<double>(r.max_iters_used),
                            r.n_failed,
                            r.failed_i,
                            r.failed_j,
                            r.failed_comp};
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

// P7-b : ecrase le bloc PARTAGE des valeurs de parametres runtime du bloc @p name. add_compiled_block
// a enregistre ce vecteur dans p_->block_params_ ET l'a capture dans les fermetures du bloc : ecrire
// dedans suffit a changer le comportement au prochain pas, SANS recompiler le .so. Erreur explicite si
// le bloc n'a pas de params runtime (vecteur absent) ou si values n'a pas la bonne taille.
void System::set_block_params(const std::string& name, const std::vector<double>& values) {
  // index() leve "System : bloc inconnu '...'" si le bloc n'existe pas (meme diagnostic que partout).
  (void)p_->blocks_.index(name);
  auto it = p_->block_params_.find(name);
  if (it == p_->block_params_.end())
    throw std::runtime_error(
        "System::set_block_params : le bloc '" + name +
        "' n'a pas de parametre runtime (declarer dsl.Param(..., kind='runtime') et brancher via "
        "backend='aot' / add_compiled_block ; les params const sont figes a la compilation)");
  std::vector<double>& pv = *it->second;
  if (values.size() != pv.size())
    throw std::runtime_error(
        "System::set_block_params : le bloc '" + name + "' attend " + std::to_string(pv.size()) +
        " parametres runtime, recu " + std::to_string(values.size()));
  pv = values;  // le vecteur est PARTAGE avec les fermetures (shared_ptr) : effet au prochain pas
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

namespace {
// Traduit le mode de transport disque Python ("none"|"staircase"|"cutcell") en GeometryMode. Erreur
// EXPLICITE sur un mode inconnu (jamais un repli silencieux). Source unique de la table des noms.
GeometryMode parse_geometry_mode(const std::string& mode, const char* err_context) {
  if (mode == "none") return GeometryMode::None;
  if (mode == "staircase") return GeometryMode::Staircase;
  if (mode == "cutcell") return GeometryMode::CutCell;
  throw std::runtime_error(std::string(err_context) + " : mode geometrie inconnu '" + mode +
                           "' (none|staircase|cutcell)");
}
}  // namespace

void System::set_disc_domain(double cx, double cy, double R, const std::string& mode) {
  Impl* P = p_.get();
  // CARTESIEN seulement : le polaire borne deja l'anneau par ses parois radiales (r_min / r_max,
  // flux radial nul) -> un masque disque cartesien n'a pas de sens sur la grille (r, theta).
  if (P->polar_)
    throw std::runtime_error(
        "System::set_disc_domain : geometrie polaire (l'anneau est deja borne par ses parois "
        "radiales r_min/r_max ; le masque disque cartesien ne s'applique pas)");
  if (!(R > 0.0))
    throw std::runtime_error("System::set_disc_domain : rayon R > 0 requis");
  // Valide le mode AVANT toute mutation (un mode inconnu ne doit pas laisser le disque a moitie pose).
  const GeometryMode gmode = parse_geometry_mode(mode, "System::set_disc_domain");
  P->disc_ = detail::DiscDomain{cx, cy, R};
  P->disc_set_ = true;
  // Materialise le masque 0/1 cellule-centre (1 ghost, pour que le transport mask-aware lise les
  // voisins i-1/i+1/j-1/j+1 jusqu'au bord). Meme layout que les blocs (ba/dm). Cellule active quand
  // son CENTRE est dans le disque (level set < 0, MEME convention que le mur conducteur).
  P->disc_mask_ = MultiFab(P->ba, P->dm, 1, 1);
  const detail::DiscDomain disc = P->disc_;
  const Geometry geom = P->geom;
  for (int li = 0; li < P->disc_mask_.local_size(); ++li) {
    Array4 m = P->disc_mask_.fab(li).array();
    // boite AVEC ghosts : on classe aussi les ghosts (le transport mask-aware lit les voisins de bord).
    const Box2D g = P->disc_mask_.fab(li).grown_box();
    for_each_cell(g, [=] ADC_HD(int i, int j) {
      m(i, j, 0) = disc.cell_active(geom.x_cell(i), geom.y_cell(j)) ? Real(1) : Real(0);
    });
  }
  // AIGUILLAGE DU TRANSPORT (chantier T5-PR3). mode == "none" : le masque est materialise (consultable
  // via disc_mask()) mais le transport reste PLEIN cartesien -> bit-identique. mode != "none" : le
  // stepper route l'avance vers assemble_rhs_masked (staircase) / assemble_rhs_eb (cutcell).
  P->geometry_mode_ = gmode;
}

void System::set_geometry_mode(const std::string& mode) {
  Impl* P = p_.get();
  const GeometryMode gmode = parse_geometry_mode(mode, "System::set_geometry_mode");
  // Un mode disque (staircase/cutcell) n'a de sens qu'avec un disque fixe : sinon le stepper retomberait
  // sur le transport plein (le masque / level set n'existe pas), un footgun silencieux -> on refuse.
  if (gmode != GeometryMode::None && !P->disc_set_)
    throw std::runtime_error(
        "System::set_geometry_mode : mode '" + mode +
        "' demande sans disque fixe ; appeler set_disc_domain(cx, cy, R) d'abord");
  P->geometry_mode_ = gmode;
}

std::vector<double> System::disc_mask() const {
  Impl* P = p_.get();
  device_fence();
  const Box2D v = P->dom;
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
  if (!P->disc_set_) {
    // CONTRAT : sans disque fixe, le sous-domaine de transport est le domaine entier -> tout actif.
    out.assign(static_cast<std::size_t>(v.nx()) * v.ny(), 1.0);
    return out;
  }
  const ConstArray4 m = P->disc_mask_.fab(0).const_array();
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(static_cast<double>(m(i, j, 0)));
  return out;
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

ADC_EXPORT void System::ensure_aux_width(int ncomp) { p_->ensure_aux_width(ncomp); }

void System::set_magnetic_field(const std::vector<double>& bz) {
  // Taille attendue du champ B_z(x) row-major (axe lent = 2nd indice de box, axe rapide = 1er) :
  //   cartesien = n * n (carre, BIT-IDENTIQUE) ; POLAIRE = nr * ntheta (anneau, i = r rapide, cf.
  //   apply_bz / set_density polaire). Le layout est le MEME que set_density (flat[j * nr + i]).
  if (p_->polar_) {
    const int nr = Impl::polar_nr(p_->cfg), nth = Impl::polar_ntheta(p_->cfg);
    if (static_cast<int>(bz.size()) != nr * nth)
      throw std::runtime_error("System::set_magnetic_field : taille != nr*ntheta (polaire)");
  } else {
    const int n = p_->cfg.n;
    if (static_cast<int>(bz.size()) != n * n)
      throw std::runtime_error("System::set_magnetic_field : taille != n*n");
  }
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
    // MPI / multi-box-safe : iteration sur les fabs LOCAUX (local_size()==0 sur un rang sans box ->
    // no-op), MEME patron qu'add_coupled_source -- plus de fab(0)/box(0) en dur, qui n'existaient que
    // sur le rang possedant la box 0 et deviendraient faux si System passait multi-box. Les blocs
    // partagent la DistributionMapping du System -> meme local_size(), fabs co-localises.
    MultiFab& Ue = P->sp[ie].U;
    for (int li = 0; li < Ue.local_size(); ++li) {
      Array4 ue = Ue.fab(li).array();
      Array4 ui = P->sp[ii].U.fab(li).array();
      Array4 ug = P->sp[ig].U.fab(li).array();
      for_each_cell(Ue.box(li), [=] ADC_HD(int i, int j) {  // sur device (lit n_e, n_g)
        const Real dn = dt * k * ue(i, j, de) * ug(i, j, dg);
        ug(i, j, dg) -= dn;
        ui(i, j, di) += dn;
        ue(i, j, de) += dn;
      });
    }
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
    // MPI / multi-box-safe : fabs LOCAUX, meme patron qu'add_coupled_source (cf. add_ionization).
    MultiFab& Ua = P->sp[ia].U;
    for (int li = 0; li < Ua.local_size(); ++li) {
      Array4 ua = Ua.fab(li).array();
      Array4 ub = P->sp[ib].U.fab(li).array();
      for_each_cell(Ua.box(li), [=] ADC_HD(int i, int j) {  // sur device
        const Real fx = dt * k * (ua(i, j, mxa) / ua(i, j, da) - ub(i, j, mxb) / ub(i, j, db));
        ua(i, j, mxa) -= fx; ub(i, j, mxb) += fx;
        const Real fy = dt * k * (ua(i, j, mya) / ua(i, j, da) - ub(i, j, myb) / ub(i, j, db));
        ua(i, j, mya) -= fy; ub(i, j, myb) += fy;
      });
    }
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
    // MPI / multi-box-safe : fabs LOCAUX, meme patron qu'add_coupled_source (cf. add_ionization).
    MultiFab& Ua = P->sp[ia].U;
    for (int li = 0; li < Ua.local_size(); ++li) {
      Array4 ua = Ua.fab(li).array();
      Array4 ub = P->sp[ib].U.fab(li).array();
      for_each_cell(Ua.box(li), [=] ADC_HD(int i, int j) {  // sur device
        const Real ra = ua(i, j, da), rb = ub(i, j, db);
        const Real pa = (ga - Real(1)) * (ua(i, j, ea) -
            Real(0.5) * (ua(i, j, mxa) * ua(i, j, mxa) + ua(i, j, mya) * ua(i, j, mya)) / ra);
        const Real pb = (gb - Real(1)) * (ub(i, j, eb) -
            Real(0.5) * (ub(i, j, mxb) * ub(i, j, mxb) + ub(i, j, myb) * ub(i, j, myb)) / rb);
        const Real q = dt * k * (pa / ra - pb / rb);  // k (T_a - T_b), T = p/rho
        ua(i, j, ea) -= q;
        ub(i, j, eb) += q;
      });
    }
  });
}

void System::add_coupled_source(const std::vector<std::string>& in_blocks,
                                const std::vector<std::string>& in_roles,
                                const std::vector<double>& consts,
                                const std::vector<std::string>& out_blocks,
                                const std::vector<std::string>& out_roles,
                                const std::vector<int>& prog_ops,
                                const std::vector<int>& prog_args,
                                const std::vector<int>& prog_lens,
                                double frequency, const std::string& label,
                                const std::vector<int>& freq_prog_ops,
                                const std::vector<int>& freq_prog_args) {
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
  // Frequence PAR CELLULE optionnelle (CoupledSource.frequency avec une Expr, raffinement de la
  // frequence CONSTANTE) : un programme bytecode mu(U) sur la MEME table de registres que les termes
  // (entrees puis constantes). Valide ICI sa FORME (opcodes / registres bornes) AVANT tout push -- la
  // borne ne doit etre enregistree qu'apres une validation complete (regle anti-borne-fantome). Vide
  // (defaut) -> aucune frequence par cellule (chemin historique).
  const bool has_freq_expr = !freq_prog_ops.empty() || !freq_prog_args.empty();
  CsProgram freq_pg;
  if (has_freq_expr) {
    if (freq_prog_ops.size() != freq_prog_args.size())
      throw std::runtime_error("System::add_coupled_source : freq_prog_ops / freq_prog_args de tailles "
                               "differentes");
    if (static_cast<int>(freq_prog_ops.size()) > kCsMaxProg)
      throw std::runtime_error("System::add_coupled_source : programme de frequence trop long (> " +
                               std::to_string(kCsMaxProg) + ")");
    freq_pg.len = static_cast<int>(freq_prog_ops.size());
    for (int k = 0; k < freq_pg.len; ++k) {
      const int opc = freq_prog_ops[static_cast<std::size_t>(k)];
      const int a = freq_prog_args[static_cast<std::size_t>(k)];
      if (opc < 0 || opc > static_cast<int>(CsOp::Sqrt))
        throw std::runtime_error("System::add_coupled_source : opcode invalide dans la frequence");
      if (opc == static_cast<int>(CsOp::PushReg) && (a < 0 || a >= n_in + n_const))
        throw std::runtime_error("System::add_coupled_source : registre hors bornes dans la frequence");
      freq_pg.op[k] = opc;
      freq_pg.arg[k] = a;
    }
  }
  // Frequence CONSTANTE declaree du couplage (audit vague 3) : enregistree pour la borne de pas de
  // step_cfl / step_adaptive (dt <= cfl/mu sur le MACRO-pas). <= 0 = pas de borne (historique). Poussee
  // APRES toute la validation (source ET frequence ont leve si invalides) : un couplage rejete ne doit
  // laisser AUCUNE borne fantome -- sinon un script qui try/except l'echec garderait un pas bride sans
  // physique correspondante.
  if (frequency > 0.0) P->coupled_freqs_.push_back(Impl::CoupledFreq{label, frequency});
  // Frequence PAR CELLULE : meme regle (push apres validation complete). Les entrees REUTILISENT la
  // resolution resolve() (ins) ; les constantes sont les memes que la source (kconsts). Le programme
  // mu(U) est reduit (MAX) a chaque pas dans step_cfl / step_adaptive.
  if (has_freq_expr) {
    Impl::CoupledFreqExpr ce;
    ce.label = label;
    ce.prog = freq_pg;
    ce.n_in = n_in;
    ce.ins.resize(static_cast<std::size_t>(n_in));
    for (int c = 0; c < n_in; ++c)
      ce.ins[static_cast<std::size_t>(c)] = {ins[static_cast<std::size_t>(c)].sidx,
                                             ins[static_cast<std::size_t>(c)].comp};
    ce.kconsts = kconsts;
    P->coupled_freq_exprs_.push_back(std::move(ce));
  }
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
                              double alpha, double krylov_tol, int krylov_max_iters,
                              const std::string& density, const std::string& momentum_x,
                              const std::string& momentum_y, const std::string& energy,
                              int bz_aux_component) {
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
  // Tolerance / budget du solve Krylov de l'etage (audit 2026-06 : les constantes 1e-10 / 400 (cart)
  // / 600 (polaire) ne sont plus figees). krylov_tol <= 0 / krylov_max_iters <= 0 = "defaut
  // historique du stepper" (on ne touche pas au reglage du stepper construit).
  if (krylov_tol > 0.0 && !(krylov_tol < 1.0))
    throw std::runtime_error("System::set_source_stage : krylov_tol doit etre dans (0, 1)");
  // GEOMETRIE : l'etage source condense est cable en CARTESIEN (CondensedSchurSourceStepper, #126) ET en
  // POLAIRE (PolarCondensedSchurSourceStepper, #212, Voie A etape 2c). Le dispatch ci-dessous construit le
  // stepper adapte a la geometrie du System. Toute autre geometrie est REJETEE explicitement (pas
  // d'ignore silencieux).
  const bool polar = (P->cfg.geometry == "polar");
  if (P->cfg.geometry != "cartesian" && !polar)
    throw std::runtime_error("System::set_source_stage : etage source condense supporte les geometries "
                             "cartesienne et polaire (recu '" + P->cfg.geometry + "')");
  // L'etage source condense POLAIRE est desormais MULTI-RANG MPI (PolarTensorKrylovSolver / Schur
  // polaire distribues par decoupage AZIMUTAL ; garde-fou de layout check_radial_columns dans le
  // solveur). Cote FACADE, le System construit pour l'instant UNE box couvrant l'anneau (P->ba mono-box),
  // donc sous MPI la box vit sur rang 0 et les autres rangs ont local_size()==0 : le solve reste CORRECT
  // (collectifs dot/project_mean appeles sur tous les rangs, contributions nulles des rangs vides) et
  // BIT-IDENTIQUE au mono-rang, mais sans parallelisme reel a ce niveau. Le decoupage theta effectif
  // (vrai scaling multi-rang) s'exerce au niveau de l'API C++ (PolarCondensedSchurSourceStepper avec un
  // BoxArray decoupe en theta) ; la repartition theta cote facade est differee (Extend). Aucun garde-fou
  // mono-rang ici : le PolarTensorKrylovSolver leve une erreur claire si jamais le layout coupe r.
  // CONTRAT roles : le bloc doit exposer Density / MomentumX / MomentumY (Energy optionnel). On lit le
  // descripteur CONSERVATIF du bloc (peuple par add_block / les .so a roles, dont le DSL compile qui
  // declare les electrons en roles). Un role requis absent leve une erreur EXPLICITE ICI (avant le pas)
  // -- le constructeur du stepper la leverait aussi, mais on diagnostique cote bloc nomme.
  const VariableSet& vs = s.cons_vars;
  // RESOLUTION DES DESCRIPTEURS (audit vague 2 : roles/champs transportes dans l'ABI). Un
  // descripteur VIDE = role canonique (historique, bit-identique). Sinon : nom de ROLE stable
  // d'abord (role_from_name), puis nom de VARIABLE du bloc. Echec = erreur explicite avec remede.
  auto resolve_field = [&](const std::string& spec, VariableRole canonical,
                           const char* label) -> int {
    if (spec.empty()) {
      const int idx = vs.index_of(canonical);
      if (idx < 0)
        throw std::runtime_error(
            "System::set_source_stage : le bloc '" + name + "' n'expose pas le role " + label +
            " requis par adc.CondensedSchur (le modele doit declarer Density / MomentumX / "
            "MomentumY ; Energy optionnel), et aucun descripteur explicite n'est fourni (passer "
            "density=/momentum=... avec un nom de role ou de variable du bloc).");
      return idx;
    }
    const VariableRole r = role_from_name(spec);
    if (r != VariableRole::Custom) {
      const int idx = vs.index_of(r);
      if (idx < 0)
        throw std::runtime_error("System::set_source_stage : le bloc '" + name +
                                 "' n'expose pas le role '" + spec + "' (" + label + ")");
      return idx;
    }
    for (std::size_t i = 0; i < vs.names.size(); ++i)
      if (vs.names[i] == spec) return static_cast<int>(i);
    throw std::runtime_error("System::set_source_stage : '" + spec +
                             "' n'est ni un role stable ni une variable du bloc '" + name +
                             "' (" + label + ")");
  };
  const int c_rho = resolve_field(density, VariableRole::Density, "Density");
  const int c_mx = resolve_field(momentum_x, VariableRole::MomentumX, "MomentumX");
  const int c_my = resolve_field(momentum_y, VariableRole::MomentumY, "MomentumY");
  const int c_E = (energy == "none")
                      ? -1
                      : (energy.empty() ? vs.index_of(VariableRole::Energy)
                                        : resolve_field(energy, VariableRole::Energy, "Energy"));
  // B_z OBLIGATOIRE : l'etage de Lorentz lit Omega = B_z. On exige set_magnetic_field appele
  // (bz_field_ renseigne) et on elargit le canal aux au canal B_z (kAuxBaseComps) pour que apply_bz le
  // peuple et que solve_fields en remplisse les ghosts. Un B_z absent leve une erreur EXPLICITE.
  if (P->fields_.bz_field_.empty())
    throw std::runtime_error(
        "System::set_source_stage : le bloc '" + name + "' n'a pas de champ B_z (aux Omega) ; "
        "adc.CondensedSchur exige set_magnetic_field(B_z) (le terme de Lorentz lit Omega = B_z).");
  // Canal aux du champ magnetique : canonique (kAuxBaseComps) par defaut, redirigeable par
  // bz_aux_component (descripteur transporte). NOTE : apply_bz peuple le canal CANONIQUE ; une
  // composante differente suppose que l'appelant la peuple lui-meme (champ aux derive/custom).
  const int c_bz = bz_aux_component >= 0 ? bz_aux_component : kAuxBaseComps;
  P->ensure_aux_width(c_bz + 1);  // garantit le canal dans l'aux partage + re-applique B_z
  // Construit l'etage source condense sur le layout REEL du System (ba/dm/geom) avec la CL du Poisson.
  // Le stepper alloue ses tampons UNE fois ; step() les reutilise (cf. son cycle de vie). alpha =
  // constante de couplage electrostatique du sous-systeme source.
  if (polar) {
    // POLAIRE (Voie A etape 2c) : PolarCondensedSchurSourceStepper sur l'anneau pgeom_, MEME CL Poisson
    // (Dirichlet/Neumann radiale, theta toujours periodique cote solveur). Preconditionneur RadialLine
    // (defaut). run_source_stage l'invoque exactement comme le cartesien (signature step() identique).
    // schur reste nullptr (chemin cartesien intouche). Composantes EXPLICITES resolues ci-dessus
    // (descripteurs vides -> roles canoniques -> bit-identique) : le stepper POLAIRE accepte les
    // overrides depuis la vague 3 (ctor a composantes explicites, parite cartesien).
    s.schur_polar = std::make_shared<PolarCondensedSchurSourceStepper>(
        vs, c_rho, c_mx, c_my, c_E, P->pgeom_, P->ba, P->fields_.poisson_bc(),
        static_cast<Real>(alpha));
    if (krylov_tol > 0.0 || krylov_max_iters > 0)
      s.schur_polar->set_krylov(krylov_tol > 0.0 ? static_cast<Real>(krylov_tol) : Real(1e-10),
                                krylov_max_iters > 0 ? krylov_max_iters : 600);
  } else {
    // CARTESIEN (#126) : composantes EXPLICITES resolues ci-dessus (descripteurs vides -> roles
    // canoniques -> memes indices que l'historique, bit-identique).
    s.schur = std::make_shared<CondensedSchurSourceStepper>(vs, c_rho, c_mx, c_my, c_E, P->geom,
                                                            P->ba, P->fields_.poisson_bc(),
                                                            static_cast<Real>(alpha));
    if (krylov_tol > 0.0 || krylov_max_iters > 0)
      s.schur->set_krylov(krylov_tol > 0.0 ? static_cast<Real>(krylov_tol) : Real(1e-10),
                          krylov_max_iters > 0 ? krylov_max_iters : 400);
  }
  s.schur_bz_comp = c_bz;
  s.schur_theta = theta;
}

void System::set_time_scheme(const std::string& scheme) {
  // Aiguille la politique de splitting du stepper de systeme (defaut Lie = bit-identique). Le schema
  // Strang reutilise les MEMES briques (s.advance pour les demi-avances de transport, run_source_stage
  // pour l'etage source plein) ; il RE-RESOUT solve_fields entre les etages (cf. SystemStepper::step_strang
  // et docs/HOFFART_STEP_SEQUENCE.md). Un schema inconnu leve une erreur EXPLICITE (pas d'ignore silencieux).
  if (scheme == "lie") {
    p_->stepper_.set_scheme(stepper::SplitScheme::Lie);
  } else if (scheme == "strang") {
    p_->stepper_.set_scheme(stepper::SplitScheme::Strang);
  } else {
    throw std::runtime_error("System::set_time_scheme : schema '" + scheme +
                             "' inconnu (attendu 'lie' ou 'strang')");
  }
}

void System::set_gauss_policy(const std::string& policy) {
  // Politique de la loi de Gauss (chantier R0, reproduction Hoffart). "restart" (defaut) : solve_fields
  // re-resout -Delta phi = f a chaque pas (bit-identique a l'historique). "evolve" : apres le premier
  // solve (phi^0), solve_fields NE re-resout plus le Poisson ; il derive l'aux du phi COURANT que
  // l'etage source Schur fait evoluer in-place dans ell_phi() -> evolution -Delta phi sans restart du
  // papier (la contrainte de Gauss n'est imposee qu'a t=0). N'a d'effet QU'avec un etage source condense
  // (sans lui phi resterait gele apres t=0). Le verrou gauss_solved_once_ est remis a zero ici pour
  // qu'un changement de politique AVANT le premier solve reste coherent (le 1er solve resout toujours).
  if (policy == "restart") {
    p_->fields_.gauss_evolve_ = false;
  } else if (policy == "evolve") {
    p_->fields_.gauss_evolve_ = true;
  } else {
    throw std::runtime_error("System::set_gauss_policy : politique '" + policy +
                             "' inconnue (attendu 'restart' ou 'evolve')");
  }
  p_->fields_.gauss_solved_once_ = false;
}

void System::set_density(const std::string& name, const std::vector<double>& rho) {
  Impl::Species& s = p_->find(name);
  const Real gm1 = Real(s.gamma) - Real(1);
  // Helper local : pose densite + etat au repos sur UNE cellule (memes formules que l'historique).
  auto set_cell = [&](Array4& u, int i, int j, Real r) {
    u(i, j, 0) = r;
    if (s.ncomp >= 3) { u(i, j, 1) = 0; u(i, j, 2) = 0; }  // qte de mvt au repos
    if (s.ncomp == 4) u(i, j, 3) = r / gm1;                // E = p/(g-1), p = rho
  };
  // MULTI-BOX (theta_boxes > 1, polaire) : @p rho est le champ GLOBAL (nr x ntheta, layout flat[j*gnx+i]
  // identique au mono-box ci-dessous). On ecrit chaque box locale a ses indices GLOBAUX. local_size() <= 1
  // (cartesien / polaire mono-box, y compris MPI mono-box) : chemin historique INCHANGE, bit-identique.
  if (s.U.local_size() > 1) {
    const int gnx = p_->dom.nx(), gny = p_->dom.ny();
    if (static_cast<int>(rho.size()) != gnx * gny)
      throw std::runtime_error("System::set_density : taille != nr*ntheta (multi-box theta)");
    for (int li = 0; li < s.U.local_size(); ++li) {
      Array4 u = s.U.fab(li).array();
      const Box2D b = s.U.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          set_cell(u, i, j, rho[static_cast<std::size_t>(j) * gnx + i]);
    }
    return;
  }
  // Layout row-major du tableau d'entree : (ni x nj) = extents de la box de l'etat. En cartesien
  // ni = nj = cfg.n (indexation et taille bit-identiques a avant). En polaire ni = nr, nj = ntheta :
  // on indexe par les extents reels de la box (et non n*n), donc nr != ntheta est correctement gere.
  const Box2D v = s.U.box(0);
  const int ni = v.nx(), nj = v.ny();
  if (static_cast<int>(rho.size()) != ni * nj)
    throw std::runtime_error("System::set_density : taille != nr*ntheta (ou n*n en cartesien)");
  Array4 u = s.U.fab(0).array();
  // CONVENTION DE LAYOUT (inchangee vs l'historique) : axe lent = 2nd indice de box (j), axe rapide =
  // 1er (i), i.e. flat[(j-lo) * ni + (i-lo)]. En cartesien ni = n, lo = 0 -> flat[j*n+i] (bit-identique
  // a avant). En polaire le tableau est donc (nr, ntheta) ligne-par-ligne radiale (i = r lent par
  // rapport a... non : j = theta lent, i = r rapide), MEME ordre que density()/copy_comp0 -> coherent.
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i)
      set_cell(u, i, j, rho[static_cast<std::size_t>(j - v.lo[1]) * ni + (i - v.lo[0])]);
}

ADC_EXPORT void System::set_block_conversion(const std::string& name, CellConvert prim_to_cons,
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

// Horloge du systeme (IO v1, audit vague 2) : macro_step est REQUIS par le restart (la cadence
// stride hold-then-catch-up lit macro_step % stride ; t seul ne suffit pas).
int System::macro_step() const { return p_->macro_step_; }

// Restauration du potentiel phi (IO v1, restart) : ecrit les cellules VALIDES de la composante 0 du
// phi du solveur (warm start multigrille ; etat physique en gauss_policy="evolve"). Mono-box
// (meme convention de marshaling que potential / set_density).
void System::set_potential(const std::vector<double>& phi) {
  Impl* P = p_.get();
  device_fence();
  if (P->polar_) {
    P->fields_.ensure_elliptic_polar();
    MultiFab& ph = P->fields_.pell_->phi();
    // Rang sans box (MPI mono-box) : NO-OP (le rang proprietaire restaure phi). Permet restart sur
    // tous les rangs avec le champ GLOBAL. Mono-rang : local_size()==1, INCHANGE.
    if (ph.local_size() == 0) return;
    const Box2D v = ph.box(0);
    if (static_cast<int>(phi.size()) != v.nx() * v.ny())
      throw std::runtime_error("System::set_potential : taille != nr*ntheta");
    Array4 a = ph.fab(0).array();
    std::size_t k = 0;
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) a(i, j, 0) = phi[k++];
    return;
  }
  P->fields_.ensure_elliptic();
  MultiFab& ph = P->fields_.ell_phi();
  if (ph.local_size() == 0) return;  // rang sans box : no-op (cf. branche polaire)
  const Box2D v = ph.box(0);
  if (static_cast<int>(phi.size()) != v.nx() * v.ny())
    throw std::runtime_error("System::set_potential : taille != n*n");
  Array4 a = ph.fab(0).array();
  std::size_t k = 0;
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) a(i, j, 0) = phi[k++];
}
void System::set_clock(double t, int macro_step) {
  if (macro_step < 0)
    throw std::runtime_error("System::set_clock : macro_step >= 0 (restart)");
  p_->t = t;
  p_->macro_step_ = macro_step;
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
// Axe LENT du champ (lignes du tableau (ny, nx)). On le lit du domaine d'INDICES (dom = nx() x ny()),
// SOURCE UNIQUE des extents pour les deux geometries : cartesien dom = n x n -> ny() == nx() == n (carre,
// INCHANGE) ; polaire dom = nr x ntheta -> nx() == nr (rapide, i), ny() == ntheta (lent, j). C'est cette
// dimension qui dimensionne le tableau numpy cote bindings : un champ polaire fait nx()*ny() = nr*ntheta
// valeurs, et avec nr != ntheta le remodelage carre (nx, nx) deborde le tampon (bug de teardown).
int System::ny() const { return p_->dom.ny(); }
double System::time() const { return p_->t; }
int System::n_species() const { return p_->blocks_.size(); }
std::vector<std::string> System::block_names() const {
  // Registre de blocs UNIQUE (store), peuple par tous les chemins d'ajout : un bloc charge via
  // add_dynamic_block / add_compiled_block (.so) y figure au meme titre qu'un add_block.
  return p_->blocks_.names();
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
    // Rang sans box (MPI mono-box) : retour VIDE (pas de fab(0)). Cf. copy_comp0 ; le champ global
    // multi-rangs passe par System::potential_global.
    if (p_->aux.local_size() == 0) return {};
    const ConstArray4 ph = p_->fields_.pell_->phi().fab(0).const_array();
    const Box2D v = p_->aux.box(0);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(ph(i, j));
    return out;
  }
  p_->fields_.ensure_elliptic();
  if (p_->aux.local_size() == 0) return {};  // rang sans box : vide (cf. potential_global)
  const ConstArray4 ph = p_->fields_.ell_phi().fab(0).const_array();
  const Box2D v = p_->aux.box(0);
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(ph(i, j));
  return out;
}

// --- accesseurs GLOBAUX (collectifs MPI-safe), IO v1 multi-rangs --------------------------------
// Patron commun (cf. en-tete system.hpp) : tampon GLOBAL de taille gny*gnx (ou nc*gny*gnx) initialise
// a 0, rempli par les fabs LOCAUX en INDICES GLOBAUX (la box porte ses indices globaux ; un rang sans
// box -> local_size()==0 -> aucune ecriture), puis all_reduce_sum_inplace : chaque cellule etant
// possedee par EXACTEMENT un rang (boites disjointes), la somme = le champ global EXACT sur chaque
// rang. Mono-rang : la box couvre tout le domaine et all_reduce = identite -> tableau bit-identique
// aux accesseurs non globaux (density / get_state / potential). Layout IDENTIQUE (density : j*gnx + i ;
// state : (c*gny + j)*gnx + i, composante-majeur ; cf. copy_comp0 / copy_state).
std::vector<double> System::density_global(const std::string& name) const {
  device_fence();
  const Impl::Species& s = p_->find(name);
  const int gnx = nx(), gny = ny();
  std::vector<double> out(static_cast<std::size_t>(gnx) * gny, 0.0);
  for (int li = 0; li < s.U.local_size(); ++li) {
    const ConstArray4 u = s.U.fab(li).const_array();
    const Box2D v = s.U.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        out[static_cast<std::size_t>(j) * gnx + i] = static_cast<double>(u(i, j, 0));
  }
  all_reduce_sum_inplace(out.data(), static_cast<int>(out.size()));
  return out;
}
std::vector<double> System::state_global(const std::string& name) const {
  device_fence();
  const Impl::Species& s = p_->find(name);
  const int nc = s.ncomp, gnx = nx(), gny = ny();
  std::vector<double> out(static_cast<std::size_t>(nc) * gnx * gny, 0.0);
  for (int li = 0; li < s.U.local_size(); ++li) {
    const ConstArray4 u = s.U.fab(li).const_array();
    const Box2D v = s.U.box(li);
    for (int c = 0; c < nc; ++c)
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          out[(static_cast<std::size_t>(c) * gny + j) * gnx + i] = static_cast<double>(u(i, j, c));
  }
  all_reduce_sum_inplace(out.data(), static_cast<int>(out.size()));
  return out;
}
std::vector<double> System::potential_global() {
  device_fence();
  const int gnx = nx(), gny = ny();
  std::vector<double> out(static_cast<std::size_t>(gnx) * gny, 0.0);
  // Resout le Poisson (polaire ou cartesien) si besoin : COLLECTIF, comme potential_global le tout.
  const MultiFab* phi = nullptr;
  if (p_->polar_) {
    p_->fields_.ensure_elliptic_polar();
    phi = &p_->fields_.pell_->phi();
  } else {
    p_->fields_.ensure_elliptic();
    phi = &p_->fields_.ell_phi();
  }
  for (int li = 0; li < phi->local_size(); ++li) {
    const ConstArray4 ph = phi->fab(li).const_array();
    const Box2D v = phi->box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        out[static_cast<std::size_t>(j) * gnx + i] = static_cast<double>(ph(i, j));
  }
  all_reduce_sum_inplace(out.data(), static_cast<int>(out.size()));
  return out;
}

// --- accesseurs LOCAUX par fab (NON collectifs) : ecriture HDF5 parallele par hyperslabs (PR-IO-3) --
// Pendant local des accesseurs _global : ils n'agregent rien (aucune comm MPI), ils exposent par rang
// les boites LOCALES (en indices GLOBAUX, tels que portes par la box du fab) et l'etat de chaque fab.
// La facade sim.write(format='hdf5', parallel=True) cree les datasets globaux puis chaque rang ecrit
// SES boites en hyperslabs. Un rang sans box -> local_size()==0 -> liste vide (jamais de fab(0) en dur).
std::vector<std::array<int, 4>> System::local_boxes(const std::string& name) const {
  device_fence();
  const Impl::Species& s = p_->find(name);
  std::vector<std::array<int, 4>> out;
  out.reserve(s.U.local_size());
  for (int li = 0; li < s.U.local_size(); ++li) {
    const Box2D v = s.U.box(li);
    out.push_back({v.lo[0], v.lo[1], v.hi[0], v.hi[1]});  // (ilo, jlo, ihi, jhi) GLOBAUX
  }
  return out;
}
std::vector<double> System::local_state(const std::string& name, int li) const {
  device_fence();
  const Impl::Species& s = p_->find(name);
  if (li < 0 || li >= s.U.local_size())
    throw std::out_of_range("System::local_state : indice de fab local hors bornes (0.." +
                            std::to_string(s.U.local_size() - 1) + ")");
  const int nc = s.ncomp;
  const ConstArray4 u = s.U.fab(li).const_array();
  const Box2D v = s.U.box(li);
  const int bnx = v.nx(), bny = v.ny();  // dimensions de la box LOCALE (cellules valides)
  std::vector<double> out(static_cast<std::size_t>(nc) * bnx * bny, 0.0);
  // Layout = state_global rapporte a la box locale : (c*bny + jl)*bnx + il, composante-majeur, donc
  // remodelable en (nc, bny, bnx) pour un hyperslab dset[:, jlo:jhi+1, ilo:ihi+1].
  for (int c = 0; c < nc; ++c)
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        out[(static_cast<std::size_t>(c) * bny + (j - v.lo[1])) * bnx + (i - v.lo[0])] =
            static_cast<double>(u(i, j, c));
  return out;
}

}  // namespace adc
