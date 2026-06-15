#pragma once

#include <adc/mesh/patch_box.hpp>    // PatchBox : empreinte index-space d'un patch fin (patch_boxes())
#include <adc/mesh/physical_bc.hpp>  // BCRec
#include <adc/numerics/time/implicit_stepper.hpp>  // NewtonOptions (options du Newton de la source IMEX)
#include <adc/runtime/export.hpp>    // ADC_EXPORT : set_compiled_block resolu par le loader natif AMR
#include <adc/runtime/model_spec.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

/// @file
/// @brief Composition multi-especes sur AMR a l'execution : le pendant raffine de System.
///
/// Un ou PLUSIEURS blocs (especes, decrites par des ModelSpec de briques generiques) portes sur une
/// hierarchie AMR. Comme System mais sur grille adaptative.
///
/// MONO-BLOC (1 add_block) : un AmrCouplerMP<Model> mono-modele (grossier + un niveau fin suivi par
/// regrid, reflux conservatif). Chemin historique, INTOUCHE -> bit-identique.
///
/// MULTI-BLOCS (>= 2 add_block, capstone, docs/AMR_MULTIBLOCK_DESIGN.md) : N blocs co-localises sur
/// UNE hierarchie AMR PARTAGEE (meme BoxArray + DistributionMapping + dx/dy par niveau, garde
/// same_layout_or_throw). Tous les blocs vivent sur TOUS les patchs. Un seul aux par niveau (phi,
/// grad phi) et un seul Poisson grossier dont le second membre est la SOMME CO-LOCALISEE des briques
/// elliptiques des blocs (f = Sum_b q_b n_b lu aux memes cellules). Conservation PAR BLOC (reflux +
/// average_down). Moteur runtime AmrRuntime (registre type-erase par nom). Blocs a schemas spatiaux
/// potentiellement DIFFERENTS et a TRAITEMENT TEMPOREL par bloc (explicite ou IMEX, source raide
/// implicite locale ; capstone vii) sur une hierarchie FIGEE (multi-blocs + regrid_every > 0 est
/// REFUSE : le regrid d'union des tags est une PR ulterieure). Multirate (substeps/stride), sources
/// couplees inter-especes : deja cables. Blocs COMPILES (add_compiled_model) multiples et MELANGE
/// compile + natif : cables (capstone v, DSL production multi-bloc). Regrid d'union : PR ulterieure.
///
/// @note Deux niveaux (ratio 2) ; traitement temporel explicite OU imex (par bloc).

namespace adc {

// Declarations anticipees du moteur multi-blocs runtime (definitions dans amr_runtime.hpp /
// amr_dsl_block.hpp). set_compiled_block stocke un BUILDER DIFFERE de bloc runtime qui, recevant le
// layout PARTAGE materialise au build paresseux, rend l'AmrRuntimeBlock type-erase du bloc compile :
// c'est ce qui permet a PLUSIEURS blocs compiles (DSL production multi-bloc) de co-exister sur la
// MEME hierarchie AMR, exactement comme add_block en multi-blocs natif. On forward-declare pour ne
// PAS alourdir ce header public (lu par bindings.cpp et les loaders) avec amr_runtime.hpp : seules
// les TU qui CONSTRUISENT/APPELLENT le builder (amr_dsl_block.hpp et python/amr_system.cpp) incluent
// les definitions completes ; un std::function de signature a types incomplets est licite tant qu'on
// ne l'instancie pas avec un callable concret hors de ces TU (recette PIMPL std::function).
struct AmrRuntimeBlock;
namespace detail {
struct SharedAmrLayout;
}

/// Maillage et cadence AMR (parametres physiques par bloc, dans la ModelSpec).
struct AmrSystemConfig {
  int n = 128;            ///< cellules du niveau grossier par direction
  double L = 1.0;         ///< taille du domaine carre [0,L]^2
  int regrid_every = 20;  ///< re-raffinement tous les N pas (0 = jamais apres l'init)
  bool periodic = true;   ///< domaine periodique
  /// POLITIQUE D'OWNERSHIP du niveau grossier (cf. AmrCouplerMP::replicated_coarse).
  /// false (DEFAUT, historique) : grossier mono-box REPLIQUE sur tous les rangs. Le Poisson
  ///   grossier et le transport grossier sont REDONDANTS sur chaque GPU (zero communication,
  ///   meilleur MG geometrique) mais NE SCALENT PAS : seuls les patchs fins se repartissent.
  /// true (mode strong-scaling) : grossier MULTI-BOX (BoxArray::from_domain, taille de tuile
  ///   coarse_max_grid) REPARTI round-robin sur les rangs. Le Poisson grossier et le transport
  ///   grossier se distribuent (chaque rang ne porte que ses tuiles), ce qui leve la redondance
  ///   et permet le strong-scaling AMR. Le MG geometrique opere alors sur un grossier multi-box
  ///   (cf. geometric_mg.hpp) : convergence a mesurer (peut demander plus de cycles).
  bool distribute_coarse = false;
  /// Taille de tuile du grossier quand distribute_coarse=true (BoxArray::from_domain). 0 => n/2
  /// (decoupage minimal 2x2, le moins agressif pour le MG). Ignore si distribute_coarse=false.
  int coarse_max_grid = 0;
};

/// Parametres figes passes au build differe du chemin compile (add_compiled_model). Materialises
/// par AmrSystem au moment de ensure_built : la geometrie + les choix refine/poisson/density connus
/// a ce moment-la. Le header amr_dsl_block les consomme pour instancier AmrCouplerMP<Model>.
struct AmrBuildParams {
  int n = 128;
  double L = 1.0;
  int regrid_every = 20;
  double gamma = 1.4;
  int substeps = 1;
  bool recon_prim = false;            ///< recon == "primitive" (fige par add_compiled_model)
  bool imex = false;                  ///< time == "imex" : source raide implicite (backward_euler)
  double refine_threshold = 1e30;     ///< 1e30 => aucun raffinement
  BCRec poisson_bc;                   ///< CL Poisson grossier (resolue par set_poisson)
  std::function<bool(Real, Real)> wall;  ///< predicat paroi conductrice (vide = aucune)
  bool has_density = false;
  std::vector<double> density;        ///< densite initiale grossiere (composante 0), n*n
  bool distribute_coarse = false;     ///< grossier multi-box reparti (strong-scaling AMR)
  int coarse_max_grid = 0;            ///< taille de tuile du grossier reparti (0 => n/2)
  // Etat conservatif initial COMPLET (toutes les composantes), prioritaire sur `density` quand
  // has_state. AJOUTE EN FIN DE STRUCT : les offsets des champs precedents sont inchanges, donc un
  // loader .so mono-bloc anterieur (qui COPIE bp dans son propre layout puis ne lit pas ces champs)
  // retombe SILENCIEUSEMENT sur le chemin densite historique -- pas de corruption (append-only).
  bool has_state = false;
  std::vector<double> state;          ///< ncomp*n*n, composante-majeur c*n*n + j*n + i ; ncomp == Model::n_vars
  // ETAGE SOURCE condense par Schur (chemin amr-schur, pendant de System::set_source_stage). AJOUTE EN
  // FIN DE STRUCT (append-only, meme raison que has_state : un loader .so anterieur ne lit pas ces
  // champs et retombe sur le chemin explicite/imex historique). schur==false -> chemin inchange.
  bool schur = false;                 ///< true : etage source condense GLOBAL (au lieu d'explicit/imex local)
  double schur_theta = 0.5;           ///< theta-schema de l'etage condense (0.5 = Crank-Nicolson)
  double schur_alpha = 1.0;           ///< constante de couplage electrostatique de l'etage condense
  bool schur_strang = false;          ///< true : splitting Strang H(dt/2) S(dt) H(dt/2) ; false : Lie H(dt) S(dt)
  std::vector<double> bz_field;       ///< champ B_z(x,y) du grossier, n*n row-major (exige par l'etage condense)
  // Reglages de l'etage condense TRANSPORTES par l'ABI (audit vague 3, append-only comme has_state).
  double schur_krylov_tol = 0.0;      ///< tolerance du solve Krylov grossier (<= 0 = defaut 1e-10)
  int schur_krylov_max_iters = 0;     ///< budget d'iterations (<= 0 = defaut 400)
  // Descripteurs des champs de l'etage ("" = role canonique, bit-identique ; sinon nom de role
  // stable OU nom de variable du bloc, resolus au build contre Model::conservative_vars()).
  std::string schur_density, schur_momentum_x, schur_momentum_y, schur_energy;
  // OPTIONS NEWTON de la source IMEX du chemin MONO-BLOC (vague 3 : options mono-bloc AMR cablees).
  // AJOUTE EN FIN DE STRUCT (append-only, meme raison que has_state / schur_* : un loader .so
  // anterieur ne lit pas ce champ et retombe sur le Newton historique a 2 iters figees). DEFAUT {} =
  // constantes historiques (2 / 0 / 0 / 1e-7 / 1.0 / none) -> backward_euler_source chemin (2a)
  // bit-identique. Consomme par build_amr_compiled (la fermeture mono-bloc le passe a cpl->step).
  NewtonOptions newton_options{};
  // METHODE TEMPORELLE du bloc, transportee sous forme d'ENTIER par l'ABI plate (0 == kEuler avant
  // historique, 1 == kSsprk3). AJOUTE EN FIN DE STRUCT (append-only, meme raison que has_state /
  // schur_* / newton_options : un loader .so anterieur ne lit pas ce champ et retombe SILENCIEUSEMENT
  // sur 0 == kEuler -- pas de corruption). Consomme par build_amr_compiled (mono-bloc -> cpl->step).
  int time_method = 0;  // adc::AmrTimeMethod : 0 kEuler (defaut), 1 kSsprk3
};

/// Fermetures type-erased d'un bloc AMR compile, produites par amr_dsl_block::build_amr_compiled et
/// installees via AmrSystem::set_compiled_block. Symetrique des hooks std::function de AmrSystem::Impl.
struct AmrCompiledHooks {
  std::shared_ptr<void> coupler_holder;   ///< maintient en vie le AmrCouplerMP<Model>
  std::function<void(double)> step;       ///< un macro-pas (regrid periodique inclus)
  std::function<double()> max_speed;      ///< vitesse d'onde max (pas CFL)
  std::function<double()> mass;           ///< masse grossiere
  std::function<int()> n_patches;         ///< nombre de patchs fins
  std::function<std::vector<double>()> density;  ///< densite grossiere, n*n row-major
  std::function<std::vector<double>()> potential;  ///< phi du niveau grossier, n*n row-major
  // AJOUTE EN QUEUE (additif, ne deplace aucun champ existant) : empreintes index-space des patchs
  // fins. Mirroir de n_patches (meme box_array(), le COMPTE devient les BOITES). Le loader .so qui
  // construit ce struct est garde par adc_native_abi_key : un .so genere AVANT cet ajout doit etre
  // recompile (le garde le diagnostique deja clairement) ; l'ajout en queue le rend purement additif.
  std::function<std::vector<PatchBox>()> patch_boxes;  ///< empreintes index-space des patchs fins
  // AJOUTES EN QUEUE (StabilityPolicy AMR, audit 2026-06, additif comme patch_boxes) : bornes de
  // pas OPTIONNELLES du bloc, evaluees sur le GROSSIER par AmrSystem::step_cfl mono-bloc. VIDES si
  // le modele ne declare pas les traits HasSourceFrequency / HasStabilityDt (bit-identique). Le
  // garde adc_native_abi_key force la regeneration des .so anterieurs (ajout purement additif).
  std::function<double()> source_frequency;  ///< max grossier de mu [1/s] (0 = ne contraint pas)
  std::function<double()> stability_dt;      ///< min grossier du pas admissible (0 = ne contraint pas)
  // AJOUTE EN QUEUE (IO v1, parite System::set_clock) : restauration du compteur de macro-pas du
  // moteur MONO-BLOC (le coupleur AmrCouplerMP porte la phase de cadence regrid dans un step_state ;
  // ce hook l'ecrit au restart). VIDE n'est jamais le cas (le builder le peuple toujours) ; le garde
  // adc_native_abi_key force la regeneration des .so anterieurs (ajout purement additif en queue).
  std::function<void(int)> set_macro_step;   ///< restaure la phase de cadence (regrid) du mono-bloc
  // AJOUTES EN QUEUE (checkpoint/restart AMR mono-rang, ADC-65 ; additif comme set_macro_step) : etat
  // CONSERVATIF COMPLET par niveau (toutes composantes) + phi (warm-start) + imposition d'une
  // hierarchie fine SAUVEE. Le coupleur mono-bloc (AmrCouplerMP) les porte ; le builder les peuple
  // toujours (jamais vides). Le garde adc_native_abi_key force la regeneration des .so anterieurs.
  std::function<int()> n_levels;                                ///< nombre de niveaux (>= 1)
  std::function<int()> n_vars;                                  ///< composantes conservees du bloc
  std::function<std::vector<double>(int)> level_state;          ///< etat complet niveau k (c*nf*nf+j*nf+i)
  std::function<void(int, const std::vector<double>&)> set_level_state;       ///< restaure l'etat niveau k
  std::function<std::vector<double>(int)> level_potential;      ///< phi niveau k (nf*nf row-major)
  std::function<void(int, const std::vector<double>&)> set_level_potential;   ///< restaure phi niveau k
  std::function<void(const std::vector<PatchBox>&)> set_hierarchy;  ///< impose les patchs fins sauves
};

/// Builder DIFFERE d'un bloc COMPILE sur la hierarchie multi-blocs : recoit le layout PARTAGE (cree
/// UNE seule fois au build paresseux, commun a tous les blocs) plus les parametres du bloc figes a
/// l'ajout (nom, densite initiale, gamma, substeps/stride, recon/imex, masque IMEX partiel resolu en
/// indices de composantes), et rend l'AmrRuntimeBlock type-erase du bloc (capture le Model/Limiter/
/// Flux CONCRETS via detail::dispatch_amr_block, le noyau reste COMPILE). Symetrique du chemin natif
/// add_block : la (4e) difference est seulement que les types sont connus a l'ajout (modele compile)
/// plutot que resolus d'une ModelSpec au build. La SIGNATURE mentionne des types FORWARD-DECLARES :
/// elle n'est instanciee avec un callable concret que dans add_compiled_model(AmrSystem&) (en-tete
/// amr_dsl_block.hpp) ou ces types sont complets, et invoquee que dans python/amr_system.cpp.
using AmrCompiledBlockBuilder = std::function<AmrRuntimeBlock(
    const detail::SharedAmrLayout& layout, const std::string& name,
    const std::vector<double>& density, bool has_density, double gamma, int substeps, bool recon_prim,
    bool imex, int stride, const std::vector<std::string>& implicit_vars,
    const std::vector<std::string>& implicit_roles)>;

/// Bloc unique porte sur une hierarchie AMR, compose a l'execution.
class AmrSystem {
 public:
  explicit AmrSystem(const AmrSystemConfig& cfg);
  ~AmrSystem();
  AmrSystem(AmrSystem&&) noexcept;
  AmrSystem& operator=(AmrSystem&&) noexcept;

  /// Borne GLOBALE de pas de temps (pendant AMR de System::add_dt_bound) : fn() evaluee UNE fois
  /// par step_cfl (hote), all_reduce_min (dt identique sur tous les rangs), <= 0 / non finie =
  /// inerte ce pas. Crochet des contraintes non locales (couplage, scheduler, rampe utilisateur).
  void add_dt_bound(const std::string& label, std::function<double()> fn);

  /// Borne ACTIVE du dernier step_cfl : "transport:<bloc>" | "source_frequency:<bloc>" |
  /// "stability_dt:<bloc>" | "global:<label>" | "degenerate" | "" (aucun pas CFL encore).
  std::string last_dt_bound() const;

  /// Ajoute un bloc porte sur l'AMR. Memes parametres de schema spatial que System
  /// (limiter x riemann x recon), appliques a chaque niveau/patch de la hierarchie. Le PREMIER
  /// add_block definit le bloc ; un 2e (ou plus) bascule sur le moteur multi-blocs (hierarchie
  /// partagee, Poisson somme co-localise). Les blocs peuvent avoir des SCHEMAS SPATIAUX DIFFERENTS.
  /// @param name    nom du bloc : INDEXE le bloc (set_density(name), mass(name), density(name)). En
  ///                multi-blocs le nom doit etre unique ; mono-bloc un nom vide cible le bloc unique.
  /// @param model   composition de briques (transport/source/elliptic + parametres)
  /// @param limiter "none" | "minmod" | "vanleer" | "weno5" (weno5 = WENO5-Z, 3 ghosts ; rusanov)
  /// @param riemann "rusanov" | "hllc" | "roe" (hllc/roe exigent un transport compressible)
  /// @param recon   "conservative" | "primitive" (variables reconstruites ; primitif plus
  ///                robuste pour Euler : positivite de rho et p)
  /// @param time    "explicit" (source en Euler avant, portee par le pas AMR) ou "imex" (source
  ///                raide traitee en IMPLICITE par backward_euler_source ; le transport reste
  ///                explicite, porte par le reflux conservatif ; cf. capstone vii). Tout autre
  ///                traitement est refuse.
  /// @param substeps sous-pas explicites du bloc (>= 1) : le pas effectif est decoupe en substeps
  ///                morceaux egaux (MULTI-BLOCS uniquement ; en mono-bloc, porte par AmrCouplerMP).
  /// @param stride  cadence HOLD-THEN-CATCH-UP du bloc (>= 1 ; defaut 1 = chaque macro-pas). stride=M
  ///                tient le bloc M-1 macro-pas puis le rattrape d'un pas effectif M*dt (multirate).
  ///                MULTI-BLOCS uniquement (un seul bloc avance toujours a chaque pas). step_cfl honore
  ///                la cadence : dt = cfl*h*min_b(substeps_b/(stride_b*w_b)), mirroir de System::step_cfl.
  /// @param implicit_vars / implicit_roles  masque IMEX partiel PORTE PAR LE BLOC (cf. System::add_block) :
  ///                composantes conservees traitees en IMPLICITE, par NOM (implicit_vars) ou par ROLE
  ///                physique (implicit_roles). VIDES (defaut) -> backward-Euler plein (toutes les
  ///                composantes implicites). N'ont de sens qu'en time="imex" : les demander en explicite
  ///                est une ERREUR (pas d'ignore silencieux). MULTI-BLOCS uniquement (le mono-bloc
  ///                AmrCouplerMP porte son IMEX sans masque ; un masque y est donc refuse).
  /// @throws std::runtime_error si un bloc est deja defini, si substeps < 1, si stride < 1, si time
  ///         n'est pas dans {explicit, imex}, si recon n'est pas dans {conservative, primitive}, ou si
  ///         un masque implicite est demande hors IMEX / avec un nom-role absent du bloc.
  /// Options Newton (vague 3, parite System::add_block) : defauts = constantes historiques,
  /// bit-identique. SUPPORT (vague 3, solde) : les OPTIONS (newton_max_iters/rel_tol/abs_tol/fd_eps/
  /// damping/fail_policy) sont desormais cablees EN MONO-BLOC (coupleur AmrCouplerMP) ET en MULTI-BLOCS
  /// (moteur AmrRuntime) ; les loaders .so les rejettent (ABI plate). Le RAPPORT newton_diagnostics
  /// (newton_report agrege) est cable en MULTI-BLOCS NATIF seulement : le mono-bloc le rejette au build,
  /// les loaders .so a la facade ; fail_policy warn/throw fonctionne partout (detection sans rapport).
  void add_block(const std::string& name, const ModelSpec& model,
                 const std::string& limiter = "minmod",
                 const std::string& riemann = "rusanov",
                 const std::string& recon = "conservative",
                 const std::string& time = "explicit", int substeps = 1, int stride = 1,
                 const std::vector<std::string>& implicit_vars = {},
                 const std::vector<std::string>& implicit_roles = {},
                 int newton_max_iters = 2, double newton_rel_tol = 0.0,
                 double newton_abs_tol = 0.0, double newton_fd_eps = 1e-7,
                 double newton_damping = 1.0, const std::string& newton_fail_policy = "none",
                 bool newton_diagnostics = false);

  /// Rapport du Newton de la source implicite (IMEX) d'un bloc, AGREGE sur les niveaux et sous-pas de
  /// la DERNIERE avance du bloc. N'existe que si le bloc a ete ajoute avec newton_diagnostics=true EN
  /// MULTI-BLOCS natif (erreur explicite sinon : mono-bloc, loader .so, ou bloc sans diagnostics).
  /// Copie a plat (pas de dependance au header numerique cote appelant), parite System::SourceNewtonReport.
  struct SourceNewtonReport {
    bool enabled;          ///< un rapport a ete calcule (au moins une avance IMEX jouee)
    bool converged;        ///< aucune cellule en echec sur la derniere avance
    double max_residual;   ///< max cellules/niveaux/sous-pas de ||F||_inf a la sortie du Newton
    double max_iters_used; ///< max cellules/niveaux/sous-pas des iterations consommees
    double n_failed;       ///< nb (cellules x niveaux x sous-pas) en echec (non-fini / pivot / non-convergence)
    double failed_i;       ///< i d'UNE cellule fautive (-1 si aucune ; index max encode)
    double failed_j;       ///< j de la meme cellule (-1 si aucune)
    double failed_comp;    ///< composante conservee du pire residu de cette cellule (-1 inconnu)
  };
  /// @throws std::runtime_error si le bloc est inconnu, en mono-bloc, sur un loader .so, ou si le bloc
  ///         n'a pas active newton_diagnostics. Force le build paresseux (ensure_built).
  SourceNewtonReport newton_report(const std::string& name);

  /// Enregistre un bloc COMPILE (chemin add_compiled_model, header amr_dsl_block.hpp). DEUX builders
  /// type-erases sont figes ici, pour les DEUX routages de la facade :
  ///  - @p mono_builder : recevant les AmrBuildParams figes au build paresseux, rend les
  ///    AmrCompiledHooks d'un AmrCouplerMP<Model> concret. Utilise EN MONO-BLOC (1 seul bloc compile)
  ///    -> chemin AmrCouplerMP historique, INTOUCHE, bit-identique.
  ///  - @p multi_builder : recevant le layout PARTAGE materialise au build paresseux (commun a tous
  ///    les blocs), rend l'AmrRuntimeBlock type-erase du bloc. Utilise EN MULTI-BLOCS (>= 2 blocs,
  ///    compiles et/ou natifs melanges) -> moteur runtime AmrRuntime, exactement comme add_block.
  /// @p recon_prim / @p imex / @p stride / @p implicit_vars / @p implicit_roles : metadonnees du bloc
  /// (schema temporel, multirate, masque IMEX partiel) figees a l'ajout, consommees par le routage
  /// multi-blocs (le mono-bloc les porte deja dans les AmrBuildParams via mono_builder).
  /// NE PAS appeler directement : passer par la fonction libre add_compiled_model(AmrSystem&, ...).
  /// @throws std::runtime_error si le systeme est deja construit.
  /// VISIBILITE DEFAUT (ADC_EXPORT) : SEULE methode appelee par le gabarit en-tete
  /// add_compiled_model(AmrSystem&) (cf. amr_dsl_block.hpp). Un loader .so genere (chemin DSL
  /// "production" cote AMR, emit_cpp_native_loader(target="amr_system") / add_native_block) inline ce
  /// gabarit et doit resoudre ce symbole depuis le module _adc deja charge ; compile en
  /// -fvisibility=hidden (pybind11), le module ne l'exporterait pas sans cette annotation et le dlopen
  /// du loader echouerait. Symetrique des methodes ADC_EXPORT de System (grid_context/install_block).
  ADC_EXPORT void set_compiled_block(
      int ncomp, double gamma, int substeps,
      std::function<AmrCompiledHooks(const AmrBuildParams&)> mono_builder,
      AmrCompiledBlockBuilder multi_builder = {}, const std::string& name = std::string(),
      bool recon_prim = false, bool imex = false, int stride = 1,
      const std::vector<std::string>& implicit_vars = {},
      const std::vector<std::string>& implicit_roles = {});

  /// Branche un bloc NATIF AMR depuis un loader .so genere par le DSL (backend "production", cible
  /// "amr_system" : dsl.compile_native(target="amr_system") / compile(backend="production",
  /// target="amr_system")). Pendant AMR de System::add_native_block : le .so inline le gabarit en-tete
  /// add_compiled_model(AmrSystem&, ...), qui materialise un AmrCouplerMP<Model> concret au build
  /// paresseux et installe ses hooks via set_compiled_block -- chemin NATIF, MEME hierarchie AMR que
  /// add_block (reflux conservatif, regrid), pas de marshaling de tableaux plats.
  ///
  /// Le module _adc est PROMU en portee globale (RTLD_NOLOAD) puis le loader est dlopen-e en
  /// RTLD_GLOBAL pour resoudre set_compiled_block ; la cle d'ABI baked dans le loader
  /// (adc_native_abi_key) est comparee a celle du module (abi_key()) -- ecart => erreur claire (pas
  /// d'UB silencieux a la frontiere C++). Memes garde-fous de schema que System (validation amont).
  ///
  /// MULTI-BLOCS (capstone v) : add_native_block PEUT desormais etre appele plusieurs fois (ou melange
  /// a add_block natif) -> les blocs compiles co-existent sur la hierarchie partagee via AmrRuntime
  /// (le loader recompile contre cet en-tete fournit le builder runtime ; cf. set_compiled_block). Le
  /// nom INDEXE alors le bloc (set_density/mass/density), comme add_block.
  /// time est cable a {explicit, imex} (imex = source raide implicite via backward_euler_source ; tout
  /// autre traitement est rejete par add_compiled_model). Le multirate (stride) et le masque IMEX
  /// partiel ne transitent PAS par l'ABI plate du loader (ABI inchangee) : ce chemin .so les REJETTE
  /// desormais au niveau de la facade Python (AmrSystem.add_equation leve ValueError sur stride>1 ou un
  /// masque IMEX non vide, plutot que de les ignorer en silence). Pour ces parametres, utiliser
  /// add_block natif (ModelSpec) ou add_compiled_model(AmrSystem&) en DIRECT (en-tete), qui exposent
  /// stride et le masque. recon "primitive" et flux "roe"/"hllc" sont CABLES a parite (#113 :
  /// dispatch_amr_compiled les accepte ; la facade Python applique un garde pression pour hllc/roe).
  /// limiter "weno5" (WENO5-Z, 3 ghosts) est CABLE sur rusanov (#105 : les niveaux du coupleur sont
  /// alloues a Limiter::n_ghost et le regrid herite n_grow() : pas de lecture hors bornes).
  /// @throws std::runtime_error si l'ABI diverge, si un symbole manque, ou substeps < 1.
  /// @param name nom du bloc : cosmetique en mono-bloc, INDEXE le bloc en multi-blocs (set_density/
  ///             mass/density ; doit etre unique et non vide des le 2e bloc, comme add_block).
  void add_native_block(const std::string& name, const std::string& so_path,
                        const std::string& limiter = "minmod",
                        const std::string& riemann = "rusanov",
                        const std::string& recon = "conservative",
                        const std::string& time = "explicit", double gamma = 1.4,
                        int substeps = 1);

  /// Raffine les cellules ou la densite (composante 0) depasse @p threshold.
  void set_refinement(double threshold);

  /// Ajoute au critere de regrid le tag de PHI sur |grad phi| (D4 du design
  /// docs/AMR_REGRID_UNION_TAGS_DESIGN.md) : raffine aussi les cellules ou la norme du gradient du
  /// potentiel electrostatique |grad phi| (composantes 1,2 de l'aux partage) depasse @p grad_threshold.
  /// MULTI-BLOCS uniquement (le moteur runtime AmrRuntime porte le regrid d'union des tags ; le chemin
  /// mono-bloc AmrCouplerMP n'a pas de predicat phi separe). Le tag de phi s'AJOUTE a l'union des tags
  /// de densite par bloc (set_refinement) : la grille raffine la ou N'IMPORTE QUEL bloc depasse son
  /// seuil de densite OU |grad phi| depasse @p grad_threshold. Critere PHYSIQUE du diocotron : le bord
  /// d'anneau suit le gradient du potentiel, pas la densite seule.
  /// @param grad_threshold seuil de |grad phi|. <= 0 (DEFAUT) -> le tag phi est DESACTIVE (phi ne
  ///        contribue pas a l'union ; bit-identique a avant cet appel). Sans regrid_every > 0, sans
  ///        effet (le regrid n'est jamais appele). A appeler AVANT le premier step.
  void set_phi_refinement(double grad_threshold);

  /// Configure le Poisson grossier (cf. System::set_poisson). Sur AMR le solveur elliptique est
  /// TOUJOURS GeometricMG et le second membre TOUJOURS f = somme des briques elliptiques du bloc.
  /// @param rhs    "charge_density" | "composite" (meme second membre compose que System)
  /// @param solver "geometric_mg" uniquement (le seul cable sur la hierarchie ; pas de FFT)
  /// @param bc     "auto" | "periodic" | "dirichlet" | "neumann"
  /// @param wall   "none" | "circle" (paroi conductrice circulaire, exige wall_radius > 0)
  /// @throws std::runtime_error si rhs, solver, bc ou wall est hors du domaine supporte.
  void set_poisson(const std::string& rhs = "charge_density",
                   const std::string& solver = "geometric_mg",
                   const std::string& bc = "auto", const std::string& wall = "none",
                   double wall_radius = 0.0);

  /// Fixe la densite initiale sur le niveau grossier (composante 0), n*n row-major.
  /// @param name etiquette cosmetique (AMR mono-bloc : la densite vise l'unique bloc).
  void set_density(const std::string& name, const std::vector<double>& rho);

  /// Fixe l'ETAT CONSERVATIF INITIAL COMPLET (toutes les composantes) sur le niveau grossier, puis
  /// le prolonge aux niveaux fins au build (injection constante, comme la densite). @p U est plat
  /// composante-majeur (c*n*n + j*n + i) de taille ncomp*n*n ; ncomp == n_vars du modele (verifie au
  /// build, ou seul Model::n_vars est connu). Prioritaire sur set_density : permet de demarrer l'AMR
  /// depuis l'etat de derive du papier (rho, rho*u, rho*v) au lieu de m=0 (Probleme 2). La conversion
  /// primitif -> conservatif (rho_u = rho*u) est faite cote Python (l'appelant fournit deja le
  /// conservatif). Cable sur les blocs NATIFS (mono-bloc comme multi-blocs : threade au builder natif,
  /// seed du grossier puis injection aux fins) ; en multi-blocs @p name indexe le bloc cible. Un bloc
  /// COMPILE (.so) porteur d'un etat leve au build en multi-blocs (le loader .so ne transporte pas
  /// l'etat) : utiliser un bloc natif adc.Model(...) ou set_density.
  /// @throws std::runtime_error si le systeme est deja construit, si U est vide, ou si sa taille
  ///         n'est pas un multiple de n*n.
  void set_conservative_state(const std::string& name, const std::vector<double>& U);

  /// Fixe le champ magnetique B_z(x, y) du niveau grossier (n*n row-major), requis par l'etage source
  /// condense par Schur (terme de Lorentz Omega = B_z). Pendant AMR de System::set_magnetic_field.
  /// MONO-BLOC uniquement (l'etage condense AMR est cable sur le coupleur mono-bloc AmrCouplerMP).
  /// @throws std::runtime_error si le systeme est deja construit ou si bz n'est pas de taille n*n.
  void set_magnetic_field(const std::vector<double>& bz);

  /// Active l'ETAGE SOURCE condense par Schur (chemin amr-schur) sur le bloc @p name. Pendant AMR de
  /// System::set_source_stage : assemble et resout l'operateur condense electrostatique/Lorentz GLOBAL
  /// (sur le grossier, en composant l'etage uniforme #126), au lieu de la source IMEX LOCALE cellule
  /// par cellule. C'est l'OPT-IN du chemin amr-schur, l'equivalent raffine du
  /// time=Strang(Explicit(ssprk3), CondensedSchur(theta, alpha)) uniforme. Le transport du bloc doit
  /// etre SOURCE-FREE (modele a brique source NoSource) : la source est jouee separement par l'etage.
  /// @param kind  seul "electrostatic_lorentz" cable (cf. CondensedSchurSourceStepper).
  /// @param theta theta-schema dans (0, 1] (0.5 = Crank-Nicolson).
  /// @param alpha constante de couplage electrostatique.
  /// @throws std::runtime_error si le systeme est deja construit, en MULTI-BLOCS, si kind/theta sont
  ///         hors domaine, ou (au build) si le bloc n'expose pas Density/MomentumX/MomentumY ou si
  ///         set_magnetic_field n'a pas ete appele.
  /// @param krylov_tol / krylov_max_iters : tolerance et budget du solve Krylov GROSSIER de
  ///        l'etage (<= 0 = defauts historiques 1e-10 / 400 ; le solve composite FAC garde ses
  ///        tolerances propres, Phase 4). @param density / momentum_x / momentum_y / energy :
  ///        descripteurs des champs ("" = role canonique, bit-identique ; sinon nom de role stable
  ///        ou de variable du bloc, resolus au build) -- parite avec System::set_source_stage.
  void set_source_stage(const std::string& name, const std::string& kind, double theta,
                        double alpha, double krylov_tol = 0.0, int krylov_max_iters = 0,
                        const std::string& density = "", const std::string& momentum_x = "",
                        const std::string& momentum_y = "", const std::string& energy = "");

  /// Choisit la politique de splitting en temps de l'etage source condense : "lie" (defaut, H(dt) S(dt))
  /// ou "strang" (H(dt/2) S(dt) H(dt/2), 2e ordre). Pendant AMR de System::set_time_scheme. Sans etage
  /// source condense (set_source_stage non appele), sans effet. @throws si schema inconnu / deja construit.
  void set_time_scheme(const std::string& scheme);

  /// Enregistre une SOURCE COUPLEE inter-especes (adc.dsl.CoupledSource compilee, ABI plate bytecode
  /// P5), pendant raffine de System::add_coupled_source mais sur la hierarchie AMR PARTAGEE. La source
  /// est appliquee a CHAQUE macro-pas APRES le transport, par splitting forward-Euler, niveau par
  /// niveau, suivi d'une cascade fin -> grossier (coherence des cellules grossieres couvertes, #169).
  /// Le couplage est cuit en machine a pile device-clean (CoupledSourceKernel) : AUCUN callback Python
  /// par cellule dans le chemin chaud. MULTI-BLOCS uniquement (>= 2 add_block : le couplage lit/ecrit
  /// PLUSIEURS blocs nommes). Doit etre appele AVANT le premier step (le moteur runtime est construit
  /// au build paresseux ; la source y est injectee).
  ///
  /// CONSERVATION : une construction add_pair (un terme +expr sur un bloc, -expr exactement sur l'autre,
  /// MEME cellule) rend la somme des deux blocs conservee PAR CELLULE (et globalement) a la precision
  /// machine. Le moteur ne l'IMPOSE pas (une ionisation creant une paire e/i est licite) : c'est une
  /// propriete du couplage construit (verify_conservation cote DSL la controle symboliquement).
  ///
  /// @throws std::runtime_error si appele en mono-bloc, si le systeme est deja construit, ou si la
  ///         forme du bytecode / un role / un bloc est invalide (memes gardes que System).
  /// @param frequency frequence CONSTANTE declaree mu [1/s] du couplage (vague 3) : borne
  ///                  dt <= cfl/mu sur le macro-pas de step_cfl ; <= 0 (defaut) = pas de borne.
  /// @param label     nom du couplage (raison "coupled_source:<label>" de last_dt_bound).
  /// @param freq_prog_ops/freq_prog_args  programme bytecode OPTIONNEL d'une frequence PAR CELLULE
  ///                  mu(U) (meme machine a pile / table de registres que la source). VIDES (defaut) =
  ///                  frequence constante seule (bit-identique). Non vides : evaluee sur le NIVEAU
  ///                  GROSSIER des blocs d'entree a chaque step_cfl (MAX + all_reduce_max, borne
  ///                  dt <= cfl / max(mu)). La borne est donc evaluee sur le grossier (pas les patchs).
  void add_coupled_source(const std::vector<std::string>& in_blocks,
                          const std::vector<std::string>& in_roles,
                          const std::vector<double>& consts,
                          const std::vector<std::string>& out_blocks,
                          const std::vector<std::string>& out_roles,
                          const std::vector<int>& prog_ops, const std::vector<int>& prog_args,
                          const std::vector<int>& prog_lens, double frequency = 0.0,
                          const std::string& label = "coupled_source",
                          const std::vector<int>& freq_prog_ops = {},
                          const std::vector<int>& freq_prog_args = {});

  void step(double dt);  ///< un macro-pas AMR (regrid periodique inclus)
  void advance(double dt, int nsteps);
  /// Avance a dt = cfl * dx_grossier / vitesse d'onde max. @return le dt utilise.
  double step_cfl(double cfl);

  int nx() const;
  double time() const;
  /// Compteur de MACRO-PAS (0-indexe ; incremente par step / advance / step_cfl), parite
  /// System::macro_step. Necessaire au checkpoint/restart (la cadence stride / regrid depend de
  /// macro_step % stride|regrid_every, pas seulement de t). Prerequis IO PR-IO-3 (audit 2026-06).
  int macro_step() const;
  /// RESTAURE l'horloge AMR (t, macro_step) -- parite System::set_clock. Pose le temps ET le compteur
  /// de macro-pas (propage a la cadence regrid/stride du moteur, mono-bloc comme multi-blocs). Utile
  /// seul (cadence stride + reprise d'horloge). @throws si macro_step < 0.
  void set_clock(double t, int macro_step);
  int n_blocks() const;           ///< nombre de blocs (1 = mono-bloc AmrCouplerMP ; >= 2 = AmrRuntime)
  /// Noms des blocs dans l'ordre d'ajout (parite System::block_names) : la facade IO itere dessus
  /// pour ecrire CHAQUE bloc par son nom (un nom vide -> bloc 0, compat mono-bloc historique).
  std::vector<std::string> block_names() const;
  int n_patches();                ///< nombre de patchs fins courants (de la hierarchie partagee)
  /// Empreintes index-space des patchs fins courants : un PatchBox (level, ilo, jlo, ihi, jhi) par
  /// box fine, pour TOUS les niveaux fins (level >= 1). Coins INCLUSIFS dans l'espace d'indices du
  /// niveau (n << level cellules/direction, ratio 2). MEME source que n_patches() (le BoxArray fin
  /// GLOBAL, toutes boites/tous rangs -> rank-independent, MPI-safe, zero communication). C'est une
  /// QUERY (entre les pas) : lecture seule des boites deja stockees, AUCUN cout chemin chaud. La
  /// conversion en [0, L]^2 se fait cote Python (qui connait n via nx() et L). Force le build
  /// paresseux (ensure_built) comme n_patches()/mass()/density().
  std::vector<PatchBox> patch_boxes();

  /// CHECKPOINT / RESTART AMR mono-rang (ADC-65), MONO-BLOC : accesseurs d'ETAT par niveau +
  /// imposition de hierarchie pour une reprise BIT-IDENTIQUE (cf. AmrSystem.checkpoint/restart cote
  /// Python). Tous REJETTENT le multi-blocs (moteur AmrRuntime : layout + aux PARTAGES, suite
  /// documentee) ; la facade rejette en plus MPI np>1 (gather par niveau : suite). Forcent le build
  /// paresseux (ensure_built) comme patch_boxes()/mass(). @p k : niveau (0 = grossier, >= 1 = fin).
  int n_levels();                 ///< nombre de niveaux de la hierarchie (>= 1)
  int n_vars();                   ///< nombre de composantes conservees (mono-bloc)
  /// Etat conservatif COMPLET du niveau @p k, plat composante-majeur c*nf*nf + j*nf + i (nf = n << k ;
  /// zeros hors patchs au niveau fin -- seul l'interieur des patchs est defini).
  std::vector<double> level_state(int k);
  void set_level_state(int k, const std::vector<double>& s);  ///< restaure l'etat du niveau @p k (tel quel)
  /// Potentiel phi du niveau @p k, plat nf*nf row-major. Niveau 0 = warm-start du multigrille
  /// (reprise bit-identique) ; niveau >= 1 = aux comp 0 (recompute a l'update).
  std::vector<double> level_potential(int k);
  void set_level_potential(int k, const std::vector<double>& p);  ///< restaure phi du niveau @p k
  /// Impose la hierarchie fine SAUVEE (au restart) au lieu du clustering Berger-Rigoutsos : @p boxes
  /// sont les empreintes patch_boxes() du checkpoint (filtrees niveau 1 en mono-bloc).
  void set_hierarchy(const std::vector<PatchBox>& boxes);

  double mass();                  ///< masse du 1er bloc sur le grossier (conservee au reflux)
  double mass(const std::string& name);     ///< masse du bloc nomme sur le grossier (conservee PAR BLOC)
  std::vector<double> density();  ///< densite grossiere du 1er bloc (composante 0), n*n row-major
  std::vector<double> density(const std::string& name);  ///< densite grossiere du bloc nomme, n*n
  /// Potentiel electrostatique phi du NIVEAU GROSSIER (base), n*n row-major. Le niveau 0 couvre
  /// tout le domaine : suffisant pour echantillonner un cercle median (FFT azimutale), MEME
  /// observable que System::potential() sur grille mono-niveau. Resout le Poisson grossier si
  /// besoin (cf. System::potential / ensure_elliptic), donc valeur courante meme avant tout step.
  /// MULTI-BLOCS : phi resulte du Poisson de SYSTEME (Sum_b q_b n_b co-localise) ; partage par tous
  /// les blocs (aux unique). Le nom du bloc n'intervient donc pas.
  std::vector<double> potential();

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace adc
