#pragma once

#include <adc/mesh/physical_bc.hpp>  // BCRec
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
/// couplees inter-especes : deja cables. DSL production multi-bloc compile et regrid d'union : PR ulterieures.
///
/// @note Deux niveaux (ratio 2) ; traitement temporel explicite OU imex (par bloc).

namespace adc {

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
};

/// Bloc unique porte sur une hierarchie AMR, compose a l'execution.
class AmrSystem {
 public:
  explicit AmrSystem(const AmrSystemConfig& cfg);
  ~AmrSystem();
  AmrSystem(AmrSystem&&) noexcept;
  AmrSystem& operator=(AmrSystem&&) noexcept;

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
  void add_block(const std::string& name, const ModelSpec& model,
                 const std::string& limiter = "minmod",
                 const std::string& riemann = "rusanov",
                 const std::string& recon = "conservative",
                 const std::string& time = "explicit", int substeps = 1, int stride = 1,
                 const std::vector<std::string>& implicit_vars = {},
                 const std::vector<std::string>& implicit_roles = {});

  /// Enregistre un bloc COMPILE (chemin add_compiled_model, header amr_dsl_block.hpp) : @p builder
  /// est une fermeture type-erased qui, recevant les AmrBuildParams figes au build paresseux, rend
  /// les AmrCompiledHooks d'un AmrCouplerMP<Model> concret. NE PAS appeler directement : passer par
  /// la fonction libre add_compiled_model(AmrSystem&, ...). @throws si un bloc est deja defini.
  /// VISIBILITE DEFAUT (ADC_EXPORT) : SEULE methode appelee par le gabarit en-tete
  /// add_compiled_model(AmrSystem&) (cf. amr_dsl_block.hpp:182). Un loader .so genere (chemin DSL
  /// "production" cote AMR, emit_cpp_native_loader(target="amr_system") / add_native_block) inline ce
  /// gabarit et doit resoudre ce symbole depuis le module _adc deja charge ; compile en
  /// -fvisibility=hidden (pybind11), le module ne l'exporterait pas sans cette annotation et le dlopen
  /// du loader echouerait. Symetrique des methodes ADC_EXPORT de System (grid_context/install_block).
  ADC_EXPORT void set_compiled_block(int ncomp, double gamma, int substeps,
                          std::function<AmrCompiledHooks(const AmrBuildParams&)> builder);

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
  /// LIMITES RESTANTES (AmrSystem mono-bloc) : time est cable a {explicit, imex} (imex = source
  /// raide implicite via backward_euler_source ; tout autre traitement est rejete par
  /// add_compiled_model). Multi-box natif (grille multi-bloc) n'est pas cable dans la facade. recon
  /// "primitive" et flux "roe"/"hllc" sont CABLES a parite (#113 :
  /// dispatch_amr_compiled les accepte ; la facade Python applique un garde pression pour hllc/roe).
  /// limiter "weno5" (WENO5-Z, 3 ghosts) est CABLE sur rusanov (#105 : les niveaux du coupleur sont
  /// alloues a Limiter::n_ghost et le regrid herite n_grow() : pas de lecture hors bornes).
  /// @throws std::runtime_error si l'ABI diverge, si un symbole manque, ou substeps < 1.
  /// @param name etiquette cosmetique (AMR mono-bloc : ne sert pas d'index).
  void add_native_block(const std::string& name, const std::string& so_path,
                        const std::string& limiter = "minmod",
                        const std::string& riemann = "rusanov",
                        const std::string& recon = "conservative",
                        const std::string& time = "explicit", double gamma = 1.4,
                        int substeps = 1);

  /// Raffine les cellules ou la densite (composante 0) depasse @p threshold.
  void set_refinement(double threshold);

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
  void add_coupled_source(const std::vector<std::string>& in_blocks,
                          const std::vector<std::string>& in_roles,
                          const std::vector<double>& consts,
                          const std::vector<std::string>& out_blocks,
                          const std::vector<std::string>& out_roles,
                          const std::vector<int>& prog_ops, const std::vector<int>& prog_args,
                          const std::vector<int>& prog_lens);

  void step(double dt);  ///< un macro-pas AMR (regrid periodique inclus)
  void advance(double dt, int nsteps);
  /// Avance a dt = cfl * dx_grossier / vitesse d'onde max. @returns le dt utilise.
  double step_cfl(double cfl);

  int nx() const;
  double time() const;
  int n_blocks() const;           ///< nombre de blocs (1 = mono-bloc AmrCouplerMP ; >= 2 = AmrRuntime)
  int n_patches();                ///< nombre de patchs fins courants (de la hierarchie partagee)
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
