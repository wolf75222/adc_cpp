#pragma once

#include <adc/core/variables.hpp>  // VariableSet (descripteur a roles porte par chaque bloc)
#include <adc/runtime/export.hpp>  // ADC_EXPORT (methodes resolues par le loader natif a travers le dlopen)
#include <adc/runtime/grid_context.hpp>  // GridContext + BlockClosures (seam bloc compile AOT)
#include <adc/runtime/model_spec.hpp>

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

/// @file
/// @brief Composition multi-especes a l'execution : un systeme couple, bloc par bloc.
///
/// Chaque bloc est une espece (un etat U) decrite par une ModelSpec (composition de briques
/// generiques : transport + source + second membre elliptique), avec son schema spatial
/// (limiteur + flux Riemann), son traitement temporel et ses sous-pas. Tous les blocs
/// partagent un Poisson dont le second membre est la somme des elliptic_rhs par bloc ; la
/// source S agit par bloc. Le coeur ne nomme aucun scenario ; ceux-ci sont des compositions
/// definies cote application (adc_cases).
///
/// Python compose (objets-briques) ; le calcul par cellule (assemble_rhs<L,F>, Newton de la
/// source implicite, multigrille/FFT) reste C++ compile et fige a l'ajout du bloc. Aucun
/// callback Python dans le hot path, sauf integrateur temporel ecrit en Python via
/// eval_rhs / get_state / set_state.

namespace adc {

/// Maillage et domaine partages par tous les blocs (les parametres physiques sont par bloc,
/// dans la ModelSpec).
///
/// GEOMETRIE (chantier "grille polaire", Phase 1) : le CHOIX de la geometrie vit ICI, dans la config
/// du maillage, PAS dans le schema (FiniteVolume reste reconstruction + Riemann + variables). Defaut
/// "cartesian" : domaine carre [0,L]^2, comportement et numerique STRICTEMENT INCHANGES (bit-identique).
/// "polar" decrit un anneau global r in [r_min, r_max] x theta in [0, 2pi) (cf. PolarGeometry) ; il est
/// porte par adc.PolarMesh cote Python et n'est PAS encore branche dans System::step (Phase 1 livre la
/// geometrie + l'operateur de transport polaire + sa validation MMS ; le transport polaire a travers
/// System, qui demanderait aussi le Poisson polaire, est une phase ulterieure). Les champs polaires sont
/// ignores tant que geometry == "cartesian".
struct SystemConfig {
  int n = 64;            ///< cellules par direction (domaine n x n) -- pour polaire : n_r = n_theta = n
  double L = 1.0;        ///< taille du domaine carre [0,L]^2 (cartesien)
  bool periodic = true;  ///< domaine periodique, sinon sortie libre en transport (cartesien)
  // --- geometrie opt-in (Phase 1) : "cartesian" (defaut, bit-identique) | "polar" (anneau global) ---
  std::string geometry = "cartesian";  ///< choix de geometrie (porte par adc.CartesianMesh / adc.PolarMesh)
  int nr = 0;            ///< cellules radiales (polaire ; 0 => prend n)
  int ntheta = 0;        ///< cellules azimutales (polaire ; 0 => prend n)
  double r_min = 0.0;    ///< rayon interieur de l'anneau (polaire)
  double r_max = 1.0;    ///< rayon exterieur de l'anneau (polaire)
  // --- decoupage multi-box du TRANSPORT polaire (decoupage en BANDES theta, ADC-67) -----------------
  // Nombre de boites de l'anneau, decoupe en theta (chaque boite couvre tout le rayon [0, nr-1] et une
  // bande azimutale). 1 (defaut) = mono-box STRICTEMENT bit-identique a l'historique. theta_boxes > 1 :
  // le transport polaire (assemble_rhs_polar + fill_ghosts collectif) tourne multi-box. CONTRAINTES (cf.
  // PolarMesh / check_geometry) : 1 <= theta_boxes <= ntheta ET theta_boxes divise ntheta. INERTE en
  // cartesien (le decoupage cartesien passe par AmrSystem / le multi-box MPI mono-box historique).
  // PORTEE : transport multi-box OK ; Poisson polaire DIRECT mono-box only (rejet AMONT clair si
  // theta_boxes > 1, cf. ensure_elliptic_polar) ; etage Schur tensoriel polaire multi-box.
  int theta_boxes = 1;   ///< boites du decoupage theta du transport polaire (1 = mono-box)
};

/// Systeme multi-especes couple, compose a l'execution a partir de briques generiques.
class System {
 public:
  explicit System(const SystemConfig& cfg);
  ~System();
  System(System&&) noexcept;
  System& operator=(System&&) noexcept;

  /// Ajoute un bloc d'equation (une espece).
  /// @param model    composition de briques (transport/source/elliptic + parametres)
  /// @param limiter  reconstruction : "none" | "minmod" | "vanleer" | "weno5"
  /// @param riemann  flux numerique : "rusanov" (generique minimal) | "hll" (generique, exige
  ///                 model.wave_speeds) | "hllc" | "roe" (hllc/roe : EULER 2D seulement, 4 variables
  ///                 + pression gaz parfait)
  /// @param recon    variables reconstruites : "conservative" | "primitive" (Euler : primitif
  ///                 plus robuste, positivite de rho et p)
  /// @param time     "explicit" (SSPRK2) | "ssprk3" | "imex" (transport explicite, source implicite
  ///                 backward-Euler local, ordre 1) | "imexrk_ars222" (famille IMEX-RK, schema
  ///                 ARS(2,2,2), ordre 2, cartesien seul ; source PLEINEMENT implicite -> incompatible
  ///                 avec implicit_vars/implicit_roles)
  /// @param substeps sous-pas par macro-pas : le bloc avance N fois par macro-pas, chaque
  ///                 sous-pas de longueur dt/N (electrons rapides : substeps=10, pas dt/10).
  /// @param stride   cadence du bloc, semantique HOLD-THEN-CATCH-UP : 1 = chaque macro-pas (defaut,
  ///                 bit-identique) ; M > 1 = bloc TENU (non avance) tant que (macro_step + 1) % M != 0,
  ///                 puis avance d'un pas effectif M*dt au macro-pas ou (macro_step + 1) % M == 0 (fin de
  ///                 fenetre de M pas), donc temporellement coherent avec les blocs rapides (bloc lent,
  ///                 p.ex. neutres sur stride=20). substeps et stride sont ORTHOGONAUX : stride=M,
  ///                 substeps=N -> N sous-pas de M*dt/N, une fois en fin de fenetre. COUPLAGE : entre deux
  ///                 rattrapages, le bloc tenu entre dans la somme du Poisson avec son etat PERIME (derniere
  ///                 avance figee). step_cfl honore la cadence (dt <= cfl*h*substeps / (stride*w)).
  /// @param evolve   false = espece GELEE (fond fixe) : non avancee en temps, mais vue par le
  ///                 Poisson de systeme (et, a venir, par les sources couplees)
  /// @param implicit_vars  IMEX seulement : noms des variables conservees a traiter en IMPLICITE dans
  ///                 le pas de source (backward-Euler) ; les autres restent explicites (Euler avant). Le
  ///                 masque est PORTE PAR LE BLOC / la politique temporelle (et NON par le modele) : le
  ///                 MEME modele peut donc etre reutilise avec des traitements implicites differents. VIDE
  ///                 (defaut) + implicit_roles VIDE -> defaut du modele (Model::is_implicit, ou tout
  ///                 implicite a defaut de trait) -> bit-identique. Resolu contre les noms conservatifs
  ///                 du bloc ; un nom absent leve une erreur EXPLICITE.
  /// @param implicit_roles IMEX seulement : meme masque implicite mais par ROLE physique ("density",
  ///                 "momentum_x", "energy", ...) au lieu du nom (cf. variable_roles). Union avec
  ///                 implicit_vars. Un role absent du bloc leve une erreur EXPLICITE.
  /// @param newton_max_iters IMEX seulement : budget d'iterations du Newton local de la source
  ///                 implicite (backward-Euler). Defaut 2 = la constante historique (bit-identique).
  /// @param newton_rel_tol / newton_abs_tol IMEX seulement : critere d'arret par CELLULE
  ///                 ||F||_inf <= abs_tol + rel_tol*||F0||_inf, evalue en tete d'iteration.
  ///                 0/0 (defaut) = desactive -> boucle historique a iterations fixes, bit-identique.
  /// @param newton_fd_eps IMEX seulement : pas (relatif + plancher) de la jacobienne par differences
  ///                 finies (defaut 1e-7 = la constante historique).
  /// @param newton_diagnostics IMEX seulement : active le rapport Newton du bloc (residu max,
  ///                 iterations max, cellules en echec -- non-fini / pivot degenere / non-convergence),
  ///                 agrege sur les sous-pas de chaque avance et consultable via newton_report(name).
  ///                 OPT-IN : false (defaut) = chemin historique sans aucun cout supplementaire.
  /// @param newton_damping IMEX seulement : amortissement de la mise a jour Newton W -= damping*delta
  ///                 dans (0, 1]. 1 (defaut) = Newton plein, bit-identique.
  /// @param newton_fail_policy IMEX seulement : reaction aux cellules en echec -- "none" (defaut,
  ///                 enregistrer seulement), "warn" (avertissement stderr par avance), "throw"
  ///                 (erreur dure avec la cellule fautive). != "none" active la detection (chemin
  ///                 instrumente, observateur pur : W inchange).
  void add_block(const std::string& name, const ModelSpec& model,
                 const std::string& limiter = "minmod",
                 const std::string& riemann = "rusanov",
                 const std::string& recon = "conservative",
                 const std::string& time = "explicit", int substeps = 1,
                 bool evolve = true, int stride = 1,
                 const std::vector<std::string>& implicit_vars = {},
                 const std::vector<std::string>& implicit_roles = {},
                 int newton_max_iters = 2, double newton_rel_tol = 0.0,
                 double newton_abs_tol = 0.0, double newton_fd_eps = 1e-7,
                 bool newton_diagnostics = false, double newton_damping = 1.0,
                 const std::string& newton_fail_policy = "none");

  /// Rapport du Newton de la source implicite (IMEX) d'un bloc, AGREGE sur les sous-pas de la
  /// DERNIERE avance du bloc. N'existe que si le bloc a ete ajoute avec newton_diagnostics=true
  /// (erreur explicite sinon). Copie a plat (pas de dependance au header numerique).
  struct SourceNewtonReport {
    bool enabled;          ///< un rapport a ete calcule (au moins une avance IMEX jouee)
    bool converged;        ///< aucune cellule en echec sur la derniere avance
    double max_residual;   ///< max cellules/sous-pas de ||F||_inf a la sortie du Newton
    double max_iters_used; ///< max cellules/sous-pas des iterations consommees
    double n_failed;       ///< nb (cellules x sous-pas) en echec (non-fini / pivot / non-convergence)
    double failed_i;       ///< i d'UNE cellule fautive (-1 si aucune ; index max encode)
    double failed_j;       ///< j de la meme cellule (-1 si aucune)
    double failed_comp;    ///< composante conservee du pire residu de cette cellule (-1 inconnu)
  };
  SourceNewtonReport newton_report(const std::string& name) const;

  /// Ajoute un bloc dont le modele est CHARGE A L'EXECUTION depuis une bibliotheque partagee (.so)
  /// generee par le DSL (emit_cpp_brick -> ModelAdapter -> fabrique extern "C"). Le .so doit exposer
  /// adc_model_nvars(), adc_make_model() (renvoie un IModel<NV>*) et adc_destroy_model(void*).
  /// CHEMIN HOTE (dispatch virtuel, Rusanov a_max global periodique, Euler explicite) : pour
  /// prototyper un modele inedit, ecrit en formules cote Python, sans recompiler le coeur. cf.
  /// dynamic_model.hpp.
  /// @param names noms des variables (introspection) ; defaut u0..u{NV-1}.
  /// @param recon reconstruction MUSCL des etats de face (conservatif) : "none" (ordre 1) | "minmod"
  ///              | "vanleer" (ordre 2, TVD). Le choix du FLUX (HLLC/Roe) reste sur le chemin compile.
  void add_dynamic_block(const std::string& name, const std::string& so_path, int substeps = 1,
                         const std::vector<std::string>& names = {},
                         const std::string& recon = "none");

  /// Ajoute un bloc dont le modele est COMPILE AOT depuis une .so generee par le DSL
  /// (dsl.compile_aot / compile_or_jit(mode="compile")). A la difference du bloc dynamique (chemin
  /// hote, dispatch virtuel IModel, Rusanov ordre 1), ce bloc tourne le chemin de PRODUCTION : la .so
  /// execute assemble_rhs<Limiter, Flux> (HLLC/Roe au choix, ordre 2) et SSPRK2/IMEX du coeur sur le
  /// modele genere ; seuls des tableaux plats transitent (ABI extern "C", cf. compiled_block_abi.hpp).
  /// @param limiter "none" | "minmod" | "vanleer"   @param riemann "rusanov" | "hll" | "hllc" | "roe"
  /// @param recon   "conservative" | "primitive"    @param time "explicit" | "imex"
  void add_compiled_block(const std::string& name, const std::string& so_path,
                          const std::string& limiter = "minmod",
                          const std::string& riemann = "rusanov",
                          const std::string& recon = "conservative",
                          const std::string& time = "explicit", int substeps = 1,
                          const std::vector<std::string>& names = {});

  /// Change les valeurs des parametres RUNTIME d'un bloc AOT (add_compiled_block) SANS recompiler le
  /// .so (P7-b). @p values est le bloc COMPLET des valeurs (ordre = ordre trie des noms cote DSL, cf.
  /// CompiledModel.runtime_param_names). Le bloc doit avoir ete ajoute via add_compiled_block ET
  /// declarer au moins un parametre runtime (dsl.Param(..., kind="runtime")) ; sinon erreur explicite
  /// (un set_param silencieux sur un bloc sans param runtime masquerait un bug). Effet au prochain pas
  /// (les fermetures du bloc lisent le bloc de valeurs PARTAGE). @throws std::runtime_error si le bloc
  /// est inconnu, n'a pas de params runtime, ou si @p values n'a pas la bonne longueur.
  void set_block_params(const std::string& name, const std::vector<double>& values);

  /// Ajoute un bloc dont le modele est compile dans un LOADER NATIF .so genere par le DSL
  /// (dsl.compile_native / compile(backend="production")). C'est le chemin de PRODUCTION : le loader
  /// inline le gabarit en-tete adc::add_compiled_model<ProdModel>, qui fabrique les fermetures sur le
  /// CONTEXTE REEL du System (grid_context) et installe le bloc via install_block. Le bloc tourne
  /// alors EXACTEMENT le chemin natif d'add_block (fill_boundary = halos MPI, assemble_rhs device),
  /// ZERO-COPIE -- a la difference d'add_compiled_block (.so + marshaling de tableaux plats, ABI
  /// extern "C" host). Le loader inline appelant des methodes hors-ligne de ce module (install_block
  /// /grid_context/ensure_aux_width, exportees ADC_EXPORT), la frontiere n'est PAS une ABI plate :
  /// loader et module DOIVENT partager la meme ABI C++. add_native_block lit la cle d'ABI du loader
  /// (adc_native_abi_key) et la COMPARE a abi_key() ; un ecart leve une erreur EXPLICITE (pas d'UB).
  /// @param limiter "none" | "minmod" | "vanleer"   @param riemann "rusanov" | "hll" | "hllc" | "roe"
  /// @param recon   "conservative" | "primitive"    @param time "explicit" | "imex"
  /// @param gamma   indice adiabatique du bloc (set_density / couplages inter-especes)
  /// @param stride  cadence du bloc (1 = chaque pas, defaut ; cf. add_block)
  void add_native_block(const std::string& name, const std::string& so_path,
                        const std::string& limiter = "minmod",
                        const std::string& riemann = "rusanov",
                        const std::string& recon = "conservative",
                        const std::string& time = "explicit", double gamma = 1.4,
                        int substeps = 1, bool evolve = true, int stride = 1);

  /// Cle d'ABI du module (compilateur + standard C++ + signature des en-tetes adc, figee a la
  /// compilation). Comparee a la cle baked dans un loader natif .so par add_native_block ; exposee
  /// aussi cote Python pour que emit_cpp_native_loader (ou un diagnostic) puisse la consulter.
  static std::string abi_key();

  /// @name Seam de bloc COMPILE AOT (parite native, sans marshaling)
  /// Pour brancher un modele genere par le DSL en COMPOSANT au moment de la COMPILATION (binaire de
  /// production Kokkos + MPI + AMR), via le gabarit libre adc::add_compiled_model<Model> de
  /// adc/runtime/dsl_block.hpp : il fabrique les fermetures avec block_builder.hpp sur le CONTEXTE
  /// REEL du System (grid_context) -- donc le bloc tourne le meme chemin que add_block (fill_boundary
  /// = halos MPI, assemble_rhs device), sans recopier les tableaux. C'est la difference avec
  /// add_compiled_block (.so + marshaling hote, prototypage runtime CPU).
  /// @{
  /// VISIBILITE DEFAUT (ADC_EXPORT) : grid_context / install_block / ensure_aux_width sont les
  /// seules methodes appelees par le gabarit en-tete add_compiled_model. Un loader .so genere (chemin
  /// DSL "production", cf. emit_cpp_native_loader / add_native_block) inline ce gabarit et doit
  /// resoudre ces symboles depuis le module _adc deja charge. Compile en -fvisibility=hidden (pybind11),
  /// le module ne les exporterait pas sans cette annotation et le dlopen du loader echouerait.
  ADC_EXPORT GridContext grid_context();  ///< maillage + CL + aux REELS du System (aux non possede)
  /// Installe un bloc a partir de fermetures deja fabriquees (cf. add_compiled_model). Les
  /// descripteurs cons/prim portent les noms ET les roles (M::conservative_vars()), exploites
  /// par les couplages inter-especes.
  ADC_EXPORT void install_block(const std::string& name, int ncomp, const VariableSet& cons_vars,
                     const VariableSet& prim_vars, double gamma, BlockClosures closures,
                     std::function<Real(const MultiFab&)> max_speed,
                     std::function<void(const MultiFab&, MultiFab&)> poisson_rhs, int substeps,
                     bool evolve, int stride = 1);
  /// Garantit que l'etat U du bloc @p name porte au moins @p n_ghost ghosts (largeur du stencil
  /// spatial). WENO5 lit 3 ghosts, > les 2 alloues par install_block ; appelee par add_compiled_model
  /// (en-tete) avec block_n_ghost(limiter) APRES install_block, pour que le chemin compile natif
  /// (loader .so) accepte weno5 -- MEME mecanisme que add_block. No-op si U a deja assez de ghosts
  /// (none/minmod/vanleer, <= 2) : allocation et donnees bit-identiques a l'historique. ADC_EXPORT :
  /// appelee par le gabarit en-tete add_compiled_model -> doit etre exportee pour le loader .so.
  ADC_EXPORT void set_block_ghosts(const std::string& name, int n_ghost);
  /// @}

  /// Configure le Poisson partage.
  /// @param rhs    seul mode : "charge_density", f = somme_s elliptic_rhs_s(u_s)
  /// @param solver "geometric_mg" (tout cas, paroi comprise) | "fft" (periodique, n = 2^k)
  /// @param bc     "auto" | "periodic" | "dirichlet" | "neumann"
  /// @param wall   "none" | "circle" : paroi conductrice en (L/2, L/2), rayon wall_radius
  /// @param epsilon permittivite CONSTANTE de l'operateur div(eps grad phi) = f. eps != 1 resout
  ///                eps lap phi = f (i.e. lap phi = f/eps). Pour une permittivite eps(x) VARIABLE,
  ///                cf. set_epsilon_field (operateur a coefficients variables, GeometricMG).
  void set_poisson(const std::string& rhs = "charge_density",
                   const std::string& solver = "geometric_mg",
                   const std::string& bc = "auto", const std::string& wall = "none",
                   double wall_radius = 0.0, double epsilon = 1.0);

  /// Fixe le DOMAINE DE TRANSPORT comme un DISQUE de centre (@p cx, @p cy) et de rayon @p R
  /// (chantier T2, CONTRAT inerte par defaut). Materialise un masque 0/1 cellule-centre (cellule
  /// active quand son centre est dans le disque, level set hypot(x-cx, y-cy) - R < 0, MEME convention
  /// que le mur conducteur du Poisson). C'est le pendant FV du mur elliptique : le papier (Hoffart et
  /// al., arXiv:2510.11808) transporte sur un vrai disque, alors qu'ADC transporte sur le carre
  /// cartesien plein avec le cercle seulement dans la paroi de Poisson (verrou "bords d'anneau
  /// cartesiens", cf. docs/HOFFART_FIDELITY.md). Le masque rend possible un transport mask-aware
  /// CONSERVATIF (flux normal nul aux faces active/inactive).
  ///
  /// MODE DE TRANSPORT DISQUE (chantier T5-PR3, @p mode) : aiguille l'avance de transport de step() vers
  /// l'operateur disque correspondant. Defaut "none" -> chemin plein cartesien (assemble_rhs), BIT-
  /// IDENTIQUE a l'historique meme apres set_disc_domain (le masque est materialise mais le transport
  /// l'ignore tant que le mode est "none"). "staircase" -> transport masque conservatif (assemble_rhs_
  /// masked, porte de face 0/1, frontiere crenelee). "cutcell" -> transport cut-cell / embedded-boundary
  /// (assemble_rhs_eb, apertures alpha_f + fraction de volume kappa, frontiere lisse, ordre 2 interieur).
  /// Le mode est honore sous Lie ET Strang (cf. set_time_scheme). Un mode != "none" sans bloc cartesien
  /// transportable leve une erreur EXPLICITE au pas (jamais un transport plein silencieux). Mode inconnu
  /// -> erreur. R > 0 requis ; cartesien seulement (le polaire borne deja l'anneau par ses parois
  /// radiales -> erreur explicite).
  void set_disc_domain(double cx, double cy, double R, const std::string& mode = "none");

  /// Fixe SEUL le mode de transport disque (sans (re)definir le disque) : "none" | "staircase" |
  /// "cutcell". Utile pour basculer le mode apres set_disc_domain, ou pour le remettre a "none"
  /// (retour au chemin plein cartesien, bit-identique). Demander un mode != "none" sans disque fixe
  /// (set_disc_domain) leve une erreur EXPLICITE (le mode seul n'a pas de geometrie a appliquer).
  void set_geometry_mode(const std::string& mode);

  /// @return le masque de domaine 0/1 cellule-centre, ny*nx row-major (j lent, i rapide). Sans
  /// set_disc_domain, renvoie un masque TOUT ACTIF (que des 1.0) : le sous-domaine de transport est
  /// le domaine entier (chemin par defaut). Diagnostic / verification du contrat.
  std::vector<double> disc_mask() const;

  /// Fixe une permittivite VARIABLE eps(x), champ n*n row-major (> 0), au CENTRE des cellules.
  /// L'operateur du Poisson de systeme passe a div(eps grad phi) = f, eps PORTE PAR L'OPERATEUR
  /// (coefficient de face harmonique, ordre 2) sans mise a l'echelle 1/eps du second membre. Seul
  /// le solveur 'geometric_mg' le supporte ; le demander avec 'fft' (coefficient constant) leve une
  /// erreur. Prevaut sur la permittivite constante de set_poisson. A appeler avant solve_fields.
  void set_epsilon_field(const std::vector<double>& eps);

  /// Fixe une permittivite ANISOTROPE eps_x(x), eps_y(x), deux champs n*n row-major (> 0), au CENTRE
  /// des cellules. L'operateur du Poisson de systeme passe a div(diag(eps_x, eps_y) grad phi) = f :
  /// les faces normales a x portent eps_x, celles normales a y portent eps_y (coefficients de face
  /// harmoniques, ordre 2), PORTES PAR L'OPERATEUR sans mise a l'echelle 1/eps du second membre.
  /// eps_x == eps_y redonne l'operateur isotrope div(eps grad phi). Seul 'geometric_mg' le supporte ;
  /// le demander avec 'fft' (coefficient constant) leve une erreur. A appeler avant solve_fields.
  void set_epsilon_anisotropic_field(const std::vector<double>& eps_x,
                                     const std::vector<double>& eps_y);

  /// Active un terme de REACTION kappa(x) >= 0 : l'operateur du Poisson de systeme passe de
  /// div(eps grad phi) = f a div(eps grad phi) - kappa phi = f (Poisson ECRANTE / Helmholtz ;
  /// kappa = 1/lambda_D^2 pour l'ecrantage de Debye). Champ n*n row-major, porte par l'operateur
  /// GeometricMG (kappa diagonal, restreint aux niveaux grossiers). Seul 'geometric_mg' le supporte
  /// (erreur avec 'fft'). Composable avec set_epsilon_field. kappa = 0 partout => Poisson inchange.
  void set_reaction_field(const std::vector<double>& kappa);

  /// Fixe un champ magnetique hors-plan B_z(x, y) PARTAGE par les blocs, n*n row-major. Peuple la
  /// composante aux supplementaire (canal B_z) lue par les modeles qui la declarent (n_aux > 3) ;
  /// inerte si aucun bloc ne lit B_z (canal aux reste a la largeur de base). B_z est statique
  /// (externe a l'elliptique) : derive_aux ne le touche pas. A appeler apres avoir ajoute le bloc
  /// (ou avant : la valeur est conservee et appliquee quand le canal aux s'elargit).
  void set_magnetic_field(const std::vector<double>& bz);

  /// Designe un bloc fluide COMPRESSIBLE (4 var) comme source de la temperature electronique T_e :
  /// le canal aux T_e (composante canonique suivante) est rempli a T = p/rho de ce bloc, RECALCULE
  /// a chaque solve_fields. N'a d'effet que si un bloc declare lire T_e (n_aux > 4) ; sinon stocke
  /// et inerte. C'est le second champ aux SUPPLEMENTAIRE (apres B_z), peuple par DERIVATION d'un
  /// bloc (et non fourni par l'utilisateur comme B_z).
  void set_electron_temperature_from(const std::string& name);

  /// Garantit que le canal aux PARTAGE a au moins @p ncomp composantes. Appele par
  /// add_compiled_model (cf. dsl_block.hpp) avec aux_comps<Model> a l'ajout d'un bloc qui lit des
  /// champs auxiliaires supplementaires. Reallouer preserve l'ADRESSE de l'aux du System (les
  /// fermetures de bloc deja installees pointent &aux), et re-applique B_z s'il a ete fourni.
  /// ADC_EXPORT : appelee par add_compiled_model (en-tete) -> doit etre exportee pour le loader .so.
  ADC_EXPORT void ensure_aux_width(int ncomp);

  /// Fixe la densite d'une espece (composante 0), tableau n*n row-major. Les autres
  /// composantes (qte de mouvement, energie) sont posees a l'equilibre au repos.
  void set_density(const std::string& name, const std::vector<double>& rho);

  /// Initialise l'etat d'un bloc depuis ses variables PRIMITIVES (rho, u, v, p ...) : @p prim est
  /// un tableau plat ncomp*n*n composante-majeur dans l'ordre de primitive_vars(name). Chaque cellule
  /// est convertie en variables CONSERVATIVES par la conversion du MODELE du bloc (M.to_conservative),
  /// puis ecrite dans l'etat. Pendant ergonomique de set_density pour un modele a plusieurs primitives
  /// (compressible 4 var : p ; isotherme 3 var ; scalaire 1 var : identite). cf. get_primitive_state.
  void set_primitive_state(const std::string& name, const std::vector<double>& prim);

  /// Lit l'etat CONSERVATIF du bloc et le convertit en variables PRIMITIVES via la conversion du
  /// modele (M.to_primitive). @return un tableau plat ncomp*n*n composante-majeur dans l'ordre de
  /// primitive_vars(name) (diagnostics : vitesses, pression). Round-trip exact avec set_primitive_state.
  std::vector<double> get_primitive_state(const std::string& name);

  /// Type-erase de la conversion PONCTUELLE (une cellule) cons <-> prim d'un bloc : in/out sont des
  /// tableaux de ncomp doubles. Installee par install_block / add_compiled_model / push_dynamic depuis
  /// le modele du bloc, consommee par set_primitive_state / get_primitive_state.
  using CellConvert = std::function<void(const double* in, double* out)>;
  /// Installe les conversions ponctuelles cons <-> prim d'un bloc (apres install_block). Appelee par
  /// le gabarit en-tete add_compiled_model (modele compile) ; le chemin natif add_block et le chemin
  /// dynamique .so les posent directement. ADC_EXPORT : resolue par le loader natif a travers le dlopen.
  ADC_EXPORT void set_block_conversion(const std::string& name, CellConvert prim_to_cons,
                                       CellConvert cons_to_prim);

  /// Installe les BORNES DE PAS optionnelles d'un bloc (apres install_block) : reduction de la
  /// frequence de source max (trait HasSourceFrequency, borne dt <= cfl*substeps/(stride*mu)) et du
  /// pas admissible min (trait HasStabilityDt, borne dt <= dt_adm*substeps/stride, sans cfl).
  /// Fonctions VIDES = le bloc n'impose pas la borne (politique de pas historique, bit-identique).
  /// Appelee par add_block et par le gabarit add_compiled_model (cf. dsl_block.hpp) avec les
  /// fermetures compilees de block_builder (make_source_frequency / make_stability_dt).
  /// ADC_EXPORT : resolue par le loader natif a travers le dlopen.
  ADC_EXPORT void set_block_dt_bounds(const std::string& name,
                                      std::function<Real(const MultiFab&)> source_frequency,
                                      std::function<Real(const MultiFab&)> stability_dt);

  /// Ajoute une borne GLOBALE de pas de temps, evaluee UNE fois par pas (hote) par step_cfl /
  /// step_adaptive : dt <= fn() quand fn() > 0 et fini (sinon la borne ne contraint pas ce pas).
  /// C'est le crochet des contraintes NON locales-cellule : couplage multi-blocs, etage
  /// Schur/Poisson, AMR/scheduler, ou une politique utilisateur (rampe de demarrage...). @p label
  /// nomme la borne dans last_dt_bound() ("global:<label>"). Une callback Python est acceptable ICI
  /// (une evaluation par pas, jamais par cellule).
  void add_dt_bound(const std::string& label, std::function<double()> fn);

  /// Nom de la borne ACTIVE (celle qui a fixe dt) du dernier step_cfl : "transport:<bloc>",
  /// "source_frequency:<bloc>", "stability_dt:<bloc>", "global:<label>", "degenerate" (aucun bloc
  /// evolutif), ou "" si aucun step_cfl n'a tourne. Diagnostic de la politique de pas.
  std::string last_dt_bound() const;

  /// Ajoute un couplage d'IONISATION (operator-split) : taux k n_e n_g ; un neutre devient un ion
  /// et un electron. Masse transferee du neutre vers l'ion (n_i + n_g conserve). Les trois blocs
  /// doivent exister. Premiere brique de source inter-especes (sur la densite, comp 0).
  void add_ionization(const std::string& electron, const std::string& ion,
                      const std::string& neutral, double rate);

  /// Ajoute une COLLISION / friction inter-especes (operator-split) : force k (u_a - u_b) sur la
  /// quantite de mouvement, opposee sur chaque espece (qte de mvt totale conservee). Les deux
  /// blocs doivent etre fluides (>= 3 variables). Echauffement par friction neglige (raffinement).
  void add_collision(const std::string& a, const std::string& b, double rate);

  /// Ajoute un ECHANGE THERMIQUE inter-especes (operator-split) : flux de chaleur k (T_a - T_b)
  /// sur l'energie, oppose sur chaque espece (energie totale conservee) ; T = p/rho. Les deux
  /// blocs doivent etre Euler compressible (4 variables, equation d'energie).
  void add_thermal_exchange(const std::string& a, const std::string& b, double rate);

  /// Active un ETAGE SOURCE condense par Schur sur le bloc @p name (splitting EXPLICITE / IMPLICITE,
  /// cf. docs/SCHUR_CONDENSATION_DESIGN.md sections 5-6). C'est l'OPT-IN de la politique adc.Split(
  /// hyperbolic=Explicit, source=CondensedSchur) : a chaque pas, le bloc fait son transport
  /// hyperbolique EXPLICITE comme aujourd'hui, PUIS cet etage source remplace la source explicite /
  /// IMEX par l'etage condense (CondensedSchurSourceStepper, #126), AUTONOME et resolu en C++ (aucun
  /// callback Python par cellule). Le chemin par defaut (sans cet appel) reste BIT-IDENTIQUE.
  ///
  /// CONTRAT (valide ICI, avant tout pas) : le bloc DOIT exposer les roles Density / MomentumX /
  /// MomentumY (Energy optionnel) dans son descripteur conservatif ; un role manquant leve une erreur
  /// EXPLICITE. Le champ B_z doit etre disponible (set_magnetic_field appele) : l'aux est elargi au
  /// canal B_z et un B_z absent leve une erreur. Le bloc reutilise le potentiel phi du Poisson de
  /// systeme comme warm start (solve_fields tourne en tete de step), mais l'etage source resout son
  /// PROPRE operateur elliptique condense (il ne duplique pas solve_fields).
  /// @param kind  seul "electrostatic_lorentz" pour l'instant (ElectrostaticLorentzCondensation).
  /// @param theta theta-schema in (0,1] (0.5 = Crank-Nicolson, 1 = Euler retrograde).
  /// @param alpha constante de couplage electrostatique du sous-systeme source (d_t(-Lap phi) =
  ///              -alpha div(rho v)).
  /// @param krylov_tol / krylov_max_iters : tolerance et budget du solve Krylov (BiCGStab) de
  ///              l'etage. <= 0 (defauts) = constantes historiques du stepper (1e-10 ; 400 en
  ///              cartesien, 600 en polaire) -- rendues configurables par l'audit 2026-06.
  /// @param density / momentum_x / momentum_y / energy : DESCRIPTEURS des champs de l'etage (audit
  ///              vague 2 : roles transportes dans l'ABI au lieu d'etre figes). Chaine VIDE
  ///              (defaut) = role canonique (Density / MomentumX / MomentumY / Energy optionnel),
  ///              bit-identique. Sinon : un NOM DE ROLE stable ("density", "momentum_x", ...) ou
  ///              un NOM DE VARIABLE du bloc -- pour un bloc a noms libres / roles Custom.
  ///              energy = "none" desactive la mise a jour d'energie. CARTESIEN seulement (le
  ///              stepper polaire ne prend pas encore d'override -> rejet explicite).
  /// @param bz_aux_component : composante du canal aux lue comme champ magnetique Omega (audit
  ///              vague 2). < 0 (defaut) = canal canonique B_z (kAuxBaseComps), bit-identique.
  void set_source_stage(const std::string& name, const std::string& kind, double theta,
                        double alpha, double krylov_tol = 0.0, int krylov_max_iters = 0,
                        const std::string& density = "", const std::string& momentum_x = "",
                        const std::string& momentum_y = "", const std::string& energy = "",
                        int bz_aux_component = -1);

  /// POLITIQUE DE SPLITTING en temps du macro-pas (transport hyperbolique H + etage source S) :
  ///  - "lie"    (defaut) : H(dt) ; S(dt) une fois (Godunov, 1er ordre). BIT-IDENTIQUE a l'historique.
  ///  - "strang"          : H(dt/2) ; S(dt) ; H(dt/2) (symetrique, 2e ordre des que H et S le sont).
  /// Le schema Strang RE-RESOUT solve_fields entre les etages pour que chaque demi-avance lise un phi
  /// coherent avec la densite courante (le solve_fields UNIQUE de tete, suffisant pour Lie, ne suffit
  /// pas a la 2nde demi-avance Strang) ; cf. docs/HOFFART_STEP_SEQUENCE.md et SystemStepper::step_strang.
  /// Reutilise les MEMES briques (s.advance, etage source) : aucun nouveau stepper. Un schema inconnu
  /// leve une erreur EXPLICITE. Sans appel, le chemin reste BIT-IDENTIQUE (Lie).
  void set_time_scheme(const std::string& scheme);

  /// Politique de la loi de Gauss (chantier R0, reproduction Hoffart). "restart" (defaut) : solve_fields
  /// re-resout -Delta phi = f a chaque pas (BIT-IDENTIQUE a l'historique). "evolve" : apres le premier
  /// solve (phi^0), solve_fields ne re-resout plus le Poisson et derive l'aux du phi COURANT que
  /// l'etage source Schur fait evoluer in-place -> evolution -Delta phi sans restart du papier (Gauss
  /// imposee qu'a t=0). N'a d'effet qu'avec un etage source condense. Schema inconnu -> erreur explicite.
  void set_gauss_policy(const std::string& policy);

  /// Ajoute une SOURCE COUPLEE GENERIQUE inter-especes decrite par un BYTECODE (adc.dsl.CoupledSource,
  /// P5 phase 1, splitting EXPLICITE forward-Euler apres le transport). A la difference des couplages
  /// nommes (add_ionization / add_collision / add_thermal_exchange) qui figent une formule, celle-ci lit
  /// des champs (bloc, role) en ENTREE et ecrit des termes de source (bloc, role) calcules par des
  /// EXPRESSIONS symboliques compilees en bytecode postfixe (machine a pile, evaluee dans le meme
  /// for_each_cell device ; aucun callback Python par cellule). Reutilise EXACTEMENT le seam
  /// d'application des couplages (P->couplings) ; MPI-safe (iteration sur les fabs locaux,
  /// local_size()==0 -> no-op).
  ///
  /// ABI PLATE (aucun objet C++ ne traverse la frontiere) :
  /// @param in_blocks  noms des blocs lus en entree (un par registre d'entree) ;
  /// @param in_roles   roles correspondants (resolus en composante par le descripteur du bloc) ;
  /// @param consts     constantes (parametres .param()), chargees apres les entrees ;
  /// @param out_blocks bloc cible de chaque terme de source ;
  /// @param out_roles  role cible de chaque terme de source ;
  /// @param prog_ops   opcodes concatenes de TOUS les termes (machine a pile, cf. CsOp) ;
  /// @param prog_args  arguments paralleles a prog_ops (indice de registre pour PushReg) ;
  /// @param prog_lens  longueur du programme de chaque terme (segmente prog_ops/prog_args dans l'ordre).
  /// @param frequency  frequence CONSTANTE declaree mu [1/s] du couplage (audit vague 3,
  ///                   CoupledSource.frequency) : borne de pas dt <= cfl / mu agregee par step_cfl /
  ///                   step_adaptive (les couplages s'appliquent UNE fois par macro-pas, la borne
  ///                   porte sur le macro-dt, sans facteur substeps/stride). <= 0 (defaut) = pas de
  ///                   borne, bit-identique.
  /// @param label      nom du couplage (raison "coupled_source:<label>" de last_dt_bound).
  /// @param freq_prog_ops/freq_prog_args  programme bytecode postfixe OPTIONNEL d'une frequence PAR
  ///                   CELLULE mu(U) (meme machine a pile que les termes, MEME table de registres :
  ///                   entrees in_blocks/in_roles puis constantes consts). VIDES (defaut) = frequence
  ///                   CONSTANTE seule (chemin historique, bit-identique). Non vides : a chaque pas,
  ///                   step_cfl / step_adaptive reduit le MAX de mu sur les cellules (all_reduce_max
  ///                   global) et borne dt <= cfl / max(mu) (raison "coupled_source:<label>", uniforme
  ///                   avec la frequence constante). max(mu) <= 0 = pas de borne ce pas.
  /// Les blocs / roles inconnus, une capacite depassee ou un programme mal forme levent une erreur
  /// EXPLICITE (avant tout pas). Sans appel, le chemin par defaut reste BIT-IDENTIQUE.
  void add_coupled_source(const std::vector<std::string>& in_blocks,
                          const std::vector<std::string>& in_roles,
                          const std::vector<double>& consts,
                          const std::vector<std::string>& out_blocks,
                          const std::vector<std::string>& out_roles,
                          const std::vector<int>& prog_ops,
                          const std::vector<int>& prog_args,
                          const std::vector<int>& prog_lens,
                          double frequency = 0.0,
                          const std::string& label = "coupled_source",
                          const std::vector<int>& freq_prog_ops = {},
                          const std::vector<int>& freq_prog_args = {});

  void solve_fields();   ///< resout Poisson puis derive aux = (phi, grad phi)
  void step(double dt);  ///< solve_fields, puis avance chaque bloc selon son schema
  void advance(double dt, int nsteps);

  /// Avance d'un pas a dt = cfl * h / vitesse d'onde max du systeme. @return le dt utilise.
  double step_cfl(double cfl);

  /// Avance d'un macro-pas MULTIRATE : le bloc le plus lent fixe le macro-pas, chaque bloc
  /// plus rapide est sous-cycle n = ceil(w_bloc / w_min) fois. @return le macro-pas.
  double step_adaptive(double cfl);

  /// @name Primitives pour un integrateur temporel ecrit en Python
  /// solve_fields(); R = eval_rhs(name); U = get_state(name); ...; set_state(name, U).
  /// @{
  std::vector<double> eval_rhs(const std::string& name);   ///< -div F + S, taille ncomp*n*n
  std::vector<double> get_state(const std::string& name);  ///< U, ncomp*n*n (composante-majeur)
  void set_state(const std::string& name, const std::vector<double>& u);
  int n_vars(const std::string& name) const;
  /// Noms des variables d'un bloc (introspection) : kind = "conservative" | "primitive".
  std::vector<std::string> variable_names(const std::string& name,
                                          const std::string& kind = "conservative") const;
  /// Roles PHYSIQUES des variables d'un bloc (parallele a variable_names) : "density",
  /// "momentum_x", "energy", ... ou "custom" si le bloc ne renseigne pas ses roles. C'est ce que
  /// resolvent les couplages inter-especes (index_of(role)) au lieu d'un indice litteral.
  std::vector<std::string> variable_roles(const std::string& name,
                                          const std::string& kind = "conservative") const;
  /// Indice adiabatique (gamma) du bloc, lu par les couplages inter-especes (collision, echange
  /// thermique, T_e). Vaut le defaut historique 1.4 sauf si le bloc le declare (add_block : ModelSpec
  /// gamma ; bloc compile / dynamique : symbole optionnel adc_compiled_gamma de l'ABI du .so).
  double block_gamma(const std::string& name) const;
  /// @}

  /// @name Diagnostics
  /// @{
  int nx() const;
  /// Compteur de MACRO-PAS (0-indexe ; incremente par step / step_cfl / step_adaptive). Necessaire
  /// au checkpoint/restart : la cadence stride (hold-then-catch-up) depend de macro_step % stride,
  /// pas seulement du temps t (audit 2026-06, IO v1).
  int macro_step() const;
  /// RESTAURE l'horloge (t, macro_step) -- reserve au RESTART (sim.restart). Restaurer macro_step
  /// est OBLIGATOIRE pour reprendre la cadence stride exactement ; un restart qui ne reposerait que
  /// t desynchroniserait les blocs a stride > 1. @throws si macro_step < 0.
  void set_clock(double t, int macro_step);
  /// Extent de l'axe LENT du champ (lignes du tableau (ny, nx) row-major rendu par density / potential
  /// / get_state). Cartesien : ny() == nx() == n (carre, INCHANGE). Polaire (anneau) : ny() == ntheta
  /// (axe azimutal lent) tandis que nx() == nr (axe radial rapide) -- avec nr != ntheta le champ fait
  /// nr*ntheta valeurs, PAS nx()^2 : c'est cette dimension qui dimensionne correctement le tableau
  /// numpy cote bindings (sans elle, un remodelage (nx, nx) deborde le tampon quand nr != ntheta).
  int ny() const;
  double time() const;
  int n_species() const;
  /// Noms des blocs, dans l'ordre d'ajout. SOURCE UNIQUE : le registre de blocs interne, peuple par
  /// TOUS les chemins d'ajout (add_block / add_dynamic_block / add_compiled_block / install_block).
  /// Un integrateur ecrit en Python itere dessus, donc il doit voir aussi les blocs charges depuis
  /// un .so (JIT / AOT) ; les compter par n_species() seul les sauterait.
  std::vector<std::string> block_names() const;
  double mass(const std::string& name) const;
  std::vector<double> density(const std::string& name) const;  ///< ny*nx row-major (j lent, i rapide)
  std::vector<double> potential();                             ///< phi, ny*nx row-major (j lent, i rapide)
  /// RESTAURE le potentiel phi (IO v1, reserve au restart) : sans lui le multigrille repartirait
  /// d'un phi vierge et la reprise ne serait pas bit-identique (warm start perdu) ; en
  /// gauss_policy="evolve", phi EST l'etat physique et sa restauration est indispensable. Champ
  /// ny*nx row-major (meme layout que potential()).
  void set_potential(const std::vector<double>& phi);

  /// @name Accesseurs GLOBAUX (collectifs MPI-safe) -- sorties / checkpoint multi-rangs (IO v1)
  /// Le System construit UNE box couvrant tout le domaine (cf. ctor : ba mono-box, dm round-robin ->
  /// box 0 sur le rang 0). Les accesseurs ci-dessus (density / get_state / potential) lisent fab(0) :
  /// VALABLES sur le rang proprietaire (mono-rang OU rang 0 sous MPI), mais fab(0) est HORS BORNES sur
  /// un rang sans box (local_size()==0). Les variantes _global remplissent un tampon GLOBAL depuis les
  /// fabs LOCAUX (en indices GLOBAUX ; rien sur un rang vide) puis all_reduce_sum_inplace -> CHAQUE
  /// rang detient le champ complet (pattern du reflux AMR, comm.hpp). Elles sont COLLECTIVES : tous les
  /// rangs DOIVENT les appeler. En mono-rang elles rendent EXACTEMENT le meme tableau que les
  /// accesseurs non-globaux (all_reduce = identite, box = domaine complet) -> sortie bit-identique.
  /// La facade IO (sim.write / sim.checkpoint) les utilise puis n'ecrit le fichier que sur le rang 0.
  /// @{
  std::vector<double> density_global(const std::string& name) const;  ///< comp0, ny*nx global
  std::vector<double> state_global(const std::string& name) const;    ///< U, ncomp*ny*nx global
  std::vector<double> potential_global();                             ///< phi, ny*nx global
  /// @}

  /// @name Accesseurs LOCAUX par fab -- ecriture HDF5 PARALLELE par hyperslabs (IO PR-IO-3, opt-in)
  /// Pendant local des accesseurs _global : au lieu de rassembler tout le champ par all_reduce_sum,
  /// ils exposent par rang la liste des boites LOCALES et l'etat de CHAQUE fab. La facade HDF5
  /// parallele (sim.write(format='hdf5', parallel=True)) cree les datasets GLOBAUX puis chaque rang
  /// ecrit SES boites en hyperslabs -- pas de gather global. Ils sont NON COLLECTIFS (purement
  /// locaux : aucune comm MPI ; un rang sans box rend une liste vide). Le System cartesien est
  /// MONO-BOX (une box couvrant le domaine, sur le rang 0) : local_boxes rend donc UNE box sur le
  /// rang 0 et rien ailleurs -- le vrai parallelisme par hyperslabs n'apparait que sur une geometrie
  /// MULTI-BOX (cf. AMR). L'API reste correcte dans le cas general (iteration sur tous les fabs
  /// locaux, indices GLOBAUX dans la box). Layout de local_state IDENTIQUE a state_global mais
  /// rapporte a la box locale : (c*bny + (j - jlo))*bnx + (i - ilo), composante-majeur.
  /// @{
  std::vector<std::array<int, 4>> local_boxes(const std::string& name) const;  ///< (ilo,jlo,ihi,jhi) par fab local
  std::vector<double> local_state(const std::string& name, int li) const;      ///< U du fab li, plat (ncomp*bny*bnx)
  /// @}
  /// @}

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace adc
