#pragma once

#include <adc/core/variables.hpp>  // VariableSet (descripteur a roles porte par chaque bloc)
#include <adc/runtime/grid_context.hpp>  // GridContext + BlockClosures (seam bloc compile AOT)
#include <adc/runtime/model_spec.hpp>

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
struct SystemConfig {
  int n = 64;            ///< cellules par direction (domaine n x n)
  double L = 1.0;        ///< taille du domaine carre [0,L]^2
  bool periodic = true;  ///< domaine periodique, sinon sortie libre en transport
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
  /// @param limiter  reconstruction : "none" | "minmod" | "vanleer"
  /// @param riemann  flux numerique : "rusanov" | "hllc" (hllc exige un transport a pression)
  /// @param recon    variables reconstruites : "conservative" | "primitive" (Euler : primitif
  ///                 plus robuste, positivite de rho et p)
  /// @param time     "explicit" (SSPRK2) | "imex" (transport explicite, source implicite)
  /// @param substeps sous-pas par macro-pas
  /// @param evolve   false = espece GELEE (fond fixe) : non avancee en temps, mais vue par le
  ///                 Poisson de systeme (et, a venir, par les sources couplees)
  void add_block(const std::string& name, const ModelSpec& model,
                 const std::string& limiter = "minmod",
                 const std::string& riemann = "rusanov",
                 const std::string& recon = "conservative",
                 const std::string& time = "explicit", int substeps = 1,
                 bool evolve = true);

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
  /// @param limiter "none" | "minmod" | "vanleer"   @param riemann "rusanov" | "hllc" | "roe"
  /// @param recon   "conservative" | "primitive"    @param time "explicit" | "imex"
  void add_compiled_block(const std::string& name, const std::string& so_path,
                          const std::string& limiter = "minmod",
                          const std::string& riemann = "rusanov",
                          const std::string& recon = "conservative",
                          const std::string& time = "explicit", int substeps = 1,
                          const std::vector<std::string>& names = {});

  /// @name Seam de bloc COMPILE AOT (parite native, sans marshaling)
  /// Pour brancher un modele genere par le DSL en COMPOSANT au moment de la COMPILATION (binaire de
  /// production Kokkos + MPI + AMR), via le gabarit libre adc::add_compiled_model<Model> de
  /// adc/runtime/dsl_block.hpp : il fabrique les fermetures avec block_builder.hpp sur le CONTEXTE
  /// REEL du System (grid_context) -- donc le bloc tourne le meme chemin que add_block (fill_boundary
  /// = halos MPI, assemble_rhs device), sans recopier les tableaux. C'est la difference avec
  /// add_compiled_block (.so + marshaling hote, prototypage runtime CPU).
  /// @{
  GridContext grid_context();  ///< maillage + CL + aux REELS du System (aux non possede)
  /// Installe un bloc a partir de fermetures deja fabriquees (cf. add_compiled_model). Les
  /// descripteurs cons/prim portent les noms ET les roles (M::conservative_vars()), exploites
  /// par les couplages inter-especes.
  void install_block(const std::string& name, int ncomp, const VariableSet& cons_vars,
                     const VariableSet& prim_vars, double gamma, BlockClosures closures,
                     std::function<Real(const MultiFab&)> max_speed,
                     std::function<void(const MultiFab&, MultiFab&)> poisson_rhs, int substeps,
                     bool evolve);
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
  void ensure_aux_width(int ncomp);

  /// Fixe la densite d'une espece (composante 0), tableau n*n row-major. Les autres
  /// composantes (qte de mouvement, energie) sont posees a l'equilibre au repos.
  void set_density(const std::string& name, const std::vector<double>& rho);

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
  /// @}

  /// @name Diagnostics
  /// @{
  int nx() const;
  double time() const;
  int n_species() const;
  /// Noms des blocs, dans l'ordre d'ajout. SOURCE UNIQUE : le registre de blocs interne, peuple par
  /// TOUS les chemins d'ajout (add_block / add_dynamic_block / add_compiled_block / install_block).
  /// Un integrateur ecrit en Python itere dessus, donc il doit voir aussi les blocs charges depuis
  /// un .so (JIT / AOT) ; les compter par n_species() seul les sauterait.
  std::vector<std::string> block_names() const;
  double mass(const std::string& name) const;
  std::vector<double> density(const std::string& name) const;  ///< n*n row-major
  std::vector<double> potential();                             ///< phi, n*n row-major
  /// @}

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace adc
