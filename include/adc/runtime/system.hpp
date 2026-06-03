#pragma once

#include <adc/runtime/model_spec.hpp>

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

  /// Configure le Poisson partage.
  /// @param rhs    seul mode : "charge_density", f = somme_s elliptic_rhs_s(u_s)
  /// @param solver "geometric_mg" (tout cas, paroi comprise) | "fft" (periodique, n = 2^k)
  /// @param bc     "auto" | "periodic" | "dirichlet" | "neumann"
  /// @param wall   "none" | "circle" : paroi conductrice en (L/2, L/2), rayon wall_radius
  /// @param epsilon permittivite CONSTANTE de l'operateur div(eps grad phi) = f. eps != 1 resout
  ///                eps lap phi = f (i.e. lap phi = f/eps). eps(x) variable demanderait un solveur
  ///                a coefficients variables (non encore disponible).
  void set_poisson(const std::string& rhs = "charge_density",
                   const std::string& solver = "geometric_mg",
                   const std::string& bc = "auto", const std::string& wall = "none",
                   double wall_radius = 0.0, double epsilon = 1.0);

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
  /// @}

  /// @name Diagnostics
  /// @{
  int nx() const;
  double time() const;
  int n_species() const;
  double mass(const std::string& name) const;
  std::vector<double> density(const std::string& name) const;  ///< n*n row-major
  std::vector<double> potential();                             ///< phi, n*n row-major
  /// @}

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace adc
