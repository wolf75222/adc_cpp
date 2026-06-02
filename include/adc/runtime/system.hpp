#pragma once

#include <memory>
#include <string>
#include <vector>

/// @file
/// @brief Composition multi-especes a l'execution : un systeme couple, bloc par bloc.
///
/// Chaque bloc est une espece (un etat U) avec son modele physique, son schema spatial
/// (limiteur + flux), son traitement temporel (explicite / IMEX) et son nombre de
/// sous-pas. Tous les blocs partagent un Poisson dont le second membre est la somme des
/// contributions par modele, f = somme_s elliptic_rhs_s(u_s) ; la source S agit par bloc.
/// La liste de blocs est dynamique, les noyaux par cellule restent compiles : aucun
/// callback Python dans le hot path, sauf integrateur temporel ecrit en Python pilotant
/// eval_rhs / get_state / set_state.
///
/// Modeles (tag, elliptic_rhs) :
///   "diocotron"      derive E x B, 1 var, fond n_i0 ;        alpha*(n - n_i0)
///   "electron_euler" Euler + force electrostatique, 4 var ;  q*n
///   "ion_isothermal" Euler isotherme + force, 3 var ;        q*n
///   "euler_poisson"  Euler + champ self-consistent, 4 var ;  signe*4piG*(rho - rho0)

namespace adc {

/// Parametres partages par tous les blocs d'un System. Un champ par modele n'est lu que
/// par le tag concerne.
struct SystemConfig {
  int n = 64;             ///< cellules par direction (domaine n x n)
  double L = 1.0;         ///< taille du domaine carre [0,L]^2
  double B0 = 1.0;        ///< champ magnetique, derive E x B ("diocotron")
  double n_i0 = 0.0;      ///< fond ionique neutralisant ("diocotron")
  double alpha = 1.0;     ///< constante de couplage Poisson ("diocotron")
  double gamma = 1.4;     ///< indice adiabatique ("electron_euler", "euler_poisson")
  double cs2 = 0.5;       ///< vitesse du son au carre, isotherme ("ion_isothermal")
  double four_pi_G = 1.0; ///< intensite de couplage ("euler_poisson")
  double rho0 = 1.0;      ///< fond neutralisant ("euler_poisson", solvabilite periodique)
  bool periodic = true;   ///< domaine periodique, sinon sortie libre en transport
};

/// Systeme multi-especes couple, compose a l'execution. Python choisit QUOI assembler,
/// le calcul par cellule reste en C++ compile.
class System {
 public:
  explicit System(const SystemConfig& cfg);
  ~System();
  System(System&&) noexcept;
  System& operator=(System&&) noexcept;

  /// Ajoute un bloc d'equation (une espece).
  /// @param charge   signe utilise dans elliptic_rhs des modeles de fluide charge
  /// @param limiter  reconstruction : "none" | "minmod" | "vanleer"
  /// @param flux     "rusanov" | "hllc" (hllc exige un modele Euler a 4 variables)
  /// @param time     "explicit" (SSPRK2) | "imex" (transport explicite, source implicite)
  /// @param substeps sous-pas par macro-pas (p.ex. 10 electrons pour 1 ion)
  void add_block(const std::string& name, const std::string& model, double charge,
                 const std::string& limiter = "minmod",
                 const std::string& flux = "rusanov",
                 const std::string& time = "explicit", int substeps = 1);

  /// Raccourci de add_block(..., "minmod", "rusanov", "explicit", 1).
  void add_species(const std::string& name, const std::string& model, double charge);

  /// Configure le Poisson partage.
  /// @param rhs    seul mode : "charge_density", f = somme_s q_s n_s
  /// @param solver "geometric_mg" (tout cas, paroi comprise) | "fft" (periodique, n = 2^k)
  /// @param bc     "auto" | "periodic" | "dirichlet" | "neumann"
  /// @param wall   "none" | "circle" : paroi conductrice en (L/2, L/2), rayon wall_radius,
  ///               imposee en embedded boundary ; exige solver = "geometric_mg"
  void set_poisson(const std::string& rhs = "charge_density",
                   const std::string& solver = "geometric_mg",
                   const std::string& bc = "auto", const std::string& wall = "none",
                   double wall_radius = 0.0);

  /// Fixe la densite d'une espece (composante 0), tableau n*n row-major. Quantite de
  /// mouvement et energie posees a l'equilibre au repos.
  void set_density(const std::string& name, const std::vector<double>& rho);

  void solve_fields();   ///< resout Poisson puis derive aux = (phi, grad phi)
  void step(double dt);  ///< solve_fields, puis avance chaque bloc selon son schema
  void advance(double dt, int nsteps);

  /// Avance d'un pas a dt = cfl * h / vitesse d'onde max du systeme. @return le dt utilise.
  double step_cfl(double cfl);

  /// @name Primitives pour un integrateur temporel ecrit en Python
  /// Pilotage depuis Python (par pas), residu et Poisson restant en C++ (par cellule) :
  /// solve_fields(); R = eval_rhs(name); U = get_state(name); ...; set_state(name, U).
  /// @{
  std::vector<double> eval_rhs(const std::string& name);   ///< -div F + S, taille ncomp*n*n
  std::vector<double> get_state(const std::string& name);  ///< U, ncomp*n*n (composante-majeur)
  void set_state(const std::string& name, const std::vector<double>& u);
  int n_vars(const std::string& name) const;
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
