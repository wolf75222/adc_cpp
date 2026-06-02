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
  /// @param time     "explicit" (SSPRK2) | "imex" (transport explicite, source implicite)
  /// @param substeps sous-pas par macro-pas
  void add_block(const std::string& name, const ModelSpec& model,
                 const std::string& limiter = "minmod",
                 const std::string& riemann = "rusanov",
                 const std::string& time = "explicit", int substeps = 1);

  /// Configure le Poisson partage.
  /// @param rhs    seul mode : "charge_density", f = somme_s elliptic_rhs_s(u_s)
  /// @param solver "geometric_mg" (tout cas, paroi comprise) | "fft" (periodique, n = 2^k)
  /// @param bc     "auto" | "periodic" | "dirichlet" | "neumann"
  /// @param wall   "none" | "circle" : paroi conductrice en (L/2, L/2), rayon wall_radius
  void set_poisson(const std::string& rhs = "charge_density",
                   const std::string& solver = "geometric_mg",
                   const std::string& bc = "auto", const std::string& wall = "none",
                   double wall_radius = 0.0);

  /// Fixe la densite d'une espece (composante 0), tableau n*n row-major. Les autres
  /// composantes (qte de mouvement, energie) sont posees a l'equilibre au repos.
  void set_density(const std::string& name, const std::vector<double>& rho);

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
