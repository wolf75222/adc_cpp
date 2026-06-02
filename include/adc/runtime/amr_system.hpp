#pragma once

#include <adc/runtime/model_spec.hpp>

#include <memory>
#include <string>
#include <vector>

/// @file
/// @brief Composition mono-espece sur AMR a l'execution : le pendant raffine de System.
///
/// Un bloc (une espece, decrite par une ModelSpec de briques generiques) porte sur une
/// hierarchie AMR (grossier + un niveau fin suivi par regrid, reflux conservatif). Comme
/// System mais sur grille adaptative. Le coeur ne nomme aucun scenario.
///
/// @note Un seul bloc (AmrCouplerMP est mono-modele) ; deux niveaux (ratio 2) ; traitement
///       temporel explicite (la source du modele est appliquee par le pas AMR).

namespace adc {

/// Maillage et cadence AMR (parametres physiques par bloc, dans la ModelSpec).
struct AmrSystemConfig {
  int n = 128;            ///< cellules du niveau grossier par direction
  double L = 1.0;         ///< taille du domaine carre [0,L]^2
  int regrid_every = 20;  ///< re-raffinement tous les N pas (0 = jamais apres l'init)
  bool periodic = true;   ///< domaine periodique
};

/// Bloc unique porte sur une hierarchie AMR, compose a l'execution.
class AmrSystem {
 public:
  explicit AmrSystem(const AmrSystemConfig& cfg);
  ~AmrSystem();
  AmrSystem(AmrSystem&&) noexcept;
  AmrSystem& operator=(AmrSystem&&) noexcept;

  /// Definit l'unique bloc porte sur l'AMR.
  /// @param model   composition de briques (transport/source/elliptic + parametres)
  /// @param limiter "none" | "minmod" | "vanleer"
  /// @param riemann "rusanov" | "hllc"
  /// @param time    "explicit" uniquement (l'IMEX sur AMR n'est pas cable ici)
  /// @throws std::runtime_error si un bloc est deja defini ou si time != "explicit".
  void add_block(const std::string& name, const ModelSpec& model,
                 const std::string& limiter = "minmod",
                 const std::string& riemann = "rusanov",
                 const std::string& time = "explicit", int substeps = 1);

  /// Raffine les cellules ou la densite (composante 0) depasse @p threshold.
  void set_refinement(double threshold);

  /// Configure le Poisson grossier (cf. System::set_poisson).
  void set_poisson(const std::string& rhs = "charge_density",
                   const std::string& solver = "geometric_mg",
                   const std::string& bc = "auto", const std::string& wall = "none",
                   double wall_radius = 0.0);

  /// Fixe la densite initiale sur le niveau grossier (composante 0), n*n row-major.
  void set_density(const std::string& name, const std::vector<double>& rho);

  void step(double dt);  ///< un macro-pas AMR (regrid periodique inclus)
  void advance(double dt, int nsteps);
  /// Avance a dt = cfl * dx_grossier / vitesse d'onde max. @returns le dt utilise.
  double step_cfl(double cfl);

  int nx() const;
  double time() const;
  int n_patches();                ///< nombre de patchs fins courants
  double mass();                  ///< masse sur le grossier (conservee au reflux)
  std::vector<double> density();  ///< densite grossiere (composante 0), n*n row-major

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace adc
