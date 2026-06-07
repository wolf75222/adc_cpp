#pragma once

#include <string>

/// @file
/// @brief Specification PLATE d'un modele : briques choisies + leurs parametres.
///
/// Decrit un PhysicalModel comme une composition de briques generiques (transport, source,
/// second membre elliptique) et porte leurs parametres. Aucun scenario nomme : c'est
/// l'application (adc_cases) qui nomme une composition de ces briques.
/// Type plat (POD) pour traverser les bindings sans friction.

namespace adc {

/// Composition de briques d'un bloc + parametres. Les champs ne sont lus que par la brique
/// concernee (cf. dispatch_model dans model_factory.hpp).
struct ModelSpec {
  std::string transport = "compressible";  ///< "exb" | "compressible" | "isothermal"
  std::string source = "none";             ///< "none" | "potential" | "gravity" | "magnetic"/"lorentz"
                                           ///< | "potential_magnetic"/"potential_lorentz"
  std::string elliptic = "charge";         ///< "charge" | "background" | "gravity"

  double B0 = 1.0;         ///< ExBVelocity : champ magnetique
  double gamma = 1.4;      ///< CompressibleFlux : indice adiabatique
  double cs2 = 0.5;        ///< IsothermalFlux : vitesse du son au carre
  double qom = 1.0;        ///< PotentialForce / MagneticLorentzForce : q/m (signe inclus)
  double q = 1.0;          ///< ChargeDensity : charge q
  double alpha = 1.0;      ///< BackgroundDensity : couplage Poisson
  double n0 = 0.0;         ///< BackgroundDensity : fond neutralisant
  double sign = 1.0;       ///< GravityCoupling : +1 gravite, -1 electrostatique
  double four_pi_G = 1.0;  ///< GravityCoupling : intensite du couplage
  double rho0 = 1.0;       ///< GravityCoupling : fond
};

}  // namespace adc
