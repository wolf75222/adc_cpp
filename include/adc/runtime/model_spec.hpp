#pragma once

#include <string>

/// @file
/// @brief Flat specification of a model: chosen bricks plus their parameters.
///
/// Describes a PhysicalModel as a composition of generic bricks (transport, source,
/// elliptic right-hand side) and carries their parameters. No named scenario: the
/// application (adc_cases) is the one that names a composition of these bricks.
/// Flat type (POD) to cross the bindings without friction.

namespace adc {

/// Brick composition of a block plus parameters. The fields are only read by the relevant
/// brick (see dispatch_model in model_factory.hpp).
struct ModelSpec {
  std::string transport = "compressible";  ///< "exb" | "compressible" | "isothermal"
  std::string source = "none";             ///< "none" | "potential" | "gravity" | "magnetic"/"lorentz"
                                           ///< | "potential_magnetic"/"potential_lorentz"
  std::string elliptic = "charge";         ///< "charge" | "background" | "gravity"

  double B0 = 1.0;         ///< ExBVelocity: magnetic field
  double gamma = 1.4;      ///< CompressibleFlux: adiabatic index
  double cs2 = 0.5;        ///< IsothermalFlux: sound speed squared
  double qom = 1.0;        ///< PotentialForce / MagneticLorentzForce: q/m (sign included)
  double q = 1.0;          ///< ChargeDensity: charge q
  double alpha = 1.0;      ///< BackgroundDensity: Poisson coupling
  double n0 = 0.0;         ///< BackgroundDensity: neutralizing background
  double sign = 1.0;       ///< GravityCoupling: +1 gravity, -1 electrostatic
  double four_pi_G = 1.0;  ///< GravityCoupling: coupling intensity
  double rho0 = 1.0;       ///< GravityCoupling: background
};

}  // namespace adc
