#pragma once

#include <string>

/// @file
/// @brief Flat specification of a model: chosen bricks plus their parameters.
///
/// Describes a PhysicalModel as a composition of generic bricks (transport, source,
/// elliptic right-hand side) and carries their parameters. No named scenario: the
/// application (adc_cases) is the one that names a composition of these bricks.
/// Flat type (POD) to cross the bindings without friction.

namespace pops {

/// Brick composition of a block plus parameters. The fields are only read by the relevant
/// brick (see dispatch_model in model_factory.hpp).
///
/// CONTRACT (ADC-290): a ModelSpec carries NO silent physics default. `transport` and `elliptic`
/// are UNSET by default (empty string) and MUST be chosen explicitly; an unset tag is rejected with
/// a clear message by validate_model_spec (model_factory.hpp) instead of silently selecting a model
/// (the old defaults `transport="compressible"` / `elliptic="charge"` made a default-constructed
/// ModelSpec mean Euler + Poisson-charge by accident). `source` keeps the only default, "none" --
/// the EXPLICIT, neutral "no source" choice, not a physics selection. The numeric parameters keep
/// their defaults: each is read only once its brick has been chosen by a tag, so it can never inject
/// physics on its own. Historical shortcuts live at the Python edge (pops.Model(...)), which always
/// sets the three tags. See docs/adr/ADR-0001-genericity-contracts.md.
struct ModelSpec {
  std::string transport;        ///< REQUIRED (unset): "exb" | "compressible" | "isothermal"
  std::string source = "none";  ///< "none" (default, neutral: no force) | "potential" | "gravity"
      ///< | "magnetic"/"lorentz" | "potential_magnetic"/"potential_lorentz"
  std::string elliptic;  ///< REQUIRED (unset): "charge" | "background" | "gravity"

  double B0 = 1.0;            ///< ExBVelocity: magnetic field
  double gamma = 1.4;         ///< CompressibleFlux: adiabatic index
  double cs2 = 0.5;           ///< IsothermalFlux: sound speed squared
  double vacuum_floor = 0.0;  ///< IsothermalFlux: quasi-vacuum density floor for u=m/max(rho,floor)
                              ///< (ADC-77). Set from pops.FluidState(vacuum_floor=...) at compose;
  ///< INDEPENDENT of the spatial positivity_floor (deliberately decoupled --
  ///< coupling them shifts the CFL dt of existing positivity_floor runs).
  ///< 0 = off (bit-identical)
  double qom = 1.0;        ///< PotentialForce / MagneticLorentzForce: q/m (sign included)
  double q = 1.0;          ///< ChargeDensity: charge q
  double alpha = 1.0;      ///< BackgroundDensity: Poisson coupling
  double n0 = 0.0;         ///< BackgroundDensity: neutralizing background
  double sign = 1.0;       ///< GravityCoupling: +1 gravity, -1 electrostatic
  double four_pi_G = 1.0;  ///< GravityCoupling: coupling intensity
  double rho0 = 1.0;       ///< GravityCoupling: background
};

}  // namespace pops
