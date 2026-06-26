#pragma once

/// @file
/// @brief Scalar advection-diffusion model (AdvectionDiffusion): VALIDATION/REFERENCE brick.
///
/// Canonical home of the validation/reference bricks (ADC-329): kept OUT of the production
/// brick surface (pops/physics/{hyperbolic,source,elliptic,composite}.hpp) and exposed under the
/// dedicated namespace pops::validation so it cannot be mistaken for a generic production brick.
///
/// Used in the core C++ tests (tests/test_weno5_ssprk3.cpp) as a reference model for WENO5/SSPRK3;
/// not used by adc_cases. Kept as a validation brick and as an example illustrating the
/// DiffusiveModel trait (nu > 0 enables the parabolic term on the core side).
///
/// The legacy include path <pops/physics/advection_diffusion.hpp> still works through a compat
/// forwarder that aliases pops::validation::AdvectionDiffusion back into pops:: (deprecated).

#include <pops/core/state/state.hpp>
#include <pops/core/foundation/types.hpp>

#include <cmath>

namespace pops::validation {

/**
 * Scalar advection-diffusion: d_t u + a . grad u = nu Lap u.
 *
 * VALIDATION/REFERENCE brick (not used by adc_cases as of 2026-06-06);
 * kept as an example and as a reference model for tests/test_weno5_ssprk3.cpp.
 *
 * Illustrates the remark "diffusion is just one more flux": we keep the
 * advection PhysicalModel and add only the diffusivity() method. The
 * parabolic term +nu Lap(u) is then provided by the core (assemble_rhs detects the
 * DiffusiveModel trait), without touching the coupler. nu = 0 recovers pure advection.
 */
struct AdvectionDiffusion {
  using State = StateVec<1>;        ///< one conservative variable: the scalar u
  using Aux = pops::Aux;             ///< auxiliary fields (unused, not coupled)
  static constexpr int n_vars = 1;  ///< number of conservative variables

  Real ax = 1.0;  ///< advection velocity in x
  Real ay = 0.0;  ///< advection velocity in y
  Real nu = 0.0;  ///< diffusivity (0 = pure advection)

  /// Advection flux F = a u in direction dir.
  POPS_HD State flux(const State& u, const Aux&, int dir) const {
    return State{(dir == 0 ? ax : ay) * u[0]};  // F = a u
  }
  /// Maximum wave speed: magnitude of the advection velocity in direction dir.
  POPS_HD Real max_wave_speed(const State&, const Aux&, int dir) const {
    const Real v = (dir == 0) ? ax : ay;
    return v < 0 ? -v : v;
  }
  /// Zero source term.
  POPS_HD State source(const State&, const Aux&) const { return State{Real(0)}; }
  /// Poisson right-hand side: u (not coupled here, present for the concept).
  POPS_HD Real elliptic_rhs(const State& u) const { return u[0]; }
  /// Diffusivity nu: enables the parabolic term on the core side (DiffusiveModel trait).
  POPS_HD Real diffusivity() const { return nu; }
};

}  // namespace pops::validation
