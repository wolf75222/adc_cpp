#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/multifab.hpp>

/// @file
/// @brief Asymptotic-preserving IMEX (implicit-explicit) integrator: imex_euler_step, order-1
///        forward-backward Euler step, U^{n+1} = U^n + dt T(U^n) + dt S(U^{n+1}).
///
/// Layer: `include/adc/numerics/time`.
/// Role: take the STIFF terms (Lorentz, Debye limit, quasi-neutrality) IMPLICITLY and the
///       transport EXPLICITLY. AP property: when the small parameter (lambda_D^2, 1/omega_c)
///       -> 0, the scheme stays stable at FIXED dt and captures the limit dynamics.
/// Contract: Texpl(U, dt) advances the transport IN PLACE (after the call U holds the known term
///           U^n + dt T(U^n)); Simpl(U, dt) solves IN PLACE U <- W with W = U + dt S(W), U being
///           the known term (linear relaxation: analytic; full Lorentz: local Newton).
///
/// Invariants:
/// - integrator agnostic of the model: Texpl/Simpl are callables (MultiFab&, Real)->void;
/// - the order is enforced -- explicit THEN implicit; no state held by the integrator.

namespace adc {

template <class TransportStep, class ImplicitSourceSolve>
void imex_euler_step(MultiFab& U, Real dt, TransportStep Texpl, ImplicitSourceSolve Simpl) {
  Texpl(U, dt);  // explicit: U becomes the known term U^n + dt T(U^n)
  Simpl(U, dt);  // implicit: solves U = known + dt S(U) (stiff source) in place
}

}  // namespace adc
