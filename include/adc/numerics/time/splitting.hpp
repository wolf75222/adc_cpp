#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/multifab.hpp>

/// @file
/// @brief Operator splitting: decomposes dU/dt = T(U) + S(U) into separate substeps.
///        lie_step (Godunov, 1st order) = T(dt) then S(dt); strang_step (2nd order) =
///        S(dt/2), T(dt), S(dt/2).
///
/// Layer: `include/adc/numerics/time`.
/// Role: handle a stiff source (relaxation, collisions, ionization) with an integrator
///       DIFFERENT from the transport one, without mixing the two stiffnesses.
/// Contract: T and S are callables (MultiFab&, Real)->void that advance their subsystem
///           IN PLACE; the integrator is agnostic to the contents.
///
/// Invariants:
/// - Strang is 2nd order as soon as each sub-integrator is (commutation error [T,S] is
///   O(dt^3) per step, O(dt^2) globally).

namespace adc {

template <class TransportStep, class SourceStep>
void lie_step(MultiFab& U, Real dt, TransportStep T, SourceStep S) {
  T(U, dt);
  S(U, dt);
}

template <class TransportStep, class SourceStep>
void strang_step(MultiFab& U, Real dt, TransportStep T, SourceStep S) {
  S(U, Real(0.5) * dt);
  T(U, dt);
  S(U, Real(0.5) * dt);
}

}  // namespace adc
