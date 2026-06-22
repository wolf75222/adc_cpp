#pragma once

/// @file
/// @brief EllipticSolver concept: common contract for elliptic solvers at the MultiFab level
///        (solve D phi = f), so couplers depend on the concept and not on a concrete class.
///
/// Layer: `include/adc/numerics/elliptic/interface`.
/// Role: express the "elliptic solver" dependency through a C++20 concept rather than a
/// hard-coded GeometricMG, which prepares swapping MG for another backend (FFT wrapper, PETSc,
/// Hypre) without touching the coupling logic.
/// Contract: an EllipticSolver exposes rhs() -> MultiFab& (right-hand side f, written before solve),
/// phi() -> MultiFab& (solution read after solve, kept between calls for the warm start),
/// solve() (solves phi from rhs in place), residual() -> Real (residual norm ||D phi - f||),
/// geom() -> const Geometry& (geometry of the solved level).
///
/// Invariants:
/// - the contract is at the MultiFab level: poisson_fft.hpp (slabs + raw vectors) does NOT model it
///   directly; PoissonFFTSolver/DistributedFFTSolver are what wrap it;
/// - phi() is kept between calls (warm start): do NOT assume an implicit reset to zero.

#include <adc/core/foundation/types.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/multifab.hpp>

#include <concepts>

namespace adc {

template <class S>
concept EllipticSolver = requires(S s) {
  { s.rhs() } -> std::same_as<MultiFab&>;
  { s.phi() } -> std::same_as<MultiFab&>;
  s.solve();
  { s.residual() } -> std::convertible_to<Real>;
  { s.geom() } -> std::convertible_to<const Geometry&>;
};

}  // namespace adc
