/// @file
/// @brief Interface reconstruction policies: MUSCL limiters and WENO5-Z.
///
/// Each policy exposes:
///   - `n_ghost`: required stencil radius (1 = first order, 2 = linear MUSCL, 3 = WENO5).
///   - `operator()(am, ap)`: slope limited from the backward (am) and forward (ap) differences.
///
/// All policies are ADC_HD (no std::, no branch to UB). The limiter is a template parameter
/// in assemble_rhs / reconstruct (static polymorphism, inlined on device). INVARIANT: a
/// reconstruction policy is POINTWISE -- it does not loop over the grid and does not access
/// any global array. The mesh stencil access lives in reconstruct (spatial_operator.hpp).

#pragma once

#include <adc/core/types.hpp>

#include <cmath>

namespace adc {

/// First-order reconstruction (piecewise constant): zero slope, 1 ghost.
///
/// Minimal policy: no slope computation, no read of a neighbor at distance >= 2.
/// The n_ghost == 1 path in reconstruct does not touch the cells at +/-2. ADC_HD.
/// INVARIANT: always returns Real(0) -- the cell state is not modified.
struct NoSlope {
  static constexpr int n_ghost = 1;
  ADC_HD Real operator()(Real, Real) const { return Real(0); }
};

/// minmod limiter: TVD (Total Variation Diminishing), 2 ghosts, order 2 in smooth regions.
///
/// Returns min(|a|,|b|)*sgn(a) if a and b have the same sign, 0 otherwise. Implemented without
/// std::min / std::abs to stay device-safe (no <cmath> required). Order 1 locally at extrema
/// (clips smooth peaks): prefer VanLeer for the Diocotron modes.
struct Minmod {
  static constexpr int n_ghost = 2;
  ADC_HD Real operator()(Real a, Real b) const {
    if (a * b <= Real(0)) return Real(0);
    const Real fa = a < 0 ? -a : a, fb = b < 0 ? -b : b;  // |.| device-safe
    return (fa < fb) ? a : b;
  }
};

/// van Leer limiter: smooth, 2 ghosts, better order at extrema than Minmod.
///
/// Harmonic average of the differences: 2ab/(a+b) if same sign, 0 otherwise. No sign branch
/// (no std::abs). Preferred over Minmod for preserving the Diocotron growth modes (less
/// dissipative at the density profile extrema).
struct VanLeer {
  static constexpr int n_ghost = 2;
  ADC_HD Real operator()(Real a, Real b) const {
    const Real ab = a * b;
    if (ab <= Real(0)) return Real(0);
    return Real(2) * ab / (a + b);
  }
};

/// weno5z: WENO5-Z reconstruction (Borges 2008) at one interface, on a 5-point stencil.
///
/// Returns the reconstructed value at the face BETWEEN v0 and vp1 (face +dir of cell v0).
/// For the -dir face, call weno5z(vp2, vp1, v0, vm1, vm2) (reversed stencil). ADC_HD.
/// INVARIANT: purely combinatorial computation, no branch on signs -- the beta and tau5
/// indicators are squares, always >= 0; only the absolute value of (b0-b2) is taken via a
/// ternary (device-safe, avoids std::abs).
/// Must NOT be called directly by a mesh user: go through the Weno5 policy and the reconstruct
/// function of spatial_operator.hpp.
ADC_HD inline Real weno5z(Real vm2, Real vm1, Real v0, Real vp1, Real vp2) {
  const Real eps = Real(1e-40);
  // three third-order reconstructions of the +x face of v0
  const Real q0 = (Real(2) * vm2 - Real(7) * vm1 + Real(11) * v0) / Real(6);
  const Real q1 = (-vm1 + Real(5) * v0 + Real(2) * vp1) / Real(6);
  const Real q2 = (Real(2) * v0 + Real(5) * vp1 - vp2) / Real(6);
  // smoothness indicators (Jiang-Shu)
  const Real b0 = Real(13) / Real(12) * (vm2 - 2 * vm1 + v0) * (vm2 - 2 * vm1 + v0) +
                  Real(0.25) * (vm2 - 4 * vm1 + 3 * v0) * (vm2 - 4 * vm1 + 3 * v0);
  const Real b1 = Real(13) / Real(12) * (vm1 - 2 * v0 + vp1) * (vm1 - 2 * v0 + vp1) +
                  Real(0.25) * (vm1 - vp1) * (vm1 - vp1);
  const Real b2 = Real(13) / Real(12) * (v0 - 2 * vp1 + vp2) * (v0 - 2 * vp1 + vp2) +
                  Real(0.25) * (3 * v0 - 4 * vp1 + vp2) * (3 * v0 - 4 * vp1 + vp2);
  // WENO-Z weights: alpha_k = d_k (1 + (tau5/(eps+beta_k))^2), tau5 = |beta0 - beta2|
  const Real tau5 = (b0 - b2 < 0 ? b2 - b0 : b0 - b2);
  const Real a0 = (Real(1) / Real(10)) * (Real(1) + (tau5 / (eps + b0)) * (tau5 / (eps + b0)));
  const Real a1 = (Real(6) / Real(10)) * (Real(1) + (tau5 / (eps + b1)) * (tau5 / (eps + b1)));
  const Real a2 = (Real(3) / Real(10)) * (Real(1) + (tau5 / (eps + b2)) * (tau5 / (eps + b2)));
  const Real inv = Real(1) / (a0 + a1 + a2);
  return (a0 * q0 + a1 * q1 + a2 * q2) * inv;
}

/// WENO5 tag policy: marks the stencil at 3 ghosts, delegates to weno5z.
///
/// Does not implement lim(am, ap) in the MUSCL sense (operator() is a no-op): the WENO5
/// reconstruction reads the 5-point stencil directly from reconstruct (n_ghost >= 3 path).
/// The dummy operator() is present to satisfy the Limiter concept (compatible with all
/// template functions that expect a limiter).
struct Weno5 {
  static constexpr int n_ghost = 3;
  ADC_HD Real operator()(Real, Real) const { return Real(0); }
};

}  // namespace adc
