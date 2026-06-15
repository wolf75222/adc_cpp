/// @file
/// @brief LorentzEliminator: 2x2 operator B of the Schur scheme for implicit velocity elimination.
///
/// Encodes B = [[1, -w], [w, 1]] and its inverse B^{-1} = (1/det)*[[1, w], [-w, 1]],
/// with w = theta * dt * B_z and det(B) = 1 + w^2 (always > 0: B invertible for any real w).
///
/// POD struct, zero allocation, trivially copyable. All accessors ADC_HD.
/// No std:: call: device-safe under Kokkos/CUDA/HIP without restriction.

#pragma once

#include <adc/core/types.hpp>

namespace adc {

/// LorentzEliminator: operator B = [[1,-w],[w,1]] and its analytic inverse.
///
/// Built from (theta, dt, B_z); encodes the implicit Lorentz term in a
/// Crank-Nicolson or theta-implicit scheme. Used to assemble the Schur operator
/// A = rho * B^{-1} in the implicit velocity solver.
///
/// SIGN CONVENTION: B_field = B_z z_hat in the (x,y) plane.
///   v x B_field = (v_y B_z, -v_x B_z) => B = [[1, -w], [w, 1]] with w = theta*dt*B_z.
///   Do not modify without re-reading the derivation below.
///
///   We work in 2D in the (x, y) plane with B oriented along z: B_field = B_z * z_hat.
///   The Lorentz term on the velocity v is:
///       F_L = q/m * (v x B_field)
///   In the 2D plane, with v = (v_x, v_y, 0) and B_field = (0, 0, B_z):
///       v x B_field = (v_y * B_z, -v_x * B_z, 0)
///   So the x component is +v_y*B_z and the y component is -v_x*B_z.
///
///   The operator B encodes the implicit term theta*dt*F_L in the time advance:
///       B v = v - theta*dt*(v x B_field)
///   which gives, by the convention above:
///       (B v)_x = v_x - theta*dt * v_y * B_z
///       (B v)_y = v_y + theta*dt * v_x * B_z
///   In 2x2 matrix form:
///       B = [[1, -w], [w, 1]]   with w = theta*dt*B_z
///
///   WARNING: the negative sign is on the upper-right entry (-w term for v_x)
///   and the positive sign is on the lower-left entry (+w term for v_y). This choice
///   follows DIRECTLY from v x B_field above and must not be modified.
///
/// Analytic inverse:
///   det(B) = 1 + w^2  (always > 0, B is invertible for any real w)
///   B^{-1} = (1/det) * [[1, w], [-w, 1]]
///
/// Intended use: assembly of the Schur operator A = rho * B^{-1} in the implicit
/// velocity solver. Generic: theta, dt, B_z are parameters; no physical dependency
/// hard-coded.
///
/// INVARIANT: trivially copyable struct (static_assert below), device-safe,
/// zero allocation, zero std:: call. Can be captured by value in a Kokkos/CUDA kernel.
struct LorentzEliminator {
  Real w;    // w = theta * dt * B_z
  Real det;  // det(B) = 1 + w^2

  // theta: implicitness of the scheme (0 = explicit, 1 = implicit, 0.5 = Crank-Nicolson).
  // dt: time step.
  // B_z: z component of the magnetic field.
  /// Builds from (theta, dt, B_z): w = theta*dt*B_z, det = 1 + w^2. ADC_HD.
  ADC_HD LorentzEliminator(Real theta, Real dt, Real B_z)
      : w(theta * dt * B_z), det(Real(1) + (theta * dt * B_z) * (theta * dt * B_z)) {}

  /// apply_B: applies B = [[1,-w],[w,1]] to (vx, vy), writes (Bx, By). ADC_HD.
  ADC_HD void apply_B(Real vx, Real vy, Real& Bx, Real& By) const {
    Bx = vx - w * vy;
    By = vy + w * vx;
  }

  /// apply_Binv: applies B^{-1} = (1/det)*[[1,w],[-w,1]] to (vx, vy), writes (vxp, vyp). ADC_HD.
  ADC_HD void apply_Binv(Real vx, Real vy, Real& vxp, Real& vyp) const {
    const Real inv = Real(1) / det;
    vxp = inv * (vx + w * vy);
    vyp = inv * (vy - w * vx);
  }

  /// @name Scalar entries of B^{-1} (to assemble the Schur operator A = rho * B^{-1}).
  /// @{
  ADC_HD Real binv_11() const { return Real(1) / det; }
  ADC_HD Real binv_12() const { return w / det; }
  ADC_HD Real binv_21() const { return -w / det; }
  ADC_HD Real binv_22() const { return Real(1) / det; }
  /// @}
};

// Static check: LorentzEliminator is trivially copyable (POD-like),
// guaranteeing it can be captured by value in a Kokkos/CUDA kernel
// without hidden allocation.
static_assert(
    __is_trivially_copyable(LorentzEliminator),
    "LorentzEliminator must be trivially copyable (zero allocation, device-safe)");

}  // namespace adc
