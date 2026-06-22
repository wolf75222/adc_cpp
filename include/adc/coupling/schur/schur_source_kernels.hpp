#pragma once

#include <adc/core/foundation/types.hpp>  // Real, ADC_HD
#include <adc/mesh/fab2d.hpp>  // Array4, ConstArray4

#include <stdexcept>
#include <string>

/// @file
/// @brief Geometry-INDEPENDENT device kernels shared by the Schur SOURCE-STAGE steppers: the
///        Cartesian condensed_schur_source_stepper.hpp, the polar
///        polar_condensed_schur_source_stepper.hpp, and (transitively) the AMR variant. These
///        functors carry NO geometry: the polar metric (1/r, r-cell math) enters ONLY the
///        reconstruct / RHS-assemble / operator-coeff / explicit-flux kernels, which stay LOCAL to
///        each stepper. Factored out (ADC-263) so the byte-identical copies stop drifting
///        independently.
///
/// MINIMAL DEPENDENCIES (the reason this header exists). It depends ONLY on the lightweight POD
///   handles (Array4 / ConstArray4) plus the scalar types -- NOT on the elliptic solver stack
///   (geometric_mg.hpp / krylov_solver.hpp). The polar path deliberately avoids that stack, so it
///   can now include these shared kernels instead of re-declaring them.
///
/// NAMED device-clean functors (recipe #93: no extended lambda first-instantiated cross-TU, nvcc
///   limit #64/#97): each captures Array4 handles (POD) plus scalars.

namespace adc {

namespace detail {

/// Linear extrapolation of a SCALAR field from the theta-stage to the full step: f^{n+1} = f^n + (1/theta)
/// (f^{n+theta} - f^n). theta = 1 -> identity (inv_theta = 1). NAMED device-clean functor.
struct SchurExtrapolateScalarKernel {
  ConstArray4 f_n;  ///< f^n
  Array4 f;         ///< IN: f^{n+theta}; OUT: f^{n+1}
  Real inv_theta;   ///< 1 / theta
  ADC_HD void operator()(int i, int j) const {
    f(i, j, 0) = f_n(i, j, 0) + inv_theta * (f(i, j, 0) - f_n(i, j, 0));
  }
};

/// Linear extrapolation of the VELOCITY (vx, vy) from the theta-stage to the full step, then recompose
/// mom = rho^n v^{n+1} into the state. rho frozen. NAMED device-clean functor.
struct SchurExtrapolateVelocityKernel {
  ConstArray4 vx_n, vy_n;  ///< v^n
  Array4 vx, vy;           ///< IN: v^{n+theta}; OUT: v^{n+1}
  Array4 st;               ///< state (WRITE mx, my; READ rho)
  Real inv_theta;          ///< 1 / theta
  int c_rho, c_mx, c_my;
  ADC_HD void operator()(int i, int j) const {
    const Real nx = vx_n(i, j, 0) + inv_theta * (vx(i, j, 0) - vx_n(i, j, 0));
    const Real ny = vy_n(i, j, 0) + inv_theta * (vy(i, j, 0) - vy_n(i, j, 0));
    vx(i, j, 0) = nx;
    vy(i, j, 0) = ny;
    const Real rho = st(i, j, c_rho);
    st(i, j, c_mx) = rho * nx;
    st(i, j, c_my) = rho * ny;
  }
};

/// Energy update: E^{n+1} = E^n + (1/2) rho^n (|v^{n+1}|^2 - |v^n|^2). NAMED device-clean
/// functor. Applied only when the Energy role is present (host-side guard).
struct SchurEnergyKernel {
  ConstArray4 vx_n, vy_n;  ///< v^n
  ConstArray4 vx, vy;      ///< v^{n+1}
  Array4 st;               ///< state (WRITE E; READ rho)
  int c_rho, c_E;
  ADC_HD void operator()(int i, int j) const {
    const Real rho = st(i, j, c_rho);
    const Real ke_old =
        Real(0.5) * rho * (vx_n(i, j, 0) * vx_n(i, j, 0) + vy_n(i, j, 0) * vy_n(i, j, 0));
    const Real ke_new = Real(0.5) * rho * (vx(i, j, 0) * vx(i, j, 0) + vy(i, j, 0) * vy(i, j, 0));
    st(i, j, c_E) += ke_new - ke_old;
  }
};

/// Extracts the velocity v = (mx, my) / rho from the state (Density / MomentumX / MomentumY roles) into two
/// scalar fields vx, vy. rho = 0 -> velocity 0 (safeguard; in practice the source freezes rho > 0).
/// NAMED device-clean functor.
struct ExtractVelocityKernel {
  ConstArray4 st;
  Array4 vx, vy;
  int c_rho, c_mx, c_my;
  ADC_HD void operator()(int i, int j) const {
    const Real rho = st(i, j, c_rho);
    const Real inv = rho != Real(0) ? Real(1) / rho : Real(0);
    vx(i, j, 0) = st(i, j, c_mx) * inv;
    vy(i, j, 0) = st(i, j, c_my) * inv;
  }
};

/// Copies the B_z field (aux channel) into an internal scalar MultiFab (0 ghost is enough, read at (i,j)).
/// NAMED device-clean functor.
struct CopyBzKernel {
  ConstArray4 bz_src;
  Array4 bz_dst;
  int c_bz;
  ADC_HD void operator()(int i, int j) const { bz_dst(i, j, 0) = bz_src(i, j, c_bz); }
};

/// Validates the Krylov tolerance / iteration budget shared by the Schur source steppers (historical
/// constants made configurable by the audit 2026-06). @p who names the calling class for the message.
/// @throws std::invalid_argument if tol <= 0 or max_iters < 1.
inline void validate_krylov_params(Real tol, int max_iters, const char* who) {
  if (!(tol > Real(0)) || max_iters < 1)
    throw std::invalid_argument(std::string(who) + ": tol > 0, max_iters >= 1");
}

}  // namespace detail

}  // namespace adc
