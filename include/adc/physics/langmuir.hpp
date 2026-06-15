#pragma once

/// @file
/// @brief 0D kernel of the linearized Langmuir mode (LangmuirMode): IMEX TEST/VALIDATION brick.
///
/// Kept as an analytic example of the IMEX scheme (explicit_step / implicit_solve): the plasma
/// term omega_p^2 is stiff (handled implicitly, A-stable), while the acoustic correction c_s^2 k^2
/// stays explicit. Not used by adc_cases as of 2026-06-06.

#include <adc/core/types.hpp>

#include <cmath>

namespace adc {

/**
 * Linearized Langmuir mode: 0D kernel of the asymptotic-preserving two-fluid scheme.
 *
 * TEST/VALIDATION brick (not used by adc_cases as of 2026-06-06);
 * kept as an analytic example of the IMEX scheme (explicit_step / implicit_solve).
 *
 * Stiff regime of a compressible magnetized fluid coupled to a self-consistent field,
 * Hoffart paper arXiv:2510.11808. Isothermal two-fluid, electrons mobile over a fixed
 * ionic background, a single Fourier mode k. The mode amplitude (a = perturbation,
 * b = da/dt) obeys a'' + (omega_p^2 + c_s^2 k^2) a = 0, that is an oscillation at
 * omega = sqrt(omega_p^2 + c_s^2 k^2) (isothermal Bohm-Gross). The plasma frequency
 * omega_p is the stiff term (it tends to infinity as the Debye length lambda_D tends to 0,
 * quasi-neutrality): an explicit scheme would require dt < 1/omega_p, the IMEX handles it
 * implicitly.
 *
 * IMEX split (see time/imex.hpp):
 *   stiff (implicit, A-stable): S(a,b) = (b, -omega_p^2 a),  oscillating pair
 *   slow (explicit): T(a,b) = (0, -c_s^2 k^2 a),  acoustic correction
 *
 * The implicit part solves the pair (a,b) together, backward Euler with factor
 * 1/sqrt(1+omega_p^2 dt^2) < 1: stable at fixed dt as omega_p tends to infinity (AP).
 */
struct LangmuirMode {
  Real omega_p = 1.0;  ///< plasma frequency (stiff term)
  Real cs2k2 = 0.0;    ///< c_s^2 k^2 (acoustic correction, slow)

  /**
   * Explicit step (slow acoustic term) in place: (a,b) <- (a,b) + dt T.
   *
   * @param[in]     a  mode amplitude
   * @param[in,out] b  velocity da/dt, updated by T = (0, -cs2k2 a)
   * @param[in]     dt time step
   */
  ADC_HD void explicit_step(Real& a, Real& b, Real dt) const {
    b += dt * (-cs2k2 * a);
  }

  /**
   * Implicit step (stiff plasma term) in place: solves (a,b) = (a*, b*) + dt S.
   *
   * With S = (b, -omega_p^2 a), analytic linear 2x2 solve (no Newton).
   *
   * @param[in,out] a  mode amplitude
   * @param[in,out] b  velocity da/dt
   * @param[in]     dt time step
   */
  ADC_HD void implicit_solve(Real& a, Real& b, Real dt) const {
    const Real as = a, bs = b, w2 = omega_p * omega_p;
    a = (as + dt * bs) / (1 + dt * dt * w2);
    b = bs - dt * w2 * a;
  }

  /// Eigenfrequency of the mode, omega = sqrt(omega_p^2 + c_s^2 k^2) (isothermal Bohm-Gross).
  Real omega() const { return std::sqrt(omega_p * omega_p + cs2k2); }
};

}  // namespace adc
