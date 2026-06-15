#pragma once

/// @file
/// @brief Linear two-fluid isothermal electrostatic kernel (TwoFluidLinear): TEST/VALIDATION brick.
///
/// Generalizes LangmuirMode to two mobile species (electrons omega_pe + ions omega_pi) with
/// isothermal pressures. Lets you check the electrostatic dispersion branches
/// (w_fast Langmuir, w_slow ion-acoustic). Not used by adc_cases as of 2026-06-06.

#include <adc/core/types.hpp>

#include <cmath>

namespace adc {

/**
 * Isothermal two-fluid electrostatic, linear mode (single Fourier k).
 *
 * TEST/VALIDATION brick (not used by adc_cases as of 2026-06-06);
 * kept as an analytical example of the two-species IMEX scheme and to check
 * the electrostatic dispersion branches (w_fast Langmuir, w_slow ion-acoustic).
 *
 * Generalizes LangmuirMode to two mobile species, electrons (omega_pe) and ions
 * (omega_pi), with isothermal pressures (sound speeds c_se, c_si). This is the linear
 * kernel of the two-fluid AP scheme (Hoffart regime) once the ions are freed.
 *
 * The density perturbation amplitudes (A_e, A_i) of a mode obey A'' = K A
 * (E eliminated by Poisson):
 *
 *     A''_e = -(c_se^2 k^2 + omega_pe^2) A_e + omega_pe^2 A_i
 *     A''_i =  omega_pi^2 A_e            - (c_si^2 k^2 + omega_pi^2) A_i
 *
 * The eigenfrequencies are the two branches, Langmuir (high frequency) and
 * ion-acoustic (low frequency), roots of the electrostatic dispersion
 * (X - c_se^2k^2 - omega_pe^2)(X - c_si^2k^2 - omega_pi^2) = omega_pe^2 omega_pi^2 with
 * X = omega^2. The plasma frequency (omega_pe, omega_pi) is the stiff term (tends to
 * infinity as lambda_D tends to 0): handled implicitly (A-stable 2x2 solve),
 * the acoustic part (c_s^2 k^2) staying explicit.
 */
struct TwoFluidLinear {
  Real omega_pe = 1, omega_pi = 0;  ///< plasma frequencies electron / ion
  Real cse2k2 = 0, csi2k2 = 0;      ///< c_se^2 k^2, c_si^2 k^2 (acoustic terms)

  /**
   * Explicit step (slow acoustic term) in place: B_s += dt (-c_s^2 k^2 A_s).
   *
   * @param[in]     Ae, Ai electron / ion amplitudes
   * @param[in,out] Be, Bi electron / ion velocities dB/dt, updated
   * @param[in]     dt     time step
   */
  ADC_HD void explicit_step(Real& Ae, Real& Ai, Real& Be, Real& Bi, Real dt) const {
    Be += dt * (-cse2k2 * Ae);
    Bi += dt * (-csi2k2 * Ai);
  }

  /**
   * Implicit step (stiff plasma term) in place: (A,B) <- (A*,B*) + dt S.
   *
   * With S = (B, M_s A) and M_s = [[-wpe2, wpe2], [wpi2, -wpi2]], solves the 2x2 system
   * (I - dt^2 M_s) A = A* + dt B* then updates B.
   *
   * @param[in,out] Ae, Ai electron / ion amplitudes
   * @param[in,out] Be, Bi electron / ion velocities dB/dt
   * @param[in]     dt     time step
   */
  ADC_HD void implicit_solve(Real& Ae, Real& Ai, Real& Be, Real& Bi, Real dt) const {
    const Real wpe2 = omega_pe * omega_pe, wpi2 = omega_pi * omega_pi, d2 = dt * dt;
    const Real re = Ae + dt * Be, ri = Ai + dt * Bi;  // known right-hand sides
    const Real a = 1 + d2 * wpe2, b = -d2 * wpe2;      // I - dt^2 M_s
    const Real c = -d2 * wpi2, dd = 1 + d2 * wpi2;
    const Real det = a * dd - b * c;
    Ae = (dd * re - b * ri) / det;
    Ai = (-c * re + a * ri) / det;
    Be = Be + dt * (-wpe2 * Ae + wpe2 * Ai);  // B* + dt M_s A^{n+1}
    Bi = Bi + dt * (wpi2 * Ae - wpi2 * Ai);
  }

  /**
   * Roots of the dispersion relation, w_fast (Langmuir) >= w_slow (ion-acoustic) >= 0.
   *
   * @param[out] w_fast high-frequency branch (Langmuir)
   * @param[out] w_slow low-frequency branch (ion-acoustic)
   */
  void dispersion(Real& w_fast, Real& w_slow) const {
    const Real wpe2 = omega_pe * omega_pe, wpi2 = omega_pi * omega_pi;
    const Real S = cse2k2 + csi2k2 + wpe2 + wpi2;  // = w_fast^2 + w_slow^2
    const Real P = cse2k2 * csi2k2 + wpe2 * csi2k2 + wpi2 * cse2k2;  // = product
    const Real disc = std::sqrt(std::fmax(S * S - 4 * P, Real(0)));
    w_fast = std::sqrt(Real(0.5) * (S + disc));
    w_slow = std::sqrt(std::fmax(Real(0.5) * (S - disc), Real(0)));
  }
};

}  // namespace adc
