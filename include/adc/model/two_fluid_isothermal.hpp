#pragma once

#include <adc/core/types.hpp>

#include <cmath>

namespace adc {

/**
 * Deux-fluides isotherme electrostatique, mode lineaire (un Fourier k).
 *
 * Generalise LangmuirMode aux deux especes mobiles, electrons (omega_pe) et ions
 * (omega_pi), avec pressions isothermes (vitesses du son c_se, c_si). C'est le noyau
 * lineaire du schema AP deux-fluides (regime Hoffart) une fois les ions liberes.
 *
 * Les amplitudes des perturbations de densite (A_e, A_i) d'un mode obeissent a A'' = K A
 * (E elimine par Poisson) :
 *
 *     A''_e = -(c_se^2 k^2 + omega_pe^2) A_e + omega_pe^2 A_i
 *     A''_i =  omega_pi^2 A_e            - (c_si^2 k^2 + omega_pi^2) A_i
 *
 * Les frequences propres sont les deux branches, Langmuir (haute frequence) et
 * ion-acoustique (basse frequence), racines de la dispersion electrostatique
 * (X - c_se^2k^2 - omega_pe^2)(X - c_si^2k^2 - omega_pi^2) = omega_pe^2 omega_pi^2 avec
 * X = omega^2. La frequence plasma (omega_pe, omega_pi) est le terme raide (tend vers
 * l'infini quand lambda_D tend vers 0) : traitee en implicite (solve 2x2 A-stable),
 * l'acoustique (c_s^2 k^2) restant explicite.
 */
struct TwoFluidLinear {
  Real omega_pe = 1, omega_pi = 0;  ///< frequences plasma electron / ion
  Real cse2k2 = 0, csi2k2 = 0;      ///< c_se^2 k^2, c_si^2 k^2 (termes acoustiques)

  /**
   * Pas explicite (terme acoustique lent) en place : B_s += dt (-c_s^2 k^2 A_s).
   *
   * @param[in]     Ae, Ai amplitudes electron / ion
   * @param[in,out] Be, Bi vitesses dB/dt electron / ion, mises a jour
   * @param[in]     dt     pas de temps
   */
  ADC_HD void explicit_step(Real& Ae, Real& Ai, Real& Be, Real& Bi, Real dt) const {
    Be += dt * (-cse2k2 * Ae);
    Bi += dt * (-csi2k2 * Ai);
  }

  /**
   * Pas implicite (terme plasma raide) en place : (A,B) <- (A*,B*) + dt S.
   *
   * Avec S = (B, M_s A) et M_s = [[-wpe2, wpe2], [wpi2, -wpi2]], resout le systeme 2x2
   * (I - dt^2 M_s) A = A* + dt B* puis met B a jour.
   *
   * @param[in,out] Ae, Ai amplitudes electron / ion
   * @param[in,out] Be, Bi vitesses dB/dt electron / ion
   * @param[in]     dt     pas de temps
   */
  ADC_HD void implicit_solve(Real& Ae, Real& Ai, Real& Be, Real& Bi, Real dt) const {
    const Real wpe2 = omega_pe * omega_pe, wpi2 = omega_pi * omega_pi, d2 = dt * dt;
    const Real re = Ae + dt * Be, ri = Ai + dt * Bi;  // membres connus
    const Real a = 1 + d2 * wpe2, b = -d2 * wpe2;      // I - dt^2 M_s
    const Real c = -d2 * wpi2, dd = 1 + d2 * wpi2;
    const Real det = a * dd - b * c;
    Ae = (dd * re - b * ri) / det;
    Ai = (-c * re + a * ri) / det;
    Be = Be + dt * (-wpe2 * Ae + wpe2 * Ai);  // B* + dt M_s A^{n+1}
    Bi = Bi + dt * (wpi2 * Ae - wpi2 * Ai);
  }

  /**
   * Racines de la relation de dispersion, w_fast (Langmuir) >= w_slow (ion-acoustique) >= 0.
   *
   * @param[out] w_fast branche haute frequence (Langmuir)
   * @param[out] w_slow branche basse frequence (ion-acoustique)
   */
  void dispersion(Real& w_fast, Real& w_slow) const {
    const Real wpe2 = omega_pe * omega_pe, wpi2 = omega_pi * omega_pi;
    const Real S = cse2k2 + csi2k2 + wpe2 + wpi2;  // = w_fast^2 + w_slow^2
    const Real P = cse2k2 * csi2k2 + wpe2 * csi2k2 + wpi2 * cse2k2;  // = produit
    const Real disc = std::sqrt(std::fmax(S * S - 4 * P, Real(0)));
    w_fast = std::sqrt(Real(0.5) * (S + disc));
    w_slow = std::sqrt(std::fmax(Real(0.5) * (S - disc), Real(0)));
  }
};

}  // namespace adc
