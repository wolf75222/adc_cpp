#pragma once

#include <adc/core/types.hpp>

#include <cmath>

namespace adc {

/**
 * Mode de Langmuir linearise : noyau 0D du schema asymptotic-preserving deux-fluides.
 *
 * Regime raide d'un fluide compressible magnetise couple a un champ self-consistant,
 * papier Hoffart arXiv:2510.11808. Deux-fluides
 * isotherme, electrons mobiles sur fond ionique fixe, un mode de Fourier k. L'amplitude
 * du mode (a = perturbation, b = da/dt) obeit a a'' + (omega_p^2 + c_s^2 k^2) a = 0, soit
 * une oscillation a omega = sqrt(omega_p^2 + c_s^2 k^2) (Bohm-Gross isotherme). La
 * frequence plasma omega_p est le terme raide (tend vers l'infini quand la longueur de
 * Debye lambda_D tend vers 0, quasi-neutralite) : un schema explicite exigerait
 * dt < 1/omega_p, l'IMEX la traite en implicite.
 *
 * Split IMEX (cf. integrator/imex.hpp) :
 *   raide (implicite, A-stable) : S(a,b) = (b, -omega_p^2 a),  paire oscillante
 *   lent (explicite)            : T(a,b) = (0, -c_s^2 k^2 a),  correction acoustique
 *
 * L'implicite resout la paire (a,b) ensemble, backward Euler de facteur
 * 1/sqrt(1+omega_p^2 dt^2) < 1 : stable a dt fixe quand omega_p tend vers l'infini (AP).
 */
struct LangmuirMode {
  Real omega_p = 1.0;  ///< frequence plasma (terme raide)
  Real cs2k2 = 0.0;    ///< c_s^2 k^2 (correction acoustique, lente)

  /**
   * Pas explicite (terme acoustique lent) en place : (a,b) <- (a,b) + dt T.
   *
   * @param[in]     a  amplitude du mode
   * @param[in,out] b  vitesse da/dt, mise a jour par T = (0, -cs2k2 a)
   * @param[in]     dt pas de temps
   */
  ADC_HD void explicit_step(Real& a, Real& b, Real dt) const {
    b += dt * (-cs2k2 * a);
  }

  /**
   * Pas implicite (terme plasma raide) en place : resout (a,b) = (a*, b*) + dt S.
   *
   * Avec S = (b, -omega_p^2 a), solve 2x2 lineaire analytique (pas de Newton).
   *
   * @param[in,out] a  amplitude du mode
   * @param[in,out] b  vitesse da/dt
   * @param[in]     dt pas de temps
   */
  ADC_HD void implicit_solve(Real& a, Real& b, Real dt) const {
    const Real as = a, bs = b, w2 = omega_p * omega_p;
    a = (as + dt * bs) / (1 + dt * dt * w2);
    b = bs - dt * w2 * a;
  }

  /// Pulsation propre du mode, omega = sqrt(omega_p^2 + c_s^2 k^2) (Bohm-Gross isotherme).
  Real omega() const { return std::sqrt(omega_p * omega_p + cs2k2); }
};

}  // namespace adc
