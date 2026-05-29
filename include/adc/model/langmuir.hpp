#pragma once

#include <adc/core/types.hpp>

#include <cmath>

// Mode de Langmuir linearise : noyau 0D du schema asymptotic-preserving deux-fluides
// (regime raide Euler-Poisson magnetique, papier Hoffart arXiv:2510.11808).
//
// Deux-fluides isotherme, electrons mobiles sur fond ionique fixe, UN mode de
// Fourier k. L'amplitude du mode (a = perturbation, b = da/dt) obeit a
//   a'' + (omega_p^2 + c_s^2 k^2) a = 0,
// oscillation a omega = sqrt(omega_p^2 + c_s^2 k^2) (Bohm-Gross isotherme). La
// frequence plasma omega_p est le terme RAIDE : omega_p -> infini quand la longueur
// de Debye lambda_D -> 0 (quasi-neutralite). Un schema explicite exigerait
// dt < 1/omega_p ; l'IMEX la traite en IMPLICITE.
//
// Split IMEX (cf. integrator/imex.hpp) :
//   - RAIDE (implicite, A-stable) : S(a,b) = (b, -omega_p^2 a)  [paire oscillante]
//   - LENT (explicite)            : T(a,b) = (0, -c_s^2 k^2 a)  [correction acoustique]
// L'implicite resout la paire (a,b) ensemble -> backward Euler de facteur
// 1/sqrt(1+omega_p^2 dt^2) < 1 : stable a dt FIXE quand omega_p -> infini (AP).

namespace adc {

struct LangmuirMode {
  Real omega_p = 1.0;  // frequence plasma (terme raide)
  Real cs2k2 = 0.0;    // c_s^2 k^2 (correction acoustique, lente)

  // partie explicite (lente) en place : (a,b) <- (a,b) + dt T,  T = (0, -cs2k2 a)
  ADC_HD void explicit_step(Real& a, Real& b, Real dt) const {
    b += dt * (-cs2k2 * a);
  }

  // partie implicite (raide) en place : resout (a,b) = (a_connu, b_connu) + dt S,
  // S = (b, -omega_p^2 a). Solve 2x2 lineaire analytique (pas de Newton).
  ADC_HD void implicit_solve(Real& a, Real& b, Real dt) const {
    const Real as = a, bs = b, w2 = omega_p * omega_p;
    a = (as + dt * bs) / (1 + dt * dt * w2);
    b = bs - dt * w2 * a;
  }

  Real omega() const { return std::sqrt(omega_p * omega_p + cs2k2); }
};

}  // namespace adc
