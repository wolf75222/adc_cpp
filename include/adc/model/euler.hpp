#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>

#include <cmath>

namespace adc {

/**
 * Euler compressible 2D pour un gaz parfait, instance du concept PhysicalModel.
 *
 * Variables conservatives U = (rho, rho u, rho v, E), avec
 * E = p/(gamma-1) + 1/2 rho (u^2 + v^2) et p = (gamma-1)(E - 1/2 rho |v|^2). Le flux
 * directionnel est F_x = (rho u, rho u^2 + p, rho u v, (E+p) u) et symetriquement en y ;
 * la vitesse d'onde maximale vaut |v_dir| + c avec c = sqrt(gamma p/rho).
 *
 * Le terme source est nul (Euler pur) et elliptic_rhs renvoie la densite de masse rho,
 * second membre de Poisson pour le futur Euler-Poisson auto-gravitant (inutilise en
 * Euler pur). L'argument aux est present pour le concept et recevra grad phi via la
 * source en Euler-Poisson, mais n'entre pas dans le flux d'Euler pur.
 *
 * @note Tout est device-callable (ADC_HD) : StateVec sur tableau C, std::sqrt
 *       (intrinseque device sous nvcc), abs manuel. Compatible kernel GPU comme le
 *       modele diocotron.
 */
struct Euler {
  using State = StateVec<4>;  ///< variables conservatives (rho, rho u, rho v, E)
  using Aux = adc::Aux;       ///< champs auxiliaires (inutilises en Euler pur)
  static constexpr int n_vars = 4;  ///< nombre de variables conservees

  Real gamma = 1.4;  ///< indice adiabatique du gaz parfait

  /// Pression du gaz parfait p = (gamma-1)(E - 1/2 rho |v|^2).
  ADC_HD Real pressure(const State& u) const {
    const Real rho = u[0];
    const Real ke = Real(0.5) * (u[1] * u[1] + u[2] * u[2]) / rho;
    return (gamma - Real(1)) * (u[3] - ke);
  }
  /// Vitesse du son c = sqrt(gamma p / rho).
  ADC_HD Real sound_speed(const State& u) const {
    return std::sqrt(gamma * pressure(u) / u[0]);
  }

  /**
   * Vitesses d'onde signees extremes dans la direction dir : v_dir - c et v_dir + c.
   *
   * Requises par les flux HLL/HLLC, au dela du seul max_wave_speed que demande Rusanov.
   *
   * @param      u    etat conservatif
   * @param      dir  direction de la face (0 = x, 1 = y)
   * @param[out] smin vitesse d'onde la plus a gauche v_dir - c
   * @param[out] smax vitesse d'onde la plus a droite v_dir + c
   */
  ADC_HD void wave_speeds(const State& u, const Aux&, int dir, Real& smin,
                          Real& smax) const {
    const Real vn = (dir == 0 ? u[1] : u[2]) / u[0];
    const Real c = sound_speed(u);
    smin = vn - c;
    smax = vn + c;
  }

  /// Flux convectif compressible dans la direction dir.
  ADC_HD State flux(const State& u, const Aux&, int dir) const {
    const Real rho = u[0];
    const Real vn = (dir == 0 ? u[1] : u[2]) / rho;  // vitesse normale a la face
    const Real p = pressure(u);
    State f{};
    f[0] = rho * vn;
    f[1] = u[1] * vn + (dir == 0 ? p : Real(0));
    f[2] = u[2] * vn + (dir == 1 ? p : Real(0));
    f[3] = (u[3] + p) * vn;
    return f;
  }

  /// Vitesse d'onde maximale |v_dir| + c (estimation Rusanov).
  ADC_HD Real max_wave_speed(const State& u, const Aux&, int dir) const {
    const Real vn = (dir == 0 ? u[1] : u[2]) / u[0];
    const Real a = vn < 0 ? -vn : vn;  // |v_dir| device-safe
    return a + sound_speed(u);
  }

  /// Terme source nul : Euler pur.
  ADC_HD State source(const State&, const Aux&) const { return State{}; }

  /// Second membre de Poisson : densite de masse rho (auto-gravite Euler-Poisson).
  ADC_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

}  // namespace adc
