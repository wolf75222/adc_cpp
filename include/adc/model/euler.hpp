#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>

#include <cmath>

// Euler compressible 2D (gaz parfait), instance du concept PhysicalModel.
//
// Variables conservatives U = (rho, rho u, rho v, E), avec
//   E = p/(gamma-1) + 1/2 rho (u^2 + v^2),   p = (gamma-1)(E - 1/2 rho |v|^2).
// Flux directionnel :
//   F_x = (rho u, rho u^2 + p, rho u v, (E+p) u)
//   F_y = (rho v, rho u v, rho v^2 + p, (E+p) v)
// Vitesse d'onde maximale dans la direction dir : |v_dir| + c, c = sqrt(gamma p/rho).
//
// source = 0 (Euler pur). elliptic_rhs = rho : densite de masse, second membre de
// Poisson pour le futur Euler-Poisson auto-gravitant ; inutilise en Euler pur. aux
// est present pour le concept (et recevra grad phi via la source en Euler-Poisson),
// mais n'entre pas dans le flux d'Euler pur.
//
// Tout est device-callable (ADC_HD) : StateVec sur tableau C, std::sqrt (intrinseque
// device sous nvcc), abs manuel. Compatible kernel GPU comme le modele diocotron.

namespace adc {

struct Euler {
  using State = StateVec<4>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 4;

  Real gamma = 1.4;

  ADC_HD Real pressure(const State& u) const {
    const Real rho = u[0];
    const Real ke = Real(0.5) * (u[1] * u[1] + u[2] * u[2]) / rho;
    return (gamma - Real(1)) * (u[3] - ke);
  }
  ADC_HD Real sound_speed(const State& u) const {
    return std::sqrt(gamma * pressure(u) / u[0]);
  }

  // Vitesses d'onde SIGNEES extremes dans la direction dir : v_dir - c et v_dir + c.
  // Requises par les flux HLL/HLLC (au dela de max_wave_speed que demande Rusanov).
  ADC_HD void wave_speeds(const State& u, const Aux&, int dir, Real& smin,
                          Real& smax) const {
    const Real vn = (dir == 0 ? u[1] : u[2]) / u[0];
    const Real c = sound_speed(u);
    smin = vn - c;
    smax = vn + c;
  }

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

  ADC_HD Real max_wave_speed(const State& u, const Aux&, int dir) const {
    const Real vn = (dir == 0 ? u[1] : u[2]) / u[0];
    const Real a = vn < 0 ? -vn : vn;  // |v_dir| device-safe
    return a + sound_speed(u);
  }

  ADC_HD State source(const State&, const Aux&) const { return State{}; }

  ADC_HD Real elliptic_rhs(const State& u) const { return u[0]; }  // densite de masse
};

}  // namespace adc
