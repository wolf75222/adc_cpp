#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/model/euler.hpp>

// Euler-Poisson auto-gravitant 2D : Euler compressible couple a la gravite via
// Poisson.
//
//   d_t U + div F(U) = S(U, grad phi),   lap phi = 4 pi G (rho - rho0)
//   g = -grad phi,   S = (0, rho g_x, rho g_y, rho u . g)
//
// La partie hydrodynamique (flux, vitesses d'onde) est DELEGUEE a Euler : seuls la
// SOURCE (force gravitationnelle, via aux = grad phi) et le second membre elliptique
// changent. C'est exactement le chemin "aux entre par la source" du concept
// PhysicalModel (contraste avec le diocotron, "aux entre par le flux").
//
// rho0 : fond neutralisant. En periodique, Poisson exige un second membre a moyenne
// nulle (solvabilite) ; 4 pi G (rho - rho0) l'assure si rho0 = <rho>. Branche tel
// quel sur Coupler<EulerPoisson> (elliptic_rhs -> multigrille -> aux=grad phi ->
// assemble_rhs avec la source). Device-callable (ADC_HD).

namespace adc {

struct EulerPoisson {
  using State = StateVec<4>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 4;

  Euler hydro{};         // partie hydrodynamique (gamma, flux, max_wave_speed)
  Real four_pi_G = 1.0;  // lap phi = four_pi_G (rho - rho0)
  Real rho0 = 1.0;       // fond neutralisant (solvabilite periodique)

  ADC_HD State flux(const State& u, const Aux& a, int dir) const {
    return hydro.flux(u, a, dir);
  }
  ADC_HD Real max_wave_speed(const State& u, const Aux& a, int dir) const {
    return hydro.max_wave_speed(u, a, dir);
  }

  // Force gravitationnelle g = -grad phi (aux = phi, d phi/dx, d phi/dy).
  ADC_HD State source(const State& u, const Aux& a) const {
    const Real gx = -a.grad_x, gy = -a.grad_y;
    State s{};
    s[0] = 0;
    s[1] = u[0] * gx;             // rho g_x
    s[2] = u[0] * gy;             // rho g_y
    s[3] = u[1] * gx + u[2] * gy;  // (rho u) . g  (travail de la gravite)
    return s;
  }

  ADC_HD Real elliptic_rhs(const State& u) const {
    return four_pi_G * (u[0] - rho0);
  }
};

}  // namespace adc
