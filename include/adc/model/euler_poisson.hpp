#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/model/euler.hpp>

// Euler-Poisson 2D : Euler compressible couple a un champ de potentiel via Poisson.
// Selon le signe du couplage, c'est de l'AUTO-GRAVITE (attractif, astrophysique) ou de
// l'ELECTROSTATIQUE mono-espece (repulsif, plasma : oscillation de Langmuir + explosion
// de Coulomb). Memes equations, signe de la source elliptique opposee.
//
//   d_t U + div F(U) = S(U, grad phi),   lap phi = s * 4 pi G (rho - rho0)
//   g = -grad phi,   S = (0, rho g_x, rho g_y, rho u . g),   s = +-1
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
  Real four_pi_G = 1.0;  // intensite du couplage (4 pi G gravite, 4 pi k_e plasma)
  Real rho0 = 1.0;       // fond neutralisant (solvabilite periodique)
  // signe du couplage : +1 = ATTRACTIF (auto-gravite -> effondrement de Jeans),
  // -1 = REPULSIF (electrostatique mono-espece -> oscillation de Langmuir + explosion
  // de Coulomb). Retourner le signe de la source elliptique retourne phi, donc la force
  // g = -grad phi : une seule ligne separe gravite et plasma.
  Real coupling_sign = 1;

  ADC_HD State flux(const State& u, const Aux& a, int dir) const {
    return hydro.flux(u, a, dir);
  }
  ADC_HD Real max_wave_speed(const State& u, const Aux& a, int dir) const {
    return hydro.max_wave_speed(u, a, dir);
  }
  ADC_HD Real pressure(const State& u) const { return hydro.pressure(u); }
  ADC_HD void wave_speeds(const State& u, const Aux& a, int dir, Real& smin,
                          Real& smax) const {
    hydro.wave_speeds(u, a, dir, smin, smax);
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
    return coupling_sign * four_pi_G * (u[0] - rho0);
  }
};

}  // namespace adc
