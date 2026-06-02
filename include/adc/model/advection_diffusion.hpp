#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>

#include <cmath>

// Advection-diffusion scalaire : d_t u + a . grad u = nu Lap u.
//
// Demontre la remarque du tuteur "la diffusion, c'est comme un flux de plus" : on
// garde le PhysicalModel d'advection et on ajoute UNE methode diffusivity(). Le terme
// parabolique +nu Lap(u) est alors fourni par le coeur (assemble_rhs detecte le trait
// DiffusiveModel), sans toucher au coupleur. nu = 0 redonne l'advection pure.

namespace adc {

struct AdvectionDiffusion {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;

  Real ax = 1.0;   // vitesse d'advection en x
  Real ay = 0.0;   // vitesse d'advection en y
  Real nu = 0.0;   // diffusivite (0 = advection pure)

  ADC_HD State flux(const State& u, const Aux&, int dir) const {
    return State{(dir == 0 ? ax : ay) * u[0]};  // F = a u
  }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int dir) const {
    const Real v = (dir == 0) ? ax : ay;
    return v < 0 ? -v : v;
  }
  ADC_HD State source(const State&, const Aux&) const { return State{Real(0)}; }
  ADC_HD Real elliptic_rhs(const State& u) const { return u[0]; }  // non couple ici
  ADC_HD Real diffusivity() const { return nu; }  // active le terme parabolique
};

}  // namespace adc
