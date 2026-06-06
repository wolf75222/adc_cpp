#pragma once

// Brique modele advection-diffusion scalaire (AdvectionDiffusion). Utilisee dans les
// tests C++ du coeur (tests/test_weno5_ssprk3.cpp) comme modele de reference pour
// WENO5/SSPRK3, mais non utilisee par adc_cases. Conservee comme brique de validation
// et brique de physique d'exemple (docs/ARCHITECTURE.md).

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>

#include <cmath>

namespace adc {

/**
 * Advection-diffusion scalaire : d_t u + a . grad u = nu Lap u.
 *
 * Brique de TEST/VALIDATION (non utilisee par adc_cases au 2026-06-06) ;
 * conservee comme exemple et comme modele de reference pour tests/test_weno5_ssprk3.cpp.
 *
 * Illustre la remarque "la diffusion, c'est comme un flux de plus" : on garde le
 * PhysicalModel d'advection et on ajoute la seule methode diffusivity(). Le terme
 * parabolique +nu Lap(u) est alors fourni par le coeur (assemble_rhs detecte le trait
 * DiffusiveModel), sans toucher au coupleur. nu = 0 redonne l'advection pure.
 */
struct AdvectionDiffusion {
  using State = StateVec<1>;  ///< une variable conservee : le scalaire u
  using Aux = adc::Aux;       ///< champs auxiliaires (inutilises, non couple)
  static constexpr int n_vars = 1;  ///< nombre de variables conservees

  Real ax = 1.0;   ///< vitesse d'advection en x
  Real ay = 0.0;   ///< vitesse d'advection en y
  Real nu = 0.0;   ///< diffusivite (0 = advection pure)

  /// Flux d'advection F = a u dans la direction dir.
  ADC_HD State flux(const State& u, const Aux&, int dir) const {
    return State{(dir == 0 ? ax : ay) * u[0]};  // F = a u
  }
  /// Vitesse d'onde maximale : module de la vitesse d'advection dans la direction dir.
  ADC_HD Real max_wave_speed(const State&, const Aux&, int dir) const {
    const Real v = (dir == 0) ? ax : ay;
    return v < 0 ? -v : v;
  }
  /// Terme source nul.
  ADC_HD State source(const State&, const Aux&) const { return State{Real(0)}; }
  /// Second membre de Poisson : u (non couple ici, present pour le concept).
  ADC_HD Real elliptic_rhs(const State& u) const { return u[0]; }
  /// Diffusivite nu : active le terme parabolique cote coeur (trait DiffusiveModel).
  ADC_HD Real diffusivity() const { return nu; }
};

}  // namespace adc
