#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>

#include <cmath>

// Modele diocotron : transport scalaire d'une densite electronique n_e par la
// derive E x B, couple a Poisson pour le potentiel.
//
//   d n_e / d t + div(n_e v_E) = 0,   v_E = (E x B) / |B|^2,   E = -grad phi
//   laplacien(phi) = alpha (n_e - n_i0)
//
// Avec B = B0 e_z et E = -grad phi dans le plan, la derive vaut
//   v_E = ( -d phi/d y , d phi/d x ) / B0
// (vitesse a divergence nulle : transport pur, pas de compression).
//
// C'est la cible de validation du solveur (instabilite diocotron). Premier
// modele a satisfaire le concept PhysicalModel. Il exerce le chemin
// "aux -> flux" : le potentiel entre par le flux, pas par la source.

namespace adc {

struct Diocotron {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;

  Real B0 = 1.0;     // champ magnetique hors-plan
  Real n_i0 = 1.0;   // densite ionique de fond (neutralisante)
  Real alpha = 1.0;  // constante de couplage Poisson

  // Composante de la vitesse de derive E x B dans la direction dir.
  ADC_HD Real drift_velocity(const Aux& a, int dir) const {
    return (dir == 0) ? (-a.grad_y / B0) : (a.grad_x / B0);
  }

  ADC_HD State flux(const State& u, const Aux& a, int dir) const {
    State f{};
    f[0] = u[0] * drift_velocity(a, dir);
    return f;
  }

  ADC_HD Real max_wave_speed(const State&, const Aux& a, int dir) const {
    const Real d = drift_velocity(a, dir);
    return d < 0 ? -d : d;  // |.| device-safe (pas de std::fabs)
  }

  ADC_HD State source(const State&, const Aux&) const { return State{}; }

  ADC_HD Real elliptic_rhs(const State& u) const { return alpha * (u[0] - n_i0); }
};

}  // namespace adc
