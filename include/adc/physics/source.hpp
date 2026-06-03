#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>

/// @file
/// @brief Briques de SOURCE S(U, aux) : terme local, generique sur la taille d'etat (travail sur
///        l'energie si 4 variables). NoSource, PotentialForce ((q/m) rho E), GravityForce (rho g).
///        Composables comme parametre Source d'un CompositeModel (physics/composite.hpp). Les sources
///        INTER-especes (ionisation, collision) vivent au niveau du systeme (operator-split).

namespace adc {

/// Pas de source.
struct NoSource {
  template <class State>
  ADC_HD State apply(const State&, const Aux&) const {
    return State{};
  }
};

/// Force du potentiel (electrostatique) (q/m) rho E sur la quantite de mouvement (+ travail
/// sur l'energie si 4 variables). E = -grad phi.
struct PotentialForce {
  Real qom = 1;  // q/m (signe inclus)
  template <class State>
  ADC_HD State apply(const State& u, const Aux& a) const {
    const Real Ex = -a.grad_x, Ey = -a.grad_y;
    State s{};
    s[1] = qom * u[0] * Ex;
    s[2] = qom * u[0] * Ey;
    if constexpr (State::size() == 4) s[3] = qom * (u[1] * Ex + u[2] * Ey);
    return s;
  }
};

/// Force gravitationnelle rho g (+ travail si 4 variables). g = -grad phi.
struct GravityForce {
  template <class State>
  ADC_HD State apply(const State& u, const Aux& a) const {
    const Real gx = -a.grad_x, gy = -a.grad_y;
    State s{};
    s[1] = u[0] * gx;
    s[2] = u[0] * gy;
    if constexpr (State::size() == 4) s[3] = u[1] * gx + u[2] * gy;
    return s;
  }
};

}  // namespace adc
