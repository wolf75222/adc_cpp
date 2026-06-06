#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>

/// @file
/// @brief Briques de SECOND MEMBRE elliptique f(U) : la contribution d'un bloc au membre de droite
///        de l'equation elliptique de systeme (Poisson). ChargeDensity (q n), BackgroundDensity
///        (alpha (n - n0)), GravityCoupling (s 4piG (rho - rho0)). Composables comme parametre
///        Elliptic d'un CompositeModel (physics/composite.hpp). L'OPERATEUR elliptique (div eps grad)
///        et la resolution vivent cote systeme (runtime) ; ici seulement le second membre par bloc.

namespace adc {

/// Densite de charge f = q n. Second membre elliptique du bloc ions ou electrons.
///
/// CONTRAT : brique ELLIPTIQUE ponctuelle, device-callable (ADC_HD), aucun etat global.
/// Lit uniquement u[0] (densite). Signe de q inclus (ion : q=+1, electron : q=-1).
struct ChargeDensity {
  Real q = 1;
  template <class State>
  ADC_HD Real rhs(const State& u) const {
    return q * u[0];
  }
};

/// Fond neutralisant f = alpha (n - n0). Modelise un fond de neutralisation fixe de densite n0.
///
/// CONTRAT : brique ELLIPTIQUE ponctuelle, device-callable (ADC_HD), aucun etat global.
/// Lit uniquement u[0]. alpha = charge effective du fond ; n0 = densite du fond neutre.
struct BackgroundDensity {
  Real alpha = 1, n0 = 0;
  template <class State>
  ADC_HD Real rhs(const State& u) const {
    return alpha * (u[0] - n0);
  }
};

/// Couplage self-consistant f = sign * 4piG * (rho - rho0).
///
/// CONTRAT : brique ELLIPTIQUE ponctuelle, device-callable (ADC_HD), aucun etat global.
/// sign = +1 gravite (Poisson standard), sign = -1 plasma (signe de Gauss). rho0 = fond.
struct GravityCoupling {
  Real sign = 1, four_pi_G = 1, rho0 = 1;
  template <class State>
  ADC_HD Real rhs(const State& u) const {
    return sign * four_pi_G * (u[0] - rho0);
  }
};

}  // namespace adc
