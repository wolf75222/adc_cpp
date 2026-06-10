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
/// Lit uniquement la densite. Signe de q inclus (ion : q=+1, electron : q=-1).
///
/// ROLE-AWARE (audit §5) : c_rho membre, defaut = layout canonique (densite en composante 0), resolu
/// par model_factory via TR::conservative_vars().index_of(Density). Canonique == defaut pour tout
/// transport natif -> STRICTEMENT bit-identique. int POD -> rhs reste device-clean.
struct ChargeDensity {
  Real q = 1;
  int c_rho = 0;  // defaut = densite en composante 0 (bit-identique)
  template <class State>
  ADC_HD Real rhs(const State& u) const {
    return q * u[c_rho];
  }
};

/// Fond neutralisant f = alpha (n - n0). Modelise un fond de neutralisation fixe de densite n0.
///
/// CONTRAT : brique ELLIPTIQUE ponctuelle, device-callable (ADC_HD), aucun etat global.
/// Lit uniquement la densite. alpha = charge effective du fond ; n0 = densite du fond neutre.
///
/// ROLE-AWARE (audit §5) : c_rho membre, defaut = composante 0, resolu par model_factory via le role
/// Density du transport. Canonique == defaut -> bit-identique. Voir ChargeDensity.
struct BackgroundDensity {
  Real alpha = 1, n0 = 0;
  int c_rho = 0;  // defaut = densite en composante 0 (bit-identique)
  template <class State>
  ADC_HD Real rhs(const State& u) const {
    return alpha * (u[c_rho] - n0);
  }
};

/// Couplage self-consistant f = sign * 4piG * (rho - rho0).
///
/// CONTRAT : brique ELLIPTIQUE ponctuelle, device-callable (ADC_HD), aucun etat global.
/// sign = +1 gravite (Poisson standard), sign = -1 plasma (signe de Gauss). rho0 = fond.
///
/// ROLE-AWARE (audit §5) : c_rho membre, defaut = composante 0, resolu par model_factory via le role
/// Density du transport. Canonique == defaut -> bit-identique. Voir ChargeDensity.
struct GravityCoupling {
  Real sign = 1, four_pi_G = 1, rho0 = 1;
  int c_rho = 0;  // defaut = densite en composante 0 (bit-identique)
  template <class State>
  ADC_HD Real rhs(const State& u) const {
    return sign * four_pi_G * (u[c_rho] - rho0);
  }
};

}  // namespace adc
