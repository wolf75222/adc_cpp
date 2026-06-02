#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>

#include <cmath>

namespace adc {

/**
 * Modele diocotron : transport scalaire d'une densite electronique par la derive E x B.
 *
 * La densite obeit a d n_e / d t + div(n_e v_E) = 0, avec v_E = (E x B) / |B|^2 et
 * E = -grad phi, le potentiel etant fourni par laplacien(phi) = alpha (n_e - n_i0).
 * Avec B = B0 e_z dans le plan, la derive vaut v_E = (-d phi/d y, d phi/d x) / B0 : un
 * champ a divergence nulle, donc transport pur sans compression.
 *
 * Cible de validation du solveur (instabilite diocotron) et premier modele a satisfaire
 * le concept PhysicalModel. Il exerce le chemin "aux vers flux" : le potentiel entre par
 * le flux et non par la source.
 */
struct Diocotron {
  using State = StateVec<1>;  ///< une variable conservee : la densite n_e
  using Aux = adc::Aux;       ///< champs auxiliaires (phi et son gradient)
  static constexpr int n_vars = 1;  ///< nombre de variables conservees

  Real B0 = 1.0;     ///< champ magnetique hors-plan
  Real n_i0 = 1.0;   ///< densite ionique de fond (neutralisante)
  Real alpha = 1.0;  ///< constante de couplage Poisson

  /**
   * Composante de la vitesse de derive E x B dans la direction dir.
   *
   * @param a   champs auxiliaires (gradient du potentiel)
   * @param dir direction de la face (0 = x, 1 = y)
   * @returns composante dir de v_E = (-d phi/d y, d phi/d x) / B0
   */
  ADC_HD Real drift_velocity(const Aux& a, int dir) const {
    return (dir == 0) ? (-a.grad_y / B0) : (a.grad_x / B0);
  }

  /// Flux d'advection n_e v_E dans la direction dir.
  ADC_HD State flux(const State& u, const Aux& a, int dir) const {
    State f{};
    f[0] = u[0] * drift_velocity(a, dir);
    return f;
  }

  /// Vitesse d'onde maximale : module de la vitesse de derive dans la direction dir.
  ADC_HD Real max_wave_speed(const State&, const Aux& a, int dir) const {
    const Real d = drift_velocity(a, dir);
    return d < 0 ? -d : d;  // |.| device-safe (pas de std::fabs)
  }

  /// Terme source nul : transport pur.
  ADC_HD State source(const State&, const Aux&) const { return State{}; }

  /// Second membre de Poisson : ecart de densite a la charge de fond, alpha (n_e - n_i0).
  ADC_HD Real elliptic_rhs(const State& u) const { return alpha * (u[0] - n_i0); }
};

}  // namespace adc
