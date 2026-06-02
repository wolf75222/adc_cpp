#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/model/euler.hpp>

namespace adc {

/**
 * Euler-Poisson 2D : Euler compressible couple a un champ de potentiel via Poisson.
 *
 * Selon le signe du couplage, le modele decrit soit l'auto-gravite (attractif,
 * astrophysique), soit l'electrostatique mono-espece (repulsif, plasma : oscillation de
 * Langmuir et explosion de Coulomb). Memes equations, signe de la source elliptique
 * oppose :
 *
 *     d_t U + div F(U) = S(U, grad phi),   lap phi = s * 4 pi G (rho - rho0)
 *     g = -grad phi,   S = (0, rho g_x, rho g_y, rho u . g),   s = +-1
 *
 * La partie hydrodynamique (flux, vitesses d'onde) est deleguee a Euler : seuls la
 * source (force gravitationnelle, via aux = grad phi) et le second membre elliptique
 * changent. C'est exactement le chemin "aux entre par la source" du concept
 * PhysicalModel, par contraste avec le diocotron ou "aux entre par le flux".
 *
 * @note rho0 est le fond neutralisant. En periodique, Poisson exige un second membre a
 *       moyenne nulle (solvabilite), assuree par 4 pi G (rho - rho0) si rho0 = <rho>.
 *       Se branche tel quel sur Coupler<EulerPoisson> (elliptic_rhs vers multigrille
 *       vers aux=grad phi vers assemble_rhs avec la source). Device-callable (ADC_HD).
 */
struct EulerPoisson {
  using State = StateVec<4>;  ///< variables conservatives (rho, rho u, rho v, E)
  using Aux = adc::Aux;       ///< champs auxiliaires (phi et son gradient)
  static constexpr int n_vars = 4;  ///< nombre de variables conservees

  Euler hydro{};         ///< partie hydrodynamique (gamma, flux, max_wave_speed)
  Real four_pi_G = 1.0;  ///< intensite du couplage (4 pi G gravite, 4 pi k_e plasma)
  Real rho0 = 1.0;       ///< fond neutralisant (solvabilite periodique)
  /// signe du couplage : +1 = attractif (auto-gravite, effondrement de Jeans), -1 =
  /// repulsif (electrostatique mono-espece, oscillation de Langmuir et explosion de
  /// Coulomb). Retourner le signe de la source elliptique retourne phi, donc la force
  /// g = -grad phi : une seule ligne separe gravite et plasma.
  Real coupling_sign = 1;

  /// Flux convectif, delegue a la partie hydrodynamique Euler.
  ADC_HD State flux(const State& u, const Aux& a, int dir) const {
    return hydro.flux(u, a, dir);
  }
  /// Vitesse d'onde maximale, deleguee a Euler.
  ADC_HD Real max_wave_speed(const State& u, const Aux& a, int dir) const {
    return hydro.max_wave_speed(u, a, dir);
  }
  /// Pression, deleguee a Euler.
  ADC_HD Real pressure(const State& u) const { return hydro.pressure(u); }
  /// Vitesses d'onde signees (HLL/HLLC), deleguees a Euler.
  ADC_HD void wave_speeds(const State& u, const Aux& a, int dir, Real& smin,
                          Real& smax) const {
    hydro.wave_speeds(u, a, dir, smin, smax);
  }

  /**
   * Source de la force gravitationnelle g = -grad phi sur quantite de mouvement et
   * energie.
   *
   * @param u etat conservatif (densite et quantite de mouvement)
   * @param a champs auxiliaires (aux = phi, d phi/dx, d phi/dy)
   * @returns S = (0, rho g_x, rho g_y, rho u . g)
   */
  ADC_HD State source(const State& u, const Aux& a) const {
    const Real gx = -a.grad_x, gy = -a.grad_y;
    State s{};
    s[0] = 0;
    s[1] = u[0] * gx;             // rho g_x
    s[2] = u[0] * gy;             // rho g_y
    s[3] = u[1] * gx + u[2] * gy;  // (rho u) . g  (travail de la gravite)
    return s;
  }

  /// Second membre de Poisson : ecart de densite signe, s 4 pi G (rho - rho0).
  ADC_HD Real elliptic_rhs(const State& u) const {
    return coupling_sign * four_pi_G * (u[0] - rho0);
  }
};

}  // namespace adc
