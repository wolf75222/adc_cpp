#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/core/variables.hpp>

#include <cmath>

namespace adc {

/**
 * Euler compressible 2D pour un gaz parfait : brique HYPERBOLIQUE (concept HyperbolicModel).
 *
 * Variables conservatives U = (rho, rho u, rho v, E), avec
 * E = p/(gamma-1) + 1/2 rho (u^2 + v^2) et p = (gamma-1)(E - 1/2 rho |v|^2). Le flux
 * directionnel est F_x = (rho u, rho u^2 + p, rho u v, (E+p) u) et symetriquement en y ;
 * la vitesse d'onde maximale vaut |v_dir| + c avec c = sqrt(gamma p/rho).
 *
 * Brique HYPERBOLIQUE pure : variables (cons U, prim P) + conversions + flux + vitesses d'onde.
 * AUCUNE source ni second membre elliptique ici : ce sont des briques SEPAREES, assemblees par
 * CompositeModel. L'argument aux est present pour le contrat (un transport a derive y lit grad
 * phi) mais n'entre pas dans le flux d'Euler.
 *
 * @note Tout est device-callable (ADC_HD) : StateVec sur tableau C, std::sqrt
 *       (intrinseque device sous nvcc), abs manuel. Compatible kernel GPU comme le
 *       modele de transport scalaire.
 */
struct Euler {
  using State = StateVec<4>;  ///< variables conservatives (rho, rho u, rho v, E)
  using Prim = StateVec<4>;   ///< variables primitives (rho, u, v, p)
  using Aux = adc::Aux;       ///< champs auxiliaires (inutilises en Euler pur)
  static constexpr int n_vars = 4;  ///< nombre de variables conservees

  Real gamma = 1.4;  ///< indice adiabatique du gaz parfait

  /// Pression du gaz parfait p = (gamma-1)(E - 1/2 rho |v|^2).
  ADC_HD Real pressure(const State& u) const {
    const Real rho = u[0];
    const Real ke = Real(0.5) * (u[1] * u[1] + u[2] * u[2]) / rho;
    return (gamma - Real(1)) * (u[3] - ke);
  }
  /// Vitesse du son c = sqrt(gamma p / rho).
  ADC_HD Real sound_speed(const State& u) const {
    return std::sqrt(gamma * pressure(u) / u[0]);
  }

  /// Conservatif -> primitif : (rho, rho u, rho v, E) -> (rho, u, v, p).
  ADC_HD Prim to_primitive(const State& u) const {
    const Real rho = u[0];
    Prim p{};
    p[0] = rho;
    p[1] = u[1] / rho;
    p[2] = u[2] / rho;
    p[3] = pressure(u);
    return p;
  }
  /// Primitif -> conservatif : (rho, u, v, p) -> (rho, rho u, rho v, E).
  ADC_HD State to_conservative(const Prim& p) const {
    const Real rho = p[0];
    State u{};
    u[0] = rho;
    u[1] = rho * p[1];
    u[2] = rho * p[2];
    u[3] = p[3] / (gamma - Real(1)) + Real(0.5) * rho * (p[1] * p[1] + p[2] * p[2]);
    return u;
  }

  /**
   * Vitesses d'onde signees extremes dans la direction dir : v_dir - c et v_dir + c.
   *
   * Requises par les flux HLL/HLLC, au dela du seul max_wave_speed que demande Rusanov.
   *
   * @param      u    etat conservatif
   * @param      dir  direction de la face (0 = x, 1 = y)
   * @param[out] smin vitesse d'onde la plus a gauche v_dir - c
   * @param[out] smax vitesse d'onde la plus a droite v_dir + c
   */
  ADC_HD void wave_speeds(const State& u, const Aux&, int dir, Real& smin,
                          Real& smax) const {
    const Prim p = to_primitive(u);
    const Real vn = (dir == 0 ? p[1] : p[2]);
    const Real c = std::sqrt(gamma * p[3] / p[0]);
    smin = vn - c;
    smax = vn + c;
  }

  /// Flux convectif compressible dans la direction dir.
  ADC_HD State flux(const State& u, const Aux&, int dir) const {
    const Real rho = u[0];
    const Real vn = (dir == 0 ? u[1] : u[2]) / rho;  // vitesse normale a la face
    const Real p = pressure(u);
    State f{};
    f[0] = rho * vn;
    f[1] = u[1] * vn + (dir == 0 ? p : Real(0));
    f[2] = u[2] * vn + (dir == 1 ? p : Real(0));
    f[3] = (u[3] + p) * vn;
    return f;
  }

  /// Spectre complet dans la direction dir : (v_dir - c, v_dir, v_dir, v_dir + c). Pendant vecteur
  /// de wave_speeds (qui ne donne que les extremes signes) ; utile aux schemas a spectre (Roe).
  ADC_HD State eigenvalues(const State& u, const Aux&, int dir) const {
    const Prim p = to_primitive(u);
    const Real vn = (dir == 0 ? p[1] : p[2]);
    const Real c = std::sqrt(gamma * p[3] / p[0]);
    State e{};
    e[0] = vn - c;
    e[1] = vn;
    e[2] = vn;
    e[3] = vn + c;
    return e;
  }

  /// Vitesse d'onde maximale |v_dir| + c (estimation Rusanov), calculee en primitif.
  ADC_HD Real max_wave_speed(const State& u, const Aux&, int dir) const {
    const Prim p = to_primitive(u);
    const Real vn = (dir == 0 ? p[1] : p[2]);
    const Real a = vn < 0 ? -vn : vn;  // |v_dir| device-safe
    return a + std::sqrt(gamma * p[3] / p[0]);
  }

  /// Descripteur des variables (contrat du modele hyperbolique ; metadonnee hote d'introspection).
  static VariableSet conservative_vars() {
    return {VariableKind::Conservative, {"rho", "rho_u", "rho_v", "E"}, 4,
            {VariableRole::Density, VariableRole::MomentumX, VariableRole::MomentumY,
             VariableRole::Energy}};
  }
  static VariableSet primitive_vars() {
    return {VariableKind::Primitive, {"rho", "u", "v", "p"}, 4,
            {VariableRole::Density, VariableRole::VelocityX, VariableRole::VelocityY,
             VariableRole::Pressure}};
  }
};

}  // namespace adc
