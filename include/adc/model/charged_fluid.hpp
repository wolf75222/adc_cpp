#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/model/euler.hpp>

#include <cmath>

/// @file
/// @brief Fluides charges pour les systemes multi-especes (CoupledSystem + SystemCoupler).
///
/// Chaque espece est un PhysicalModel (loi locale) : transport hydrodynamique et force
/// electrostatique (q/m) rho E, avec E = -grad phi lu dans aux. Le potentiel phi est
/// partage entre especes ; son second membre est une quantite de systeme, somme des
/// charges Sum_s q_s n_s, assemblee par ChargeDensityRhs (coeur). elliptic_rhs ne sert
/// donc ici qu'au cas mono-bloc (densite de charge de l'espece) ; en multi-especes c'est
/// ChargeDensityRhs qui pilote Poisson.
///
/// Deux modeles sont fournis :
///   ChargedEuler            Euler complet (rho, rho u, rho v, E), 4 variables.
///   ChargedEulerIsothermal  Euler isotherme p = cs^2 rho (rho, rho u, rho v), 3 variables.
///
/// Le couple canonique "electrons Euler + ions Euler isothermes + Poisson" se compose
/// alors comme deux EquationBlock heterogenes (4 et 3 variables) d'un meme CoupledSystem.

namespace adc {

/**
 * Euler compressible avec force electrostatique, espece chargee complete.
 *
 * Delegue le flux et la vitesse d'onde a Euler, et ajoute la source (q/m) rho E sur la
 * quantite de mouvement ainsi que le travail correspondant sur l'energie. Le signe est
 * inclus dans qom = q/m (electrons < 0).
 */
struct ChargedEuler {
  using State = StateVec<4>;  ///< variables conservatives (rho, rho u, rho v, E)
  using Aux = adc::Aux;       ///< champs auxiliaires (phi et son gradient)
  static constexpr int n_vars = 4;  ///< nombre de variables conservees

  Euler hydro{};           ///< partie hydrodynamique deleguee
  Real qom = Real(-1);     ///< rapport charge/masse q/m (signe inclus)
  Real charge = Real(-1);  ///< charge q (densite de charge q*rho pour elliptic_rhs mono-bloc)

  /// Flux convectif, delegue a Euler.
  ADC_HD State flux(const State& u, const Aux& a, int dir) const {
    return hydro.flux(u, a, dir);
  }
  /// Vitesse d'onde maximale, deleguee a Euler.
  ADC_HD Real max_wave_speed(const State& u, const Aux& a, int dir) const {
    return hydro.max_wave_speed(u, a, dir);
  }
  /// Pression, deleguee a Euler (debloque HLL/HLLC sur ce modele).
  ADC_HD Real pressure(const State& u) const { return hydro.pressure(u); }
  /// Vitesses d'onde signees (HLL/HLLC), deleguees a Euler.
  ADC_HD void wave_speeds(const State& u, const Aux& a, int dir, Real& smin,
                          Real& smax) const {
    hydro.wave_speeds(u, a, dir, smin, smax);
  }
  /**
   * Source electrostatique (q/m) rho E sur quantite de mouvement et energie.
   *
   * @param u etat conservatif (densite et quantite de mouvement)
   * @param a champs auxiliaires (E = -grad phi)
   * @returns S = (0, (q/m) rho E_x, (q/m) rho E_y, (q/m) (rho v).E)
   */
  ADC_HD State source(const State& u, const Aux& a) const {
    const Real Ex = -a.grad_x, Ey = -a.grad_y;  // E = -grad phi
    State s{};
    s[1] = qom * u[0] * Ex;            // (q/m) rho E_x
    s[2] = qom * u[0] * Ey;
    s[3] = qom * (u[1] * Ex + u[2] * Ey);  // travail de la force : (q/m) (rho v).E
    return s;
  }
  /// Second membre de Poisson mono-bloc : densite de charge q rho.
  ADC_HD Real elliptic_rhs(const State& u) const { return charge * u[0]; }
};

/**
 * Euler isotherme (p = cs^2 rho) avec force electrostatique, espece chargee sans energie.
 *
 * Trois variables (rho, rho u, rho v), pression fermee par cs2 sans equation d'energie.
 * Ajoute la meme source electrostatique (q/m) rho E que ChargedEuler.
 */
struct ChargedEulerIsothermal {
  using State = StateVec<3>;  ///< variables conservatives (rho, rho u, rho v)
  using Aux = adc::Aux;       ///< champs auxiliaires (phi et son gradient)
  static constexpr int n_vars = 3;  ///< nombre de variables conservees

  Real cs2 = Real(1);      ///< vitesse du son au carre (temperature isotherme)
  Real qom = Real(1);      ///< rapport charge/masse q/m
  Real charge = Real(1);   ///< charge q

  /// Flux convectif isotherme (pression p = cs^2 rho).
  ADC_HD State flux(const State& u, const Aux&, int dir) const {
    const Real rho = u[0];
    const Real vn = (dir == 0 ? u[1] : u[2]) / rho;
    const Real p = cs2 * rho;
    State f{};
    f[0] = (dir == 0 ? u[1] : u[2]);
    f[1] = u[1] * vn + (dir == 0 ? p : Real(0));
    f[2] = u[2] * vn + (dir == 1 ? p : Real(0));
    return f;
  }
  /// Vitesse d'onde maximale |v_dir| + c_s (estimation Rusanov).
  ADC_HD Real max_wave_speed(const State& u, const Aux&, int dir) const {
    const Real vn = (dir == 0 ? u[1] : u[2]) / u[0];
    const Real a = vn < 0 ? -vn : vn;
    return a + std::sqrt(cs2);
  }
  /// Vitesses d'onde signees (HLL/HLLC) : isotherme, v_dir -+ c_s.
  ADC_HD void wave_speeds(const State& u, const Aux&, int dir, Real& smin,
                          Real& smax) const {
    const Real vn = (dir == 0 ? u[1] : u[2]) / u[0];
    const Real c = std::sqrt(cs2);
    smin = vn - c;
    smax = vn + c;
  }
  /// Source electrostatique (q/m) rho E sur la quantite de mouvement (pas d'energie).
  ADC_HD State source(const State& u, const Aux& a) const {
    const Real Ex = -a.grad_x, Ey = -a.grad_y;
    State s{};
    s[1] = qom * u[0] * Ex;
    s[2] = qom * u[0] * Ey;
    return s;
  }
  /// Second membre de Poisson mono-bloc : densite de charge q rho.
  ADC_HD Real elliptic_rhs(const State& u) const { return charge * u[0]; }
};

}  // namespace adc
