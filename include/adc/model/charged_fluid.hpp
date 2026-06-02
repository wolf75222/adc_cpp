#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/model/euler.hpp>

#include <cmath>

// Fluides CHARGES pour les systemes multi-especes (CoupledSystem + SystemCoupler).
//
// Chaque espece est un PhysicalModel (loi locale) : transport hydrodynamique + force
// electrostatique (q/m) rho E, avec E = -grad phi lu dans aux. Le potentiel phi est
// PARTAGE entre especes ; son second membre est une quantite de SYSTEME, somme des
// charges Sum_s q_s n_s, assemblee par ChargeDensityRhs (coeur). elliptic_rhs ici ne
// sert donc qu'au cas MONO-bloc (densite de charge de l'espece) ; en multi-especes
// c'est ChargeDensityRhs qui pilote Poisson.
//
//   ChargedEuler            : Euler complet (rho, rho u, rho v, E), 4 variables.
//   ChargedEulerIsothermal  : Euler isotherme p = cs^2 rho (rho, rho u, rho v), 3 variables.
//
// Le couple canonique "electrons Euler + ions Euler isothermes + Poisson" se compose
// alors comme deux EquationBlock heterogenes (4 et 3 variables) d'un meme CoupledSystem.

namespace adc {

// Euler compressible + force electrostatique. qom = q/m (signe inclus : electrons < 0).
// Delegue le flux et la vitesse d'onde a Euler ; ajoute la source (q/m) rho E sur la
// quantite de mouvement et le travail correspondant sur l'energie.
struct ChargedEuler {
  using State = StateVec<4>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 4;

  Euler hydro{};
  Real qom = Real(-1);     // q/m
  Real charge = Real(-1);  // q (densite de charge q*rho pour elliptic_rhs mono-bloc)

  ADC_HD State flux(const State& u, const Aux& a, int dir) const {
    return hydro.flux(u, a, dir);
  }
  ADC_HD Real max_wave_speed(const State& u, const Aux& a, int dir) const {
    return hydro.max_wave_speed(u, a, dir);
  }
  // Delegues a Euler -> debloque HLL/HLLC (vitesses signees + pression) sur ce modele.
  ADC_HD Real pressure(const State& u) const { return hydro.pressure(u); }
  ADC_HD void wave_speeds(const State& u, const Aux& a, int dir, Real& smin,
                          Real& smax) const {
    hydro.wave_speeds(u, a, dir, smin, smax);
  }
  ADC_HD State source(const State& u, const Aux& a) const {
    const Real Ex = -a.grad_x, Ey = -a.grad_y;  // E = -grad phi
    State s{};
    s[1] = qom * u[0] * Ex;            // (q/m) rho E_x
    s[2] = qom * u[0] * Ey;
    s[3] = qom * (u[1] * Ex + u[2] * Ey);  // travail de la force : (q/m) (rho v).E
    return s;
  }
  ADC_HD Real elliptic_rhs(const State& u) const { return charge * u[0]; }
};

// Euler ISOTHERME (p = cs^2 rho) + force electrostatique. 3 variables (pas d'energie).
struct ChargedEulerIsothermal {
  using State = StateVec<3>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 3;

  Real cs2 = Real(1);      // vitesse du son au carre (temperature isotherme)
  Real qom = Real(1);      // q/m
  Real charge = Real(1);   // q

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
  ADC_HD Real max_wave_speed(const State& u, const Aux&, int dir) const {
    const Real vn = (dir == 0 ? u[1] : u[2]) / u[0];
    const Real a = vn < 0 ? -vn : vn;
    return a + std::sqrt(cs2);
  }
  // Vitesses d'onde signees (requises par HLL/HLLC) : isotherme -> v_dir ∓ c_s.
  ADC_HD void wave_speeds(const State& u, const Aux&, int dir, Real& smin,
                          Real& smax) const {
    const Real vn = (dir == 0 ? u[1] : u[2]) / u[0];
    const Real c = std::sqrt(cs2);
    smin = vn - c;
    smax = vn + c;
  }
  ADC_HD State source(const State& u, const Aux& a) const {
    const Real Ex = -a.grad_x, Ey = -a.grad_y;
    State s{};
    s[1] = qom * u[0] * Ex;
    s[2] = qom * u[0] * Ey;
    return s;
  }
  ADC_HD Real elliptic_rhs(const State& u) const { return charge * u[0]; }
};

}  // namespace adc
