#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/model/euler.hpp>  // Euler : reutilise comme brique de transport CompressibleFlux

#include <cmath>

/// @file
/// @brief Briques physiques GENERIQUES composables, et le CompositeModel qui les assemble.
///
/// Le coeur ne connait AUCUN scenario nomme. Il fournit des
/// briques mathematiques reutilisables (transport, source, second membre elliptique), et un
/// `CompositeModel<Transport, Source, Elliptic>` qui les combine en un PhysicalModel compile.
/// Un scenario nomme est une COMPOSITION de briques, choisie depuis l'application (adc_cases).
///
/// Transport (hyperbolique) : ExBVelocity (1 var), CompressibleFlux (4 var, = Euler),
/// IsothermalFlux (3 var). Source : NoSource, PotentialForce ((q/m) rho E), GravityForce
/// (rho g). Elliptique : ChargeDensity (q n), BackgroundDensity (alpha (n - n0)),
/// GravityCoupling (s 4piG (rho - rho0)).

namespace adc {

// ---------------------------------------------------------------------------
// Briques de TRANSPORT (flux hyperbolique). Chacune : n_vars, flux, max_wave_speed
// (+ pressure / wave_speeds quand le flux HLLC s'applique).
// ---------------------------------------------------------------------------

/// Advection scalaire par la derive E x B : v = (-d_y phi, d_x phi)/B0 (a divergence nulle).
struct ExBVelocity {
  static constexpr int n_vars = 1;
  Real B0 = 1;
  ADC_HD Real velocity(const Aux& a, int dir) const {
    return (dir == 0) ? (-a.grad_y / B0) : (a.grad_x / B0);
  }
  ADC_HD StateVec<1> flux(const StateVec<1>& u, const Aux& a, int dir) const {
    StateVec<1> f{};
    f[0] = u[0] * velocity(a, dir);
    return f;
  }
  ADC_HD Real max_wave_speed(const StateVec<1>&, const Aux& a, int dir) const {
    const Real d = velocity(a, dir);
    return d < 0 ? -d : d;
  }
};

/// Flux d'Euler compressible 2D (reutilise Euler : gamma, pression, vitesses d'onde signees).
using CompressibleFlux = Euler;

/// Flux d'Euler ISOTHERME (p = cs^2 rho), 3 variables (rho, rho u, rho v).
struct IsothermalFlux {
  static constexpr int n_vars = 3;
  Real cs2 = 1;
  ADC_HD StateVec<3> flux(const StateVec<3>& u, const Aux&, int dir) const {
    const Real rho = u[0];
    const Real vn = (dir == 0 ? u[1] : u[2]) / rho;
    const Real p = cs2 * rho;
    StateVec<3> f{};
    f[0] = (dir == 0 ? u[1] : u[2]);
    f[1] = u[1] * vn + (dir == 0 ? p : Real(0));
    f[2] = u[2] * vn + (dir == 1 ? p : Real(0));
    return f;
  }
  ADC_HD Real max_wave_speed(const StateVec<3>& u, const Aux&, int dir) const {
    const Real vn = (dir == 0 ? u[1] : u[2]) / u[0];
    const Real a = vn < 0 ? -vn : vn;
    return a + std::sqrt(cs2);
  }
  /// Vitesses signees (HLL/HLLC) : v_dir -+ c_s.
  ADC_HD void wave_speeds(const StateVec<3>& u, const Aux&, int dir, Real& smin,
                          Real& smax) const {
    const Real vn = (dir == 0 ? u[1] : u[2]) / u[0];
    const Real c = std::sqrt(cs2);
    smin = vn - c;
    smax = vn + c;
  }
};

// ---------------------------------------------------------------------------
// Briques de SOURCE S(U, aux). Generique sur la taille d'etat (energie si 4 var).
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Briques de SECOND MEMBRE elliptique f(U) (densite de charge / fond / gravite).
// ---------------------------------------------------------------------------

/// Densite de charge f = q n.
struct ChargeDensity {
  Real q = 1;
  template <class State>
  ADC_HD Real rhs(const State& u) const {
    return q * u[0];
  }
};

/// Fond neutralisant f = alpha (n - n0).
struct BackgroundDensity {
  Real alpha = 1, n0 = 0;
  template <class State>
  ADC_HD Real rhs(const State& u) const {
    return alpha * (u[0] - n0);
  }
};

/// Couplage self-consistant f = sign 4piG (rho - rho0) (sign = +1 gravite, -1 plasma).
struct GravityCoupling {
  Real sign = 1, four_pi_G = 1, rho0 = 1;
  template <class State>
  ADC_HD Real rhs(const State& u) const {
    return sign * four_pi_G * (u[0] - rho0);
  }
};

// ---------------------------------------------------------------------------
// Composition : assemble (transport, source, elliptic) en un PhysicalModel compile.
// ---------------------------------------------------------------------------

/// Modele physique compose de trois briques. Satisfait le concept PhysicalModel ; expose
/// pressure / wave_speeds quand la brique de transport les fournit (necessaire au flux HLLC).
template <class Transport, class Source, class Elliptic>
struct CompositeModel {
  using State = StateVec<Transport::n_vars>;
  using Aux = adc::Aux;
  static constexpr int n_vars = Transport::n_vars;

  Transport tr{};
  Source src{};
  Elliptic ell{};

  ADC_HD State flux(const State& u, const Aux& a, int dir) const { return tr.flux(u, a, dir); }
  ADC_HD Real max_wave_speed(const State& u, const Aux& a, int dir) const {
    return tr.max_wave_speed(u, a, dir);
  }
  ADC_HD State source(const State& u, const Aux& a) const { return src.apply(u, a); }
  ADC_HD Real elliptic_rhs(const State& u) const { return ell.rhs(u); }

  ADC_HD Real pressure(const State& u) const
    requires requires(const Transport t, const State s) { t.pressure(s); }
  {
    return tr.pressure(u);
  }
  ADC_HD void wave_speeds(const State& u, const Aux& a, int dir, Real& smin, Real& smax) const
    requires requires(const Transport t, const State s, const Aux aa, int d, Real& lo,
                      Real& hi) { t.wave_speeds(s, aa, d, lo, hi); }
  {
    tr.wave_speeds(u, a, dir, smin, smax);
  }
};

}  // namespace adc
