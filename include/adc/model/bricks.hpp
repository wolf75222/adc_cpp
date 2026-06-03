#pragma once

#include <adc/core/physical_model.hpp>  // HyperbolicModel : contrat de la brique hyperbolique
#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/core/variables.hpp>
#include <adc/model/euler.hpp>  // Euler : reutilise comme brique hyperbolique CompressibleFlux

#include <cmath>

/// @file
/// @brief Briques physiques GENERIQUES composables, et le CompositeModel qui les assemble.
///
/// Le coeur ne connait AUCUN scenario nomme. Il fournit des
/// briques mathematiques reutilisables (hyperbolique, source, second membre elliptique), et un
/// `CompositeModel<Hyperbolic, Source, Elliptic>` qui les combine en un PhysicalModel compile.
/// Un scenario nomme est une COMPOSITION de briques, choisie depuis l'application (adc_cases).
///
/// Hyperbolique (Vars + flux + vitesses d'onde + conversions) : ExBVelocity (1 var),
/// CompressibleFlux (4 var, = Euler),
/// IsothermalFlux (3 var). Source : NoSource, PotentialForce ((q/m) rho E), GravityForce
/// (rho g). Elliptique : ChargeDensity (q n), BackgroundDensity (alpha (n - n0)),
/// GravityCoupling (s 4piG (rho - rho0)).

namespace adc {

// ---------------------------------------------------------------------------
// Briques HYPERBOLIQUES (Vars + flux + vitesses d'onde + conversions cons<->prim). Chacune :
// State, Prim, n_vars, flux, max_wave_speed, to_primitive/to_conservative, conservative_vars/
// primitive_vars (+ pressure/wave_speeds si flux HLLC). Cf. le concept HyperbolicModel.
// ---------------------------------------------------------------------------

/// Advection scalaire par la derive E x B : v = (-d_y phi, d_x phi)/B0 (a divergence nulle).
struct ExBVelocity {
  static constexpr int n_vars = 1;
  using State = StateVec<1>;
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
  /// Spectre : une onde, la vitesse de derive dans la direction dir.
  ADC_HD StateVec<1> eigenvalues(const StateVec<1>&, const Aux& a, int dir) const {
    StateVec<1> e{};
    e[0] = velocity(a, dir);
    return e;
  }
  // Scalaire : variables primitives = conservatives (densite transportee).
  using Prim = StateVec<1>;
  ADC_HD Prim to_primitive(const StateVec<1>& u) const { return u; }
  ADC_HD StateVec<1> to_conservative(const Prim& p) const { return p; }
  static Variables conservative_vars() {
    return {VariableKind::Conservative, {"n"}, 1, {VariableRole::Density}};
  }
  static Variables primitive_vars() {
    return {VariableKind::Primitive, {"n"}, 1, {VariableRole::Density}};
  }
};

/// Flux d'Euler compressible 2D (reutilise Euler : gamma, pression, vitesses d'onde signees).
using CompressibleFlux = Euler;

/// Flux d'Euler ISOTHERME (p = cs^2 rho), 3 variables (rho, rho u, rho v).
struct IsothermalFlux {
  static constexpr int n_vars = 3;
  using State = StateVec<3>;  ///< variables conservatives (rho, rho u, rho v)
  using Prim = StateVec<3>;   ///< variables primitives (rho, u, v)
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
  /// Conservatif -> primitif : (rho, rho u, rho v) -> (rho, u, v).
  ADC_HD Prim to_primitive(const StateVec<3>& u) const {
    Prim p{};
    p[0] = u[0];
    p[1] = u[1] / u[0];
    p[2] = u[2] / u[0];
    return p;
  }
  /// Primitif -> conservatif : (rho, u, v) -> (rho, rho u, rho v).
  ADC_HD StateVec<3> to_conservative(const Prim& p) const {
    StateVec<3> u{};
    u[0] = p[0];
    u[1] = p[0] * p[1];
    u[2] = p[0] * p[2];
    return u;
  }
  ADC_HD Real max_wave_speed(const StateVec<3>& u, const Aux&, int dir) const {
    const Prim p = to_primitive(u);
    const Real vn = (dir == 0 ? p[1] : p[2]);
    const Real a = vn < 0 ? -vn : vn;
    return a + std::sqrt(cs2);
  }
  /// Spectre complet : (v_dir - c, v_dir, v_dir + c), c = sqrt(cs2).
  ADC_HD StateVec<3> eigenvalues(const StateVec<3>& u, const Aux&, int dir) const {
    const Prim p = to_primitive(u);
    const Real vn = (dir == 0 ? p[1] : p[2]);
    const Real c = std::sqrt(cs2);
    StateVec<3> e{};
    e[0] = vn - c;
    e[1] = vn;
    e[2] = vn + c;
    return e;
  }
  /// Vitesses signees (HLL/HLLC) : v_dir -+ c_s.
  ADC_HD void wave_speeds(const StateVec<3>& u, const Aux&, int dir, Real& smin,
                          Real& smax) const {
    const Prim p = to_primitive(u);
    const Real vn = (dir == 0 ? p[1] : p[2]);
    const Real c = std::sqrt(cs2);
    smin = vn - c;
    smax = vn + c;
  }
  static Variables conservative_vars() {
    return {VariableKind::Conservative, {"rho", "rho_u", "rho_v"}, 3,
            {VariableRole::Density, VariableRole::MomentumX, VariableRole::MomentumY}};
  }
  static Variables primitive_vars() {
    return {VariableKind::Primitive, {"rho", "u", "v"}, 3,
            {VariableRole::Density, VariableRole::VelocityX, VariableRole::VelocityY}};
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
// Composition : assemble (hyperbolique, source, elliptic) en un PhysicalModel compile. La brique
// HYPERBOLIQUE porte Vars + flux + vitesses d'onde + conversions (indissociables) ; source et
// elliptique sont des briques separees, librement composables.
// ---------------------------------------------------------------------------

/// Modele physique compose : une brique HYPERBOLIQUE + une source + un second membre elliptique.
/// Satisfait le concept PhysicalModel ; les Vars (conversions + descripteur), le flux et les
/// vitesses d'onde viennent de l'hyperbolique ; pressure / wave_speeds sont exposes quand
/// l'hyperbolique les fournit (necessaire au flux HLLC).
template <class Hyperbolic, class Source, class Elliptic>
struct CompositeModel {
  static_assert(HyperbolicModel<Hyperbolic>,
                "CompositeModel : la 1ere brique doit etre un modele HYPERBOLIQUE (Vars + "
                "conversions cons<->prim + flux + max_wave_speed), cf. concept HyperbolicModel");
  using State = StateVec<Hyperbolic::n_vars>;
  using Prim = typename Hyperbolic::Prim;
  using Aux = adc::Aux;
  static constexpr int n_vars = Hyperbolic::n_vars;

  Hyperbolic hyp{};
  Source src{};
  Elliptic ell{};

  ADC_HD State flux(const State& u, const Aux& a, int dir) const { return hyp.flux(u, a, dir); }
  ADC_HD Real max_wave_speed(const State& u, const Aux& a, int dir) const {
    return hyp.max_wave_speed(u, a, dir);
  }
  ADC_HD State source(const State& u, const Aux& a) const { return src.apply(u, a); }
  ADC_HD Real elliptic_rhs(const State& u) const { return ell.rhs(u); }
  ADC_HD Prim to_primitive(const State& u) const { return hyp.to_primitive(u); }
  ADC_HD State to_conservative(const Prim& p) const { return hyp.to_conservative(p); }
  static Variables conservative_vars() { return Hyperbolic::conservative_vars(); }
  static Variables primitive_vars() { return Hyperbolic::primitive_vars(); }

  ADC_HD Real pressure(const State& u) const
    requires requires(const Hyperbolic h, const State s) { h.pressure(s); }
  {
    return hyp.pressure(u);
  }
  ADC_HD void wave_speeds(const State& u, const Aux& a, int dir, Real& smin, Real& smax) const
    requires requires(const Hyperbolic h, const State s, const Aux aa, int d, Real& lo,
                      Real& hi) { h.wave_speeds(s, aa, d, lo, hi); }
  {
    hyp.wave_speeds(u, a, dir, smin, smax);
  }
};

}  // namespace adc
