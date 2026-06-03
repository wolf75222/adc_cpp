#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/core/variables.hpp>
#include <adc/physics/euler.hpp>  // Euler : reutilise comme brique hyperbolique CompressibleFlux

#include <cmath>

/// @file
/// @brief Briques HYPERBOLIQUES generiques : Vars (cons U / prim P + conversions + descripteur) +
///        flux + vitesses d'onde. Chacune satisfait le concept HyperbolicPhysicalModel : State, Prim,
///        n_vars, flux, max_wave_speed, to_primitive/to_conservative, conservative_vars/primitive_vars
///        (+ pressure/wave_speeds si flux HLLC). Source et second membre elliptique sont des briques
///        SEPAREES (physics/source.hpp, physics/elliptic.hpp) ; CompositeModel (physics/composite.hpp)
///        les assemble. ExBVelocity (1 var), CompressibleFlux (= Euler, 4 var), IsothermalFlux (3 var).

namespace adc {

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
  static VariableSet conservative_vars() {
    return {VariableKind::Conservative, {"n"}, 1, {VariableRole::Density}};
  }
  static VariableSet primitive_vars() {
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
  static VariableSet conservative_vars() {
    return {VariableKind::Conservative, {"rho", "rho_u", "rho_v"}, 3,
            {VariableRole::Density, VariableRole::MomentumX, VariableRole::MomentumY}};
  }
  static VariableSet primitive_vars() {
    return {VariableKind::Primitive, {"rho", "u", "v"}, 3,
            {VariableRole::Density, VariableRole::VelocityX, VariableRole::VelocityY}};
  }
};

}  // namespace adc
