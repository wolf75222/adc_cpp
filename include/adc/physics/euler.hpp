#pragma once

/// @file
/// @brief 2D compressible Euler model (ideal gas): pure HYPERBOLIC brick satisfying
///        the HyperbolicPhysicalModel concept. Source and elliptic right-hand side are
///        separate bricks (physics/source.hpp, physics/elliptic.hpp); this file contains only
///        Vars + flux + wave speeds + cons<->prim conversions.

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/core/variables.hpp>

#include <cmath>

namespace adc {

/**
 * 2D compressible Euler for an ideal gas: HYPERBOLIC brick (HyperbolicModel concept).
 *
 * Conservative variables U = (rho, rho u, rho v, E), with
 * E = p/(gamma-1) + 1/2 rho (u^2 + v^2) and p = (gamma-1)(E - 1/2 rho |v|^2). The
 * directional flux is F_x = (rho u, rho u^2 + p, rho u v, (E+p) u) and symmetrically in y;
 * the maximum wave speed is |v_dir| + c with c = sqrt(gamma p/rho).
 *
 * Pure HYPERBOLIC brick: variables (cons U, prim P) + conversions + flux + wave speeds.
 * NO source or elliptic right-hand side here: those are SEPARATE bricks, assembled by
 * CompositeModel. The aux argument is present for the contract (a drift transport reads grad
 * phi) but does not enter the Euler flux.
 *
 * @note Everything is device-callable (ADC_HD): StateVec over a C array, std::sqrt
 *       (device intrinsic under nvcc), manual abs. Compatible with a GPU kernel like the
 *       scalar transport model.
 */
struct Euler {
  using State = StateVec<4>;        ///< conservative variables (rho, rho u, rho v, E)
  using Prim = StateVec<4>;         ///< primitive variables (rho, u, v, p)
  using Aux = adc::Aux;             ///< auxiliary fields (unused in pure Euler)
  static constexpr int n_vars = 4;  ///< number of conserved variables

  Real gamma = 1.4;  ///< adiabatic index of the ideal gas

  /// Ideal-gas pressure p = (gamma-1)(E - 1/2 rho |v|^2).
  ADC_HD Real pressure(const State& u) const {
    const Real rho = u[0];
    const Real ke = Real(0.5) * (u[1] * u[1] + u[2] * u[2]) / rho;
    return (gamma - Real(1)) * (u[3] - ke);
  }
  /// Sound speed c = sqrt(gamma p / rho).
  ADC_HD Real sound_speed(const State& u) const { return std::sqrt(gamma * pressure(u) / u[0]); }

  /// Conservative -> primitive: (rho, rho u, rho v, E) -> (rho, u, v, p).
  ADC_HD Prim to_primitive(const State& u) const {
    const Real rho = u[0];
    Prim p{};
    p[0] = rho;
    p[1] = u[1] / rho;
    p[2] = u[2] / rho;
    p[3] = pressure(u);
    return p;
  }
  /// Primitive -> conservative: (rho, u, v, p) -> (rho, rho u, rho v, E).
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
   * Extreme signed wave speeds in direction dir: v_dir - c and v_dir + c.
   *
   * Required by the HLL/HLLC fluxes, beyond the single max_wave_speed that Rusanov needs.
   *
   * @param      u    conservative state
   * @param      dir  face direction (0 = x, 1 = y)
   * @param[out] smin leftmost wave speed v_dir - c
   * @param[out] smax rightmost wave speed v_dir + c
   */
  ADC_HD void wave_speeds(const State& u, const Aux&, int dir, Real& smin, Real& smax) const {
    const Prim p = to_primitive(u);
    const Real vn = (dir == 0 ? p[1] : p[2]);
    const Real c = std::sqrt(gamma * p[3] / p[0]);
    smin = vn - c;
    smax = vn + c;
  }

  /// Compressible convective flux in direction dir.
  ADC_HD State flux(const State& u, const Aux&, int dir) const {
    const Real rho = u[0];
    const Real vn = (dir == 0 ? u[1] : u[2]) / rho;  // velocity normal to the face
    const Real p = pressure(u);
    State f{};
    f[0] = rho * vn;
    f[1] = u[1] * vn + (dir == 0 ? p : Real(0));
    f[2] = u[2] * vn + (dir == 1 ? p : Real(0));
    f[3] = (u[3] + p) * vn;
    return f;
  }

  /// Full spectrum in direction dir: (v_dir - c, v_dir, v_dir, v_dir + c). Vector counterpart
  /// of wave_speeds (which only gives the signed extremes); useful for spectrum schemes (Roe).
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

  /// Maximum wave speed |v_dir| + c (Rusanov estimate), computed in primitive variables.
  ADC_HD Real max_wave_speed(const State& u, const Aux&, int dir) const {
    const Prim p = to_primitive(u);
    const Real vn = (dir == 0 ? p[1] : p[2]);
    const Real a = vn < 0 ? -vn : vn;  // |v_dir| device-safe
    return a + std::sqrt(gamma * p[3] / p[0]);
  }

  /// Variable descriptor (hyperbolic model contract; host introspection metadata).
  static VariableSet conservative_vars() {
    return {VariableKind::Conservative,
            {"rho", "rho_u", "rho_v", "E"},
            4,
            {VariableRole::Density, VariableRole::MomentumX, VariableRole::MomentumY,
             VariableRole::Energy}};
  }
  static VariableSet primitive_vars() {
    return {VariableKind::Primitive,
            {"rho", "u", "v", "p"},
            4,
            {VariableRole::Density, VariableRole::VelocityX, VariableRole::VelocityY,
             VariableRole::Pressure}};
  }
};

}  // namespace adc
