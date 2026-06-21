#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/core/variables.hpp>
#include <adc/physics/euler.hpp>  // Euler: reused as the CompressibleFlux hyperbolic brick

#include <cmath>

/// @file
/// @brief Generic HYPERBOLIC bricks: Vars (cons U / prim P + conversions + descriptor) +
///        flux + wave speeds. Each one satisfies the HyperbolicPhysicalModel concept: State, Prim,
///        n_vars, flux, max_wave_speed, to_primitive/to_conservative, conservative_vars/primitive_vars
///        (+ pressure/wave_speeds if HLLC flux). Source and elliptic right-hand side are SEPARATE
///        bricks (physics/source.hpp, physics/elliptic.hpp); CompositeModel (physics/composite.hpp)
///        assembles them. ExBVelocity (1 var), CompressibleFlux (= Euler, 4 var), IsothermalFlux (3 var).

namespace adc {

/// Scalar advection by the E x B drift: v = (-d_y phi, d_x phi)/B0 (divergence-free).
///
/// 1-variable HYPERBOLIC brick (scalar density n). Satisfies HyperbolicPhysicalModel.
/// CONTRACT: purely pointwise functions, device-callable (ADC_HD). No MultiFab,
/// no allocation, no global access. The divergence-free E x B drift ensures
/// exact conservation (no compression term in this flux).
/// Variables cons = prim = {n} (scalar: no nontrivial conversion).
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
  /// Spectrum: one wave, the drift speed in direction dir.
  ADC_HD StateVec<1> eigenvalues(const StateVec<1>&, const Aux& a, int dir) const {
    StateVec<1> e{};
    e[0] = velocity(a, dir);
    return e;
  }
  // Scalar: primitive variables = conservative (transported density).
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

/// Scalar advection by the E x B drift in POLAR coordinates (r, theta) -- "annular polar grid"
/// effort, Phase 1. This is a brick SEPARATE from ExBVelocity (Cartesian), not a
/// modification: the polar solver (assemble_rhs_polar) uses it on a PolarGeometry.
///
/// aux CHANNEL LAYOUT IN POLAR (documented, contract of this brick) -- the base components
/// [0..2] carry the E field in the LOCAL ORTHONORMAL BASIS (e_r, e_theta):
///   aux.phi    [0] = phi (potential; unused by the flux, present for symmetry)
///   aux.grad_x [1] = grad_r     = d phi / d r            (radial component of grad phi)
///   aux.grad_y [2] = grad_theta = (1/r) d phi / d theta  (PHYSICAL AZIMUTHAL component of grad phi)
/// We REUSE the two grad_x/grad_y slots of adc::Aux for grad_r/grad_theta (no new aux
/// field); the MEANING is polar and carried by this brick alone. grad_theta is the
/// PHYSICAL derivative (already divided by r): thus the velocity below is symmetric to the
/// Cartesian one (vr <- -grad_theta/B, vtheta <- grad_r/B) and the caller that fills aux carries the 1/r.
///
/// E x B VELOCITY IN POLAR (PHYSICAL components in the local basis):
///   v_r     = -(1/(B r)) d phi/d theta = -grad_theta / B   (dir == 0, radial)
///   v_theta =  (1/B)     d phi/d r     =  grad_r     / B   (dir == 1, azimuthal)
/// The returned flux (dir 0 = F_r = n v_r; dir 1 = F_theta = n v_theta) is PHYSICAL; the 1/r metric
/// and the divergence (1/r) d_r(r F_r) + (1/r) d_theta(F_theta) are carried by assemble_rhs_polar,
/// NOT by this brick. The brick thus stays a pure physics (no box, no r).
struct ExBVelocityPolar {
  static constexpr int n_vars = 1;
  using State = StateVec<1>;
  Real B0 = 1;
  /// PHYSICAL component of the drift velocity in direction index dir (0 = r, 1 = theta).
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
  /// Spectrum: one wave, the drift speed in direction dir.
  ADC_HD StateVec<1> eigenvalues(const StateVec<1>&, const Aux& a, int dir) const {
    StateVec<1> e{};
    e[0] = velocity(a, dir);
    return e;
  }
  // Scalar: primitive variables = conservative (transported density).
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

/// Compressible 2D Euler flux (reuses Euler: gamma, pressure, signed wave speeds).
/// Compat alias: CompressibleFlux == Euler; the complete hyperbolic brick.
using CompressibleFlux = Euler;

/// ISOTHERMAL Euler flux (p = cs2 rho), 3 variables (rho, rho u, rho v).
///
/// 3-variable HYPERBOLIC brick (density + momenta). Satisfies
/// HyperbolicPhysicalModel. Isothermal closure law: p = cs2 * rho (no energy
/// equation). CONTRACT: purely pointwise functions, device-callable (ADC_HD).
/// No MultiFab, no allocation, no global access.
/// Invariant: cs2 > 0 so that the wave speed sqrt(cs2) is real.
struct IsothermalFlux {
  static constexpr int n_vars = 3;
  using State = StateVec<3>;  ///< conservative variables (rho, rho u, rho v)
  using Prim = StateVec<3>;   ///< primitive variables (rho, u, v)
  Real cs2 = 1;
  /// Quasi-vacuum density floor (ADC-77). When > 0, the velocity is computed as u = m / max(rho,
  /// vacuum_floor) so it stays bounded where the rollup evacuates the background (rho -> ~0); this
  /// bounds BOTH the CFL wave speed and the advective flux in one place (max_wave_speed and flux both
  /// divide by rho here). Mass and momentum are NOT modified -- only the velocity ESTIMATE is bounded,
  /// so the conservative state is untouched (unlike a cell density clamp). <= 0: inactive, and the raw
  /// 1/rho path is taken verbatim (bit-identical, including for rho <= 0).
  Real vacuum_floor = 0;
  /// rho clamped from below by vacuum_floor for the velocity division ONLY. Manual max (device-safe,
  /// no std:: in the kernel path). floor <= 0 -> returns rho unchanged (bit-identical).
  ADC_HD Real velocity_rho(Real rho) const {
    return (vacuum_floor > Real(0) && rho < vacuum_floor) ? vacuum_floor : rho;
  }
  ADC_HD StateVec<3> flux(const StateVec<3>& u, const Aux&, int dir) const {
    const Real rho = u[0];
    const Real vn = (dir == 0 ? u[1] : u[2]) / velocity_rho(rho);
    const Real p = cs2 * rho;
    StateVec<3> f{};
    f[0] = (dir == 0 ? u[1] : u[2]);
    f[1] = u[1] * vn + (dir == 0 ? p : Real(0));
    f[2] = u[2] * vn + (dir == 1 ? p : Real(0));
    return f;
  }
  /// Conservative -> primitive: (rho, rho u, rho v) -> (rho, u, v). The velocity uses the quasi-vacuum
  /// floored density (velocity_rho); rho itself (p[0]) stays the raw conserved value.
  ADC_HD Prim to_primitive(const StateVec<3>& u) const {
    Prim p{};
    p[0] = u[0];
    const Real rho_v = velocity_rho(u[0]);
    p[1] = u[1] / rho_v;
    p[2] = u[2] / rho_v;
    return p;
  }
  /// Primitive -> conservative: (rho, u, v) -> (rho, rho u, rho v).
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
  /// Full spectrum: (v_dir - c, v_dir, v_dir + c), c = sqrt(cs2).
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
  /// Signed speeds (HLL/HLLC): v_dir -+ c_s.
  ADC_HD void wave_speeds(const StateVec<3>& u, const Aux&, int dir, Real& smin, Real& smax) const {
    const Prim p = to_primitive(u);
    const Real vn = (dir == 0 ? p[1] : p[2]);
    const Real c = std::sqrt(cs2);
    smin = vn - c;
    smax = vn + c;
  }
  static VariableSet conservative_vars() {
    return {VariableKind::Conservative,
            {"rho", "rho_u", "rho_v"},
            3,
            {VariableRole::Density, VariableRole::MomentumX, VariableRole::MomentumY}};
  }
  static VariableSet primitive_vars() {
    return {VariableKind::Primitive,
            {"rho", "u", "v"},
            3,
            {VariableRole::Density, VariableRole::VelocityX, VariableRole::VelocityY}};
  }
};

/// ISOTHERMAL Euler flux in POLAR geometry (ring r, theta), 3 variables (rho, rho v_r,
/// rho v_theta) -- "polar fluid grid" effort, Path A step 1. This is a brick SEPARATE
/// from IsothermalFlux (Cartesian): the PHYSICAL flux and the conversions are IDENTICAL (the
/// components 1, 2 are the momentum in the LOCAL ORTHONORMAL BASIS (e_r, e_theta);
/// dir 0 = radial, dir 1 = azimuthal), but this brick adds the GEOMETRIC CURVATURE TERM
/// carried by the polar metric. We inherit IsothermalFlux to NOT duplicate flux /
/// conversions / wave speeds (Cartesian strictly intact, bit-identical) and add ONLY
/// the polar_geom_source method.
///
/// WHY AN EXPLICIT GEOMETRIC TERM (and not a plain conservative divergence):
/// the vector momentum equation d_t(rho v) + div(rho v (x) v) + grad p = 0,
/// projected onto the LOCAL polar basis (e_r, e_theta) which ROTATES with theta, gives for the
/// PHYSICAL components m_r = rho v_r, m_theta = rho v_theta:
///   d_t m_r     + (1/r) d_r(r (rho v_r^2 + p)) + (1/r) d_theta(rho v_r v_theta)
///                 - (rho v_theta^2 + p)/r            = 0      (CENTRIFUGAL + pressure term)
///   d_t m_theta + (1/r) d_r(r rho v_r v_theta)     + (1/r) d_theta(rho v_theta^2 + p)
///                 + (rho v_r v_theta)/r             = 0      (cross CURVATURE term)
/// The assemble_rhs_polar operator computes EXACTLY -(1/r) d_r(r F_r) - (1/r) d_theta(F_theta)
/// with F_r, F_theta = IsothermalFlux::flux: it thus reproduces the divergences, but NOT the
/// algebraic terms -(rho v_theta^2 + p)/r and +(rho v_r v_theta)/r. These terms are NOT
/// captured by the conservative divergence (proof: on the cell (rho, v_r=0, v_theta(r)) in
/// rotational equilibrium d_r p = rho v_theta^2/r, the radial divergence alone would yield
/// d_t m_r = -(d_r p + p/r) != 0, breaking the equilibrium). An explicit GEOMETRIC SOURCE
/// is therefore REQUIRED, provided here and added per cell by assemble_rhs_polar (which alone knows r):
///   S_geom = ( 0 , (rho v_theta^2 + p)/r , -(rho v_r v_theta)/r ).
/// With this source the rotational equilibrium is preserved to the scheme order (cf.
/// test_polar_fluid_equilibrium). r > 0 (ring, r_min > 0): no axis singularity.
///
/// CONTRACT: pointwise PHYSICAL brick, device-callable (ADC_HD), no box, no allocation.
/// polar_geom_source takes ONLY the state and r (no aux): it is pure metric.
struct IsothermalFluxPolar : IsothermalFlux {
  /// GEOMETRIC curvature source term in a cell of radius r > 0 (ring). See the @file block
  /// above for the derivation. S_geom = (0, (rho v_theta^2 + p)/r, -(rho v_r v_theta)/r),
  /// p = cs2 rho. Component 0 (mass) is zero: mass is purely conservative in polar.
  ADC_HD StateVec<3> polar_geom_source(const StateVec<3>& u, Real r) const {
    const Real rho = u[0];
    const Real inv_rho =
        Real(1) / velocity_rho(rho);   // quasi-vacuum floored (ADC-77; bit-identical if off)
    const Real mr = u[1], mth = u[2];  // rho v_r, rho v_theta (local basis (e_r, e_theta))
    const Real p = cs2 * rho;
    const Real inv_r = Real(1) / r;
    StateVec<3> s{};
    s[0] = Real(0);
    s[1] = (mth * mth * inv_rho + p) * inv_r;  // (rho v_theta^2 + p)/r: centrifugal + pressure
    s[2] = -(mr * mth * inv_rho) * inv_r;      // -(rho v_r v_theta)/r: cross curvature
    return s;
  }
};

}  // namespace adc
