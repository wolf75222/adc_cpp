#pragma once

#include <adc/core/foundation/types.hpp>
#include <adc/core/state/variables.hpp>
#include <adc/mesh/execution/for_each.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/mesh/boundary/physical_bc.hpp>
#include <adc/numerics/elliptic/poisson/poisson_operator.hpp>  // apply_laplacian (Lap phi^n)
#include <adc/numerics/linalg/lorentz_eliminator.hpp>         // closed 2x2 B^{-1}

#include <stdexcept>

/// @file
/// @brief BUILDER (NOT solver) of the Schur-condensed source stage of the implicit source
///        coupling potential / velocity / Lorentz (design and references in
///        docs/SCHUR_CONDENSATION_DESIGN.md). This header ONLY ASSEMBLES the coefficients of the tensor
///        elliptic operator A_op and the condensed right-hand side; it does NOT solve and does NOT
///        reconstruct the velocity (stage PR4, which will call the EllipticSolver concept, e.g. TensorKrylovSolver).
///
/// FIXED SIGN CONVENTION (docs/SCHUR_CONDENSATION_DESIGN.md section 2.1):
///   We theta-discretize the source. The unknown is phi^{n+theta}. The condensed operator is
///       L_schur(phi) = -Lap phi - theta^2 dt^2 alpha div( rho B^{-1} grad phi )
///   and its right-hand side
///       RHS = -Lap phi^n - theta dt alpha div( rho B^{-1} v^n ),  v^n = (mx, my) / rho.
///   We write c = theta^2 dt^2 alpha (coefficient of the condensed term).
///
///   WRITING AS A FULL-TENSOR OPERATOR (poisson_operator.hpp). The elliptic path solves
///   L(phi) = -div(A grad phi) + kappa phi, A = [[eps_x, a_xy], [a_yx, eps_y]]. We identify
///       -Lap phi - c div(rho B^{-1} grad phi) = -div( (I + c rho B^{-1}) grad phi ),
///   thus A = I + c rho B^{-1}, which gives PER CELL:
///       eps_x = 1 + c rho binv_11      a_xy = c rho binv_12
///       a_yx  = c rho binv_21          eps_y = 1 + c rho binv_22
///   with B^{-1} = (1/det) [[1, w], [-w, 1]], w = theta dt B_z, det = 1 + w^2 (LorentzEliminator).
///   At B_z = 0: w = 0, binv = I, thus a_xy = a_yx = 0 and eps_x = eps_y = 1 + c rho. If in addition
///   c = 0 (theta=0 or dt=0 or alpha=0), A = I and L_schur degenerates EXACTLY into the canonical
///   Laplacian (standard Poisson). kappa stays NULL (no mass term in the condensation).
///
///   RIGHT-HAND SIDE DISCRETIZATION. -Lap phi^n is computed by apply_laplacian WITHOUT coefficient
///   (canonical 5-point Laplacian = div(grad phi^n)), then negated. The divergence of the EXPLICIT flux
///   F = rho B^{-1} v^n (vector field AT THE CELL CENTERS) is a centered FV divergence:
///       div F (i,j) = (Fx(i+1,j) - Fx(i-1,j)) / (2 dx) + (Fy(i,j+1) - Fy(i,j-1)) / (2 dy),
///   second order, consistent with the centered discretization of the operator (poisson_operator.hpp).
///
/// GENERICITY (docs/SCHUR_CONDENSATION_DESIGN.md section 4). This builder names NO scenario:
///   it reads the ROLES Density / MomentumX / MomentumY (and Energy ignored here) of a VariableSet, plus an
///   aux field B_z, alpha, theta, dt. Any fluid block that exposes these roles is eligible (ExB-drift polar ring OR
///   magnetized multi-species) without one extra line of Schur code. The contract is validated at
///   assembly (roles present, otherwise explicit exception on the HOST side, before any kernel).
///
/// DEVICE / MPI (docs/GPU_RUNTIME_PORT.md). All kernels are device-clean NAMED FUNCTORS
///   (no extended lambda first-instantiated cross-TU, nvcc limitation #64/#97). The coefficients live
///   in MultiFab (one field per entry). All loops iterate over local_size(): a rank WITHOUT box
///   (local_size()==0, MPI decomposition) runs no kernel and never dereferences fab(0) -> MPI-clean.

namespace adc {

namespace detail {

/// Coefficients of the tensor operator A_op = I + c rho B^{-1} ASSEMBLED per cell from the fluid
/// state and the B_z field. Writes eps_x, eps_y (diagonal) and a_xy, a_yx (cross) in component 0 of
/// their respective MultiFab. NAMED functor (device-clean): captures Array4 handles (POD) + scalars.
///   s: fluid state (reads rho = s(.,.,c_rho));
///   bz: B_z field at the center (reads B_z = bz(.,.,0));
///   c_rho: component of the Density role in the state;
///   c: theta^2 dt^2 alpha; th_dt: theta dt (for w = theta dt B_z).
struct SchurOperatorCoeffKernel {
  ConstArray4 s;    ///< fluid state
  ConstArray4 bz;   ///< B_z field at the center
  Array4 ex, ey;    ///< output: eps_x, eps_y (diagonal of A)
  Array4 axy, ayx;  ///< output: cross terms a_xy, a_yx
  Real c;           ///< c = theta^2 dt^2 alpha
  Real th_dt;  ///< theta * dt (for LorentzEliminator: w = th_dt * B_z, and binv depends only on w)
  int c_rho;   ///< Density component
  ADC_HD void operator()(int i, int j) const {
    const Real rho = s(i, j, c_rho);
    // Closed B^{-1} (2x2 rotation-dilation): only w = theta dt B_z enters binv. We reconstruct
    // the eliminator via (theta=th_dt, dt=1, B_z) to obtain w = th_dt * B_z (same binv).
    const LorentzEliminator le(th_dt, Real(1), bz(i, j, 0));
    const Real cr = c * rho;  // c rho: common factor of the 4 entries of A - I
    ex(i, j, 0) = Real(1) + cr * le.binv_11();
    ey(i, j, 0) = Real(1) + cr * le.binv_22();
    axy(i, j, 0) = cr * le.binv_12();
    ayx(i, j, 0) = cr * le.binv_21();
  }
};

/// EXPLICIT flux F = rho B^{-1} v^n (v = (mx,my)/rho) ASSEMBLED per cell, WRITTEN at the center into fx/fy.
/// rho B^{-1} v = B^{-1} (rho v) = B^{-1} (mx, my): we apply B^{-1} DIRECTLY to the momentum
/// (mx, my), which avoids the division by rho (and its rho=0 case). Device-clean NAMED functor.
struct SchurExplicitFluxKernel {
  ConstArray4 s;   ///< fluid state (mx, my read at components c_mx, c_my)
  ConstArray4 bz;  ///< B_z field at the center
  Array4 fx, fy;   ///< output: components of the flux F = B^{-1} (mx, my)
  Real th_dt;      ///< theta * dt (w = th_dt * B_z)
  int c_mx, c_my;  ///< MomentumX / MomentumY components
  ADC_HD void operator()(int i, int j) const {
    const LorentzEliminator le(th_dt, Real(1), bz(i, j, 0));
    Real Fx, Fy;
    le.apply_Binv(s(i, j, c_mx), s(i, j, c_my), Fx, Fy);  // B^{-1} (mx, my) = rho B^{-1} v
    fx(i, j, 0) = Fx;
    fy(i, j, 0) = Fy;
  }
};

/// rhs(i,j) = lap(i,j) (= -Lap phi^n, already negated by the caller) - g * div F, second-order centered
/// divergence of a flux F at the center (fx, fy, ghosts filled). g = theta dt alpha. Device-clean NAMED functor.
struct SchurRhsAssembleKernel {
  ConstArray4 neg_lap;      ///< -Lap phi^n (already negated)
  ConstArray4 fx, fy;       ///< flux F at the center (ghosts filled)
  Array4 rhs;               ///< output: condensed right-hand side
  Real g;                   ///< theta dt alpha
  Real half_idx, half_idy;  ///< 1/(2 dx), 1/(2 dy)
  ADC_HD void operator()(int i, int j) const {
    const Real divF = (fx(i + 1, j, 0) - fx(i - 1, j, 0)) * half_idx +
                      (fy(i, j + 1, 0) - fy(i, j - 1, 0)) * half_idy;
    rhs(i, j, 0) = neg_lap(i, j, 0) - g * divF;
  }
};

/// neg(i,j) = -src(i,j) (negation of component 0). Device-clean NAMED functor.
struct NegateKernel {
  ConstArray4 src;
  Array4 dst;
  ADC_HD void operator()(int i, int j) const { dst(i, j, 0) = -src(i, j, 0); }
};

}  // namespace detail

/// Result of the Schur assembly: the coefficient MultiFab of the tensor operator A_op and the
/// condensed right-hand side. Owns its own fields (move-only via the MultiFab internal std::vector):
/// the caller (PR4) then passes them to GeometricMG::set_epsilon_anisotropic(eps_x, eps_y) +
/// set_cross_terms-per-field / to the TensorKrylovSolver, and solves L_schur(phi) = rhs.
///
/// kappa stays ABSENT (the condensation produces no mass term): the operator is
/// -div(A grad phi), not Helmholtz. eps_x == eps_y == (1 + c rho) and a_xy == a_yx == 0 when B_z == 0.
struct SchurCondensationOperator {
  MultiFab eps_x;  ///< diagonal A_xx = 1 + c rho binv_11
  MultiFab eps_y;  ///< diagonal A_yy = 1 + c rho binv_22
  MultiFab a_xy;   ///< cross A_xy = c rho binv_12
  MultiFab a_yx;   ///< cross A_yx = c rho binv_21
  MultiFab rhs;    ///< condensed right-hand side -Lap phi^n - theta dt alpha div(rho B^{-1} v^n)
};

/// GENERIC builder of the condensed source stage for the electrostatic + Lorentz source
/// (kind="electrostatic_lorentz" on the future Python facade side, PR5). Reads the roles of a fluid block, does NOT
/// solve. This is the "level 1+4 partial" object of docs/SCHUR_CONDENSATION_DESIGN.md restricted to
/// ASSEMBLY (operator + RHS), without the solve nor the velocity reconstruction.
class ElectrostaticLorentzCondensation {
 public:
  /// @p vars: descriptor of the fluid block; MUST expose the roles Density / MomentumX / MomentumY.
  /// @p alpha: coupling constant (electrostatic).
  /// @p theta: implicitness of the theta-scheme (0.5 = Crank-Nicolson).
  /// @p dt: time step of the source stage.
  /// The role contract is validated HERE (host, before any kernel): missing roles -> clear exception.
  ElectrostaticLorentzCondensation(const VariableSet& vars, Real alpha, Real theta, Real dt)
      : c_rho_(vars.index_of(VariableRole::Density)),
        c_mx_(vars.index_of(VariableRole::MomentumX)),
        c_my_(vars.index_of(VariableRole::MomentumY)),
        alpha_(alpha),
        theta_(theta),
        dt_(dt) {
    if (c_rho_ < 0 || c_mx_ < 0 || c_my_ < 0)
      throw std::runtime_error(
          "ElectrostaticLorentzCondensation: the fluid block must expose the roles Density, "
          "MomentumX and MomentumY (VariableSet.roles populated).");
  }

  Real c_coeff() const { return theta_ * theta_ * dt_ * dt_ * alpha_; }  ///< c = theta^2 dt^2 alpha
  int density_comp() const { return c_rho_; }
  int momentum_x_comp() const { return c_mx_; }
  int momentum_y_comp() const { return c_my_; }

  /// Assembles ONLY the coefficients of the tensor operator A_op = I + c rho B^{-1} into
  /// MultiFab eps_x/eps_y (1 ghost, for the operator face harmonic mean) and a_xy/a_yx
  /// (1 ghost, for the face arithmetic mean of the cross fluxes). The ghosts are filled by the
  /// BC @p bc (zero-gradient extrapolation on the physical boundaries, like the existing eps wiring).
  ///   @p state: fluid state (reads rho at the center);
  ///   @p bz: B_z field at the center (1 ghost minimum recommended; read at (i,j) only here).
  /// state/bz/eps_x/... share BoxArray + DistributionMapping (same decomposition). MPI-clean.
  void assemble_operator(const MultiFab& state, const MultiFab& bz, const Geometry& geom,
                         const BCRec& bc, MultiFab& eps_x, MultiFab& eps_y, MultiFab& a_xy,
                         MultiFab& a_yx) const {
    const Real c = c_coeff();
    const Real th_dt = theta_ * dt_;
    for (int li = 0; li < state.local_size(); ++li) {
      const ConstArray4 s = state.fab(li).const_array();
      const ConstArray4 b = bz.fab(li).const_array();
      Array4 ex = eps_x.fab(li).array();
      Array4 ey = eps_y.fab(li).array();
      Array4 fxy = a_xy.fab(li).array();
      Array4 fyx = a_yx.fab(li).array();
      for_each_cell(eps_x.box(li),
                    detail::SchurOperatorCoeffKernel{s, b, ex, ey, fxy, fyx, c, th_dt, c_rho_});
    }
    // coefficient ghosts: the operator face mean reads the neighbor at +-1 (box boundary AND
    // physical boundary). We extend by zero-gradient on the physical boundaries (Foextrap), periodic otherwise
    // (eps_bc of GeometricMG). a_xy/a_yx: face arithmetic mean -> same extension.
    const BCRec ebc = coeff_bc(bc);
    fill_ghosts(eps_x, geom.domain, ebc);
    fill_ghosts(eps_y, geom.domain, ebc);
    fill_ghosts(a_xy, geom.domain, ebc);
    fill_ghosts(a_yx, geom.domain, ebc);
  }

  /// Assembles ONLY the condensed right-hand side -Lap phi^n - theta dt alpha div(rho B^{-1} v^n).
  ///   @p phi_n: current potential phi^n (ghosts filled by the BC @p bc before the Laplacian);
  ///   @p state: fluid state (reads mx, my); @p bz: B_z field at the center;
  ///   @p rhs: output (0 ghost is enough).
  /// Internal buffers (Lap phi^n, flux F) allocated on the layout of @p rhs: transient memory, but
  /// the API stays pure-assembly (no persistent state between calls). MPI-clean (local_size() loops).
  void assemble_rhs(MultiFab& phi_n, const MultiFab& state, const MultiFab& bz,
                    const Geometry& geom, const BCRec& bc, MultiFab& rhs) const {
    const Real th_dt = theta_ * dt_;
    const Real g =
        theta_ * dt_ * alpha_;  // theta dt alpha (coefficient of the div(rho B^{-1} v) term)
    const BoxArray& ba = rhs.box_array();
    const DistributionMapping& dm = rhs.dmap();

    // 1) -Lap phi^n: apply_laplacian WITHOUT coefficient = div(grad phi^n) (canonical Laplacian), negated.
    //    We first fill the ghosts of phi^n (physical BC) so that the boundary stencil is correct.
    device_fence();
    fill_ghosts(phi_n, geom.domain, bc);
    MultiFab lap(ba, dm, 1, 0);
    apply_laplacian(phi_n, geom, lap);  // lap = Lap phi^n (A=I, kappa=0)
    MultiFab neg_lap(ba, dm, 1, 0);
    for (int li = 0; li < neg_lap.local_size(); ++li)
      for_each_cell(neg_lap.box(li),
                    detail::NegateKernel{lap.fab(li).const_array(), neg_lap.fab(li).array()});

    // 2) EXPLICIT flux F = rho B^{-1} v^n = B^{-1} (mx, my), at the center (1 ghost for the centered div).
    MultiFab fx(ba, dm, 1, 1), fy(ba, dm, 1, 1);
    for (int li = 0; li < state.local_size(); ++li) {
      const ConstArray4 s = state.fab(li).const_array();
      const ConstArray4 b = bz.fab(li).const_array();
      for_each_cell(fx.box(li),
                    detail::SchurExplicitFluxKernel{s, b, fx.fab(li).array(), fy.fab(li).array(),
                                                    th_dt, c_mx_, c_my_});
    }
    // ghosts of F: the centered divergence reads Fx(i+-1), Fy(j+-1) (box boundary AND physical boundary).
    const BCRec ebc = coeff_bc(bc);
    fill_ghosts(fx, geom.domain, ebc);
    fill_ghosts(fy, geom.domain, ebc);

    // 3) rhs = -Lap phi^n - g div F (second-order centered divergence).
    const Real half_idx = Real(1) / (Real(2) * geom.dx());
    const Real half_idy = Real(1) / (Real(2) * geom.dy());
    for (int li = 0; li < rhs.local_size(); ++li)
      for_each_cell(rhs.box(li),
                    detail::SchurRhsAssembleKernel{
                        neg_lap.fab(li).const_array(), fx.fab(li).const_array(),
                        fy.fab(li).const_array(), rhs.fab(li).array(), g, half_idx, half_idy});
  }

  /// COMPLETE assembly (operator + RHS) into one SchurCondensationOperator object allocated on the layout of
  /// @p state. Convenience for the PR4 caller; the coefficients carry 1 ghost (operator faces),
  /// the RHS 0 ghost. @p phi_n is modified (ghosts filled).
  SchurCondensationOperator assemble(MultiFab& phi_n, const MultiFab& state, const MultiFab& bz,
                                     const Geometry& geom, const BCRec& bc) const {
    const BoxArray& ba = state.box_array();
    const DistributionMapping& dm = state.dmap();
    SchurCondensationOperator op{MultiFab(ba, dm, 1, 1), MultiFab(ba, dm, 1, 1),
                                 MultiFab(ba, dm, 1, 1), MultiFab(ba, dm, 1, 1),
                                 MultiFab(ba, dm, 1, 0)};
    assemble_operator(state, bz, geom, bc, op.eps_x, op.eps_y, op.a_xy, op.a_yx);
    assemble_rhs(phi_n, state, bz, geom, bc, op.rhs);
    return op;
  }

 private:
  /// BC of the coefficient / flux fields: periodic preserved, physical boundary -> zero-gradient
  /// (Foextrap). Identical to GeometricMG::eps_bc: the face value at the domain boundary equals the interior
  /// value (coefficient continuous at the contour), consistent with the elliptic operator.
  static BCRec coeff_bc(const BCRec& bc) {
    auto fo = [](BCType t) { return t == BCType::Periodic ? t : BCType::Foextrap; };
    BCRec b;
    b.xlo = fo(bc.xlo);
    b.xhi = fo(bc.xhi);
    b.ylo = fo(bc.ylo);
    b.yhi = fo(bc.yhi);
    return b;
  }

  int c_rho_, c_mx_, c_my_;  ///< components of the roles Density / MomentumX / MomentumY
  Real alpha_, theta_, dt_;
};

}  // namespace adc
