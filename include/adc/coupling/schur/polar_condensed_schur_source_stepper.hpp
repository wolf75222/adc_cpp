#pragma once

#include <adc/core/foundation/types.hpp>
#include <adc/core/state/variables.hpp>
#include <adc/coupling/schur/schur_source_kernels.hpp>  // shared geometry-free kernels + validate_krylov_params (#263)
#include <adc/mesh/execution/for_each.hpp>
#include <adc/mesh/geometry/geometry.hpp>  // PolarGeometry
#include <adc/mesh/storage/mf_arith.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/mesh/boundary/physical_bc.hpp>
#include <adc/numerics/elliptic/polar/polar_tensor_operator.hpp>  // PolarTensorKrylovSolver, apply_polar_tensor (#210)
#include <adc/numerics/lorentz_eliminator.hpp>  // B^{-1} closed form (#118)
#include <adc/parallel/comm.hpp>

#include <stdexcept>

/// @file
/// @brief PolarCondensedSchurSourceStepper: SOURCE STAGE condensed by Schur in POLAR geometry
///        (ring (r, theta)). POLAR counterpart of the Cartesian CondensedSchurSourceStepper (#126,
///        condensed_schur_source_stepper.hpp), for the implicit source coupling potential / velocity /
///        Lorentz of a STIFF magnetized fluid on a ring mesh.
///        It is a STANDALONE SOURCE stage (transport frozen). WIRED into the facade since
///        Path A step 2c: System::set_source_stage builds it when the geometry is polar and
///        SystemStepper::run_source_stage invokes it after transport (cf. python/system.cpp) --
///        the old "does not get wired in" mention was stale (audit 2026-06).
///
/// SEPARATE, ADDITIVE PATH (Path A step 2b). The CARTESIAN Schur stays BIT-IDENTICAL
/// (condensed_schur_source_stepper.hpp UNTOUCHED); this header touches no existing path. It COMPOSES
/// the bricks already on master:
///   - PolarTensorKrylovSolver (polar_tensor_operator.hpp, #210): matrix-free BiCGStab for
///     L_int(phi) = div(A grad phi) in POLAR METRIC, FULL non-symmetric tensor A = [[a_rr, a_rt], [a_tr, a_tt]]
///     (RadialLine preconditioner);
///   - LorentzEliminator (lorentz_eliminator.hpp, #118): B^{-1} closed form (2x2 rotation-dilation).
///
/// WHY THE SAME 2x2 B^{-1} AS IN CARTESIAN. The polar velocity (v_r, v_theta) is expressed in the
/// LOCAL ORTHONORMAL frame (e_r, e_theta). The magnetic field is carried by z (B = B_z z_hat,
/// orthogonal to the (r, theta) plane). The Lorentz force v x B is a ROTATION in the plane, INDEPENDENT
/// of the orientation of the local orthonormal frame: (v x B)_r = +B_z v_theta, (v x B)_theta = -B_z v_r,
/// exactly the Cartesian form (v x B)_x = +B_z v_y, (v x B)_y = -B_z v_x. Thus the SAME
/// LorentzEliminator applies to (v_r, v_theta), and the condensed tensor A = I + c rho B^{-1} (c =
/// theta^2 dt^2 alpha) has the same entries, simply RE-LABELED (x -> r, y -> theta):
///   a_rr = 1 + c rho binv_11      a_rt = c rho binv_12
///   a_tr = c rho binv_21          a_tt = 1 + c rho binv_22
/// with B^{-1} = (1/det)[[1, w], [-w, 1]], w = theta dt B_z, det = 1 + w^2. This is EXACTLY the FULL
/// non-symmetric tensor that PolarTensorKrylovSolver (#210) accepts: the cross term a_rt/a_tr IS the
/// Lorentz rotation. The POLAR METRIC (1/r, 1/r^2) is carried by the solver operator; the
/// stepper only assembles the cell-centered coefficients and the right-hand side.
///
/// SEQUENCE (once per source stage), polar counterpart of docs/SCHUR_CONDENSATION_DESIGN.md level 4:
///   1. ASSEMBLE A = I + c rho^n B^{-1} (4 cell-centered coefficient fields) and the condensed
///      right-hand side rhs_polar = Lap_polar phi^n + theta dt alpha div_polar(rho^n B^{-1} v^n);
///   2. SOLVE L_int(phi) = div(A grad phi) = rhs_polar via PolarTensorKrylovSolver;
///   3. RECONSTRUCT v^{n+theta} = B^{-1}(v^n - theta dt grad_polar phi^{n+theta}), mom = rho^n v;
///   4. ENERGY (if Energy role): E^{n+1} = E^n + (1/2) rho (|v^{n+1}|^2 - |v^n|^2);
///   5. EXTRAPOLATE U^{n+theta} -> U^{n+1}: f^{n+1} = f^n + (1/theta)(f^{n+theta} - f^n);
///   6. FILL the ghosts (theta periodic, r physical) before returning.
///
/// SOLVE SIGN CONVENTION (read before any modification). The Cartesian builder (#124) writes
///   the stage L_schur(phi) = rhs_schur with L_schur(phi) = -div(A grad phi), and
///       rhs_schur = -Lap phi^n - theta dt alpha div(rho B^{-1} v^n).
///   The PolarTensorKrylovSolver (#210) solves L_int(phi) = +div(A grad phi) = rhs (POSITIVE sign,
///   cf. test_polar_tensor_elliptic_mms: the RHS there equals directly f = div(A grad phi_exact)). We have
///   L_schur = -L_int, so solving L_schur(phi) = rhs_schur amounts to solving L_int(phi) = -rhs_schur.
///   The POLAR right-hand side fed to the solver is therefore -rhs_schur:
///       rhs_polar = -rhs_schur = +Lap_polar phi^n + theta dt alpha div_polar(rho B^{-1} v^n).
///   Safeguard: c = 0 and B_z = 0 -> A = I, the solve becomes Lap_polar phi^{n+theta} = Lap_polar phi^n
///   -> phi^{n+theta} = phi^n (up to the gauge), and the reconstruction degenerates to the explicit
///   electrostatic push v^{n+theta} = v^n - theta dt grad_polar phi^n (case B = 0).
///
/// RIGHT-HAND SIDE DISCRETIZATION (consistent with the polar operator #210 and assemble_rhs_polar).
///   - Lap_polar phi^n: applies apply_polar_tensor with A = I (coefficients at 1) = (1/r) d_r(r d_r phi)
///     + (1/r^2) d_theta^2 phi, conservative radial FV stencil + 2-point azimuthal FD (the SAME as
///     the solve operator, guaranteeing the exact degeneracy at c=0, B_z=0).
///   - div_polar(F), F = rho B^{-1} v^n = B^{-1}(mr, mtheta) at the center: centered FV polar divergence
///       div F (i,j) = (1/r_i) (r_{i+1/2} - r_{i-1/2} spread by centered difference) ... discretized by
///       div F = (1/r_i)(d_r(r F_r))_center + (1/r_i) (d_theta F_theta)_center, that is
///       div F = (1/r_i) [ (r_{i+1} F_r(i+1) - r_{i-1} F_r(i-1)) / (2 dr) + (F_th(i,j+1) - F_th(i,j-1)) / (2 dtheta) ].
///     Centered, second order (polar counterpart of the Cartesian centered divergence of builder #124).
///
/// POLAR GRADIENT (reconstruction). grad_polar phi = (d_r phi, (1/r) d_theta phi), CENTERED differences:
///       (grad phi)_r     = (phi(i+1,j) - phi(i-1,j)) / (2 dr)
///       (grad phi)_theta = (phi(i,j+1) - phi(i,j-1)) / (2 r_i dtheta).
///   Same centered differences as the RHS divergence (term-by-term consistency at solve precision,
///   as the Cartesian SchurReconstructKernel uses a centered gradient).
///
/// SCOPE: MULTI-RANK MPI AND MULTI-BOX (several boxes PER RANK) by AZIMUTHAL split (theta only),
///   like PolarTensorKrylovSolver (#210). The body of step() already iterates over local_size() everywhere and
///   goes through fill_ghosts (MPI halo exchange + physical BC, DIAGONAL CORNERS included for the cross
///   terms of the 9-point stencil) for all fields (bz_/a_rr_/a_tt_/a_rt_/a_tr_/fr_/ft_/phi/state)
///   : it is structurally distributed and multi-box. The elliptic solve (PolarTensorKrylovSolver)
///   supports multi-rank AND multi-box under the theta-split constraint (each box covers the full
///   radial range for the RadialLine preconditioner; the solver safeguard check_radial_columns
///   enforces it -- Jacobi fallback without constraint for a 2D tiling that cuts r). Single-rank / single
///   box: BIT-IDENTICAL path. Multi-box validation: test_polar_schur_multibox (multi-box vs mono-box parity
///   bit-close, cross terms AND 2D Jacobi tiling). Device: all kernels of the path
///   are NAMED device-clean functors (recipe #93); CUDA-device validation = ci-full.
///
/// DEVICE. All kernels are NAMED device-clean FUNCTORS (recipe #93: no extended lambda
///   first-instantiated cross-TU, nvcc limit #64/#97). The buffers are ALLOCATED ONCE at
///   construction and REUSED on each step(). The Krylov solver is created per call (fixed buffers).

namespace adc {

namespace detail {

/// POLAR condensed tensor coefficients A = I + c rho B^{-1} at cell centers. Polar re-labeling
/// of the Cartesian coefficients (#124, SchurOperatorCoeffKernel): x -> r, y -> theta. The SAME
/// 2x2 B^{-1} (Lorentz rotation) because the force v x B is independent of the orientation of the local
/// orthonormal frame (cf. header). NAMED device-clean functor.
struct PolarSchurOperatorCoeffKernel {
  ConstArray4 s;    ///< fluid state (read rho)
  ConstArray4 bz;   ///< B_z field at the center
  Array4 arr, att;  ///< output: a_rr, a_tt (diagonal of A)
  Array4 art, atr;  ///< output: cross terms a_rt, a_tr
  Real c;           ///< c = theta^2 dt^2 alpha
  Real th_dt;       ///< theta * dt (w = th_dt * B_z, binv depends only on w)
  int c_rho;        ///< Density component
  ADC_HD void operator()(int i, int j) const {
    const Real rho = s(i, j, c_rho);
    const LorentzEliminator le(th_dt, Real(1), bz(i, j, 0));  // w = th_dt * B_z
    const Real cr = c * rho;
    arr(i, j, 0) = Real(1) + cr * le.binv_11();
    att(i, j, 0) = Real(1) + cr * le.binv_22();
    art(i, j, 0) = cr * le.binv_12();
    atr(i, j, 0) = cr * le.binv_21();
  }
};

/// EXPLICIT flux F = rho B^{-1} v^n = B^{-1}(mr, mtheta) at the center (physical components (e_r,
/// e_theta)). We apply B^{-1} DIRECTLY to the momentum (avoids the division by rho).
/// NAMED device-clean functor. Polar re-labeling of SchurExplicitFluxKernel (#124).
struct PolarSchurExplicitFluxKernel {
  ConstArray4 s;   ///< fluid state (mr, mtheta read at components c_mx, c_my)
  ConstArray4 bz;  ///< B_z field at the center
  Array4 fr, ft;   ///< output: F_r, F_theta = B^{-1}(mr, mtheta)
  Real th_dt;      ///< theta * dt (w = th_dt * B_z)
  int c_mx, c_my;  ///< MomentumX (= radial), MomentumY (= azimuthal) components
  ADC_HD void operator()(int i, int j) const {
    const LorentzEliminator le(th_dt, Real(1), bz(i, j, 0));
    Real Fr, Ft;
    le.apply_Binv(s(i, j, c_mx), s(i, j, c_my), Fr, Ft);  // B^{-1}(mr, mtheta) = rho B^{-1} v
    fr(i, j, 0) = Fr;
    ft(i, j, 0) = Ft;
  }
};

/// rhs_polar(i,j) = lap_polar(i,j) (= Lap_polar phi^n) + g * div_polar F, second-order centered POLAR
/// divergence of a vector field F = (F_r, F_theta) at the center (ghosts filled). g = theta dt alpha.
///   div_polar F = (1/r_i) [ (r_{i+1} F_r(i+1) - r_{i-1} F_r(i-1)) / (2 dr)
///                          + (F_th(i,j+1) - F_th(i,j-1)) / (2 dtheta) ].
/// The POSITIVE sign of div (and of lap) is the opposite of the Cartesian schur convention: we solve
/// L_int = +div(A grad phi) = rhs_polar = -rhs_schur (cf. header, sign convention). NAMED functor.
struct PolarSchurRhsAssembleKernel {
  ConstArray4 lap;           ///< Lap_polar phi^n (positive sign, A=I)
  ConstArray4 fr, ft;        ///< flux F at the center (ghosts filled)
  Array4 rhs;                ///< output: condensed right-hand side (L_int sign)
  Real g;                    ///< theta dt alpha
  Real half_idr, half_idth;  ///< 1/(2 dr), 1/(2 dtheta)
  Real r_min, dr;            ///< for r_cell(i), r_cell(i+-1)
  ADC_HD void operator()(int i, int j) const {
    const Real ri = r_min + (i + Real(0.5)) * dr;
    const Real rip = r_min + (i + Real(1.5)) * dr;  // r_cell(i+1)
    const Real rim = r_min + (i - Real(0.5)) * dr;  // r_cell(i-1)
    const Real inv_r = Real(1) / ri;
    const Real div_r =
        (rip * fr(i + 1, j, 0) - rim * fr(i - 1, j, 0)) * half_idr;      // d_r(r F_r) centered
    const Real div_t = (ft(i, j + 1, 0) - ft(i, j - 1, 0)) * half_idth;  // d_theta(F_th) centered
    const Real divF = inv_r * (div_r + div_t);
    rhs(i, j, 0) = lap(i, j, 0) + g * divF;
  }
};

/// Reconstructs v^{n+theta} = B^{-1}(v^n - theta dt grad_polar phi^{n+theta}) and writes mom = rho^n
/// v^{n+theta}. grad_polar phi is the CENTERED difference: (d_r phi, (1/r) d_theta phi). rho read from
/// the state (Density role), FROZEN. NAMED device-clean functor. Polar counterpart of SchurReconstructKernel.
struct PolarSchurReconstructKernel {
  ConstArray4 phi;           ///< phi^{n+theta} (ghosts filled: centered grad reads i+-1, j+-1)
  ConstArray4 vr, vt;        ///< v^n (components 0: velocity, NOT momentum)
  ConstArray4 bz;            ///< B_z field at the center
  Array4 st;                 ///< fluid state (WRITE mr, mtheta; READ rho)
  Array4 nvr, nvt;           ///< output: v^{n+theta} (component 0) for the energy / the diagnostic
  Real th_dt;                ///< theta * dt (w = th_dt * B_z, and gradient factor)
  Real half_idr, half_idth;  ///< 1/(2 dr), 1/(2 dtheta)
  Real r_min, dr;            ///< for r_cell(i) (azimuthal metric 1/r)
  int c_rho, c_mx, c_my;     ///< Density / MomentumX (radial) / MomentumY (azimuthal) components
  ADC_HD void operator()(int i, int j) const {
    const Real ri = r_min + (i + Real(0.5)) * dr;
    const Real gr = (phi(i + 1, j, 0) - phi(i - 1, j, 0)) * half_idr;          // d_r phi
    const Real gt = (phi(i, j + 1, 0) - phi(i, j - 1, 0)) * (half_idth / ri);  // (1/r) d_theta phi
    const Real rhsr = vr(i, j, 0) - th_dt * gr;  // (v^n - theta dt grad_polar phi)_r
    const Real rhst = vt(i, j, 0) - th_dt * gt;
    const LorentzEliminator le(th_dt, Real(1), bz(i, j, 0));  // w = th_dt * B_z
    Real nr, nt;
    le.apply_Binv(rhsr, rhst, nr, nt);  // v^{n+theta} = B^{-1}(v^n - theta dt grad_polar phi)
    nvr(i, j, 0) = nr;
    nvt(i, j, 0) = nt;
    const Real rho = st(i, j, c_rho);  // rho^n (frozen in the source)
    st(i, j, c_mx) = rho * nr;         // mom^{n+theta} = rho^n v^{n+theta}
    st(i, j, c_my) = rho * nt;
  }
};

// The geometry-free extrapolate / energy / extract-velocity / copy-Bz kernels are shared with the
// Cartesian stepper via <adc/coupling/schur/schur_source_kernels.hpp> (#263): detail::SchurExtrapolateScalarKernel,
// SchurExtrapolateVelocityKernel, SchurEnergyKernel, ExtractVelocityKernel, CopyBzKernel. Their member
// fields are named vx/vy but hold the polar velocity (vr/vtheta): the math is frame-independent. Only the
// metric-bearing kernels above (operator-coeff, explicit-flux, RHS-assemble, reconstruct) stay local.

/// dst <- src (component 0). NAMED device-clean functor (local: the polar header does not depend on
/// geometric_mg.hpp, like polar_tensor_operator.hpp).
struct PolarSchurCopyComp0Kernel {
  Array4 d;
  ConstArray4 s;
  ADC_HD void operator()(int i, int j) const { d(i, j, 0) = s(i, j, 0); }
};

}  // namespace detail

/// SOURCE STAGE condensed by Schur in POLAR geometry, STANDALONE (transport frozen), GENERIC over any
/// polar fluid block that exposes the roles Density / MomentumX (radial) / MomentumY (azimuthal)
/// (+ optional Energy). Polar counterpart of CondensedSchurSourceStepper; SEPARATE path, the Cartesian
/// one stays BIT-IDENTICAL.
///
/// ROLE CONVENTION IN POLAR: the RADIAL momentum (rho v_r) carries the MomentumX role,
/// the AZIMUTHAL momentum (rho v_theta) carries the MomentumY role (cf. IsothermalFluxPolar:
/// component 1 = radial, component 2 = azimuthal, local orthonormal basis (e_r, e_theta)).
///
/// LIFECYCLE: built ONCE on a fixed (PolarGeometry + BoxArray); allocates its buffers at
/// construction; step() REUSES them. theta/dt may change between calls.
class PolarCondensedSchurSourceStepper {
 public:
  /// @p vars: descriptor of the fluid block; MUST expose Density / MomentumX / MomentumY (Energy
  ///            optional). Contract validated HERE (host), before any kernel.
  /// @p geom: POLAR geometry (ring); @p ba: SINGLE box; @p bcPhi: radial BC of the potential
  ///            (xlo/xhi: Dirichlet or Foextrap; theta always periodic on the solver side).
  /// @p alpha: electrostatic coupling constant.
  /// @p precond: preconditioner of the PolarTensorKrylovSolver (RadialLine by default).
  PolarCondensedSchurSourceStepper(const VariableSet& vars, const PolarGeometry& geom,
                                   const BoxArray& ba, const BCRec& bcPhi, Real alpha,
                                   PolarPrecond precond = PolarPrecond::RadialLine)
      : PolarCondensedSchurSourceStepper(
            vars, vars.index_of(VariableRole::Density), vars.index_of(VariableRole::MomentumX),
            vars.index_of(VariableRole::MomentumY), vars.index_of(VariableRole::Energy), geom, ba,
            bcPhi, alpha, precond) {}

  /// EXPLICIT-COMPONENT variant (audit wave 3, parity with the Cartesian stepper):
  /// the caller DESIGNATES the components (rho, m_r, m_theta[, E]) -- block with free names / Custom
  /// roles. The canonical ctor above DELEGATES here (role resolution unchanged).
  PolarCondensedSchurSourceStepper(const VariableSet& vars, int c_rho, int c_mx, int c_my, int c_E,
                                   const PolarGeometry& geom, const BoxArray& ba,
                                   const BCRec& bcPhi, Real alpha,
                                   PolarPrecond precond = PolarPrecond::RadialLine)
      : vars_(vars),
        c_rho_(c_rho),
        c_mx_(c_mx),
        c_my_(c_my),
        c_E_(c_E),
        alpha_(alpha),
        geom_(geom),
        bcPhi_(bcPhi),
        precond_(precond),
        ba_(ba),
        dm_(ba.size(), n_ranks()),
        // condensed tensor coefficients A = I + c rho B^{-1} (1 ghost: operator faces)
        a_rr_(ba, dm_, 1, 1),
        a_tt_(ba, dm_, 1, 1),
        a_rt_(ba, dm_, 1, 1),
        a_tr_(ba, dm_, 1, 1),
        // condensed RHS buffers
        lap_(ba, dm_, 1, 0),
        rhs_(ba, dm_, 1, 0),
        fr_(ba, dm_, 1, 1),
        ft_(ba, dm_, 1, 1),
        // a MultiFab at 1 (A=I) for the scalar Lap_polar of the RHS
        one_rr_(ba, dm_, 1, 1),
        one_tt_(ba, dm_, 1, 1),
        bz_(ba, dm_, 1, 1),
        phi_n_(ba, dm_, 1, 1),
        vr_n_(ba, dm_, 1, 0),
        vt_n_(ba, dm_, 1, 0),
        vr_t_(ba, dm_, 1, 0),
        vt_t_(ba, dm_, 1, 0) {
    // MULTI-RANK MPI (theta-only split): no more single-rank safeguard. The layout constraint
    // (each box covers the full radial range, required by the RadialLine preconditioner of the elliptic
    // solve) is checked by the PolarTensorKrylovSolver built in step() (check_radial_columns)
    // -> a clear error is raised on all ranks if the split cuts r. Single-rank / single box:
    // path unchanged (the check passes trivially, all_reduce = identity in serial).
    if (c_rho_ < 0 || c_mx_ < 0 || c_my_ < 0)
      throw std::runtime_error(
          "PolarCondensedSchurSourceStepper: the fluid block must expose the roles Density, "
          "MomentumX "
          "(radial) and MomentumY (azimuthal).");
    one_rr_.set_val(Real(1));
    one_tt_.set_val(Real(1));
  }

  /// true if the model carries an Energy role (energy update active).
  bool has_energy() const { return c_E_ >= 0; }

  /// POLAR condensed SOURCE STAGE, IN-PLACE on @p state and @p phi.
  ///   @p state: fluid state (rho, mom_r, mom_theta [, E]); WRITES mom (+ E); rho FROZEN.
  ///   @p phi: potential; INPUT phi^n (warm start of the solve); OUTPUT phi^{n+1}.
  ///   @p bz_field: B_z field (aux channel), component @p c_bz read at the center. theta/dt: theta-scheme.
  void step(MultiFab& state, MultiFab& phi, const MultiFab& bz_field, int c_bz, Real theta,
            Real dt) {
    const Real th_dt = theta * dt;
    const Real c = theta * theta * dt * dt * alpha_;  // c = theta^2 dt^2 alpha
    const Real g = theta * dt * alpha_;               // theta dt alpha
    const Real dr = geom_.dr();
    const Real dth = geom_.dtheta();
    const Real half_idr = Real(1) / (Real(2) * dr);
    const Real half_idth = Real(1) / (Real(2) * dth);

    // -1) freeze phi^n (final extrapolation).
    copy_comp0(phi_n_, phi);

    // 0) extract v^n = (mr, mtheta)/rho and copy B_z into the internal buffer.
    for (int li = 0; li < state.local_size(); ++li) {
      const ConstArray4 s = state.fab(li).const_array();
      for_each_cell(state.box(li),
                    detail::ExtractVelocityKernel{s, vr_n_.fab(li).array(), vt_n_.fab(li).array(),
                                                  c_rho_, c_mx_, c_my_});
      for_each_cell(bz_.box(li), detail::CopyBzKernel{bz_field.fab(li).const_array(),
                                                      bz_.fab(li).array(), c_bz});
    }
    const BCRec ebc = coeff_bc(bcPhi_);
    device_fence();
    fill_ghosts(bz_, geom_.domain, ebc);

    // 1a) ASSEMBLE the coefficients A = I + c rho B^{-1} at the center (4 fields).
    for (int li = 0; li < state.local_size(); ++li) {
      const ConstArray4 s = state.fab(li).const_array();
      const ConstArray4 b = bz_.fab(li).const_array();
      for_each_cell(a_rr_.box(li),
                    detail::PolarSchurOperatorCoeffKernel{
                        s, b, a_rr_.fab(li).array(), a_tt_.fab(li).array(), a_rt_.fab(li).array(),
                        a_tr_.fab(li).array(), c, th_dt, c_rho_});
    }
    // ghosts of the coefficients (the operator face average reads the neighbor at +-1): theta
    // periodic, radial Foextrap. PolarTensorKrylovSolver::set_coefficients also fills these ghosts, but
    // we fill them here for robustness (assemble_rhs reads fr/ft -> consistency).
    device_fence();
    fill_ghosts(a_rr_, geom_.domain, ebc);
    fill_ghosts(a_tt_, geom_.domain, ebc);
    fill_ghosts(a_rt_, geom_.domain, ebc);
    fill_ghosts(a_tr_, geom_.domain, ebc);

    // 1b) ASSEMBLE the condensed right-hand side rhs_polar = Lap_polar phi^n + g div_polar(rho B^{-1} v^n).
    //     Lap_polar phi^n: apply_polar_tensor with A = I (coefficients at 1), SAME stencil as the solve.
    device_fence();
    fill_ghosts(phi, geom_.domain, phi_bc());  // ghosts of phi^n for the boundary Laplacian
    apply_polar_tensor(phi, geom_, lap_, &one_rr_, &one_tt_, nullptr,
                       nullptr);  // lap_ = Lap_polar phi^n
    // explicit flux F = B^{-1}(mr, mtheta) at the center (1 ghost for the centered div).
    for (int li = 0; li < state.local_size(); ++li) {
      const ConstArray4 s = state.fab(li).const_array();
      const ConstArray4 b = bz_.fab(li).const_array();
      for_each_cell(fr_.box(li),
                    detail::PolarSchurExplicitFluxKernel{s, b, fr_.fab(li).array(),
                                                         ft_.fab(li).array(), th_dt, c_mx_, c_my_});
    }
    device_fence();
    fill_ghosts(fr_, geom_.domain, ebc);  // centered div reads F(i+-1), F(j+-1)
    fill_ghosts(ft_, geom_.domain, ebc);
    // rhs_polar = Lap_polar phi^n + g div_polar F (sign L_int = -rhs_schur).
    for (int li = 0; li < rhs_.local_size(); ++li)
      for_each_cell(rhs_.box(li), detail::PolarSchurRhsAssembleKernel{
                                      lap_.fab(li).const_array(), fr_.fab(li).const_array(),
                                      ft_.fab(li).const_array(), rhs_.fab(li).array(), g, half_idr,
                                      half_idth, geom_.r_min, dr});

    // 2) SOLVE L_int(phi) = div(A grad phi) = rhs_polar via PolarTensorKrylovSolver.
    PolarTensorKrylovSolver kry(geom_, ba_, bcPhi_, precond_);
    const bool cross =
        (c != Real(0));  // non-trivial cross terms as soon as c != 0 (B_z != 0 -> binv_12 != 0)
    if (cross)
      kry.set_coefficients(&a_rr_, &a_tt_, &a_rt_, &a_tr_);
    else
      kry.set_coefficients(&a_rr_, &a_tt_);
    copy_comp0(kry.phi(), phi);  // warm start: phi^n -> kry.phi()
    copy_comp0(kry.rhs(), rhs_);
    last_result_ = kry.solve(krylov_tol_, krylov_max_iters_);
    copy_comp0(phi, kry.phi());  // phi <- phi^{n+theta}

    // 3) RECONSTRUCT v^{n+theta} = B^{-1}(v^n - theta dt grad_polar phi^{n+theta}); mom = rho v.
    device_fence();
    fill_ghosts(phi, geom_.domain, phi_bc());  // centered grad_polar reads phi(i+-1), phi(j+-1)
    for (int li = 0; li < state.local_size(); ++li)
      for_each_cell(state.box(li),
                    detail::PolarSchurReconstructKernel{
                        phi.fab(li).const_array(), vr_n_.fab(li).const_array(),
                        vt_n_.fab(li).const_array(), bz_.fab(li).const_array(),
                        state.fab(li).array(), vr_t_.fab(li).array(), vt_t_.fab(li).array(), th_dt,
                        half_idr, half_idth, geom_.r_min, dr, c_rho_, c_mx_, c_my_});
    // vr_t_/vt_t_ carry v^{n+theta}.

    // 5) EXTRAPOLATE phi and v from the theta-stage to the full step: f^{n+1} = f^n + (1/theta)(f^{n+theta}-f^n).
    const Real inv_theta = Real(1) / theta;
    for (int li = 0; li < phi.local_size(); ++li)
      for_each_cell(phi.box(li), detail::SchurExtrapolateScalarKernel{
                                     phi_n_.fab(li).const_array(), phi.fab(li).array(), inv_theta});
    for (int li = 0; li < state.local_size(); ++li)
      for_each_cell(state.box(li), detail::SchurExtrapolateVelocityKernel{
                                       vr_n_.fab(li).const_array(), vt_n_.fab(li).const_array(),
                                       vr_t_.fab(li).array(), vt_t_.fab(li).array(),
                                       state.fab(li).array(), inv_theta, c_rho_, c_mx_, c_my_});

    // 4) ENERGY (if role present): E^{n+1} = E^n + (1/2) rho (|v^{n+1}|^2 - |v^n|^2).
    if (c_E_ >= 0)
      for (int li = 0; li < state.local_size(); ++li)
        for_each_cell(state.box(li), detail::SchurEnergyKernel{
                                         vr_n_.fab(li).const_array(), vt_n_.fab(li).const_array(),
                                         vr_t_.fab(li).const_array(), vt_t_.fab(li).const_array(),
                                         state.fab(li).array(), c_rho_, c_E_});

    // 6) FILL the ghosts of the state and the potential before returning.
    device_fence();
    fill_ghosts(state, geom_.domain, bcU_default());
    fill_ghosts(phi, geom_.domain, phi_bc());
  }

  /// Diagnostic of the last solve (BiCGStab iterations, relative residual, convergence).
  const PolarKrylovResult& last_solve() const { return last_result_; }

  /// Tolerance / iteration budget of the polar Krylov solve. DEFAULTS = historical constants
  /// (1e-10, 600), made configurable by the audit 2026-06. @throws std::invalid_argument.
  void set_krylov(Real tol, int max_iters) {
    detail::validate_krylov_params(tol, max_iters, "PolarCondensedSchurSourceStepper::set_krylov");
    krylov_tol_ = tol;
    krylov_max_iters_ = max_iters;
  }

  int density_comp() const { return c_rho_; }
  int momentum_x_comp() const { return c_mx_; }
  int momentum_y_comp() const { return c_my_; }
  int energy_comp() const { return c_E_; }

 private:
  /// Default state BC for the final ghost fill: theta periodic kept, radial
  /// Foextrap. The source stage is local in space; the state BC only affects the published ghosts.
  BCRec bcU_default() const { return coeff_bc(bcPhi_); }

  /// BC of the coefficient / flux fields: theta periodic kept, radial physical boundary -> Foextrap.
  static BCRec coeff_bc(const BCRec& bc) {
    auto fo = [](BCType t) { return t == BCType::Periodic ? t : BCType::Foextrap; };
    BCRec b;
    b.xlo = fo(bc.xlo);
    b.xhi = fo(bc.xhi);
    b.ylo = BCType::Periodic;
    b.yhi = BCType::Periodic;  // theta always periodic
    return b;
  }

  /// POTENTIAL BC for the stepper fill_ghosts: caller radial (type + Dirichlet value)
  /// kept, theta FORCED periodic. On the ring, theta has no physical boundary; the ctor contract
  /// ("theta always periodic on the solver side") is applied by PolarTensorKrylovSolver and
  /// coeff_bc, but the ghosts of phi were filled with RAW bcPhi_: a caller setting Dirichlet
  /// in y (System::poisson_bc sets Dirichlet on the 4 faces) obtained at the theta=0/2pi seam
  /// ghosts by ODD REFLECTION (ghost = 2*0 - mirror = -phi) instead of the wrap. The centered azimuthal
  /// gradient of the reconstruction there read an error ~2 phi/(2 r dtheta), i.e. a radial momentum
  /// kick O(1/(theta dtheta)) as an anti-symmetric dipole at the two seam columns --
  /// a spurious O(1/h) seam drift that, left unhandled, makes a perturbed run diverge (measured
  /// magnitude and provenance: docs/validation/HEADER_PROVENANCE.md).
  BCRec phi_bc() const {
    BCRec b = bcPhi_;
    b.ylo = BCType::Periodic;
    b.yhi = BCType::Periodic;  // theta always periodic
    b.ylo_val = Real(0);
    b.yhi_val = Real(0);
    return b;
  }

  /// dst <- src (component 0, valid cells). NAMED local functor (decoupled from the Cartesian one).
  void copy_comp0(MultiFab& dst, const MultiFab& src) {
    for (int li = 0; li < dst.local_size(); ++li)
      for_each_cell(dst.box(li), detail::PolarSchurCopyComp0Kernel{dst.fab(li).array(),
                                                                   src.fab(li).const_array()});
  }

  VariableSet vars_;
  int c_rho_, c_mx_, c_my_, c_E_;
  Real alpha_;
  PolarGeometry geom_;
  BCRec bcPhi_;
  PolarPrecond precond_;
  BoxArray ba_;
  DistributionMapping dm_;
  MultiFab a_rr_, a_tt_, a_rt_, a_tr_;  ///< coefficients A = I + c rho B^{-1} (reused)
  MultiFab lap_, rhs_;                  ///< Lap_polar phi^n + condensed RHS
  MultiFab fr_, ft_;                    ///< explicit flux F = B^{-1}(mr, mtheta) at the center
  MultiFab one_rr_, one_tt_;            ///< fields at 1 (A=I) for the Lap_polar of the RHS
  MultiFab bz_;                         ///< B_z at the center (internal buffer, 1 ghost)
  MultiFab phi_n_;                      ///< phi^n frozen (final extrapolation)
  MultiFab vr_n_, vt_n_;                ///< v^n (extracted at the start of step)
  MultiFab vr_t_, vt_t_;                ///< v^{n+theta} then v^{n+1}
  PolarKrylovResult last_result_;       ///< diagnostic of the last solve
  Real krylov_tol_ = Real(1e-10);       ///< solve tolerance (historical default, cf. set_krylov)
  int krylov_max_iters_ = 600;          ///< iteration budget (historical default, cf. set_krylov)
};

}  // namespace adc
