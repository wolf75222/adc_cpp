#pragma once

#include <adc/core/foundation/types.hpp>
#include <adc/core/state/variables.hpp>
#include <adc/coupling/schur/schur_condensation.hpp>  // ElectrostaticLorentzCondensation (builder #124)
#include <adc/coupling/schur/schur_source_kernels.hpp>  // shared geometry-free kernels + validate_krylov_params (#263)
#include <adc/mesh/execution/for_each.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/mf_arith.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/mesh/boundary/physical_bc.hpp>
#include <adc/numerics/elliptic/mg/geometric_mg.hpp>   // operator + preconditioner (#120)
#include <adc/numerics/elliptic/linear/krylov_solver.hpp>  // TensorKrylovSolver (BiCGStab, #122)
#include <adc/numerics/lorentz_eliminator.hpp>      // closed-form B^{-1} (#118)

#include <stdexcept>

/// @file
/// @brief CondensedSchurSourceStepper: Schur-condensed SOURCE STAGE (level 4 of
///        docs/SCHUR_CONDENSATION_DESIGN.md) for the implicit source coupling potential / velocity /
///        Lorentz. This is a STANDALONE SOURCE stage (transport frozen):
///        it does NOT replace the System::step path and does NOT hook into it (facade wiring = PR5). It
///        COMPOSES the bricks already on master:
///          - ElectrostaticLorentzCondensation (schur_condensation.hpp, #124): assembles A_op = I + c
///            rho B^{-1} (eps_x/eps_y diag, a_xy/a_yx cross) plus the condensed RHS;
///          - TensorKrylovSolver (krylov_solver.hpp, #122): matrix-free BiCGStab preconditioned by
///            the GeometricMG V-cycle on the SYMMETRIC part (the operator is NON self-adjoint as soon as
///            B_z != 0);
///          - LorentzEliminator (lorentz_eliminator.hpp, #118): closed-form B^{-1} for the reconstruction.
///
/// SEQUENCE (docs/SCHUR_CONDENSATION_DESIGN.md section 5, level 4), once per source stage:
///   1. ASSEMBLE A_op = I + c rho^n B^{-1} (c = theta^2 dt^2 alpha) and the condensed RHS
///      rhs_schur = -Lap phi^n - theta dt alpha div(rho^n B^{-1} v^n)         [#124]
///   2. SOLVE L_schur(phi^{n+theta}) = rhs_schur, L_schur = -div(A_op grad phi)              [#122]
///   3. RECONSTRUCT v^{n+theta} = B^{-1}(v^n - theta dt grad phi^{n+theta}), mom = rho^n v     [#118]
///   4. ENERGY (if Energy role present): see CHOICE below
///   5. EXTRAPOLATE U^{n+theta} -> U^{n+1}: see CHOICE below
///   6. FILL the ghosts (fill_boundary, MPI halos) and publish phi^{n+1} + grad phi^{n+1}.
///
/// SOLVE SIGN CONVENTION (read before any modification). The builder #124 writes the stage in the
///   form L_schur(phi) = rhs_schur with L_schur(phi) = -div(A_op grad phi) (cf. the header of
///   schur_condensation.hpp). The TensorKrylovSolver, on the other hand, solves L_int(phi) = rhs_kry with the
///   poisson_operator convention L_int = div(A grad phi) - kappa phi (kappa = 0 here). So L_schur = -L_int, and
///   solving L_schur(phi) = rhs_schur amounts to solving L_int(phi) = -rhs_schur. We thus pass
///   rhs_kry = -rhs_schur to the solver (rhs() field negated). Safeguard: c = 0 and B_z = 0 -> A = I, the
///   solve becomes Lap phi^{n+theta} = Lap phi^n -> phi^{n+theta} = phi^n (up to a constant), and the
///   reconstruction degenerates into the explicit electrostatic push v^{n+theta} = v^n - theta dt grad
///   phi^n (case B = 0). The builder already tests this safeguard on the assembly side (test C2).
///
/// EXTRAPOLATION CHOICE (step 5), DOCUMENTED. theta-stage -> n+1 by LINEAR EXTRAPOLATION:
///       U^{n+1} = U^n + (1/theta) (U^{n+theta} - U^n).
///   This is the form cited by docs/SCHUR_CONDENSATION_DESIGN.md (section 5, level 4, step 5). At
///   theta = 1 (pure implicit) the stage lands DIRECTLY at n+1 and the extrapolation is the identity
///   (factor 1/theta = 1). At theta = 1/2 (Crank-Nicolson) the factor is 2: the half-step state is
///   extrapolated to the full step. Applied to phi and v (and thus mom = rho v); rho is FROZEN in the source
///   (all its transport is in the hyperbolic stage), so rho^{n+1} = rho^n.
///
/// ENERGY CHOICE (step 4), DOCUMENTED. The source d_t v = -grad phi + v x Omega does WORK on the
///   fluid via the electrostatic force -grad phi (the Lorentz rotation v x Omega does no work:
///   |B v|^2 = |v|^2 up to a det dilation; in the absence of dissipation the kinetic energy
///   changes only through the field). With rho frozen, the INTERNAL energy is conserved in the source; only
///   the KINETIC energy varies. We thus update the total energy by the kinetic-energy INCREMENT:
///       E^{n+1} = E^n + (1/2) rho^n ( |v^{n+1}|^2 - |v^n|^2 ).
///   Consistent with a total energy E = E_internal + (1/2) rho |v|^2 where only the kinetic term moves
///   under the source. Applied ONLY if the Energy role is present; otherwise the energy is ignored
///   (isothermal model / no energy equation), like the builder #124 which already ignores Energy.
///
/// DEVICE / MPI (docs/GPU_RUNTIME_PORT.md). All kernels of this stage are NAMED FUNCTORS
///   device-clean (no extended lambda first-instantiated cross-TU, nvcc limit #64/#97). The
///   MultiFab buffers (operator A_op, RHS, phi^n, grad phi, v^n) are ALLOCATED ONCE at
///   construction and REUSED on each step() call (the #124 design notes the cost of a
///   reallocation per step). All loops iterate over local_size(): a rank WITHOUT a box
///   (local_size()==0, MPI partition) executes no kernel and never derefs fab(0). The Krylov
///   solve is COLLECTIVE (dot/all_reduce over all ranks, including empty ones): no deadlock.

namespace adc {

namespace detail {

/// Reconstructs v^{n+theta} = B^{-1}(v^n - theta dt grad phi^{n+theta}) and writes mom = rho^n v^{n+theta}
/// into the state. grad phi is the CENTERED difference (same discretization as the divergence of the
/// condensed RHS #124 and as field_postprocess): the implicit relation B v = v^n - theta dt grad phi then holds
/// to the precision of the SOLVE, term by term. rho is read from the state (Density role) and FROZEN.
/// NAMED device-clean functor: captures Array4 handles (POD) plus scalars.
struct SchurReconstructKernel {
  ConstArray4 phi;          ///< phi^{n+theta} (ghosts filled: centered grad reads i+-1, j+-1)
  ConstArray4 vx, vy;       ///< v^n (component 0 of their MultiFab: velocity, NOT momentum)
  ConstArray4 bz;           ///< B_z field at the center
  Array4 st;                ///< fluid state (WRITE mx, my; READ rho)
  Array4 nvx, nvy;          ///< output: v^{n+theta} (component 0) for the energy / the diagnostic
  Real th_dt;               ///< theta * dt (w = th_dt * B_z, and gradient factor)
  Real half_idx, half_idy;  ///< 1/(2 dx), 1/(2 dy) (centered gradient)
  int c_rho, c_mx, c_my;    ///< Density / MomentumX / MomentumY components
  ADC_HD void operator()(int i, int j) const {
    const Real gx = (phi(i + 1, j, 0) - phi(i - 1, j, 0)) * half_idx;  // d_x phi^{n+theta}
    const Real gy = (phi(i, j + 1, 0) - phi(i, j - 1, 0)) * half_idy;  // d_y phi^{n+theta}
    const Real rhsx = vx(i, j, 0) - th_dt * gx;  // (v^n - theta dt grad phi)_x
    const Real rhsy = vy(i, j, 0) - th_dt * gy;
    const LorentzEliminator le(th_dt, Real(1), bz(i, j, 0));  // w = th_dt * B_z
    Real nx, ny;
    le.apply_Binv(rhsx, rhsy, nx, ny);  // v^{n+theta} = B^{-1}(v^n - theta dt grad phi)
    nvx(i, j, 0) = nx;
    nvy(i, j, 0) = ny;
    const Real rho = st(i, j, c_rho);  // rho^n (frozen in the source)
    st(i, j, c_mx) = rho * nx;         // mom^{n+theta} = rho^n v^{n+theta}
    st(i, j, c_my) = rho * ny;
  }
};

// The 5 geometry-free source kernels (extrapolate-scalar/velocity, energy, extract-velocity, copy-Bz)
// live in <adc/coupling/schur/schur_source_kernels.hpp> and are shared with the polar / AMR steppers (#263).
// Only SchurReconstructKernel stays local: it carries the discretization-specific centered gradient.

}  // namespace detail

/// Schur-condensed SOURCE STAGE, STANDALONE (transport frozen), GENERIC over any fluid block that
/// exposes the Density / MomentumX / MomentumY roles (+ optional Energy). Names no scenario:
/// a polar-ring model with ExB drift is just ONE client, a two-species magnetized model would be another.
///
/// LIFE CYCLE: built ONCE on a fixed layout (BoxArray + DistributionMapping + Geometry);
/// allocates all its buffers at construction (GeometricMG operator + preconditioner + A_op,
/// RHS, phi^n, v^n, grad fields, + the Krylov solver kry_ and its MultiFab handle). step() REUSES them
/// without reallocation: kry_.solve() recomputes the ENTIRETY of its state per solve (prepare_solve, r0,
/// descent directions), so the persistent buffer is bit-identical to the old per-call construction of a
/// TensorKrylovSolver. theta/dt may change between calls (step() parameters).
class CondensedSchurSourceStepper {
 public:
  /// @p vars: descriptor of the fluid block; MUST expose Density / MomentumX / MomentumY (Energy
  ///            optional). The contract is validated HERE (host), before any kernel.
  /// @p geom: geometry (cartesian); @p ba: partition; @p bcPhi: BC of the potential phi.
  /// @p alpha: electrostatic coupling constant.
  /// @p n_precond_vcycles: N MG V-cycles per application of the BiCGStab preconditioner (1 or 2).
  CondensedSchurSourceStepper(const VariableSet& vars, const Geometry& geom, const BoxArray& ba,
                              const BCRec& bcPhi, Real alpha, int n_precond_vcycles = 1)
      : CondensedSchurSourceStepper(
            vars, vars.index_of(VariableRole::Density), vars.index_of(VariableRole::MomentumX),
            vars.index_of(VariableRole::MomentumY), vars.index_of(VariableRole::Energy), geom, ba,
            bcPhi, alpha, n_precond_vcycles) {}

  /// Variant with EXPLICIT COMPONENTS (audit 2026-06, wave 2: roles/fields carried in
  /// the ABI). The caller DESIGNATES the components (rho, mx, my[, E]) instead of letting the stepper
  /// resolve the canonical roles -- for a block whose descriptor does not expose Density/
  /// MomentumX/MomentumY (free names, Custom roles) or stores its fields elsewhere. @p c_E < 0 =
  /// no energy. The canonical ctor above DELEGATES here (role resolution unchanged,
  /// bit-identical).
  CondensedSchurSourceStepper(const VariableSet& vars, int c_rho, int c_mx, int c_my, int c_E,
                              const Geometry& geom, const BoxArray& ba, const BCRec& bcPhi,
                              Real alpha, int n_precond_vcycles = 1)
      : vars_(vars),
        c_rho_(c_rho),
        c_mx_(c_mx),
        c_my_(c_my),
        c_E_(c_E),
        alpha_(alpha),
        geom_(geom),
        bcPhi_(bcPhi),
        n_precond_(n_precond_vcycles),
        ba_(ba),
        dm_(ba.size(), n_ranks()),
        // FULL operator (BiCGStab matvec) + SYMMETRIC preconditioner (without cross terms).
        op_(geom, ba, bcPhi),
        precond_(geom, ba, bcPhi),
        // reused buffers (allocated ONCE):
        eps_x_(ba, dm_, 1, 1),
        eps_y_(ba, dm_, 1, 1),
        a_xy_(ba, dm_, 1, 1),
        a_yx_(ba, dm_, 1, 1),
        rhs_schur_(ba, dm_, 1, 0),
        bz_(ba, dm_, 1, 1),
        vx_n_(ba, dm_, 1, 0),
        vy_n_(ba, dm_, 1, 0),
        vx_t_(ba, dm_, 1, 0),
        vy_t_(ba, dm_, 1, 0),
        phi_n_(ba, dm_, 1, 1),
        kry_(op_, precond_, n_precond_) {
    if (c_rho_ < 0 || c_mx_ < 0 || c_my_ < 0)
      throw std::runtime_error(
          "CondensedSchurSourceStepper: the fluid block must expose the Density, MomentumX "
          "and MomentumY roles (VariableSet.roles populated).");
  }

  /// true if the model carries an Energy role (energy update active).
  bool has_energy() const { return c_E_ >= 0; }

  /// Condensed SOURCE STAGE, IN-PLACE on @p state and @p phi.
  ///   @p state: fluid state (rho, mom_x, mom_y [, E]); WRITES mom (+ E); rho FROZEN.
  ///   @p phi: potential; INPUT phi^n (warm start of the solve); OUTPUT phi^{n+1}.
  ///   @p bz_field: B_z field (aux channel), component @p c_bz read at the center. theta/dt: theta-scheme.
  /// No transport: this is the SOURCE stage alone (the (2)-(3) implicit stage of docs/SCHUR_CONDENSATION_DESIGN.md).
  void step(MultiFab& state, MultiFab& phi, const MultiFab& bz_field, int c_bz, Real theta,
            Real dt) {
    const Real th_dt = theta * dt;

    // -1) freeze phi^n (for the final extrapolation; op_'s phi() will be overwritten by the solve).
    copy_comp0(phi_n_, phi);

    // 0) extract v^n = (mx, my)/rho and copy B_z into the internal buffer (1 ghost filled afterwards).
    for (int li = 0; li < state.local_size(); ++li) {
      const ConstArray4 s = state.fab(li).const_array();
      for_each_cell(state.box(li),
                    detail::ExtractVelocityKernel{s, vx_n_.fab(li).array(), vy_n_.fab(li).array(),
                                                  c_rho_, c_mx_, c_my_});
      for_each_cell(bz_.box(li), detail::CopyBzKernel{bz_field.fab(li).const_array(),
                                                      bz_.fab(li).array(), c_bz});
    }
    const BCRec ebc = coeff_bc(bcPhi_);
    fill_ghosts(bz_, geom_.domain, ebc);  // B_z read at (i,j) only here, but 1 ghost = MPI-clean

    // 1) ASSEMBLE A_op = I + c rho B^{-1} + condensed RHS (#124). phi gets its ghosts filled (BC).
    ElectrostaticLorentzCondensation builder(vars_, alpha_, theta, dt);
    builder.assemble_operator(state, bz_, geom_, bcPhi_, eps_x_, eps_y_, a_xy_, a_yx_);
    builder.assemble_rhs(phi, state, bz_, geom_, bcPhi_, rhs_schur_);

    // 2) configure the FULL operator (op_) and the SYMMETRIC preconditioner (precond_), then
    //    SOLVE L_int(phi) = -rhs_schur (cf. the sign convention of the header) by BiCGStab.
    op_.set_epsilon_anisotropic(eps_x_, eps_y_);
    op_.set_cross_terms(a_xy_, a_yx_);
    precond_.set_epsilon_anisotropic(eps_x_, eps_y_);  // symmetric part: cross terms DROPPED.
    // warm start: phi^n -> op_.phi(); rhs = -rhs_schur.
    copy_comp0(op_.phi(), phi);
    negate_into(op_.rhs(), rhs_schur_);
    // kry_ is a persistent MEMBER (its ~10 MultiFab are allocated ONCE at construction, not
    // per call): solve() recomputes the entirety of its state per solve (prepare_solve, r0, directions),
    // so the result is BIT-IDENTICAL to the old per-call construction of a local TensorKrylovSolver here.
    // Tolerances configurable via set_krylov (audit 2026-06); defaults = historical 1e-10/400.
    last_result_ = kry_.solve(krylov_tol_, krylov_max_iters_);
    copy_comp0(phi, op_.phi());  // phi <- phi^{n+theta}

    // 3) RECONSTRUCT v^{n+theta} = B^{-1}(v^n - theta dt grad phi^{n+theta}); mom = rho v.
    device_fence();
    fill_ghosts(phi, geom_.domain, bcPhi_);  // centered grad reads phi(i+-1), phi(j+-1)
    const Real half_idx = Real(1) / (Real(2) * geom_.dx());
    const Real half_idy = Real(1) / (Real(2) * geom_.dy());
    for (int li = 0; li < state.local_size(); ++li) {
      for_each_cell(
          state.box(li),
          detail::SchurReconstructKernel{
              phi.fab(li).const_array(), vx_n_.fab(li).const_array(), vy_n_.fab(li).const_array(),
              bz_.fab(li).const_array(), state.fab(li).array(), vx_t_.fab(li).array(),
              vy_t_.fab(li).array(), th_dt, half_idx, half_idy, c_rho_, c_mx_, c_my_});
    }
    // vx_t_/vy_t_ now carry v^{n+theta}.

    // 5) EXTRAPOLATE phi and v from the theta-stage to the full step: f^{n+1} = f^n + (1/theta)(f^{n+theta}-f^n).
    //    theta = 1 -> identity. phi^n is frozen in phi_n_ (step -1), v^n in vx_n_/vy_n_.
    const Real inv_theta = Real(1) / theta;
    for (int li = 0; li < phi.local_size(); ++li)
      for_each_cell(phi.box(li), detail::SchurExtrapolateScalarKernel{
                                     phi_n_.fab(li).const_array(), phi.fab(li).array(), inv_theta});
    // v + mom: linear extrapolation, recompose mom = rho v^{n+1}. vx_t_/vy_t_ then carry v^{n+1}.
    for (int li = 0; li < state.local_size(); ++li)
      for_each_cell(state.box(li), detail::SchurExtrapolateVelocityKernel{
                                       vx_n_.fab(li).const_array(), vy_n_.fab(li).const_array(),
                                       vx_t_.fab(li).array(), vy_t_.fab(li).array(),
                                       state.fab(li).array(), inv_theta, c_rho_, c_mx_, c_my_});

    // 4) ENERGY (if role present): E^{n+1} = E^n + (1/2) rho (|v^{n+1}|^2 - |v^n|^2).
    if (c_E_ >= 0)
      for (int li = 0; li < state.local_size(); ++li)
        for_each_cell(state.box(li), detail::SchurEnergyKernel{
                                         vx_n_.fab(li).const_array(), vy_n_.fab(li).const_array(),
                                         vx_t_.fab(li).const_array(), vy_t_.fab(li).const_array(),
                                         state.fab(li).array(), c_rho_, c_E_});

    // 6) FILL the ghosts of the state and of the potential (MPI halos / physical BC) before returning control.
    device_fence();
    fill_ghosts(state, geom_.domain, bcU_default());
    fill_ghosts(phi, geom_.domain, bcPhi_);
  }

  /// Diagnostic of the last solve (BiCGStab iterations, relative residual, convergence).
  const KrylovResult& last_solve() const { return last_result_; }

  /// Tolerance / iteration budget of the stage Krylov solve (BiCGStab). DEFAULTS = historical
  /// constants (1e-10, 400), made configurable by the audit 2026-06 (explicit numeric
  /// constants). @throws std::invalid_argument out of domain.
  void set_krylov(Real tol, int max_iters) {
    detail::validate_krylov_params(tol, max_iters, "CondensedSchurSourceStepper::set_krylov");
    krylov_tol_ = tol;
    krylov_max_iters_ = max_iters;
  }

  int density_comp() const { return c_rho_; }
  int momentum_x_comp() const { return c_mx_; }
  int momentum_y_comp() const { return c_my_; }
  int energy_comp() const { return c_E_; }

 private:
  /// Default state BC for the final ghost fill: Foextrap everywhere except periodic
  /// preserved (zero-gradient, consistent with the coeff_bc of the coefficients). The source stage is local in
  /// space; the state BC only affects the ghosts published for the next transport stage.
  BCRec bcU_default() const { return coeff_bc(bcPhi_); }

  /// BC of the coefficient fields: periodic preserved, physical boundary -> zero-gradient (Foextrap).
  static BCRec coeff_bc(const BCRec& bc) {
    auto fo = [](BCType t) { return t == BCType::Periodic ? t : BCType::Foextrap; };
    BCRec b;
    b.xlo = fo(bc.xlo);
    b.xhi = fo(bc.xhi);
    b.ylo = fo(bc.ylo);
    b.yhi = fo(bc.yhi);
    return b;
  }

  /// dst <- src (component 0, valid cells). NAMED functor (reuses CopyComp0Kernel #93).
  void copy_comp0(MultiFab& dst, const MultiFab& src) {
    for (int li = 0; li < dst.local_size(); ++li)
      for_each_cell(dst.box(li),
                    detail::CopyComp0Kernel{dst.fab(li).array(), src.fab(li).const_array()});
  }

  /// dst <- -src (component 0). NAMED functor (reuses NegateKernel from schur_condensation.hpp).
  void negate_into(MultiFab& dst, const MultiFab& src) {
    for (int li = 0; li < dst.local_size(); ++li)
      for_each_cell(dst.box(li),
                    detail::NegateKernel{src.fab(li).const_array(), dst.fab(li).array()});
  }

  VariableSet vars_;               ///< descriptor of the fluid block (roles) passed to builder #124
  int c_rho_, c_mx_, c_my_, c_E_;  ///< role components (c_E_ < 0 if Energy absent)
  Real alpha_;
  Geometry geom_;
  BCRec bcPhi_;
  int n_precond_;
  BoxArray ba_;
  DistributionMapping dm_;
  GeometricMG op_, precond_;              ///< full operator (matvec) + symmetric preconditioner
  MultiFab eps_x_, eps_y_, a_xy_, a_yx_;  ///< coefficients A_op = I + c rho B^{-1} (reused)
  MultiFab rhs_schur_;                    ///< condensed RHS (schur sign; negated before the solve)
  MultiFab bz_;                           ///< B_z at the center (internal buffer, 1 ghost)
  MultiFab vx_n_, vy_n_;                  ///< v^n (extracted at the start of step)
  MultiFab vx_t_, vy_t_;  ///< v^{n+theta} then v^{n+1} (reconstruction + extrapolation)
  MultiFab
      phi_n_;  ///< phi^n frozen (extrapolation); allocated at construction, copied at the start of step
  KrylovResult last_result_;       ///< diagnostic of the last solve
  Real krylov_tol_ = Real(1e-10);  ///< tolerance of the solve (historical default, cf. set_krylov)
  int krylov_max_iters_ = 400;     ///< iteration budget (historical default, cf. set_krylov)
  // PERSISTENT Krylov solver (BiCGStab + MG precond). Allocates its buffers (r/rhat/p/v/s/t/phat/
  // shat + BC offsets) ONCE; step() reuses kry_ without reallocation. MUST be declared AFTER
  // op_/precond_/n_precond_ (which it references): init order (after them) and destruction order (before them)
  // correct. Bit-identical to the old per-call construction (solve() reinitializes its entire state).
  TensorKrylovSolver kry_;
};

}  // namespace adc
