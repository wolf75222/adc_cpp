#pragma once

#include <adc/coupling/schur/condensed_schur_source_stepper.hpp>  // CondensedSchurSourceStepper (#126) + detail kernels
#include <adc/amr/refinement_ratio.hpp>
#include <adc/coupling/schur/schur_condensation.hpp>  // ElectrostaticLorentzCondensation (assemble per level)
#include <adc/numerics/elliptic/composite_fac_poisson.hpp>  // CompositeFacPoisson (composite FAC elliptic solve)
#include <adc/numerics/time/amr_reflux_mf.hpp>   // mf_average_down_mb (fine -> coarse cascade)
#include <adc/numerics/time/amr_subcycling.hpp>  // AmrLevelMP (multi-patch hierarchy)

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

/// @file
/// @brief AmrCondensedSchurSourceStepper: AMR counterpart of the Schur-condensed SOURCE stage
///        (CondensedSchurSourceStepper, #126), carried over a HIERARCHY of levels (AmrLevelMP) rather
///        than over a uniform grid. This is the GLOBAL electrostatic/Lorentz source stage of the
///        "amr-schur" path -- the refined equivalent of the uniform path
///          System(...).add_equation(time=Strang(hyperbolic=Explicit(ssprk3),
///                                                source=CondensedSchur(theta, alpha)))
///        and NOT a local cell-by-cell source (cf. the local IMEX backward_euler_source of the
///        amr-imex path, which is NOT the quantitatively-validated reference source treatment;
///        cf. docs/HOFFART_FIDELITY.md).
///
/// STRATEGY (option A, mirror of the existing AMR Poisson compute_aux/solve_fields). The AMR elliptic
/// solver of this code solves Poisson on the COARSE LEVEL then injects grad phi to the fine levels (the
/// fine patches refine TRANSPORT, not the elliptic solve). The condensed source stage follows the
/// SAME approach: it assembles and solves the condensed operator A_op = I + theta^2 dt^2 alpha rho B^{-1}
/// on the coarse level (by COMPOSING the uniform stage #126, bit-for-bit), then -- for a multi-level
/// hierarchy -- injects grad phi^{n+theta} to the fine levels and reconstructs the velocities there, ending with
/// the fine -> coarse cascade (average_down) which restores the consistency of the covered coarse cells
/// (invariant #169). A spatially constant state (mono-level) degenerates EXACTLY into the uniform stage:
/// this is the parity criterion (Step 2).
///
/// SCOPE (updated Phase 4a, multi-patch fine). The MONO-LEVEL path is complete and bit-identical
/// to the uniform stage #126. The MULTI-LEVEL path is IMPLEMENTED (COMPOSITE condensed source stage:
/// the tensor Schur elliptic is solved by FAC on coarse + fine, velocity reconstruction
/// PER LEVEL then the average_down cascade -- cf. step_multilevel), in the FRAME of 2 levels + 1..N fine
/// patches that are disjoint NON ADJACENT (separated by at least one coarse cell) + coarse replicated mono-block
/// (mono-rank). ONE patch (N == 1) degenerates EXACTLY into Phase 3c (bit-identical). Beyond that (ADJACENT
/// patches / fine-fine join, > 2 levels, MPI, multi-block), step() explicitly REFUSES (clear error)
/// rather than silently applying a partial source: this is Phase 4b.
///
/// LIFE CYCLE / DEVICE / MPI. Built ONCE on the COARSE layout (BoxArray + Geometry + Poisson BC);
/// all buffers of the coarse uniform stage are allocated at construction and reused
/// by step(). The coarse Krylov solve is COLLECTIVE (dot/all_reduce over all ranks, including
/// empty ones) -- like the uniform stage: no deadlock. theta/dt may change between calls.

namespace adc {

/// Schur-condensed SOURCE stage over an AMR hierarchy. GENERIC over any fluid block that exposes
/// the Density / MomentumX / MomentumY roles (+ optional Energy), exactly like the uniform stage.
class AmrCondensedSchurSourceStepper {
 public:
  /// @p vars: descriptor of the fluid block (MUST expose Density / MomentumX / MomentumY; Energy
  ///            optional). Validated HERE (host) by the ctor of the coarse uniform stage.
  /// @p coarse_geom: geometry of the COARSE LEVEL (cartesian).
  /// @p coarse_ba: decomposition of the coarse level (replicated mono-box or distributed multi-box).
  /// @p bcPhi: BC of the potential phi (same as the coarse Poisson).
  /// @p alpha: electrostatic coupling constant.
  /// @p n_precond_vcycles: N MG V-cycles per application of the BiCGStab preconditioner (1 or 2).
  AmrCondensedSchurSourceStepper(const VariableSet& vars, const Geometry& coarse_geom,
                                 const BoxArray& coarse_ba, const BCRec& bcPhi, Real alpha,
                                 int n_precond_vcycles = 1)
      : AmrCondensedSchurSourceStepper(
            vars, vars.index_of(VariableRole::Density), vars.index_of(VariableRole::MomentumX),
            vars.index_of(VariableRole::MomentumY), vars.index_of(VariableRole::Energy),
            coarse_geom, coarse_ba, bcPhi, alpha, n_precond_vcycles) {}

  /// EXPLICIT-COMPONENT variant (audit wave 3, parity with the System steppers): roles
  /// carried by the ABI instead of being resolved canonically. The canonical ctor DELEGATES here.
  AmrCondensedSchurSourceStepper(const VariableSet& vars, int c_rho, int c_mx, int c_my, int c_E,
                                 const Geometry& coarse_geom, const BoxArray& coarse_ba,
                                 const BCRec& bcPhi, Real alpha, int n_precond_vcycles = 1)
      : vars_(vars),
        coarse_geom_(coarse_geom),
        coarse_ba_(coarse_ba),
        bcPhi_(bcPhi),
        alpha_(alpha),
        c_rho_(c_rho),
        c_mx_(c_mx),
        c_my_(c_my),
        c_E_(c_E),
        coarse_(vars, c_rho, c_mx, c_my, c_E, coarse_geom, coarse_ba, bcPhi, alpha,
                n_precond_vcycles) {}

  /// Tolerance / budget of the COARSE stage Krylov solve (delegated to the uniform stage #126;
  /// historical defaults 1e-10 / 400). The COMPOSITE multi-level solve (FAC, Phase 3c) keeps its
  /// own tolerances (Phase 4 follow-up).
  void set_krylov(Real tol, int max_iters) { coarse_.set_krylov(tol, max_iters); }

  /// true if the model carries an Energy role (energy update active in the coarse stage).
  bool has_energy() const { return coarse_.energy_comp() >= 0; }

  /// Condensed SOURCE stage, IN-PLACE on the hierarchy @p levels and the coarse potential @p coarse_phi.
  ///   @p levels: multi-patch hierarchy; levels[0] = COARSE (level 0), levels[k>=1] = FINE
  ///                  (ratio 2). The conservative state of each level is levels[k].U (rho FROZEN,
  ///                  mom/E updated; same convention as the uniform stage).
  ///   @p coarse_phi: potential of the coarse level. INPUT phi^n (warm start of the solve); OUTPUT
  ///                  phi^{n+1}. Same object as the coarse Poisson (mg_.phi() of the coupler) on the facade side.
  ///   @p coarse_bz: B_z field of the coarse level (aux channel), component @p c_bz read at the center.
  ///   @p theta / @p dt: theta-scheme (theta in (0, 1]); dt = effective step (stride factor included
  ///                  by the caller, like s.advance / run_source_stage of the uniform path).
  void step(std::vector<AmrLevelMP>& levels, MultiFab& coarse_phi, const MultiFab& coarse_bz,
            int c_bz, Real theta, Real dt) {
    if (levels.empty())
      return;
    // A fine level EFFECTIVELY POPULATED (>= one patch) signals a multi-level hierarchy. NB: the
    // compiled path (build_amr_compiled) ALWAYS allocates a seed fine level, EMPTY after regrid when
    // no refinement is requested (refine_threshold disabled) -> levels.size() is 2 but the
    // hierarchy is EFFECTIVELY mono-level. So we gate on the NUMBER OF fine PATCHES, not on
    // levels.size(), to avoid refusing the mono-level case with an allocated but empty fine level.
    int n_fine_patches = 0;
    for (std::size_t k = 1; k < levels.size(); ++k)
      n_fine_patches += static_cast<int>(levels[k].U.box_array().size());
    if (n_fine_patches == 0) {
      // MONO-LEVEL (no fine patch): COMPLETE uniform stage on the coarse level (assemble + solve +
      // reconstruction + extrapolation + energy + ghosts), bit-for-bit identical to #126.
      coarse_.step(levels[0].U, coarse_phi, coarse_bz, c_bz, theta, dt);
      return;
    }
    // MULTI-LEVEL (Phase 4a): COMPOSITE condensed source stage -- the fine patches REALLY refine
    // the elliptic (tensor Schur operator solved by FAC on coarse + fine), then velocity
    // reconstruction PER LEVEL and average_down cascade. Phase 4a frame: 2 levels, 1..N fine patches
    // disjoint NON ADJACENT (separated by at least one coarse cell -- guard imposed by the FAC),
    // coarse replicated mono-block, MONO-RANK. Beyond that -> clear error (> 2 levels / MPI / multi-block
    // = Phase 4b). The fine-fine join between adjacent patches is rejected at the FAC ctor (Phase 4b).
    if (levels.size() != 2 || n_ranks() != 1)
      throw std::runtime_error(
          "AmrCondensedSchurSourceStepper: COMPOSITE condensed source stage wired for 2 levels + "
          "NON ADJACENT multi-box fine patches, mono-rank ; > 2 levels / MPI / multi-block = Phase "
          "4b.");
    step_multilevel(levels, coarse_phi, coarse_bz, c_bz, theta, dt);
  }

  /// Diagnostic of the last coarse stage solve (BiCGStab iterations, relative residual, convergence).
  const KrylovResult& last_solve() const { return coarse_.last_solve(); }

  int density_comp() const { return coarse_.density_comp(); }
  int momentum_x_comp() const { return coarse_.momentum_x_comp(); }
  int momentum_y_comp() const { return coarse_.momentum_y_comp(); }
  int energy_comp() const { return coarse_.energy_comp(); }

 private:
  /// COMPOSITE 2-level condensed source stage (1 mono-box fine patch). Assembles the Schur condensed
  /// operator (A = I + c rho B^{-1}, full tensor) + the condensed RHS PER LEVEL (ElectrostaticLorentzCondensation),
  /// solves the COMPOSITE elliptic (CompositeFacPoisson: the fine patch refines the elliptic), reconstructs
  /// the velocity PER LEVEL (v^{n+theta} = B^{-1}(v^n - theta dt grad phi^{n+theta})), extrapolates phi/v to
  /// the full step, updates the energy, then cascades fine -> coarse (average_down, covered cells).
  void step_multilevel(std::vector<AmrLevelMP>& levels, MultiFab& coarse_phi,
                       const MultiFab& coarse_bz, int c_bz, Real theta, Real dt) {
    // COMPLETE fine BoxArray (1..N patches): the FAC is built on this tiling; the patches being separated
    // by at least one coarse cell (FAC ctor guard), each edge is a true C-F join.
    const BoxArray& fine_ba = levels[1].U.box_array();
    ensure_fac(fine_ba);
    const Geometry geom_c = coarse_geom_;
    const Geometry geom_f = coarse_geom_.refine(kAmrRefRatio);
    ElectrostaticLorentzCondensation builder(vars_, alpha_, theta, dt);

    MultiFab& Uc = levels[0].U;
    MultiFab& Uf = levels[1].U;
    const BoxArray bac = Uc.box_array();
    const DistributionMapping dmc = Uc.dmap();
    const BoxArray baf = Uf.box_array();
    const DistributionMapping dmf = Uf.dmap();

    // --- B_z 1-component per level (coarse: extracted from coarse_bz; fine: bilerp of the coarse) ---
    MultiFab bz_c(bac, dmc, 1, 1), bz_f(baf, dmf, 1, 1);
    copy_comp(bz_c, coarse_bz, c_bz);
    device_fence();
    fill_ghosts(bz_c, geom_c.domain, coeff_bc(bcPhi_));
    bilerp_coarse_to_fine(bz_f, bz_c);  // fine B_z from the coarse (B0 uniform -> exact)

    // --- phi^n per level (coarse = coarse_phi; fine = injected aux, levels[1].aux comp 0) ---
    MultiFab phi_n_c(bac, dmc, 1, 1), phi_n_f(baf, dmf, 1, 1);
    copy0(phi_n_c, coarse_phi);
    copy0(phi_n_f, *levels[1].aux);

    // --- v^n per level (before the solve: the reconstruction overwrites mom) ---
    MultiFab vx_n_c(bac, dmc, 1, 0), vy_n_c(bac, dmc, 1, 0);
    MultiFab vx_n_f(baf, dmf, 1, 0), vy_n_f(baf, dmf, 1, 0);
    extract_v(Uc, vx_n_c, vy_n_c);
    extract_v(Uf, vx_n_f, vy_n_f);

    // --- operator + condensed RHS assembly PER LEVEL, into the composite solver fields ---
    // eps_x == eps_y for the Schur (A_xx = A_yy = 1 + c rho/det): we write eps_x into the single eps of the
    // composite and eps_y into a discarded scratch. f_composite = -rhs_schur (sign convention #126).
    MultiFab eps_y_c(bac, dmc, 1, 1), eps_y_f(baf, dmf, 1, 1);
    MultiFab rhs_c(bac, dmc, 1, 0), rhs_f(baf, dmf, 1, 0);
    builder.assemble_operator(Uc, bz_c, geom_c, bcPhi_, fac_->eps_coarse(), eps_y_c,
                              fac_->a_xy_coarse(), fac_->a_yx_coarse());
    builder.assemble_operator(Uf, bz_f, geom_f, bcPhi_, fac_->eps_fine(), eps_y_f,
                              fac_->a_xy_fine(), fac_->a_yx_fine());
    {
      MultiFab pn(bac, dmc, 1, 1);
      copy0(pn, phi_n_c);
      builder.assemble_rhs(pn, Uc, bz_c, geom_c, bcPhi_, rhs_c);
      negate_into(fac_->rhs_coarse(), rhs_c);
    }
    {
      MultiFab pn(baf, dmf, 1, 1);
      copy0(pn, phi_n_f);
      builder.assemble_rhs(pn, Uf, bz_f, geom_f, bcPhi_, rhs_f);
      negate_into(fac_->rhs_fine(), rhs_f);
    }

    // --- COMPOSITE SOLVE: phi^{n+theta} per level (the fine patch refines the elliptic) ---
    fac_->use_variable_coefficient(true);
    fac_->use_cross_terms(true);
    fac_->solve();

    // --- velocity reconstruction + phi/v extrapolation + energy, PER LEVEL ---
    reconstruct_level(Uc, fac_->phi_coarse(), phi_n_c, bz_c, vx_n_c, vy_n_c, geom_c, theta, dt,
                      /*fill_phi_ghosts=*/true);
    reconstruct_level(Uf, fac_->phi_fine(), phi_n_f, bz_f, vx_n_f, vy_n_f, geom_f, theta, dt,
                      /*fill_phi_ghosts=*/false);  // C-F ghosts already set by the composite solve

    // coarse phi^{n+1} (extrapolated in place into fac_->phi_coarse()) -> published into coarse_phi.
    copy0(coarse_phi, fac_->phi_coarse());

    // --- fine -> coarse cascade: the COVERED coarse cells = 2x2 average of the fine cells (#169) ---
    device_fence();
    mf_average_down_mb(Uf, Uc);
    device_fence();
    fill_ghosts(coarse_phi, geom_c.domain, bcPhi_);
  }

  /// Reconstructs v^{n+theta} = B^{-1}(v^n - theta dt grad phi^{n+theta}) (CENTERED grad), writes mom = rho v;
  /// extrapolates phi and v from the theta-stage to the full step (f^{n+1} = f^n + (1/theta)(f^{n+theta}-f^n)); updates
  /// the energy (if the role is present). @p fill_phi_ghosts: fill the physical ghosts of phi (coarse)
  /// ; false for the fine level (the C-F ghosts are already set by the composite solve -- do NOT overwrite them).
  void reconstruct_level(MultiFab& state, MultiFab& phi_nt, const MultiFab& phi_n,
                         const MultiFab& bz, const MultiFab& vx_n, const MultiFab& vy_n,
                         const Geometry& geom, Real theta, Real dt, bool fill_phi_ghosts) {
    const Real th_dt = theta * dt, inv_theta = Real(1) / theta;
    const Real half_idx = Real(1) / (Real(2) * geom.dx());
    const Real half_idy = Real(1) / (Real(2) * geom.dy());
    device_fence();
    if (fill_phi_ghosts)
      fill_ghosts(phi_nt, geom.domain, bcPhi_);
    device_fence();
    MultiFab vx_t(state.box_array(), state.dmap(), 1, 0),
        vy_t(state.box_array(), state.dmap(), 1, 0);
    for (int li = 0; li < state.local_size(); ++li)
      for_each_cell(
          state.box(li),
          detail::SchurReconstructKernel{
              phi_nt.fab(li).const_array(), vx_n.fab(li).const_array(), vy_n.fab(li).const_array(),
              bz.fab(li).const_array(), state.fab(li).array(), vx_t.fab(li).array(),
              vy_t.fab(li).array(), th_dt, half_idx, half_idy, c_rho_, c_mx_, c_my_});
    for (int li = 0; li < phi_nt.local_size(); ++li)
      for_each_cell(phi_nt.box(li),
                    detail::SchurExtrapolateScalarKernel{phi_n.fab(li).const_array(),
                                                         phi_nt.fab(li).array(), inv_theta});
    for (int li = 0; li < state.local_size(); ++li)
      for_each_cell(state.box(li), detail::SchurExtrapolateVelocityKernel{
                                       vx_n.fab(li).const_array(), vy_n.fab(li).const_array(),
                                       vx_t.fab(li).array(), vy_t.fab(li).array(),
                                       state.fab(li).array(), inv_theta, c_rho_, c_mx_, c_my_});
    if (c_E_ >= 0)
      for (int li = 0; li < state.local_size(); ++li)
        for_each_cell(state.box(li), detail::SchurEnergyKernel{
                                         vx_n.fab(li).const_array(), vy_n.fab(li).const_array(),
                                         vx_t.fab(li).const_array(), vy_t.fab(li).const_array(),
                                         state.fab(li).array(), c_rho_, c_E_});
    device_fence();
    fill_ghosts(state, geom.domain, coeff_bc(bcPhi_));
  }

  /// Builds (or rebuilds if the fine tiling changes) the composite elliptic solver on the fine
  /// patches. We compare the current fine BoxArray to the previous one (same boxes AND same order) to avoid an
  /// unnecessary rebuild (the FAC is reused as long as the hierarchy does not change).
  void ensure_fac(const BoxArray& fine_ba) {
    if (fac_ && fac_fine_boxes_ == fine_ba.boxes())
      return;
    fac_ = std::make_unique<CompositeFacPoisson>(coarse_geom_, coarse_ba_, bcPhi_, fine_ba,
                                                 kAmrRefRatio);
    fac_fine_boxes_ = fine_ba.boxes();
  }

  /// BC of the coefficients (eps/B_z) and of the published state: periodic preserved, physical edge zero-gradient.
  static BCRec coeff_bc(const BCRec& b) {
    auto fo = [](BCType t) { return t == BCType::Periodic ? t : BCType::Foextrap; };
    BCRec c;
    c.xlo = fo(b.xlo);
    c.xhi = fo(b.xhi);
    c.ylo = fo(b.ylo);
    c.yhi = fo(b.yhi);
    return c;
  }

  void copy0(MultiFab& dst, const MultiFab& src) {
    device_fence();
    for (int li = 0; li < dst.local_size(); ++li)
      for_each_cell(dst.box(li),
                    detail::CopyComp0Kernel{dst.fab(li).array(), src.fab(li).const_array()});
  }
  void copy_comp(MultiFab& dst, const MultiFab& src, int c) {  // dst comp0 <- src comp c
    device_fence();
    for (int li = 0; li < dst.local_size(); ++li)
      for_each_cell(dst.box(li),
                    detail::CopyBzKernel{src.fab(li).const_array(), dst.fab(li).array(), c});
  }
  void negate_into(MultiFab& dst, const MultiFab& src) {
    device_fence();
    for (int li = 0; li < dst.local_size(); ++li)
      for_each_cell(dst.box(li),
                    detail::NegateKernel{src.fab(li).const_array(), dst.fab(li).array()});
  }
  void extract_v(const MultiFab& state, MultiFab& vx, MultiFab& vy) {
    device_fence();
    for (int li = 0; li < state.local_size(); ++li)
      for_each_cell(state.box(li),
                    detail::ExtractVelocityKernel{state.fab(li).const_array(), vx.fab(li).array(),
                                                  vy.fab(li).array(), c_rho_, c_mx_, c_my_});
  }
  /// Fills valid + ghosts of EACH fine patch @p fine by bilerp of the coarse field @p coarse (B_z,
  /// etc.). Coarse mono-box replicated; we loop over the local fine patches (multi-patch).
  void bilerp_coarse_to_fine(MultiFab& fine, const MultiFab& coarse) {
    device_fence();
    const ConstArray4 C = coarse.fab(0).const_array();
    const int ng = fine.n_grow();
    for (int li = 0; li < fine.local_size(); ++li) {
      Array4 F = fine.fab(li).array();
      const Box2D vb = fine.box(li);
      for (int j = vb.lo[1] - ng; j <= vb.hi[1] + ng; ++j)
        for (int i = vb.lo[0] - ng; i <= vb.hi[0] + ng; ++i)
          F(i, j, 0) = detail::fac_bilerp_coarse(C, i, j, kAmrRefRatio);
    }
  }

  VariableSet vars_;
  Geometry coarse_geom_;
  BoxArray coarse_ba_;
  BCRec bcPhi_;
  Real alpha_;
  int c_rho_, c_mx_, c_my_, c_E_;
  /// Uniform condensed source stage carried over the COARSE LEVEL (MONO-LEVEL path, parity #126).
  CondensedSchurSourceStepper coarse_;
  /// Composite elliptic solver (MULTI-LEVEL path), built lazily on the fine patches.
  std::unique_ptr<CompositeFacPoisson> fac_;
  /// Fine tiling (boxes + order) of the last built FAC: used to detect a hierarchy change.
  std::vector<Box2D> fac_fine_boxes_;
};

}  // namespace adc
