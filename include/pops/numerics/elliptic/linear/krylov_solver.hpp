#pragma once

/// @file
/// @brief TensorKrylovSolver: MATRIX-FREE Krylov solver (BiCGStab) with MG preconditioning, for the
///        FULL-TENSOR elliptic operator L(phi) = -div(A grad phi) + kappa phi, A possibly non-symmetric.
///
/// Layer: `include/pops/numerics/elliptic/linear`.
/// Role: solve the elliptic problem when A is NON-SYMMETRIC (cross term Axy != Ayx, e.g. the rotation
/// B^{-1} arising from a Schur condensation), the case where the GeometricMG V-cycle ALONE
/// (5-point Gauss-Seidel smoother, diagonal block, explicit cross terms) stalls or diverges. BiCGStab
/// (and not GMRES): FIXED memory footprint, no restart. The matvec is apply_laplacian (full operator,
/// strictly matrix-free); the preconditioner is N V-cycles of GeometricMG on the SYMMETRIC PART
/// (diagonal block). Models the EllipticSolver concept.
/// Contract: convention aligned with poisson_operator.hpp -- we solve L_int(phi) = rhs with
/// L_int = div(A grad phi) - kappa phi; the case A = I, kappa = 0 recovers the canonical Laplacian and
/// converges to the SAME solution as GeometricMG (to the tolerance).
///
/// Invariants / constraints:
/// - @p op and @p precond MUST be DISTINCT GeometricMG objects (assert in the constructor): apply_precond
///   OVERWRITES precond.rhs()/phi() at every iteration; confusing them would destroy the solve iterate;
/// - precond carries the SYMMETRIC part (same eps/eps_y/kappa, set_cross_terms NOT called);
/// - DEVICE/MPI: named functors only; the dot products (dot, norm) are COLLECTIVE
///   (all_reduce_sum) and called on ALL ranks, including a rank WITHOUT a box (no deadlock);
/// - ADDITIVE: no existing path (GeometricMG / Poisson) goes through this header (opt-in).

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/elliptic/linear/krylov_result.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>
#include <pops/parallel/comm.hpp>

#include <cassert>
#include <cmath>

namespace pops {

// KrylovResult is defined in krylov_result.hpp (included above), shared with generic_krylov.hpp.

// Matrix-free BiCGStab, preconditioned by N V-cycles of GeometricMG on the SYMMETRIC part.
//
// @p op: GeometricMG carrying the FULL operator (configured via set_cross_terms / set_epsilon*
//               / set_reaction as appropriate). Serves (a) as storage for the fine-level phi/rhs fields
//               (phi()/rhs()/geom()), and (b) as the source of the operator coefficients for the matvec
//               (op_eps(), op_a_xy(), ...). This is the object that models the EllipticSolver concept.
// @p precond: GeometricMG carrying the SYMMETRIC part (same eps/eps_y/kappa, BUT set_cross_terms
//               NOT called -> diagonal block). Its phi()/rhs() serve as a preconditioning buffer.
//               MUST be an object DISTINCT from @p op: apply_precond OVERWRITES precond_.rhs() (copy_into)
//               and precond_.phi() (set_val(0) then V-cycle) at every iteration. If precond_ == op_, these
//               writes would overwrite the iterate phi() and the real rhs() of BiCGStab, destroying the solve.
//               The constructor enforces this via assert(&op != &precond). A SEPARATE object without cross
//               terms is in any case the proper symmetric preconditioner.
class TensorKrylovSolver {
 public:
  // @p n_precond_vcycles: number N of MG V-cycles per preconditioner application (1 or 2).
  TensorKrylovSolver(GeometricMG& op, GeometricMG& precond, int n_precond_vcycles = 1)
      : op_(op),
        precond_(precond),
        n_precond_(n_precond_vcycles),
        ba_(op.box_array()),
        dm_(op.dmap()),
        r_(ba_, dm_, 1, 0),
        rhat_(ba_, dm_, 1, 0),
        p_(ba_, dm_, 1, 0),
        v_(ba_, dm_, 1, 0),
        s_(ba_, dm_, 1, 0),
        t_(ba_, dm_, 1, 0),
        phat_(ba_, dm_, 1, 1),
        shat_(ba_, dm_, 1, 1),
        op_offset_(ba_, dm_, 1, 0),
        bc_offset_(ba_, dm_, 1, 0) {
    // op_ and precond_ MUST be distinct: apply_precond overwrites precond_.rhs()/phi() at every
    // iteration; confusing them with op_ would overwrite the iterate and the right-hand side of the solve (see header).
    assert(&op_ != &precond_ && "TensorKrylovSolver: op and precond must be distinct objects");
  }

  // --- EllipticSolver concept ---
  MultiFab& phi() { return op_.phi(); }
  MultiFab& rhs() { return op_.rhs(); }
  const Geometry& geom() const { return op_.geom(); }
  // current GLOBAL L2 residual ||rhs - L_int(phi)|| (collective). L2 counterpart of GeometricMG's norm_inf.
  Real residual() {
    apply_operator(phi(), r_);                  // r_ = L_int(phi)
    lincomb(r_, Real(1), rhs(), Real(-1), r_);  // r_ = rhs - L_int(phi)
    return l2_norm(r_);
  }
  void solve() { solve(Real(1e-10), 200); }

  // Preconditioned BiCGStab. phi() is the unknown (warm start: incoming value = starting point);
  // rhs() the right-hand side. Returns iterations + relative residual + convergence.
  KrylovResult solve(Real rel_tol, int max_iters) {
    prepare_solve();  // compute (once) the inhomogeneous Dirichlet BC offsets (matvec + precond)
    // r0 = rhs - L_int(phi)  (true INHOMOGENEOUS residual, warm start respected). Here we KEEP the AFFINE
    // operator: it folds the Dirichlet data into the residual, exactly what we want for r0. The
    // equivalent linear system is L_lin x = rhs - c_bc (c_bc = apply_operator(0)); since
    // r0 = rhs - apply_operator(phi) = (rhs - c_bc) - L_lin(phi), r0 is UNCHANGED. The IN-LOOP matvecs,
    // by contrast, act on correction DIRECTIONS and must be LINEAR (apply_operator_lin).
    apply_operator(phi(), v_);                  // v_ = L_int(phi)
    lincomb(r_, Real(1), rhs(), Real(-1), v_);  // r_ = rhs - L_int(phi)
    const Real bnorm = l2_norm(rhs());
    const Real norm0 = bnorm > Real(0) ? bnorm : Real(1);  // relative base (zero rhs -> absolute)
    Real rnorm = l2_norm(r_);
    KrylovResult res;
    res.rel_residual = rnorm / norm0;
    if (rnorm <= rel_tol * norm0) {
      res.converged = true;
      return res;
    }  // already converged

    // rhat = frozen r0 (BiCGStab shadow vector); p, v <- 0.
    copy_into(rhat_, r_);
    p_.set_val(Real(0));
    v_.set_val(Real(0));
    Real rho_prev = Real(1), alpha = Real(1), omega = Real(1);

    for (int k = 1; k <= max_iters; ++k) {
      const Real rho = dot(rhat_, r_);  // COLLECTIVE (all ranks)
      // guard: BiCGStab breakdown (rho or omega ~ 0). We return the current best effort.
      if (std::fabs(rho) < kTiny || std::fabs(omega) < kTiny) {
        res.iters = k - 1;
        res.rel_residual = rnorm / norm0;
        return res;
      }
      const Real beta = (rho / rho_prev) * (alpha / omega);
      // p <- r + beta (p - omega v)
      lincomb(p_, Real(1), p_, -omega, v_);  // p <- p - omega v
      lincomb(p_, beta, p_, Real(1), r_);    // p <- r + beta p
      // phat = M^{-1} p  (N MG V-cycles on the symmetric part)
      apply_precond(p_, phat_);
      apply_operator_lin(phat_, v_);  // v = L_lin(phat) (LINEAR matvec: phat is a direction)
      const Real rhat_dot_v = dot(rhat_, v_);  // COLLECTIVE
      if (std::fabs(rhat_dot_v) < kTiny) {
        res.iters = k - 1;
        res.rel_residual = rnorm / norm0;
        return res;
      }
      alpha = rho / rhat_dot_v;
      // s <- r - alpha v
      lincomb(s_, Real(1), r_, -alpha, v_);
      // phi <- phi + alpha phat   (partial correction; buffer before the test on ||s||)
      saxpy(phi(), alpha, phat_);
      const Real snorm = l2_norm(s_);
      if (snorm <= rel_tol * norm0) {  // convergence at mid-iteration
        rnorm = snorm;
        res.iters = k;
        res.rel_residual = rnorm / norm0;
        res.converged = true;
        return res;
      }
      // shat = M^{-1} s; t = L_lin(shat) (LINEAR matvec: shat is a correction direction)
      apply_precond(s_, shat_);
      apply_operator_lin(shat_, t_);
      const Real tt = dot(t_, t_);  // COLLECTIVE
      omega = tt > kTiny ? dot(t_, s_) / tt : Real(0);
      // phi <- phi + omega shat; r <- s - omega t
      saxpy(phi(), omega, shat_);
      lincomb(r_, Real(1), s_, -omega, t_);
      rnorm = l2_norm(r_);
      res.iters = k;
      res.rel_residual = rnorm / norm0;
      if (rnorm <= rel_tol * norm0) {
        res.converged = true;
        return res;
      }
      rho_prev = rho;
    }
    return res;  // max_iters reached without convergence: best effort (converged=false)
  }

 private:
  static constexpr Real kTiny = Real(1e-300);  // guard against breakdown / division by 0

  // INHOMOGENEOUS MATRIX-FREE matvec: out = L_int(in) = div(A grad in) - kappa in, ghosts of in filled
  // with op_.bc() (the FULL BC). Reuses the coefficients of op_'s FULL operator (same pointers as
  // current_residual). AFFINE in in when bcPhi carries a nonzero Dirichlet value: the boundary ghost
  // equals 2 v - in_interior, so the stencil of boundary cells receives a CONSTANT term c_bc =
  // apply_operator(0). We use it ONLY for the true residual r0 / residual() (where that constant term
  // is exactly the Dirichlet data folded into the residual, which is what we want). The IN-LOOP matvecs
  // (on correction directions) go through apply_operator_lin (LINEAR operator).
  void apply_operator(MultiFab& in, MultiFab& out) {
    device_fence();  // a kernel may have written in; wait before the host read of fill_ghosts
    fill_ghosts(in, op_.geom().domain, op_.bc());
    apply_laplacian(in, op_.geom(), out, op_.op_coef(), op_.op_eps(), op_.op_kappa(),
                    op_.op_eps_y(), op_.op_a_xy(), op_.op_a_yx());
    // conductor cells (mask==0): L_int is 0 there (Dirichlet phi=0), like poisson_residual.
    if (const MultiFab* mk = op_.op_mask())
      mask_zero(out, *mk);
  }

  // LINEAR MATRIX-FREE matvec: out = L_lin(in) = apply_operator(in) - c_bc, with c_bc =
  // apply_operator(0) the inhomogeneous boundary part (constant). BiCGStab applies the matvec to correction
  // DIRECTIONS (phat = M^{-1} p, shat = M^{-1} s), NOT to the iterate; the operator must be linear there,
  // otherwise the constant term c_bc injected at each matvec breaks the BiCGStab relations (residual that
  // oscillates / diverges). We therefore subtract c_bc, just as we subtract d_bc in the preconditioner. Zero
  // Dirichlet BC => has_op_offset_ stays false => apply_operator_lin == apply_operator, bit-identical.
  void apply_operator_lin(MultiFab& in, MultiFab& out) {
    apply_operator(in, out);
    if (has_op_offset_)
      lincomb(out, Real(1), out, Real(-1), op_offset_);  // out <- L_lin(in)
  }

  // preconditioner M^{-1}: out = (N MG V-cycles on the symmetric part) applied to in, with HOMOGENEOUS
  // BC (start out = 0, no warm start: M^{-1} is a LINEAR operator frozen for the BiCGStab iteration).
  // precond_ does NOT carry the cross terms -> diagonal block.
  //
  // The V-cycle of precond_ (precond_.vcycle()) runs at level 0 with its FULL BC bc_. Now if bcPhi
  // carries a NONZERO Dirichlet value (xlo_val/... != 0), fill_physical_bc fills the ghost by
  // ghost = 2 v - interior: starting from phi=0, the first pass injects a FIXED source term ~2 v,
  // INDEPENDENT of in. The raw V-cycle is therefore AFFINE: precond_raw(in) = M^{-1} in + d_bc, with
  // d_bc = precond_raw(0) (constant offset, function of the BC and coefficients alone, NOT of in).
  // This offset injected into phat/shat would make phi += alpha phat + omega shat drift by a spurious
  // term alpha d_bc + omega d_bc at each iteration and would break convergence. We therefore SUBTRACT the offset:
  //   M^{-1} in = precond_raw(in) - precond_raw(0),
  // which is EXACTLY the V-cycle with HOMOGENEOUS BC (the MG recursion is affine, its homogeneous part does
  // not depend on v). Bit-identical to an internal vcycle_homogeneous(), without touching GeometricMG.
  //
  // Same reason and remedy as apply_operator_lin (linear matvec): the operator AND the preconditioner
  // are AFFINE under nonzero Dirichlet BC, and BiCGStab applies them to correction DIRECTIONS;
  // we linearize both by subtracting their respective offset (c_bc for the matvec, d_bc for the precond).
  // Only the true residual r0 / residual() keeps the operator affine (the Dirichlet data is folded in there).
  //
  // d_bc is computed ONCE per solve (prepare_solve). Zero Dirichlet BC (xlo_val=... =0) =>
  // has_bc_offset_ stays false => path STRICTLY UNCHANGED (no subtraction), bit-identical.
  void apply_precond(MultiFab& in, MultiFab& out) {
    precond_raw(in, out);
    if (has_bc_offset_)
      lincomb(out, Real(1), out, Real(-1), bc_offset_);  // out <- M^{-1} in (homogeneous)
  }

  // RAW V-cycle of the preconditioner: out = (N MG V-cycles) applied to in, starting from phi=0. AFFINE in in
  // when bcPhi carries a nonzero Dirichlet value (offset d_bc = precond_raw(0)).
  void precond_raw(MultiFab& in, MultiFab& out) {
    copy_into(precond_.rhs(), in);
    precond_.phi().set_val(Real(0));
    for (int i = 0; i < n_precond_; ++i)
      precond_.vcycle();
    copy_into(out, precond_.phi());
  }

  // Prepare the inhomogeneous BC offsets, ONCE per solve: c_bc = apply_operator(0) for the matvec and
  // d_bc = precond_raw(0) for the preconditioner. A BC without a nonzero Dirichlet value leaves both
  // offsets at 0 (has_*_offset_ = false): apply_operator_lin and apply_precond revert to the historical
  // path (no subtraction), STRICTLY bit-identical. We detect inhomogeneity on op_.bc()
  // (matvec) and precond_.bc() (precond) separately. We use phat_ (1 ghost, required by fill_ghosts
  // of apply_operator) as the NULL input; phat_ is overwritten at the 1st iteration (apply_precond(p_, phat_)).
  void prepare_solve() {
    auto inhomog = [](const BCRec& b) {
      return b.xlo_val != Real(0) || b.xhi_val != Real(0) || b.ylo_val != Real(0) ||
             b.yhi_val != Real(0);
    };
    has_op_offset_ = inhomog(op_.bc());
    has_bc_offset_ = inhomog(precond_.bc());
    if (has_op_offset_) {
      phat_.set_val(Real(0));  // null input (1 ghost for fill_ghosts; phat_ overwritten afterward)
      apply_operator(
          phat_,
          op_offset_);  // op_offset_ <- apply_operator(0) = c_bc (inhomogeneous boundary part)
    }
    if (has_bc_offset_) {
      phat_.set_val(Real(0));
      precond_raw(phat_, bc_offset_);  // bc_offset_ <- precond_raw(0) = d_bc
    }
  }

  // GLOBAL L2 norm sqrt(sum x^2), collective (dot). Host sqrt, identical on all ranks.
  Real l2_norm(const MultiFab& x) { return std::sqrt(dot(x, x)); }

  // copy component 0 of the valid cells (src -> dst), named functor (device-clean).
  void copy_into(MultiFab& dst, const MultiFab& src) {
    for (int li = 0; li < dst.local_size(); ++li) {
      Array4 d = dst.fab(li).array();
      const ConstArray4 s = src.fab(li).const_array();
      for_each_cell(dst.box(li), detail::CopyComp0Kernel{d, s});
    }
  }

  // freeze out=0 on the conductor cells (mask==0), named functor (reuses ZeroConductorKernel).
  void mask_zero(MultiFab& out, const MultiFab& mask) {
    for (int li = 0; li < out.local_size(); ++li) {
      Array4 o = out.fab(li).array();
      const ConstArray4 m = mask.fab(li).const_array();
      for_each_cell(out.box(li), detail::ZeroConductorKernel{o, m});
    }
  }

  GeometricMG& op_;
  GeometricMG& precond_;
  int n_precond_;
  BoxArray ba_;
  DistributionMapping dm_;
  MultiFab r_, rhat_, p_, v_, s_, t_;  // 0 ghost: pointwise ops
  MultiFab phat_, shat_;               // 1 ghost: inputs of apply_operator (fill_ghosts)
  MultiFab op_offset_;  // c_bc = apply_operator(0): inhomogeneous boundary part of the matvec
  MultiFab bc_offset_;  // d_bc = precond_raw(0): Dirichlet BC offset of the precond
  bool has_op_offset_ = false;  // true if the operator BC carries a nonzero Dirichlet value
  bool has_bc_offset_ = false;  // true if the preconditioner BC carries a nonzero value
};

}  // namespace pops
