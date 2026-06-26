#pragma once

/// @file
/// @brief Generic MATRIX-FREE Krylov solver loops -- richardson_solve, cg_solve, bicgstab_solve,
///        gmres_solve -- operating on pops::MultiFab with the operator supplied as a CALLBACK (ApplyFn).
///
/// Layer: `include/pops/numerics/elliptic/linear`.
/// Role: the reusable Krylov core. Where TensorKrylovSolver (krylov_solver.hpp) hardwires the
/// operator and preconditioner to a GeometricMG (the 5-point Laplacian matvec + an MG V-cycle
/// preconditioner on the symmetric part), this header LIFTS the operator/preconditioner to generic
/// std::function callbacks: the caller supplies ANY matrix-free apply `out <- A(in)`. The BiCGStab
/// math and its fixed scratch structure are COPIED from TensorKrylovSolver; only the matvec /
/// preconditioner indirection differs. Richardson and CG are added alongside. These loops are the
/// dynamic solver primitives that a later slice wires into the compiled time-program
/// (ProgramContext / codegen); this header is pure C++ numerics and has no Python / program
/// dependency.
///
/// Convention: each loop solves `A x = b` (A = the @p A callback). @p phi is the unknown IN/OUT:
/// the incoming value is the initial guess (warm start), the outgoing value is the solution. @p rhs
/// is the right-hand side, never modified. Convergence is the RELATIVE L2 residual
/// `||r|| <= rel_tol * ||b||` (||.|| = sqrt of the Krylov inner product, a GLOBAL L2 norm), or
/// @p max_iters reached (returns converged=false, best effort). When ||b|| == 0 the base is taken as 1.
///
/// MULTI-COMPONENT (vector / state) FIELDS: every inner product goes through detail::krylov_dot, which
/// reduces over ALL components (pops::dot_all) when the field has ncomp>1 and over component 0 only
/// (pops::dot, BIT-IDENTICAL) when it is scalar. A vector solve thus drives the residual / search-
/// direction norms and the CG / BiCGStab scalar recurrences over every component -- a component-0-only
/// dot would converge on component 0 alone and leave the others unsolved. The pointwise updates
/// (saxpy / lincomb) already span all components, so no other change is needed; the scalar (ncomp==1)
/// path is unchanged.
///
/// COLLECTIVE / ALL-RANKS CONTRACT: every reduction goes through krylov_dot -> pops::dot / dot_all,
/// which perform a COLLECTIVE all_reduce_sum and MUST run on EVERY rank (including a rank with no box),
/// otherwise MPI deadlocks. Because the residual driving the stopping test is collective, all ranks
/// observe the SAME residual and break at the SAME iteration: the loops NEVER short-circuit a dot() and
/// the trip count is identical on every rank. CG and BiCGStab use the same all_reduce'd dot, so they
/// are also rank-divergence free.
///
/// ALLOC-ONCE / REUSE DISCIPLINE: every scratch MultiFab is allocated ONCE at the START of each call,
/// co-distributed with @p phi (its BoxArray / DistributionMapping / ncomp / nghost), and reused
/// across the iteration loop -- there is NO heap allocation inside the loop. Richardson needs r (1
/// scratch); CG needs r, p, Ap (3); BiCGStab needs r, rhat, p, v, s, t, plus phat/shat when a
/// preconditioner is supplied (mirrors TensorKrylovSolver's fixed footprint).
///
/// Constraints: @p max_iters <= 0 throws std::invalid_argument (a dynamic loop with no iteration
/// budget is a configuration error; spec error 13). std::function lives only at the HOST loop level
/// (never inside a device kernel), so it does not violate the device-clean kernel rules.

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/linear/krylov_result.hpp>  // pops::KrylovResult (shared)

#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

namespace pops {

// KrylovResult is shared via krylov_result.hpp (included above), so this header and the
// GeometricMG-coupled krylov_solver.hpp never carry hand-synchronised copies of the struct.

/// Matrix-free operator callback: `out <- A(in)`. The caller supplies any apply (e.g. fill ghosts +
/// a stencil matvec). @p in is logically read-only (a const ref); a matvec that needs to refresh
/// @p in's ghosts may do so, but must not change its valid cells. @p out is fully overwritten.
using ApplyFn = std::function<void(MultiFab& out, const MultiFab& in)>;

namespace detail {

/// Krylov inner product x.y, COLLECTIVE (all_reduce_sum). For a MULTI-component (vector / state)
/// operator it reduces over ALL components (pops::dot_all) so the residual / search-direction norms and
/// the CG / BiCGStab scalar recurrences cover every component -- a component-0-only dot would converge
/// on component 0 alone and leave the others unsolved. For a single-component field it is exactly
/// pops::dot(x, y) (component 0), so the scalar path stays BIT-IDENTICAL. Must run on every rank.
inline Real krylov_dot(const MultiFab& x, const MultiFab& y) {
  return x.ncomp() > 1 ? dot_all(x, y) : dot(x, y);
}

/// GLOBAL L2 norm sqrt(sum x.x), collective (all_reduce_sum). Full-component for a vector / state
/// field, component-0 (bit-identical) for a scalar field. Identical on all ranks; wraps krylov_dot.
inline Real krylov_l2_norm(const MultiFab& x) {
  return std::sqrt(krylov_dot(x, x));
}

/// Guards a dynamic solver loop: a non-positive iteration budget is a configuration error.
inline void require_max_iter(int max_iters) {
  if (max_iters <= 0)
    throw std::invalid_argument("dynamic solver loops require max_iter");
}

/// Tiny breakdown guard for the BiCGStab scalar recurrences (division by ~0).
inline constexpr Real kKrylovTiny = Real(1e-300);

}  // namespace detail

/// Richardson iteration x <- x + omega (b - A x), solving A x = b. Simplest matrix-free relaxation;
/// converges for any A whose eigenvalues lie in (0, 2/omega) (e.g. an SPD operator with omega below
/// 1/lambda_max). Allocates ONE scratch (r), reused across the loop.
///
/// @param A         matrix-free operator `out <- A(in)`.
/// @param phi       unknown, IN (initial guess) / OUT (solution).
/// @param rhs       right-hand side b (unchanged).
/// @param omega     relaxation factor.
/// @param rel_tol   stop when ||r|| <= rel_tol * ||b||.
/// @param max_iters iteration budget; <= 0 throws std::invalid_argument.
/// @return iterations, relative residual, convergence flag.
inline KrylovResult richardson_solve(const ApplyFn& A, MultiFab& phi, const MultiFab& rhs,
                                     Real omega, Real rel_tol, int max_iters) {
  detail::require_max_iter(max_iters);
  // Scratch allocated ONCE, co-distributed with phi; reused every iteration (no in-loop alloc).
  MultiFab r(phi.box_array(), phi.dmap(), phi.ncomp(), phi.n_grow());

  const Real bnorm = detail::krylov_l2_norm(rhs);        // COLLECTIVE
  const Real norm0 = bnorm > Real(0) ? bnorm : Real(1);  // zero rhs -> absolute base

  A(r, phi);                               // r = A(phi)
  lincomb(r, Real(1), rhs, Real(-1), r);   // r = b - A(phi)
  Real rnorm = detail::krylov_l2_norm(r);  // COLLECTIVE
  KrylovResult res;
  res.rel_residual = rnorm / norm0;
  if (rnorm <= rel_tol * norm0) {  // already converged (warm start)
    res.converged = true;
    return res;
  }

  for (int k = 1; k <= max_iters; ++k) {
    saxpy(phi, omega, r);                   // phi <- phi + omega r
    A(r, phi);                              // r = A(phi)
    lincomb(r, Real(1), rhs, Real(-1), r);  // r = b - A(phi)
    rnorm = detail::krylov_l2_norm(r);      // COLLECTIVE (all ranks, same value)
    res.iters = k;
    res.rel_residual = rnorm / norm0;
    if (rnorm <= rel_tol * norm0) {
      res.converged = true;
      return res;
    }
  }
  return res;  // max_iters reached: best effort (converged=false)
}

/// Conjugate Gradient, solving A x = b for an SPD operator A. Standard preconditioner-free CG.
/// Allocates THREE scratch fields (r, p, Ap), reused across the loop. The dot products (rho, p.Ap)
/// are COLLECTIVE (pops::dot): called on every rank, identical trip count.
///
/// NOTE: A must be symmetric positive-definite. A non-SPD operator may stall or diverge (use
/// bicgstab_solve instead).
///
/// @param A         SPD matrix-free operator `out <- A(in)`.
/// @param phi       unknown, IN (initial guess) / OUT (solution).
/// @param rhs       right-hand side b (unchanged).
/// @param rel_tol   stop when ||r|| <= rel_tol * ||b||.
/// @param max_iters iteration budget; <= 0 throws std::invalid_argument.
/// @return iterations, relative residual, convergence flag.
inline KrylovResult cg_solve(const ApplyFn& A, MultiFab& phi, const MultiFab& rhs, Real rel_tol,
                             int max_iters) {
  detail::require_max_iter(max_iters);
  // Scratch allocated ONCE, co-distributed with phi; reused every iteration. p carries phi's ghost
  // width because A(Ap, p) may read p's ghosts; r and Ap need none but share the layout for clarity.
  MultiFab r(phi.box_array(), phi.dmap(), phi.ncomp(), phi.n_grow());
  MultiFab p(phi.box_array(), phi.dmap(), phi.ncomp(), phi.n_grow());
  MultiFab Ap(phi.box_array(), phi.dmap(), phi.ncomp(), phi.n_grow());

  const Real bnorm = detail::krylov_l2_norm(rhs);  // COLLECTIVE
  const Real norm0 = bnorm > Real(0) ? bnorm : Real(1);

  A(r, phi);                               // r = A(phi)
  lincomb(r, Real(1), rhs, Real(-1), r);   // r = b - A(phi)
  Real rnorm = detail::krylov_l2_norm(r);  // COLLECTIVE
  KrylovResult res;
  res.rel_residual = rnorm / norm0;
  if (rnorm <= rel_tol * norm0) {  // already converged (warm start)
    res.converged = true;
    return res;
  }

  lincomb(p, Real(1), r, Real(0), r);      // p_0 = r_0 (copy via lincomb)
  Real rs_old = detail::krylov_dot(r, r);  // COLLECTIVE: ||r||^2 (all components if ncomp>1)

  for (int k = 1; k <= max_iters; ++k) {
    A(Ap, p);                                    // Ap = A(p)
    const Real pAp = detail::krylov_dot(p, Ap);  // COLLECTIVE (all components if ncomp>1)
    if (std::fabs(pAp) < detail::kKrylovTiny) {  // breakdown (A not SPD / null direction)
      res.iters = k - 1;
      res.rel_residual = rnorm / norm0;
      return res;
    }
    const Real alpha = rs_old / pAp;
    saxpy(phi, alpha, p);                          // phi <- phi + alpha p
    saxpy(r, -alpha, Ap);                          // r   <- r - alpha Ap
    const Real rs_new = detail::krylov_dot(r, r);  // COLLECTIVE: new ||r||^2 (all components)
    rnorm = std::sqrt(rs_new);
    res.iters = k;
    res.rel_residual = rnorm / norm0;
    if (rnorm <= rel_tol * norm0) {
      res.converged = true;
      return res;
    }
    const Real beta = rs_new / rs_old;
    lincomb(p, Real(1), r, beta, p);  // p <- r + beta p
    rs_old = rs_new;
  }
  return res;  // max_iters reached: best effort (converged=false)
}

/// Preconditioned BiCGStab, solving A x = b for a general (possibly non-symmetric) operator A. The
/// algorithm and the fixed scratch footprint are copied from TensorKrylovSolver::solve; only the
/// operator/preconditioner are lifted to callbacks. The preconditioner @p precond is OPTIONAL: an
/// empty std::function means the identity (unpreconditioned BiCGStab), in which case phat = p and
/// shat = s directly and the two phat/shat scratch fields are not allocated.
///
/// All dot products (rho, rhat.v, t.t, t.s) are COLLECTIVE (pops::dot -> all_reduce_sum): called on
/// every rank, identical trip count, no rank-divergent break. Scratch (r, rhat, p, v, s, t, and
/// phat/shat when preconditioned) is allocated ONCE and reused across the loop.
///
/// @param A         matrix-free operator `out <- A(in)`.
/// @param precond   matrix-free preconditioner `out <- M^{-1}(in)`; EMPTY -> identity.
/// @param phi       unknown, IN (initial guess) / OUT (solution).
/// @param rhs       right-hand side b (unchanged).
/// @param rel_tol   stop when ||r|| <= rel_tol * ||b||.
/// @param max_iters iteration budget; <= 0 throws std::invalid_argument.
/// @return iterations, relative residual, convergence flag.
inline KrylovResult bicgstab_solve(const ApplyFn& A, const ApplyFn& precond, MultiFab& phi,
                                   const MultiFab& rhs, Real rel_tol, int max_iters) {
  detail::require_max_iter(max_iters);
  const bool has_precond = static_cast<bool>(precond);

  // Scratch allocated ONCE, co-distributed with phi; reused every iteration (mirror of
  // TensorKrylovSolver's fixed fields). The matvec inputs (p / s, or phat / shat when preconditioned)
  // carry phi's ghost width since A may read their ghosts; the pointwise scratch (r, rhat, v, t)
  // shares the layout. phat/shat are only allocated when a preconditioner is supplied.
  const BoxArray& ba = phi.box_array();
  const DistributionMapping& dm = phi.dmap();
  const int nc = phi.ncomp();
  const int ng = phi.n_grow();
  MultiFab r(ba, dm, nc, ng), rhat(ba, dm, nc, ng), p(ba, dm, nc, ng);
  MultiFab v(ba, dm, nc, ng), s(ba, dm, nc, ng), t(ba, dm, nc, ng);
  MultiFab phat, shat;
  if (has_precond) {
    phat = MultiFab(ba, dm, nc, ng);
    shat = MultiFab(ba, dm, nc, ng);
  }

  const Real bnorm = detail::krylov_l2_norm(rhs);  // COLLECTIVE
  const Real norm0 = bnorm > Real(0) ? bnorm : Real(1);

  A(v, phi);                               // v = A(phi) (scratch reuse for r0)
  lincomb(r, Real(1), rhs, Real(-1), v);   // r = b - A(phi)
  Real rnorm = detail::krylov_l2_norm(r);  // COLLECTIVE
  KrylovResult res;
  res.rel_residual = rnorm / norm0;
  if (rnorm <= rel_tol * norm0) {  // already converged (warm start)
    res.converged = true;
    return res;
  }

  lincomb(rhat, Real(1), r, Real(0), r);  // rhat = frozen r0 (shadow vector)
  p.set_val(Real(0));
  v.set_val(Real(0));
  Real rho_prev = Real(1), alpha = Real(1), omega = Real(1);

  for (int k = 1; k <= max_iters; ++k) {
    const Real rho = detail::krylov_dot(rhat, r);  // COLLECTIVE (all components if ncomp>1)
    if (std::fabs(rho) < detail::kKrylovTiny || std::fabs(omega) < detail::kKrylovTiny) {
      res.iters = k - 1;  // breakdown: best effort
      res.rel_residual = rnorm / norm0;
      return res;
    }
    const Real beta = (rho / rho_prev) * (alpha / omega);
    lincomb(p, Real(1), p, -omega, v);            // p <- p - omega v
    lincomb(p, beta, p, Real(1), r);              // p <- r + beta p
    MultiFab& phat_ref = has_precond ? phat : p;  // identity precond: phat = p
    if (has_precond)
      precond(phat, p);                                   // phat = M^{-1} p
    A(v, phat_ref);                                       // v = A(phat)
    const Real rhat_dot_v = detail::krylov_dot(rhat, v);  // COLLECTIVE (all components if ncomp>1)
    if (std::fabs(rhat_dot_v) < detail::kKrylovTiny) {
      res.iters = k - 1;
      res.rel_residual = rnorm / norm0;
      return res;
    }
    alpha = rho / rhat_dot_v;
    lincomb(s, Real(1), r, -alpha, v);             // s <- r - alpha v
    saxpy(phi, alpha, phat_ref);                   // phi <- phi + alpha phat (partial)
    const Real snorm = detail::krylov_l2_norm(s);  // COLLECTIVE
    if (snorm <= rel_tol * norm0) {                // mid-iteration convergence
      rnorm = snorm;
      res.iters = k;
      res.rel_residual = rnorm / norm0;
      res.converged = true;
      return res;
    }
    MultiFab& shat_ref = has_precond ? shat : s;  // identity precond: shat = s
    if (has_precond)
      precond(shat, s);                        // shat = M^{-1} s
    A(t, shat_ref);                            // t = A(shat)
    const Real tt = detail::krylov_dot(t, t);  // COLLECTIVE (all components if ncomp>1)
    omega = tt > detail::kKrylovTiny ? detail::krylov_dot(t, s) / tt : Real(0);  // COLLECTIVE
    saxpy(phi, omega, shat_ref);        // phi <- phi + omega shat
    lincomb(r, Real(1), s, -omega, t);  // r <- s - omega t
    rnorm = detail::krylov_l2_norm(r);  // COLLECTIVE
    res.iters = k;
    res.rel_residual = rnorm / norm0;
    if (rnorm <= rel_tol * norm0) {
      res.converged = true;
      return res;
    }
    rho_prev = rho;
  }
  return res;  // max_iters reached: best effort (converged=false)
}

/// Left-preconditioned restarted GMRES(m), solving A x = b for a GENERAL (possibly NON-symmetric)
/// operator A. Where CG needs an SPD A and BiCGStab can break down on a strongly non-symmetric one,
/// GMRES minimises the (preconditioned) residual over the growing Krylov subspace and is the robust
/// choice for a non-self-adjoint operator (e.g. an advection-diffusion / condensed-Schur block).
///
/// Math: classic restarted GMRES. The inner cycle builds an Arnoldi basis of M^{-1}A by MODIFIED
/// Gram-Schmidt, accumulating the upper-Hessenberg matrix H; Givens rotations triangularise H
/// incrementally so the least-squares residual ||beta e1 - H y|| is read off the last rotated
/// component WITHOUT a matvec. Every @p restart steps (or at convergence) the iterate is updated from
/// the back-substituted y and the cycle restarts on the fresh residual. The preconditioner @p precond
/// is OPTIONAL: an empty std::function means the identity (unpreconditioned GMRES). Left
/// preconditioning is used (the minimised residual is M^{-1}(b - A x)); with the identity that is the
/// true residual, so the stopping test ||r|| <= rel_tol * ||b|| matches CG / BiCGStab exactly.
///
/// ALLOC-ONCE: the (restart+1) Arnoldi basis MultiFabs, plus w / r / Mb scratch, are allocated ONCE
/// (co-distributed with @p phi) before the restart loop and reused across every cycle -- no MultiFab
/// allocation inside the loop. The small (restart+1) x restart Hessenberg least-squares lives on
/// fixed-size stack arrays (H, the Givens cs/sn, the rotated rhs g, the solution y): no Eigen / LAPACK
/// and no device-side dynamic allocation; @p restart is capped at kGmresRestartMax (50) so the stack
/// footprint stays bounded. All inner products go through detail::krylov_dot (COLLECTIVE all_reduce,
/// multi-component aware), so the loop is rank-divergence free and solves EVERY component of a vector /
/// state field.
///
/// @param A         matrix-free operator `out <- A(in)`.
/// @param precond   matrix-free preconditioner `out <- M^{-1}(in)`; EMPTY -> identity.
/// @param phi       unknown, IN (initial guess) / OUT (solution).
/// @param rhs       right-hand side b (unchanged).
/// @param rel_tol   stop when ||r|| <= rel_tol * ||b||.
/// @param max_iters total matvec budget across restarts; <= 0 throws std::invalid_argument.
/// @param restart   GMRES restart length m (basis size); clamped to [1, kGmresRestartMax]. Default 30.
/// @return iterations, relative residual, convergence flag.
inline KrylovResult gmres_solve(const ApplyFn& A, const ApplyFn& precond, MultiFab& phi,
                                const MultiFab& rhs, Real rel_tol, int max_iters,
                                int restart = 30) {
  detail::require_max_iter(max_iters);
  constexpr int kGmresRestartMax = 50;  // caps the stack Hessenberg system (restart+1 entries)
  const int m = restart < 1 ? 1 : (restart > kGmresRestartMax ? kGmresRestartMax : restart);
  const bool has_precond = static_cast<bool>(precond);

  const BoxArray& ba = phi.box_array();
  const DistributionMapping& dm = phi.dmap();
  const int nc = phi.ncomp();
  const int ng = phi.n_grow();

  // Scratch allocated ONCE, co-distributed with phi; reused across every restart cycle. The Arnoldi
  // basis V holds m+1 vectors; w is the matvec / MGS workspace (carries phi's ghost width since A reads
  // its ghosts); r is the (preconditioned) residual; Mb is M^{-1} b for the relative-residual base.
  std::vector<MultiFab> V;
  V.reserve(static_cast<std::size_t>(m) + 1);
  for (int i = 0; i <= m; ++i)
    V.emplace_back(ba, dm, nc, ng);
  MultiFab w(ba, dm, nc, ng);
  MultiFab r(ba, dm, nc, ng);
  MultiFab Mb;
  if (has_precond)
    Mb = MultiFab(ba, dm, nc, ng);

  // Preconditioned right-hand-side norm: the left-preconditioned residual is M^{-1}(b - A x), so its
  // base is ||M^{-1} b||. With the identity Mb aliases rhs and this is the true ||b||, matching CG.
  if (has_precond)
    precond(Mb, rhs);
  const MultiFab& Mb_ref = has_precond ? Mb : rhs;
  const Real bnorm = detail::krylov_l2_norm(Mb_ref);  // COLLECTIVE
  const Real norm0 = bnorm > Real(0) ? bnorm : Real(1);

  // Fixed-size stack Hessenberg least-squares: H column-major (m columns of m+1 rows), the Givens
  // rotation cosines/sines, the rotated rhs g (beta e1 after rotations), and the back-substituted y.
  // restart <= kGmresRestartMax keeps every array bounded; no heap, no Eigen / LAPACK.
  Real H[kGmresRestartMax + 1][kGmresRestartMax];
  Real cs[kGmresRestartMax];
  Real sn[kGmresRestartMax];
  Real g[kGmresRestartMax + 1];
  Real y[kGmresRestartMax];

  KrylovResult res;
  int total_iters = 0;

  while (total_iters < max_iters) {
    // r = M^{-1}(b - A phi): residual of the current iterate, then left-preconditioned.
    A(w, phi);                              // w = A(phi)
    lincomb(w, Real(1), rhs, Real(-1), w);  // w = b - A(phi)
    if (has_precond)
      precond(r, w);  // r = M^{-1}(b - A phi)
    else
      lincomb(r, Real(1), w, Real(0), w);   // r = b - A phi (copy via lincomb)
    Real beta = detail::krylov_l2_norm(r);  // COLLECTIVE
    res.rel_residual = beta / norm0;
    if (beta <= rel_tol * norm0) {  // converged (also the warm-start early-out)
      res.converged = true;
      return res;
    }

    lincomb(V[0], Real(1) / beta, r, Real(0), r);  // v_0 = r / ||r||
    g[0] = beta;
    for (int i = 1; i <= m; ++i)
      g[i] = Real(0);

    int k = 0;  // Krylov steps taken this cycle
    for (; k < m && total_iters < max_iters; ++k) {
      ++total_iters;
      // Arnoldi: w = M^{-1} A v_k, orthogonalised against v_0..v_k by MODIFIED Gram-Schmidt.
      if (has_precond) {
        A(r, V[k]);     // r = A v_k (reuse r as the unpreconditioned matvec scratch)
        precond(w, r);  // w = M^{-1} A v_k
      } else {
        A(w, V[k]);  // w = A v_k
      }
      for (int i = 0; i <= k; ++i) {
        H[i][k] = detail::krylov_dot(w, V[i]);  // COLLECTIVE (all components if ncomp>1)
        saxpy(w, -H[i][k], V[i]);               // w <- w - H[i][k] v_i
      }
      H[k + 1][k] = detail::krylov_l2_norm(w);  // COLLECTIVE
      if (H[k + 1][k] > detail::kKrylovTiny)
        lincomb(V[k + 1], Real(1) / H[k + 1][k], w, Real(0), w);  // v_{k+1} = w / h
      // else: lucky breakdown (the exact solution lies in the current subspace); v_{k+1} stays unused.

      // Apply the previous Givens rotations to the new Hessenberg column, then a fresh rotation to
      // annihilate H[k+1][k]; carry the rotation through g (the rotated beta e1).
      for (int i = 0; i < k; ++i) {
        const Real t = cs[i] * H[i][k] + sn[i] * H[i + 1][k];
        H[i + 1][k] = -sn[i] * H[i][k] + cs[i] * H[i + 1][k];
        H[i][k] = t;
      }
      const Real denom = std::sqrt(H[k][k] * H[k][k] + H[k + 1][k] * H[k + 1][k]);
      if (denom > detail::kKrylovTiny) {
        cs[k] = H[k][k] / denom;
        sn[k] = H[k + 1][k] / denom;
      } else {  // degenerate column: identity rotation (no division by ~0)
        cs[k] = Real(1);
        sn[k] = Real(0);
      }
      H[k][k] = cs[k] * H[k][k] + sn[k] * H[k + 1][k];
      H[k + 1][k] = Real(0);
      const Real g_next = -sn[k] * g[k];
      g[k] = cs[k] * g[k];
      g[k + 1] = g_next;

      const Real resid = std::fabs(g[k + 1]);  // ||M^{-1}(b - A x)|| WITHOUT a matvec
      res.iters = total_iters;
      res.rel_residual = resid / norm0;
      if (resid <= rel_tol * norm0) {
        ++k;  // include this step in the back-substitution
        break;
      }
    }

    // Back-substitution: solve the k x k upper-triangular R y = g, then phi <- phi + sum y_i v_i.
    for (int i = k - 1; i >= 0; --i) {
      Real sum = g[i];
      for (int j = i + 1; j < k; ++j)
        sum -= H[i][j] * y[j];
      y[i] = std::fabs(H[i][i]) > detail::kKrylovTiny ? sum / H[i][i] : Real(0);
    }
    for (int i = 0; i < k; ++i)
      saxpy(phi, y[i], V[i]);  // phi <- phi + y_i v_i

    // True residual of the updated iterate decides convergence / continuation (the rotated estimate
    // g[k] guides the inner cycle; this recompute keeps the stopping test honest across restarts).
    A(w, phi);                              // w = A(phi)
    lincomb(w, Real(1), rhs, Real(-1), w);  // w = b - A(phi)
    if (has_precond)
      precond(r, w);
    else
      lincomb(r, Real(1), w, Real(0), w);
    const Real rnorm = detail::krylov_l2_norm(r);  // COLLECTIVE
    res.rel_residual = rnorm / norm0;
    if (rnorm <= rel_tol * norm0) {
      res.converged = true;
      return res;
    }
  }
  return res;  // max_iters reached: best effort (converged=false)
}

}  // namespace pops
