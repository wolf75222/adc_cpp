#pragma once

/// @file
/// @brief Generic MATRIX-FREE Krylov solver loops -- richardson_solve, cg_solve, bicgstab_solve --
///        operating on adc::MultiFab with the operator supplied as a CALLBACK (ApplyFn).
///
/// Layer: `include/adc/numerics/elliptic/linear`.
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
/// reduces over ALL components (adc::dot_all) when the field has ncomp>1 and over component 0 only
/// (adc::dot, BIT-IDENTICAL) when it is scalar. A vector solve thus drives the residual / search-
/// direction norms and the CG / BiCGStab scalar recurrences over every component -- a component-0-only
/// dot would converge on component 0 alone and leave the others unsolved. The pointwise updates
/// (saxpy / lincomb) already span all components, so no other change is needed; the scalar (ncomp==1)
/// path is unchanged.
///
/// COLLECTIVE / ALL-RANKS CONTRACT: every reduction goes through krylov_dot -> adc::dot / dot_all,
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

#include <adc/core/foundation/types.hpp>
#include <adc/mesh/storage/mf_arith.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/numerics/elliptic/linear/krylov_result.hpp>  // adc::KrylovResult (shared)

#include <cmath>
#include <functional>
#include <stdexcept>

namespace adc {

// KrylovResult is shared via krylov_result.hpp (included above), so this header and the
// GeometricMG-coupled krylov_solver.hpp never carry hand-synchronised copies of the struct.

/// Matrix-free operator callback: `out <- A(in)`. The caller supplies any apply (e.g. fill ghosts +
/// a stencil matvec). @p in is logically read-only (a const ref); a matvec that needs to refresh
/// @p in's ghosts may do so, but must not change its valid cells. @p out is fully overwritten.
using ApplyFn = std::function<void(MultiFab& out, const MultiFab& in)>;

namespace detail {

/// Krylov inner product x.y, COLLECTIVE (all_reduce_sum). For a MULTI-component (vector / state)
/// operator it reduces over ALL components (adc::dot_all) so the residual / search-direction norms and
/// the CG / BiCGStab scalar recurrences cover every component -- a component-0-only dot would converge
/// on component 0 alone and leave the others unsolved. For a single-component field it is exactly
/// adc::dot(x, y) (component 0), so the scalar path stays BIT-IDENTICAL. Must run on every rank.
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
/// are COLLECTIVE (adc::dot): called on every rank, identical trip count.
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
/// All dot products (rho, rhat.v, t.t, t.s) are COLLECTIVE (adc::dot -> all_reduce_sum): called on
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

}  // namespace adc
