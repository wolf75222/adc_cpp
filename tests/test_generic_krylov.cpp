// Generic MATRIX-FREE Krylov layer (generic_krylov.hpp): three solver loops -- Richardson, CG,
// BiCGStab -- that take the operator as a CALLBACK (ApplyFn), so any matrix-free apply can be
// plugged in. This is the reusable core that a later slice wires into the compiled time-program
// (ProgramContext / codegen); this test is PURE C++ and validates the loops in isolation.
//
// OPERATOR: an SPD Helmholtz operator A = I - alpha*Lap (alpha = 0.1), supplied as an ApplyFn that
//   fills the ghosts of `in` (periodic), applies the SHARED discrete 5-point Laplacian
//   (apply_laplacian, all optional coefficients null -> bit-identical bare Laplacian), then forms
//   out = in - alpha*Lap(in). The bare periodic Laplacian has a constant null space that breaks CG;
//   I - alpha*Lap is symmetric POSITIVE-DEFINITE (its spectrum is 1 + alpha*lambda, lambda >= 0 the
//   non-negative eigenvalues of -Lap), so CG is well-defined and the loop is well-conditioned.
//
// MANUFACTURED SOLUTION: phi_exact(x,y) = sin(2 pi x) sin(2 pi y) (periodic on the unit square). We
//   do NOT use the continuous eigenvalue: to test the SOLVER and not the discretization, we form
//   rhs = A(phi_exact) by APPLYING the same discrete operator to the sampled phi_exact. Then we
//   solve A x = rhs from x = 0 and require max|x - phi_exact| < 1e-8 (tight: same discrete A).
//
// We validate the four loops:
//   - cg_solve        (SPD operator),
//   - bicgstab_solve  (identity preconditioner -- empty ApplyFn),
//   - richardson_solve(omega = 1/(1 + alpha*8*pi^2) ~ 1/spectral-max, more iters allowed),
//   - gmres_solve     (restarted GMRES(m), identity preconditioner): on the SPD operator it matches
//                      CG, and on a NON-symmetric operator (Helmholtz + a one-sided advection term,
//                      where CG STAGNATES) it converges to phi_exact. The non-symmetric case is the
//                      gmres-specific guard -- cg_solve on the same operator must NOT recover phi_exact.
// Each must converge (converged == true, iters > 1, small residual) and recover phi_exact. We also
// assert that max_iters = 0 throws std::invalid_argument (spec error 13).
//
// SERIAL test: no MPI (single box, DistributionMapping(1, 1)); the dot products in the loops are
// nonetheless COLLECTIVE (pops::dot -> all_reduce_sum), the identity in serial.

#include <pops/numerics/elliptic/linear/generic_krylov.hpp>
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>  // apply_laplacian (shared 5-point matvec)
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>  // fill_ghosts (periodic ghost exchange)

#include "test_harness.hpp"  // pops::test::Checker + kPi

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace pops;
using pops::test::kPi;

namespace {

constexpr int kN = 32;        // 32 x 32 periodic grid
constexpr Real kAlpha = 0.1;  // Helmholtz coefficient: A = I - alpha*Lap (SPD, well-conditioned)

// Named functor (device-clean): out(i,j) = in(i,j) - alpha*lap(i,j). Same recipe as the elliptic
// kernels (#93): a plain lambda is fine on the host Serial path here, but a named functor keeps the
// kernel emission robust on every backend.
struct HelmholtzCombineKernel {
  Array4 outv;
  ConstArray4 inv, lapv;
  Real alpha;
  POPS_HD void operator()(int i, int j) const { outv(i, j) = inv(i, j) - alpha * lapv(i, j); }
};

// Non-symmetric combine: out = in - alpha*Lap(in) + beta * (in(i) - in(i-1)) / h, a FIRST-order
// upwind x-derivative added to the SPD Helmholtz operator. The one-sided difference is NOT
// self-adjoint (its transpose is the opposite-sided difference), so the whole operator is
// non-symmetric -- CG stagnates on it while GMRES (and BiCGStab) converge. `in`'s ghosts are
// periodic (filled before the matvec), so in(i-1) wraps at the low edge.
struct AdvectionHelmholtzKernel {
  Array4 outv;
  ConstArray4 inv, lapv;
  Real alpha, beta, inv_h;
  POPS_HD void operator()(int i, int j) const {
    outv(i, j) = inv(i, j) - alpha * lapv(i, j) + beta * (inv(i, j) - inv(i - 1, j)) * inv_h;
  }
};

// phi_exact(x,y) = sum of several periodic sine modes. A SINGLE mode is an eigenvector of the
// discrete Laplacian, so CG/BiCGStab would converge in ONE step (masking the iteration loop); a SUM
// of modes with DISTINCT eigenvalues forces several Krylov steps and a genuine Richardson sweep,
// which is what we want to exercise. All modes are periodic on the unit square. POPS_HD so it is
// device-callable from the init kernel (else nvcc returns garbage on device, like phi_exact in
// test_krylov_solver).
POPS_HD Real phi_exact(Real x, Real y) {
  return std::sin(2 * kPi * x) * std::sin(2 * kPi * y) +
         Real(0.5) * std::sin(4 * kPi * x) * std::sin(2 * kPi * y) +
         Real(0.3) * std::cos(2 * kPi * x) * std::cos(6 * kPi * y) +
         Real(0.2) * std::sin(6 * kPi * x) * std::cos(4 * kPi * y);
}

struct SampleExactKernel {
  Array4 af;
  Geometry geom;
  POPS_HD void operator()(int i, int j) const {
    af(i, j) = phi_exact(geom.x_cell(i), geom.y_cell(j));
  }
};

// max|a - b| over the valid cells, reduced over all ranks (serial: identity). Host loop (a tiny
// grid; this is a correctness check, not a hot path).
Real max_abs_diff(const MultiFab& a, const MultiFab& b) {
  Real d = 0;
  for (int li = 0; li < a.local_size(); ++li) {
    const ConstArray4 pa = a.fab(li).const_array();
    const ConstArray4 pb = b.fab(li).const_array();
    const Box2D bx = a.box(li);
    for (int j = bx.lo[1]; j <= bx.hi[1]; ++j)
      for (int i = bx.lo[0]; i <= bx.hi[0]; ++i)
        d = std::fmax(d, std::fabs(pa(i, j) - pb(i, j)));
  }
  return static_cast<Real>(all_reduce_max(static_cast<double>(d)));
}

}  // namespace

int main() {
  pops::test::Checker chk(pops::test::Checker::Style::Terse);

  Box2D dom = Box2D::from_extents(kN, kN);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, 1);
  BCRec bc;  // all faces default to Periodic -> fill_ghosts wraps in x and y

  // SPD Helmholtz operator A(in) = in - alpha*Lap(in), matrix-free: fill periodic ghosts of `in`,
  // apply the shared discrete 5-point Laplacian into a scratch, then combine. `in` needs >= 1 ghost
  // for the stencil; `tmp` here is captured once and reused across every matvec (no per-call alloc).
  MultiFab lap_tmp(ba, dm, 1, 0);
  ApplyFn A = [&](MultiFab& out, const MultiFab& in) {
    // in is const at the API level, but fill_ghosts / apply_laplacian take a mutable MultiFab&
    // (they only WRITE the ghosts of `in`, never the valid cells). Casting away const is the same
    // contract the solver loops rely on; the valid data of `in` is unchanged.
    MultiFab& in_mut = const_cast<MultiFab&>(in);
    fill_ghosts(in_mut, geom.domain, bc);
    apply_laplacian(in_mut, geom,
                    lap_tmp);  // lap_tmp = Lap(in) (all coeffs null -> bare Laplacian)
    for (int li = 0; li < out.local_size(); ++li) {
      Array4 ov = out.fab(li).array();
      const ConstArray4 iv = in.fab(li).const_array();
      const ConstArray4 lv = lap_tmp.fab(li).const_array();
      for_each_cell(out.box(li), HelmholtzCombineKernel{ov, iv, lv, kAlpha});
    }
  };

  // Manufactured solution phi_exact and the discretization-exact rhs = A(phi_exact).
  MultiFab phi_exact_mf(ba, dm, 1, 1);  // >= 1 ghost (input of A)
  for (int li = 0; li < phi_exact_mf.local_size(); ++li) {
    Array4 af = phi_exact_mf.fab(li).array();
    for_each_cell(phi_exact_mf.box(li), SampleExactKernel{af, geom});
  }
  MultiFab rhs(ba, dm, 1, 0);
  A(rhs, phi_exact_mf);  // rhs <- A(phi_exact): discrete RHS (tests the SOLVER, not the scheme)

  // NON-symmetric operator A_ns(in) = in - alpha*Lap(in) + beta * upwind dx(in): the Helmholtz part is
  // SPD, the one-sided advection term breaks symmetry. beta is large enough that the operator is
  // strongly non-self-adjoint (CG stagnates), but the spectrum stays in the right half-plane so GMRES
  // converges. Reuses lap_tmp; `in`'s periodic ghosts feed the upwind in(i-1).
  constexpr Real kBeta =
      2.0;  // advection strength (CFL-irrelevant: this is a linear solve, not a step)
  const Real inv_h = Real(1) / geom.dx();
  ApplyFn A_ns = [&](MultiFab& out, const MultiFab& in) {
    MultiFab& in_mut = const_cast<MultiFab&>(in);
    fill_ghosts(in_mut, geom.domain, bc);
    apply_laplacian(in_mut, geom, lap_tmp);
    for (int li = 0; li < out.local_size(); ++li) {
      Array4 ov = out.fab(li).array();
      const ConstArray4 iv = in.fab(li).const_array();
      const ConstArray4 lv = lap_tmp.fab(li).const_array();
      for_each_cell(out.box(li), AdvectionHelmholtzKernel{ov, iv, lv, kAlpha, kBeta, inv_h});
    }
  };
  MultiFab rhs_ns(ba, dm, 1, 0);
  A_ns(rhs_ns,
       phi_exact_mf);  // rhs_ns <- A_ns(phi_exact): discrete RHS for the non-symmetric solve

  const Real rel_tol = 1e-12;
  const Real recover_tol = 1e-8;

  // --- CG (SPD operator) ---
  {
    MultiFab x(ba, dm, 1, 1);
    x.set_val(0.0);
    const KrylovResult r = cg_solve(A, x, rhs, rel_tol, 500);
    const Real err = max_abs_diff(x, phi_exact_mf);
    std::printf("CG        : %s in %d iters (rel=%.2e) | max|x - exact| = %.3e\n",
                r.converged ? "CONVERGED" : "FAILED", r.iters, r.rel_residual, err);
    chk(r.converged, "cg_converged");
    chk(r.iters > 1, "cg_iters_gt_1");
    chk(r.rel_residual <= rel_tol * 10, "cg_residual_small");
    chk(err < recover_tol, "cg_recovers_exact");
  }

  // --- BiCGStab (identity preconditioner = empty ApplyFn) ---
  {
    MultiFab x(ba, dm, 1, 1);
    x.set_val(0.0);
    const KrylovResult r = bicgstab_solve(A, ApplyFn{}, x, rhs, rel_tol, 500);
    const Real err = max_abs_diff(x, phi_exact_mf);
    std::printf("BiCGStab  : %s in %d iters (rel=%.2e) | max|x - exact| = %.3e\n",
                r.converged ? "CONVERGED" : "FAILED", r.iters, r.rel_residual, err);
    chk(r.converged, "bicgstab_converged");
    chk(r.iters > 1, "bicgstab_iters_gt_1");
    chk(r.rel_residual <= rel_tol * 10, "bicgstab_residual_small");
    chk(err < recover_tol, "bicgstab_recovers_exact");
  }

  // --- Richardson (omega ~ 1/spectral-max; needs many more iters) ---
  {
    MultiFab x(ba, dm, 1, 1);
    x.set_val(0.0);
    // Richardson is stable only for omega < 2/lambda_max(A). The DISCRETE -Lap has largest
    // eigenvalue 8/h^2 (NOT the continuous 8*pi^2: the continuous value would over-relax by the grid
    // factor and DIVERGE), so lambda_max(A) = 1 + alpha*8/h^2. omega = 1/lambda_max under-relaxes
    // safely for every mode; convergence is slow (the low modes have rate ~1 - omega) so we grant a
    // large iteration budget.
    const Real h = geom.dx();  // = geom.dy() on the unit square (uniform)
    const Real lambda_max = Real(1) + kAlpha * Real(8) / (h * h);
    const Real omega = Real(1) / lambda_max;
    const KrylovResult r = richardson_solve(A, x, rhs, omega, rel_tol, 200000);
    const Real err = max_abs_diff(x, phi_exact_mf);
    std::printf("Richardson: %s in %d iters (rel=%.2e, omega=%.4f) | max|x - exact| = %.3e\n",
                r.converged ? "CONVERGED" : "FAILED", r.iters, r.rel_residual, omega, err);
    chk(r.converged, "richardson_converged");
    chk(r.iters > 1, "richardson_iters_gt_1");
    chk(r.rel_residual <= rel_tol * 10, "richardson_residual_small");
    chk(err < recover_tol, "richardson_recovers_exact");
  }

  // --- GMRES on the SPD operator (identity preconditioner): must recover phi_exact like CG ---
  {
    MultiFab x(ba, dm, 1, 1);
    x.set_val(0.0);
    const KrylovResult r = gmres_solve(A, ApplyFn{}, x, rhs, rel_tol, 500, 30);
    const Real err = max_abs_diff(x, phi_exact_mf);
    std::printf("GMRES(SPD): %s in %d iters (rel=%.2e) | max|x - exact| = %.3e\n",
                r.converged ? "CONVERGED" : "FAILED", r.iters, r.rel_residual, err);
    chk(r.converged, "gmres_spd_converged");
    chk(r.iters > 1, "gmres_spd_iters_gt_1");
    chk(r.rel_residual <= rel_tol * 10, "gmres_spd_residual_small");
    chk(err < recover_tol, "gmres_spd_recovers_exact");
  }

  // --- GMRES on the NON-symmetric operator: the gmres-specific guard. CG STAGNATES on A_ns (it is not
  //     self-adjoint), so we first confirm CG fails to recover phi_exact, then GMRES does. ---
  {
    MultiFab x_cg(ba, dm, 1, 1);
    x_cg.set_val(0.0);
    const KrylovResult rc = cg_solve(A_ns, x_cg, rhs_ns, rel_tol, 500);
    const Real err_cg = max_abs_diff(x_cg, phi_exact_mf);
    std::printf(
        "CG(nonsym): %s in %d iters (rel=%.2e) | max|x - exact| = %.3e (expected to NOT "
        "recover)\n",
        rc.converged ? "CONVERGED" : "FAILED", rc.iters, rc.rel_residual, err_cg);
    chk(!(rc.converged && err_cg < recover_tol), "cg_stagnates_on_nonsymmetric");

    MultiFab x(ba, dm, 1, 1);
    x.set_val(0.0);
    const KrylovResult r = gmres_solve(A_ns, ApplyFn{}, x, rhs_ns, rel_tol, 500, 30);
    const Real err = max_abs_diff(x, phi_exact_mf);
    std::printf("GMRES(nsy): %s in %d iters (rel=%.2e) | max|x - exact| = %.3e\n",
                r.converged ? "CONVERGED" : "FAILED", r.iters, r.rel_residual, err);
    chk(r.converged, "gmres_nonsym_converged");
    chk(r.iters > 1, "gmres_nonsym_iters_gt_1");
    chk(r.rel_residual <= rel_tol * 10, "gmres_nonsym_residual_small");
    chk(err < recover_tol, "gmres_nonsym_recovers_exact");
  }

  // --- max_iters <= 0 must throw (spec error 13) ---
  {
    MultiFab x(ba, dm, 1, 1);
    x.set_val(0.0);
    bool threw_cg = false, threw_rich = false, threw_bicg = false, threw_gmres = false;
    try {
      cg_solve(A, x, rhs, rel_tol, 0);
    } catch (const std::invalid_argument&) {
      threw_cg = true;
    }
    try {
      richardson_solve(A, x, rhs, Real(0.1), rel_tol, 0);
    } catch (const std::invalid_argument&) {
      threw_rich = true;
    }
    try {
      bicgstab_solve(A, ApplyFn{}, x, rhs, rel_tol, 0);
    } catch (const std::invalid_argument&) {
      threw_bicg = true;
    }
    try {
      gmres_solve(A, ApplyFn{}, x, rhs, rel_tol, 0);
    } catch (const std::invalid_argument&) {
      threw_gmres = true;
    }
    chk(threw_cg, "cg_max_iters_0_throws");
    chk(threw_rich, "richardson_max_iters_0_throws");
    chk(threw_bicg, "bicgstab_max_iters_0_throws");
    chk(threw_gmres, "gmres_max_iters_0_throws");
  }

  if (chk.fails() == 0)
    std::printf("OK test_generic_krylov\n");
  return chk.failed();
}
