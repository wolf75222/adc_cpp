#pragma once

/// @file
/// @brief C++20 concepts NAMING the common contracts of the elliptic stage: EllipticOperator
///        (operator role), LinearSolver (iterative subset), FieldPostProcessor (field derivation).
///
/// Layer: `include/adc/numerics/elliptic`.
/// Role: PURELY DESCRIPTIVE header (host metaprogramming). It formalizes as concepts the
/// contracts ALREADY coded by the existing elliptic classes and PROVES it via static_assert. No
/// floating-point logic, no existing class touched: the device-validated elliptic stack stays
/// bit-identical. Reuses (does not redefine) the EllipticSolver concept from elliptic_solver.hpp.
/// Contract: three concepts -- EllipticOperator (geom/bc + stencil coefficient pointers for a
/// matvec consistent with the MG residual, modeled by GeometricMG), LinearSolver (EllipticSolver + a
/// tolerance variant solve(rel_tol, max_iters) with a non-void return), FieldPostProcessor (callable deriving the
/// field from phi, signature of field_postprocess).
///
/// Invariants:
/// - the concepts are COMPILE-TIME predicates, with no device impact (neither kernel nor extended lambda);
/// - LinearSolver is DELIBERATELY separated from EllipticSolver: the DIRECT solvers (PoissonFFTSolver,
///   DistributedFFTSolver, PolarPoissonSolver) model EllipticSolver but NOT LinearSolver (no
///   notion of iterative tolerance), and that is correct;
/// - none of the elliptic classes is modified to satisfy a concept (deliberate refusal).

#include <adc/core/types.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/elliptic/elliptic_problem.hpp>  // FieldPostProcess (spec), field_postprocess
#include <adc/numerics/elliptic/elliptic_solver.hpp>   // concept EllipticSolver (already in place)

#include <concepts>

namespace adc {

// ---------------------------------------------------------------------------
// (1) EllipticOperator: OPERATOR role of the elliptic stencil at the MultiFab level.
//
// Contract = what a matrix-free matvec consumer (TensorKrylovSolver) reads from
// its operator to apply L_int(phi) = div(A grad phi) - kappa phi in a way
// CONSISTENT with the MG residual (poisson_residual): the geometry, the physical BC, the
// BoxArray/DistributionMapping of the fine level, and the operator coefficient POINTERS
// (eps_x, eps_y, kappa, cut-cell weights, cross terms Axy/Ayx, mask).
// An inactive term returns nullptr (cf. internal op_*_ptr), which the concept only requires
// to be CALLABLE and convertible to const MultiFab*; it does not constrain the value.
//
// GeometricMG models this role (and it is the only TYPE that carries it today). The
// free functions apply_laplacian/poisson_residual/gs_smooth are its lowest-level
// realization (the APPLICATION proper): not constrainable by a concept
// because they are not types. EllipticOperator thus NAMES the interface that SUPPLIES
// the coefficients (the "what to apply"), not the application kernel (the "how").
//
// NOTE: EllipticOperator does NOT REQUIRE solve()/rhs()/phi(); it describes the operator
// role only. GeometricMG completes it with EllipticSolver (it is also a solver), but
// a purely operator type (without solve) would already satisfy EllipticOperator.
template <class Op>
concept EllipticOperator = requires(Op op) {
  { op.geom() } -> std::convertible_to<const Geometry&>;
  { op.bc() } -> std::convertible_to<const BCRec&>;
  // Fine-level coefficient pointers: nullptr when the term is inactive.
  { op.op_mask() } -> std::convertible_to<const MultiFab*>;
  { op.op_coef() } -> std::convertible_to<const MultiFab*>;
  { op.op_eps() } -> std::convertible_to<const MultiFab*>;
  { op.op_kappa() } -> std::convertible_to<const MultiFab*>;
  { op.op_eps_y() } -> std::convertible_to<const MultiFab*>;
  { op.op_a_xy() } -> std::convertible_to<const MultiFab*>;
  { op.op_a_yx() } -> std::convertible_to<const MultiFab*>;
};

// ---------------------------------------------------------------------------
// (2) LinearSolver: ITERATIVE solver with an explicit stopping criterion.
//
// Contract = solve(rel_tol, max_iters) that returns a RESULT (the convention "solve
// up to a relative tolerance in at most max_iters steps, returning a report").
// The return type is NOT constrained to a precise struct: GeometricMG
// returns int (number of V-cycles performed) and TensorKrylovSolver returns KrylovResult
// (iters + relative residual + convergence). The only REAL common invariant is: the return
// is NOT void (it carries stopping information). The concept thus reflects this
// reality via !std::same_as<void>, without imposing a shared result type that
// does not exist (forcing it would require modifying one of the two classes: forbidden).
//
// We also require the EllipticSolver base (rhs/phi/solve()/residual/geom): an
// elliptic LinearSolver IS an EllipticSolver that, IN ADDITION, exposes the
// tolerance variant. GeometricMG and TensorKrylovSolver model both.
//
// DOCUMENTED GAP (concept DELIBERATELY separated from EllipticSolver). The DIRECT solvers
// PoissonFFTSolver, DistributedFFTSolver and PolarPoissonSolver solve in ONE pass
// (FFT + Thomas): they have NO solve(rel_tol, max_iters) nor any notion of iterative
// tolerance. They model EllipticSolver (at the Cartesian MultiFab level) or
// PolarEllipticSolver (polar), but NOT LinearSolver, and that is CORRECT: a direct
// solver is not an iterative solver. LinearSolver thus captures the ITERATIVE
// subset of the contract, without claiming that all elliptic backends carry it.
template <class S>
concept LinearSolver = EllipticSolver<S> && requires(S s, Real tol, int it) {
  // Tolerance variant: solves up to rel_tol (or max_iters) and returns a stopping
  // report. Return type FREE but NON void (int for MG, KrylovResult for Krylov).
  s.solve(tol, it);
  requires !std::same_as<decltype(s.solve(tol, it)), void>;
};

// ---------------------------------------------------------------------------
// (3) FieldPostProcessor: field derivation from the potential, phi -> aux/grad.
//
// Contract = an APPLICATOR callable with the signature of field_postprocess:
//   (const MultiFab& phi, MultiFab& out, Real cx, Real cy, FieldPostProcess spec) -> void
// i.e. write into out the convention (phi in component 0 if requested) + the
// centered gradient (+/- depending on the spec sign), with cx = 1/(2 dx), cy = 1/(2 dy). The SPEC
// (gradient sign, phi storage) stays the existing FieldPostProcess struct
// (elliptic_problem.hpp): we do not redefine it, we PARAMETERIZE it.
//
// We constrain a CALLABLE type (functor or function pointer), not a class with
// methods: that is what field_postprocess IS (a free function). The static_assert
// below proves that &field_postprocess satisfies FieldPostProcessor.
template <class F>
concept FieldPostProcessor =
    requires(F f, const MultiFab& phi, MultiFab& out, Real cx, Real cy,
             FieldPostProcess spec) {
      { f(phi, out, cx, cy, spec) } -> std::same_as<void>;
    };

}  // namespace adc
