#pragma once

/// @file
/// @brief DESCRIPTIVE types of the elliptic stage: EllipticProblem (problem definition) and
///        FieldPostProcess (field derivation convention), plus additive factories.
///
/// Layer: `include/pops/numerics/elliptic/interface`.
/// Role: NAME values and conventions ALREADY coded, without modifying a single floating-point operation
/// (bit-identical structural refactor). EllipticProblem gathers what defines lap(eps phi) = f solved
/// by GeometricMG/PoissonFFTSolver (coefficient eps, physical BCRec, nullspace_const flag).
/// FieldPostProcess names the derivation E = -grad phi and the sign convention (GradSign::Plus for the
/// coupler that stores +grad phi, Minus for a consumer storing -grad phi).
/// Contract: make_elliptic_solver(geom, ba, problem, ...) builds an EllipticSolver from a named
/// EllipticProblem; field_postprocess(phi, out, cx, cy, spec) writes out = (phi if requested,
/// s*grad phi centered) with s = +/-1 per the spec, cx = 1/(2 dx), cy = 1/(2 dy).
///
/// Invariants:
/// - eps is PURELY DESCRIPTIVE: the 5-point stencil does not read it (eps = 1 implicit); eps != 1 raises
///   std::invalid_argument in make_elliptic_solver (forbidden trap, not silently ignored);
/// - field_postprocess reproduces detail::coupler_grad_phi character for character (same order, same
///   factors *cx / *cy): only the sign s is a degree of freedom;
/// - FieldPostprocessKernel is a NAMED functor (and not an POPS_HD lambda) because it is first instantiated
///   from an external TU: an extended lambda would break the device kernel emission under nvcc.

#include <pops/core/foundation/types.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>  // homogeneous(const BCRec&)
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>

#include <stdexcept>
#include <utility>

namespace pops {

// (a) Elliptic problem: coefficient, physical BC, nullspace.
struct EllipticProblem {
  Real eps = 1;                  // descriptive: current stencil state (eps = 1)
  BCRec bc{};                    // physical BC already propagated
  bool nullspace_const = false;  // solution defined up to an additive constant
};

// Homogeneous BCRec associated with the problem: delegates to homogeneous(const BCRec&)
// already in geometric_mg.hpp (multigrid correction with homogeneous BC).
inline BCRec homogeneous_bc(const EllipticProblem& p) {
  return homogeneous(p.bc);
}

// Additive factory: builds an EllipticSolver (GeometricMG, PoissonFFTSolver)
// from a named EllipticProblem. Delegates to the existing (geom, ba, bc,
// active, ...) constructor by extracting problem.bc. eps and nullspace_const are
// descriptive at this stage (eps = 1 already in the stencil; nullspace handled by
// the bottom-solve + demean on the MG side, by the k=0 mode set to zero on the FFT side):
// no existing caller is touched, no numerical value changes. Template
// to avoid an include cycle (this header already includes geometric_mg.hpp).
template <class Solver, class... Args>
inline Solver make_elliptic_solver(const Geometry& geom, const BoxArray& ba,
                                   const EllipticProblem& problem, Args&&... args) {
  // Scientific guard: eps is NOT read by the 5-point stencil
  // (apply_laplacian / poisson_residual / gs_color write lap without a factor,
  // so eps = 1). Setting eps != 1 would suggest solving a variable-coefficient
  // Laplacian with no effect on the values: we forbid this trap instead of
  // ignoring it silently.
  if (problem.eps != Real(1))
    throw std::invalid_argument(
        "EllipticProblem::eps != 1 unsupported (constant-coefficient Laplacian "
        "operator)");
  return Solver(geom, ba, problem.bc, std::forward<Args>(args)...);
}

// (b) Field post-processing: derivation convention E = -grad phi.
struct FieldPostProcess {
  enum class GradSign { Plus, Minus };
  GradSign sign = GradSign::Plus;  // +grad (coupler) or -grad (two_fluid)
  bool store_phi = true;           // phi in component 0 (coupler convention)
};

namespace detail {
// Derives aux = (phi, s grad phi) from phi (field_postprocess). Named functor
// (and not an POPS_HD lambda): same reasons as the elliptic path (#93) -- this kernel is
// first instantiated from an external TU and an extended lambda breaks the device kernel
// emission under nvcc. Body identical to the former lambda -> bit-identical.
struct FieldPostprocessKernel {
  bool store_phi;
  Array4 a;
  ConstArray4 p;
  int gx;
  Real s, cx, cy;
  POPS_HD void operator()(int i, int j) const {
    if (store_phi)
      a(i, j, 0) = p(i, j);
    a(i, j, gx) = s * (p(i + 1, j) - p(i - 1, j)) * cx;
    a(i, j, gx + 1) = s * (p(i, j + 1) - p(i, j - 1)) * cy;
  }
};
}  // namespace detail

// Derives the field from phi per spec. The body reproduces EXACTLY
// detail::coupler_grad_phi (coupler.hpp): same operation order, same
// multiplicative factor *cx / *cy, with the sign s = +1 (Plus) or -1 (Minus) being the only
// degree of freedom. cx, cy remain the centered factors already computed by
// the caller (1/(2 dx), 1/(2 dy)).
inline void field_postprocess(const MultiFab& phi, MultiFab& out, Real cx, Real cy,
                              FieldPostProcess spec) {
  const Real s = (spec.sign == FieldPostProcess::GradSign::Plus) ? Real(1) : Real(-1);
  const bool store_phi = spec.store_phi;
  for (int li = 0; li < out.local_size(); ++li) {
    const ConstArray4 p = phi.fab(li).const_array();
    Array4 a = out.fab(li).array();
    const Box2D v = out.box(li);
    const int gx = store_phi ? 1 : 0;  // component offset if phi is stored
    for_each_cell(v, detail::FieldPostprocessKernel{store_phi, a, p, gx, s, cx, cy});
  }
}

}  // namespace pops
