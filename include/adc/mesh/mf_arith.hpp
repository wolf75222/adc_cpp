/// @file
/// @brief MultiFab arithmetic (saxpy, lincomb, norm_inf, dot) over VALID cells.
///
/// Building blocks for integrator stages and Krylov solvers. Assumes IDENTICAL layouts
/// (same BoxArray, same DistributionMapping). Pointwise operations -> ALIASING is safe
/// (x or y == z allowed). norm_inf / dot go through the reducer seam (true Kokkos reduction).
/// dot performs a COLLECTIVE all_reduce: it MUST be called on EVERY rank (including a rank with no
/// box) under MPI, otherwise deadlock. FP NOTE: dot/sum are re-associated per tile (Kokkos::Sum,
/// deterministic/idempotent but not bit-identical to a lexicographic sum, for all Kokkos
/// spaces); norm_inf is exact everywhere. The kernels are device-clean NAMED FUNCTORS (nvcc cross-TU).

#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/parallel/comm.hpp>  // all_reduce_sum: COLLECTIVE dot product (Krylov under MPI)

#include <algorithm>

namespace adc {

namespace detail {
// NAMED FUNCTORS (not ADC_HD lambdas) for the MultiFab arithmetic kernels. Same recipe as
// the block path (#64): these operations are first-instantiated from the MG V-cycle, itself pulled
// from an external TU (native harness/loader); an extended lambda at this spot makes nvcc stumble on
// device kernel emission (null kernel-stub -> Cuda segfault in -O Release without -g, #93). Body
// strictly identical to the old lambdas -> bit-identical on CPU and device.
struct SaxpyKernel {
  Array4 Y;
  ConstArray4 X;
  Real a;
  int c;
  ADC_HD void operator()(int i, int j) const { Y(i, j, c) += a * X(i, j, c); }
};

struct LincombKernel {
  Array4 Z;
  ConstArray4 X, Y;
  Real a, b;
  int c;
  ADC_HD void operator()(int i, int j) const { Z(i, j, c) = a * X(i, j, c) + b * Y(i, j, c); }
};

// Reducer |f(i,j,comp)| -> max, passed DIRECTLY to reduce_max_cell (no wrapping extended
// lambda, unlike for_each_cell_reduce_max). This is the device-clean path documented
// in for_each.hpp. Reducer signature (i, j, Real& acc); same Kokkos::Max / same sequential host
// loop -> bit-identical to the old norm_inf (max and fabs without rounding).
struct NormInfKernel {
  ConstArray4 a;
  int comp;
  ADC_HD void operator()(int i, int j, Real& acc) const {
    const Real v = a(i, j, comp);
    const Real av = v < 0 ? -v : v;
    if (av > acc)
      acc = av;
  }
};

// Reducer x(i,j,comp) * y(i,j,comp) -> sum, passed DIRECTLY to reduce_sum_cell (no wrapping
// extended lambda). Device-clean NAMED functor (same recipe as NormInfKernel) for the Krylov
// solver dot product, pulled from an external TU. Reducer signature (i, j, Real& acc).
struct DotKernel {
  ConstArray4 x, y;
  int comp;
  ADC_HD void operator()(int i, int j, Real& acc) const { acc += x(i, j, comp) * y(i, j, comp); }
};
}  // namespace detail

/// y <- y + a x over ALL components of the valid cells. Identical layouts required.
inline void saxpy(MultiFab& y, Real a, const MultiFab& x) {
  const int nc = y.ncomp();
  for (int li = 0; li < y.local_size(); ++li) {
    Array4 Y = y.fab(li).array();
    const ConstArray4 X = x.fab(li).const_array();
    const Box2D b = y.fab(li).box();
    for (int c = 0; c < nc; ++c)
      for_each_cell(b, detail::SaxpyKernel{Y, X, a, c});
  }
}

// Infinity norm over the valid cells of one component. Each local fab is
// reduced by for_each_cell_reduce_max over |f(i,j,comp)| (true Kokkos reduction,
// Kokkos::Max), aggregated by host max over the fabs.
//
// No more device_fence() up front: under Kokkos parallel_reduce is blocking and
// absorbs the barrier. EXACT everywhere: max and fabs are without rounding and max
// is associative/commutative in IEEE754, so bit-identical to the old norm_inf
// regardless of backend (the reduction order changes no bit).
/// Infinity norm max |f(.,.,comp)| over the valid cells (LOCAL, without MPI all_reduce). EXACT
/// on all backends (max without rounding, associative/commutative) -> bit-identical everywhere.
inline Real norm_inf(const MultiFab& mf, int comp = 0) {
  Real m = 0;
  for (int li = 0; li < mf.local_size(); ++li) {
    const ConstArray4 a = mf.fab(li).const_array();
    m = std::max(m, reduce_max_cell(mf.box(li), detail::NormInfKernel{a, comp}));
  }
  return m;  // MPI all-reduce max later (iso-behavior, not added here)
}

/// z <- a x + b y over ALL components of the valid cells. Identical layouts; aliasing safe.
inline void lincomb(MultiFab& z, Real a, const MultiFab& x, Real b, const MultiFab& y) {
  const int nc = z.ncomp();
  for (int li = 0; li < z.local_size(); ++li) {
    Array4 Z = z.fab(li).array();
    const ConstArray4 X = x.fab(li).const_array();
    const ConstArray4 Y = y.fab(li).const_array();
    const Box2D bb = z.fab(li).box();
    for (int c = 0; c < nc; ++c)
      for_each_cell(bb, detail::LincombKernel{Z, X, Y, a, b, c});
  }
}

// Dot product sum_cells x . y over the VALID cells of component comp, reduced over all
// ranks (all-reduce). Building block of Krylov solvers (BiCGStab: rho, alpha, omega, betas). Each
// local fab is reduced by reduce_sum_cell (true Kokkos reduction, Kokkos::Sum), the local fabs
// aggregated by host sum, then all_reduce_sum aggregates the ranks.
//
// COLLECTIVE, MANDATORY UNDER MPI: all_reduce_sum is called on EVERY rank, including a rank
// WITH NO box (local_size()==0, which then contributes 0 to the local sum). Without this call on all
// ranks, MPI_Allreduce deadlocks (desynchronized collective); the Krylov solver must therefore
// NEVER short-circuit dot() on an empty rank. In serial all_reduce_sum is the identity.
//
// FP NOTE (like sum()): Kokkos::Sum re-associates the sum per tile, so dot is not bit-identical
// to a lexicographic sum (deterministic/idempotent nonetheless, all Kokkos spaces). Under MPI, the all-reduce
// returns the SAME value to all ranks (MPI_SUM over one same set of local contributions), so the
// Krylov stopping criterion triggers at the SAME iteration everywhere (no desynchronization).
/// Dot product Sum_cells x.y over component comp, reduced over ALL ranks (all_reduce).
/// COLLECTIVE, MANDATORY UNDER MPI: must be called on every rank (including empty), otherwise
/// deadlock. FP NOTE: not bit-identical across backends under Kokkos; the all-reduce returns the same
/// value to all ranks (no desynchronization of the Krylov stopping criterion).
inline Real dot(const MultiFab& x, const MultiFab& y, int comp = 0) {
  Real s = 0;
  for (int li = 0; li < x.local_size(); ++li) {
    const ConstArray4 X = x.fab(li).const_array();
    const ConstArray4 Y = y.fab(li).const_array();
    s += reduce_sum_cell(x.box(li), detail::DotKernel{X, Y, comp});
  }
  return static_cast<Real>(all_reduce_sum(static_cast<double>(s)));
}

}  // namespace adc
