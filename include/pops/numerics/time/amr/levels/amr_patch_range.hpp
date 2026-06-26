#pragma once

#include <pops/numerics/time/amr/reflux/amr_flux_helpers.hpp>
#include <pops/amr/hierarchy/refinement_ratio.hpp>
#include <pops/parallel/comm.hpp>  // all_reduce_sum_inplace (distributed multi-patch reflux)

#include <algorithm>

/// @file
/// @brief Named types of the multi-patch coarse-fine interface: PatchRange (coarse footprint
///        of a fine patch), FluxRegister (GLOBAL-indexed coarse buffer + all_reduce), CoverageMask
///        (cells shadowed by a patch), SubcyclingSchedule (Berger-Oliger cadence) and
///        CoarseFineInterface (coverage + reflux routing), with the multi-box fill/avgdown
///        helpers (mf_fill_fine_ghosts_multi, mf_average_down_multi, fill_periodic_local).
///
/// Layer: `include/pops/numerics/time`.
/// Role: promote to TYPES the roles previously inlined/duplicated in the multi-patch
///        subcycling (amr_subcycling.hpp). Centralization with strictly preserved arithmetic.
///
/// Invariants:
/// - PatchRange uses the historical upper bound (hi-1)/2 (NOT Box2D::coarsen, floor of both
///   bounds) -> bit-identical to the original inline footprints;
/// - FluxRegister / CoverageMask are built on the GLOBAL box_array (known to all ranks):
///   MPI-safe. Each rank fills its LOCAL contributions (0 elsewhere), gather() sums them via
///   all_reduce_sum_inplace; in serial all_reduce is the identity -> bit-for-bit identical;
/// - CoverageMask prevents double-reflux of a fine-fine joint (only true fine-coarse interfaces
///   are corrected);
/// - AvgDownMultiKernel / route_reflux are NAMED functors/functions (no generic lambda)
///   -> safe under nvcc;
/// - fill_periodic_local serves the REPLICATED coarse (per-rank dmap): purely local folding
///   without an MPI plan, reads valid / writes ghost (no race).

namespace pops {

static_assert(kAmrRefRatio == 2, "ratio-2-structural kernels below assume kAmrRefRatio == 2");

// PatchRange (review, point 5: role promoted to a type). COARSE footprint [I0..I1]x[J0..J1]
// of a fine patch under ratio 2: I0 = lo/2, I1 = (hi-1)/2 (aligned patch, lo even / hi odd).
// Centralizes the footprint computation repeated inline in average_down, the coverage and the
// init of the reflux registers. NB: this is NOT Box2D::coarsen (floor of both bounds) but the
// historical upper bound (hi-1)/2; the exact arithmetic is preserved (bit-identical).
struct PatchRange {
  int I0, I1, J0, J1;
  explicit PatchRange(const Box2D& fine)
      : I0(fine.lo[0] / 2),
        I1((fine.hi[0] - 1) / 2),
        J0(fine.lo[1] / 2),
        J1((fine.hi[1] - 1) / 2) {}
  Box2D box() const { return Box2D{{I0, J0}, {I1, J1}}; }  // coarse footprint (cells)
};

// multi-box fine ghosts from the coarse (space+time interp), THEN fill_boundary
// (fine-fine) will overwrite the ghosts covered by a neighbor box. coarse mono-box.
inline void mf_fill_fine_ghosts_multi(MultiFab& Uf, const MultiFab& Uc_old, const MultiFab& Uc_new,
                                      Real frac) {
  device_fence();
  const int nc = Uf.ncomp();
  const ConstArray4 co = Uc_old.fab(0).const_array();
  const ConstArray4 cn = Uc_new.fab(0).const_array();
  for (int li = 0; li < Uf.local_size(); ++li) {
    Array4 f = Uf.fab(li).array();
    const Box2D v = Uf.box(li), g = Uf.fab(li).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i)
        if (!v.contains(i, j))
          fill_cf_ghost_cell(f, co, cn, i, j, nc, frac);
  }
}

namespace detail {
// NAMED device-clean functor (extended lambda -> trips nvcc cross-TU): fine -> coarse average
// (ratio 2) of a fine box over the PatchRange coarse footprint. Body bit-identical to the old
// lambda of mf_average_down_multi.
struct AvgDownMultiKernel {
  ConstArray4 f;
  Array4 c;
  int nc;
  POPS_HD void operator()(int I, int J) const {
    for (int k = 0; k < nc; ++k)
      c(I, J, k) = Real(0.25) * (f(2 * I, 2 * J, k) + f(2 * I + 1, 2 * J, k) +
                                 f(2 * I, 2 * J + 1, k) + f(2 * I + 1, 2 * J + 1, k));
  }
};
}  // namespace detail

// fine -> coarse average over the footprint of EACH fine box (multi-box).
inline void mf_average_down_multi(const MultiFab& Uf, MultiFab& Uc) {
  const int nc = Uc.ncomp();
  Array4 c = Uc.fab(0).array();
  for (int li = 0; li < Uf.local_size(); ++li) {
    const ConstArray4 f = Uf.fab(li).const_array();
    const PatchRange pr(Uf.box(li));
    for_each_cell(pr.box(), detail::AvgDownMultiKernel{f, c, nc});
  }
}

// PURELY LOCAL periodic fill of the ghosts of a mono-box coarse (self-folding).
// Equivalent to a periodic fill_boundary for a single box, but WITHOUT the MPI plan: serves
// the REPLICATED coarse (per-rank copy), whose per-rank DistributionMapping would violate the
// replicated-metadata assumption of fill_boundary. Reads valid cells (indices
// folded into [0,N)) and writes only ghosts: no read/write race.
inline void fill_periodic_local(MultiFab& mf, const Box2D& dom) {
  device_fence();
  const int nc = mf.ncomp(), NX = dom.nx(), NY = dom.ny();
  auto wrap = [](int x, int n) { return (x % n + n) % n; };
  for (int li = 0; li < mf.local_size(); ++li) {
    Array4 a = mf.fab(li).array();
    const Box2D g = mf.fab(li).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i)
        if (i < 0 || i >= NX || j < 0 || j >= NY)
          for (int k = 0; k < nc; ++k)
            a(i, j, k) = a(wrap(i, NX), wrap(j, NY), k);
  }
}

// FluxRegister (review, point 2: role promoted to a type). Coarse register with GLOBAL indexing
// over a REGION (box, with origin), to lift average_down (overwrite of covered cells, set)
// and reflux (addition to bordering cells, bounded add) across ranks.
// Each rank fills its LOCAL contributions (0 elsewhere), gather() sums them via
// all_reduce_sum_inplace, then each rank reads the total via at(). In serial all_reduce is
// the identity -> bit-for-bit identical. Region with origin (0,0) = full coarse grid; region
// = bounding box = multi-box average_down path. Same index formulas as before.
struct FluxRegister {
  int I0, J0, NX, NY, nc;
  std::vector<Real> buf;
  FluxRegister(const Box2D& region, int ncomp)
      : I0(region.lo[0]),
        J0(region.lo[1]),
        NX(region.hi[0] - region.lo[0] + 1),
        NY(region.hi[1] - region.lo[1] + 1),
        nc(ncomp),
        buf(static_cast<std::size_t>(NX) * NY * ncomp, Real(0)) {}
  std::size_t idx(int I, int J, int k) const {
    return (static_cast<std::size_t>(J - J0) * NX + (I - I0)) * nc + k;
  }
  bool in(int I, int J) const { return I >= I0 && I < I0 + NX && J >= J0 && J < J0 + NY; }
  void set(int I, int J, int k, Real v) { buf[idx(I, J, k)] = v; }  // overwrite (average_down)
  void add(int I, int J, int k, Real v) {                           // bordering addition (reflux)
    if (in(I, J))
      buf[idx(I, J, k)] += v;
  }
  Real at(int I, int J, int k) const { return buf[idx(I, J, k)]; }
  void gather() { all_reduce_sum_inplace(buf.data(), static_cast<int>(buf.size())); }
};

// CoverageMask (review, point 2: "coverage" part of CoarseFineInterface). Coarse mask
// over a REGION saying which cells are SHADOWED by a fine patch. Built on the
// GLOBAL box_array (known to all ranks) -> MPI-safe. mark(box) marks the coarse footprint
// of a patch (intersected with the region); covered(I,J) is bounded (false outside region).
// This is what prevents the double-reflux of a fine-fine joint. Same cells as before.
struct CoverageMask {
  int I0, J0, NX, NY;
  std::vector<char> cov;
  explicit CoverageMask(const Box2D& region)
      : I0(region.lo[0]),
        J0(region.lo[1]),
        NX(region.hi[0] - region.lo[0] + 1),
        NY(region.hi[1] - region.lo[1] + 1),
        cov(static_cast<std::size_t>(NX) * NY, 0) {}
  void mark(const Box2D& b) {  // marks the cells of b intersected with the region
    const int i0 = std::max(b.lo[0], I0), i1 = std::min(b.hi[0], I0 + NX - 1);
    const int j0 = std::max(b.lo[1], J0), j1 = std::min(b.hi[1], J0 + NY - 1);
    for (int J = j0; J <= j1; ++J)
      for (int I = i0; I <= i1; ++I)
        cov[(static_cast<std::size_t>(J - J0) * NX) + (I - I0)] = 1;
  }
  bool covered(int I, int J) const {
    if (I < I0 || I >= I0 + NX || J < J0 || J >= J0 + NY)
      return false;
    return cov[(static_cast<std::size_t>(J - J0) * NX) + (I - I0)] != 0;
  }
};

// SubcyclingSchedule (review, point 5: role promoted to a type). Berger-Oliger cadence of a
// level: temporal refinement ratio r, substep dt/r, and temporal position frac(s)
// = s/r of substep s in the parent step. Centralizes the `const int r = kAmrRefRatio`, `dt / r` and
// `Real(s) / r` scattered across the subcycling loops. Arithmetic strictly preserved:
// dt_sub(dt) == dt / r and frac(s) == Real(s) / r at the same types, thus bit-identical.
struct SubcyclingSchedule {
  int r;
  explicit SubcyclingSchedule(int ratio = kAmrRefRatio) : r(ratio) {}
  int count() const { return r; }                 // number of substeps
  Real dt_sub(Real dt) const { return dt / r; }   // fine step = parent step / r
  Real frac(int s) const { return Real(s) / r; }  // temporal position of substep s
};

// CoarseFineInterface (review, point 2). The coarse-fine interface of a level: coverage
// (which coarse cells are shadowed by a fine patch, via CoverageMask) + bordering ROUTING
// of the reflux (which coarse cell borders which fine-patch face, and the conservative
// correction poured into it). Centralizes the two inline logics previously duplicated
// in amr_step_2level_multipatch and subcycle_level_mp. Builds the mask on the GLOBAL
// box_array() of the fine patches (MPI-safe). route_reflux is a template on the register
// type (Reg / RegMP, same field layout): named function (no generic lambda), thus
// safe under nvcc. Arithmetic bit-identical to the previous bodies.
struct CoarseFineInterface {
  CoverageMask cmask;
  // region = coarse footprint of the level (origin (0,0), dims NX x NY); fine_ba = GLOBAL
  // fine patches (all the boxes, known to all ranks). We mark the coarse PatchRange footprint
  // of each patch.
  CoarseFineInterface(const Box2D& coarse_region, const BoxArray& fine_ba) : cmask(coarse_region) {
    for (int g = 0; g < fine_ba.size(); ++g)
      cmask.mark(PatchRange(fine_ba[g]).box());
  }
  bool covered(int I, int J) const { return cmask.covered(I, J); }

  // Pours the reflux correction of ONE fine patch (register g, parent coords) into the
  // coarse register ref: on each BORDERING coarse cell not covered by another patch,
  // (time-integrated fine flux - coarse flux x dt) / dx|dy. Same formulas, same order
  // (left/right in x, bottom/top in y) as the original inline bodies.
  template <class Reg>
  void route_reflux(const Reg& g, Real dx, Real dy, Real dt, FluxRegister& ref, int nc) const {
    for (int J = g.J0; J <= g.J1; ++J)
      for (int k = 0; k < nc; ++k) {
        if (!covered(g.I0 - 1, J))
          ref.add(g.I0 - 1, J, k,
                  -(g.fL[(J - g.J0) * nc + k] - g.cL[(J - g.J0) * nc + k] * dt) / dx);
        if (!covered(g.I1 + 1, J))
          ref.add(g.I1 + 1, J, k,
                  +(g.fR[(J - g.J0) * nc + k] - g.cR[(J - g.J0) * nc + k] * dt) / dx);
      }
    for (int I = g.I0; I <= g.I1; ++I)
      for (int k = 0; k < nc; ++k) {
        if (!covered(I, g.J0 - 1))
          ref.add(I, g.J0 - 1, k,
                  -(g.fB[(I - g.I0) * nc + k] - g.cB[(I - g.I0) * nc + k] * dt) / dy);
        if (!covered(I, g.J1 + 1))
          ref.add(I, g.J1 + 1, k,
                  +(g.fT[(I - g.I0) * nc + k] - g.cT[(I - g.I0) * nc + k] * dt) / dy);
      }
  }
};

}  // namespace pops
