/// @file
/// @brief Global wave-speed / step-bound reductions and the HLL wave-speed cache.
///
/// CONTRACT: the CFL and step-bound reductions over a whole MultiFab (device reduction + MPI
/// all_reduce), plus the OPT-IN per-cell wave-speed scratch fill.
///   - max_wave_speed_mf: global max CFL speed (collective under MPI).
///   - max_wave_speed_hotspot_mf: cell dominating the CFL bound (diagnostic, ADC-182).
///   - max_stability_speed_mf / max_source_frequency_mf / min_stability_dt_mf: optional
///     step-bound traits (HasStabilitySpeed / HasSourceFrequency / HasStabilityDt).
///   - fill_wave_speed_cache: per-cell (lo_x, hi_x, lo_y, hi_y) scratch for the HLL cache path
///     (the scratch is CONSUMED in cartesian_operator.hpp).
///
/// COLLECTIVE UNDER MPI: every reduction aggregates via all_reduce over ALL ranks; without it
/// each rank would choose a different dt and the simulation desynchronizes (see notes below).

#pragma once

#include <adc/core/state/state.hpp>
#include <adc/core/foundation/types.hpp>
#include <adc/mesh/storage/fab2d.hpp>
#include <adc/mesh/execution/for_each.hpp>  // reduce_max_cell, reduce_min_cell
#include <adc/mesh/storage/multifab.hpp>
#include <adc/numerics/spatial/state_access.hpp>  // load_state, load_aux, aux_comps
#include <adc/parallel/comm.hpp>                  // all_reduce_max, all_reduce_min

#include <algorithm>
#include <limits>

namespace adc {

namespace detail {
/// MaxWaveSpeedKernel<Model>: device reduction functor for max_wave_speed_mf.
///
/// Accumulates the max of the wave speeds in both directions at cell (i,j).
/// Named functor (and not an extended lambda): robust device emission from an external TU
/// (add_compiled_model). Body bit-identical to the former lambda. ADC_HD.
template <class Model>
struct MaxWaveSpeedKernel {
  Model model;
  ConstArray4 u, a;
  ADC_HD void operator()(int i, int j, Real& acc) const {
    const auto s = load_state<Model>(u, i, j);
    const Aux ax = load_aux<aux_comps<Model>()>(a, i, j);
    const Real wx = model.max_wave_speed(s, ax, 0);
    const Real wy = model.max_wave_speed(s, ax, 1);
    const Real w = wx > wy ? wx : wy;
    if (w > acc)
      acc = w;
  }
};
}  // namespace detail

/// max_wave_speed_mf: global max of the wave speed over the whole MultiFab (CFL).
///
/// Reduce over all local boxes then all_reduce_max over all MPI ranks.
/// Without the all_reduce, each rank only sees its boxes and step_cfl computes a different
/// dt per rank (desynchronization / divergence). In serial all_reduce_max is the identity.
/// For a model without transport (max_wave_speed = 0 everywhere) -> returns 0 (step unconstrained).
//
// COLLECTIVE UNDER MPI: we aggregate via all_reduce_max over ALL ranks (same convention as
// AmrCouplerMp::max_wave_speed and GeometricMG::current_residual). Without this all-reduce, each
// rank only sees the max of ITS boxes: step_cfl / step_adaptive then choose a DIFFERENT dt per
// rank (the rank whose local max is lower takes too large a step) and the simulation diverges or
// desynchronizes the ranks. In serial all_reduce_max is the identity (behavior unchanged).
template <class Model>
inline Real max_wave_speed_mf(const Model& model, const MultiFab& U, const MultiFab& aux) {
  Real m = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    m = std::max(m, reduce_max_cell(U.box(li), detail::MaxWaveSpeedKernel<Model>{model, u, a}));
  }
  return static_cast<Real>(all_reduce_max(static_cast<double>(m)));
}

namespace detail {
/// Locates the cell DOMINATING the CFL (dt_hotspot diagnostic, ADC-182): EQUALITY scan
/// of the recomputed w -- same functor and same data as MaxWaveSpeedKernel, hence bit-equal
/// to the max returned by max_wave_speed_mf -- which encodes the GLOBAL index j*nx + i as
/// Real (exact as long as nx*ny < 2^53) and reduces to the MIN (first cell in lexicographic
/// order: deterministic). NAMED functor (cross-TU instantiation under nvcc).
template <class Model>
struct WaveSpeedMatchKernel {
  Model model;
  ConstArray4 u, a;
  Real target;
  Real nx;  // encoding stride (nx of the DOMAIN, global indices)
  ADC_HD void operator()(int i, int j, Real& acc) const {
    const auto s = load_state<Model>(u, i, j);
    const Aux ax = load_aux<aux_comps<Model>()>(a, i, j);
    const Real wx = model.max_wave_speed(s, ax, 0);
    const Real wy = model.max_wave_speed(s, ax, 1);
    const Real w = wx > wy ? wx : wy;
    if (w == target) {
      const Real idx = static_cast<Real>(j) * nx + static_cast<Real>(i);
      if (idx < acc)
        acc = idx;
    }
  }
};
}  // namespace detail

/// dt_hotspot diagnostic (ADC-182): the cell (GLOBAL indices) that dominates the block's transport
/// CFL bound, and its speed w = max(wx, wy). ON DEMAND only -- two full passes (max then location
/// by bit-exact equality), step_cfl does not touch it (bit-identical). MPI: all_reduce of the max
/// then all_reduce_min of the encoded index (+inf on the non-holder ranks). @p nx: domain width
/// (encoding j*nx + i).
template <class Model>
inline void max_wave_speed_hotspot_mf(const Model& model, const MultiFab& U, const MultiFab& aux,
                                      int nx, Real& w_out, int& i_out, int& j_out) {
  const Real w = max_wave_speed_mf(model, U, aux);
  Real best = std::numeric_limits<Real>::infinity();
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    best = std::min(best, reduce_min_cell(U.box(li), detail::WaveSpeedMatchKernel<Model>{
                                                         model, u, a, w, static_cast<Real>(nx)}));
  }
  best = static_cast<Real>(all_reduce_min(static_cast<double>(best)));
  w_out = w;
  // identity of Kokkos::Min = max_real (finite): a rank/box without a cell equaling the max
  // leaves this value -> we only decode if a REAL index was encoded.
  if (best >= Real(0) && best < std::numeric_limits<Real>::max() * Real(0.5)) {
    const long long idx = static_cast<long long>(best);
    i_out = static_cast<int>(idx % nx);
    j_out = static_cast<int>(idx / nx);
  } else {  // empty domain / degenerate state: no cell (w may be 0)
    i_out = -1;
    j_out = -1;
  }
}

// ============================================================================
// OPTIONAL STEP-BOUND REDUCTIONS (audit 2026-06, step_cfl effort).
// Counterparts of max_wave_speed_mf for the HasStabilitySpeed / HasSourceFrequency /
// HasStabilityDt traits (cf. core/physical_model.hpp). Same conventions: reduction via the seam
// (device under Kokkos), MPI all_reduce (without which each rank would choose a different dt).
// Instantiated ONLY for a model declaring the trait (if constexpr on the block_builder side):
// zero codegen, zero cost for a legacy model.
// ============================================================================

namespace detail {
/// StabilitySpeedKernel: max over cells/directions of model.stability_speed (replaces
/// MaxWaveSpeedKernel when the trait is declared). Named functor (device-clean cross-TU).
template <class Model>
struct StabilitySpeedKernel {
  Model model;
  ConstArray4 u, a;
  ADC_HD void operator()(int i, int j, Real& acc) const {
    const auto s = load_state<Model>(u, i, j);
    const Aux ax = load_aux<aux_comps<Model>()>(a, i, j);
    const Real wx = model.stability_speed(s, ax, 0);
    const Real wy = model.stability_speed(s, ax, 1);
    const Real w = wx > wy ? wx : wy;
    if (w > acc)
      acc = w;
  }
};

/// SourceFrequencyKernel: max over cells of model.source_frequency (mu >= 0, 1/s).
template <class Model>
struct SourceFrequencyKernel {
  Model model;
  ConstArray4 u, a;
  ADC_HD void operator()(int i, int j, Real& acc) const {
    const auto s = load_state<Model>(u, i, j);
    const Aux ax = load_aux<aux_comps<Model>()>(a, i, j);
    const Real mu = model.source_frequency(s, ax);
    if (mu > acc)
      acc = mu;
  }
};

/// InvStabilityDtKernel: max over cells of 1/model.stability_dt. We reduce the INVERSE (a
/// frequency) because the seam only provides a MAX reduction initialized to 0 (reduce_max_cell):
/// min(dt) == 1/max(1/dt) for dt > 0. A stability_dt <= 0 or non-finite is ignored (does not
/// constrain) -- the model signals "no bound here" by returning +inf.
template <class Model>
struct InvStabilityDtKernel {
  Model model;
  ConstArray4 u, a;
  ADC_HD void operator()(int i, int j, Real& acc) const {
    const auto s = load_state<Model>(u, i, j);
    const Aux ax = load_aux<aux_comps<Model>()>(a, i, j);
    const Real db = model.stability_dt(s, ax);
    if (db > Real(0)) {
      const Real inv = Real(1) / db;
      if (inv > acc)
        acc = inv;
    }
  }
};
}  // namespace detail

/// Global max of the STABILITY speed (HasStabilitySpeed trait) -- counterpart of max_wave_speed_mf.
template <class Model>
inline Real max_stability_speed_mf(const Model& model, const MultiFab& U, const MultiFab& aux) {
  Real m = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    m = std::max(m, reduce_max_cell(U.box(li), detail::StabilitySpeedKernel<Model>{model, u, a}));
  }
  return static_cast<Real>(all_reduce_max(static_cast<double>(m)));
}

/// Global max of the source frequency (HasSourceFrequency trait). 0 if the source does not constrain.
template <class Model>
inline Real max_source_frequency_mf(const Model& model, const MultiFab& U, const MultiFab& aux) {
  Real m = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    m = std::max(m, reduce_max_cell(U.box(li), detail::SourceFrequencyKernel<Model>{model, u, a}));
  }
  return static_cast<Real>(all_reduce_max(static_cast<double>(m)));
}

/// Global min of the declared admissible step (HasStabilityDt trait), via max(1/dt) (cf.
/// InvStabilityDtKernel). @return 0 if NO cell constrains (the block imposes no bound).
template <class Model>
inline Real min_stability_dt_mf(const Model& model, const MultiFab& U, const MultiFab& aux) {
  Real inv = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    inv =
        std::max(inv, reduce_max_cell(U.box(li), detail::InvStabilityDtKernel<Model>{model, u, a}));
  }
  inv = static_cast<Real>(all_reduce_max(static_cast<double>(inv)));
  return inv > Real(0) ? Real(1) / inv : Real(0);
}

// ============================================================================
// HLL WAVE SPEED CACHE (OPT-IN -- default path strictly untouched)
// ============================================================================
// The HLL flux bounds each face by the signal speeds of BOTH adjacent cells (Davis estimates, cf.
// hll_speeds). The default path recalls model.wave_speeds per face: for a model whose wave speeds are
// expensive (moment hierarchy + factorizations at each call), wave_speeds is recomputed several times
// per cell and per RK stage. This OPT-IN path evaluates wave_speeds ONCE per cell and per direction in
// a scratch (4 components lo_x/hi_x/lo_y/hi_y), then assembles the residual by reading the scratch: the
// face speed at i-1/2 becomes min/max of the signed speeds of the two neighbor cells (union over the
// neighbors, as in the per-cell reference). The scratch is CONSUMED by assemble_rhs_hll_cached
// (cartesian_operator.hpp).

namespace detail {
/// WaveSpeedCacheKernel: evaluates model.wave_speeds per cell in both directions and stores
/// (lo_x, hi_x, lo_y, hi_y) in a 4-component scratch. Named functor (device-clean cross-TU, same
/// emission constraint as MaxWaveSpeedKernel). ADC_HD.
template <class Model>
struct WaveSpeedCacheKernel {
  Model model;
  ConstArray4 u, a;
  Array4 ws;
  ADC_HD void operator()(int i, int j) const {
    const auto s = load_state<Model>(u, i, j);
    const Aux ax = load_aux<aux_comps<Model>()>(a, i, j);
    Real lox, hix, loy, hiy;
    model.wave_speeds(s, ax, 0, lox, hix);
    model.wave_speeds(s, ax, 1, loy, hiy);
    ws(i, j, 0) = lox;
    ws(i, j, 1) = hix;
    ws(i, j, 2) = loy;
    ws(i, j, 3) = hiy;
  }
};
}  // namespace detail

/// fill_wave_speed_cache: fills the per-cell wave speed scratch (lo_x, hi_x, lo_y, hi_y).
///
/// Evaluated on the VALID box grown by one ghost (grow(v, 1)): a valid cell's face speed reads the
/// cached speed of its neighbor, so the scratch must cover one ghost on each side. @p cache must have
/// the SAME layout as @p U (same BoxArray / DistributionMapping), 4 components and >= 1 ghost. @p U
/// carries its ghosts already filled (fill_ghosts); @p aux carries at least 1 ghost (read at the
/// neighbor cells, like assemble_rhs).
template <class Model>
inline void fill_wave_speed_cache(const Model& model, const MultiFab& U, const MultiFab& aux,
                                  MultiFab& cache) {
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    Array4 ws = cache.fab(li).array();
    for_each_cell(U.box(li).grow(1), detail::WaveSpeedCacheKernel<Model>{model, u, a, ws});
  }
}

}  // namespace adc
