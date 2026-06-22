/// @file
/// @brief for_each_cell and reductions: the parallelism SEAM over the cells of a Box2D;
///        sync_host / sync_device: the residency COHERENCE seam (counterpart for host accesses).
///
/// KOKKOS IS THE ONLY on-node backend: this seam compiles ONLY under ADC_HAS_KOKKOS (cf. CMake, which
/// makes Kokkos mandatory; without it, #error below). The functor is taken BY VALUE and receives
/// (i, j); it captures Array4 handles by value (POD), never the Fab nor anything virtual:
/// exactly the constraint of a device kernel. The on-node target (sequential = Kokkos Serial, CPU
/// multi-thread = Kokkos OpenMP, GPU = Kokkos Cuda/HIP) is chosen AT KOKKOS INSTALLATION, not
/// by an adc flag: a single for_each_cell call (Kokkos::parallel_for over MDRangePolicy<Rank<2>>)
/// covers all three. The CPU -> GPU switch therefore does NOT change the call sites.
/// FP CHOICE: the SUM reduction (Kokkos::Sum) reassociates the addition per tile -> DETERMINISTIC per
/// tile (idempotent: same data, same backend -> same bits) but NOT bit-identical to a lexicographic
/// sum; this holds for Serial, OpenMP and Cuda (a single path, Kokkos). The MAX reduction
/// is exact everywhere (max associative/commutative in IEEE754). sync_host() = a targeted device_fence()
/// before a host access; sync_device() = no-op under unified memory (scaffolding for a future
/// non-unified path).

#pragma once

#include <adc/core/foundation/kokkos_env.hpp>  // detail::ensure_kokkos_initialized + device_fence (life cycle)
#include <adc/core/foundation/types.hpp>
#include <adc/mesh/index/box2d.hpp>

#include <cstdint>      // std::int64_t: cell counts (LLP64 portability, no-op on LP64)
#include <cstdlib>      // getenv / strtol: overridable serial fallback threshold (#165)
#include <type_traits>  // std::is_same_v: compile-time guard host vs device exec space (#165)

#ifndef ADC_HAS_KOKKOS
// adc_cpp is KOKKOS-ONLY: there is no longer a standalone OpenMP backend nor a manual host loop
// as a production path. Configure with -DADC_USE_KOKKOS=ON (+ -DKokkos_ROOT=...); serial
// goes through a Kokkos install with Kokkos_ENABLE_SERIAL=ON.
#error \
    "adc_cpp is Kokkos-only: for_each_cell requires ADC_HAS_KOKKOS. Configure with -DADC_USE_KOKKOS=ON and a Kokkos Serial/OpenMP/Cuda install."
#endif

#include <Kokkos_Core.hpp>

namespace adc {

// detail::ensure_kokkos_initialized() and device_fence(): defined in adc/core/kokkos_env.hpp
// (Kokkos life cycle shared with the unified allocator, which must also initialize Kokkos BEFORE
// its first kokkos_malloc, otherwise the Kokkos build crashes when constructing a Fab).

// SERIAL FALLBACK THRESHOLD for for_each_cell (#165). Under a HOST Kokkos execution space
// (Serial/OpenMP), launching a Kokkos::parallel_for(MDRangePolicy) on a tiny box pays a
// fork/join (and the policy construction) that OVERWHELMS the useful work. The multigrid
// V-cycle descends down to ~2x2/4x4 grids; on those levels the GS smoother, the residual,
// the restriction/prolongation and the copies chain dozens of parallel_for over a few
// cells, and this launch overhead DOMINATES the solve time. Below this threshold we run
// a SEQUENTIAL host loop (internal to the Kokkos path, this is NOT a separate backend),
// above it we keep Kokkos parallel_for for the fine grids.
//
// BIT-IDENTITY. for_each_cell has NO inter-iteration dependency: each f(i, j)
// writes only cell (i, j) of its destination and reads cells IT DOES NOT WRITE
// in the same call (the GS smoother is RED-BLACK colored -- one color only reads
// the other; residual/restriction/prolongation/copies/saxpy write a destination
// distinct from the source). The result is therefore INDEPENDENT OF the traversal ORDER:
// the sequential loop yields exactly the same bits as MDRangePolicy<Rank<2>>.
// The threshold touches ONLY for_each_cell (not the reductions for_each_cell_reduce_*:
// the Kokkos parallel sum reassociates the addition, so switching them to serial would
// NOT be bit-identical -- we leave them intact; the max is exact but the smoother itself
// does go through for_each_cell, where the overhead of the small grids concentrates).
//
// Overridable at run time via ADC_FOREACH_SERIAL_THRESHOLD (read once) to
// resweep the threshold without recompiling; default 4096 (same fork/join vs computation
// trade-off as the old if() clause of the removed OpenMP path).
namespace detail {
inline std::int64_t foreach_serial_threshold() {
  static const std::int64_t thr = []() -> std::int64_t {
    if (const char* e = std::getenv("ADC_FOREACH_SERIAL_THRESHOLD")) {
      char* end = nullptr;
      const std::int64_t v = std::strtol(e, &end, 10);
      if (end != e && v >= 0)
        return v;
    }
    return 4096;
  }();
  return thr;
}
}  // namespace detail

// ---------------------------------------------------------------------------
// Data residency: sync_host() / sync_device(). The COHERENCE seam, the
// counterpart of for_each_cell for host accesses.
//
// Today the Fab storage lives in UNIFIED memory (Kokkos::SharedSpace, cf.
// allocator.hpp): the same buffer serves the host code (operator(), loops) AND
// the device kernels. Coherence therefore does NOT require a copy, only
// ORDERING: a host access must not read/write a buffer while
// an async kernel still touches it. Until now this ordering was laid down by
// hand by scattered device_fence() calls, without ever saying WHICH residency one
// wants to make valid.
//
// sync_host()/sync_device() ENCODE that intent:
//   - sync_host(): "I am going to read/write this data FROM THE HOST;
//                      make it valid host-side". Under SharedSpace = a
//                      targeted device_fence() (wait for in-flight kernels), so
//                      host accesses are then race-free (data race).
//   - sync_device(): "I am going to read/write this data FROM THE DEVICE
//                      (a kernel); make it valid device-side". Under
//                      SharedSpace the preceding host writes are visible
//                      from the device without a barrier (no async host pipeline to
//                      drain), so it is a REAL no-op today; the
//                      function exists to MARK the intent at the call site.
//
// SEMANTICS UNDER SHAREDSPACE (current state): these calls are at most a fence,
// never a copy. The behavior therefore stays BIT-IDENTICAL to the old code
// (sync_host == the old device_fence() laid before a host access; sync_device
// == nothing). This is deliberately SCAFFOLDING: under unified memory there is
// nothing else to do.
//
// FUTURE NON-UNIFIED PATH (separate host/device buffers + deep_copy): this is WHERE
// the migration would plug in. sync_host() would do a Kokkos::deep_copy
// device->host (and a fence) if the device is the last residency written;
// sync_device() a deep_copy host->device in the other direction. Tracking "who
// owns the up-to-date data" (per-residency dirty flag) would live on the MultiFab,
// not here: this seam stays stateless, the MultiFab overloads carry the state.
// Since all the host-access sites already go through sync_host(), switching to
// that path will NOT touch the operators, exactly as for_each_cell
// isolates the CPU -> GPU switch from the call sites.

/// Makes the HOST residency valid before a host access (read/write from the host). Under unified memory
/// = a targeted device_fence() (waits for in-flight kernels).
inline void sync_host() {
  device_fence();
}

/// Marks a DEVICE residency (upcoming kernel). Under unified memory: NO-OP (host writes
/// are already visible from the device); exists to document the intent and to accommodate a future
/// deep_copy host->device on a non-unified path.
inline void sync_device() {}

/// Applies @p f to EACH cell (i, j) of box @p b (bounds inclusive), via Kokkos::parallel_for
/// (Serial / OpenMP / Cuda depending on the Kokkos install). @p f is taken by value and MUST be
/// device-callable (annotated ADC_HD, captures POD by value). No order guarantee.
template <class F>
void for_each_cell(const Box2D& b, F f) {
  // SMALL BOXES (#165): under a HOST Kokkos execution space (Serial/OpenMP), the
  // fork/join of a parallel_for on a tiny grid (coarse V-cycle levels,
  // ~2x2..32x32) overwhelms the computation. We then run a sequential host loop (internal
  // to the Kokkos path). BIT-IDENTICAL: no inter-iteration dependency (cf. the threshold note),
  // so the order affects no bit.
  //
  // DEVICE GUARD (if constexpr): the serial fallback is taken ONLY if the default execution space
  // of Kokkos IS the host space (Serial/OpenMP). Under a DEVICE space (Cuda
  // on a CUDA device), DefaultExecutionSpace != DefaultHostExecutionSpace: the host loop
  // would run on the CPU while the preceding device kernels are in flight (no
  // fence laid here) -- data race. We therefore keep parallel_for on device WHATEVER
  // THE size -> GPU path STRICTLY unchanged (the if constexpr evaporates at
  // compile time, zero overhead). Under SharedSpace + host execution, the loop is
  // race-free: the existing coherence seams (gs_rb_sweep lays its device_fence around the
  // sweeps, sync_host before the host accesses) stay in place and unchanged.
  if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>) {
    const std::int64_t n_cells =
        static_cast<std::int64_t>(b.hi[0] - b.lo[0] + 1) * (b.hi[1] - b.lo[1] + 1);
    if (n_cells < detail::foreach_serial_threshold()) {
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          f(i, j);
      return;
    }
  }
  detail::ensure_kokkos_initialized();
  // IndexType<int>: SIGNED indices. Ghost boxes have negative low
  // bounds (e.g. lo = -ng for copy_shifted); without an explicit signed type,
  // MDRangePolicy rejects the bound -1 (implicit conversion deemed unsafe).
  Kokkos::parallel_for("adc_for_each_cell",
                       Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<int>>(
                           {b.lo[0], b.lo[1]}, {b.hi[0] + 1, b.hi[1] + 1}),
                       f);
}

// Device reductions: the reducing counterpart of for_each_cell. Same constraints
// on the functor (device-callable POD, taken by value, captures a ConstArray4,
// never the Fab); it receives (i, j) and returns the value to accumulate. The seam carries
// the device ordering: under Kokkos the scalar is ready on return without a
// prior device_fence() (parallel_reduce is blocking host-side and orders itself
// after the parallel_for already submitted in the same space).
//
// IMPORTANT FP CHOICE. A true parallel reduction reassociates floating-point
// addition (non-associative in IEEE754): the result of the sum depends on
// the traversal order.
//   - SUM: Kokkos::Sum, DETERMINISTIC per-tile reduction (no floating-point
//     atomics). Two calls on identical data return exactly the
//     same bit -> idempotence (sum_unchanged) holds. But the per-tile order
//     DIFFERS from a lexicographic sum: the value is NOT bit-identical to
//     a hand-written (i, j) loop. Since Kokkos is the only backend, this
//     holds for ALL spaces (Serial, OpenMP, Cuda).
//   - MAX: Kokkos::Max, exact everywhere (max is associative/commutative and
//     rounding-free in IEEE754) -> bit-identical across Kokkos spaces.
// Summary: the SUM is deterministic-per-tile (idempotent) but reassociated; the
// MAX (norm_inf) is exact.

/// SUM reduction of @p f(i, j) over box @p b. @p f device-callable (ADC_HD) returning the value
/// to accumulate. FP WARNING: Kokkos::Sum reassociates the sum per tile (deterministic/idempotent but
/// not bit-identical to a lexicographic sum), for all Kokkos spaces. Blocking host-side.
template <class F>
Real for_each_cell_reduce_sum(const Box2D& b, F f) {
  detail::ensure_kokkos_initialized();
  Real result = 0;
  Kokkos::parallel_reduce(
      "adc_reduce_sum",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<int>>({b.lo[0], b.lo[1]},
                                                                     {b.hi[0] + 1, b.hi[1] + 1}),
      KOKKOS_LAMBDA(int i, int j, Real& acc) { acc += f(i, j); }, Kokkos::Sum<Real>{result});
  return result;  // blocking host-side: valid on return, without device_fence()
}

/// MAX reduction of @p f(i, j) over box @p b. @p f device-callable (ADC_HD). EXACT everywhere (max
/// is associative/commutative in IEEE754, rounding-free) -> bit-identical across Kokkos spaces. Blocking.
template <class F>
Real for_each_cell_reduce_max(const Box2D& b, F f) {
  detail::ensure_kokkos_initialized();
  Real result = 0;
  Kokkos::parallel_reduce(
      "adc_reduce_max",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<int>>({b.lo[0], b.lo[1]},
                                                                     {b.hi[0] + 1, b.hi[1] + 1}),
      KOKKOS_LAMBDA(int i, int j, Real& acc) {
        const Real v = f(i, j);
        if (v > acc)
          acc = v;
      },
      Kokkos::Max<Real>{result});
  return result;  // exact max (associative/commutative IEEE754), no fence
}

// MAX variant with a REDUCING FUNCTOR: @p f is passed DIRECTLY to Kokkos::parallel_reduce and
// receives (i, j, Real& acc) to update acc (acc = max(acc, value)). Unlike
// for_each_cell_reduce_max, NO extended lambda wraps @p f: this is the device-clean path
// for a Model-template kernel instantiated from an EXTERNAL TRANSLATION UNIT (add_compiled_model),
// where nvcc does not reliably emit an extended lambda (cf. the named functors of spatial_operator.hpp).
// Determinism and bit-exactness IDENTICAL to for_each_cell_reduce_max (same Kokkos::Max): only the
// carrier of the computation changes (named functor instead of a lambda wrapper).
/// MAX reduction with a REDUCING FUNCTOR: @p f receives (i, j, Real& acc) and updates acc, passed
/// DIRECTLY to Kokkos::parallel_reduce without a wrapper lambda (device-clean path for a kernel
/// instantiated cross-TU). Bit-exactness identical to for_each_cell_reduce_max.
template <class F>
Real reduce_max_cell(const Box2D& b, F f) {
  detail::ensure_kokkos_initialized();
  Real result = 0;
  Kokkos::parallel_reduce("adc_reduce_max_cell",
                          Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<int>>(
                              {b.lo[0], b.lo[1]}, {b.hi[0] + 1, b.hi[1] + 1}),
                          f, Kokkos::Max<Real>{result});
  return result;
}

// MIN variant: exact counterpart of reduce_max_cell for Kokkos::Min (dt_hotspot diagnostic,
// ADC-182: reduction of the smallest index encoded among the cells that equal the max).
template <class F>
Real reduce_min_cell(const Box2D& b, F f) {
  detail::ensure_kokkos_initialized();
  Real result = 0;
  Kokkos::parallel_reduce("adc_reduce_min_cell",
                          Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<int>>(
                              {b.lo[0], b.lo[1]}, {b.hi[0] + 1, b.hi[1] + 1}),
                          f, Kokkos::Min<Real>{result});
  return result;
}

// SUM variant with a REDUCING FUNCTOR: exact counterpart of reduce_max_cell for Kokkos::Sum. @p f
// receives (i, j, Real& acc) and accumulates (acc += value), passed DIRECTLY to parallel_reduce WITHOUT
// a wrapping extended lambda (unlike for_each_cell_reduce_sum, which lays one). This is
// the device-clean path required by a kernel instantiated from an EXTERNAL TRANSLATION UNIT (the
// Krylov solver drawn from the native harness/loader): nvcc does not reliably emit an extended lambda
// first-instantiated cross-TU (cf. the named functors of mf_arith.hpp / spatial_operator.hpp).
// Determinism and FP IDENTICAL to for_each_cell_reduce_sum: same per-tile deterministic Kokkos::Sum.
/// SUM reduction with a REDUCING FUNCTOR: @p f receives (i, j, Real& acc) and accumulates, passed DIRECTLY
/// to Kokkos::parallel_reduce without a wrapper lambda (device-clean cross-TU path). Same FP
/// guarantees as for_each_cell_reduce_sum (Kokkos::Sum reassociated per tile, deterministic/idempotent).
template <class F>
Real reduce_sum_cell(const Box2D& b, F f) {
  detail::ensure_kokkos_initialized();
  Real result = 0;
  Kokkos::parallel_reduce("adc_reduce_sum_cell",
                          Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<int>>(
                              {b.lo[0], b.lo[1]}, {b.hi[0] + 1, b.hi[1] + 1}),
                          f, Kokkos::Sum<Real>{result});
  return result;
}

}  // namespace adc
