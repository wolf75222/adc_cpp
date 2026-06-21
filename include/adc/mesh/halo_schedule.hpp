/// @file
/// @brief HaloSchedule: memoized intra-level halo-exchange plan for fill_boundary (ADC-260).
///
/// fill_boundary_begin used to enumerate, on EVERY call, the neighbor-box job schedule: a BoxHash
/// build plus a local (and, under MPI, a global) enumeration over the BoxArray. That schedule is a
/// pure function of the LAYOUT (BoxArray, DistributionMapping, n_grow) and the per-call (Periodicity,
/// domain); only the copy/pack/MPI/unpack of the LIVE data must rerun. This header holds the
/// cacheable plan so the enumeration runs ONCE per (layout, Periodicity, domain). Jobs carry GLOBAL
/// box indices (resolved to local fabs at replay), so a plan is valid for any MultiFab over the same
/// layout. MPI-free: the in-flight buffers and MPI_Request stay in HaloExchange (fill_boundary.hpp).

#pragma once

#include <adc/mesh/box2d.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace adc {

/// One halo copy/transfer: the ghost @p region of box @p dst is filled from the shifted valid region
/// of box @p src (shift sx, sy in cells for the periodic wrap; 0 for an interior neighbor). @p src and
/// @p dst are GLOBAL box indices into the BoxArray (resolved to local fabs when the job is replayed).
struct HaloJob {
  int src = 0;
  int dst = 0;
  int sx = 0;
  int sy = 0;
  Box2D region{};
};

/// Memoized schedule for ONE (Periodicity, domain) over a fixed layout. @p local holds the copies
/// whose dst AND src are owned by this rank; @p send[r]/@p recv[r] hold the jobs exchanged with rank
/// r (both empty unless built under MPI with n_ranks() > 1). The fingerprint (per_x, per_y, domain)
/// identifies the (Periodicity, domain) it was built for; the LAYOUT is implicit because the plan is
/// stored on the MultiFab that owns the BoxArray/DistributionMapping/n_grow (see HaloScheduleCache).
/// The jobs carry only Box2D regions, so the plan is INDEPENDENT of ncomp (the component count is
/// supplied at replay via mf.ncomp() to size buffers); ncomp is intentionally absent from the key.
struct HaloSchedule {
  bool per_x = false;
  bool per_y = false;
  Box2D domain{};
  std::vector<HaloJob> local;
  std::vector<std::vector<HaloJob>> send;  // [rank]; empty unless MPI && n_ranks() > 1
  std::vector<std::vector<HaloJob>> recv;  // [rank]
};

/// Small per-MultiFab cache of halo schedules, one entry per distinct (Periodicity, domain). In
/// practice a MultiFab is filled with a single (Periodicity, domain) for its role, so this holds one
/// or two entries; lookup is a short linear scan. Entries are shared_ptr so an in-flight
/// HaloExchange can hold a stable handle to the plan it is replaying even if a later call appends a
/// new entry. The cache LIVES ON the MultiFab (multifab.hpp); it is dropped when the MultiFab is
/// reassigned (e.g. AMR regrid builds a fresh MultiFab and move-assigns it over the slot), which is
/// the only way the layout changes, so a stale schedule can never be served. NOT thread-safe (the
/// fill_boundary path is driven from a single host thread; Kokkos parallelism lives inside for_each).
class HaloScheduleCache {
 public:
  /// Existing schedule for (px, py, dom), or nullptr if none is cached yet.
  std::shared_ptr<const HaloSchedule> find(bool px, bool py, const Box2D& dom) const {
    for (const auto& s : entries_) {
      if (s->per_x == px && s->per_y == py && s->domain == dom) {
        return s;
      }
    }
    return nullptr;
  }

  /// Appends a fresh, empty schedule and returns it for the caller to populate.
  std::shared_ptr<HaloSchedule> add() {
    entries_.push_back(std::make_shared<HaloSchedule>());
    return entries_.back();
  }

  /// Drops every cached schedule, forcing a rebuild on the next fill_boundary. Used by tests to
  /// compare the cached path against a fresh rebuild; not needed in production (regrid drops the
  /// whole cache by reassigning the MultiFab).
  void clear() { entries_.clear(); }

  /// Number of cached schedules (test/instrumentation hook).
  std::size_t size() const { return entries_.size(); }

 private:
  std::vector<std::shared_ptr<HaloSchedule>> entries_;
};

namespace detail {
/// Process-wide count of halo-schedule (re)builds. A single instance across translation units (an
/// inline function with a function-local static). NOT thread-safe; instrumentation only.
inline std::int64_t& halo_schedule_build_counter() {
  static std::int64_t n = 0;
  return n;
}
}  // namespace detail

/// Number of times fill_boundary has BUILT (enumerated) a halo schedule. A reused (cached) schedule
/// does NOT increment it, so a stable layout filled K times reports 1. Test hook for cache
/// engagement; not part of the public numerical API.
inline std::int64_t halo_schedule_build_count() {
  return detail::halo_schedule_build_counter();
}

/// Resets the build counter (tests).
inline void reset_halo_schedule_build_count() {
  detail::halo_schedule_build_counter() = 0;
}

}  // namespace adc
