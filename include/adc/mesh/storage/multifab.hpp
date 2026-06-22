/// @file
/// @brief MultiFab: a field DISTRIBUTED over a level (equivalent of AMReX's MultiFab).
///
/// Carries the decomposition (BoxArray), the distribution (DistributionMapping), the component and
/// ghost counts, and allocates only the Fab2D OWNED by this rank. This is where data parallelism
/// lives; the physics layer never sees it. Iteration runs over the LOCAL fabs:
/// for (int li = 0; li < mf.local_size(); ++li) { auto a = mf.fab(li).array(); for_each_cell(...); }.
/// sync_host()/sync_device() encode the access intent (data residence, see for_each.hpp); under
/// unified memory sync_host = a targeted device_fence(), sync_device = no-op. sum() reduces over all
/// ranks (all_reduce): Kokkos::Sum reassociates per tile (deterministic/idempotent, not bit-identical
/// to a lexicographic sum).

#pragma once

#include <adc/core/foundation/types.hpp>
#include <adc/mesh/index/box2d.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/layout/distribution_mapping.hpp>
#include <adc/mesh/storage/fab2d.hpp>
#include <adc/mesh/execution/for_each.hpp>       // device_fence, sync_host, sync_device
#include <adc/mesh/boundary/halo_schedule.hpp>  // memoized fill_boundary schedule (ADC-260)
#include <adc/parallel/comm.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace adc {

/// Field distributed over a level: decomposition (BoxArray) + distribution (DistributionMapping) +
/// ncomp components + ngrow ghosts. Allocates only the fabs owned by THIS rank; iteration runs over
/// local_size() (LOCAL indices). global_index/local_index_of bridge local <-> global.
class MultiFab {
 public:
  MultiFab() = default;

  /// Builds the field: allocates one Fab2D (ncomp components, ngrow ghosts) for EACH box that this
  /// rank owns according to dm. Boxes belonging to other ranks are not allocated here.
  MultiFab(BoxArray ba, DistributionMapping dm, int ncomp, int ngrow)
      : ba_(std::move(ba)),
        dm_(std::move(dm)),
        ncomp_(ncomp),
        ngrow_(ngrow),
        local_index_(ba_.size(), -1) {
    const int me = my_rank();
    for (int i = 0; i < ba_.size(); ++i) {
      if (dm_[i] == me) {
        local_index_[i] = static_cast<int>(fabs_.size());
        global_of_local_.push_back(i);
        fabs_.emplace_back(ba_[i], ncomp_, ngrow_);
      }
    }
  }

  /// GLOBAL decomposition of the level (all boxes, all ranks).
  const BoxArray& box_array() const { return ba_; }
  /// GLOBAL distribution (owner rank per box).
  const DistributionMapping& dmap() const { return dm_; }
  /// Number of components.
  int ncomp() const { return ncomp_; }
  /// Number of ghost layers.
  int n_grow() const { return ngrow_; }

  /// Number of fabs OWNED by this rank (bound on local indices).
  int local_size() const { return static_cast<int>(fabs_.size()); }
  /// Local fab at index li (0 <= li < local_size()), for writing.
  Fab2D& fab(int li) { return fabs_[li]; }
  /// Local fab at index li, for reading.
  const Fab2D& fab(int li) const { return fabs_[li]; }
  /// VALID box of local fab li.
  const Box2D& box(int li) const { return fabs_[li].box(); }
  /// GLOBAL index (in box_array) of local fab li.
  int global_index(int li) const { return global_of_local_[li]; }
  /// LOCAL index of the global box @p global, or -1 if it is not owned by this rank.
  int local_index_of(int global) const { return local_index_[global]; }

  /// Makes the HOST residence valid (before a host access: operator(), loop, set_val). Under unified
  /// memory = a targeted device_fence().
  void sync_host() { adc::sync_host(); }
  /// Marks a DEVICE residence (before a kernel). No-op under unified memory.
  void sync_device() { adc::sync_device(); }

  /// Fills all cells (valid + ghosts) of every local fab with v. Synchronizes host residence first
  /// (a kernel may have written these fabs).
  void set_val(Real v) {
    sync_host();  // a kernel may have written these fabs; make the host residence
                  // valid before the host fill (otherwise a host/kernel write
                  // race). Under unified memory = a device_fence().
    for (auto& f : fabs_)
      f.set_val(v);
  }

  /// Internal (ADC-260): memoized halo-exchange schedule used by fill_boundary. Lazily created on
  /// first use. The schedule is a pure function of (box_array, dmap, n_grow) for a given
  /// (Periodicity, domain); since none of ba_/dm_/ngrow_ has an in-place setter, the cache can only
  /// go stale through whole-object (re)assignment (e.g. AMR regrid builds a fresh MultiFab and
  /// move-assigns it over the level slot), which drops the cache with the object. It is shared on
  /// copy (a copy has the same layout), which keeps copies consistent. Not part of the public
  /// numerical API. Returned by reference so fill_boundary can populate it.
  HaloScheduleCache& halo_cache() const {
    if (!halo_cache_)
      halo_cache_ = std::make_shared<HaloScheduleCache>();
    return *halo_cache_;
  }

 private:
  BoxArray ba_{};
  DistributionMapping dm_{};
  int ncomp_{1};
  int ngrow_{0};
  std::vector<Fab2D> fabs_{};           // locally owned fabs
  std::vector<int> local_index_{};      // global box -> local index (-1 otherwise)
  std::vector<int> global_of_local_{};  // local index -> global box
  // Memoized fill_boundary schedule (ADC-260). mutable: caching is logically const; lazily built.
  mutable std::shared_ptr<HaloScheduleCache> halo_cache_{};
};

/// Sum of the VALID cells of component comp, reduced over ALL ranks (all_reduce). COLLECTIVE under
/// MPI. FP NOTE: Kokkos::Sum reassociates per tile (deterministic/idempotent, not bit-identical to a
/// lexicographic sum).
inline Real sum(const MultiFab& mf, int comp = 0) {
  Real s = 0;
  for (int li = 0; li < mf.local_size(); ++li) {
    const ConstArray4 a = mf.fab(li).const_array();
    s += for_each_cell_reduce_sum(mf.box(li),
                                  [a, comp] ADC_HD(int i, int j) { return a(i, j, comp); });
  }
  return static_cast<Real>(all_reduce_sum(static_cast<double>(s)));
}

}  // namespace adc
