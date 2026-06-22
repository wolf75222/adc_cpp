/// @file
/// @brief Fab2D: single-grid data on a Box2D (in-house equivalent of AMReX's FArrayBox);
///        Array4 / ConstArray4: lightweight POD device-copyable handles over its buffer.
///
/// Contiguous buffer covering the VALID box grown by n_ghost layers, with n_comp components.
/// Component-SLOW layout (like AMReX Array4): for a given component the (i, j) plane is
/// contiguous with i as the fast index -> each variable is a contiguous SoA slice (good
/// per-variable vectorization). Index: c*(nx_tot*ny_tot) + (j-jg0)*nx_tot + (i-ig0). Array4 /
/// ConstArray4 are lightweight handles (raw pointer + strides), trivially copyable and
/// capturable BY VALUE in a functor (Kokkos view semantics): you capture the handle, not the
/// Fab; operator() is ADC_HD (device-callable). Storage lives in UNIFIED memory (cf.
/// allocator.hpp).

#pragma once

#include <adc/core/foundation/allocator.hpp>
#include <adc/core/foundation/types.hpp>
#include <adc/mesh/index/box2d.hpp>

#include <cassert>
#include <cstdint>
#include <vector>

namespace adc {

/// WRITE POD handle (raw pointer + strides) over a Fab2D buffer, indexed by (i, j, c)
/// IN GLOBAL INDICES (ig0/jg0 = lower corner of the grown box). Trivially copyable, capturable by
/// value in a device kernel. INVARIANT: owns NOTHING; valid as long as the source Fab is.
struct Array4 {
  Real* p{nullptr};
  int nx_tot{0};
  std::int64_t comp_stride{0};
  int ig0{0}, jg0{0};  // global indices of the lower corner of the grown box

  /// Reference to cell (i, j) of component c (global indices). ADC_HD. No bounds checking
  /// (hot path / device): the caller guarantees (i, j, c) is inside the grown box.
  ADC_HD Real& operator()(int i, int j, int c = 0) const {
    return p[c * comp_stride + static_cast<std::int64_t>(j - jg0) * nx_tot + (i - ig0)];
  }
};

/// READ-only handle (const counterpart of Array4): same layout and same contract (POD
/// device-copyable, global indices, no bounds checking). ADC_HD.
struct ConstArray4 {
  const Real* p{nullptr};
  int nx_tot{0};
  std::int64_t comp_stride{0};
  int ig0{0}, jg0{0};

  /// Value of cell (i, j) of component c (global indices). ADC_HD, no bounds checking.
  ADC_HD Real operator()(int i, int j, int c = 0) const {
    return p[c * comp_stride + static_cast<std::int64_t>(j - jg0) * nx_tot + (i - ig0)];
  }
};

/// Single-grid data on a Box2D: VALID box + ng ghost layers, ncomp components, component-slow
/// layout. OWNS its buffer (unified memory). Exposes Array4 / ConstArray4 handles to kernels
/// (capture by value), never the Fab itself.
class Fab2D {
 public:
  Fab2D() = default;

  /// Allocates the valid box grown by ng ghosts, ncomp components, initialized to 0.
  Fab2D(const Box2D& valid, int ncomp, int ng)
      : valid_(valid),
        ng_(ng),
        ncomp_(ncomp),
        gbox_(valid.grow(ng)),
        nx_tot_(gbox_.nx()),
        ny_tot_(gbox_.ny()),
        data_(static_cast<std::int64_t>(nx_tot_) * ny_tot_ * ncomp, Real{0}) {}

  /// VALID box (without ghosts).
  const Box2D& box() const { return valid_; }
  /// Grown box (valid + ng ghosts) = actual memory footprint.
  const Box2D& grown_box() const { return gbox_; }
  /// Number of components.
  int ncomp() const { return ncomp_; }
  /// Number of ghost layers.
  int n_ghost() const { return ng_; }
  /// Buffer size (nx_tot * ny_tot * ncomp).
  std::int64_t size() const { return static_cast<std::int64_t>(data_.size()); }

  /// HOST write access (i, j, c) (bounds assert in debug). Do not call inside a device kernel:
  /// go through array() (POD handle).
  Real& operator()(int i, int j, int c = 0) { return data_[idx(i, j, c)]; }
  /// HOST read access (i, j, c) (bounds assert in debug).
  Real operator()(int i, int j, int c = 0) const { return data_[idx(i, j, c)]; }

  /// WRITE handle (POD device-copyable) over this Fab. Valid as long as the Fab lives.
  Array4 array() {
    return Array4{data_.data(), nx_tot_, static_cast<std::int64_t>(nx_tot_) * ny_tot_, gbox_.lo[0],
                  gbox_.lo[1]};
  }
  /// READ handle (POD device-copyable) over this Fab. Valid as long as the Fab lives.
  ConstArray4 const_array() const {
    return ConstArray4{data_.data(), nx_tot_, static_cast<std::int64_t>(nx_tot_) * ny_tot_,
                       gbox_.lo[0], gbox_.lo[1]};
  }

  /// Raw pointer to the buffer (passed directly to MPI in unified memory, for instance).
  Real* data() { return data_.data(); }
  const Real* data() const { return data_.data(); }
  /// Fills the whole buffer (valid + ghosts) with value v.
  void set_val(Real v) { std::fill(data_.begin(), data_.end(), v); }

 private:
  // linear index (i, j, c) in the component-slow layout; bounds assert in debug.
  std::int64_t idx(int i, int j, int c) const {
    assert(gbox_.contains(i, j) && c >= 0 && c < ncomp_);
    return c * static_cast<std::int64_t>(nx_tot_) * ny_tot_ +
           static_cast<std::int64_t>(j - gbox_.lo[1]) * nx_tot_ + (i - gbox_.lo[0]);
  }

  Box2D valid_{};
  int ng_{0};
  int ncomp_{1};
  Box2D gbox_{};
  int nx_tot_{0}, ny_tot_{0};
  // storage: host (std::allocator) or CUDA unified memory (cf. allocator.hpp).
  std::vector<Real, fab_allocator<Real>> data_{};
};

}  // namespace adc
