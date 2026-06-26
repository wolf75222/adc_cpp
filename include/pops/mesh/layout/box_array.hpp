/// @file
/// @brief BoxArray: the set of boxes tiling a level (disjoint, covering).
///
/// Equivalent of AMReX's BoxArray. from_domain splits a domain into tiles of at most
/// max_grid_size per direction, distributed as EVENLY as possible (better balancing than
/// greedy chunks). Carries NO field data and no MPI distribution (cf. MultiFab /
/// DistributionMapping): it is only the geometric decomposition of the level.

#pragma once

#include <pops/mesh/index/box2d.hpp>

#include <cstdint>
#include <utility>
#include <vector>

namespace pops {

/// Ordered list of boxes tiling a level. Expected INVARIANT (not checked): boxes
/// disjoint and covering the domain. The ORDER is significant (global box index = position
/// in the vector; shared by MultiFab / DistributionMapping). Copyable (vector of Box2D).
class BoxArray {
 public:
  BoxArray() = default;
  /// Build from an already-computed list of boxes (move). The order is kept as is.
  explicit BoxArray(std::vector<Box2D> boxes) : boxes_(std::move(boxes)) {}

  /// Tile the domain into tiles of at most max_grid_size per direction, distributed evenly.
  /// Traversal order is y outer, x inner (deterministic, identical on all ranks).
  static BoxArray from_domain(const Box2D& domain, int max_grid_size) {
    auto sx = split_range(domain.lo[0], domain.hi[0], max_grid_size);
    auto sy = split_range(domain.lo[1], domain.hi[1], max_grid_size);
    std::vector<Box2D> boxes;
    boxes.reserve(sx.size() * sy.size());
    for (auto [ylo, yhi] : sy)
      for (auto [xlo, xhi] : sx)
        boxes.push_back(Box2D{{xlo, ylo}, {xhi, yhi}});
    return BoxArray{std::move(boxes)};
  }

  /// Number of boxes in the tiling.
  int size() const { return static_cast<int>(boxes_.size()); }
  /// Box at global index i (0 <= i < size()); the index is the box identity throughout the code.
  const Box2D& operator[](int i) const { return boxes_[i]; }
  /// View on the underlying vector (element-by-element equality = same boxes AND same order).
  const std::vector<Box2D>& boxes() const { return boxes_; }

  /// Total number of valid cells (sum of num_cells over all boxes).
  std::int64_t num_cells() const {
    std::int64_t n = 0;
    for (const auto& b : boxes_)
      n += b.num_cells();
    return n;
  }

  /// Smallest box enclosing all boxes (empty box if the tiling is empty).
  Box2D bounding_box() const {
    if (boxes_.empty())
      return Box2D{};
    Box2D b = boxes_[0];
    for (const auto& o : boxes_) {
      b.lo[0] = std::min(b.lo[0], o.lo[0]);
      b.lo[1] = std::min(b.lo[1], o.lo[1]);
      b.hi[0] = std::max(b.hi[0], o.hi[0]);
      b.hi[1] = std::max(b.hi[1], o.hi[1]);
    }
    return b;
  }

 private:
  // Split [lo, hi] into segments of length <= m, distributed evenly:
  // n = ceil(len/m) segments, the first `rem` of them one notch longer.
  static std::vector<std::pair<int, int>> split_range(int lo, int hi, int m) {
    std::vector<std::pair<int, int>> segs;
    int len = hi - lo + 1;
    if (len <= 0 || m <= 0)
      return segs;
    int n = (len + m - 1) / m;
    int base = len / n, rem = len % n;
    int cur = lo;
    for (int k = 0; k < n; ++k) {
      int l = base + (k < rem ? 1 : 0);
      segs.push_back({cur, cur + l - 1});
      cur += l;
    }
    return segs;
  }

  std::vector<Box2D> boxes_{};
};

}  // namespace pops
