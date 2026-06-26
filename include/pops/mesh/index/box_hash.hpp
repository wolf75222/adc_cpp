/// @file
/// @brief BoxHash: spatial hash for fast lookup of boxes intersecting a query.
///
/// A uniform bin grid (classic spatial-hash technique) maps each bin to the list of boxes that
/// touch it: finding the boxes whose region may intersect a query box becomes ~O(1) per query
/// (the halo-pair search drops from O(N) to ~O(n), n << N). Built ONCE per mesh, reusable as long
/// as it does not change. Non-omission INVARIANT: if a box intersects the query, they share a cell
/// hence a bin -> query() returns a sorted SUPERSET without duplicates, the caller tests the exact
/// intersection.

#pragma once

#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace pops {

/// Spatial index of a BoxArray's boxes via a bin grid. References the BoxArray by INDEX
/// (the indices returned by query() are global indices into the original BoxArray);
/// valid as long as that BoxArray does not change.
class BoxHash {
 public:
  /// Builds the index: bin = side of a bin (cells); bin <= 0 forces bin = 1. Each box is
  /// inserted into all the bins it covers. Cost proportional to the total area in bins.
  BoxHash(const BoxArray& ba, int bin) : bin_(bin > 0 ? bin : 1) {
    for (int i = 0; i < ba.size(); ++i) {
      const Box2D& b = ba[i];
      for (int by = fdiv(b.lo[1]); by <= fdiv(b.hi[1]); ++by)
        for (int bx = fdiv(b.lo[0]); bx <= fdiv(b.hi[0]); ++bx)
          bins_[key(bx, by)].push_back(i);
    }
  }

  /// Indices (SORTED, without duplicates) of the boxes that may intersect q: guaranteed SUPERSET
  /// (no intersecting box is omitted). The caller tests the exact intersection. Empty if q is empty.
  std::vector<int> query(const Box2D& q) const {
    std::vector<int> out;
    if (q.empty())
      return out;
    for (int by = fdiv(q.lo[1]); by <= fdiv(q.hi[1]); ++by)
      for (int bx = fdiv(q.lo[0]); bx <= fdiv(q.hi[0]); ++bx) {
        auto it = bins_.find(key(bx, by));
        if (it != bins_.end())
          out.insert(out.end(), it->second.begin(), it->second.end());
      }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
  }

 private:
  // bin index = integer division by bin_ rounded down (toward -inf) (handles negative coords).
  // Thin adapter over floor_div (box2d.hpp): bin_ > 0 (forced by the constructor) -> result
  // bit-identical to the old x >= 0 ? x / bin_: -((-x + bin_ - 1) / bin_).
  int fdiv(int x) const { return floor_div(x, bin_); }
  static std::int64_t key(int bx, int by) {
    return (static_cast<std::int64_t>(bx) << 32) |
           (static_cast<std::int64_t>(static_cast<std::uint32_t>(by)) & INT64_C(0xffffffff));
  }

  int bin_;
  std::unordered_map<std::int64_t, std::vector<int>> bins_;
};

/// Recommended bin size for a BoxArray: the largest box extent (at least 1), so that
/// neighboring boxes fall into adjacent bins (memory / selectivity trade-off).
inline int suggest_bin(const BoxArray& ba) {
  int m = 1;
  for (int i = 0; i < ba.size(); ++i)
    m = std::max({m, ba[i].nx(), ba[i].ny()});
  return m;
}

}  // namespace pops
