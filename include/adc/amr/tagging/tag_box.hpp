/// @file
/// @brief TagBox: dense grid of markers (0/1) over a region, input to Berger-Rigoutsos clustering.
///
/// Layer: `include/adc/amr` (AMR geometric primitives).
/// Role: marker structure for cells to refine, produced by tagging and consumed by
/// berger_rigoutsos. Pure integer arithmetic on indices, no physics, no parallelism.
/// Contract: indexed in the index space of its box (lo/hi corners INCLUSIVE, Box2D convention);
/// dense i-fast (j-slow) storage.
///
/// Invariants:
/// - a TagBox covers EXACTLY its box; linear indexing assumes i in [lo[0], hi[0]] and
///   j in [lo[1], hi[1]];
/// - markers are 0 (not tagged) or 1 (tagged);
/// - for MPI later, distributed tags will be gathered onto this grid before clustering (clustering
///   is cheap compared to the rest).

#pragma once

#include <adc/mesh/box2d.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace adc {

/// Dense grid of 0/1 markers over a box, input to Berger-Rigoutsos clustering.
///
/// Usage: filled by tagging (tag_cells), optionally dilated (grow_tags) then merged
/// (tag_union), finally clustered (berger_rigoutsos).
/// Contract: accesses (i, j) are in the index space of `box` (lo/hi corners INCLUSIVE).
/// Invariants: `t` has exactly box.num_cells() entries (0 if the box is empty), rows i-fast.
struct TagBox {
  Box2D box{};
  std::vector<char> t{};

  TagBox() = default;
  /// Build a TagBox covering b, all markers at 0 (buffer sized on b.num_cells()).
  explicit TagBox(const Box2D& b)
      : box(b), t(static_cast<std::size_t>(std::max<std::int64_t>(0, b.num_cells())), 0) {}

  /// Write access to marker (i, j); (i, j) MUST be in box (no bound check).
  char& operator()(int i, int j) { return t[idx(i, j)]; }
  /// Read access to marker (i, j); (i, j) MUST be in box (no bound check).
  char operator()(int i, int j) const { return t[idx(i, j)]; }
  /// true if (i, j) is in box AND tagged; safe (bound check included, unlike operator()).
  bool tagged(int i, int j) const { return box.contains(i, j) && t[idx(i, j)] != 0; }

  /// Number of tagged cells (sum of markers).
  std::int64_t count() const {
    std::int64_t c = 0;
    for (char x : t)
      c += x;
    return c;
  }

 private:
  // i-fast linear index of marker (i, j) in `t`; assumes (i, j) in box.
  std::size_t idx(int i, int j) const {
    return static_cast<std::size_t>(j - box.lo[1]) * box.nx() + (i - box.lo[0]);
  }
};

/// Union (cell-by-cell logical OR) of several TagBox sharing EXACTLY the same box.
///
/// Building block of the multi-block tag-union regrid (conservative regrid = common hierarchy,
/// co-located cells, union of tags; docs/AMR_REGRID_UNION_TAGS_DESIGN.md step R3).
/// @param parts TagBox to merge; all MUST cover the same parent domain.
/// @return TagBox on the common box, marked where at least one member was; empty list -> empty TagBox.
/// @throws std::runtime_error if a box disagrees (linear indexing would mix two geometries).
// No physics dependency (a few lines): i-fast storage -> simple |= over the buffer.
inline TagBox tag_union(const std::vector<TagBox>& parts) {
  if (parts.empty())
    return TagBox{};
  TagBox out(parts[0].box);
  for (const TagBox& tb : parts) {
    if (tb.box.lo[0] != out.box.lo[0] || tb.box.lo[1] != out.box.lo[1] ||
        tb.box.hi[0] != out.box.hi[0] || tb.box.hi[1] != out.box.hi[1])
      throw std::runtime_error(
          "tag_union: all TagBox must share EXACTLY the same box (same parent "
          "domain) for cell-by-cell union");
    const std::size_t n = std::min(out.t.size(), tb.t.size());
    for (std::size_t k = 0; k < n; ++k)
      out.t[k] |= tb.t[k];
  }
  return out;
}

}  // namespace adc
