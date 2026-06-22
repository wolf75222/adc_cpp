/// @file
/// @brief Box2D: the integer index space of a 2D cell-centered Cartesian grid.
///
/// Building block of the AMR stack, inspired by AMReX's Box. Corners lo / hi INCLUSIVE (AMReX
/// convention); box EMPTY if hi < lo along a direction. Pure integer arithmetic: no data, no
/// parallelism, fully testable. Indices may be NEGATIVE (ghost layers), hence the FLOOR division
/// in coarsen (consistent on both sides of zero). length/nx/ny are ADC_HD (called from
/// Geometry::dx()/dy() inside a device kernel). Concrete 2D to match the physical targets: 2D is an
/// official, introspectable invariant of the core (adc.capabilities()["dimension"] == 2, ADR-0001
/// Decision 1; see docs/sphinx/reference/known-limitations.md), not an oversight. The move to a
/// Dim-template (BoxND) is a generalization deferred to a future milestone (Option B).

#pragma once

#include <adc/core/foundation/types.hpp>  // ADC_HD: nx/ny/length called from Geometry::dx() inside a device kernel

#include <algorithm>
#include <cstdint>

namespace adc {

// Integer division rounded down (toward -inf), consistent on both sides of zero: the only correct
// division for NEGATIVE indices (ghost layers) during coarsen / spatial hash. C++ division truncates
// toward zero; we subtract 1 when the remainder is non-zero and of opposite sign to the divisor (the
// truncated quotient was then rounded up). Shared low-level building block (Box2D coarsen, BoxHash bin
// hashing, coarse->fine indices in refinement.hpp). b != 0 expected.
/// Integer division of a by b rounded down (handles a < 0 AND b < 0). ADC_HD constexpr (kernels).
ADC_HD constexpr int floor_div(int a, int b) {
  const int q = a / b, rem = a % b;
  return (rem != 0 && ((rem < 0) != (b < 0))) ? q - 1 : q;
}

/// 2D integer index space, cell-centered. Corners lo/hi INCLUSIVE; box empty if hi < lo.
/// Pure POD (no field data): trivially copyable, capturable by value inside a kernel.
/// INVARIANT: indices may be negative (ghosts); refine/coarsen are block-wise bijections
/// (refine then coarsen gives back the box, but coarsen then refine rounds it to the block).
struct Box2D {
  int lo[2]{0, 0};
  int hi[2]{-1, -1};  // empty by default (hi < lo)

  /// Box [0, nx-1] x [0, ny-1] covering nx*ny cells from the index origin.
  static Box2D from_extents(int nx, int ny) { return Box2D{{0, 0}, {nx - 1, ny - 1}}; }

  // ADC_HD: Geometry::dx()/dy() (themselves ADC_HD) read domain.nx()/ny(); a device kernel that
  // calls geom.x_cell(i) descends down to here. Without ADC_HD this is a __host__ from __device__ ->
  // nvcc yields GARBAGE (often 0) with no error. Pure integer arithmetic, device-safe, host unchanged.
  /// Number of cells in direction d (= hi[d] - lo[d] + 1); negative if the box is empty. ADC_HD.
  ADC_HD int length(int d) const { return hi[d] - lo[d] + 1; }
  /// Width (direction 0). ADC_HD (called from Geometry::dx() in a device kernel).
  ADC_HD int nx() const { return length(0); }
  /// Height (direction 1). ADC_HD (called from Geometry::dy() in a device kernel).
  ADC_HD int ny() const { return length(1); }
  /// Total number of cells (nx*ny, floored at 0 per direction): 0 if the box is empty.
  std::int64_t num_cells() const {
    return static_cast<std::int64_t>(std::max(0, nx())) * std::max(0, ny());
  }
  /// true if the box contains no cell (hi < lo in one direction).
  bool empty() const { return hi[0] < lo[0] || hi[1] < lo[1]; }

  /// true if cell (i, j) is inside the box (lo/hi bounds inclusive).
  bool contains(int i, int j) const { return i >= lo[0] && i <= hi[0] && j >= lo[1] && j <= hi[1]; }
  /// true if box b (non-empty) is entirely contained in *this.
  bool contains(const Box2D& b) const {
    return !b.empty() && b.lo[0] >= lo[0] && b.hi[0] <= hi[0] && b.lo[1] >= lo[1] &&
           b.hi[1] <= hi[1];
  }

  /// Grows the box by n cells in ALL directions (uniform ghost layer).
  Box2D grow(int n) const { return {{lo[0] - n, lo[1] - n}, {hi[0] + n, hi[1] + n}}; }
  /// Grows by n cells in the SINGLE direction d (n may be negative to shrink).
  Box2D grow(int d, int n) const {
    Box2D b = *this;
    b.lo[d] -= n;
    b.hi[d] += n;
    return b;
  }
  /// Translates the box by s cells in direction d (lo and hi shifted by the same s).
  Box2D shift(int d, int s) const {
    Box2D b = *this;
    b.lo[d] += s;
    b.hi[d] += s;
    return b;
  }

  /// Refines by a ratio r: each cell becomes an r x r block ([lo, hi] -> [lo*r, hi*r + r-1]).
  Box2D refine(int r) const {
    return {{lo[0] * r, lo[1] * r}, {hi[0] * r + r - 1, hi[1] * r + r - 1}};
  }
  /// Coarsens by a ratio r via FLOOR division of each corner (handles the negative ghost indices).
  Box2D coarsen(int r) const {
    return {{floor_div(lo[0], r), floor_div(lo[1], r)}, {floor_div(hi[0], r), floor_div(hi[1], r)}};
  }

  /// Intersection of the two boxes (possibly empty: hi < lo if they do not overlap).
  Box2D intersect(const Box2D& o) const {
    return {{std::max(lo[0], o.lo[0]), std::max(lo[1], o.lo[1])},
            {std::min(hi[0], o.hi[0]), std::min(hi[1], o.hi[1])}};
  }

  bool operator==(const Box2D&) const = default;
};

}  // namespace adc
