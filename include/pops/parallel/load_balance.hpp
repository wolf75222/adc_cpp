#pragma once

#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

/// @file
/// @brief AMR load balancing: builds a DistributionMapping (box -> rank) from a
///        BoxArray, using a Z-order space-filling curve (SFC) or LPT knapsack.
///
/// Layer: `include/pops/parallel`.
/// Role: distribute boxes over ranks with replicated metadata (AMReX style). Two
/// strategies: make_sfc_distribution (contiguous segments of equal load along the
/// Morton curve -> spatial locality) and make_knapsack_distribution (heaviest box
/// to the least loaded rank -> minimizes the maximum imbalance). load_imbalance
/// measures the max load / average load ratio. Morton tooling exposed: part1by1,
/// morton_key, morton_order.
/// Contract: a box weight is its cell count (proxy for compute cost).
///
/// Invariants:
/// - PURE functions, no MPI: testable in serial; they will feed the comm seam once
///   an MPI backend is wired in;
/// - SFC guarantees that with nboxes >= nranks each rank receives at least one box;
/// - degenerate cases (n == 0, nranks <= 1): everything is assigned to rank 0.

namespace pops {

// Spread the bits of x (16 useful bits) onto the even positions of a 64-bit word.
inline std::uint64_t part1by1(std::uint64_t x) {
  x &= 0xffffffffULL;
  x = (x | (x << 16)) & 0x0000ffff0000ffffULL;
  x = (x | (x << 8)) & 0x00ff00ff00ff00ffULL;
  x = (x | (x << 4)) & 0x0f0f0f0f0f0f0f0fULL;
  x = (x | (x << 2)) & 0x3333333333333333ULL;
  x = (x | (x << 1)) & 0x5555555555555555ULL;
  return x;
}

// Morton key (Z-order) interleaving (x, y): x on the even bits, y on the odd bits.
inline std::uint64_t morton_key(std::uint32_t x, std::uint32_t y) {
  return part1by1(x) | (part1by1(y) << 1);
}

// Box indices sorted along the Morton curve (low corner, shifted by the bounding
// box to stay positive).
inline std::vector<int> morton_order(const BoxArray& ba) {
  const int n = ba.size();
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  if (n == 0)
    return order;
  const Box2D bb = ba.bounding_box();
  std::vector<std::uint64_t> key(n);
  for (int i = 0; i < n; ++i)
    key[i] = morton_key(static_cast<std::uint32_t>(ba[i].lo[0] - bb.lo[0]),
                        static_cast<std::uint32_t>(ba[i].lo[1] - bb.lo[1]));
  std::sort(order.begin(), order.end(), [&](int a, int b) { return key[a] < key[b]; });
  return order;
}

// Z-order distribution: contiguous segments of ~equal load along the SFC.
// Guarantees that with nboxes >= nranks each rank receives at least one box.
inline DistributionMapping make_sfc_distribution(const BoxArray& ba, int nranks) {
  const int n = ba.size();
  std::vector<int> rank(n, 0);
  if (n == 0 || nranks <= 1)
    return DistributionMapping(std::move(rank));

  const std::vector<int> order = morton_order(ba);
  std::int64_t total = ba.num_cells();
  const double target = double(total) / nranks;  // target load per rank

  std::int64_t acc = 0;
  int r = 0;
  for (int k = 0; k < n; ++k) {
    const int b = order[k];
    rank[b] = r;
    acc += ba[b].num_cells();
    // advance to the next rank if the target share is reached AND enough boxes
    // remain to give at least one box to each remaining rank.
    const int boxes_left = n - 1 - k;
    const int ranks_left = nranks - 1 - r;
    if (r < nranks - 1 && acc >= target * (r + 1) && boxes_left >= ranks_left)
      ++r;
  }
  return DistributionMapping(std::move(rank));
}

// Knapsack distribution (LPT): heaviest box -> least loaded rank.
inline DistributionMapping make_knapsack_distribution(const BoxArray& ba, int nranks) {
  const int n = ba.size();
  std::vector<int> rank(n, 0);
  if (n == 0 || nranks <= 1)
    return DistributionMapping(std::move(rank));

  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return ba[a].num_cells() > ba[b].num_cells(); });

  std::vector<std::int64_t> load(nranks, 0);
  for (int b : order) {
    int r = 0;
    for (int q = 1; q < nranks; ++q)
      if (load[q] < load[r])
        r = q;
    rank[b] = r;
    load[r] += ba[b].num_cells();
  }
  return DistributionMapping(std::move(rank));
}

// Imbalance = max load / average load (1.0 = perfect).
inline double load_imbalance(const BoxArray& ba, const DistributionMapping& dm, int nranks) {
  if (nranks <= 0 || ba.size() == 0)
    return 1.0;
  std::vector<std::int64_t> load(nranks, 0);
  for (int i = 0; i < ba.size(); ++i)
    load[dm[i]] += ba[i].num_cells();
  std::int64_t mx = 0, sum = 0;
  for (std::int64_t l : load) {
    mx = std::max(mx, l);
    sum += l;
  }
  const double avg = double(sum) / nranks;
  return avg > 0 ? mx / avg : 1.0;
}

}  // namespace pops
