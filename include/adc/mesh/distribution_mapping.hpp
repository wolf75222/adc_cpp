/// @file
/// @brief DistributionMapping: maps each box (by global index) to its owning MPI rank.
///
/// SEPARATE from BoxArray (AMReX convention): boxes can be redistributed without rebuilding the
/// decomposition. The (nboxes, nranks) ctor yields a round-robin; balanced strategies (Z-order SFC,
/// knapsack) live in parallel/load_balance.hpp and build a DistributionMapping from an explicit rank
/// vector. REPLICATED metadata: every rank knows the full assignment (key to the distributed
/// fill_boundary / parallel_copy paths that enumerate the same jobs on all ranks).

#pragma once

#include <utility>
#include <vector>

namespace adc {

/// Owning MPI rank of each box, indexed by GLOBAL box index (parallel to a BoxArray). Metadata
/// replicated on every rank.
class DistributionMapping {
 public:
  DistributionMapping() = default;

  /// Round-robin: box i -> rank i % nranks (rank 0 if nranks <= 0). Default distribution.
  DistributionMapping(int nboxes, int nranks) {
    rank_.resize(nboxes);
    for (int i = 0; i < nboxes; ++i)
      rank_[i] = (nranks > 0) ? i % nranks : 0;
  }

  /// EXPLICIT assignment: rank[i] = owning rank of box i (move). For external balanced strategies
  /// (load_balance.hpp).
  explicit DistributionMapping(std::vector<int> rank) : rank_(std::move(rank)) {}

  /// Owning rank of the box with global index i.
  int operator[](int i) const { return rank_[i]; }
  /// Number of boxes covered (= size of the associated BoxArray).
  int size() const { return static_cast<int>(rank_.size()); }
  /// View on the rank vector (element-by-element equality = same assignment).
  const std::vector<int>& ranks() const { return rank_; }

 private:
  std::vector<int> rank_{};
};

}  // namespace adc
