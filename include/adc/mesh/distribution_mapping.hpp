#pragma once

#include <utility>
#include <vector>

// DistributionMapping : associe a chaque box (par indice global) le rang MPI
// qui la possede. Le ctor (nboxes, nranks) donne un round-robin ; les strategies
// equilibrees (Z-order SFC, knapsack) sont dans parallel/load_balance.hpp et
// construisent un DistributionMapping a partir d'un vecteur de rangs explicite.
// Separe de BoxArray (convention AMReX) : on peut redistribuer sans rebatir le
// decoupage.

namespace adc {

class DistributionMapping {
 public:
  DistributionMapping() = default;

  DistributionMapping(int nboxes, int nranks) {
    rank_.resize(nboxes);
    for (int i = 0; i < nboxes; ++i) rank_[i] = (nranks > 0) ? i % nranks : 0;
  }

  explicit DistributionMapping(std::vector<int> rank) : rank_(std::move(rank)) {}

  int operator[](int i) const { return rank_[i]; }
  int size() const { return static_cast<int>(rank_.size()); }
  const std::vector<int>& ranks() const { return rank_; }

 private:
  std::vector<int> rank_{};
};

}  // namespace adc
