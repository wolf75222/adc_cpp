/// @file
/// @brief AmrHierarchy: the stack of refinement levels (container of the AMR hierarchy).
///
/// Layer: `include/adc/amr` (AMR geometric primitives).
/// Role: carries, per level, the domain in index space, the BoxArray and the MultiFab field.
/// Level 0 = the coarsest; fixed integer refinement ratio (2 by default).
/// Contract: pure container; POPULATING the fine levels (tagging + Berger-Rigoutsos clustering)
/// is the job of regrid.hpp. Here we provide the explicit add/replace of a level, sufficient
/// for static refinement.
///
/// Invariants:
/// - domain(lev) == domain(lev-1).refine(ref_ratio): domains nested by the fixed ratio;
/// - the three vectors (domain_, ba_, data_) always have the same size = num_levels();
/// - replacing or clearing a level invalidates and removes all finer levels.

#pragma once

#include <adc/amr/hierarchy/refinement_ratio.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/parallel/comm.hpp>

#include <cassert>
#include <utility>
#include <vector>

namespace adc {

/// Stack of refined levels (domain + BoxArray + MultiFab per level), level 0 the coarsest.
///
/// Usage: built with only level 0 (coarse), then expanded by add_level / install_level
/// (static) or by regrid_level (dynamic).
/// Contract: fixed integer ref_ratio; domain(lev) follows from domain(lev-1) by refine(ref_ratio).
/// Invariants: domain_, ba_ and data_ stay of length num_levels(); install/clear truncate the
/// finer levels to keep the stack consistent.
class AmrHierarchy {
 public:
  /// Builds the hierarchy with only its level 0 (coarse).
  /// @param coarse_domain domain of level 0 in index space (INCLUSIVE lo/hi corners).
  /// @param max_grid_size max box size for the split into BoxArray.
  /// @param ncomp number of field components per cell.
  /// @param ngrow number of ghost layers of the MultiFab.
  /// @param ref_ratio integer refinement ratio; only kAmrRefRatio (2) is supported today
  ///        and any other value is rejected at construction (see refinement_ratio.hpp).
  AmrHierarchy(const Box2D& coarse_domain, int max_grid_size, int ncomp, int ngrow,
               int ref_ratio = kAmrRefRatio)
      : ref_ratio_(ref_ratio), ncomp_(ncomp), ngrow_(ngrow) {
    require_supported_ref_ratio(ref_ratio);
    BoxArray ba = BoxArray::from_domain(coarse_domain, max_grid_size);
    domain_.push_back(coarse_domain);
    ba_.push_back(ba);
    data_.emplace_back(ba, DistributionMapping(ba.size(), n_ranks()), ncomp, ngrow);
  }

  /// Adds a fine level defined by its BoxArray (in fine index space).
  /// @param fine_ba boxes of the new level, expressed in the refined index space.
  /// The level domain is deduced by refine(ref_ratio) from the previous level domain.
  void add_level(const BoxArray& fine_ba) {
    const int lev = num_levels();
    domain_.push_back(domain_[lev - 1].refine(ref_ratio_));
    ba_.push_back(fine_ba);
    data_.emplace_back(fine_ba, DistributionMapping(fine_ba.size(), n_ranks()), ncomp_, ngrow_);
  }

  /// Installs (adds or replaces) a fine level at index lev. Used by the regrid.
  /// @param lev index of the level to install; must satisfy 1 <= lev <= num_levels().
  /// @param fine_ba boxes of the level in the refined index space.
  /// @param data MultiFab already built for this level (transferred by move).
  /// Replacing an existing level INVALIDATES and removes all finer levels.
  void install_level(int lev, const BoxArray& fine_ba, MultiFab data) {
    assert(lev >= 1 && lev <= num_levels());
    const Box2D dom = domain_[lev - 1].refine(ref_ratio_);
    if (lev == num_levels()) {
      domain_.push_back(dom);
      ba_.push_back(fine_ba);
      data_.push_back(std::move(data));
    } else {
      domain_[lev] = dom;
      ba_[lev] = fine_ba;
      data_[lev] = std::move(data);
      domain_.resize(lev + 1);
      ba_.resize(lev + 1);
      data_.resize(lev + 1);
    }
  }

  /// Removes all levels strictly finer than lev (no-op if lev is already the finest).
  void clear_above(int lev) {
    if (lev + 1 < num_levels()) {
      domain_.resize(lev + 1);
      ba_.resize(lev + 1);
      data_.resize(lev + 1);
    }
  }

  /// Number of levels present (>= 1: level 0 always exists).
  int num_levels() const { return static_cast<int>(data_.size()); }
  /// Integer refinement ratio between consecutive levels.
  int ref_ratio() const { return ref_ratio_; }
  /// Number of field components per cell.
  int ncomp() const { return ncomp_; }
  /// Number of ghost layers of the MultiFab.
  int n_grow() const { return ngrow_; }

  /// Domain of level lev in index space (INCLUSIVE lo/hi corners).
  const Box2D& domain(int lev) const { return domain_[lev]; }
  /// BoxArray (split into boxes) of level lev.
  const BoxArray& boxes(int lev) const { return ba_[lev]; }
  /// Field of level lev (mutable access).
  MultiFab& data(int lev) { return data_[lev]; }
  /// Field of level lev (const access).
  const MultiFab& data(int lev) const { return data_[lev]; }

 private:
  int ref_ratio_;
  int ncomp_;
  int ngrow_;
  std::vector<Box2D> domain_{};
  std::vector<BoxArray> ba_{};
  std::vector<MultiFab> data_{};
};

}  // namespace adc
