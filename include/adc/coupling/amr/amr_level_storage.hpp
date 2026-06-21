/// @file
/// @brief AmrLevelStack: AMR hierarchy storage (levels + aux) extracted from the couplers.
///
/// The coupler now only ORDERS the operations; this stack OWNS the level stack
/// std::vector<Level>, the parallel aux MultiFab stack, and carries the wiring L_[k].aux = &aux_[k].
/// Generic over Level (AmrLevelMF mono-box or AmrLevelMP multi-box): only the U (MultiFab)
/// and aux (const MultiFab*) members are touched here; the distribution lives in the MultiFab (no
/// mono-rank assumption). ADDRESS INVARIANT: aux_ is sized ONCE at the ctor then never
/// resized (the L_[k].aux pointers point into aux_); reattach_aux(k) replaces aux_[k] IN
/// PLACE and rewires. aux width propagated as a parameter (default kAuxBaseComps = 3, bit-identical).

#pragma once

#include <adc/core/state.hpp>  // kAuxBaseComps: default aux width (base phi/grad channel)
#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/multifab.hpp>

#include <utility>
#include <vector>

namespace adc {

/// Owns the AMR level stack and the parallel aux stack. @tparam Level: level type carrying
/// U (MultiFab) and aux (const MultiFab*) (AmrLevelMF or AmrLevelMP). INVARIANT: aux_ has a size
/// fixed at the ctor (stable addresses for L_[k].aux).
template <class Level>
class AmrLevelStack {
 public:
  /// Builds the stack: takes ownership of @p levels, allocates an aux (aux_ncomp components, 1
  /// ghost) on the layout of each U, and wires L_[k].aux = &aux_[k]. @p dom: domain of level 0.
  AmrLevelStack(const Box2D& dom, std::vector<Level> levels,
                int aux_ncomp = kAuxBaseComps)
      : dom_(dom), L_(std::move(levels)), aux_ncomp_(aux_ncomp) {
    nlev_ = static_cast<int>(L_.size());
    aux_.resize(nlev_);  // stable addresses: aux_ is no longer resized
    for (int k = 0; k < nlev_; ++k) {
      aux_[k] = MultiFab(L_[k].U.box_array(), L_[k].U.dmap(), aux_ncomp_, 1);
      L_[k].aux = &aux_[k];
    }
  }

  std::vector<Level>& levels() { return L_; }
  const std::vector<Level>& levels() const { return L_; }
  MultiFab& coarse() { return L_[0].U; }
  const MultiFab& coarse() const { return L_[0].U; }
  const Box2D& domain() const { return dom_; }
  int nlev() const { return nlev_; }

  std::vector<Level>& L() { return L_; }
  std::vector<MultiFab>& aux() { return aux_; }
  MultiFab& aux(int k) { return aux_[k]; }
  const MultiFab& aux(int k) const { return aux_[k]; }

  // aux channel width (components), as sized at the ctor.
  int aux_ncomp() const { return aux_ncomp_; }

  // In-place realloc of aux_[k] on the current box of L_[k].U + rewiring of the
  // pointer. Keeps the aux channel width (aux_ncomp_); default 3 -> bit-identical
  // to the original inline block (same MultiFab(..., 3, 1)).
  void reattach_aux(int k) {
    aux_[k] = MultiFab(L_[k].U.box_array(), L_[k].U.dmap(), aux_ncomp_, 1);
    L_[k].aux = &aux_[k];
  }

 private:
  Box2D dom_;
  std::vector<Level> L_;
  std::vector<MultiFab> aux_;
  int nlev_ = 0;
  int aux_ncomp_ = kAuxBaseComps;  // aux channel width (default: base contract)
};

}  // namespace adc
