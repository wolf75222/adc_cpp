/// @file
/// @brief Dynamic regrid: (re)builds a fine level from the tagging of a coarse level.
///
/// Layer: `include/adc/amr` (AMR geometric primitives).
/// Role: tags a level, clusters the tagged cells into boxes (Berger-Rigoutsos), refines into the
/// BoxArray of the next level, rebuilds its MultiFab and fills it (interpolation from the
/// coarse level, then a copy of the old fine level where it existed to preserve accuracy).
/// Contract: the tagging criterion is a generic predicate on (ConstArray4, i, j); we stay
/// agnostic of the physics. For a gradient criterion, the caller fills the ghosts beforehand.
///
/// Invariants:
/// - conservative regrid = common hierarchy, co-located cells, regrid by union of tags;
/// - the fine level lives in the refined index space of the coarse level (refine(ref_ratio));
/// - without any tag, the fine level (and the finer ones) is removed.

#pragma once

#include <adc/amr/amr_hierarchy.hpp>
#include <adc/amr/cluster.hpp>
#include <adc/amr/tag_box.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/refinement.hpp>
#include <adc/parallel/comm.hpp>

#include <utility>
#include <vector>

namespace adc {

/// Marks the valid cells where the predicate is true, on a TagBox covering the domain.
/// @tparam Crit predicate (ConstArray4, i, j) -> bool, evaluated on the valid cells of each fab.
/// @param mf source field (local: only iterates over the rank's local fabs).
/// @param domain domain covered by the returned TagBox (level index space).
/// @return TagBox over domain, marked where crit is true.
template <class Crit>
TagBox tag_cells(const MultiFab& mf, const Box2D& domain, Crit crit) {
  TagBox tb(domain);
  for (int li = 0; li < mf.local_size(); ++li) {
    const Fab2D& f = mf.fab(li);
    const ConstArray4 a = f.const_array();
    const Box2D v = f.box();
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        if (crit(a, i, j)) tb(i, j) = 1;
  }
  return tb;
}

/// Grows the tags by n cells (square neighborhood), staying within the domain.
/// @param n dilation radius (buffer); used for nesting and to anticipate the motion of structures.
/// @param domain bounds the neighborhood: no tag is placed outside the domain.
/// @return new TagBox over in.box, marked over the union of the square neighborhoods of the tagged cells.
inline TagBox grow_tags(const TagBox& in, int n, const Box2D& domain) {
  TagBox out(in.box);
  const Box2D& b = in.box;
  for (int j = b.lo[1]; j <= b.hi[1]; ++j)
    for (int i = b.lo[0]; i <= b.hi[0]; ++i)
      if (in(i, j))
        for (int dj = -n; dj <= n; ++dj)
          for (int di = -n; di <= n; ++di) {
            const int ii = i + di, jj = j + dj;
            if (b.contains(ii, jj) && domain.contains(ii, jj)) out(ii, jj) = 1;
          }
  return out;
}

/// Regrid parameters (configuration object).
struct RegridParams {
  int n_buffer = 1;        ///< tag dilation radius (grow_tags) before clustering.
  ClusterParams cluster{};  ///< Berger-Rigoutsos clustering parameters.
};

/// (Re)builds level coarse_lev+1 from the tagging of level coarse_lev.
/// @tparam Crit tagging predicate (ConstArray4, i, j) -> bool.
/// @param h hierarchy modified in place (fine level installed, or finer levels removed if no tag).
/// @param coarse_lev coarse level source of the tagging; the built fine level is coarse_lev+1.
/// @param rp parameters (tag buffer, clustering).
/// Steps: tag -> grow -> Berger-Rigoutsos -> refine(ref_ratio) -> interpolation from the coarse level,
/// then a copy of the old fine level where it existed to preserve accuracy.
template <class Crit>
void regrid_level(AmrHierarchy& h, int coarse_lev, Crit crit,
                  const RegridParams& rp = {}) {
  const Box2D cdom = h.domain(coarse_lev);
  TagBox tags = tag_cells(h.data(coarse_lev), cdom, crit);
  TagBox grown = grow_tags(tags, rp.n_buffer, cdom);

  if (grown.count() == 0) {  // nothing left to refine
    h.clear_above(coarse_lev);
    return;
  }

  // coarse boxes -> fine boxes (index space of level coarse_lev+1)
  std::vector<Box2D> cboxes = berger_rigoutsos(grown, rp.cluster);
  std::vector<Box2D> fboxes;
  fboxes.reserve(cboxes.size());
  for (const auto& b : cboxes) fboxes.push_back(b.refine(h.ref_ratio()));
  BoxArray fba(std::move(fboxes));

  MultiFab newfine(fba, DistributionMapping(fba.size(), n_ranks()), h.ncomp(),
                   h.n_grow());
  interpolate(h.data(coarse_lev), newfine, h.ref_ratio());
  if (h.num_levels() > coarse_lev + 1)
    parallel_copy(newfine, h.data(coarse_lev + 1));  // preserve the old fine level

  h.install_level(coarse_lev + 1, fba, std::move(newfine));
}

}  // namespace adc
