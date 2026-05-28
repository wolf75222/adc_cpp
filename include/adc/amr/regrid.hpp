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

// Regrid dynamique : tague un niveau, regroupe les cellules taguees en boxes
// (Berger-Rigoutsos), raffine en BoxArray du niveau suivant, reconstruit son
// MultiFab et le remplit (interpolation depuis le grossier, puis copie de
// l'ancien fin la ou il existait pour preserver la precision).
//
// Le critere de tagging est un predicat generique sur (ConstArray4, i, j) : on
// reste agnostique de la physique. Pour un critere a gradient, l'appelant
// remplit les ghosts avant.

namespace adc {

// Marque les cellules valides ou le predicat est vrai, sur une TagBox couvrant
// le domaine.
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

// Dilate les tags de n cellules (voisinage carre), en restant dans le domaine.
// Sert au nesting et a anticiper le deplacement des structures.
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

struct RegridParams {
  int n_buffer = 1;
  ClusterParams cluster{};
};

template <class Crit>
void regrid_level(AmrHierarchy& h, int coarse_lev, Crit crit,
                  const RegridParams& rp = {}) {
  const Box2D cdom = h.domain(coarse_lev);
  TagBox tags = tag_cells(h.data(coarse_lev), cdom, crit);
  TagBox grown = grow_tags(tags, rp.n_buffer, cdom);

  if (grown.count() == 0) {  // plus rien a raffiner
    h.clear_above(coarse_lev);
    return;
  }

  // boxes grossieres -> boxes fines (espace d'indices du niveau coarse_lev+1)
  std::vector<Box2D> cboxes = berger_rigoutsos(grown, rp.cluster);
  std::vector<Box2D> fboxes;
  fboxes.reserve(cboxes.size());
  for (const auto& b : cboxes) fboxes.push_back(b.refine(h.ref_ratio()));
  BoxArray fba(std::move(fboxes));

  MultiFab newfine(fba, DistributionMapping(fba.size(), n_ranks()), h.ncomp(),
                   h.n_grow());
  interpolate(h.data(coarse_lev), newfine, h.ref_ratio());
  if (h.num_levels() > coarse_lev + 1)
    parallel_copy(newfine, h.data(coarse_lev + 1));  // preserver l'ancien fin

  h.install_level(coarse_lev + 1, fba, std::move(newfine));
}

}  // namespace adc
