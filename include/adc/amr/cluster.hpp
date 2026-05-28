#pragma once

#include <adc/amr/tag_box.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <vector>

// Clustering Berger-Rigoutsos : transforme une grille de tags en un petit nombre
// de boxes couvrant les cellules taguees avec une bonne efficacite (fraction de
// cellules taguees dans les boxes).
//
// Recursion :
//   1. reduire la region au bounding box des tags (trim)
//   2. si efficacite >= seuil, ou region non splittable -> accepter la box
//   3. sinon choisir une coupe : trou (colonne/ligne vide) en priorite, sinon
//      inflexion (max du changement de Laplacien de la signature), sinon milieu
//   4. recurser sur les deux moities
// Puis chop final selon max_box_size.

namespace adc {

struct ClusterParams {
  double min_efficiency = 0.7;
  int min_box_size = 1;
  int max_box_size = 32;
};

namespace detail {

inline Box2D tag_bbox(const TagBox& tb, const Box2D& region) {
  int lo0 = INT_MAX, lo1 = INT_MAX, hi0 = INT_MIN, hi1 = INT_MIN;
  for (int j = region.lo[1]; j <= region.hi[1]; ++j)
    for (int i = region.lo[0]; i <= region.hi[0]; ++i)
      if (tb(i, j)) {
        lo0 = std::min(lo0, i);
        hi0 = std::max(hi0, i);
        lo1 = std::min(lo1, j);
        hi1 = std::max(hi1, j);
      }
  if (hi0 < lo0) return Box2D{{0, 0}, {-1, -1}};  // vide
  return Box2D{{lo0, lo1}, {hi0, hi1}};
}

inline long count_in(const TagBox& tb, const Box2D& r) {
  long c = 0;
  for (int j = r.lo[1]; j <= r.hi[1]; ++j)
    for (int i = r.lo[0]; i <= r.hi[0]; ++i) c += tb(i, j);
  return c;
}

inline std::vector<long> signature(const TagBox& tb, const Box2D& r, int axis) {
  const int len = (axis == 0) ? r.nx() : r.ny();
  std::vector<long> s(len, 0);
  for (int j = r.lo[1]; j <= r.hi[1]; ++j)
    for (int i = r.lo[0]; i <= r.hi[0]; ++i)
      if (tb(i, j)) s[(axis == 0) ? (i - r.lo[0]) : (j - r.lo[1])] += 1;
  return s;
}

// trou (zero) interieur le plus proche du centre, dans [mb, len-mb]. -1 sinon.
inline int best_hole(const std::vector<long>& s, int mb) {
  const int len = static_cast<int>(s.size());
  int best = -1, bestd = INT_MAX, c = len / 2;
  for (int k = mb; k <= len - mb; ++k)
    if (s[k] == 0) {
      int d = std::abs(k - c);
      if (d < bestd) {
        bestd = d;
        best = k;
      }
    }
  return best;
}

// inflexion : max |D[k] - D[k-1]| avec D le Laplacien discret, dans la plage
// valide. score renvoye via `score`. -1 si aucune.
inline int best_inflection(const std::vector<long>& s, int mb, long& score) {
  const int len = static_cast<int>(s.size());
  score = 0;
  if (len < 3) return -1;
  std::vector<long> D(len, 0);
  for (int k = 1; k < len - 1; ++k) D[k] = s[k + 1] - 2 * s[k] + s[k - 1];
  int best = -1;
  const int klo = std::max(mb, 2), khi = std::min(len - mb, len - 2);
  for (int k = klo; k <= khi; ++k) {
    long delta = std::labs(D[k] - D[k - 1]);
    if (delta > score) {
      score = delta;
      best = k;
    }
  }
  return best;
}

inline void cluster_rec(const TagBox& tb, Box2D region, const ClusterParams& p,
                        std::vector<Box2D>& out) {
  region = tag_bbox(tb, region);
  if (region.empty()) return;

  const long ntag = count_in(tb, region);
  const double eff = double(ntag) / double(region.num_cells());
  const int mb = std::max(1, p.min_box_size);
  const bool sx = region.nx() >= 2 * mb;
  const bool sy = region.ny() >= 2 * mb;
  if (eff >= p.min_efficiency || (!sx && !sy)) {
    out.push_back(region);
    return;
  }

  const auto Sx = signature(tb, region, 0);
  const auto Sy = signature(tb, region, 1);
  const int hx = sx ? best_hole(Sx, mb) : -1;
  const int hy = sy ? best_hole(Sy, mb) : -1;

  int axis = -1, kcut = -1;
  if (hx >= 0 && hy >= 0) {
    axis = (region.nx() >= region.ny()) ? 0 : 1;
    kcut = (axis == 0) ? hx : hy;
  } else if (hx >= 0) {
    axis = 0;
    kcut = hx;
  } else if (hy >= 0) {
    axis = 1;
    kcut = hy;
  } else {
    long scx = 0, scy = 0;
    const int ix = sx ? best_inflection(Sx, mb, scx) : -1;
    const int iy = sy ? best_inflection(Sy, mb, scy) : -1;
    if (ix >= 0 && (scx >= scy || iy < 0)) {
      axis = 0;
      kcut = ix;
    } else if (iy >= 0) {
      axis = 1;
      kcut = iy;
    } else {  // milieu de la plus grande dimension splittable
      if (sx && (region.nx() >= region.ny() || !sy)) {
        axis = 0;
        kcut = region.nx() / 2;
      } else {
        axis = 1;
        kcut = region.ny() / 2;
      }
    }
  }

  Box2D left = region, right = region;
  if (axis == 0) {
    left.hi[0] = region.lo[0] + kcut - 1;
    right.lo[0] = region.lo[0] + kcut;
  } else {
    left.hi[1] = region.lo[1] + kcut - 1;
    right.lo[1] = region.lo[1] + kcut;
  }
  cluster_rec(tb, left, p, out);
  cluster_rec(tb, right, p, out);
}

}  // namespace detail

inline std::vector<Box2D> berger_rigoutsos(const TagBox& tags,
                                           const ClusterParams& p = {}) {
  std::vector<Box2D> raw;
  detail::cluster_rec(tags, tags.box, p, raw);
  std::vector<Box2D> result;
  for (const auto& b : raw) {
    BoxArray chopped = BoxArray::from_domain(b, p.max_box_size);
    for (int k = 0; k < chopped.size(); ++k) result.push_back(chopped[k]);
  }
  return result;
}

}  // namespace adc
