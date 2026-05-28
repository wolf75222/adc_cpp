#pragma once

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

// Equilibrage de charge pour l'AMR a metadonnees repliquees (facon AMReX,
// cf. la bibliographie sect. 4.2). Deux strategies construisent un
// DistributionMapping a partir d'un BoxArray et d'un nombre de rangs :
//
//   - Z-order (courbe de remplissage de l'espace, SFC) : on ordonne les boxes
//     le long de la courbe de Morton, puis on coupe en segments contigus de
//     charge egale. Avantage : localite spatiale (les boxes d'un rang sont
//     voisines), donc moins de communication ; Burstedde et al. mesurent ~11,5
//     voisins distincts meme a 200 000 coeurs.
//
//   - Knapsack (LPT, longest-processing-time) : on assigne chaque box, par
//     poids decroissant, au rang le moins charge. Avantage : minimise le
//     desequilibre maximal ; inconvenient : ignore la localite.
//
// Le poids d'une box est son nombre de cellules (proxy du cout de calcul). Ces
// algos sont purs (pas de MPI) : testables en serie, ils alimenteront le seam
// comm quand un vrai backend MPI sera branche.

namespace adc {

// Etale les bits de x (16 bits utiles) sur les positions paires d'un 64 bits.
inline std::uint64_t part1by1(std::uint64_t x) {
  x &= 0xffffffffULL;
  x = (x | (x << 16)) & 0x0000ffff0000ffffULL;
  x = (x | (x << 8)) & 0x00ff00ff00ff00ffULL;
  x = (x | (x << 4)) & 0x0f0f0f0f0f0f0f0fULL;
  x = (x | (x << 2)) & 0x3333333333333333ULL;
  x = (x | (x << 1)) & 0x5555555555555555ULL;
  return x;
}

// Cle de Morton (Z-order) entrelacant (x, y) : x sur les bits pairs, y impairs.
inline std::uint64_t morton_key(std::uint32_t x, std::uint32_t y) {
  return part1by1(x) | (part1by1(y) << 1);
}

// Indices des boxes tries le long de la courbe de Morton (coin bas, decale par
// le bounding box pour rester positif).
inline std::vector<int> morton_order(const BoxArray& ba) {
  const int n = ba.size();
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  if (n == 0) return order;
  const Box2D bb = ba.bounding_box();
  std::vector<std::uint64_t> key(n);
  for (int i = 0; i < n; ++i)
    key[i] = morton_key(static_cast<std::uint32_t>(ba[i].lo[0] - bb.lo[0]),
                        static_cast<std::uint32_t>(ba[i].lo[1] - bb.lo[1]));
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return key[a] < key[b]; });
  return order;
}

// Distribution Z-order : segments contigus de charge ~egale le long de la SFC.
// Garantit qu'avec nboxes >= nranks chaque rang recoit au moins une box.
inline DistributionMapping make_sfc_distribution(const BoxArray& ba,
                                                 int nranks) {
  const int n = ba.size();
  std::vector<int> rank(n, 0);
  if (n == 0 || nranks <= 1) return DistributionMapping(std::move(rank));

  const std::vector<int> order = morton_order(ba);
  long total = ba.num_cells();
  const double target = double(total) / nranks;  // charge cible par rang

  long acc = 0;
  int r = 0;
  for (int k = 0; k < n; ++k) {
    const int b = order[k];
    rank[b] = r;
    acc += ba[b].num_cells();
    // avance de rang si la part visee est atteinte ET qu'il reste assez de
    // boxes pour donner au moins une box a chaque rang restant.
    const int boxes_left = n - 1 - k;
    const int ranks_left = nranks - 1 - r;
    if (r < nranks - 1 && acc >= target * (r + 1) && boxes_left >= ranks_left)
      ++r;
  }
  return DistributionMapping(std::move(rank));
}

// Distribution knapsack (LPT) : box la plus lourde -> rang le moins charge.
inline DistributionMapping make_knapsack_distribution(const BoxArray& ba,
                                                      int nranks) {
  const int n = ba.size();
  std::vector<int> rank(n, 0);
  if (n == 0 || nranks <= 1) return DistributionMapping(std::move(rank));

  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return ba[a].num_cells() > ba[b].num_cells(); });

  std::vector<long> load(nranks, 0);
  for (int b : order) {
    int r = 0;
    for (int q = 1; q < nranks; ++q)
      if (load[q] < load[r]) r = q;
    rank[b] = r;
    load[r] += ba[b].num_cells();
  }
  return DistributionMapping(std::move(rank));
}

// Desequilibre = charge max / charge moyenne (1.0 = parfait).
inline double load_imbalance(const BoxArray& ba, const DistributionMapping& dm,
                             int nranks) {
  if (nranks <= 0 || ba.size() == 0) return 1.0;
  std::vector<long> load(nranks, 0);
  for (int i = 0; i < ba.size(); ++i) load[dm[i]] += ba[i].num_cells();
  long mx = 0, sum = 0;
  for (long l : load) {
    mx = std::max(mx, l);
    sum += l;
  }
  const double avg = double(sum) / nranks;
  return avg > 0 ? mx / avg : 1.0;
}

}  // namespace adc
