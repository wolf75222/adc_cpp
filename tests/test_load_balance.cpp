// Equilibrage de charge : Z-order (SFC) vs knapsack. On verifie la validite
// (toutes les boxes assignees, tous les rangs utilises), la propriete de
// localite de la SFC (segments contigus le long de la courbe de Morton -> il y
// a exactement nranks-1 transitions de rang dans l'ordre de Morton), et que le
// knapsack equilibre au moins aussi bien que la SFC (il optimise le desequilibre
// max, la SFC le troque contre la localite).

#include <adc/mesh/index/box2d.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/parallel/load_balance.hpp>

#include <cstdio>
#include <vector>

using namespace adc;

// nombre de changements de rang le long de l'ordre de Morton.
static int rank_transitions(const BoxArray& ba, const DistributionMapping& dm) {
  const std::vector<int> order = morton_order(ba);
  int t = 0;
  for (std::size_t k = 1; k < order.size(); ++k)
    if (dm[order[k]] != dm[order[k - 1]])
      ++t;
  return t;
}

static bool all_in_range(const DistributionMapping& dm, int nranks) {
  for (int r : dm.ranks())
    if (r < 0 || r >= nranks)
      return false;
  return true;
}

static int n_ranks_used(const DistributionMapping& dm, int nranks) {
  std::vector<char> seen(nranks, 0);
  for (int r : dm.ranks())
    seen[r] = 1;
  int u = 0;
  for (char c : seen)
    u += c;
  return u;
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  // --- cles de Morton : valeurs de reference ---
  chk(morton_key(0, 0) == 0, "morton_00");
  chk(morton_key(1, 0) == 1, "morton_10");
  chk(morton_key(0, 1) == 2, "morton_01");
  chk(morton_key(1, 1) == 3, "morton_11");
  chk(morton_key(2, 0) == 4, "morton_20");
  chk(morton_key(0, 2) == 8, "morton_02");

  const int nranks = 4;

  // --- cas uniforme : 8x8 = 64 boxes de 16x16 (charge egale) ---
  BoxArray ba = BoxArray::from_domain(Box2D::from_extents(128, 128), 16);
  chk(ba.size() == 64, "uniform_64_boxes");

  DistributionMapping sfc = make_sfc_distribution(ba, nranks);
  DistributionMapping knap = make_knapsack_distribution(ba, nranks);

  chk(all_in_range(sfc, nranks), "sfc_ranks_in_range");
  chk(all_in_range(knap, nranks), "knap_ranks_in_range");
  chk(n_ranks_used(sfc, nranks) == nranks, "sfc_all_ranks_used");
  chk(n_ranks_used(knap, nranks) == nranks, "knap_all_ranks_used");

  const double sfc_imb = load_imbalance(ba, sfc, nranks);
  const double knap_imb = load_imbalance(ba, knap, nranks);
  std::printf("uniforme : sfc_imb=%.4f knap_imb=%.4f\n", sfc_imb, knap_imb);
  chk(sfc_imb <= 1.001, "sfc_uniform_balanced");
  chk(knap_imb <= 1.001, "knap_uniform_balanced");

  // localite : la SFC fait des segments contigus (nranks-1 transitions), le
  // knapsack disperse (beaucoup plus de transitions).
  const int sfc_t = rank_transitions(ba, sfc);
  const int knap_t = rank_transitions(ba, knap);
  std::printf("localite : sfc_transitions=%d knap_transitions=%d\n", sfc_t, knap_t);
  chk(sfc_t == nranks - 1, "sfc_contiguous_segments");
  chk(knap_t > sfc_t, "knap_less_local_than_sfc");

  // --- cas non-uniforme concu : poids [5,4,3,2,2,2], 3 rangs, places le long
  // de l'axe x pour que l'ordre de Morton = l'ordre d'insertion. Le knapsack
  // (LPT) doit equilibrer strictement mieux que la coupe contigue SFC. ---
  const int w[] = {5, 4, 3, 2, 2, 2};
  std::vector<Box2D> bx;
  for (int k = 0; k < 6; ++k)
    bx.push_back(Box2D{{k * 100, 0}, {k * 100, w[k] - 1}});  // 1 x w_k cellules
  BoxArray ban(std::move(bx));
  for (int k = 0; k < 6; ++k)
    chk(ban[k].num_cells() == w[k], "nonuniform_weights");

  DistributionMapping sfc3 = make_sfc_distribution(ban, 3);
  DistributionMapping knap3 = make_knapsack_distribution(ban, 3);
  chk(all_in_range(sfc3, 3) && all_in_range(knap3, 3), "nonuniform_in_range");
  chk(n_ranks_used(sfc3, 3) == 3 && n_ranks_used(knap3, 3) == 3, "nonuniform_all_ranks_used");

  const double sfc3_imb = load_imbalance(ban, sfc3, 3);
  const double knap3_imb = load_imbalance(ban, knap3, 3);
  std::printf("non-uniforme : sfc_imb=%.4f knap_imb=%.4f\n", sfc3_imb, knap3_imb);
  chk(knap3_imb <= sfc3_imb + 1e-9, "knap_balances_at_least_as_well");
  chk(knap3_imb < sfc3_imb, "knap_strictly_better_here");

  if (fails == 0)
    std::printf("OK test_load_balance\n");
  return fails == 0 ? 0 : 1;
}
