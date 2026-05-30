// Injection d'aux multi-patch parent -> enfant DISTRIBUEE (coupler_inject_aux_mb,
// chemin replicated_parent=false via parallel_copy), verifiee contre la valeur
// ANALYTIQUE attendue : independante du nombre de rangs.
//
// Au-dela de 2 niveaux, le parent (aux du niveau intermediaire) est multi-box REPARTI :
// un patch enfant peut avoir son parent sur un AUTRE rang. Sans le FillPatch parallel_copy,
// mf_find_box rendrait -1 et ces cellules resteraient a leur sentinelle (-12345). On verifie
// que chaque cellule (valides + ghosts), entierement couverte par le parent, recoit la bonne
// valeur f(coarsen(i), coarsen(j), k). A np=1 le chemin reste exerce (parallel_copy = copie
// memoire). DIST == analytique a np=1/2/4 prouve l'invariance a la distribution.

#include <adc/coupling/amr_coupler_mp.hpp>  // detail::coupler_inject_aux_mb
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/refinement.hpp>  // coarsen_index
#include <adc/parallel/comm.hpp>

#include <cstdio>
#include <vector>

using namespace adc;

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank();
  long fails = 0;

  // f analytique sur la grille parente (niveau 1), exacte en double.
  auto fval = [](int ci, int cj, int k) { return double(ci * 1000 + cj * 10 + k); };

  // parent : niveau 1, 4 boites pavant [0,32)^2 (quadrants 16x16), reparti round-robin.
  std::vector<Box2D> pb;
  for (int qy = 0; qy < 2; ++qy)
    for (int qx = 0; qx < 2; ++qx) {
      const int x0 = 16 * qx, y0 = 16 * qy;
      pb.push_back(Box2D{{x0, y0}, {x0 + 15, y0 + 15}});
    }
  BoxArray pba(pb);
  DistributionMapping pdm(static_cast<int>(pb.size()), n_ranks());  // round-robin
  MultiFab parent(pba, pdm, 3, 1);
  for (int li = 0; li < parent.local_size(); ++li) {
    Array4 a = parent.fab(li).array();
    const Box2D g = parent.fab(li).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i)
        for (int k = 0; k < 3; ++k) a(i, j, k) = fval(i, j, k);
  }

  // enfant : niveau 2, 2 patchs interieurs (region niveau 1 [8,24)^2 -> niveau 2 [16,48)^2),
  // reparti round-robin. Le grown footprint coarsen reste dans [0,32) -> couverture totale.
  std::vector<Box2D> cb = {Box2D{{16, 16}, {31, 47}}, Box2D{{32, 16}, {47, 47}}};
  BoxArray cba(cb);
  DistributionMapping cdm(static_cast<int>(cb.size()), n_ranks());
  MultiFab child(cba, cdm, 3, 1);
  child.set_val(-12345.0);  // sentinelle : doit etre ecrasee partout (couverture totale)

  detail::coupler_inject_aux_mb(parent, child, /*replicated_parent=*/false);

  // verification locale contre l'analytique (coarsen du grown box).
  for (int lc = 0; lc < child.local_size(); ++lc) {
    const ConstArray4 c = child.fab(lc).const_array();
    const Box2D g = child.fab(lc).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
        const int ci = coarsen_index(i, 2), cj = coarsen_index(j, 2);
        for (int k = 0; k < 3; ++k)
          if (c(i, j, k) != fval(ci, cj, k)) ++fails;
      }
  }

  fails = all_reduce_sum(fails);
  if (me == 0) {
    std::printf("inject aux multi-patch distribue (np=%d) : %ld cellules fausses\n",
                n_ranks(), fails);
    std::printf(fails == 0 ? "OK test_mpi_coupler_inject\n"
                           : "FAIL test_mpi_coupler_inject\n");
  }
  comm_finalize();
  return fails == 0 ? 0 : 1;
}
