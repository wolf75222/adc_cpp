// Smoke test du backend MPI (lance via mpirun -np N). Verifie le seam complet :
// distribution SFC d'un BoxArray sur N rangs, MultiFab n'allouant que les fabs
// locaux, et reduction collective. Chaque rang ne possede qu'une partie des
// boxes ; la somme globale (all_reduce) doit retrouver le total serie.
//
// Tourne aussi en serie (np=1) : all_reduce devient l'identite. Le resultat est
// identique quel que soit N (invariance au nombre de rangs).

#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/parallel/comm.hpp>
#include <adc/parallel/load_balance.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank(), np = n_ranks();
  long fails = 0;

  BoxArray ba = BoxArray::from_domain(Box2D::from_extents(128, 128), 16);
  DistributionMapping dm = make_sfc_distribution(ba, np);
  MultiFab mf(ba, dm, 1, 0);
  mf.set_val(1.0);

  // 1. somme globale (all-reduce) = nombre total de cellules valides
  const double s = sum(mf, 0);
  const double expect = double(ba.num_cells());
  if (std::fabs(s - expect) > 1e-9) {
    ++fails;
    if (me == 0)
      std::printf("FAIL sum %.1f != %.1f\n", s, expect);
  }

  // 2. la somme des fabs locaux sur tous les rangs = nombre total de boxes
  const long tot_local = all_reduce_sum(static_cast<long>(mf.local_size()));
  if (tot_local != ba.size()) {
    ++fails;
    if (me == 0)
      std::printf("FAIL local_size sum %ld != %d\n", tot_local, ba.size());
  }

  // 3. chaque box possedee localement est bien attribuee a ce rang par dm
  for (int li = 0; li < mf.local_size(); ++li)
    if (dm[mf.global_index(li)] != me) {
      ++fails;
      break;
    }

  // 4. equilibrage raisonnable (64 boxes egales)
  if (me == 0) {
    const double imb = load_imbalance(ba, dm, np);
    std::printf("np=%d  boxes=%d  somme=%.0f  imbalance=%.3f\n", np, ba.size(), s, imb);
    if (imb > 1.2) {
      ++fails;
      std::printf("FAIL imbalance %.3f > 1.2\n", imb);
    }
  }

  const long gfails = all_reduce_sum(fails);
  if (me == 0 && gfails == 0)
    std::printf("OK test_mpi_smoke (np=%d)\n", np);
  comm_finalize();
  return gfails == 0 ? 0 : 1;
}
