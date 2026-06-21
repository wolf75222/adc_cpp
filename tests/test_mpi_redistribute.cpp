// Redistribution distribuee entre decompositions differentes (lance via mpirun
// -np N) : parallel_copy(dst, src) copie les regions valides qui se recouvrent,
// y compris quand src et dst vivent sur des rangs differents. On remplit une
// grille en TUILES 2D, on la redistribue en BANDES (layout du solveur FFT), on
// verifie les valeurs, puis on redistribue les bandes vers de NOUVELLES tuiles
// et on verifie l'identite du round-trip. Invariant au nombre de rangs.

#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/refinement.hpp>  // parallel_copy
#include <adc/parallel/comm.hpp>
#include <adc/parallel/load_balance.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank(), np = n_ranks();
  long fails = 0;

  const int L = 64;
  Box2D dom = Box2D::from_extents(L, L);
  auto f = [&](int i, int j) { return 1.0 + i + 1000.0 * j; };  // valeur unique/cellule

  // decomposition A : tuiles 16x16 (16 boxes), repartition SFC
  BoxArray tiles = BoxArray::from_domain(dom, 16);
  DistributionMapping dmA = make_sfc_distribution(tiles, np);
  MultiFab MA(tiles, dmA, 1, 0);
  for (int li = 0; li < MA.local_size(); ++li) {
    Fab2D& F = MA.fab(li);
    const Box2D b = F.box();
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        F(i, j) = f(i, j);
  }

  // decomposition B : bandes (np boxes, full x, L/np lignes), box r -> rang r
  const int nyl = L / np;
  std::vector<Box2D> slabs;
  for (int r = 0; r < np; ++r)
    slabs.push_back(Box2D{{0, r * nyl}, {L - 1, (r + 1) * nyl - 1}});
  BoxArray bands(std::move(slabs));
  DistributionMapping dmB(np, np);
  MultiFab MB(bands, dmB, 1, 0);
  MB.set_val(-1.0);

  // tuiles -> bandes (cross-rang)
  parallel_copy(MB, MA);
  for (int li = 0; li < MB.local_size(); ++li) {
    const Fab2D& F = MB.fab(li);
    const Box2D b = F.box();
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        if (std::fabs(F(i, j) - f(i, j)) > 1e-12)
          ++fails;
  }

  // bandes -> nouvelles tuiles (round-trip), identite attendue
  MultiFab MA2(tiles, dmA, 1, 0);
  MA2.set_val(-1.0);
  parallel_copy(MA2, MB);
  for (int li = 0; li < MA2.local_size(); ++li) {
    const Fab2D& F = MA2.fab(li);
    const Box2D b = F.box();
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        if (std::fabs(F(i, j) - f(i, j)) > 1e-12)
          ++fails;
  }

  const long gfails = all_reduce_sum(fails);
  if (me == 0) {
    if (gfails == 0)
      std::printf("OK test_mpi_redistribute (np=%d : tuiles<->bandes)\n", np);
    else
      std::printf("FAIL test_mpi_redistribute : %ld cellules fausses (np=%d)\n", gfails, np);
  }
  comm_finalize();
  return gfails == 0 ? 0 : 1;
}
