// Echange de halos distribue (lance via mpirun -np N). On remplit les cellules
// valides avec une valeur ne dependant QUE des coordonnees globales repliees
// (periodiques) : val(i,j,c) = wrap(i) + 0.001*wrap(j) + 100*c. Apres
// fill_boundary periodique, chaque cellule fantome doit valoir la meme chose
// (sa source periodique a la meme valeur repliee). Avec np>1, ces ghosts
// proviennent de fabs distants : le test valide donc le transfert cross-rang.
//
// Invariant au nombre de rangs : reussit en serie (np=1, chemin local) comme en
// distribue (np>1, chemin MPI). Couvre aussi les coins (shifts diagonaux).

#include <adc/mesh/index/box2d.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/boundary/fill_boundary.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/parallel/comm.hpp>
#include <adc/parallel/load_balance.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank(), np = n_ranks();
  long fails = 0;

  const int L = 64, ng = 1, ncomp = 2;
  Box2D dom = Box2D::from_extents(L, L);
  auto wrap = [&](int x) { return ((x % L) + L) % L; };
  auto val = [&](int i, int j, int c) {
    return double(wrap(i)) + 0.001 * double(wrap(j)) + 100.0 * c;
  };

  BoxArray ba = BoxArray::from_domain(dom, 16);  // 4x4 = 16 boxes
  DistributionMapping dm = make_sfc_distribution(ba, np);
  MultiFab mf(ba, dm, ncomp, ng);

  // remplir les cellules valides locales
  for (int li = 0; li < mf.local_size(); ++li) {
    Fab2D& F = mf.fab(li);
    const Box2D b = F.box();
    for (int c = 0; c < ncomp; ++c)
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          F(i, j, c) = val(i, j, c);
  }

  fill_boundary(mf, dom, Periodicity{true, true});

  // verifier toutes les cellules (valides + fantomes) contre la valeur repliee
  for (int li = 0; li < mf.local_size(); ++li) {
    const Fab2D& F = mf.fab(li);
    const Box2D g = F.box().grow(ng);
    for (int c = 0; c < ncomp; ++c)
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i)
          if (std::fabs(F(i, j, c) - val(i, j, c)) > 1e-12)
            ++fails;
  }

  const long gfails = all_reduce_sum(fails);
  if (me == 0) {
    if (gfails == 0)
      std::printf("OK test_mpi_fillboundary (np=%d, boxes=%d)\n", np, ba.size());
    else
      std::printf("FAIL test_mpi_fillboundary : %ld cellules fausses (np=%d)\n", gfails, np);
  }
  comm_finalize();
  return gfails == 0 ? 0 : 1;
}
