// Valide all_reduce_sum_inplace (somme element par element d'un tableau sur tous les
// rangs), la brique de comm qui manquait pour le reflux AMR multi-patch distribue :
// chaque rang remplit les contributions de ses patchs locaux (0 ailleurs), un
// all-reduce donne a chaque rang le registre complet. Lance via mpirun -np N ; le
// resultat doit etre le MEME quel que soit N (invariance au nombre de rangs). En serie
// (np=1) l'operation est l'identite.

#include <pops/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace pops;

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank(), np = n_ranks();
  long fails = 0;

  // chaque rang contribue (me+1)*(i+1) ; la somme attendue est S*(i+1) avec
  // S = sum_{r=1..np} r = np(np+1)/2 (invariant du nombre de rangs cote resultat).
  const int n = 17;
  std::vector<double> buf(n);
  for (int i = 0; i < n; ++i)
    buf[i] = double(me + 1) * (i + 1);
  all_reduce_sum_inplace(buf.data(), n);

  const double S = double(np) * (np + 1) / 2.0;
  for (int i = 0; i < n; ++i)
    if (std::fabs(buf[i] - S * (i + 1)) > 1e-9) {
      ++fails;
      if (me == 0)
        std::printf("FAIL i=%d : %.3f != %.3f\n", i, buf[i], S * (i + 1));
    }

  // n <= 0 : no-op sans crash (buffer nul autorise).
  all_reduce_sum_inplace(nullptr, 0);

  if (me == 0) {
    if (fails == 0)
      std::printf("OK test_mpi_array_reduce (np=%d)\n", np);
    else
      std::printf("FAIL test_mpi_array_reduce : %ld\n", fails);
  }
  comm_finalize();
  return fails == 0 ? 0 : 1;
}
