// DataWriter HDF5 : ecrit un MultiFab multi-boites dans un dataset [ny][nx] global,
// relit le fichier et compare cellule par cellule. En serie (ce test) toutes les
// boites sont sur le rang 0 ; le chemin parallele MPI-IO (hyperslabs disjointes,
// H5FD_MPIO_INDEPENDENT) est exerce sur ROMEO via mpirun. Valide que la selection
// par hyperslab place bien chaque boite au bon endroit du domaine global.

#include <adc/analysis/hdf5_writer.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/parallel/comm.hpp>

#include <hdf5.h>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;

static double fval(int i, int j) { return i + 100.0 * j; }  // valeur globale deterministe

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int N = 16, bs = 8;  // 2x2 boites de 8x8
  Box2D dom = Box2D::from_extents(N, N);
  BoxArray ba = BoxArray::from_domain(dom, bs);
  DistributionMapping dm(ba.size(), n_ranks());
  MultiFab mf(ba, dm, 1, 0);

  for (int li = 0; li < mf.local_size(); ++li) {
    Array4 a = mf.fab(li).array();
    const Box2D b = mf.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) a(i, j, 0) = fval(i, j);
  }

  // Chemin RELATIF (cwd) : en parallele il doit etre sur un systeme de fichiers
  // PARTAGE (GPFS/Lustre). MPI-IO collectif sur un /tmp local par noeud bloque.
  const std::string fn = "adc_test_hdf5.h5";
  write_hdf5(mf, N, N, fn, "field", 0);
  barrier();  // tous les rangs ont ecrit avant que le rang 0 ne relise

  // relecture (rang 0) et comparaison au domaine global complet
  int fails = 0;
  if (my_rank() == 0) {
    std::vector<double> buf(static_cast<std::size_t>(N) * N, -1.0);
    hid_t file = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t dset = H5Dopen2(file, "field", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Dclose(dset);
    H5Fclose(file);

    double maxdiff = 0;
    for (int j = 0; j < N; ++j)
      for (int i = 0; i < N; ++i)
        maxdiff = std::fmax(maxdiff,
                            std::fabs(buf[static_cast<std::size_t>(j) * N + i] -
                                      fval(i, j)));
    std::printf("HDF5 round-trip N=%d boites=%d maxdiff=%.3e\n", N, ba.size(),
                maxdiff);
    if (maxdiff != 0.0) {
      std::printf("FAIL hdf5_roundtrip\n");
      ++fails;
    }
  }

  if (fails == 0 && my_rank() == 0) std::printf("OK test_hdf5_io\n");
  comm_finalize();
  return fails == 0 ? 0 : 1;
}
