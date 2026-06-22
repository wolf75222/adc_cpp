// DistributedFFTSolver : FFT spectrale distribuee par bandes, comme EllipticSolver AUTONOME.
// On resout lap(phi) = rho sur un domaine periodique avec rho = mode de Fourier a moyenne nulle.
// Le solve direct etant EXACT pour le Laplacien discret 5 points (PoissonFFT divise par la valeur
// propre discrete), le residu discret ||lap phi - rho|| est machine-zero. Lance np=1/2/4 : le
// decoupage en bandes et la transposee MPI_Alltoall de PoissonFFT ne changent pas le resultat.
// Prouve que la FFT distribuee est un EllipticSolver reutilisable, pas un rouage enferme dans
// SpectralCoupler.

#include <adc/numerics/elliptic/poisson/poisson_fft_solver.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  long fails = 0;
  const int N = 64;  // puissance de 2, divisible par np = 1/2/4
  Geometry geom{Box2D::from_extents(N, N), 0.0, 1.0, 0.0, 1.0};
  const double dx = geom.dx(), dy = geom.dy();

  DistributedFFTSolver fft(geom);
  // rho = sin(2 pi x) sin(2 pi y) : mode periodique a moyenne nulle (solvabilite Poisson).
  for (int li = 0; li < fft.rhs().local_size(); ++li) {
    Array4 r = fft.rhs().fab(li).array();
    const Box2D b = fft.rhs().box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        r(i, j) = std::sin(2 * kPi * (i + 0.5) * dx) * std::sin(2 * kPi * (j + 0.5) * dy);
  }
  fft.solve();
  const double res = fft.residual();
  if (my_rank() == 0)
    std::printf("DistributedFFTSolver (np=%d) : residu discret ||lap phi - rho|| = %.3e\n",
                n_ranks(), res);
  if (res > 1e-9) {
    if (my_rank() == 0)
      std::printf("FAIL fft_dist_residu\n");
    ++fails;
  }

  fails = all_reduce_sum(fails);
  if (fails == 0 && my_rank() == 0)
    std::printf("OK test_mpi_fft_distributed\n");
  comm_finalize();
  return fails == 0 ? 0 : 1;
}
