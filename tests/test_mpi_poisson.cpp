// Solveur de Poisson periodique spectral distribue (lance via mpirun -np N).
// On choisit un RHS de moyenne nulle rho(i,j), on resout lap_h phi = rho en
// reparti par bandes, on rassemble phi sur le rang 0, puis on applique le
// Laplacien 5-points periodique a phi et on verifie qu'il redonne rho a
// l'arrondi (verification du residu, independante de la constante additive).
// Resultat identique quel que soit N (Nx, Ny divisibles par N).

#include <adc/numerics/elliptic/poisson/poisson_fft.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#ifdef ADC_HAS_MPI
#include <mpi.h>
#endif

using namespace adc;

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank(), np = n_ranks();
  constexpr double kPi = 3.14159265358979323846;

  const int Nx = 64, Ny = 64;  // puissances de 2, divisibles par np <= 64
  const double Lx = 1.0, Ly = 1.0;
  PoissonFFT solver(Nx, Ny, Lx, Ly);
  const int nyl = solver.ny_local(), y0 = solver.y_begin();

  // RHS de moyenne nulle : produit de cosinus (somme nulle sur les periodes).
  auto rho = [&](int i, int j) {
    return std::cos(2 * kPi * 2 * i / Nx) * std::cos(2 * kPi * 3 * j / Ny) +
           0.5 * std::cos(2 * kPi * 5 * i / Nx) * std::cos(2 * kPi * 1 * j / Ny);
  };

  std::vector<double> rho_local(static_cast<std::size_t>(nyl) * Nx), phi_local;
  for (int jl = 0; jl < nyl; ++jl)
    for (int i = 0; i < Nx; ++i)
      rho_local[jl * Nx + i] = rho(i, y0 + jl);

  solver.solve(rho_local, phi_local);

  // rassembler phi sur le rang 0 (slabs contigues -> grille complete).
  std::vector<double> phi_full;
  if (np == 1) {
    phi_full = phi_local;
  } else {
#ifdef ADC_HAS_MPI
    if (me == 0)
      phi_full.resize(static_cast<std::size_t>(Ny) * Nx);
    MPI_Gather(phi_local.data(), nyl * Nx, MPI_DOUBLE, phi_full.data(), nyl * Nx, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);
#endif
  }

  long fails = 0;
  if (me == 0) {
    const double dx = Lx / Nx, dy = Ly / Ny;
    auto P = [&](int i, int j) { return phi_full[((j % Ny + Ny) % Ny) * Nx + (i % Nx + Nx) % Nx]; };
    double maxres = 0, meanphi = 0;
    for (std::size_t t = 0; t < phi_full.size(); ++t)
      meanphi += phi_full[t];
    meanphi /= double(Nx) * Ny;
    for (int j = 0; j < Ny; ++j)
      for (int i = 0; i < Nx; ++i) {
        const double lap = (P(i + 1, j) - 2 * P(i, j) + P(i - 1, j)) / (dx * dx) +
                           (P(i, j + 1) - 2 * P(i, j) + P(i, j - 1)) / (dy * dy);
        maxres = std::max(maxres, std::fabs(lap - rho(i, j)));
      }
    std::printf("np=%d  max|lap_h(phi)-rho|=%.3e  mean(phi)=%.3e\n", np, maxres, meanphi);
    if (maxres > 1e-10)
      ++fails;
    if (std::fabs(meanphi) > 1e-10)
      ++fails;
  }

#ifdef ADC_HAS_MPI
  if (np > 1)
    MPI_Bcast(&fails, 1, MPI_LONG, 0, MPI_COMM_WORLD);
#endif
  if (me == 0 && fails == 0)
    std::printf("OK test_mpi_poisson (np=%d)\n", np);
  comm_finalize();
  return fails == 0 ? 0 : 1;
}
