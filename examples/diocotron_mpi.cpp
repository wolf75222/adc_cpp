// Diocotron distribue MPI, version demo. Le pas couple (transport E x B + Poisson
// spectral distribue) est porte par le composant reutilisable SpectralExBStepper
// (coupling/spectral_coupler.hpp) : cet exemple ne fait que CONFIGURER (condition
// initiale, n_i0) et TOURNER (boucle step + dump). Plus aucune boucle couplee
// reecrite a la main.
//
// Aux instantanes, la densite est rassemblee sur le rang 0 (MPI_Gather des bandes)
// et ecrite en dens_XXXX.txt (rendu par scripts/make_diocotron_gif.py).
//
// Run : mpirun -np 4 ./build-mpi/bin/diocotron_mpi /tmp/dio_mpi [nc] [nsteps]

#include <adc/coupling/spectral_coupler.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifdef ADC_HAS_MPI
#include <mpi.h>
#endif

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank(), np = n_ranks();
  const std::string out = (argc > 1) ? argv[1] : "dio_mpi";
  const int nc = (argc > 2) ? std::atoi(argv[2]) : 128;
  const int nsteps = (argc > 3) ? std::atoi(argv[3]) : 600;
  if (nc % np != 0) {
    if (me == 0) std::printf("nc=%d doit etre divisible par np=%d\n", nc, np);
    comm_finalize();
    return 1;
  }
  if (me == 0) std::filesystem::create_directories(out);

  const double Lx = 1.0, Ly = 1.0, dx = Lx / nc, dy = Ly / nc;
  const double A = 1.0, w = 0.05, eta = 0.02;
  const int m = 2;
  auto ne0 = [&](int i, int j) {
    const double x = (i + 0.5) * dx, y = (j + 0.5) * dy;
    const double y0 = 0.5 + eta * std::cos(2 * kPi * m * x);
    return 1.0 + A * std::exp(-((y - y0) * (y - y0)) / (w * w));
  };
  double mean = 0;
  for (int j = 0; j < nc; ++j)
    for (int i = 0; i < nc; ++i) mean += ne0(i, j);

  Diocotron model;
  model.B0 = 1.0;
  model.alpha = 1.0;
  model.n_i0 = mean / (double(nc) * nc);

  // --- composant pret a l'emploi ---
  SpectralExBStepper<Diocotron> sim(model, nc, nc, Lx, Ly);
  const int nyl = sim.ny_local(), y0 = sim.y_begin();
  {
    Fab2D& F = sim.local();
    for (int j = y0; j < y0 + nyl; ++j)
      for (int i = 0; i < nc; ++i) F(i, j) = ne0(i, j);
  }

  // --- I/O : rassemble la densite sur le rang 0 et l'ecrit ---
  std::vector<double> loc(static_cast<std::size_t>(nyl) * nc), full;
  if (me == 0) full.resize(static_cast<std::size_t>(nc) * nc);
  auto dump = [&](int frame) {
    const Fab2D& F = sim.local();
    for (int jl = 0; jl < nyl; ++jl)
      for (int i = 0; i < nc; ++i) loc[jl * nc + i] = F(i, y0 + jl);
    if (np == 1) {
      full = loc;
    } else {
#ifdef ADC_HAS_MPI
      MPI_Gather(loc.data(), nyl * nc, MPI_DOUBLE, full.data(), nyl * nc,
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    }
    if (me != 0) return;
    char nm[64];
    std::snprintf(nm, sizeof(nm), "/dens_%04d.txt", frame);
    std::ofstream f(out + nm);
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i)
        f << full[j * nc + i] << (i + 1 < nc ? ' ' : '\n');
  };

  sim.solve_aux();
  const double M0 = sim.mass();
  double dt = 0.4 * dx / sim.max_drift_speed();
  const int snap = std::max(1, nsteps / 30);
  if (me == 0)
    std::printf("diocotron MPI (np=%d) nc=%d Poisson spectral dt=%.2e\n", np, nc,
                dt);

  int frame = 0;
  for (int s = 0; s <= nsteps; ++s) {
    if (s % snap == 0) {
      dump(frame++);
      if (me == 0)
        std::printf("  s=%4d  drift=%.2e\n", s, std::fabs(sim.mass() - M0));
    }
    if (s == nsteps) break;
    sim.step(dt);
    if (s % 20 == 0) dt = 0.4 * dx / sim.max_drift_speed();
  }
  if (me == 0)
    std::printf("ecrit %s + %d instantanes ; drift final=%.2e\n", out.c_str(),
                frame, std::fabs(sim.mass() - M0));
  comm_finalize();
  return 0;
}
