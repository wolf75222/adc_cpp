// Diocotron distribue MPI bout-en-bout, version demo (produit des instantanes).
// Transport (advection E x B, Rusanov) + Poisson periodique SPECTRAL distribue,
// sur une decomposition en bandes (1 box/rang, layout du solveur FFT). A chaque
// pas : rho = alpha (n_e - n_i0) -> solve FFT distribue -> phi -> halos -> aux =
// grad phi -> halos -> advance par bande. Aux instantanes, la densite est
// rassemblee sur le rang 0 (MPI_Gather des bandes) et ecrite en dens_XXXX.txt,
// au meme format que les autres demos (rendu par scripts/make_diocotron_gif.py).
//
// Pendant visuel, cote passage a l'echelle, du demo AMR 3 niveaux : ici c'est la
// distribution MPI (et un vrai Poisson spectral) qui porte le calcul.
//
// Run : mpirun -np 4 ./build-mpi/bin/diocotron_mpi /tmp/dio_mpi [nc] [nsteps]
//       python scripts/make_diocotron_gif.py /tmp/dio_mpi docs/anim_diocotron_mpi.gif

#include <adc/elliptic/poisson_fft.hpp>
#include <adc/integrator/amr_reflux.hpp>  // advance_fab_1c, *face_box
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/parallel/comm.hpp>

#include <algorithm>
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
  if (me == 0) std::filesystem::create_directories(out);
#ifdef ADC_HAS_MPI
  if (np > 1) MPI_Barrier(MPI_COMM_WORLD);
#endif
  if (nc % np != 0) {
    if (me == 0) std::printf("nc=%d doit etre divisible par np=%d\n", nc, np);
    comm_finalize();
    return 1;
  }

  const int Nx = nc, Ny = nc;
  const double Lx = 1.0, Ly = 1.0, dx = Lx / Nx, dy = Ly / Ny;
  Box2D dom = Box2D::from_extents(Nx, Ny);

  Diocotron model;
  model.B0 = 1.0;
  model.alpha = 1.0;
  const double A = 1.0, w = 0.05, eta = 0.02;
  const int m = 2;
  auto ne0 = [&](int i, int j) {
    const double x = (i + 0.5) * dx, y = (j + 0.5) * dy;
    const double y0 = 0.5 + eta * std::cos(2 * kPi * m * x);
    return 1.0 + A * std::exp(-((y - y0) * (y - y0)) / (w * w));
  };
  double mean = 0;
  for (int j = 0; j < Ny; ++j)
    for (int i = 0; i < Nx; ++i) mean += ne0(i, j);
  model.n_i0 = mean / (double(Nx) * Ny);

  // bandes : 1 box/rang (full x, Ny/np lignes), box r -> rang r.
  const int nyl = Ny / np, y0 = me * nyl;
  std::vector<Box2D> slabs;
  for (int r = 0; r < np; ++r)
    slabs.push_back(Box2D{{0, r * nyl}, {Nx - 1, (r + 1) * nyl - 1}});
  BoxArray sba(std::move(slabs));
  DistributionMapping sdm(np, np);
  MultiFab Une(sba, sdm, 1, 1), Uphi(sba, sdm, 1, 1), Uaux(sba, sdm, 3, 1);
  {
    Fab2D& F = Une.fab(0);
    for (int j = y0; j < y0 + nyl; ++j)
      for (int i = 0; i < Nx; ++i) F(i, j) = ne0(i, j);
  }
  PoissonFFT solver(Nx, Ny, Lx, Ly);
  std::vector<double> rho_local(static_cast<std::size_t>(nyl) * Nx), phi_local;

  // solve + phi + halos + aux + halos (prepare aux pour l'advance et le dt).
  auto compute_aux = [&]() {
    Fab2D& Fn = Une.fab(0);
    for (int jl = 0; jl < nyl; ++jl)
      for (int i = 0; i < Nx; ++i)
        rho_local[jl * Nx + i] = model.alpha * (Fn(i, y0 + jl) - model.n_i0);
    solver.solve(rho_local, phi_local);
    Fab2D& Fp = Uphi.fab(0);
    for (int jl = 0; jl < nyl; ++jl)
      for (int i = 0; i < Nx; ++i) Fp(i, y0 + jl) = phi_local[jl * Nx + i];
    fill_boundary(Uphi, dom, Periodicity{true, true});
    const Fab2D& P = Uphi.fab(0);
    Fab2D& Ax = Uaux.fab(0);
    for (int j = y0; j < y0 + nyl; ++j)
      for (int i = 0; i < Nx; ++i) {
        Ax(i, j, 0) = P(i, j);
        Ax(i, j, 1) = (P(i + 1, j) - P(i - 1, j)) / (2 * dx);
        Ax(i, j, 2) = (P(i, j + 1) - P(i, j - 1)) / (2 * dy);
      }
    fill_boundary(Uaux, dom, Periodicity{true, true});
  };
  auto vmax = [&]() {
    const Fab2D& Ax = Uaux.fab(0);
    double v = 0;
    for (int j = y0; j < y0 + nyl; ++j)
      for (int i = 0; i < Nx; ++i)
        v = std::max(v, std::hypot(Ax(i, j, 1), Ax(i, j, 2)) / model.B0);
    return std::max(all_reduce_max(v), 1e-12);
  };
  auto mass = [&]() {
    double s = 0;
    const Fab2D& Fn = Une.fab(0);
    for (int j = y0; j < y0 + nyl; ++j)
      for (int i = 0; i < Nx; ++i) s += Fn(i, j);
    return all_reduce_sum(s) * dx * dy;
  };

  // rassemble la densite sur le rang 0 et l'ecrit (format des autres demos).
  std::vector<double> loc(static_cast<std::size_t>(nyl) * Nx), full;
  if (me == 0) full.resize(static_cast<std::size_t>(Ny) * Nx);
  auto dump = [&](int frame) {
    const Fab2D& Fn = Une.fab(0);
    for (int jl = 0; jl < nyl; ++jl)
      for (int i = 0; i < Nx; ++i) loc[jl * Nx + i] = Fn(i, y0 + jl);
    if (np == 1) {
      full = loc;
    } else {
#ifdef ADC_HAS_MPI
      MPI_Gather(loc.data(), nyl * Nx, MPI_DOUBLE, full.data(), nyl * Nx,
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    }
    if (me != 0) return;
    char nm[64];
    std::snprintf(nm, sizeof(nm), "/dens_%04d.txt", frame);
    std::ofstream f(out + nm);
    for (int j = 0; j < Ny; ++j)
      for (int i = 0; i < Nx; ++i)
        f << full[j * Nx + i] << (i + 1 < Nx ? ' ' : '\n');
  };

  compute_aux();
  const double M0 = mass();
  double dt = 0.4 * dx / vmax();
  const int snap = std::max(1, nsteps / 30);
  if (me == 0)
    std::printf("diocotron MPI (np=%d) nc=%d Poisson spectral dt=%.2e\n", np, nc,
                dt);

  int frame = 0;
  for (int s = 0; s <= nsteps; ++s) {
    if (s % snap == 0) {
      dump(frame++);
      if (me == 0)
        std::printf("  s=%4d  drift=%.2e\n", s, std::fabs(mass() - M0));
    }
    if (s == nsteps) break;
    compute_aux();
    Fab2D fxf(xface_box(Une.box(0)), 1, 0), fyf(yface_box(Une.box(0)), 1, 0);
    fill_boundary(Une, dom, Periodicity{true, true});
    advance_fab_1c(model, Une.fab(0), Uaux.fab(0), dx, dy, dt, fxf, fyf);
    if (s % 20 == 0) dt = 0.4 * dx / vmax();
  }
  if (me == 0)
    std::printf("ecrit %s + %d instantanes ; drift final=%.2e\n", out.c_str(),
                frame, std::fabs(mass() - M0));
  comm_finalize();
  return 0;
}
