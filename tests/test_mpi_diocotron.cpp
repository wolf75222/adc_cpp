// Diocotron distribue bout-en-bout (lance via mpirun -np N) : K pas couples
//   transport (advection E x B, Rusanov) + Poisson periodique SPECTRAL distribue
// sur une decomposition en bandes (slabs), exactement le layout du solveur FFT.
// Chaque pas : rho = alpha (n_e - n_i0) -> solve FFT distribue -> phi -> halos
// (fill_boundary) -> aux = grad phi -> halos -> advance par bande.
//
// Validation : on compare a une reference SERIE calculee sur la grille complete
// dans le meme process (meme sequence de FFT 1D + meme advance), donc l'accord
// doit etre bit a bit. Invariant au nombre de rangs (Nx, Ny divisibles par N).

#include <adc/elliptic/poisson_fft.hpp>
#include <adc/integrator/amr_reflux.hpp>  // advance_fab_1c, fill_periodic_fab, *face_box
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#ifdef ADC_HAS_MPI
#include <mpi.h>
#endif

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

// aux = (phi, d phi/dx, d phi/dy) sur les cellules valides (phi a ses ghosts).
static void aux_from_phi(const Fab2D& phi, Fab2D& aux, double dx, double dy) {
  const Box2D b = aux.box();
  for (int j = b.lo[1]; j <= b.hi[1]; ++j)
    for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
      aux(i, j, 0) = phi(i, j);
      aux(i, j, 1) = (phi(i + 1, j) - phi(i - 1, j)) / (2 * dx);
      aux(i, j, 2) = (phi(i, j + 1) - phi(i, j - 1)) / (2 * dy);
    }
}

// ghosts periodiques multi-composantes pour un Fab couvrant le domaine.
static void fill_periodic_mc_fab(Fab2D& F, const Box2D& dom) {
  const int ng = F.n_ghost(), nc = F.ncomp();
  for (int c = 0; c < nc; ++c) {
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int g = 1; g <= ng; ++g) {
        F(dom.lo[0] - g, j, c) = F(dom.hi[0] - g + 1, j, c);
        F(dom.hi[0] + g, j, c) = F(dom.lo[0] + g - 1, j, c);
      }
    for (int i = dom.lo[0] - ng; i <= dom.hi[0] + ng; ++i)
      for (int g = 1; g <= ng; ++g) {
        F(i, dom.lo[1] - g, c) = F(i, dom.hi[1] - g + 1, c);
        F(i, dom.hi[1] + g, c) = F(i, dom.lo[1] + g - 1, c);
      }
  }
}

// Poisson periodique spectral SERIE sur la grille complete (meme math que
// PoissonFFT : FFT-x, FFT-y par colonne, division, inverses).
static void poisson_serial(const std::vector<double>& rho, std::vector<double>& phi,
                           int Nx, int Ny, double dx, double dy) {
  std::vector<cplx> A(static_cast<std::size_t>(Ny) * Nx);
  for (std::size_t t = 0; t < A.size(); ++t) A[t] = cplx(rho[t], 0.0);
  for (int j = 0; j < Ny; ++j) fft1d(&A[j * Nx], Nx, false);
  std::vector<cplx> col(Ny);
  for (int i = 0; i < Nx; ++i) {
    for (int j = 0; j < Ny; ++j) col[j] = A[j * Nx + i];
    fft1d(col.data(), Ny, false);
    const double lx = (2 * std::cos(2 * kPi * i / Nx) - 2) / (dx * dx);
    for (int ky = 0; ky < Ny; ++ky) {
      const double ly = (2 * std::cos(2 * kPi * ky / Ny) - 2) / (dy * dy);
      const double lam = lx + ly;
      col[ky] = (std::abs(lam) < 1e-14) ? cplx(0, 0) : col[ky] / lam;
    }
    fft1d(col.data(), Ny, true);
    for (int j = 0; j < Ny; ++j) A[j * Nx + i] = col[j];
  }
  for (int j = 0; j < Ny; ++j) fft1d(&A[j * Nx], Nx, true);
  phi.resize(A.size());
  for (std::size_t t = 0; t < A.size(); ++t) phi[t] = A[t].real();
}

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank(), np = n_ranks();

  const int Nx = 64, Ny = 64, K = 5;
  const double Lx = 1.0, Ly = 1.0, dx = Lx / Nx, dy = Ly / Ny, dt = 0.1 * dx;
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
  // n_i0 = moyenne (calculee sur toute la grille, deterministe, identique partout)
  double mean = 0;
  for (int j = 0; j < Ny; ++j)
    for (int i = 0; i < Nx; ++i) mean += ne0(i, j);
  model.n_i0 = mean / (double(Nx) * Ny);

  // ===== reference SERIE (grille complete) =====
  Fab2D Une_ref(dom, 1, 1), phi_ref(dom, 1, 1), aux_ref(dom, 3, 1);
  for (int j = 0; j < Ny; ++j)
    for (int i = 0; i < Nx; ++i) Une_ref(i, j) = ne0(i, j);
  {
    std::vector<double> rho(static_cast<std::size_t>(Ny) * Nx), phi;
    Fab2D fxr(xface_box(dom), 1, 0), fyr(yface_box(dom), 1, 0);
    for (int s = 0; s < K; ++s) {
      for (int j = 0; j < Ny; ++j)
        for (int i = 0; i < Nx; ++i)
          rho[j * Nx + i] = model.alpha * (Une_ref(i, j) - model.n_i0);
      poisson_serial(rho, phi, Nx, Ny, dx, dy);
      for (int j = 0; j < Ny; ++j)
        for (int i = 0; i < Nx; ++i) phi_ref(i, j) = phi[j * Nx + i];
      fill_periodic_fab(phi_ref, dom);
      aux_from_phi(phi_ref, aux_ref, dx, dy);
      fill_periodic_mc_fab(aux_ref, dom);
      fill_periodic_fab(Une_ref, dom);
      advance_fab_1c(model, Une_ref, aux_ref, dx, dy, dt, fxr, fyr);
    }
  }

  // ===== distribue : bandes (1 box/rang), meme layout que la FFT =====
  const int nyl = Ny / np, y0 = me * nyl;
  std::vector<Box2D> slabs;
  for (int r = 0; r < np; ++r)
    slabs.push_back(Box2D{{0, r * nyl}, {Nx - 1, (r + 1) * nyl - 1}});
  BoxArray sba(std::move(slabs));
  DistributionMapping sdm(np, np);  // box r -> rang r
  MultiFab Une(sba, sdm, 1, 1), Uphi(sba, sdm, 1, 1), Uaux(sba, sdm, 3, 1);
  {
    Fab2D& F = Une.fab(0);
    const Box2D b = F.box();
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) F(i, j) = ne0(i, j);
  }
  PoissonFFT solver(Nx, Ny, Lx, Ly);
  std::vector<double> rho_local(static_cast<std::size_t>(nyl) * Nx), phi_local;
  for (int s = 0; s < K; ++s) {
    Fab2D& Fn = Une.fab(0);
    for (int jl = 0; jl < nyl; ++jl)
      for (int i = 0; i < Nx; ++i)
        rho_local[jl * Nx + i] = model.alpha * (Fn(i, y0 + jl) - model.n_i0);
    solver.solve(rho_local, phi_local);
    Fab2D& Fp = Uphi.fab(0);
    for (int jl = 0; jl < nyl; ++jl)
      for (int i = 0; i < Nx; ++i) Fp(i, y0 + jl) = phi_local[jl * Nx + i];
    fill_boundary(Uphi, dom, Periodicity{true, true});
    aux_from_phi(Uphi.fab(0), Uaux.fab(0), dx, dy);
    fill_boundary(Uaux, dom, Periodicity{true, true});
    fill_boundary(Une, dom, Periodicity{true, true});
    Fab2D fxf(xface_box(Une.box(0)), 1, 0), fyf(yface_box(Une.box(0)), 1, 0);
    advance_fab_1c(model, Une.fab(0), Uaux.fab(0), dx, dy, dt, fxf, fyf);
  }

  // ===== comparaison : rassembler la grille distribuee et confronter au serie =====
  std::vector<double> loc(static_cast<std::size_t>(nyl) * Nx);
  {
    const Fab2D& F = Une.fab(0);
    for (int jl = 0; jl < nyl; ++jl)
      for (int i = 0; i < Nx; ++i) loc[jl * Nx + i] = F(i, y0 + jl);
  }
  std::vector<double> full;
  if (np == 1) {
    full = loc;
  } else {
#ifdef ADC_HAS_MPI
    if (me == 0) full.resize(static_cast<std::size_t>(Ny) * Nx);
    MPI_Gather(loc.data(), nyl * Nx, MPI_DOUBLE, full.data(), nyl * Nx,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  }

  long fails = 0;
  if (me == 0) {
    double maxdiff = 0;
    for (int j = 0; j < Ny; ++j)
      for (int i = 0; i < Nx; ++i)
        maxdiff = std::max(maxdiff, std::fabs(full[j * Nx + i] - Une_ref(i, j)));
    std::printf("np=%d  K=%d  maxdiff(distribue vs serie)=%.3e\n", np, K, maxdiff);
    if (maxdiff > 1e-11) ++fails;
  }
#ifdef ADC_HAS_MPI
  if (np > 1) MPI_Bcast(&fails, 1, MPI_LONG, 0, MPI_COMM_WORLD);
#endif
  if (me == 0 && fails == 0) std::printf("OK test_mpi_diocotron (np=%d)\n", np);
  comm_finalize();
  return fails == 0 ? 0 : 1;
}
