// Instabilite diocotron canonique : colonne de charge creuse (anneau) dans une
// cavite conductrice. Reproduit le cas de Hoffart-Maier-Shadid-Tomas (2025,
// arXiv:2510.11808, fig. 5.1-5.3) et de Davidson-Felice (1998).
//
// Notre modele guiding-center resout DIRECTEMENT la limite de derive magnetique
// que leur schema implicite cherche a atteindre depuis le modele complet : la
// densite n_e est transportee par la derive E x B, phi resolu par Poisson.
//
// Anneau de charge r in [r0, r1] perturbe azimutalement (mode l), paroi
// conductrice phi=0 (Dirichlet). Le mode l croit en l tourbillons en anneau.
//
// Sortie : <out>/dens_XXXX.txt (instantanes) + <out>/ring_amp.csv
//          (t, amplitude du mode l de phi echantillonne a r=r0).
//
// Run : ./build/bin/diocotron_column /tmp/dio_col 256 1500 3

#include <adc/coupling/coupler.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/operator/reconstruction.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>

using namespace adc;

static constexpr double kPi = 3.14159265358979323846;

int main(int argc, char** argv) {
  const std::string out = (argc > 1) ? argv[1] : "dio_col";
  const int n = (argc > 2) ? std::atoi(argv[2]) : 256;
  const int nsteps = (argc > 3) ? std::atoi(argv[3]) : 1500;
  const int ell = (argc > 4) ? std::atoi(argv[4]) : 3;  // mode azimutal
  const double Rwall = (argc > 5) ? std::atof(argv[5]) : 0.0;  // 0 = bord carre
  std::filesystem::create_directories(out);

  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), n_ranks());

  // geometrie de l'anneau (ratios proches du papier : r1/r0 ~ 4/3)
  const double cx = 0.5, cy = 0.5;
  const double r0 = 0.18, r1 = 0.24;  // anneau de charge
  const double rhomax = 1.0, rhomin = 1e-3;
  const double delta = 0.1;  // amplitude de la perturbation azimutale

  Diocotron model;
  model.B0 = 1.0;
  model.alpha = 1.0;
  model.n_i0 = 0.0;  // colonne d'electrons pure

  MultiFab U(ba, dm, 1, 2);  // 2 ghosts pour MUSCL
  Array4 a = U.fab(0).array();
  for_each_cell(dom, [=](int i, int j) {
    const double x = geom.x_cell(i) - cx, y = geom.y_cell(j) - cy;
    const double r = std::hypot(x, y);
    const double th = std::atan2(y, x);
    double ne = rhomin;
    if (r > r0 && r < r1)
      ne = rhomax * (1.0 - delta + delta * std::sin(ell * th));
    a(i, j, 0) = ne;
  });

  BCRec bc;  // paroi conductrice : phi Dirichlet 0 ; fluide : sortie libre
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;  // pour phi
  BCRec bcU;
  bcU.xlo = bcU.xhi = bcU.ylo = bcU.yhi = BCType::Foextrap;  // pour U

  // paroi conductrice circulaire optionnelle (sinon bord carre Dirichlet)
  std::function<bool(Real, Real)> active{};
  if (Rwall > 0)
    active = [=](Real x, Real y) { return std::hypot(x - cx, y - cy) < Rwall; };
  Coupler<Diocotron> cpl(model, geom, ba, bcU, bc, active);

  auto vmax_estimate = [&]() {
    const Fab2D& ax = cpl.aux().fab(0);
    double v = 0;
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        const double gx = ax(i, j, 1), gy = ax(i, j, 2);
        v = std::max(v, std::sqrt(gx * gx + gy * gy) / model.B0);
      }
    return std::max(v, 1e-12);
  };

  // amplitude du mode l : module du l-ieme coefficient de Fourier de phi
  // echantillonne sur le cercle r = r0 (bilineaire).
  auto sample = [&](const Fab2D& f, double x, double y) {
    const double gx = (x - geom.xlo) / geom.dx() - 0.5;
    const double gy = (y - geom.ylo) / geom.dy() - 0.5;
    int i = (int)std::floor(gx), j = (int)std::floor(gy);
    i = std::clamp(i, 0, n - 2);
    j = std::clamp(j, 0, n - 2);
    const double tx = gx - i, ty = gy - j;
    return (1 - tx) * (1 - ty) * f(i, j, 0) + tx * (1 - ty) * f(i + 1, j, 0) +
           (1 - tx) * ty * f(i, j + 1, 0) + tx * ty * f(i + 1, j + 1, 0);
  };
  auto mode_amplitude = [&]() {
    const Fab2D& p = cpl.phi().fab(0);
    const int K = 256;
    double re = 0, im = 0;
    for (int k = 0; k < K; ++k) {
      const double th = 2 * kPi * k / K;
      const double val = sample(p, cx + r0 * std::cos(th), cy + r0 * std::sin(th));
      re += val * std::cos(ell * th);
      im += val * std::sin(ell * th);
    }
    return 2.0 * std::sqrt(re * re + im * im) / K;
  };

  auto dump_density = [&](int frame) {
    char name[64];
    std::snprintf(name, sizeof(name), "/dens_%04d.txt", frame);
    std::ofstream f(out + name);
    const Fab2D& U0 = U.fab(0);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) f << U0(i, j, 0) << (i + 1 < n ? ' ' : '\n');
  };

  std::ofstream amp(out + "/ring_amp.csv");
  amp << "# diocotron colonne n=" << n << " l=" << ell << " r0=" << r0
      << " r1=" << r1 << " delta=" << delta << "\n";
  amp << "t,amplitude\n";

  cpl.solve_fields(U);
  const Real CFL = 0.4;
  Real dt = CFL * geom.dx() / vmax_estimate();

  const int snap_every = std::max(1, nsteps / 30);
  double t = 0;
  int frame = 0;
  std::printf("diocotron colonne n=%d l=%d nsteps=%d dt0=%.3e\n", n, ell, nsteps,
              dt);

  for (int s = 0; s <= nsteps; ++s) {
    const double am = mode_amplitude();
    amp << t << ',' << am << '\n';
    if (s % snap_every == 0) {
      dump_density(frame++);
      std::printf("  s=%5d t=%7.3f a_l=%.4e\n", s, t, am);
    }
    if (s == nsteps) break;
    cpl.advance<VanLeer>(U, dt);
    t += dt;
    if (s % 25 == 0) dt = CFL * geom.dx() / vmax_estimate();
  }

  amp.close();
  std::printf("ecrit %s/ring_amp.csv + %d instantanes\n", out.c_str(), frame);
  return 0;
}
