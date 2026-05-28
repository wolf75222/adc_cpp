// Instabilite diocotron en boite periodique : une bande de charge cree un
// ecoulement E x B cisaille (deux couches de cisaillement avec point
// d'inflexion), instable a une perturbation le long de la bande. Les bords
// s'enroulent en "cat's eyes", phenomenologie classique du diocotron / KH.
//
// Cle physique : c'est le couplage qui rend l'ecoulement. La densite n_e fixe
// phi par Poisson, phi fixe la derive v_E = (E x B)/B^2, qui transporte n_e.
//
// Sortie :
//   <out>/diocotron_amp.csv      : t, amplitude du mode, densite min/max
//   <out>/dens_XXXX.txt          : instantanes de densite (grille n x n)
// L'amplitude = norme L2 de l'ecart au profil moyenne en x ; sa pente
// logarithmique dans la phase lineaire donne le taux de croissance gamma.
//
// Build : cmake --build build --target diocotron
// Run   : ./build/bin/diocotron /tmp/diocotron_out [n] [nsteps]

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
#include <string>

using namespace adc;

static constexpr double kPi = 3.14159265358979323846;

int main(int argc, char** argv) {
  const std::string out = (argc > 1) ? argv[1] : "diocotron_out";
  const int n = (argc > 2) ? std::atoi(argv[2]) : 192;
  const int nsteps = (argc > 3) ? std::atoi(argv[3]) : 700;
  std::filesystem::create_directories(out);

  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), n_ranks());

  // parametres de la bande de charge et de la perturbation (CLI : argv[4]=m, [5]=w)
  const double A = 1.0;                                    // amplitude de la bande
  const double w = (argc > 5) ? std::atof(argv[5]) : 0.05;  // demi-largeur
  const int m = (argc > 4) ? std::atoi(argv[4]) : 2;       // mode seme en x
  const double eta = 0.02;                                 // deplacement de la bande

  Diocotron model;
  model.B0 = 1.0;
  model.alpha = 1.0;

  // condition initiale : bande gaussienne dont l'axe est deplace de eta cos(2pi m x)
  MultiFab U(ba, dm, 1, 2);  // 2 ghosts pour MUSCL
  Array4 a = U.fab(0).array();
  for_each_cell(dom, [a, geom, A, w, m, eta](int i, int j) {
    const double x = geom.x_cell(i), y = geom.y_cell(j);
    const double y0 = 0.5 + eta * std::cos(2 * kPi * m * x);
    a(i, j, 0) = 1.0 + A * std::exp(-((y - y0) * (y - y0)) / (w * w));
  });

  // neutralite globale : n_i0 = moyenne de n_e (source de Poisson a moyenne nulle)
  double mean = 0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) mean += a(i, j, 0);
  mean /= double(n) * n;
  model.n_i0 = mean;

  BCRec bc;  // periodique partout (U et phi)
  Coupler<Diocotron> cpl(model, geom, ba, bc, bc);

  // estimation de la vitesse E x B pour fixer le pas de temps
  auto vmax_estimate = [&]() {
    const Fab2D& ax = cpl.aux().fab(0);
    double vmax = 0;
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        const double gx = ax(i, j, 1), gy = ax(i, j, 2);
        vmax = std::max(vmax, std::sqrt(gx * gx + gy * gy) / model.B0);
      }
    return std::max(vmax, 1e-12);
  };

  // amplitude du mode : ecart L2 au profil moyenne en x
  auto amplitude = [&](double& nmin, double& nmax) {
    const Fab2D& f = U.fab(0);
    nmin = 1e300;
    nmax = -1e300;
    double s = 0;
    for (int j = 0; j < n; ++j) {
      double nbar = 0;
      for (int i = 0; i < n; ++i) nbar += f(i, j, 0);
      nbar /= n;
      for (int i = 0; i < n; ++i) {
        const double d = f(i, j, 0) - nbar;
        s += d * d;
        nmin = std::min(nmin, f(i, j, 0));
        nmax = std::max(nmax, f(i, j, 0));
      }
    }
    return std::sqrt(s / (double(n) * n));
  };

  auto dump_density = [&](int frame) {
    char name[64];
    std::snprintf(name, sizeof(name), "/dens_%04d.txt", frame);
    std::ofstream f(out + name);
    const Fab2D& U0 = U.fab(0);
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) f << U0(i, j, 0) << (i + 1 < n ? ' ' : '\n');
    }
  };

  std::ofstream amp(out + "/diocotron_amp.csv");
  amp << "# diocotron n=" << n << " A=" << A << " w=" << w << " m=" << m
      << " eta=" << eta << " n_i0=" << mean << "\n";
  amp << "t,amplitude,nmin,nmax\n";

  cpl.solve_fields(U);
  const Real CFL = 0.4;
  Real dt = CFL * geom.dx() / vmax_estimate();

  const int snap_every = std::max(1, nsteps / 30);
  double t = 0;
  int frame = 0;
  std::printf("diocotron n=%d nsteps=%d dt0=%.3e v=%.3e\n", n, nsteps, dt,
              vmax_estimate());

  for (int s = 0; s <= nsteps; ++s) {
    double nmin, nmax;
    const double amp_t = amplitude(nmin, nmax);
    amp << t << ',' << amp_t << ',' << nmin << ',' << nmax << '\n';
    if (s % snap_every == 0) {
      dump_density(frame++);
      std::printf("  s=%5d t=%7.3f amp=%.4e nmin=%.3f nmax=%.3f\n", s, t, amp_t,
                  nmin, nmax);
    }
    if (s == nsteps) break;

    cpl.advance<VanLeer>(U, dt);
    t += dt;
    if (s % 25 == 0) dt = CFL * geom.dx() / vmax_estimate();  // reactualise le pas
  }

  amp.close();
  std::printf("ecrit %s/diocotron_amp.csv + %d instantanes\n", out.c_str(),
              frame);
  return 0;
}
