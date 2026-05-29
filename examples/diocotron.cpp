// Instabilite diocotron en boite periodique : une bande de charge cree un
// ecoulement E x B cisaille, instable a une perturbation le long de la bande ; les
// bords s'enroulent en "cat's eyes" (phenomenologie diocotron / Kelvin-Helmholtz).
//
// Pilote MINCE au-dessus de la facade compilee `adc::solver` : toute la physique
// (CI bande, neutralite n_i0, Poisson, derive E x B, SSPRK2 couple) vit dans
// DiocotronSolver (ic = Band). Ici on ne fait que : choisir la config, lire les
// champs (density()) pour les diagnostics, et ecrire les sorties.
//
// Sortie :
//   <out>/diocotron_amp.csv : t, amplitude du mode (L2 de l'ecart au profil
//                             moyenne en x), densite min/max
//   <out>/dens_XXXX.txt     : instantanes de densite (grille n x n)
//
// Build : cmake --build build --target diocotron
// Run   : ./build/bin/diocotron /tmp/diocotron_out [n] [nsteps] [mode] [width]

#include <adc/solver/diocotron_solver.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace adc;

int main(int argc, char** argv) {
  const std::string out = (argc > 1) ? argv[1] : "diocotron_out";
  DiocotronConfig cfg;
  cfg.ic = DiocotronIC::Band;
  cfg.n = (argc > 2) ? std::atoi(argv[2]) : 192;
  const int nsteps = (argc > 3) ? std::atoi(argv[3]) : 700;
  cfg.band_mode = (argc > 4) ? std::atoi(argv[4]) : 2;
  cfg.band_width = (argc > 5) ? std::atof(argv[5]) : 0.05;
  std::filesystem::create_directories(out);

  DiocotronSolver s(cfg);  // <- toute la physique est dans la facade
  const int n = s.nx();

  // amplitude du mode : norme L2 de l'ecart au profil moyenne en x (par ligne y)
  auto amplitude = [&](const std::vector<double>& d, double& nmin, double& nmax) {
    nmin = 1e300; nmax = -1e300;
    double s2 = 0;
    for (int j = 0; j < n; ++j) {
      double nbar = 0;
      for (int i = 0; i < n; ++i) nbar += d[j * n + i];
      nbar /= n;
      for (int i = 0; i < n; ++i) {
        const double dev = d[j * n + i] - nbar;
        s2 += dev * dev;
        nmin = std::min(nmin, d[j * n + i]);
        nmax = std::max(nmax, d[j * n + i]);
      }
    }
    return std::sqrt(s2 / (double(n) * n));
  };
  auto dump = [&](const std::vector<double>& d, int frame) {
    char name[64];
    std::snprintf(name, sizeof(name), "/dens_%04d.txt", frame);
    std::ofstream f(out + name);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) f << d[j * n + i] << (i + 1 < n ? ' ' : '\n');
  };

  std::ofstream amp(out + "/diocotron_amp.csv");
  amp << "# diocotron (facade, ic=Band) n=" << n << " mode=" << cfg.band_mode
      << " width=" << cfg.band_width << "\n";
  amp << "t,amplitude,nmin,nmax\n";

  const int snap_every = std::max(1, nsteps / 30);
  int frame = 0;
  std::printf("diocotron n=%d nsteps=%d v_derive=%.3e\n", n, nsteps,
              s.max_drift_speed());

  for (int step = 0; step <= nsteps; ++step) {
    const auto d = s.density();
    double nmin, nmax;
    const double a = amplitude(d, nmin, nmax);
    amp << s.time() << ',' << a << ',' << nmin << ',' << nmax << '\n';
    if (step % snap_every == 0) {
      dump(d, frame++);
      std::printf("  s=%5d t=%7.3f amp=%.4e nmin=%.3f nmax=%.3f\n", step, s.time(),
                  a, nmin, nmax);
    }
    if (step == nsteps) break;
    s.step_cfl(0.4);  // pas stable choisi par la facade (CFL sur la derive E x B)
  }
  amp.close();
  std::printf("ecrit %s/diocotron_amp.csv + %d instantanes\n", out.c_str(), frame);
  return 0;
}
