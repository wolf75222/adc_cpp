// Instabilite diocotron canonique : colonne de charge creuse (anneau) dans une
// cavite conductrice. Reproduit Hoffart-Maier-Shadid-Tomas (2025, arXiv:2510.11808,
// fig. 5.1-5.3) et Davidson-Felice (1998). Anneau de charge perturbe azimutalement
// (mode l), paroi conductrice phi = 0 (Dirichlet) ; le mode l croit en l tourbillons.
//
// Pilote MINCE au-dessus de la facade `adc::solver` : la CI anneau, les CL Dirichlet
// (phi) / Foextrap (U), la paroi conductrice circulaire optionnelle (embedded
// boundary) et le pas couple sont TOUS dans DiocotronSolver (ic = Ring). Ici on ne
// fait que lire phi (potential()) pour le diagnostic azimutal et ecrire les sorties.
//
// Sortie : <out>/dens_XXXX.txt (instantanes) + <out>/ring_amp.csv
//          (t, amplitude du mode l de phi echantillonne a r = r0).
//
// Run : ./build/bin/diocotron_column /tmp/dio_col [n] [nsteps] [l] [Rwall]

#include <adc/solver/diocotron_solver.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

int main(int argc, char** argv) {
  const std::string out = (argc > 1) ? argv[1] : "dio_col";
  DiocotronConfig cfg;
  cfg.ic = DiocotronIC::Ring;
  cfg.n = (argc > 2) ? std::atoi(argv[2]) : 256;
  const int nsteps = (argc > 3) ? std::atoi(argv[3]) : 1500;
  cfg.ring_mode = (argc > 4) ? std::atoi(argv[4]) : 3;       // mode azimutal l
  cfg.wall_radius = (argc > 5) ? std::atof(argv[5]) : 0.0;   // 0 = bord carre
  std::filesystem::create_directories(out);

  DiocotronSolver s(cfg);  // <- CI anneau + BC Dirichlet/Foextrap + paroi : dans la facade
  const int n = s.nx();
  const double L = cfg.L, dx = s.dx(), cx = 0.5 * L, cy = 0.5 * L, r0 = cfg.ring_r0;
  const int ell = cfg.ring_mode;

  // echantillonnage bilineaire de phi (row-major, domaine [0,L], xlo = 0)
  auto sample = [&](const std::vector<double>& p, double x, double y) {
    const double gx = x / dx - 0.5, gy = y / dx - 0.5;
    int i = std::clamp((int)std::floor(gx), 0, n - 2);
    int j = std::clamp((int)std::floor(gy), 0, n - 2);
    const double tx = gx - i, ty = gy - j;
    return (1 - tx) * (1 - ty) * p[j * n + i] + tx * (1 - ty) * p[j * n + i + 1] +
           (1 - tx) * ty * p[(j + 1) * n + i] + tx * ty * p[(j + 1) * n + i + 1];
  };
  // amplitude du mode l : module du l-ieme coefficient de Fourier de phi sur r = r0
  auto mode_amplitude = [&](const std::vector<double>& p) {
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
  auto dump = [&](const std::vector<double>& d, int frame) {
    char name[64];
    std::snprintf(name, sizeof(name), "/dens_%04d.txt", frame);
    std::ofstream f(out + name);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) f << d[j * n + i] << (i + 1 < n ? ' ' : '\n');
  };

  std::ofstream amp(out + "/ring_amp.csv");
  amp << "# diocotron colonne (facade, ic=Ring) n=" << n << " l=" << ell
      << " r0=" << cfg.ring_r0 << " r1=" << cfg.ring_r1 << "\n";
  amp << "t,amplitude\n";

  const int snap_every = std::max(1, nsteps / 30);
  int frame = 0;
  std::printf("diocotron colonne n=%d l=%d nsteps=%d\n", n, ell, nsteps);

  for (int step = 0; step <= nsteps; ++step) {
    const double a = mode_amplitude(s.potential());
    amp << s.time() << ',' << a << '\n';
    if (step % snap_every == 0) {
      dump(s.density(), frame++);
      std::printf("  s=%5d t=%7.3f a_l=%.4e\n", step, s.time(), a);
    }
    if (step == nsteps) break;
    s.step_cfl(0.4);
  }
  amp.close();
  std::printf("ecrit %s/ring_amp.csv + %d instantanes\n", out.c_str(), frame);
  return 0;
}
