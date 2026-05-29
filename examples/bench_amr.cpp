// Banc de mesure : verifie que ca tourne VRAIMENT, et a quelle vitesse. Chronometre
// sans I/O :
//   1. deux-fluides AP mono-grille (transport 2 especes + Poisson). Cible de scaling
//      OpenMP. Si n est une puissance de 2, compare AUSSI l'elliptique multigrille
//      (GeometricMG, iteratif) vs FFT (PoissonFFTSolver, direct) : meme physique,
//      debit different. Reporte M mailles-MAJ/s + conservation.
//   2. coupleur AMR multi-patch (AmrCouplerMP) + regrid Berger-Rigoutsos : verifie que
//      l'AMR couple tourne et CONSERVE, et son debit.
//
// Run : OMP_NUM_THREADS=k ./build-omp/bin/bench_amr [n] [nsteps]

#include <adc/coupling/amr_coupler_mp.hpp>
#include <adc/elliptic/geometric_mg.hpp>
#include <adc/elliptic/poisson_fft_solver.hpp>
#include <adc/integrator/two_fluid_ap.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/model/diocotron.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;
using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// chronometre nsteps pas du deux-fluides AP avec l'elliptique Elliptic. Renvoie le temps ;
// remplit drift (conservation masse) et dev (max|n_e - 1|, pour comparer les physiques).
template <class Elliptic>
static double bench_tfap(int n, int nsteps, double& drift, double& dev) {
  TwoFluidAP2D<Elliptic> d(n, 2 * kPi, 1.0, 0.04, 5.0, 1.0);
  d.init(1e-3);
  const double m0 = sum(d.e, 0);
  const double dt = 0.4 * (2 * kPi / n) / std::sqrt(1.0);  // CFL acoustique (c_s = 1)
  d.step(dt, true);  // warm-up
  const auto t0 = clk::now();
  for (int s = 0; s < nsteps; ++s) d.step(dt, true);
  const auto t1 = clk::now();
  drift = std::fabs(sum(d.e, 0) - m0);
  dev = 0;
  const ConstArray4 f = d.e.fab(0).const_array();
  for (int j = d.dom.lo[1]; j <= d.dom.hi[1]; ++j)
    for (int i = d.dom.lo[0]; i <= d.dom.hi[0]; ++i)
      dev = std::fmax(dev, std::fabs(f(i, j, 0) - 1.0));
  return secs(t0, t1);
}

int main(int argc, char** argv) {
  const int n = (argc > 1) ? std::atoi(argv[1]) : 256;
  const int nsteps = (argc > 2) ? std::atoi(argv[2]) : 100;
  const std::string mode = (argc > 3) ? argv[3] : "both";  // "tf" | "amr" | "both"
  const double cells = double(n) * n * nsteps;

  // --- 1. deux-fluides AP mono-grille (scaling OpenMP, backend multigrille) ---
  if (mode != "amr") {
    double drift, dev;
    const double t = bench_tfap<GeometricMG>(n, nsteps, drift, dev);
    std::printf("two-fluid AP (MG) n=%d, %d pas : %.3f s | %.1f M mailles-MAJ/s | "
                "%.2f us/pas | drift_masse=%.2e\n",
                n, nsteps, t, cells / t / 1e6, t / nsteps * 1e6, drift);

    // backend FFT direct (CL periodique, n puissance de 2) : meme physique, plus rapide.
    if ((n & (n - 1)) == 0) {
      double dF, devF;
      const double tF = bench_tfap<PoissonFFTSolver>(n, nsteps, dF, devF);
      std::printf("two-fluid AP (FFT) n=%d, %d pas : %.3f s | %.1f M mailles-MAJ/s | "
                  "speedup vs MG x%.2f | ecart physique |dev_MG-dev_FFT|=%.2e\n",
                  n, nsteps, tF, cells / tF / 1e6, t / tF, std::fabs(dev - devF));
    }
  }

  // --- 2. coupleur AMR multi-patch + regrid Berger-Rigoutsos ---
  if (mode != "tf") {
    Box2D dom = Box2D::from_extents(n, n);
    Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
    const double dxc = geom.dx(), dyc = geom.dy();
    BoxArray ba(std::vector<Box2D>{dom});
    DistributionMapping dm(1, n_ranks());
    BCRec bc;
    Diocotron model; model.B0 = 1.0; model.alpha = 1.0; model.n_i0 = 1.0;
    auto blob = [&](double x, double y) {
      return 1.0 + 0.6 * std::exp(-((x - 0.35) * (x - 0.35) + (y - 0.5) * (y - 0.5)) / 0.004) +
             0.6 * std::exp(-((x - 0.65) * (x - 0.65) + (y - 0.5) * (y - 0.5)) / 0.004);
    };
    auto crit = [&](const ConstArray4& a, int i, int j) { return a(i, j, 0) > model.n_i0 + 0.05; };
    Box2D seed{{2 * (n / 4), 2 * (n / 4)}, {2 * (3 * n / 4) - 1, 2 * (3 * n / 4) - 1}};
    MultiFab Uc(ba, dm, 1, 1), Uf(BoxArray(std::vector<Box2D>{seed}), dm, 1, 1);
    {
      Array4 u = Uc.fab(0).array();
      const Box2D g = Uc.fab(0).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) u(i, j, 0) = blob((i + 0.5) * dxc, (j + 0.5) * dyc);
      Array4 uf = Uf.fab(0).array();
      const Box2D b = Uf.box(0);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i) uf(i, j, 0) = blob((i + 0.5) * dxc / 2, (j + 0.5) * dyc / 2);
    }
    std::vector<AmrLevelMP> LP;
    LP.push_back({std::move(Uc), nullptr, dxc, dyc});
    LP.push_back({std::move(Uf), nullptr, dxc / 2, dyc / 2});
    AmrCouplerMP<Diocotron> sim(model, geom, ba, bc, std::move(LP));
    sim.regrid(crit);
    sim.update();
    const double m0 = sim.mass();
    const double dt = 0.4 * dxc / sim.max_drift_speed();
    sim.step(dt);  // warm-up
    const auto t0 = clk::now();
    for (int s = 0; s < nsteps; ++s) {
      if (s % 10 == 0) sim.regrid(crit);
      sim.step(dt);
    }
    const auto t1 = clk::now();
    const double t = secs(t0, t1);
    const double drift = std::fabs(sim.mass() - m0);
    std::printf("AMR multi-patch couple n=%d (+1 niveau fin), %d pas : %.3f s | %.2f ms/pas | "
                "npatch=%d | drift_masse=%.2e\n",
                n, nsteps, t, t / nsteps * 1e3, sim.levels()[1].U.local_size(), drift);
  }
  return 0;
}
