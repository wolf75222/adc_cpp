// Banc de mesure : verifie que ca tourne VRAIMENT, et a quelle vitesse. Deux charges,
// chronometrees sans I/O :
//   1. deux-fluides AP mono-grille (transport 2 especes + Poisson multigrille) : cible
//      de scaling OpenMP propre (compute pur, pas de desequilibre AMR). Reporte le debit
//      en M mailles-MAJ/s et la conservation de la masse.
//   2. coupleur AMR multi-patch (AmrCouplerMP) avec regrid Berger-Rigoutsos : verifie
//      que l'AMR couple tourne et CONSERVE, et son debit.
//
// Run : OMP_NUM_THREADS=k ./build-omp/bin/bench_amr [n] [nsteps]

#include <adc/coupling/amr_coupler_mp.hpp>
#include <adc/elliptic/geometric_mg.hpp>
#include <adc/integrator/two_fluid_ap.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/model/diocotron.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;
using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char** argv) {
  const int n = (argc > 1) ? std::atoi(argv[1]) : 256;
  const int nsteps = (argc > 2) ? std::atoi(argv[2]) : 100;

  // --- 1. deux-fluides AP mono-grille (scaling OpenMP) ---
  {
    TwoFluidAP2D<GeometricMG> d(n, 2 * kPi, 1.0, 0.04, 5.0, 1.0);
    d.init(1e-3);
    const double m0 = sum(d.e, 0);
    const double dt = 0.4 * (2 * kPi / n) / std::sqrt(1.0);  // CFL acoustique (c_s = 1)
    d.step(dt, true);  // warm-up (alloc, premier V-cycle)
    const auto t0 = clk::now();
    for (int s = 0; s < nsteps; ++s) d.step(dt, true);
    const auto t1 = clk::now();
    const double t = secs(t0, t1);
    const double cells = double(n) * n * nsteps;
    const double drift = std::fabs(sum(d.e, 0) - m0);
    std::printf("two-fluid AP mono-grille n=%d, %d pas : %.3f s | %.1f M mailles-MAJ/s | "
                "%.2f us/pas | drift_masse=%.2e\n",
                n, nsteps, t, cells / t / 1e6, t / nsteps * 1e6, drift);
  }

  // --- 2. coupleur AMR multi-patch + regrid Berger-Rigoutsos ---
  {
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
