// Phase 7 du portage runtime : un CAS COMPLET via le System (orchestration), pas une piece isolee.
// euler_poisson : transport compressible (HLLC) + force de gravite + solve Poisson a CHAQUE pas +
// pas de temps CFL. Exerce les phases 1 (transport/MultiFab), 2 (BCs) et 3 (Poisson) INTEGREES par le
// System. On lie system.cpp et on compare CPU vs GPU. Portable seriel / Kokkos+CUDA.
#include <pops/runtime/config/model_spec.hpp>
#include <pops/runtime/system.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

int main(int argc, char** argv) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::initialize(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  double mass = 0, sum_phi = 0, max_phi = 0;
  {
    const int n = 64;
    const double L = 1.0;
    SystemConfig cfg;
    cfg.n = n;
    cfg.L = L;
    cfg.periodic = true;
    System sim(cfg);

    ModelSpec spec;  // euler_poisson : compressible + gravite (source + couplage elliptique)
    spec.transport = "compressible";
    spec.source = "gravity";
    spec.elliptic = "gravity";
    spec.gamma = 1.4;
    spec.sign = -1.0;
    spec.four_pi_G = 1.0;
    spec.rho0 = 1.0;
    sim.add_block("gas", spec, "minmod", "hllc", "conservative", "explicit", 1, true);
    sim.set_poisson("charge_density", "fft");

    std::vector<double> rho(static_cast<std::size_t>(n) * n);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        const double x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5;
        rho[static_cast<std::size_t>(j) * n + i] = 1.0 + 0.3 * std::exp(-(x * x + y * y) / 0.02);
      }
    sim.set_density("gas", rho);

    for (int s = 0; s < 20; ++s)
      sim.step_cfl(0.4);  // transport + Poisson + force, par pas

    mass = sim.mass("gas");
    const std::vector<double> phi = sim.potential();
    for (double v : phi) {
      sum_phi += v;
      max_phi = std::fmax(max_phi, std::fabs(v));
    }
  }
#if defined(POPS_HAS_KOKKOS)
  const char* space = Kokkos::DefaultExecutionSpace::name();
#else
  const char* space = "Serial(host)";
#endif
  std::printf("exec=%s n=64 steps=20  mass=%.12f  sum(phi)=%.12f  max|phi|=%.12f\n", space, mass,
              sum_phi, max_phi);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::finalize();
#endif
  return 0;
}
