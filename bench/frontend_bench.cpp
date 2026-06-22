#include <adc/runtime/config/model_spec.hpp>
#include <adc/runtime/system.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace adc;

namespace {

std::vector<double> initial_state(int n, double gamma) {
  const double pi = 3.14159265358979323846;
  std::vector<double> U(static_cast<std::size_t>(4) * n * n, 0.0);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n;
      const double y = (j + 0.5) / n;
      const double rho = 1.0 + 0.1 * std::sin(2.0 * pi * x) * std::cos(2.0 * pi * y);
      const double u = 0.2 + 0.05 * std::sin(2.0 * pi * y);
      const double v = -0.1 + 0.05 * std::cos(2.0 * pi * x);
      const double p = 1.0;
      const std::size_t q = static_cast<std::size_t>(j) * n + i;
      U[q] = rho;
      U[static_cast<std::size_t>(n) * n + q] = rho * u;
      U[static_cast<std::size_t>(2) * n * n + q] = rho * v;
      U[static_cast<std::size_t>(3) * n * n + q] =
          p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v);
    }
  }
  return U;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(ADC_HAS_KOKKOS)
  Kokkos::initialize(argc, argv);
#endif

  const int n = argc > 1 ? std::atoi(argv[1]) : 128;
  const int steps = argc > 2 ? std::atoi(argv[2]) : 40;
  const int warmup = argc > 3 ? std::atoi(argv[3]) : 5;
  const double dt = argc > 4 ? std::atof(argv[4]) : 1e-4;
  const double gamma = 1.4;
  double advance_ms = 0.0;
  double extract_ms = 0.0;
  double mass = 0.0;

  {
    SystemConfig cfg;
    cfg.n = n;
    cfg.L = 1.0;
    cfg.periodic = true;
    System sim(cfg);

    ModelSpec spec;
    spec.transport = "compressible";
    spec.source = "none";
    spec.elliptic = "background";
    spec.gamma = gamma;
    spec.alpha = 0.0;
    spec.n0 = 0.0;
    sim.add_block("gas", spec, "minmod", "rusanov", "conservative", "explicit", 1, true);
    sim.set_poisson("charge_density", "fft");
    sim.set_state("gas", initial_state(n, gamma));

    for (int s = 0; s < warmup; ++s) sim.step(dt);
#if defined(ADC_HAS_KOKKOS)
    Kokkos::fence();
#endif
    const auto t0 = std::chrono::steady_clock::now();
    sim.advance(dt, steps);
#if defined(ADC_HAS_KOKKOS)
    Kokkos::fence();
#endif
    const auto t1 = std::chrono::steady_clock::now();
    advance_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const auto e0 = std::chrono::steady_clock::now();
    const auto U = sim.get_state("gas");
    const auto e1 = std::chrono::steady_clock::now();
    extract_ms = std::chrono::duration<double, std::milli>(e1 - e0).count();
    for (std::size_t q = 0; q < static_cast<std::size_t>(n) * n; ++q) mass += U[q];
  }

  std::printf(
      "FRONTEND_CPP n=%d steps=%d warmup=%d dt=%.17g advance_ms=%.6f extract_ms=%.6f "
      "total_ms=%.6f mass=%.17e\n",
      n, steps, warmup, dt, advance_ms, extract_ms, advance_ms + extract_ms, mass);

#if defined(ADC_HAS_KOKKOS)
  Kokkos::finalize();
#endif
  return 0;
}
