// Scaling AMR SYNTHETIQUE (4 bulles) de la campagne de perf. Adapte du test integre
// python/tests/gpu/amrmpi_integrated.cpp : MEME cas (modele euler_poisson COMPILE via
// add_compiled_model -- chemin foncteur nomme qui PASSE nvcc, contrairement au chemin generique),
// hierarchie AMR avec regrid + reflux + Poisson, niveau fin multi-patch distribue sur les rangs.
// Ici on emet le JSONL adc_perf_v1 (workload=amr) au lieu de la sortie de validation : percentiles
// du pas, cells/s (proxy = n^2 grossier), invariants (conservation de masse, finitude, n_patches).
//
// Mode d'ownership du grossier : --distribute 1 => grossier multi-box reparti (vrai strong-scaling
// multi-GPU) ; 0 => grossier replique. Defaut : reparti si n_ranks()>1. Le grossier reparti utilise
// coarse_max_grid = n/2 (decoupage 2x2, le moins agressif pour le MG geometrique).
//
// Portable : Kokkos initialise SEULEMENT si compile avec Kokkos (sinon serie, AmrSystem tourne hote).
// Lance via bench/run_scaling.sh (kokkos-omp / kokkos-cuda / mpi-cuda) en srun -n 1/2/4.

#include <adc/parallel/comm.hpp>
#include <adc/physics/bricks/bricks.hpp>
#include <adc/physics/fluids/euler.hpp>
#include <adc/runtime/builders/compiled/amr_dsl_block.hpp>
#include <adc/runtime/amr_system.hpp>

#ifdef ADC_HAS_KOKKOS
#include <Kokkos_Core.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace adc;
using Clock = std::chrono::steady_clock;
// euler_poisson auto-gravitant/plasma : Euler + force de gravite + couplage self-consistant (Gauss).
using Model = CompositeModel<Euler, GravityForce, GravityCoupling>;

#ifndef ADC_BUILD_SHA
#define ADC_BUILD_SHA "unknown"
#endif
#ifndef ADC_BUILD_BRANCH
#define ADC_BUILD_BRANCH "unknown"
#endif

static void fence() {
#ifdef ADC_HAS_KOKKOS
  Kokkos::fence();
#endif
}

static double percentile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double idx = q * (v.size() - 1);
  const size_t lo = static_cast<size_t>(idx);
  const size_t hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (idx - lo) * (v[hi] - v[lo]);
}

// 4 bulles gaussiennes de densite (convention de amrmpi_integrated) : declenche le raffinement.
static std::vector<double> four_bubbles(int n) {
  std::vector<double> rho(static_cast<std::size_t>(n) * n);
  const double cx[4] = {0.25, 0.75, 0.25, 0.75};
  const double cy[4] = {0.25, 0.25, 0.75, 0.75};
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n, y = (j + 0.5) / n;
      double r = 1.0;
      for (int b = 0; b < 4; ++b) {
        const double dx = x - cx[b], dy = y - cy[b];
        r += 0.5 * std::exp(-(dx * dx + dy * dy) / 0.004);
      }
      rho[static_cast<std::size_t>(j) * n + i] = r;
    }
  return rho;
}

static int run(int argc, char** argv) {
  int n = 128, steps = 20, warmup = 4, distribute = -1;
  std::string backend = "serial", machine = "unknown", scaling = "strong";
  for (int a = 1; a < argc; ++a) {
    auto eat = [&](const char* key, auto& out) {
      if (std::strcmp(argv[a], key) == 0 && a + 1 < argc) {
        using T = std::decay_t<decltype(out)>;
        if constexpr (std::is_same_v<T, std::string>)
          out = argv[++a];
        else
          out = std::atoi(argv[++a]);
        return true;
      }
      return false;
    };
    if (eat("--n", n)) continue;
    if (eat("--steps", steps)) continue;
    if (eat("--warmup", warmup)) continue;
    if (eat("--distribute", distribute)) continue;
    if (eat("--scaling", scaling)) continue;
    if (eat("--backend", backend)) continue;
    if (eat("--machine", machine)) continue;
  }
  const bool dist = (distribute < 0) ? (n_ranks() > 1) : (distribute != 0);

  AmrSystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;
  cfg.regrid_every = 8;
  cfg.distribute_coarse = dist;
  cfg.coarse_max_grid = dist ? n / 2 : 0;  // 2x2 : decoupage qui ne degrade pas le MG

  AmrSystem sys(cfg);
  add_compiled_model(sys, "gas", Model{Euler{1.4}, GravityForce{}, GravityCoupling{-1.0, 1.0, 1.0}},
                     "minmod", "rusanov", "conservative", "explicit", /*gamma=*/1.4);
  sys.set_poisson("charge_density", "geometric_mg");
  sys.set_refinement(1.2);
  sys.set_density("gas", four_bubbles(n));

  const double dt = 5e-4;
  const double m0 = sys.mass();  // force le build paresseux (regrid initial)
  const int np0 = sys.n_patches();

  for (int s = 0; s < warmup; ++s) sys.step(dt);
  fence();

  std::vector<double> ms;
  ms.reserve(steps);
  const auto wall0 = Clock::now();
  for (int s = 0; s < steps; ++s) {
    const auto s0 = Clock::now();
    sys.step(dt);
    fence();
    ms.push_back(1e3 * std::chrono::duration<double>(Clock::now() - s0).count());
  }
  fence();
  const double wall = std::chrono::duration<double>(Clock::now() - wall0).count();

  const double mass = sys.mass();
  const int npf = sys.n_patches();
  const double drift = std::fabs(mass - m0) / std::max(std::fabs(m0), 1e-30);

  auto rmax = [](double x) { return all_reduce_max(x); };
  const double med = rmax(percentile(ms, 0.5));
  const double p10 = rmax(percentile(ms, 0.10));
  const double p90 = rmax(percentile(ms, 0.90));
  double mean = 0;
  for (double x : ms) mean += x;
  mean /= std::max<size_t>(ms.size(), 1);
  double var = 0;
  for (double x : ms) var += (x - mean) * (x - mean);
  var /= std::max<size_t>(ms.size(), 1);
  const double cv = mean > 0 ? std::sqrt(var) / mean : 0.0;
  const bool finite = std::isfinite(mass) && std::isfinite(med);
  // cells/s : proxy = cellules GROSSIERES n^2 / pas median (le fin ajoute des cellules : voir patches).
  const double cells_per_s = med > 0 ? (double(n) * n) / (med / 1e3) : 0.0;

  if (my_rank() == 0) {
    std::printf(
        "{\"schema\":\"adc_perf_v1\",\"front\":\"cpp_scaling\","
        "\"adc_cpp_sha\":\"%s\",\"adc_cpp_branch\":\"%s\","
        "\"backend\":\"%s\",\"machine\":\"%s\",\"ranks\":%d,\"threads\":%d,\"gpus\":%d,"
        "\"workload\":\"amr\",\"scaling\":\"%s\",\"nx\":%d,\"ny\":%d,\"distribute_coarse\":%s,"
        "\"patches0\":%d,\"patches\":%d,\"warmup\":%d,\"steps\":%d,"
        "\"hot_ms_per_step\":{\"median\":%.6e,\"p10\":%.6e,\"p90\":%.6e,\"cv\":%.6e},"
        "\"cells_per_s\":%.6e,"
        "\"invariants\":{\"mass\":%.10e,\"mass_drift\":%.3e,\"patches\":%d,\"nan\":%s}}\n",
        ADC_BUILD_SHA, ADC_BUILD_BRANCH, backend.c_str(), machine.c_str(), n_ranks(),
        std::atoi(std::getenv("OMP_NUM_THREADS") ? std::getenv("OMP_NUM_THREADS") : "1"), 0,
        scaling.c_str(), n, n, dist ? "true" : "false", np0, npf, warmup, steps, med, p10, p90, cv,
        cells_per_s, mass, drift, npf, finite ? "false" : "true");
    (void)wall;
  }
  return finite ? 0 : 1;
}

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
#ifdef ADC_HAS_KOKKOS
  Kokkos::initialize(argc, argv);
#endif
  int rc = run(argc, argv);
#ifdef ADC_HAS_KOKKOS
  Kokkos::finalize();
#endif
  comm_finalize();
  return rc;
}
