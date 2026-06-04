// Harness GPU de la VALIDATION INTEGREE AmrSystem + MPI + GPU (deliverable C). Superset du test de
// regression tests/test_mpi_amr_compiled_parity.cpp : meme cas (4 bulles, modele euler_poisson
// COMPILE via add_compiled_model, hierarchie AMR avec regrid + reflux + Poisson, grossier replique +
// niveau fin multi-patch distribue sur n_ranks() GPU) MAIS instrumente pour la PERF full-device :
//   - imprime mass / csum / csumsq / cmax (comparables bit a bit entre np=1/2/4 par le script) ;
//   - verifie la consistance cross-rang (grossier replique : spread == 0) ;
//   - mesure le temps PAR PAS (apres warmup + Kokkos::fence pour capturer le travail device async).
//
// Lance par amrmpi_romeo_build.sh en srun -n 1/2/4 --gpus-per-task=1 (un GH200 par rang). Sous Cuda,
// for_each_cell est async ; density()/mass() de l'AmrSystem fencent en interne avant la lecture hote,
// et on encadre la mesure de temps par Kokkos::fence() pour ne pas sous-estimer le cout device.
#include <adc/physics/bricks.hpp>
#include <adc/physics/euler.hpp>
#include <adc/runtime/amr_dsl_block.hpp>
#include <adc/runtime/amr_system.hpp>
#include <adc/parallel/comm.hpp>

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
using Model = CompositeModel<Euler, GravityForce, GravityCoupling>;

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

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fails = 0;
  {
    const int me = my_rank(), np = n_ranks();
    const int n = 128;  // grossier 128^2, fin 256^2 sous les patchs : charge GPU non triviale
    const std::vector<double> rho = four_bubbles(n);

    AmrSystemConfig cfg;
    cfg.n = n;
    cfg.L = 1.0;
    cfg.periodic = true;
    cfg.regrid_every = 8;

    AmrSystem sys(cfg);
    add_compiled_model(sys, "gas",
                       Model{Euler{1.4}, GravityForce{}, GravityCoupling{-1.0, 1.0, 1.0}}, "minmod",
                       "rusanov", "conservative", "explicit", /*gamma=*/1.4);
    sys.set_poisson("charge_density", "geometric_mg");
    sys.set_refinement(1.2);
    sys.set_density("gas", rho);

    const double m0 = sys.mass();  // build paresseux (regrid initial distribue)
    const int np0 = sys.n_patches();

    const double dt = 5e-4;
    const int warmup = 4, measured = 40;
    for (int s = 0; s < warmup; ++s) sys.step(dt);  // warmup (JIT/cache/alloc)
    Kokkos::fence();
    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < measured; ++s) sys.step(dt);
    Kokkos::fence();  // capturer le travail device async avant de stopper le chrono
    const auto t1 = std::chrono::steady_clock::now();
    const double per_step_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / measured;

    Kokkos::fence();
    const std::vector<double> dens = sys.density();
    const double mass = sys.mass();
    const int npf = sys.n_patches();

    double csum = 0, csumsq = 0, cmax = 0;
    for (double v : dens) {
      csum += v;
      csumsq += v * v;
      const double a = std::fabs(v);
      if (a > cmax) cmax = a;
    }
    const double smax = all_reduce_max(csum), smin = -all_reduce_max(-csum);
    const double qmax = all_reduce_max(csumsq), qmin = -all_reduce_max(-csumsq);
    const double mmax = all_reduce_max(mass), mmin = -all_reduce_max(-mass);
    const double xmax = all_reduce_max(cmax), xmin = -all_reduce_max(-cmax);
    const double spread = std::fmax(std::fmax(smax - smin, qmax - qmin),
                                    std::fmax(mmax - mmin, xmax - xmin));
    const double maxstep = all_reduce_max(per_step_ms);  // pas le plus lent (le mur)

    if (me == 0) {
      std::printf("AMRMPI np=%d patches0=%d patchesF=%d | mass=%.17e | csum=%.17e csumsq=%.17e "
                  "cmax=%.17e | crossrank_spread=%.3e\n",
                  np, np0, npf, mass, csum, csumsq, cmax, spread);
      std::printf("AMRMPI exec=%s m0=%.17e (conservation: dm=%.3e) | per_step_ms=%.4f "
                  "(max over ranks, n=%d, measured=%d)\n",
                  Kokkos::DefaultExecutionSpace::name(), m0, std::fabs(mass - m0), maxstep, n,
                  measured);
      if (!(dens.size() == static_cast<std::size_t>(n) * n)) { std::printf("FAIL taille\n"); ++fails; }
      if (!(cmax > 1e-6)) { std::printf("FAIL densite triviale\n"); ++fails; }
      if (!(npf >= 2)) { std::printf("FAIL < 2 patchs fins\n"); ++fails; }
      if (!(spread == 0.0)) { std::printf("FAIL grossier non bit-identique entre rangs\n"); ++fails; }
      if (fails == 0)
        std::printf("OK amrmpi_integrated np=%d (AmrSystem+MPI+GPU compile : bit-identique cross-rang)\n",
                    np);
    }
  }
  Kokkos::finalize();
  comm_finalize();
  return fails ? 1 : 0;
}
