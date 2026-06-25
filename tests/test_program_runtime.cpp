// Compiled time-program runtime seam (epic ADC-399 / ADC-401 Phase 2b): a Forward-Euler Program,
// installed as a macro-step closure via adc::runtime::program::ProgramContext, runs C++-side during
// sim.step(dt). This test proves the seam end-to-end WITHOUT codegen or a .so: it builds the closure
// in C++ (the role the generated problem.so will later fill) and checks bit-parity against a reference
// Forward-Euler step computed from the SAME existing primitives (solve_fields + eval_rhs + U + dt*R).
//
// Model: a compressible Euler gas with a NON-UNIFORM pressure IC (u = v = 0), so -div F has a non-zero
// momentum component -> the step actually changes the state (parity is not vacuous). No source, no
// charge (NoEll), so the result is pure gas dynamics and deterministic across two System instances.

#include <adc/mesh/storage/multifab.hpp>
#include <adc/physics/bricks/source.hpp>                // NoSource
#include <adc/physics/composition/composite.hpp>        // CompositeModel
#include <adc/physics/fluids/euler.hpp>                 // Euler
#include <adc/runtime/builders/compiled/dsl_block.hpp>  // add_compiled_model
#include <adc/runtime/program/program_context.hpp>      // ProgramContext (the seam under test)
#include <adc/runtime/system.hpp>

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace adc;

// Elliptic brick that contributes nothing (no charge): the Poisson RHS stays zero, phi = 0, and the
// Euler flux ignores aux -> the residual is pure gas dynamics.
struct NoEll {
  template <class State>
  ADC_HD Real rhs(const State&) const {
    return Real(0);
  }
};
using GasModel = CompositeModel<Euler, NoSource, NoEll>;

static void fill_ic(std::vector<double>& U, int n, double gamma) {
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  const double pi = 3.14159265358979323846;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const std::size_t k =
          static_cast<std::size_t>(j) * n + i;  // j slow, i fast (get_state layout)
      const double x = (i + 0.5) / n, y = (j + 0.5) / n;
      const double p =
          3.0 + 0.5 * std::cos(2 * pi * x) * std::cos(2 * pi * y);  // periodic, non-uniform
      U[0 * nn + k] = 1.0;                                          // rho
      U[1 * nn + k] = 0.0;                                          // rho u
      U[2 * nn + k] = 0.0;                                          // rho v
      U[3 * nn + k] = p / (gamma - 1.0);                            // E (u = v = 0)
    }
}

static void add_gas(System& s, double gamma) {
  add_compiled_model(s, "gas", GasModel{Euler{gamma}, NoSource{}, NoEll{}}, "minmod", "rusanov",
                     "conservative", "explicit", gamma);
  s.set_poisson("charge_density", "geometric_mg");
}

int main(int argc, char** argv) {
#if defined(ADC_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  const int n = 16;
  const double gamma = 1.4, dt = 1e-3;
  const std::size_t nn = static_cast<std::size_t>(n) * n;

  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;

  std::vector<double> U0(4 * nn);
  fill_ic(U0, n, gamma);

  // Reference: one Forward-Euler step via the existing primitives, combined on the host.
  System ref(cfg);
  add_gas(ref, gamma);
  ref.set_state("gas", U0);
  ref.solve_fields();
  const std::vector<double> R0 = ref.eval_rhs("gas");
  std::vector<double> Uref(4 * nn);
  for (std::size_t k = 0; k < Uref.size(); ++k)
    Uref[k] = U0[k] + dt * R0[k];

  // Program: the SAME step expressed as a ProgramContext closure and driven by sim.step(dt).
  System sim(cfg);
  add_gas(sim, gamma);
  sim.set_state("gas", U0);

  runtime::program::ProgramContext ctx(&sim);
  ctx.install([ctx](double h) {
    ctx.solve_fields();
    for (int b = 0; b < ctx.n_blocks(); ++b) {
      MultiFab& U = ctx.state(b);
      MultiFab R = ctx.rhs_scratch_like(U);
      ctx.rhs_into(b, U, R);
      ctx.axpy(U, Real(h), R);  // U <- U + h * R  (Forward Euler)
    }
  });

  // Profiling counters (ADC-459, Spec 3 section 29): enable the System Profiler, so the ProgramContext
  // seam ops the step body calls (solve_fields, rhs_into, axpy) bump "kernels" and rhs_scratch_like
  // records the scratch peak. This is the HOST-validatable path (a ProgramContext built directly in
  // C++, no compiled .so); the cache hit/skip counters need a held schedule the codegen emits, so they
  // are exercised on the Kokkos/ROMEO compiled-.so runtime, not here.
  sim.enable_profiling();
  const int step0 = sim.macro_step();
  sim.step(dt);
  const std::vector<double> Up = sim.get_state("gas");

  int fails = 0;
  double err = 0, change = 0;
  for (std::size_t k = 0; k < Up.size(); ++k) {
    err = std::fmax(err, std::fabs(Up[k] - Uref[k]));
    change = std::fmax(change, std::fabs(Up[k] - U0[k]));
  }
  if (!(err < 1e-12)) {
    std::printf("FAIL parity: max|Up - Uref| = %.3e\n", err);
    ++fails;
  }
  if (sim.macro_step() != step0 + 1) {
    std::printf("FAIL macro_step not advanced (%d -> %d)\n", step0, sim.macro_step());
    ++fails;
  }
  if (!(change > 1e-9)) {
    std::printf("FAIL program step did not change the state (change = %.3e)\n", change);
    ++fails;
  }

  // ADC-459 counters: one step ran solve_fields + (1 block) rhs_into + axpy = EXACTLY 3 kernel-
  // dispatching seam ops (no double-count: solve_fields counts once, via Impl::solve_fields). Pinning
  // the exact value guards against a seam double-counting (a >0 check would not).
  const runtime::program::Profiler& prof = sim.profiler();
  if (prof.counter("kernels") != 3) {
    std::printf("FAIL kernels counter = %lld, expected 3 (solve_fields + rhs_into + axpy, no double)\n",
                static_cast<long long>(prof.counter("kernels")));
    ++fails;
  }
  if (!(prof.counter("scratch_allocs") > 0)) {
    std::printf("FAIL scratch_allocs counter not incremented (= %lld)\n",
                static_cast<long long>(prof.counter("scratch_allocs")));
    ++fails;
  }
  if (!(prof.counter("scratch_peak_bytes") > 0)) {
    std::printf("FAIL scratch_peak_bytes not recorded (= %lld)\n",
                static_cast<long long>(prof.counter("scratch_peak_bytes")));
    ++fails;
  }
  // The cache hit/skip counters never fire on this native ProgramContext step (no held schedule); they
  // exist as counters only after the compiled scheduler emits cache_should_update. Assert they read 0.
  if (prof.counter("cache_hits") != 0 || prof.counter("cache_misses") != 0) {
    std::printf("FAIL cache counters moved on the native path (hits=%lld misses=%lld)\n",
                static_cast<long long>(prof.counter("cache_hits")),
                static_cast<long long>(prof.counter("cache_misses")));
    ++fails;
  }
  {
    const std::string report = sim.profile_report();
    if (report.find("kernels=") == std::string::npos) {
      std::printf("FAIL profile_report omits the kernels counter line\n");
      ++fails;
    }
  }

  if (fails == 0)
    std::printf(
        "OK test_program_runtime (program Forward Euler == eval_rhs reference; "
        "max|d| = %.2e, change = %.2e)\n",
        err, change);
  return fails ? 1 : 0;
}
