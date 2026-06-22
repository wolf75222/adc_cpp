// Compiled time-program LOADER path (epic ADC-399 / ADC-401 Phase 2c-i): System::install_program
// dlopens a generated problem.so and installs its compiled time Program across the ABI boundary.
//
// We compile AT RUNTIME a stub problem.so -- the role the codegen (Phase 2c-ii) will fill -- that
// exports adc_program_abi_key() + adc_install_program(void* sys); the installer wraps the System in a
// ProgramContext and installs the SAME Forward-Euler closure as the in-process test_program_runtime.
// We then sim.install_program(so) + sim.step(dt) and check bit-parity against a reference Forward-Euler
// step computed from the same primitives (solve_fields + eval_rhs + U + dt*R). This validates the
// dlopen + ABI-key guard + RTLD_GLOBAL resolution of the seam accessors, end to end.
//
// Skips (exit 0) under Kokkos (a nu CPU loader is ABI-incompatible with the device module) or when no
// C++ compiler is known to the build -- same policy as test_amr_native_loader. CMake injects
// ADC_TEST_CXX / ADC_TEST_INCLUDE / ADC_TEST_CXX_STD / ADC_TEST_TMPDIR and sets ENABLE_EXPORTS so the
// .so resolves the exported System seam symbols against this executable.

#include <adc/mesh/storage/multifab.hpp>
#include <adc/physics/bricks/source.hpp>                // NoSource
#include <adc/physics/composition/composite.hpp>        // CompositeModel
#include <adc/physics/fluids/euler.hpp>                 // Euler
#include <adc/runtime/builders/compiled/dsl_block.hpp>  // add_compiled_model
#include <adc/runtime/system.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace adc;

namespace {

struct NoEll {
  template <class State>
  ADC_HD Real rhs(const State&) const {
    return Real(0);
  }
};
using GasModel = CompositeModel<Euler, NoSource, NoEll>;
constexpr double kGamma = 1.4;

void fill_ic(std::vector<double>& U, int n) {
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  const double pi = 3.14159265358979323846;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const std::size_t k = static_cast<std::size_t>(j) * n + i;
      const double x = (i + 0.5) / n, y = (j + 0.5) / n;
      const double p = 3.0 + 0.5 * std::cos(2 * pi * x) * std::cos(2 * pi * y);
      U[0 * nn + k] = 1.0;
      U[1 * nn + k] = 0.0;
      U[2 * nn + k] = 0.0;
      U[3 * nn + k] = p / (kGamma - 1.0);
    }
}

void add_gas(System& s) {
  add_compiled_model(s, "gas", GasModel{Euler{kGamma}, NoSource{}, NoEll{}}, "minmod", "rusanov",
                     "conservative", "explicit", kGamma);
  s.set_poisson("charge_density", "geometric_mg");
}

// The generated problem.so: a Forward-Euler Program installed via ProgramContext. This is exactly the
// source the Phase 2c-ii codegen will emit (here hand-written for an autonomous C++ test). The ABI key
// is the preprocessor LITERAL (not the inline abi_key_string(), which would be interposed via RTLD).
std::string loader_source() {
  // clang-format off
  return R"CPP(
#include <adc/runtime/program/program_context.hpp>
#include <adc/runtime/dynamic/abi_key.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/core/foundation/types.hpp>
extern "C" const char* adc_program_abi_key() { return ADC_ABI_KEY_LITERAL; }
extern "C" const char* adc_program_name() { return "forward_euler_stub"; }
extern "C" void adc_install_program(void* sys) {
  adc::runtime::program::ProgramContext ctx(sys);
  ctx.install([ctx](double dt) {
    ctx.solve_fields();
    for (int b = 0; b < ctx.n_blocks(); ++b) {
      adc::MultiFab& U = ctx.state(b);
      adc::MultiFab R = ctx.rhs_scratch_like(U);
      ctx.rhs_into(b, U, R);
      ctx.axpy(U, static_cast<adc::Real>(dt), R);
    }
  });
}
)CPP";
  // clang-format on
}

bool compile_loader(const std::string& src_path, const std::string& so_path) {
#if defined(__APPLE__)
  const std::string cc =
      "/usr/bin/c++";  // xcrun wrapper: resolves the SDK sysroot (same clang family)
#else
  const std::string cc = ADC_TEST_CXX;
#endif
  std::string cmd = cc + " -shared -fPIC -std=" + ADC_TEST_CXX_STD + " -O2 -I " + ADC_TEST_INCLUDE +
                    " " + src_path + " -o " + so_path;
#if defined(__APPLE__)
  cmd += " -undefined dynamic_lookup";  // undefined System symbols resolved at load from the exe
#endif
  cmd += " 2> /dev/null";
  return std::system(cmd.c_str()) == 0;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(ADC_HAS_KOKKOS)
  (void)argc;
  (void)argv;
  std::printf("skip test_program_loader (backend Kokkos : nu CPU loader incompatible)\n");
  return 0;
#else
  (void)argc;
  (void)argv;
  const char* cxx = ADC_TEST_CXX;
  if (!cxx || cxx[0] == '\0') {
    std::printf("skip test_program_loader (aucun compilateur C++ connu du build)\n");
    return 0;
  }

  const int n = 16;
  const double dt = 1e-3;
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  std::vector<double> U0(4 * nn);
  fill_ic(U0, n);

  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;

  // Reference: one Forward-Euler step via the existing primitives, combined on the host.
  System ref(cfg);
  add_gas(ref);
  ref.set_state("gas", U0);
  ref.solve_fields();
  const std::vector<double> R0 = ref.eval_rhs("gas");
  std::vector<double> Uref(4 * nn);
  for (std::size_t k = 0; k < Uref.size(); ++k)
    Uref[k] = U0[k] + dt * R0[k];

  // Compile the stub problem.so and load it via System::install_program.
  const std::string tmp = std::string(ADC_TEST_TMPDIR) + "/program_loader_" +
                          std::to_string(static_cast<long>(std::clock()));
  const std::string src = tmp + ".cpp";
  const std::string so = tmp + ".so";
  {
    std::ofstream f(src);
    f << loader_source();
  }
  if (!compile_loader(src, so)) {
    std::printf("skip test_program_loader (echec de compilation du stub .so -- en-tetes/std ?)\n");
    return 0;
  }

  System sim(cfg);
  add_gas(sim);
  sim.set_state("gas", U0);
  sim.install_program(so);  // dlopen + ABI check + adc_install_program(this)
  const int step0 = sim.macro_step();
  sim.step(dt);  // SystemStepper dispatches to the installed Program
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
    std::printf("FAIL loaded program did not change the state (change = %.3e)\n", change);
    ++fails;
  }

  if (fails == 0)
    std::printf(
        "OK test_program_loader (problem.so Forward Euler via install_program == reference; "
        "max|d| = %.2e, change = %.2e)\n",
        err, change);
  return fails ? 1 : 0;
#endif
}
