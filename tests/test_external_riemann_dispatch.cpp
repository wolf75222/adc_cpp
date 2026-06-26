// End-to-end external C++ Riemann brick: build a real .so, dlopen it, dispatch its flux (ADC-463).
//
// test_external_brick.cpp covers the host identity registry in isolation. THIS test closes the
// deferred half of ADC-463 (Spec 3 section 21-22, criterion 20): an external brick shipped in a
// standalone .so is dlopen'd and its flux DISPATCHED into the finite-volume machinery in the SAME
// type system as a native flux -- statically (build_block<Limiter, UserFlux> inside the .so), never a
// per-cell string lookup. It mirrors test_amr_native_loader.cpp: the brick source is written and
// compiled to a .so at run time (so no committed binary), then loaded.
//
// VALIDATIONS:
//   1. ExternalBrickHandle dlopens the .so, reads its manifest, and exposes the brick's id +
//      requirements (the manifest is visible in the same registry native bricks would use).
//   2. The resolved residual entry point runs the brick's flux. The brick wraps pops::RusanovFlux, so
//      its residual is compared BIT-FOR-BIT against the native rusanov path (make_block "rusanov"):
//      the external brick is dispatched into identical numerics -> dmax == 0.
//   3. An unknown id is rejected with a clear error.

#include <pops/runtime/program/external_riemann_brick.hpp>

#include <pops/runtime/builders/compiled/compiled_block_abi.hpp>  // compiled_block::residual (native ref)
#include <pops/physics/bricks/bricks.hpp>                         // CompositeModel / Euler / ...

#include "test_harness.hpp"  // pops::test::Checker

#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>

using pops::runtime::program::ExternalBrickHandle;

namespace {

constexpr double kGamma = 1.4;

// The C++ an advanced user ships in my_riemann.cpp: a NumericalFlux policy (here a thin wrapper over
// the native RusanovFlux so the test can prove BIT-IDENTICAL dispatch) + the two macros that register
// the brick and emit its static-dispatch ABI. The Model is fixed in the .so (Euler), exactly as a
// real external brick would pin its target model.
std::string brick_source() {
  // clang-format off
  return R"CPP(
#include <pops/runtime/program/external_riemann_brick.hpp>
#include <pops/physics/bricks/bricks.hpp>

// The user's flux: same single-interface contract as pops::RusanovFlux (numerical_flux.hpp). Here it
// forwards to RusanovFlux so the test can assert bit-identical dispatch; a real brick would compute
// its own interface flux. POPS_HD: device-callable, no virtuals.
struct UserRusanov {
  template <class Model>
  POPS_HD typename Model::State operator()(const Model& m, const typename Model::State& UL,
                                          const pops::Aux& AL, const typename Model::State& UR,
                                          const pops::Aux& AR, int dir) const {
    return pops::RusanovFlux{}(m, UL, AL, UR, AR, dir);
  }
};

namespace user_brick {
using Model = pops::CompositeModel<pops::Euler, pops::NoSource, pops::BackgroundDensity>;
}

POPS_DEFINE_EXTERNAL_RIEMANN_BRICK("my_riemann", UserRusanov, user_brick::Model, "max_wave_speed");
POPS_DEFINE_BRICK_MANIFEST();
)CPP";
  // clang-format on
}

// Splices a space-separated list of include dirs into `-I dir` flags (the Kokkos interface include
// dirs CMake hands us as one space-joined string). Empty input -> "" (the non-Kokkos host path).
std::string include_flags(const char* dirs) {
  std::string out;
  std::string token;
  const std::string s = (dirs != nullptr) ? dirs : "";
  for (std::size_t i = 0; i <= s.size(); ++i) {
    if (i == s.size() || s[i] == ' ') {
      if (!token.empty())
        out += " -I " + token;
      token.clear();
    } else {
      token.push_back(s[i]);
    }
  }
  return out;
}

// Compile the brick to a .so (g++/c++ at run time). Returns true on success. Mirrors
// test_amr_native_loader.cpp::compile_loader (same SDK / std handling on macOS) AND, under Kokkos,
// adds this module's Kokkos interface include dirs / defines / options so the brick .so is ABI-
// compatible with the test binary (for_each.hpp #errors without POPS_HAS_KOKKOS).
bool compile_brick(const std::string& src, const std::string& so) {
#if defined(__APPLE__)
  const std::string cc = "/usr/bin/c++";
#else
  const std::string cc = POPS_TEST_CXX;
#endif
  std::string cmd = cc + " -shared -fPIC -std=" + POPS_TEST_CXX_STD + " -O2 -I " + POPS_TEST_INCLUDE +
                    " " + src + " -o " + so;
#if defined(POPS_HAS_KOKKOS)
  // Match the module's Kokkos ABI: its interface includes (Kokkos + libomp), defines and options.
  cmd += include_flags(POPS_TEST_KOKKOS_INC);
  // INTERFACE_COMPILE_OPTIONS may carry CMake's `SHELL:` escape (e.g. "SHELL:-Xpreprocessor -fopenmp
  // -I.../libomp/include" for AppleClang OpenMP); strip the literal `SHELL:` token, the rest is a
  // plain flag list the compiler accepts as-is.
  {
    std::string opts = POPS_TEST_KOKKOS_OPTS;
    for (std::size_t p = opts.find("SHELL:"); p != std::string::npos; p = opts.find("SHELL:"))
      opts.erase(p, 6);
    cmd += " " + opts;
  }
  // INTERFACE_COMPILE_DEFINITIONS are bare tokens (KOKKOS_DEPENDENCE, ...); prefix each with -D.
  {
    const std::string defs = POPS_TEST_KOKKOS_DEFS;
    std::string token;
    for (std::size_t i = 0; i <= defs.size(); ++i) {
      if (i == defs.size() || defs[i] == ' ') {
        if (!token.empty())
          cmd += " -D" + token;
        token.clear();
      } else {
        token.push_back(defs[i]);
      }
    }
  }
  cmd += " -DPOPS_HAS_KOKKOS";
#endif
#if defined(__APPLE__)
  cmd += " -undefined dynamic_lookup";
#endif
  cmd += " 2> " + so + ".log";
  return std::system(cmd.c_str()) == 0;
}

// A smooth periodic Euler state (rho, mx, my, E) in component-major layout c*n*n + j*n + i.
std::vector<double> euler_state(int n) {
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  std::vector<double> U(4 * nn);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5;
      const double rho = 1.0 + 0.3 * std::exp(-(x * x + y * y) / 0.02);
      const double u = 0.1, v = -0.05, p = 1.0;
      const std::size_t k = static_cast<std::size_t>(j) * n + i;
      U[0 * nn + k] = rho;
      U[1 * nn + k] = rho * u;
      U[2 * nn + k] = rho * v;
      U[3 * nn + k] = p / (kGamma - 1.0) + 0.5 * rho * (u * u + v * v);
    }
  return U;
}

using RefModel = pops::CompositeModel<pops::Euler, pops::NoSource, pops::BackgroundDensity>;

}  // namespace

int main() {
  const char* cxx = POPS_TEST_CXX;
  if (cxx == nullptr || cxx[0] == '\0') {
    std::printf("skip test_external_riemann_dispatch (no C++ compiler known to the build)\n");
    return 0;
  }

  const std::string tmp = std::string(POPS_TEST_TMPDIR) + "/external_riemann_" +
                          std::to_string(static_cast<long>(std::clock()));
  const std::string src = tmp + ".cpp", so = tmp + ".so";
  {
    std::ofstream f(src);
    f << brick_source();
  }
  if (!compile_brick(src, so)) {
    std::printf(
        "skip test_external_riemann_dispatch (brick .so failed to compile -- headers/std?)\n");
    return 0;  // self-skip on a toolchain that cannot build the brick (cf. test_amr_native_loader)
  }

  pops::test::Checker chk;

  // (1) dlopen + manifest visibility + requirements surface.
  ExternalBrickHandle handle(so, "my_riemann");
  chk(handle.id() == "my_riemann", "handle_id");
  chk(handle.requirements() == "max_wave_speed", "requirements_surface");
  chk(handle.residual() != nullptr, "residual_resolved");
  // The dlopen registered the manifest in this image's process catalog too.
  const auto* entry = pops::runtime::program::BrickRegistry::instance().lookup("my_riemann");
  chk(entry != nullptr && entry->category == "riemann", "manifest_visible_in_registry");

  // (2) BIT-IDENTICAL dispatch: external brick residual == native rusanov residual.
  const int n = 48;
  const double dx = 1.0 / n, dy = 1.0 / n;
  const std::vector<double> U = euler_state(n);
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  std::vector<double> Rext(4 * nn, 0.0), Rnat(4 * nn, 0.0);

  // External brick: the resolved entry point dispatches build_block<Minmod, UserRusanov> inside the .so.
  handle.residual()(U.data(), Rext.data(), /*aux=*/nullptr, n, dx, dy, /*periodic=*/1, "minmod",
                    /*recon_prim=*/0, /*pos_floor=*/0.0);
  // Native reference: the SAME path with the native rusanov flux (compiled_block::residual -> make_block).
  pops::compiled_block::residual<RefModel>(U.data(), Rnat.data(), /*aux=*/nullptr, n, dx, dy,
                                          /*periodic=*/true, "minmod", "rusanov",
                                          /*recon_prim=*/false);
  double dmax = 0.0, nrm = 0.0;
  for (std::size_t k = 0; k < Rext.size(); ++k) {
    const double d = std::fabs(Rext[k] - Rnat[k]);
    if (d > dmax)
      dmax = d;
    const double a = std::fabs(Rnat[k]);
    if (a > nrm)
      nrm = a;
  }
  chk(nrm > 1e-8, "native_residual_nontrivial");
  chk(dmax == 0.0, "external_dispatch_bit_identical_to_native_rusanov");

  // (3) Unknown id -> clear error.
  bool threw = false;
  try {
    ExternalBrickHandle bad(so, "no_such_brick");
  } catch (const std::runtime_error& e) {
    threw = true;
    const std::string msg = e.what();
    chk(msg.find("no_such_brick") != std::string::npos, "unknown_id_names_id");
  }
  chk(threw, "unknown_id_rejected");

  return chk.failed();
}
