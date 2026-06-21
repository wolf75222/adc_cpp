// HLLC/Roe + reconstruction primitive sur le chemin "production" NATIF cote AMR (Gap 1 parite).
// Pendant de tests/test_amr_weno5_native.cpp : prouve que riemann=hllc/roe ET recon=primitive
// tournent la MEME hierarchie AMR (AmrCouplerMP<Model> + reflux + regrid), BIT-IDENTIQUE, sur les
// trois chemins, et que hllc/roe produisent un resultat DIFFERENT de rusanov (flux actif, non muet).
//
//   (A) add_compiled_model(AmrSystem&) == add_block(ModelSpec)   -- direct, sans .so : tourne sous
//       TOUS les backends (hote, Kokkos Serial), c'est la parite decisive qui ne casse pas nvcc.
//       4 combos : (hllc/roe) x (conservative/primitive), chacun dmax==0.
//       + rusanov conservatif (oracle) -> hllc != rusanov et roe != rusanov (flux actif).
//
//   (B) add_native_block(loader.so) == add_compiled_model(AmrSystem&) -- chemin .so, autocompile a
//       l'execution (source du loader AMR, cf. dsl.emit_cpp_native_loader(target="amr_system")).
//       Auto-saute sous backend Kokkos (ABI incompatible) ; la parite CPU reste couverte par (A).
//
// Le modele est un Euler PUR (CompositeModel<Euler, NoSource, BackgroundDensity{alpha=0}>) : la
// brique elliptique vaut 0, phi=0 (zero bruit FP), parite STRICTE. La pression est requise pour
// hllc/roe (cf. Euler::pressure()), d'ou le choix d'Euler.
//
// CMake injecte ADC_TEST_CXX, ADC_TEST_INCLUDE, ADC_TEST_CXX_STD, ADC_TEST_TMPDIR (meme pattern
// que test_amr_weno5_native).
#include <adc/physics/bricks.hpp>  // CompositeModel, Euler, NoSource, BackgroundDensity
#include <adc/runtime/amr_dsl_block.hpp>
#include <adc/runtime/amr_system.hpp>
#include <adc/runtime/model_spec.hpp>

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

using ProdModel = CompositeModel<Euler, NoSource, BackgroundDensity>;
constexpr double kGamma = 1.4;

ProdModel make_model() {
  // alpha=0 : elliptic_rhs nul -> phi=0, parite stricte.
  return ProdModel{Euler{static_cast<Real>(kGamma)}, NoSource{},
                   BackgroundDensity{Real(0), Real(0)}};
}

// ModelSpec equivalente pour le chemin add_block (dispatch_amr_compiled -> MEME type concret).
ModelSpec make_spec() {
  ModelSpec spec;
  spec.transport = "compressible";
  spec.source = "none";
  spec.elliptic = "background";
  spec.gamma = kGamma;
  spec.alpha = 0.0;
  spec.n0 = 0.0;
  return spec;
}

std::vector<double> bubble(int n) {
  std::vector<double> rho(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5;
      rho[static_cast<std::size_t>(j) * n + i] = 1.0 + 0.5 * std::exp(-(x * x + y * y) / 0.02);
    }
  return rho;
}

AmrSystemConfig make_cfg(int n) {
  AmrSystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;
  cfg.regrid_every = 4;
  return cfg;
}

struct Snap {
  std::vector<double> density;
  double mass = 0;
  int n_patches = 0;
};

Snap run(AmrSystem& s, int nsteps) {
  s.set_poisson("charge_density", "geometric_mg");
  s.set_refinement(1.2);
  const double dt = 2e-4;
  for (int k = 0; k < nsteps; ++k)
    s.step(dt);
  return Snap{s.density(), s.mass(), s.n_patches()};
}

double maxdiff(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0;
  for (std::size_t k = 0; k < a.size() && k < b.size(); ++k)
    d = std::fmax(d, std::fabs(a[k] - b[k]));
  return d;
}
double maxabs(const std::vector<double>& a) {
  double m = 0;
  for (double v : a)
    m = std::fmax(m, std::fabs(v));
  return m;
}

// Source du loader AMR : MEME forme que dsl.emit_cpp_native_loader(target="amr_system"), modele en dur.
std::string loader_source() {
  // Generated C++ source raw string: clang-format would reindent (or, with the
  // interleaved R"CPP( delimiters, runaway-indent) the inner content. Fence it to keep the
  // emitted source verbatim.
  // clang-format off
  return R"CPP(
#include <adc/runtime/amr_dsl_block.hpp>
#include <adc/runtime/abi_key.hpp>
#include <adc/physics/bricks.hpp>
#include <string>
namespace adc_generated {
using ProdModel = adc::CompositeModel<adc::Euler, adc::NoSource, adc::BackgroundDensity>;
}
// LITTERAL preprocesseur (PAS abi_key_string() : une inline serait interposee, ELF/RTLD_GLOBAL,
// vers la copie du module deja charge -> cle du module renvoyee -> garde d'ABI tautologique).
extern "C" const char* adc_native_abi_key() { return ADC_ABI_KEY_LITERAL; }
extern "C" void adc_install_native_amr(void* sys, const char* name, const char* limiter,
                                       const char* riemann, const char* recon, const char* time,
                                       double gamma, int substeps) {
  adc::AmrSystem* s = reinterpret_cast<adc::AmrSystem*>(sys);
  adc::add_compiled_model<adc_generated::ProdModel>(
      *s, name,
      adc_generated::ProdModel{adc::Euler{static_cast<adc::Real>(gamma)}, adc::NoSource{},
                               adc::BackgroundDensity{adc::Real(0), adc::Real(0)}},
      limiter, riemann, recon, time, gamma, substeps);
}
)CPP";
  // clang-format on
}

bool compile_loader(const std::string& src_path, const std::string& so_path) {
#if defined(__APPLE__)
  const std::string cc = "/usr/bin/c++";
#else
  const std::string cc = ADC_TEST_CXX;
#endif
  std::string cmd = cc + " -shared -fPIC -std=" + ADC_TEST_CXX_STD + " -O2 -I " + ADC_TEST_INCLUDE +
                    " " + src_path + " -o " + so_path;
#if defined(__APPLE__)
  cmd += " -undefined dynamic_lookup";
#endif
  cmd += " 2> /dev/null";
  return std::system(cmd.c_str()) == 0;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(ADC_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  const int n = 64;
  const int nsteps = 12;
  const std::vector<double> rho = bubble(n);

  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  // ============================================================================================
  // (A) PARITE DECISIVE (sans .so) : add_compiled_model(AmrSystem&) == add_block(ModelSpec)
  //     pour riemann=hllc/roe et recon=conservative/primitive (4 combos).
  //     Tourne sous TOUS les backends (hote, Kokkos Serial) -> ne casse pas nvcc.
  // ============================================================================================

  // Calcule l'oracle rusanov conservatif (pour le NO-SILENT-FALLBACK hllc/roe != rusanov).
  std::vector<double> d_rusanov;
  {
    AmrSystem Ref(make_cfg(n));
    add_compiled_model(Ref, "gas", make_model(), "minmod", "rusanov", "conservative", "explicit",
                       kGamma);
    Ref.set_density("gas", rho);
    d_rusanov = run(Ref, nsteps).density;
  }

  // Parite add_compiled_model == add_block pour les 4 combos.
  // Renvoie la densite add_compiled_model (pour le test rusanov != riemann).
  auto parity_direct = [&](const char* riem, const char* recon) -> std::vector<double> {
    AmrSystem A(make_cfg(n));  // bloc COMPILE
    add_compiled_model(A, "gas", make_model(), "minmod", riem, recon, "explicit", kGamma);
    A.set_density("gas", rho);
    const Snap sa = run(A, nsteps);

    AmrSystem B(make_cfg(n));  // bloc NATIF via add_block
    B.add_block("gas", make_spec(), "minmod", riem, recon, "explicit", 1);
    B.set_density("gas", rho);
    const Snap sb = run(B, nsteps);

    const double nrm = maxabs(sb.density), dmax = maxdiff(sa.density, sb.density);
    char w[200];
    std::snprintf(w, sizeof w, "[%s/%s] densite non triviale", riem, recon);
    chk(nrm > 1e-6, w);
    std::snprintf(w, sizeof w, "[%s/%s] add_compiled_model == add_block (dmax==0)", riem, recon);
    chk(dmax == 0.0, w);
    std::snprintf(w, sizeof w, "[%s/%s] masse add_compiled_model == add_block", riem, recon);
    chk(std::fabs(sa.mass - sb.mass) < 1e-12 * (std::fabs(sb.mass) + 1.0), w);
    std::snprintf(w, sizeof w, "[%s/%s] n_patches identique (regrid deterministe)", riem, recon);
    chk(sa.n_patches == sb.n_patches, w);
    std::printf("OK  [%s/%s] dmax=%.0f\n", riem, recon, dmax);
    return sa.density;
  };

  const std::vector<double> d_hllc_cons = parity_direct("hllc", "conservative");
  const std::vector<double> d_hllc_prim = parity_direct("hllc", "primitive");
  const std::vector<double> d_roe_cons = parity_direct("roe", "conservative");
  const std::vector<double> d_roe_prim = parity_direct("roe", "primitive");

  // NO-SILENT-FALLBACK : hllc et roe doivent differer de rusanov sur ce meme etat (la bulle
  // est un ecoulement compressible non-trivial : hllc/roe ne se reduisent PAS a rusanov).
  chk(maxdiff(d_hllc_cons, d_rusanov) > 1e-12,
      "hllc != rusanov (le flux hllc est actif, non silencieux)");
  chk(maxdiff(d_roe_cons, d_rusanov) > 1e-12,
      "roe != rusanov (le flux roe est actif, non silencieux)");

  std::printf(
      "OK  (A) 4 combos hllc/roe x conservative/primitive BIT-IDENTIQUES (dmax==0) ; "
      "hllc != rusanov, roe != rusanov (flux actifs)\n");

  // ============================================================================================
  // (B) CHEMIN .so : add_native_block(loader) == add_compiled_model(AmrSystem&), hllc ET roe.
  //     Le loader est recompile par un g++ nu -> incompatible avec un module Kokkos : SAUTE.
  //     (A) a deja couvert la parite CPU complete.
  // ============================================================================================
#if defined(ADC_HAS_KOKKOS)
  std::printf("skip (B) loader .so (backend Kokkos : loader CPU nu incompatible)\n");
#else
  const char* cxx = ADC_TEST_CXX;
  if (!cxx || cxx[0] == '\0') {
    std::printf("skip (B) loader .so (aucun compilateur C++ connu du build)\n");
  } else {
    const std::string tmp = std::string(ADC_TEST_TMPDIR) + "/amr_riemann_native_" +
                            std::to_string(static_cast<long>(std::clock()));
    const std::string src = tmp + ".cpp";
    const std::string so = tmp + ".so";
    {
      std::ofstream f(src);
      f << loader_source();
    }
    if (!compile_loader(src, so)) {
      std::printf("skip (B) loader .so (echec de compilation du loader -- en-tetes/std ?)\n");
    } else {
      auto parity_loader = [&](const char* riem, const char* recon) {
        AmrSystem A(make_cfg(n));  // chemin "production" : loader .so -> add_native_block
        A.add_native_block("gas", so, "minmod", riem, recon, "explicit", kGamma, 1);
        A.set_density("gas", rho);
        const Snap sa = run(A, nsteps);

        AmrSystem B(make_cfg(n));  // MEME modele installe EN DIRECT
        add_compiled_model(B, "gas", make_model(), "minmod", riem, recon, "explicit", kGamma);
        B.set_density("gas", rho);
        const Snap sb = run(B, nsteps);

        const double dmax = maxdiff(sa.density, sb.density);
        char w[200];
        std::snprintf(w, sizeof w, "[%s/%s] add_native_block == add_compiled_model (dmax==0)", riem,
                      recon);
        chk(dmax == 0.0, w);
        std::snprintf(w, sizeof w, "[%s/%s] n_patches loader == direct", riem, recon);
        chk(sa.n_patches == sb.n_patches, w);
      };
      parity_loader("hllc", "conservative");
      parity_loader("hllc", "primitive");
      parity_loader("roe", "conservative");
      parity_loader("roe", "primitive");
      std::printf("OK (B) add_native_block(hllc/roe, cons/prim) == add_compiled_model (dmax==0)\n");
    }
  }
#endif  // ADC_HAS_KOKKOS

  if (fails == 0)
    std::printf(
        "OK test_amr_riemann_native (hllc/roe x conservative/primitive : "
        "add_compiled_model == add_block, bit-identique ; hllc/roe actifs vs rusanov)\n");
  return fails ? 1 : 0;
}
