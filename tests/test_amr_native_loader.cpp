// Chemin "production" NATIF cote AMR (Plan Ideal etape 5 / DSL Phase D) : AmrSystem::add_native_block.
//
// On compile A L'EXECUTION un LOADER .so qui inline le gabarit en-tete add_compiled_model(AmrSystem&,
// Model{...}) derriere l'ABI extern "C" MINCE attendue par AmrSystem::add_native_block (symbole
// adc_install_native_amr + cle adc_native_abi_key). C'est exactement la source qu'emet le DSL
// (dsl.emit_cpp_native_loader(target="amr_system")), mais ecrite ici pour un test C++ AUTONOME
// (lance en ctest, sans le module Python). On le branche via AmrSystem::add_native_block puis on
// compare a un AmrSystem ou le MEME modele est installe par add_compiled_model(AmrSystem&) EN DIRECT :
// memes types, meme schema -> densite grossiere / masse / n_patches BIT-IDENTIQUES (dmax == 0).
//
// On verifie aussi le GARDE-FOU ABI : un loader dont la cle adc_native_abi_key est falsifiee (recompile
// avec une signature d'en-tetes bidon) est REJETE explicitement par add_native_block (pas d'UB).
//
// Le modele est un transport pur (CompositeModel<Euler, NoSource, BackgroundDensity{alpha=0}>) : la
// brique elliptique vaut 0, donc le solve MG donne phi=0 des deux cotes (zero bruit FP), parite stricte.
//
// CMake injecte ADC_TEST_CXX (compilateur), ADC_TEST_INCLUDE (dossier des en-tetes adc) et
// ADC_TEST_CXX_STD (norme C++ du build, pour que la cle d'ABI du loader concorde avec celle du test).
#include <adc/physics/bricks/bricks.hpp>  // CompositeModel, Euler, NoSource, BackgroundDensity
#include <adc/runtime/builders/compiled/amr_dsl_block.hpp>
#include <adc/runtime/amr_system.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>  // std::clock : suffixe unique pour les fichiers temporaires du test
#include <fstream>
#include <string>
#include <vector>

using namespace adc;

namespace {

using ProdModel = CompositeModel<Euler, NoSource, BackgroundDensity>;

// Parametres du modele FIGES dans le loader (alpha=0 : elliptic_rhs nul -> phi=0, parite stricte).
constexpr double kGamma = 1.4;

std::vector<double> bubble(int n) {  // bulle de densite lisse, periodique
  std::vector<double> rho(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5;
      rho[static_cast<std::size_t>(j) * n + i] = 1.0 + 0.5 * std::exp(-(x * x + y * y) / 0.02);
    }
  return rho;
}

// Source du loader AMR : MEME forme que dsl.emit_cpp_native_loader(target="amr_system"). Le modele est
// ecrit en dur (CompositeModel<Euler, NoSource, BackgroundDensity>) pour un test autonome. @p fake_sig
// remplace -DADC_HEADER_SIG par une valeur bidon (cle d'ABI fausse) quand vrai.
std::string loader_source() {
  // Generated C++ source raw string: clang-format would reindent (or, with the
  // interleaved R"CPP( delimiters, runaway-indent) the inner content. Fence it to keep the
  // emitted source verbatim.
  // clang-format off
  return R"CPP(
#include <adc/runtime/builders/compiled/amr_dsl_block.hpp>
#include <adc/runtime/dynamic/abi_key.hpp>
#include <adc/physics/bricks/bricks.hpp>
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

// Compile le loader en .so (g++/c++ a l'execution). @p extra : flags supplementaires (-DADC_HEADER_SIG
// bidon pour le test de rejet d'ABI). Renvoie true si la compilation reussit.
bool compile_loader(const std::string& src_path, const std::string& so_path,
                    const std::string& extra) {
  // Compilateur : ADC_TEST_CXX (= CMAKE_CXX_COMPILER) en general. Sur macOS, ce chemin pointe le c++
  // de la toolchain Xcode SANS sysroot SDK -> '<algorithm> file not found' ; on prefere alors le
  // wrapper /usr/bin/c++ (xcrun) qui resout l'SDK. Meme famille de compilateur (clang) donc __VERSION__
  // identique -> cle d'ABI concordante avec ce binaire.
#if defined(__APPLE__)
  const std::string cc = "/usr/bin/c++";
#else
  const std::string cc = ADC_TEST_CXX;
#endif
  // ADC_TEST_CXX_STD : meme norme que le build du test -> __cplusplus identique -> cle d'ABI concordante.
  std::string cmd = cc + " -shared -fPIC -std=" + ADC_TEST_CXX_STD + " -O2 -I " + ADC_TEST_INCLUDE +
                    " " + extra + " " + src_path + " -o " + so_path;
#if defined(__APPLE__)
  cmd +=
      " -undefined dynamic_lookup";  // macOS : indefinis resolus a l'execution (set_compiled_block)
#endif
  cmd += " 2> /dev/null";
  return std::system(cmd.c_str()) == 0;
}

AmrSystemConfig make_cfg(int n) {
  AmrSystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;
  cfg.regrid_every = 4;
  return cfg;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(ADC_HAS_KOKKOS)
  // Backend Kokkos : le loader serait recompile a l'execution par un g++ NU (sans les flags/headers
  // Kokkos ni -DADC_HAS_KOKKOS) -> ABI incompatible avec ce module Kokkos (kernels for_each device),
  // donc on SAUTE. La parite CPU complete est couverte par le job Release (backend hote, sans Kokkos)
  // et par le test Python test_dsl_production_amr (qui compile le loader via le DSL). exit 0.
  (void)argc;
  (void)argv;
  std::printf("skip test_amr_native_loader (backend Kokkos : loader CPU nu incompatible)\n");
  return 0;
#else
  (void)argc;
  (void)argv;

  const char* cxx = ADC_TEST_CXX;
  if (!cxx || cxx[0] == '\0') {  // pas de compilateur connu du build : on saute (exit 0)
    std::printf("skip test_amr_native_loader (aucun compilateur C++ connu du build)\n");
    return 0;
  }

  const int n = 64;
  const std::vector<double> rho = bubble(n);

  // Ecrit la source du loader puis compile le .so VALIDE (cle d'ABI concordante).
  const std::string tmp = std::string(ADC_TEST_TMPDIR) + "/amr_native_loader_" +
                          std::to_string(static_cast<long>(std::clock()));
  const std::string src = tmp + ".cpp";
  const std::string so = tmp + ".so";
  {
    std::ofstream f(src);
    f << loader_source();
  }
  if (!compile_loader(src, so, "")) {
    std::printf("skip test_amr_native_loader (echec de compilation du loader -- en-tetes/std ?)\n");
    return 0;
  }

  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  // (A) bloc "production" : loader .so -> AmrSystem::add_native_block (chemin natif AMR).
  AmrSystem A(make_cfg(n));
  A.add_native_block("gas", so, "minmod", "rusanov", "conservative", "explicit", kGamma, 1);
  A.set_poisson("charge_density", "geometric_mg");
  A.set_refinement(1.2);
  A.set_density("gas", rho);

  // (B) MEME modele installe EN DIRECT par add_compiled_model(AmrSystem&) (le chemin que le loader
  // inline). Memes types + meme schema -> parite STRICTE attendue (bit-identique).
  AmrSystem B(make_cfg(n));
  add_compiled_model(
      B, "gas",
      ProdModel{Euler{static_cast<Real>(kGamma)}, NoSource{}, BackgroundDensity{Real(0), Real(0)}},
      "minmod", "rusanov", "conservative", "explicit", kGamma, 1);
  B.set_poisson("charge_density", "geometric_mg");
  B.set_refinement(1.2);
  B.set_density("gas", rho);

  chk(A.n_patches() == B.n_patches(), "n_patches initial loader == add_compiled_model");
  const double m0 = A.mass();
  chk(std::fabs(m0 - B.mass()) < 1e-12 * (std::fabs(m0) + 1.0), "masse initiale loader == direct");

  const double dt = 2e-4;
  for (int s = 0; s < 12; ++s) {
    A.step(dt);
    B.step(dt);
  }
  const std::vector<double> da = A.density(), db = B.density();
  chk(da.size() == db.size() && !da.empty(), "densite non vide, memes tailles");
  double dmax = 0, nrm = 0;
  for (std::size_t k = 0; k < da.size() && k < db.size(); ++k) {
    dmax = std::fmax(dmax, std::fabs(da[k] - db[k]));
    nrm = std::fmax(nrm, std::fabs(db[k]));
  }
  chk(nrm > 1e-6, "densite directe non triviale");
  chk(dmax == 0.0, "densite grossiere loader == add_compiled_model (bit-identique, dmax==0)");
  chk(A.n_patches() == B.n_patches(), "n_patches final loader == direct (regrid identique)");

  // (C) GARDE-FOU ABI : un loader a cle adc_native_abi_key falsifiee est REJETE par add_native_block.
  const std::string so_bad = tmp + "_bad.so";
  const bool built_bad =
      compile_loader(src, so_bad, "-DADC_HEADER_SIG=\\\"deadbeef_fausse_signature\\\"");
  if (built_bad) {
    AmrSystem C(make_cfg(n));
    bool raised = false;
    try {
      C.add_native_block("gas", so_bad, "minmod", "rusanov", "conservative", "explicit", kGamma, 1);
    } catch (const std::runtime_error& e) {
      // ADC-283 : le message anglais (ADC-272) est "incompatible ABI", pas "ABI incompatible".
      raised = (std::string(e.what()).find("incompatible ABI") != std::string::npos);
    }
    chk(raised, "cle d'ABI falsifiee REJETEE par add_native_block");
  } else {
    std::printf("note : loader a cle fausse non compile (garde-fou ABI non exerce)\n");
  }

  if (fails == 0)
    std::printf(
        "OK test_amr_native_loader (add_native_block == add_compiled_model(AmrSystem&) ; "
        "dmax=%.1e ; ABI falsifiee rejetee)\n",
        dmax);
  return fails ? 1 : 0;
#endif  // ADC_HAS_KOKKOS
}
