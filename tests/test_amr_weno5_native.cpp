// WENO5 sur le chemin "production" NATIF cote AMR (Plan Ideal etape 5 / DSL Phase D). Pendant raffine
// de tests/test_weno5_compiled_model.cpp (System) : on prouve que le limiteur WENO5-Z (stencil 5
// points, 3 ghosts) tourne EXACTEMENT la meme hierarchie AMR (AmrCouplerMP<Model> + reflux conservatif
// + regrid) sur les trois chemins, BIT-IDENTIQUE :
//
//   (A) add_compiled_model(AmrSystem&) == add_block(ModelSpec)   -- direct, sans .so : tourne sous TOUS
//       les backends (hote, Kokkos Serial), c'est la parite decisive qui ne casse pas nvcc.
//   (B) add_native_block(loader.so) == add_compiled_model(AmrSystem&) -- chemin .so (la source qu'emet
//       dsl.emit_cpp_native_loader(target="amr_system"), ecrite ici pour un test C++ autonome). Le
//       loader est recompile a l'execution par un g++ nu, donc cette partie SE SAUTE sous backend
//       Kokkos (ABI incompatible avec ce module) ; la parite CPU complete reste couverte par (A).
//
// Le verrou : les niveaux du coupleur sont alloues a Limiter::n_ghost (= Weno5::n_ghost = 3, cf.
// build_amr_compiled) et le regrid HERITE n_grow() (amr_regrid_finest : ngf = L[fk].U.n_grow()), donc
// le stencil 5 points de WENO5 ne lit pas hors bornes -- MEME mecanisme ghost que System (PR #22/#88),
// pas de logique dupliquee. On exige aussi le NO-DEFAULT-CHANGE : none/minmod (<= 2 ghosts) restent
// BIT-IDENTIQUES (le branchement weno5 n'a change ni l'allocation ni le resultat des autres limiteurs).
//
// Le modele est un transport pur (CompositeModel<Euler, NoSource, BackgroundDensity{alpha=0}>) : la
// brique elliptique vaut 0, donc le solve MG donne phi=0 partout (zero bruit FP), parite STRICTE.
//
// CMake injecte ADC_TEST_CXX (compilateur), ADC_TEST_INCLUDE (en-tetes adc), ADC_TEST_CXX_STD (norme
// C++ du build : la cle d'ABI du loader concorde avec celle du test) et ADC_TEST_TMPDIR.
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
#include <Kokkos_Core.hpp>  // ScopeGuard : la partie (A) tourne le coupleur AMR (for_each/kokkos_malloc)
#endif

using namespace adc;

namespace {

using ProdModel = CompositeModel<Euler, NoSource, BackgroundDensity>;
constexpr double kGamma = 1.4;

ProdModel make_model() {
  // alpha=0 : elliptic_rhs nul -> phi=0, parite stricte. n0 sans effet (alpha=0).
  return ProdModel{Euler{static_cast<Real>(kGamma)}, NoSource{},
                   BackgroundDensity{Real(0), Real(0)}};
}

// ModelSpec equivalente (dispatch_model -> CompositeModel<CompressibleFlux=Euler, NoSource,
// BackgroundDensity{alpha=0}>) pour le chemin natif add_block : MEME type concret -> parite exacte.
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

std::vector<double> bubble(int n) {  // bulle de densite lisse, periodique (WENO5 pleinement actif)
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

// Avance @p s de N pas et renvoie (densite grossiere, masse, n_patches). dt petit : reste stable WENO5.
struct Snap {
  std::vector<double> density;
  double mass = 0;
  int n_patches = 0;
};
Snap run(AmrSystem& s, int nsteps) {
  s.set_poisson("charge_density", "geometric_mg");
  s.set_refinement(1.2);  // raffine la bulle
  const double dt = 2e-4;
  for (int k = 0; k < nsteps; ++k) s.step(dt);
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
  for (double v : a) m = std::fmax(m, std::fabs(v));
  return m;
}

// Source du loader AMR : MEME forme que dsl.emit_cpp_native_loader(target="amr_system"), modele en dur.
std::string loader_source() {
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
}

bool compile_loader(const std::string& src_path, const std::string& so_path) {
#if defined(__APPLE__)
  const std::string cc = "/usr/bin/c++";  // resout l'SDK (cf. test_amr_native_loader)
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
  Kokkos::ScopeGuard guard(argc, argv);  // initialise/finalise le backend (partie A : coupleur AMR)
#else
  (void)argc;
  (void)argv;
#endif
  const int n = 64;
  const int nsteps = 12;
  const std::vector<double> rho = bubble(n);

  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  // ============================================================================================
  // (A) PARITE DECISIVE (sans .so) : add_compiled_model(AmrSystem&) == add_block(ModelSpec), pour
  //     weno5 ET minmod. Tourne sous TOUS les backends (hote, Kokkos Serial) -> ne casse pas nvcc.
  // ============================================================================================
  auto parity_direct = [&](const char* lim) {
    AmrSystem A(make_cfg(n));  // bloc COMPILE : modele connu a la compilation
    add_compiled_model(A, "gas", make_model(), lim, "rusanov", "conservative", "explicit", kGamma);
    A.set_density("gas", rho);
    const Snap sa = run(A, nsteps);

    AmrSystem B(make_cfg(n));  // bloc NATIF : meme modele via dispatch d'une ModelSpec
    B.add_block("gas", make_spec(), lim, "rusanov", "conservative", "explicit", 1);
    B.set_density("gas", rho);
    const Snap sb = run(B, nsteps);

    const double nrm = maxabs(sb.density), dmax = maxdiff(sa.density, sb.density);
    char w[160];
    std::snprintf(w, sizeof w, "[%s] densite directe non triviale", lim);
    chk(nrm > 1e-6, w);
    std::snprintf(w, sizeof w, "[%s] add_compiled_model == add_block sur AMR (dmax==0)", lim);
    chk(dmax == 0.0, w);
    std::snprintf(w, sizeof w, "[%s] masse add_compiled_model == add_block", lim);
    chk(std::fabs(sa.mass - sb.mass) < 1e-12 * (std::fabs(sb.mass) + 1.0), w);
    std::snprintf(w, sizeof w, "[%s] n_patches add_compiled_model == add_block (regrid identique)",
                  lim);
    chk(sa.n_patches == sb.n_patches, w);
    return sa.density;  // pour le test no-default-change (weno5 != minmod)
  };

  const std::vector<double> dw = parity_direct("weno5");
  const std::vector<double> dm = parity_direct("minmod");
  parity_direct("none");

  // NO-DEFAULT-CHANGE croise : WENO5 (ordre 5) DOIT differer de minmod (ordre 2) sur un meme etat
  // lisse evolue (sinon le branchement weno5 retomberait sur minmod -> dispatch inerte). Le residu
  // n'est pas trivial, donc une difference > 0 prouve que Weno5 est bien instancie et actif.
  chk(maxdiff(dw, dm) > 1e-9, "weno5 != minmod (la reconstruction WENO5 est bien active sur AMR)");

  // ============================================================================================
  // (B) CHEMIN .so : add_native_block(loader) == add_compiled_model(AmrSystem&), weno5 ET minmod.
  //     Le loader est recompile par un g++ nu -> incompatible avec un module Kokkos : on SAUTE (A
  //     a deja couvert la parite CPU). exit 0.
  // ============================================================================================
#if defined(ADC_HAS_KOKKOS)
  std::printf("skip (B) loader .so (backend Kokkos : loader CPU nu incompatible)\n");
#else
  const char* cxx = ADC_TEST_CXX;
  if (!cxx || cxx[0] == '\0') {
    std::printf("skip (B) loader .so (aucun compilateur C++ connu du build)\n");
  } else {
    const std::string tmp = std::string(ADC_TEST_TMPDIR) + "/amr_weno5_native_" +
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
      auto parity_loader = [&](const char* lim) {
        AmrSystem A(make_cfg(n));  // chemin "production" : loader .so -> add_native_block
        A.add_native_block("gas", so, lim, "rusanov", "conservative", "explicit", kGamma, 1);
        A.set_density("gas", rho);
        const Snap sa = run(A, nsteps);

        AmrSystem B(make_cfg(n));  // MEME modele installe EN DIRECT (le chemin que le loader inline)
        add_compiled_model(B, "gas", make_model(), lim, "rusanov", "conservative", "explicit",
                           kGamma);
        B.set_density("gas", rho);
        const Snap sb = run(B, nsteps);

        const double dmax = maxdiff(sa.density, sb.density);
        char w[160];
        std::snprintf(w, sizeof w, "[%s] add_native_block == add_compiled_model (dmax==0)", lim);
        chk(dmax == 0.0, w);
        std::snprintf(w, sizeof w, "[%s] n_patches loader == direct", lim);
        chk(sa.n_patches == sb.n_patches, w);
      };
      parity_loader("weno5");
      parity_loader("minmod");
      std::printf("OK (B) add_native_block(weno5/minmod) == add_compiled_model(AmrSystem&)\n");
    }
  }
#endif  // ADC_HAS_KOKKOS

  if (fails == 0)
    std::printf("OK test_amr_weno5_native (weno5 AMR : add_native_block == add_compiled_model == "
                "add_block, bit-identique ; no-default-change none/minmod)\n");
  return fails ? 1 : 0;
}
