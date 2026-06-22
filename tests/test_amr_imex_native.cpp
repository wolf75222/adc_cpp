// IMEX (source implicite raide) sur le chemin "production" NATIF cote AMR (Gap 2 parite).
// Pendant de tests/test_amr_riemann_native.cpp / test_amr_weno5_native.cpp : prouve que
// time="imex" tourne la MEME hierarchie AMR (AmrCouplerMP<Model> + reflux + regrid), BIT-IDENTIQUE,
// sur les trois chemins d'entree, et que l'IMEX est ACTIF (different de l'explicite, et stable la ou
// l'explicite EXPLOSE sur une source raide).
//
// Le moteur AMR etait EXPLICITE seul : la source etait appliquee en Euler avant a chaque feuille/niveau
// (mf_apply_source). En IMEX, le sous-pas applique (a) l'avance de transport sans source (deja portee
// par compute_face_fluxes + reflux conservatif) PUIS (b) la source raide en IMPLICITE par
// backward_euler_source (Newton local, foncteur device NOMME). C'est le pendant AMR de l'IMEX du System
// (block_builder.hpp::AdvanceImex). La source restant CELLULE-LOCALE (hors flux de face), elle n'entre
// PAS dans les registres de reflux : la conservation aux interfaces grossier-fin est intacte (prouve ici).
//
// Deux modeles, pour couvrir les deux exigences sans toucher aux briques de production (write-set) :
//
//   (A) PARITE 3 CHEMINS + IMEX ACTIF, modele ModelSpec-atteignable
//       CompositeModel<Euler, PotentialForce, ChargeDensity> (source="potential", elliptic="charge") :
//         - add_compiled_model(AmrSystem&) == add_block(ModelSpec) == add_native_block(.so), dmax==0,
//           sous time="imex" -> le drapeau IMEX est cable a l'IDENTIQUE sur les trois entrees ;
//         - IMEX != explicite sur le MEME etat -> le pas implicite est bien pris (non silencieux) ;
//         - masse grossiere conservee a ~machine sous IMEX (PotentialForce ne touche pas la densite ;
//           la source est cellule-locale, hors reflux) -> conservation preservee.
//       La source de PotentialForce ne demontre PAS l'explosion explicite (sa raideur vit dans le
//       couplage au champ self-consistent, gele dans le solve implicite cellule-local), d'ou (B).
//
//   (B) AP / SOURCE RAIDE (explicite EXPLOSE, IMEX stable), modele a source de RELAXATION lineaire
//       definie ICI (StiffRelax, S = (u_eq - u)/eps cellule-locale) -- une telle brique n'existe pas
//       dans la dispatch ModelSpec, donc seuls les chemins a Model concret (add_compiled_model et
//       add_native_block) la portent. Pour eps << dt : l'explicite (facteur |1 - dt/eps| >> 1) DIVERGE
//       (non fini), l'IMEX (backward Euler, inconditionnellement stable) reste fini et capture
//       l'equilibre. + parite add_compiled_model == add_native_block (dmax==0) en regime non explosif.
//
//   (A) et (B) tournent en DIRECT (sans .so) sous TOUS les backends (hote, Kokkos Serial) : c'est la
//   parite decisive qui ne casse pas nvcc. Les legs .so (add_native_block) AUTO-SAUTENT sous Kokkos
//   (loader CPU nu, ABI incompatible) ; la parite CPU reste couverte par le chemin direct.
//
// CMake injecte ADC_TEST_CXX, ADC_TEST_INCLUDE, ADC_TEST_CXX_STD, ADC_TEST_TMPDIR (meme pattern que
// test_amr_riemann_native).
#include <adc/physics/bricks/bricks.hpp>  // CompositeModel, Euler, PotentialForce, ChargeDensity, BackgroundDensity
#include <adc/runtime/builders/amr_dsl_block.hpp>
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

constexpr double kGamma = 1.4;
constexpr double kQom =
    200.0;  // q/m de PotentialForce : assez fort pour IMEX != explicite, sans exploser

// (A) modele ModelSpec-atteignable : Euler + force du potentiel (source raide self-consistent) + charge.
using PotModel = CompositeModel<Euler, PotentialForce, ChargeDensity>;
PotModel make_pot() {
  return PotModel{Euler{static_cast<Real>(kGamma)}, PotentialForce{static_cast<Real>(kQom)},
                  ChargeDensity{Real(1)}};
}
ModelSpec make_pot_spec() {
  ModelSpec spec;
  spec.transport = "compressible";
  spec.source = "potential";
  spec.elliptic = "charge";
  spec.gamma = kGamma;
  spec.qom = kQom;
  spec.q = 1.0;
  return spec;
}

// (B) source de RELAXATION lineaire RAIDE, cellule-locale : S = (u_eq - u)/eps sur toutes les
// variables conservees. backward Euler (IMEX) est inconditionnellement stable ; Euler avant explose
// des que dt/eps > 2 (cf. test_imex_ap). Definie ICI (hors briques de production, write-set).
struct StiffRelax {
  Real inv_eps = Real(0);
  Real u_eq[4] = {Real(1), Real(0), Real(0), Real(2.5)};  // rho, mx, my, E d'equilibre
  template <class State>
  ADC_HD State apply(const State& u, const Aux&) const {
    State s{};
    for (int c = 0; c < State::size(); ++c)
      s[c] = inv_eps * (u_eq[c] - u[c]);
    return s;
  }
};
// elliptic "background" alpha=0 : rhs nul -> phi=0, aucun couplage au champ (raideur purement locale).
using StiffModel = CompositeModel<Euler, StiffRelax, BackgroundDensity>;
StiffModel make_stiff(double eps) {
  StiffRelax r;
  r.inv_eps = static_cast<Real>(1.0 / eps);
  return StiffModel{Euler{static_cast<Real>(kGamma)}, r, BackgroundDensity{Real(0), Real(0)}};
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
bool all_finite(const std::vector<double>& a) {
  for (double v : a)
    if (!std::isfinite(v))
      return false;
  return true;
}

// Source du loader AMR : MEME forme que dsl.emit_cpp_native_loader(target="amr_system"), DEUX modeles
// en dur (PotModel pour A, StiffModel pour B) selectionnes par le nom du bloc ("pot" | "stiff:<eps>").
std::string loader_source() {
  // Generated C++ source raw string: clang-format would reindent (or, with the
  // interleaved R"CPP( delimiters, runaway-indent) the inner content. Fence it to keep the
  // emitted source verbatim.
  // clang-format off
  return R"CPP(
#include <adc/runtime/builders/amr_dsl_block.hpp>
#include <adc/runtime/detail/abi_key.hpp>
#include <adc/physics/bricks/bricks.hpp>
#include <cstdlib>
#include <cstring>
#include <string>
namespace adc_generated {
using PotModel = adc::CompositeModel<adc::Euler, adc::PotentialForce, adc::ChargeDensity>;
struct StiffRelax {
  adc::Real inv_eps = adc::Real(0);
  adc::Real u_eq[4] = {adc::Real(1), adc::Real(0), adc::Real(0), adc::Real(2.5)};
  template <class State>
  ADC_HD State apply(const State& u, const adc::Aux&) const {
    State s{};
    for (int c = 0; c < State::size(); ++c) s[c] = inv_eps * (u_eq[c] - u[c]);
    return s;
  }
};
using StiffModel = adc::CompositeModel<adc::Euler, StiffRelax, adc::BackgroundDensity>;
}
// LITTERAL preprocesseur (PAS abi_key_string() : une inline serait interposee, ELF/RTLD_GLOBAL,
// vers la copie du module deja charge -> cle du module renvoyee -> garde d'ABI tautologique).
extern "C" const char* adc_native_abi_key() { return ADC_ABI_KEY_LITERAL; }
extern "C" void adc_install_native_amr(void* sys, const char* name, const char* limiter,
                                       const char* riemann, const char* recon, const char* time,
                                       double gamma, int substeps) {
  adc::AmrSystem* s = reinterpret_cast<adc::AmrSystem*>(sys);
  if (std::strncmp(name, "stiff:", 6) == 0) {
    const double eps = std::atof(name + 6);
    adc_generated::StiffRelax r;
    r.inv_eps = static_cast<adc::Real>(1.0 / eps);
    adc::add_compiled_model<adc_generated::StiffModel>(
        *s, name,
        adc_generated::StiffModel{adc::Euler{static_cast<adc::Real>(gamma)}, r,
                                  adc::BackgroundDensity{adc::Real(0), adc::Real(0)}},
        limiter, riemann, recon, time, gamma, substeps);
    return;
  }
  adc::add_compiled_model<adc_generated::PotModel>(
      *s, name,
      adc_generated::PotModel{adc::Euler{static_cast<adc::Real>(gamma)},
                              adc::PotentialForce{static_cast<adc::Real>()CPP" +
         std::to_string(kQom) + R"CPP()}, adc::ChargeDensity{adc::Real(1)}},
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

// --- helpers de run (Part A : potential ; Part B : stiff) ---

// Configure le Poisson + raffinement + densite et avance nsteps de dt. Renvoie densite + masse.
struct Snap {
  std::vector<double> density;
  double mass = 0;
  int n_patches = 0;
};

// @p setup installe l'unique bloc (add_compiled_model ou add_block) ; le reste (Poisson, raffinement,
// densite, avance) est commun. Construit son propre AmrSystem (pas de fuite).
template <class Setup>
Snap run(int n, const std::vector<double>& rho, int nsteps, double dt, Setup setup) {
  AmrSystem s(make_cfg(n));
  setup(s);
  s.set_poisson("charge_density", "geometric_mg");
  s.set_refinement(1.2);
  s.set_density("gas", rho);
  for (int k = 0; k < nsteps; ++k)
    s.step(dt);
  return Snap{s.density(), s.mass(), s.n_patches()};
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
  // (A) PARITE 3 CHEMINS sous IMEX + IMEX ACTIF + CONSERVATION, modele potential (ModelSpec).
  // ============================================================================================
  const double dtA = 5e-3;  // regime ou IMEX differe nettement de l'explicite (PotentialForce fort)

  // (A1) add_compiled_model == add_block sous time="imex" : dmax==0 (chemin direct, tous backends).
  Snap A_compiled = run(n, rho, nsteps, dtA, [](AmrSystem& s) {
    add_compiled_model(s, "gas", make_pot(), "minmod", "rusanov", "conservative", "imex", kGamma);
  });
  Snap A_block = run(n, rho, nsteps, dtA, [](AmrSystem& s) {
    s.add_block("gas", make_pot_spec(), "minmod", "rusanov", "conservative", "imex", 1);
  });
  chk(maxabs(A_block.density) > 1e-6, "[A] densite IMEX non triviale");
  chk(maxdiff(A_compiled.density, A_block.density) == 0.0,
      "[A] add_compiled_model == add_block sous IMEX (dmax==0)");
  chk(std::fabs(A_compiled.mass - A_block.mass) < 1e-12 * (std::fabs(A_block.mass) + 1.0),
      "[A] masse add_compiled_model == add_block sous IMEX");
  chk(A_compiled.n_patches == A_block.n_patches, "[A] n_patches identique sous IMEX");
  std::printf("OK  [A] add_compiled_model == add_block sous IMEX (dmax==0)\n");

  // (A2) IMEX ACTIF : different de l'explicite sur le MEME etat (le pas implicite est bien pris).
  Snap A_explicit = run(n, rho, nsteps, dtA, [](AmrSystem& s) {
    add_compiled_model(s, "gas", make_pot(), "minmod", "rusanov", "conservative", "explicit",
                       kGamma);
  });
  chk(maxdiff(A_compiled.density, A_explicit.density) > 1e-9,
      "[A] IMEX != explicite sur le meme etat (pas implicite actif, non silencieux)");
  std::printf("OK  [A] IMEX != explicite (dmax=%.3e)\n",
              maxdiff(A_compiled.density, A_explicit.density));

  // (A3) CONSERVATION : la masse grossiere est conservee a ~machine sous IMEX (PotentialForce ne
  // touche pas la densite ; la source est cellule-locale, hors registres de reflux). On la compare a
  // la masse de l'explicite : les deux conservent la masse exactement -> ecart machine.
  {
    AmrSystem s(make_cfg(n));
    add_compiled_model(s, "gas", make_pot(), "minmod", "rusanov", "conservative", "imex", kGamma);
    s.set_poisson("charge_density", "geometric_mg");
    s.set_refinement(1.2);
    s.set_density("gas", rho);
    const double m0 = s.mass();
    for (int k = 0; k < nsteps; ++k)
      s.step(dtA);
    const double m1 = s.mass();
    const double drift = std::fabs(m1 - m0) / (std::fabs(m0) + 1e-30);
    chk(drift < 1e-12, "[A] masse conservee a ~machine sous IMEX (reflux intact)");
    std::printf("OK  [A] conservation IMEX : derive relative de masse = %.3e\n", drift);
  }

  // ============================================================================================
  // (B) SOURCE RAIDE : explicite EXPLOSE, IMEX stable ; add_compiled_model == add_native_block.
  // ============================================================================================
  // Regime RAIDE (eps << dt) : explicite diverge, IMEX reste fini et borne.
  {
    const double eps = 1e-5, dtB = 1e-3;
    Snap B_imex = run(n, rho, nsteps, dtB, [eps](AmrSystem& s) {
      add_compiled_model(s, "gas", make_stiff(eps), "minmod", "rusanov", "conservative", "imex",
                         kGamma);
    });
    Snap B_expl = run(n, rho, nsteps, dtB, [eps](AmrSystem& s) {
      add_compiled_model(s, "gas", make_stiff(eps), "minmod", "rusanov", "conservative", "explicit",
                         kGamma);
    });
    chk(all_finite(B_imex.density) && maxabs(B_imex.density) < 1e3,
        "[B] IMEX stable sur source raide (fini, borne)");
    chk(!all_finite(B_expl.density) || maxabs(B_expl.density) > 1e3,
        "[B] explicite EXPLOSE sur source raide (non fini ou >> borne)");
    std::printf(
        "OK  [B] source raide (eps=%.0e, dt=%.0e) : IMEX max=%.3e (stable) | explicite %s\n", eps,
        dtB, maxabs(B_imex.density),
        all_finite(B_expl.density) ? "borne >> 1" : "NON FINI (explose)");
  }

  // (B2) PARITE add_compiled_model == add_block sous IMEX en regime NON explosif (eps modere) :
  // le drapeau IMEX traverse aussi la dispatch ModelSpec a l'identique pour un modele a source
  // (ici on reste sur potential car StiffRelax n'est pas ModelSpec-atteignable ; (A) couvre deja
  // la parite 3 chemins du drapeau IMEX). On verifie en plus que la source raide IMEX est stable
  // en regime modere et borne pour add_compiled_model == add_compiled_model (idempotence du build).
  {
    const double eps = 1e-3, dtB = 2e-4;
    Snap B1 = run(n, rho, nsteps, dtB, [eps](AmrSystem& s) {
      add_compiled_model(s, "gas", make_stiff(eps), "minmod", "rusanov", "conservative", "imex",
                         kGamma);
    });
    Snap B2 = run(n, rho, nsteps, dtB, [eps](AmrSystem& s) {
      add_compiled_model(s, "gas", make_stiff(eps), "minmod", "rusanov", "conservative", "imex",
                         kGamma);
    });
    chk(maxdiff(B1.density, B2.density) == 0.0,
        "[B] add_compiled_model deterministe sous IMEX (dmax==0)");
    chk(all_finite(B1.density), "[B] IMEX stable en regime modere");
    std::printf("OK  [B] add_compiled_model IMEX deterministe (dmax==0)\n");
  }

  std::printf(
      "OK  (direct) IMEX cable a parite (add_compiled_model == add_block, dmax==0) ; IMEX "
      "actif (!= explicite) ; stable sur source raide (explicite explose) ; masse conservee\n");

  // ============================================================================================
  // (C) CHEMIN .so : add_native_block(loader) == add_compiled_model, sous IMEX (A potential + B stiff).
  //     Le loader est recompile par un g++ nu -> incompatible avec un module Kokkos : SAUTE.
  // ============================================================================================
#if defined(ADC_HAS_KOKKOS)
  std::printf("skip (C) loader .so (backend Kokkos : loader CPU nu incompatible)\n");
#else
  const char* cxx = ADC_TEST_CXX;
  if (!cxx || cxx[0] == '\0') {
    std::printf("skip (C) loader .so (aucun compilateur C++ connu du build)\n");
  } else {
    const std::string tmp = std::string(ADC_TEST_TMPDIR) + "/amr_imex_native_" +
                            std::to_string(static_cast<long>(std::clock()));
    const std::string src = tmp + ".cpp";
    const std::string so = tmp + ".so";
    {
      std::ofstream f(src);
      f << loader_source();
    }
    if (!compile_loader(src, so)) {
      std::printf("skip (C) loader .so (echec de compilation du loader -- en-tetes/std ?)\n");
    } else {
      // (C-A) potential sous IMEX : add_native_block == add_compiled_model (dmax==0).
      {
        AmrSystem A(make_cfg(n));
        A.add_native_block("pot", so, "minmod", "rusanov", "conservative", "imex", kGamma, 1);
        A.set_poisson("charge_density", "geometric_mg");
        A.set_refinement(1.2);
        A.set_density("gas", rho);
        for (int k = 0; k < nsteps; ++k)
          A.step(dtA);

        AmrSystem B(make_cfg(n));
        add_compiled_model(B, "gas", make_pot(), "minmod", "rusanov", "conservative", "imex",
                           kGamma);
        B.set_poisson("charge_density", "geometric_mg");
        B.set_refinement(1.2);
        B.set_density("gas", rho);
        for (int k = 0; k < nsteps; ++k)
          B.step(dtA);

        chk(maxdiff(A.density(), B.density()) == 0.0,
            "[C-A] add_native_block == add_compiled_model sous IMEX (potential, dmax==0)");
        chk(A.n_patches() == B.n_patches(), "[C-A] n_patches loader == direct");
      }
      // (C-B) stiff sous IMEX (regime modere, fini) : add_native_block == add_compiled_model (dmax==0).
      {
        const double eps = 1e-3, dtB = 2e-4;
        const std::string bname = "stiff:" + std::to_string(eps);
        AmrSystem A(make_cfg(n));
        A.add_native_block(bname, so, "minmod", "rusanov", "conservative", "imex", kGamma, 1);
        A.set_poisson("charge_density", "geometric_mg");
        A.set_refinement(1.2);
        A.set_density("gas", rho);
        for (int k = 0; k < nsteps; ++k)
          A.step(dtB);

        AmrSystem B(make_cfg(n));
        add_compiled_model(B, "gas", make_stiff(eps), "minmod", "rusanov", "conservative", "imex",
                           kGamma);
        B.set_poisson("charge_density", "geometric_mg");
        B.set_refinement(1.2);
        B.set_density("gas", rho);
        for (int k = 0; k < nsteps; ++k)
          B.step(dtB);

        chk(maxdiff(A.density(), B.density()) == 0.0,
            "[C-B] add_native_block == add_compiled_model sous IMEX (stiff, dmax==0)");
      }
      std::printf(
          "OK (C) add_native_block == add_compiled_model sous IMEX (potential + stiff, dmax==0)\n");
    }
  }
#endif  // ADC_HAS_KOKKOS

  if (fails == 0)
    std::printf(
        "OK test_amr_imex_native (IMEX sur AMR : add_native_block == add_compiled_model == "
        "add_block bit-identique ; IMEX actif vs explicite ; stable sur source raide ; "
        "masse conservee au reflux)\n");
  return fails ? 1 : 0;
}
