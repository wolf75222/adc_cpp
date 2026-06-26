// AMR MULTI-BLOCS COMPILES (capstone "v", DSL production multi-bloc) : la FACADE RUNTIME AmrSystem
// accepte desormais PLUSIEURS blocs COMPILES (add_compiled_model, modeles connus a la compilation)
// co-localises sur UNE hierarchie AMR PARTAGEE. Avant ce capstone, le 2e bloc compile LEVAIT
// ("set_compiled_block : un seul bloc compile est supporte") : la facade ne portait qu'un unique
// AmrCouplerMP mono-bloc pour le chemin compile. Desormais add_compiled_model fige DEUX builders
// (un mono-bloc AmrCouplerMP + un multi-blocs AmrRuntimeBlock) et la facade route a ensure_built :
// un seul bloc -> AmrCouplerMP (bit-identique) ; deux blocs ou plus -> AmrRuntime (file de specs
// construite paresseusement sur le layout PARTAGE, exactement comme add_block natif).
//
// Ce que le test verrouille :
//   (A) DEUX blocs compiles a SCHEMAS DIFFERENTS sur la hierarchie partagee : pas de crash au 2e bloc,
//       les DEUX blocs EVOLUENT (transport E x B), masse conservee PAR BLOC, potentiel de systeme
//       (Poisson somme co-localise q0 n0 + q1 n1) non trivial.
//   (B) COUPLAGE entre les deux blocs compiles : une source ionisation-like CONSERVATIVE (+S sur un
//       bloc, -S exactement sur l'autre, meme cellule) transfere de la masse entre blocs tout en
//       conservant la masse COMPOSITE globale a ~machine.
//   (C) MELANGE compile + natif : un bloc compile (add_compiled_model) et un bloc natif (add_block)
//       co-existent sur la meme hierarchie (le melange etait refuse avant ce capstone).
//   (D) MONO-BLOC COMPILE BIT-IDENTIQUE : un seul bloc compile route TOUJOURS par AmrCouplerMP (jamais
//       par AmrRuntime), donc un meme cas joue deux fois donne le MEME resultat au bit pres (dmax==0).
//   (E) NON-REGRESSION du refus : le 2e bloc compile NE LEVE PLUS (preuve directe que la file de specs
//       a remplace l'ancien throw "un seul bloc compile").
//
// CHOIX DE COMPILABILITE (limitation connue nvcc, cf. tache). Un test AMR complet avec concept + lambda
// GENERIQUE (auto m) NE COMPILE PAS sous nvcc. Ici on utilise donc des FONCTEURS / TYPES CONCRETS :
// les modeles sont des CompositeModel<ExBVelocity, NoSource, ChargeDensity> instancies a la main (pas
// de dispatch_model generique), et add_compiled_model capture ces types concrets. Le noyau AMR
// (advance_amr<Limiter, Flux>) reste capture par dispatch_amr_block via une fonction template NOMMEE
// (recette device-clean #64/#97), jamais une lambda etendue cross-TU. Le test compile donc partout
// (CPU + Kokkos Serial/OpenMP/Cuda), comme test_amr_coupled_source_role_strict et add_compiled_model #16/#18.

#include <pops/coupling/source/coupled_source_program.hpp>  // CsOp (opcodes du bytecode P5)
#include <pops/physics/bricks/bricks.hpp>                   // CompositeModel
#include <pops/physics/bricks/elliptic.hpp>                 // ChargeDensity
#include <pops/physics/bricks/hyperbolic.hpp>               // ExBVelocity
#include <pops/physics/bricks/source.hpp>                   // NoSource
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>            // add_compiled_model(AmrSystem&, ...)
#include <pops/runtime/amr_system.hpp>               // facade AmrSystem
#include <pops/runtime/config/model_spec.hpp>  // ModelSpec (bloc natif, melange compile + natif)

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

// Modele ExB scalaire (1 var) a charge q, CONNU A LA COMPILATION (type concret, pas de dispatch). q
// (signe inclus) distingue ions (+1) / electrons (-1) ; q=0 -> bloc neutre (pas de contribution Poisson).
using ExBModel = CompositeModel<ExBVelocity, NoSource, ChargeDensity>;
static ExBModel exb_model(double q, double B0) {
  return ExBModel{ExBVelocity{Real(B0)}, NoSource{}, ChargeDensity{Real(q)}};
}

// Spec ExB scalaire NATIVE (pour le melange compile + natif du point C) : meme physique, via ModelSpec.
static ModelSpec exb_spec(double q, double B0) {
  ModelSpec s;
  s.transport = "exb";
  s.source = "none";
  s.elliptic = "charge";
  s.q = q;
  s.B0 = B0;
  return s;
}

// Source RAIDE CELLULE-LOCALE qui NE TOUCHE PAS la densite (composante 0) NI l'energie (composante 3) :
// relaxation de la SEULE quantite de mouvement (mx, my) vers zero, de raideur 1/eps (parente de
// StiffMomentumRelax de test_amr_multiblock_imex.cpp ; ici on ne stiffenne que mx/my pour que le MASQUE
// IMEX PARTIEL {momentum_x, momentum_y} couvre EXACTEMENT les composantes raides). En EXPLICITE (Euler
// avant) mx <- mx (1 - dt/eps) DIVERGE des que dt/eps > 2 ; en IMEX (backward Euler) mx <- mx /
// (1 + dt/eps) reste BORNE pour tout dt > 0. rho (comp 0) a source NULLE -> masse conservee a la machine.
struct StiffMomentumRelax {
  Real inv_eps = Real(0);
  template <class State>
  POPS_HD State apply(const State& u, const Aux&) const {
    State s{};
    if (State::size() > 1)
      s[1] = -inv_eps * u[1];  // -mx / eps
    if (State::size() > 2)
      s[2] = -inv_eps * u[2];  // -my / eps
    return s;
  }
};

// Modele COMPILE 4 variables a SOURCE RAIDE LOCALE (pour exercer le chemin IMEX du bloc compile multi-
// bloc en DIRECT, add_compiled_model). elliptic "background" alpha=0 : rhs nul (raideur PUREMENT locale,
// aucun couplage Poisson) -> on separe la stabilite de la source de tout couplage. TYPE CONCRET (pas de
// dispatch generique) : reste compilable sous nvcc, comme le reste du fichier.
using StiffCModel = CompositeModel<Euler, StiffMomentumRelax, BackgroundDensity>;
static StiffCModel stiff_cmodel(double eps) {
  StiffMomentumRelax r;
  r.inv_eps = static_cast<Real>(1.0 / eps);
  return StiffCModel{Euler{Real(1.4)}, r, BackgroundDensity{Real(0), Real(0)}};
}

// Modele COMPILE 4 variables NEUTRE (Euler sans source) : un bloc voisin explicite, MEME nombre de
// variables que le bloc raide (layout coherent sur la hierarchie partagee).
using NeutralCModel = CompositeModel<Euler, NoSource, BackgroundDensity>;
static NeutralCModel neutral_cmodel() {
  return NeutralCModel{Euler{Real(1.4)}, NoSource{}, BackgroundDensity{Real(0), Real(0)}};
}

// densite "bulle" gaussienne (gradients non triviaux -> le transport engendre de l'impulsion que la
// source raide relaxe). Sert au cas (F) sur les blocs compiles 4 variables.
static std::vector<double> bubble(int n) {
  std::vector<double> rho(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5;
      rho[static_cast<std::size_t>(j) * n + i] = 1.0 + 0.5 * std::exp(-(x * x + y * y) / 0.02);
    }
  return rho;
}

// densite de charge a moyenne nulle (solvable en periodique) : un creneau centre +/- amplitude, n*n.
static std::vector<double> bump(int n, double base, double amp) {
  std::vector<double> r(static_cast<std::size_t>(n) * n, base);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const bool in = (i >= n / 4 && i < 3 * n / 4 && j >= n / 4 && j < 3 * n / 4);
      r[static_cast<std::size_t>(j) * n + i] = base + (in ? amp : -amp / 3.0);
    }
  return r;
}

template <class F>
static bool raises(F&& f) {
  try {
    f();
  } catch (const std::runtime_error&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

static double dmax_field(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0;
  const std::size_t nn = a.size() < b.size() ? a.size() : b.size();
  for (std::size_t i = 0; i < nn; ++i)
    d = std::max(d, std::fabs(a[i] - b[i]));
  return d;
}

static bool all_finite(const std::vector<double>& v) {
  for (double x : v)
    if (!std::isfinite(x))
      return false;
  return true;
}

static double maxabs(const std::vector<double>& v) {
  double m = 0;
  for (double x : v)
    m = std::max(m, std::fabs(x));
  return m;
}

// Enregistre une source ionisation-like CONSERVATIVE entre deux blocs nommes sur le role density :
// S = k * n_a * n_b ; gain +S sur @p block_gain, perte -S (gain + Neg) sur @p block_loss, MEME cellule.
// Bytecode postfixe construit a la main, EXACTEMENT comme pops.dsl.CoupledSource.add_pair (cf.
// test_amr_multiblock_coupled_source). Conserve n_a + n_b par cellule a ~machine.
static void register_ionization(AmrSystem& sim, const std::string& block_a,
                                const std::string& block_b, const std::string& block_gain,
                                const std::string& block_loss, double k) {
  const std::vector<std::string> in_blocks = {block_a, block_b};
  const std::vector<std::string> in_roles = {"density", "density"};
  const std::vector<double> consts = {k};
  const std::vector<std::string> out_blocks = {block_gain, block_loss};
  const std::vector<std::string> out_roles = {"density", "density"};
  const int P = static_cast<int>(CsOp::PushReg), MUL = static_cast<int>(CsOp::Mul),
            NEG = static_cast<int>(CsOp::Neg);
  // registres : r0 = n_a (entree), r1 = n_b (entree), r2 = k (constante)
  // gain  : PushReg 0, PushReg 1, Mul, PushReg 2, Mul        -> k * n_a * n_b
  // perte : <gain> puis Neg                                  -> -(k * n_a * n_b)
  std::vector<int> prog_ops = {P, P, MUL, P, MUL,        // gain
                               P, P, MUL, P, MUL, NEG};  // perte
  std::vector<int> prog_args = {0, 1, 0, 2, 0, 0, 1, 0, 2, 0, 0};
  std::vector<int> prog_lens = {5, 6};
  // ADC-214 : description bytecode regroupee dans le POD CoupledSourceProgram (appel auto-documente).
  pops::CoupledSourceProgram prog;
  prog.in_blocks = in_blocks;
  prog.in_roles = in_roles;
  prog.consts = consts;
  prog.out_blocks = out_blocks;
  prog.out_roles = out_roles;
  prog.prog_ops = prog_ops;
  prog.prog_args = prog_args;
  prog.prog_lens = prog_lens;
  sim.add_coupled_source(prog);
}

int main(int argc, char** argv) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    std::printf("  [%s] %s\n", c ? "OK " : "XX ", w);
    if (!c)
      ++fails;
  };

  const int N = 32;
  const double L = 1.0, B0 = 1.0, k = 0.5;
  const double q0 = +1.0, q1 = -1.0;  // ions (block 0), electrons (block 1)
  const std::vector<double> rho0 = bump(N, 1.0, 0.40);
  const std::vector<double> rho1 = bump(N, 1.0, 0.20);

  // ============================================================================================
  // (A) DEUX BLOCS COMPILES a schemas DIFFERENTS, sans couplage. Le 2e add_compiled_model ne leve
  //     PLUS ; les deux blocs evoluent ; masse conservee par bloc ; potentiel de systeme non trivial.
  // ============================================================================================
  {
    AmrSystemConfig cfg;
    cfg.n = N;
    cfg.L = L;
    cfg.periodic = true;
    cfg.regrid_every = 0;  // multi-blocs : hierarchie figee

    AmrSystem sim(cfg);
    // bloc 0 : ions q=+1, schema none/rusanov.
    add_compiled_model(sim, "ions", exb_model(q0, B0), "none", "rusanov", "conservative",
                       "explicit",
                       /*gamma=*/1.4);
    // bloc 1 : electrons q=-1, schema minmod/rusanov (DIFFERENT du bloc 0). NE LEVE PLUS.
    add_compiled_model(sim, "electrons", exb_model(q1, B0), "minmod", "rusanov", "conservative",
                       "explicit", /*gamma=*/1.4);
    sim.set_poisson("charge_density", "geometric_mg", "periodic");
    sim.set_density("ions", rho0);
    sim.set_density("electrons", rho1);

    chk(sim.n_blocks() == 2, "A_two_compiled_blocks");

    const std::vector<double> d0_before = sim.density("ions");
    const std::vector<double> d1_before = sim.density("electrons");
    const double m0_before = sim.mass("ions");
    const double m1_before = sim.mass("electrons");

    sim.advance(0.01, 5);

    const std::vector<double> d0_after = sim.density("ions");
    const std::vector<double> d1_after = sim.density("electrons");
    chk(all_finite(d0_after) && all_finite(d1_after), "A_state_finite");
    chk(dmax_field(d0_after, d0_before) > 1e-6, "A_block0_evolved");
    chk(dmax_field(d1_after, d1_before) > 1e-6, "A_block1_evolved");

    chk(std::fabs(sim.mass("ions") - m0_before) < 1e-10, "A_block0_mass_conserved");
    chk(std::fabs(sim.mass("electrons") - m1_before) < 1e-10, "A_block1_mass_conserved");

    const std::vector<double> phi = sim.potential();
    double pmax = 0;
    for (double v : phi)
      pmax = std::max(pmax, std::fabs(v));
    chk(pmax > 1e-8, "A_system_potential_nonzero");
    chk(sim.n_patches() >= 1, "A_shared_hierarchy_has_fine_patch");
  }

  // ============================================================================================
  // (B) COUPLAGE entre deux blocs compiles : ionisation-like +S/-S (ions gagnent, neutrals perdent).
  //     La masse COMPOSITE n_ions + n_neutrals est conservee globalement ; la masse ions AUGMENTE.
  //     neutrals q=0 (pas de contribution Poisson) -> couplage pur entre deux blocs compiles.
  // ============================================================================================
  {
    AmrSystemConfig cfg;
    cfg.n = N;
    cfg.L = L;
    cfg.periodic = true;
    cfg.regrid_every = 0;

    AmrSystem sim(cfg);
    add_compiled_model(sim, "ions", exb_model(q0, B0), "minmod", "rusanov", "conservative",
                       "explicit", /*gamma=*/1.4);
    add_compiled_model(sim, "neutrals", exb_model(0.0, B0), "minmod", "rusanov", "conservative",
                       "explicit", /*gamma=*/1.4);
    sim.set_poisson("charge_density", "geometric_mg", "periodic");
    sim.set_density("ions", rho0);
    sim.set_density("neutrals", rho1);
    // gain sur "ions", perte sur "neutrals" (echange conservatif par cellule).
    register_ionization(sim, "ions", "neutrals", "ions", "neutrals", k);

    const double tot0 = sim.mass("ions") + sim.mass("neutrals");
    const double mi0 = sim.mass("ions");

    sim.advance(0.01, 6);

    chk(all_finite(sim.density("ions")) && all_finite(sim.density("neutrals")), "B_state_finite");
    const double tot1 = sim.mass("ions") + sim.mass("neutrals");
    chk(std::fabs(tot1 - tot0) < 1e-9, "B_composite_mass_conserved");
    chk(sim.mass("ions") > mi0 + 1e-6, "B_source_transfers_to_ions");
  }

  // ============================================================================================
  // (C) MELANGE compile (add_compiled_model) + natif (add_block) sur la meme hierarchie partagee.
  //     Le melange etait REFUSE avant ce capstone ; il doit desormais construire et evoluer.
  // ============================================================================================
  {
    AmrSystemConfig cfg;
    cfg.n = N;
    cfg.L = L;
    cfg.periodic = true;
    cfg.regrid_every = 0;

    AmrSystem sim(cfg);
    add_compiled_model(sim, "ions", exb_model(q0, B0), "minmod", "rusanov", "conservative",
                       "explicit", /*gamma=*/1.4);                                   // bloc COMPILE
    sim.add_block("electrons", exb_spec(q1, B0), "none", "rusanov", "conservative",  // bloc NATIF
                  "explicit", 1);
    sim.set_poisson("charge_density", "geometric_mg", "periodic");
    sim.set_density("ions", rho0);
    sim.set_density("electrons", rho1);

    chk(sim.n_blocks() == 2, "C_mixed_two_blocks");
    const std::vector<double> d0_before = sim.density("ions");
    const std::vector<double> d1_before = sim.density("electrons");
    sim.advance(0.01, 5);
    chk(dmax_field(sim.density("ions"), d0_before) > 1e-6, "C_compiled_block_evolved");
    chk(dmax_field(sim.density("electrons"), d1_before) > 1e-6, "C_native_block_evolved");
    chk(std::fabs(sim.mass("ions") - 0.0) >= 0.0, "C_mass_queryable");  // n'a pas crash
  }

  // ============================================================================================
  // (D) MONO-BLOC COMPILE BIT-IDENTIQUE : un seul bloc compile route TOUJOURS par AmrCouplerMP
  //     (jamais AmrRuntime). Meme cas joue deux fois -> dmax == 0 (le routage facade ne devie pas le
  //     mono-bloc compile vers le nouveau moteur, qui differe sur l'ordre des operations flottantes).
  // ============================================================================================
  {
    auto run_mono = [&]() {
      AmrSystemConfig cfg;
      cfg.n = N;
      cfg.L = L;
      cfg.periodic = true;
      cfg.regrid_every = 0;
      AmrSystem sim(cfg);
      add_compiled_model(sim, "ne", exb_model(q0, B0), "none", "rusanov", "conservative",
                         "explicit",
                         /*gamma=*/1.4);
      sim.set_poisson("charge_density", "geometric_mg", "periodic");
      sim.set_density("ne", rho0);
      sim.advance(0.01, 5);
      return sim.density("ne");
    };
    const std::vector<double> a = run_mono();
    const std::vector<double> b = run_mono();
    chk(dmax_field(a, b) == 0.0, "D_monoblock_compiled_bit_identical_dmax0");
  }

  // ============================================================================================
  // (E) NON-REGRESSION DU REFUS : le 2e add_compiled_model NE LEVE PLUS (preuve directe que la file
  //     de specs + build paresseux ont remplace l'ancien throw "un seul bloc compile").
  // ============================================================================================
  {
    const bool threw = raises([&] {
      AmrSystemConfig cfg;
      cfg.n = N;
      cfg.L = L;
      cfg.periodic = true;
      cfg.regrid_every = 0;
      AmrSystem sim(cfg);
      add_compiled_model(sim, "a", exb_model(q0, B0), "minmod", "rusanov", "conservative",
                         "explicit", 1.4);
      add_compiled_model(sim, "b", exb_model(q1, B0), "minmod", "rusanov", "conservative",
                         "explicit", 1.4);  // 2e bloc compile : NE DOIT PLUS lever
    });
    chk(!threw, "E_second_compiled_block_no_longer_throws");
  }

  // ============================================================================================
  // (F) IMEX + MULTIRATE (stride) + MASQUE IMEX PARTIEL sur un bloc COMPILE multi-bloc, via le chemin
  //     C++ DIRECT add_compiled_model(AmrSystem&) (qui, lui, SUPPORTE time="imex"/stride/masque ; le
  //     chemin .so loader ne les transporte PAS et la facade Python les REJETTE, cf. revue #195). Tous
  //     les autres cas du fichier utilisent time="explicit" : sans (F), le chemin IMEX des blocs
  //     compiles multi-blocs etait ENTIEREMENT NON teste. On verrouille trois proprietes :
  //       (F1) GENUINE IMEX : un bloc compile raide (1/eps >> 1/dt) en IMEX reste FINI/BORNE la ou le
  //            MEME bloc en EXPLICITE EXPLOSE (la source raide n'attaque que mx/my/E ; l'explosion
  //            contamine rho via le transport, observable a la facade density()).
  //       (F2) STRIDE EFFECTIF : stride=2 donne une trajectoire DIFFERENTE de stride=1 (memes eps/dt/
  //            macro-pas) -> la cadence est REELLEMENT prise en compte par le bloc compile (pas ignoree).
  //       (F3) MASQUE PARTIEL : implicit_roles={"momentum_x"} (sous-ensemble des composantes) est RESOLU
  //            contre l'Euler concret et le bloc tourne FINI ; un masque demande en EXPLICITE LEVE.
  //     Le 2e bloc (neutre, explicite) force la route MULTI-BLOC (AmrRuntime), pas le mono-bloc AmrCouplerMP.
  // ============================================================================================
  {
    const int Nf = 32;
    const double eps = 1e-5, dtF = 1e-3;
    const int KF = 12;
    const std::vector<double> rhoF = bubble(Nf);

    // Construit un AmrSystem a DEUX blocs COMPILES : un bloc raide (traitement/stride/masque variables)
    // + un bloc neutre explicite. Renvoie la densite du bloc raide apres KF macro-pas (peut lever au
    // build paresseux si le masque demande en explicite, capte par l'appelant).
    auto run_stiff_compiled = [&](bool imex, int stride,
                                  const std::vector<std::string>& impl_roles) {
      AmrSystemConfig cfg;
      cfg.n = Nf;
      cfg.L = L;
      cfg.periodic = true;
      cfg.regrid_every = 0;  // multi-blocs : hierarchie figee
      AmrSystem sim(cfg);
      add_compiled_model(sim, "stiff", stiff_cmodel(eps), "minmod", "rusanov", "conservative",
                         imex ? "imex" : "explicit", /*gamma=*/1.4, /*substeps=*/1, stride,
                         /*implicit_vars=*/{}, impl_roles);
      add_compiled_model(sim, "neutral", neutral_cmodel(), "minmod", "rusanov", "conservative",
                         "explicit", /*gamma=*/1.4);  // voisin explicite -> route MULTI-BLOC
      sim.set_poisson("charge_density", "geometric_mg", "periodic");
      sim.set_density("stiff", rhoF);
      sim.set_density("neutral", rhoF);
      const double m0 = sim.mass("stiff");
      for (int s = 0; s < KF; ++s)
        sim.advance(dtF, 1);
      const std::vector<double> d = sim.density("stiff");
      return std::make_pair(d, std::fabs(sim.mass("stiff") - m0));
    };

    // (F1) IMEX stable vs EXPLICITE qui explose (genuine IMEX, pas un no-op silencieux).
    const auto imex_res = run_stiff_compiled(/*imex=*/true, /*stride=*/1, {});
    chk(all_finite(imex_res.first) && maxabs(imex_res.first) < 1e3,
        "F1_compiled_imex_block_finite_and_bounded");
    chk(imex_res.second < 1e-10, "F1_compiled_imex_mass_conserved");  // source ne touche pas rho

    const auto expl_res = run_stiff_compiled(/*imex=*/false, /*stride=*/1, {});
    const bool expl_blew_up = !all_finite(expl_res.first) || maxabs(expl_res.first) > 1e3;
    chk(expl_blew_up, "F1_compiled_explicit_block_BLOWS_UP (disable-and-fail : IMEX requis)");
    std::printf("      (F) compile IMEX : max(rho)=%.3e ; EXPLICITE : %s\n", maxabs(imex_res.first),
                all_finite(expl_res.first) ? "borne >> 1" : "NON FINI (explose)");

    // (F2) stride=2 DIFFERE de stride=1 (memes eps/dt/macro-pas) : la cadence est effective.
    const auto imex_s2 = run_stiff_compiled(/*imex=*/true, /*stride=*/2, {});
    chk(all_finite(imex_s2.first), "F2_compiled_imex_stride2_finite");
    const double d_stride = dmax_field(imex_res.first, imex_s2.first);
    chk(d_stride > 0.0, "F2_compiled_imex_stride2_DIFFERS_from_stride1 (multirate effectif)");
    std::printf("      (F2) stride : dmax(rho ; s1 vs s2)=%.3e\n", d_stride);

    // (F3) masque IMEX PARTIEL implicit_roles={"momentum_x", "momentum_y"} : resolu contre l'Euler
    //      concret (4 var) en les indices 1/2, et applique en IMPLICITE a ces SEULES composantes. Comme
    //      la source raide n'attaque QUE mx/my, ce masque (qui couvre exactement les composantes raides)
    //      donne un run FINI/BORNE -> preuve que le masque a ete RESOLU et APPLIQUE.
    const auto imex_mask =
        run_stiff_compiled(/*imex=*/true, /*stride=*/1, {"momentum_x", "momentum_y"});
    chk(all_finite(imex_mask.first) && maxabs(imex_mask.first) < 1e3,
        "F3_compiled_imex_partial_mask_mxmy_resolved_and_runs");

    // (F3-bis) le masque est GENUINEMENT PARTIEL (pas un backward-Euler plein deguise) : un masque
    //          {momentum_x} SEUL laisse my en EXPLICITE -> my (raide) EXPLOSE et contamine rho. Si le
    //          masque etait ignore (tout implicite) le run resterait borne ; l'explosion PROUVE que la
    //          selection par composante est reellement honoree (mirroir de l'esprit disable-and-fail).
    const auto imex_mask_mx = run_stiff_compiled(/*imex=*/true, /*stride=*/1, {"momentum_x"});
    const bool partial_blew_up =
        !all_finite(imex_mask_mx.first) || maxabs(imex_mask_mx.first) > 1e3;
    chk(partial_blew_up,
        "F3_compiled_partial_mask_mx_only_leaves_my_EXPLICIT_and_blows_up (masque honore par "
        "composante)");

    // (F3-neg) masque PARTIEL demande en EXPLICITE -> LEVE (pas d'ignore silencieux ; meme garde que
    //          add_block / set_compiled_block, amr_system.cpp / amr_dsl_block.hpp).
    const bool mask_in_explicit_threw =
        raises([&] { (void)run_stiff_compiled(/*imex=*/false, /*stride=*/1, {"momentum_x"}); });
    chk(mask_in_explicit_threw, "F3_compiled_partial_mask_rejected_in_explicit");
  }

  if (fails == 0)
    std::printf("OK test_amr_multiblock_compiled\n");
  else
    std::printf("FAIL test_amr_multiblock_compiled : %d echec(s)\n", fails);
  return fails == 0 ? 0 : 1;
}
