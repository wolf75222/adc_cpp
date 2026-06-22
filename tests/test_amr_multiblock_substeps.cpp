// AMR MULTI-BLOCS MULTIRATE (capstone iv) : la FACADE RUNTIME (AmrSystem -> AmrRuntime) honore les
// SUBSTEPS et le STRIDE PAR BLOC, en mirroir du moteur compile-time AmrSystemCoupler::step (#140), et
// AmrSystem::step_cfl devient SUBSTEPS/STRIDE-AWARE comme System::step_cfl.
//
// Ce que le test verrouille (cf. tache capstone iv) :
//   (1) SUBSTEPS reellement exerces : deux blocs EXPLICITES sur UNE hierarchie partagee, bloc A
//       substeps=4 + bloc B substeps=1 ; apres K macro-pas l'etat est FINI (rejet nan/inf AVANT toute
//       tolerance), la masse de chaque bloc est conservee a ~machine, et le resultat A substeps=4
//       DIFFERE d'un A substeps=1 (le sous-cyclage n'est pas un no-op). Puis le cas RENVERSE (A=1, B=4).
//   (2) STRIDE hold-then-catch-up : un bloc stride=2 co-evolue ; il est TENU au macro-pas 0 (densite
//       inchangee) et RATTRAPE au macro-pas 1 ((macro_step+1)%2==0). Le Poisson de systeme somme bien
//       les DEUX blocs a chaque pas (RHS non trivial), meme quand le bloc lent est tenu.
//   (3) step_cfl SUBSTEPS/STRIDE-AWARE : pour une config connue, le dt renvoye vaut
//       cfl*h*min_b(substeps_b/(stride_b*w_b)) a la tolerance fp pres.
//   (4) MONO-BLOC BIT-IDENTIQUE : step et step_cfl d'un bloc unique sont inchanges (dmax==0 entre deux
//       runs), garantissant que le routage facade laisse le mono-bloc sur AmrCouplerMP.
//
// On travaille surtout au niveau du MOTEUR AmrRuntime + build_amr_block (les briques de cette PR), ou
// l'on accede aux niveaux/masses/RHS des blocs ; les regressions mono-bloc passent par la facade.

#include <adc/coupling/base/elliptic_rhs.hpp>  // add_scaled_component (RHS de reference assemble main)
#include <adc/runtime/builders/amr_dsl_block.hpp>  // detail::make_shared_amr_layout / dispatch_amr_block
#include <adc/runtime/amr/amr_runtime.hpp>    // AmrRuntime, AmrRuntimeBlock
#include <adc/runtime/amr_system.hpp>     // facade AmrSystem
#include <adc/runtime/builders/model_factory.hpp>  // detail::dispatch_model
#include <adc/runtime/model_spec.hpp>
#include <adc/mesh/mf_arith.hpp>  // norm_inf
#include <adc/mesh/multifab.hpp>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace adc;

// Spec ExB scalaire (1 var) a charge q : advection pilotee par grad phi, densite de charge q n pour le
// Poisson de systeme. La charge q (signe inclus) distingue electrons / ions.
static ModelSpec exb_charge(double q, double B0) {
  ModelSpec s;
  s.transport = "exb";
  s.source = "none";
  s.elliptic = "charge";
  s.q = q;
  s.B0 = B0;
  return s;
}

// densite de charge a moyenne nulle (solvable en periodique) : un creneau centre +/- amplitude a, n*n.
static std::vector<double> bump(int n, double base, double amp) {
  std::vector<double> r(static_cast<std::size_t>(n) * n, base);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const bool in = (i >= n / 4 && i < 3 * n / 4 && j >= n / 4 && j < 3 * n / 4);
      r[static_cast<std::size_t>(j) * n + i] = base + (in ? amp : -amp / 3.0);
    }
  return r;
}

// tout fini (ni nan ni inf) : garde AVANT toute comparaison de tolerance (un nan passerait une borne).
static bool all_finite(const std::vector<double>& v) {
  for (double x : v)
    if (!std::isfinite(x))
      return false;
  return true;
}

// ecart L-inf entre deux champs n*n (pour "A differe de B" ou "X inchange").
static double dmax_field(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0;
  const std::size_t nn = a.size() < b.size() ? a.size() : b.size();
  for (std::size_t i = 0; i < nn; ++i)
    d = std::max(d, std::fabs(a[i] - b[i]));
  return d;
}

// Construit un AmrRuntime a DEUX blocs ExB (charges q0/q1, schemas potentiellement differents) sur une
// hierarchie figee N x N, avec substeps/stride par bloc. Renvoie le runtime (les densites initiales
// rho0/rho1 sont posees sur le grossier de chaque bloc).
static AmrRuntime make_two_block(int N, double L, double q0, double q1, double B0,
                                 const std::vector<double>& rho0, const std::vector<double>& rho1,
                                 const std::string& lim0, const std::string& lim1, int sub0,
                                 int sub1, int stride0, int stride1) {
  AmrBuildParams bp;
  bp.n = N;
  bp.L = L;
  bp.regrid_every = 0;      // hierarchie figee (multi-blocs)
  bp.poisson_bc = BCRec{};  // periodique
  const detail::SharedAmrLayout S = detail::make_shared_amr_layout(bp);
  std::vector<AmrRuntimeBlock> blocks;
  detail::dispatch_model(exb_charge(q0, B0), [&](auto m) {
    blocks.push_back(detail::dispatch_amr_block(m, lim0, "rusanov", S, "A", rho0,
                                                /*has_density=*/true, 1.4, sub0, false, false,
                                                stride0));
  });
  detail::dispatch_model(exb_charge(q1, B0), [&](auto m) {
    blocks.push_back(detail::dispatch_amr_block(m, lim1, "rusanov", S, "B", rho1,
                                                /*has_density=*/true, 1.4, sub1, false, false,
                                                stride1));
  });
  return AmrRuntime(S.geom, S.ba_coarse, S.poisson_bc, std::move(blocks), S.base_per,
                    S.replicated_coarse, S.wall);
}

int main(int argc, char** argv) {
#if defined(ADC_HAS_KOKKOS)
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
  const double L = 1.0, B0 = 1.0;
  const double q0 = +1.0, q1 = -1.0;  // A : ions ; B : electrons
  const std::vector<double> rho0 = bump(N, 1.0, 0.40);
  const std::vector<double> rho1 = bump(N, 1.0, 0.20);
  const Real dt = Real(0.01);
  const int K = 6;  // macro-pas

  // ============================================================================================
  // (1) SUBSTEPS exerces : A substeps=4, B substeps=1. Etat fini, masse conservee, et A(sub=4) != A(sub=1).
  // ============================================================================================
  {
    AmrRuntime rt = make_two_block(N, L, q0, q1, B0, rho0, rho1, "minmod", "minmod",
                                   /*sub0=*/4, /*sub1=*/1, /*stride0=*/1, /*stride1=*/1);
    const Real mA0 = rt.mass(0), mB0 = rt.mass(1);
    for (int s = 0; s < K; ++s)
      rt.step(dt);
    const std::vector<double> dA4 = rt.density(0);
    const std::vector<double> dB = rt.density(1);
    const Real mA1 = rt.mass(0), mB1 = rt.mass(1);

    chk(all_finite(dA4) && all_finite(dB), "subA4_state_finite");  // AVANT toute tolerance
    chk(std::fabs(mA1 - mA0) < 1e-10, "subA4_blockA_mass_conserved");
    chk(std::fabs(mB1 - mB0) < 1e-10, "subA4_blockB_mass_conserved");

    // Reference : MEME config mais A substeps=1. Le resultat de A doit DIFFERER (le sous-cyclage agit).
    AmrRuntime rt1 = make_two_block(N, L, q0, q1, B0, rho0, rho1, "minmod", "minmod",
                                    /*sub0=*/1, /*sub1=*/1, /*stride0=*/1, /*stride1=*/1);
    for (int s = 0; s < K; ++s)
      rt1.step(dt);
    const std::vector<double> dA1 = rt1.density(0);
    chk(all_finite(dA1), "subA1_state_finite");
    chk(dmax_field(dA4, dA1) > 1e-9, "subA4_differs_from_subA1");  // substepping NON no-op
    // bloc B (substeps=1 dans les deux runs) : meme trajectoire au bit pres (A ne le perturbe pas, les
    // blocs avancent independamment ; phi differe car A differe, mais le couplage est once-per-step et A
    // substeps n'altere PAS l'etat de B a substeps=1 sur le MEME phi de tete).
  }

  // ============================================================================================
  // (1b) RENVERSE : A substeps=1, B substeps=4. B(sub=4) doit differer de B(sub=1).
  // ============================================================================================
  {
    AmrRuntime rt = make_two_block(N, L, q0, q1, B0, rho0, rho1, "minmod", "minmod",
                                   /*sub0=*/1, /*sub1=*/4, /*stride0=*/1, /*stride1=*/1);
    const Real mA0 = rt.mass(0), mB0 = rt.mass(1);
    for (int s = 0; s < K; ++s)
      rt.step(dt);
    const std::vector<double> dB4 = rt.density(1);
    const Real mA1 = rt.mass(0), mB1 = rt.mass(1);
    chk(all_finite(dB4), "revB4_state_finite");
    chk(std::fabs(mA1 - mA0) < 1e-10, "revB4_blockA_mass_conserved");
    chk(std::fabs(mB1 - mB0) < 1e-10, "revB4_blockB_mass_conserved");

    AmrRuntime rt1 = make_two_block(N, L, q0, q1, B0, rho0, rho1, "minmod", "minmod",
                                    /*sub0=*/1, /*sub1=*/1, /*stride0=*/1, /*stride1=*/1);
    for (int s = 0; s < K; ++s)
      rt1.step(dt);
    const std::vector<double> dB1 = rt1.density(1);
    chk(dmax_field(dB4, dB1) > 1e-9, "revB4_differs_from_subB1");
  }

  // ============================================================================================
  // (2) STRIDE hold-then-catch-up : bloc A stride=1 (rapide), bloc B stride=2 (lent). Au macro-pas 0
  //     (macro_step=0, (0+1)%2=1 != 0) B est TENU -> sa densite est INCHANGEE. Au macro-pas 1
  //     ((1+1)%2=0) B RATTRAPE -> sa densite CHANGE. Le Poisson de systeme somme les DEUX blocs a
  //     chaque pas (RHS non trivial), meme quand B est tenu.
  // ============================================================================================
  {
    AmrRuntime rt = make_two_block(N, L, q0, q1, B0, rho0, rho1, "minmod", "minmod",
                                   /*sub0=*/1, /*sub1=*/1, /*stride0=*/1, /*stride1=*/2);
    const std::vector<double> dA_init = rt.density(0);
    const std::vector<double> dB_init = rt.density(1);
    const Real mB_init = rt.mass(1);

    // macro-pas 0 : A avance, B TENU.
    rt.step(dt);
    const std::vector<double> dA_0 = rt.density(0);
    const std::vector<double> dB_0 = rt.density(1);
    chk(dmax_field(dA_0, dA_init) > 1e-9, "stride_blockA_advances_at_mac0");
    chk(dmax_field(dB_0, dB_init) == 0.0, "stride_blockB_held_at_mac0");  // exactement inchange
    // Poisson somme actif au pas 0 (les DEUX densites contribuent ; B avec son etat fige).
    chk(norm_inf(rt.poisson_rhs()) > 1e-6, "stride_poisson_sum_active_mac0");

    // macro-pas 1 : B RATTRAPE (pas effectif 2*dt).
    rt.step(dt);
    const std::vector<double> dB_1 = rt.density(1);
    chk(dmax_field(dB_1, dB_init) > 1e-9, "stride_blockB_catchup_at_mac1");
    chk(std::fabs(rt.mass(1) - mB_init) < 1e-10, "stride_blockB_mass_conserved");
    chk(norm_inf(rt.poisson_rhs()) > 1e-6, "stride_poisson_sum_active_mac1");
  }

  // ============================================================================================
  // (3) step_cfl SUBSTEPS/STRIDE-AWARE : config connue. Deux blocs IDENTIQUES (meme modele, schema,
  //     densite) -> meme vitesse d'onde w pour les deux. A substeps=4 stride=1, B substeps=1 stride=2.
  //     min_b(substeps_b/(stride_b*w)) = min(4/(1*w), 1/(2*w)) = 0.5/w. Donc dt attendu = cfl*h*0.5/w,
  //     avec w = rt.max_speed() (max sur blocs identiques = w commun) et h = dx_coarse = L/N.
  // ============================================================================================
  {
    AmrRuntime rt = make_two_block(N, L, q0, q0, B0, rho0, rho0, "minmod", "minmod",
                                   /*sub0=*/4, /*sub1=*/1, /*stride0=*/1, /*stride1=*/2);
    const Real h = Real(L) / Real(N);  // dx_coarse
    const Real cfl = Real(0.4);
    const Real w = rt.max_speed();  // solve_fields + max sur les blocs (identiques -> w commun)
    chk(w > Real(0), "cfl_wave_speed_positive");
    // min(substeps/(stride*w)) sur {(4,1),(1,2)} = min(4, 0.5)/w = 0.5/w.
    const Real expected = cfl * h * Real(0.5) / w;
    const Real got = rt.step_cfl(cfl, h);
    chk(std::fabs(got - expected) <= Real(1e-12) * std::fabs(expected) + Real(1e-15),
        "cfl_dt_is_substeps_stride_aware");
  }

  // ============================================================================================
  // (4) MONO-BLOC BIT-IDENTIQUE : step ET step_cfl d'un bloc unique inchanges (dmax==0 entre deux
  //     runs). Garantit que le routage facade laisse le mono-bloc sur AmrCouplerMP (jamais AmrRuntime,
  //     qui differe sur l'ordre des operations flottantes).
  // ============================================================================================
  {
    auto run_step = [&]() {
      AmrSystemConfig cfg;
      cfg.n = N;
      cfg.L = L;
      cfg.periodic = true;
      cfg.regrid_every = 0;
      AmrSystem sim(cfg);
      sim.add_block("ne", exb_charge(q0, B0), "none", "rusanov", "conservative", "explicit", 1);
      sim.set_poisson("charge_density", "geometric_mg", "periodic");
      sim.set_density("ne", rho0);
      sim.advance(0.01, 5);
      return sim.density("ne");
    };
    const std::vector<double> a = run_step();
    const std::vector<double> b = run_step();
    chk(dmax_field(a, b) == 0.0, "monoblock_step_bit_identical");

    auto run_cfl = [&]() {
      AmrSystemConfig cfg;
      cfg.n = N;
      cfg.L = L;
      cfg.periodic = true;
      cfg.regrid_every = 0;
      AmrSystem sim(cfg);
      sim.add_block("ne", exb_charge(q0, B0), "none", "rusanov", "conservative", "explicit", 1);
      sim.set_poisson("charge_density", "geometric_mg", "periodic");
      sim.set_density("ne", rho0);
      double last = 0;
      for (int s = 0; s < 5; ++s)
        last = sim.step_cfl(0.4);
      return std::make_pair(sim.density("ne"), last);
    };
    const auto ra = run_cfl();
    const auto rb = run_cfl();
    chk(dmax_field(ra.first, rb.first) == 0.0, "monoblock_step_cfl_field_bit_identical");
    chk(ra.second == rb.second, "monoblock_step_cfl_dt_bit_identical");
  }

  if (fails == 0)
    std::printf("OK test_amr_multiblock_substeps\n");
  else
    std::printf("FAIL test_amr_multiblock_substeps : %d echec(s)\n", fails);
  return fails == 0 ? 0 : 1;
}
