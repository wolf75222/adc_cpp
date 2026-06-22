// Validation DEVICE (GH200, Kokkos Cuda) + MPI du CHEMIN NATIF DE PRODUCTION du DSL : le backend
// "production" (#85, add_native_block / emit_cpp_native_loader) branche un modele genere par le DSL
// dans le System via le gabarit en-tete adc::add_compiled_model<ProdModel>(System&, ...). C'est
// EXACTEMENT la fonction qu'inline le loader natif genere (adc_install_native) ; le bloc tourne
// ensuite le MEME chemin que add_block (fill_boundary = halos MPI, assemble_rhs<Limiter, Flux> device,
// SSPRK2 du coeur), ZERO-COPIE, device-clean via les FONCTEURS NOMMES de block_builder.hpp (#64).
//
// CE harness exerce add_compiled_model directement (la piece numerique du chemin de production : ce
// que le loader natif appelle), avec un CompositeModel CONCRET identique a ce qu'emet le DSL pour un
// modele euler compressible couple au Poisson (euler_poisson) : CompositeModel<Euler, PotentialForce,
// ChargeDensity>. Euler (4 var) expose pressure/wave_speeds -> flux HLLC disponible (chemin de
// production ordre 2 + Riemann complet, avec equation d'energie) ; PotentialForce ajoute (q/m) rho E
// (E = -grad phi du canal aux, peuple par solve_fields) sur la quantite de mouvement + le travail sur
// l'energie ; ChargeDensity couple le bloc au Poisson de systeme (elliptic_rhs = q rho). Le harness
// instancie donc le chemin de production ENTIER : Poisson + derivation aux + transport HLLC + source
// E. C'est exactement le modele des references CPU/GPU (test_compiled_model_parity, phase7_system).
// Le binding Python (add_native_block via dlopen d'un loader nvcc-Cuda sur aarch64) est un montage
// SEPARE plus lourd, hors perimetre ici (cf. note de suivi dans le rapport).
//
// On lie python/system.cpp (definitions de System::install_block / grid_context / ensure_aux_width,
// appelees par add_compiled_model), comme le harness frere phase7_system.cpp et le test CPU
// tests/test_compiled_model_parity.cpp. Le MEME source compile en exec=Cuda (nvcc_wrapper + Kokkos)
// et en oracle exec=Serial (g++, ADC_HAS_KOKKOS off). --dump=<prefixe> ecrit U (par cellule, tous les
// comp) -> diff binaire Cuda vs Serial : on rapporte dmax_abs par cellule avec TOLERANCE (PAS
// bit-exact apres quelques pas : les reductions Kokkos reassocient les sommes FP, ecart ~1e-13 attendu).
//
// MPI : le System a un domaine MONO-BOITE (DistributionMapping(1, n_ranks())) -> la boite 0 appartient
// au rang 0 (round-robin : 0 % np == 0). Le bloc n'est donc PAS decoupe entre rangs (a la difference du
// chemin AMR distribue, deja valide). Tous les rangs construisent le System et appellent step()
// (collectifs du Poisson coherents) ; seul le rang 0 (proprietaire de la boite) lit / dumpe l'etat.
// Les invariants du rang proprietaire doivent etre INVARIANTS au nombre de rangs : le script ROMEO
// compare np=1 (oracle mono-GPU) a np=2/4. C'est la validation realiste du chemin natif System sous
// MPI (la seam MPI fill_boundary + collectifs Poisson tourne ; le multi-GPU par decoupage de domaine
// est le chemin AMR, couvert ailleurs).

#include <adc/physics/bricks.hpp>     // CompositeModel + Euler + PotentialForce + ChargeDensity
#include <adc/runtime/builders/dsl_block.hpp>  // add_compiled_model<Model> (gabarit natif du chemin production)
#include <adc/runtime/system.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>  // getenv (trace d'etape optionnelle)
#include <cstring>
#include <string>
#include <vector>

#include <adc/parallel/comm.hpp>  // comm_init, my_rank, n_ranks, all_reduce_max

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace adc;

// Modele genere par le DSL pour un euler_poisson couple : transport compressible (4 var : rho, rho u,
// rho v, E ; pression de gaz parfait -> HLLC disponible) + force du potentiel (q/m) rho E (E = -grad
// phi, lue dans aux) + densite de charge q rho au second membre du Poisson. C'est la forme exacte d'un
// CompositeModel<Hyp, Src, Ell> qu'emet _emit_bricks (cf. python/adc/dsl.py).
using ProdModel = CompositeModel<Euler, PotentialForce, ChargeDensity>;

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(ADC_HAS_KOKKOS)
  // L'allocateur unifie (kokkos_malloc<SharedSpace>) exige Kokkos initialise AVANT la 1ere allocation
  // (le ctor de System alloue l'aux) : ScopeGuard (RAII) avant toute construction de System.
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  const int me = my_rank(), np = n_ranks();

  std::string dump_prefix;
  for (int k = 1; k < argc; ++k)
    if (std::strncmp(argv[k], "--dump=", 7) == 0)
      dump_prefix = argv[k] + 7;

#if defined(ADC_HAS_KOKKOS)
  const char* space = Kokkos::DefaultExecutionSpace::name();
#else
  const char* space = "Serial(host)";
#endif

  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c && me == 0) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };
  // Trace d'etape OPTIONNELLE (ADC_DSLPROD_TRACE=1) : ecrit sur stderr, flush immediat, pour localiser
  // un eventuel crash device entre construction / add_compiled_model / solve_fields / step. Inerte par
  // defaut (aucun effet sur les sorties machine-parsables). Utile au diagnostic GH200.
  const bool trace = (me == 0) && std::getenv("ADC_DSLPROD_TRACE");
  auto step_mark = [&](const char* w) {
    if (trace) {
      std::fprintf(stderr, "[trace] %s\n", w);
      std::fflush(stderr);
    }
  };

  // n = 64 (= 2^6) : compatible avec le solveur FFT periodique (n = 2^k) ET GeometricMG.
  const int n = 64;
  const double L = 1.0;
  const double gamma = 1.4;    // indice adiabatique du gaz parfait (Euler)
  const double qom = 1.0;      // (q/m) de la force du potentiel
  const double qcharge = 1.0;  // densite de charge q rho

  SystemConfig cfg;
  cfg.n = n;
  cfg.L = L;
  cfg.periodic = true;
  step_mark("avant System(cfg)");
  System sim(cfg);
  step_mark("apres System(cfg)");

  // CHEMIN DE PRODUCTION : le modele genere par le DSL branche comme bloc natif (ce que le loader
  // natif inline). HLLC (Riemann complet) + minmod (ordre 2) + reconstruction primitive (rho, u, v)
  // + explicite SSPRK2 : numerique de production identique a un add_block.
  add_compiled_model<ProdModel>(
      sim, "gas",
      ProdModel{Euler{Real(gamma)}, PotentialForce{Real(qom)}, ChargeDensity{Real(qcharge)}},
      "minmod", "hllc", "primitive", "explicit", /*gamma=*/gamma);
  step_mark("apres add_compiled_model");
  // Solveur Poisson : geometric_mg par defaut (tout cas) ; ADC_DSLPROD_SOLVER=fft pour le FFT
  // periodique (n = 2^k) -- utile au diagnostic device. n = 64 satisfait les deux.
  const char* solver_env = std::getenv("ADC_DSLPROD_SOLVER");
  const std::string solver = solver_env ? solver_env : "geometric_mg";
  sim.set_poisson("charge_density", solver);
  step_mark("apres set_poisson");

  // Densite a moyenne ~1 + perturbation gaussienne centree (charge nette -> phi non trivial -> force E
  // non triviale). set_density / lectures d'etat ne touchent fab(0) que sur le rang proprietaire.
  if (me == 0) {
    std::vector<double> rho(static_cast<std::size_t>(n) * n);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        const double x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5;
        rho[static_cast<std::size_t>(j) * n + i] = 1.0 + 0.3 * std::exp(-(x * x + y * y) / 0.02);
      }
    sim.set_density("gas", rho);
  }
  step_mark("apres set_density");

  // solve_fields() + step() embarquent des collectifs (Poisson MG, reductions) : TOUS les rangs les
  // appellent pour que les collectifs concordent. Le transport / la source ne touchent que la boite
  // locale (vide hors du rang proprietaire). Quelques pas explicites a dt fixe (sous-CFL : c ~
  // sqrt(gamma) ~ 1.18, dx=1/48 -> dt_CFL ~ 0.018 ; dt=5e-3 conserve la stabilite et reste
  // deterministe a dt fixe). Cuda vs Serial : reproductible SOUS TOLERANCE dmax_abs, PAS bit-exact
  // (les reductions Kokkos reassocient les sommes FP). np vs np : checksums du rang proprietaire
  // invariants au nombre de rangs (bloc non decoupe, un seul proprietaire).
  sim.solve_fields();
  step_mark("apres solve_fields");
  const double dt = 5e-3;
  const int nsteps = 5;
  for (int s = 0; s < nsteps; ++s) {
    sim.step(dt);
    if (trace) {
      std::fprintf(stderr, "[trace] apres step %d\n", s);
      std::fflush(stderr);
    }
  }
  step_mark("apres tous les step");

#if defined(ADC_HAS_KOKKOS)
  Kokkos::
      fence();  // ceinture avant la lecture hote (get_state / potential fencent deja en interne)
#endif

  // Lecture / checksum SUR LE RANG PROPRIETAIRE (boite 0 -> rang 0). U = (rho, rho u, rho v, E),
  // composante-majeur, n*n par composante. Le potentiel phi est lu de meme (n*n).
  double mass = 0, csum = 0, csumsq = 0, cmax = 0, pmax = 0;
  std::vector<double> U, phi;
  // mass() est COLLECTIF (all_reduce) : appele par TOUS les rangs AVANT le bloc rang-0. Sinon le rang 0
  // entre dans un collectif que les autres rangs ne font pas (ils filent vers l'all_reduce_max plus bas)
  // -> collectifs desalignes -> deadlock / crash en np>1.
  mass = sim.mass("gas");
  if (me == 0) {
    U = sim.get_state("gas");
    phi = sim.potential();
    for (double v : U) {
      csum += v;
      csumsq += v * v;
      const double a = std::fabs(v);
      if (a > cmax)
        cmax = a;
    }
    for (double v : phi)
      pmax = std::fmax(pmax, std::fabs(v));

    chk(U.size() == static_cast<std::size_t>(4) * n * n, "state_size_4xn2");
    chk(phi.size() == static_cast<std::size_t>(n) * n, "phi_size_n2");
    chk(cmax > 1e-6, "state_nontrivial");
    chk(pmax > 1e-9, "poisson_phi_nontrivial");  // le couplage elliptic_rhs a bien produit un phi
    chk(std::isfinite(mass) && std::isfinite(csum) && std::isfinite(csumsq), "finite");

    // Ligne machine-parsable (le script ROMEO DIFF ces lignes entre np=1/2/4 ; np=1 = oracle mono-GPU)
    // ET entre exec=Cuda et exec=Serial pour la parite device.
    std::printf(
        "DSLPROD np=%d exec=%s solver=%s n=%d steps=%d | mass=%.17e csum=%.17e csumsq=%.17e "
        "cmax=%.17e pmax=%.17e\n",
        np, space, solver.c_str(), n, nsteps, mass, csum, csumsq, cmax, pmax);
  }

  // Parite cross-rang du checksum proprietaire : le bloc n'etant pas decoupe, les invariants du rang 0
  // doivent etre invariants au nombre de rangs. On reduit l'ecart max-min de chaque quantite sur tous
  // les rangs (vaut 0 ssi tous identiques ; les rangs non proprietaires portent 0, donc on compare
  // explicitement au rang 0 via broadcast par all_reduce du couple max/-max sur la valeur du rang 0).
  // Pour rester simple ET correct avec un seul proprietaire : on diffuse la valeur du rang 0 par
  // all_reduce_max (les autres portent -inf via -all_reduce_max(-x) ; ici seul le rang 0 a une valeur,
  // donc all_reduce_max rend la valeur du rang 0). spread doit valoir 0 a la machine (un seul ecrivain).
  const double msum_owner = all_reduce_max(me == 0 ? mass : -1e300);
  const double csum_owner = all_reduce_max(me == 0 ? csum : -1e300);
  if (me == 0) {
    const double dspread = std::fmax(std::fabs(mass - msum_owner), std::fabs(csum - csum_owner));
    std::printf("DSLPROD np=%d crossrank_owner_spread=%.3e\n", np, dspread);
    chk(dspread == 0.0, "crossrank_owner_consistent");
  }

  // Dump binaire de U (par cellule, tous comp) sur le rang proprietaire -> diff_bin Cuda vs Serial.
  if (me == 0 && !dump_prefix.empty()) {
    const std::string path = dump_prefix + "_dslprod.bin";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f) {
      std::fwrite(U.data(), 1, U.size() * sizeof(double), f);
      std::fclose(f);
      std::printf("  dump %s (%zu doubles)\n", path.c_str(), U.size());
    }
  }

  if (me == 0 && fails == 0)
    std::printf("OK gpu_dsl_production_validate (exec=%s, np=%d)\n", space, np);

  comm_finalize();
  return fails == 0 ? 0 : 1;
}
