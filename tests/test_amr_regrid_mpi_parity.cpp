// PARITE MPI du REGRID D'UNION DES TAGS multi-blocs (T4 du design
// docs/AMR_REGRID_UNION_TAGS_DESIGN.md, suivi #199). C'est le verrou de parite cross-rang manquant :
// le regrid d'union reduit les tags cross-rang par all_reduce_or_inplace (etape R4) AVANT le
// clustering Berger-Rigoutsos, de sorte que TOUS les rangs partent de la MEME grille de tags et
// produisent EXACTEMENT le meme BoxArray fin -> meme DistributionMapping -> hierarchie IDENTIQUE quel
// que soit le nombre de rangs. Si la reduction (R4) etait omise ou buguee, deux rangs partiraient de
// grilles de tags differentes, le clustering divergerait par rang et MPI desynchroniserait (risques
// X1/X2 du design).
//
// SCENARIO (le MEME a np=1/2/4) : deux blocs ExB a charges opposees (Poisson de systeme somme), un
// blob a gauche et un a droite, sur une hierarchie 2 niveaux. GROSSIER REPARTI (distribute_coarse=true,
// BoxArray multi-box round-robin) : c'est le seul chemin ou (R4) est active (en grossier REPLIQUE,
// chaque rang a deja la grille de tags complete, all_reduce_or serait l'identite). regrid_every=2 :
// la grille se re-grille effectivement pendant la sequence, en suivant l'union des tags densite par
// bloc + le tag de phi sur |grad phi| (set_phi_refinement, le predicat cable depuis la facade par CE
// suivi). On avance plusieurs macro-pas (donc plusieurs regrids), puis on observe la hierarchie finale.
//
// ASSERTIONS :
//   (1) CONSISTANCE CROSS-RANG (dans CHAQUE run) : la densite grossiere de chaque bloc est reconstruite
//       GLOBALEMENT (all_reduce des boites disjointes du grossier reparti), donc n*n sur chaque rang ;
//       son checksum, le potentiel de systeme et n_patches sont des grandeurs GLOBALES -> spread max
//       cross-rang == 0 (insensible a l'ordre via all_reduce_max). Un bug de halo / Poisson somme /
//       layout fin divergent le casserait.
//   (2) PARITE AU NB DE RANGS : on imprime n_patches + des checksums (densite par bloc + potentiel) ;
//       la CI relance le MEME binaire en np=1/2/4 et DIFFE la ligne AMRREGRID (np=1 = oracle ;
//       np=2/4 doivent etre BIT-IDENTIQUES). Le n_patches identique cross-np = layout fin identique
//       cross-np (LE point du regrid d'union : un seul fb/dmap pour tous les rangs).
//   (3) CONSERVATION PAR BLOC a travers les regrids : la masse de chaque bloc est conservee (reflux +
//       report fin exact + interp parent piecewise-constant conservatif au sens integral).
//
// Independant du backend (Kokkos Serial CI, Kokkos Cuda GH200). Compile le runtime AmrSystem comme
// test_mpi_amr_twoblock_parity (avec python/amr_system.cpp).
#include <adc/runtime/amr_system.hpp>
#include <adc/runtime/model_spec.hpp>
#include <adc/parallel/comm.hpp>  // comm_init, my_rank, n_ranks, all_reduce_*

#include "test_harness.hpp"  // adc::test::checksum (somme des carres partagee)

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace adc;

static ModelSpec exb_charge(double q, double B0) {
  ModelSpec s;
  s.transport = "exb";
  s.source = "none";
  s.elliptic = "charge";
  s.q = q;
  s.B0 = B0;
  return s;
}

// Disque gaussien centre en (cx, cy) du domaine [0,1]^2, amplitude amp sur une base, n*n row-major.
// Le maximum (base + amp) depasse le seuil de raffinement -> la region taguee suit le blob (regrid).
static std::vector<double> blob(int n, double cx, double cy, double amp, double base,
                                double width) {
  std::vector<double> rho(static_cast<std::size_t>(n) * n, base);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n, y = (j + 0.5) / n;
      const double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
      rho[static_cast<std::size_t>(j) * n + i] = base + amp * std::exp(-r2 / (width * width));
    }
  return rho;
}

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(ADC_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  const int me = my_rank(), np = n_ranks();
  const int n = 32;
  const double B0 = 1.0, q0 = +1.0, q1 = -1.0;
  const std::vector<double> rho0 = blob(n, 0.30, 0.5, 1.0, 1.0, 0.07);  // bloc a gauche
  const std::vector<double> rho1 = blob(n, 0.70, 0.5, 1.0, 1.0, 0.07);  // bloc a droite

  AmrSystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;
  cfg.regrid_every = 2;          // REGRID ACTIF : la hierarchie se re-grille pendant la sequence
  cfg.distribute_coarse = true;  // GROSSIER REPARTI : active la reduction collective des tags (R4)
  // coarse_max_grid = 0 -> n/2 (decoupage 2x2 multi-box, le moins agressif pour le MG geometrique).

  AmrSystem sys(cfg);
  sys.add_block("a", exb_charge(q0, B0), "minmod", "rusanov", "conservative", "explicit", 1);
  sys.add_block("b", exb_charge(q1, B0), "minmod", "rusanov", "conservative", "explicit", 1);
  sys.set_poisson("charge_density", "geometric_mg", "periodic");
  sys.set_refinement(1.5);  // tag densite > 1.5 (union des deux blobs, par bloc)
  sys.set_phi_refinement(
      1e-3);  // tag |grad phi| > 1e-3 (bord d'anneau ; predicat phi cable facade)
  sys.set_density("a", rho0);
  sys.set_density("b", rho1);

  const double m0a = sys.mass("a");  // declenche le build paresseux
  const double m0b = sys.mass("b");

  const double dt = 1e-3;
  for (int s = 0; s < 16; ++s)
    sys.step(dt);  // 16 macro-pas, regrid tous les 2 -> plusieurs regrids

#if defined(ADC_HAS_KOKKOS)
  Kokkos::fence();
#endif
  const std::vector<double> da = sys.density("a");
  const std::vector<double> db = sys.density("b");
  const std::vector<double> phi = sys.potential();
  const double ma = sys.mass("a"), mb = sys.mass("b");
  const int npatch = sys.n_patches();  // nombre de patchs fins = signature du layout fin d'union

  using adc::test::checksum;  // somme des carres partagee (signature deterministe d'un champ)
  const double ca = checksum(da), cb = checksum(db), cp = checksum(phi);

  // (1) CONSISTANCE CROSS-RANG : densite reconstruite globalement + potentiel + n_patches sont des
  // grandeurs GLOBALES identiques sur tout rang. spread = max - min cross-rang (insensible a l'ordre).
  auto spread = [](double x) { return all_reduce_max(x) - (-all_reduce_max(-x)); };
  const double sp = std::fmax(
      std::fmax(spread(ca), spread(cb)),
      std::fmax(spread(cp),
                std::fmax(spread(ma), std::fmax(spread(mb), spread(static_cast<double>(npatch))))));

  int fails = 0;
  if (me == 0) {
    // Ligne PARITE (diffee cross-np par la CI) : n_patches + checksums imprimes en %.17e bit-exact.
    std::printf(
        "AMRREGRID np=%d | n_patches=%d | csum_a=%.17e csum_b=%.17e csum_phi=%.17e | "
        "crossrank_spread=%.3e\n",
        np, npatch, ca, cb, cp, sp);
    std::printf("AMRREGRID conservation: dm_a=%.3e dm_b=%.3e | mass_a=%.17e mass_b=%.17e\n",
                std::fabs(ma - m0a), std::fabs(mb - m0b), ma, mb);

    if (!(da.size() == static_cast<std::size_t>(n) * n)) {
      std::printf("FAIL taille densite (%zu != %d)\n", da.size(), n * n);
      ++fails;
    }
    if (!(cp > 1e-12)) {
      std::printf("FAIL potentiel trivial (Poisson somme inactif)\n");
      ++fails;
    }
    if (!(npatch >= 1)) {
      std::printf("FAIL aucun patch fin (le regrid n'a pas raffine)\n");
      ++fails;
    }
    if (!std::isfinite(ca) || !std::isfinite(cb) || !std::isfinite(cp)) {
      std::printf("FAIL champ non fini (MG diverge / regrid casse ?)\n");
      ++fails;
    }
    // (3) masse de CHAQUE bloc conservee a travers les regrids (report fin exact + interp parent
    // piecewise-constant conservatif au sens integral + reflux conservatif).
    if (!(std::fabs(ma - m0a) < 1e-9)) {
      std::printf("FAIL masse bloc a non conservee a travers le regrid\n");
      ++fails;
    }
    if (!(std::fabs(mb - m0b) < 1e-9)) {
      std::printf("FAIL masse bloc b non conservee a travers le regrid\n");
      ++fails;
    }
    // (1) grossier reparti reconstruit GLOBALEMENT + layout fin d'union UNIQUE -> tout bit-identique
    // cross-rang (spread exactement 0). Le n_patches dans le spread = meme layout fin sur tous les rangs.
    if (!(sp == 0.0)) {
      std::printf(
          "FAIL grandeurs non bit-identiques entre rangs (spread=%.3e) : la reduction des tags "
          "(R4) ou le layout fin d'union diverge par rang\n",
          sp);
      ++fails;
    }
    if (fails == 0)
      std::printf(
          "OK test_amr_regrid_mpi_parity np=%d (regrid d'union : layout fin IDENTIQUE "
          "cross-rang, masse par bloc conservee ; CI diffe np=1/2/4)\n",
          np);
  } else {
    (void)sp;
  }
  comm_finalize();
  return fails ? 1 : 0;
}
