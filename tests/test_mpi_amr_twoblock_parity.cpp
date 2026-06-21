// PARITE MPI du capstone AMR MULTI-BLOCS (PR1). Pendant multi-blocs de test_mpi_amr_compiled_parity :
// DEUX blocs EXPLICITES a schemas DIFFERENTS co-localises sur UNE hierarchie AMR PARTAGEE (Poisson de
// systeme a second membre SOMME q0 n0 + q1 n1), DISTRIBUES sur n_ranks(). Propriete verifiee :
//   (1) CONSISTANCE CROSS-RANG : le grossier etant REPLIQUE, la masse de CHAQUE bloc, la densite de
//       chaque bloc et le potentiel de systeme sont des grandeurs GLOBALES identiques sur tous les
//       rangs (spread max reduit == 0). Un bug de halo / Poisson somme / aux distant le casserait.
//   (2) PARITE AU NB DE RANGS : on imprime des checksums (par bloc + potentiel) ; le script de build
//       relance le MEME binaire en np=1/2/4 et DIFF (np=1 = oracle ; np=2/4 BIT-IDENTIQUES).
//
// Hierarchie FIGEE (regrid_every=0) : multi-blocs PR1 n'a pas de regrid (AmrRuntime ; le regrid
// d'union des tags est une PR ulterieure). On exerce neanmoins le grossier replique + le patch fin
// central multi-patch + le Poisson somme co-localise distribues. Independant du backend (Kokkos
// Serial CI, Kokkos Cuda GH200).
#include <adc/runtime/amr_system.hpp>
#include <adc/runtime/model_spec.hpp>
#include <adc/parallel/comm.hpp>  // comm_init, my_rank, n_ranks, all_reduce_*

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

// creneau lisse a moyenne (offset) nulle, n*n row-major (charge totale solvable en periodique).
static std::vector<double> bump(int n, double amp) {
  std::vector<double> r(static_cast<std::size_t>(n) * n, 1.0);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n, y = (j + 0.5) / n;
      const double dx = x - 0.5, dy = y - 0.5;
      r[static_cast<std::size_t>(j) * n + i] = 1.0 + amp * std::exp(-(dx * dx + dy * dy) / 0.01);
    }
  // retire l'offset moyen -> Sum q n a moyenne nulle (Poisson periodique solvable).
  double mean = 0;
  for (double v : r)
    mean += v;
  mean /= static_cast<double>(r.size());
  for (double& v : r)
    v += (1.0 - mean);
  return r;
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
  const std::vector<double> rho0 = bump(n, 0.40);
  const std::vector<double> rho1 = bump(n, 0.20);

  AmrSystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;
  cfg.regrid_every = 0;  // multi-blocs PR1 : hierarchie FIGEE

  AmrSystem sys(cfg);
  sys.add_block("ions", exb_charge(q0, B0), "none", "rusanov", "conservative", "explicit", 1);
  sys.add_block("electrons", exb_charge(q1, B0), "minmod", "rusanov", "conservative", "explicit",
                1);  // SCHEMA DIFFERENT
  sys.set_poisson("charge_density", "geometric_mg", "periodic");
  sys.set_density("ions", rho0);
  sys.set_density("electrons", rho1);

  const double m0i = sys.mass("ions");  // declenche le build paresseux
  const double m0e = sys.mass("electrons");

  const double dt = 1e-3;
  for (int s = 0; s < 16; ++s)
    sys.step(dt);

#if defined(ADC_HAS_KOKKOS)
  Kokkos::fence();
#endif
  const std::vector<double> di = sys.density("ions");
  const std::vector<double> de = sys.density("electrons");
  const std::vector<double> phi = sys.potential();
  const double mi = sys.mass("ions"), mass_e = sys.mass("electrons");

  auto checksum = [](const std::vector<double>& v) {
    double s = 0;
    for (double x : v)
      s += x * x;
    return s;
  };
  const double ci = checksum(di), ce = checksum(de), cp = checksum(phi);

  // (1) CONSISTANCE CROSS-RANG : grossier replique -> chaque grandeur globale identique sur tout
  // rang. spread = max(max - min) sur les checksums + masses ; == 0 ssi bit-identique cross-rang.
  auto spread = [](double x) { return all_reduce_max(x) - (-all_reduce_max(-x)); };
  const double sp = std::fmax(std::fmax(spread(ci), spread(ce)),
                              std::fmax(spread(cp), std::fmax(spread(mi), spread(mass_e))));

  int fails = 0;
  if (me == 0) {
    std::printf(
        "AMRMB np=%d | mass_ions=%.17e mass_elec=%.17e | csum_ions=%.17e csum_elec=%.17e "
        "csum_phi=%.17e | crossrank_spread=%.3e\n",
        np, mi, mass_e, ci, ce, cp, sp);
    std::printf("AMRMB conservation: dm_ions=%.3e dm_elec=%.3e\n", std::fabs(mi - m0i),
                std::fabs(mass_e - m0e));
    if (!(di.size() == static_cast<std::size_t>(n) * n)) {
      std::printf("FAIL taille densite\n");
      ++fails;
    }
    if (!(cp > 1e-12)) {
      std::printf("FAIL potentiel trivial (Poisson somme inactif)\n");
      ++fails;
    }
    // masse de CHAQUE bloc conservee (transport conservatif periodique, par bloc).
    if (!(std::fabs(mi - m0i) < 1e-9)) {
      std::printf("FAIL masse ions non conservee\n");
      ++fails;
    }
    if (!(std::fabs(mass_e - m0e) < 1e-9)) {
      std::printf("FAIL masse electrons non conservee\n");
      ++fails;
    }
    // grossier replique : tout bit-identique cross-rang (spread exactement 0).
    if (!(sp == 0.0)) {
      std::printf("FAIL grandeurs non bit-identiques entre rangs\n");
      ++fails;
    }
    if (fails == 0)
      std::printf(
          "OK test_mpi_amr_twoblock_parity np=%d (multi-blocs AMR : Poisson somme "
          "co-localise, masse par bloc, bit-identique cross-rang)\n",
          np);
  } else {
    (void)sp;
  }
  comm_finalize();
  return fails ? 1 : 0;
}
