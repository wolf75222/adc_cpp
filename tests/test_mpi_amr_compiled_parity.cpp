// VALIDATION INTEGREE AmrSystem + MPI + GPU (deliverable C). Un SEUL run combine, pour la premiere
// fois, les trois axes qui n'avaient ete valides que SEPAREMENT sur GH200 (cf docs/GPU_RUNTIME_PORT.md,
// phases 5/6/9) :
//   - une HIERARCHIE AMR reelle (AmrSystem : grossier replique + niveau fin multi-patch suivi par
//     regrid Berger-Rigoutsos, reflux conservatif, Poisson grossier a chaque pas) ;
//   - un MODELE COMPILE branche par add_compiled_model(AmrSystem, ...) (CompositeModel connu a la
//     compilation, chemin amr_dsl_block.hpp, PR #45) ;
//   - une DISTRIBUTION MPI : les patchs fins sont repartis sur n_ranks() GPU (un par rang), halos
//     cross-rang via fill_boundary, reflux et masse reduits par all_reduce.
//
// Propriete verifiee : l'evolution de la hierarchie est INVARIANTE AU NOMBRE DE RANGS. Le grossier
// etant REPLIQUE (defaut AmrCouplerMP), density() et mass() sont des grandeurs GLOBALES identiques
// sur chaque rang ; le decoupage du niveau fin entre rangs (et donc le chemin MPI : halos distants,
// injection parallel_copy, reflux route vers la box parente distante) ne doit RIEN changer au
// resultat bit a bit. On le controle de deux manieres complementaires :
//   (1) CONSISTANCE CROSS-RANG dans le run : tous les rangs voient la MEME densite grossiere et la
//       MEME masse (diff max reduite sur les rangs == 0). Sans cela, un bug de halo/reflux distant
//       casserait silencieusement la replication.
//   (2) PARITE AU NB DE RANGS : on imprime un checksum de la densite + la masse ; le script de build
//       relance le MEME binaire en np=1/2/4 et DIFF les sorties. np=1 est l'oracle MONO-GPU ; np=2/4
//       doivent etre BIT-IDENTIQUES (dmax=0).
//
// Independant du backend : vert sous Kokkos Serial (CI, CPU) ET sous Kokkos Cuda (ROMEO GH200,
// multi-GPU). Sous Cuda, for_each_cell ne fence pas (async) : density()/mass() de l'AmrSystem font
// deja un device_fence() interne avant la lecture hote (read_coarse / amr_read_coarse), donc la
// lecture hote ici est sure. On insere malgre tout un Kokkos::fence() de ceinture avant les diffs.
#include <adc/physics/bricks/bricks.hpp>         // CompositeModel, GravityForce, GravityCoupling
#include <adc/physics/fluids/euler.hpp>          // Euler (transport compressible)
#include <adc/runtime/builders/amr_dsl_block.hpp>  // add_compiled_model(AmrSystem, ...)
#include <adc/runtime/amr_system.hpp>
#include <adc/parallel/comm.hpp>  // comm_init, my_rank, n_ranks, all_reduce_*

#include <cmath>
#include <cstdio>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace adc;
using Model = CompositeModel<Euler, GravityForce, GravityCoupling>;

// QUATRE bulles de densite lisses, periodiques, bien separees : chacune depasse le seuil de
// raffinement -> Berger-Rigoutsos produit PLUSIEURS patchs fins disjoints, que le regrid REPARTIT
// sur les rangs (round-robin DistributionMapping(nfine, n_ranks())). C'est ce qui distribue
// reellement le niveau fin sur plusieurs GPU (et non un seul patch central sur un seul rang).
static std::vector<double> four_bubbles(int n) {
  std::vector<double> rho(static_cast<std::size_t>(n) * n);
  const double cx[4] = {0.25, 0.75, 0.25, 0.75};
  const double cy[4] = {0.25, 0.25, 0.75, 0.75};
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n, y = (j + 0.5) / n;
      double r = 1.0;
      for (int b = 0; b < 4; ++b) {
        const double dx = x - cx[b], dy = y - cy[b];
        r += 0.5 * std::exp(-(dx * dx + dy * dy) / 0.004);
      }
      rho[static_cast<std::size_t>(j) * n + i] = r;
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
  const int n = 64;
  const std::vector<double> rho = four_bubbles(n);

  AmrSystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;
  cfg.regrid_every = 4;  // re-raffinement periodique : exerce le regrid distribue plusieurs fois

  // Modele euler_poisson COMPILE branche sur la hierarchie AMR (chemin de production add_compiled_model).
  AmrSystem sys(cfg);
  add_compiled_model(sys, "gas", Model{Euler{1.4}, GravityForce{}, GravityCoupling{-1.0, 1.0, 1.0}},
                     "minmod", "rusanov", "conservative", "explicit", /*gamma=*/1.4);
  sys.set_poisson("charge_density", "geometric_mg");
  sys.set_refinement(1.2);  // raffine la bulle (rho > 1.2 au coeur)
  sys.set_density("gas", rho);

  const double m0 = sys.mass();  // declenche le build paresseux (regrid initial distribue)
  const int np0 = sys.n_patches();

  // Plusieurs macro-pas AMR : chaque pas = Poisson grossier + injection vers les fins + transport
  // multi-niveaux + reflux conservatif ; tous les 4 pas, regrid Berger-Rigoutsos (redistribue les
  // patchs sur les rangs). C'est la totalite du chemin AMR + MPI exercee ensemble.
  const double dt = 1e-3;
  const int nsteps = 16;
  for (int s = 0; s < nsteps; ++s)
    sys.step(dt);

#if defined(ADC_HAS_KOKKOS)
  Kokkos::fence();  // ceinture avant la lecture hote (density()/mass() fencent deja en interne)
#endif
  const std::vector<double> dens = sys.density();  // grossier REPLIQUE : identique sur chaque rang
  const double mass = sys.mass();
  const int npf = sys.n_patches();

  // Checksum de la densite grossiere (somme + somme des carres + max) : signature bit-sensible du
  // champ final, comparable entre nombres de rangs par le script de build.
  double csum = 0, csumsq = 0, cmax = 0;
  for (double v : dens) {
    csum += v;
    csumsq += v * v;
    const double a = std::fabs(v);
    if (a > cmax)
      cmax = a;
  }

  // (1) CONSISTANCE CROSS-RANG : le grossier replique impose que chaque rang ait EXACTEMENT le meme
  // champ. On reduit l'ecart max entre les checksums locaux et ceux du rang 0 (max - min == 0 ssi
  // tous egaux). On compare via all_reduce_max/min des memes quantites.
  const double smax = all_reduce_max(csum), smin = -all_reduce_max(-csum);
  const double qmax = all_reduce_max(csumsq), qmin = -all_reduce_max(-csumsq);
  const double mmax = all_reduce_max(mass), mmin = -all_reduce_max(-mass);
  const double xmax = all_reduce_max(cmax), xmin = -all_reduce_max(-cmax);
  const double spread =
      std::fmax(std::fmax(smax - smin, qmax - qmin), std::fmax(mmax - mmin, xmax - xmin));

  int fails = 0;
  if (me == 0) {
    // Sortie machine-parsable (le script DIFF ces lignes entre np=1/2/4 ; np=1 = oracle mono-GPU).
    std::printf(
        "AMRMPI np=%d patches0=%d patchesF=%d | mass=%.17e | csum=%.17e csumsq=%.17e "
        "cmax=%.17e | crossrank_spread=%.3e\n",
        np, np0, npf, mass, csum, csumsq, cmax, spread);
#if defined(ADC_HAS_KOKKOS)
    const char* space = Kokkos::DefaultExecutionSpace::name();
#else
    const char* space = "Serial(host)";
#endif
    std::printf("AMRMPI exec=%s m0=%.17e (conservation: dm=%.3e)\n", space, m0,
                std::fabs(mass - m0));

    if (!(dens.size() == static_cast<std::size_t>(n) * n)) {
      std::printf("FAIL densite grossiere de mauvaise taille\n");
      ++fails;
    }
    if (!(cmax > 1e-6)) {
      std::printf("FAIL densite triviale (pas de signal)\n");
      ++fails;
    }
    // >= 2 patchs fins : sous np>=2 ils se repartissent sur plusieurs rangs/GPU (round-robin),
    // exercant le chemin fin DISTRIBUE (halos cross-rang, reflux route vers la box parente distante)
    // et pas seulement le grossier replique. Les 4 bulles produisent typiquement 4 patchs.
    if (!(npf >= 2)) {
      std::printf("FAIL < 2 patchs fins (niveau fin non distribuable)\n");
      ++fails;
    }
    // Le grossier replique DOIT etre bit-identique sur tous les rangs (spread exactement 0).
    if (!(spread == 0.0)) {
      std::printf("FAIL grossier non bit-identique entre rangs\n");
      ++fails;
    }
    if (fails == 0)
      std::printf(
          "OK test_mpi_amr_compiled_parity np=%d (AmrSystem+MPI+compile : grossier "
          "bit-identique cross-rang)\n",
          np);
  } else {
    // Les rangs non-0 valident aussi la consistance (spread doit etre 0 partout) mais ne FAILent que
    // via le rang 0 (sortie unique). On garde le code symetrique : aucune assertion divergente.
    (void)spread;
  }
  comm_finalize();
  return fails ? 1 : 0;
}
