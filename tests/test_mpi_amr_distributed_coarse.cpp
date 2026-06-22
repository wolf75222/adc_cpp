// STRONG-SCALING AMR : grossier REPARTI cable dans AmrSystem (deliverable C, perf full-device).
//
// test_mpi_amr_compiled_parity valide la hierarchie AMR + MPI + modele compile avec le grossier
// REPLIQUE (defaut) : le Poisson grossier et le transport grossier sont REDONDANTS sur chaque rang,
// donc le run NE SCALE PAS (cf docs/GPU_RUNTIME_PORT.md phase 10). Ce test exerce le MODE SCALABLE
// (AmrSystemConfig::distribute_coarse=true) : le niveau grossier devient MULTI-BOX (BoxArray::from_domain)
// REPARTI round-robin sur les rangs, le Poisson grossier (GeometricMG multi-box) et le transport
// grossier se DISTRIBUENT. C'est le chemin du strong-scaling AMR.
//
// Ce qu'on verifie (criteres d'HONNETETE du deliverable) :
//   (1) CORRECTION PHYSIQUE : le grossier reparti donne le MEME champ que le grossier replique a
//       l'arrondi pres. On construit DEUX AmrSystem dans le meme binaire (replique = oracle,
//       reparti) avec exactement la meme init et la meme sequence de pas, et on compare la densite
//       grossiere finale. La densite est reconstruite GLOBALEMENT (chaque rang n'a que ses tuiles ->
//       coupler_read_coarse all_reduce les boites disjointes), donc dens.size()==n*n sur chaque rang.
//   (2) MAX CROSS-RANG BIT-IDENTIQUE : cmax (reduction max, INSENSIBLE a l'ordre de sommation) doit
//       etre identique a tous les np. C'est le critere bit-exact que la doc exige pour le reparti
//       (les sommes additives, elles, dependent de l'ordre de reduction FMA quand le grossier est
//       genuinement decoupe -- documente pour #59 ; on ne l'exige donc PAS bit a bit ici).
//   (3) CONSERVATION : masse conservee a l'arrondi (reflux conservatif + all_reduce_sum).
//   (4) MG CONVERGE : phi reste fini et le champ non trivial (pas de divergence du multigrille
//       geometrique sur le grossier multi-box). Couvert par (1) : un MG diverge -> NaN -> echec.
//
// Independant du backend : Kokkos Serial (CI, CPU) et Cuda (GH200). Le script ROMEO relance le MEME
// binaire en np=1/2/4 et diff cmax (bit-identique attendu).
#include <adc/physics/bricks.hpp>         // CompositeModel, GravityForce, GravityCoupling
#include <adc/physics/euler.hpp>          // Euler
#include <adc/runtime/builders/amr_dsl_block.hpp>  // add_compiled_model(AmrSystem, ...)
#include <adc/runtime/amr_system.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace adc;
using Model = CompositeModel<Euler, GravityForce, GravityCoupling>;

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

// Construit un AmrSystem (4 bulles, euler_poisson compile), avance nsteps, rend la densite grossiere
// GLOBALE (n*n) + masse finale + m0. distribute => grossier multi-box reparti (sinon replique).
struct Result {
  std::vector<double> dens;
  double mass, m0;
  int npf;
};

static Result run(int n, int nsteps, double dt, bool distribute) {
  const std::vector<double> rho = four_bubbles(n);
  AmrSystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;
  cfg.regrid_every = 4;
  cfg.distribute_coarse = distribute;  // <-- le mode scalable cable dans AmrSystem
  // coarse_max_grid = 0 -> n/2 (decoupage 2x2, le moins agressif pour le MG geometrique).

  AmrSystem sys(cfg);
  add_compiled_model(sys, "gas", Model{Euler{1.4}, GravityForce{}, GravityCoupling{-1.0, 1.0, 1.0}},
                     "minmod", "rusanov", "conservative", "explicit", /*gamma=*/1.4);
  sys.set_poisson("charge_density", "geometric_mg");
  sys.set_refinement(1.2);
  sys.set_density("gas", rho);

  Result R;
  R.m0 = sys.mass();
  for (int s = 0; s < nsteps; ++s)
    sys.step(dt);
#if defined(ADC_HAS_KOKKOS)
  Kokkos::fence();
#endif
  R.dens = sys.density();
  R.mass = sys.mass();
  R.npf = sys.n_patches();
  return R;
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
  const int nsteps = 16;
  const double dt = 1e-3;

  const Result rep = run(n, nsteps, dt, /*distribute=*/false);  // oracle : grossier replique
  const Result dis = run(n, nsteps, dt, /*distribute=*/true);   // mode scalable : grossier reparti

  // (1) ecart REPARTI vs REPLIQUE sur la densite grossiere globale (n*n sur chaque rang).
  double dmax = 0;
  if (dis.dens.size() == rep.dens.size())
    for (std::size_t k = 0; k < dis.dens.size(); ++k)
      dmax = std::fmax(dmax, std::fabs(dis.dens[k] - rep.dens[k]));

  // checksums du champ reparti.
  double csum = 0, csumsq = 0, cmax = 0;
  for (double v : dis.dens) {
    csum += v;
    csumsq += v * v;
    const double a = std::fabs(v);
    if (a > cmax)
      cmax = a;
  }
  // (2) cmax cross-rang : max insensible a l'ordre -> doit etre identique sur tous les rangs.
  const double xmax = all_reduce_max(cmax), xmin = -all_reduce_max(-cmax);
  const double cmax_spread = xmax - xmin;
  // dmax reduit sur les rangs (chaque rang a le meme champ global reconstruit, mais on est defensif).
  const double dmax_g = all_reduce_max(dmax);

  int fails = 0;
  if (me == 0) {
    std::printf(
        "AMRDIST np=%d distribute_npf=%d replicated_npf=%d | cmax=%.17e | "
        "dist_vs_repl_dmax=%.3e | cmax_crossrank_spread=%.3e\n",
        np, dis.npf, rep.npf, cmax, dmax_g, cmax_spread);
#if defined(ADC_HAS_KOKKOS)
    const char* space = Kokkos::DefaultExecutionSpace::name();
#else
    const char* space = "Serial(host)";
#endif
    std::printf(
        "AMRDIST exec=%s | conservation: dm_dist=%.3e dm_repl=%.3e | csum=%.17e csumsq=%.17e\n",
        space, std::fabs(dis.mass - dis.m0), std::fabs(rep.mass - rep.m0), csum, csumsq);

    if (!(dis.dens.size() == static_cast<std::size_t>(n) * n)) {
      std::printf("FAIL densite repartie de mauvaise taille\n");
      ++fails;
    }
    if (!(cmax > 1e-6)) {
      std::printf("FAIL densite repartie triviale\n");
      ++fails;
    }
    if (!std::isfinite(cmax) || !std::isfinite(csum)) {
      std::printf("FAIL champ non fini (MG diverge ?)\n");
      ++fails;
    }
    // (4) MG converge => champ fini ET proche du replique : le grossier reparti doit retrouver le
    // meme physique a l'arrondi pres (la difference vient de l'ordre de reduction du Poisson +
    // transport multi-box, pas d'un schema different). Seuil large mais ferme : un MG qui diverge
    // ou un transport casse exploserait bien au-dela.
    if (!(dmax_g < 1e-9)) {
      std::printf("FAIL reparti != replique au-dela de l'arrondi (dmax=%.3e)\n", dmax_g);
      ++fails;
    }
    // (3) conservation des deux modes.
    if (!(std::fabs(dis.mass - dis.m0) < 1e-10)) {
      std::printf("FAIL conservation grossier reparti (dm=%.3e)\n", std::fabs(dis.mass - dis.m0));
      ++fails;
    }
    // (2) cmax bit-identique cross-rang.
    if (!(cmax_spread == 0.0)) {
      std::printf("FAIL cmax non bit-identique entre rangs (spread=%.3e)\n", cmax_spread);
      ++fails;
    }
    if (fails == 0)
      std::printf(
          "OK test_mpi_amr_distributed_coarse np=%d (grossier reparti == replique a "
          "l'arrondi, cmax bit-identique cross-rang, masse conservee)\n",
          np);
  }
  comm_finalize();
  return fails ? 1 : 0;
}
