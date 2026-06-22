// Run diocotron HyQMOM-15 sur AmrSystem MULTI-BOITE, decomposition de domaine MPI reelle (halos
// inter-GPU). ADC-320, suite directe de ADC-181.
//
// ADC-181 a valide la branche multi-GPU de hyqmom15 par la decomposition MPI de System en
// topologie MONO-BOITE round-robin (diocotron_mpi.sbatch) : toute la boite reste sur le rang 0, les
// autres rangs ont local_size()==0 et n'ajoutent que des zeros a l'all-reduce. Cela valide le chemin
// collectif/MPI + la parite de masse mais N'EXERCE PAS l'echange de halos inter-GPU pour hyqmom15.
//
// Ce driver trace la validation DIRECTE : un vrai run hyqmom15 DOMAINE-DECOMPOSE ou les halos
// transportent l'etat des 15 moments entre GPU. On cable le composite hyqmom15 (briques emises par
// la DSL Hyqmom15Hyp/Src/Ell + Poisson geometric_mg) sur AmrSystem avec distribute_coarse=true : le
// niveau grossier devient un BoxArray MULTI-BOITE reparti round-robin sur n_ranks() GPU. Le transport
// grossier appelle alors fill_boundary sur ce MultiFab multi-boite (amr_subcycling.hpp), qui a np>1
// est un VRAI echange MPI cross-rang des 15 composantes conservees sur tampons SharedHostPinnedSpace
// (le fix CUDA-IPC #254). Le niveau fin (regrid + reflux conservatif) se repartit AUSSI round-robin.
// distribute_coarse devient effectif depuis ADC-319 (#140 : les loaders DSL compilent avec ADC_HAS_MPI).
//
// Topologie / preuve de repartition (ADC-319) : coarse_local_boxes() < coarse_total_boxes() par rang
// a np>1 prouve que le grossier s'est genuinement decoupe entre rangs (a np=1 ou en mode replique,
// local == total). Le driver AVORTE si la repartition n'a pas eu lieu a np>1.
//
// On rejoue, dans le MEME binaire, DEUX modes d'ownership depuis le MEME etat initial :
//   - REPARTI  (distribute_coarse=true)  : le grossier multi-boite reparti, halos inter-GPU reels ;
//   - REPLIQUE (distribute_coarse=false) : oracle, grossier mono-boite replique sur chaque rang.
// Critere DIRECT de correction des halos (a np FIXE) : la densite grossiere REPARTIE == REPLIQUEE
// BIT-POUR-BIT (dist_vs_repl_dmax == 0, max compris). A np fixe les deux modes partagent la MEME
// trajectoire (meme ordre de reduction, donc memes tags et memes patchs), si bien que l'echange
// multi-boite inter-GPU des 15 moments est prouve transparent. Conservation : masse conservee par
// run, ET masse globale np=2/4 vs np=1 au dernier ulp (reassociation FMA cross-rang, REELLE ici
// contrairement au round-robin mono-boite de ADC-181 qui la masquait). EN REVANCHE cmax (pic
// ponctuel) et le nombre de patchs DIVERGENT entre np : le diocotron est INSTABLE et le regrid AMR
// reagit a la reassociation ulp du Poisson cross-rang (maillages differents -> pic resolu
// differemment) -- l'invariant CONSERVE (masse) n'en est pas affecte. On ne peut donc PAS exiger un
// cmax bit-identique entre np sur un ecoulement instable raffine (le "bit-exact sur le max" de
// ADC-181 portait sur un champ B_z STATIQUE) ; le bit-exact se lit sur dist_vs_repl a np fixe. dt
// FIXE (comme test_mpi_amr_distributed_coarse) : la sequence de pas est identique a tous les np et
// dans les deux modes, seule la reassociation spatiale du Poisson (amplifiee par l'instabilite + le
// regrid) distingue les trajectoires entre np.
//
// L'etat initial (15*n*n doubles, comp-major c*n*n + j*n + i) est LU en binaire (ic_<n>.raw, calcule
// par le python valide diocotron_state) et seme par set_conservative_state -- meme artefact que le
// driver System (make_brick_and_ic.py). Seul le rang 0 imprime ; les accesseurs grossiers
// mass()/density()/potential() de l'AmrSystem all-reduce en interne (chaque rang detient le champ
// global reconstruit), donc aucun deadlock.

#include <adc/parallel/comm.hpp>
#include <adc/physics/composition/composite.hpp>
#include <adc/runtime/builders/compiled/amr_dsl_block.hpp>  // add_compiled_model(AmrSystem&, ...)
#include <adc/runtime/amr_system.hpp>

#include "hyqmom15_brick.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

static double arg_d(int argc, char** argv, const char* key, double dflt) {
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], key)) return std::atof(argv[i + 1]);
  return dflt;
}
static std::string arg_s(int argc, char** argv, const char* key, const char* dflt) {
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], key)) return argv[i + 1];
  return dflt;
}

// Resultat d'un run pour un mode d'ownership donne. Champs grossiers GLOBAUX (n*n, reconstruits par
// les accesseurs all-reduce de l'AmrSystem), invariants suivis pour le gate.
struct Result {
  std::vector<double> dens;  // densite grossiere M00 reconstruite (n*n)
  double mass = 0, m0 = 0;   // masse finale / initiale (composante 0 sur le grossier)
  double csum = 0, csumsq = 0, cmax = 0;  // checksums du champ grossier
  double max_abs_phi = 0;    // max|phi| grossier (MG fini ?)
  int npatches = 0;          // nombre de patchs fins (la decomposition fine)
  int coarse_local = 0, coarse_total = 0;  // boites grossieres locales / totales (preuve ADC-319)
  bool finite = true;
};

// Un run complet hyqmom15 sur AmrSystem pour un mode d'ownership. nsteps pas a dt FIXE depuis l'IC
// commun U0. Renvoie les invariants grossiers GLOBAUX. distribute => grossier multi-boite reparti.
template <class Model>
static Result run_mode(const std::vector<double>& U0, int n, bool distribute, int nsteps,
                       double dt, int regrid_every, double refine_thr, int coarse_max_grid) {
  adc::AmrSystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;
  cfg.regrid_every = regrid_every;
  cfg.distribute_coarse = distribute;  // reparti => grossier multi-box reparti (halos inter-GPU)
  // 2x2 (coarse_max_grid = n/2) : le decoupage le moins agressif pour le MG geometrique multi-box
  // (cf. AmrSystemConfig + amrmpi_integrated : 2x2 converge en autant de cycles que le mono-box).
  cfg.coarse_max_grid = distribute ? (coarse_max_grid > 0 ? coarse_max_grid : n / 2) : 0;

  adc::AmrSystem sys(cfg);
  // composite emis par la DSL (flux + vitesses exactes par real_eig_minmax / source Lorentz / rhs de
  // Poisson), branche par le SEAM DE COMPILATION add_compiled_model -- bloc UNIQUE => chemin
  // mono-bloc AmrCouplerMP<Model> (jamais la facade AmrSystemCoupler). Memes schemas que le driver
  // System (diocotron_gpu.cpp) : limiteur none, riemann hll exact, recon conservative, explicite.
  adc::add_compiled_model(sys, "mom", Model{}, "none", "hll", "conservative", "explicit");
  sys.set_poisson("charge_density", "geometric_mg");
  sys.set_refinement(refine_thr);  // raffine le grossier ou M00 > seuil -> patchs fins distribues
  sys.set_conservative_state("mom", U0);  // sema les 15 moments (prolonge a la grille fine au build)

  Result R;
  R.m0 = sys.mass();  // build paresseux (regrid initial : decoupe + repartit le grossier ET le fin)
  for (int s = 0; s < nsteps; ++s) sys.step(dt);
#if defined(ADC_HAS_KOKKOS)
  Kokkos::fence();  // capture le travail device async avant les lectures hote
#endif

  R.dens = sys.density();      // grossier M00 GLOBAL (reparti : all_reduce_sum interne)
  R.mass = sys.mass();
  R.npatches = sys.n_patches();
  R.coarse_local = sys.coarse_local_boxes();  // ADC-319 : boites grossieres OWNED par ce rang
  R.coarse_total = sys.coarse_total_boxes();  // total des boites grossieres (identique a tous les rangs)
  const std::vector<double> phi = sys.potential();
  for (double v : R.dens) {
    if (!std::isfinite(v)) R.finite = false;
    R.csum += v;
    R.csumsq += v * v;
    const double a = std::fabs(v);
    if (a > R.cmax) R.cmax = a;
  }
  for (double v : phi) {
    if (!std::isfinite(v)) R.finite = false;
    R.max_abs_phi = std::max(R.max_abs_phi, std::fabs(v));
  }
  return R;
}

// Imprime (rang 0) la ligne machine-parsable consommee par diocotron_amr_mpi.sbatch + les
// diagnostics de repartition. cmax cross-rang : max insensible a l'ordre -> identique a tous les rangs.
static void emit(const char* tag, int np, int n, const Result& R) {
  const double xmax = adc::all_reduce_max(R.cmax), xmin = -adc::all_reduce_max(-R.cmax);
  const double cmax_spread = xmax - xmin;  // 0 attendu : tous les rangs voient le meme champ global
  if (adc::my_rank() != 0) return;
  std::printf("HYQMOMAMR mode=%s np=%d n=%d | mass=%.17e massdrift=%.3e | "
              "csum=%.17e csumsq=%.17e cmax=%.17e | cmax_crossrank_spread=%.3e | "
              "patches=%d clocal=%d ctotal=%d maxabs_phi=%.6e\n",
              tag, np, n, R.mass, R.m0 != 0 ? std::fabs(R.mass - R.m0) / std::fabs(R.m0) : 0.0,
              R.csum, R.csumsq, R.cmax, cmax_spread, R.npatches, R.coarse_local, R.coarse_total,
              R.max_abs_phi);
  std::fflush(stdout);
}

int main(int argc, char** argv) {
  adc::comm_init(&argc, &argv);  // MPI_Init si ADC_HAS_MPI ; no-op en serie
#if defined(ADC_HAS_KOKKOS)
  Kokkos::initialize(argc, argv);
#endif
  int rc = 0;
  {
    const int me = adc::my_rank(), np = adc::n_ranks();
    const int n = static_cast<int>(arg_d(argc, argv, "--n", 128));
    const int nsteps = static_cast<int>(arg_d(argc, argv, "--steps", 80));
    const double dt = arg_d(argc, argv, "--dt", 4e-4);  // FIXE : sequence de pas identique a tous les np
    const int regrid_every = static_cast<int>(arg_d(argc, argv, "--regrid-every", 8));
    const int coarse_max_grid = static_cast<int>(arg_d(argc, argv, "--coarse-max-grid", 0));
    const double refine_arg = arg_d(argc, argv, "--refine", -1.0);  // <0 => derive du M00 de l'IC
    const int do_compare = static_cast<int>(arg_d(argc, argv, "--compare", 1));  // run replique oracle
    const std::string ic = arg_s(argc, argv, "--ic", "ic.raw");
    // Gardes identiques sur tous les rangs (aucun collectif encore) : sortie commune sans deadlock.
    if (n < 2 || nsteps < 1 || dt <= 0) {
      if (me == 0) std::fprintf(stderr, "[amr] --n>=2 --steps>=1 --dt>0 requis\n");
      rc = 1;
    }

    std::vector<double> U0;
    if (rc == 0) {
      U0.assign(static_cast<std::size_t>(15) * n * n, 0.0);
      std::ifstream f(ic, std::ios::binary);
      if (!f) {
        if (me == 0) std::fprintf(stderr, "IC introuvable : %s\n", ic.c_str());
        rc = 2;
      } else {
        f.read(reinterpret_cast<char*>(U0.data()),
               static_cast<std::streamsize>(U0.size() * sizeof(double)));
        if (!f) {
          if (me == 0) std::fprintf(stderr, "IC tronquee : %s\n", ic.c_str());
          rc = 2;
        }
      }
    }

    if (rc == 0) {
      // Seuil de raffinement : par defaut derive de la composante 0 (M00) de l'IC -- moyenne +
      // 0.4*(max-moyenne) tague la couronne dense du diocotron, robuste a la discretisation. La
      // valeur est IDENTIQUE sur tous les rangs (chacun lit l'IC complete) -> meme hierarchie partout.
      double refine_thr = refine_arg;
      if (refine_thr < 0) {
        double s = 0, mx = 0;
        for (std::size_t k = 0; k < static_cast<std::size_t>(n) * n; ++k) {
          s += U0[k];
          mx = std::max(mx, U0[k]);
        }
        const double mean = s / (static_cast<double>(n) * n);
        refine_thr = mean + 0.4 * (mx - mean);
      }
      if (me == 0) {
        std::printf("[amr] n=%d np=%d steps=%d dt=%.3e regrid_every=%d coarse_max_grid=%d "
                    "refine=%.6e ic=%s compare=%d\n",
                    n, np, nsteps, dt, regrid_every,
                    coarse_max_grid > 0 ? coarse_max_grid : n / 2, refine_thr, ic.c_str(), do_compare);
        std::fflush(stdout);
      }

      // Une brique elliptique PAR n (rho_background cave, depend de la discretisation) ; Hyp/Src
      // partagees. Le composite est selectionne ici, run_mode est template sur le Model.
      Result rep{}, dis{};
      bool ran = true;
      if (n == 128) {
        using Model = adc::CompositeModel<adc_generated::Hyqmom15Hyp, adc_generated::Hyqmom15Src,
                                          adc_generated::Hyqmom15Ell128>;
        dis = run_mode<Model>(U0, n, /*distribute=*/true, nsteps, dt, regrid_every, refine_thr,
                              coarse_max_grid);
        if (do_compare)
          rep = run_mode<Model>(U0, n, /*distribute=*/false, nsteps, dt, regrid_every, refine_thr,
                                coarse_max_grid);
      } else if (n == 256) {
        using Model = adc::CompositeModel<adc_generated::Hyqmom15Hyp, adc_generated::Hyqmom15Src,
                                          adc_generated::Hyqmom15Ell256>;
        dis = run_mode<Model>(U0, n, /*distribute=*/true, nsteps, dt, regrid_every, refine_thr,
                              coarse_max_grid);
        if (do_compare)
          rep = run_mode<Model>(U0, n, /*distribute=*/false, nsteps, dt, regrid_every, refine_thr,
                                coarse_max_grid);
      } else {
        if (me == 0) std::fprintf(stderr, "n=%d sans brique elliptique emise (128|256)\n", n);
        ran = false;
        rc = 2;
      }

      if (ran) {
        emit("reparti", np, n, dis);
        if (do_compare) emit("replique", np, n, rep);

        // Ecart REPARTI vs REPLIQUE sur la densite grossiere globale (n*n sur chaque rang). Un MG qui
        // diverge ou un transport casse exploserait bien au-dela du seuil d'arrondi.
        double dmax = 0;
        if (do_compare && dis.dens.size() == rep.dens.size())
          for (std::size_t k = 0; k < dis.dens.size(); ++k)
            dmax = std::fmax(dmax, std::fabs(dis.dens[k] - rep.dens[k]));
        const double dmax_g = adc::all_reduce_max(dmax);

        if (me == 0) {
          if (do_compare)
            std::printf("HYQMOMAMR dist_vs_repl_dmax=%.3e (reparti vs replique, grossier global)\n",
                        dmax_g);
          // Preuve de repartition (ADC-319) : a np>1 le grossier reparti DOIT avoir local < total.
          const bool distributed = (np == 1) || (dis.coarse_local < dis.coarse_total);
          std::printf("HYQMOMAMR distributed_coarse=%s (np=%d clocal=%d ctotal=%d)\n",
                      distributed ? "yes" : "NO", np, dis.coarse_local, dis.coarse_total);
          int fails = 0;
          if (!dis.finite || (do_compare && !rep.finite)) {
            std::printf("FAIL champ non fini (MG diverge / etat non realisable ?)\n");
            ++fails;
          }
          if (!(dis.cmax > 1e-6)) { std::printf("FAIL densite repartie triviale\n"); ++fails; }
          if (!distributed) {
            std::printf("FAIL grossier NON reparti a np>1 (clocal>=ctotal : ADC-319 inactif ?)\n");
            ++fails;
          }
          if (!(dis.npatches >= 1)) { std::printf("FAIL pas de patch fin (refine inactif ?)\n"); ++fails; }
          if (do_compare && !(dmax_g < 1e-9)) {
            std::printf("FAIL reparti != replique au-dela de l'arrondi (dmax=%.3e)\n", dmax_g);
            ++fails;
          }
          if (fails == 0)
            std::printf("HYQMOMAMR_RUN_OK np=%d (grossier reparti : halos inter-GPU 15 moments)\n", np);
          else
            rc = 3;
        }
      }
    }
  }
#if defined(ADC_HAS_KOKKOS)
  Kokkos::finalize();
#endif
  adc::comm_finalize();
  return rc;
}
