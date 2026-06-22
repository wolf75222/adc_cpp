// Run diocotron HyQMOM-15 sur GPU (Kokkos-Cuda, noeud armgpu H100) -- campagne + validation
// device de dense_eig (prepare ADC-181).
//
// La brique modele est EMISE par la DSL (emit_cpp_brick, fichier hyqmom15_brick.hpp genere
// cote Mac depuis le modele VALIDE : flux + vitesses exactes par jacobien autodiff +
// adc::real_eig_minmax + sources electriques) et branchee par le SEAM DE COMPILATION
// adc::add_compiled_model - le meme chemin natif que add_block (assemble_rhs device, halos),
// zero marshaling. L'etat initial est LU en binaire (ic.raw : 15*n*n doubles row-major
// (comp, y, x)) - calcule par le python valide (diocotron_state), jamais re-porte.
//
// Sorties dans --out : snap_<k>.raw (15*n*n + n*n phi, doubles), growth.csv (t, dt, masse,
// |a_l| l=2..6 sur l'anneau), un par-pas leger. Pas de checkpoint HDF5 cote C++ : la
// trajectoire est courte (walltime short) et les snapshots suffisent pour reprendre une
// analyse ; la reprise bit-stable reste le chemin python (CPU).
//
// Multi-GPU + MPI (optionnel, -DADC_VALIDATION_MPI=ON, cf. diocotron_mpi.sbatch) : le MEME
// driver tourne sous srun -n N. comm_init/comm_finalize ouvrent/ferment MPI ; les lectures de
// diagnostic et de snapshot passent par sys.state_global / sys.potential_global (all-reduce
// collectif, system.cpp) et le pas sys.step_cfl / sys.solve_fields est collectif, donc chaque
// rang detient le champ complet et seul le rang 0 ecrit fichiers + stdout. Sans ADC_HAS_MPI les
// stubs de comm.hpp donnent my_rank()=0 / n_ranks()=1 : le chemin serie (parity181) est inchange
// au bit pres. Topologie = System mono-boite round-robin (boite 0 au rang 0, autres rangs
// local_size()==0 -> contribuent 0 a l'all-reduce), identique a la topologie CPU de run_mpi.py.

#include <adc/parallel/comm.hpp>
#include <adc/physics/composition/composite.hpp>
#include <adc/runtime/builders/compiled/dsl_block.hpp>
#include <adc/runtime/system.hpp>

#include "hyqmom15_brick.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

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

int main(int argc, char** argv) {
  adc::comm_init(&argc, &argv);  // MPI_Init si ADC_HAS_MPI et pas deja initialise ; no-op en serie
  const int me = adc::my_rank(), np = adc::n_ranks();
  const int n = static_cast<int>(arg_d(argc, argv, "--n", 128));
  const double tend = arg_d(argc, argv, "--tend", 4.0);
  const double cfl = arg_d(argc, argv, "--cfl", 0.4);
  const int snap_every = static_cast<int>(arg_d(argc, argv, "--snap-every", 200));
  const int diag_every = static_cast<int>(arg_d(argc, argv, "--diag-every", 5));
  const double max_steps = arg_d(argc, argv, "--max-steps", 1e9);
  const std::string ic = arg_s(argc, argv, "--ic", "ic.raw");
  const std::string out = arg_s(argc, argv, "--out", "out_gpu");
  // rho_background / debye / omega_p sont CAVES dans les briques emises (make_brick_and_ic.py).
  // Tous les rangs parsent les MEMES arguments : cette garde est identique partout, ils sortent
  // ensemble (aucun collectif n'a encore eu lieu) -- pas de deadlock.
  if (n < 1 || snap_every < 1 || diag_every < 1) {
    if (me == 0) std::fprintf(stderr, "[gpu] --n/--snap-every/--diag-every doivent etre >= 1\n");
    adc::comm_finalize();
    return 1;
  }

  // mkdir : seul le rang 0 ecrit dans --out (les autres rangs n'ont pas de boite a dumper).
  if (me == 0 && std::system(("mkdir -p " + out).c_str()) != 0) {
    std::fprintf(stderr, "[gpu] mkdir -p %s a echoue\n", out.c_str());
    adc::comm_finalize();
    return 1;
  }

  // --- etat initial : 15*n*n doubles (comp, y, x), produit par make_ic.py (python valide) ---
  // Chaque rang lit l'IC complete (echec identique sur tous les rangs -- meme fichier -- donc
  // sortie collective sans deadlock) ; le round-robin de System place la boite 0 sur le rang 0.
  std::vector<double> U0(static_cast<std::size_t>(15) * n * n);
  {
    std::ifstream f(ic, std::ios::binary);
    if (!f) {
      if (me == 0) std::fprintf(stderr, "IC introuvable : %s\n", ic.c_str());
      adc::comm_finalize();
      return 2;
    }
    f.read(reinterpret_cast<char*>(U0.data()),
           static_cast<std::streamsize>(U0.size() * sizeof(double)));
    if (!f) {
      if (me == 0) std::fprintf(stderr, "IC tronquee : %s\n", ic.c_str());
      adc::comm_finalize();
      return 2;
    }
  }

  adc::SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;
  adc::System sys(cfg);

  // briques emises par la DSL (flux + vitesses exactes / source Lorentz / rhs de Poisson),
  // assemblees par CompositeModel et branchees par le seam de COMPILATION (chemin natif
  // complet : assemble_rhs device, halos) - le pattern des tests test_dsl_{brick,source,
  // elliptic,compose} du coeur.
  // une brique elliptique PAR n (rho_background cave, depend de la discretisation du
  // scenario) ; Hyp/Src partagees.
  if (n == 128) {
    using Model = adc::CompositeModel<adc_generated::Hyqmom15Hyp, adc_generated::Hyqmom15Src,
                                      adc_generated::Hyqmom15Ell128>;
    adc::add_compiled_model(sys, "mom", Model{}, "none", "hll", "conservative", "explicit");
  } else if (n == 256) {
    using Model = adc::CompositeModel<adc_generated::Hyqmom15Hyp, adc_generated::Hyqmom15Src,
                                      adc_generated::Hyqmom15Ell256>;
    adc::add_compiled_model(sys, "mom", Model{}, "none", "hll", "conservative", "explicit");
  } else {
    if (me == 0) std::fprintf(stderr, "n=%d sans brique elliptique emise (128|256)\n", n);
    adc::comm_finalize();
    return 2;
  }
  sys.set_poisson("charge_density", "geometric_mg");

  sys.set_state("mom", U0);
  sys.solve_fields();

  // Seul le rang 0 ouvre/ecrit growth.csv ; les autres rangs gardent gf=nullptr et n'ecrivent
  // jamais (ils n'ont pas de boite et ne doivent pas courir en ecriture sur le fichier).
  std::FILE* gf = nullptr;
  if (me == 0) {
    gf = std::fopen((out + "/growth.csv").c_str(), "w");
    if (!gf) {
      std::fprintf(stderr, "[gpu] impossible d'ouvrir %s/growth.csv\n", out.c_str());
      adc::comm_finalize();
      return 1;
    }
    std::fprintf(gf, "step,t,dt,mass,a2,a3,a4,a5,a6\n");
  }

  double t = 0.0;
  long k = 0;
  double mass0 = 0.0;
  for (std::size_t i = 0; i < static_cast<std::size_t>(n) * n; ++i) mass0 += U0[i];

  // Invariants suivis pour la ligne finale machine-parsable (gate du diocotron_mpi.sbatch).
  double mass = mass0;        // masse globale du dernier diagnostic (sinon mass0)
  double max_abs_phi = 0.0;   // max|phi| du dernier snapshot (informatif)
  double last_a4 = 0.0;       // |a_4|/ringsum du dernier diagnostic (mode physique l=4)
  double dt0 = 0.0;
  while (t < tend && k < static_cast<long>(max_steps)) {
    const double dt = sys.step_cfl(cfl);
    t += dt;
    ++k;
    if (k == 1) dt0 = dt;
    // GARDE dt : sans relaxation par pas (flagrelax=1 du MATLAB), les vitesses exactes
    // explosent quand l'etat approche le bord de realisabilite -> dt s'effondre et le run
    // gele en marquant le temps (observe : n=256 fige a t=0.089). On sort PROPREMENT avec
    // un diagnostic plutot que de bruler le walltime (la projection compilee = ADC-177).
    // En MPI, dt vient de step_cfl (deja reduit GLOBAL), donc la garde est identique sur tous
    // les rangs ; on vote tout de meme via all_reduce_max pour ne JAMAIS sortir un seul rang
    // de la boucle (les collectifs restants -- state_global -- bloqueraient les autres rangs).
    int collapse = (dt < 1e-4 * dt0) ? 1 : 0;
    collapse = static_cast<int>(adc::all_reduce_max(static_cast<double>(collapse)));
    if (collapse) {
      if (me == 0)
        std::fprintf(stderr,
                     "[DT_COLLAPSE] pas %ld t=%.6f : dt=%.3e < 1e-4*dt0 (%.3e) -- etat au "
                     "bord de realisabilite, projection requise (ADC-177). Sortie propre.\n",
                     k, t, dt, dt0);
      break;
    }

    if (k % diag_every == 0 || t >= tend) {
      // state_global = all-reduce collectif : tous les rangs detiennent le MEME champ U, donc
      // mass / a[] / finite sont identiques partout (les rangs sortent ensemble en cas de NaN).
      const std::vector<double> U = sys.state_global("mom");
      double dmass = 0.0;
      std::complex<double> a[5] = {};
      double ringsum = 0.0;
      bool finite = true;
      for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
          const double m00 = U[static_cast<std::size_t>(j) * n + i];
          if (!std::isfinite(m00)) finite = false;
          dmass += m00;
          const double x = -0.5 + (i + 0.5) / n, y = -0.5 + (j + 0.5) / n;
          const double r = std::sqrt(x * x + y * y);
          if (r > 0.30 && r < 0.45) {
            const double th = std::atan2(y, x);
            ringsum += m00;
            for (int l = 0; l < 5; ++l)
              a[l] += m00 * std::exp(std::complex<double>(0.0, -(l + 2.0) * th));
          }
        }
      }
      if (!finite) {
        if (me == 0) std::fprintf(stderr, "[FATAL] M00 non fini au pas %ld\n", k);
        if (gf) std::fclose(gf);
        adc::comm_finalize();
        return 3;
      }
      mass = dmass;
      last_a4 = std::abs(a[2]) / ringsum;  // a[2] = mode l=4 (l index 0..4 -> l=2..6)
      if (me == 0) {
        std::fprintf(gf, "%ld,%.8f,%.3e,%.15e", k, t, dt, mass);
        for (int l = 0; l < 5; ++l) std::fprintf(gf, ",%.6e", std::abs(a[l]) / ringsum);
        std::fprintf(gf, "\n");
        std::fflush(gf);
      }
    }
    if (k % snap_every == 0 || t >= tend) {
      // Accesseurs collectifs : TOUS les rangs appellent state_global/potential_global ;
      // seul le rang 0 ecrit le fichier (les autres rangs detiennent le meme champ mais ne
      // dumpent pas, pour ne pas courir en ecriture sur snap_*.raw).
      const std::vector<double> U = sys.state_global("mom");
      const std::vector<double> phi = sys.potential_global();
      double pmax = 0.0;
      for (double v : phi) pmax = std::max(pmax, std::abs(v));
      max_abs_phi = pmax;
      if (me == 0) {
        char path[512];
        std::snprintf(path, sizeof path, "%s/snap_%06ld.raw", out.c_str(), k);
        std::ofstream f(path, std::ios::binary);
        const double hdr[3] = {static_cast<double>(n), t, static_cast<double>(k)};
        f.write(reinterpret_cast<const char*>(hdr), sizeof hdr);
        f.write(reinterpret_cast<const char*>(U.data()),
                static_cast<std::streamsize>(U.size() * sizeof(double)));
        f.write(reinterpret_cast<const char*>(phi.data()),
                static_cast<std::streamsize>(phi.size() * sizeof(double)));
        std::printf("[snap] pas %ld t=%.5f -> %s\n", k, t, path);
        std::fflush(stdout);
      }
    }
  }

  const std::vector<double> U = sys.state_global("mom");  // collectif (tous les rangs)
  double fmass = 0.0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) fmass += U[static_cast<std::size_t>(j) * n + i];
  mass = fmass;
  if (me == 0) {
    std::printf("[fin] %ld pas, t=%.5f, derive de masse %.2e\n", k, t,
                std::abs(mass - mass0) / mass0);
    // Ligne machine-parsable consommee par diocotron_mpi.sbatch (gate masse + parite ulp np).
    std::printf("HYQMOMPI np=%d n=%d t=%.8f steps=%ld mass=%.17e massdrift=%.3e "
                "maxabs_phi=%.6e l4=%.6e\n",
                np, n, t, k, mass, std::abs(mass - mass0) / mass0, max_abs_phi, last_a4);
    std::fflush(stdout);
  }
  if (gf) std::fclose(gf);
  adc::comm_finalize();
  return 0;
}
