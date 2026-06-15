// Bug device+MPI du solveur Poisson FFT (#93) : avec set_poisson(..., "fft") sous mpirun -np>1,
// System repartit UNE box unique en round-robin (DistributionMapping(1, n_ranks())), donc certains
// rangs ont local_size()==0. PoissonFFTSolver::solve() dereferencait alors fab(0) / phi_.fab(0) sur
// un rang sans box locale -> SIGSEGV (fault addr 0x28). L'assert(n_ranks()==1...) du constructeur
// disparaissait en Release (NDEBUG) et la protection s'evanouissait en silence.
//
// Le fix (chemin B) REFUSE explicitement "fft" sous MPI (n_ranks>1) avec une erreur claire, levee sur
// TOUS les rangs (ensure_elliptic est appele collectivement par solve_fields) : pas de segfault, pas
// d'interblocage. Le periodique distribue passe par DistributedFFTSolver (teste a part dans
// test_mpi_fft_distributed). Garde-fou dur redondant dans le constructeur de PoissonFFTSolver.
//
// Ce test, sous -np {1,2,4} :
//   - np=1 : "fft" marche (solve_fields() tourne, potentiel fini non trivial), bit-coherent avec le
//            chemin geometric_mg du meme probleme (a la tolerance FP du solveur direct vs iteratif).
//   - np>1 : solve_fields() leve std::runtime_error sur CHAQUE rang (message clair), sans crash ni
//            blocage. On verifie que TOUS les rangs ont bien leve (all_reduce), donc collectif.

#include <adc/physics/composite.hpp>
#include <adc/physics/hyperbolic.hpp>  // ExBVelocity
#include <adc/physics/source.hpp>      // NoSource
#include <adc/runtime/dsl_block.hpp>   // add_compiled_model
#include <adc/runtime/system.hpp>

#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

#ifdef ADC_HAS_MPI
#include <mpi.h>
#endif

using namespace adc;

// Bloc de CHARGE : alimente le second membre du Poisson (elliptic_rhs = densite de charge q n).
struct ChargeEll {
  template <class State>
  ADC_HD Real rhs(const State& u) const { return u[0]; }  // rho = comp 0
};

using ProbeModel = CompositeModel<ExBVelocity, NoSource, ChargeEll>;  // transporte + charge le Poisson

// Construit le probleme (charge a moyenne nulle pour le Poisson periodique soluble) et cable "fft".
// Le rang proprietaire (0) ecrit la densite. Renvoie le System pret a solve_fields().
static void build_problem(System& sys, int n, bool owns) {
  add_compiled_model(sys, "probe", ProbeModel{}, "minmod", "rusanov", "conservative", "explicit");
  sys.set_poisson("composite", "fft");  // f = somme des briques elliptiques (ici la charge)
  if (owns) {
    const std::size_t nn = static_cast<std::size_t>(n) * n;
    std::vector<double> q(nn, 0.0);  // charge a moyenne nulle : +1 gauche, -1 droite (somme = 0)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) q[static_cast<std::size_t>(j) * n + i] = (i < n / 2) ? 1.0 : -1.0;
    sys.set_density("probe", q);
  }
}

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(ADC_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  const int me = my_rank(), np = n_ranks();

  long fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("[rank %d/%d] FAIL %s\n", me, np, w); ++fails; }
  };

  const int n = 16;
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  const bool owns = (me == 0);  // box 0 -> rang 0 sous DistributionMapping(1, np)

  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;

  if (np == 1) {
    // np=1 : "fft" doit marcher. solve_fields() construit le solveur FFT mono-rang et resout.
    System sys(cfg);
    build_problem(sys, n, owns);
    bool threw = false;
    try {
      sys.solve_fields();
      sys.solve_fields();  // construit-puis-reutilise
    } catch (const std::exception& e) {
      threw = true;
      std::printf("[rank 0/1] np=1 inattendu : fft a leve : %s\n", e.what());
    }
    chk(!threw, "np1_fft_runs");

    const std::vector<double> phi = sys.potential();
    chk(phi.size() == nn, "np1_potential_size");
    double maxabs = 0;
    bool finite = true;
    for (double v : phi) {
      if (!std::isfinite(v)) finite = false;
      maxabs = std::fmax(maxabs, std::fabs(v));
    }
    chk(finite, "np1_potential_finite");
    chk(maxabs > 0.0, "np1_potential_nonzero");

    // Reference : meme probleme via geometric_mg. La FFT directe et le MG iteratif resolvent le MEME
    // Laplacien 5-points periodique -> meme potentiel a la tolerance FP (jauge : moyenne nulle des
    // deux cotes). On compare apres recentrage de moyenne pour neutraliser la constante de jauge.
    System ref(cfg);
    add_compiled_model(ref, "probe", ProbeModel{}, "minmod", "rusanov", "conservative", "explicit");
    ref.set_poisson("composite", "geometric_mg");
    std::vector<double> q(nn, 0.0);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) q[static_cast<std::size_t>(j) * n + i] = (i < n / 2) ? 1.0 : -1.0;
    ref.set_density("probe", q);
    ref.solve_fields();
    const std::vector<double> phi_ref = ref.potential();
    chk(phi_ref.size() == nn, "np1_ref_size");

    double mean_fft = 0, mean_ref = 0;
    for (std::size_t k = 0; k < nn; ++k) { mean_fft += phi[k]; mean_ref += phi_ref[k]; }
    mean_fft /= static_cast<double>(nn);
    mean_ref /= static_cast<double>(nn);
    double linf = 0, ampl = 0;
    for (std::size_t k = 0; k < nn; ++k) {
      const double a = phi[k] - mean_fft, b = phi_ref[k] - mean_ref;
      linf = std::fmax(linf, std::fabs(a - b));
      ampl = std::fmax(ampl, std::fabs(b));
    }
    // Tolerance relative a l'amplitude du potentiel de reference (MG converge a sa tolerance interne).
    const double rel = (ampl > 0) ? linf / ampl : linf;
    std::printf("[rank 0/1] np=1 fft OK : |phi|max=%.3e  ||phi_fft - phi_mg||inf/ampl=%.3e\n",
                maxabs, rel);
    chk(rel < 5e-3, "np1_fft_matches_mg");
  } else {
    // np>1 : "fft" doit etre REFUSE proprement. solve_fields() (-> ensure_elliptic) leve sur TOUS les
    // rangs. On capture l'exception : pas de segfault, pas d'interblocage. Le message doit etre clair.
    System sys(cfg);
    build_problem(sys, n, owns);  // set_poisson ne construit pas encore le solveur (paresseux)
    bool threw = false;
    std::string msg;
    try {
      sys.solve_fields();  // construit le solveur -> doit lever ici, sur CE rang
    } catch (const std::exception& e) {
      threw = true;
      msg = e.what();
    }
    chk(threw, "npN_fft_refused");
    // Message clair et explicite (sous-chaine stable du libelle verbatim).
    chk(msg.find("fft solver unsupported under MPI") != std::string::npos, "npN_clear_message");
    std::printf("[rank %d/%d] np=%d fft refuse (attendu) : %s\n", me, np, np,
                threw ? msg.c_str() : "(pas d'exception !)");

    // TOUS les rangs doivent avoir leve (sinon un rang serait entre dans le solve collectif et
    // bloquerait). On reduit le compte de "a leve" : il doit valoir np.
#ifdef ADC_HAS_MPI
    long threw_local = threw ? 1 : 0, threw_all = 0;
    MPI_Allreduce(&threw_local, &threw_all, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    chk(threw_all == np, "npN_all_ranks_threw");
#endif
  }

#ifdef ADC_HAS_MPI
  if (np > 1) {
    long g = 0;
    MPI_Allreduce(&fails, &g, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    fails = g;
  }
#endif
  if (me == 0 && fails == 0)
    std::printf("OK test_mpi_system_fft (np=%d)\n", np);
  comm_finalize();
  return fails == 0 ? 0 : 1;
}
