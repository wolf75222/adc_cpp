// Poisson FFT sous MPI via remap box<->slabs (ADC-287). Avec set_poisson(..., "fft") sous mpirun
// -np>1, System repartit UNE box unique en round-robin (DistributionMapping(1, n_ranks())), donc
// certains rangs ont local_size()==0. Historiquement PoissonFFTSolver::solve() dereferencait fab(0)
// sur un rang sans box -> SIGSEGV (#93), et le chemin (#93) REFUSAIT explicitement "fft" sous MPI.
//
// ADC-287 remplace ce refus par un chemin distribue : RemappedFFTSolver presente vers l'exterieur la
// MEME box unique (rhs()/phi() sur ba/dm, alignes avec l'aux) et cache un scatter/gather box<->slabs
// autour de PoissonFFT dans solve(). Le chemin solve_fields reste donc intact. C'est un changement
// STRUCTUREL en attente du vote de design ADC-273.
//
// Ce test, sous -np {1,2,4} :
//   - np=1 : "fft" marche (solve_fields() tourne, potentiel fini non trivial), bit-coherent avec le
//            chemin geometric_mg du meme probleme (a la tolerance FP du solveur direct vs iteratif).
//   - np>1 : "fft" marche aussi (RemappedFFTSolver) : solve_fields() ne leve pas, le potentiel est
//            fini non trivial, et il coincide avec geometric_mg du meme probleme (apres recentrage de
//            moyenne pour neutraliser la jauge), a la tolerance FP. n=16 -> Ny=16 divisible par 1/2/4,
//            donc la garde Ny % n_ranks() de RemappedFFTSolver passe.
//
// ADC-316 : en complement de la parite phi vs geometric_mg ci-dessus, on assure aussi le residu
// discret du RemappedFFTSolver (residual(), DIRECTEMENT car l'API System ne l'expose pas) sur deux
// tailles -- 16 (radix-2) et 12 (NON puissance de 2 -> repli DFT directe O(n^2) de PoissonFFT, le
// chemin du remap jusqu'ici non couvert) -- toutes deux machine-zero sous np = 1/2/4.

#include <adc/numerics/elliptic/poisson_fft_solver.hpp>  // RemappedFFTSolver, BoxArray (direct residual check)
#include <adc/mesh/geometry.hpp>                         // Geometry, Box2D
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
static constexpr double kPi = 3.14159265358979323846;

// Residu discret machine-zero du RemappedFFTSolver (le solveur "fft"/"fft_spectral" sous MPI) sur la
// MEME disposition mono-box que System (ba.size()==1, dm(1, np) -> box sur le rang proprietaire, fab
// vide ailleurs). L'API System n'expose pas residual(), on teste donc le solveur DIRECTEMENT.
// rho = sin(2 pi x) sin(2 pi y) : mode periodique a moyenne nulle (solvabilite Poisson). Le solve
// direct etant EXACT pour le Laplacien discret 5 points, residual() (COLLECTIF : poisson_residual sur
// l'owner + all_reduce_max) est machine-zero. N divisible par np ; radix-2 si puissance de 2, sinon
// repli DFT directe O(n^2) de PoissonFFT.
static double remap_residual(int N) {
  Geometry geom{Box2D::from_extents(N, N), 0.0, 1.0, 0.0, 1.0};
  BoxArray ba(std::vector<Box2D>{Box2D::from_extents(N, N)});  // box unique = disposition System
  const double dx = geom.dx(), dy = geom.dy();
  RemappedFFTSolver fft(geom, ba);
  for (int li = 0; li < fft.rhs().local_size(); ++li) {  // seul le proprietaire possede une fab
    Array4 r = fft.rhs().fab(li).array();
    const Box2D b = fft.rhs().box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        r(i, j) = std::sin(2 * kPi * (i + 0.5) * dx) * std::sin(2 * kPi * (j + 0.5) * dy);
  }
  fft.solve();            // scatter/gather box<->bandes + PoissonFFT + wrap periodique des ghosts
  return fft.residual();  // COLLECTIF : poisson_residual (owner) + all_reduce_max
}

// Bloc de CHARGE : alimente le second membre du Poisson (elliptic_rhs = densite de charge q n).
struct ChargeEll {
  template <class State>
  ADC_HD Real rhs(const State& u) const {
    return u[0];
  }  // rho = comp 0
};

using ProbeModel =
    CompositeModel<ExBVelocity, NoSource, ChargeEll>;  // transporte + charge le Poisson

// Construit le probleme (charge a moyenne nulle pour le Poisson periodique soluble) et cable "fft".
// Le rang proprietaire (0) ecrit la densite. Renvoie le System pret a solve_fields().
static void build_problem(System& sys, int n, bool owns) {
  add_compiled_model(sys, "probe", ProbeModel{}, "minmod", "rusanov", "conservative", "explicit");
  sys.set_poisson("composite", "fft");  // f = somme des briques elliptiques (ici la charge)
  if (owns) {
    const std::size_t nn = static_cast<std::size_t>(n) * n;
    std::vector<double> q(nn, 0.0);  // charge a moyenne nulle : +1 gauche, -1 droite (somme = 0)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        q[static_cast<std::size_t>(j) * n + i] = (i < n / 2) ? 1.0 : -1.0;
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
    if (!c) {
      std::printf("[rank %d/%d] FAIL %s\n", me, np, w);
      ++fails;
    }
  };

  // RemappedFFTSolver residual() machine-zero (ADC-316). Le solveur "fft" sous MPI est
  // RemappedFFTSolver ; on verifie son residu discret DIRECTEMENT (l'API System n'expose pas
  // residual()) sur la grille mono-box System a deux tailles : 16 (puissance de 2 -> radix-2) et
  // 12 (NON puissance de 2 -> repli DFT directe O(n^2) de PoissonFFT, chemin du remap jusqu'ici non
  // couvert). Les deux doivent etre divisibles par np (transposee a bandes : Nx ET Ny divisibles par
  // n_ranks()) ; 12 = 2^2 * 3 valide np = 1/2/4.
  for (int Nr : {16, 12}) {
    if (Nr % np != 0) {
      if (me == 0)
        std::printf("[rank 0/%d] SKIP remap residual N=%d (non divisible par np)\n", np, Nr);
      continue;
    }
    const double res = remap_residual(Nr);
    if (me == 0)
      std::printf("[rank 0/%d] RemappedFFTSolver residual (N=%d, %s) = %.3e\n", np, Nr,
                  is_pow2(Nr) ? "radix-2" : "DFT O(n^2)", res);
    chk(res < 1e-9, "remap_residual_machine_zero");
  }

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
      if (!std::isfinite(v))
        finite = false;
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
      for (int i = 0; i < n; ++i)
        q[static_cast<std::size_t>(j) * n + i] = (i < n / 2) ? 1.0 : -1.0;
    ref.set_density("probe", q);
    ref.solve_fields();
    const std::vector<double> phi_ref = ref.potential();
    chk(phi_ref.size() == nn, "np1_ref_size");

    double mean_fft = 0, mean_ref = 0;
    for (std::size_t k = 0; k < nn; ++k) {
      mean_fft += phi[k];
      mean_ref += phi_ref[k];
    }
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
    // np>1 : "fft" doit MARCHER (RemappedFFTSolver). solve_fields() (-> ensure_elliptic) construit le
    // solveur distribue sur TOUS les rangs (collectif) ; aucun ne leve. Le scatter/gather box<->slabs
    // est interne a solve(). On compare ensuite a geometric_mg du meme probleme.
    System sys(cfg);
    build_problem(sys, n, owns);  // set_poisson ne construit pas encore le solveur (paresseux)
    bool threw = false;
    std::string msg;
    try {
      sys.solve_fields();
      sys.solve_fields();  // construit-puis-reutilise (le solveur paresseux n'est bati qu'une fois)
    } catch (const std::exception& e) {
      threw = true;
      msg = e.what();
    }
    if (threw)
      std::printf("[rank %d/%d] np=%d inattendu : fft a leve : %s\n", me, np, np, msg.c_str());
    chk(!threw, "npN_fft_runs");

    // potential_global() est COLLECTIF : il renvoie le champ global n*n sur CHAQUE rang (la box vit
    // sur le rang 0, all_reduce_sum_inplace le diffuse). On verifie qu'il est fini et non trivial.
    const std::vector<double> phi = sys.potential_global();
    chk(phi.size() == nn, "npN_potential_size");
    double maxabs = 0;
    bool finite = true;
    for (double v : phi) {
      if (!std::isfinite(v))
        finite = false;
      maxabs = std::fmax(maxabs, std::fabs(v));
    }
    chk(finite, "npN_potential_finite");
    chk(maxabs > 0.0, "npN_potential_nonzero");

    // Reference : MEME probleme distribue via geometric_mg. La FFT remappee et le MG iteratif resolvent
    // le MEME Laplacien 5-points periodique -> meme potentiel a la tolerance FP (jauge : moyenne nulle).
    // On compare apres recentrage de moyenne pour neutraliser la constante de jauge.
    System ref(cfg);
    add_compiled_model(ref, "probe", ProbeModel{}, "minmod", "rusanov", "conservative", "explicit");
    ref.set_poisson("composite", "geometric_mg");
    if (owns) {
      std::vector<double> q(nn, 0.0);
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
          q[static_cast<std::size_t>(j) * n + i] = (i < n / 2) ? 1.0 : -1.0;
      ref.set_density("probe", q);
    }
    ref.solve_fields();
    const std::vector<double> phi_ref = ref.potential_global();
    chk(phi_ref.size() == nn, "npN_ref_size");

    double mean_fft = 0, mean_ref = 0;
    for (std::size_t k = 0; k < nn; ++k) {
      mean_fft += phi[k];
      mean_ref += phi_ref[k];
    }
    mean_fft /= static_cast<double>(nn);
    mean_ref /= static_cast<double>(nn);
    double linf = 0, ampl = 0;
    for (std::size_t k = 0; k < nn; ++k) {
      const double a = phi[k] - mean_fft, b = phi_ref[k] - mean_ref;
      linf = std::fmax(linf, std::fabs(a - b));
      ampl = std::fmax(ampl, std::fabs(b));
    }
    const double rel = (ampl > 0) ? linf / ampl : linf;
    if (me == 0)
      std::printf("[rank 0/%d] np=%d fft OK : |phi|max=%.3e  ||phi_fft - phi_mg||inf/ampl=%.3e\n",
                  np, np, maxabs, rel);
    chk(rel < 5e-3, "npN_fft_matches_mg");
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
