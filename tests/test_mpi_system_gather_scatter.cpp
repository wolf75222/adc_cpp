// Verrou de non-regression du bug "finding 8" : acces fab(0) SANS garde local_size dans les
// fermetures rhs_into / advance / max_speed des blocs DSL hote (dynamique .so ET compile).
//
// System repartit UNE box unique en round-robin (DistributionMapping(1, n_ranks())), donc a np>1
// un seul rang la possede ; les autres ont local_size()==0. Avant ce fix, les fermetures
// rhs_into / advance / max_speed des chemins dynamique (push_dynamic) et compile (add_compiled_block)
// appelaient copy_state(U, ...) / write_state(U, ...) sans tester local_size() -> fab(0) hors-bornes
// -> crash UB silencieux ou segfault sur les rangs vides.
//
// Ce test exerce System::add_compiled_model (chemin compile, fermetures type-erased) sous mpirun
// -np {1,2,4} et exige :
//   (a) aucun crash (fermetures no-op sur les rangs sans box) ;
//   (b) resultat INVARIANT au nombre de rangs (la box vit sur rang 0 quel que soit np) ;
//   (c) step() / step_cfl() COLLECTIFS (tous les rangs participent aux all_reduce internes).
//
// RAISONNEMENT COLLECTIF MPI : rhs_into / advance / max_speed ne contiennent AUCUNE operation MPI
// collective (contrairement au solve Poisson qui est collectif). La garde "if (U.local_size() == 0)
// return;" est un no-op unilateral sur les rangs vides -> aucun risque d'interblocage. Le seul
// collectif est l'all_reduce_max implicite dans step_cfl() (appele APRES les max_speed des blocs),
// que TOUS les rangs atteignent (la garde early-return est AVANT ce collectif). De meme, mass()
// est un all_reduce_sum appele collectivement en fin de test.
//
// AVANT le fix : segfault / UB a np=2/4 sur les rangs sans box locale (rhs_into / advance /
// max_speed dereferencaient fab(0) inexistant). APRES : np=1/2/4 verts, resultats identiques.

#include <pops/physics/composition/composite.hpp>
#include <pops/physics/bricks/hyperbolic.hpp>  // ExBVelocity (scalaire 1 var)
#include <pops/physics/bricks/source.hpp>      // NoSource
#include <pops/runtime/builders/compiled/dsl_block.hpp>   // add_compiled_model
#include <pops/runtime/system.hpp>

#include <pops/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif
#ifdef POPS_HAS_MPI
#include <mpi.h>
#endif

using namespace pops;

// Brique elliptique nulle : Poisson avec second membre nul -> phi=0 -> pas de derive.
// On teste l'avance en temps, pas la physique ; la densite reste uniforme.
struct NoEll {
  template <class State>
  POPS_HD Real rhs(const State&) const {
    return Real(0);
  }
};
// Modele scalaire : transport E x B (vitesse nulle ici car phi=0) + source nulle + elliptic nul.
using ScalarModel = CompositeModel<ExBVelocity, NoSource, NoEll>;

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
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

  const int n = 16;
  const double rho0 = 1.5, dt = 0.01;
  const int nsteps = 5;
  const std::size_t nn = static_cast<std::size_t>(n) * n;

  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;

  System sys(cfg);
  // add_compiled_model branche le chemin COMPILE (fermetures rhs_into / advance / max_speed
  // de add_compiled_block dans native_loader.hpp) : ce sont exactement les sites du finding 8.
  add_compiled_model(sys, "u", ScalarModel{}, "none", "rusanov", "conservative", "explicit");
  sys.set_poisson("composite", "geometric_mg");

  // Init uniforme sur le rang proprietaire (box 0 = rang 0 sous DistributionMapping(1, np)).
  // Les rangs vides ne touchent rien (set_density itere local_size() = 0 -> no-op).
  const bool owns = (me == 0);
  if (owns) {
    sys.set_density("u", std::vector<double>(nn, rho0));
  }

  // --- Exerce les 3 chemins du finding 8 sur TOUS les rangs ---
  //
  // (1) step() : appelle solve_fields (collectif) puis s.advance(U, dt, nsub) sur TOUS les rangs.
  //     advance (finding 8, chemin compile) appelle copy_state / write_state sans garde -> crash
  //     hors-bornes sur les rangs vides avant le fix. APRES : no-op sur les rangs vides.
  for (int s = 0; s < nsteps; ++s)
    sys.step(dt);

  // (2) step_cfl(cfl) : appelle s.max_speed(U) sur TOUS les rangs, puis all_reduce_max du dt CFL
  //     (collectif, APRES les max_speed). max_speed (finding 8) appelait copy_state sans garde.
  //     APRES : no-op sur les rangs vides, 0 local -> l'all_reduce prend le vrai max du proprietaire.
  const double dt_cfl = sys.step_cfl(0.5);
  chk(std::isfinite(dt_cfl) && dt_cfl > 0, "dt_cfl_valide");
  // dt_cfl INVARIANT en np : la box vit sur rang 0 quel que soit np ; le max_speed du proprietaire
  // est diffuse par all_reduce_max -> meme dt_cfl sur tous les rangs (seul rang 0 a une vraie vitesse,
  // les autres contribuent 0 a l'all_reduce, sans impact sur le max).
  chk(std::isfinite(dt_cfl), "dt_cfl_fini");

  // (3) eval_rhs() exerce rhs_into (finding 8) : copie U -> applique le residu -> copie retour.
  //     Avant le fix : crash sur les rangs vides. Apres : no-op sur les rangs vides ; le resultat
  //     est lit uniquement par le rang proprietaire.
  if (owns) {
    const std::vector<double> R = sys.eval_rhs("u");
    bool rfin = (R.size() == nn);
    for (double r : R)
      rfin = rfin && std::isfinite(r);
    chk(rfin, "rhs_fini");
  }

  // --- Verification du resultat physique (etat uniforme reste uniforme : transport nul, phi=0) ---
  // Appelee UNIQUEMENT sur le rang proprietaire (densite stockee localement).
  if (owns) {
    const std::vector<double> d = sys.density("u");
    chk(d.size() == nn, "densite_taille");
    bool finite = true;
    double dmin = d[0], dmax = d[0];
    for (double v : d) {
      finite = finite && std::isfinite(v);
      if (v < dmin)
        dmin = v;
      if (v > dmax)
        dmax = v;
    }
    chk(finite, "densite_finie");
    // etat uniforme, phi=0 -> transport nul -> densite INCHANGEE (a la precision machine).
    chk(std::fabs(dmin - rho0) < 1e-10, "densite_min_invariante");
    chk(std::fabs(dmax - rho0) < 1e-10, "densite_max_invariante");
    std::printf("[rank %d/%d] np=%d  rho_min=%.12f  rho_max=%.12f  dt_cfl=%.6e\n", me, np, np, dmin,
                dmax, dt_cfl);
  }

  // mass() est COLLECTIVE (sum -> all_reduce) : TOUS les rangs l'appellent (sinon interblocage).
  // La masse totale = rho0 * n*n INDEPENDANTE de np (la box vit toujours sur rang 0).
  const double mtot = sys.mass("u");
  chk(std::isfinite(mtot), "masse_finie");
  chk(std::fabs(mtot - rho0 * static_cast<double>(nn)) < 1e-9, "masse_conservee");

#ifdef POPS_HAS_MPI
  if (np > 1) {
    long g = 0;
    MPI_Allreduce(&fails, &g, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    fails = g;
  }
#endif
  if (me == 0 && fails == 0)
    std::printf("OK test_mpi_system_gather_scatter (np=%d)\n", np);
  comm_finalize();
  return fails == 0 ? 0 : 1;
}
