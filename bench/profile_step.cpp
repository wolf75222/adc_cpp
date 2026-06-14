// Harnais de PROFILAGE (mesure seule, ZERO optimisation). Reconstruit a la main, a partir des
// SEAMS PUBLICS de la bibliotheque (assemble_rhs, GeometricMG / PoissonFFTSolver, fill_boundary,
// max_wave_speed_mf, dot, device_fence), un pas de temps REPRESENTATIF du diocotron tel que le joue
// System::step (python/system.cpp), et CHRONOMETRE separement chaque phase :
//
//   - poisson    : solve elliptique (GeometricMG V-cycles ou PoissonFFTSolver)
//   - aux_derive : assemblage du second membre q n + derivation (phi, grad phi) -> canal aux
//                  (les deux boucles HOTE par cellule de solve_fields)
//   - halos      : fill_boundary / fill_ghosts (echange MPI + ghosts physiques) sur aux ET U
//   - transport  : fill_ghosts(U) compte a part ; assemble_rhs<L,F> (operateur FV) + lincomb SSPRK
//   - reduction  : max_wave_speed_mf (CFL) + dot (norme, brique des reductions Krylov)
//   - fence      : device_fence() isole (cout d'une barriere device sous Kokkos ; ~0 en serie)
//   - alloc_tmp  : (re)allocation des MultiFab temporaires par pas (R, etage SSPRK U1)
//
// Le pas reproduit l'ORDONNANCEMENT de System::step : solve_fields (poisson + aux_derive + halos),
// puis avance du bloc (SSPRK2 = 2 etages : chacun fill_ghosts + assemble_rhs + lincomb), avec un pas
// CFL (max_wave_speed_mf). On NE touche NI system.cpp NI aucun en-tete du hot path : tout est ici,
// via l'API publique header-only -- exactement le code que System::step orchestre, instrumente.
//
// Ce binaire est COMPILE avec le meme backend que la bibliotheque (Serie / Kokkos OpenMP / Kokkos
// Cuda) et lance sous MPI (np>1) sans changement : le seam for_each_cell et le seam comm portent le
// backend. La grille est UNE box repartie en round-robin (DistributionMapping(1, n_ranks)), comme
// System -- donc sous MPI un seul rang possede la box ; le profil MPI mesure alors le surcout des
// COLLECTIVES (all_reduce dans dot / max_wave_speed_mf, halos) sur ce decoupage mono-box, fidele a
// System (lui-meme mono-box). On mesure le pas de System tel quel, pas un cas a charge repartie.

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/elliptic/geometric_mg.hpp>
#include <adc/numerics/elliptic/poisson_fft_solver.hpp>
#include <adc/numerics/spatial_operator.hpp>
#include <adc/parallel/comm.hpp>
#include <adc/physics/bricks.hpp>  // CompositeModel, ExBVelocity, NoSource, ChargeDensity

#include "common.hpp"  // adc::bench::{timed, PhaseTimers, eat} (briques de mesure partagees)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

using namespace adc;
using adc::bench::Clock;        // std::chrono::steady_clock (horloge des harnais)
using adc::bench::PhaseTimers;  // accumulateur de temps par phase (poisson/aux/halos/transport/...)
using adc::bench::timed;        // chronometre une phase (device_fence avant/apres)
static constexpr double kPi = 3.14159265358979323846;

// Modele DIOCOTRON : advection scalaire ExB (n_vars=1) + pas de source + densite de charge q n au
// second membre du Poisson. C'est EXACTEMENT la composition que monte le cas diocotron (cf. adc_cases)
// et que System branche par add_block ; le chemin numerique (assemble_rhs<Limiter, Rusanov>, SSPRK2,
// GeometricMG) est identique.
using Diocotron = CompositeModel<ExBVelocity, NoSource, ChargeDensity>;

// Initialise un anneau de densite (profil diocotron : couronne de charge module azimutalement).
static void init_ring(MultiFab& U, const Geometry& geom) {
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 a = U.fab(li).array();
    const Box2D v = U.box(li);
    const double cx = 0.5 * (geom.xlo + geom.xhi), cy = 0.5 * (geom.ylo + geom.yhi);
    const double r0 = 0.18, dr = 0.04;
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        const double x = geom.x_cell(i) - cx, y = geom.y_cell(j) - cy;
        const double r = std::hypot(x, y);
        const double th = std::atan2(y, x);
        const double bump = std::exp(-(r - r0) * (r - r0) / (2 * dr * dr));
        a(i, j, 0) = 0.05 + bump * (1.0 + 0.1 * std::cos(4 * th));  // graine du mode diocotron
      }
  }
}

struct StepResult {
  PhaseTimers t;
};

// Un pas REPRESENTATIF, instrumente phase par phase. solver : pointeur non nul = celui utilise.
static StepResult one_step(Diocotron& model, MultiFab& U, MultiFab& aux, GeometricMG* mg,
                           PoissonFFTSolver* fft, const Geometry& geom, const Box2D& dom,
                           const BCRec& bc, const Periodicity& per, bool periodic, double cfl,
                           const BoxArray& ba, const DistributionMapping& dm,
                           const std::string& limiter) {
  StepResult res;
  PhaseTimers& t = res.t;
  const Real dx = geom.dx(), dy = geom.dy();
  const int ncomp = Diocotron::n_vars;

  // --- solve_fields (1) : assemblage du rhs = q n (boucle hote elliptic_rhs) -------------------
  MultiFab& rhs = mg ? mg->rhs() : fft->rhs();
  MultiFab& phi = mg ? mg->phi() : fft->phi();
  t.aux_derive += timed([&] {
    rhs.set_val(Real(0));
    for (int li = 0; li < rhs.local_size(); ++li) {
      Array4 r = rhs.fab(li).array();
      const ConstArray4 u = U.fab(li).const_array();
      const Box2D b = rhs.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          r(i, j) += model.elliptic_rhs(load_state<Diocotron>(u, i, j));
    }
  });
  // (2) solve elliptique = la phase poisson
  t.poisson += timed([&] {
    if (mg)
      mg->solve();
    else
      fft->solve();
  });
  // (3) derivation aux = (phi, grad phi) par cellule (boucle hote de solve_fields)
  t.aux_derive += timed([&] {
    for (int li = 0; li < aux.local_size(); ++li) {
      const ConstArray4 p = phi.fab(li).const_array();
      Array4 a = aux.fab(li).array();
      const Box2D v = aux.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
          a(i, j, 0) = p(i, j);
          a(i, j, 1) = (p(i + 1, j) - p(i - 1, j)) / (2 * dx);
          a(i, j, 2) = (p(i, j + 1) - p(i, j - 1)) / (2 * dy);
        }
    }
  });
  // (4) halos de aux (echange MPI + ghosts physiques) -- comme la fin de solve_fields
  t.halos += timed([&] {
    if (periodic)
      fill_boundary(aux, dom, per);
    else
      fill_ghosts(aux, dom, bc);
  });

  // --- pas CFL : reduction max vitesse d'onde (max_wave_speed_mf, all_reduce_max sous MPI) ------
  Real w = 0;
  t.reduction += timed([&] { w = std::max(max_wave_speed_mf(model, U, aux), Real(1e-30)); });
  const Real h = std::min(dx, dy);
  const Real dt = static_cast<Real>(cfl) * h / w;

  // --- avance SSPRK2 : etage temporaire R + etat intermediaire U1 (alloc par pas) ---------------
  MultiFab R, U1;
  t.alloc_tmp += timed([&] {
    R = MultiFab(ba, dm, ncomp, 0);
    U1 = MultiFab(ba, dm, ncomp, U.n_grow());
  });

  const Diocotron& m = model;
  const bool prim = false;  // scalaire : prim == cons
  auto assemble = [&](MultiFab& Uin, MultiFab& Rout) {
    t.halos += timed([&] {  // halos de U (echange + ghosts physiques)
      if (periodic)
        fill_boundary(Uin, dom, per);
      else
        fill_ghosts(Uin, dom, bc);
    });
    t.transport += timed([&] {  // operateur FV : assemble_rhs<Limiter, Rusanov>
      if (limiter == "none")
        assemble_rhs<NoSlope, RusanovFlux>(m, Uin, aux, geom, Rout, prim);
      else if (limiter == "vanleer")
        assemble_rhs<VanLeer, RusanovFlux>(m, Uin, aux, geom, Rout, prim);
      else if (limiter == "weno5")
        assemble_rhs<Weno5, RusanovFlux>(m, Uin, aux, geom, Rout, prim);
      else
        assemble_rhs<Minmod, RusanovFlux>(m, Uin, aux, geom, Rout, prim);
    });
  };

  // etage 1 : U1 = U + dt R(U)
  assemble(U, R);
  t.transport += timed([&] {
    for (int li = 0; li < U.local_size(); ++li) {
      Array4 u1 = U1.fab(li).array();
      const ConstArray4 u = U.fab(li).const_array();
      const ConstArray4 r = R.fab(li).const_array();
      const Box2D v = U.box(li);
      for (int c = 0; c < ncomp; ++c)
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i) u1(i, j, c) = u(i, j, c) + dt * r(i, j, c);
    }
  });
  // etage 2 : U = 0.5 U + 0.5 (U1 + dt R(U1))
  assemble(U1, R);
  t.transport += timed([&] {
    for (int li = 0; li < U.local_size(); ++li) {
      Array4 u = U.fab(li).array();
      const ConstArray4 u1 = U1.fab(li).const_array();
      const ConstArray4 r = R.fab(li).const_array();
      const Box2D v = U.box(li);
      for (int c = 0; c < ncomp; ++c)
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i)
            u(i, j, c) = Real(0.5) * u(i, j, c) + Real(0.5) * (u1(i, j, c) + dt * r(i, j, c));
    }
  });

  // --- reduction supplementaire (dot, brique des produits scalaires Krylov / diagnostics) -------
  t.reduction += timed([&] {
    volatile Real s = dot(U, U, 0);
    (void)s;
  });

  // --- fence isole : cout d'une barriere device par pas (no-op en serie / OpenMP) ---------------
  t.fence += timed([&] { device_fence(); });

  return res;
}

int main(int argc, char** argv) {
  comm_init(&argc, &argv);  // MPI_Init si compile avec MPI ; no-op en serie

  int n = 256, steps = 50, warmup = 5;
  double cfl = 0.4;
  std::string solver = "geometric_mg", limiter = "minmod", bcmode = "periodic";
  for (int a = 1; a < argc; ++a) {
    using adc::bench::eat;  // consomme un argument "--cle valeur" (avance a, convertit selon le type)
    if (eat(argc, argv, a, "--n", n)) continue;
    if (eat(argc, argv, a, "--steps", steps)) continue;
    if (eat(argc, argv, a, "--warmup", warmup)) continue;
    if (eat(argc, argv, a, "--cfl", cfl)) continue;
    if (eat(argc, argv, a, "--solver", solver)) continue;
    if (eat(argc, argv, a, "--limiter", limiter)) continue;
    if (eat(argc, argv, a, "--bc", bcmode)) continue;
  }
  const bool periodic = (bcmode == "periodic");

  // --- maillage UNE box repartie round-robin (comme System) ------------------------------------
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, n_ranks());
  BCRec bc;  // transport : periodique par defaut
  if (!periodic) bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Foextrap;
  Periodicity per{periodic, periodic};
  BCRec pbc;  // Poisson : periodique ou Dirichlet
  if (!periodic) pbc.xlo = pbc.xhi = pbc.ylo = pbc.yhi = BCType::Dirichlet;

  const int n_ghost = (limiter == "weno5") ? 3 : (limiter == "none" ? 1 : 2);
  MultiFab U(ba, dm, Diocotron::n_vars, n_ghost);
  MultiFab aux(ba, dm, kAuxBaseComps, 1);
  U.set_val(Real(0));
  aux.set_val(Real(0));
  init_ring(U, geom);

  Diocotron model{ExBVelocity{1.0}, NoSource{}, ChargeDensity{1.0}};

  // --- garde-fous solveur (alignes sur System::ensure_elliptic) --------------------------------
  if (solver == "fft") {
    if (n_ranks() > 1) {
      if (my_rank() == 0)
        std::fprintf(stderr, "[bench] solver=fft non supporte en MPI (np>1) -> geometric_mg\n");
      solver = "geometric_mg";
    } else if (!periodic) {
      if (my_rank() == 0)
        std::fprintf(stderr, "[bench] solver=fft exige periodic -> geometric_mg\n");
      solver = "geometric_mg";
    }
  }

  // --- solveur elliptique construit une fois (comme System) ------------------------------------
  GeometricMG* mg = nullptr;
  PoissonFFTSolver* fft = nullptr;
  GeometricMG mg_storage(geom, ba, pbc, std::function<bool(Real, Real)>{});
  PoissonFFTSolver* fft_storage = nullptr;
  if (solver == "fft") {
    fft_storage = new PoissonFFTSolver(geom, ba, pbc, std::function<bool(Real, Real)>{});
    fft = fft_storage;
  } else {
    mg = &mg_storage;
  }

  // --- warmup (hors mesure : pagination, premier kernel Kokkos, premier V-cycle) ---------------
  for (int s = 0; s < warmup; ++s)
    one_step(model, U, aux, mg, fft, geom, dom, bc, per, periodic, cfl, ba, dm, limiter);

  // --- mesure ----------------------------------------------------------------------------------
  PhaseTimers acc;
  const auto wall0 = Clock::now();
  for (int s = 0; s < steps; ++s) {
    StepResult r =
        one_step(model, U, aux, mg, fft, geom, dom, bc, per, periodic, cfl, ba, dm, limiter);
    acc.add(r.t);
  }
  device_fence();
  const auto wall1 = Clock::now();
  const double wall = std::chrono::duration<double>(wall1 - wall0).count();

  // --- agregation MPI : MAX sur les rangs (chemin critique d'un pas collectif) ------------------
  auto rmax = [](double x) { return all_reduce_max(x); };
  PhaseTimers mx;
  mx.poisson = rmax(acc.poisson);
  mx.aux_derive = rmax(acc.aux_derive);
  mx.halos = rmax(acc.halos);
  mx.transport = rmax(acc.transport);
  mx.reduction = rmax(acc.reduction);
  mx.fence = rmax(acc.fence);
  mx.alloc_tmp = rmax(acc.alloc_tmp);
  const double wall_max = rmax(wall);

  if (my_rank() == 0) {
    const double tot = mx.total();
    const double per_step_ms = 1e3 * tot / steps;
    std::printf("# adc_cpp profile_step\n");
    std::printf(
        "# backend ranks=%d  grid=%dx%d  steps=%d  warmup=%d  solver=%s  limiter=%s  bc=%s  "
        "cfl=%.3f\n",
        n_ranks(), n, n, steps, warmup, solver.c_str(), limiter.c_str(), bcmode.c_str(), cfl);
    std::printf("# wall_total=%.4f s  sum_phases=%.4f s  per_step=%.4f ms\n", wall_max, tot,
                per_step_ms);
    std::printf("%-12s %14s %14s %8s\n", "phase", "total_s", "per_step_ms", "pct");
    auto row = [&](const char* name, double v) {
      std::printf("%-12s %14.6f %14.6f %7.1f%%\n", name, v, 1e3 * v / steps,
                  tot > 0 ? 100.0 * v / tot : 0.0);
    };
    row("transport", mx.transport);
    row("poisson", mx.poisson);
    row("halos", mx.halos);
    row("aux_derive", mx.aux_derive);
    row("reduction", mx.reduction);
    row("fence", mx.fence);
    row("alloc_tmp", mx.alloc_tmp);
    row("TOTAL", tot);
  }

  delete fft_storage;
  comm_finalize();
  return 0;
}
