// Front C++ DIRECT de la campagne de perf (cf. adc_cases/perf). Joue le CAS SUR -- Euler compressible
// PUR, periodique, bulle de pression lisse de faible amplitude (rho>0, p>0 garantis) -- exactement la
// MEME physique et les MEMES reglages numeriques que les fronts Python (briques add_block / DSL
// add_equation), pour mesurer l'ECART de cout entre fronts a calcul identique.
//
// Ce binaire reconstruit a la main, depuis les SEAMS PUBLICS de la bibliotheque (assemble_rhs,
// GeometricMG, fill_boundary, max_wave_speed_mf, dot, device_fence), le pas que System::step
// orchestre -- comme bench/profile_step.cpp, dont il reprend la structure et l'instrumentation par
// phase (System n'a AUCUN timer interne ; tout le phase-timing vit ici). Il ajoute :
//   - le decoupage du TEMPS UTILISATEUR cold-cache en etages (model_build / addblock / state_init /
//     first_step / warmup / run_loop / diag) -- import et dsl_compile sont N/A en C++ (= 0) ;
//   - les percentiles de la boucle chaude (median / p10 / p90 / cv) sur echantillons par pas ;
//   - une sortie JSONL (schema "adc_perf_v1") agregee rang 0, consommee par perf/plot_frontend.py ;
//   - un bascule Poisson (--poisson on|off) : off = transport pur (signal frontend propre, defaut) ;
//     on = solve elliptique inerte (charge=0) a chaque pas, regime MG-domine (idiome two_euler).
//
// Compile/lance avec le MEME backend que la lib (Kokkos Serial / OpenMP / Cuda) et sous MPI (np>1)
// sans changement, via bench/run_frontend.sh. ZERO optimisation : on ne fait que MESURER.

#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/layout/distribution_mapping.hpp>
#include <adc/mesh/boundary/fill_boundary.hpp>
#include <adc/mesh/execution/for_each.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/mf_arith.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/mesh/boundary/physical_bc.hpp>
#include <adc/numerics/elliptic/mg/geometric_mg.hpp>
#include <adc/numerics/spatial_operator.hpp>
#include <adc/parallel/comm.hpp>
#include <adc/physics/bricks/bricks.hpp>  // CompositeModel, NoSource, ChargeDensity, kAuxBaseComps
#include <adc/physics/fluids/euler.hpp>   // Euler (brique hyperbolique compressible 4 var)

#include "common.hpp"  // adc::bench::{timed, PhaseTimers, percentile, eat} (briques de mesure partagees)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using namespace adc;
using adc::bench::Clock;        // std::chrono::steady_clock (horloge des harnais)
using adc::bench::PhaseTimers;  // accumulateur de temps par phase (total() inutilise ici)
using adc::bench::timed;        // chronometre une phase (device_fence avant/apres)
using adc::bench::percentile;   // percentile interpole des temps par pas

// ====================================================================================================
// CONTRAT DU CAS SUR -- ces constantes DOIVENT coincider bit-a-bit avec adc_cases/perf/frontend_compare.py
// (memes IC, memes reglages numeriques) sinon l'identite numerique inter-fronts tombe. Toute
// modification doit etre repercutee des deux cotes.
// ----------------------------------------------------------------------------------------------------
namespace safecase {
constexpr double kL = 1.0;        // domaine [0,L]^2 periodique
constexpr double kGamma = 1.4;    // gaz parfait
constexpr double kRho0 = 1.0;     // densite uniforme (rho>0 garanti)
constexpr double kP0 = 1.0;       // pression de fond
constexpr double kDp = 0.1;       // amplitude de la bulle de pression (faible)
constexpr double kSigma2 = 0.02;  // largeur^2 (en unites de L^2) de la gaussienne
constexpr double kCflForDt = 0.4;  // SEULEMENT pour deriver un dt FIXE (pas d'adaptatif : identite numerique)

// Borne de vitesse d'onde FIXE (c_max au pic de pression, vitesse nulle) -> dt deterministe identique
// sur les 3 fronts. dt = kCflForDt * (L/n) / wmax. Aucune reduction CFL par pas.
inline double wmax() { return std::sqrt(kGamma * (kP0 + kDp) / kRho0); }
inline double dt_for(int n) { return kCflForDt * (kL / n) / wmax(); }
}  // namespace safecase

#ifndef ADC_BUILD_SHA
#define ADC_BUILD_SHA "unknown"  // injecte par CMake (-DADC_BUILD_SHA=...) via run_frontend.sh
#endif
#ifndef ADC_BUILD_BRANCH
#define ADC_BUILD_BRANCH "unknown"
#endif

using SafeEuler = CompositeModel<Euler, NoSource, ChargeDensity>;

// Etat conservatif U = (rho, rho u, rho v, E) : bulle de pression lisse au centre, gaz au repos.
// Pendant C++ EXACT de adc_cases.common.initial_conditions.euler_pressure_blob (convention field[j,i]
// cote Python ; ici a(i,j,c)). rho = rho0 (>0), u = v = 0, p = p0 + dp exp(-r^2/(sigma2 L^2)) (>0),
// E = p/(gamma-1).
static void init_pressure_blob(MultiFab& U, const Geometry& geom) {
  using namespace safecase;
  const double cx = 0.5 * (geom.xlo + geom.xhi), cy = 0.5 * (geom.ylo + geom.yhi);
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 a = U.fab(li).array();
    const Box2D v = U.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        const double x = geom.x_cell(i) - cx, y = geom.y_cell(j) - cy;
        const double r2 = x * x + y * y;
        const double p = kP0 + kDp * std::exp(-r2 / (kSigma2 * kL * kL));
        a(i, j, 0) = Real(kRho0);                       // rho
        a(i, j, 1) = Real(0);                           // rho u
        a(i, j, 2) = Real(0);                           // rho v
        a(i, j, 3) = Real(p / (kGamma - 1.0));          // E (v = 0)
      }
  }
}

// Un pas SSPRK2 a dt FIXE, instrumente phase par phase. `with_poisson` : solve elliptique inerte
// (charge=0) + derivation aux, comme System::step quand un Poisson est branche ; sinon transport pur.
static PhaseTimers one_step(SafeEuler& model, MultiFab& U, MultiFab& aux, GeometricMG* mg,
                            const Geometry& geom, const Box2D& dom, const Periodicity& per,
                            double dt, const BoxArray& ba, const DistributionMapping& dm,
                            bool with_poisson) {
  PhaseTimers t;
  const Real dx = geom.dx(), dy = geom.dy();
  const int ncomp = SafeEuler::n_vars;

  if (with_poisson && mg) {
    MultiFab& rhs = mg->rhs();
    MultiFab& phi = mg->phi();
    // (1) assemblage rhs = q n (q=0 -> rhs identiquement nul, mais la boucle hote a le meme cout)
    t.aux_derive += timed([&] {
      rhs.set_val(Real(0));
      for (int li = 0; li < rhs.local_size(); ++li) {
        Array4 r = rhs.fab(li).array();
        const ConstArray4 u = U.fab(li).const_array();
        const Box2D b = rhs.box(li);
        for (int j = b.lo[1]; j <= b.hi[1]; ++j)
          for (int i = b.lo[0]; i <= b.hi[0]; ++i)
            r(i, j) += model.elliptic_rhs(load_state<SafeEuler>(u, i, j));
      }
    });
    // (2) solve elliptique
    t.poisson += timed([&] { mg->solve(); });
    // (3) derivation aux = (phi, grad phi) par cellule
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
    // (4) halos de aux
    t.halos += timed([&] { fill_boundary(aux, dom, per); });
  }

  // --- avance SSPRK2 (dt FIXE) : etage temporaire R + etat intermediaire U1 (alloc par pas) --------
  MultiFab R, U1;
  t.alloc_tmp += timed([&] {
    R = MultiFab(ba, dm, ncomp, 0);
    U1 = MultiFab(ba, dm, ncomp, U.n_grow());
  });

  const SafeEuler& m = model;
  const bool prim = false;  // reconstruction CONSERVATIVE (recon="conservative" cote Python)
  auto assemble = [&](MultiFab& Uin, MultiFab& Rout) {
    t.halos += timed([&] { fill_boundary(Uin, dom, per); });
    t.transport +=
        timed([&] { assemble_rhs<Minmod, RusanovFlux>(m, Uin, aux, geom, Rout, prim); });
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
          for (int i = v.lo[0]; i <= v.hi[0]; ++i) u1(i, j, c) = u(i, j, c) + Real(dt) * r(i, j, c);
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
            u(i, j, c) = Real(0.5) * u(i, j, c) + Real(0.5) * (u1(i, j, c) + Real(dt) * r(i, j, c));
    }
  });

  t.fence += timed([&] { device_fence(); });
  return t;
}

// Invariants physiques agreges (somme/min sur tous les rangs). nan = au moins un non-fini.
struct Invariants {
  double mass = 0, rho_min = 0, p_min = 0;
  bool nan = false;
};
static Invariants diagnose(const SafeEuler& model, const MultiFab& U) {
  double mass = 0, rho_min = 1e300, p_min = 1e300;
  bool finite = true;
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const Box2D v = U.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        const auto s = load_state<SafeEuler>(u, i, j);
        const double rho = s[0];
        const double p = model.pressure(s);
        mass += rho;
        rho_min = std::min(rho_min, rho);
        p_min = std::min(p_min, p);
        for (int c = 0; c < SafeEuler::n_vars; ++c)
          if (!std::isfinite(double(s[c]))) finite = false;
      }
  }
  Invariants inv;
  inv.mass = all_reduce_sum(mass);
  inv.rho_min = -all_reduce_max(-rho_min);  // pas de all_reduce_min : min(x) = -max(-x)
  inv.p_min = -all_reduce_max(-p_min);
  char bad = finite ? 0 : 1;
  all_reduce_or_inplace(&bad, 1);
  inv.nan = (bad != 0);
  return inv;
}

int main(int argc, char** argv) {
  comm_init(&argc, &argv);

  int n = 256, steps = 50, warmup = 5;
  std::string poisson = "off", backend = "serial", machine = "unknown";
  double dt_override = -1.0;
  for (int a = 1; a < argc; ++a) {
    using adc::bench::eat;  // consomme un argument "--cle valeur" (avance a, convertit selon le type)
    if (eat(argc, argv, a, "--n", n)) continue;
    if (eat(argc, argv, a, "--steps", steps)) continue;
    if (eat(argc, argv, a, "--warmup", warmup)) continue;
    if (eat(argc, argv, a, "--poisson", poisson)) continue;
    if (eat(argc, argv, a, "--dt", dt_override)) continue;
    if (eat(argc, argv, a, "--backend", backend)) continue;  // libelle informatif (le vrai backend = celui du build)
    if (eat(argc, argv, a, "--machine", machine)) continue;
  }
  const bool with_poisson = (poisson == "on");
  const double dt = (dt_override > 0) ? dt_override : safecase::dt_for(n);

  // --- maillage UNE box repartie round-robin (comme System / profile_step) ----------------------
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, safecase::kL, 0.0, safecase::kL};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, n_ranks());
  Periodicity per{true, true};
  BCRec pbc;           // Poisson periodique

  // (1) construction du modele -- etage cold-cache "model_build"
  SafeEuler model{Euler{safecase::kGamma}, NoSource{}, ChargeDensity{0.0}};
  double t_model_build = 0;
  {
    const auto t0 = Clock::now();
    SafeEuler probe{Euler{safecase::kGamma}, NoSource{}, ChargeDensity{0.0}};
    (void)probe;
    t_model_build = std::chrono::duration<double>(Clock::now() - t0).count();
  }

  // (2) allocation etat + solveur -- etage "addblock"
  const int n_ghost = 2;  // minmod
  MultiFab U, aux;
  GeometricMG* mg = nullptr;
  GeometricMG mg_storage(geom, ba, pbc, std::function<bool(Real, Real)>{});
  double t_addblock = timed([&] {
    U = MultiFab(ba, dm, SafeEuler::n_vars, n_ghost);
    aux = MultiFab(ba, dm, kAuxBaseComps, 1);
    U.set_val(Real(0));
    aux.set_val(Real(0));
    if (with_poisson) mg = &mg_storage;
  });

  // (3) initialisation de l'etat -- etage "state_init"
  double t_state_init = timed([&] {
    init_pressure_blob(U, geom);
    fill_boundary(U, dom, per);
  });

  // (4) premier pas (inclut premier kernel / premier V-cycle) -- etage "first_step"
  double t_first_step =
      timed([&] { one_step(model, U, aux, mg, geom, dom, per, dt, ba, dm, with_poisson); });

  // (5) warmup (hors mesure chaude)
  double t_warmup = timed([&] {
    for (int s = 0; s < warmup; ++s)
      one_step(model, U, aux, mg, geom, dom, per, dt, ba, dm, with_poisson);
  });

  // (6) boucle de run chronometree -- echantillon ms/pas + phases
  PhaseTimers acc;
  std::vector<double> ms_per_step;
  ms_per_step.reserve(steps);
  const auto wall0 = Clock::now();
  for (int s = 0; s < steps; ++s) {
    const auto s0 = Clock::now();
    PhaseTimers t = one_step(model, U, aux, mg, geom, dom, per, dt, ba, dm, with_poisson);
    device_fence();
    ms_per_step.push_back(1e3 * std::chrono::duration<double>(Clock::now() - s0).count());
    acc.add(t);
  }
  device_fence();
  const double t_run_loop = std::chrono::duration<double>(Clock::now() - wall0).count();

  // (7) diagnostics finals -- etage "diag"
  Invariants inv;
  double t_diag = timed([&] { inv = diagnose(model, U); });

  // --- agregation MPI : MAX sur les rangs (chemin critique d'un pas collectif) -------------------
  auto rmax = [](double x) { return all_reduce_max(x); };
  const double poisson_ms = 1e3 * rmax(acc.poisson) / steps;
  const double aux_ms = 1e3 * rmax(acc.aux_derive) / steps;
  const double halos_ms = 1e3 * rmax(acc.halos) / steps;
  const double transport_ms = 1e3 * rmax(acc.transport) / steps;
  const double reduction_ms = 1e3 * rmax(acc.reduction) / steps;
  const double fence_ms = 1e3 * rmax(acc.fence) / steps;
  const double alloc_ms = 1e3 * rmax(acc.alloc_tmp) / steps;

  // statistiques hot-loop (sur le wall par pas, rang 0 ; agrege en MAX pour median/p10/p90)
  const double med = percentile(ms_per_step, 0.5);
  const double p10 = percentile(ms_per_step, 0.10);
  const double p90 = percentile(ms_per_step, 0.90);
  double mean = 0;
  for (double x : ms_per_step) mean += x;
  mean /= std::max<size_t>(ms_per_step.size(), 1);
  double var = 0;
  for (double x : ms_per_step) var += (x - mean) * (x - mean);
  var /= std::max<size_t>(ms_per_step.size(), 1);
  const double cv = mean > 0 ? std::sqrt(var) / mean : 0.0;

  const double med_max = rmax(med);
  // rmax() == all_reduce_max() = COLLECTIVE : TOUS les rangs l'appellent (jamais sous if(rank0),
  // sinon deadlock). On hisse p10/p90 hors du printf rang 0.
  const double p10_max = rmax(p10);
  const double p90_max = rmax(p90);
  const long long cells = static_cast<long long>(n) * n;
  const double cells_per_s = med_max > 0 ? (double(cells) * n_ranks() / (med_max / 1e3)) : 0.0;

  const double total_cold =
      rmax(t_model_build + t_addblock + t_state_init + t_first_step + t_warmup + t_run_loop +
           t_diag);

  if (my_rank() == 0) {
    // Une ligne JSON (schema adc_perf_v1) consommee par perf/plot_frontend.py. import/dsl_compile = 0.
    std::printf(
        "{\"schema\":\"adc_perf_v1\",\"front\":\"cpp\","
        "\"adc_cpp_sha\":\"%s\",\"adc_cpp_branch\":\"%s\",\"adc_cases_sha\":null,"
        "\"backend\":\"%s\",\"machine\":\"%s\",\"ranks\":%d,\"threads\":%d,\"gpus\":%d,"
        "\"nx\":%d,\"ny\":%d,\"boxes\":1,\"max_grid\":%d,"
        "\"workload\":\"euler_safe\",\"limiter\":\"minmod\",\"flux\":\"rusanov\","
        "\"recon\":\"conservative\",\"time\":\"ssprk2\",\"poisson\":\"%s\",\"dt\":%.10e,"
        "\"warmup\":%d,\"steps\":%d,"
        "\"stages\":{\"import\":0.0,\"model_build\":%.6e,\"dsl_compile\":0.0,\"addblock\":%.6e,"
        "\"state_init\":%.6e,\"first_step\":%.6e,\"warmup\":%.6e,\"run_loop\":%.6e,\"diag\":%.6e},"
        "\"total_cold_user_s\":%.6e,"
        "\"hot_ms_per_step\":{\"median\":%.6e,\"p10\":%.6e,\"p90\":%.6e,\"cv\":%.6e},"
        "\"phases_ms_per_step\":{\"poisson\":%.6e,\"aux_derive\":%.6e,\"halos\":%.6e,"
        "\"transport\":%.6e,\"reduction\":%.6e,\"fence\":%.6e,\"alloc_tmp\":%.6e},"
        "\"cells_per_s\":%.6e,"
        "\"invariants\":{\"mass\":%.10e,\"rho_min\":%.6e,\"p_min\":%.6e,\"nan\":%s}}\n",
        ADC_BUILD_SHA, ADC_BUILD_BRANCH, backend.c_str(), machine.c_str(), n_ranks(),
        std::atoi(std::getenv("OMP_NUM_THREADS") ? std::getenv("OMP_NUM_THREADS") : "1"), 0, n, n, n,
        poisson.c_str(), dt, warmup, steps, t_model_build, t_addblock, t_state_init, t_first_step,
        t_warmup, t_run_loop, t_diag, total_cold, med_max, p10_max, p90_max, cv, poisson_ms,
        aux_ms, halos_ms, transport_ms, reduction_ms, fence_ms, alloc_ms, cells_per_s, inv.mass,
        inv.rho_min, inv.p_min, inv.nan ? "true" : "false");
  }

  comm_finalize();
  return 0;
}
