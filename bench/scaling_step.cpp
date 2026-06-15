// Driver de SCALING (campagne de perf, cf. adc_cases/perf). Mesure le debit (cells/s) et le ms/pas
// d'un noyau a grande taille, decoupe MULTI-BOX (contrairement a profile_step / frontend_cpp qui sont
// mono-box, fideles a System). Le multi-box exerce les vrais halos MPI (fill_boundary entre boites) et
// l'equilibrage round-robin. Trois charges, choisies par --workload :
//
//   transport : Euler compressible PUR (SafeEuler), SSPRK2 a dt fixe, AUCUN Poisson dans la boucle
//               chronometree -> isole le cout transport FV pur. Taille forte typique 4096x4096.
//   poisson   : un solve elliptique GeometricMG isole (RHS lisse fixe, phi remis a 0 par iteration)
//               -> isole le cout du V-cycle. Taille forte typique 1024x1024.
//   amr       : NON IMPLEMENTE dans ce binaire (l'entree AmrSystem C++ depuis un bench n'est pas
//               cablee, et le chemin generique ne compile pas sous nvcc, cf. memoire). Emet une ligne
//               de diagnostic explicite plutot que de FAUX chiffres -- le scaling AMR est un suivi.
//
// Strong vs weak : ce binaire joue UNE grille ; le balayage (n, ranks, threads, GPUs) est pilote par
// bench/run_scaling.sh. Le libelle --scaling est juste recopie dans le JSON. Compile/lance avec le MEME
// backend que la lib (Serie / Kokkos OpenMP / Cuda) et sous MPI sans changement. ZERO optimisation.

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>  // dot (brique des reductions)
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/elliptic/geometric_mg.hpp>
#include <adc/numerics/spatial_operator.hpp>
#include <adc/parallel/comm.hpp>
#include <adc/physics/bricks.hpp>
#include <adc/physics/euler.hpp>

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
using Clock = std::chrono::steady_clock;
static constexpr double kPi = 3.14159265358979323846;

#ifndef ADC_BUILD_SHA
#define ADC_BUILD_SHA "unknown"
#endif
#ifndef ADC_BUILD_BRANCH
#define ADC_BUILD_BRANCH "unknown"
#endif

using SafeEuler = CompositeModel<Euler, NoSource, ChargeDensity>;
static constexpr double kGamma = 1.4;

template <class F>
double timed_s(F&& f) {
  device_fence();
  const auto t0 = Clock::now();
  f();
  device_fence();
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

static double percentile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double idx = q * (v.size() - 1);
  const size_t lo = static_cast<size_t>(idx);
  const size_t hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (idx - lo) * (v[hi] - v[lo]);
}

// Etat Euler initial lisse (densite uniforme, bulle de pression) pour le transport.
static void init_blob(MultiFab& U, const Geometry& geom) {
  const double cx = 0.5 * (geom.xlo + geom.xhi), cy = 0.5 * (geom.ylo + geom.yhi);
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 a = U.fab(li).array();
    const Box2D v = U.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        const double x = geom.x_cell(i) - cx, y = geom.y_cell(j) - cy;
        const double p = 1.0 + 0.1 * std::exp(-(x * x + y * y) / 0.02);
        a(i, j, 0) = Real(1.0);
        a(i, j, 1) = Real(0);
        a(i, j, 2) = Real(0);
        a(i, j, 3) = Real(p / (kGamma - 1.0));
      }
  }
}

// Un pas transport pur (SSPRK2, dt fixe, sans Poisson), multi-box. Accumule les temps par phase :
// halos, transport (operateur FV + lincomb), alloc_tmp (MultiFab temporaires par pas), reduction (dot).
static void transport_step(const SafeEuler& m, MultiFab& U, MultiFab& aux, const Geometry& geom,
                           const Box2D& dom, const Periodicity& per, double dt, const BoxArray& ba,
                           const DistributionMapping& dm, double& halos_s, double& transport_s,
                           double& alloc_s, double& reduction_s) {
  const int ncomp = SafeEuler::n_vars;
  MultiFab R, U1;
  alloc_s += timed_s([&] {
    R = MultiFab(ba, dm, ncomp, 0);
    U1 = MultiFab(ba, dm, ncomp, U.n_grow());
  });
  const bool prim = false;
  auto assemble = [&](MultiFab& Uin, MultiFab& Rout) {
    halos_s += timed_s([&] { fill_boundary(Uin, dom, per); });
    transport_s += timed_s([&] { assemble_rhs<Minmod, RusanovFlux>(m, Uin, aux, geom, Rout, prim); });
  };
  assemble(U, R);
  transport_s += timed_s([&] {
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
  assemble(U1, R);
  transport_s += timed_s([&] {
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
  // reduction (brique des produits scalaires / diagnostics) : un dot global par pas.
  reduction_s += timed_s([&] {
    volatile Real s = dot(U, U, 0);
    (void)s;
  });
}

// Remplit un RHS lisse periodique a moyenne nulle (solubilite) : sin(2pi x) sin(2pi y).
static void fill_smooth_rhs(MultiFab& rhs, const Geometry& geom) {
  for (int li = 0; li < rhs.local_size(); ++li) {
    Array4 r = rhs.fab(li).array();
    const Box2D b = rhs.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        r(i, j) = Real(std::sin(2 * kPi * geom.x_cell(i)) * std::sin(2 * kPi * geom.y_cell(j)));
  }
}

// Checkpoint de debug rang-tagge (gate par $ADC_DBG) pour localiser un deadlock multi-rang.
static void dbg(const char* m) {
  if (std::getenv("ADC_DBG")) {
    std::fprintf(stderr, "[r%d] %s\n", my_rank(), m);
    std::fflush(stderr);
  }
}

int main(int argc, char** argv) {
  comm_init(&argc, &argv);

  std::string workload = "transport", scaling = "strong", backend = "serial", machine = "unknown";
  int n = 4096, steps = 20, warmup = 3, max_grid = 256;
  for (int a = 1; a < argc; ++a) {
    auto eat = [&](const char* key, auto& out) {
      if (std::strcmp(argv[a], key) == 0 && a + 1 < argc) {
        using T = std::decay_t<decltype(out)>;
        if constexpr (std::is_same_v<T, std::string>)
          out = argv[++a];
        else
          out = std::atoi(argv[++a]);
        return true;
      }
      return false;
    };
    if (eat("--workload", workload)) continue;
    if (eat("--scaling", scaling)) continue;
    if (eat("--n", n)) continue;
    if (eat("--steps", steps)) continue;
    if (eat("--warmup", warmup)) continue;
    if (eat("--max-grid", max_grid)) continue;
    if (eat("--backend", backend)) continue;
    if (eat("--machine", machine)) continue;
  }

  const int threads = std::atoi(std::getenv("OMP_NUM_THREADS") ? std::getenv("OMP_NUM_THREADS") : "1");

  if (workload == "amr") {
    if (my_rank() == 0)
      std::printf(
          "{\"schema\":\"adc_perf_v1\",\"front\":\"cpp_scaling\",\"workload\":\"amr\","
          "\"status\":\"not_implemented\",\"reason\":\"AmrSystem C++ entry not wired into bench; "
          "generic AMR path does not compile under nvcc (see notes). Run AMR scaling via a dedicated "
          "AmrSystem program.\",\"adc_cpp_sha\":\"%s\",\"adc_cpp_branch\":\"%s\",\"n\":%d}\n",
          ADC_BUILD_SHA, ADC_BUILD_BRANCH, n);
    comm_finalize();
    return 0;
  }

  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, max_grid);
  DistributionMapping dm(ba.size(), n_ranks());
  Periodicity per{true, true};
  BCRec pbc;

  std::vector<double> ms;
  ms.reserve(steps);
  double halos_acc = 0, transport_acc = 0, poisson_acc = 0, alloc_acc = 0, reduction_acc = 0;
  double diag_s = 0;
  char invbuf[192];
  std::snprintf(invbuf, sizeof(invbuf), "{\"nan\":false}");

  if (workload == "transport") {
    SafeEuler model{Euler{kGamma}, NoSource{}, ChargeDensity{0.0}};
    MultiFab U(ba, dm, SafeEuler::n_vars, 2), aux(ba, dm, kAuxBaseComps, 1);
    { char b[96]; std::snprintf(b, sizeof b, "mesh boxes=%d local=%d", ba.size(), U.local_size()); dbg(b); }
    U.set_val(Real(0));
    aux.set_val(Real(0));
    init_blob(U, geom);
    dbg("init fill_boundary BEGIN");
    fill_boundary(U, dom, per);
    dbg("init fill_boundary END");
    const double dt = 0.4 * (1.0 / n) / std::sqrt(kGamma * 1.1 / 1.0);  // dt fixe deterministe
    double hh = 0, tt = 0, aa = 0, rr = 0;
    dbg("warmup BEGIN");
    for (int s = 0; s < warmup; ++s) {
      transport_step(model, U, aux, geom, dom, per, dt, ba, dm, hh, tt, aa, rr);
      { char b[64]; std::snprintf(b, sizeof b, "warmup step %d done", s); dbg(b); }
    }
    dbg("warmup END");
    for (int s = 0; s < steps; ++s) {
      double h = 0, t = 0, a = 0, r = 0;
      const auto s0 = Clock::now();
      transport_step(model, U, aux, geom, dom, per, dt, ba, dm, h, t, a, r);
      device_fence();
      ms.push_back(1e3 * std::chrono::duration<double>(Clock::now() - s0).count());
      halos_acc += h;
      transport_acc += t;
      alloc_acc += a;
      reduction_acc += r;
    }
    diag_s = timed_s([&] {
      double mass = 0, rmn = 1e300, pmn = 1e300;
      bool fin = true;
      for (int li = 0; li < U.local_size(); ++li) {
        const ConstArray4 u = U.fab(li).const_array();
        const Box2D v = U.box(li);
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
            const auto st = load_state<SafeEuler>(u, i, j);
            mass += st[0];
            rmn = std::min(rmn, double(st[0]));
            pmn = std::min(pmn, double(model.pressure(st)));
            for (int c = 0; c < SafeEuler::n_vars; ++c)
              if (!std::isfinite(double(st[c]))) fin = false;
          }
      }
      mass = all_reduce_sum(mass);
      rmn = -all_reduce_max(-rmn);
      pmn = -all_reduce_max(-pmn);
      char bad = fin ? 0 : 1;
      all_reduce_or_inplace(&bad, 1);
      std::snprintf(invbuf, sizeof(invbuf),
                    "{\"mass\":%.10e,\"rho_min\":%.6e,\"p_min\":%.6e,\"nan\":%s}", mass, rmn, pmn,
                    bad ? "true" : "false");
    });
  } else if (workload == "poisson") {
    GeometricMG mg(geom, ba, pbc, std::function<bool(Real, Real)>{});
    fill_smooth_rhs(mg.rhs(), geom);
    for (int s = 0; s < warmup; ++s) {
      mg.phi().set_val(Real(0));
      mg.solve();
    }
    for (int s = 0; s < steps; ++s) {
      mg.phi().set_val(Real(0));
      const auto s0 = Clock::now();
      double dts = timed_s([&] { mg.solve(); });
      ms.push_back(1e3 * std::chrono::duration<double>(Clock::now() - s0).count());
      poisson_acc += dts;
    }
    diag_s = timed_s([&] {
      MultiFab& phi = mg.phi();
      double mn = 1e300, mx = -1e300;
      bool fin = true;
      for (int li = 0; li < phi.local_size(); ++li) {
        const ConstArray4 p = phi.fab(li).const_array();
        const Box2D v = phi.box(li);
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
            const double val = p(i, j);
            mn = std::min(mn, val);
            mx = std::max(mx, val);
            if (!std::isfinite(val)) fin = false;
          }
      }
      mn = -all_reduce_max(-mn);
      mx = all_reduce_max(mx);
      char bad = fin ? 0 : 1;
      all_reduce_or_inplace(&bad, 1);
      std::snprintf(invbuf, sizeof(invbuf), "{\"phi_min\":%.6e,\"phi_max\":%.6e,\"nan\":%s}", mn, mx,
                    bad ? "true" : "false");
    });
  } else {
    if (my_rank() == 0)
      std::fprintf(stderr, "workload inconnu : %s (transport|poisson|amr)\n", workload.c_str());
    comm_finalize();
    return 2;
  }

  auto rmax = [](double x) { return all_reduce_max(x); };
  const double med = rmax(percentile(ms, 0.5));
  const double p10 = rmax(percentile(ms, 0.10));
  const double p90 = rmax(percentile(ms, 0.90));
  double mean = 0;
  for (double x : ms) mean += x;
  mean /= std::max<size_t>(ms.size(), 1);
  double var = 0;
  for (double x : ms) var += (x - mean) * (x - mean);
  var /= std::max<size_t>(ms.size(), 1);
  const double cv = mean > 0 ? std::sqrt(var) / mean : 0.0;

  // Debit GLOBAL : cellules du domaine entier / temps median par pas. Pour le weak scaling, le
  // lanceur fait croitre n avec les ressources ; cells/s global doit alors rester ~constant.
  const long long cells = static_cast<long long>(n) * n;
  const double cells_per_s = med > 0 ? double(cells) / (med / 1e3) : 0.0;
  // FIX DEADLOCK np>=2 : rmax() == all_reduce_max() est une COLLECTIVE -> TOUS les rangs doivent
  // l'appeler, le MEME nombre de fois. Les appeler dans le printf sous `if (my_rank()==0)` ne les
  // executait que sur le rang 0 -> les autres rangs filaient vers comm_finalize() -> rang 0 bloque
  // a jamais dans MPI_Allreduce. On hisse donc TOUTES les reductions hors du bloc rang 0.
  const double halos_ms = 1e3 * rmax(halos_acc) / steps;
  const double transport_ms = 1e3 * rmax(transport_acc) / steps;
  const double poisson_ms = 1e3 * rmax(poisson_acc) / steps;
  const double reduction_ms = 1e3 * rmax(reduction_acc) / steps;
  const double alloc_ms = 1e3 * rmax(alloc_acc) / steps;
  const double diag_ms = 1e3 * rmax(diag_s);

  if (my_rank() == 0) {
    std::printf(
        "{\"schema\":\"adc_perf_v1\",\"front\":\"cpp_scaling\","
        "\"adc_cpp_sha\":\"%s\",\"adc_cpp_branch\":\"%s\","
        "\"backend\":\"%s\",\"machine\":\"%s\",\"ranks\":%d,\"threads\":%d,\"gpus\":%d,"
        "\"workload\":\"%s\",\"scaling\":\"%s\",\"nx\":%d,\"ny\":%d,\"boxes\":%d,\"max_grid\":%d,"
        "\"warmup\":%d,\"steps\":%d,"
        "\"hot_ms_per_step\":{\"median\":%.6e,\"p10\":%.6e,\"p90\":%.6e,\"cv\":%.6e},"
        "\"phases_ms_per_step\":{\"halos\":%.6e,\"transport\":%.6e,\"poisson\":%.6e,"
        "\"reduction\":%.6e,\"alloc_tmp\":%.6e,\"diag\":%.6e},"
        "\"cells_per_s\":%.6e,\"invariants\":%s}\n",
        ADC_BUILD_SHA, ADC_BUILD_BRANCH, backend.c_str(), machine.c_str(), n_ranks(), threads, 0,
        workload.c_str(), scaling.c_str(), n, n, ba.size(), max_grid, warmup, steps, med, p10, p90,
        cv, halos_ms, transport_ms, poisson_ms, reduction_ms, alloc_ms, diag_ms, cells_per_s, invbuf);
  }

  comm_finalize();
  return 0;
}
