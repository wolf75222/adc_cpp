// Benchmark TRANSPORT pur multi-box/MPI, hors Hoffart et hors Poisson.
//
// Objectif : mesurer un cas "sain" pour le scaling distribue. Contrairement a profile_step
// (system-like, mono-box, Poisson/MG dominant), ce harnais decoupe vraiment le domaine en boites,
// les distribue sur les rangs MPI, puis separe :
//   - halos : fill_boundary periodique sur les ghosts ;
//   - cfl_reduction : max_wave_speed_mf, donc reduction globale ;
//   - transport : assemble_rhs Euler FV + mises a jour SSPRK2 ;
//   - diagnostic_reduction : dot, proxy des reductions globales de diagnostics/Krylov ;
//   - fence : barriere device isolee.
//
// Le cas physique est Euler 2D periodique lisse, sans disque, sans Schur, sans AMR, sans Poisson.
// Il sert a comparer C++/Kokkos/MPI et frontends Python sans melanger le verrou scientifique Hoffart.

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/numerics/spatial_operator.hpp>
#include <adc/parallel/comm.hpp>
#include <adc/parallel/load_balance.hpp>
#include <adc/physics/bricks/bricks.hpp>
#include <adc/physics/fluids/euler.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>

using namespace adc;
using Clock = std::chrono::steady_clock;

namespace {

constexpr double kPi = 3.14159265358979323846;
using Model = CompositeModel<Euler, NoSource, BackgroundDensity>;

struct Timers {
  double halos = 0.0;
  double cfl_reduction = 0.0;
  double transport = 0.0;
  double diagnostic_reduction = 0.0;
  double fence = 0.0;

  double total() const {
    return halos + cfl_reduction + transport + diagnostic_reduction + fence;
  }

  void add(const Timers& other) {
    halos += other.halos;
    cfl_reduction += other.cfl_reduction;
    transport += other.transport;
    diagnostic_reduction += other.diagnostic_reduction;
    fence += other.fence;
  }
};

template <class F>
double timed(F&& f) {
  device_fence();
  const auto t0 = Clock::now();
  f();
  device_fence();
  const auto t1 = Clock::now();
  return std::chrono::duration<double>(t1 - t0).count();
}

template <class T>
bool eat_arg(int& a, int argc, char** argv, const char* key, T& out) {
  if (std::strcmp(argv[a], key) != 0 || a + 1 >= argc) return false;
  using U = std::decay_t<T>;
  if constexpr (std::is_same_v<U, std::string>) {
    out = argv[++a];
  } else if constexpr (std::is_same_v<U, double>) {
    out = std::atof(argv[++a]);
  } else {
    out = std::atoi(argv[++a]);
  }
  return true;
}

Euler::State smooth_euler_state(const Euler& eul, const Geometry& geom, int i, int j) {
  const double x = geom.x_cell(i);
  const double y = geom.y_cell(j);
  Euler::Prim p{};
  p[0] = 1.0 + 0.10 * std::sin(2.0 * kPi * x) * std::cos(2.0 * kPi * y);
  p[1] = 0.20 + 0.05 * std::sin(2.0 * kPi * y);
  p[2] = -0.10 + 0.05 * std::cos(2.0 * kPi * x);
  p[3] = 1.0 + 0.05 * std::sin(2.0 * kPi * (x + y));
  return eul.to_conservative(p);
}

void initialize_state(MultiFab& U, MultiFab& aux, const Geometry& geom, const Euler& eul) {
  U.sync_host();
  aux.sync_host();
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 u = U.fab(li).array();
    Array4 a = aux.fab(li).array();
    const Box2D valid = U.box(li);
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j) {
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i) {
        const Euler::State s = smooth_euler_state(eul, geom, i, j);
        for (int c = 0; c < Model::n_vars; ++c) u(i, j, c) = s[c];
        a(i, j, 0) = 0.0;
        a(i, j, 1) = 0.0;
        a(i, j, 2) = 0.0;
      }
    }
  }
  U.sync_device();
  aux.sync_device();
}

void update_stage1(MultiFab& U1, const MultiFab& U, const MultiFab& R, Real dt) {
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 u1 = U1.fab(li).array();
    ConstArray4 u = U.fab(li).const_array();
    ConstArray4 r = R.fab(li).const_array();
    const Box2D b = U.box(li);
    for (int c = 0; c < Model::n_vars; ++c) {
      for_each_cell(b, [=] ADC_HD(int i, int j) { u1(i, j, c) = u(i, j, c) + dt * r(i, j, c); });
    }
  }
}

void update_stage2(MultiFab& U, const MultiFab& U1, const MultiFab& R, Real dt) {
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 u = U.fab(li).array();
    ConstArray4 u1 = U1.fab(li).const_array();
    ConstArray4 r = R.fab(li).const_array();
    const Box2D b = U.box(li);
    for (int c = 0; c < Model::n_vars; ++c) {
      for_each_cell(b, [=] ADC_HD(int i, int j) {
        u(i, j, c) = Real(0.5) * u(i, j, c) + Real(0.5) * (u1(i, j, c) + dt * r(i, j, c));
      });
    }
  }
}

template <class Limiter>
void assemble_with(Model& model, MultiFab& U, MultiFab& aux, const Geometry& geom, MultiFab& R,
                   bool reconstruct_primitive) {
  assemble_rhs<Limiter, RusanovFlux>(model, U, aux, geom, R, reconstruct_primitive);
}

void assemble_selected(const std::string& limiter, Model& model, MultiFab& U, MultiFab& aux,
                       const Geometry& geom, MultiFab& R, bool reconstruct_primitive) {
  if (limiter == "none") {
    assemble_with<NoSlope>(model, U, aux, geom, R, reconstruct_primitive);
  } else if (limiter == "vanleer") {
    assemble_with<VanLeer>(model, U, aux, geom, R, reconstruct_primitive);
  } else if (limiter == "weno5") {
    assemble_with<Weno5>(model, U, aux, geom, R, reconstruct_primitive);
  } else {
    assemble_with<Minmod>(model, U, aux, geom, R, reconstruct_primitive);
  }
}

Timers one_step(Model& model, MultiFab& U, MultiFab& aux, MultiFab& R, MultiFab& U1,
                const Geometry& geom, const Box2D& dom, const Periodicity& per,
                const std::string& limiter, double cfl, bool reconstruct_primitive) {
  Timers t;
  t.halos += timed([&] { fill_boundary(aux, dom, per); });
  Real w = 0.0;
  t.cfl_reduction += timed([&] {
    w = std::max(max_wave_speed_mf(model, U, aux), Real(1.0e-30));
  });
  const Real dt = static_cast<Real>(cfl) * std::min(geom.dx(), geom.dy()) / w;

  auto assemble_stage = [&](MultiFab& stage) {
    t.halos += timed([&] { fill_boundary(stage, dom, per); });
    t.transport += timed([&] {
      assemble_selected(limiter, model, stage, aux, geom, R, reconstruct_primitive);
    });
  };

  assemble_stage(U);
  t.transport += timed([&] { update_stage1(U1, U, R, dt); });
  assemble_stage(U1);
  t.transport += timed([&] { update_stage2(U, U1, R, dt); });

  t.diagnostic_reduction += timed([&] {
    volatile Real s = dot(U, U, 0);
    (void)s;
  });
  t.fence += timed([&] { device_fence(); });
  return t;
}

}  // namespace

int main(int argc, char** argv) {
  comm_init(&argc, &argv);

  int n = 512;
  int max_grid = 128;
  int steps = 20;
  int warmup = 5;
  double cfl = 0.4;
  double length = 1.0;
  std::string limiter = "minmod";
  std::string distribution = "sfc";

  for (int a = 1; a < argc; ++a) {
    if (eat_arg(a, argc, argv, "--n", n) || eat_arg(a, argc, argv, "--max-grid", max_grid) ||
        eat_arg(a, argc, argv, "--steps", steps) || eat_arg(a, argc, argv, "--warmup", warmup) ||
        eat_arg(a, argc, argv, "--cfl", cfl) || eat_arg(a, argc, argv, "--L", length) ||
        eat_arg(a, argc, argv, "--limiter", limiter) ||
        eat_arg(a, argc, argv, "--distribution", distribution)) {
      continue;
    }
    if (my_rank() == 0) std::printf("Unknown argument: %s\n", argv[a]);
    comm_finalize();
    return 2;
  }

  const int me = my_rank();
  const int np = n_ranks();
  const Box2D dom = Box2D::from_extents(n, n);
  const Geometry geom{dom, 0.0, length, 0.0, length};
  const int ngrow = limiter == "weno5" ? 3 : (limiter == "none" ? 1 : 2);
  const BoxArray ba = BoxArray::from_domain(dom, max_grid);
  const DistributionMapping dm = distribution == "knapsack" ? make_knapsack_distribution(ba, np)
                                                             : make_sfc_distribution(ba, np);
  const Periodicity per{true, true};
  const Model model0{Euler{Real(1.4)}, NoSource{}, BackgroundDensity{Real(0), Real(0)}};
  Model model = model0;

  MultiFab U(ba, dm, Model::n_vars, ngrow);
  MultiFab aux(ba, dm, Model::n_aux, 1);
  MultiFab R(ba, dm, Model::n_vars, 0);
  MultiFab U1(ba, dm, Model::n_vars, ngrow);
  initialize_state(U, aux, geom, model.hyp);
  fill_boundary(U, dom, per);
  fill_boundary(aux, dom, per);

  for (int s = 0; s < warmup; ++s) {
    (void)one_step(model, U, aux, R, U1, geom, dom, per, limiter, cfl, true);
  }

  Timers total;
  const auto wall0 = Clock::now();
  for (int s = 0; s < steps; ++s) {
    total.add(one_step(model, U, aux, R, U1, geom, dom, per, limiter, cfl, true));
  }
  device_fence();
  const auto wall1 = Clock::now();
  const double wall_local = std::chrono::duration<double>(wall1 - wall0).count();
  const double wall = all_reduce_max(wall_local);

  const double local_cells = static_cast<double>([&] {
    long c = 0;
    for (int li = 0; li < U.local_size(); ++li) c += U.box(li).num_cells();
    return c;
  }());
  const double max_local_cells = all_reduce_max(local_cells);
  const double min_local_cells = -all_reduce_max(-local_cells);
  const double total_cells = all_reduce_sum(local_cells);
  const double checksum = sum(U, 0);

  if (me == 0) {
    auto print_phase = [&](const char* name, double seconds) {
      const double per_step_ms = steps > 0 ? 1000.0 * seconds / steps : 0.0;
      const double pct = total.total() > 0.0 ? 100.0 * seconds / total.total() : 0.0;
      std::printf("%-22s %.9e %.6f %.3f%%\n", name, seconds, per_step_ms, pct);
    };

    std::printf("# adc_cpp profile_transport_mbox\n");
    std::printf("# ranks=%d boxes=%d local_boxes_rank0=%d n=%d max_grid=%d steps=%d warmup=%d\n", np,
                ba.size(), U.local_size(), n, max_grid, steps, warmup);
    std::printf("# limiter=%s distribution=%s periodic=xy reconstruct=primitive checksum=%.17e\n",
                limiter.c_str(), distribution.c_str(), checksum);
    std::printf("# load_cells min=%.0f max=%.0f total=%.0f imbalance=%.6f\n", min_local_cells,
                max_local_cells, total_cells,
                total_cells > 0.0 ? max_local_cells / (total_cells / std::max(np, 1)) : 1.0);
    std::printf("# wall_total=%.9e s per_step=%.6f ms sum_phases=%.9e s\n", wall,
                steps > 0 ? 1000.0 * wall / steps : 0.0, total.total());
    std::printf("phase                  total_s per_step_ms pct\n");
    print_phase("halos", total.halos);
    print_phase("cfl_reduction", total.cfl_reduction);
    print_phase("transport", total.transport);
    print_phase("diagnostic_reduction", total.diagnostic_reduction);
    print_phase("fence", total.fence);
    print_phase("TOTAL", total.total());
  }

  comm_finalize();
  return 0;
}
