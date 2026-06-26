// ADC-262: the red-black Gauss-Seidel smoother (gs_rb_sweep, the 86%-dominant Poisson path) must
// REUSE the cached halo schedule (ADC-260) instead of re-enumerating the exchange jobs on every
// sweep. Each GeometricMG V-cycle runs nu1 + nu2 sweeps per level plus nbottom (=50) at the bottom,
// and each sweep does two fill_ghosts -> hundreds of fill_ghosts per cycle. With the cache engaged
// the halo schedule is BUILT once per distinct MultiFab layout (a handful of MG levels) and reused
// across all sweeps AND all cycles; without it the build count would grow with the sweep count.
//
// This locks the lever-(a) win measured for ADC-262 (the per-sweep re-enumeration is already gone via
// ADC-260): the decisive invariant is that running MORE V-cycles adds ZERO further schedule builds.
//
// SCOPE: this pins the schedule ENUMERATION cost (serial, single box). The ADC-260 cache does NOT
// remove the per-sweep pack/MPI exchange under MPI (the bottom level alone is ~nbottom*2 exchanges per
// V-cycle), so a distributed face-subset / fused-exchange smoother (lever b) remains a possible MPI
// follow-up; it is out of scope here. The `< 50` bound below assumes the default nbottom=50.

#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/boundary/halo_schedule.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>

#include <cmath>
#include <cstdio>

using namespace pops;

static constexpr double kPi = 3.14159265358979323846;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  // Dirichlet MMS problem big enough for a multi-level V-cycle (deep enough that nbottom=50 bottom
  // sweeps run): lap(phi) = -2 pi^2 sin(pi x) sin(pi y).
  const int n = 64;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;

  GeometricMG mg(geom, ba, bc);
  Array4 af = mg.rhs().fab(0).array();
  for_each_cell(dom, [af, geom](int i, int j) {
    af(i, j, 0) = -2 * kPi * kPi * std::sin(kPi * geom.x_cell(i)) * std::sin(kPi * geom.y_cell(j));
  });
  mg.phi().set_val(0.0);

  // One V-cycle builds the schedule for each level's phi layout (first fill_ghosts at that level);
  // every later sweep at that level reuses it.
  reset_halo_schedule_build_count();
  mg.vcycle();
  const std::int64_t builds_after_1 = halo_schedule_build_count();
  chk(builds_after_1 >= 1, "smoother does fill_ghosts (schedule built at least once)");

  // DECISIVE: four more V-cycles -- hundreds more sweeps, each with two fill_ghosts -- must add ZERO
  // schedule builds. If gs_rb_sweep re-enumerated the exchange per sweep, this would grow ~5x.
  for (int c = 0; c < 4; ++c)
    mg.vcycle();
  const std::int64_t builds_after_5 = halo_schedule_build_count();
  chk(builds_after_5 == builds_after_1,
      "RB-GS reuses the cached halo schedule across sweeps and cycles (no per-sweep "
      "re-enumeration)");

  // The first-cycle build count reflects distinct layouts (a few MG levels), NOT the sweep count: a
  // single cycle already runs nbottom=50 bottom sweeps, so a per-sweep rebuild would exceed 50.
  chk(builds_after_1 < 50, "schedule builds scale with MG levels, not with the sweep count");

  std::printf("  halo-schedule builds: after 1 vcycle = %lld, after 5 = %lld (reused)\n",
              static_cast<long long>(builds_after_1), static_cast<long long>(builds_after_5));

  // Sanity: the solve still converges (the cache must not have broken the smoother).
  const Real r0 = mg.current_residual();
  for (int c = 0; c < 20; ++c)
    mg.vcycle();
  chk(mg.current_residual() < 1e-6 * r0 || mg.current_residual() < 1e-10,
      "GeometricMG still converges with the cached smoother");

  if (fails == 0)
    std::printf("OK test_poisson_smoother_cache\n");
  return fails ? 1 : 0;
}
