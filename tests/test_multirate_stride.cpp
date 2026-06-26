// Multirate par CADENCE (retour tuteur sec.8.2 C) : une espece "lente" (un gaz) n'est pas
// resolue a chaque pas. `stride` = elle n'avance qu'1 macro-pas sur stride, alors d'un pas
// effectif stride*dt (elle rattrape le temps). Au total elle avance autant, mais calculee
// stride fois moins souvent.
//
// Ici : bloc rapide (stride 1) + bloc lent (stride 3), production constante. Apres 1 pas le
// lent a deja fait son pas de 3*dt (!= rapide) ; apres 3 pas les deux sont synchronises.

#include <pops/core/model/coupled_system.hpp>
#include <pops/core/state/state.hpp>
#include <pops/coupling/system/system_coupler.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <cmath>
#include <cstdio>

using namespace pops;

struct Production {
  using State = StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;
  Real rate = Real(1);
  POPS_HD State flux(const State&, const Aux&, int) const { return State{}; }
  POPS_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  POPS_HD State source(const State&, const Aux&) const { return State{rate}; }
  POPS_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

struct ZeroSystemRhs {
  template <class System>
  void operator()(const System&, MultiFab& rhs) const {
    rhs.set_val(Real(0));
  }
};

using FastBlk = EquationBlock<Production, FirstOrder, ExplicitTime<SSPRK2, 1, 1>>;  // stride 1
using SlowBlk = EquationBlock<Production, FirstOrder, ExplicitTime<SSPRK2, 1, 3>>;  // stride 3
static_assert(FastBlk::Time::stride == 1);
static_assert(SlowBlk::Time::stride == 3);

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const Box2D dom = Box2D::from_extents(4, 4);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const BoxArray ba(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  const int ncell = 16;
  BCRec bc;
  const Real dt = Real(0.1);

  MultiFab Uf(ba, dm, 1, 2), Us(ba, dm, 1, 2);
  Uf.set_val(Real(0));
  Us.set_val(Real(0));
  FastBlk fast{"fast", Production{Real(1)}, Uf, bc};
  SlowBlk slow{"slow", Production{Real(1)}, Us, bc};
  CoupledSystem system{fast, slow};
  auto sim = make_system_coupler(system, geom, ba, bc, ZeroSystemRhs{});

  // 1er macro-pas : rapide avance de dt (0.1) ; lent avance de 3*dt (0.3) en une fois.
  sim.step(dt);
  chk(std::fabs(sum(Uf, 0) - Real(0.1) * ncell) < Real(1e-12), "fast_after_1");
  chk(std::fabs(sum(Us, 0) - Real(0.3) * ncell) < Real(1e-12), "slow_after_1_big_step");

  // 2e et 3e macro-pas : rapide +0.1 chacun ; lent saute (1%3, 2%3 != 0) -> reste 0.3.
  sim.step(dt);
  chk(std::fabs(sum(Us, 0) - Real(0.3) * ncell) < Real(1e-12), "slow_skips_step2");
  sim.step(dt);

  // Apres 3 macro-pas : les deux ont avance d'un temps total 3*dt -> synchronises a 0.3.
  chk(std::fabs(sum(Uf, 0) - Real(0.3) * ncell) < Real(1e-12), "fast_after_3");
  chk(std::fabs(sum(Us, 0) - Real(0.3) * ncell) < Real(1e-12), "slow_after_3_synced");

  if (fails == 0)
    std::printf("OK test_multirate_stride\n");
  return fails == 0 ? 0 : 1;
}
