// Multirate PLEINEMENT ADAPTATIF (sec.8.2 C) : SystemDriver::step_adaptive(cfl). Le pas macro
// est fixe par l'espece la plus rapide (CFL) ; le `stride` de chaque espece est derive AU
// RUNTIME du ratio w_max/w_s. Une espece 4x plus lente avance donc 1 fois sur 4, par un pas
// 4x plus grand (= son dt stable), en une seule resolution.
//
// Modele transport (fixe la vitesse d'onde -> le stride) + production constante (observable).
// Champ uniforme : l'advection ne change rien, seule la source fait grandir n -> on lit
// directement le pas effectif de chaque espece.

#include <adc/core/model/coupled_system.hpp>
#include <adc/core/state/state.hpp>
#include <adc/coupling/system/system_coupler.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/layout/distribution_mapping.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/multifab.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace adc;

struct AdvectProduce {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  Real a = Real(1), rate = Real(1);
  ADC_HD State flux(const State& u, const Aux&, int dir) const {
    return State{dir == 0 ? a * u[0] : Real(0)};
  }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return a < 0 ? -a : a; }
  ADC_HD State source(const State&, const Aux&) const { return State{rate}; }
  ADC_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

struct ZeroSystemRhs {
  template <class System>
  void operator()(const System&, MultiFab& rhs) const {
    rhs.set_val(Real(0));
  }
};

using Blk = EquationBlock<AdvectProduce, FirstOrder, ExplicitTime<SSPRK2, 1>>;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int n = 16;
  const Box2D dom = Box2D::from_extents(n, n);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const BoxArray ba(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  const int ncell = n * n;
  BCRec bc;

  MultiFab Uf(ba, dm, 1, 2), Us(ba, dm, 1, 2);
  Uf.set_val(Real(1));
  Us.set_val(Real(1));
  // rapide (a=4) -> stride 1 ; lent (a=1) -> stride floor(4/1)=4. Production rate=1.
  Blk fast{"fast", AdvectProduce{Real(4), Real(1)}, Uf, bc};
  Blk slow{"slow", AdvectProduce{Real(1), Real(1)}, Us, bc};
  CoupledSystem system{fast, slow};
  auto sim = make_system_coupler(system, geom, ba, bc, ZeroSystemRhs{});

  const Real cfl = Real(0.4);
  const Real h = std::min(geom.dx(), geom.dy());
  const Real macro_dt = cfl * h / Real(4);  // w_max = 4

  const Real dt = sim.step_adaptive(cfl);
  chk(std::fabs(dt - macro_dt) < Real(1e-14), "adaptive_macro_dt_from_fastest");
  // rapide : stride 1 -> avance de macro_dt ; production -> +macro_dt.
  // tol 1e-10 : arrondi SSPRK vs valeur directe, cumule sur 256 cellules (~1e-12).
  chk(std::fabs(sum(Uf, 0) - (Real(1) + macro_dt) * ncell) < Real(1e-10), "fast_one_macro_step");
  // lent : stride 4 -> avance de 4*macro_dt en une fois ; production -> +4*macro_dt.
  chk(std::fabs(sum(Us, 0) - (Real(1) + Real(4) * macro_dt) * ncell) < Real(1e-10),
      "slow_big_adaptive_step");

  if (fails == 0)
    std::printf("OK test_adaptive_multirate\n");
  return fails == 0 ? 0 : 1;
}
