// Pas macro choisi par CFL multi-especes (§8.2 C) : SystemDriver::step_cfl. Le pas est
// dt = cfl * min(dx,dy) / w_max, ou w_max est la plus grande vitesse d'onde sur TOUTES les
// especes -> l'espece la plus rapide contraint le pas. Combine au Stride d'une espece lente,
// cela donne le multirate pratique.

#include <adc/core/coupled_system.hpp>
#include <adc/core/state.hpp>
#include <adc/coupling/system_coupler.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace adc;

// Advection a vitesse constante a : vitesse d'onde max = |a|.
struct AdvectX {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  Real a = Real(1);
  ADC_HD State flux(const State& u, const Aux&, int dir) const {
    return State{dir == 0 ? a * u[0] : Real(0)};
  }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return a < 0 ? -a : a; }
  ADC_HD State source(const State&, const Aux&) const { return State{}; }
  ADC_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

struct ZeroSystemRhs {
  template <class System>
  void operator()(const System&, MultiFab& rhs) const { rhs.set_val(Real(0)); }
};

using Blk = EquationBlock<AdvectX, FirstOrder, ExplicitTime<SSPRK2, 1>>;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  const int n = 16;
  const Box2D dom = Box2D::from_extents(n, n);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const BoxArray ba(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  BCRec bc;

  MultiFab Uf(ba, dm, 1, 2), Us(ba, dm, 1, 2);
  Uf.set_val(Real(1));
  Us.set_val(Real(1));
  // espece rapide (a=2) + espece lente (a=0.5) : le pas CFL est fixe par la rapide.
  Blk fast{"fast", AdvectX{Real(2)}, Uf, bc};
  Blk slow{"slow", AdvectX{Real(0.5)}, Us, bc};
  CoupledSystem system{fast, slow};
  SystemCoupler sim(system, geom, ba, bc, ZeroSystemRhs{});

  const Real cfl = Real(0.4);
  const Real h = std::min(geom.dx(), geom.dy());
  const Real expected = cfl * h / Real(2);  // w_max = max(2, 0.5) = 2

  const Real dt = sim.step_cfl(cfl);
  chk(std::fabs(dt - expected) < Real(1e-14), "cfl_dt_from_fastest_species");
  chk(dt > Real(0), "cfl_dt_positive");
  // le systeme a avance (etat fini, masse conservee pour l'advection periodique).
  chk(std::fabs(sum(Uf, 0) - Real(1) * n * n) < Real(1e-10), "fast_mass_conserved");
  chk(std::fabs(sum(Us, 0) - Real(1) * n * n) < Real(1e-10), "slow_mass_conserved");

  if (fails == 0) std::printf("OK test_cfl_dt\n");
  return fails == 0 ? 0 : 1;
}
