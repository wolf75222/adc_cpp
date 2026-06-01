// Squelette architecture multi-blocs : PhysicalModel local, EquationBlock,
// CoupledSystem, scheduler par sous-pas, RHS elliptique multi-champs.

#include <adc/core/coupled_system.hpp>
#include <adc/core/state.hpp>
#include <adc/coupling/elliptic_rhs.hpp>
#include <adc/integrator/scheduler.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>

#include <cmath>
#include <cstdio>
#include <type_traits>

using namespace adc;

struct ElectronToy {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  ADC_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State&, const Aux&) const { return State{Real(0)}; }
  ADC_HD Real elliptic_rhs(const State& u) const { return -u[0]; }
};

struct IonToy {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  ADC_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State&, const Aux&) const { return State{Real(0)}; }
  ADC_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

using ElectronBlock =
    EquationBlock<ElectronToy, MusclVanLeerHLLC, ImplicitTime<UserTimeIntegrator, 10>>;
using IonBlock = EquationBlock<IonToy, MusclMinmod, ExplicitTime<SSPRK2, 1>>;

static_assert(EquationBlockLike<ElectronBlock>);
static_assert(EquationBlockLike<IonBlock>);
static_assert(ElectronBlock::Time::treatment == TimeTreatment::Implicit);
static_assert(ElectronBlock::Time::substeps == 10);
static_assert(IonBlock::Time::treatment == TimeTreatment::Explicit);
static_assert(std::is_same_v<ElectronBlock::Spatial::NumericalFlux, HLLCFlux>);
static_assert(std::is_same_v<IonBlock::Spatial::NumericalFlux, RusanovFlux>);

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
  const BoxArray ba = BoxArray::from_domain(dom, 4);
  const DistributionMapping dm(ba.size(), n_ranks());
  BCRec bc;

  MultiFab Ue(ba, dm, 1, 0), Ui(ba, dm, 1, 0), rhs(ba, dm, 1, 0);
  Ue.set_val(2.0);
  Ui.set_val(5.0);

  ElectronBlock electrons{"electrons", ElectronToy{}, Ue, bc};
  IonBlock ions{"ions", IonToy{}, Ui, bc};
  CoupledSystem system{electrons, ions};
  static_assert(CoupledSystemLike<decltype(system)>);
  chk(decltype(system)::n_blocks == 2, "two_blocks");

  int ne = 0, ni = 0;
  Real dte = 0, dti = 0;
  advance_subcycled(system, Real(0.2), [&](auto& block, Real h, int, int) {
    using M = typename std::decay_t<decltype(block)>::Model;
    if constexpr (std::is_same_v<M, ElectronToy>) {
      ++ne;
      dte += h;
    } else if constexpr (std::is_same_v<M, IonToy>) {
      ++ni;
      dti += h;
    }
  });
  chk(ne == 10, "electron_substeps");
  chk(ni == 1, "ion_substeps");
  chk(std::fabs(dte - 0.2) < 1e-12, "electron_dt_sum");
  chk(std::fabs(dti - 0.2) < 1e-12, "ion_dt_sum");

  TwoFieldChargeDensityRhs charge;
  charge.q0 = Real(-1);  // electrons
  charge.q1 = Real(1);   // ions
  charge(Ue, Ui, rhs);
  chk(std::fabs(sum(rhs) - Real(3 * 16)) < 1e-12, "charge_density_rhs");

  if (fails == 0) std::printf("OK test_system_abstraction\n");
  return fails == 0 ? 0 : 1;
}
