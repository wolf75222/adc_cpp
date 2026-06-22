// Execution mono-niveau d'un CoupledSystem : RHS elliptique global, bloc
// implicite delegue, bloc explicite avance par le coeur.

#include <adc/core/coupled_system.hpp>
#include <adc/core/state.hpp>
#include <adc/coupling/static_system/system_coupler.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>

#include <cmath>
#include <cstdio>
#include <type_traits>

using namespace adc;

struct ElectronSource {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;

  Real rate = Real(2);

  ADC_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State&, const Aux&) const { return State{rate}; }
  ADC_HD Real elliptic_rhs(const State& u) const { return -u[0]; }
};

struct IonSource {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;

  Real rate = Real(3);

  ADC_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State&, const Aux&) const { return State{rate}; }
  ADC_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

struct ZeroSystemRhs {
  template <class System>
  void operator()(const System&, MultiFab& rhs) const {
    rhs.set_val(Real(0));
  }
};

static void add_constant(MultiFab& mf, Real value) {
  for (int li = 0; li < mf.local_size(); ++li) {
    Array4 a = mf.fab(li).array();
    const Box2D b = mf.box(li);
    for_each_cell(b, [=] ADC_HD(int i, int j) { a(i, j, 0) += value; });
  }
}

using ElectronBlock =
    EquationBlock<ElectronSource, MusclVanLeer, ImplicitTime<UserTimeIntegrator, 2>>;
using IonBlock = EquationBlock<IonSource, MusclMinmod, ExplicitTime<SSPRK3, 1>>;

static_assert(EquationBlockLike<ElectronBlock>);
static_assert(EquationBlockLike<IonBlock>);
static_assert(ElectronBlock::Time::treatment == TimeTreatment::Implicit);
static_assert(IonBlock::Time::treatment == TimeTreatment::Explicit);

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

  MultiFab Ue(ba, dm, 1, 2), Ui(ba, dm, 1, 2);
  Ue.set_val(Real(0));
  Ui.set_val(Real(0));

  ElectronBlock electrons{"electrons", ElectronSource{}, Ue, bc};
  IonBlock ions{"ions", IonSource{}, Ui, bc};
  CoupledSystem system{electrons, ions};
  auto sim = make_system_coupler(system, geom, ba, bc, ZeroSystemRhs{});

  int implicit_calls = 0;
  sim.step(Real(0.1), [&](auto&, auto& block, Real h, int, int) {
    using Model = typename std::decay_t<decltype(block)>::Model;
    if constexpr (std::is_same_v<Model, ElectronSource>) {
      ++implicit_calls;
      add_constant(block.U(), block.model.rate * h);
    }
  });

  chk(implicit_calls == 2, "implicit_substeps");
  chk(std::fabs(sum(Ue) - Real(2 * 0.1 * 16)) < Real(1e-12), "electron_implicit_update");
  chk(std::fabs(sum(Ui) - Real(3 * 0.1 * 16)) < Real(1e-12), "ion_explicit_update");

  if (fails == 0)
    std::printf("OK test_system_coupler\n");
  return fails == 0 ? 0 : 1;
}
