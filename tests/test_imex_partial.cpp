// IMEX PARTIEL (jalon 2.2) : un modele declare quelles variables sont implicites.
//
// Modele a 2 variables, toutes deux en relaxation S_c = -k_c (u_c - eq_c) :
//   var 0 RAIDE (k0 grand) -> traitee IMPLICITE ;
//   var 1 douce (k1 petit) -> traitee EXPLICITE (Euler avant).
// On verifie que le defaut ImplicitSourceStepper applique bien backward-Euler a la
// var 0 et forward-Euler a la var 1, en comparant a un modele SANS trait (tout
// implicite) : la var 0 coincide, la var 1 differe (3.5 explicite vs 4.0 implicite).

#include <adc/core/coupled_system.hpp>
#include <adc/core/state.hpp>
#include <adc/coupling/system_coupler.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

// Deux relaxations independantes. eq = (1, 2), k = (100, 1).
struct TwoVarRelax {
  using State = StateVec<2>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 2;
  ADC_HD State flux(const State&, const Aux&, int) const { return State{}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State& u, const Aux&) const {
    return State{-Real(100) * (u[0] - Real(1)), -Real(1) * (u[1] - Real(2))};
  }
  ADC_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

// Variante IMEX partiel : seule la var 0 (raide) est implicite.
struct TwoVarRelaxPartial : TwoVarRelax {
  static constexpr bool is_implicit(int c) { return c == 0; }
};

struct ZeroSystemRhs {
  template <class System>
  void operator()(const System&, MultiFab& rhs) const { rhs.set_val(Real(0)); }
};

static_assert(PartiallyImplicitModel<TwoVarRelaxPartial>);
static_assert(!PartiallyImplicitModel<TwoVarRelax>);

template <class Model>
static void run(MultiFab& U, const Geometry& geom, const BoxArray& ba,
                const DistributionMapping&, Real dt) {
  using Blk = EquationBlock<Model, FirstOrder, IMEXTime<UserTimeIntegrator, 1>>;
  BCRec bc;
  Blk block{"relax", Model{}, U, bc};
  CoupledSystem system{block};
  SystemCoupler sim(system, geom, ba, bc, ZeroSystemRhs{});
  sim.step(dt, ImplicitSourceStepper{});
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  const Box2D dom = Box2D::from_extents(4, 4);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const BoxArray ba = BoxArray::from_domain(dom, 4);
  const DistributionMapping dm(ba.size(), n_ranks());
  const int ncell = 16;
  const Real dt = Real(0.5);

  // valeurs de reference :
  //  var 0 implicite (dt*k0 = 50) : W0 = (5 + dt k0 eq0)/(1 + dt k0) = 55/51.
  //  var 1 explicite (dt*k1 = 0.5) : W1 = 5 + dt*(-k1 (5 - eq1)) = 5 - 0.5*3 = 3.5.
  //  var 1 si IMPLICITE (modele sans trait) : (5 + dt k1 eq1)/(1 + dt k1) = 6/1.5 = 4.0.
  const Real W0 = (Real(5) + Real(0.5) * Real(100)) / (Real(1) + Real(0.5) * Real(100));
  const Real W1_expl = Real(3.5), W1_impl = Real(4.0);

  MultiFab Up(ba, dm, 2, 1), Uf(ba, dm, 2, 1);
  Up.set_val(Real(5));
  Uf.set_val(Real(5));

  run<TwoVarRelaxPartial>(Up, geom, ba, dm, dt);  // var0 implicite, var1 explicite
  run<TwoVarRelax>(Uf, geom, ba, dm, dt);         // tout implicite

  // var 0 raide : implicite dans les deux -> meme valeur bornee (pas d'explosion).
  chk(std::fabs(sum(Up, 0) - W0 * ncell) < Real(1e-9), "partial_var0_implicit");
  chk(std::fabs(sum(Uf, 0) - W0 * ncell) < Real(1e-9), "full_var0_implicit");
  // var 1 : explicite (3.5) en partiel, implicite (4.0) en plein -> le trait CHANGE bien
  // le traitement de cette seule variable.
  chk(std::fabs(sum(Up, 1) - W1_expl * ncell) < Real(1e-12), "partial_var1_explicit");
  chk(std::fabs(sum(Uf, 1) - W1_impl * ncell) < Real(1e-9), "full_var1_implicit");
  chk(std::fabs(sum(Up, 1) - sum(Uf, 1)) > Real(1), "partial_vs_full_differ");

  if (fails == 0) std::printf("OK test_imex_partial\n");
  return fails == 0 ? 0 : 1;
}
