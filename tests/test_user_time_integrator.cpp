// Integrateur en temps fourni par l'UTILISATEUR (retour tuteur sec.8.2 A3/A5). Le coeur
// fournit SSPRK2/SSPRK3 ; mais on peut aussi ecrire son propre objet a `take_step` et le
// passer comme Method d'un bloc explicite, le coupleur l'instancie et l'appelle, sans
// rien changer au coeur. La physique (rhs_eval = flux + source) reste compilee.

#include <adc/core/model/coupled_system.hpp>
#include <adc/core/state/state.hpp>
#include <adc/coupling/static_system/system_coupler.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/layout/distribution_mapping.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/mf_arith.hpp>  // saxpy
#include <adc/mesh/storage/multifab.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

// Production constante : n -> dt*rate quel que soit le schema (Euler avant exact ici).
struct Production {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  Real rate = Real(3);
  ADC_HD State flux(const State&, const Aux&, int) const { return State{}; }
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

// Integrateur ECRIT PAR L'UTILISATEUR : Euler avant. Il ne voit que rhs_eval + U.
struct UserForwardEuler {
  template <class RhsEval>
  void take_step(RhsEval&& rhs, MultiFab& U, Real dt) const {
    MultiFab R(U.box_array(), U.dmap(), U.ncomp(), 0);
    rhs(U, R);
    saxpy(U, dt, R);
  }
};

static_assert(TimeStepper<UserForwardEuler>, "l'integrateur utilisateur doit modeler TimeStepper");

// Le Method du bloc EST le type de l'integrateur utilisateur.
using UserBlock = EquationBlock<Production, FirstOrder, ExplicitTime<UserForwardEuler, 1>>;
static_assert(UserBlock::Time::treatment == TimeTreatment::Explicit);

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

  MultiFab U(ba, dm, 1, 2);
  U.set_val(Real(0));
  UserBlock blk{"prod", Production{Real(3)}, U, bc};
  CoupledSystem system{blk};
  auto sim = make_system_coupler(system, geom, ba, bc, ZeroSystemRhs{});

  const Real dt = Real(0.1);
  sim.step(dt);  // tout explicite : le coupleur appelle UserForwardEuler::take_step

  // Euler avant sur une source constante : n = dt * rate, exact.
  chk(std::fabs(sum(U, 0) - dt * Real(3) * ncell) < Real(1e-12), "user_integrator_advances");

  if (fails == 0)
    std::printf("OK test_user_time_integrator\n");
  return fails == 0 ? 0 : 1;
}
