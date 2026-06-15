// EXEMPLE C++ MINIMAL deux especes, SANS Python (jalon 2.4).
//
// Le test "est-ce qu'un utilisateur peut construire son cas ?" : electrons
// IMPLICITES (source de relaxation raide) + ions EXPLICITES (SSPRK2) + Poisson
// rhs = n_i - n_e, assemble par ChargeDensityRhs a N especes. L'utilisateur ne
// compose que des briques (modele local, schema spatial, politique temps, charge)
// et appelle SystemCoupler ; aucun solveur implicite n'est ecrit a la main (le
// defaut ImplicitSourceStepper s'en charge).
//
// Couvre aussi : RHS Poisson non nul a N blocs (jalon 2.1.1 / 2.5.1) et le defaut
// implicite inconditionnellement stable sur source raide (jalon 2.2.1 / 2.5.3).

#include <adc/core/coupled_system.hpp>
#include <adc/core/state.hpp>
#include <adc/coupling/system_coupler.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>

#include <cmath>
#include <cstdio>
#include <type_traits>

using namespace adc;

// Electrons : densite scalaire qui RELAXE vers neq a un taux RAIDE k. Pas de flux
// (drift gele dans ce squelette). C'est exactement le terme qu'on veut implicite :
// en explicite il imposerait dt < 1/k.
struct ElectronRelax {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;

  Real k = Real(1000);    // raideur
  Real neq = Real(1);     // densite d'equilibre

  ADC_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State& u, const Aux&) const {
    return State{-k * (u[0] - neq)};
  }
  ADC_HD Real elliptic_rhs(const State& u) const { return -u[0]; }
};

// Ions : production constante, explicite. Pas de flux.
struct IonProduction {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;

  Real rate = Real(3);

  ADC_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State&, const Aux&) const { return State{rate}; }
  ADC_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

using ElectronBlock =
    EquationBlock<ElectronRelax, FirstOrder, ImplicitTime<UserTimeIntegrator, 1>>;
using IonBlock = EquationBlock<IonProduction, FirstOrder, ExplicitTime<SSPRK2, 1>>;

static_assert(EquationBlockLike<ElectronBlock>);
static_assert(EquationBlockLike<IonBlock>);
static_assert(ElectronBlock::Time::treatment == TimeTreatment::Implicit);
static_assert(IonBlock::Time::treatment == TimeTreatment::Explicit);

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
  BCRec bc;  // periodique partout (le defaut)

  MultiFab Ue(ba, dm, 1, 2), Ui(ba, dm, 1, 2);
  Ue.set_val(Real(5));  // loin de neq=1 : la relaxation doit etre forte
  Ui.set_val(Real(0));

  // --- composition du cas : trois briques par espece, rien de plus ---
  ElectronBlock electrons{"electrons", ElectronRelax{}, Ue, bc};
  IonBlock ions{"ions", IonProduction{}, Ui, bc};
  CoupledSystem system{electrons, ions};

  // Poisson rhs = Sum_s q_s n_s = (+1) n_i + (-1) n_e = n_i - n_e.
  ChargeDensityRhs charge{{{Real(-1), 0}, {Real(1), 0}}};  // [electrons, ions]
  SystemCoupler sim(system, geom, ba, bc, charge);

  // Un pas : electrons via le defaut implicite, ions explicites par le coeur.
  const Real dt = Real(0.1);
  sim.step(dt, ImplicitSourceStepper{});

  // (1) Ions explicites : production constante exacte, n_i = dt * rate = 0.3.
  chk(std::fabs(sum(Ui) - Real(0.3) * ncell) < Real(1e-12), "ion_explicit");

  // (2) Electrons implicites : backward-Euler exact pour la relaxation lineaire,
  //     n_e = (n0 + dt k neq) / (1 + dt k). dt*k = 100 : un schema explicite
  //     EXPLOSERAIT (n_e ~ 5 - 400). Ici la valeur reste bornee et proche de neq.
  const Real ne_be = (Real(5) + dt * Real(1000) * Real(1)) / (Real(1) + dt * Real(1000));
  chk(std::fabs(sum(Ue) - ne_be * ncell) < Real(1e-9), "electron_implicit_exact");
  chk(sum(Ue) > Real(0) && sum(Ue) < Real(5) * ncell, "electron_implicit_bounded");

  // (3) RHS Poisson a N especes, non nul (jalon 2.1.1 / 2.5.1) : f = n_i - n_e,
  //     et l'assembleur somme bien sur tous les blocs.
  MultiFab rhs(ba, dm, 1, 0);
  charge(system, rhs);
  chk(std::fabs(sum(rhs) - (sum(Ui) - sum(Ue))) < Real(1e-12), "charge_density_rhs");
  chk(std::fabs(sum(rhs)) > Real(1), "poisson_rhs_nonzero");

  if (fails == 0) std::printf("OK test_two_species_minimal\n");
  return fails == 0 ? 0 : 1;
}
