// Source de COUPLAGE inter-especes (jalon 2.1.2).
//
// model.source(u, aux) est locale au bloc. Une CoupledSource lit PLUSIEURS blocs
// pour exprimer un echange entre especes. Ici : un echange lineaire qui transfere
// de la masse du bloc 1 vers le bloc 0 a un taux k(n1 - n0). Conservatif par
// construction (ce que les sources locales ne pourraient pas garantir : elles ne
// voient pas l'autre espece). Applique par SystemCoupler::coupled_source_step.

#include <adc/core/coupled_system.hpp>
#include <adc/core/state.hpp>
#include <adc/coupling/coupled_source.hpp>
#include <adc/coupling/system_coupler.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

// Espece inerte : densite scalaire, aucune dynamique locale. Toute l'evolution
// vient de la source de couplage, pas de model.source.
struct Inert {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  ADC_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State&, const Aux&) const { return State{Real(0)}; }
  ADC_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

// Echange lineaire entre les deux premiers blocs : n0 += dt k (n1 - n0),
// n1 -= dt k (n1 - n0). Lit l'etat des DEUX blocs -> c'est tout l'interet d'une
// CoupledSource. Conserve n0 + n1 exactement (le flux quitte 1 et entre dans 0).
struct LinearExchange {
  Real k = Real(0.5);

  template <CoupledSystemLike System>
  void apply(System& sys, const MultiFab& /*aux*/, Real dt) const {
    MultiFab& U0 = sys.template block<0>().U();
    MultiFab& U1 = sys.template block<1>().U();
    const Real coef = k * dt;
    for (int li = 0; li < U0.local_size(); ++li) {
      Array4 a0 = U0.fab(li).array();
      Array4 a1 = U1.fab(li).array();
      const Box2D b = U0.box(li);
      const Real c = coef;
      for_each_cell(b, [=] ADC_HD(int i, int j) {
        const Real flux = c * (a1(i, j, 0) - a0(i, j, 0));
        a0(i, j, 0) += flux;
        a1(i, j, 0) -= flux;
      });
    }
  }
};

using BlockA = EquationBlock<Inert, FirstOrder, ExplicitTime<SSPRK2, 1>>;
using BlockB = EquationBlock<Inert, FirstOrder, ExplicitTime<SSPRK2, 1>>;

static_assert(CoupledSourceFor<LinearExchange, CoupledSystem<BlockA, BlockB>>);
static_assert(CoupledSourceFor<NoCoupledSource, CoupledSystem<BlockA, BlockB>>);

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
  BCRec bc;

  MultiFab U0(ba, dm, 1, 2), U1(ba, dm, 1, 2);
  U0.set_val(Real(1));
  U1.set_val(Real(3));
  const Real total0 = sum(U0) + sum(U1);

  BlockA a{"a", Inert{}, U0, bc};
  BlockB b{"b", Inert{}, U1, bc};
  CoupledSystem system{a, b};
  SystemCoupler sim(system, geom, ba, bc, ChargeDensityRhs{{{Real(1), 0}, {Real(-1), 0}}});

  const Real dt = Real(0.1);
  sim.coupled_source_step(LinearExchange{Real(0.5)}, dt);

  // Echange : flux = 0.5*0.1*(3-1) = 0.1 par cellule. n0 -> 1.1, n1 -> 2.9.
  chk(std::fabs(sum(U0) - Real(1.1) * ncell) < Real(1e-12), "block0_gained");
  chk(std::fabs(sum(U1) - Real(2.9) * ncell) < Real(1e-12), "block1_lost");
  // Conservation : la source de couplage ne cree ni ne detruit de masse totale.
  chk(std::fabs((sum(U0) + sum(U1)) - total0) < Real(1e-12), "total_conserved");

  // NoCoupledSource : no-op, etat inchange.
  const Real s0 = sum(U0), s1 = sum(U1);
  sim.coupled_source_step(NoCoupledSource{}, dt);
  chk(std::fabs(sum(U0) - s0) < Real(1e-14) && std::fabs(sum(U1) - s1) < Real(1e-14),
      "no_coupled_source_noop");

  if (fails == 0) std::printf("OK test_coupled_source\n");
  return fails == 0 ? 0 : 1;
}
