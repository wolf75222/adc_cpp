// SystemCoupler : deux blocs EXPLICITES heterogenes (jalon 2.5.2) et conditions aux
// limites PAR BLOC reellement appliquees (jalon 2.1.3).
//
// Partie A : deux blocs explicites avec des schemas ET des sous-pas differents
//   (SSPRK2 1 sous-pas vs SSPRK3 4 sous-pas) avancent chacun selon SA politique.
// Partie B : deux blocs identiques au seul detail de la BC (periodique vs outflow)
//   divergent apres un pas -> le coeur remplit les halos avec block.bc, pas une BC
//   globale unique.

#include <adc/core/model/coupled_system.hpp>
#include <adc/core/state/state.hpp>
#include <adc/coupling/static_system/system_coupler.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/layout/distribution_mapping.hpp>
#include <adc/mesh/execution/for_each.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/mf_arith.hpp>
#include <adc/mesh/storage/multifab.hpp>

#include <cmath>
#include <cstdio>
#include <type_traits>

using namespace adc;

// Production constante, sans flux : n -> dt*rate quel que soit le schema/sous-pas.
struct Production {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  Real rate = Real(1);
  ADC_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State&, const Aux&) const { return State{rate}; }
  ADC_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

// Advection horizontale a vitesse constante a : F_x = a u, F_y = 0. Le seul terme
// sensible a la BC (le flux au bord depend du ghost rempli selon block.bc).
struct AdvectX {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  Real a = Real(1);
  ADC_HD State flux(const State& u, const Aux&, int dir) const {
    return State{dir == 0 ? a * u[0] : Real(0)};
  }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return a < 0 ? -a : a; }
  ADC_HD State source(const State&, const Aux&) const { return State{Real(0)}; }
  ADC_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

struct ZeroSystemRhs {
  template <class System>
  void operator()(const System&, MultiFab& rhs) const {
    rhs.set_val(Real(0));
  }
};

static void fill_ramp_x(MultiFab& mf) {
  for (int li = 0; li < mf.local_size(); ++li) {
    Array4 a = mf.fab(li).array();
    const Box2D b = mf.box(li);
    for_each_cell(b, [=] ADC_HD(int i, int j) { a(i, j, 0) = Real(i); });
  }
}

using ProdSSP2 = EquationBlock<Production, FirstOrder, ExplicitTime<SSPRK2, 1>>;
using ProdSSP3 = EquationBlock<Production, FirstOrder, ExplicitTime<SSPRK3, 4>>;
using AdvBlock = EquationBlock<AdvectX, FirstOrder, ExplicitTime<SSPRK2, 1>>;

static_assert(std::is_same_v<ProdSSP2::Time::Method, SSPRK2>);
static_assert(std::is_same_v<ProdSSP3::Time::Method, SSPRK3>);
static_assert(ProdSSP2::Time::substeps == 1);
static_assert(ProdSSP3::Time::substeps == 4);

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
  const int ncell = 16;
  const Real dt = Real(0.05);

  // --- Partie A : deux schemas explicites differents, memes equations ---
  {
    MultiFab Ua(ba, dm, 1, 2), Ub(ba, dm, 1, 2);
    Ua.set_val(Real(0));
    Ub.set_val(Real(0));
    ProdSSP2 a{"ssp2", Production{Real(2)}, Ua, BCRec{}};
    ProdSSP3 b{"ssp3", Production{Real(5)}, Ub, BCRec{}};
    CoupledSystem system{a, b};
    auto sim = make_system_coupler(system, geom, ba, BCRec{}, ZeroSystemRhs{});
    sim.step(dt);  // tout explicite : pas de callback
    chk(std::fabs(sum(Ua) - dt * Real(2) * ncell) < Real(1e-12), "ssp2_block");
    chk(std::fabs(sum(Ub) - dt * Real(5) * ncell) < Real(1e-12), "ssp3_block");
  }

  // --- Partie B : meme equation, BC differentes par bloc ---
  // Controle : deux blocs periodiques identiques -> resultat identique.
  {
    BCRec per;  // periodique partout
    MultiFab Up1(ba, dm, 1, 2), Up2(ba, dm, 1, 2);
    fill_ramp_x(Up1);
    fill_ramp_x(Up2);
    AdvBlock b1{"p1", AdvectX{}, Up1, per};
    AdvBlock b2{"p2", AdvectX{}, Up2, per};
    CoupledSystem system{b1, b2};
    auto sim = make_system_coupler(system, geom, ba, per, ZeroSystemRhs{});
    sim.step(dt);
    MultiFab d(ba, dm, 1, 0);
    lincomb(d, Real(1), Up1, Real(-1), Up2);
    chk(norm_inf(d) < Real(1e-12), "same_bc_identical");
  }

  // Test : periodique vs outflow (foextrap), meme donnee initiale -> divergent.
  {
    BCRec per;
    BCRec out;
    out.xlo = out.xhi = BCType::Foextrap;
    out.ylo = out.yhi = BCType::Foextrap;
    MultiFab Uper(ba, dm, 1, 2), Uout(ba, dm, 1, 2);
    fill_ramp_x(Uper);
    fill_ramp_x(Uout);
    AdvBlock bp{"per", AdvectX{}, Uper, per};
    AdvBlock bo{"out", AdvectX{}, Uout, out};
    CoupledSystem system{bp, bo};
    auto sim = make_system_coupler(system, geom, ba, per, ZeroSystemRhs{});
    sim.step(dt);
    MultiFab d(ba, dm, 1, 0);
    lincomb(d, Real(1), Uper, Real(-1), Uout);
    chk(norm_inf(d) > Real(1e-3), "per_block_bc_differs");
  }

  if (fails == 0)
    std::printf("OK test_system_two_explicit\n");
  return fails == 0 ? 0 : 1;
}
