// Vrai IMEX (revue Codex 9.1) : un bloc IMEX a flux non nul voit son TRANSPORT
// explicite avance par le coeur (puis sa source par le callback implicite). Avant le
// correctif, un bloc IMEX partait entierement dans le callback : avec un stepper
// source-seule (ImplicitSourceStepper), son transport n'etait PAS avance (champ fige).
//
// Ici : deux blocs d'advection identiques, l'un IMEX, l'autre explicite. Apres un pas
// (avec ImplicitSourceStepper, source nulle = no-op), les DEUX ont advecte (champ change).

#include <pops/core/model/coupled_system.hpp>
#include <pops/core/state/state.hpp>
#include <pops/coupling/system/system_coupler.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <cstdio>

using namespace pops;

struct AdvectX {
  using State = StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;
  Real a = Real(1);
  POPS_HD State flux(const State& u, const Aux&, int dir) const {
    return State{dir == 0 ? a * u[0] : Real(0)};
  }
  POPS_HD Real max_wave_speed(const State&, const Aux&, int) const { return a < 0 ? -a : a; }
  POPS_HD State source(const State&, const Aux&) const { return State{}; }
  POPS_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

struct ZeroSystemRhs {
  template <class System>
  void operator()(const System&, MultiFab& rhs) const {
    rhs.set_val(Real(0));
  }
};

static void fill_ramp(MultiFab& mf) {
  for (int li = 0; li < mf.local_size(); ++li) {
    Array4 a = mf.fab(li).array();
    const Box2D b = mf.box(li);
    for_each_cell(b, [=] POPS_HD(int i, int j) { a(i, j, 0) = Real(i); });
  }
}

using ImexBlk = EquationBlock<AdvectX, FirstOrder, IMEXTime<UserTimeIntegrator, 1>>;
using ExplBlk = EquationBlock<AdvectX, FirstOrder, ExplicitTime<SSPRK2, 1>>;
static_assert(ImexBlk::Time::treatment == TimeTreatment::IMEX);

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const Box2D dom = Box2D::from_extents(8, 8);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const BoxArray ba(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  BCRec bc;  // periodique

  MultiFab Ui(ba, dm, 1, 2), Ue(ba, dm, 1, 2);
  fill_ramp(Ui);
  fill_ramp(Ue);
  MultiFab Ui0 = Ui;  // copie de la donnee initiale

  ImexBlk imex{"imex", AdvectX{Real(1)}, Ui, bc};
  ExplBlk expl{"expl", AdvectX{Real(1)}, Ue, bc};
  CoupledSystem system{imex, expl};
  auto sim = make_system_coupler(system, geom, ba, bc, ZeroSystemRhs{});

  sim.step(Real(0.05), ImplicitSourceStepper{});  // source nulle -> seul le transport agit

  MultiFab d(ba, dm, 1, 0);
  lincomb(d, Real(1), Ui, Real(-1), Ui0);
  // 9.1 : le bloc IMEX a bien ete TRANSPORTE (avant le correctif : champ fige -> 0).
  chk(norm_inf(d) > Real(1e-3), "imex_block_transported");
  // sanity : le bloc explicite a transporte aussi.
  lincomb(d, Real(1), Ue, Real(-1), Ui0);
  chk(norm_inf(d) > Real(1e-3), "explicit_block_transported");

  if (fails == 0)
    std::printf("OK test_imex_transport\n");
  return fails == 0 ? 0 : 1;
}
