// AmrSystemCoupler (TODO 2.3 / 2.5.4) : un CoupledSystem porte sur AMR.
//
// Partie A : deux blocs EXPLICITES (schemas spatiaux differents : electrons MUSCL,
//   ions ordre 1) advectes sur une hierarchie 2 niveaux periodique. On verifie la
//   CONSERVATION par bloc a travers un pas (reflux + average_down du moteur AMR),
//   que le RHS Poisson de SYSTEME lit bien n_i - n_e sur le grossier, et que phi
//   resulte non nul.
// Partie B : electrons IMPLICITES (relaxation raide, sans flux) + ions explicites.
//   Le defaut AmrImplicitSourceStepper resout backward-Euler sur CHAQUE niveau :
//   grossier ET fin relaxent, bornes, stables a grand dt.

#include <adc/core/coupled_system.hpp>
#include <adc/core/state.hpp>
#include <adc/coupling/amr_system_coupler.hpp>
#include <adc/integrator/amr_reflux_mf.hpp>  // AmrLevelMP
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/refinement.hpp>  // coarsen_index

#include <cmath>
#include <cstdio>
#include <type_traits>
#include <vector>

using namespace adc;

struct AdvectX {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  Real a = Real(1);
  ADC_HD State flux(const State& u, const Aux&, int dir) const {
    return State{dir == 0 ? a * u[0] : Real(0)};
  }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const {
    return a < 0 ? -a : a;
  }
  ADC_HD State source(const State&, const Aux&) const { return State{Real(0)}; }
  ADC_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

struct ElectronRelax {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  Real k = Real(1000), neq = Real(1);
  ADC_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State& u, const Aux&) const {
    return State{-k * (u[0] - neq)};
  }
  ADC_HD Real elliptic_rhs(const State& u) const { return -u[0]; }
};

struct IonProd {
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
  void operator()(const System&, MultiFab& rhs) const { rhs.set_val(Real(0)); }
};

// remplit U (composante 0) par une fonction de l'indice GROSSIER (le fin echantillonne
// la meme fonction via coarsen_index) -> grossier et fin coherents a l'init.
template <class F>
static void fill_by_coarse_i(MultiFab& U, int ratio, F f) {
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 a = U.fab(li).array();
    const Box2D g = U.fab(li).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
        const int ci = (ratio == 1) ? i : coarsen_index(i, ratio);
        a(i, j, 0) = f(ci);
      }
  }
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  const int NC = 16;
  const Box2D dom = Box2D::from_extents(NC, NC);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const BoxArray ba_coarse(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  const Real dxc = geom.dx(), dyc = geom.dy();

  // patch fin sur les cellules grossieres [4..11]^2 -> box fine {{8,8},{23,23}}.
  const Box2D fbox{{8, 8}, {23, 23}};
  const BoxArray ba_fine(std::vector<Box2D>{fbox});

  auto make_level = [&](MultiFab&& U, Real dx, Real dy) {
    return AmrLevelMP{std::move(U), nullptr, dx, dy};
  };

  // --- Partie A : deux blocs explicites advectes, conservation + RHS systeme ---
  {
    using ElecBlk = EquationBlock<AdvectX, MusclMinmod, ExplicitTime<SSPRK2, 1>>;
    using IonBlk = EquationBlock<AdvectX, FirstOrder, ExplicitTime<SSPRK2, 1>>;
    static_assert(!std::is_same_v<ElecBlk::Spatial, IonBlk::Spatial>,
                  "les deux blocs doivent avoir des schemas spatiaux distincts");

    // charge a moyenne nulle (solvable en periodique) : n_e = 1 + 0.25 p, n_i = 1 - 0.25 p,
    // p(i) = +1 si i<8 sinon -1 (somme nulle). n_i - n_e = -0.5 p.
    auto pat = [](int ci) { return ci < 8 ? Real(1) : Real(-1); };
    auto ne_fn = [&](int ci) { return Real(1) + Real(0.25) * pat(ci); };
    auto ni_fn = [&](int ci) { return Real(1) - Real(0.25) * pat(ci); };

    MultiFab UeC(ba_coarse, dm, 1, 2), UeF(ba_fine, dm, 1, 2);
    MultiFab UiC(ba_coarse, dm, 1, 2), UiF(ba_fine, dm, 1, 2);
    fill_by_coarse_i(UeC, 1, ne_fn);  fill_by_coarse_i(UeF, 2, ne_fn);
    fill_by_coarse_i(UiC, 1, ni_fn);  fill_by_coarse_i(UiF, 2, ni_fn);

    // blocs : state pointe (provisoirement) sur le grossier ; recable par le coupleur.
    ElecBlk e{"electrons", AdvectX{Real(1)}, UeC, BCRec{}};
    IonBlk ion{"ions", AdvectX{Real(1)}, UiC, BCRec{}};
    CoupledSystem system{e, ion};

    std::vector<std::vector<AmrLevelMP>> block_levels;
    block_levels.emplace_back();
    block_levels.back().push_back(make_level(std::move(UeC), dxc, dyc));
    block_levels.back().push_back(make_level(std::move(UeF), dxc / 2, dyc / 2));
    block_levels.emplace_back();
    block_levels.back().push_back(make_level(std::move(UiC), dxc, dyc));
    block_levels.back().push_back(make_level(std::move(UiF), dxc / 2, dyc / 2));

    ChargeDensityRhs charge{{{Real(-1), 0}, {Real(1), 0}}};  // [electrons, ions]
    AmrSystemCoupler sim(system, geom, ba_coarse, BCRec{}, charge,
                         std::move(block_levels));

    sim.solve_fields();  // sync + Poisson : grossier coherent, phi a jour

    // RHS de systeme a N blocs sur le grossier : f = n_i - n_e = -0.5 p (|.| = 0.5).
    MultiFab rhs(ba_coarse, dm, 1, 0);
    charge(sim.system(), rhs);
    chk(std::fabs(norm_inf(rhs) - Real(0.5)) < Real(1e-12), "amr_system_rhs_nidne");
    chk(norm_inf(sim.phi()) > Real(1e-6), "amr_poisson_phi_nonzero");

    const Real m0e = sim.mass(0), m0i = sim.mass(1);
    sim.step(Real(0.01));  // tout explicite
    const Real m1e = sim.mass(0), m1i = sim.mass(1);
    // conservation (reflux + average_down) : la masse grossiere totale est invariante
    // sous advection periodique, malgre l'interface coarse-fine.
    chk(std::fabs(m1e - m0e) < Real(1e-11), "amr_electron_mass_conserved");
    chk(std::fabs(m1i - m0i) < Real(1e-11), "amr_ion_mass_conserved");
  }

  // --- Partie B : electrons implicites (relaxation raide) + ions explicites ---
  {
    using ElecImpl =
        EquationBlock<ElectronRelax, FirstOrder, ImplicitTime<UserTimeIntegrator, 1>>;
    using IonExpl = EquationBlock<IonProd, FirstOrder, ExplicitTime<SSPRK2, 1>>;
    static_assert(ElecImpl::Time::treatment == TimeTreatment::Implicit);

    MultiFab UeC(ba_coarse, dm, 1, 2), UeF(ba_fine, dm, 1, 2);
    MultiFab UiC(ba_coarse, dm, 1, 2), UiF(ba_fine, dm, 1, 2);
    UeC.set_val(Real(5));  UeF.set_val(Real(5));   // loin de neq = 1
    UiC.set_val(Real(0));  UiF.set_val(Real(0));

    ElecImpl e{"electrons", ElectronRelax{}, UeC, BCRec{}};
    IonExpl ion{"ions", IonProd{}, UiC, BCRec{}};
    CoupledSystem system{e, ion};

    std::vector<std::vector<AmrLevelMP>> block_levels;
    block_levels.emplace_back();
    block_levels.back().push_back(make_level(std::move(UeC), dxc, dyc));
    block_levels.back().push_back(make_level(std::move(UeF), dxc / 2, dyc / 2));
    block_levels.emplace_back();
    block_levels.back().push_back(make_level(std::move(UiC), dxc, dyc));
    block_levels.back().push_back(make_level(std::move(UiF), dxc / 2, dyc / 2));

    AmrSystemCoupler sim(system, geom, ba_coarse, BCRec{}, ZeroSystemRhs{},
                         std::move(block_levels));

    const Real dt = Real(0.1);  // dt*k = 100 : un explicite exploserait
    sim.step(dt, AmrImplicitSourceStepper{});

    const Real ne_be =
        (Real(5) + dt * Real(1000)) / (Real(1) + dt * Real(1000));  // 105/101
    // grossier ET fin relaxes (backward-Euler exact, applique a chaque niveau).
    const MultiFab& eC = sim.levels(0)[0].U;
    const MultiFab& eF = sim.levels(0)[1].U;
    chk(std::fabs(sum(eC) - ne_be * 256) < Real(1e-9), "amr_electron_coarse_relaxed");
    chk(std::fabs(sum(eF) - ne_be * 256) < Real(1e-9), "amr_electron_fine_relaxed");
    chk(norm_inf(eC) < Real(5), "amr_electron_bounded");
    // ions explicites : production constante sur le grossier (dt * rate = 0.3).
    chk(std::fabs(sum(sim.levels(1)[0].U) - Real(0.3) * 256) < Real(1e-12),
        "amr_ion_coarse_produced");
  }

  if (fails == 0) std::printf("OK test_amr_system_coupler\n");
  return fails == 0 ? 0 : 1;
}
