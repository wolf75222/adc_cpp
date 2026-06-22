// AmrSystemCoupler (jalon 2.3 / 2.5.4) : un CoupledSystem porte sur AMR.
//
// Partie A : deux blocs EXPLICITES (schemas spatiaux differents : electrons MUSCL,
//   ions ordre 1) advectes sur une hierarchie 2 niveaux periodique. On verifie la
//   CONSERVATION par bloc a travers un pas (reflux + average_down du moteur AMR),
//   que le RHS Poisson de SYSTEME lit bien n_i - n_e sur le grossier, et que phi
//   resulte non nul.
// Partie B : electrons IMPLICITES (relaxation raide, sans flux) + ions explicites.
//   Le defaut AmrImplicitSourceStepper resout backward-Euler sur CHAQUE niveau :
//   grossier ET fin relaxent, bornes, stables a grand dt.

#include <adc/core/model/coupled_system.hpp>
#include <adc/core/state/state.hpp>
#include <adc/coupling/system/amr_system_coupler.hpp>
#include <adc/numerics/time/amr/reflux/amr_reflux_mf.hpp>  // AmrLevelMP
#include <adc/mesh/index/box2d.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/layout/distribution_mapping.hpp>
#include <adc/mesh/execution/for_each.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/mf_arith.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/mesh/layout/refinement.hpp>  // coarsen_index

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
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return a < 0 ? -a : a; }
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
  ADC_HD State source(const State& u, const Aux&) const { return State{-k * (u[0] - neq)}; }
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
  void operator()(const System&, MultiFab& rhs) const {
    rhs.set_val(Real(0));
  }
};

// Echange lineaire conservatif entre les deux premiers blocs (lit les DEUX blocs) :
// n0 += dt k (n1 - n0), n1 -= idem. Conserve n0 + n1 par cellule.
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
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
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
    fill_by_coarse_i(UeC, 1, ne_fn);
    fill_by_coarse_i(UeF, 2, ne_fn);
    fill_by_coarse_i(UiC, 1, ni_fn);
    fill_by_coarse_i(UiF, 2, ni_fn);

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
    AmrSystemCoupler sim(system, geom, ba_coarse, BCRec{}, charge, std::move(block_levels));

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
    using ElecImpl = EquationBlock<ElectronRelax, FirstOrder, ImplicitTime<UserTimeIntegrator, 1>>;
    using IonExpl = EquationBlock<IonProd, FirstOrder, ExplicitTime<SSPRK2, 1>>;
    static_assert(ElecImpl::Time::treatment == TimeTreatment::Implicit);

    MultiFab UeC(ba_coarse, dm, 1, 2), UeF(ba_fine, dm, 1, 2);
    MultiFab UiC(ba_coarse, dm, 1, 2), UiF(ba_fine, dm, 1, 2);
    UeC.set_val(Real(5));
    UeF.set_val(Real(5));  // loin de neq = 1
    UiC.set_val(Real(0));
    UiF.set_val(Real(0));

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

    const Real ne_be = (Real(5) + dt * Real(1000)) / (Real(1) + dt * Real(1000));  // 105/101
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

  // --- Partie C : cadence Poisson (jalon 2.2.3) ---
  // Un bloc explicite a 4 sous-pas. OncePerStep -> phi resolu 1 fois par macro-pas ;
  // PerSubstep -> re-resolu avant chaque sous-pas suivant (1 + 3 = 4 resolutions).
  {
    using Blk4 = EquationBlock<AdvectX, FirstOrder, ExplicitTime<SSPRK2, 4>>;
    static_assert(Blk4::Time::substeps == 4);
    auto build = [&](PoissonCadence cadence) {
      MultiFab Uc(ba_coarse, dm, 1, 2);
      Uc.set_val(Real(1));
      Blk4 blk{"adv", AdvectX{Real(1)}, Uc, BCRec{}};
      CoupledSystem system{blk};
      std::vector<std::vector<AmrLevelMP>> bl;
      bl.emplace_back();
      bl.back().push_back(make_level(std::move(Uc), dxc, dyc));  // 1 niveau (grossier seul)
      return AmrSystemCoupler(system, geom, ba_coarse, BCRec{}, ZeroSystemRhs{}, std::move(bl),
                              Periodicity{true, true}, true, cadence);
    };
    auto once = build(PoissonCadence::OncePerStep);
    once.step(Real(0.01));
    chk(once.solve_count() == 1, "cadence_once_per_step");

    auto each = build(PoissonCadence::PerSubstep);
    each.step(Real(0.01));
    chk(each.solve_count() == 4, "cadence_per_substep");
  }

  // --- Partie D : source de couplage inter-especes sur AMR (revue Codex 9.5) ---
  // Deux blocs, echange lineaire conservatif applique PAR NIVEAU : la masse totale
  // (somme des deux especes) est conservee a chaque niveau ; chaque espece change.
  {
    using Blk = EquationBlock<AdvectX, FirstOrder, ExplicitTime<SSPRK2, 1>>;
    MultiFab U0c(ba_coarse, dm, 1, 2), U0f(ba_fine, dm, 1, 2);
    MultiFab U1c(ba_coarse, dm, 1, 2), U1f(ba_fine, dm, 1, 2);
    U0c.set_val(Real(1));
    U0f.set_val(Real(1));
    U1c.set_val(Real(3));
    U1f.set_val(Real(3));
    Blk b0{"a", AdvectX{Real(0)}, U0c, BCRec{}};
    Blk b1{"b", AdvectX{Real(0)}, U1c, BCRec{}};
    CoupledSystem system{b0, b1};
    std::vector<std::vector<AmrLevelMP>> bl;
    bl.emplace_back();
    bl.back().push_back(make_level(std::move(U0c), dxc, dyc));
    bl.back().push_back(make_level(std::move(U0f), dxc / 2, dyc / 2));
    bl.emplace_back();
    bl.back().push_back(make_level(std::move(U1c), dxc, dyc));
    bl.back().push_back(make_level(std::move(U1f), dxc / 2, dyc / 2));
    AmrSystemCoupler sim(system, geom, ba_coarse, BCRec{},
                         ChargeDensityRhs{{{Real(1), 0}, {Real(-1), 0}}}, std::move(bl));

    const Real tot0 = sim.mass(0) + sim.mass(1);
    const Real m0_before = sim.mass(0);
    sim.coupled_source_step(LinearExchange{Real(0.5)}, Real(0.1));
    chk(std::fabs((sim.mass(0) + sim.mass(1)) - tot0) < Real(1e-12),
        "amr_coupled_source_conserves");
    chk(sim.mass(0) > m0_before + Real(1e-6), "amr_coupled_source_transfers");  // b0 gagne (n1>n0)
    // NoCoupledSource : no-op.
    const Real m0_now = sim.mass(0);
    sim.coupled_source_step(NoCoupledSource{}, Real(0.1));
    chk(std::fabs(sim.mass(0) - m0_now) < Real(1e-14), "amr_no_coupled_source_noop");
  }

  if (fails == 0)
    std::printf("OK test_amr_system_coupler\n");
  return fails == 0 ? 0 : 1;
}
