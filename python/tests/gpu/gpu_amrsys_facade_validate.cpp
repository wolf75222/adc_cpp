// Validation DEVICE (GH200, Kokkos Cuda) de la FACADE AmrSystemCoupler elle-meme (limite device (b)).
//
// CONTRASTE avec gpu_amr_bz_validate.cpp : celui-la valide le MOTEUR advance_amr en direct, parce que
// la facade AmrSystemCoupler<CoupledSystemLike System> ne s'instanciait PAS sous nvcc -- le concept
// CoupledSystemLike sondait for_each_block avec une LAMBDA GENERIQUE en contexte non evalue
// (s.for_each_block([](auto&){})), que le frontend nvcc/EDG refuse. Corrige en remplacant la sonde
// par un FONCTEUR NOMME (detail::ForEachBlockProbe), meme recette que les foncteurs nommes de
// block_builder.hpp (#64). CE harness instancie la facade ENTIERE (CoupledSystem 2 blocs + AMR 2
// niveaux + Poisson de systeme + solve_fields + step) : il NE COMPILE PAS sans le correctif, et
// produit un device bit-identique au Serial avec.
//
// Imprime exec=, masses par bloc, checksums grossier+fin par bloc (%.17g) ; --dump=<prefixe> ecrit
// U(grossier+fin, par bloc) pour un diff binaire Cuda vs Serial (dmax par cellule).

#include <adc/core/model/coupled_system.hpp>
#include <adc/core/model/equation_block.hpp>
#include <adc/core/state/state.hpp>
#include <adc/core/foundation/types.hpp>  // ADC_HD
#include <adc/coupling/static_system/amr_system_coupler.hpp>
#include <adc/coupling/base/elliptic_rhs.hpp>  // ChargeDensityRhs
#include <adc/mesh/index/box2d.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/layout/distribution_mapping.hpp>
#include <adc/mesh/execution/for_each.hpp>  // device_fence
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/mf_arith.hpp>  // sum, norm_inf
#include <adc/mesh/storage/multifab.hpp>
#include <adc/mesh/layout/refinement.hpp>                  // coarsen_index
#include <adc/numerics/fv/spatial_discretisation.hpp>  // FirstOrder, MusclMinmod
#include <adc/numerics/time/amr/reflux/amr_reflux_mf.hpp>      // AmrLevelMP
#include <adc/numerics/time/integrators/time_integrator.hpp>    // ExplicitTime
#include <adc/numerics/time/integrators/time_steppers.hpp>      // SSPRK2
#include <adc/parallel/comm.hpp>                    // n_ranks

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace adc;

// Advection lineaire 1 variable, couplee a Poisson par elliptic_rhs = u (densite de charge ponderee
// par le poids du bloc dans ChargeDensityRhs). ADC_HD : memes fermetures hote/device.
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

// Remplit U(comp 0) par une fonction de l'indice GROSSIER (le fin echantillonne la meme fonction via
// coarsen_index) -> grossier et fin coherents a l'init. Ecriture HOTE (memoire unifiee).
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

static void collect(const MultiFab& U, std::vector<double>& out) {
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 a = U.fab(li).const_array();
    const Box2D b = U.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        out.push_back(a(i, j, 0));
  }
}

int main(int argc, char** argv) {
#if defined(ADC_HAS_KOKKOS)
  Kokkos::initialize(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  std::string dump_prefix;
  for (int k = 1; k < argc; ++k)
    if (std::strncmp(argv[k], "--dump=", 7) == 0)
      dump_prefix = argv[k] + 7;

  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };
#if defined(ADC_HAS_KOKKOS)
  const char* space = Kokkos::DefaultExecutionSpace::name();
#else
  const char* space = "Serial(host)";
#endif

  {
    const int NC = 16;
    const Box2D dom = Box2D::from_extents(NC, NC);
    const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
    const BoxArray ba_coarse(std::vector<Box2D>{dom});
    const DistributionMapping dm(1, n_ranks());
    const Real dxc = geom.dx(), dyc = geom.dy();
    const Box2D fbox{{8, 8}, {23, 23}};  // patch fin sur [4..11]^2 grossier
    const BoxArray ba_fine(std::vector<Box2D>{fbox});

    // charge a moyenne nulle (solvable en periodique) : n_e = 1 + 0.25 p, n_i = 1 - 0.25 p,
    // p(i) = +1 si i<8 sinon -1. n_i - n_e = -0.5 p -> phi non nul.
    auto pat = [](int ci) { return ci < 8 ? Real(1) : Real(-1); };
    auto ne_fn = [&](int ci) { return Real(1) + Real(0.25) * pat(ci); };
    auto ni_fn = [&](int ci) { return Real(1) - Real(0.25) * pat(ci); };

    MultiFab UeC(ba_coarse, dm, 1, 2), UeF(ba_fine, dm, 1, 2);
    MultiFab UiC(ba_coarse, dm, 1, 2), UiF(ba_fine, dm, 1, 2);
    fill_by_coarse_i(UeC, 1, ne_fn);
    fill_by_coarse_i(UeF, 2, ne_fn);
    fill_by_coarse_i(UiC, 1, ni_fn);
    fill_by_coarse_i(UiF, 2, ni_fn);

    using ElecBlk = EquationBlock<AdvectX, MusclMinmod, ExplicitTime<SSPRK2, 1>>;
    using IonBlk = EquationBlock<AdvectX, FirstOrder, ExplicitTime<SSPRK2, 1>>;
    ElecBlk e{"electrons", AdvectX{Real(1)}, UeC, BCRec{}};
    IonBlk ion{"ions", AdvectX{Real(1)}, UiC, BCRec{}};
    CoupledSystem system{e, ion};  // <- exige CoupledSystemLike (le point qui faisait buter nvcc)

    std::vector<std::vector<AmrLevelMP>> block_levels;
    block_levels.emplace_back();
    block_levels.back().push_back(AmrLevelMP{std::move(UeC), nullptr, dxc, dyc});
    block_levels.back().push_back(AmrLevelMP{std::move(UeF), nullptr, dxc / 2, dyc / 2});
    block_levels.emplace_back();
    block_levels.back().push_back(AmrLevelMP{std::move(UiC), nullptr, dxc, dyc});
    block_levels.back().push_back(AmrLevelMP{std::move(UiF), nullptr, dxc / 2, dyc / 2});

    ChargeDensityRhs charge{{{Real(-1), 0}, {Real(1), 0}}};  // [electrons, ions]
    // INSTANCIATION DE LA FACADE : c'est ici que le concept CoupledSystemLike est evalue par nvcc.
    AmrSystemCoupler sim(system, geom, ba_coarse, BCRec{}, charge, std::move(block_levels));

    sim.solve_fields();
    device_fence();
    chk(norm_inf(sim.phi()) > Real(1e-6), "facade_poisson_phi_nonzero");

    const Real m0e = sim.mass(0), m0i = sim.mass(1);
    const Real dt = Real(0.01);
    for (int s = 0; s < 3; ++s)
      sim.step(dt);  // tout explicite, advection periodique
    device_fence();
    const Real m1e = sim.mass(0), m1i = sim.mass(1);
    // conservation (reflux + average_down) : masse grossiere totale invariante sous advection periodique.
    chk(std::fabs(m1e - m0e) < Real(1e-10), "facade_electron_mass_conserved");
    chk(std::fabs(m1i - m0i) < Real(1e-10), "facade_ion_mass_conserved");

    const MultiFab& eC = sim.levels(0)[0].U;
    const MultiFab& eF = sim.levels(0)[1].U;
    const MultiFab& iC = sim.levels(1)[0].U;
    const MultiFab& iF = sim.levels(1)[1].U;
    std::printf("[AMRSYS facade] exec=%s\n", space);
    std::printf("[AMRSYS facade] mass e: %.17g -> %.17g   mass i: %.17g -> %.17g\n", m0e, m1e, m0i,
                m1i);
    std::printf("[AMRSYS facade] sum eC=%.17g eF=%.17g iC=%.17g iF=%.17g  ninf(phi)=%.17g\n",
                sum(eC), sum(eF), sum(iC), sum(iF), norm_inf(sim.phi()));

    if (!dump_prefix.empty()) {
      std::vector<double> v;
      collect(eC, v);
      collect(eF, v);
      collect(iC, v);
      collect(iF, v);
      const std::string path = dump_prefix + "_amrsys.bin";
      FILE* f = std::fopen(path.c_str(), "wb");
      if (f) {
        std::fwrite(v.data(), 1, v.size() * sizeof(double), f);
        std::fclose(f);
      }
    }
  }

#if defined(ADC_HAS_KOKKOS)
  Kokkos::finalize();
#endif
  if (fails == 0)
    std::printf("OK gpu_amrsys_facade_validate (exec=%s)\n", space);
  return fails == 0 ? 0 : 1;
}
