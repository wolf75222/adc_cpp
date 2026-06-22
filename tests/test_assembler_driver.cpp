// Scission Assembleur / Driver (retour tuteur sec.8.2 B). Un SystemAssembler ASSEMBLE les
// champs (Poisson de systeme + aux + residu de bloc) SANS avancer ; un SystemDriver AVANCE
// (et possede un assembleur). "advance un coupleur" devient "advance un driver".

#include <adc/core/coupled_system.hpp>
#include <adc/core/state.hpp>
#include <adc/coupling/static_system/system_coupler.hpp>  // SystemAssembler, SystemDriver, SystemCoupler
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>

#include <cmath>
#include <cstdio>
#include <type_traits>

using namespace adc;

struct Scalar {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  ADC_HD State flux(const State&, const Aux&, int) const { return State{}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State&, const Aux&) const { return State{}; }
  ADC_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

using Blk = EquationBlock<Scalar, FirstOrder, ExplicitTime<SSPRK2, 1>>;

// SystemCoupler reste un alias du Driver (compat).
static_assert(std::is_same_v<SystemCoupler<CoupledSystem<Blk, Blk>, ChargeDensityRhs>,
                             SystemDriver<CoupledSystem<Blk, Blk>, ChargeDensityRhs>>);

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int n = 16;
  const Box2D dom = Box2D::from_extents(n, n);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const BoxArray ba(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  BCRec bc;

  // charge a moyenne nulle : n0 = 1 + 0.25*signe(i<n/2), n1 = 1 -> f = -0.5*signe.
  MultiFab U0(ba, dm, 1, 2), U1(ba, dm, 1, 2);
  {
    Array4 a0 = U0.fab(0).array(), a1 = U1.fab(0).array();
    const Box2D g = U0.fab(0).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
        a0(i, j, 0) = Real(1) + (i < n / 2 ? Real(0.25) : Real(-0.25));
        a1(i, j, 0) = Real(1);
      }
  }
  Blk b0{"a", Scalar{}, U0, bc}, b1{"b", Scalar{}, U1, bc};
  CoupledSystem system{b0, b1};
  ChargeDensityRhs charge{{{Real(-1), 0}, {Real(1), 0}}};

  // --- ASSEMBLEUR seul : resout les champs, expose phi/aux, ne fait AUCUN pas. ---
  SystemAssembler assembler(system, geom, ba, bc, charge);
  assembler.solve_fields();
  chk(norm_inf(assembler.phi()) > Real(1e-6), "assembler_phi_nonzero");
  // residu d'un bloc : Scalar -> flux et source nuls -> R = 0 (l'evaluateur tourne).
  MultiFab R(ba, dm, 1, 0);
  assembler.block_residual<NoSlope, RusanovFlux>(assembler.system().block<0>(),
                                                 assembler.system().block<0>().U(), R,
                                                 /*recompute_aux=*/false);
  chk(norm_inf(R) < Real(1e-14), "assembler_block_residual_zero");

  // --- DRIVER : avance (et possede un assembleur). ---
  MultiFab V0(ba, dm, 1, 2), V1(ba, dm, 1, 2);
  V0.set_val(Real(1));
  V1.set_val(Real(1));
  Blk d0{"a", Scalar{}, V0, bc}, d1{"b", Scalar{}, V1, bc};
  CoupledSystem dsys{d0, d1};
  SystemDriver driver(dsys, geom, ba, bc, charge);
  driver.step(Real(0.1));  // blocs explicites, flux/source nuls -> etat inchange, mais tourne
  chk(std::fabs(sum(V0, 0) - Real(1) * n * n) < Real(1e-12), "driver_step_runs");
  chk(norm_inf(driver.phi()) < Real(1e-9), "driver_phi_zero_for_neutral_balance");

  if (fails == 0)
    std::printf("OK test_assembler_driver\n");
  return fails == 0 ? 0 : 1;
}
