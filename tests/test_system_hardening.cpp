// Durcissement (revue Codex 2026-06-01) : garde-fous de construction/usage.
//   9.2 ChargeDensityRhs : defaut charge = 0 (neutre ne pollue pas Poisson) ;
//       exige une entree par bloc (sinon throw).
//   9.3 AmrSystemCoupler : refuse une hierarchie mal formee (throw au ctor).

#include <adc/core/model/coupled_system.hpp>
#include <adc/core/state/state.hpp>
#include <adc/coupling/static_system/amr_system_coupler.hpp>
#include <adc/coupling/static_system/system_coupler.hpp>
#include <adc/numerics/time/amr_reflux_mf.hpp>  // AmrLevelMP
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

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

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };
  auto throws = [&](auto&& f) {
    try {
      f();
      return false;
    } catch (const std::exception&) {
      return true;
    }
  };

  const Box2D dom = Box2D::from_extents(4, 4);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const BoxArray ba(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  BCRec bc;

  MultiFab Ue(ba, dm, 1, 2), Un(ba, dm, 1, 2);
  Ue.set_val(Real(2));
  Un.set_val(Real(7));  // "neutre" : densite quelconque, ne doit PAS toucher Poisson
  Blk e{"electrons", Scalar{}, Ue, bc};
  Blk neutral{"neutres", Scalar{}, Un, bc};
  CoupledSystem system{e, neutral};
  MultiFab rhs(ba, dm, 1, 0);

  // 9.2a : entree manquante (1 charge pour 2 blocs) -> throw.
  chk(throws([&] {
        ChargeDensityRhs bad{{{Real(-1), 0}}};
        bad(system, rhs);
      }),
      "charge_rhs_requires_one_per_block");

  // 9.2b : neutre declare avec charge = 0 -> ne contribue pas (rhs = -n_e seulement).
  ChargeDensityRhs good{{{Real(-1), 0}, {Real(0), 0}}};
  good(system, rhs);
  chk(std::fabs(sum(rhs) - (Real(-1) * sum(Ue, 0))) < Real(1e-12), "neutral_zero_charge");
  // defaut de SpeciesCharge bien a 0 (et non +1).
  chk(SpeciesCharge{}.charge == Real(0), "species_charge_default_zero");

  // 9.3 : AmrSystemCoupler refuse un block_levels mal dimensionne (1 pour 2 blocs).
  chk(throws([&] {
        std::vector<std::vector<AmrLevelMP>> bl;  // un seul bloc fourni
        bl.emplace_back();
        bl.back().push_back(AmrLevelMP{MultiFab(ba, dm, 1, 2), nullptr, geom.dx(), geom.dy()});
        AmrSystemCoupler sim(system, geom, ba, bc, ChargeDensityRhs{{{Real(-1), 0}, {Real(0), 0}}},
                             std::move(bl));
      }),
      "amr_rejects_wrong_block_count");

  if (fails == 0)
    std::printf("OK test_system_hardening\n");
  return fails == 0 ? 0 : 1;
}
