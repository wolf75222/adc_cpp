// Pas macro choisi par CFL multi-especes (sec.8.2 C) : SystemDriver::step_cfl. Le pas est
// dt = cfl * min(dx,dy) / w_max, ou w_max est la plus grande vitesse d'onde sur TOUTES les
// especes -> l'espece la plus rapide contraint le pas. Combine au Stride d'une espece lente,
// cela donne le multirate pratique.

#include <adc/core/model/coupled_system.hpp>
#include <adc/core/state/state.hpp>
#include <adc/coupling/static_system/system_coupler.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

using namespace adc;

// Advection a vitesse constante a : vitesse d'onde max = |a|.
struct AdvectX {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  Real a = Real(1);
  ADC_HD State flux(const State& u, const Aux&, int dir) const {
    return State{dir == 0 ? a * u[0] : Real(0)};
  }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return a < 0 ? -a : a; }
  ADC_HD State source(const State&, const Aux&) const { return State{}; }
  ADC_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

// ADC-267 : modele dont la vitesse d'onde est NaN (flux physique fini, 0). Verifie que le calcul
// du pas CFL avale un NaN : system_max_wave_speed fait std::max(wmax_acc, .) avec wmax_acc a 0,
// donc le NaN (2e argument) est avale (std::max(0, NaN) = 0) -> w_max reste fini -> dt fini.
struct NanSpeed {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  ADC_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  ADC_HD State source(const State&, const Aux&) const { return State{}; }
  ADC_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

struct ZeroSystemRhs {
  template <class System>
  void operator()(const System&, MultiFab& rhs) const {
    rhs.set_val(Real(0));
  }
};

using Blk = EquationBlock<AdvectX, FirstOrder, ExplicitTime<SSPRK2, 1>>;

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

  MultiFab Uf(ba, dm, 1, 2), Us(ba, dm, 1, 2);
  Uf.set_val(Real(1));
  Us.set_val(Real(1));
  // espece rapide (a=2) + espece lente (a=0.5) : le pas CFL est fixe par la rapide.
  Blk fast{"fast", AdvectX{Real(2)}, Uf, bc};
  Blk slow{"slow", AdvectX{Real(0.5)}, Us, bc};
  CoupledSystem system{fast, slow};
  auto sim = make_system_coupler(system, geom, ba, bc, ZeroSystemRhs{});

  const Real cfl = Real(0.4);
  const Real h = std::min(geom.dx(), geom.dy());
  const Real expected = cfl * h / Real(2);  // w_max = max(2, 0.5) = 2

  const Real dt = sim.step_cfl(cfl);
  chk(std::fabs(dt - expected) < Real(1e-14), "cfl_dt_from_fastest_species");
  chk(dt > Real(0), "cfl_dt_positive");
  // le systeme a avance (etat fini, masse conservee pour l'advection periodique).
  chk(std::fabs(sum(Uf, 0) - Real(1) * n * n) < Real(1e-10), "fast_mass_conserved");
  chk(std::fabs(sum(Us, 0) - Real(1) * n * n) < Real(1e-10), "slow_mass_conserved");

  // --- ADC-267 : systeme au repos (w_max = 0) -> le garde CFL clampe le denominateur a 1e-30 ---
  // dt = cfl*h / max(w_max, 1e-30) ; sans le plancher, dt = +inf sur un etat au repos.
  {
    MultiFab Uq(ba, dm, 1, 2);
    Uq.set_val(Real(1));
    Blk quiet{"quiet", AdvectX{Real(0)}, Uq, bc};  // a = 0 -> w_max = 0
    CoupledSystem qsys{quiet};
    SystemCoupler qsim(qsys, geom, ba, bc, ZeroSystemRhs{});
    const Real dtq = qsim.step_cfl(cfl);
    const Real floor_dt = cfl * h / Real(1e-30);  // denominateur clampe au plancher
    chk(std::isfinite(dtq), "quiescent_dt_finite");
    chk(std::fabs(dtq - floor_dt) <= floor_dt * Real(1e-12), "quiescent_dt_clamped_to_floor");
    chk(std::isfinite(sum(Uq, 0)), "quiescent_state_finite");  // a = 0 -> aucune advection
  }

  // --- ADC-267 : une vitesse d'onde NaN est avalee par le garde CFL -> le PAS reste fini. ---
  // On ne fait PAS avancer l'etat ici : le flux numerique (Rusanov) utilise aussi la vitesse
  // d'onde, donc un NaN s'y propagerait (comportement attendu du schema, pas du garde). Ce qu'on
  // verifie, c'est la robustesse du calcul du pas (cfl_dt sans avancer).
  {
    MultiFab Un(ba, dm, 1, 2);
    Un.set_val(Real(1));
    using NanBlk = EquationBlock<NanSpeed, FirstOrder, ExplicitTime<SSPRK2, 1>>;
    NanBlk nblk{"nan", NanSpeed{}, Un, bc};
    CoupledSystem nsys{nblk};
    SystemCoupler nsim(nsys, geom, ba, bc, ZeroSystemRhs{});
    const Real dtn = nsim.cfl_dt(cfl);               // calcule le pas SANS avancer l'etat
    chk(std::isfinite(dtn), "nan_speed_dt_finite");  // std::max(0, NaN) = 0 -> w_max = 0 -> dt fini
    chk(dtn > Real(0), "nan_speed_dt_positive");
  }

  if (fails == 0)
    std::printf("OK test_cfl_dt\n");
  return fails == 0 ? 0 : 1;
}
