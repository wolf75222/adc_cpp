// Cadence HOLD-THEN-CATCH-UP dans AmrSystemCoupler : un bloc lent (stride=M>1)
// doit etre TENU aux macro-pas 0..M-2 et avancer AU macro-pas M-1 (rattrapage),
// de facon a ne jamais etre en avance sur les blocs rapides au point de couplage.
//
// Bug corrige : l'ancienne condition `macro_step_ % stride == 0` declenchait
// l'avance du bloc lent AU PREMIER macro-pas (macro_step_=0, 0%stride==0), ce
// qui plagait le bloc dans le futur relativement aux blocs rapides.
// Condition correcte : `(macro_step_ + 1) % stride == 0`.
//
// Test A (cadence) : bloc rapide (stride=1) + bloc lent (stride=M), production
//   constante, SANS flux (uniformite spatiale). On compte les avances explicites
//   en mesurant l'integrale de la densite apres chaque macro-pas :
//   - macro-pas 0..M-2 : le bloc lent reste a u=0 (non avance).
//   - macro-pas M-1    : le bloc lent rattrape en un seul pas effectif M*dt.
//   - au bout de M macro-pas, les deux blocs ont avance du meme temps total M*dt.
//   Invariant de couplage : a chaque macro-pas, bloc_lent.time <= bloc_rapide.time.
//
// Test B (bit-identique stride=1) : un bloc stride=1 sur AmrSystemCoupler suit
//   exactement la meme trajectoire qu'un stride=1 au premier macro-pas (avance
//   a mac=0), et produit la meme valeur apres N pas qu'un calcul de reference.

#include <adc/core/model/coupled_system.hpp>
#include <adc/core/state/state.hpp>
#include <adc/coupling/static_system/amr_system_coupler.hpp>
#include <adc/numerics/time/amr_reflux_mf.hpp>  // AmrLevelMP
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;

// Modele de production constante (sans flux) : source = +rate, elliptic_rhs = u[0].
struct Prod {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  Real rate = Real(1);
  ADC_HD State flux(const State&, const Aux&, int) const { return State{}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State&, const Aux&) const { return State{rate}; }
  ADC_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

struct ZeroRhs {
  template <class System>
  void operator()(const System&, MultiFab& rhs) const {
    rhs.set_val(Real(0));
  }
};

// Construit un AmrSystemCoupler mono-niveau (grossier seul) pour un CoupledSystem
// {FastBlk, SlowBlk}. Les MultiFab sont construits localement et DEPLACE dans les
// niveaux ; les references retournees dans out_fast/out_slow permettent d'interroger
// la valeur apres l'avance.
template <class FastBlk, class SlowBlk>
static auto make_sim(const Geometry& geom, const BoxArray& ba, const DistributionMapping& dm,
                     MultiFab& out_fast,  // sortie : pointe sur le MultiFab du bloc rapide
                     MultiFab& out_slow)  // sortie : pointe sur le MultiFab du bloc lent
{
  MultiFab Uf(ba, dm, 1, 2), Us(ba, dm, 1, 2);
  Uf.set_val(Real(0));
  Us.set_val(Real(0));

  // Copies pour exposer les valeurs au test (le coupler possede les MultiFab internes).
  out_fast = MultiFab(ba, dm, 1, 2);
  out_slow = MultiFab(ba, dm, 1, 2);
  out_fast.set_val(Real(0));
  out_slow.set_val(Real(0));

  FastBlk fast{"fast", Prod{Real(1)}, Uf, BCRec{}};
  SlowBlk slow{"slow", Prod{Real(1)}, Us, BCRec{}};
  CoupledSystem system{fast, slow};

  // Mono-niveau : un seul niveau (grossier), pas de raffinement.
  const Real dx = geom.dx(), dy = geom.dy();
  std::vector<std::vector<AmrLevelMP>> bl;
  bl.emplace_back();
  bl.back().push_back(AmrLevelMP{std::move(Uf), nullptr, dx, dy});
  bl.emplace_back();
  bl.back().push_back(AmrLevelMP{std::move(Us), nullptr, dx, dy});

  return AmrSystemCoupler(system, geom, ba, BCRec{}, ZeroRhs{}, std::move(bl));
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int NC = 4;
  const Box2D dom = Box2D::from_extents(NC, NC);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const BoxArray ba(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  const int ncell = NC * NC;  // 16 cellules
  const Real dt = Real(0.1);

  // -----------------------------------------------------------------------
  // Test A : bloc rapide stride=1, bloc lent stride=3.
  //   ExplicitTime<SSPRK2, substeps=1, stride>.
  // -----------------------------------------------------------------------
  {
    constexpr int M = 3;
    using FastBlk = EquationBlock<Prod, FirstOrder, ExplicitTime<SSPRK2, 1, 1>>;
    using SlowBlk = EquationBlock<Prod, FirstOrder, ExplicitTime<SSPRK2, 1, M>>;
    static_assert(FastBlk::Time::stride == 1);
    static_assert(SlowBlk::Time::stride == M);

    MultiFab Uf_out, Us_out;  // non utilise ici -- on lit via sim.coarse()
    FastBlk fast{"fast", Prod{Real(1)}, Uf_out, BCRec{}};
    SlowBlk slow{"slow", Prod{Real(1)}, Us_out, BCRec{}};

    MultiFab Uf(ba, dm, 1, 2), Us(ba, dm, 1, 2);
    Uf.set_val(Real(0));
    Us.set_val(Real(0));
    FastBlk f2{"fast", Prod{Real(1)}, Uf, BCRec{}};
    SlowBlk s2{"slow", Prod{Real(1)}, Us, BCRec{}};
    CoupledSystem system{f2, s2};

    const Real dx = geom.dx(), dy = geom.dy();
    std::vector<std::vector<AmrLevelMP>> bl;
    bl.emplace_back();
    bl.back().push_back(AmrLevelMP{std::move(Uf), nullptr, dx, dy});
    bl.emplace_back();
    bl.back().push_back(AmrLevelMP{std::move(Us), nullptr, dx, dy});

    AmrSystemCoupler sim(system, geom, ba, BCRec{}, ZeroRhs{}, std::move(bl));

    // -- macro-pas 0 (macro_step_=0 avant ++, verifie apres) :
    //    (0+1)%3 = 1 != 0 -> bloc lent TENU.
    //    Bloc rapide avance de dt.
    sim.step(dt);
    const Real uf0 = sum(sim.coarse(0), 0) / Real(ncell);  // valeur moyenne rapide
    const Real us0 = sum(sim.coarse(1), 0) / Real(ncell);  // valeur moyenne lente
    chk(std::fabs(uf0 - Real(0.1)) < Real(1e-12), "A_fast_advances_at_mac0");
    chk(std::fabs(us0 - Real(0.0)) < Real(1e-12),
        "A_slow_held_at_mac0");  // BUG : ancienne version donnait 0.3 ici

    // -- macro-pas 1 (macro_step_=1) : (1+1)%3 = 2 != 0 -> bloc lent encore TENU.
    sim.step(dt);
    const Real uf1 = sum(sim.coarse(0), 0) / Real(ncell);
    const Real us1 = sum(sim.coarse(1), 0) / Real(ncell);
    chk(std::fabs(uf1 - Real(0.2)) < Real(1e-12), "A_fast_advances_at_mac1");
    chk(std::fabs(us1 - Real(0.0)) < Real(1e-12), "A_slow_held_at_mac1");

    // -- macro-pas 2 (macro_step_=2) : (2+1)%3 = 0 -> bloc lent RATTRAPE (3*dt).
    sim.step(dt);
    const Real uf2 = sum(sim.coarse(0), 0) / Real(ncell);
    const Real us2 = sum(sim.coarse(1), 0) / Real(ncell);
    chk(std::fabs(uf2 - Real(0.3)) < Real(1e-12), "A_fast_at_mac2");
    chk(std::fabs(us2 - Real(0.3)) < Real(1e-12),
        "A_slow_catchup_at_mac2");  // synchronises a 0.3

    // Invariant de couplage : le bloc lent n'est JAMAIS en avance sur le rapide.
    // Apres chaque macro-pas, slow_time <= fast_time.
    // (Verifie implicitement : us0=0 <= uf0=0.1, us1=0 <= uf1=0.2, us2=0.3 == uf2=0.3)
    chk(us0 <= uf0 + Real(1e-14), "A_coupling_invariant_mac0");
    chk(us1 <= uf1 + Real(1e-14), "A_coupling_invariant_mac1");
    chk(us2 <= uf2 + Real(1e-14), "A_coupling_invariant_mac2");
  }

  // -----------------------------------------------------------------------
  // Test A2 : stride=5 (M=5). Le lent est tenu 4 pas, rattrape au 5eme.
  // -----------------------------------------------------------------------
  {
    constexpr int M = 5;
    using FastBlk = EquationBlock<Prod, FirstOrder, ExplicitTime<SSPRK2, 1, 1>>;
    using SlowBlk = EquationBlock<Prod, FirstOrder, ExplicitTime<SSPRK2, 1, M>>;

    MultiFab Uf(ba, dm, 1, 2), Us(ba, dm, 1, 2);
    Uf.set_val(Real(0));
    Us.set_val(Real(0));
    FastBlk f{"fast", Prod{Real(1)}, Uf, BCRec{}};
    SlowBlk s{"slow", Prod{Real(1)}, Us, BCRec{}};
    CoupledSystem system{f, s};

    const Real dx = geom.dx(), dy = geom.dy();
    std::vector<std::vector<AmrLevelMP>> bl;
    bl.emplace_back();
    bl.back().push_back(AmrLevelMP{std::move(Uf), nullptr, dx, dy});
    bl.emplace_back();
    bl.back().push_back(AmrLevelMP{std::move(Us), nullptr, dx, dy});
    AmrSystemCoupler sim(system, geom, ba, BCRec{}, ZeroRhs{}, std::move(bl));

    // Pas 0..3 : lent tenu.
    for (int mac = 0; mac < M - 1; ++mac) {
      sim.step(dt);
      const Real us = sum(sim.coarse(1), 0) / Real(ncell);
      chk(std::fabs(us - Real(0.0)) < Real(1e-12), "A2_slow_held");
    }
    // Pas 4 (mac=M-1) : lent rattrape.
    sim.step(dt);
    const Real uf = sum(sim.coarse(0), 0) / Real(ncell);
    const Real us = sum(sim.coarse(1), 0) / Real(ncell);
    chk(std::fabs(uf - Real(0.5)) < Real(1e-12), "A2_fast_final");
    chk(std::fabs(us - Real(0.5)) < Real(1e-12), "A2_slow_catchup");
    chk(us <= uf + Real(1e-14), "A2_coupling_invariant");
  }

  // -----------------------------------------------------------------------
  // Test B : stride=1 -> avance a CHAQUE macro-pas (bit-identique historique).
  //   Apres N pas, u = N * dt * rate (production constante, SSPRK2 exact sur
  //   une source constante).
  // -----------------------------------------------------------------------
  {
    using Blk = EquationBlock<Prod, FirstOrder, ExplicitTime<SSPRK2, 1, 1>>;
    static_assert(Blk::Time::stride == 1);

    MultiFab Uc(ba, dm, 1, 2);
    Uc.set_val(Real(0));
    Blk blk{"b", Prod{Real(1)}, Uc, BCRec{}};
    CoupledSystem system{blk};

    const Real dx = geom.dx(), dy = geom.dy();
    std::vector<std::vector<AmrLevelMP>> bl;
    bl.emplace_back();
    bl.back().push_back(AmrLevelMP{std::move(Uc), nullptr, dx, dy});
    AmrSystemCoupler sim(system, geom, ba, BCRec{}, ZeroRhs{}, std::move(bl));

    const int N = 5;
    for (int i = 0; i < N; ++i)
      sim.step(dt);
    const Real u = sum(sim.coarse(0), 0) / Real(ncell);
    // SSPRK2 exact pour u' = 1 : u(N*dt) = N*dt.
    chk(std::fabs(u - Real(N) * dt) < Real(1e-12), "B_stride1_advances_every_step");
  }

  if (fails == 0)
    std::printf("OK test_amr_stride_cadence\n");
  return fails == 0 ? 0 : 1;
}
