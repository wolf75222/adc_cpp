// A3 : coherence des cellules grossieres COUVERTES apres une source appliquee par niveau.
//
// Contexte. AmrSystemCoupler applique la source de couplage (coupled_source_step) et la
// source implicite (AmrImplicitSourceStepper) INDEPENDAMMENT sur chaque niveau de la
// hierarchie. Une cellule grossiere COUVERTE par un patch fin ne represente pas de matiere
// propre : elle doit, par construction conservative, valoir la moyenne 2x2 de ses quatre
// enfants fins. Quand la source est NON LINEAIRE ou le champ NON UNIFORME, appliquer la
// source au grossier donne S(moyenne) alors que les enfants recoivent moyenne(S) : les deux
// different, et la cellule couverte derive de la moyenne de ses enfants. Le diagnostic
// amr_mass (somme du seul niveau grossier) compte alors une source grossiere fantome sous le
// patch. La cascade fin -> grossier (mf_average_down_mb) ajoutee a la fin de la source
// retablit l'invariant. Ce test echoue AVANT le correctif (covered != moyenne des enfants).
//
// On verifie l'invariant sur les DEUX chemins : la source de couplage explicite
// (coupled_source_step) et la source implicite par defaut (AmrImplicitSourceStepper).

#include <adc/core/coupled_system.hpp>
#include <adc/core/state.hpp>
#include <adc/coupling/amr_system_coupler.hpp>
#include <adc/numerics/time/amr_reflux_mf.hpp>  // AmrLevelMP
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/refinement.hpp>  // coarsen_index

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;

// Bloc inerte (pas de flux) : on isole l'effet de la source par niveau, sans transport.
struct Inert {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  Real rate = Real(1);
  ADC_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  // source NON LINEAIRE : -rate * u^2. Stiff-friendly (decroissance), exploite par le
  // chemin implicite (backward_euler_source). S(moyenne) != moyenne(S) en general.
  ADC_HD State source(const State& u, const Aux&) const {
    return State{-rate * u[0] * u[0]};
  }
  ADC_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

struct ZeroSystemRhs {
  template <class System>
  void operator()(const System&, MultiFab& rhs) const { rhs.set_val(Real(0)); }
};

// Source de couplage NON LINEAIRE sur un bloc : n0 -= dt * k * n0^2 (par cellule). Non
// lineaire en l'etat -> S(moyenne) != moyenne(S), ce qui rend la cellule couverte
// incoherente avec ses enfants si on n'average_down pas apres.
struct NonlinearSelfSink {
  Real k = Real(0.7);
  template <CoupledSystemLike System>
  void apply(System& sys, const MultiFab& /*aux*/, Real dt) const {
    MultiFab& U0 = sys.template block<0>().U();
    const Real coef = k * dt;
    for (int li = 0; li < U0.local_size(); ++li) {
      Array4 a0 = U0.fab(li).array();
      const Box2D b = U0.box(li);
      const Real c = coef;
      for_each_cell(b, [=] ADC_HD(int i, int j) {
        const Real u = a0(i, j, 0);
        a0(i, j, 0) = u - c * u * u;
      });
    }
  }
};

// Remplit U (comp 0) par une fonction des indices DE CE NIVEAU : grossier et fin portent
// des profils distincts mais coherents a l'init (le fin echantillonne au centre fin), de
// sorte que les 4 enfants d'une cellule couverte DIFFERENT entre eux (champ non uniforme).
template <class F>
static void fill_by_index(MultiFab& U, F f) {
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 a = U.fab(li).array();
    const Box2D g = U.fab(li).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) a(i, j, 0) = f(i, j);
  }
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };
  // garde anti faux-positif : rejette nan/inf AVANT toute comparaison de tolerance, car en
  // C++ "nan < tol" est faux et passerait silencieusement le test.
  auto finite_close = [&](Real got, Real want, Real tol, const char* w) {
    if (!std::isfinite(got) || !std::isfinite(want)) {
      std::printf("FAIL %s (non finite : got=%g want=%g)\n", w, got, want);
      ++fails;
      return;
    }
    chk(std::fabs(got - want) < tol, w);
  };

  const int NC = 16;
  const Box2D dom = Box2D::from_extents(NC, NC);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const BoxArray ba_coarse(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  const Real dxc = geom.dx(), dyc = geom.dy();

  // patch fin sur les cellules grossieres [4..11]^2 (ratio 2) -> box fine {{8,8},{23,23}}.
  const Box2D fbox{{8, 8}, {23, 23}};
  const BoxArray ba_fine(std::vector<Box2D>{fbox});
  // domaine d'indices du niveau fin (pour borner la lecture des enfants).
  const int CLO = 4, CHI = 11;  // plage des cellules grossieres couvertes

  auto make_level = [&](MultiFab&& U, Real dx, Real dy) {
    return AmrLevelMP{std::move(U), nullptr, dx, dy};
  };

  // champ NON UNIFORME : varie avec i et j -> les 4 enfants d'une cellule couverte different.
  auto profile = [](int i, int j) {
    return Real(1) + Real(0.05) * static_cast<Real>(i) + Real(0.03) * static_cast<Real>(j);
  };

  // verifie, pour chaque cellule grossiere couverte, que covered == moyenne 2x2 des enfants.
  auto check_coherence = [&](const MultiFab& Uc, const MultiFab& Uf, const char* tag) {
    device_fence();
    const ConstArray4 c = Uc.fab(0).const_array();
    const ConstArray4 f = Uf.fab(0).const_array();
    Real worst = 0;
    for (int cj = CLO; cj <= CHI; ++cj)
      for (int ci = CLO; ci <= CHI; ++ci) {
        const Real avg = Real(0.25) * (f(2 * ci, 2 * cj, 0) + f(2 * ci + 1, 2 * cj, 0) +
                                       f(2 * ci, 2 * cj + 1, 0) + f(2 * ci + 1, 2 * cj + 1, 0));
        const Real cov = c(ci, cj, 0);
        if (!std::isfinite(cov) || !std::isfinite(avg)) {
          std::printf("FAIL %s (non finite a (%d,%d) : cov=%g avg=%g)\n", tag, ci, cj, cov, avg);
          ++fails;
          return;
        }
        worst = std::max(worst, std::fabs(cov - avg));
      }
    chk(worst < Real(1e-12), tag);
  };

  // --- chemin A : source de COUPLAGE explicite (coupled_source_step) ---
  {
    using Blk = EquationBlock<Inert, FirstOrder, ExplicitTime<SSPRK2, 1>>;
    MultiFab Uc(ba_coarse, dm, 1, 2), Uf(ba_fine, dm, 1, 2);
    fill_by_index(Uc, profile);
    fill_by_index(Uf, profile);  // coherent a l'init : covered == moyenne enfants
    Blk b0{"a", Inert{}, Uc, BCRec{}};
    CoupledSystem system{b0};
    std::vector<std::vector<AmrLevelMP>> bl;
    bl.emplace_back();
    bl.back().push_back(make_level(std::move(Uc), dxc, dyc));
    bl.back().push_back(make_level(std::move(Uf), dxc / 2, dyc / 2));
    AmrSystemCoupler sim(system, geom, ba_coarse, BCRec{}, ZeroSystemRhs{}, std::move(bl));

    // source non lineaire par niveau : sans average_down, covered = S(grossier) derive de la
    // moyenne des enfants = moyenne(S(fin)).
    sim.coupled_source_step(NonlinearSelfSink{Real(0.7)}, Real(0.1));
    check_coherence(sim.levels(0)[0].U, sim.levels(0)[1].U, "coupled_covered_eq_fine_avg");

    // sanity : la source a REELLEMENT bouge l'etat (sinon coherence triviale a l'init).
    device_fence();
    const Real uf_lo = sim.levels(0)[1].U.fab(0).const_array()(2 * CLO, 2 * CLO, 0);
    chk(std::fabs(uf_lo - profile(2 * CLO, 2 * CLO)) > Real(1e-4), "coupled_source_active");
  }

  // --- chemin B : source IMPLICITE par defaut (AmrImplicitSourceStepper via step) ---
  {
    using BlkImpl = EquationBlock<Inert, FirstOrder, ImplicitTime<UserTimeIntegrator, 1>>;
    static_assert(BlkImpl::Time::treatment == TimeTreatment::Implicit);
    MultiFab Uc(ba_coarse, dm, 1, 2), Uf(ba_fine, dm, 1, 2);
    fill_by_index(Uc, profile);
    fill_by_index(Uf, profile);
    BlkImpl b0{"a", Inert{Real(1)}, Uc, BCRec{}};
    CoupledSystem system{b0};
    std::vector<std::vector<AmrLevelMP>> bl;
    bl.emplace_back();
    bl.back().push_back(make_level(std::move(Uc), dxc, dyc));
    bl.back().push_back(make_level(std::move(Uf), dxc / 2, dyc / 2));
    AmrSystemCoupler sim(system, geom, ba_coarse, BCRec{}, ZeroSystemRhs{}, std::move(bl));

    // backward_euler_source resout -rate*u^2 par niveau (non lineaire) -> meme incoherence.
    sim.step(Real(0.2), AmrImplicitSourceStepper{});
    check_coherence(sim.levels(0)[0].U, sim.levels(0)[1].U, "implicit_covered_eq_fine_avg");
  }

  // --- chemin C : hierarchie MONO-NIVEAU -> aucune cellule couverte, average_down no-op.
  // On verifie que l'etat apres source est EXACTEMENT S(u) (bit-identique a l'historique :
  // la cascade fin->grossier ajoutee ne doit rien changer sans niveau fin).
  {
    using Blk = EquationBlock<Inert, FirstOrder, ExplicitTime<SSPRK2, 1>>;
    MultiFab Uc(ba_coarse, dm, 1, 2);
    fill_by_index(Uc, profile);
    Blk b0{"a", Inert{}, Uc, BCRec{}};
    CoupledSystem system{b0};
    std::vector<std::vector<AmrLevelMP>> bl;
    bl.emplace_back();
    bl.back().push_back(make_level(std::move(Uc), dxc, dyc));  // grossier seul
    AmrSystemCoupler sim(system, geom, ba_coarse, BCRec{}, ZeroSystemRhs{}, std::move(bl));

    const Real k = Real(0.7), dt = Real(0.1);
    sim.coupled_source_step(NonlinearSelfSink{k}, dt);
    device_fence();
    const ConstArray4 c = sim.levels(0)[0].U.fab(0).const_array();
    Real worst = 0;
    for (int j = 0; j < NC; ++j)
      for (int i = 0; i < NC; ++i) {
        const Real u0 = profile(i, j);
        const Real want = u0 - k * dt * u0 * u0;  // S applique une fois, pas d'average_down
        const Real got = c(i, j, 0);
        if (!std::isfinite(got)) {
          std::printf("FAIL single_level_bit_identical (non finite a (%d,%d))\n", i, j);
          ++fails;
          j = NC; break;
        }
        worst = std::max(worst, std::fabs(got - want));
      }
    finite_close(worst, Real(0), Real(1e-14), "single_level_bit_identical");
  }

  if (fails == 0) std::printf("OK test_amr_source_covered_cells\n");
  return fails == 0 ? 0 : 1;
}
