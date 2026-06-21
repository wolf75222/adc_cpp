// A3 : conservation de la masse totale par la source de couplage sur AMR, mesuree par une
// somme FEUILLE-CONSCIENTE (pas la somme naive du seul niveau grossier).
//
// Une source de couplage conservative verifie S_0 + S_1 = 0 par cellule (la matiere passe
// d'une espece a l'autre, rien ne se cree). La masse physique d'une espece sur la hierarchie
// est sa somme FEUILLE-CONSCIENTE : cellules grossieres NON couvertes a dVc + region couverte
// comptee depuis le niveau FIN a dVf (le fin est la verite la ou il existe).
//
// Deux mesures, deux roles :
//   (1) garde-fou PHYSIQUE : la masse totale feuille-consciente (n0 + n1) est invariante par
//       une source conservative appliquee par niveau, AVANT comme APRES le correctif (la
//       conservation cellule-locale a chaque niveau ne depend pas de l'average_down).
//   (2) discriminant du CORRECTIF : la masse GROSSIERE naive d'une espece (le diagnostic
//       amr_mass = mass(b), somme du seul niveau grossier) doit EGALER sa masse feuille-
//       consciente. Sous champ non uniforme et source NON LINEAIRE, la cellule grossiere
//       couverte recoit S(grossier) alors que ses enfants recoivent moyenne(S(fin)) ; sans
//       l'average_down trailing, mass(b) derive de la masse feuille-consciente. Le correctif
//       ramene les cellules couvertes a la moyenne de leurs enfants -> les deux coincident.
//       (n0 + n1 seul ne discrimine PAS : conservatif par cellule, donc invariant meme sur une
//       cellule couverte incoherente ; il faut mesurer une espece a la fois.)

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

struct Inert {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  ADC_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State&, const Aux&) const { return State{Real(0)}; }
  ADC_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

struct ZeroSystemRhs {
  template <class System>
  void operator()(const System&, MultiFab& rhs) const {
    rhs.set_val(Real(0));
  }
};

// Echange NON LINEAIRE mais CONSERVATIF entre deux blocs : flux = dt*k*(n1^2 - n0^2) par
// cellule ; n0 += flux, n1 -= flux. n0 + n1 invariant cellule par cellule (conservatif), mais
// la non-linearite fait S(moyenne) != moyenne(S) -> sans average_down la cellule couverte
// porte une masse grossiere fantome qui casse la somme grossiere naive.
struct NonlinearExchange {
  Real k = Real(0.4);
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
        const Real n0 = a0(i, j, 0), n1 = a1(i, j, 0);
        const Real flux = c * (n1 * n1 - n0 * n0);
        a0(i, j, 0) = n0 + flux;
        a1(i, j, 0) = n1 - flux;
      });
    }
  }
};

template <class F>
static void fill_by_index(MultiFab& U, F f) {
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 a = U.fab(li).array();
    const Box2D g = U.fab(li).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i)
        a(i, j, 0) = f(i, j);
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
  const Real dVc = dxc * dyc, dVf = (dxc / 2) * (dyc / 2);

  const Box2D fbox{{8, 8}, {23, 23}};  // couvre coarse [4..11]^2 a ratio 2
  const BoxArray ba_fine(std::vector<Box2D>{fbox});
  const int CLO = 4, CHI = 11;

  auto make_level = [&](MultiFab&& U, Real dx, Real dy) {
    return AmrLevelMP{std::move(U), nullptr, dx, dy};
  };
  auto covered = [&](int ci, int cj) { return ci >= CLO && ci <= CHI && cj >= CLO && cj <= CHI; };

  // masse FEUILLE-CONSCIENTE d'UNE espece (comp 0) : grossier NON couvert a dVc + fin a dVf
  // (le fin est la verite la ou il existe). C'est la masse physique de l'espece sur la
  // hierarchie, independante de la coherence des cellules couvertes.
  auto leaf_mass = [&](const MultiFab& Uc, const MultiFab& Uf) {
    device_fence();
    const ConstArray4 c = Uc.fab(0).const_array();
    const ConstArray4 f = Uf.fab(0).const_array();
    Real M = 0;
    for (int cj = 0; cj < NC; ++cj)
      for (int ci = 0; ci < NC; ++ci)
        if (!covered(ci, cj))
          M += c(ci, cj, 0) * dVc;
    for (int J = fbox.lo[1]; J <= fbox.hi[1]; ++J)
      for (int I = fbox.lo[0]; I <= fbox.hi[0]; ++I)
        M += f(I, J, 0) * dVf;
    return M;
  };

  using Blk = EquationBlock<Inert, FirstOrder, ExplicitTime<SSPRK2, 1>>;

  // champ non uniforme et DIFFERENT par espece -> n1^2 - n0^2 non trivial, enfants distincts.
  auto p0 = [](int i, int j) {
    return Real(1) + Real(0.04) * static_cast<Real>(i) - Real(0.02) * static_cast<Real>(j);
  };
  auto p1 = [](int i, int j) {
    return Real(2) - Real(0.03) * static_cast<Real>(i) + Real(0.05) * static_cast<Real>(j);
  };

  MultiFab U0c(ba_coarse, dm, 1, 2), U0f(ba_fine, dm, 1, 2);
  MultiFab U1c(ba_coarse, dm, 1, 2), U1f(ba_fine, dm, 1, 2);
  fill_by_index(U0c, p0);
  fill_by_index(U0f, p0);
  fill_by_index(U1c, p1);
  fill_by_index(U1f, p1);

  Blk b0{"a", Inert{}, U0c, BCRec{}};
  Blk b1{"b", Inert{}, U1c, BCRec{}};
  CoupledSystem system{b0, b1};
  std::vector<std::vector<AmrLevelMP>> bl;
  bl.emplace_back();
  bl.back().push_back(make_level(std::move(U0c), dxc, dyc));
  bl.back().push_back(make_level(std::move(U0f), dxc / 2, dyc / 2));
  bl.emplace_back();
  bl.back().push_back(make_level(std::move(U1c), dxc, dyc));
  bl.back().push_back(make_level(std::move(U1f), dxc / 2, dyc / 2));
  AmrSystemCoupler sim(system, geom, ba_coarse, BCRec{}, ZeroSystemRhs{}, std::move(bl));

  // solve_fields synchronise (average_down) avant la mesure de reference.
  sim.solve_fields();
  // masse FEUILLE-CONSCIENTE totale (n0 + n1) avant la source : invariant physique de
  // reference d'une source conservative (la matiere passe d'une espece a l'autre).
  const Real leaf0_b0 = leaf_mass(sim.levels(0)[0].U, sim.levels(0)[1].U);
  const Real leaf0_b1 = leaf_mass(sim.levels(1)[0].U, sim.levels(1)[1].U);
  const Real leaf_tot0 = leaf0_b0 + leaf0_b1;

  // garde anti faux-positif : la reference doit etre finie et non triviale.
  chk(std::isfinite(leaf_tot0), "ref_mass_finite");
  chk(std::fabs(leaf_tot0) > Real(1e-6), "ref_mass_nonzero");
  // a l'init coherent (covered = moyenne des enfants), la masse grossiere naive d'une espece
  // egale sa masse feuille-consciente.
  chk(std::fabs(sim.mass(0) - leaf0_b0) < Real(1e-12), "ref_b0_coarse_eq_leaf");
  chk(std::fabs(sim.mass(1) - leaf0_b1) < Real(1e-12), "ref_b1_coarse_eq_leaf");

  // plusieurs pas de source conservative non lineaire.
  for (int s = 0; s < 3; ++s)
    sim.coupled_source_step(NonlinearExchange{Real(0.4)}, Real(0.05));

  const Real leaf1_b0 = leaf_mass(sim.levels(0)[0].U, sim.levels(0)[1].U);
  const Real leaf1_b1 = leaf_mass(sim.levels(1)[0].U, sim.levels(1)[1].U);
  const Real leaf_tot1 = leaf1_b0 + leaf1_b1;
  const Real coarse1_b0 = sim.mass(0), coarse1_b1 = sim.mass(1);

  // (garde) rejette nan/inf avant toute tolerance.
  chk(std::isfinite(leaf_tot1) && std::isfinite(coarse1_b0) && std::isfinite(coarse1_b1),
      "post_mass_finite");
  // (1) garde-fou physique : la masse TOTALE feuille-consciente (n0 + n1) est conservee a
  // ~machine par la source conservative (vrai independamment du correctif : la source
  // conserve cellule par cellule a chaque niveau).
  chk(std::fabs(leaf_tot1 - leaf_tot0) < Real(1e-12), "leaf_total_mass_conserved");
  // (2) c'est ICI que mord le correctif : la masse GROSSIERE naive d'une espece (le
  // diagnostic amr_mass) doit egaler sa masse FEUILLE-CONSCIENTE. Sous champ non uniforme +
  // source non lineaire, sans l'average_down trailing la cellule grossiere couverte porte
  // S(grossier) au lieu de moyenne(S(fin)), donc mass(b) (grossier seul) derive de la masse
  // feuille-consciente. Le correctif ramene les cellules couvertes a la moyenne des enfants
  // -> les deux mesures coincident a ~machine.
  chk(std::fabs(coarse1_b0 - leaf1_b0) < Real(1e-12), "b0_coarse_eq_leaf_after_source");
  chk(std::fabs(coarse1_b1 - leaf1_b1) < Real(1e-12), "b1_coarse_eq_leaf_after_source");

  // sanity : la source a REELLEMENT redistribue (sinon conservation triviale).
  device_fence();
  const Real n0_lo = sim.levels(0)[1].U.fab(0).const_array()(2 * CLO, 2 * CLO, 0);
  chk(std::fabs(n0_lo - p0(2 * CLO, 2 * CLO)) > Real(1e-4), "source_active");

  if (fails == 0)
    std::printf("OK test_amr_composite_source_conservation\n");
  return fails == 0 ? 0 : 1;
}
