// Second membre du Poisson de SYSTEME : f = somme_s elliptic_rhs_s(u_s).
//
// Le second membre assemble par le System (solve_fields : rhs.set_val(0) puis, pour chaque bloc,
// add_poisson_rhs(U, rhs) += elliptic_rhs(U)) est GENERIQUE : il somme la brique elliptique de
// chaque bloc, sans supposer la forme densite de charge q n. On le prouve ici a l'echelle de la
// brique make_poisson_rhs(model) (block_builder.hpp), qui produit exactement cette accumulation :
//
//   (A) make_poisson_rhs(BackgroundDensity) ecrit f = alpha (n - n0) cellule par cellule
//       (brique != charge), bit-identique a une reference manuelle.
//   (B) La SOMME de deux briques DIFFERENTES (charge q0 n + gravite sign 4piG (rho - rho0)) sur
//       deux blocs, accumulee comme le fait solve_fields, egale la somme manuelle, bit a bit.
//
// Aucune tolerance : operator!= strict (le test echoue au moindre ecart de dernier bit).

#include <adc/physics/composite.hpp>
#include <adc/physics/elliptic.hpp>
#include <adc/runtime/block_builder.hpp>  // make_poisson_rhs

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;

// Modele scalaire minimal (1 var) : seule la brique elliptique compte ici. Le transport et la
// source sont triviaux (la brique make_poisson_rhs ne lit que elliptic_rhs).
template <class Elliptic>
struct ScalarElliptic {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  Elliptic ell{};
  ADC_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State&, const Aux&) const { return State{Real(0)}; }
  ADC_HD Real elliptic_rhs(const State& u) const { return ell.rhs(u); }
};

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  const int N = 32;
  Box2D dom = Box2D::from_extents(N, N);
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, 1);

  // Densites de test (deux champs distincts).
  MultiFab U0(ba, dm, 1, 0), U1(ba, dm, 1, 0);
  {
    Array4 a0 = U0.fab(0).array(), a1 = U1.fab(0).array();
    const Box2D v = U0.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        a0(i, j, 0) = Real(1) + Real(0.3) * std::sin(Real(0.2) * i) * std::cos(Real(0.1) * j);
        a1(i, j, 0) = Real(0.7) + Real(0.4) * std::cos(Real(0.15) * i + Real(0.05) * j);
      }
  }

  // Parametres des briques (non triviaux : alpha != 1, n0 != 0, sign = -1).
  const Real q0 = Real(-0.8);
  const Real alpha = Real(1.3), n0 = Real(0.4);
  const Real sign = Real(-1), fourpiG = Real(2.5), rho0 = Real(0.6);

  // (A) BackgroundDensity (brique != charge) : f = alpha (n - n0).
  {
    ScalarElliptic<BackgroundDensity> m{BackgroundDensity{alpha, n0}};
    auto rhs_fn = make_poisson_rhs(m);
    MultiFab rhs(ba, dm, 1, 0);
    rhs.set_val(Real(0));
    rhs_fn(U0, rhs);  // f = alpha (n - n0)

    bool bit_eq = true;
    const ConstArray4 r = rhs.fab(0).const_array();
    const ConstArray4 a0 = U0.fab(0).const_array();
    const Box2D v = rhs.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        const Real ref = alpha * (a0(i, j, 0) - n0);
        if (r(i, j, 0) != ref) bit_eq = false;
      }
    chk(bit_eq, "background_rhs_bit_identique");
  }

  // (B) Somme de DEUX briques DIFFERENTES, accumulee comme solve_fields :
  //     rhs.set_val(0) ; bloc0 (charge q0 n) puis bloc1 (gravite sign 4piG (rho - rho0)).
  {
    ScalarElliptic<ChargeDensity> m0{ChargeDensity{q0}};
    ScalarElliptic<GravityCoupling> m1{GravityCoupling{sign, fourpiG, rho0}};
    auto f0 = make_poisson_rhs(m0);
    auto f1 = make_poisson_rhs(m1);

    MultiFab rhs(ba, dm, 1, 0);
    rhs.set_val(Real(0));
    f0(U0, rhs);  // += q0 n0
    f1(U1, rhs);  // += sign 4piG (n1 - rho0)

    // Reference : la MEME accumulation, mais en appelant DIRECTEMENT les bricks (elliptic_rhs) dans
    // le meme ordre que solve_fields. On valide la LOGIQUE de composition / sommation (set_val(0)
    // puis += brique de chaque bloc), bit a bit, sans re-deriver la formule a la main (une seconde
    // ecriture de l'expression peut differer d'un ULP par contraction FP, ce qui ne dit rien de la
    // composition). m0 / m1 sont les memes briques que celles capturees par f0 / f1.
    bool bit_eq = true;
    const ConstArray4 r = rhs.fab(0).const_array();
    const ConstArray4 a0 = U0.fab(0).const_array();
    const ConstArray4 a1 = U1.fab(0).const_array();
    const Box2D v = rhs.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        Real ref = Real(0);
        ref += m0.elliptic_rhs(StateVec<1>{a0(i, j, 0)});  // charge q0 n0
        ref += m1.elliptic_rhs(StateVec<1>{a1(i, j, 0)});  // gravite sign 4piG (n1 - rho0)
        if (r(i, j, 0) != ref) bit_eq = false;
      }
    chk(bit_eq, "somme_charge_plus_gravite_bit_identique");

    // Garde-fou : la somme n'est PAS celle de deux blocs de charge (sinon le second membre serait
    // fige sur q n). Au moins une cellule doit differer de q0 n0 + q0 n1.
    bool differs = false;
    for (int j = v.lo[1]; j <= v.hi[1] && !differs; ++j)
      for (int i = v.lo[0]; i <= v.hi[0] && !differs; ++i) {
        const Real charge_only = q0 * a0(i, j, 0) + q0 * a1(i, j, 0);
        if (r(i, j, 0) != charge_only) differs = true;
      }
    chk(differs, "gravite_distincte_de_charge");
  }

  if (fails == 0) std::printf("OK test_elliptic_composite_rhs\n");
  return fails == 0 ? 0 : 1;
}
