// Identite NUMERIQUE EXACTE des types nommes de include/adc/elliptic :
// EllipticProblem et FieldPostProcess sont des descripteurs structurels qui
// NOMMENT des valeurs et conventions deja codees, sans changer une seule
// operation flottante. Le test prouve la bit-identite par operator== strict
// (pas une tolerance) : il echouerait au moindre ecart de dernier bit.
//
//   (A) EllipticProblem : GeometricMG construit via le constructeur BCRec et via
//       la fabrique make_elliptic_solver(EllipticProblem) sur le MEME cas
//       manufacture donne phi identique cellule par cellule. homogeneous_bc d'un
//       probleme egale homogeneous(problem.bc).
//   (B) FieldPostProcess : la boucle de reference recopiee de
//       detail::coupler_grad_phi egale field_postprocess(GradSign::Plus) bit a
//       bit. Le cas GradSign::Minus egale -reference (convention two_fluid).
//   (C) garde-fou eps : EllipticProblem{}.eps vaut exactement 1.

#include <adc/numerics/elliptic/interface/elliptic_problem.hpp>
#include <adc/numerics/elliptic/mg/geometric_mg.hpp>
#include <adc/numerics/elliptic/poisson/poisson_operator.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/layout/distribution_mapping.hpp>
#include <adc/mesh/execution/for_each.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/mf_arith.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/mesh/boundary/physical_bc.hpp>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int N = 64;
  Box2D dom = Box2D::from_extents(N, N);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, 1);
  BCRec bc;  // periodique

  auto fr = [&](int i, int j) {
    return std::sin(2 * kPi * geom.x_cell(i)) * std::sin(2 * kPi * geom.y_cell(j));
  };

  // (C) garde-fou eps : etat implicite actuel du stencil.
  chk(EllipticProblem{}.eps == Real(1), "eps_implicite_vaut_1");

  // (C bis) eps != 1 n'est pas supporte (stencil a coefficient constant) :
  // make_elliptic_solver doit LANCER plutot que de l'ignorer en silence.
  {
    EllipticProblem prob_eps2{Real(2), bc, false};
    bool a_lance = false;
    try {
      GeometricMG bad = make_elliptic_solver<GeometricMG>(geom, ba, prob_eps2);
      (void)bad;
    } catch (const std::invalid_argument&) {
      a_lance = true;
    }
    chk(a_lance, "eps_different_de_1_lance");
  }

  // homogeneous_bc(probleme) == homogeneous(probleme.bc), champ par champ.
  {
    BCRec mixed;
    mixed.xlo = BCType::Dirichlet;
    mixed.xlo_val = 3.0;
    mixed.yhi = BCType::Foextrap;
    EllipticProblem prob{Real(1), mixed, false};
    BCRec h0 = homogeneous(prob.bc);
    BCRec h1 = homogeneous_bc(prob);
    chk(h0.xlo == h1.xlo && h0.xhi == h1.xhi && h0.ylo == h1.ylo && h0.yhi == h1.yhi,
        "homogeneous_bc_meme_types");
    chk(h0.xlo_val == h1.xlo_val && h0.xhi_val == h1.xhi_val && h0.ylo_val == h1.ylo_val &&
            h0.yhi_val == h1.yhi_val,
        "homogeneous_bc_meme_valeurs");
  }

  // (A) EllipticProblem : constructeur BCRec vs fabrique EllipticProblem.
  auto solve_into = [&](auto& solver) {
    Array4 f = solver.rhs().fab(0).array();
    const Box2D v = solver.rhs().box(0);
    for_each_cell(v, [=] ADC_HD(int i, int j) { f(i, j) = fr(i, j); });
    solver.phi().set_val(0.0);
    solver.solve();
  };

  GeometricMG mg_ref(geom, ba, bc);
  EllipticProblem prob{Real(1), bc, true};  // nullspace_const : etiquette
  GeometricMG mg_named = make_elliptic_solver<GeometricMG>(geom, ba, prob);
  solve_into(mg_ref);
  solve_into(mg_named);

  // memes BCRec -> meme suite d'operations -> memes bits.
  bool phi_bit_eq = true;
  const ConstArray4 pr = mg_ref.phi().fab(0).const_array();
  const ConstArray4 pn = mg_named.phi().fab(0).const_array();
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i)
      if (pr(i, j) != pn(i, j))
        phi_bit_eq = false;
  chk(phi_bit_eq, "EllipticProblem_phi_bit_identique");

  // (B) FieldPostProcess : reference recopiee vs field_postprocess.
  // phi connu (1 ghost) periodique.
  MultiFab phi(ba, dm, 1, 1);
  {
    Array4 p = phi.fab(0).array();
    const Box2D v = phi.box(0);
    for_each_cell(v, [=] ADC_HD(int i, int j) { p(i, j) = fr(i, j); });
    fill_ghosts(phi, dom, bc);
  }
  const Real cx = Real(1) / (2 * geom.dx());
  const Real cy = Real(1) / (2 * geom.dy());

  // reference : recopie exacte de detail::coupler_grad_phi.
  MultiFab ref(ba, dm, 3, 1);
  {
    const ConstArray4 p = phi.fab(0).const_array();
    Array4 a = ref.fab(0).array();
    const Box2D v = ref.box(0);
    for_each_cell(v, [=] ADC_HD(int i, int j) {
      a(i, j, 0) = p(i, j);
      a(i, j, 1) = (p(i + 1, j) - p(i - 1, j)) * cx;
      a(i, j, 2) = (p(i, j + 1) - p(i, j - 1)) * cy;
    });
  }

  // nomme : GradSign::Plus, store_phi=true.
  MultiFab plus(ba, dm, 3, 1);
  field_postprocess(phi, plus, cx, cy, FieldPostProcess{FieldPostProcess::GradSign::Plus, true});

  bool plus_bit_eq = true;
  {
    const ConstArray4 ar = ref.fab(0).const_array();
    const ConstArray4 ap = plus.fab(0).const_array();
    const Box2D v = ref.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        for (int c = 0; c < 3; ++c)
          if (ar(i, j, c) != ap(i, j, c))
            plus_bit_eq = false;
  }
  chk(plus_bit_eq, "FieldPostProcess_Plus_bit_identique");

  // GradSign::Minus, store_phi=false : 2 composantes E = -grad phi (two_fluid).
  // chaque composante vaut -ref de la composante de gradient correspondante.
  MultiFab minus(ba, dm, 2, 1);
  field_postprocess(phi, minus, cx, cy, FieldPostProcess{FieldPostProcess::GradSign::Minus, false});

  bool minus_bit_eq = true;
  {
    const ConstArray4 ar = ref.fab(0).const_array();
    const ConstArray4 am = minus.fab(0).const_array();
    const Box2D v = minus.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        if (am(i, j, 0) != -ar(i, j, 1))
          minus_bit_eq = false;
        if (am(i, j, 1) != -ar(i, j, 2))
          minus_bit_eq = false;
      }
  }
  chk(minus_bit_eq, "FieldPostProcess_Minus_egale_moins_grad");

  if (fails == 0)
    std::printf("OK test_elliptic_problem\n");
  return fails == 0 ? 0 : 1;
}
