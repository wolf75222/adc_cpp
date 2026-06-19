// Operateur elliptique ECRANTE / Helmholtz : div(eps grad phi) - kappa phi = f
// (GeometricMG::set_reaction). C'est l'extension de l'OPERATEUR composable au-dela de eps(x) :
// un terme de reaction kappa(x) (Poisson de Debye : kappa = 1/lambda_D^2). kappa >= 0 ->
// operateur plus diagonalement dominant (la multigrille converge au moins aussi bien). Solution
// manufacturee LISSE, second membre f calcule ANALYTIQUEMENT, Dirichlet exact, gate ordre 2.
//
//   phi(x,y) = sin(pi x) sin(pi y),  lap(phi) = -2 pi^2 phi.
// (A) kappa constant, eps=1 : lap(phi) - kappa phi = f -> f = -(2 pi^2 + kappa) phi. Ordre 2.
// (B) NON-REGRESSION : kappa=0 -> operateur strictement identique au Poisson (ecart residu nul).
// (C) COMPOSABILITE eps(x) + kappa : div(eps grad phi) - kappa phi, eps=1+0.5x, ordre 2.
// (D)/(E) kappa(x,y) VARIABLE, via set_reaction(fn) puis set_reaction(MultiFab) (ADC-251) : kappa
//   est DIAGONAL, lu en (i,j) seul -> alloue 0 ghost et ses ghosts ne sont JAMAIS remplis (a la
//   difference de eps). Un kappa CONSTANT (A/C) ne peut pas le verifier ; un kappa qui varie en x ET
//   y exerce reellement la lecture par cellule. La convergence ordre 2 prouve que 0 ghost SUFFIT
//   (lecture en (i,j)), et verrouille l'invariant : un futur stencil qui lirait kappa sur ses ghosts
//   (non remplis) casserait ces deux cas.

#include <adc/numerics/elliptic/geometric_mg.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;
static constexpr double KAPPA = 50.0;  // 1/lambda_D^2 (ecrantage modere : lambda_D ~ 0.14)

static double phi_exact(double x, double y) {
  return std::sin(kPi * x) * std::sin(kPi * y);
}
static double eps_field(double x, double /*y*/) { return 1.0 + 0.5 * x; }

// f = div(eps grad phi) - kappa phi (analytique). eps_on -> eps=1+0.5x sinon eps=1.
static double rhs_exact(double x, double y, bool eps_on) {
  const double s = std::sin(kPi * x) * std::sin(kPi * y);
  double div_eps_grad;
  if (eps_on)
    div_eps_grad = -(1.0 + 0.5 * x) * 2.0 * kPi * kPi * s +
                   0.5 * kPi * std::cos(kPi * x) * std::sin(kPi * y);
  else
    div_eps_grad = -2.0 * kPi * kPi * s;
  return div_eps_grad - KAPPA * s;  // - kappa phi
}

// Resout div(eps grad phi) - kappa phi = f sur n x n (Dirichlet exact), renvoie l'erreur L-inf.
static double solve_mms(int n, bool eps_on) {
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;  // phi=0 au bord (exact)

  GeometricMG mg(geom, ba, bc);
  if (eps_on) mg.set_epsilon([](Real x, Real y) { return Real(eps_field(x, y)); });
  mg.set_reaction([](Real, Real) { return Real(KAPPA); });  // kappa constant

  Array4 af = mg.rhs().fab(0).array();
  for_each_cell(dom, [af, geom, eps_on](int i, int j) {
    af(i, j, 0) = rhs_exact(geom.x_cell(i), geom.y_cell(j), eps_on);
  });
  mg.phi().set_val(0.0);

  const Real r0 = mg.current_residual();
  Real rn = r0;
  for (int c = 0; c < 80 && rn > 1e-11 * r0; ++c) {
    mg.vcycle();
    rn = mg.current_residual();
  }

  Fab2D& p = mg.phi().fab(0);
  double eInf = 0;
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
      eInf = std::max(eInf,
                      std::fabs(p(i, j, 0) - phi_exact(geom.x_cell(i), geom.y_cell(j))));
  return eInf;
}

// Non-regression : kappa=0 -> l'operateur doit etre IDENTIQUE au Poisson (sans set_reaction).
static double zero_kappa_residual_gap(int n) {
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;

  auto fill = [&](GeometricMG& mg) {
    Array4 ap = mg.phi().fab(0).array();
    Array4 af = mg.rhs().fab(0).array();
    for_each_cell(dom, [ap, af, geom](int i, int j) {
      const double x = geom.x_cell(i), y = geom.y_cell(j);
      ap(i, j, 0) = std::sin(kPi * x) * std::sin(2 * kPi * y);  // phi non trivial
      af(i, j, 0) = std::cos(kPi * x) * std::sin(kPi * y);
    });
  };

  GeometricMG mg_pois(geom, ba, bc);
  fill(mg_pois);
  const Real r_pois = mg_pois.current_residual();

  GeometricMG mg_k0(geom, ba, bc);
  mg_k0.set_reaction([](Real, Real) { return Real(0); });  // kappa=0
  fill(mg_k0);
  const Real r_k0 = mg_k0.current_residual();

  return std::fabs(r_pois - r_k0);
}

// kappa(x,y) LISSE, strictement positif (>= 30 -> diagonalement dominant), variable en x ET y. Un
// kappa constant ne distingue pas une lecture (i,j) correcte d'une lecture decalee / sur ghost.
static double kappa_var(double x, double y) { return 30.0 * (1.0 + 0.5 * x + 0.3 * y); }
// f = lap(phi) - kappa(x,y) phi, eps = 1, phi = sin(pi x) sin(pi y).
static double rhs_varkappa(double x, double y) {
  const double s = std::sin(kPi * x) * std::sin(kPi * y);
  return -2.0 * kPi * kPi * s - kappa_var(x, y) * s;
}

// kappa(x,y) VARIABLE : exerce la lecture diagonale par cellule du coefficient de reaction (ADC-251).
// use_field selectionne la surcharge : false -> set_reaction(fn) (sample_per_level), true ->
// set_reaction(MultiFab) (restrict_and_fill). Les deux cablent kappa avec 0 ghost / do_fill=false ;
// la lecture etant en (i,j), 0 ghost SUFFIT -- l'ordre 2 le prouve. Renvoie l'erreur L-inf.
static double solve_mms_varkappa(int n, bool use_field, double* rel_resid = nullptr) {
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;

  GeometricMG mg(geom, ba, bc);
  if (use_field) {
    // Champ kappa fin discretise (1 composante, 0 ghost), copie/restreint par set_reaction(MultiFab).
    MultiFab kf(ba, mg.dmap(), 1, 0);
    Array4 ak = kf.fab(0).array();
    for_each_cell(dom, [ak, geom](int i, int j) {
      ak(i, j, 0) = Real(kappa_var(geom.x_cell(i), geom.y_cell(j)));
    });
    mg.set_reaction(kf);
  } else {
    mg.set_reaction([](Real x, Real y) { return Real(kappa_var(x, y)); });
  }

  Array4 af = mg.rhs().fab(0).array();
  for_each_cell(dom, [af, geom](int i, int j) {
    af(i, j, 0) = rhs_varkappa(geom.x_cell(i), geom.y_cell(j));
  });
  mg.phi().set_val(0.0);

  const Real r0 = mg.current_residual();
  Real rn = r0;
  for (int c = 0; c < 80 && rn > 1e-11 * r0; ++c) {
    mg.vcycle();
    rn = mg.current_residual();
  }
  // Residu relatif final : permet a l'appelant de verifier que le solveur a CONVERGE, donc que
  // eInf mesure l'erreur de DISCRETISATION (et non un residu solveur residuel qui fausserait le ratio).
  if (rel_resid) *rel_resid = (r0 > Real(0)) ? double(rn / r0) : double(rn);

  Fab2D& p = mg.phi().fab(0);
  double eInf = 0;
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
      eInf = std::max(eInf, std::fabs(p(i, j, 0) - phi_exact(geom.x_cell(i), geom.y_cell(j))));
  return eInf;
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  // (A) kappa constant, eps=1 : convergence ordre 2.
  const double e32 = solve_mms(32, false), e64 = solve_mms(64, false), e128 = solve_mms(128, false);
  const double r1 = e32 / e64, r2 = e64 / e128;
  std::printf("ecrante (kappa=%.0f) MMS : Linf e32=%.3e e64=%.3e e128=%.3e | ratios %.2f %.2f\n",
              KAPPA, e32, e64, e128, r1, r2);
  chk(r1 > 3.5 && r1 < 4.5, "ordre2_ratio_32_64");
  chk(r2 > 3.5 && r2 < 4.5, "ordre2_ratio_64_128");

  // (B) non-regression kappa=0.
  const double gap = zero_kappa_residual_gap(64);
  std::printf("kappa=0 : ecart residu vs Poisson = %.3e\n", gap);
  chk(gap < 1e-12, "kappa0_non_regression");

  // (C) composabilite eps(x) + kappa : ordre 2.
  const double c64 = solve_mms(64, true), c128 = solve_mms(128, true);
  const double rc = c64 / c128;
  std::printf("eps(x) + kappa MMS : Linf c64=%.3e c128=%.3e | ratio %.2f\n", c64, c128, rc);
  chk(rc > 3.5 && rc < 4.5, "ordre2_eps_plus_kappa");

  // (D) kappa(x,y) VARIABLE via set_reaction(fn) : lecture diagonale par cellule (ADC-251), ordre 2.
  double rr_d = 0;
  const double d64 = solve_mms_varkappa(64, false), d128 = solve_mms_varkappa(128, false, &rr_d);
  const double rd = d64 / d128;
  std::printf("kappa(x,y) fn MMS : Linf d64=%.3e d128=%.3e | ratio %.2f (resid %.1e)\n", d64, d128,
              rd, rr_d);
  chk(rr_d < 1e-9, "conv_varkappa_fn_128");  // le solveur a converge -> eInf = erreur de discretisation
  chk(rd > 3.5 && rd < 4.5, "ordre2_varkappa_fn");

  // (E) kappa(x,y) VARIABLE via set_reaction(MultiFab) : meme invariant, chemin restrict_and_fill.
  double rr_f = 0;
  const double f64 = solve_mms_varkappa(64, true), f128 = solve_mms_varkappa(128, true, &rr_f);
  const double rf = f64 / f128;
  std::printf("kappa(x,y) field MMS : Linf f64=%.3e f128=%.3e | ratio %.2f (resid %.1e)\n", f64, f128,
              rf, rr_f);
  chk(rr_f < 1e-9, "conv_varkappa_field_128");
  chk(rf > 3.5 && rf < 4.5, "ordre2_varkappa_field");

  if (fails == 0) std::printf("OK test_screened_poisson\n");
  return fails == 0 ? 0 : 1;
}
