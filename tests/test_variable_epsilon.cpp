// Validation de l'operateur elliptique a PERMITTIVITE VARIABLE div(eps grad phi) = f
// (multigrille geometrique, GeometricMG::set_epsilon). Solution manufacturee LISSE non
// triviale, second membre f calcule ANALYTIQUEMENT, CL Dirichlet exactes au bord.
//
// MMS : phi(x,y) = sin(pi x) sin(pi y),  eps(x,y) = 1 + 0.5 x  (lisse, gradient non nul).
//   div(eps grad phi) = eps lap(phi) + grad(eps) . grad(phi)
//   lap(phi)          = -2 pi^2 sin(pi x) sin(pi y)
//   grad(eps).grad(phi) = (d eps/dx)(d phi/dx) = 0.5 * pi cos(pi x) sin(pi y)
//   f(x,y) = -(1 + 0.5 x) 2 pi^2 sin(pi x) sin(pi y) + 0.5 pi cos(pi x) sin(pi y).
// phi s'annule sur le bord du carre unite -> Dirichlet homogene EXACT.
//
// Gate STRICT : on raffine n = 32, 64, 128 et on exige une convergence d'ORDRE 2 en norme
// L-inf (erreur divisee par ~4 a chaque raffinement, ratio dans [3.5, 4.5]). On verifie
// aussi la NON-REGRESSION : avec eps UNIFORME=1 le residu (donc l'operateur) est identique
// au Laplacien constant existant.

#include <adc/numerics/elliptic/mg/geometric_mg.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/storage/fab2d.hpp>
#include <adc/mesh/execution/for_each.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/multifab.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

static double phi_exact(double x, double y) {
  return std::sin(kPi * x) * std::sin(kPi * y);
}
static double eps_field(double x, double /*y*/) {
  return 1.0 + 0.5 * x;
}
static double rhs_exact(double x, double y) {
  const double s = std::sin(kPi * x) * std::sin(kPi * y);
  return -(1.0 + 0.5 * x) * 2.0 * kPi * kPi * s + 0.5 * kPi * std::cos(kPi * x) * std::sin(kPi * y);
}

// Resout div(eps grad phi) = f sur n x n (Dirichlet exact), renvoie l'erreur L-inf.
static double solve_mms(int n) {
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);

  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;  // phi=0 au bord (exact)

  GeometricMG mg(geom, ba, bc);
  mg.set_epsilon([](Real x, Real y) { return Real(eps_field(x, y)); });

  Array4 af = mg.rhs().fab(0).array();
  for_each_cell(
      dom, [af, geom](int i, int j) { af(i, j, 0) = rhs_exact(geom.x_cell(i), geom.y_cell(j)); });
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
      eInf = std::max(eInf, std::fabs(p(i, j, 0) - phi_exact(geom.x_cell(i), geom.y_cell(j))));
  return eInf;
}

// Non-regression : avec eps uniforme=1, le residu initial (phi=0, donc -lap(0)+f = f sur
// cellules actives) ne change pas, mais surtout l'OPERATEUR doit etre identique au
// Laplacien constant. On compare le residu sur un phi non trivial entre le solveur eps=1 et
// le solveur sans eps : ils doivent coincider a la precision machine.
static double uniform_eps_residual_gap(int n) {
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;

  auto fill_phi_rhs = [&](GeometricMG& mg) {
    Array4 ap = mg.phi().fab(0).array();
    Array4 af = mg.rhs().fab(0).array();
    for_each_cell(dom, [ap, af, geom](int i, int j) {
      const double x = geom.x_cell(i), y = geom.y_cell(j);
      ap(i, j, 0) = std::sin(kPi * x) * std::sin(2 * kPi * y);  // phi arbitraire non trivial
      af(i, j, 0) = std::cos(kPi * x) * std::sin(kPi * y);
    });
  };

  GeometricMG mg_const(geom, ba, bc);
  fill_phi_rhs(mg_const);
  const Real r_const = mg_const.current_residual();

  GeometricMG mg_eps(geom, ba, bc);
  mg_eps.set_epsilon([](Real, Real) { return Real(1); });  // eps uniforme=1
  fill_phi_rhs(mg_eps);
  const Real r_eps = mg_eps.current_residual();

  return std::fabs(r_const - r_eps);
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const double e32 = solve_mms(32);
  const double e64 = solve_mms(64);
  const double e128 = solve_mms(128);
  const double r1 = e32 / e64, r2 = e64 / e128;
  std::printf("eps variable MMS : Linf  e32=%.3e e64=%.3e e128=%.3e | ratios %.2f %.2f\n", e32, e64,
              e128, r1, r2);
  chk(r1 > 3.5 && r1 < 4.5, "ordre2_ratio_32_64");
  chk(r2 > 3.5 && r2 < 4.5, "ordre2_ratio_64_128");

  const double gap = uniform_eps_residual_gap(64);
  std::printf("eps uniforme=1 : ecart residu vs operateur constant = %.3e\n", gap);
  chk(gap < 1e-12, "eps_uniforme_non_regression");

  if (fails == 0)
    std::printf("OK test_variable_epsilon\n");
  return fails == 0 ? 0 : 1;
}
