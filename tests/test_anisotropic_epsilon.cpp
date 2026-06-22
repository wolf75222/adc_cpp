// Validation de l'operateur elliptique ANISOTROPE div(diag(eps_x, eps_y) grad phi) = f
// (multigrille geometrique, GeometricMG::set_epsilon_anisotropic). Les faces NORMALES A X
// utilisent eps_x, les faces NORMALES A Y utilisent eps_y. Solution manufacturee LISSE non
// triviale, second membre f calcule ANALYTIQUEMENT, CL Dirichlet exactes au bord.
//
// MMS : phi(x,y) = sin(pi x) sin(pi y),  eps_x(x,y) = 1 + 0.5 x,  eps_y(x,y) = 1 + 0.3 y.
//   div(diag(eps_x, eps_y) grad phi) = d/dx(eps_x d phi/dx) + d/dy(eps_y d phi/dy).
//   d phi/dx = pi cos(pi x) sin(pi y),   d phi/dy = pi sin(pi x) cos(pi y).
//   d/dx(eps_x d phi/dx) = (d eps_x/dx) d phi/dx + eps_x d^2 phi/dx^2
//        = 0.5 pi cos(pi x) sin(pi y) - (1 + 0.5 x) pi^2 sin(pi x) sin(pi y).
//   d/dy(eps_y d phi/dy) = (d eps_y/dy) d phi/dy + eps_y d^2 phi/dy^2
//        = 0.3 pi sin(pi x) cos(pi y) - (1 + 0.3 y) pi^2 sin(pi x) sin(pi y).
//   f = 0.5 pi cos(pi x) sin(pi y) + 0.3 pi sin(pi x) cos(pi y)
//       - pi^2 sin(pi x) sin(pi y) [ (1 + 0.5 x) + (1 + 0.3 y) ].
// phi s'annule sur le bord du carre unite -> Dirichlet homogene EXACT.
//
// Gate STRICT : on raffine n = 32, 64, 128 et on exige une convergence d'ORDRE 2 en norme
// L-inf (erreur divisee par ~4 a chaque raffinement, ratio dans [3.5, 4.5]). On verifie
// aussi la NON-REGRESSION ISOTROPE DEGENERE : avec eps_y = eps_x (anisotropie degeneree) le
// residu (donc l'operateur) doit etre IDENTIQUE A LA PRECISION MACHINE a l'operateur
// isotrope set_epsilon(eps_x). Composabilite avec kappa egalement verifiee.

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
static double eps_x_field(double x, double /*y*/) {
  return 1.0 + 0.5 * x;
}
static double eps_y_field(double /*x*/, double y) {
  return 1.0 + 0.3 * y;
}

// f = div(diag(eps_x, eps_y) grad phi) (analytique).
static double rhs_exact(double x, double y) {
  const double s = std::sin(kPi * x) * std::sin(kPi * y);
  return 0.5 * kPi * std::cos(kPi * x) * std::sin(kPi * y) +
         0.3 * kPi * std::sin(kPi * x) * std::cos(kPi * y) -
         kPi * kPi * s * ((1.0 + 0.5 * x) + (1.0 + 0.3 * y));
}

// Resout div(diag(eps_x, eps_y) grad phi) = f sur n x n (Dirichlet exact), erreur L-inf.
static double solve_mms(int n) {
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);

  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;  // phi=0 au bord (exact)

  GeometricMG mg(geom, ba, bc);
  mg.set_epsilon_anisotropic([](Real x, Real y) { return Real(eps_x_field(x, y)); },
                             [](Real x, Real y) { return Real(eps_y_field(x, y)); });

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

// Non-regression : eps_y == eps_x (anisotropie DEGENEREE) doit donner exactement l'operateur
// isotrope set_epsilon(eps_x). On compare le residu sur un phi non trivial entre le solveur
// anisotrope degenere et le solveur isotrope : coincidence a la precision machine.
static double degenerate_aniso_residual_gap(int n) {
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

  // operateur isotrope eps = eps_x (faces x et y partagent eps_x)
  GeometricMG mg_iso(geom, ba, bc);
  mg_iso.set_epsilon([](Real x, Real y) { return Real(eps_x_field(x, y)); });
  fill_phi_rhs(mg_iso);
  const Real r_iso = mg_iso.current_residual();

  // operateur anisotrope DEGENERE eps_x = eps_y = eps_x : doit etre bit-identique a l'isotrope
  GeometricMG mg_aniso(geom, ba, bc);
  mg_aniso.set_epsilon_anisotropic([](Real x, Real y) { return Real(eps_x_field(x, y)); },
                                   [](Real x, Real y) { return Real(eps_x_field(x, y)); });
  fill_phi_rhs(mg_aniso);
  const Real r_aniso = mg_aniso.current_residual();

  return std::fabs(r_iso - r_aniso);
}

// Composabilite anisotrope + reaction kappa : div(diag(eps_x, eps_y) grad phi) - kappa phi = f,
// kappa constant. f gagne le terme - kappa phi. Convergence ordre 2 attendue.
static constexpr double KAPPA = 50.0;
static double solve_mms_kappa(int n) {
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;

  GeometricMG mg(geom, ba, bc);
  mg.set_epsilon_anisotropic([](Real x, Real y) { return Real(eps_x_field(x, y)); },
                             [](Real x, Real y) { return Real(eps_y_field(x, y)); });
  mg.set_reaction([](Real, Real) { return Real(KAPPA); });

  Array4 af = mg.rhs().fab(0).array();
  for_each_cell(dom, [af, geom](int i, int j) {
    const double x = geom.x_cell(i), y = geom.y_cell(j);
    af(i, j, 0) = rhs_exact(x, y) - KAPPA * phi_exact(x, y);  // - kappa phi
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
      eInf = std::max(eInf, std::fabs(p(i, j, 0) - phi_exact(geom.x_cell(i), geom.y_cell(j))));
  return eInf;
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  // (A) MMS anisotrope : convergence ordre 2.
  const double e32 = solve_mms(32);
  const double e64 = solve_mms(64);
  const double e128 = solve_mms(128);
  const double r1 = e32 / e64, r2 = e64 / e128;
  std::printf("aniso MMS : Linf e32=%.3e e64=%.3e e128=%.3e | ratios %.2f %.2f\n", e32, e64, e128,
              r1, r2);
  chk(r1 > 3.5 && r1 < 4.5, "ordre2_ratio_32_64");
  chk(r2 > 3.5 && r2 < 4.5, "ordre2_ratio_64_128");

  // (B) non-regression : eps_y = eps_x (degenere) == isotrope eps_x au bit.
  const double gap = degenerate_aniso_residual_gap(64);
  std::printf("eps_y=eps_x : ecart residu vs isotrope eps_x = %.3e\n", gap);
  chk(gap < 1e-12, "aniso_degenere_non_regression");

  // (C) composabilite aniso + kappa : ordre 2.
  const double c64 = solve_mms_kappa(64), c128 = solve_mms_kappa(128);
  const double rc = c64 / c128;
  std::printf("aniso + kappa MMS : Linf c64=%.3e c128=%.3e | ratio %.2f\n", c64, c128, rc);
  chk(rc > 3.5 && rc < 4.5, "ordre2_aniso_plus_kappa");

  if (fails == 0)
    std::printf("OK test_anisotropic_epsilon\n");
  return fails == 0 ? 0 : 1;
}
