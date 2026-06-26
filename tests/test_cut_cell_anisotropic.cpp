// Bord embedded cut-cell (Shortley-Weller) AVEC operateur elliptique ANISOTROPE :
// div(diag(eps_x, eps_y) grad phi) = f sur un disque, phi=0 sur le cercle. Le stencil cut-cell
// multiplie chaque poids Shortley-Weller par sa permittivite de FACE directionnelle (eps_x pour les
// faces normales a x, eps_y pour les faces normales a y) : ce test le valide par MMS.
//
// MMS : phi = R^2 - r^2 (nulle sur r = R). Pour eps_x, eps_y CONSTANTS,
//   div(diag(eps_x, eps_y) grad phi) = eps_x phi_xx + eps_y phi_yy = -2 eps_x - 2 eps_y,
// constant ; phi = R^2 - r^2 reste donc la solution EXACTE avec f = -2(eps_x + eps_y).
// (A) eps_x=1.5, eps_y=0.7 (anisotrope) : cut-cell converge a l'ORDRE ~2 en L2 (Shortley-Weller).
// (B) NON-REGRESSION : eps_x=eps_y=1 (anisotropie degeneree) == cut-cell SANS eps (operateur lap).

#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>

#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

using namespace pops;
static constexpr double kCx = 0.5, kCy = 0.5, kR = 0.4;

static GeometricMG make_mg(int nc) {
  Box2D dom = Box2D::from_extents(nc, nc);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba(std::vector<Box2D>{dom});
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;
  std::function<Real(Real, Real)> ls = [](Real x, Real y) {
    return std::hypot(x - kCx, y - kCy) - kR;
  };
  std::function<bool(Real, Real)> active = [](Real x, Real y) {
    return std::hypot(x - kCx, y - kCy) < kR;
  };
  return GeometricMG(geom, ba, bc, active, false, 2, 2, 2, 50, /*cut_cell=*/true, ls);
}

// Erreur L2 de phi vs R^2 - r^2 sur le disque, operateur anisotrope (eps_x, eps_y constants).
static double solve_err_aniso(int nc, double ex, double ey) {
  GeometricMG mg = make_mg(nc);
  mg.set_epsilon_anisotropic([ex](Real, Real) { return Real(ex); },
                             [ey](Real, Real) { return Real(ey); });
  mg.rhs().set_val(Real(-2.0 * (ex + ey)));  // f = div(diag(ex,ey) grad(R^2-r^2)) = -2(ex+ey)
  mg.phi().set_val(0.0);
  mg.solve_robust(1e-10, 300);
  const double dx = 1.0 / nc;
  const ConstArray4 p = mg.phi().fab(0).const_array();
  double s = 0;
  long cnt = 0;
  for (int j = 0; j < nc; ++j)
    for (int i = 0; i < nc; ++i) {
      const double x = (i + 0.5) * dx, y = (j + 0.5) * dx;
      const double r2 = (x - kCx) * (x - kCx) + (y - kCy) * (y - kCy);
      if (r2 < kR * kR) {
        const double e = std::fabs(p(i, j) - (kR * kR - r2));
        s += e * e;
        ++cnt;
      }
    }
  return std::sqrt(s / cnt);
}

static double order(double e1, double e2, int n1, int n2) {
  return std::log(e1 / e2) / std::log(double(n2) / n1);
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  // (A) anisotrope eps_x=1.5, eps_y=0.7 : convergence cut-cell ordre ~2 en L2.
  const double ex = 1.5, ey = 0.7;
  const double a128 = solve_err_aniso(128, ex, ey);
  const double a256 = solve_err_aniso(256, ex, ey);
  const double a512 = solve_err_aniso(512, ex, ey);
  const double o = order(a128, a512, 128, 512);
  std::printf("cut-cell anisotrope (ex=%.1f ey=%.1f) L2 : %.3e %.3e %.3e  ordre=%.2f\n", ex, ey,
              a128, a256, a512, o);
  chk(o > 1.7, "cutcell_aniso_ordre2_L2");
  chk(std::isfinite(a512) && a512 > 0, "cutcell_aniso_fini");

  // (B) non-regression : eps_x=eps_y=1 (degenere) == cut-cell sans eps (operateur lap, f=-4).
  const int nc = 256;
  GeometricMG mg_a = make_mg(nc);
  mg_a.set_epsilon_anisotropic([](Real, Real) { return Real(1); },
                               [](Real, Real) { return Real(1); });
  mg_a.rhs().set_val(Real(-4.0));
  mg_a.phi().set_val(0.0);
  mg_a.solve_robust(1e-10, 300);

  GeometricMG mg_p = make_mg(nc);  // sans eps : operateur lap
  mg_p.rhs().set_val(Real(-4.0));
  mg_p.phi().set_val(0.0);
  mg_p.solve_robust(1e-10, 300);

  const ConstArray4 pa = mg_a.phi().fab(0).const_array();
  const ConstArray4 pp = mg_p.phi().fab(0).const_array();
  double gap = 0;
  for (int j = 0; j < nc; ++j)
    for (int i = 0; i < nc; ++i)
      gap = std::max(gap, std::fabs(pa(i, j) - pp(i, j)));
  std::printf("cut-cell : eps_x=eps_y=1 vs sans eps, ecart max = %.3e\n", gap);
  chk(gap < 1e-12, "cutcell_aniso_degenere_non_regression");

  if (fails == 0)
    std::printf("OK test_cut_cell_anisotropic\n");
  return fails == 0 ? 0 : 1;
}
