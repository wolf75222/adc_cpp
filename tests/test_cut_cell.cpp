// Bord embedded cut-cell (Shortley-Weller) vs escalier, sur une solution manufacturee.
//
// Probleme : lap(phi) = -4 dans le disque r < R, phi = 0 sur le cercle r = R.
// Solution exacte : phi(x, y) = R^2 - r^2  (lap(R^2 - r^2) = -4, nulle sur r = R).
//
// L'escalier impose phi = 0 aux CENTRES de cellules juste sous le cercle, ou la solution
// exacte vaut R^2 - r_c^2 > 0 : erreur de bord O(dx) -> ordre global ~ 1. Le cut-cell impose
// phi = 0 sur le CERCLE (distance theta*dx), donc reproduit la solution a l'ordre 2 (l'erreur
// residuelle vient de la linearisation du level-set courbe). On verifie : ordre L2 cut-cell ~ 2,
// ordre escalier ~ 1, et erreur cut-cell de plusieurs ordres de grandeur sous l'escalier.
//
// Norme L2 (et non L_inf) : la convergence cut-cell cell-centree est polluee par la pire
// cellule coupee (probleme du small-cell), ce qui rend L_inf erratique ; l'ordre 2 propre
// est en norme L2 (supraconvergence de Shortley-Weller). On l'imprime aussi a titre indicatif.

#include <adc/numerics/elliptic/mg/geometric_mg.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>

#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

using namespace adc;
static constexpr double kCx = 0.5, kCy = 0.5, kR = 0.4;

static GeometricMG make_mg(int nc, bool cut) {
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
  // (geom, ba, bc, active, replicated, min_coarse, nu1, nu2, nbottom, cut_cell, levelset)
  return GeometricMG(geom, ba, bc, active, false, 2, 2, 2, 50, cut,
                     cut ? ls : std::function<Real(Real, Real)>{});
}

// erreur L2 (et L_inf) de phi_MG vs R^2 - r^2 sur les cellules interieures au disque.
static void solve_err(int nc, bool cut, double& l2, double& linf) {
  GeometricMG mg = make_mg(nc, cut);
  const double dx = 1.0 / nc;
  mg.rhs().set_val(-4.0);  // f = lap(phi) = -4 ; cellules conductrices ignorees (masquees)
  mg.phi().set_val(0.0);
  mg.solve_robust(1e-10, 300);
  const ConstArray4 p = mg.phi().fab(0).const_array();
  double s = 0, mx = 0;
  long cnt = 0;
  for (int j = 0; j < nc; ++j)
    for (int i = 0; i < nc; ++i) {
      const double x = (i + 0.5) * dx, y = (j + 0.5) * dx;
      const double r2 = (x - kCx) * (x - kCx) + (y - kCy) * (y - kCy);
      if (r2 < kR * kR) {  // interieur du disque
        const double e = std::fabs(p(i, j) - (kR * kR - r2));
        s += e * e;
        mx = std::max(mx, e);
        ++cnt;
      }
    }
  l2 = std::sqrt(s / cnt);
  linf = mx;
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

  double s128, s256, s512, si128, si256, si512, c128, c256, c512, ci128, ci256, ci512;
  solve_err(128, false, s128, si128);
  solve_err(256, false, s256, si256);
  solve_err(512, false, s512, si512);
  solve_err(128, true, c128, ci128);
  solve_err(256, true, c256, ci256);
  solve_err(512, true, c512, ci512);
  const double o_s = order(s128, s512, 128, 512);     // escalier, L2
  const double o_c = order(c128, c512, 128, 512);     // cut-cell, L2 (ordre propre)
  const double o_ci = order(ci128, ci512, 128, 512);  // cut-cell, L_inf (indicatif)

  std::printf("escalier L2 : %.3e %.3e %.3e  ordre=%.2f\n", s128, s256, s512, o_s);
  std::printf("cut-cell L2 : %.3e %.3e %.3e  ordre=%.2f\n", c128, c256, c512, o_c);
  std::printf("cut-cell Linf: %.3e %.3e %.3e  ordre=%.2f (indicatif, small-cell)\n", ci128, ci256,
              ci512, o_ci);
  std::printf("gain L2 a nc=512 : escalier / cutcell = %.0fx\n", s512 / c512);

  chk(o_c > 1.7, "cut_cell_ordre_2_L2");                 // Shortley-Weller : ~2 en L2
  chk(o_s < 1.4, "escalier_ordre_1");                    // escalier : ~1
  chk(c512 < s512 / 50.0, "cut_cell_bien_plus_precis");  // gain de plusieurs ordres
  chk(std::isfinite(c512) && c512 > 0, "cut_cell_fini");

  if (fails == 0)
    std::printf("OK test_cut_cell\n");
  return fails == 0 ? 0 : 1;
}
