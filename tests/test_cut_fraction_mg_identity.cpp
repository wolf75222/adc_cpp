// Bit-identite du REFACTOR cut_fraction au niveau GeometricMG (chantier T5-PR1).
//
// Le refactor a remplace la lambda 'cut' INLINE de GeometricMG par la primitive partagee
// detail::cut_fraction + detail::shortley_weller (cut_fraction.hpp). Ce test prouve qu'il n'y a
// AUCUN changement de comportement sur un cas de mur-disque :
//
//  (A) le champ de coefficients cut-cell ASSEMBLE par GeometricMG (5 composantes, niveau fin via
//      op_coef()) est EXACTEMENT egal (diff 0.0, operator!=) a une reference recalculee a la main
//      avec l'ANCIENNE formule inline. C'est la garantie "coef byte-identique" demandee.
//  (B) la resolution elliptique converge et le residu final est fini et < tolerance (le solveur
//      reste fonctionnel apres le refactor ; on capture aussi le residu pour comparaison eventuelle).
//
// Probleme : lap(phi) = -4 dans le disque r < R, phi = 0 sur r = R. Solution exacte R^2 - r^2.

#include <adc/numerics/elliptic/geometric_mg.hpp>
#include <adc/numerics/elliptic/cut_fraction.hpp>
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

// Oracle : ancienne formule inline (lambda 'cut' + 2/(axm*(axm+axp)) ...) reproduite a l'identique.
static void ref_coef(Real lc_x, Real lc_y, Real dx, Real dy,
                     const std::function<Real(Real, Real)>& ls, Real out[5]) {
  auto cut = [](Real lc, Real ln, Real h) -> Real {
    if (ln < Real(0)) return h;
    Real th = lc / (lc - ln);
    if (th < Real(1e-3)) th = Real(1e-3);
    if (th > Real(1)) th = Real(1);
    return th * h;
  };
  const Real lc = ls(lc_x, lc_y);
  const Real axm = cut(lc, ls(lc_x - dx, lc_y), dx);
  const Real axp = cut(lc, ls(lc_x + dx, lc_y), dx);
  const Real aym = cut(lc, ls(lc_x, lc_y - dy), dy);
  const Real ayp = cut(lc, ls(lc_x, lc_y + dy), dy);
  out[0] = Real(2) / (axm * (axm + axp));
  out[1] = Real(2) / (axp * (axm + axp));
  out[2] = Real(2) / (aym * (aym + ayp));
  out[3] = Real(2) / (ayp * (aym + ayp));
  out[4] = Real(2) / (axm * axp) + Real(2) / (aym * ayp);
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) { if (!c) { std::printf("FAIL %s\n", w); ++fails; } };

  const int nc = 64;
  Box2D dom = Box2D::from_extents(nc, nc);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba(std::vector<Box2D>{dom});
  BCRec bc; bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;
  std::function<Real(Real, Real)> ls = [](Real x, Real y) { return std::hypot(x - kCx, y - kCy) - kR; };
  std::function<bool(Real, Real)> active = [](Real x, Real y) { return std::hypot(x - kCx, y - kCy) < kR; };
  // (geom, ba, bc, active, replicated, min_coarse, nu1, nu2, nbottom, cut_cell=true, levelset)
  GeometricMG mg(geom, ba, bc, active, false, 2, 2, 2, 50, true, ls);

  // (A) coef assemble par GeometricMG == reference inline, EXACTEMENT (niveau fin, 5 composantes).
  const MultiFab* coef = mg.op_coef();
  chk(coef != nullptr, "op_coef_disponible");
  const MultiFab* mask = mg.op_mask();
  chk(mask != nullptr, "op_mask_disponible");
  if (coef && mask) {
    const ConstArray4 c = coef->fab(0).const_array();
    const ConstArray4 m = mask->fab(0).const_array();
    const Real dx = geom.dx(), dy = geom.dy();
    Real max_diff = Real(0);
    long active_cnt = 0, mismatches = 0;
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) {
        if (m(i, j) == Real(0)) {  // conducteur : GeometricMG met coef = 0
          for (int k = 0; k < 5; ++k)
            if (c(i, j, k) != Real(0)) { ++mismatches; max_diff = Real(1); }
          continue;
        }
        ++active_cnt;
        Real ref[5];
        ref_coef(geom.x_cell(i), geom.y_cell(j), dx, dy, ls, ref);
        for (int k = 0; k < 5; ++k) {
          if (c(i, j, k) != ref[k]) {  // EXACT : operator!=, aucune tolerance
            ++mismatches;
            max_diff = std::max(max_diff, std::fabs(c(i, j, k) - ref[k]));
          }
        }
      }
    std::printf("(A) coef : %ld cellules actives, %ld ecarts, max_diff=%.3e\n",
                active_cnt, mismatches, static_cast<double>(max_diff));
    chk(active_cnt > 1800, "balayage_disque_couvert");
    chk(mismatches == 0, "coef_byte_identique_a_la_reference_inline");
    chk(max_diff == Real(0), "max_diff_exactement_0");
  }

  // (B) le solveur reste fonctionnel : residu fini, convergence sous tolerance.
  mg.rhs().set_val(-4.0);
  mg.phi().set_val(0.0);
  const int cycles = mg.solve_robust(1e-10, 300);
  const Real res = mg.residual();
  std::printf("(B) solve_robust : %d cycles, residu final = %.3e\n", cycles, static_cast<double>(res));
  chk(std::isfinite(res), "residu_fini");
  chk(res < Real(1e-6), "residu_sous_tolerance");

  // verification physique : phi reproduit R^2 - r^2 a l'ordre 2 (le refactor n'a pas casse la solution).
  const ConstArray4 p = mg.phi().fab(0).const_array();
  const Real dx = geom.dx();
  double l2 = 0; long cnt = 0;
  for (int j = 0; j < nc; ++j)
    for (int i = 0; i < nc; ++i) {
      const double x = (i + 0.5) * dx, y = (j + 0.5) * dx;
      const double r2 = (x - kCx) * (x - kCx) + (y - kCy) * (y - kCy);
      if (r2 < kR * kR) { const double e = p(i, j) - (kR * kR - r2); l2 += e * e; ++cnt; }
    }
  l2 = std::sqrt(l2 / cnt);
  std::printf("    erreur L2 vs R^2 - r^2 = %.3e\n", l2);
  chk(l2 < 1e-3, "solution_physique_correcte");

  if (fails == 0) std::printf("OK test_cut_fraction_mg_identity\n");
  return fails == 0 ? 0 : 1;
}
