// Regrid : un niveau fin est cree autour de la region taguee, les donnees fines
// sont interpolees depuis le grossier, le buffer dilate la region, un re-regrid
// preserve l'ancien fin, et un tagging vide supprime le niveau fin.

#include <adc/amr/hierarchy/amr_hierarchy.hpp>
#include <adc/amr/tagging/cluster.hpp>
#include <adc/amr/regridding/regrid.hpp>
#include <adc/mesh/index/box2d.hpp>
#include <adc/mesh/storage/fab2d.hpp>
#include <adc/mesh/execution/for_each.hpp>
#include <adc/mesh/storage/multifab.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

// feature centrale : 1 dans [6..9]^2, 0 ailleurs
static double feature(int i, int j) {
  return (i >= 6 && i <= 9 && j >= 6 && j <= 9) ? 1.0 : 0.0;
}

static auto threshold_crit() {
  return [](const ConstArray4& a, int i, int j) { return a(i, j, 0) > 0.5; };
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };
  auto close = [](Real x, Real y) { return std::fabs(x - y) < 1e-9; };

  Box2D cdom = Box2D::from_extents(16, 16);

  // --- regrid sans buffer : box fine = refine de la feature ---
  {
    AmrHierarchy h(cdom, 16, 1, 1, 2);
    Array4 a = h.data(0).fab(0).array();
    for_each_cell(cdom, [a](int i, int j) { a(i, j, 0) = feature(i, j); });

    RegridParams rp;
    rp.n_buffer = 0;
    regrid_level(h, 0, threshold_crit(), rp);

    chk(h.num_levels() == 2, "level_created");
    chk(h.domain(1) == cdom.refine(2), "fine_domain");
    chk(h.boxes(1).size() == 1, "one_fine_box");
    chk(h.boxes(1)[0] == (Box2D{{12, 12}, {19, 19}}), "fine_box_extent");
    // interpolation injective : fine(12,12)=coarse(6,6)=1, fine(19,19)=coarse(9,9)=1
    chk(close(h.data(1).fab(0)(12, 12, 0), 1.0), "interp_lo");
    chk(close(h.data(1).fab(0)(19, 19, 0), 1.0), "interp_hi");
    // conservation de l'injection : 16 cellules grossieres -> 64 fines a 1
    chk(close(sum(h.data(1)), 64.0), "interp_sum");
  }

  // --- buffer dilate la region taguee ---
  {
    AmrHierarchy h(cdom, 16, 1, 1, 2);
    Array4 a = h.data(0).fab(0).array();
    for_each_cell(cdom, [a](int i, int j) { a(i, j, 0) = feature(i, j); });

    RegridParams rp;
    rp.n_buffer = 1;
    regrid_level(h, 0, threshold_crit(), rp);
    // tags [6..9] dilates -> [5..10], refine -> [10..21]
    chk(h.boxes(1)[0] == (Box2D{{10, 10}, {21, 21}}), "buffered_box");
  }

  // --- re-regrid : l'ancien fin est preserve la ou il recouvre ---
  {
    AmrHierarchy h(cdom, 16, 1, 1, 2);
    Array4 a = h.data(0).fab(0).array();
    for_each_cell(cdom, [a](int i, int j) { a(i, j, 0) = feature(i, j); });

    RegridParams rp;
    rp.n_buffer = 0;
    regrid_level(h, 0, threshold_crit(), rp);
    h.data(1).fab(0)(12, 12, 0) = 999.0;  // marqueur dans le fin

    regrid_level(h, 0, threshold_crit(), rp);  // memes boxes
    chk(close(h.data(1).fab(0)(12, 12, 0), 999.0), "old_fine_preserved");
    chk(close(h.data(1).fab(0)(19, 19, 0), 1.0), "rest_interpolated");
  }

  // --- tagging vide : le niveau fin disparait ---
  {
    AmrHierarchy h(cdom, 16, 1, 1, 2);
    Array4 a = h.data(0).fab(0).array();
    for_each_cell(cdom, [a](int i, int j) { a(i, j, 0) = feature(i, j); });
    regrid_level(h, 0, threshold_crit(), RegridParams{});
    chk(h.num_levels() == 2, "before_clear");

    h.data(0).set_val(0.0);  // plus aucune cellule au-dessus du seuil
    regrid_level(h, 0, threshold_crit(), RegridParams{});
    chk(h.num_levels() == 1, "fine_removed");
  }

  if (fails == 0)
    std::printf("OK test_regrid\n");
  return fails == 0 ? 0 : 1;
}
