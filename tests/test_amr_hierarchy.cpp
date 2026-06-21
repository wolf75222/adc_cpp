// AmrHierarchy : construction du niveau grossier, ajout d'un niveau fin imbrique,
// et interpolation grossier->fin sur la region raffinee.

#include <adc/amr/amr_hierarchy.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/refinement.hpp>

#include "test_harness.hpp"  // adc::test::Checker (compteur + assertion partages)

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;

int main() {
  adc::test::Checker chk;  // style terse : n'imprime que les echecs (FAIL <libelle>)
  auto close = [](Real x, Real y) { return std::fabs(x - y) < 1e-9; };  // tolerance absolue locale

  Box2D cdom = Box2D::from_extents(8, 8);  // [0..7]
  AmrHierarchy h(cdom, /*max_grid_size=*/8, /*ncomp=*/1, /*ngrow=*/1,
                 /*ref_ratio=*/2);

  chk(h.num_levels() == 1, "lev0_count");
  chk(h.ref_ratio() == 2, "ref_ratio");
  chk(h.domain(0) == cdom, "lev0_domain");
  chk(h.data(0).local_size() == 1, "lev0_one_box");

  // niveau fin imbrique : cellules grossieres [2..5]^2 raffinees -> [4..11]^2
  BoxArray fba(std::vector<Box2D>{Box2D{{4, 4}, {11, 11}}});
  h.add_level(fba);

  chk(h.num_levels() == 2, "lev1_count");
  chk(h.domain(1) == cdom.refine(2), "lev1_domain");  // [0..15]
  chk(h.boxes(1)[0] == (Box2D{{4, 4}, {11, 11}}), "lev1_box");
  chk(h.data(1).n_grow() == 1, "lev1_ghost");

  // remplir le grossier puis interpoler vers le fin imbrique
  Array4 ac = h.data(0).fab(0).array();
  for_each_cell(cdom, [ac](int I, int J) { ac(I, J, 0) = I + 100.0 * J; });
  interpolate(h.data(0), h.data(1), h.ref_ratio());

  // fine(i,j) = gc(i/2, j/2)
  chk(close(h.data(1).fab(0)(4, 4, 0), 202.0), "interp_44");      // gc(2,2)
  chk(close(h.data(1).fab(0)(11, 11, 0), 505.0), "interp_1111");  // gc(5,5)

  if (chk.fails() == 0)
    std::printf("OK test_amr_hierarchy\n");
  return chk.failed();
}
