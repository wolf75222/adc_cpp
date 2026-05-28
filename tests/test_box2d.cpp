// Box2D : arithmetique entiere de l'espace d'indices (grow, refine/coarsen,
// intersect, contains), y compris la division plancher du coarsen sur les
// indices negatifs des ghosts.

#include <adc/mesh/box2d.hpp>

#include <cstdio>

using namespace adc;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  Box2D b = Box2D::from_extents(4, 3);  // [0..3] x [0..2]
  chk(b.nx() == 4 && b.ny() == 3, "extents");
  chk(b.num_cells() == 12, "num_cells");
  chk(b.contains(3, 2) && !b.contains(4, 0), "contains");

  Box2D g = b.grow(1);  // [-1..4] x [-1..3]
  chk(g.lo[0] == -1 && g.hi[0] == 4 && g.lo[1] == -1 && g.hi[1] == 3, "grow");
  chk(g.nx() == 6 && g.ny() == 5, "grow_extents");

  Box2D r = b.refine(2);  // [0..7] x [0..5]
  chk(r.lo[0] == 0 && r.hi[0] == 7 && r.hi[1] == 5, "refine");
  chk(r.coarsen(2) == b, "coarsen_roundtrip");

  // coarsen avec indices negatifs : floor(-1/2) = -1, floor(2/2) = 1
  Box2D neg{{-1, -1}, {2, 2}};
  Box2D c = neg.coarsen(2);
  chk(c.lo[0] == -1 && c.hi[0] == 1, "coarsen_floor");

  Box2D a{{0, 0}, {5, 5}};
  Box2D d{{3, 3}, {9, 9}};
  Box2D in = a.intersect(d);  // [3..5] x [3..5]
  chk(in.lo[0] == 3 && in.hi[0] == 5 && in.lo[1] == 3 && in.hi[1] == 5,
      "intersect");
  chk(a.intersect(Box2D{{10, 10}, {12, 12}}).empty(), "intersect_empty");

  chk(a.contains(in) && !a.contains(d), "contains_box");

  if (fails == 0) std::printf("OK test_box2d\n");
  return fails == 0 ? 0 : 1;
}
