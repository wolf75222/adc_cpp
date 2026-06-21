// BoxArray::from_domain : decoupage en tuiles disjointes et couvrantes,
// reparties egalement (cas divisible et non divisible).

#include <adc/mesh/box_array.hpp>

#include <cstdio>
#include <vector>

using namespace adc;

// Verifie que les boxes pavent exactement le domaine : chaque cellule couverte
// une et une seule fois.
static bool tiles_exactly(const BoxArray& ba, const Box2D& dom) {
  std::vector<int> count(static_cast<size_t>(dom.nx()) * dom.ny(), 0);
  for (int b = 0; b < ba.size(); ++b) {
    const Box2D& box = ba[b];
    for (int j = box.lo[1]; j <= box.hi[1]; ++j)
      for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
        if (!dom.contains(i, j))
          return false;
        ++count[static_cast<size_t>(j - dom.lo[1]) * dom.nx() + (i - dom.lo[0])];
      }
  }
  for (int c : count)
    if (c != 1)
      return false;
  return true;
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  // cas divisible : 8x8, max_grid_size 4 -> 2x2 = 4 boxes de 4x4
  Box2D dom = Box2D::from_extents(8, 8);
  BoxArray ba = BoxArray::from_domain(dom, 4);
  chk(ba.size() == 4, "div_count");
  for (int b = 0; b < ba.size(); ++b)
    chk(ba[b].nx() == 4 && ba[b].ny() == 4, "div_tile_size");
  chk(tiles_exactly(ba, dom), "div_tiles_exactly");
  chk(ba.num_cells() == 64, "div_num_cells");
  chk(ba.bounding_box() == dom, "div_bbox");

  // cas non divisible : 10x10, max_grid_size 4 -> 3x3 = 9 boxes, tailles 4,3,3
  Box2D dom2 = Box2D::from_extents(10, 10);
  BoxArray ba2 = BoxArray::from_domain(dom2, 4);
  chk(ba2.size() == 9, "ndiv_count");
  chk(tiles_exactly(ba2, dom2), "ndiv_tiles_exactly");
  chk(ba2.num_cells() == 100, "ndiv_num_cells");
  chk(ba2.bounding_box() == dom2, "ndiv_bbox");
  // premiere tuile en x doit faire 4 (base 3 + reste 1), les suivantes 3
  chk(ba2[0].nx() == 4 && ba2[1].nx() == 3 && ba2[2].nx() == 3, "ndiv_even_split");

  if (fails == 0)
    std::printf("OK test_box_array\n");
  return fails == 0 ? 0 : 1;
}
