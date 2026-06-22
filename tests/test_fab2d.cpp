// Fab2D + Array4 + for_each_cell : remplissage de l'interieur via le dispatch
// (handle Array4 capture par valeur, comme un kernel Kokkos), ghosts intacts,
// coherence du handle avec operator(), et layout composante-lente.

#include <adc/mesh/index/box2d.hpp>
#include <adc/mesh/storage/fab2d.hpp>
#include <adc/mesh/execution/for_each.hpp>

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

  Box2D valid = Box2D::from_extents(4, 3);  // [0..3] x [0..2]
  Fab2D fab(valid, /*ncomp=*/2, /*ng=*/1);
  chk(fab.grown_box().nx() == 6 && fab.grown_box().ny() == 5, "grown");
  chk(fab.size() == 6 * 5 * 2, "alloc_size");
  chk(fab(-1, -1, 0) == 0.0 && fab(4, 3, 1) == 0.0, "zero_init_ghost");

  // remplir l'interieur via le dispatch, handle capture par valeur
  Array4 a = fab.array();
  for_each_cell(valid, [a](int i, int j) {
    a(i, j, 0) = i + 10.0 * j;
    a(i, j, 1) = -(i + 10.0 * j);
  });

  chk(fab(0, 0, 0) == 0.0 && fab(3, 2, 0) == 23.0, "fill_c0");
  chk(fab(3, 2, 1) == -23.0, "fill_c1");
  chk(fab(-1, 0, 0) == 0.0 && fab(4, 2, 0) == 0.0 && fab(0, -1, 0) == 0.0, "ghost_untouched");

  ConstArray4 ca = fab.const_array();
  chk(ca(2, 1, 0) == fab(2, 1, 0) && ca(2, 1, 1) == fab(2, 1, 1), "array4_matches");

  // composante-lente : le plan c=1 est un bloc contigu apres c=0,
  // de stride nx_tot * ny_tot = 6 * 5 = 30
  chk(&fab(0, 0, 1) - &fab(0, 0, 0) == 30, "comp_slowest");

  if (fails == 0)
    std::printf("OK test_fab2d\n");
  return fails == 0 ? 0 : 1;
}
