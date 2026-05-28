// Geometry : pas d'espace dx/dy, centres de cellule, raffinement.

#include <adc/mesh/box2d.hpp>
#include <adc/mesh/geometry.hpp>

#include <cmath>
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
  auto close = [](Real x, Real y) { return std::fabs(x - y) < 1e-12; };

  Geometry g{Box2D::from_extents(4, 2), 0.0, 1.0, 0.0, 1.0};
  chk(close(g.dx(), 0.25), "dx");
  chk(close(g.dy(), 0.5), "dy");
  chk(close(g.x_cell(0), 0.125), "x_cell0");
  chk(close(g.x_cell(3), 0.875), "x_cell3");
  chk(close(g.x_cell(-1), -0.125), "x_cell_ghost");
  chk(close(g.y_cell(0), 0.25), "y_cell0");

  Geometry gf = g.refine(2);
  chk(gf.domain == Box2D::from_extents(8, 4), "refine_domain");
  chk(close(gf.dx(), 0.125), "refine_dx");
  chk(close(gf.xhi, 1.0), "refine_extent");

  if (fails == 0) std::printf("OK test_geometry\n");
  return fails == 0 ? 0 : 1;
}
