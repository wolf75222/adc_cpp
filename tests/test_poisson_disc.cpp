// Paroi conductrice circulaire (embedded boundary) dans la multigrille.
// Solution manufacturee dans un disque de rayon R centre en (0.5,0.5) :
//   phi = R^2 - r^2  satisfait  lap(phi) = -4  et  phi = 0  sur le cercle.
// On resout lap(phi) = -4 avec phi=0 hors du disque (masque) et on compare a
// l'interieur (loin de la frontiere en escalier, O(dx) la-bas).

#include <adc/numerics/elliptic/mg/geometric_mg.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/layout/distribution_mapping.hpp>
#include <adc/mesh/storage/fab2d.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/multifab.hpp>

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

  const double cx = 0.5, cy = 0.5, R = 0.4;
  auto run = [&](int n, int& cycles, double& err) {
    Box2D dom = Box2D::from_extents(n, n);
    Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
    BoxArray ba = BoxArray::from_domain(dom, n);
    BCRec bc;
    bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;
    auto active = [=](Real x, Real y) { return std::hypot(x - cx, y - cy) < R; };
    GeometricMG mg(geom, ba, bc, active);
    mg.rhs().set_val(-4.0);
    mg.phi().set_val(0.0);

    // convergence degradee par la frontiere en escalier (le grossier ne la
    // represente pas) : on vise une reduction 1e-6 du residu, suffisante pour
    // la precision de troncature. En usage couple, le warm start ramene a
    // 1-2 cycles par pas.
    const Real r0 = mg.current_residual();
    Real rn = r0;
    cycles = 0;
    while (rn > 1e-6 * r0 && cycles < 200) {
      mg.vcycle();
      rn = mg.current_residual();
      ++cycles;
    }

    const Fab2D& p = mg.phi().fab(0);
    err = 0;
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
        const double x = geom.x_cell(i) - cx, y = geom.y_cell(j) - cy;
        const double r = std::hypot(x, y);
        if (r < 0.8 * R)  // interieur, loin de l'escalier
          err = std::max(err, std::fabs(p(i, j, 0) - (R * R - r * r)));
      }
  };

  int c128 = 0, c256 = 0;
  double e128 = 0, e256 = 0;
  run(128, c128, e128);
  run(256, c256, e256);
  std::printf("disc : n=128 cycles=%d err=%.3e | n=256 cycles=%d err=%.3e\n", c128, e128, c256,
              e256);

  chk(c128 < 200, "converged_128");
  chk(c256 < 200, "converged_256");
  chk(e256 < 5e-3, "accurate");
  chk(e256 < e128, "converges_with_resolution");

  if (fails == 0)
    std::printf("OK test_poisson_disc\n");
  return fails == 0 ? 0 : 1;
}
