// Lisseur Gauss-Seidel red-black sur une solution manufacturee Dirichlet :
//   phi = sin(pi x) sin(pi y) sur [0,1]^2 (nulle au bord),
//   lap(phi) = -2 pi^2 phi  ->  on resout lap(phi) = f avec f = -2 pi^2 phi.
// On verifie que le residu chute fortement et que la solution converge vers la
// solution exacte a O(dx^2).

#include <adc/numerics/elliptic/poisson_operator.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

static constexpr double kPi = 3.14159265358979323846;

static double phi_exact(double x, double y) {
  return std::sin(kPi * x) * std::sin(kPi * y);
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int n = 16;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), n_ranks());

  MultiFab phi(ba, dm, 1, 1), f(ba, dm, 1, 0), res(ba, dm, 1, 0);

  // second membre f = -2 pi^2 sin(pi x) sin(pi y)
  Array4 af = f.fab(0).array();
  for_each_cell(dom, [af, geom](int i, int j) {
    af(i, j, 0) = -2 * kPi * kPi * phi_exact(geom.x_cell(i), geom.y_cell(j));
  });

  phi.set_val(0.0);  // initial nul (ghosts compris)

  BCRec bc;  // Dirichlet 0 sur les 4 faces
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;

  poisson_residual(phi, f, geom, bc, res);
  const Real r0 = norm_inf(res);

  gs_smooth(phi, f, geom, bc, 800);

  poisson_residual(phi, f, geom, bc, res);
  const Real rN = norm_inf(res);
  chk(rN / r0 < 1e-4, "residual_reduced");

  // erreur vs solution exacte (au noeud du laplacien discret) : O(dx^2)
  Real err = 0;
  const Fab2D& p = phi.fab(0);
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
      Real e = std::fabs(p(i, j, 0) - phi_exact(geom.x_cell(i), geom.y_cell(j)));
      err = std::max(err, e);
    }
  chk(err < 0.02, "solution_accurate");

  if (fails == 0)
    std::printf("OK test_poisson_smoother (r0=%.3e rN=%.3e err=%.3e)\n", r0, rN, err);
  return fails == 0 ? 0 : 1;
}
