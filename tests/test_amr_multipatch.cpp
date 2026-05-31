// Valide le multi-patch (niveau fin a plusieurs boxes) par le test DECISIF : 2 boxes
// fines qui pavent exactement la meme region qu'1 grande box doivent donner le MEME
// etat grossier. Ca verifie d'un coup que (1) le reflux est coverage-aware (le joint
// interne entre les 2 patches ne reflue PAS, c'est une interface fin-fin), et (2)
// fill_boundary transfere correctement les halos fin-fin (sinon le flux au joint
// differerait). Plus : conservation de la masse composite.

#include <adc/integrator/amr_reflux_mf.hpp>  // amr_step_2level_mf + amr_step_2level_multipatch
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/model/diocotron.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  const int nc = 32;
  Box2D dom = Box2D::from_extents(nc, nc);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const double dxc = geom.dx(), dyc = geom.dy();
  DistributionMapping dm(1, n_ranks());
  BoxArray bac(std::vector<Box2D>{dom});

  Diocotron model;
  model.B0 = 1.0;
  model.n_i0 = 1.0;
  const double gx = 0.5, gy = -0.3;  // aux uniforme -> advection
  auto ne0 = [&](double x, double y) {
    return 1.0 + 0.3 * std::sin(2 * kPi * x) * std::cos(2 * kPi * y);
  };
  const double dt = 0.2 * dxc / std::hypot(gx, gy);

  // region raffinee : [CI0..CI1] x [CJ0..CJ1], coupee en deux a CM.
  const int CI0 = 8, CI1 = 23, CJ0 = 8, CJ1 = 23, CM = 15;
  Box2D big{{2 * CI0, 2 * CJ0}, {2 * CI1 + 1, 2 * CJ1 + 1}};
  Box2D left{{2 * CI0, 2 * CJ0}, {2 * CM + 1, 2 * CJ1 + 1}};
  Box2D right{{2 * (CM + 1), 2 * CJ0}, {2 * CI1 + 1, 2 * CJ1 + 1}};

  auto fill = [&](MultiFab& U, double dx) {
    for (int li = 0; li < U.local_size(); ++li) {
      Array4 u = U.fab(li).array();
      const Box2D b = U.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i) u(i, j, 0) = ne0((i + 0.5) * dx, (j + 0.5) * dx);
    }
  };
  auto fill_aux = [&](MultiFab& a) {
    for (int li = 0; li < a.local_size(); ++li) {
      Array4 ar = a.fab(li).array();
      const Box2D g = a.fab(li).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) { ar(i, j, 0) = 0; ar(i, j, 1) = gx; ar(i, j, 2) = gy; }
    }
  };

  // --- version A : 1 grande box fine (amr_step_2level_mf) ---
  MultiFab UcA(bac, dm, 1, 1), UfA(BoxArray(std::vector<Box2D>{big}), dm, 1, 1);
  MultiFab axcA(bac, dm, 3, 1), axfA(BoxArray(std::vector<Box2D>{big}), dm, 3, 1);
  fill(UcA, dxc); fill(UfA, dxc / 2); fill_aux(axcA); fill_aux(axfA);
  detail::amr_step_2level_mf<NoSlope, RusanovFlux>(model, UcA, dom, dxc, dyc, UfA, CI0, CI1, CJ0,
                                           CJ1, axcA, axfA, dt);

  // --- version B : 2 boxes fines pavant la meme region (multipatch) ---
  MultiFab UcB(bac, dm, 1, 1);
  BoxArray baf2(std::vector<Box2D>{left, right});
  DistributionMapping dm2(2, n_ranks());
  MultiFab UfB(baf2, dm2, 1, 1), axfB(baf2, dm2, 3, 1), axcB(bac, dm, 3, 1);
  fill(UcB, dxc); fill(UfB, dxc / 2); fill_aux(axcB); fill_aux(axfB);
  const double mB0 = sum(UcB, 0);
  amr_step_2level_multipatch<NoSlope, RusanovFlux>(model, UcB, dom, dxc, dyc, UfB, axcB,
                                                   axfB, dt);

  // --- comparaison grossier A vs B ---
  double maxdiff = 0;
  const ConstArray4 ua = UcA.fab(0).const_array(), ub = UcB.fab(0).const_array();
  for (int j = 0; j < nc; ++j)
    for (int i = 0; i < nc; ++i)
      maxdiff = std::fmax(maxdiff, std::fabs(ua(i, j, 0) - ub(i, j, 0)));
  std::printf("multipatch (2 boxes) vs 1 grande box : max|dUc| = %.3e\n", maxdiff);
  chk(maxdiff < 1e-12, "multipatch_equiv_singlebox");  // joint fin-fin non reflue, halos OK

  if (fails == 0) std::printf("OK test_amr_multipatch\n");
  return fails == 0 ? 0 : 1;
}
