// Valide compute_face_fluxes (brique reflux AMR) : la divergence des flux de FACE
// doit redonner EXACTEMENT le residu d'assemble_rhs (qui calcule -div F directement
// puis jette les flux). C'est l'invariant qui garantit qu'on peut batir le reflux
// AMR sur compute_face_fluxes sans changer la physique. Teste sur Euler (source
// nulle), en NoSlope/Rusanov ET MUSCL/HLLC, bit a bit.

#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/euler.hpp>
#include <adc/operator/numerical_flux.hpp>
#include <adc/operator/reconstruction.hpp>
#include <adc/operator/spatial_operator.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

template <class Limiter, class Flux>
static double max_div_mismatch(int n) {
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, n_ranks());
  Euler model;

  MultiFab U(ba, dm, 4, 2), aux(ba, dm, 3, 1), R(ba, dm, 4, 0);
  aux.set_val(0.0);
  Array4 u = U.fab(0).array();
  for (int j = -2; j < n + 2; ++j)
    for (int i = -2; i < n + 2; ++i)
      if (U.fab(0).grown_box().contains(i, j)) {
        const double x = (i + 0.5) / n, y = (j + 0.5) / n;
        const double rho = 1.0 + 0.2 * std::sin(2 * kPi * x) * std::cos(2 * kPi * y);
        const double ux = 0.3 * std::cos(2 * kPi * x), uy = -0.2 * std::sin(2 * kPi * y);
        const double p = 1.0 + 0.1 * std::sin(2 * kPi * (x + y));
        u(i, j, 0) = rho;
        u(i, j, 1) = rho * ux;
        u(i, j, 2) = rho * uy;
        u(i, j, 3) = p / (1.4 - 1) + 0.5 * rho * (ux * ux + uy * uy);
      }
  fill_boundary(U, dom, Periodicity{true, true});

  // reference : assemble_rhs (source Euler nulle -> R = -div F)
  assemble_rhs<Limiter, Flux>(model, U, aux, geom, R);

  // candidat : -div(compute_face_fluxes)
  BoxArray bax(std::vector<Box2D>{xface_box(dom)});
  BoxArray bay(std::vector<Box2D>{yface_box(dom)});
  MultiFab Fx(bax, dm, 4, 0), Fy(bay, dm, 4, 0);
  compute_face_fluxes<Limiter, Flux>(model, U, aux, Fx, Fy);

  const ConstArray4 r = R.fab(0).const_array();
  const ConstArray4 fx = Fx.fab(0).const_array();
  const ConstArray4 fy = Fy.fab(0).const_array();
  const double dx = geom.dx(), dy = geom.dy();
  double maxerr = 0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      for (int c = 0; c < 4; ++c) {
        const double div = -(fx(i + 1, j, c) - fx(i, j, c)) / dx -
                           (fy(i, j + 1, c) - fy(i, j, c)) / dy;
        maxerr = std::fmax(maxerr, std::fabs(div - r(i, j, c)));
      }
  return maxerr;
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  const double e1 = max_div_mismatch<NoSlope, RusanovFlux>(48);
  const double e2 = max_div_mismatch<Minmod, HLLCFlux>(48);
  std::printf("div(face_fluxes) vs assemble_rhs : NoSlope/Rusanov=%.2e Minmod/HLLC=%.2e\n",
              e1, e2);
  chk(e1 == 0.0, "faces_equiv_noslope_rusanov");  // bit-identique
  chk(e2 == 0.0, "faces_equiv_muscl_hllc");

  if (fails == 0) std::printf("OK test_face_fluxes\n");
  return fails == 0 ? 0 : 1;
}
