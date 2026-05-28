// Coupleur Euler-Poisson ferme : Poisson -> aux -> advance, sur le diocotron.
//   - equilibre neutre (n_e = n_i0) : phi constant, derive nulle, U inchange
//   - perturbation 2D : la boucle couplee fait bouger le fluide, mais la masse
//     est conservee a l'arrondi (flux conservatif, derive E x B) et la
//     positivite preservee

#include <adc/coupling/coupler.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/diocotron.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

static constexpr double kPi = 3.14159265358979323846;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int n = 64;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), n_ranks());

  Diocotron model;
  model.B0 = 1.0;
  model.n_i0 = 1.0;
  model.alpha = 1.0;

  BCRec bc;  // periodique sur U et phi (diocotron periodique)

  // --- equilibre neutre : n_e = n_i0 -> aucune dynamique ---
  {
    Coupler<Diocotron> cpl(model, geom, ba, bc, bc);
    MultiFab U(ba, dm, 1, 1);
    U.set_val(1.0);
    const Real dx = geom.dx();
    for (int s = 0; s < 3; ++s) cpl.advance(U, 0.5 * dx);
    Real maxdev = 0;
    const Fab2D& f = U.fab(0);
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        maxdev = std::max(maxdev, std::fabs(f(i, j, 0) - 1.0));
    chk(maxdev < 1e-9, "neutral_equilibrium");
  }

  // --- perturbation 2D : masse conservee, positivite, dynamique non triviale ---
  {
    Coupler<Diocotron> cpl(model, geom, ba, bc, bc);
    MultiFab U(ba, dm, 1, 1), U0(ba, dm, 1, 0);
    Array4 a = U.fab(0).array();
    // deux blobs decentres : profil multi-modes (phi non proportionnel a rho),
    // donc derive E x B non perpendiculaire a grad rho -> dynamique reelle
    // (les deux blobs s'entrainent mutuellement, comme en diocotron)
    for_each_cell(dom, [a, geom](int i, int j) {
      const double x = geom.x_cell(i), y = geom.y_cell(j);
      auto blob = [&](double cx, double cy) {
        const double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
        return std::exp(-r2 / 0.01);
      };
      a(i, j, 0) = 1.0 + 0.5 * blob(0.35, 0.5) + 0.5 * blob(0.65, 0.5);
    });
    // copie de l'etat initial
    Array4 a0 = U0.fab(0).array();
    for_each_cell(dom, [a, a0](int i, int j) { a0(i, j, 0) = a(i, j, 0); });

    const Real m0 = sum(U);
    const Real dx = geom.dx();
    for (int s = 0; s < 20; ++s) cpl.advance(U, 0.5 * dx);

    chk(std::fabs(sum(U) - m0) < 1e-8, "mass_conserved");

    Real mn = 1e300, moved = 0;
    const Fab2D& f = U.fab(0);
    const Fab2D& f0 = U0.fab(0);
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
        mn = std::min(mn, f(i, j, 0));
        moved = std::max(moved, std::fabs(f(i, j, 0) - f0(i, j, 0)));
      }
    chk(mn > 0.0, "positivity");
    chk(moved > 1e-4, "coupled_dynamics");
  }

  if (fails == 0) std::printf("OK test_coupler\n");
  return fails == 0 ? 0 : 1;
}
