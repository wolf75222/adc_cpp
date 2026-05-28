// Advection bout-en-bout : transport d'un cosinus par derive E x B a vitesse
// constante (aux prescrit), domaine periodique, SSPRK2 + Rusanov.
//   - masse conservee a l'arrondi (schema conservatif, halos periodiques)
//   - positivite preservee
//   - la solution est plus proche du profil TRANSLATE que du profil initial
//     (preuve que ca advecte a la bonne vitesse, pas un schema fige)

#include <adc/integrator/ssprk.hpp>
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

static double rho_profile(double x) { return 1.0 + 0.5 * std::cos(2 * kPi * x); }

static void fill_aux_uniform(MultiFab& aux, Real phi, Real gx, Real gy) {
  for (int li = 0; li < aux.local_size(); ++li) {
    Fab2D& f = aux.fab(li);
    const Box2D gb = f.grown_box();
    for (int j = gb.lo[1]; j <= gb.hi[1]; ++j)
      for (int i = gb.lo[0]; i <= gb.hi[0]; ++i) {
        f(i, j, 0) = phi;
        f(i, j, 1) = gx;
        f(i, j, 2) = gy;
      }
  }
}

// L1 moyen entre U et un profil analytique en x (translate de `shift`)
static double l1_to_profile(const MultiFab& U, const Geometry& g, double shift) {
  const Box2D dom = g.domain;
  double s = 0;
  long n = 0;
  const Fab2D& f = U.fab(0);
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
      double xe = g.x_cell(i) - shift;
      s += std::fabs(f(i, j, 0) - rho_profile(xe));
      ++n;
    }
  return s / n;
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int n = 32;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  Diocotron model;
  model.B0 = 1.0;
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), n_ranks());

  // vx = -grad_y/B0 = 1
  MultiFab aux(ba, dm, 3, 1);
  fill_aux_uniform(aux, 0.0, 0.0, -1.0);

  MultiFab U(ba, dm, 1, 1);
  Array4 a = U.fab(0).array();
  for_each_cell(dom, [a, geom](int i, int j) {
    a(i, j, 0) = rho_profile(geom.x_cell(i));
  });

  BCRec bc;  // periodique par defaut sur les 4 faces
  const double V = 1.0;
  const Real dt = 0.4 * geom.dx() / V;

  const double m0 = sum(U);
  double t = 0;
  const int steps = static_cast<int>(0.25 / dt);
  for (int s = 0; s < steps; ++s) {
    advance_ssprk2(model, U, aux, geom, bc, dt);
    t += dt;
  }

  // masse conservee
  chk(std::fabs(sum(U) - m0) < 1e-9, "mass_conserved");

  // positivite
  double mn = 1e300;
  const Fab2D& f = U.fab(0);
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) mn = std::min(mn, f(i, j, 0));
  chk(mn > 0.0, "positivity");

  // advection a la bonne vitesse : plus proche du profil translate que de l'initial
  const double err_shift = l1_to_profile(U, geom, V * t);
  const double err_static = l1_to_profile(U, geom, 0.0);
  chk(err_shift < err_static, "advected_not_static");
  chk(err_shift < 0.05, "shift_error_small");

  if (fails == 0) std::printf("OK test_advect\n");
  return fails == 0 ? 0 : 1;
}
