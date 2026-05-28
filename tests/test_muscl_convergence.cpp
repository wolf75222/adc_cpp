// Convergence de la reconstruction : advection d'un cosinus lisse a vitesse
// constante (aux prescrit), domaine periodique. On mesure l'ordre par
// raffinement n=32 -> 64. VanLeer (MUSCL) doit donner ~ordre 2, le Rusanov 1er
// ordre (NoSlope) ~ordre 1. La positivite est preservee dans les deux cas.

#include <adc/integrator/ssprk.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/operator/reconstruction.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

static constexpr double kPi = 3.14159265358979323846;
static double rho0(double x) { return 1.0 + 0.5 * std::cos(2 * kPi * x); }

static void fill_aux_uniform(MultiFab& aux, Real gx, Real gy) {
  for (int li = 0; li < aux.local_size(); ++li) {
    Fab2D& f = aux.fab(li);
    const Box2D gb = f.grown_box();
    for (int j = gb.lo[1]; j <= gb.hi[1]; ++j)
      for (int i = gb.lo[0]; i <= gb.hi[0]; ++i) {
        f(i, j, 0) = 0;
        f(i, j, 1) = gx;
        f(i, j, 2) = gy;
      }
  }
}

template <class Lim>
static double advect_l1(int n, double& min_rho) {
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), n_ranks());
  Diocotron model;
  model.B0 = 1.0;

  MultiFab aux(ba, dm, 3, 2);
  fill_aux_uniform(aux, 0.0, -1.0);  // vx = -gy/B0 = 1

  MultiFab U(ba, dm, 1, 2);
  Array4 a = U.fab(0).array();
  for_each_cell(dom, [a, geom](int i, int j) { a(i, j, 0) = rho0(geom.x_cell(i)); });

  BCRec bc;  // periodique
  const Real dx = geom.dx();
  const Real dt = 0.4 * dx;
  const double t_end = 0.5;
  const int steps = static_cast<int>(t_end / dt);
  for (int s = 0; s < steps; ++s) advance_ssprk2<Lim>(model, U, aux, geom, bc, dt);
  const double t = steps * dt;

  const Fab2D& f = U.fab(0);
  double s = 0;
  long cnt = 0;
  min_rho = 1e300;
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
      s += std::fabs(f(i, j, 0) - rho0(geom.x_cell(i) - t));
      min_rho = std::min(min_rho, f(i, j, 0));
      ++cnt;
    }
  return s / cnt;
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  double m1, m2;
  const double e32_vl = advect_l1<VanLeer>(32, m1);
  const double e64_vl = advect_l1<VanLeer>(64, m2);
  const double order_vl = std::log(e32_vl / e64_vl) / std::log(2.0);

  double m3, m4;
  const double e32_fo = advect_l1<NoSlope>(32, m3);
  const double e64_fo = advect_l1<NoSlope>(64, m4);
  const double order_fo = std::log(e32_fo / e64_fo) / std::log(2.0);

  std::printf("VanLeer : e32=%.3e e64=%.3e ordre=%.2f\n", e32_vl, e64_vl, order_vl);
  std::printf("Rusanov : e32=%.3e e64=%.3e ordre=%.2f\n", e32_fo, e64_fo, order_fo);

  chk(order_vl > 1.7, "muscl_second_order");
  chk(order_fo < 1.3, "rusanov_first_order");
  chk(e64_vl < e64_fo, "muscl_more_accurate");
  chk(m1 > 0 && m2 > 0 && m3 > 0 && m4 > 0, "positivity");

  if (fails == 0) std::printf("OK test_muscl_convergence\n");
  return fails == 0 ? 0 : 1;
}
