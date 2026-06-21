// Diagnostics AMR (coupling/amr_diagnostics.hpp) contre des valeurs analytiques.
// amr_mass (somme u*dV, routee par le seam reducteur for_each_cell_reduce_sum) et
// amr_max_drift_speed (max |grad phi| / B0, boucle hote). Pin les deux independamment
// des coupleurs : garde la conversion seam de amr_mass bit-identique a la boucle hote
// en serie, et fige le contrat de amr_max_drift_speed avant sa propre conversion.

#include <adc/coupling/amr_diagnostics.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/multifab.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int nc = 16;
  Box2D dom = Box2D::from_extents(nc, nc);
  const Real dx = Real(1) / nc, dy = Real(1) / nc;
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, n_ranks());

  // amr_mass, champ constant u = 2.5 : masse = 2.5 * Lx * Ly = 2.5 (Lx = Ly = 1).
  {
    MultiFab U(ba, dm, 1, 1);
    Array4 u = U.fab(0).array();
    const Box2D g = U.fab(0).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i)
        u(i, j, 0) = Real(2.5);
    const Real M = amr_mass(U, dom, dx, dy);
    std::printf("amr_mass(const 2.5) = %.15e (attendu 2.5)\n", M);
    chk(std::fabs(M - Real(2.5)) < 1e-13, "amr_mass_constant");
  }

  // amr_mass, champ varie : EXACTEMENT la boucle hote de reference en serie (meme dV
  // multiplie dans le noyau, meme ordre lexicographique j-externe/i-interne).
  {
    MultiFab U(ba, dm, 1, 1);
    Array4 u = U.fab(0).array();
    const Real dV = dx * dy;
    Real ref = 0;
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) {
        const Real v = i + 2 * j - Real(0.3) * i * j;
        u(i, j, 0) = v;
        ref += v * dV;
      }
    const Real M = amr_mass(U, dom, dx, dy);
    std::printf("amr_mass(varie) = %.15e (ref hote %.15e | diff %.2e)\n", M, ref,
                std::fabs(M - ref));
    chk(M == ref, "amr_mass_bit_identique_hote");
  }

  // amr_max_drift_speed, aux a gradient constant (gx, gy) : max = hypot(gx,gy) / B0.
  {
    const Real gx = Real(0.3), gy = Real(-0.4), B0 = Real(2);  // hypot = 0.5 -> 0.25
    MultiFab aux(ba, dm, 3, 1);
    Array4 a = aux.fab(0).array();
    const Box2D g = aux.fab(0).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
        a(i, j, 0) = 0;
        a(i, j, 1) = gx;
        a(i, j, 2) = gy;
      }
    const Real v = amr_max_drift_speed(aux, dom, B0);
    const Real exp = std::hypot(gx, gy) / B0;
    std::printf("amr_max_drift_speed(const) = %.15e (attendu %.15e)\n", v, exp);
    chk(std::fabs(v - exp) < 1e-13, "amr_max_drift_speed_const");
  }

  // amr_max_drift_speed, plancher 1e-12 quand le champ est nul (garde-fou CFL).
  {
    MultiFab aux(ba, dm, 3, 1);
    Array4 a = aux.fab(0).array();
    const Box2D g = aux.fab(0).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
        a(i, j, 0) = 0;
        a(i, j, 1) = 0;
        a(i, j, 2) = 0;
      }
    const Real v = amr_max_drift_speed(aux, dom, Real(1));
    chk(v == Real(1e-12), "amr_max_drift_speed_plancher");
  }

  if (fails == 0)
    std::printf("OK test_amr_diagnostics\n");
  return fails == 0 ? 0 : 1;
}
