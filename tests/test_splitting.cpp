// Ordre du splitting d'operateur sur un systeme lineaire 2x2 NON commutant, dont
// le flot exact est connu (matrice exp). On evite ainsi toute erreur spatiale : on
// mesure UNIQUEMENT l'ordre temporel du splitting.
//
//   dU/dt = (A + B) U,  U in R^2,  A = [[0,1],[0,0]], B = [[0,0],[1,0]]
//   exp(A h) U : x += h y          (operateur "transport")
//   exp(B h) U : y += h x          (operateur "source")
//   [A,B] != 0  ->  le splitting a une vraie erreur de commutation
//   exact : exp((A+B)h), A+B = [[0,1],[1,0]] -> (x,y) = (x0 ch + y0 sh, x0 sh + y0 ch)
//
// Attendu : Strang 2e ordre (erreur /4 quand dt /2), Lie 1er ordre (/2).

#include <pops/numerics/time/schemes/splitting.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <cmath>
#include <cstdio>

using namespace pops;

static void fill_ic(MultiFab& U, double x0, double y0) {
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 a = U.fab(li).array();
    const Box2D b = U.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        a(i, j, 0) = x0;
        a(i, j, 1) = y0;
      }
  }
}

// flot exact exp(A h) : x += h y  ("transport")
static void stepA(MultiFab& U, Real h) {
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 a = U.fab(li).array();
    const Box2D b = U.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        a(i, j, 0) += h * a(i, j, 1);
  }
}
// flot exact exp(B h) : y += h x  ("source")
static void stepB(MultiFab& U, Real h) {
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 a = U.fab(li).array();
    const Box2D b = U.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        a(i, j, 1) += h * a(i, j, 0);
  }
}

// erreur max vs solution exacte a T, en n pas, schema = "strang" ou "lie".
static double run(bool strang, int n, double T, double x0, double y0) {
  const int N = 4;
  Box2D dom = Box2D::from_extents(N, N);
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, 1);
  MultiFab U(ba, dm, 2, 0);
  fill_ic(U, x0, y0);

  const Real dt = T / n;
  for (int s = 0; s < n; ++s) {
    if (strang)
      strang_step(U, dt, stepA, stepB);
    else
      lie_step(U, dt, stepA, stepB);
  }

  const double xe = x0 * std::cosh(T) + y0 * std::sinh(T);
  const double ye = x0 * std::sinh(T) + y0 * std::cosh(T);
  double err = 0;
  const ConstArray4 a = U.fab(0).const_array();
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
      err = std::fmax(err, std::fabs(a(i, j, 0) - xe));
      err = std::fmax(err, std::fabs(a(i, j, 1) - ye));
    }
  return err;
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const double T = 0.8, x0 = 1.0, y0 = 0.0;
  const double sE1 = run(true, 20, T, x0, y0), sE2 = run(true, 40, T, x0, y0);
  const double lE1 = run(false, 20, T, x0, y0), lE2 = run(false, 40, T, x0, y0);
  const double sOrder = std::log2(sE1 / sE2);
  const double lOrder = std::log2(lE1 / lE2);

  std::printf("Strang : err(20)=%.3e err(40)=%.3e ordre=%.2f\n", sE1, sE2, sOrder);
  std::printf("Lie    : err(20)=%.3e err(40)=%.3e ordre=%.2f\n", lE1, lE2, lOrder);

  chk(sOrder > 1.8 && sOrder < 2.2, "strang_ordre_2");
  chk(lOrder > 0.8 && lOrder < 1.3, "lie_ordre_1");
  chk(sE1 < lE1, "strang_plus_precis_que_lie");

  if (fails == 0)
    std::printf("OK test_splitting\n");
  return fails == 0 ? 0 : 1;
}
