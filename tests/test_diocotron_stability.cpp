// Invariants de stabilite du schema diocotron (transport E x B incompressible, div v = 0).
// Au-dela de la masse, le schema volumes finis limite doit respecter, meme quand
// l'instabilite croit (le mode s'amplifie mais les VALEURS sont transportees, pas creees) :
//   1. masse : sum(n) conservee a l'arrondi (forme flux + periodique) ;
//   2. principe du maximum : min/max de n preserves (le limiteur MUSCL evite les overshoots),
//      donc pas de densite negative ;
//   3. enstrophie : sum(n^2) NON CROISSANTE. En continu l'advection incompressible conserve
//      l'integrale de n^2 ; le schema upwind/MUSCL la dissipe -> elle ne peut que baisser.
// C'est un test de STABILITE (pas un ordre de convergence) : aucune croissance parasite.

#include <adc/coupling/coupler.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/operator/reconstruction.hpp>

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

  const int n = 64;
  const double L = 1.0, n_i0 = 1.0, eps = 0.3;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, L, 0.0, L};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, 1);

  Diocotron model;
  model.B0 = 1.0;
  model.n_i0 = n_i0;
  model.alpha = 1.0;

  MultiFab U(ba, dm, 1, 2);  // scalaire n_e
  {
    Fab2D& f = U.fab(0);
    const Box2D v = U.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        f(i, j, 0) = n_i0 + eps * std::sin(2 * kPi * geom.x_cell(i)) *
                                std::sin(2 * kPi * geom.y_cell(j));
  }

  auto stats = [&](double& nmin, double& nmax, double& enstro) {
    const ConstArray4 u = U.fab(0).const_array();
    const Box2D v = U.box(0);
    nmin = 1e30; nmax = -1e30; enstro = 0;
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        const double val = u(i, j, 0);
        nmin = std::min(nmin, val); nmax = std::max(nmax, val);
        enstro += val * val;
      }
  };

  double nmin0, nmax0, ens0;
  stats(nmin0, nmax0, ens0);
  const double mass0 = sum(U, 0);
  // derive E x B estimee ~ alpha eps / (4 pi B0) ~ 0.024 ; pas conservateur (CFL ~ 0.1).
  const double dt = 0.06;

  Coupler<Diocotron> cpl(model, geom, ba, BCRec{}, BCRec{});
  double ens_max = ens0, nmin_run = nmin0, nmax_run = nmax0;
  for (int s = 0; s < 150; ++s) {
    cpl.advance<Minmod>(U, dt);
    double a, b, e;
    stats(a, b, e);
    nmin_run = std::min(nmin_run, a);
    nmax_run = std::max(nmax_run, b);
    ens_max = std::max(ens_max, e);  // l'enstrophie ne doit jamais depasser l'initiale
  }
  double nminF, nmaxF, ensF;
  stats(nminF, nmaxF, ensF);
  const double dmass = std::fabs(sum(U, 0) - mass0);
  std::printf("diocotron : dmasse=%.2e | n in [%.4f, %.4f] (init [%.4f, %.4f]) | "
              "enstrophie %.4f -> %.4f (max %.4f)\n",
              dmass, nmin_run, nmax_run, nmin0, nmax0, ens0, ensF, ens_max);

  chk(dmass < 1e-10, "masse_conservee");
  chk(nmin_run > nmin0 - 1e-3 && nmax_run < nmax0 + 1e-3, "principe_du_maximum");
  chk(nmin_run > 0, "positivite");
  chk(ens_max < ens0 * (1 + 1e-9), "enstrophie_non_croissante");

  if (fails == 0) std::printf("OK test_diocotron_stability\n");
  return fails == 0 ? 0 : 1;
}
