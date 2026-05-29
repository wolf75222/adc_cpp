// Deux-fluides isotherme 2D AP sur la couche MultiFab, via le header partage et
// PORTABLE GPU integrator/two_fluid_ap.hpp (memes kernels que l'exemple Kokkos
// examples/gpu/two_fluid_ap_kokkos.cpp). Etat deux-especes en MultiFab, transport
// par for_each_cell, halos par fill_boundary, champ par PoissonFFTSolver.
//
// Validation : stabilite AP (borne + quasi-neutre vs non stabilise qui explose),
// dispersion isotrope sur mode diagonal, conservation de la masse par espece.
// Mono-rang / boite unique (contrainte PoissonFFTSolver).

#include <adc/elliptic/poisson_fft_solver.hpp>
#include <adc/integrator/two_fluid_ap.hpp>
#include <adc/model/two_fluid_isothermal.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

using Driver = TwoFluidAP2D<PoissonFFTSolver>;

static double maxdev(const Driver& d) {
  double m = 0;
  const Fab2D& f = d.e.fab(0);
  for (int j = d.dom.lo[1]; j <= d.dom.hi[1]; ++j)
    for (int i = d.dom.lo[0]; i <= d.dom.hi[0]; ++i)
      m = std::fmax(m, std::fabs(f(i, j, 0) - 1.0));
  return m;
}
static double maxcharge(const Driver& d) {
  double m = 0;
  const Fab2D& fe = d.e.fab(0); const Fab2D& fi = d.ion.fab(0);
  for (int j = d.dom.lo[1]; j <= d.dom.hi[1]; ++j)
    for (int i = d.dom.lo[0]; i <= d.dom.hi[0]; ++i)
      m = std::fmax(m, std::fabs(fi(i, j, 0) - fe(i, j, 0)));
  return m;
}
static bool finite(const Driver& d) {
  const Fab2D& f = d.e.fab(0);
  for (int j = d.dom.lo[1]; j <= d.dom.hi[1]; ++j)
    for (int i = d.dom.lo[0]; i <= d.dom.hi[0]; ++i)
      if (!std::isfinite(f(i, j, 0)) || std::fabs(f(i, j, 0) - 1.0) > 1e3) return false;
  return true;
}
static double ampmode(const Driver& d, double k) {
  double s = 0;
  const Fab2D& f = d.e.fab(0);
  for (int j = d.dom.lo[1]; j <= d.dom.hi[1]; ++j)
    for (int i = d.dom.lo[0]; i <= d.dom.hi[0]; ++i)
      s += (f(i, j, 0) - 1.0) * std::cos(k * d.geom.x_cell(i) + k * d.geom.y_cell(j));
  return 2.0 * s / (d.n * d.n);
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  // --- 1. stabilite AP : raide, mode diagonal ---
  {
    Driver d(64, 2 * kPi, 1.0, 0.04, 1e3, 20.0);
    const double dt = 5.0 / 1e3;
    d.init(1e-3);
    const double m0e = sum(d.e, 0), m0i = sum(d.ion, 0);
    for (int t = 0; t < 300; ++t) d.step(dt, true);
    const double dev = maxdev(d), chg = maxcharge(d);
    const bool fin = finite(d);
    const double de = std::fabs(sum(d.e, 0) - m0e), di = std::fabs(sum(d.ion, 0) - m0i);

    Driver u(64, 2 * kPi, 1.0, 0.04, 1e3, 20.0);  // non stabilise
    u.init(1e-3);
    bool blew = false;
    for (int t = 0; t < 60; ++t) { u.step(dt, false); if (!finite(u)) blew = true; }

    std::printf("MF AP 2D raide (dt*omega_pe=5) : stabilise max|dne|=%.3e max|charge|=%.3e | "
                "non stabilise %s\n", dev, chg, blew ? "EXPLOSE" : "borne");
    chk(fin && dev < 0.1, "MF_AP2D_borne");
    chk(chg < 0.1, "MF_AP2D_quasi_neutre");
    chk(blew, "MF_non_stabilise_explose");
    chk(de < 1e-7 && di < 1e-7, "MF_AP2D_masse_conservee");
  }

  // --- 2. dispersion isotrope (mode diagonal k=(1,1), k^2=2) ---
  {
    const double L = 2 * kPi, k = 2 * kPi / L, k2 = 2 * k * k;
    Driver d(64, L, 1.0, 0.04, 5.0, 1.0);
    d.init(1e-3);
    TwoFluidLinear mode{5.0, 1.0, 1.0 * k2, 0.04 * k2};
    double wf, ws;
    mode.dispersion(wf, ws);
    const double dt = 0.02;
    double t = 0, prev = ampmode(d, k), tz = -1;
    for (int it = 0; it < 60 && tz < 0; ++it) {
      d.step(dt, true);
      t += dt;
      const double a = ampmode(d, k);
      if (a < 0 && prev > 0) tz = (t - dt) + dt * prev / (prev - a);
      prev = a;
    }
    const double wm = (tz > 0) ? kPi / (2 * tz) : 0;
    std::printf("MF AP 2D dispersion (mode diagonal) : w_fast theorique=%.3f mesure=%.3f "
                "(ecart %.1f%%)\n", wf, wm, 100 * std::fabs(wm - wf) / wf);
    chk(tz > 0 && std::fabs(wm - wf) / wf < 0.08, "MF_AP2D_dispersion_isotrope");
  }

  if (fails == 0) std::printf("OK test_two_fluid_ap_2d_mf\n");
  return fails == 0 ? 0 : 1;
}
