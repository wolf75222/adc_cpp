// Deux-fluides isotherme 2D AP, PORTE sur la couche MultiFab du depot. C'est le
// meme schema que test_two_fluid_ap_2d, mais l'etat deux-especes vit desormais dans
// des MultiFab (3 composantes n, mx, my), le transport passe par for_each_cell, les
// halos periodiques par fill_boundary, et le champ par PoissonFFTSolver (le wrapper
// EllipticSolver autour de PoissonFFT). Une fois ici, le solveur herite "gratuitement"
// du seam de parallelisme (serial/OpenMP/Kokkos) et de la couche distribuee.
//
// Validation : on reproduit la reference self-contained -> stabilite AP (borne +
// quasi-neutre vs non stabilise qui explose), dispersion isotrope sur mode diagonal,
// conservation de la masse par espece. Mono-rang / boite unique (contrainte
// PoissonFFTSolver).

#include <adc/elliptic/poisson_fft_solver.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/model/two_fluid_isothermal.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

// Solveur AP 2D sur MultiFab. coup_s = omega_ps^2 ; z_e = -1, z_i = +1 ; n0 = 1.
struct Solver {
  int n;
  double L, dx, dy, cse2, csi2, ce, ci;  // ce = wpe^2, ci = wpi^2
  Box2D dom;
  Geometry geom;
  BoxArray ba;
  DistributionMapping dm;
  MultiFab e, ion;     // n, mx, my (ngrow 1)
  MultiFab mse, msi;   // mx*, my*  (ngrow 1)
  MultiFab nte, nti;   // densites tentatives (ngrow 0)
  MultiFab Ef;         // Ex, Ey    (ngrow 0)
  MultiFab mne, mni;   // mx, my finaux (ngrow 1)
  PoissonFFTSolver pf;
  Periodicity per{true, true};

  Solver(int n_, double L_, double cse2_, double csi2_, double wpe, double wpi)
      : n(n_), L(L_), dx(L_ / n_), dy(L_ / n_), cse2(cse2_), csi2(csi2_),
        ce(wpe * wpe), ci(wpi * wpi),
        dom(Box2D::from_extents(n_, n_)),
        geom{dom, 0.0, L_, 0.0, L_},
        ba(BoxArray::from_domain(dom, n_)),
        dm(ba.size(), n_ranks()),
        e(ba, dm, 3, 1), ion(ba, dm, 3, 1),
        mse(ba, dm, 2, 1), msi(ba, dm, 2, 1),
        nte(ba, dm, 1, 0), nti(ba, dm, 1, 0),
        Ef(ba, dm, 2, 0), mne(ba, dm, 2, 1), mni(ba, dm, 2, 1),
        pf(geom, ba) {}

  void init(double eps) {
    e.set_val(0);
    ion.set_val(0);
    Array4 ae = e.fab(0).array(), ai = ion.fab(0).array();
    const Geometry g = geom;
    const double k = 2 * kPi / L;
    for_each_cell(dom, [=](int i, int j) {
      ae(i, j, 0) = 1.0 * (1 + eps * std::cos(k * g.x_cell(i) + k * g.y_cell(j)));
      ai(i, j, 0) = 1.0;
    });
  }

  // -div des flux de qte de mvt (Rusanov dim-scinde) -> m* = m + dt*(-divF)
  void mstar(MultiFab& sp, MultiFab& ms, double c2, double dt) {
    ConstArray4 s = sp.fab(0).const_array();
    Array4 m = ms.fab(0).array();
    const double dx_ = dx, dy_ = dy;
    for_each_cell(dom, [=](int i, int j) {
      auto N = [&](int a, int b) { return s(a, b, 0); };
      auto MX = [&](int a, int b) { return s(a, b, 1); };
      auto MY = [&](int a, int b) { return s(a, b, 2); };
      const double cs = std::sqrt(c2);
      auto Fxx = [&](int a, int b) { return MX(a, b) * MX(a, b) / N(a, b) + c2 * N(a, b); };
      auto Fyx = [&](int a, int b) { return MX(a, b) * MY(a, b) / N(a, b); };
      auto Fxy = [&](int a, int b) { return MX(a, b) * MY(a, b) / N(a, b); };
      auto Fyy = [&](int a, int b) { return MY(a, b) * MY(a, b) / N(a, b) + c2 * N(a, b); };
      const double aR = std::fmax(std::fabs(MX(i, j) / N(i, j)) + cs,
                                  std::fabs(MX(i + 1, j) / N(i + 1, j)) + cs);
      const double aL = std::fmax(std::fabs(MX(i - 1, j) / N(i - 1, j)) + cs,
                                  std::fabs(MX(i, j) / N(i, j)) + cs);
      const double bU = std::fmax(std::fabs(MY(i, j) / N(i, j)) + cs,
                                  std::fabs(MY(i, j + 1) / N(i, j + 1)) + cs);
      const double bD = std::fmax(std::fabs(MY(i, j - 1) / N(i, j - 1)) + cs,
                                  std::fabs(MY(i, j) / N(i, j)) + cs);
      const double fxxR = 0.5 * (Fxx(i, j) + Fxx(i + 1, j)) - 0.5 * aR * (MX(i + 1, j) - MX(i, j));
      const double fxxL = 0.5 * (Fxx(i - 1, j) + Fxx(i, j)) - 0.5 * aL * (MX(i, j) - MX(i - 1, j));
      const double fyxR = 0.5 * (Fyx(i, j) + Fyx(i + 1, j)) - 0.5 * aR * (MY(i + 1, j) - MY(i, j));
      const double fyxL = 0.5 * (Fyx(i - 1, j) + Fyx(i, j)) - 0.5 * aL * (MY(i, j) - MY(i - 1, j));
      const double fxyU = 0.5 * (Fxy(i, j) + Fxy(i, j + 1)) - 0.5 * bU * (MX(i, j + 1) - MX(i, j));
      const double fxyD = 0.5 * (Fxy(i, j - 1) + Fxy(i, j)) - 0.5 * bD * (MX(i, j) - MX(i, j - 1));
      const double fyyU = 0.5 * (Fyy(i, j) + Fyy(i, j + 1)) - 0.5 * bU * (MY(i, j + 1) - MY(i, j));
      const double fyyD = 0.5 * (Fyy(i, j - 1) + Fyy(i, j)) - 0.5 * bD * (MY(i, j) - MY(i, j - 1));
      m(i, j, 0) = MX(i, j) + dt * (-((fxxR - fxxL) / dx_ + (fxyU - fxyD) / dy_));
      m(i, j, 1) = MY(i, j) + dt * (-((fyxR - fyxL) / dx_ + (fyyU - fyyD) / dy_));
    });
  }

  void ntilde(MultiFab& sp, MultiFab& ms, MultiFab& nt, double dt) {
    ConstArray4 s = sp.fab(0).const_array(), m = ms.fab(0).const_array();
    Array4 o = nt.fab(0).array();
    const double dx_ = dx, dy_ = dy;
    for_each_cell(dom, [=](int i, int j) {
      const double div = (m(i + 1, j, 0) - m(i - 1, j, 0)) / (2 * dx_) +
                         (m(i, j + 1, 1) - m(i, j - 1, 1)) / (2 * dy_);
      o(i, j, 0) = s(i, j, 0) - dt * div;
    });
  }

  void mnew(MultiFab& ms, MultiFab& mn, double z, double coup, double dt) {
    ConstArray4 m = ms.fab(0).const_array(), ef = Ef.fab(0).const_array();
    Array4 o = mn.fab(0).array();
    for_each_cell(dom, [=](int i, int j) {
      o(i, j, 0) = m(i, j, 0) + dt * z * coup * ef(i, j, 0);
      o(i, j, 1) = m(i, j, 1) + dt * z * coup * ef(i, j, 1);
    });
  }

  void update(MultiFab& sp, MultiFab& mn, double dt) {
    Array4 s = sp.fab(0).array();
    ConstArray4 m = mn.fab(0).const_array();
    const double dx_ = dx, dy_ = dy;
    for_each_cell(dom, [=](int i, int j) {
      const double div = (m(i + 1, j, 0) - m(i - 1, j, 0)) / (2 * dx_) +
                         (m(i, j + 1, 1) - m(i, j - 1, 1)) / (2 * dy_);
      s(i, j, 0) -= dt * div;
      s(i, j, 1) = m(i, j, 0);
      s(i, j, 2) = m(i, j, 1);
    });
  }

  void step(double dt, bool stabilize) {
    fill_boundary(e, dom, per);
    fill_boundary(ion, dom, per);
    mstar(e, mse, cse2, dt);
    mstar(ion, msi, csi2, dt);
    fill_boundary(mse, dom, per);
    fill_boundary(msi, dom, per);
    ntilde(e, mse, nte, dt);
    ntilde(ion, msi, nti, dt);
    const double beta0 = stabilize ? dt * dt * (ce + ci) : 0.0;  // n0 = 1
    {  // RHS du Poisson reformule : Lap(phi) = (ne* - ni*)/(1+beta0)
      ConstArray4 ne = nte.fab(0).const_array(), ni = nti.fab(0).const_array();
      Array4 r = pf.rhs().fab(0).array();
      for_each_cell(dom, [=](int i, int j) {
        r(i, j, 0) = (ne(i, j, 0) - ni(i, j, 0)) / (1 + beta0);
      });
    }
    pf.solve();
    fill_boundary(pf.phi(), dom, per);
    {  // E = -grad phi
      ConstArray4 p = pf.phi().fab(0).const_array();
      Array4 ef = Ef.fab(0).array();
      const double dx_ = dx, dy_ = dy;
      for_each_cell(dom, [=](int i, int j) {
        ef(i, j, 0) = -(p(i + 1, j, 0) - p(i - 1, j, 0)) / (2 * dx_);
        ef(i, j, 1) = -(p(i, j + 1, 0) - p(i, j - 1, 0)) / (2 * dy_);
      });
    }
    mnew(mse, mne, -1.0, ce, dt);  // electron
    mnew(msi, mni, +1.0, ci, dt);  // ion
    fill_boundary(mne, dom, per);
    fill_boundary(mni, dom, per);
    update(e, mne, dt);
    update(ion, mni, dt);
  }

  double maxdev() {
    double mx = 0;
    const Fab2D& f = e.fab(0);
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        mx = std::fmax(mx, std::fabs(f(i, j, 0) - 1.0));
    return mx;
  }
  double maxcharge() {
    double mx = 0;
    const Fab2D& fe = e.fab(0); const Fab2D& fi = ion.fab(0);
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        mx = std::fmax(mx, std::fabs(fi(i, j, 0) - fe(i, j, 0)));
    return mx;
  }
  bool finite() {
    const Fab2D& f = e.fab(0);
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        if (!std::isfinite(f(i, j, 0)) || std::fabs(f(i, j, 0) - 1.0) > 1e3) return false;
    return true;
  }
  double amp(double k) {
    double s = 0;
    const Fab2D& f = e.fab(0);
    const Geometry g = geom;
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        s += (f(i, j, 0) - 1.0) * std::cos(k * g.x_cell(i) + k * g.y_cell(j));
    return 2.0 * s / (n * n);
  }
};

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  // --- 1. stabilite AP : raide, mode diagonal, sur MultiFab ---
  {
    Solver s(64, 2 * kPi, 1.0, 0.04, 1e3, 20.0);
    const double dt = 5.0 / 1e3;
    s.init(1e-3);
    const double m0e = sum(s.e, 0), m0i = sum(s.ion, 0);
    for (int t = 0; t < 300; ++t) s.step(dt, true);
    const double dev = s.maxdev(), chg = s.maxcharge();
    const bool fin = s.finite();
    const double de = std::fabs(sum(s.e, 0) - m0e), di = std::fabs(sum(s.ion, 0) - m0i);

    Solver u(64, 2 * kPi, 1.0, 0.04, 1e3, 20.0);  // copie non stabilisee
    u.init(1e-3);
    bool blew = false;
    for (int t = 0; t < 60; ++t) { u.step(dt, false); if (!u.finite()) blew = true; }

    std::printf("MF AP 2D raide (dt*omega_pe=5) : stabilise max|dne|=%.3e max|charge|=%.3e | "
                "non stabilise %s\n", dev, chg, blew ? "EXPLOSE" : "borne");
    chk(fin && dev < 0.1, "MF_AP2D_borne");
    chk(chg < 0.1, "MF_AP2D_quasi_neutre");
    chk(blew, "MF_non_stabilise_explose");
    chk(de < 1e-7 && di < 1e-7, "MF_AP2D_masse_conservee");
  }

  // --- 2. dispersion isotrope (mode diagonal k=(1,1), k^2=2) sur MultiFab ---
  {
    const double L = 2 * kPi, k = 2 * kPi / L, k2 = 2 * k * k;
    Solver s(64, L, 1.0, 0.04, 5.0, 1.0);
    s.init(1e-3);
    TwoFluidLinear mode{5.0, 1.0, 1.0 * k2, 0.04 * k2};
    double wf, ws;
    mode.dispersion(wf, ws);
    const double dt = 0.02;
    double t = 0, prev = s.amp(k), tz = -1;
    for (int it = 0; it < 60 && tz < 0; ++it) {
      s.step(dt, true);
      t += dt;
      const double a = s.amp(k);
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
