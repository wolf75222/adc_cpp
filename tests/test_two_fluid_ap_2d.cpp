// Deux-fluides isotherme SPATIAL 2D, asymptotic-preserving, branche sur le VRAI
// solveur spectral du depot (elliptic/poisson_fft.hpp, le meme que PoissonFFTSolver).
// Avec beta0 = dt^2 (omega_pe^2 + omega_pi^2) CONSTANT (exact en regime lineaire,
// n ~ n0), la reformulation d_x[(1+beta)E] = charge* devient un Poisson a coefficient
// CONSTANT : (1+beta0) Lap(phi) = (ne* - ni*), donc
//   Lap(phi) = (ne* - ni*) / (1+beta0),  E = -grad phi.
// Le solveur de Poisson existant s'utilise tel quel, on divise juste le RHS par
// (1+beta0). L'etape de champ AP retombe sur l'infrastructure elliptique du code.
//
// Validation 2D : (1) stabilite AP a omega_pe=1e3, dt*omega_pe=5 (borne + quasi-neutre
// vs version non stabilisee qui explose) ; (2) dispersion isotrope sur un mode
// DIAGONAL (k = (1,1), donc d_x ET d_y actifs) vs TwoFluidLinear ; (3) conservation
// de la masse par espece.

#include <adc/elliptic/poisson_fft.hpp>
#include <adc/model/two_fluid_isothermal.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

struct Sp2D {
  double c2, coup, z;                  // c_s^2, omega_ps^2, signe de charge
  std::vector<double> n, mx, my;       // densite, qte de mvt x et y (row-major j*Nx+i)
};

// -div des flux de quantite de mouvement (Rusanov par direction), 2D periodique.
static void mom_flux_rhs(const Sp2D& s, int Nx, int Ny, double dx, double dy,
                         std::vector<double>& dmx, std::vector<double>& dmy) {
  const double cs = std::sqrt(s.c2);
  auto id = [&](int i, int j) { return j * Nx + i; };
  std::vector<double> fxx(Nx * Ny), fyx(Nx * Ny), fxy(Nx * Ny), fyy(Nx * Ny);
  for (int j = 0; j < Ny; ++j)
    for (int i = 0; i < Nx; ++i) {
      const int k = id(i, j), kxp = id((i + 1) % Nx, j), kyp = id(i, (j + 1) % Ny);
      // flux en x (interface i+1/2) : Fxx = mx^2/n + c^2 n, Fyx = mx my / n
      const double ax = std::fmax(std::fabs(s.mx[k] / s.n[k]) + cs,
                                  std::fabs(s.mx[kxp] / s.n[kxp]) + cs);
      const double Fxx_k = s.mx[k] * s.mx[k] / s.n[k] + s.c2 * s.n[k];
      const double Fxx_p = s.mx[kxp] * s.mx[kxp] / s.n[kxp] + s.c2 * s.n[kxp];
      const double Fyx_k = s.mx[k] * s.my[k] / s.n[k];
      const double Fyx_p = s.mx[kxp] * s.my[kxp] / s.n[kxp];
      fxx[k] = 0.5 * (Fxx_k + Fxx_p) - 0.5 * ax * (s.mx[kxp] - s.mx[k]);
      fyx[k] = 0.5 * (Fyx_k + Fyx_p) - 0.5 * ax * (s.my[kxp] - s.my[k]);
      // flux en y (interface j+1/2) : Fxy = mx my / n, Fyy = my^2/n + c^2 n
      const double ay = std::fmax(std::fabs(s.my[k] / s.n[k]) + cs,
                                  std::fabs(s.my[kyp] / s.n[kyp]) + cs);
      const double Fxy_k = s.mx[k] * s.my[k] / s.n[k];
      const double Fxy_p = s.mx[kyp] * s.my[kyp] / s.n[kyp];
      const double Fyy_k = s.my[k] * s.my[k] / s.n[k] + s.c2 * s.n[k];
      const double Fyy_p = s.my[kyp] * s.my[kyp] / s.n[kyp] + s.c2 * s.n[kyp];
      fxy[k] = 0.5 * (Fxy_k + Fxy_p) - 0.5 * ay * (s.mx[kyp] - s.mx[k]);
      fyy[k] = 0.5 * (Fyy_k + Fyy_p) - 0.5 * ay * (s.my[kyp] - s.my[k]);
    }
  for (int j = 0; j < Ny; ++j)
    for (int i = 0; i < Nx; ++i) {
      const int k = id(i, j), kxm = id((i - 1 + Nx) % Nx, j), kym = id(i, (j - 1 + Ny) % Ny);
      dmx[k] = -((fxx[k] - fxx[kxm]) / dx + (fxy[k] - fxy[kym]) / dy);
      dmy[k] = -((fyx[k] - fyx[kxm]) / dx + (fyy[k] - fyy[kym]) / dy);
    }
}

// divergence centree de (mx,my), 2D periodique.
static void divergence(const std::vector<double>& mx, const std::vector<double>& my,
                       int Nx, int Ny, double dx, double dy, std::vector<double>& d) {
  auto id = [&](int i, int j) { return j * Nx + i; };
  for (int j = 0; j < Ny; ++j)
    for (int i = 0; i < Nx; ++i)
      d[id(i, j)] = (mx[id((i + 1) % Nx, j)] - mx[id((i - 1 + Nx) % Nx, j)]) / (2 * dx) +
                    (my[id(i, (j + 1) % Ny)] - my[id(i, (j - 1 + Ny) % Ny)]) / (2 * dy);
}

// Un pas AP 2D. stabilize=false : beta0=0 (schema non stabilise, explose au raide).
static void ap_step_2d(Sp2D& e, Sp2D& ion, int Nx, int Ny, double dx, double dy,
                       double dt, bool stabilize, PoissonFFT& pf) {
  const int M = Nx * Ny;
  std::vector<double> dmex(M), dmey(M), dmix(M), dmiy(M);
  mom_flux_rhs(e, Nx, Ny, dx, dy, dmex, dmey);
  mom_flux_rhs(ion, Nx, Ny, dx, dy, dmix, dmiy);
  std::vector<double> mes(M), meys(M), mis(M), miys(M);  // m* (apres flux explicite)
  for (int k = 0; k < M; ++k) {
    mes[k] = e.mx[k] + dt * dmex[k];
    meys[k] = e.my[k] + dt * dmey[k];
    mis[k] = ion.mx[k] + dt * dmix[k];
    miys[k] = ion.my[k] + dt * dmiy[k];
  }
  std::vector<double> dive(M), divi(M), nte(M), nti(M);
  divergence(mes, meys, Nx, Ny, dx, dy, dive);
  divergence(mis, miys, Nx, Ny, dx, dy, divi);
  for (int k = 0; k < M; ++k) {
    nte[k] = e.n[k] - dt * dive[k];
    nti[k] = ion.n[k] - dt * divi[k];
  }
  // champ reformule : Lap(phi) = (nte - nti)/(1+beta0)
  const double beta0 = stabilize ? dt * dt * (e.coup + ion.coup) : 0.0;  // n0 = 1
  std::vector<double> rho(M), phi, Ex(M), Ey(M);
  for (int k = 0; k < M; ++k) rho[k] = (nte[k] - nti[k]) / (1 + beta0);
  pf.solve(rho, phi);
  auto id = [&](int i, int j) { return j * Nx + i; };
  for (int j = 0; j < Ny; ++j)
    for (int i = 0; i < Nx; ++i) {  // E = -grad phi (centre)
      Ex[id(i, j)] = -(phi[id((i + 1) % Nx, j)] - phi[id((i - 1 + Nx) % Nx, j)]) / (2 * dx);
      Ey[id(i, j)] = -(phi[id(i, (j + 1) % Ny)] - phi[id(i, (j - 1 + Ny) % Ny)]) / (2 * dy);
    }
  // Lorentz implicite (coefficient n0 = 1)
  std::vector<double> men(M), meyn(M), min_(M), miyn(M);
  for (int k = 0; k < M; ++k) {
    men[k] = mes[k] + dt * e.z * e.coup * Ex[k];
    meyn[k] = meys[k] + dt * e.z * e.coup * Ey[k];
    min_[k] = mis[k] + dt * ion.z * ion.coup * Ex[k];
    miyn[k] = miys[k] + dt * ion.z * ion.coup * Ey[k];
  }
  // continuite conservative avec la quantite de mouvement finale
  divergence(men, meyn, Nx, Ny, dx, dy, dive);
  divergence(min_, miyn, Nx, Ny, dx, dy, divi);
  for (int k = 0; k < M; ++k) {
    e.n[k] -= dt * dive[k];
    ion.n[k] -= dt * divi[k];
  }
  e.mx = men; e.my = meyn; ion.mx = min_; ion.my = miyn;
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  const int Nx = 64, Ny = 64;
  const double L = 2 * kPi, dx = L / Nx, dy = L / Ny;
  const double n0 = 1, eps = 1e-3, kx = 2 * kPi / L, ky = 2 * kPi / L;  // mode diagonal (1,1)
  PoissonFFT pf(Nx, Ny, L, L);
  auto id = [&](int i, int j) { return j * Nx + i; };
  auto make = [&](double wpe, double wpi, double cse2, double csi2) {
    Sp2D e{cse2, wpe * wpe, -1.0, std::vector<double>(Nx * Ny), std::vector<double>(Nx * Ny, 0.0),
           std::vector<double>(Nx * Ny, 0.0)};
    Sp2D ion{csi2, wpi * wpi, +1.0, std::vector<double>(Nx * Ny, n0),
             std::vector<double>(Nx * Ny, 0.0), std::vector<double>(Nx * Ny, 0.0)};
    for (int j = 0; j < Ny; ++j)
      for (int i = 0; i < Nx; ++i)
        e.n[id(i, j)] = n0 * (1 + eps * std::cos(kx * (i + 0.5) * dx + ky * (j + 0.5) * dy));
    return std::pair{e, ion};
  };
  auto mass = [&](const std::vector<double>& a) {
    double s = 0; for (double v : a) s += v; return s;
  };
  auto max_charge = [&](const Sp2D& e, const Sp2D& ion) {
    double m = 0;
    for (int k = 0; k < Nx * Ny; ++k) m = std::fmax(m, std::fabs(ion.n[k] - e.n[k]));
    return m;
  };

  // --- 1. stabilite AP : raide, mode diagonal ---
  {
    const double wpe = 1e3, wpi = 20, cse2 = 1, csi2 = 0.04, dt = 5.0 / wpe;
    auto [e, ion] = make(wpe, wpi, cse2, csi2);
    auto eu = e, ionu = ion;  // copie non stabilisee
    const double me0 = mass(e.n), mi0 = mass(ion.n);
    bool apfin = true;
    for (int s = 0; s < 300; ++s) {
      ap_step_2d(e, ion, Nx, Ny, dx, dy, dt, true, pf);
      for (double v : e.n) if (!std::isfinite(v)) apfin = false;
    }
    bool blew = false;
    for (int s = 0; s < 60; ++s) {
      ap_step_2d(eu, ionu, Nx, Ny, dx, dy, dt, false, pf);  // beta0 = 0
      for (double v : eu.n) if (!std::isfinite(v) || std::fabs(v - n0) > 1e3) blew = true;
    }
    double apdev = 0;
    for (double v : e.n) apdev = std::fmax(apdev, std::fabs(v - n0));
    std::printf("AP 2D raide (dt*omega_pe=5) : stabilise max|dne|=%.3e max|charge|=%.3e | "
                "non stabilise %s\n", apdev, max_charge(e, ion), blew ? "EXPLOSE" : "borne");
    chk(apfin && apdev < 0.1, "AP2D_borne");
    chk(max_charge(e, ion) < 0.1, "AP2D_quasi_neutre");
    chk(blew, "non_stabilise_explose");
    chk(std::fabs(mass(e.n) - me0) < 1e-8 && std::fabs(mass(ion.n) - mi0) < 1e-8,
        "AP2D_masse_conservee");
  }

  // --- 2. dispersion isotrope sur mode diagonal k=(1,1), k^2 = 2 ---
  {
    const double wpe = 5, wpi = 1, cse2 = 1, csi2 = 0.04, k2 = kx * kx + ky * ky;
    auto [e, ion] = make(wpe, wpi, cse2, csi2);
    TwoFluidLinear mode{wpe, wpi, cse2 * k2, csi2 * k2};
    double wf, ws;
    mode.dispersion(wf, ws);
    const double dt = 0.02;  // resolu + CFL acoustique + beta0 petit (~0.01)
    auto amp = [&]() {
      double s = 0;
      for (int j = 0; j < Ny; ++j)
        for (int i = 0; i < Nx; ++i)
          s += (e.n[id(i, j)] - n0) * std::cos(kx * (i + 0.5) * dx + ky * (j + 0.5) * dy);
      return 2.0 * s / (Nx * Ny);
    };
    double t = 0, prev = amp(), tz = -1;
    for (int s = 0; s < 60 && tz < 0; ++s) {
      ap_step_2d(e, ion, Nx, Ny, dx, dy, dt, true, pf);
      t += dt;
      const double a = amp();
      if (a < 0 && prev > 0) tz = (t - dt) + dt * prev / (prev - a);
      prev = a;
    }
    const double wm = (tz > 0) ? kPi / (2 * tz) : 0;
    std::printf("AP 2D dispersion (mode diagonal, k^2=%.1f) : w_fast theorique=%.3f "
                "mesure=%.3f (ecart %.1f%%)\n", k2, wf, wm, 100 * std::fabs(wm - wf) / wf);
    chk(tz > 0 && std::fabs(wm - wf) / wf < 0.08, "AP2D_dispersion_isotrope");
  }

  if (fails == 0) std::printf("OK test_two_fluid_ap_2d\n");
  return fails == 0 ? 0 : 1;
}
