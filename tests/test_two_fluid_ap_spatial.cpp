// Deux-fluides isotherme SPATIAL 1D, schema ASYMPTOTIC-PRESERVING. On rend la
// continuite + Lorentz IMPLICITES : en substituant, l'equation du champ devient
//   d_x[(1+beta) E] = charge*,  beta = dt^2 (omega_pe^2 n_e + omega_pi^2 n_i),
// "Poisson reformule". Quand omega_pe -> infini (lambda_D -> 0), beta -> infini et
// E -> charge*/beta : la quasi-neutralite est capturee a dt FIXE, sans resoudre
// l'echelle plasma. C'est le coeur de APIMEXTwoFluidIsothermal (MUFFIN).
//
// Validation : a omega_pe = 1e3 et dt*omega_pe = 5 (raide), le schema AP reste BORNE
// et quasi-neutre la ou l'explicite EXPLOSE. Regime non raide : reproduit la
// dispersion. Conservation de la masse par espece. 1D, regime lineaire (diff.
// centrees) ; le non lineaire upwind + 2D + MultiFab restent la suite.

#include <adc/model/two_fluid_isothermal.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

struct Grid {
  int N;
  double dx, n0, cse2, csi2, coup_e, coup_i;  // coup_s = omega_ps^2
  std::vector<double> ne, me, ni, mi;
};

static double Dx(const std::vector<double>& f, int i, double dx) {  // central, periodique
  const int N = static_cast<int>(f.size());
  return (f[(i + 1) % N] - f[(i - 1 + N) % N]) / (2 * dx);
}

// Pas AP : continuite + Lorentz implicites via Poisson reformule.
static void ap_step(Grid& g, double dt) {
  const int N = g.N;
  const double dx = g.dx;
  std::vector<double> mes(N), mis(N), nte(N), nti(N), E(N), H(N), beta(N), rhs(N);
  // 1. flux de quantite de mouvement explicite (centre)
  auto Fm = [&](const std::vector<double>& n, const std::vector<double>& m,
                double c2, int i) { return m[i] * m[i] / n[i] + c2 * n[i]; };
  std::vector<double> Fme(N), Fmi(N);
  for (int i = 0; i < N; ++i) {
    Fme[i] = Fm(g.ne, g.me, g.cse2, i);
    Fmi[i] = Fm(g.ni, g.mi, g.csi2, i);
  }
  for (int i = 0; i < N; ++i) {
    mes[i] = g.me[i] - dt * Dx(Fme, i, dx);
    mis[i] = g.mi[i] - dt * Dx(Fmi, i, dx);
  }
  // 2. densites tentatives (continuite avec m*)
  for (int i = 0; i < N; ++i) {
    nte[i] = g.ne[i] - dt * Dx(mes, i, dx);
    nti[i] = g.ni[i] - dt * Dx(mis, i, dx);
  }
  // 3. champ reformule : d_x[(1+beta)E] = (nti - nte)
  double meanrhs = 0;
  for (int i = 0; i < N; ++i) {
    beta[i] = dt * dt * (g.coup_i * g.ni[i] + g.coup_e * g.ne[i]);
    rhs[i] = nti[i] - nte[i];
    meanrhs += rhs[i];
  }
  meanrhs /= N;
  double acc = 0;
  for (int i = 0; i < N; ++i) {
    rhs[i] -= meanrhs;
    H[i] = acc + 0.5 * dx * rhs[i];
    acc += dx * rhs[i];
  }
  double s1 = 0, s2 = 0;  // jauge : moyenne(E) = 0
  for (int i = 0; i < N; ++i) {
    s1 += H[i] / (1 + beta[i]);
    s2 += 1.0 / (1 + beta[i]);
  }
  const double C = -s1 / s2;
  for (int i = 0; i < N; ++i) E[i] = (H[i] + C) / (1 + beta[i]);
  // 4. Lorentz implicite (n^n gele en coefficient)
  std::vector<double> men(N), min_(N);
  for (int i = 0; i < N; ++i) {
    men[i] = mes[i] + dt * (-g.coup_e) * g.ne[i] * E[i];
    min_[i] = mis[i] + dt * (+g.coup_i) * g.ni[i] * E[i];
  }
  // 5. continuite conservative avec la quantite de mouvement finale
  std::vector<double> ne2 = g.ne, ni2 = g.ni;
  for (int i = 0; i < N; ++i) {
    ne2[i] = g.ne[i] - dt * Dx(men, i, dx);
    ni2[i] = g.ni[i] - dt * Dx(min_, i, dx);
  }
  g.ne = ne2;
  g.ni = ni2;
  g.me = men;
  g.mi = min_;
}

// Pas explicite (pour contraste) : Lorentz explicite, champ d_x E = charge.
static void explicit_step(Grid& g, double dt) {
  const int N = g.N;
  const double dx = g.dx;
  std::vector<double> Fme(N), Fmi(N), E(N), H(N);
  auto Fm = [&](const std::vector<double>& n, const std::vector<double>& m,
                double c2, int i) { return m[i] * m[i] / n[i] + c2 * n[i]; };
  for (int i = 0; i < N; ++i) {
    Fme[i] = Fm(g.ne, g.me, g.cse2, i);
    Fmi[i] = Fm(g.ni, g.mi, g.csi2, i);
  }
  double mean = 0, acc = 0;
  std::vector<double> rhs(N);
  for (int i = 0; i < N; ++i) { rhs[i] = g.ni[i] - g.ne[i]; mean += rhs[i]; }
  mean /= N;
  double sE = 0;
  for (int i = 0; i < N; ++i) { rhs[i] -= mean; H[i] = acc + 0.5 * dx * rhs[i]; acc += dx * rhs[i]; sE += H[i]; }
  sE /= N;
  for (int i = 0; i < N; ++i) E[i] = H[i] - sE;
  std::vector<double> ne2 = g.ne, me2 = g.me, ni2 = g.ni, mi2 = g.mi;
  for (int i = 0; i < N; ++i) {
    ne2[i] = g.ne[i] - dt * Dx(g.me, i, dx);
    ni2[i] = g.ni[i] - dt * Dx(g.mi, i, dx);
    me2[i] = g.me[i] - dt * Dx(Fme, i, dx) + dt * (-g.coup_e) * g.ne[i] * E[i];
    mi2[i] = g.mi[i] - dt * Dx(Fmi, i, dx) + dt * (+g.coup_i) * g.ni[i] * E[i];
  }
  g.ne = ne2; g.ni = ni2; g.me = me2; g.mi = mi2;
}

static Grid make_grid(int N, double L, double n0, double wpe, double wpi,
                      double cse2, double csi2, double eps) {
  Grid g{N, L / N, n0, cse2, csi2, wpe * wpe, wpi * wpi, {}, {}, {}, {}};
  g.ne.resize(N); g.me.assign(N, 0); g.ni.assign(N, n0); g.mi.assign(N, 0);
  const double k = 2 * kPi / L;
  for (int i = 0; i < N; ++i) g.ne[i] = n0 * (1 + eps * std::cos(k * (i + 0.5) * g.dx));
  return g;
}
static double maxdev(const Grid& g) {
  double m = 0;
  for (int i = 0; i < g.N; ++i) m = std::fmax(m, std::fabs(g.ne[i] - g.n0));
  return m;
}
static double max_charge(const Grid& g) {
  double m = 0;
  for (int i = 0; i < g.N; ++i) m = std::fmax(m, std::fabs(g.ni[i] - g.ne[i]));
  return m;
}
static double mass(const std::vector<double>& a) {
  double s = 0; for (double v : a) s += v; return s;
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  const int N = 128;
  const double L = 2 * kPi, n0 = 1, eps = 1e-3;

  // --- AP : raide (omega_pe=1e3), dt*omega_pe = 5 ---
  {
    const double wpe = 1e3, wpi = 20, cse2 = 1, csi2 = 0.04;
    const double dt = 5.0 / wpe;  // dt*omega_pe = 5 : explicite instable
    Grid ap = make_grid(N, L, n0, wpe, wpi, cse2, csi2, eps);
    Grid ex = ap;
    const double me0 = mass(ap.ne), mi0 = mass(ap.ni);
    bool apfin = true;
    for (int s = 0; s < 300; ++s) {
      ap_step(ap, dt);
      for (double v : ap.ne) if (!std::isfinite(v)) apfin = false;
    }
    bool exblew = false;  // std::fmax ignore les NaN -> on detecte l'explosion a la main
    for (int s = 0; s < 50; ++s) {                       // explose vite
      explicit_step(ex, dt);
      for (double v : ex.ne)
        if (!std::isfinite(v) || std::fabs(v - n0) > 1e3) exblew = true;
    }
    std::printf("AP raide (dt*omega_pe=5) : AP max|dne|=%.3e max|charge|=%.3e | "
                "explicite %s\n",
                maxdev(ap), max_charge(ap), exblew ? "EXPLOSE" : "borne");
    chk(apfin && maxdev(ap) < 0.1, "AP_borne");                 // reste O(eps), pas d'explosion
    chk(max_charge(ap) < 0.1, "AP_quasi_neutre");
    chk(exblew, "explicite_explose");
    chk(std::fabs(mass(ap.ne) - me0) < 1e-9 && std::fabs(mass(ap.ni) - mi0) < 1e-9,
        "masse_conservee");
  }

  // --- non raide : dispersion (le schema AP reste consistant) ---
  {
    const double wpe = 5, wpi = 1, cse2 = 1, csi2 = 0.04, k = 1.0;
    Grid g = make_grid(N, L, n0, wpe, wpi, cse2, csi2, eps);
    TwoFluidLinear mode{wpe, wpi, cse2 * k * k, csi2 * k * k};
    double wf, ws;
    mode.dispersion(wf, ws);
    const double dt = 0.5 / wf;  // resolu
    auto amp = [&]() {
      double s = 0;
      for (int i = 0; i < N; ++i) s += (g.ne[i] - n0) * std::cos(k * (i + 0.5) * g.dx);
      return 2.0 * s / N;
    };
    double t = 0, prev = amp(), tz = -1;
    for (int s = 0; s < 400; ++s) {
      ap_step(g, dt);
      t += dt;
      const double a = amp();
      if (tz < 0 && a < 0 && prev > 0) tz = (t - dt) + dt * prev / (prev - a);
      prev = a;
    }
    const double wm = (tz > 0) ? kPi / (2 * tz) : 0;
    std::printf("AP non raide : w_fast theorique=%.3f mesure=%.3f (ecart %.1f%%)\n",
                wf, wm, 100 * std::fabs(wm - wf) / wf);
    chk(tz > 0 && std::fabs(wm - wf) / wf < 0.15, "AP_dispersion_resolue");
  }

  if (fails == 0) std::printf("OK test_two_fluid_ap_spatial\n");
  return fails == 0 ? 0 : 1;
}
