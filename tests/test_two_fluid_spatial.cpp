// Deux-fluides isotherme SPATIAL 1D (non lineaire) : transport conservatif de CHAQUE
// espece (Euler isotherme, flux de Rusanov) + Poisson 1D periodique pour le champ +
// force de Lorentz. Premier solveur reellement spatial multi-especes (sort du mode
// de Fourier). Explicite (CFL plasma : dt < 1/omega_pe) ; l'AP implicite + 2D +
// MultiFab restent l'etape suivante.
//
// Validation :
//   1. conservation : masse de chaque espece conservee (continuite conservative).
//   2. dispersion : l'oscillation plasma reproduit la branche de Langmuir
//       omega_fast (TwoFluidLinear), mesuree sur la grille.
//   3. positivite / finitude.
//
// Normalisation : kappa = e/eps0 = 1, n0 = 1 -> coup_s = omega_ps^2,
// d_x E = (n_i - n_e), accel_s = z_s coup_s E (z_e = -1, z_i = +1). La linearisation
// redonne exactement Ä = K A de TwoFluidLinear.

#include <adc/model/two_fluid_isothermal.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

struct Species {
  double c2;    // vitesse du son isotherme au carre
  double coup;  // omega_ps^2 (constante de couplage, n0 = kappa = 1)
  double z;     // signe de charge (electron -1, ion +1)
  std::vector<double> n, m;  // densite, quantite de mouvement n*u (cellules)
};

// RHS de transport -div F (Euler isotherme, Rusanov) pour une espece, periodique.
static void transport_rhs(const std::vector<double>& n, const std::vector<double>& m,
                          double c2, double dx, std::vector<double>& dn,
                          std::vector<double>& dm) {
  const int N = static_cast<int>(n.size());
  const double cs = std::sqrt(c2);
  auto Fn = [&](int i) { return m[i]; };
  auto Fm = [&](int i) { return m[i] * m[i] / n[i] + c2 * n[i]; };
  auto spd = [&](int i) { return std::fabs(m[i] / n[i]) + cs; };
  std::vector<double> fn(N), fm(N);  // flux a l'interface i+1/2
  for (int i = 0; i < N; ++i) {
    const int ip = (i + 1) % N;
    const double a = std::fmax(spd(i), spd(ip));
    fn[i] = 0.5 * (Fn(i) + Fn(ip)) - 0.5 * a * (n[ip] - n[i]);
    fm[i] = 0.5 * (Fm(i) + Fm(ip)) - 0.5 * a * (m[ip] - m[i]);
  }
  for (int i = 0; i < N; ++i) {
    const int im = (i - 1 + N) % N;
    dn[i] = -(fn[i] - fn[im]) / dx;
    dm[i] = -(fm[i] - fm[im]) / dx;
  }
}

// Champ E tel que d_x E = g (g de moyenne nulle), 1D periodique : integrale
// cumulative recentree (E de moyenne nulle = jauge physique).
static void solve_field(const std::vector<double>& g, double dx,
                        std::vector<double>& E) {
  const int N = static_cast<int>(g.size());
  double acc = 0, mean = 0;
  for (int i = 0; i < N; ++i) {
    E[i] = acc + 0.5 * dx * g[i];  // valeur au centre i
    acc += dx * g[i];
    mean += E[i];
  }
  mean /= N;
  for (int i = 0; i < N; ++i) E[i] -= mean;
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  const int N = 128;
  const double L = 2 * kPi, dx = L / N, k = 2 * kPi / L;  // k = 1 (fondamental)
  const double n0 = 1.0, wpe = 5.0, wpi = 1.0, cse2 = 1.0, csi2 = 0.04, eps = 1e-3;

  Species e{cse2, wpe * wpe, -1.0, std::vector<double>(N), std::vector<double>(N)};
  Species ion{csi2, wpi * wpi, +1.0, std::vector<double>(N), std::vector<double>(N)};
  for (int i = 0; i < N; ++i) {
    const double x = (i + 0.5) * dx;
    e.n[i] = n0 * (1 + eps * std::cos(k * x));  // electrons perturbes
    ion.n[i] = n0;
    e.m[i] = ion.m[i] = 0;
  }

  // dispersion theorique (TwoFluidLinear) au mode k
  TwoFluidLinear mode;
  mode.omega_pe = wpe;
  mode.omega_pi = wpi;
  mode.cse2k2 = cse2 * k * k;
  mode.csi2k2 = csi2 * k * k;
  double wf, ws;
  mode.dispersion(wf, ws);

  auto mode_amp = [&]() {  // amplitude du mode cos(kx) de (ne - n0)
    double s = 0;
    for (int i = 0; i < N; ++i) s += (e.n[i] - n0) * std::cos(k * (i + 0.5) * dx);
    return 2.0 * s / N;
  };
  auto total = [&](const std::vector<double>& a) {
    double s = 0;
    for (double v : a) s += v;
    return s;
  };
  const double me0 = total(e.n), mi0 = total(ion.n), amp0 = mode_amp();

  const double dt = 0.4 * (L / N) / (std::sqrt(cse2) + 0.3);  // CFL transport
  // (dt ~ 0.0085 << 1/wpe = 0.2 : plasma resolu)
  const double T = 0.6;
  std::vector<double> g(N), E(N), dne(N), dme(N), dni(N), dmi(N);
  double t = 0, prev = amp0, tzero = -1;
  bool finite = true, positive = true;
  while (t < T) {
    for (int i = 0; i < N; ++i) g[i] = ion.n[i] - e.n[i];
    solve_field(g, dx, E);
    transport_rhs(e.n, e.m, e.c2, dx, dne, dme);
    transport_rhs(ion.n, ion.m, ion.c2, dx, dni, dmi);
    for (int i = 0; i < N; ++i) {
      e.n[i] += dt * dne[i];
      e.m[i] += dt * dme[i];
      ion.n[i] += dt * dni[i];
      ion.m[i] += dt * dmi[i];
      e.m[i] += dt * e.z * e.coup * e.n[i] * E[i];      // Lorentz electron
      ion.m[i] += dt * ion.z * ion.coup * ion.n[i] * E[i];  // Lorentz ion
      if (!std::isfinite(e.n[i]) || !std::isfinite(e.m[i])) finite = false;
      if (e.n[i] <= 0 || ion.n[i] <= 0) positive = false;
    }
    t += dt;
    const double a = mode_amp();
    if (tzero < 0 && a < 0 && prev > 0) tzero = (t - dt) + dt * prev / (prev - a);
    prev = a;
  }

  const double w_meas = (tzero > 0) ? kPi / (2 * tzero) : 0.0;
  std::printf("dispersion grille : w_fast theorique=%.3f mesure=%.3f (ecart %.1f%%) "
              "[w_slow=%.3f]\n",
              wf, w_meas, 100 * std::fabs(w_meas - wf) / wf, ws);
  std::printf("conservation : d(masse_e)=%.2e d(masse_i)=%.2e  positivite=%d\n",
              std::fabs(total(e.n) - me0), std::fabs(total(ion.n) - mi0), positive);

  chk(finite && positive, "stable_positif");
  chk(std::fabs(total(e.n) - me0) < 1e-9 && std::fabs(total(ion.n) - mi0) < 1e-9,
      "masse_conservee_par_espece");
  chk(tzero > 0 && std::fabs(w_meas - wf) / wf < 0.10, "dispersion_langmuir_sur_grille");

  if (fails == 0) std::printf("OK test_two_fluid_spatial\n");
  return fails == 0 ? 0 : 1;
}
