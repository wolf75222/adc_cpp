// Rotation cyclotron du deux-fluides (champ magnetique uniforme hors-plan B_z).
// Test analytique isolant l'operateur magnetique : plasma UNIFORME (n_e = n_i = 1 ->
// charge nulle -> E = 0) avec une quantite de mouvement electronique uniforme. Le
// transport est inerte (gradients nuls) et le champ est nul, donc la SEULE dynamique
// est la rotation de Lorentz : (m_x, m_y) doit tourner a la frequence cyclotron wce,
// en conservant |m| (rotation) et la masse. On mesure wce par le premier passage a zero
// de m_x ~ m0 cos(wce t) et on verifie l'ecart a la theorie.

#include <adc/elliptic/poisson_fft_solver.hpp>
#include <adc/integrator/two_fluid_ap.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;
using Driver = TwoFluidAP2D<PoissonFFTSolver>;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  Driver d(32, 2 * kPi, 1.0, 1.0, 0.0, 0.0);  // wpe = wpi = 0 : pas de couplage (E = 0)
  const double m0 = 0.3;
  d.e.set_val(0); d.ion.set_val(0);
  {  // etat uniforme : n = 1 partout, electrons (mx = m0, my = 0), ions au repos
    Array4 ae = d.e.fab(0).array(), ai = d.ion.fab(0).array();
    const Box2D g = d.e.fab(0).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
        ae(i, j, 0) = 1.0; ae(i, j, 1) = m0; ae(i, j, 2) = 0.0;
        ai(i, j, 0) = 1.0; ai(i, j, 1) = 0.0; ai(i, j, 2) = 0.0;
      }
  }
  const double wce = 5.0;
  d.wce = wce; d.wci = 0.0;
  const double dt = 0.01;

  const int i0 = d.dom.lo[0], j0 = d.dom.lo[1];  // cellule temoin (champ uniforme)
  auto emx = [&]() { return d.e.fab(0)(i0, j0, 1); };
  auto emy = [&]() { return d.e.fab(0)(i0, j0, 2); };

  const double m0e = sum(d.e, 0);
  double t = 0, prev = emx(), tz = -1, norm_err = 0, dev = 0;
  for (int it = 0; it < 400 && tz < 0; ++it) {
    d.step(dt, true);
    t += dt;
    const double mx = emx(), my = emy();
    norm_err = std::fmax(norm_err, std::fabs(std::sqrt(mx * mx + my * my) - m0));
    dev = std::fmax(dev, std::fabs(d.e.fab(0)(i0, j0, 0) - 1.0));  // n_e doit rester 1
    if (mx < 0 && prev > 0) tz = (t - dt) + dt * prev / (prev - mx);  // m_x ~ cos(wce t)
    prev = mx;
  }
  const double wmeas = (tz > 0) ? kPi / (2 * tz) : 0;  // 1er zero a wce*t = pi/2
  const double dm = std::fabs(sum(d.e, 0) - m0e);
  std::printf("cyclotron : wce theorique=%.3f mesure=%.3f (ecart %.2f%%) |m| err=%.2e "
              "n_e dev=%.2e d masse=%.2e\n",
              wce, wmeas, 100 * std::fabs(wmeas - wce) / wce, norm_err, dev, dm);
  chk(tz > 0 && std::fabs(wmeas - wce) / wce < 0.02, "cyclotron_frequence");
  chk(norm_err < 1e-9, "cyclotron_norme_preservee");  // la rotation conserve |m|
  chk(dev < 1e-12, "cyclotron_densite_inerte");        // E = 0 -> n inchange
  chk(dm < 1e-9, "cyclotron_masse_conservee");

  if (fails == 0) std::printf("OK test_two_fluid_cyclotron\n");
  return fails == 0 ? 0 : 1;
}
