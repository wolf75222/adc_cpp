// Enveloppe de robustesse du schema deux-fluides AP en fonction de l'amplitude
// initiale. Le transport de quantite de mouvement (tfap_mstar) est Rusanov ordre 1
// et la continuite (tfap_div_update) est CENTREE (non upwindee) : adapte au regime
// lineaire AP, mais la continuite centree est dispersive et peut perdre la positivite
// a grande amplitude. Ce test CARTOGRAPHIE ou le schema tient, pour decider si une
// reconstruction MUSCL/limitee est reellement necessaire (et la valider plus tard).
//
// On balaye eps croissant a regime raide (dt*omega_pe = 5, stabilise) et on mesure :
// densite min (positivite), max|charge| (quasi-neutralite), derive de masse, finitude.

#include <adc/elliptic/poisson_fft_solver.hpp>
#include <adc/integrator/two_fluid_ap.hpp>
#include <adc/model/two_fluid_isothermal.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;
using Driver = TwoFluidAP2D<PoissonFFTSolver>;

static double mindens(const Driver& d) {
  double m = 1e300;
  const Fab2D& f = d.e.fab(0);
  for (int j = d.dom.lo[1]; j <= d.dom.hi[1]; ++j)
    for (int i = d.dom.lo[0]; i <= d.dom.hi[0]; ++i) m = std::fmin(m, f(i, j, 0));
  return m;
}
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
static bool allfinite(const Driver& d) {
  const Fab2D& f = d.e.fab(0);
  for (int j = d.dom.lo[1]; j <= d.dom.hi[1]; ++j)
    for (int i = d.dom.lo[0]; i <= d.dom.hi[0]; ++i)
      if (!std::isfinite(f(i, j, 0))) return false;
  return true;
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  const double eps_list[] = {1e-3, 1e-2, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8};
  const double dt = 5.0 / 1e3;
  std::printf("enveloppe AP 2D (n=64, dt*omega_pe=5, 300 pas, stabilise) :\n");
  std::printf("   eps    min(n_e)   max|dne|  max|charge|   d masse_e   etat\n");
  double last_positive_eps = 0;
  for (double eps : eps_list) {
    Driver d(64, 2 * kPi, 1.0, 0.04, 1e3, 20.0);
    d.init(eps);
    const double m0 = sum(d.e, 0);
    for (int t = 0; t < 300; ++t) d.step(dt, true);
    const bool fin = allfinite(d);
    const double mn = fin ? mindens(d) : -1.0;
    const double dev = fin ? maxdev(d) : INFINITY;
    const double chg = fin ? maxcharge(d) : INFINITY;
    const double dm = std::fabs(sum(d.e, 0) - m0);
    const char* etat = !fin ? "NON-FINI" : (mn <= 0 ? "n<=0" : "ok");
    std::printf("  %5.3f  %9.3e  %9.3e  %10.3e  %9.2e   %s\n", eps, mn, dev, chg, dm, etat);
    if (fin && mn > 0) last_positive_eps = eps;
    // garde de regression : tant que la solution reste finie, la masse est conservee
    // (le schema est conservatif par construction : div centree + flux Rusanov).
    if (fin) chk(dm < 1e-7, "masse_conservee_si_fini");
  }
  std::printf("amplitude max gardant n_e > 0 : eps = %.3f\n", last_positive_eps);
  // garde minimale : le regime lineaire/faiblement non-lineaire reste positif et borne.
  chk(last_positive_eps >= 0.1, "positif_jusqua_eps_0.1");

  // --- scenario 2 : transport d'une bosse acoustique (couplage faible ~= Euler) ---
  // Bosse gaussienne quasi-neutre (e == ion -> E ~= 0) lancant des ondes acoustiques.
  // La continuite CENTREE est un flux central pur ; la continuite UPWIND est Rusanov
  // a reconstruction MINMOD (MUSCL ordre 2, dissipation O(dx^2) en lisse). On joue le
  // MEME scenario dans les deux modes, a deux largeurs, et on lit honnetement :
  //  - bosse ETROITE : le sous-depassement (min n_e sous le fond) est presque IDENTIQUE
  //    centre vs MUSCL -> il est surtout PHYSIQUE (rarefaction acoustique), pas du Gibbs
  //    numerique. (Le 1er ordre Rusanov le reduisait artificiellement par sur-diffusion.)
  //  - bosse LARGE (lisse, bien resolue) : MUSCL preserve le pic comme le centre (pas de
  //    sur-diffusion) -> c'est un schema upwind ordre 2 PRECIS, l'apport reel ici.
  auto run_bump = [&](bool upwind, double wcell, double& minover, double& endpeak,
                      double& dm, bool& fin) {
    Driver d(96, 2 * kPi, 1.0, 1.0, 2.0, 0.4);  // couplage faible
    d.upwind_continuity = upwind;
    d.e.set_val(0); d.ion.set_val(0);
    Array4 ae = d.e.fab(0).array(), ai = d.ion.fab(0).array();
    const double xc = kPi, yc = kPi, w = wcell * (2 * kPi / 96);
    for (int j = d.dom.lo[1]; j <= d.dom.hi[1]; ++j)
      for (int i = d.dom.lo[0]; i <= d.dom.hi[0]; ++i) {
        const double x = d.geom.x_cell(i), y = d.geom.y_cell(j);
        const double r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc);
        const double bump = 1.0 + 1.0 * std::exp(-r2 / (w * w));  // pic n = 2
        ae(i, j, 0) = bump; ai(i, j, 0) = bump;
      }
    const double m0 = sum(d.e, 0);
    const double dt = 0.3 * (2 * kPi / 96) / std::sqrt(1.0);  // CFL acoustique
    minover = 1e300;
    for (int t = 0; t < 200; ++t) {
      d.step(dt, true);
      minover = std::fmin(minover, mindens(d));
    }
    endpeak = 1.0 + maxdev(d);  // densite max finale (= pic restant)
    dm = std::fabs(sum(d.e, 0) - m0);
    fin = allfinite(d);
  };
  {
    double minC, pkC, dmC, minU, pkU, dmU; bool finC, finU;
    run_bump(false, 6.0, minC, pkC, dmC, finC);  // front raide
    run_bump(true, 6.0, minU, pkU, dmU, finU);
    const double underC = 1.0 - minC, underU = 1.0 - minU;
    std::printf("front raide (bosse etroite) : sous-depassement CENTRE=%.1f%% MUSCL=%.1f%% "
                "(quasi identiques -> physique, pas Gibbs) masse %.0e/%.0e\n",
                100 * underC, 100 * underU, dmC, dmU);
    chk(finC && finU, "front_raide_fini");
    chk(dmC < 1e-7 && dmU < 1e-7, "front_raide_masse_conservee");
    // MUSCL precis : n'aggrave PAS le sous-depassement (il l'egale ; les deux schemas
    // precis s'accordent sur la rarefaction physique). marge robuste cross-plateforme.
    chk(underU < underC + 0.01, "muscl_naggrave_pas_le_front");

    double minS, pkS, dmS, minM, pkM, dmM; bool finS, finM;
    run_bump(false, 16.0, minS, pkS, dmS, finS);  // bosse lisse, bien resolue
    run_bump(true, 16.0, minM, pkM, dmM, finM);
    std::printf("bosse lisse (bien resolue) : pic final CENTRE=%.4f MUSCL=%.4f "
                "(perte de pic MUSCL=%.1f%%) -> ordre 2, pas de sur-diffusion\n",
                pkS, pkM, 100 * (pkS - pkM) / (pkS - 1.0));
    // MUSCL ordre 2 : pas de sur-diffusion -> conserve l'essentiel du pic du schema centre.
    chk(pkM > 1.0 + 0.95 * (pkS - 1.0), "muscl_pas_de_sur_diffusion");
  }

  if (fails == 0) std::printf("OK test_two_fluid_ap_amplitude\n");
  return fails == 0 ? 0 : 1;
}
