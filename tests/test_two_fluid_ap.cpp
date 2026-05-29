// Schema asymptotic-preserving deux-fluides, noyau 0D : mode de Langmuir
// (model/langmuir.hpp) integre par l'IMEX d'Euler (integrator/imex.hpp) sur un
// champ uniforme MultiFab<2> = (a, b). On valide :
//   1. AP : a omega_p tres grand (raide) et dt >> 1/omega_p, l'IMEX reste BORNE
//      (oscillation rapide amortie par le solve implicite A-stable) la ou un schema
//      explicite EXPLOSE (facteur sqrt(1+omega^2 dt^2) > 1).
//   2. consistance : en regime non raide, l'IMEX est 1er ordre en dt.
//
// C'est le terme raide (frequence plasma) du futur deux-fluides isotherme magnetique
// traite en implicite, avant le couplage spatial complet.

#include <adc/integrator/imex.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/model/langmuir.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;

static MultiFab make_mode(double a0, double b0) {
  Box2D dom = Box2D::from_extents(4, 4);
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, 1);
  MultiFab U(ba, dm, 2, 0);
  Array4 u = U.fab(0).array();
  const Box2D v = U.box(0);
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
      u(i, j, 0) = a0;
      u(i, j, 1) = b0;
    }
  return U;
}

// IMEX sur le mode ; renvoie max|a| sur le run et a final.
static void run_imex(const LangmuirMode& m, double dt, int n, double& maxa,
                     double& finala) {
  MultiFab U = make_mode(1.0, 0.0);
  auto apply = [&](MultiFab& V, Real h, bool implicit) {
    Array4 u = V.fab(0).array();
    const Box2D v = V.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        Real a = u(i, j, 0), b = u(i, j, 1);
        if (implicit) m.implicit_solve(a, b, h);
        else m.explicit_step(a, b, h);
        u(i, j, 0) = a;
        u(i, j, 1) = b;
      }
  };
  auto Texpl = [&](MultiFab& V, Real h) { apply(V, h, false); };
  auto Simpl = [&](MultiFab& V, Real h) { apply(V, h, true); };

  maxa = 1.0;
  for (int s = 0; s < n; ++s) {
    imex_euler_step(U, dt, Texpl, Simpl);
    const double a = U.fab(0).const_array()(0, 0, 0);
    maxa = std::fmax(maxa, std::fabs(a));
  }
  finala = U.fab(0).const_array()(0, 0, 0);
}

// Euler explicite sur le systeme complet a''=-omega^2 a ; renvoie max|a|.
static double run_explicit_maxa(const LangmuirMode& m, double dt, int n) {
  double a = 1, b = 0, maxa = 1;
  const double w2 = m.omega_p * m.omega_p + m.cs2k2;
  for (int s = 0; s < n; ++s) {
    const double an = a + dt * b, bn = b - dt * w2 * a;
    a = an;
    b = bn;
    maxa = std::fmax(maxa, std::fabs(a));
  }
  return maxa;
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  // --- 1. AP : raide (omega_p = 1e3), dt = 0.01 >> 1/omega_p ---
  {
    LangmuirMode m;
    m.omega_p = 1e3;
    m.cs2k2 = 1.0;
    const double dt = 0.01;
    double maxa, fa;
    run_imex(m, dt, 200, maxa, fa);
    const double maxe = run_explicit_maxa(m, dt, 200);
    std::printf("AP raide (omega_p=%.0e, dt*omega_p=%.0f) : IMEX max|a|=%.4f | "
                "explicite max|a|=%.3e\n",
                m.omega_p, dt * m.omega_p, maxa, maxe);
    chk(std::isfinite(fa) && maxa < 2.0, "imex_AP_borne");
    chk(!std::isfinite(maxe) || maxe > 1e6, "explicite_explose");
  }

  // --- 2. consistance (non raide) : ordre 1 en dt ---
  {
    LangmuirMode m;
    m.omega_p = 2.0;
    m.cs2k2 = 1.0;  // omega = sqrt(5)
    const double T = 0.5, w = m.omega();
    const double exact = std::cos(w * T);
    double maxa, a1, a2;
    run_imex(m, T / 40, 40, maxa, a1);
    run_imex(m, T / 80, 80, maxa, a2);
    const double e1 = std::fabs(a1 - exact), e2 = std::fabs(a2 - exact);
    const double order = std::log2(e1 / e2);
    std::printf("consistance : err(dt)=%.3e err(dt/2)=%.3e ordre=%.2f (exact=%.4f)\n",
                e1, e2, order, exact);
    chk(order > 0.7 && order < 1.4, "imex_ordre_1");
  }

  if (fails == 0) std::printf("OK test_two_fluid_ap\n");
  return fails == 0 ? 0 : 1;
}
