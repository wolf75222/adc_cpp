// Propriete asymptotic-preserving (AP) de l'IMEX d'Euler sur une source raide de
// relaxation lineaire (modele jouet du regime raide Euler-Poisson : Debye -> 0,
// Lorentz, quasi-neutralite) :
//
//   du/dt = (u_eq - u) / eps           (transport nul : champ uniforme)
//   exact : u(t) = u_eq + (u0 - u_eq) e^{-t/eps}
//
// IMEX (implicite sur la source) : u^{n+1} = (u^n + (dt/eps) u_eq) / (1 + dt/eps).
//   - AP : eps -> 0 a dt FIXE -> u^{n+1} -> u_eq, stable, capture l'equilibre.
//   - explicite : u^{n+1} = u^n + (dt/eps)(u_eq - u^n), facteur |1 - dt/eps| >> 1
//     pour dt >> eps -> EXPLOSE.
// En regime non raide (eps ~ 1) l'IMEX est 1er ordre en dt.

#include <adc/integrator/imex.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/multifab.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

static MultiFab make_uniform(double u0) {
  Box2D dom = Box2D::from_extents(4, 4);
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, 1);
  MultiFab U(ba, dm, 1, 0);
  U.set_val(u0);
  return U;
}

static double value(const MultiFab& U) { return U.fab(0).const_array()(0, 0, 0); }

// transport nul (champ uniforme) : la partie explicite ne fait rien
static void no_transport(MultiFab&, Real) {}

static double run_imex(double eps, double u0, double u_eq, double dt, int n) {
  MultiFab U = make_uniform(u0);
  auto simpl = [=](MultiFab& V, Real h) {
    const Real c = h / eps;
    for (int li = 0; li < V.local_size(); ++li) {
      Array4 a = V.fab(li).array();
      const Box2D b = V.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          a(i, j, 0) = (a(i, j, 0) + c * u_eq) / (1 + c);  // solve implicite
    }
  };
  for (int s = 0; s < n; ++s) imex_euler_step(U, dt, no_transport, simpl);
  return value(U);
}

static double run_explicit(double eps, double u0, double u_eq, double dt, int n) {
  double u = u0;
  for (int s = 0; s < n; ++s) u = u + dt * (u_eq - u) / eps;
  return u;
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  const double u0 = 2.0, u_eq = 1.0;

  // --- AP : raide (eps << dt), IMEX stable et -> equilibre ---
  {
    const double eps = 1e-6, dt = 0.1;
    const double ui = run_imex(eps, u0, u_eq, dt, 20);
    const double ue = run_explicit(eps, u0, u_eq, dt, 5);
    std::printf("AP raide (eps=%.0e, dt=%.2f) : IMEX u=%.6f (u_eq=%.1f) | explicite |u|=%.3e\n",
                eps, dt, ui, u_eq, std::fabs(ue));
    chk(std::isfinite(ui) && std::fabs(ui - u_eq) < 1e-3, "imex_AP_vers_equilibre");
    chk(!std::isfinite(ue) || std::fabs(ue) > 1e3, "explicite_explose");
  }

  // --- non raide (eps = 1) : IMEX 1er ordre en dt ---
  {
    const double eps = 1.0, T = 1.0;
    const double exact = u_eq + (u0 - u_eq) * std::exp(-T / eps);
    const double e1 = std::fabs(run_imex(eps, u0, u_eq, T / 20, 20) - exact);
    const double e2 = std::fabs(run_imex(eps, u0, u_eq, T / 40, 40) - exact);
    const double order = std::log2(e1 / e2);
    std::printf("non raide : err(dt)=%.3e err(dt/2)=%.3e ordre=%.2f (exact=%.6f)\n",
                e1, e2, order, exact);
    chk(order > 0.8 && order < 1.3, "imex_ordre_1");
  }

  if (fails == 0) std::printf("OK test_imex_ap\n");
  return fails == 0 ? 0 : 1;
}
