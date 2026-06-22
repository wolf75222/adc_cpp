// Limite asymptotic-preserving QUANTIFIEE : on balaie la raideur eps sur 8 decades a pas de
// temps FIXE et on montre que l'IMEX reste UNIFORMEMENT borne (stabilite independante de la
// raideur) et capture de mieux en mieux l'equilibre quand eps -> 0. C'est la propriete qui
// definit un schema AP, au-dela d'une seule valeur de eps (cf. test_imex_ap).
//
//   du/dt = (u_eq - u)/eps ,  IMEX implicite : u^{n+1} = (u^n + (dt/eps) u_eq)/(1 + dt/eps).
// Erreur apres n pas : |u_n - u_eq| = |u0 - u_eq| / (1 + dt/eps)^n  -> 0 quand eps -> 0.
// L'explicite, lui, a un facteur |1 - dt/eps| >> 1 des que dt >> eps : il explose.

#include <adc/numerics/time/schemes/imex.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/layout/distribution_mapping.hpp>
#include <adc/mesh/storage/multifab.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;

static void no_transport(MultiFab&, Real) {}

// n pas d'IMEX implicite sur la relaxation, renvoie |u_n - u_eq|.
static double imex_err(double eps, double u0, double u_eq, double dt, int n) {
  Box2D dom = Box2D::from_extents(4, 4);
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, 1);
  MultiFab U(ba, dm, 1, 0);
  U.set_val(u0);
  auto simpl = [=](MultiFab& V, Real h) {
    const Real c = h / eps;
    for (int li = 0; li < V.local_size(); ++li) {
      Array4 a = V.fab(li).array();
      const Box2D b = V.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          a(i, j, 0) = (a(i, j, 0) + c * u_eq) / (1 + c);
    }
  };
  for (int s = 0; s < n; ++s)
    imex_euler_step(U, dt, no_transport, simpl);
  return std::fabs(U.fab(0).const_array()(0, 0, 0) - u_eq);
}

// explicite naif : facteur d'amplification |1 - dt/eps| par pas.
static double explicit_val(double eps, double u0, double u_eq, double dt, int n) {
  double u = u0;
  for (int s = 0; s < n; ++s)
    u = u + dt * (u_eq - u) / eps;
  return u;
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const double u0 = 2.0, u_eq = 1.0, dt = 0.1;
  const int n = 50;
  const std::vector<double> epss = {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8};

  double prev = 1e30, worst = 0;
  bool monotone = true;
  std::printf("limite AP (dt=%.2f fixe, %d pas) :\n", dt, n);
  for (double eps : epss) {
    const double e = imex_err(eps, u0, u_eq, dt, n);
    std::printf("  eps=%.0e (dt/eps=%.0e) : |u-u_eq|=%.3e\n", eps, dt / eps, e);
    worst = std::fmax(worst, e);
    if (!std::isfinite(e))
      monotone = false;
    if (e > prev + 1e-14)
      monotone = false;  // erreur non croissante quand eps decroit
    prev = e;
  }
  const double e_stiff = imex_err(1e-8, u0, u_eq, dt, n);
  const double e_expl = explicit_val(1e-6, u0, u_eq, dt, 10);

  // stabilite UNIFORME : borne independante de la raideur (jamais > |u0 - u_eq|).
  chk(std::isfinite(worst) && worst <= std::fabs(u0 - u_eq) + 1e-12, "AP_borne_uniforme");
  // capture de l'equilibre dans la limite raide.
  chk(e_stiff < 1e-6, "AP_capture_equilibre");
  // l'erreur ne croit pas quand on durcit (de plus en plus AP).
  chk(monotone, "AP_monotone_en_raideur");
  // contraste : l'explicite explose au meme regime.
  chk(!std::isfinite(e_expl) || std::fabs(e_expl) > 1e3, "explicite_explose");

  if (fails == 0)
    std::printf("OK test_ap_limit\n");
  return fails == 0 ? 0 : 1;
}
