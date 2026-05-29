// Validation d'Euler-Poisson auto-gravitant (model/euler_poisson.hpp) via le
// Coupler<Model> existant (elliptic_rhs -> multigrille -> aux = grad phi ->
// assemble_rhs avec la source de gravite).
//
//   1. concept : EulerPoisson modele PhysicalModel.
//   2. theorie de Jeans (regime STABLE) : une perturbation acoustique-gravitationnelle
//      au repos oscille a omega = sqrt(c_s^2 k^2 - 4 pi G rho0). On mesure omega par
//      le premier passage a zero du mode et on compare a la theorie.
//   3. conservation : la gravite est interne -> quantite de mouvement totale ~ 0,
//      masse conservee.

#include <adc/core/physical_model.hpp>
#include <adc/coupling/coupler.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/euler_poisson.hpp>
#include <adc/operator/reconstruction.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

static_assert(PhysicalModel<EulerPoisson>, "EulerPoisson modele PhysicalModel");

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  const int N = 96;
  const double L = 1.0, rho0 = 1.0, p0 = 1.0, gamma = 5.0 / 3.0;
  const double four_pi_G = 20.0, eps = 1e-3;
  const double k = 2 * kPi / L;
  const double cs2 = gamma * p0 / rho0;
  const double omega_th = std::sqrt(cs2 * k * k - four_pi_G * rho0);  // regime stable

  Box2D dom = Box2D::from_extents(N, N);
  Geometry geom{dom, 0.0, L, 0.0, L};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, 1);

  EulerPoisson model;
  model.hydro.gamma = gamma;
  model.four_pi_G = four_pi_G;
  model.rho0 = rho0;

  BCRec bcU, bcPhi;  // periodique
  Coupler<EulerPoisson> cpl(model, geom, ba, bcU, bcPhi);

  // perturbation acoustique-gravitationnelle au repos : delta rho = eps rho0 cos(kx),
  // delta p = c_s^2 delta rho (adiabatique, pas de mode entropique), u = v = 0.
  MultiFab U(ba, dm, 4, 2);
  {
    Fab2D& f = U.fab(0);
    const Box2D v = U.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        const double drho = eps * rho0 * std::cos(k * geom.x_cell(i));
        const double rho = rho0 + drho;
        const double p = p0 + cs2 * drho;
        f(i, j, 0) = rho;
        f(i, j, 1) = 0.0;
        f(i, j, 2) = 0.0;
        f(i, j, 3) = p / (gamma - 1);  // E = p/(gamma-1), energie cinetique nulle
      }
  }

  auto mode_amp = [&]() {  // amplitude du mode cos(kx) de (rho - rho0)
    const ConstArray4 u = U.fab(0).const_array();
    const Box2D v = U.box(0);
    double m = 0;
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        m += (u(i, j, 0) - rho0) * std::cos(k * geom.x_cell(i));
    return 2.0 * m / (static_cast<double>(N) * N);
  };
  auto momentum = [&](int comp) {
    const ConstArray4 u = U.fab(0).const_array();
    const Box2D v = U.box(0);
    double s = 0;
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) s += u(i, j, comp);
    return s;
  };

  const double m0 = mode_amp(), mass0 = sum(U, 0);
  const double T = 0.35, cfl = 0.4;  // T > quart de periode (~0.23)
  const double dt = cfl * (L / N) / (std::sqrt(cs2) + 0.1);

  double t = 0, mprev = m0, tprev = 0, tzero = -1;
  while (t < T) {
    cpl.advance<Minmod>(U, dt);
    t += dt;
    const double m = mode_amp();
    if (tzero < 0 && m < 0 && mprev > 0)  // premier passage a zero du mode
      tzero = tprev + dt * mprev / (mprev - m);
    mprev = m;
    tprev = t;
  }

  const double omega_meas = (tzero > 0) ? kPi / (2 * tzero) : 0.0;
  const double rel = std::fabs(omega_meas - omega_th) / omega_th;
  const double dmom = std::fabs(momentum(1)) + std::fabs(momentum(2));
  const double dmass = std::fabs(sum(U, 0) - mass0);

  std::printf("Jeans stable : omega_th=%.4f omega_mesure=%.4f (ecart %.1f%%)\n",
              omega_th, omega_meas, 100 * rel);
  std::printf("conservation : |p_tot|=%.3e  dmasse=%.3e\n", dmom, dmass);

  chk(tzero > 0, "oscillation_detectee");
  chk(rel < 0.08, "frequence_de_Jeans");
  chk(dmom < 1e-9, "qte_mouvement_conservee");
  chk(dmass < 1e-9, "masse_conservee");

  if (fails == 0) std::printf("OK test_euler_poisson\n");
  return fails == 0 ? 0 : 1;
}
