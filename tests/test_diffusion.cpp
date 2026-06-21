// Terme parabolique du coeur : un modele qui declare diffusivity() recoit +nu Lap(U)
// dans assemble_rhs (la diffusion "comme un flux de plus"). On valide l'equation de la
// chaleur pure (flux nul) : un mode cos(kx) decroit a exp(-lambda t) avec lambda le
// taux du laplacien DISCRET. Modele jouet inline (le coeur ne connait aucune physique).

#include <adc/core/physical_model.hpp>
#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/numerics/time/ssprk.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/spatial_operator.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

// Chaleur pure : aucun flux hyperbolique, seule la diffusivite agit.
struct Heat {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  Real nu = 0.0;
  ADC_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State&, const Aux&) const { return State{Real(0)}; }
  ADC_HD Real elliptic_rhs(const State&) const { return Real(0); }
  ADC_HD Real diffusivity() const { return nu; }
};

static_assert(PhysicalModel<Heat>, "Heat modele PhysicalModel");
static_assert(DiffusiveModel<Heat>, "Heat est diffusif");

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int n = 48;
  const double L = 1.0, eps = 1e-3, nu = 0.05, k = 2 * kPi / L;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, L, 0.0, L};
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), n_ranks());
  BCRec bc;  // periodique
  const double dx = geom.dx();

  MultiFab U(ba, dm, 1, 1), aux(ba, dm, 3, 1);
  aux.set_val(0.0);  // flux ignore aux ; alloue pour load_aux

  auto amp = [&]() {  // amplitude du mode cos(kx)
    const ConstArray4 u = U.fab(0).const_array();
    double m = 0;
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        m += (u(i, j, 0) - 1.0) * std::cos(k * geom.x_cell(i));
    return 2.0 * m / (double(n) * n);
  };
  auto init = [&]() {
    Array4 a = U.fab(0).array();
    const double e = eps, kk = k;
    for_each_cell(dom, [a, geom, kk, e](int i, int j) {
      a(i, j, 0) = 1.0 + e * std::cos(kk * geom.x_cell(i));
    });
  };

  // --- nu = 0 : controle, l'etat ne bouge pas ---
  {
    Heat m;
    m.nu = 0.0;
    init();
    const double a0 = amp(), mass0 = sum(U);
    for (int s = 0; s < 50; ++s)
      advance_ssprk2(m, U, aux, geom, bc, 1e-3);
    chk(std::fabs(amp() - a0) < 1e-12, "nu0_static");
    chk(std::fabs(sum(U) - mass0) < 1e-10, "nu0_mass");
  }

  // --- nu > 0 : le mode decroit a exp(-lambda t), lambda = nu*(2-2cos(k dx))/dx^2 ---
  {
    Heat m;
    m.nu = nu;
    init();
    const double a0 = amp(), mass0 = sum(U);
    const double dt = 1e-3;
    const int K = 300;
    for (int s = 0; s < K; ++s)
      advance_ssprk2(m, U, aux, geom, bc, dt);
    const double t = K * dt;
    const double lambda = nu * (2.0 - 2.0 * std::cos(k * dx)) / (dx * dx);
    const double a_th = a0 * std::exp(-lambda * t);
    const double rel = std::fabs(amp() - a_th) / std::fabs(a_th);
    std::printf("  diffusion: A/A0=%.4f  theorie=%.4f  err=%.2e\n", amp() / a0, a_th / a0, rel);
    chk(rel < 0.02, "heat_decay_matches_theory");
    chk(amp() < 0.7 * a0, "heat_decays");
    chk(std::fabs(sum(U) - mass0) < 1e-9, "heat_mass_conserved");
  }

  if (fails == 0)
    std::printf("OK test_diffusion\n");
  return fails == 0 ? 0 : 1;
}
