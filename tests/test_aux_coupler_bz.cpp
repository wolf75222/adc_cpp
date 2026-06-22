// Chantier "Aux extensible", increment 2 : PEUPLEMENT d'un champ aux supplementaire cote
// coupleur. Le Coupler mono-bloc alloue le canal aux a la largeur DU MODELE (aux_comps) et
// remplit la composante B_z depuis une fonction B_z(x, y) fournie par l'utilisateur. On
// exerce le chemin COMPLET Poisson -> aux -> B_z -> source -> integration :
//   modele jouet n_aux=4, flux nul, elliptic_rhs nul (phi = 0), source S = B_z * u.
//   -> du/dt = B_z u ; B_z constant c -> u(t) = u0 * A^N avec A l'amplification SSPRK2.

#include <adc/core/model/physical_model.hpp>
#include <adc/core/state/state.hpp>
#include <adc/core/foundation/types.hpp>
#include <adc/coupling/single/coupler.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/layout/distribution_mapping.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/mesh/boundary/physical_bc.hpp>
#include <adc/numerics/fv/reconstruction.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <functional>

using namespace adc;

// Modele jouet : croissance pilotee par B_z. flux nul, pas de couplage elliptique (phi = 0).
struct BzGrow {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  static constexpr int n_aux = 4;  // phi, grad_x, grad_y, B_z
  ADC_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State& u, const Aux& a) const {
    State s{};
    s[0] = a.B_z * u[0];
    return s;
  }
  ADC_HD Real elliptic_rhs(const State&) const { return Real(0); }
};

static_assert(PhysicalModel<BzGrow>, "BzGrow modele PhysicalModel");
static_assert(aux_comps<BzGrow>() == 4, "BzGrow declare n_aux = 4");

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int n = 16;
  const Real L = 1.0, c = 0.5;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, L, 0.0, L};
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), n_ranks());
  BCRec bc;  // periodique (transport et Poisson)

  BzGrow model;
  // active vide ; B_z = constante c.
  Coupler<BzGrow> cpl(model, geom, ba, bc, bc, std::function<bool(Real, Real)>{},
                      [c](Real, Real) { return c; });

  MultiFab U(ba, dm, 1, 1);
  U.set_val(1.0);

  // --- (A) le canal aux est alloue a la largeur du modele (4) et B_z est peuple ---
  chk(cpl.aux().fab(0).ncomp() == 4, "aux_width_is_model_width");
  cpl.solve_fields(U);  // Poisson (f=0 -> phi=0) puis derive aux
  {
    double maxphi = 0, maxbz = 0;
    const MultiFab& A = cpl.aux();
    for (int li = 0; li < A.local_size(); ++li) {
      const ConstArray4 a = A.fab(li).const_array();
      const Box2D v = A.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
          maxphi = std::max(maxphi, std::fabs(a(i, j, 0)));    // phi = 0
          maxbz = std::max(maxbz, std::fabs(a(i, j, 3) - c));  // B_z = c
        }
    }
    std::printf("  (A) apres solve : max|phi|=%.2e  max|B_z - c|=%.2e\n", maxphi, maxbz);
    chk(maxphi < 1e-12, "phi_is_zero");
    chk(maxbz < 1e-14, "Bz_populated");
  }

  // --- (B) le chemin complet applique B_z : du/dt = c u -> u = A^N (amplification SSPRK2) ---
  const int N = 20;
  const Real dt = 0.01;
  for (int s = 0; s < N; ++s)
    cpl.advance(U, dt);  // SSPRK2 PerStage, NoSlope
  const double cdt = c * dt;
  const double A = 1.0 + cdt + 0.5 * cdt * cdt;  // SSPRK2 (Heun) sur u' = c u
  const double expected = std::pow(A, N);
  {
    double maxerr = 0, val = 0;
    for (int li = 0; li < U.local_size(); ++li) {
      const ConstArray4 u = U.fab(li).const_array();
      const Box2D v = U.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
          maxerr = std::max(maxerr, std::fabs(u(i, j, 0) - expected));
          val = u(i, j, 0);
        }
    }
    std::printf("  (B) apres %d pas : u=%.6f  attendu A^N=%.6f  err=%.2e\n", N, val, expected,
                maxerr);
    chk(maxerr < 1e-10, "Bz_drives_growth_ssprk2");
    chk(val > 1.05, "u_grew");  // c>0 -> croissance nette
  }

  if (fails == 0)
    std::printf("OK test_aux_coupler_bz\n");
  return fails == 0 ? 0 : 1;
}
