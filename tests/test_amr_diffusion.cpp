// Diffusion sur AMR comme FLUX de face (jalon 4) : un DiffusiveModel ajoute le flux
// Fickien -nu grad u dans compute_face_fluxes. Vu par le reflux, il reste donc
// CONSERVATIF aux interfaces coarse-fine (un Laplacien direct, hors flux, serait
// ignore par le reflux et fuirait de la masse a l'interface). On verifie : (1) la
// masse totale (grossier average-down) est conservee a travers un pas AMR diffusif,
// (2) la diffusion AGIT (le champ se lisse, le pic baisse).

#include <adc/core/state/state.hpp>
#include <adc/numerics/time/amr_reflux_mf.hpp>  // AmrLevelMP, advance_amr, mf_average_down_mb
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;

// Scalaire purement diffusif : pas de flux hyperbolique, pas de source ; seule la
// diffusivite agit (via le flux de face Fickien de compute_face_fluxes).
struct DiffuseScalar {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  Real nu = Real(0.1);
  ADC_HD State flux(const State&, const Aux&, int) const { return State{}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State&, const Aux&) const { return State{}; }
  ADC_HD Real elliptic_rhs(const State& u) const { return u[0]; }
  ADC_HD Real diffusivity() const { return nu; }
};

static_assert(DiffusiveModel<DiffuseScalar>);

template <class F>
static void fill(MultiFab& U, Real dx, F f) {
  Array4 a = U.fab(0).array();
  const Box2D g = U.fab(0).grown_box();
  for (int j = g.lo[1]; j <= g.hi[1]; ++j)
    for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
      const Real x = (Real(i) + Real(0.5)) * dx, y = (Real(j) + Real(0.5)) * dx;
      a(i, j, 0) = f(x, y);
    }
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int NC = 16;
  const Box2D dom = Box2D::from_extents(NC, NC);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const BoxArray ba_c(std::vector<Box2D>{dom});
  const Box2D fbox{{8, 8}, {23, 23}};
  const BoxArray ba_f(std::vector<Box2D>{fbox});
  const DistributionMapping dm(1, n_ranks());
  const Real dxc = geom.dx(), dxf = dxc / 2;

  MultiFab Uc(ba_c, dm, 1, 2), Uf(ba_f, dm, 1, 2);
  MultiFab auxc(ba_c, dm, 3, 1), auxf(ba_f, dm, 3, 1);
  auxc.set_val(Real(0));
  auxf.set_val(Real(0));

  // champ lisse non uniforme : 1 + 0.5 cos(2 pi x) cos(2 pi y) (la diffusion l'aplatit).
  auto f = [](Real x, Real y) {
    return Real(1) +
           Real(0.5) * std::cos(Real(2) * Real(M_PI) * x) * std::cos(Real(2) * Real(M_PI) * y);
  };
  fill(Uc, dxc, f);
  fill(Uf, dxf, f);

  std::vector<AmrLevelMP> levels;
  levels.push_back(AmrLevelMP{std::move(Uc), &auxc, dxc, dxc});
  levels.push_back(AmrLevelMP{std::move(Uf), &auxf, dxf, dxf});

  DiffuseScalar model{Real(0.1)};

  // rend le grossier coherent (couvert = moyenne du fin) avant de mesurer la masse.
  mf_average_down_mb(levels[1].U, levels[0].U);
  const Real M0 = sum(levels[0].U, 0);
  const Real peak0 = norm_inf(levels[0].U);

  const Real dt = Real(0.001);
  for (int s = 0; s < 5; ++s)
    advance_amr<NoSlope, RusanovFlux>(model, levels, dom, dt, Periodicity{true, true});

  const Real M1 = sum(levels[0].U, 0);
  const Real peak1 = norm_inf(levels[0].U);

  // conservation : la diffusion en flux de face conserve l'integrale, AUSSI a travers
  // l'interface coarse-fine (c'est ce que le reflux garantit).
  chk(std::fabs(M1 - M0) < Real(1e-12), "amr_diffusion_mass_conserved");
  // la diffusion a bien agi : amplitude reduite (pic plus proche de la moyenne 1).
  chk(peak1 < peak0 - Real(1e-6), "amr_diffusion_smooths");

  if (fails == 0)
    std::printf("OK test_amr_diffusion\n");
  return fails == 0 ? 0 : 1;
}
