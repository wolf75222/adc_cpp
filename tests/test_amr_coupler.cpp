// Test de caracterisation du AmrCoupler (le coupleur AMR de production, ce qui tourne
// dans examples/diocotron_amr.cpp), desormais PORTE sur la pile MultiFab + le seam
// (AmrLevelMF + amr_step_multilevel_mf + compute_face_fluxes). Fige son comportement :
// conservation de la masse sur la hierarchie 2 niveaux (reflux coarse-fine a l'arrondi),
// finitude, couplage Poisson effectif. Le meme invariant que la version Fab2D.

#include <adc/coupling/amr_coupler.hpp>       // AmrCoupler + AmrLevelMF + mf_average_down
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/diocotron.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  const int nc = 32;
  Box2D dom = Box2D::from_extents(nc, nc);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const double dxc = geom.dx(), dyc = geom.dy(), dxf = dxc / 2, dyf = dyc / 2;
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, n_ranks());

  Diocotron model;
  model.B0 = 1.0;
  model.alpha = 1.0;
  model.n_i0 = 1.0;
  auto ne0 = [&](double x, double y) {
    return 1.0 + 0.3 * std::sin(2 * kPi * x) * std::sin(2 * kPi * y);
  };

  const int CI0 = nc / 4, CI1 = 3 * nc / 4 - 1, CJ0 = nc / 4, CJ1 = 3 * nc / 4 - 1;
  Box2D fbox{{2 * CI0, 2 * CJ0}, {2 * CI1 + 1, 2 * CJ1 + 1}};

  MultiFab Uc(ba, dm, 1, 1), Uf(BoxArray(std::vector<Box2D>{fbox}), dm, 1, 1);
  {
    Array4 uc = Uc.fab(0).array();
    const Box2D gc = Uc.fab(0).grown_box();
    for (int j = gc.lo[1]; j <= gc.hi[1]; ++j)
      for (int i = gc.lo[0]; i <= gc.hi[0]; ++i)
        uc(i, j, 0) = ne0((i + 0.5) * dxc, (j + 0.5) * dyc);
    Array4 uf = Uf.fab(0).array();
    for (int j = fbox.lo[1]; j <= fbox.hi[1]; ++j)
      for (int i = fbox.lo[0]; i <= fbox.hi[0]; ++i)
        uf(i, j, 0) = ne0((i + 0.5) * dxf, (j + 0.5) * dyf);
  }
  mf_average_down(Uf, Uc, CI0, CI1, CJ0, CJ1);

  std::vector<AmrLevelMF> L0;
  L0.push_back({std::move(Uc), nullptr, dxc, dyc, CI0, CI1, CJ0, CJ1, true});
  L0.push_back({std::move(Uf), nullptr, dxf, dyf, 0, 0, 0, 0, false});

  BCRec bc;  // periodique
  AmrCoupler<Diocotron> sim(model, geom, ba, bc, std::move(L0));

  sim.update();
  const double v0 = sim.max_drift_speed();
  const double m0 = sim.mass();
  const double dt = 0.4 * dxc / v0;

  bool finite = true;
  for (int s = 0; s < 20; ++s) {
    sim.step(dt);
    if (!std::isfinite(sim.mass())) finite = false;
  }
  const double m1 = sim.mass();
  std::printf("AmrCoupler 2 niveaux (MultiFab) : v_derive=%.3e masse0=%.10e masse=%.10e "
              "dmasse=%.2e\n", v0, m0, m1, std::fabs(m1 - m0));

  chk(v0 > 1e-6, "couplage_poisson_effectif");
  chk(finite && std::isfinite(m1), "stable_fini");
  chk(std::fabs(m1 - m0) < 1e-9, "masse_conservee_hierarchie");

  if (fails == 0) std::printf("OK test_amr_coupler\n");
  return fails == 0 ? 0 : 1;
}
