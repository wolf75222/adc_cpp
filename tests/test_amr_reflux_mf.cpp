// Valide amr_step_2level_mf (pas AMR 2-niveaux sur MultiFab, via compute_face_fluxes)
// par EQUIVALENCE a amr_step_2level (Fab2D, deja teste conservatif dans
// test_amr_reflux) : meme CI, meme aux uniforme, meme modele scalaire Diocotron ->
// avec NoSlope/Rusanov les deux doivent donner le MEME etat grossier a l'arrondi.
// Comme la version Fab2D conserve a la machine, l'equivalence prouve que la version
// MultiFab conserve aussi.

#include <adc/integrator/amr_reflux.hpp>     // amr_step_2level (Fab2D, reference)
#include <adc/integrator/amr_reflux_mf.hpp>  // amr_step_2level_mf (MultiFab, candidat)
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
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

  const int nc = 16;
  Box2D dom = Box2D::from_extents(nc, nc);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const double dxc = geom.dx(), dyc = geom.dy();
  const int CI0 = 4, CI1 = 11, CJ0 = 4, CJ1 = 11;
  Box2D fbox{{2 * CI0, 2 * CJ0}, {2 * CI1 + 1, 2 * CJ1 + 1}};

  Diocotron model;
  model.B0 = 1.0;
  model.alpha = 1.0;
  model.n_i0 = 1.0;
  const double gx = 0.5, gy = -0.3;  // aux uniforme -> advection a vitesse constante
  auto ne0 = [&](double x, double y) {
    return 1.0 + 0.3 * std::sin(2 * kPi * x) * std::cos(2 * kPi * y);
  };
  const double dt = 0.2 * dxc / (std::hypot(gx, gy) + 1e-12);

  // --- reference Fab2D ---
  Fab2D Uc_f(dom, 1, 1), Uf_f(fbox, 1, 1), axc_f(dom, 3, 1), axf_f(fbox, 3, 1);
  {
    const Box2D g = axc_f.grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
        axc_f(i, j, 0) = 0; axc_f(i, j, 1) = gx; axc_f(i, j, 2) = gy;
      }
    const Box2D gf = axf_f.grown_box();
    for (int j = gf.lo[1]; j <= gf.hi[1]; ++j)
      for (int i = gf.lo[0]; i <= gf.hi[0]; ++i) {
        axf_f(i, j, 0) = 0; axf_f(i, j, 1) = gx; axf_f(i, j, 2) = gy;
      }
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) Uc_f(i, j) = ne0((i + 0.5) * dxc, (j + 0.5) * dyc);
    for (int j = fbox.lo[1]; j <= fbox.hi[1]; ++j)
      for (int i = fbox.lo[0]; i <= fbox.hi[0]; ++i)
        Uf_f(i, j) = ne0((i + 0.5) * dxc / 2, (j + 0.5) * dyc / 2);
  }
  amr_step_2level(model, Uc_f, dom, dxc, dyc, Uf_f, CI0, CI1, CJ0, CJ1, axc_f, axf_f, dt);

  // --- candidat MultiFab ---
  BoxArray bac(std::vector<Box2D>{dom}), baf(std::vector<Box2D>{fbox});
  DistributionMapping dm(1, n_ranks());
  MultiFab Uc_m(bac, dm, 1, 1), Uf_m(baf, dm, 1, 1), axc_m(bac, dm, 3, 1), axf_m(baf, dm, 3, 1);
  {
    Array4 ac = axc_m.fab(0).array(), af = axf_m.fab(0).array();
    const Box2D g = axc_m.fab(0).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) { ac(i, j, 0) = 0; ac(i, j, 1) = gx; ac(i, j, 2) = gy; }
    const Box2D gf = axf_m.fab(0).grown_box();
    for (int j = gf.lo[1]; j <= gf.hi[1]; ++j)
      for (int i = gf.lo[0]; i <= gf.hi[0]; ++i) { af(i, j, 0) = 0; af(i, j, 1) = gx; af(i, j, 2) = gy; }
    Array4 uc = Uc_m.fab(0).array(), uf = Uf_m.fab(0).array();
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) uc(i, j, 0) = ne0((i + 0.5) * dxc, (j + 0.5) * dyc);
    for (int j = fbox.lo[1]; j <= fbox.hi[1]; ++j)
      for (int i = fbox.lo[0]; i <= fbox.hi[0]; ++i)
        uf(i, j, 0) = ne0((i + 0.5) * dxc / 2, (j + 0.5) * dyc / 2);
  }
  amr_step_2level_mf<NoSlope, RusanovFlux>(model, Uc_m, dom, dxc, dyc, Uf_m, CI0, CI1,
                                           CJ0, CJ1, axc_m, axf_m, dt);

  // --- comparaison etat grossier ---
  double maxdiff = 0;
  const ConstArray4 um = Uc_m.fab(0).const_array();
  for (int j = 0; j < nc; ++j)
    for (int i = 0; i < nc; ++i)
      maxdiff = std::fmax(maxdiff, std::fabs(um(i, j, 0) - Uc_f(i, j)));
  std::printf("amr_step_2level_mf vs Fab2D : max|dUc| = %.3e\n", maxdiff);
  chk(maxdiff < 1e-12, "mf_equiv_fab2d");  // meme algorithme/flux -> arrondi

  if (fails == 0) std::printf("OK test_amr_reflux_mf\n");
  return fails == 0 ? 0 : 1;
}
