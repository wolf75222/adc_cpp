// Moteur AMR unifie (revue, point 5) : on fait avancer une hierarchie AMR par l'ENTREE
// nommee advance_amr(m, hierarchy, dt) + le type LevelHierarchy, au lieu d'appeler une
// fonction amr_step_* dont le cas est encode dans le nom. On verifie (1) que la voie nommee
// donne EXACTEMENT le meme resultat que l'appel direct a amr_step_multilevel_multipatch
// (l'entree est une facade fidele), et (2) que le pas reste conservatif a travers cette API.

#include <adc/integrator/amr_reflux_mf.hpp>  // advance_amr, LevelHierarchy
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
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
  const double dxc = geom.dx(), dyc = geom.dy();
  DistributionMapping dm(1, 1), dm2(2, 1);
  BoxArray bac(std::vector<Box2D>{dom});

  Diocotron model;
  model.B0 = 1.0;
  model.n_i0 = 1.0;
  const double gx = 0.5, gy = -0.3;
  auto ne0 = [&](double x, double y) {
    return 1.0 + 0.3 * std::sin(2 * kPi * x) * std::cos(2 * kPi * y);
  };
  const double dt = 0.2 * dxc / std::hypot(gx, gy);

  const int CI0 = 8, CI1 = 23, CJ0 = 8, CJ1 = 23, CM = 15;
  Box2D left{{2 * CI0, 2 * CJ0}, {2 * CM + 1, 2 * CJ1 + 1}};
  Box2D right{{2 * (CM + 1), 2 * CJ0}, {2 * CI1 + 1, 2 * CJ1 + 1}};
  BoxArray baf(std::vector<Box2D>{left, right});

  auto fill = [&](MultiFab& U, double dx) {
    for (int li = 0; li < U.local_size(); ++li) {
      Array4 u = U.fab(li).array();
      const Box2D b = U.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i) u(i, j, 0) = ne0((i + 0.5) * dx, (j + 0.5) * dx);
    }
  };
  auto fill_aux = [&](MultiFab& a) {
    for (int li = 0; li < a.local_size(); ++li) {
      Array4 ar = a.fab(li).array();
      const Box2D g = a.fab(li).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) { ar(i, j, 0) = 0; ar(i, j, 1) = gx; ar(i, j, 2) = gy; }
    }
  };

  MultiFab axc(bac, dm, 3, 1), axf(baf, dm2, 3, 1);
  fill_aux(axc); fill_aux(axf);

  // --- voie DIRECTE : amr_step_multilevel_multipatch ---
  std::vector<AmrLevelMP> Ld(2);
  { MultiFab Uc(bac, dm, 1, 1), Uf(baf, dm2, 1, 1); fill(Uc, dxc); fill(Uf, dxc / 2);
    Ld[0] = {std::move(Uc), &axc, dxc, dyc};
    Ld[1] = {std::move(Uf), &axf, dxc / 2, dyc / 2}; }
  for (int s = 0; s < 15; ++s)
    amr_step_multilevel_multipatch<NoSlope, RusanovFlux>(model, Ld, dom, dt);

  // --- voie NOMMEE : advance_amr(m, LevelHierarchy, dt) ---
  LevelHierarchy h;
  h.base_dom = dom;
  h.base_per = Periodicity{true, true};
  { MultiFab Uc(bac, dm, 1, 1), Uf(baf, dm2, 1, 1); fill(Uc, dxc); fill(Uf, dxc / 2);
    h.levels.resize(2);
    h.levels[0] = {std::move(Uc), &axc, dxc, dyc};
    h.levels[1] = {std::move(Uf), &axf, dxc / 2, dyc / 2}; }
  const double m0 = sum(h.levels[0].U, 0);
  for (int s = 0; s < 15; ++s) advance_amr<NoSlope, RusanovFlux>(model, h, dt);
  const double mF = sum(h.levels[0].U, 0);

  double maxdiff = 0;
  const ConstArray4 ud = Ld[0].U.fab(0).const_array(), un = h.levels[0].U.fab(0).const_array();
  for (int j = 0; j < nc; ++j)
    for (int i = 0; i < nc; ++i)
      maxdiff = std::fmax(maxdiff, std::fabs(ud(i, j, 0) - un(i, j, 0)));
  std::printf("advance_amr vs amr_step direct : maxdiff=%.3e | derive masse=%.3e\n",
              maxdiff, std::fabs(mF - m0));

  chk(maxdiff == 0.0, "advance_amr_facade_fidele");   // facade : strictement identique
  chk(std::fabs(mF - m0) < 1e-10, "advance_amr_conservatif");

  // --- 3 niveaux : le niveau 1 multi-box joue parent ET enfant. Meme facade unifiee. ---
  // Niveau 2 = un patch interieur a la box gauche du niveau 1 ([16,31]x[16,47]).
  {
    BoxArray ba2(std::vector<Box2D>{Box2D{{40, 40}, {55, 55}}});
    MultiFab ax2(ba2, dm, 3, 1);
    fill_aux(ax2);
    auto build = [&](std::vector<AmrLevelMP>& L) {
      MultiFab U0(bac, dm, 1, 1), U1(baf, dm2, 1, 1), U2(ba2, dm, 1, 1);
      fill(U0, dxc); fill(U1, dxc / 2); fill(U2, dxc / 4);
      L.resize(3);
      L[0] = {std::move(U0), &axc, dxc, dyc};
      L[1] = {std::move(U1), &axf, dxc / 2, dyc / 2};
      L[2] = {std::move(U2), &ax2, dxc / 4, dyc / 4};
      mf_average_down_mb(L[2].U, L[1].U);  // sync init : niveau 1 <- niveau 2
      mf_average_down_mb(L[1].U, L[0].U);  // niveau 0 <- niveau 1
    };

    std::vector<AmrLevelMP> Ld3;
    build(Ld3);
    for (int s = 0; s < 12; ++s)
      amr_step_multilevel_multipatch<NoSlope, RusanovFlux>(model, Ld3, dom, dt);

    LevelHierarchy h3;
    h3.base_dom = dom;
    h3.base_per = Periodicity{true, true};
    build(h3.levels);
    const double m0_3 = sum(h3.levels[0].U, 0);
    for (int s = 0; s < 12; ++s) advance_amr<NoSlope, RusanovFlux>(model, h3, dt);
    const double mF_3 = sum(h3.levels[0].U, 0);

    double md3 = 0;
    const ConstArray4 d3 = Ld3[0].U.fab(0).const_array(),
                      n3 = h3.levels[0].U.fab(0).const_array();
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i)
        md3 = std::fmax(md3, std::fabs(d3(i, j, 0) - n3(i, j, 0)));
    std::printf("advance_amr 3 niveaux : maxdiff=%.3e | derive masse=%.3e\n", md3,
                std::fabs(mF_3 - m0_3));
    chk(md3 == 0.0, "advance_amr_3niv_facade");
    chk(std::fabs(mF_3 - m0_3) < 1e-10, "advance_amr_3niv_conservatif");
  }

  if (fails == 0) std::printf("OK test_advance_amr\n");
  return fails == 0 ? 0 : 1;
}
