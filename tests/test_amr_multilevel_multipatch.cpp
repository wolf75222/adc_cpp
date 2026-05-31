// Valide amr_step_multilevel_multipatch (recursion AMR N-niveaux, MULTI-BOX a chaque
// niveau) par trois gardes :
//   1. 3 niveaux mono-box -> BIT A BIT identique a amr_step_multilevel_mf (reference
//      mono-box deja validee vs Fab2D). Prouve que la generalisation se reduit au cas
//      mono-box.
//   2. 2 niveaux, niveau fin a 2 boxes -> BIT A BIT identique a amr_step_2level_multipatch
//      (reference multipatch deja validee). Prouve le reflux coverage-aware multi-box.
//   3. 3 niveaux avec un niveau intermediaire MULTI-BOX (capacite reellement nouvelle :
//      reflux route vers la box parente + sous-cyclage recursif) -> masse conservee.

#include <adc/integrator/amr_reflux_mf.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
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
  const double dxc = 1.0 / nc, dyc = 1.0 / nc;

  Diocotron m;
  m.B0 = 1.0;
  m.n_i0 = 1.0;
  auto gxv = [&](double dx, int i) { return 0.2 * std::sin(2 * kPi * (i + 0.5) * dx); };
  auto blob = [](double x, double y) {
    return 1.0 + 0.5 * std::exp(-((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) / 0.02);
  };

  auto MFv = [&](std::vector<Box2D> bs, int nco) {
    return MultiFab(BoxArray(bs), DistributionMapping((int)bs.size(), n_ranks()), nco, 1);
  };
  auto fill_aux = [&](MultiFab& a, double dx) {
    for (int li = 0; li < a.local_size(); ++li) {
      Array4 ar = a.fab(li).array();
      const Box2D g = a.fab(li).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
          ar(i, j, 0) = 0; ar(i, j, 1) = gxv(dx, i); ar(i, j, 2) = -1.0;
        }
    }
  };
  auto init = [&](MultiFab& U, double dx) {
    for (int li = 0; li < U.local_size(); ++li) {
      Array4 u = U.fab(li).array();
      const Box2D b = U.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i) u(i, j, 0) = blob((i + 0.5) * dx, (j + 0.5) * dx);
    }
  };
  auto coarse_mass = [&](const MultiFab& U) {
    double M = 0;
    const ConstArray4 u = U.fab(0).const_array();
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) M += u(i, j, 0) * dxc * dyc;
    return M;
  };

  // ===== Garde 1 : 3 niveaux mono-box, multipatch vs amr_step_multilevel_mf =====
  {
    const int I0 = 8, I1 = 23, J0 = 8, J1 = 23, K0 = 24, K1 = 39, L0 = 24, L1 = 39;
    Box2D fb1{{2 * I0, 2 * J0}, {2 * I1 + 1, 2 * J1 + 1}};
    Box2D fb2{{2 * K0, 2 * L0}, {2 * K1 + 1, 2 * L1 + 1}};
    MultiFab a0 = MFv({dom}, 3), a1 = MFv({fb1}, 3), a2 = MFv({fb2}, 3);
    fill_aux(a0, dxc); fill_aux(a1, dxc / 2); fill_aux(a2, dxc / 4);

    // reference AmrLevelMF
    MultiFab R0 = MFv({dom}, 1), R1 = MFv({fb1}, 1), R2 = MFv({fb2}, 1);
    init(R0, dxc); init(R1, dxc / 2); init(R2, dxc / 4);
    mf_average_down(R2, R1, K0, K1, L0, L1);
    mf_average_down(R1, R0, I0, I1, J0, J1);
    std::vector<detail::AmrLevelMF> LR(3);
    LR[0] = {std::move(R0), &a0, dxc, dyc, I0, I1, J0, J1, true};
    LR[1] = {std::move(R1), &a1, dxc / 2, dyc / 2, K0, K1, L0, L1, true};
    LR[2] = {std::move(R2), &a2, dxc / 4, dyc / 4, 0, 0, 0, 0, false};

    // candidat AmrLevelMP (memes donnees)
    MultiFab P0 = MFv({dom}, 1), P1 = MFv({fb1}, 1), P2 = MFv({fb2}, 1);
    init(P0, dxc); init(P1, dxc / 2); init(P2, dxc / 4);
    mf_average_down_mb(P2, P1);
    mf_average_down_mb(P1, P0);
    std::vector<AmrLevelMP> LP(3);
    LP[0] = {std::move(P0), &a0, dxc, dyc};
    LP[1] = {std::move(P1), &a1, dxc / 2, dyc / 2};
    LP[2] = {std::move(P2), &a2, dxc / 4, dyc / 4};

    const double dt = 0.4 * dxc;
    for (int s = 0; s < 40; ++s) {
      detail::amr_step_multilevel_mf<NoSlope, RusanovFlux>(m, LR, dom, dt);
      detail::amr_step_multilevel_multipatch<NoSlope, RusanovFlux>(m, LP, dom, dt);
    }
    double maxdiff = 0;
    const ConstArray4 ur = LR[0].U.fab(0).const_array(), up = LP[0].U.fab(0).const_array();
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i)
        maxdiff = std::fmax(maxdiff, std::fabs(ur(i, j, 0) - up(i, j, 0)));
    std::printf("garde 1 (3 niv mono-box) : max|dUc| vs multilevel_mf = %.3e\n", maxdiff);
    chk(maxdiff < 1e-12, "mp_equiv_multilevel_mf");
  }

  // ===== Garde 2 : 2 niveaux, fin a 2 boxes, vs amr_step_2level_multipatch =====
  {
    const int CI0 = 8, CI1 = 23, CJ0 = 8, CJ1 = 23, CM = 15;
    Box2D left{{2 * CI0, 2 * CJ0}, {2 * CM + 1, 2 * CJ1 + 1}};
    Box2D right{{2 * (CM + 1), 2 * CJ0}, {2 * CI1 + 1, 2 * CJ1 + 1}};
    MultiFab axc = MFv({dom}, 3), axf = MFv({left, right}, 3);
    fill_aux(axc, dxc); fill_aux(axf, dxc / 2);
    const double dt = 0.2 * dxc / std::hypot(0.2, 1.0);

    // reference amr_step_2level_multipatch
    MultiFab Rc = MFv({dom}, 1), Rf = MFv({left, right}, 1);
    init(Rc, dxc); init(Rf, dxc / 2);
    for (int s = 0; s < 20; ++s)
      amr_step_2level_multipatch<NoSlope, RusanovFlux>(m, Rc, dom, dxc, dyc, Rf, axc, axf, dt);

    // candidat (hierarchie 2 niveaux, fin multi-box)
    MultiFab Pc = MFv({dom}, 1), Pf = MFv({left, right}, 1);
    init(Pc, dxc); init(Pf, dxc / 2);
    std::vector<AmrLevelMP> LP(2);
    LP[0] = {std::move(Pc), &axc, dxc, dyc};
    LP[1] = {std::move(Pf), &axf, dxc / 2, dyc / 2};
    for (int s = 0; s < 20; ++s)
      detail::amr_step_multilevel_multipatch<NoSlope, RusanovFlux>(m, LP, dom, dt);

    double maxdiff = 0;
    const ConstArray4 ur = Rc.fab(0).const_array(), up = LP[0].U.fab(0).const_array();
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i)
        maxdiff = std::fmax(maxdiff, std::fabs(ur(i, j, 0) - up(i, j, 0)));
    std::printf("garde 2 (2 niv, fin 2 boxes) : max|dUc| vs 2level_multipatch = %.3e\n", maxdiff);
    chk(maxdiff < 1e-12, "mp_equiv_2level_multipatch");
  }

  // ===== Garde 3 : 3 niveaux, niveau 1 MULTI-BOX (capacite nouvelle) -> conservation =====
  {
    // niveau 1 = region L0 [8,23]x[8,23] coupee a la colonne 15 (2 patches), raffinee.
    Box2D l1L{{16, 16}, {31, 47}}, l1R{{32, 16}, {47, 47}};  // coords niveau 1
    // niveau 2 = patch interne au patch GAUCHE du niveau 1 : region L1 [20,27]x[20,27].
    Box2D l2{{40, 40}, {55, 55}};  // coords niveau 2 (= L1 [20,27]^2 x2)
    MultiFab a0 = MFv({dom}, 3), a1 = MFv({l1L, l1R}, 3), a2 = MFv({l2}, 3);
    fill_aux(a0, dxc); fill_aux(a1, dxc / 2); fill_aux(a2, dxc / 4);

    MultiFab U0 = MFv({dom}, 1), U1 = MFv({l1L, l1R}, 1), U2 = MFv({l2}, 1);
    init(U0, dxc); init(U1, dxc / 2); init(U2, dxc / 4);
    mf_average_down_mb(U2, U1);  // sync init : niveau 1 <- niveau 2
    mf_average_down_mb(U1, U0);  // niveau 0 <- niveau 1 (avant de mesurer la masse)
    std::vector<AmrLevelMP> LP(3);
    LP[0] = {std::move(U0), &a0, dxc, dyc};
    LP[1] = {std::move(U1), &a1, dxc / 2, dyc / 2};
    LP[2] = {std::move(U2), &a2, dxc / 4, dyc / 4};

    const double M0 = coarse_mass(LP[0].U);
    const double dt = 0.4 * dxc;
    for (int s = 0; s < 30; ++s)
      detail::amr_step_multilevel_multipatch<NoSlope, RusanovFlux>(m, LP, dom, dt);
    const double drift = std::fabs(coarse_mass(LP[0].U) - M0);
    std::printf("garde 3 (3 niv, niv 1 multi-box) : drift_masse = %.3e\n", drift);
    chk(drift < 1e-12, "mp_3level_multibox_conservation");
  }

  if (fails == 0) std::printf("OK test_amr_multilevel_multipatch\n");
  return fails == 0 ? 0 : 1;
}
