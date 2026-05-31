// Valide amr_step_multilevel_mf (recursion AMR N-niveaux sur MultiFab) par
// EQUIVALENCE a amr_step_multilevel (Fab2D, deja teste conservatif) : meme hierarchie
// 3 niveaux, meme CI, meme aux -> avec NoSlope/Rusanov les deux doivent donner le MEME
// niveau grossier a l'arrondi, sur plusieurs pas. Plus : la masse MultiFab conservee.

#include <adc/integrator/amr_multilevel.hpp>     // Fab2D (reference)
#include <adc/integrator/amr_reflux_mf.hpp>      // MultiFab (candidat)
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
  const int I0 = 8, I1 = 23, J0 = 8, J1 = 23;    // niveau 1 dans 0
  const int K0 = 24, K1 = 39, L0 = 24, L1 = 39;  // niveau 2 dans 1
  Box2D fbox1{{2 * I0, 2 * J0}, {2 * I1 + 1, 2 * J1 + 1}};
  Box2D fbox2{{2 * K0, 2 * L0}, {2 * K1 + 1, 2 * L1 + 1}};

  Diocotron m;
  m.B0 = 1.0;
  auto gxv = [&](double dx, int i) { return 0.2 * std::sin(2 * kPi * (i + 0.5) * dx); };
  auto blob = [](double x, double y) {
    return 1.0 + 0.5 * std::exp(-((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) / 0.02);
  };

  // --- reference Fab2D ---
  Fab2D U0(dom, 1, 1), U1(fbox1, 1, 1), U2(fbox2, 1, 1);
  Fab2D a0(dom, 3, 1), a1(fbox1, 3, 1), a2(fbox2, 3, 1);
  auto fill_aux_f = [&](Fab2D& a, double dx) {
    const Box2D g = a.grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) { a(i, j, 0) = 0; a(i, j, 1) = gxv(dx, i); a(i, j, 2) = -1.0; }
  };
  fill_aux_f(a0, dxc); fill_aux_f(a1, dxc / 2); fill_aux_f(a2, dxc / 4);
  auto init_f = [&](Fab2D& U, double dx) {
    const Box2D b = U.box();
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) U(i, j) = blob((i + 0.5) * dx, (j + 0.5) * dx);
  };
  init_f(U0, dxc); init_f(U1, dxc / 2); init_f(U2, dxc / 4);
  average_down_fab(U2, U1, K0, K1, L0, L1);
  average_down_fab(U1, U0, I0, I1, J0, J1);
  std::vector<AmrLevel> LF(3);
  LF[0] = {std::move(U0), &a0, dxc, dyc, I0, I1, J0, J1, true};
  LF[1] = {std::move(U1), &a1, dxc / 2, dyc / 2, K0, K1, L0, L1, true};
  LF[2] = {std::move(U2), &a2, dxc / 4, dyc / 4, 0, 0, 0, 0, false};

  // --- candidat MultiFab (memes donnees) ---
  DistributionMapping dm(1, n_ranks());
  auto MF = [&](const Box2D& b, int nco) {
    return MultiFab(BoxArray(std::vector<Box2D>{b}), dm, nco, 1);
  };
  MultiFab M0 = MF(dom, 1), M1 = MF(fbox1, 1), M2 = MF(fbox2, 1);
  MultiFab b0 = MF(dom, 3), b1 = MF(fbox1, 3), b2 = MF(fbox2, 3);
  auto fill_aux_m = [&](MultiFab& a, double dx) {
    Array4 ar = a.fab(0).array();
    const Box2D g = a.fab(0).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) { ar(i, j, 0) = 0; ar(i, j, 1) = gxv(dx, i); ar(i, j, 2) = -1.0; }
  };
  fill_aux_m(b0, dxc); fill_aux_m(b1, dxc / 2); fill_aux_m(b2, dxc / 4);
  auto init_m = [&](MultiFab& U, double dx) {
    Array4 u = U.fab(0).array();
    const Box2D b = U.box(0);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) u(i, j, 0) = blob((i + 0.5) * dx, (j + 0.5) * dx);
  };
  init_m(M0, dxc); init_m(M1, dxc / 2); init_m(M2, dxc / 4);
  mf_average_down(M2, M1, K0, K1, L0, L1);
  mf_average_down(M1, M0, I0, I1, J0, J1);
  std::vector<detail::AmrLevelMF> LM(3);
  LM[0] = {std::move(M0), &b0, dxc, dyc, I0, I1, J0, J1, true};
  LM[1] = {std::move(M1), &b1, dxc / 2, dyc / 2, K0, K1, L0, L1, true};
  LM[2] = {std::move(M2), &b2, dxc / 4, dyc / 4, 0, 0, 0, 0, false};

  auto mass_m = [&]() {
    double M = 0;
    const ConstArray4 u = LM[0].U.fab(0).const_array();
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) M += u(i, j, 0) * dxc * dyc;
    return M;
  };
  const double Mm0 = mass_m();

  const double dt = 0.4 * dxc;
  for (int s = 0; s < 40; ++s) {
    amr_step_multilevel(m, LF, dom, dt);
    detail::amr_step_multilevel_mf<NoSlope, RusanovFlux>(m, LM, dom, dt);
  }

  double maxdiff = 0;
  const ConstArray4 um = LM[0].U.fab(0).const_array();
  for (int j = 0; j < nc; ++j)
    for (int i = 0; i < nc; ++i)
      maxdiff = std::fmax(maxdiff, std::fabs(um(i, j, 0) - LF[0].U(i, j)));
  const double drift = std::fabs(mass_m() - Mm0);
  std::printf("multilevel_mf vs Fab2D (40 pas, 3 niveaux) : max|dUc|=%.3e  drift_masse=%.3e\n",
              maxdiff, drift);
  chk(maxdiff < 1e-12, "mf_equiv_fab2d_multilevel");
  chk(drift < 1e-12, "mf_mass_conserved_multilevel");

  if (fails == 0) std::printf("OK test_amr_multilevel_mf\n");
  return fails == 0 ? 0 : 1;
}
