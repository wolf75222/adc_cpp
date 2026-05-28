// AMR multi-niveaux dans le temps : sous-cyclage Berger-Oliger recursif sur 3
// niveaux emboites (ratio 2 a chaque etage) avec reflux a chaque interface. Le
// meme blob advecte que test_amr_reflux, mais raffine deux fois au centre. Le
// niveau le plus fin fait r*r = 4 sous-pas par pas grossier. On verifie que la
// masse reste conservee a l'arrondi (telescopage average_down + reflux a tous
// les etages) et que la solution reste bornee.

#include <adc/integrator/amr_multilevel.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/model/diocotron.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int nc = 32;
  Box2D dom = Box2D::from_extents(nc, nc);
  const double dxc = 1.0 / nc, dyc = 1.0 / nc;

  // region raffinee par le niveau 1 (coords niveau 0) et par le niveau 2
  // (coords niveau 1, strictement interieure a la box du niveau 1).
  const int I0 = 8, I1 = 23, J0 = 8, J1 = 23;       // niveau 1 dans niveau 0
  const int K0 = 24, K1 = 39, L0 = 24, L1 = 39;     // niveau 2 dans niveau 1
  Box2D fbox1{{2 * I0, 2 * J0}, {2 * I1 + 1, 2 * J1 + 1}};
  Box2D fbox2{{2 * K0, 2 * L0}, {2 * K1 + 1, 2 * L1 + 1}};

  Diocotron m;
  m.B0 = 1.0;
  constexpr double kPi = 3.14159265358979323846;
  auto fill_aux = [&](Fab2D& aux, double dx) {
    const Box2D g = aux.grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
        aux(i, j, 0) = 0.0;                                      // phi (inutilise)
        aux(i, j, 1) = 0.2 * std::sin(2 * kPi * (i + 0.5) * dx);  // gx
        aux(i, j, 2) = -1.0;                                     // gy
      }
  };
  auto blob = [](double x, double y) {
    const double r2 = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5);
    return 1.0 + 0.5 * std::exp(-r2 / 0.02);
  };

  Fab2D U0(dom, 1, 1), U1(fbox1, 1, 1), U2(fbox2, 1, 1);
  Fab2D a0(dom, 3, 1), a1(fbox1, 3, 1), a2(fbox2, 3, 1);
  fill_aux(a0, dxc);
  fill_aux(a1, dxc / 2);
  fill_aux(a2, dxc / 4);
  auto init = [&](Fab2D& U, double dx) {
    const Box2D b = U.box();
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        U(i, j) = blob((i + 0.5) * dx, (j + 0.5) * dx);
  };
  init(U0, dxc);
  init(U1, dxc / 2);
  init(U2, dxc / 4);
  average_down_fab(U2, U1, K0, K1, L0, L1);  // sync 2 -> 1
  average_down_fab(U1, U0, I0, I1, J0, J1);  // sync 1 -> 0

  std::vector<AmrLevel> L(3);
  L[0] = {std::move(U0), &a0, dxc,     dyc,     I0, I1, J0, J1, true};
  L[1] = {std::move(U1), &a1, dxc / 2, dyc / 2, K0, K1, L0, L1, true};
  L[2] = {std::move(U2), &a2, dxc / 4, dyc / 4, 0,  0,  0,  0,  false};

  auto mass = [&]() {
    double M = 0;
    const Box2D b = L[0].U.box();
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) M += L[0].U(i, j) * dxc * dyc;
    return M;
  };
  const double M0 = mass();

  const double dt = 0.4 * dxc;  // CFL grossier ; le plus fin fait 4 sous-pas
  for (int s = 0; s < 60; ++s) amr_step_multilevel(m, L, dom, dt);

  const double M1 = mass();
  std::printf("masse : M0=%.10f M1=%.10f  drift=%.3e\n", M0, M1,
              std::fabs(M1 - M0));
  chk(std::fabs(M1 - M0) < 1e-12, "mass_conserved_multilevel_reflux");

  double mn = 1e300, mx = -1e300;
  const Box2D b = L[0].U.box();
  for (int j = b.lo[1]; j <= b.hi[1]; ++j)
    for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
      mn = std::min(mn, L[0].U(i, j));
      mx = std::max(mx, L[0].U(i, j));
    }
  chk(mn > 0.999 && mx < 1.5, "stable_and_bounded_multilevel");

  if (fails == 0) std::printf("OK test_amr_multilevel\n");
  return fails == 0 ? 0 : 1;
}
