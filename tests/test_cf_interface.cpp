// Contrat des types SubcyclingSchedule et CoarseFineInterface (revue, points 5 et 2). Ces
// deux abstractions ont ete extraites de la recursion AMR coeur (amr_step_2level_multipatch
// et subcycle_level_mp) ; leur integration est couverte bit a bit par les tests de reflux
// (np=1/2/4 identiques). Ici on fige les mecaniques locales :
//   - SubcyclingSchedule : ratio, dt/r, frac s/r (arithmetique exactement preservee) ;
//   - CoarseFineInterface : couverture batie depuis un BoxArray fin (empreinte PatchRange)
//     et routage bordant du reflux (formules et garde de couverture).

#include <pops/numerics/time/amr/reflux/amr_reflux_mf.hpp>  // pops::SubcyclingSchedule, pops::CoarseFineInterface
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>

#include <cstdio>
#include <vector>

using namespace pops;

// registre minimal a la disposition de Reg / RegMP (champs lus par route_reflux).
struct RegLite {
  int I0, I1, J0, J1;
  std::vector<Real> cL, cR, cB, cT, fL, fR, fB, fT;
};

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  // --- SubcyclingSchedule : cadence Berger-Oliger ratio 2 ---
  SubcyclingSchedule s2(2);
  chk(s2.count() == 2, "sched_count");
  chk(s2.dt_sub(Real(1)) == Real(1) / 2, "sched_dt_sub");  // bit-identique a dt / r
  chk(s2.frac(0) == Real(0) / 2 && s2.frac(1) == Real(1) / 2, "sched_frac");
  SubcyclingSchedule s_def;  // ratio par defaut = 2
  chk(s_def.count() == 2, "sched_defaut");

  // --- CoarseFineInterface : couverture depuis un BoxArray fin ---
  // patch fin [4..11]^2 -> empreinte grossiere [2..5]^2 (PatchRange). Region grossiere 8x8.
  BoxArray fine(std::vector<Box2D>{Box2D{{4, 4}, {11, 11}}});
  CoarseFineInterface cfi(Box2D{{0, 0}, {7, 7}}, fine);
  chk(cfi.covered(2, 2) && cfi.covered(5, 5) && cfi.covered(3, 4), "cfi_couvert_dedans");
  chk(!cfi.covered(1, 2) && !cfi.covered(6, 5), "cfi_non_couvert_bordant");
  chk(!cfi.covered(-1, -1) && !cfi.covered(8, 8), "cfi_hors_region");

  // --- route_reflux : correction bordante coverage-aware ---
  // un patch, 1 composante, empreinte grossiere [2..5]^2. On pose des flux simples et on
  // verifie la formule -(fL - cL*dt)/dx sur le bord gauche et +(fR - cR*dt)/dx a droite.
  const int nc = 1;
  RegLite g;
  g.I0 = 2;
  g.I1 = 5;
  g.J0 = 2;
  g.J1 = 5;
  const int nJ = g.J1 - g.J0 + 1, nI = g.I1 - g.I0 + 1;
  g.cL.assign(nJ * nc, Real(1));
  g.cR.assign(nJ * nc, Real(2));
  g.cB.assign(nI * nc, Real(3));
  g.cT.assign(nI * nc, Real(4));
  g.fL.assign(nJ * nc, Real(10));
  g.fR.assign(nJ * nc, Real(20));
  g.fB.assign(nI * nc, Real(30));
  g.fT.assign(nI * nc, Real(40));
  const Real dx = Real(0.5), dy = Real(0.25), dt = Real(2);

  // registre grossier sur l'interface (boite englobante crue de 1, clampee).
  FluxRegister ref(Box2D{{1, 1}, {6, 6}}, nc);
  cfi.route_reflux(g, dx, dy, dt, ref, nc);

  // bord gauche I0-1 = 1 (non couvert) : -(fL - cL*dt)/dx = -(10 - 1*2)/0.5 = -16.
  chk(ref.at(1, 2, 0) == -(Real(10) - Real(1) * dt) / dx, "reflux_gauche");
  // bord droit I1+1 = 6 (non couvert) : +(fR - cR*dt)/dx = (20 - 2*2)/0.5 = 32.
  chk(ref.at(6, 2, 0) == +(Real(20) - Real(2) * dt) / dx, "reflux_droite");
  // bord bas J0-1 = 1 : -(fB - cB*dt)/dy = -(30 - 3*2)/0.25 = -96.
  chk(ref.at(2, 1, 0) == -(Real(30) - Real(3) * dt) / dy, "reflux_bas");
  // bord haut J1+1 = 6 : +(fT - cT*dt)/dy = (40 - 4*2)/0.25 = 128.
  chk(ref.at(2, 6, 0) == +(Real(40) - Real(4) * dt) / dy, "reflux_haut");

  // garde de couverture : un patch dont le bord gauche tombe sur une cellule COUVERTE par un
  // second patch ne doit PAS verser de reflux la (joint fin-fin). Deux patchs adjacents
  // [4..11]x[4..11] et [12..19]x[4..11] -> empreintes [2..5] et [6..9] en x. Le bord droit du
  // premier (I1+1 = 6) est alors couvert par le second.
  BoxArray two(std::vector<Box2D>{Box2D{{4, 4}, {11, 11}}, Box2D{{12, 4}, {19, 11}}});
  CoarseFineInterface cfi2(Box2D{{0, 0}, {15, 7}}, two);
  chk(cfi2.covered(6, 2), "cfi2_joint_couvert");
  FluxRegister ref2(Box2D{{1, 1}, {10, 6}}, nc);
  cfi2.route_reflux(g, dx, dy, dt, ref2, nc);  // g = premier patch (empreinte [2..5])
  chk(ref2.at(6, 2, 0) == Real(0), "reflux_joint_supprime");  // bord droit couvert -> rien
  chk(ref2.at(1, 2, 0) == -(Real(10) - Real(1) * dt) / dx, "reflux_gauche_libre");  // gauche libre

  if (fails == 0)
    std::printf("OK test_cf_interface\n");
  return fails == 0 ? 0 : 1;
}
