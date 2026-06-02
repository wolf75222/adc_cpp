// GeometricMG::solve_robust : durcissement anti-divergence au bord embedded haute resolution.
// On verifie trois proprietes du contrat :
//   1. CONVERGENCE sur un cas qui DIVERGE en nu nominal (paroi conductrice circulaire a eff 640,
//      ou le V-cycle nu=2 a un rayon spectral > 1) : solve_robust doit ramener le residu sous tol ;
//   2. LOCALITE du durcissement (non sticky) : deux solves a froid identiques font le MEME travail
//      (meme nombre de cycles). Un durcissement STICKY ferait converger le 2e plus vite (nu reste
//      eleve) ; la version locale restaure nu, donc le 2e refait exactement le chemin du 1er ;
//   3. BIT-IDENTIQUE a solve() sur un cas convergent (eff 224) : la phase 1 est le corps de solve(),
//      donc phi doit etre identique au bit pres.

#include <adc/elliptic/geometric_mg.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>

#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

using namespace adc;

// remplit le RHS = anneau de charge (meme CI que le transport a derive en colonne) et remet phi a 0.
static void set_ring_rhs(GeometricMG& mg, int nc, double dx) {
  const double cx = 0.5, cy = 0.5, r0 = 0.15, r1 = 0.20, delta = 0.1, floor = 1e-3;
  mg.phi().set_val(0.0);
  Array4 f = mg.rhs().fab(0).array();
  for (int j = 0; j < nc; ++j)
    for (int i = 0; i < nc; ++i) {
      const double x = (i + 0.5) * dx, y = (j + 0.5) * dx;
      const double r = std::hypot(x - cx, y - cy), th = std::atan2(y - cy, x - cx);
      f(i, j) = (r > r0 && r < r1) ? (1.0 - delta + delta * std::sin(4 * th)) : floor;
    }
}

static GeometricMG make_mg(int nc) {
  const double cx = 0.5, cy = 0.5, Rwall = 0.40;
  Box2D dom = Box2D::from_extents(nc, nc);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba(std::vector<Box2D>{dom});
  std::function<bool(Real, Real)> active = [=](Real x, Real y) { return std::hypot(x - cx, y - cy) < Rwall; };
  BCRec bc; bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;
  return GeometricMG(geom, ba, bc, active);
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) { if (!c) { std::printf("FAIL %s\n", w); ++fails; } };

  // 1 + 2 : cas DIVERGENT (eff 640). solve_robust converge ; deux solves a froid font le meme travail.
  {
    GeometricMG mg = make_mg(640);
    const double dx = 1.0 / 640;
    set_ring_rhs(mg, 640, dx);
    const double r0 = mg.residual();
    const int n1 = mg.solve_robust(1e-8, 50);
    const double rfin = mg.residual();
    chk(rfin < 1e-6 * r0, "convergence_sur_cas_divergent");

    set_ring_rhs(mg, 640, dx);  // remise a froid (phi=0, meme RHS)
    const int n2 = mg.solve_robust(1e-8, 50);
    chk(n1 == n2, "durcissement_local_non_sticky");  // sticky -> n2 < n1
    std::printf("solve_robust eff640 : r0=%.2e rfin=%.2e ratio=%.2e cyc=%d puis %d (egaux=%d)\n",
                r0, rfin, rfin / r0, n1, n2, int(n1 == n2));
  }

  // 3 : cas CONVERGENT (eff 224), solve() vs solve_robust bit-a-bit identiques.
  {
    GeometricMG a = make_mg(224), b = make_mg(224);
    const double dx = 1.0 / 224;
    set_ring_rhs(a, 224, dx);
    set_ring_rhs(b, 224, dx);
    a.solve(1e-8, 50);
    b.solve_robust(1e-8, 50);
    MultiFab diff = a.phi();
    saxpy(diff, Real(-1), b.phi());
    const double md = norm_inf(diff);
    chk(md == 0.0, "bit_identique_a_solve_si_convergent");
    std::printf("solve vs solve_robust (eff224, convergent) : maxdiff(phi)=%.3e\n", md);
  }

  if (fails == 0) std::printf("OK test_solve_robust\n");
  return fails == 0 ? 0 : 1;
}
