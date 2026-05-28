// Operateur spatial Rusanov sur le modele diocotron a aux prescrit.
//   - champ uniforme + vitesse constante -> R = 0 (divergence d'un flux constant)
//   - rampe lineaire rho = x, vitesse vx = 1 -> R = -v . grad rho = -1 (exact a
//     l'ordre 1 amont sur donnee lineaire), verifie a l'interieur.

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/operator/spatial_operator.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

// aux uniforme (ghosts compris) : phi, grad_x, grad_y constants
static void fill_aux_uniform(MultiFab& aux, Real phi, Real gx, Real gy) {
  for (int li = 0; li < aux.local_size(); ++li) {
    Fab2D& f = aux.fab(li);
    const Box2D gb = f.grown_box();
    for (int j = gb.lo[1]; j <= gb.hi[1]; ++j)
      for (int i = gb.lo[0]; i <= gb.hi[0]; ++i) {
        f(i, j, 0) = phi;
        f(i, j, 1) = gx;
        f(i, j, 2) = gy;
      }
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

  Box2D dom = Box2D::from_extents(8, 8);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  Diocotron model;
  model.B0 = 1.0;
  BoxArray ba = BoxArray::from_domain(dom, 8);
  DistributionMapping dm(ba.size(), n_ranks());

  // vx = -grad_y / B0 = 1, vy = grad_x / B0 = 0
  MultiFab aux(ba, dm, 3, 1);
  fill_aux_uniform(aux, 0.0, 0.0, -1.0);

  // --- champ uniforme -> R = 0 ---
  {
    MultiFab U(ba, dm, 1, 1), R(ba, dm, 1, 0);
    U.set_val(2.0);  // remplit aussi les ghosts
    assemble_rhs(model, U, aux, geom, R);
    Real maxabs = 0;
    const Fab2D& r = R.fab(0);
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        maxabs = std::max(maxabs, std::fabs(r(i, j, 0)));
    chk(maxabs < 1e-12, "uniform_zero");
  }

  // --- rampe rho = x -> R = -1 a l'interieur ---
  {
    MultiFab U(ba, dm, 1, 1), R(ba, dm, 1, 0);
    Array4 a = U.fab(0).array();
    for_each_cell(dom, [a, geom](int i, int j) { a(i, j, 0) = geom.x_cell(i); });

    BCRec bc;  // foextrap partout (les ghosts ne sont valides qu'au bord)
    bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Foextrap;
    fill_ghosts(U, dom, bc);

    assemble_rhs(model, U, aux, geom, R);
    const Fab2D& r = R.fab(0);
    Real maxerr = 0;
    for (int j = 1; j <= 6; ++j)
      for (int i = 1; i <= 6; ++i)
        maxerr = std::max(maxerr, std::fabs(r(i, j, 0) - (-1.0)));
    chk(maxerr < 1e-12, "ramp_minus_v_grad");
  }

  if (fails == 0) std::printf("OK test_spatial_operator\n");
  return fails == 0 ? 0 : 1;
}
