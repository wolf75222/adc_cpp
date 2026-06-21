// CL physiques : Foextrap (gradient nul), Dirichlet (reflexion autour de la
// valeur de face), et remplissage correct des coins par composition
// faces-x puis faces-y.
//
// + COIN du stencil a 9 points en MULTI-BOX : le coin (x-physique CROISE y-PERIODIQUE) doit etre rempli
//   par fill_ghosts (fill_boundary remplit la halo y, puis fill_physical_bc etend la CL radiale dans la
//   halo y via la plage j ETENDUE). Sans cette extension, ce coin reste a 0 et un terme croise d'un
//   operateur a 9 points le lit faux au bord de box (regression historique multi-box polaire).

#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>

#include <cstdio>
#include <vector>

using namespace adc;

static double g(int i, int j) {
  return i + 10.0 * j;
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  Box2D dom = Box2D::from_extents(4, 4);  // [0..3] x [0..3]
  BoxArray ba = BoxArray::from_domain(dom, 4);
  MultiFab mf(ba, DistributionMapping(ba.size(), n_ranks()), 1, 1);

  Array4 a = mf.fab(0).array();
  for_each_cell(dom, [a](int i, int j) { a(i, j, 0) = g(i, j); });

  // melange : xlo foextrap, xhi dirichlet(0), ylo foextrap, yhi dirichlet(0)
  BCRec bc;
  bc.xlo = BCType::Foextrap;
  bc.xhi = BCType::Dirichlet;
  bc.xhi_val = 0.0;
  bc.ylo = BCType::Foextrap;
  bc.yhi = BCType::Dirichlet;
  bc.yhi_val = 0.0;

  fill_physical_bc(mf, dom, bc);
  const Fab2D& f = mf.fab(0);

  // foextrap xlo : ghost = cellule interne la plus proche
  chk(f(-1, 2, 0) == g(0, 2), "foextrap_xlo");
  // dirichlet xhi val 0 : ghost = -interne miroir
  chk(f(4, 2, 0) == -g(3, 2), "dirichlet_xhi");
  // foextrap ylo
  chk(f(2, -1, 0) == g(2, 0), "foextrap_ylo");
  // dirichlet yhi
  chk(f(2, 4, 0) == -g(2, 3), "dirichlet_yhi");

  // coin bas-gauche : face-y foextrap sur i etendu, lit le ghost-x deja rempli
  // ghost(-1,-1) = a(-1, 0) = g(0,0)
  chk(f(-1, -1, 0) == g(0, 0), "corner_foextrap_foextrap");
  // coin bas-droit : ylo foextrap lit le ghost xhi dirichlet a i=4
  // ghost(4,-1) = a(4,0) = -g(3,0)
  chk(f(4, -1, 0) == -g(3, 0), "corner_dirichlet_foextrap");

  // -------------------------------------------------------------------------------------------------
  // COIN (x-physique CROISE y-PERIODIQUE) en MULTI-BOX (deux boites decoupees en y, y PERIODIQUE).
  // fill_ghosts complet (fill_boundary periodique y PUIS fill_physical_bc) doit remplir le coin
  // diagonal (x-ghost physique x y-ghost periodique-d'une-autre-box). Avant le correctif, ce coin
  // restait a 0 (la CL x ne couvrait que la plage y VALIDE, pas la halo y).
  // -------------------------------------------------------------------------------------------------
  {
    const int nx = 4, ny = 8;
    Box2D dom2 = Box2D::from_extents(nx, ny);  // [0..3] x [0..7]
    // 2 boites en y : [0..3] et [4..7], chacune pleine en x.
    std::vector<Box2D> bx = {Box2D{{0, 0}, {nx - 1, 3}}, Box2D{{0, 4}, {nx - 1, 7}}};
    BoxArray ba2(std::move(bx));
    MultiFab mf2(ba2, DistributionMapping(ba2.size(), n_ranks()), 1, 1);
    for (int li = 0; li < mf2.local_size(); ++li) {
      Box2D vb = mf2.box(li);
      Array4 av = mf2.fab(li).array();
      for (int j = vb.lo[1]; j <= vb.hi[1]; ++j)
        for (int i = vb.lo[0]; i <= vb.hi[0]; ++i)
          av(i, j, 0) = g(i, j);
    }
    BCRec bc2;  // x physique (Foextrap bas, Dirichlet 0 haut), y PERIODIQUE
    bc2.xlo = BCType::Foextrap;
    bc2.xhi = BCType::Dirichlet;
    bc2.xhi_val = 0.0;
    bc2.ylo = bc2.yhi = BCType::Periodic;
    fill_ghosts(mf2, dom2, bc2);

    // Le test ne vaut qu'en mono-rang (les deux boites sont locales). Sous MPI le coin cross-rang est
    // couvert par test_mpi_fillboundary + test_polar_schur_multibox ; ici on cible le multi-box LOCAL.
    if (n_ranks() == 1) {
      const Fab2D& f0 = mf2.fab(0);  // box 0 : y valide [0..3]
      // Coin (i=-1, j=4) : x-ghost Foextrap (lit a(0, 4)) x y-ghost periodique (box1 valide j=4).
      // Foextrap xlo : ghost = a(0, 4) = g(0, 4).
      chk(f0(-1, 4, 0) == g(0, 4), "corner_xfoextrap_yperiodic_multibox");
      // Coin (i=4, j=4) : x-ghost Dirichlet 0 (ghost = -a(3, 4)) x y-ghost periodique (box1 valide).
      chk(f0(4, 4, 0) == -g(3, 4), "corner_xdirichlet_yperiodic_multibox");
      // Cote y-bas de box0 (j=-1 = periodique wrap vers j=7 de box1) croise x-physique :
      // coin (i=-1, j=-1) : Foextrap lit a(0, -1) = a(0, 7) (deja rempli periodique) = g(0, 7).
      chk(f0(-1, -1, 0) == g(0, 7), "corner_xfoextrap_yperiodic_wrap");
    }
  }

  if (fails == 0)
    std::printf("OK test_physical_bc\n");
  return fails == 0 ? 0 : 1;
}
