// fill_boundary : echange de halos intra-niveau.
//   - 4 boxes, non periodique : aretes et coin interieurs remplis depuis les
//     voisins, ghosts hors domaine laisses a 0.
//   - 1 box, periodique : wrapping correct des deux cotes et des coins.
//   - la somme sur les cellules valides ne change pas (on n'ecrit que les ghosts).

#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <cmath>
#include <cstdio>

using namespace pops;

// champ global continu a travers les frontieres de boxes
static double g(int i, int j) {
  return i + 100.0 * j;
}

static void fill_valid(MultiFab& mf) {
  for (int li = 0; li < mf.local_size(); ++li) {
    Array4 a = mf.fab(li).array();
    for_each_cell(mf.box(li), [a](int i, int j) { a(i, j, 0) = g(i, j); });
  }
}

// recupere le fab local dont le coin bas vaut (lo0, lo1)
static const Fab2D& fab_with_lo(const MultiFab& mf, int lo0, int lo1) {
  for (int li = 0; li < mf.local_size(); ++li)
    if (mf.box(li).lo[0] == lo0 && mf.box(li).lo[1] == lo1)
      return mf.fab(li);
  return mf.fab(0);
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  // --- 4 boxes, non periodique ---
  {
    Box2D dom = Box2D::from_extents(8, 8);
    BoxArray ba = BoxArray::from_domain(dom, 4);  // boxes 4x4
    MultiFab mf(ba, DistributionMapping(ba.size(), n_ranks()), 1, 1);
    fill_valid(mf);
    Real s_before = sum(mf);

    fill_boundary(mf, dom, Periodicity{false, false});

    const Fab2D& b0 = fab_with_lo(mf, 0, 0);     // box [0..3]x[0..3]
    chk(b0(4, 2, 0) == g(4, 2), "edge_right");   // depuis le voisin x
    chk(b0(2, 4, 0) == g(2, 4), "edge_top");     // depuis le voisin y
    chk(b0(4, 4, 0) == g(4, 4), "corner_diag");  // depuis le voisin diagonal
    chk(b0(-1, 2, 0) == 0.0, "phys_left_zero");  // bord physique : intact
    chk(b0(2, -1, 0) == 0.0, "phys_bottom_zero");
    chk(b0(-1, -1, 0) == 0.0, "phys_corner_zero");

    chk(std::fabs(sum(mf) - s_before) < 1e-12, "sum_unchanged");
  }

  // --- 1 box couvrant tout le domaine, periodique ---
  {
    Box2D dom = Box2D::from_extents(8, 8);
    BoxArray ba = BoxArray::from_domain(dom, 8);  // une seule box [0..7]x[0..7]
    chk(ba.size() == 1, "single_box");
    MultiFab mf(ba, DistributionMapping(ba.size(), n_ranks()), 1, 1);
    fill_valid(mf);

    fill_boundary(mf, dom, Periodicity{true, true});

    const Fab2D& f = mf.fab(0);
    chk(f(-1, 3, 0) == g(7, 3), "wrap_left");    // i=-1 <- i=7
    chk(f(8, 3, 0) == g(0, 3), "wrap_right");    // i=8  <- i=0
    chk(f(3, -1, 0) == g(3, 7), "wrap_bottom");  // j=-1 <- j=7
    chk(f(3, 8, 0) == g(3, 0), "wrap_top");      // j=8  <- j=0
    chk(f(-1, -1, 0) == g(7, 7), "wrap_corner");
    chk(f(8, 8, 0) == g(0, 0), "wrap_corner2");
  }

  if (fails == 0)
    std::printf("OK test_fill_boundary\n");
  return fails == 0 ? 0 : 1;
}
