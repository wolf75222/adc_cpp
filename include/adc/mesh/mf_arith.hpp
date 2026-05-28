#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/multifab.hpp>

#include <algorithm>

// Combinaisons lineaires de MultiFab sur les cellules valides, pour les etages
// des integrateurs. Suppose des layouts identiques (meme BoxArray, meme
// DistributionMapping). Operations point a point, donc l'aliasing (x ou y == z)
// est sans danger.

namespace adc {

// y <- y + a x
inline void saxpy(MultiFab& y, Real a, const MultiFab& x) {
  for (int li = 0; li < y.local_size(); ++li) {
    Fab2D& Y = y.fab(li);
    const Fab2D& X = x.fab(li);
    const Box2D b = Y.box();
    for (int c = 0; c < y.ncomp(); ++c)
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i) Y(i, j, c) += a * X(i, j, c);
  }
}

// norme infinie sur les cellules valides d'une composante
inline Real norm_inf(const MultiFab& mf, int comp = 0) {
  Real m = 0;
  for (int li = 0; li < mf.local_size(); ++li) {
    const Fab2D& f = mf.fab(li);
    const Box2D b = f.box();
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        const Real a = f(i, j, comp);
        m = std::max(m, a < 0 ? -a : a);
      }
  }
  return m;  // all-reduce max MPI plus tard
}

// z <- a x + b y
inline void lincomb(MultiFab& z, Real a, const MultiFab& x, Real b,
                    const MultiFab& y) {
  for (int li = 0; li < z.local_size(); ++li) {
    Fab2D& Z = z.fab(li);
    const Fab2D& X = x.fab(li);
    const Fab2D& Y = y.fab(li);
    const Box2D bb = Z.box();
    for (int c = 0; c < z.ncomp(); ++c)
      for (int j = bb.lo[1]; j <= bb.hi[1]; ++j)
        for (int i = bb.lo[0]; i <= bb.hi[0]; ++i)
          Z(i, j, c) = a * X(i, j, c) + b * Y(i, j, c);
  }
}

}  // namespace adc
