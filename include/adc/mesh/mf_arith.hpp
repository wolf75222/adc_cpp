#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/multifab.hpp>

#include <algorithm>

// Combinaisons lineaires de MultiFab sur les cellules valides, pour les etages
// des integrateurs. Suppose des layouts identiques (meme BoxArray, meme
// DistributionMapping). Operations point a point, donc l'aliasing (x ou y == z)
// est sans danger.

namespace adc {

// y <- y + a x
inline void saxpy(MultiFab& y, Real a, const MultiFab& x) {
  const int nc = y.ncomp();
  for (int li = 0; li < y.local_size(); ++li) {
    Array4 Y = y.fab(li).array();
    const ConstArray4 X = x.fab(li).const_array();
    const Box2D b = y.fab(li).box();
    for (int c = 0; c < nc; ++c)
      for_each_cell(b, [=] ADC_HD(int i, int j) { Y(i, j, c) += a * X(i, j, c); });
  }
}

// norme infinie sur les cellules valides d'une composante. Chaque fab local est
// reduit par for_each_cell_reduce_max sur |f(i,j,comp)| (vraie reduction device
// sous Kokkos, boucle hote en serie/OpenMP), agrege par max hote sur les fabs.
//
// Plus de device_fence() en tete : sous Kokkos parallel_reduce est bloquant et
// absorbe la barriere. EXACT partout : max et fabs sont sans arrondi et le max
// est associatif/commutatif en IEEE754, donc bit-identique a l'ancien norm_inf
// quel que soit le backend (l'ordre de reduction ne change aucun bit).
inline Real norm_inf(const MultiFab& mf, int comp = 0) {
  Real m = 0;
  for (int li = 0; li < mf.local_size(); ++li) {
    const ConstArray4 a = mf.fab(li).const_array();
    m = std::max(m, for_each_cell_reduce_max(
                        mf.box(li), [a, comp] ADC_HD(int i, int j) {
                          const Real v = a(i, j, comp);
                          return v < 0 ? -v : v;
                        }));
  }
  return m;  // all-reduce max MPI plus tard (iso-comportement, non ajoute ici)
}

// z <- a x + b y
inline void lincomb(MultiFab& z, Real a, const MultiFab& x, Real b,
                    const MultiFab& y) {
  const int nc = z.ncomp();
  for (int li = 0; li < z.local_size(); ++li) {
    Array4 Z = z.fab(li).array();
    const ConstArray4 X = x.fab(li).const_array();
    const ConstArray4 Y = y.fab(li).const_array();
    const Box2D bb = z.fab(li).box();
    for (int c = 0; c < nc; ++c)
      for_each_cell(bb, [=] ADC_HD(int i, int j) {
        Z(i, j, c) = a * X(i, j, c) + b * Y(i, j, c);
      });
  }
}

}  // namespace adc
