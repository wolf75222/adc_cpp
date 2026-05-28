#pragma once

#include <adc/mesh/box2d.hpp>

// for_each_cell : le dispatch maison sur les cellules d'une Box2D. C'est le
// seam de parallelisme, calque sur Kokkos::parallel_for(MDRangePolicy<Rank<2>>).
//
// Le fonctor est pris par valeur et recoit (i, j). Il capture par valeur des
// handles Array4 (POD), jamais le Fab ni rien de virtuel : exactement la
// contrainte d'un kernel device. Basculer sur Kokkos plus tard revient a
// remplacer le corps de cette fonction, sans toucher aux appelants.
//
// Backend actuel : OpenMP (collapse 2D, ordonnancement statique). Sans -fopenmp
// la pragma est ignoree et l'execution est sequentielle.

namespace adc {

template <class F>
void for_each_cell(const Box2D& b, F f) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (int j = b.lo[1]; j <= b.hi[1]; ++j)
    for (int i = b.lo[0]; i <= b.hi[0]; ++i) f(i, j);
}

}  // namespace adc
