#pragma once

#include <adc/mesh/box2d.hpp>

#ifdef ADC_HAS_KOKKOS
#include <Kokkos_Core.hpp>
#endif

// for_each_cell : le seam de parallelisme sur les cellules d'une Box2D. Le
// fonctor est pris par valeur et recoit (i, j) ; il capture par valeur des
// handles Array4 (POD), jamais le Fab ni rien de virtuel : exactement la
// contrainte d'un kernel device.
//
// Backend selectionnable a la compilation (par unite de traduction) :
//   - Kokkos (ADC_HAS_KOKKOS) : Kokkos::parallel_for(MDRangePolicy<Rank<2>>),
//     s'execute sur l'espace par defaut (p.ex. Cuda sur GH200). Le fonctor doit
//     alors etre device-callable (lambda annotee ADC_HD) et operer sur une
//     donnee device-residente. C'est le MEME appel for_each_cell que le CPU.
//   - OpenMP (_OPENMP) : collapse(2), ordonnancement statique.
//   - sinon : sequentiel.
//
// Le passage CPU -> GPU ne change donc PAS les sites d'appel : on remplace le
// backend ici, les operateurs (assemble_rhs, coupleurs, demos) restent inchanges.

namespace adc {

// Barriere device : attend la fin des kernels en vol avant qu'un acces HOTE a la
// memoire (unifiee) ne lise des donnees encore en cours d'ecriture par un kernel.
// No-op hors Kokkos. A appeler avant toute lecture/ecriture hote (fill_ghosts,
// transferts, normes) suivant un for_each_cell sur GPU.
inline void device_fence() {
#if defined(ADC_HAS_KOKKOS)
  Kokkos::fence();
#endif
}

template <class F>
void for_each_cell(const Box2D& b, F f) {
#if defined(ADC_HAS_KOKKOS)
  // IndexType<int> : indices SIGNES. Les boites de ghosts ont des bornes basses
  // negatives (p.ex. lo = -ng pour copy_shifted) ; sans type signe explicite,
  // MDRangePolicy rejette la borne -1 (conversion implicite jugee non sure).
  Kokkos::parallel_for(
      "adc_for_each_cell",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<int>>(
          {b.lo[0], b.lo[1]}, {b.hi[0] + 1, b.hi[1] + 1}),
      f);
#elif defined(_OPENMP)
  // Clause if() : on ne paie le fork/join que si la boite est assez grande. Sinon
  // (niveaux grossiers de la multigrille : 2x2, 4x4...) le cout d'ouverture de la
  // region parallele ecraserait le calcul (regression x40 mesuree sans le seuil).
  const long n_cells = static_cast<long>(b.hi[0] - b.lo[0] + 1) *
                       (b.hi[1] - b.lo[1] + 1);
#pragma omp parallel for collapse(2) schedule(static) if (n_cells >= 4096)
  for (int j = b.lo[1]; j <= b.hi[1]; ++j)
    for (int i = b.lo[0]; i <= b.hi[0]; ++i) f(i, j);
#else
  for (int j = b.lo[1]; j <= b.hi[1]; ++j)
    for (int i = b.lo[0]; i <= b.hi[0]; ++i) f(i, j);
#endif
}

}  // namespace adc
