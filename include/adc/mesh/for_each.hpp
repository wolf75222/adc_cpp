#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>

#include <algorithm>

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

// Reductions device : le pendant reducteur de for_each_cell. Memes contraintes
// sur le fonctor (POD device-callable, pris par valeur, capture un ConstArray4,
// jamais le Fab) ; il recoit (i, j) et rend la valeur a accumuler. Le seam porte
// l'ordonnancement device : sous Kokkos le scalaire est pret au retour sans
// device_fence() prealable (parallel_reduce est bloquant cote hote et s'ordonne
// apres les parallel_for deja soumis dans le meme espace).
//
// CHOIX FP IMPORTANT. Une vraie reduction parallele reassocie l'addition
// flottante (non associative en IEEE754) : le resultat de la somme depend de
// l'ordre, donc d'un ordre device il differe de la boucle hote au dernier bit.
//   - Kokkos : Kokkos::Sum / Kokkos::Max, reduction par tuile DETERMINISTE (pas
//     d'atomics flottants), donc deux appels sur des donnees identiques rendent
//     exactement le meme bit -> l'idempotence (sum_unchanged) tient.
//   - OpenMP : on NE prend PAS reduction(+:acc). Le repo garantit OpenMP
//     bit-identique a la serie ; or reduction(+:) reordonne la somme par thread
//     et casserait cette garantie. On garde donc la boucle hote sequentielle,
//     identique a la serie, pour OpenMP comme pour le backend serie.
//   - Serie : boucle hote sequentielle, SEUL chemin bit-identique a l'ancien sum.
// Bilan : sum n'est plus bit-identique a la boucle hote UNIQUEMENT sous Kokkos ;
// serie et OpenMP restent exacts. norm_inf (un max) est exact partout (max et
// fabs sont sans arrondi et associatifs/commutatifs en IEEE754).

template <class F>
Real for_each_cell_reduce_sum(const Box2D& b, F f) {
#if defined(ADC_HAS_KOKKOS)
  Real result = 0;
  Kokkos::parallel_reduce(
      "adc_reduce_sum",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<int>>(
          {b.lo[0], b.lo[1]}, {b.hi[0] + 1, b.hi[1] + 1}),
      KOKKOS_LAMBDA(int i, int j, Real& acc) { acc += f(i, j); },
      Kokkos::Sum<Real>{result});
  return result;  // bloquant cote hote : valide au retour, sans device_fence()
#else
  // Serie ET OpenMP : meme ordre lexicographique que l'ancien sum (j externe, i
  // interne). Bit-identique a la boucle hote, donc OpenMP reste identique a la
  // serie (pas de reduction(+:) qui reordonnerait la somme par thread).
  Real acc = 0;
  for (int j = b.lo[1]; j <= b.hi[1]; ++j)
    for (int i = b.lo[0]; i <= b.hi[0]; ++i) acc += f(i, j);
  return acc;
#endif
}

template <class F>
Real for_each_cell_reduce_max(const Box2D& b, F f) {
#if defined(ADC_HAS_KOKKOS)
  Real result = 0;
  Kokkos::parallel_reduce(
      "adc_reduce_max",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<int>>(
          {b.lo[0], b.lo[1]}, {b.hi[0] + 1, b.hi[1] + 1}),
      KOKKOS_LAMBDA(int i, int j, Real& acc) {
        const Real v = f(i, j);
        if (v > acc) acc = v;
      },
      Kokkos::Max<Real>{result});
  return result;  // max exact (associatif/commutatif IEEE754), pas de fence
#else
  // Serie ET OpenMP : max sequentiel. Exact dans tous les cas (max invariant par
  // reordonnancement IEEE754), donc bit-identique a l'ancien norm_inf partout.
  Real acc = 0;
  for (int j = b.lo[1]; j <= b.hi[1]; ++j)
    for (int i = b.lo[0]; i <= b.hi[0]; ++i) acc = std::max(acc, f(i, j));
  return acc;
#endif
}

}  // namespace adc
