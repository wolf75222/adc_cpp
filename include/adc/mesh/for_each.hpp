#pragma once

#include <adc/core/kokkos_env.hpp>  // detail::ensure_kokkos_initialized + device_fence (cycle de vie)
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

// detail::ensure_kokkos_initialized() et device_fence() : definis dans adc/core/kokkos_env.hpp
// (cycle de vie Kokkos partage avec l'allocateur unifie, qui doit aussi initialiser Kokkos AVANT
// son premier kokkos_malloc, sans quoi le build Kokkos plante a la construction d'un Fab).

// ---------------------------------------------------------------------------
// Residence des donnees : sync_host() / sync_device(). Le seam de COHERENCE, le
// pendant de for_each_cell pour les acces hote.
//
// Aujourd'hui le stockage Fab vit en memoire UNIFIEE (Kokkos::SharedSpace, cf.
// allocator.hpp) : un meme buffer sert au code hote (operator(), boucles) ET
// aux kernels device. La coherence ne demande donc PAS de copie, seulement de
// l'ORDONNANCEMENT : un acces hote ne doit pas lire/ecrire un buffer pendant
// qu'un kernel async le touche encore. Jusqu'ici cet ordonnancement etait pose a
// la main par des device_fence() epars, sans jamais dire QUELLE residence on
// veut rendre valide.
//
// sync_host()/sync_device() ENCODENT cette intention :
//   - sync_host()   : "je vais lire/ecrire ces donnees DEPUIS L'HOTE ;
//                      rends-les valides cote hote". Sous SharedSpace = un
//                      device_fence() cible (attendre les kernels en vol), donc
//                      acces hote sur sont sans course.
//   - sync_device() : "je vais lire/ecrire ces donnees DEPUIS LE DEVICE
//                      (un kernel) ; rends-les valides cote device". Sous
//                      SharedSpace les ecritures hote precedentes sont visibles
//                      du device sans barriere (pas de pipeline hote async a
//                      drainer), donc c'est un VRAI no-op aujourd'hui ; la
//                      fonction existe pour MARQUER l'intention au site d'appel.
//
// SEMANTIQUE SOUS SHAREDSPACE (etat actuel) : ces appels sont au plus un fence,
// jamais une copie. Le comportement reste donc BIT-IDENTIQUE a l'ancien code
// (sync_host == l'ancien device_fence() pose avant un acces hote ; sync_device
// == rien). C'est volontairement du SCAFFOLDING : sous memoire unifiee il n'y a
// rien d'autre a faire.
//
// CHEMIN FUTUR NON UNIFIE (buffers hote/device separes + deep_copy) : c'est ICI
// que se brancherait la migration. sync_host() ferait un Kokkos::deep_copy
// device->hote (et un fence) si le device est la derniere residence ecrite ;
// sync_device() un deep_copy hote->device dans l'autre sens. Le suivi "qui
// possede la donnee a jour" (dirty flag par residence) vivrait sur le MultiFab,
// pas ici : ce seam reste sans etat, les surcharges MultiFab portent l'etat.
// Comme tous les sites d'acces hote passent deja par sync_host(), basculer vers
// ce chemin ne touchera PAS les operateurs, exactement comme for_each_cell
// isole le passage CPU -> GPU des sites d'appel.

// Rend la residence HOTE valide avant un acces hote. Sous memoire unifiee : un
// device_fence() cible (attend les kernels en vol). No-op hors Kokkos.
inline void sync_host() { device_fence(); }

// Marque une residence DEVICE (kernel a venir). Sous memoire unifiee : no-op
// (les ecritures hote sont deja visibles du device, rien a drainer). Existe pour
// documenter l'intention au site d'appel et accueillir un futur deep_copy
// hote->device sur un chemin non unifie.
inline void sync_device() {}

template <class F>
void for_each_cell(const Box2D& b, F f) {
#if defined(ADC_HAS_KOKKOS)
  detail::ensure_kokkos_initialized();
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
  detail::ensure_kokkos_initialized();
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
  detail::ensure_kokkos_initialized();
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

// Variante MAX a FONCTEUR REDUCTEUR : @p f est passe DIRECTEMENT a Kokkos::parallel_reduce et
// recoit (i, j, Real& acc) pour mettre acc a jour (acc = max(acc, valeur)). A la difference de
// for_each_cell_reduce_max, AUCUNE lambda etendue n'enveloppe @p f : c'est le chemin device-clean
// pour un noyau Model-template instancie depuis une UNITE DE TRADUCTION EXTERNE (add_compiled_model),
// ou nvcc n'emet pas fiablement une lambda etendue (cf. les foncteurs nommes de spatial_operator.hpp).
// Determinisme et bit-exactitude IDENTIQUES a for_each_cell_reduce_max (meme Kokkos::Max, meme boucle
// hote sequentielle) : seul le porteur du calcul change (foncteur nomme au lieu d'un wrapper lambda).
template <class F>
Real reduce_max_cell(const Box2D& b, F f) {
#if defined(ADC_HAS_KOKKOS)
  detail::ensure_kokkos_initialized();
  Real result = 0;
  Kokkos::parallel_reduce(
      "adc_reduce_max_cell",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<int>>(
          {b.lo[0], b.lo[1]}, {b.hi[0] + 1, b.hi[1] + 1}),
      f, Kokkos::Max<Real>{result});
  return result;
#else
  Real acc = 0;
  for (int j = b.lo[1]; j <= b.hi[1]; ++j)
    for (int i = b.lo[0]; i <= b.hi[0]; ++i) f(i, j, acc);
  return acc;
#endif
}

}  // namespace adc
