/// @file
/// @brief for_each_cell et reductions : le SEAM de parallelisme sur les cellules d'une Box2D ;
///        sync_host / sync_device : le seam de COHERENCE de residence (pendant pour les acces hote).
///
/// KOKKOS EST LE SEUL backend on-node : ce seam ne compile QUE sous ADC_HAS_KOKKOS (cf. CMake, qui
/// rend Kokkos obligatoire ; sans lui, #error plus bas). Le fonctor est pris PAR VALEUR et recoit
/// (i, j) ; il capture par valeur des handles Array4 (POD), jamais le Fab ni rien de virtuel :
/// exactement la contrainte d'un kernel device. La cible on-node (sequentiel = Kokkos Serial, CPU
/// multi-thread = Kokkos OpenMP, GPU = Kokkos Cuda/HIP) est choisie A L'INSTALLATION DE KOKKOS, pas
/// par un drapeau adc : un seul appel for_each_cell (Kokkos::parallel_for sur MDRangePolicy<Rank<2>>)
/// couvre les trois. Le passage CPU -> GPU ne change donc PAS les sites d'appel.
/// CHOIX FP : la reduction SOMME (Kokkos::Sum) reassocie l'addition par tuile -> DETERMINISTE par
/// tuile (idempotent : memes donnees, meme backend -> memes bits) mais PAS bit-identique a une somme
/// lexicographique ; ceci vaut pour Serial, OpenMP et Cuda (un seul chemin, Kokkos). La reduction MAX
/// est exacte partout (max associatif/commutatif en IEEE754). sync_host() = device_fence() cible
/// avant un acces hote ; sync_device() = no-op sous memoire unifiee (scaffolding pour un futur chemin
/// non unifie).

#pragma once

#include <adc/core/kokkos_env.hpp>  // detail::ensure_kokkos_initialized + device_fence (cycle de vie)
#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>

#include <cstdlib>      // getenv / strtol : seuil de bascule serie surchargeable (#165)
#include <type_traits>  // std::is_same_v : garde compile-time exec space hote vs device (#165)

#ifndef ADC_HAS_KOKKOS
// adc_cpp est KOKKOS-ONLY : il n'existe plus de backend OpenMP autonome ni de boucle hote manuelle
// comme chemin de production. Configurer avec -DADC_USE_KOKKOS=ON (+ -DKokkos_ROOT=...) ; le serie
// passe par une install Kokkos avec Kokkos_ENABLE_SERIAL=ON.
#error "adc_cpp is Kokkos-only: for_each_cell requires ADC_HAS_KOKKOS. Configure with -DADC_USE_KOKKOS=ON and a Kokkos Serial/OpenMP/Cuda install."
#endif

#include <Kokkos_Core.hpp>

// for_each_cell : le seam de parallelisme sur les cellules d'une Box2D. Le
// fonctor est pris par valeur et recoit (i, j) ; il capture par valeur des
// handles Array4 (POD), jamais le Fab ni rien de virtuel : exactement la
// contrainte d'un kernel device.
//
// Backend UNIQUE : Kokkos (ADC_HAS_KOKKOS obligatoire). Kokkos::parallel_for sur
// MDRangePolicy<Rank<2>> s'execute sur l'espace par defaut de l'install Kokkos :
//   - Kokkos Serial  : sequentiel mono-thread (Kokkos_ENABLE_SERIAL=ON) ;
//   - Kokkos OpenMP  : CPU multi-thread       (Kokkos_ENABLE_OPENMP=ON) ;
//   - Kokkos Cuda/HIP: GPU (p.ex. Cuda sur GH200, Kokkos_ENABLE_CUDA=ON).
// Le fonctor doit etre device-callable (lambda annotee ADC_HD, capture POD par
// valeur) et operer sur une donnee device-residente : c'est le MEME appel
// for_each_cell quel que soit l'espace, CPU comme GPU.
//
// Le passage CPU -> GPU ne change donc PAS les sites d'appel : on change l'espace
// d'execution a l'install Kokkos, les operateurs (assemble_rhs, coupleurs, demos)
// restent inchanges.

namespace adc {

// detail::ensure_kokkos_initialized() et device_fence() : definis dans adc/core/kokkos_env.hpp
// (cycle de vie Kokkos partage avec l'allocateur unifie, qui doit aussi initialiser Kokkos AVANT
// son premier kokkos_malloc, sans quoi le build Kokkos plante a la construction d'un Fab).

// SEUIL DE BASCULE SERIE pour for_each_cell (#165). Sous un espace d'execution Kokkos HOTE
// (Serial/OpenMP), lancer un Kokkos::parallel_for(MDRangePolicy) sur une box minuscule paie un
// fork/join (et la construction de la politique) qui ECRASE le calcul utile. Le V-cycle
// multigrille descend jusqu'a des grilles ~2x2/4x4 ; sur ces niveaux le smoother GS, le residu,
// la restriction/prolongation et les copies enchainent des dizaines de parallel_for sur quelques
// cellules, et ce surcout de lancement DOMINE le temps de solve. En-dessous de ce seuil on execute
// une boucle hote SEQUENTIELLE (interne au chemin Kokkos, ce n'est PAS un backend separe),
// au-dessus on garde Kokkos parallel_for pour les grilles fines.
//
// BIT-IDENTITE. for_each_cell n'a AUCUNE dependance inter-iteration : chaque f(i, j)
// ecrit la seule cellule (i, j) de sa destination et lit des cellules QU'IL N'ECRIT PAS
// dans le meme appel (le smoother GS est ROUGE-NOIR colore -- une couleur ne lit que
// l'autre ; residu/restriction/prolongation/copies/saxpy ecrivent une destination
// distincte de la source). Le resultat est donc INDEPENDANT DE L'ORDRE de parcours :
// la boucle sequentielle rend exactement les memes bits que MDRangePolicy<Rank<2>>.
// Le seuil ne touche QUE for_each_cell (pas les reductions for_each_cell_reduce_* :
// la somme parallele Kokkos reassocie l'addition, donc y basculer en serie NE serait
// PAS bit-identique -- on les laisse intactes ; le max est exact mais le smoother, lui,
// passe bien par for_each_cell, ou se concentre le surcout des petites grilles).
//
// Surchargeable a l'execution par ADC_FOREACH_SERIAL_THRESHOLD (lu une fois) pour
// rebalayer le seuil sans recompiler ; defaut 4096 (meme arbitrage fork/join vs calcul
// que l'ancienne clause if() du chemin OpenMP retire).
namespace detail {
inline long foreach_serial_threshold() {
  static const long thr = [] {
    if (const char* e = std::getenv("ADC_FOREACH_SERIAL_THRESHOLD")) {
      char* end = nullptr;
      const long v = std::strtol(e, &end, 10);
      if (end != e && v >= 0) return v;
    }
    return 4096L;
  }();
  return thr;
}
}  // namespace detail

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
//                      les acces hote sont alors sans course (data race).
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
// device_fence() cible (attend les kernels en vol).
/// Rend la residence HOTE valide avant un acces hote (lecture/ecriture depuis l'hote). Sous memoire
/// unifiee = un device_fence() cible (attend les kernels en vol).
inline void sync_host() { device_fence(); }

// Marque une residence DEVICE (kernel a venir). Sous memoire unifiee : no-op
// (les ecritures hote sont deja visibles du device, rien a drainer). Existe pour
// documenter l'intention au site d'appel et accueillir un futur deep_copy
// hote->device sur un chemin non unifie.
/// Marque une residence DEVICE (kernel a venir). Sous memoire unifiee : NO-OP (les ecritures hote
/// sont deja visibles du device) ; existe pour documenter l'intention et accueillir un futur
/// deep_copy hote->device sur un chemin non unifie.
inline void sync_device() {}

/// Applique @p f a CHAQUE cellule (i, j) de la box @p b (bornes incluses), via Kokkos::parallel_for
/// (Serial / OpenMP / Cuda selon l'install Kokkos). @p f est pris par valeur et DOIT etre
/// device-callable (annote ADC_HD, capture des POD par valeur). Aucune garantie d'ordre.
template <class F>
void for_each_cell(const Box2D& b, F f) {
  // PETITES BOITES (#165) : sous un espace d'execution Kokkos HOTE (Serial/OpenMP), le
  // fork/join d'un parallel_for sur une grille minuscule (niveaux grossiers du V-cycle,
  // ~2x2..32x32) ecrase le calcul. On execute alors une boucle hote sequentielle (interne
  // au chemin Kokkos). BIT-IDENTIQUE : aucune dependance inter-iteration (cf. note du seuil),
  // donc l'ordre n'affecte aucun bit.
  //
  // GARDE DEVICE (if constexpr) : la bascule serie n'est prise QUE si l'espace d'execution
  // par defaut de Kokkos EST l'espace hote (Serial/OpenMP). Sous un espace DEVICE (Cuda
  // sur GH200), DefaultExecutionSpace != DefaultHostExecutionSpace : la boucle hote
  // tournerait sur le CPU pendant que les kernels device precedents sont en vol (pas de
  // fence pose ici) -- course de donnees. On garde donc parallel_for sur device QUELLE QUE
  // SOIT la taille -> chemin GPU STRICTEMENT inchange (le if constexpr s'evapore a la
  // compilation, zero surcout). Sous SharedSpace + execution hote, la boucle est sans
  // course : les seams de coherence existants (gs_rb_sweep pose ses device_fence autour des
  // sweeps, sync_host avant les acces hote) restent en place et inchanges.
  if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace,
                               Kokkos::DefaultHostExecutionSpace>) {
    const long n_cells = static_cast<long>(b.hi[0] - b.lo[0] + 1) *
                         (b.hi[1] - b.lo[1] + 1);
    if (n_cells < detail::foreach_serial_threshold()) {
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i) f(i, j);
      return;
    }
  }
  detail::ensure_kokkos_initialized();
  // IndexType<int> : indices SIGNES. Les boites de ghosts ont des bornes basses
  // negatives (p.ex. lo = -ng pour copy_shifted) ; sans type signe explicite,
  // MDRangePolicy rejette la borne -1 (conversion implicite jugee non sure).
  Kokkos::parallel_for(
      "adc_for_each_cell",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<int>>(
          {b.lo[0], b.lo[1]}, {b.hi[0] + 1, b.hi[1] + 1}),
      f);
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
// l'ordre de parcours.
//   - SOMME : Kokkos::Sum, reduction par tuile DETERMINISTE (pas d'atomics
//     flottants). Deux appels sur des donnees identiques rendent exactement le
//     meme bit -> l'idempotence (sum_unchanged) tient. Mais l'ordre par tuile
//     DIFFERE d'une somme lexicographique : la valeur n'est PAS bit-identique a
//     une boucle (i, j) ecrite a la main. Comme Kokkos est le seul backend, ceci
//     vaut pour TOUS les espaces (Serial, OpenMP, Cuda).
//   - MAX : Kokkos::Max, exact partout (le max est associatif/commutatif et sans
//     arrondi en IEEE754) -> bit-identique entre espaces Kokkos.
// Bilan : la SOMME est deterministe-par-tuile (idempotente) mais reassociee ; le
// MAX (norm_inf) est exact.

/// Reduction SOMME de @p f(i, j) sur la box @p b. @p f device-callable (ADC_HD) renvoyant la valeur
/// a accumuler. ATTENTION FP : Kokkos::Sum reassocie la somme par tuile (deterministe/idempotent mais
/// non bit-identique a une somme lexicographique), pour tous les espaces Kokkos. Bloquant cote hote.
template <class F>
Real for_each_cell_reduce_sum(const Box2D& b, F f) {
  detail::ensure_kokkos_initialized();
  Real result = 0;
  Kokkos::parallel_reduce(
      "adc_reduce_sum",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<int>>(
          {b.lo[0], b.lo[1]}, {b.hi[0] + 1, b.hi[1] + 1}),
      KOKKOS_LAMBDA(int i, int j, Real& acc) { acc += f(i, j); },
      Kokkos::Sum<Real>{result});
  return result;  // bloquant cote hote : valide au retour, sans device_fence()
}

/// Reduction MAX de @p f(i, j) sur la box @p b. @p f device-callable (ADC_HD). EXACT partout (le max
/// est associatif/commutatif en IEEE754, sans arrondi) -> bit-identique entre espaces Kokkos. Bloquant.
template <class F>
Real for_each_cell_reduce_max(const Box2D& b, F f) {
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
}

// Variante MAX a FONCTEUR REDUCTEUR : @p f est passe DIRECTEMENT a Kokkos::parallel_reduce et
// recoit (i, j, Real& acc) pour mettre acc a jour (acc = max(acc, valeur)). A la difference de
// for_each_cell_reduce_max, AUCUNE lambda etendue n'enveloppe @p f : c'est le chemin device-clean
// pour un noyau Model-template instancie depuis une UNITE DE TRADUCTION EXTERNE (add_compiled_model),
// ou nvcc n'emet pas fiablement une lambda etendue (cf. les foncteurs nommes de spatial_operator.hpp).
// Determinisme et bit-exactitude IDENTIQUES a for_each_cell_reduce_max (meme Kokkos::Max) : seul le
// porteur du calcul change (foncteur nomme au lieu d'un wrapper lambda).
/// Reduction MAX a FONCTEUR REDUCTEUR : @p f recoit (i, j, Real& acc) et met acc a jour, passe
/// DIRECTEMENT a Kokkos::parallel_reduce sans lambda d'enveloppe (chemin device-clean pour un noyau
/// instancie cross-TU). Bit-exactitude identique a for_each_cell_reduce_max.
template <class F>
Real reduce_max_cell(const Box2D& b, F f) {
  detail::ensure_kokkos_initialized();
  Real result = 0;
  Kokkos::parallel_reduce(
      "adc_reduce_max_cell",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<int>>(
          {b.lo[0], b.lo[1]}, {b.hi[0] + 1, b.hi[1] + 1}),
      f, Kokkos::Max<Real>{result});
  return result;
}

// Variante MIN : pendant exact de reduce_max_cell pour Kokkos::Min (diagnostic dt_hotspot,
// ADC-182 : reduction du plus petit indice encode parmi les cellules qui egalent le max).
template <class F>
Real reduce_min_cell(const Box2D& b, F f) {
  detail::ensure_kokkos_initialized();
  Real result = 0;
  Kokkos::parallel_reduce(
      "adc_reduce_min_cell",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<int>>(
          {b.lo[0], b.lo[1]}, {b.hi[0] + 1, b.hi[1] + 1}),
      f, Kokkos::Min<Real>{result});
  return result;
}

// Variante SOMME a FONCTEUR REDUCTEUR : pendant exact de reduce_max_cell pour Kokkos::Sum. @p f
// recoit (i, j, Real& acc) et accumule (acc += valeur), passe DIRECTEMENT a parallel_reduce SANS
// lambda etendue d'enveloppe (a la difference de for_each_cell_reduce_sum, qui en pose une). C'est
// le chemin device-clean exige par un noyau instancie depuis une UNITE DE TRADUCTION EXTERNE (le
// solveur de Krylov tire du harness/loader natif) : nvcc n'emet pas fiablement une lambda etendue
// premiere-instanciee cross-TU (cf. les foncteurs nommes de mf_arith.hpp / spatial_operator.hpp).
// Determinisme et FP IDENTIQUES a for_each_cell_reduce_sum : meme Kokkos::Sum deterministe par tuile.
/// Reduction SOMME a FONCTEUR REDUCTEUR : @p f recoit (i, j, Real& acc) et accumule, passe DIRECTEMENT
/// a Kokkos::parallel_reduce sans lambda d'enveloppe (chemin device-clean cross-TU). Memes garanties
/// FP que for_each_cell_reduce_sum (Kokkos::Sum reassocie par tuile, deterministe/idempotent).
template <class F>
Real reduce_sum_cell(const Box2D& b, F f) {
  detail::ensure_kokkos_initialized();
  Real result = 0;
  Kokkos::parallel_reduce(
      "adc_reduce_sum_cell",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<int>>(
          {b.lo[0], b.lo[1]}, {b.hi[0] + 1, b.hi[1] + 1}),
      f, Kokkos::Sum<Real>{result});
  return result;
}

}  // namespace adc
