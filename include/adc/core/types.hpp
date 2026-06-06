#pragma once

/// @file
/// @brief Types scalaires de base et macro ADC_HD (portabilite host+device). Socle minimal
///        sans dependance externe; la bascule vers pde_core::Real attendra le maillage distribue.
///
/// `Real` : alias double centralise. Tout le calcul numerique l'utilise; ne pas ecrire `double`
/// directement dans la couche physique ou les kernels.
///
/// `ADC_HD` : annotation portant les fonctions appelees dans les kernels Kokkos sur host ET device.
/// - Kokkos  : KOKKOS_FUNCTION (portable Cuda/HIP/SYCL/CPU, sans syntaxe CUDA manuelle).
///   KOKKOS_FUNCTION est prefere a KOKKOS_INLINE_FUNCTION pour ne pas ajouter d'`inline` implicite
///   sur les sites deja notes `ADC_HD inline ...`.
/// - CUDA/HIP directs (sans Kokkos) : __host__ __device__.
/// - CPU pur : expansion vide.
/// INVARIANT : ADC_HD ne peut entourer que du code device-clean (pas d'objet hote,
/// pas de std::vector, pas de vtable).

// Types scalaires de base. Volontairement local et minimal pour garder le
// premier socle sans dependance externe. La bascule vers pde_core::Real
// (partage avec advection_cpp / euler_cpp / poisson_cpp) se fera quand le
// maillage distribue arrivera, pas avant.

// ADC_HD : annotation host+device pour les accesseurs appeles dans les kernels.
// Sous Kokkos on delegue a KOKKOS_FUNCTION (le pendant PORTABLE de __host__ __device__ :
// Cuda, HIP, SYCL... le meme code source vise CPU et GPU selon le backend) -> aucune syntaxe
// CUDA ecrite a la main. KOKKOS_FUNCTION (et non KOKKOS_INLINE_FUNCTION) reproduit exactement
// l'ancien comportement (pas de 'inline' ajoute, pour les sites ecrits 'ADC_HD inline ...').
// Hors Kokkos : repli compilateur device, sinon vide (build CPU inchange).
#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Macros.hpp>
#define ADC_HD KOKKOS_FUNCTION
#elif defined(__CUDACC__) || defined(__HIPCC__)
#define ADC_HD __host__ __device__
#else
#define ADC_HD
#endif

namespace adc {

using Real = double;

}  // namespace adc
