#pragma once

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
