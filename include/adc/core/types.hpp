#pragma once

// Types scalaires de base. Volontairement local et minimal pour garder le
// premier socle sans dependance externe. La bascule vers pde_core::Real
// (partage avec advection_cpp / euler_cpp / poisson_cpp) se fera quand le
// maillage distribue arrivera, pas avant.

// ADC_HD : annotation host+device pour les accesseurs appeles dans les kernels
// GPU (CUDA/HIP). Vide hors compilateur device -> aucun effet sur le build CPU.
// C'est le pendant de AMREX_GPU_HOST_DEVICE / KOKKOS_FUNCTION : le meme code
// (Array4, indexation) se compile pour l'hote et le device.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define ADC_HD __host__ __device__
#else
#define ADC_HD
#endif

namespace adc {

using Real = double;

}  // namespace adc
