/// @file
/// @brief Cycle de vie Kokkos partage : init paresseuse + barriere device. Indispensable depuis que
///        l'allocateur unifie (kokkos_malloc<SharedSpace>) est appele DES la construction d'un Fab,
///        AVANT tout for_each. L'init ne peut donc plus etre confiee au seul premier kernel : c'est
///        ce meme garde qui s'applique a l'allocation ET aux kernels, pour que le build Kokkos
///        (Serial/OpenMP/Cuda) marche sans Kokkos::initialize explicite dans chaque main.
///
/// INVARIANT de sequencement : detail::ensure_kokkos_initialized() est appelee par ManagedArena
/// avant tout kokkos_malloc ; device_fence() est appelee par l'hote avant tout acces a la memoire
/// unifiee apres un kernel. Ces deux points d'entree sont les SEULS endroits ou le cycle de vie
/// Kokkos est pilote; ne pas appeler Kokkos::initialize/finalize ailleurs.

#pragma once

#ifdef ADC_HAS_KOKKOS
#include <Kokkos_Core.hpp>

#include <cstdlib>  // std::atexit
#endif

namespace adc {

#ifdef ADC_HAS_KOKKOS
namespace detail {
/// Initialise Kokkos au PREMIER besoin (allocation d'un Fab OU premier kernel), finalize via atexit.
/// No-op si l'appelant a deja fait son propre Kokkos::initialize / ScopeGuard, ou si Kokkos est deja
/// finalize. Un seul atexit est pose (les appels suivants voient is_initialized()). Sequence de
/// destruction : les MultiFab LOCAUX sont detruits a la fin de main, donc AVANT le finalize atexit.
inline void ensure_kokkos_initialized() {
  if (!Kokkos::is_initialized() && !Kokkos::is_finalized()) {
    Kokkos::initialize();
    std::atexit([] {
      if (Kokkos::is_initialized()) Kokkos::finalize();
    });
  }
}
}  // namespace detail
#endif

/// Barriere device : attend la fin des kernels en vol avant un acces HOTE a la memoire unifiee.
/// No-op hors Kokkos (et si rien n'a ete lance).
inline void device_fence() {
#ifdef ADC_HAS_KOKKOS
  if (Kokkos::is_initialized()) Kokkos::fence();
#endif
}

}  // namespace adc
