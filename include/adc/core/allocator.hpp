#pragma once

#include <cstddef>
#include <memory>

// Allocateur du stockage Fab2D, selectionnable a la compilation.
//
//   - Kokkos (ADC_HAS_KOKKOS) : memoire UNIFIEE PORTABLE via Kokkos::SharedSpace (alias
//     CudaUVMSpace / HIPManagedSpace / SYCLSharedUSMSpace / HostSpace selon le backend ; AUCUNE
//     API CUDA ecrite a la main), accessible de facon coherente depuis l'hote ET le device. Sur GH200
//     (Grace+Hopper, NVLink-C2C) c'est materiellement coherent : un meme buffer
//     sert au code hote (operator(), boucles) ET aux kernels device lances par
//     for_each_cell, sans deep_copy ni migration. Et std::vector conserve sa
//     SEMANTIQUE VALEUR (copie profonde dans une nouvelle allocation managee),
//     donc les algorithmes qui copient un Fab (reflux, average_down) restent
//     corrects.
//   - sinon : std::allocator (hote). Le build CPU est byte-identique a avant.
//
// ARENA (pool memoire) : un kokkos_malloc<SharedSpace> par petit Fab temporaire est lent
// (synchronisation, mise en place de la table de pages). Les etages d'integration
// (SSPRK) et les niveaux de la multigrille allouent/liberent en boucle des Fabs de
// MEMES tailles. ManagedArena est un cache : a la liberation on RECYCLE le bloc
// (free-list par taille en octets) au lieu de le rendre tout de suite ; a l'allocation
// on reutilise un bloc libre de la meme taille s'il existe. Apres un rodage (les
// premieres tailles distinctes), les pas suivants ne font plus aucun
// kokkos_malloc. C'est l'Arena d'AMReX / l'allocateur a cache de PyTorch. La
// semantique valeur est intacte : chaque vecteur possede un bloc distinct ; le pool
// est un singleton partage, donc l'allocateur reste sans etat (stateless).

namespace adc {
struct ArenaStats {
  long hits = 0;       // allocations servies par le pool (pas de kokkos_malloc)
  long misses = 0;     // allocations ayant declenche un kokkos_malloc<SharedSpace>
  long fences = 0;     // barrieres Kokkos::fence en lot (recyclage des blocs en attente)
  std::size_t reserved_bytes = 0;  // total managee detenu par le pool
};
}  // namespace adc

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>

#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

namespace adc {

static_assert(Kokkos::has_shared_space,
              "adc : le backend Kokkos doit fournir SharedSpace (memoire unifiee) pour le Fab "
              "device ; activer un backend Cuda/HIP/SYCL (ou un backend hote, ou SharedSpace est HostSpace)");

// Cache d'allocations en memoire unifiee (Kokkos::SharedSpace), free-list par taille (octets).
//
// SECURITE async : un kernel peut encore lire/ecrire un Fab au moment de sa destruction. Avant on
// s'appuyait sur la synchro implicite de cudaFree ; ici on reproduit cette barriere mais EN LOT et
// PORTABLE (Kokkos::fence) : un bloc libere va dans `pending_` (pas encore reutilisable) ; quand une
// allocation manque de bloc pret, un seul Kokkos::fence() draine le device et bascule TOUS les
// pending vers `ready_`. Un bloc de `ready_` a donc forcement vu sa derniere utilisation device
// terminee avant que l'hote (value-init du vector) ne le reecrive.
class ManagedArena {
 public:
  static ManagedArena& instance() {
    static ManagedArena a;
    return a;
  }

  void* allocate(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    std::lock_guard<std::mutex> lk(m_);
    std::call_once(hook_once_, [] {  // rendre les blocs a Kokkos::finalize (sinon allocation "fuitee")
      Kokkos::push_finalize_hook([] { ManagedArena::instance().release_all(); });
    });
    if (void* p = pop_ready(bytes)) return p;
    if (pending_count_ > 0) {
      Kokkos::fence();  // barriere en lot, portable (draine les kernels en vol ; ancien cudaDeviceSynchronize)
      ++fences_;
      for (auto& kv : pending_) {
        auto& r = ready_[kv.first];
        r.insert(r.end(), kv.second.begin(), kv.second.end());
        kv.second.clear();
      }
      pending_count_ = 0;
      if (void* p = pop_ready(bytes)) return p;
    }
    void* p = Kokkos::kokkos_malloc<Kokkos::SharedSpace>("adc_fab", bytes);  // memoire unifiee portable
    if (!p) throw std::bad_alloc();
    ++misses_;
    reserved_ += bytes;
    return p;
  }

  // Bloc libere : en attente (pas reutilisable avant la prochaine barriere en lot). Pas de
  // kokkos_free immediat (le pool vit jusqu'a la fin du process ; release_all rend tout a finalize).
  void deallocate(void* p, std::size_t bytes) {
    if (!p) return;
    std::lock_guard<std::mutex> lk(m_);
    pending_[bytes].push_back(p);
    ++pending_count_;
  }

  // Hook de Kokkos::finalize : libere tous les blocs (ready + pending) via kokkos_free AVANT l'arret
  // des espaces memoire, pour ne laisser aucune allocation Kokkos non rendue (le pool ne free jamais
  // en cours de route). Appele une seule fois, hors de toute allocation concurrente.
  void release_all() {
    std::lock_guard<std::mutex> lk(m_);
    for (auto& kv : ready_)
      for (void* p : kv.second) Kokkos::kokkos_free<Kokkos::SharedSpace>(p);
    for (auto& kv : pending_)
      for (void* p : kv.second) Kokkos::kokkos_free<Kokkos::SharedSpace>(p);
    ready_.clear();
    pending_.clear();
    pending_count_ = 0;
  }

  ArenaStats stats() {
    std::lock_guard<std::mutex> lk(m_);
    return ArenaStats{hits_, misses_, fences_, reserved_};
  }

 private:
  void* pop_ready(std::size_t bytes) {
    auto it = ready_.find(bytes);
    if (it != ready_.end() && !it->second.empty()) {
      void* p = it->second.back();
      it->second.pop_back();
      ++hits_;
      return p;
    }
    return nullptr;
  }

  std::mutex m_;
  std::once_flag hook_once_;  // enregistrement unique du finalize hook Kokkos
  std::unordered_map<std::size_t, std::vector<void*>> ready_;    // surs, reutilisables
  std::unordered_map<std::size_t, std::vector<void*>> pending_;  // liberes, a draîner
  long pending_count_ = 0;
  long hits_ = 0, misses_ = 0, fences_ = 0;
  std::size_t reserved_ = 0;
};

inline ArenaStats arena_stats() { return ManagedArena::instance().stats(); }

template <class T>
struct ManagedAllocator {
  using value_type = T;
  ManagedAllocator() noexcept = default;
  template <class U>
  ManagedAllocator(const ManagedAllocator<U>&) noexcept {}

  T* allocate(std::size_t n) {
    return static_cast<T*>(ManagedArena::instance().allocate(n * sizeof(T)));
  }
  void deallocate(T* p, std::size_t n) noexcept {
    ManagedArena::instance().deallocate(p, n * sizeof(T));
  }
};

template <class A, class B>
bool operator==(const ManagedAllocator<A>&, const ManagedAllocator<B>&) noexcept {
  return true;  // sans etat : tous egaux (pool singleton partage)
}
template <class A, class B>
bool operator!=(const ManagedAllocator<A>&, const ManagedAllocator<B>&) noexcept {
  return false;
}

template <class T>
using fab_allocator = ManagedAllocator<T>;

}  // namespace adc

#else

namespace adc {
template <class T>
using fab_allocator = std::allocator<T>;

// Stub hors memoire unifiee : pas de pool, pas de stats (build CPU inchange).
inline ArenaStats arena_stats() { return ArenaStats{}; }
}  // namespace adc

#endif
