/// @file
/// @brief Allocateur du stockage Fab2D, selectionnable a la compilation.
///
/// Deux strategies selon le build :
///   - Kokkos (ADC_HAS_KOKKOS) : `ManagedAllocator<T>` backed par `ManagedArena` (pool de
///     blocs en memoire unifiee Kokkos::SharedSpace). std::vector<T, ManagedAllocator<T>>
///     conserve la semantique valeur (copie profonde dans une nouvelle allocation managee).
///   - CPU pur : `std::allocator<T>` -- byte-identique a l'ancien comportement.
///
/// `fab_allocator<T>` est l'alias canonique a utiliser; ne pas instancier ManagedAllocator
/// directement dans le code numerique.
///
/// INVARIANT securite async : un bloc libere passe dans `pending_` (ManagedArena) et n'est
/// reutilise qu'apres la prochaine Kokkos::fence en lot. Cela reproduit la barriere implicite
/// de cudaFree de facon portable. Voir ManagedArena::deallocate et ManagedArena::allocate.

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

/// Statistiques du pool ManagedArena : hits/misses/fences et octets retenus.
/// Lecture sans effets de bord via adc::arena_stats() (snapshot sous verrou).
struct ArenaStats {
  long hits = 0;       // allocations servies par le pool (pas de kokkos_malloc)
  long misses = 0;     // allocations ayant declenche un kokkos_malloc<SharedSpace>
  long fences = 0;     // barrieres Kokkos::fence en lot (recyclage des blocs en attente)
  std::size_t reserved_bytes = 0;  // total managee detenu par le pool
};
}  // namespace adc

#if defined(ADC_HAS_KOKKOS)
#include <adc/core/kokkos_env.hpp>  // detail::ensure_kokkos_initialized : Kokkos init AVANT kokkos_malloc
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
/// Pool de memoire unifiee (Kokkos::SharedSpace) avec free-list par taille (octets).
///
/// Singleton (ManagedArena::instance()). Stateless du point de vue de std::allocator :
/// tous les ManagedAllocator<T> partagent le meme pool, l'operateur == renvoie true.
///
/// INVARIANT async : un bloc rendu par deallocate() passe en `pending_` et n'est
/// reutilise qu'apres Kokkos::fence() en lot (dans allocate(), si la free-list ready_
/// est vide). Un bloc de `ready_` a donc forcement ete vu finalize par le device.
///
/// Cycle de vie : les blocs ne sont jamais retournes a Kokkos au fil du calcul; seul
/// release_all() (hook Kokkos::finalize) les rend via kokkos_free. Ne pas appeler
/// release_all() manuellement.
class ManagedArena {
 public:
  // Singleton de DUREE DE VIE PROCESSUS, JAMAIS detruit (fuite intentionnelle d'un objet et de
  // ses tables -- l'OS reclame tout a l'exit). POURQUOI pas une statique locale ordinaire (bug
  // reel, issue #271, gdb sur CI glibc) : instance() est inline et le module pybind11 _adc est
  // compile en visibilite HIDDEN -> chaque loader .so du DSL (dlopen, jamais decharge) possede SA
  // copie de la statique (verifie par LD_DEBUG=bindings : tous les symboles ManagedArena du .so
  // se lient au .so lui-meme). A l'exit, les destructeurs de ces copies (enregistres TARD, donc
  // executes TOT, LIFO) detruisaient les tables AVANT le Kokkos::finalize de l'atexit du module
  // (enregistre tot, execute tard), dont les finalize hooks rappelaient release_all() sur des
  // arenes DETRUITES -> frees de pointeurs poubelle -> "free(): corrupted unsorted chunks" /
  // SIGSEGV au teardown. Avec le singleton jamais detruit, l'instance reste valide a TOUT moment
  // de la fin du process : plus aucune dependance a l'ordre des handlers d'exit. Les blocs du
  // pool restent rendus a Kokkos par release_all (hook de finalize) ; seules les TABLES (maps de
  // pointeurs) sont fuites, par construction.
  static ManagedArena& instance() {
    static ManagedArena* a = new ManagedArena();
    return *a;
  }

  void* allocate(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    // CRUCIAL : un Fab peut etre construit AVANT tout for_each (donc avant l'init paresseuse cote
    // kernel). kokkos_malloc exige Kokkos initialise -> on garantit l'init ICI aussi. Sans cela, le
    // build Kokkos plante des la 1ere allocation (regression identifiee sur build-kokkos). Hors lock.
    detail::ensure_kokkos_initialized();
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
  std::unordered_map<std::size_t, std::vector<void*>> pending_;  // liberes, a drainer
  long pending_count_ = 0;
  long hits_ = 0, misses_ = 0, fences_ = 0;
  std::size_t reserved_ = 0;
};

inline ArenaStats arena_stats() { return ManagedArena::instance().stats(); }

/// Adaptateur std::allocator_traits backed par ManagedArena.
/// Stateless : tous les ManagedAllocator<T> sont egaux (pool singleton partage).
/// Utiliser l'alias `fab_allocator<T>` plutot que ce template directement.
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

// Allocateur des TAMPONS DE COMMUNICATION MPI (sbuf/rbuf de fill_boundary), DISTINCT de
// fab_allocator : surtout PAS de memoire managee/device ici. Une MPI CUDA-aware (sur ROMEO :
// OpenMPI 4.1.7, PML ob1 + BTL smcuda) detecte un pointeur device/managed (cuPointerGetAttribute)
// et tente un transfert device->device par CUDA IPC (cuIpcOpenMemHandle). Sous isolation cgroup GPU
// (srun --gpus-per-task=1, chaque rang ne voit QUE son GPU comme device 0), l'IPC handle exporte par
// le pair pointe un GPU invisible -> l'ouverture ne peut pas aboutir -> DEADLOCK du rendez-vous.
// On alloue donc en memoire HOTE EPINGLEE (Kokkos::SharedHostPinnedSpace) : accessible du device
// (les kernels pack/unpack for_each ecrivent dedans directement, comme SharedSpace) MAIS vue comme
// memoire HOTE par MPI -> chemin hote normal, JAMAIS d'IPC, robuste quel que soit l'env de lancement.
// Backend hote Kokkos : SharedHostPinnedSpace == HostSpace (rien ne change). Voir fill_boundary.hpp.
static_assert(Kokkos::has_shared_host_pinned_space,
              "adc : le backend Kokkos doit fournir SharedHostPinnedSpace (memoire hote epinglee) "
              "pour les tampons de communication MPI de fill_boundary");

/// Adaptateur std::allocator_traits sur Kokkos::SharedHostPinnedSpace (hote epingle, device-accessible).
/// Stateless. Utiliser l'alias `comm_allocator<T>`. PAS de pool a free differe (a la difference de
/// ManagedArena) : fill_boundary_end pose donc un device_fence() apres l'unpack avant que les tampons
/// ne soient liberes (les kernels d'unpack lisent rbuf de facon asynchrone).
template <class T>
struct PinnedAllocator {
  using value_type = T;
  PinnedAllocator() noexcept = default;
  template <class U>
  PinnedAllocator(const PinnedAllocator<U>&) noexcept {}

  T* allocate(std::size_t n) {
    if (n == 0) return nullptr;
    detail::ensure_kokkos_initialized();  // kokkos_malloc exige Kokkos initialise
    void* p = Kokkos::kokkos_malloc<Kokkos::SharedHostPinnedSpace>("adc_comm", n * sizeof(T));
    if (!p) throw std::bad_alloc();
    return static_cast<T*>(p);
  }
  void deallocate(T* p, std::size_t) noexcept {
    if (p) Kokkos::kokkos_free<Kokkos::SharedHostPinnedSpace>(p);
  }
};
template <class A, class B>
bool operator==(const PinnedAllocator<A>&, const PinnedAllocator<B>&) noexcept { return true; }
template <class A, class B>
bool operator!=(const PinnedAllocator<A>&, const PinnedAllocator<B>&) noexcept { return false; }

template <class T>
using comm_allocator = PinnedAllocator<T>;

}  // namespace adc

#else

namespace adc {
template <class T>
using fab_allocator = std::allocator<T>;

// Hors Kokkos : tampons MPI en memoire hote ordinaire (build CPU inchange, byte-identique a avant).
template <class T>
using comm_allocator = std::allocator<T>;

// Stub hors memoire unifiee : pas de pool, pas de stats (build CPU inchange).
inline ArenaStats arena_stats() { return ArenaStats{}; }
}  // namespace adc

#endif
