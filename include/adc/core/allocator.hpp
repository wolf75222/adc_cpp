/// @file
/// @brief Fab2D storage allocator, selectable at compile time.
///
/// Two strategies depending on the build:
///   - Kokkos (ADC_HAS_KOKKOS): `ManagedAllocator<T>` backed by `ManagedArena` (a pool of
///     blocks in Kokkos::SharedSpace unified memory). std::vector<T, ManagedAllocator<T>>
///     keeps value semantics (deep copy into a new managed allocation).
///   - Pure CPU: `std::allocator<T>` -- byte-identical to the old behavior.
///
/// `fab_allocator<T>` is the canonical alias to use; do not instantiate ManagedAllocator
/// directly in numerical code.
///
/// Async safety INVARIANT: a freed block goes into `pending_` (ManagedArena) and is only
/// reused after the next batched Kokkos::fence. This reproduces the implicit barrier
/// of cudaFree in a portable way. See ManagedArena::deallocate and ManagedArena::allocate.

#pragma once

#include <cstddef>
#include <memory>

namespace adc {

/// ManagedArena pool statistics: hits/misses/fences and retained bytes.
/// Side-effect-free read via adc::arena_stats() (snapshot under lock).
struct ArenaStats {
  long hits = 0;                   // allocations served by the pool (no kokkos_malloc)
  long misses = 0;                 // allocations that triggered a kokkos_malloc<SharedSpace>
  long fences = 0;                 // batched Kokkos::fence barriers (recycling pending blocks)
  std::size_t reserved_bytes = 0;  // total managed memory held by the pool
};
}  // namespace adc

#if defined(ADC_HAS_KOKKOS)
#include <adc/core/kokkos_env.hpp>  // detail::ensure_kokkos_initialized: Kokkos init BEFORE kokkos_malloc
#include <Kokkos_Core.hpp>

#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

namespace adc {

static_assert(
    Kokkos::has_shared_space,
    "adc: the Kokkos backend must provide SharedSpace (unified memory) for the device "
    "Fab; enable a Cuda/HIP/SYCL backend (or a host backend, where SharedSpace is HostSpace)");

// Cache of unified-memory allocations (Kokkos::SharedSpace), free-list by size (bytes).
//
// Async SAFETY: a kernel may still read/write a Fab at the moment of its destruction. Previously we
// relied on the implicit synchronization of cudaFree; here we reproduce that barrier but BATCHED and
// PORTABLE (Kokkos::fence): a freed block goes into `pending_` (not yet reusable); when an
// allocation lacks a ready block, a single Kokkos::fence() drains the device and moves ALL
// pending blocks to `ready_`. A block from `ready_` has therefore necessarily had its last device
// use finished before the host (value-init of the vector) overwrites it.
/// Unified-memory pool (Kokkos::SharedSpace) with a free-list by size (bytes).
///
/// Singleton (ManagedArena::instance()). Stateless from std::allocator's point of view:
/// all ManagedAllocator<T> share the same pool, operator == returns true.
///
/// Async INVARIANT: a block returned by deallocate() goes into `pending_` and is only
/// reused after a batched Kokkos::fence() (in allocate(), if the ready_ free-list
/// is empty). A block from `ready_` has therefore necessarily been seen finalized by the device.
///
/// Lifecycle: blocks are never returned to Kokkos during the computation; only
/// release_all() (Kokkos::finalize hook) returns them via kokkos_free. Do not call
/// release_all() manually.
class ManagedArena {
 public:
  // PROCESS-LIFETIME singleton, NEVER destroyed (intentional leak of an object and of
  // its tables -- the OS reclaims everything at exit). WHY not an ordinary local static (real bug,
  // issue #271, gdb on CI glibc): instance() is inline and the pybind11 module _adc is
  // compiled with HIDDEN visibility -> each DSL .so loader (dlopen, never unloaded) has ITS
  // own copy of the static (verified with LD_DEBUG=bindings: all ManagedArena symbols of the .so
  // bind to the .so itself). At exit, the destructors of these copies (registered LATE, hence
  // run EARLY, LIFO) destroyed the tables BEFORE the module's atexit Kokkos::finalize
  // (registered early, run late), whose finalize hooks called release_all() back on
  // DESTROYED arenas -> frees of garbage pointers -> "free(): corrupted unsorted chunks" /
  // SIGSEGV at teardown. With the never-destroyed singleton, the instance stays valid at ANY moment
  // of the process shutdown: no longer any dependency on the order of exit handlers. The pool
  // blocks are still returned to Kokkos by release_all (finalize hook); only the TABLES (maps of
  // pointers) are leaked, by construction.
  static ManagedArena& instance() {
    static ManagedArena* a = new ManagedArena();
    return *a;
  }

  void* allocate(std::size_t bytes) {
    if (bytes == 0)
      return nullptr;
    // CRUCIAL: a Fab can be constructed BEFORE any for_each (hence before the lazy init on the
    // kernel side). kokkos_malloc requires Kokkos initialized -> we guarantee the init HERE too.
    // Without this, the Kokkos build crashes at the very first allocation (regression identified on
    // build-kokkos). Outside the lock.
    detail::ensure_kokkos_initialized();
    std::lock_guard<std::mutex> lk(m_);
    std::call_once(hook_once_,
                   [] {  // return the blocks at Kokkos::finalize (otherwise "leaked" allocation)
                     Kokkos::push_finalize_hook([] { ManagedArena::instance().release_all(); });
                   });
    if (void* p = pop_ready(bytes))
      return p;
    if (pending_count_ > 0) {
      Kokkos::
          fence();  // batched, portable barrier (drains in-flight kernels; former cudaDeviceSynchronize)
      ++fences_;
      for (auto& kv : pending_) {
        auto& r = ready_[kv.first];
        r.insert(r.end(), kv.second.begin(), kv.second.end());
        kv.second.clear();
      }
      pending_count_ = 0;
      if (void* p = pop_ready(bytes))
        return p;
    }
    void* p =
        Kokkos::kokkos_malloc<Kokkos::SharedSpace>("adc_fab", bytes);  // portable unified memory
    if (!p)
      throw std::bad_alloc();
    ++misses_;
    reserved_ += bytes;
    return p;
  }

  // Freed block: pending (not reusable before the next batched barrier). No
  // immediate kokkos_free (the pool lives until the end of the process; release_all returns everything at finalize).
  void deallocate(void* p, std::size_t bytes) {
    if (!p)
      return;
    std::lock_guard<std::mutex> lk(m_);
    pending_[bytes].push_back(p);
    ++pending_count_;
  }

  // Kokkos::finalize hook: frees all blocks (ready + pending) via kokkos_free BEFORE the memory
  // spaces shut down, so as to leave no unreturned Kokkos allocation (the pool never frees
  // along the way). Called only once, outside any concurrent allocation.
  void release_all() {
    std::lock_guard<std::mutex> lk(m_);
    for (auto& kv : ready_)
      for (void* p : kv.second)
        Kokkos::kokkos_free<Kokkos::SharedSpace>(p);
    for (auto& kv : pending_)
      for (void* p : kv.second)
        Kokkos::kokkos_free<Kokkos::SharedSpace>(p);
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
  std::once_flag hook_once_;  // single registration of the Kokkos finalize hook
  std::unordered_map<std::size_t, std::vector<void*>> ready_;    // safe, reusable
  std::unordered_map<std::size_t, std::vector<void*>> pending_;  // freed, to drain
  long pending_count_ = 0;
  long hits_ = 0, misses_ = 0, fences_ = 0;
  std::size_t reserved_ = 0;
};

inline ArenaStats arena_stats() {
  return ManagedArena::instance().stats();
}

/// std::allocator_traits adapter backed by ManagedArena.
/// Stateless: all ManagedAllocator<T> are equal (shared singleton pool).
/// Use the alias `fab_allocator<T>` rather than this template directly.
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
  return true;  // stateless: all equal (shared singleton pool)
}
template <class A, class B>
bool operator!=(const ManagedAllocator<A>&, const ManagedAllocator<B>&) noexcept {
  return false;
}

template <class T>
using fab_allocator = ManagedAllocator<T>;

// Allocator for the MPI COMMUNICATION BUFFERS (sbuf/rbuf of fill_boundary), DISTINCT from
// fab_allocator: definitely NO managed/device memory here. A CUDA-aware MPI (e.g.
// OpenMPI 4.1.7, PML ob1 + BTL smcuda) detects a device/managed pointer (cuPointerGetAttribute)
// and attempts a device->device transfer via CUDA IPC (cuIpcOpenMemHandle). Under GPU cgroup
// isolation (srun --gpus-per-task=1, each rank sees ONLY its GPU as device 0), the IPC handle exported by
// the peer points to an invisible GPU -> the open cannot succeed -> DEADLOCK of the rendezvous.
// So we allocate in PINNED HOST memory (Kokkos::SharedHostPinnedSpace): accessible from the device
// (the pack/unpack for_each kernels write into it directly, like SharedSpace) BUT seen as
// HOST memory by MPI -> normal host path, NEVER IPC, robust whatever the launch environment.
// Kokkos host backend: SharedHostPinnedSpace == HostSpace (nothing changes). See fill_boundary.hpp.
static_assert(Kokkos::has_shared_host_pinned_space,
              "adc: the Kokkos backend must provide SharedHostPinnedSpace (pinned host memory) "
              "for the MPI communication buffers of fill_boundary");

/// std::allocator_traits adapter over Kokkos::SharedHostPinnedSpace (pinned host, device-accessible).
/// Stateless. Use the alias `comm_allocator<T>`. NO deferred-free pool (unlike
/// ManagedArena): fill_boundary_end therefore places a device_fence() after the unpack before the buffers
/// are freed (the unpack kernels read rbuf asynchronously).
template <class T>
struct PinnedAllocator {
  using value_type = T;
  PinnedAllocator() noexcept = default;
  template <class U>
  PinnedAllocator(const PinnedAllocator<U>&) noexcept {}

  T* allocate(std::size_t n) {
    if (n == 0)
      return nullptr;
    detail::ensure_kokkos_initialized();  // kokkos_malloc requires Kokkos initialized
    void* p = Kokkos::kokkos_malloc<Kokkos::SharedHostPinnedSpace>("adc_comm", n * sizeof(T));
    if (!p)
      throw std::bad_alloc();
    return static_cast<T*>(p);
  }
  void deallocate(T* p, std::size_t) noexcept {
    if (p)
      Kokkos::kokkos_free<Kokkos::SharedHostPinnedSpace>(p);
  }
};
template <class A, class B>
bool operator==(const PinnedAllocator<A>&, const PinnedAllocator<B>&) noexcept {
  return true;
}
template <class A, class B>
bool operator!=(const PinnedAllocator<A>&, const PinnedAllocator<B>&) noexcept {
  return false;
}

template <class T>
using comm_allocator = PinnedAllocator<T>;

}  // namespace adc

#else

namespace adc {
template <class T>
using fab_allocator = std::allocator<T>;

// Outside Kokkos: MPI buffers in ordinary host memory (CPU build unchanged, byte-identical to before).
template <class T>
using comm_allocator = std::allocator<T>;

// Stub outside unified memory: no pool, no stats (CPU build unchanged).
inline ArenaStats arena_stats() {
  return ArenaStats{};
}
}  // namespace adc

#endif
