#pragma once

#include <cstddef>
#include <memory>

// Allocateur du stockage Fab2D, selectionnable a la compilation.
//
//   - CUDA (ADC_HAS_KOKKOS + __CUDACC__) : memoire UNIFIEE (cudaMallocManaged),
//     accessible de facon coherente depuis l'hote ET le device. Sur GH200
//     (Grace+Hopper, NVLink-C2C) c'est materiellement coherent : un meme buffer
//     sert au code hote (operator(), boucles) ET aux kernels device lances par
//     for_each_cell, sans deep_copy ni migration. Et std::vector conserve sa
//     SEMANTIQUE VALEUR (copie profonde dans une nouvelle allocation managee),
//     donc les algorithmes qui copient un Fab (reflux, average_down) restent
//     corrects.
//   - sinon : std::allocator (hote). Le build CPU est byte-identique a avant.

#if defined(ADC_HAS_KOKKOS) && defined(__CUDACC__)
#include <cuda_runtime.h>
#include <new>

namespace adc {

template <class T>
struct ManagedAllocator {
  using value_type = T;
  ManagedAllocator() noexcept = default;
  template <class U>
  ManagedAllocator(const ManagedAllocator<U>&) noexcept {}

  T* allocate(std::size_t n) {
    void* p = nullptr;
    if (n != 0 && cudaMallocManaged(&p, n * sizeof(T)) != cudaSuccess)
      throw std::bad_alloc();
    return static_cast<T*>(p);
  }
  void deallocate(T* p, std::size_t) noexcept {
    if (p) cudaFree(p);
  }
};

template <class A, class B>
bool operator==(const ManagedAllocator<A>&, const ManagedAllocator<B>&) noexcept {
  return true;
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
}  // namespace adc

#endif
