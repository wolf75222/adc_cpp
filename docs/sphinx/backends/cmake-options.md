# CMake options


Verified in [`CMakeLists.txt`](../../../CMakeLists.txt):

| CMake option | Effect | Default |
|--------------|-------|--------|
| `ADC_USE_KOKKOS` | Only on-node backend, required (CPU Serial/OpenMP + GPU Cuda/HIP). Configuring with `OFF` is a fatal CMake error. | `ON` |
| `ADC_USE_MPI` | Distributed comm seam (`comm.hpp` -> MPI collectives). | `OFF` |
| `ADC_BUILD_PYTHON` | Python `adc` module (pybind11), serial only. | `OFF` |

The Kokkos sub-backend (Serial / OpenMP / Cuda) is not an `adc_cpp` option: it is
chosen at the moment Kokkos is installed (`Kokkos_ENABLE_SERIAL`, `Kokkos_ENABLE_OPENMP`,
`Kokkos_ENABLE_CUDA` + `Kokkos_ARCH_HOPPER90`), then pointed at by `-DKokkos_ROOT=...`. That is what
distinguishes configurations 1/2/5 below.

Notes:

- Kokkos is the only on-node backend and it is required: configuring without it
  (`-DADC_USE_KOKKOS=OFF`) is a fatal CMake error, and the
  [`for_each.hpp`](https://github.com/wolf75222/adc_cpp/blob/master/include/adc/mesh/execution/for_each.hpp)
  seam does not compile without `ADC_HAS_KOKKOS` (`#error`).
- **Kokkos does not need to be pre-installed**: CMake does `find_package(Kokkos)` then, as a fallback,
  fetches + builds it via FetchContent (version `ADC_KOKKOS_FETCH_VERSION`, default 4.4.01, tarball verified by SHA256). The
  `-DKokkos_ROOT=...` commands below reuse an install (faster); without them, Kokkos is fetched.
- The C++ standard is C++20 (nvcc CUDA 12.x does not offer `-std=c++23`).
