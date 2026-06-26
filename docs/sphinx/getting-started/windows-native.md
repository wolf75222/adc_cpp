# Native Windows (without WSL2)

> **Status: native Windows works for the CPU** -- Serial, OpenMP, the `pops` Python module
> (`import pops`, `System`, `AmrSystem`), **DSL production** (`.dll`, bit-identical results) and
> compilation of custom C++ (`adc_cases.common.native`). **The GPU stays on [WSL2](windows-wsl2.md)**
> (Kokkos CUDA is not supported natively on Windows).

This page complements the [WSL2](windows-wsl2.md) guide. WSL2 remains the simplest path and the only
one that covers the **GPU**. The native path is useful if you want to avoid WSL2 for CPU work.

## Toolchain

- **Visual Studio 2022** (MSVC, `cl.exe`, C++23 via `/std:c++20`/`/std:c++latest`) + **CMake**.
- **LLVM/clang-cl** (`winget install LLVM.LLVM`) **only for OpenMP**: MSVC keeps
  `_OPENMP=2.0` even with `/openmp:llvm`, so Kokkos OpenMP (>=3.0) requires clang-cl (which reports 5.1).
- **Python** (Miniforge recommended) + `numpy`, `pybind11`.
- Any compilation at runtime (DSL `production`, `adc_cases.common.native`) launches `cl`/`clang-cl`:
  launch Python from an **"x64 Native Tools"** prompt (vcvars) to have `INCLUDE`/`LIB`.

## Kokkos as a shared DLL (key to the native path)

The DSL and `adc_cases.common.native` generate a `.dll` that shares the Kokkos runtime of the `_pops`
module: you therefore need **Kokkos as a shared library** (a single runtime, otherwise double-singleton ->
crash). MSVC does not export the Kokkos symbols by default -> use `WINDOWS_EXPORT_ALL_SYMBOLS`:

```bat
:: Serial (CPU sequentiel) -- compilateur cl
cmake -S kokkos -B build-kokkos -G "Visual Studio 17 2022" -A x64 ^
  -DKokkos_ENABLE_SERIAL=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON ^
  -DCMAKE_INSTALL_PREFIX=kokkos-shared
cmake --build build-kokkos --config Release --target install

:: + OpenMP (CPU parallele) -- compilateur clang-cl (dans un invite vcvars), generateur Ninja
cmake -S kokkos -B build-omp -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang-cl ^
  -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=ON ^
  -DBUILD_SHARED_LIBS=ON -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON -DCMAKE_INSTALL_PREFIX=kokkos-omp
cmake --build build-omp --target install
```

`kokkos*.dll` must be loadable at runtime: place it **next to the `.pyd`** (Python 3.8+
no longer uses the `PATH` for extension dependencies) or via `os.add_dll_directory(...)`.

## Python module `_pops`

```bat
cmake -S adc_cpp -B build-pywin -G "Visual Studio 17 2022" -A x64 ^
  -DPOPS_BUILD_PYTHON=ON -DPOPS_USE_KOKKOS=ON -DKokkos_ROOT=...\kokkos-shared ^
  -DPython_EXECUTABLE=...\python.exe -DCMAKE_CXX_FLAGS="/DPOPS_EXPORT_BUILDING_MODULE /DNOMINMAX /bigobj"
cmake --build build-pywin --config Release --target _pops
```
`/DPOPS_EXPORT_BUILDING_MODULE` exports the `System` methods (`__declspec(dllexport)`, see
`export.hpp`) -> produces the **import library `_pops.lib`** against which the DSL `.dll` links. Copy
`_pops.cp3XX-win_amd64.pyd`, `_pops.lib` and `kokkos*.dll` next to `python/pops/`.

```python
import pops; pops.doctor(); s = pops.System(n=64); a = pops.AmrSystem(n=64)   # OK natif
```

## DSL production (`.dll`) and `adc_cases`

`model.compile(..., backend="production")` compiles a `.dll` (cl/clang-cl, `/LD`, linked to
`kokkoscore.lib` + `_pops.lib`) that `System.add_native_block` loads (`LoadLibraryW`) -- no Unix
`RTLD_GLOBAL`, the `_pops` symbols are resolved at link time via `_pops.lib`. Validated bit-identical to the
brick path. `adc_cases.common.native.build_shared` produces standalone `.dll` files (ctypes) in the
same way.

> Warning: The custom `.cpp` files of the cases must export their entry points as `__declspec(dllexport)` on
> Windows (`extern "C"` alone does not export under `/LD`).

## Notable differences vs Linux

| Topic | Linux/WSL2 | Native Windows |
|---|---|---|
| Dynamic loading | `dlopen`/`RTLD_GLOBAL` | `LoadLibraryW` + **import library** (`_pops.lib`) |
| Kokkos cross-`.dll` | symbols via `RTLD_GLOBAL` | **Kokkos as a shared DLL** (`WINDOWS_EXPORT_ALL_SYMBOLS`) |
| OpenMP | gcc/clang `-fopenmp` | **clang-cl** (MSVC `_OPENMP` capped at 2.0) |
| DSL compilation at runtime | g++/clang on the `PATH` | `cl`/`clang-cl` -> **vcvars env required** |
| GPU CUDA | yes (WSL2, sm_86) | **no** (Kokkos CUDA not supported natively) -> use WSL2 |

## Limitations
- **Native CUDA GPU: not available** (Kokkos has no native CUDA support on Windows -- neither VS+CUDA, nor
  nvcc-as-CXX for lack of a Windows `nvcc_wrapper`). The GPU goes through [WSL2](windows-wsl2.md).
  CMake refuses this combination early: configuring with `Kokkos_ENABLE_CUDA=ON` on Windows stops at
  a fatal error pointing to WSL2 (no broken half-build deep in the Kokkos configure). Refs ADC-168.
- CMake industrialization (shared Kokkos option, install of the import libs, clang-cl presets) to finalize
  so that these builds are reproducible from a single command.
