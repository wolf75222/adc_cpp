# Windows natif (sans WSL2)

> **Statut : le Windows natif fonctionne pour le CPU** -- Serial, OpenMP, module Python `adc`
> (`import adc`, `System`, `AmrSystem`), **DSL production** (`.dll`, resultats bit-identiques) et
> compilation de C++ custom (`adc_cases.common.native`). **Le GPU reste sur [WSL2](windows_wsl2.md)**
> (Kokkos CUDA n'est pas supporte en natif Windows).

Cette page complete le guide [WSL2](windows_wsl2.md). WSL2 reste le chemin le plus simple et le seul
qui couvre le **GPU**. Le natif est utile si tu veux eviter WSL2 pour du travail CPU.

## Toolchain

- **Visual Studio 2022** (MSVC, `cl.exe`, C++23 via `/std:c++20`/`/std:c++latest`) + **CMake**.
- **LLVM/clang-cl** (`winget install LLVM.LLVM`) **uniquement pour OpenMP** : MSVC garde
  `_OPENMP=2.0` meme avec `/openmp:llvm`, donc Kokkos OpenMP (>=3.0) exige clang-cl (qui rapporte 5.1).
- **Python** (Miniforge conseille) + `numpy`, `pybind11`.
- Toute compilation au runtime (DSL `production`, `adc_cases.common.native`) lance `cl`/`clang-cl` :
  lancer Python depuis un invite **"x64 Native Tools"** (vcvars) pour avoir `INCLUDE`/`LIB`.

## Kokkos en DLL partagee (cle du natif)

Le DSL et `adc_cases.common.native` generent une `.dll` qui partage le runtime Kokkos du module
`_adc` : il faut donc **Kokkos en bibliotheque partagee** (un seul runtime, sinon double-singleton ->
crash). MSVC n'exporte pas les symboles Kokkos par defaut -> utiliser `WINDOWS_EXPORT_ALL_SYMBOLS` :

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

`kokkos*.dll` doit etre chargeable au runtime : le placer **a cote du `.pyd`** (Python 3.8+
n'utilise plus le `PATH` pour les dependances d'extension) ou via `os.add_dll_directory(...)`.

## Module Python `_adc`

```bat
cmake -S adc_cpp -B build-pywin -G "Visual Studio 17 2022" -A x64 ^
  -DADC_BUILD_PYTHON=ON -DADC_USE_KOKKOS=ON -DKokkos_ROOT=...\kokkos-shared ^
  -DPython_EXECUTABLE=...\python.exe -DCMAKE_CXX_FLAGS="/DADC_EXPORT_BUILDING_MODULE /DNOMINMAX /bigobj"
cmake --build build-pywin --config Release --target _adc
```
`/DADC_EXPORT_BUILDING_MODULE` exporte les methodes `System` (`__declspec(dllexport)`, cf.
`export.hpp`) -> produit l'**import library `_adc.lib`** contre laquelle la `.dll` DSL se lie. Copier
`_adc.cp3XX-win_amd64.pyd`, `_adc.lib` et `kokkos*.dll` a cote de `python/adc/`.

```python
import adc; adc.doctor(); s = adc.System(n=64); a = adc.AmrSystem(n=64)   # OK natif
```

## DSL production (`.dll`) et `adc_cases`

`model.compile(..., backend="production")` compile une `.dll` (cl/clang-cl, `/LD`, liee a
`kokkoscore.lib` + `_adc.lib`) que `System.add_native_block` charge (`LoadLibraryW`) -- pas de
`RTLD_GLOBAL` Unix, les symboles `_adc` sont resolus au lien via `_adc.lib`. Valide bit-identique au
chemin briques. `adc_cases.common.native.build_shared` produit des `.dll` standalone (ctypes) de la
meme facon.

> Attention : Les `.cpp` custom des cas doivent exporter leurs entry points en `__declspec(dllexport)` sous
> Windows (`extern "C"` seul n'exporte pas sous `/LD`).

## Ecarts notables vs Linux

| Sujet | Linux/WSL2 | Windows natif |
|---|---|---|
| Chargement dynamique | `dlopen`/`RTLD_GLOBAL` | `LoadLibraryW` + **import library** (`_adc.lib`) |
| Kokkos cross-`.dll` | symboles via `RTLD_GLOBAL` | **Kokkos en DLL partagee** (`WINDOWS_EXPORT_ALL_SYMBOLS`) |
| OpenMP | gcc/clang `-fopenmp` | **clang-cl** (MSVC `_OPENMP` bloque a 2.0) |
| Compilation DSL au runtime | g++/clang sur le `PATH` | `cl`/`clang-cl` -> **env vcvars requis** |
| GPU CUDA | oui (WSL2, sm_86) | **non** (Kokkos CUDA non supporte natif) -> utiliser WSL2 |

## Limites
- **GPU CUDA natif : non disponible** (Kokkos n'a pas de support CUDA natif Windows -- ni VS+CUDA, ni
  nvcc-as-CXX faute de `nvcc_wrapper` Windows). Le GPU passe par [WSL2](windows_wsl2.md).
- Industrialisation CMake (option Kokkos shared, install des import libs, presets clang-cl) a finaliser
  pour rendre ces builds reproductibles d'une commande.
