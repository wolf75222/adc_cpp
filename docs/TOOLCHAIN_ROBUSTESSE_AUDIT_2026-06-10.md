# Toolchain and install robustness audit, status (2026-06-10)

**Trigger:** three real bugs at third-party users (colleagues' machines), all of the
same class "**build environment != runtime environment**":

| # | Observed symptom | Root cause | Status |
|---|---|---|---|
| 1 | `error: invalid value 'c++23' in '-std=c++23'` (mambaforge env) | the DSL compiled its runtime `.so` files with the compiler from the **PATH** (old conda gcc/clang) and the literal **`c++23` spelling**, whereas `_pops` is built in `-std=c++2b` by another compiler | **FIXED** (3 safeguards, validated by repro) |
| 2 | `dlopen : symbol not found in flat namespace '__ZN3adc6System13install_block...'` | `_pops` module **stale** vs headers (build before a `git pull`): the DSL loader references a C++ signature that the old `.so` does not export; the ABI guard **never** runs because the dlopen fails before it ([native_loader.hpp:634](../include/pops/runtime/builders/compiled/native_loader.hpp) dlopen < line 647 key read) | **FIXED** (pre-dlopen guard, validated by repro) |
| 3 | `subprocess.CalledProcessError: Command [...] returned non-zero exit status 1` | `subprocess.run(check=True)` without capture: the compiler error is not surfaced in the exception | **FIXED** (`_run_compile`, stderr + remedies) |

Audit conducted by a multi-agent workflow (4 lenses: `dsl.py`, CMake, env bug class, conda-forge
reality) + **adversarial counter-verification** of the critical findings. Notable point: both
HIGH findings on the compiler/std were *refuted* by the verifiers **because the fixes
delivered during the audit already corrected them**, an independent confirmation of the root causes.

---

## 1. What the build bakes into `_pops` (after this session)

| Python attribute | CMake source | Role |
|---|---|---|
| `__cxx_std__` (20/23) | `POPS_CXX_STD` | C++ standard of the DSL loader (already present) |
| `__has_kokkos__` | `POPS_HAS_KOKKOS` | compiled compute backend (set_threads/doctor/loader parity) |
| **`__cxx_compiler__`** *(new)* | `CMAKE_CXX_COMPILER` | **the only ABI-compatible guaranteed compiler** for the DSL loaders (the ABI key encodes `__VERSION__`) |
| `abi_key()` | `__VERSION__`+`__cplusplus`+`POPS_HEADER_SIG` | C++ ABI guard of the production path |

## 2. DSL compiler resolution chain (new, centralized)

`_default_cxx()` ([codegen/toolchain.py](../python/pops/codegen/toolchain.py)): explicit `cxx=` -> `$POPS_CXX` -> **build
compiler** (`__cxx_compiler__`, if present on the machine) -> PATH (`c++`/`g++`/`clang++`,
legacy). The Kokkos/CUDA path keeps its priority (`POPS_KOKKOS_CXX`, explicit `nvcc_wrapper`).

Before **each** compilation: `_probe_cxx_std()` checks `-std=` (probe `-fsyntax-only`), downgrades
`c++23`->`c++2b` (same level, same `__cplusplus` on recent compilers) if needed, otherwise raises an
**actionable** error (compiler used, build compiler, 3 remedies). Compilation failure ->
`_run_compile` surfaces the **compiler output** in the exception.

## 3. Pre-dlopen guards (bug #2)

- `_check_headers_match_module(include)`: compares the header signature **baked** into `_pops`
  (`headers=` token of `abi_key()`) with the one of the `include/` tree in use. Divergence -> clear
  error "stale module, rebuild" BEFORE any compilation/dlopen. Wired into `compile_native` and
  `HybridModel.compile` (native paths).
- `pops.doctor()` exposes the same check (`headers_sync`).
- A failing import of `pops._pops` now gives an actionable message (extension missing /
  wrong cpython-3XY interpreter, with the exact remedy) instead of the raw `ModuleNotFoundError`.

## 4. Kokkos parity and cache (.so), hardening

- **Confirmed by adversarial verification**: the C++ ABI key does NOT encode `POPS_HAS_KOKKOS`
  (divergent `allocator.hpp`/`types.hpp` layouts = potential UB) nor libc++/libstdc++. C++ work item
  proposed (sec.6). In the meantime, **Python mitigation**: `_warn_kokkos_parity()` warns when
  `_pops.__has_kokkos__` and `POPS_KOKKOS_ROOT` diverge (Kokkos module + serial loader = silent perf;
  the reverse = real risk).
- `.so` cache key: addition of the **CPU architecture + `POPS_DSL_OPTFLAGS`**
  (`_platform_cache_key`), an x86_64/`-march=native` `.so` will no longer be reused on another
  machine via a shared cache (silent SIGILL).

## 5. conda-forge reality (network lens; to reconfirm at the first `env create`)

- **`cxx-compiler`**: clang 19 (osx-arm64) / gcc 14 (linux-64), **both C++23-capable**.
  A C++23 toolchain guaranteed by conda is therefore possible, BUT in **fully consistent all-conda**
  mode (build `_pops` with it AND `POPS_CXX=$CONDA_PREFIX/bin/...`; do not mix with AppleClang).
- **`kokkos` conda-forge = Serial backend ONLY** (no OpenMP; CUDA = separate package).
  Enough to compile/validate `POPS_USE_KOKKOS=ON`, but **does not scale in threads**. Real
  multi-thread: Kokkos compiled `-DKokkos_ENABLE_OPENMP=ON` (local) or Spack/ROMEO. Documented in
  `environment.yml` + `installation.md` (check `Kokkos found ... = (...)` at configure).
- The `# [linux]`/`# [osx]` selectors **do not work** in `environment.yml`
  (conda-build only) -> a single portable yml, variants via separate envs or conda-lock.
- The `gcc` of a macOS mambaforge env is a shim/disguised clang, often old -> exactly the
  pitfall of bug #1; covered by the resolution chain sec.2.

## 6. Remaining work items (proposals, not done here)

| Prio | Work item | Why | Nature |
|---|---|---|---|
| P1 | `kokkos=<0|1>` token **in `abi_key_string()`** (C++ + CMake + dsl) | Kokkos mismatch = undetected UB (confirmed); the Python warning is a mitigation, not a guarantee | dedicated PR (changes the ABI key -> invalidates caches, touches the ABI tests) |
| P1 | `stdlib=` token (`_LIBCPP_VERSION`/`__GLIBCXX__`) in the ABI key | libc++ vs libstdc++ = undetected divergent std::string/function ABI (confirmed) | same PR as above |
| P2 | lazy `import numpy` in `dsl.py` | numpy missing kills the whole `import pops` whereas the production path does not need it | small dsl refactor |
| P2 | Sign the **Kokkos version** in the cache feature-key | a different Kokkos between build and runtime does not invalidate the `.so` cache | dsl.py |
| P2 | configure warning if `CONDA_PREFIX` is empty when a conda preset is used | parallel/mpi presets assume the env is activated | CMakePresets/CMake |
| P3 | macOS SDK in the cache key | low risk, documented | optional |

## 7. Target user journey (after this session)

```bash
# 1x : all the tooling (recent cmake, ninja, ccache, python, numpy, pybind11, kokkos*, openmpi)
conda env create -f environment.yml && conda activate pops

# one command per mode (flags baked into CMakePresets.json)
cmake --preset python          && cmake --build --preset python            # serie
cmake --preset python-parallel && cmake --build --preset python-parallel   # Kokkos*

# diagnostic en 1 commande (a donner aux collegues au moindre souci)
python -c "import pops; pops.doctor()"

# threads en 1 ligne (plus de variables d'env a connaitre)
python -c "import pops; pops.set_threads(); ..."
```

\* subject to the Kokkos-Serial caveat of sec.5 for real scaling.

**Every error of the fragile paths is now actionable**: compiler too old (probe),
stale module (signature guard), wrong interpreter (import guard), compile failure (stderr
surfaced), Kokkos divergence (warning), and `pops.doctor()` checks all of it up front.
