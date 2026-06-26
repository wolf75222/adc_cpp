# Build and install UX, complete audit and roadmap (2026-06-10)

Follow-up to [TOOLCHAIN_ROBUSTESSE_AUDIT_2026-06-10.md](TOOLCHAIN_ROBUSTESSE_AUDIT_2026-06-10.md).
Audit by multi-agent workflow (6 lenses: similar projects **via web**, modern CMake
**via web**, full build system, CI, **adversarial critique of our own changes**,
user journey across 3 personas) + adversarial cross-check of the critical findings.

---

## 1. What the twin projects do (web research, primary sources)

Projects studied: **pyAMReX / WarpX** (AMR+GPU+MPI+pybind11, the closest to adc_cpp),
openPMD-api, PyKokkos, FEniCSx/dolfinx, OpenMC, PyBaMM, official pybind11 doc, Scientific
Python Development Guide.

| Topic | Dominant 2025-2026 pattern | adc_cpp today |
|---|---|---|
| Install of the Python module | **`pip install .` via scikit-build-core** (1st choice in the pybind11 doc) or CMake target **`pip_install`** (pyAMReX/WarpX) -> site-packages | manual PYTHONPATH (dev crutch at the twins, not a distribution mode) |
| Optional backends (GPU/MPI) | env vars mapped 1:1 onto CMake `-D` at `pip install` time (`WARPX_COMPUTE=CUDA WARPX_MPI=ON pip install .`) ; conda variants `=*=mpi_openmpi` ; Spack variants | CMake options + presets (local equivalent, not pip) |
| Bugs "JIT/runtime with another toolchain" | a REAL and documented class (cppyy/Cling ABI GCC vs Clang, Numba/JAX PTX vs toolkit). Anti-pattern: **freeze/bake the toolchain** | exactly what we implemented (`__cxx_compiler__` baked, probes, enriched ABI key) ok |
| Env reproducibility | `environment.yml` judged non-reproducible (no lock) -> **pixi** (auto multi-platform lockfile, env inside the project, HPC-friendly) or conda-lock | environment.yml (enough to get started, lock to consider) |
| Exemplary install doc | WarpX/openPMD: 3 paths (Users / Developers / **HPC with sourceable machine profiles** `Tools/machines/<cluster>/*.profile.example`) | 1 installation page + tutorial ; no ROMEO profile |

## 2. Fixed TODAY (2nd wave, after the full audit)

| Fix | Detail | Validation |
|---|---|---|
| **ABI key enriched with `kokkos=` + `stdlib=`** (the 2 confirmed UB holes) | [abi_key.hpp](../include/pops/runtime/dynamic/abi_key.hpp): `;kokkos=<0|1>` (POPS_HAS_KOKKOS, allocator/types layouts) + `;stdlib=libc++_NNN|libstdc++_NNN` (libc++/libstdc++ mix). A single inline function -> module AND loader consistent automatically | key printed with/without `-DADC_HAS_KOKKOS` ; Python parsers (`std=`, `headers=`) insensitive (tested) |
| **Kokkos OpenMP "via conda" in 1 command** | [scripts/kokkos_openmp_conda.sh](../scripts/kokkos_openmp_conda.sh): builds Kokkos Serial+OpenMP into `$CONDA_PREFIX` (~2 min, tooling already provided by the env). Answer to the conda-forge Serial-only package | **actually tested**: build OK, `KOKKOS_ENABLE_OPENMP` present, configure adc_cpp -> `Kokkos found = (OPENMP;SERIAL)` |
| libomp hints before `find_package(Kokkos)` | `KokkosConfig.cmake` does `find_dependency(OpenMP REQUIRED)` -> was failing on macOS ; macro `pops_apple_libomp_hints()` factored out (conda then brew), called for both OpenMP AND Kokkos | configure against the test Kokkos OpenMP: OK |
| **FetchContent consumer no longer compiles the tests** (HIGH confirmed) | `option(POPS_BUILD_TESTS ${PROJECT_IS_TOP_LEVEL})` + bump `cmake_minimum_required(3.21)` (aligned with presets) | test super-project: **0 test target** pulled, `app` links `pops::pops` and runs ; top-level unchanged (120 targets) |
| **Version unified to 0.1.0 + `pops.__version__`** | `project(VERSION 0.1.0)` (Doxygen/Sphinx already said 0.1.0, CMake said 0.0.1) -> baked `POPS_VERSION` -> `pops.__version__` | bindings compile with `POPS_VERSION="0.1.0"` |
| Hardened presets | hidden preset `conda` with **`condition: CONDA_PREFIX != ""`** (clear failure instead of silently empty roots) ; `CMAKE_EXPORT_COMPILE_COMMANDS=ON` (clangd/IDE) ; honest `python-parallel` description (Serial-only note + remedy) | `cmake --list-presets` OK |
| ctest labels + timeouts | `pops_add_test` -> `LABELS core TIMEOUT 600` ; `pops_add_mpi_test` -> `LABELS mpi` (an MPI deadlock no longer blocks the runner) -> `ctest -L mpi` possible | top-level configure OK |
| **cache-HIT** guard (hole found by self-critique) | `check_compiled_matches_module()` at the **wiring point** (System+AmrSystem `add_equation`): a cached `.so` + stale module -> actionable error, no more cryptic dlopen. The adversarial verifier then **confirmed it as already sufficient** | simulated scenario: raises/no-ops correctly |
| Misc confirmed | a single walk+sha256 (reuse of the signature) ; `.pops_cache*/` gitignored ; phantom Catch2 mention removed ; `build-master/python` -> `build-py-kokkos/python` ; tutorial Step 16: conda path `$CONDA_PREFIX` vs custom `$KOKKOS_ROOT` ; pybind11 pin `<3` lifted (the build takes the active env, 3.0.4 validated) ; memoization of the `-std` probe ; `OMP_PROC_BIND=false` **only on macOS** (cluster NUMA preserved) | py_compile/configure/0 warning |

## 3. Self-critique: verdicts on our own work

- **Refuted-in-our-favor x3**: the adversarial verifiers tried to break our guards
  and concluded that they already cover the reported cases (compiler resolution, c++2b probe,
  cache-HIT guard).
- **Confirmed against us and fixed**: presets without a CONDA_PREFIX condition ; inconsistent
  pybind11 pin ; `OMP_PROC_BIND=false` penalizing the cluster ; non-memoized probe ; displayName
  `python-parallel` misleading. -> all fixed (section 2).
- **Known and accepted**: `set_threads` relies on a Python flag (`_first_system_built`) that
  does not see a Kokkos init triggered by another path (direct DSL .so). Clean fix =
  expose `Kokkos::is_initialized()` in the bindings (roadmap P2).
- **Generous but real**: `generator: Ninja` forced in the presets, accepted (the conda env
  provides ninja ; documented).

## 4. Remaining roadmap (prioritized)

### P1, the "distribution" step (design decision, dedicated PR)
- **`pyproject.toml` + scikit-build-core**: `pip install .` drives the existing CMake (a single
  source of truth), `pip install -e .` replaces PYTHONPATH. Backends via env vars -> `-D`
  (WarpX pattern: `POPS_KOKKOS=ON POPS_MPI=ON pip install .`). This is THE 2025-2026 standard
  (pybind11, Scientific Python Guide, pyAMReX/WarpX/PyBaMM). PyBaMM caveat: test early on
  aarch64 (ROMEO Grace).
- **`install()`/export package-config** (~30 lines): `find_package(adc)` for consumers
  outside FetchContent + prerequisite for a possible conda/Spack feedstock. Guarded by `POPS_INSTALL`.

### P2, incremental robustness/convenience
- `_pops.kokkos_is_initialized()` exposed -> `set_threads` reliable on all init paths.
- CI: use `cmake --preset` (single source of truth ; the CI rewrites the flags by hand,
  duplication confirmed, counted differently across the jobs) ; extend
  `pops_cap_opt_for_kokkos_ram`/pool to the ~39 targets that compile system/amr_system.cpp.
- configure+build step of `bench/` in ci-full (silent rot today).
- Versioned ROMEO machine profile (`Tools/machines/romeo/...profile.example`, WarpX pattern) +
  "Spack cluster without root" section in installation.md.
- One-command doc (target `docs` or script: sphinx+doxygen+conda-forge deps).

### P3, opportunistic
- **pixi** (or conda-lock) for the reproducible multi-platform lock ; pixi stores the env inside
  the project (cluster $HOME quotas).
- cibuildwheel/PyPI wheels the day public distribution matters ; conda-forge feedstock
  (with `pybind11-abi` for inter-module ABI consistency).
- lazy `import numpy` in dsl.py (production does not need it) ; debug/asan presets ;
  Kokkos version in the DSL cache feature-key.

## 5. Clear answer to "parallel Kokkos with conda?"

The conda-forge `kokkos` **package**: no (Serial-only, CUDA = separate package, no OpenMP).
But **yes in practice**: `bash scripts/kokkos_openmp_conda.sh` builds a Kokkos Serial+OpenMP
**inside the conda env** in ~2 min with the env tooling, tested end to end on this
machine (build + `Kokkos found = (OPENMP;SERIAL)` at adc_cpp configure). The GPU stays
ROMEO/Spack.
