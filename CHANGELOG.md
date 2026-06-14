# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
[SemVer](https://semver.org/lang/en/) (0.y.z while the public API still moves).

## Versioning policy

- **Single source**: `project(VERSION x.y.z)` in `CMakeLists.txt`. Everything derives from it:
  `adc.__version__` (bakes `ADC_VERSION` into `_adc`), the pip wheel (regex in `pyproject.toml`),
  `adcConfigVersion.cmake` (install/export). NEVER duplicate the number elsewhere; the docs build
  derives it too (`scripts/build_docs.sh` injects `PROJECT_NUMBER`, `docs/sphinx/conf.py` reads
  `project(VERSION)`), so nothing is bumped by hand outside `CMakeLists.txt`.
- **Bump**: PATCH = fixes with no API change; MINOR = backward-compatible API/brick additions;
  MAJOR (post-1.0) = API or ABI break of the DSL production path.
- **Tag**: set `git tag vX.Y.Z` on master when the bumping PR merges, then `git push --tags`.
- On every notable PR: one line in `[Unreleased]` below; on a bump, the section becomes
  `[x.y.z] - YYYY-MM-DD`.

## [Unreleased]

### Added

- **Quality tooling / static analysis** (ADC-105): dedicated CI workflow `.github/workflows/quality.yml`,
  off the PR critical path (weekly Sunday cron + `workflow_dispatch` + `quality` label). Five
  *informative* (non-blocking) jobs: `clang-format` (`.clang-format`), strict warnings
  (`ADC_ENABLE_WARNINGS`), `clang-tidy` (`.clang-tidy`), ASan+UBSan sanitizers (`ADC_ENABLE_SANITIZERS`,
  `ci-warnings`/`ci-asan` presets) and CodeQL. CMake options OFF by default (empty `adc_dev_options`
  target), so `ci.yml`, local builds and `adc_cases` are unchanged. See `docs/QUALITY_TOOLING.md`.

### Changed

- **Docs version single-sourced** (ADC-233): the docs build derives the version from
  `project(VERSION)` in `CMakeLists.txt` (`scripts/build_docs.sh` injects the Doxygen
  `PROJECT_NUMBER`, `docs/sphinx/conf.py` reads it), so Doxygen and Sphinx no longer drift.

## [0.1.0] - 2026-06-10

First numbered release (previously `0.0.1`, never exposed; Doxygen/Sphinx already announced
0.1.0, and this bump aligns the CMake single source with it).

### Added
- `pip install .` via scikit-build-core: module in site-packages, no PYTHONPATH; backends via
  environment variables (`ADC_USE_KOKKOS=ON Kokkos_ROOT=... pip install .`).
- `find_package(adc)`: install/export rules for the header-only core (`ADC_INSTALL`).
- `adc.__version__`, `adc.doctor()` (full diagnostic), `adc.set_threads()` /
  `adc.parallel_info()` / `adc.has_kokkos()`, `_adc.kokkos_is_initialized()`.
- CMake presets (`python`, `python-parallel`, `serial`, `parallel`, `mpi` plus the `ci-*` series
  used by CI: single source of the flags).
- Conda environment (`environment.yml`) plus `scripts/setup_env.sh` (per-platform toolchain
  pinned in the env) plus `pixi.toml` (reproducible cross-platform lockfile).
- `scripts/kokkos_openmp_conda.sh` (Kokkos Serial+OpenMP in the conda env, ~2 min);
  `scripts/build_docs.sh` (lint + Sphinx + Doxygen + site in one command); machine profile
  `Tools/machines/romeo/romeo_adc.profile.example`.
- Extended ABI key of the DSL production path: `kokkos=` and `stdlib=` tokens (divergences
  previously undetected), as a preprocessor literal (insensitive to ELF interposition).
- DSL runtime toolchain guards: build compiler baked in (`__cxx_compiler__`) and preferred over
  PATH, standard probe (`c++23`->`c++2b`), pre-dlopen module/header guard (including on cache HIT),
  compilation errors surfaced with the compiler output.

### Changed
- `ADC_BUILD_TESTS` follows `PROJECT_IS_TOP_LEVEL`: a FetchContent consumer no longer builds the
  test suite.
- `import adc` works without numpy (`adc.dsl` is lazy, with a targeted error at use).
- DSL `.so` cache: machine-aware key (arch + optflags) and fingerprint of the Kokkos install
  (`KokkosCore_config.h`); a different Kokkos invalidates the cache.
- Tests: ctest labels (`core`/`mpi`) plus timeouts; memory guard `-O0` plus the ninja pool
  extended automatically to any target compiling `system.cpp`/`amr_system.cpp` (39 objects).
- `pybind11` taken from the environment before any FetchContent; ccache auto-detected;
  `ADC_PY_LTO` option (OFF by default).

### Fixed
- Three real user bugs of the DSL production path: conda PATH compiler rejecting `-std=c++23`,
  stale module leading to a cryptic dlopen `symbol not found`, `CalledProcessError` without the
  compiler output.
- `find_package(Kokkos)` failed on macOS against a Kokkos OpenMP (libomp hints set before
  KokkosConfig's `find_dependency(OpenMP)`).
- Docs: pip/PYTHONPATH contradiction, phantom Catch2 mention, `$KOKKOS_ROOT` undefined in the
  tutorial, stale numpy claim, inconsistent versions (0.0.1 vs 0.1.0).
