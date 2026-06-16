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

- **Documentation rebuilt around a Diataxis navigation** (ADC-248): Getting started, Tutorials,
  Concepts (11 pages, ADC-249), How-to (12 task pages), Simulation, AMR, Running, Advanced topics,
  Reference and Development sections, plus a Quickstart page and an internal documentation style
  guide (ADC-250). Pages are kebab-case.
- **Embedded C++ reference** in the Sphinx site via doxysphinx (ADC-149), with a modern Doxygen
  theme and full dot diagrams (ADC-239).
- **Documentation-as-code tooling**: per-page `docmap.toml` with owner and freshness plus an
  example harness (ADC-147), a doc taxonomy and a stack ADR (ADC-146), and CI doc lanes (a light PR
  lane and a weekly heavy lane with lld and ccache, ADC-151/225).
- **Quality tooling / static analysis** (ADC-105): dedicated CI workflow `.github/workflows/quality.yml`,
  off the PR critical path (weekly Sunday cron + `workflow_dispatch` + `quality` label). Five
  *informative* (non-blocking) jobs: `clang-format` (`.clang-format`), strict warnings
  (`ADC_ENABLE_WARNINGS`), `clang-tidy` (`.clang-tidy`), ASan+UBSan sanitizers (`ADC_ENABLE_SANITIZERS`,
  `ci-warnings`/`ci-asan` presets) and CodeQL. CMake options OFF by default (empty `adc_dev_options`
  target), so `ci.yml`, local builds and `adc_cases` are unchanged. See `docs/QUALITY_TOOLING.md`.
- **Fuzzing, coverage and developer automation** (ADC-113): invariant-checked libFuzzer harnesses in
  `fuzz/` (Box2D, Berger-Rigoutsos clustering, `real_eig_minmax`; option `ADC_BUILD_FUZZING` +
  `ci-fuzz` preset, clang), gcov/gcovr coverage (`ADC_ENABLE_COVERAGE` + `ci-coverage` preset),
  Python ruff lint (`[tool.ruff]`), opt-in pre-commit hooks (`.pre-commit-config.yaml`: clang-format
  and ruff at commit), and two more `quality.yml` jobs (`fuzz`, `coverage`) on the same weekly
  informative cadence; the default build stays bit-unchanged.
- **Repository health and GitHub hygiene**: BSD-3 `LICENSE` and license declaration, `CONTRIBUTING`,
  `SECURITY`, `.gitattributes`, PR and issue templates (ADC-223/224/244/246), a root-hygiene guard
  (ADC-169), Dependabot weekly Actions bumps (ADC-117), a release workflow that turns a `v*` tag
  into a GitHub Release from this changelog (ADC-234), and `docs/VERSIONING.md` with the bump rules
  (ADC-232/237).
- **Shared C++ test harness** (`test_harness.hpp`, `bench/common.hpp`, ADC-215); a Schur-split test
  with an AMR guard on `add_block` and auto-skip without Kokkos (ADC-207).
- **CI**: `ci.yml` split into parallel `gate-cpp` / `gate-python` lanes with `mold` and path
  routing (ADC-171).
- **Moment-model documentation**: a step-by-step HyQMOM tutorial, a moments-and-closures concept
  page, a moment-models reference, and `adc.moments` (`build_moment_model`, `gaussian_closure`,
  `lorentz_sources`) added to the Python API reference (autodoc).

### Changed

- **Documentation translated to English**: the whole Sphinx site (ADC-228), the root design guides
  and this changelog (ADC-241), the remaining French docs (ADC-245), `CONTRIBUTING` / `SECURITY` /
  `DOC_QUALITY` (ADC-227), and a restructured English README with a translation glossary
  (ADC-119/236/218).
- **BREAKING (C++)**: facade parameters regrouped into a POD struct, source-incompatible for C++
  callers; the Python API is unchanged (ADC-214).
- **Docs version single-sourced** (ADC-233): the docs build derives the version from
  `project(VERSION)` in `CMakeLists.txt` (`scripts/build_docs.sh` injects the Doxygen
  `PROJECT_NUMBER`, `docs/sphinx/conf.py` reads it), so Doxygen and Sphinx no longer drift.
- **Portability (LLP64 / Windows data model)**: `long` to `int64_t` in `mesh/` and the `box_hash`
  key, removing undefined `<< 32` shifts (ADC-209/216).
- **Internal C++ cleanups**: rule of five on owning types (ADC-212), bit-exact factorizations of
  duplicated FV/MG and Newton-Jacobian code (ADC-213), residual extended device lambdas converted
  to named functors (ADC-210), hardened runtime guards (ADC-211), and a coding-standards and
  comments audit (ADC-124/125).
- **Code comments and Python docstrings translated to English** (ADC-272): all of `include/adc/**`,
  `python/adc/**` (including the pybind bindings) and the `CODE_DOCUMENTATION_CONVENTION` guide; the
  published `/cpp/` Doxygen reference and the Python autodoc now render in English. Code structure is
  byte-identical; codegen-template strings and cross-TU dispatch tokens are kept verbatim.

### Fixed

- Heap-buffer-overflow masked on disc geometry, caught by a ghost-width guard at FV operator entry
  (ADC-163).
- Comment rot: 10 inaccurate comments corrected (ADC-208); broken sister-solver links in the docs
  index.
- **Docs (Pages) build**: `suppress_warnings = ["docutils"]` set unconditionally in
  `docs/sphinx/conf.py` so `sphinx -W` no longer fails on autodoc-rendered Doxygen `@param`
  docstrings (the meaningful broken-reference and toctree checks stay strict).
- **`master` gate unblocked** (ADC-281): `test_dispatch_tags` greps the dispatch-registry error
  fragments, which ADC-272 (#105) had translated to English in `dispatch_tags.hpp` without updating
  the test -- reconciled the grepped fragments (`unknown limiter` / `unknown Riemann flux` /
  `unsupported` / `polar`).
- **Python test message-assertions reconciled with ADC-272 translation** (ADC-283): ~26 assertions
  across 21 `python/tests/` files grepped now-translated French error fragments (masked by the
  gate-python swallow, ADC-112). Reconciled to assert language-stable substrings (echoed values,
  quoted tokens) so they survive the ongoing translation; also reconciled the order-sensitive
  `ABI incompatible` greps with the source's `incompatible ABI` -- the `test_dsl_production{,_amr}.py`
  asserts and the C++ `test_amr_native_loader.cpp` guard (`test_native_abi_std.py` rides along under
  this umbrella) -- plus the stale forward-order wording in comments and hints (`bindings.cpp`,
  `__init__.py`, `dsl.py`, `python/CMakeLists.txt`).

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
