# Versioning

`adc_cpp` follows [Semantic Versioning 2.0.0](https://semver.org). This document declares
the public API that the version number tracks, and the rules for bumping it.

## Single source of the version number

The version lives in one place: `project(VERSION x.y.z)` in `CMakeLists.txt`. Everything
derives from it: `pops.__version__` (baked as `POPS_VERSION` into `_pops`), the pip wheel
(scikit-build-core regex on `pyproject.toml`), and `adcConfigVersion.cmake`. Do not
duplicate the number elsewhere. The docs build derives it too: `scripts/build_docs.sh` injects
`PROJECT_NUMBER` into Doxygen from `project(VERSION)`, and `docs/sphinx/conf.py` reads the same
value, so the published docs never drift from `CMakeLists.txt`.

## Public API (under SemVer guarantee)

What a version bump is allowed to break is exactly this surface:

- C++ runtime facade: `pops::System`, `pops::AmrSystem` and their public methods (block
  composition, `set_poisson`, `set_refinement`, stepping).
- The concepts a model composes against: `PhysicalModel`, `NumericalFlux`, `EllipticSolver`,
  and the named generic bricks in `include/pops/physics/`.
- Python bindings: the documented `pops.*` surface (`pops.Model`, `pops.System`,
  `pops.AmrSystem`, `pops.dsl.Model`, the brick classes, `pops.doctor`, `pops.set_threads`,
  `pops.parallel_info`, `pops.has_kokkos`, `pops.__version__`).
- DSL surface: the fixed aux names (`phi`, `grad_x`, `grad_y`, `B_z`, `T_e`) and the
  documented builders.
- Consumable CMake: the `pops::pops` target, `find_package(adc)`, and the documented options
  (`POPS_USE_MPI`, `POPS_USE_HDF5`, `POPS_USE_KOKKOS`, ...) and presets.

## Internal (no guarantee, may change in any release)

Private helpers, memory layouts and `Fab` / `MultiFab` internals, the DSL code-generation
internals and the production `.so` ABI key, test harnesses, benchmarks, and anything not in
the list above. The ABI key intentionally invalidates the DSL cache across toolchains; that
is not a SemVer-relevant break.

## Bump rules

- PATCH (`x.y.Z`): bug fixes with no change to the public API.
- MINOR (`x.Y.0`): backward-compatible additions to the public API (new bricks, new options,
  new Python surface).
- MAJOR (`X.0.0`, post-1.0): a break of the public API or of the production DSL ABI.

While in `0.y.z` initial development the public API may still change; a `0.y` bump can carry
breaking changes until `1.0.0`.

## Releasing

1. Bump `project(VERSION x.y.z)` in `CMakeLists.txt`. Everything else (the Python `__version__`, the
   pip wheel, Doxygen and Sphinx) derives from it automatically; nothing else is edited by hand.
2. Move the `## [Unreleased]` entries of [CHANGELOG.md](../CHANGELOG.md) into a
   `## [x.y.z] - YYYY-MM-DD` section.
3. Merge, then `git tag vx.y.z` on master and `git push --tags`. The `release.yml` workflow
   turns the tag into a GitHub Release built from that CHANGELOG section.
