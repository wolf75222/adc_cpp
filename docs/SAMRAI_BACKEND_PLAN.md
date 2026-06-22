# Technical plan: SAMRAI AMR backend for adc_cpp

Date: 2026-06-11

**Status (2026-06-12).** Plan reviewed by a multi-agent panel (code verification +
> critique + vote). Verdict: **spike-first**, a gate must be cleared before any heavy
> coding: (1) **technical feasibility** (Kokkos device blocker `SharedSpace` host==device ->
> v1 CPU/MPI-only); (2) **proof of the intended value = scaling** (quantified crossover point
> native vs SAMRAI: SAMRAI lifts the coarse-replicated / `DistributionMapping`
> round-robin / global-collective ceilings; native stays more capable on the *physics*
> side, anisotropic elliptic/Schur). The 7 open decisions (section 10) are settled unanimously
> (CPU/MPI-first, `distribute_coarse=false`->error, constant C/F interp, adc reflux kept,
> regrid coverage/nesting, restart out of scope). **Consequence**: distributed elliptic
> (scalar HYPRE/FAC) is *scaling-critical*, a long pole, not optional; M8 = parity tier
> only. Tracking: Linear milestone "SAMRAI AMR Backend", gate **ADC-126**,
> value **ADC-130**, scaling benchmark **ADC-162**; decision in
> `SAMRAI_BACKEND_DECISION.md`.

Goal: plug SAMRAI in as an optional AMR backend in `adc_cpp`, without breaking the existing
public C++/Python APIs or the native backend parity tests.

Sources read:

- Local `adc_cpp` code: `include/adc/amr`, `include/adc/mesh`,
  `include/adc/numerics/time`, `include/adc/coupling`, `include/adc/runtime`,
  `python/`, `tests/`, `docs/ARCHITECTURE.md`,
  `docs/AMR_MULTIBLOCK_DESIGN.md`, `docs/AMR_REGRID_UNION_TAGS_DESIGN.md`,
  `docs/AMR_CONDENSED_SCHUR_DESIGN.md`, `docs/BACKEND_COVERAGE.md`.
- Official SAMRAI documentation:
  [GitHub LLNL/SAMRAI](https://github.com/LLNL/SAMRAI),
  [LLNL project page](https://computing.llnl.gov/projects/samrai),
  [LLNL software page](https://computing.llnl.gov/projects/samrai/software),
  [SAMRAI Concepts](https://computing.llnl.gov/sites/default/files/SAMRAI-Concepts_0.pdf).
- SAMRAI source verified locally: `hier::PatchHierarchy`, `hier::PatchLevel`,
  `hier::Patch`, `hier::VariableDatabase`, `pdat::CellVariable`,
  `pdat::CellData`, `pdat::FaceVariable`, `pdat::SideVariable`,
  `mesh::GriddingAlgorithm`, `mesh::TagAndInitializeStrategy`,
  `mesh::BergerRigoutsos`, `mesh::TreeLoadBalancer`,
  `xfer::RefineAlgorithm`, `xfer::CoarsenAlgorithm`,
  `solv::CellPoissonFACSolver`, `solv::CellPoissonFACOps`,
  `solv::CellPoissonHypreSolver`.

## 1. Current state to preserve

### Public C++ entries

To keep as the high-level surface:

- `include/adc/runtime/amr_system.hpp`
  - `adc::runtime::AmrSystemConfig`
  - `adc::runtime::AmrSystem`
- `include/adc/runtime/amr/amr_runtime.hpp`
  - runtime multi-block engine, shared hierarchy, regrid by union of tags.
- `include/adc/coupling/amr/amr_coupler_mp.hpp`
  - historical mono-block path.
- `include/adc/coupling/system/amr_system_coupler.hpp`
  - compile-time multi-block path.
- `include/adc/numerics/time/amr/advance/amr_advance.hpp`
  - `LevelHierarchy`, `advance_amr`.

These APIs must keep compiling with the native backend without SAMRAI.

### Public Python entries

To keep with no default behavioral change:

- `python/bindings/core/bindings.cpp`
  - `AmrSystemConfig` bindings.
- `python/bindings/amr/amr_system.cpp`
  - `AmrSystem` bindings.
- `python/adc/__init__.py`
  - Python `AmrSystem` facade, sugar `add_block`, `add_equation`,
    `patch_rectangles`, `write`, `checkpoint`, `restart`.

Python methods to keep:

- configuration: `n`, `L`, `periodic`, `regrid_every`, `distribute_coarse`,
  `coarse_max_grid`.
- models: `add_block`, `add_native_block`, `add_equation`.
- refinement: `set_refinement`, `set_phi_refinement`.
- elliptic: `set_poisson`, `set_density`, `set_conservative_state`,
  `potential`.
- coupling: `add_coupled_source`, `set_source_stage`, `set_magnetic_field`,
  `add_dt_bound`, `last_dt_bound`.
- advancing: `step`, `advance`, `step_cfl`, `time`.
- diagnostics: `nx`, `n_blocks`, `block_names`, `n_patches`,
  `patch_boxes`, `mass`, `density`.

Proposed addition: an optional backend selector, for example
`AmrSystemConfig::amr_backend = "native" | "samrai"`, exposed in Python as
`cfg.amr_backend`. The default value stays `"native"`.

## 2. Current adc_cpp structures to replace or adapt

| Need | Current implementation | Current files |
|---|---|---|
| AMR levels | `AmrHierarchy`, `AmrLevelStack`, `AmrLevelMP` | `include/adc/amr/hierarchy/amr_hierarchy.hpp`, `include/adc/coupling/amr/amr_level_storage.hpp`, `include/adc/numerics/time/amr/levels/amr_subcycling.hpp` |
| Patches | `Box2D`, `BoxArray`, local/global index of `MultiFab` | `include/adc/mesh/index/box2d.hpp`, `include/adc/mesh/layout/box_array.hpp`, `include/adc/mesh/storage/multifab.hpp` |
| Hierarchy | vectors of levels and `MultiFab`, fixed ref ratio 2 | `amr_hierarchy.hpp`, `amr_runtime.hpp`, `amr_coupler_mp.hpp` |
| Distribution | `DistributionMapping` round-robin/explicit | `include/adc/mesh/layout/distribution_mapping.hpp` |
| Data | `MultiFab` -> contiguous `Fab2D`, ghosts, `sync_host/device` | `include/adc/mesh/storage/multifab.hpp`, `include/adc/mesh/storage/fab2d.hpp` |
| Intra-level ghost fill | hand-rolled MPI halos | `include/adc/mesh/boundary/fill_boundary.hpp` |
| Physical BC | `BCRec`, `fill_physical_bc`, `fill_ghosts` | `include/adc/mesh/boundary/physical_bc.hpp` |
| Interpolation/restriction | `interpolate`, `average_down`, `parallel_copy` | `include/adc/mesh/layout/refinement.hpp` |
| Tags and clustering | `TagBox`, `tag_cells`, `grow_tags`, `berger_rigoutsos` | `include/adc/amr/tagging/tag_box.hpp`, `include/adc/amr/regridding/regrid.hpp`, `include/adc/amr/tagging/cluster.hpp` |
| Regrid production | fine layout imposed, multi-block union already implemented | `include/adc/coupling/amr/amr_regrid_coupler.hpp`, `include/adc/runtime/amr/amr_runtime.hpp` |
| Reflux | `FluxRegister`, `CoverageMask`, `CoarseFineInterface` | `include/adc/numerics/time/amr/levels/amr_patch_range.hpp`, `include/adc/numerics/time/amr_reflux*.hpp` |
| Subcycling | Berger-Oliger ratio 2, average-down, reflux | `include/adc/numerics/time/amr/levels/amr_subcycling.hpp` |
| Elliptic | `GeometricMG`, limited `CompositeFacPoisson`, coarse solve + injection by default | `include/adc/numerics/elliptic/*`, `include/adc/coupling/amr/amr_coupler_mp.hpp` |
| Physics | local models, fluxes, sources, CFL, DSL/native | `include/adc/physics`, `include/adc/numerics`, `include/adc/runtime` |

Conclusion: SAMRAI must replace hierarchy management, patches,
AMR transfers and regridding. The physics models, numerical kernels,
DSL/Python, coupled sources and adc diagnostics must stay on top.

## 3. What SAMRAI brings

Per the LLNL documentation, SAMRAI structures an AMR computation into nested
levels, each level being covered by logical rectangular patches with contiguous
data and ghost cells. The software page lists the useful packages:

- `hier`: `Patch`, `PatchLevel`, `PatchHierarchy`, `VariableDatabase`,
  box computation and distributed metadata.
- `pdat`: cell/node/face/side centered data, in particular
  `CellData<T>`, `FaceData<T>`, `SideData<T>`.
- `mesh`: hierarchy construction, regridding, clustering of tagged
  cells, load balancing.
- `xfer`: intra-level communications, refinement/coarsening between levels,
  time/space interpolation, physical BC hooks.
- `algs`: level integrators, local time cycling. Not to adopt at the
  first milestone to limit risk.
- `solv`: data vectors on the hierarchy, PETSc/Sundials/HYPRE interfaces,
  FAC Poisson solver.
- `geom`: cartesian geometry and refine/coarsen operator registers.

SAMRAI classes to use directly:

- Hierarchy: `SAMRAI::hier::PatchHierarchy`, `PatchLevel`, `Patch`,
  `Box`, `BoxContainer`, `IntVector`, `Index`.
- Variables: `SAMRAI::hier::VariableDatabase`, `VariableContext`,
  `PatchData`.
- Data: `SAMRAI::pdat::CellVariable<double>`,
  `CellData<double>`, `FaceVariable<double>` or `SideVariable<double>`.
- Geometry: `SAMRAI::geom::CartesianGridGeometry`.
- Regrid: `SAMRAI::mesh::GriddingAlgorithm`,
  `TagAndInitializeStrategy`, `BergerRigoutsos`, `TreeLoadBalancer`.
- Transfers: `SAMRAI::xfer::RefineAlgorithm`, `RefineSchedule`,
  `RefinePatchStrategy`, `CoarsenAlgorithm`, `CoarsenSchedule`,
  `CoarsenPatchStrategy`.
- Standard operators: `CONSTANT_REFINE`, `LINEAR_REFINE`,
  `CONSERVATIVE_LINEAR_REFINE`, `CONSERVATIVE_COARSEN` or weighted average
  via `geom::CartesianCellDoubleWeightedAverage`, depending on the field.
- Elliptic: `SAMRAI::solv::CellPoissonFACSolver`,
  `CellPoissonFACOps`, `CellPoissonHypreSolver`,
  `PoissonSpecifications`.

## 4. Target architecture

Principle: introduce an AMR backend boundary. The application code keeps the
`AmrSystem` facade; the native backend stays the reference; SAMRAI becomes an
alternative implementation behind an adapter.

```text
Python adc.AmrSystem / C++ adc::runtime::AmrSystem
        |
        v
AmrSystemBackend (new boundary)
        |
        +-- NativeAmrBackend
        |     reuses AmrRuntime, AmrCouplerMP, MultiFab, AmrHierarchy
        |
        +-- SamraiAmrBackend
              owns PatchHierarchy + VariableDatabase + xfer schedules
              exposes views compatible with the adc_cpp kernels
```

Proposed new components:

- `include/adc/amr/backend.hpp`
  - enum `AmrBackendKind { Native, Samrai }`.
  - minimal common interface for `step`, `advance`, `mass`,
    `density`, `potential`, `patch_boxes`, `n_patches`.
- `include/adc/samrai/samrai_fwd.hpp`
  - confined SAMRAI includes and `ADC_USE_SAMRAI` guard.
- `include/adc/samrai/box_adapter.hpp`
  - conversions `Box2D <-> hier::Box`, `BoxArray <-> BoxContainer`.
- `include/adc/samrai/cell_data_view.hpp`
  - view from `pdat::CellData<double>` to a `Fab2D` /
    `Array4` compatible access to reuse the adc kernels.
- `include/adc/samrai/hierarchy_adapter.hpp`
  - owner of `PatchHierarchy`, `CartesianGridGeometry`,
    `CURRENT`, `NEW`, `SCRATCH` contexts, patch data ids.
- `include/adc/samrai/transfer_adapter.hpp`
  - `RefineAlgorithm`, `CoarsenAlgorithm`, schedule cache, physical BC.
- `include/adc/samrai/regrid_adapter.hpp`
  - adc implementation of `mesh::TagAndInitializeStrategy`.
- `include/adc/samrai/flux_register_adapter.hpp`
  - phase 1: reuse the adc logic on SAMRAI views.
  - phase 2: migration to SAMRAI face/side flux if worthwhile.
- `include/adc/samrai/elliptic_adapter.hpp`
  - phase 1: parity with the current adc solver.
  - phase 2: SAMRAI/HYPRE FAC.
- `include/adc/runtime/amr_system_samrai.hpp`
  - concrete backend called by `AmrSystem`.

Responsibilities transferred to SAMRAI:

- creation and ownership of the patch hierarchy;
- distributed metadata and neighborhoods;
- load balancing;
- generation of fine patches via `GriddingAlgorithm`;
- patch data allocation per variable/context;
- ghost fills and coarse/fine transfers via schedules;
- average-down via `CoarsenAlgorithm`;
- FAC elliptic solver only after baseline parity.

Responsibilities remaining in adc_cpp:

- C++/Python API;
- definition of physics models, DSL, native ABI;
- flux, reconstruction, source, CFL kernels;
- multi-block coupled sources;
- choice of refinement criteria;
- regrid cadence exposed to the user;
- `mass`, `density`, `potential`, `patch_boxes` diagnostics;
- parity tests.

## 5. Mapping adc_cpp -> SAMRAI

| adc_cpp current | SAMRAI target | adc_cpp adapter | Note |
|---|---|---|---|
| `Box2D` | `hier::Box`, `hier::Index` | `samrai/box_adapter.hpp` | Inclusive indices on both sides; set `tbox::Dimension(2)`. |
| `BoxArray` | `PatchLevel` boxes, `hier::BoxContainer` | `samrai/box_adapter.hpp` | In SAMRAI, the layout is produced by `GriddingAlgorithm`; `BoxArray` becomes mostly a diagnostic view. |
| `DistributionMapping` | `PatchLevel` processor mapping, `TreeLoadBalancer` | `hierarchy_adapter.hpp` | Stop driving the distribution by hand; only expose the result if needed. |
| `MultiFab` | `VariableDatabase` + `PatchData` ids + `CellData<double>` per patch | `samrai_multifab_view.hpp` or `cell_data_view.hpp` | Adapt only the useful subset (`ncomp`, `ngrow`, patch iteration, pointer). |
| `Fab2D` | `pdat::CellData<double>` | `cell_data_view.hpp` | Check SAMRAI column-major memory layout vs `Fab2D`; encapsulate stride. |
| `AmrHierarchy` | `hier::PatchHierarchy` | `hierarchy_adapter.hpp` | The native backend keeps `AmrHierarchy`; the SAMRAI backend exposes an equivalent facade. |
| `AmrLevelMP` | `{level_number, data_id, aux_id, dx, dy}` | `level_view.hpp` | Must no longer own `MultiFab`; it references the SAMRAI patch data. |
| `AmrLevelStack` | `PatchHierarchy` + contexts | `hierarchy_adapter.hpp` | Replaces the vectors of levels and aux. |
| `TagBox` | integer tag patch data (`CellData<int>`) | `regrid_adapter.hpp` | `tagCellsForRefinement` calls the adc predicates then sets `tag_index` to 1. |
| `ClusterParams`, `berger_rigoutsos` | `BergerRigoutsos` + `GriddingAlgorithm` input DB | `regrid_adapter.hpp` | Map `max_grid`, `grow`, `margin`, proper nesting and efficiency. |
| `fill_boundary` | intra-level `xfer::RefineSchedule` | `transfer_adapter.hpp` | Schedules cached and invalidated after regrid. |
| `fill_physical_bc` | `xfer::RefinePatchStrategy` | `transfer_adapter.hpp` | Port `BCRec` to SAMRAI callbacks. |
| `fill_cf_ghost_cell`, `mf_fill_fine_ghosts_*` | `RefineAlgorithm` coarse->fine with time/space interpolation | `transfer_adapter.hpp` | First reproduce the constant adc interpolation; add linear afterward. |
| `parallel_copy` | xfer schedule between data ids/contexts | `transfer_adapter.hpp` | Useful for regrid and old/new contexts. |
| `average_down` | `CoarsenAlgorithm`, `CoarsenSchedule` | `transfer_adapter.hpp` | Conservative operator for conserved fields. |
| `FluxRegister` | phase 1: adc adapter; phase 2: `FaceData`/`SideData` + SAMRAI sync | `flux_register_adapter.hpp` | Risk of numerical change; keep adc for initial parity. |
| `CoverageMask` | `PatchHierarchy` + regions covered by fine levels | `coverage_adapter.hpp` | Can be derived from the SAMRAI hierarchy; keep current tests. |
| `GeometricMG` | phase 1: adc solver; phase 2: `CellPoissonFACSolver` | `elliptic_adapter.hpp` | SAMRAI composite FAC is a separate decision. |
| `CompositeFacPoisson` | `CellPoissonFACSolver`/`CellPoissonFACOps` | `elliptic_adapter.hpp` | Long-term target for multi-level composite solve. |
| `AmrRuntime` | orchestration kept, storage delegated | `amr_system_samrai.hpp` | Keep tag union and multi-block; change data/hierarchy backend. |
| Physics models | no SAMRAI mapping | `cell_data_view.hpp` | The kernels keep reading/writing adc views. |

## 6. APIs to preserve and acceptable changes

### Stable

- `adc::runtime::AmrSystemConfig` keeps its existing fields.
- `adc::runtime::AmrSystem` keeps its public methods.
- Python `adc.AmrSystem` keeps its methods and return values.
- `patch_boxes()` still returns `PatchBox`/rectangles in the exposed order.
- `density()` and `potential()` keep the same evaluation conventions.
- `regrid_every == 0` still means frozen hierarchy.
- Native backend stays bit-identical and tested.

### Compatible additions

- `AmrSystemConfig::amr_backend`.
- `AmrSystemConfig::poisson_backend`, later:
  `"adc"` by default, `"samrai_fac"` opt-in.
- `ADC_USE_SAMRAI` in CMake; without this option, requesting `"samrai"` raises an
  explicit error.
- Parity tests comparing `"native"` and `"samrai"`.

### Acceptable internal changes

- The AMR algorithms become generic over a data facade instead
  of taking `MultiFab` directly.
- `DistributionMapping` is no longer the source-of-truth object in the SAMRAI backend.
- `BoxArray` becomes a view/diagnostic in the SAMRAI backend.
- The communication schedules are cached and rebuilt in
  `resetHierarchyConfiguration`.

### Changes to avoid at the first milestone

- Replace the physics kernels with the `SAMRAI::algs` integrators.
- Introduce a per-species hierarchy.
- Enable SAMRAI FAC as the default solver.
- Change the Python semantics of `add_block`, `set_refinement`, `step`.

## 7. Concrete implementation order

### M0 - Build and SAMRAI dependency

Files:

- `CMakeLists.txt`
- `python/CMakeLists.txt`
- new `cmake/FindSAMRAI.cmake` if `find_package(SAMRAI CONFIG)` is not
  available in the target environment.
- `docs/HPC_SPACK_GUIDE.md`

Work:

- Add `option(ADC_USE_SAMRAI "Enable SAMRAI AMR backend" OFF)`.
- Link SAMRAI only if the option is active.
- Check the SAMRAI variants: MPI, HDF5, HYPRE, RAJA, Umpire.
- Document a reproducible Spack/CMake command.

Exit criterion:

- Build without SAMRAI unchanged.
- Build with `ADC_USE_SAMRAI=ON` compiles an empty test that includes
  `SAMRAI/hier/PatchHierarchy.h`.

### M1 - Backend selector with no behavior change

Files:

- `include/adc/runtime/amr_system.hpp`
- `python/bindings/core/bindings.cpp`
- `python/bindings/amr/amr_system.cpp`
- `python/adc/__init__.py`
- `tests/test_amr_system_contract.cpp`
- `python/tests/test_bindings.py`

Work:

- Add `amr_backend`, default value `"native"`.
- If `"samrai"` is requested without `ADC_USE_SAMRAI`, raise a clear error.
- Wire `NativeAmrBackend` as the current path.

Exit criterion:

- All current tests stay green on the native backend.
- New Python test: `cfg.amr_backend = "native"` and `"samrai"` without a SAMRAI
  build produce the expected behaviors.

### M2 - Box, geometry and data view adapters

New files:

- `include/adc/samrai/samrai_fwd.hpp`
- `include/adc/samrai/box_adapter.hpp`
- `include/adc/samrai/cell_data_view.hpp`
- `tests/test_samrai_box_adapter.cpp`
- `tests/test_samrai_cell_data_view.cpp`
- `tests/CMakeLists.txt`

Work:

- Convert `Box2D` to `hier::Box` and back.
- Build a view over `pdat::CellData<double>::getPointer(depth)`.
- Validate strides, ghosts, inclusive bounds and components.

Exit criterion:

- Serial unit tests.
- Cell-by-cell write/read comparison with `Fab2D`.

### M3 - One-level SAMRAI hierarchy

Files:

- `include/adc/samrai/hierarchy_adapter.hpp`
- `include/adc/runtime/amr_system_samrai.hpp`
- `include/adc/runtime/amr_system.hpp`
- `python/bindings/amr/amr_system.cpp`

Work:

- Create `CartesianGridGeometry`.
- Create `PatchHierarchy`.
- Register `state`, `aux`, `phi`, `rhs` variables via
  `VariableDatabase::registerVariableAndContext`.
- Allocate patch data on level 0.
- Implement `patch_boxes`, `n_patches`, `density`, `mass` reading from SAMRAI.

Exit criterion:

- One-level SAMRAI `AmrSystem`, no regrid, no subcycling.
- Parity with `test_amr_spatial_parity` or a minimal equivalent test.

### M4 - Intra-level ghost fill and physical BC

Files:

- `include/adc/samrai/transfer_adapter.hpp`
- `include/adc/samrai/physical_bc_strategy.hpp`
- `tests/test_samrai_fill_boundary.cpp`
- `tests/test_samrai_physical_bc.cpp`

Work:

- Register `RefineAlgorithm` for same-level copy.
- Implement `RefinePatchStrategy` that calls the `BCRec` logic.
- Cache `RefineSchedule` per level/data_id/context.
- Invalidate the schedules after regrid.

Exit criterion:

- Parity with `test_fill_boundary`, `test_physical_bc`.
- MPI variant: parity with `test_mpi_fillboundary`.

### M5 - Coarse/fine transfers

Files:

- `include/adc/samrai/transfer_adapter.hpp`
- `include/adc/numerics/time/amr/reflux/amr_flux_helpers.hpp` if a generic interface
  is needed.
- `tests/test_samrai_refinement.cpp`
- `tests/test_samrai_cf_interface.cpp`

Work:

- Map `interpolate` to `RefineAlgorithm`.
- Map `average_down` to `CoarsenAlgorithm`.
- First reproduce the current constant interpolation for parity.
- Then add `LINEAR_REFINE` / `CONSERVATIVE_LINEAR_REFINE` as an option.

Exit criterion:

- Parity with `test_refinement`, `test_cf_interface`.
- average-down conservation on conserved fields.

### M6 - SAMRAI regrid via TagAndInitializeStrategy

Files:

- `include/adc/samrai/regrid_adapter.hpp`
- `include/adc/runtime/amr_system_samrai.hpp`
- `tests/test_samrai_regrid.cpp`
- `tests/test_samrai_multiblock_regrid_union.cpp`

Work:

- Implement `SamraiTagAndInitializeStrategy`.
- In `tagCellsForRefinement`, call the adc predicates per block and do
  the union of the tags.
- In `initializeLevelData`, allocate the data, interpolate from coarse
  and copy the old fine if available.
- In `resetHierarchyConfiguration`, rebuild schedules and caches.
- Use `GriddingAlgorithm::makeCoarsestLevel`, `makeFinerLevel`,
  `regridAllFinerLevels`.
- Configure `BergerRigoutsos` and `TreeLoadBalancer`.

Exit criterion:

- Structural parity with `test_regrid`.
- Multi-block parity with `test_amr_multiblock_regrid_union`.
- Shared layout invariant verified after each regrid.

### M7 - Subcycling and reflux

Files:

- `include/adc/samrai/flux_register_adapter.hpp`
- `include/adc/numerics/time/amr/levels/amr_subcycling.hpp`
- `include/adc/numerics/time/amr/levels/amr_patch_range.hpp`
- `include/adc/runtime/amr_system_samrai.hpp`
- `tests/test_samrai_flux_register.cpp`
- `tests/test_samrai_amr_diffusion.cpp`

Work:

- Phase 1: keep the adc flux registers on SAMRAI views.
- Adapt `PatchRange`, `CoverageMask`, `CoarseFineInterface`.
- Keep ratio 2 and the current cadence.
- Compare the reflux corrections against the native backend.

Exit criterion:

- Parity with `test_flux_register`, `test_coverage_mask`,
  `test_patch_range`, `test_amr_diffusion`.
- Mass conservation on coarse/fine interfaces.

### M8 - Elliptic, adc parity phase

Files:

- `include/adc/samrai/elliptic_adapter.hpp`
- `include/adc/coupling/amr/amr_coupler_mp.hpp`
- `include/adc/runtime/amr/amr_runtime.hpp`
- `tests/test_samrai_amr_potential.cpp`

Work:

- Keep the current behavior: coarse solve + aux injection, or temporary native
  mirror for `GeometricMG`.
- Copy RHS/phi between SAMRAI patch data and adc views/mirrors.
- Do not enable SAMRAI FAC by default.

Exit criterion:

- Parity with `test_amr_potential`, `test_poisson_convergence` at the
  expected level.
- Python tests `test_poisson_composite.py`, `test_dsl_elliptic.py` adapted
  to a parameterized backend mode.

### M9 - Elliptic, SAMRAI/HYPRE FAC phase

Files:

- `include/adc/samrai/elliptic_adapter.hpp`
- `tests/test_samrai_fac_poisson.cpp`
- `tests/test_samrai_fac_variable_eps.cpp`
- backend documentation.

Work:

- Add `poisson_backend = "samrai_fac"`.
- Use `CellPoissonHypreSolver`, `CellPoissonFACOps`,
  `CellPoissonFACSolver`.
- Map Poisson boundary conditions to `SimpleCellRobinBcCoefs` or a
  dedicated BC strategy.
- Map `D`, `C`, scalar/tensor epsilon coefficients if supported.

Exit criterion:

- 2D manufactured solution.
- Composite coarse/fine test.
- Comparison with the current `CompositeFacPoisson` on the edge cases already
  covered.

### M10 - Python, CI and documentation

Files:

- `python/tests/test_amr_multiblock.py`
- `python/tests/test_dsl_hybrid_amr.py`
- `python/tests/test_dsl_production_amr.py`
- `docs/BACKEND_COVERAGE.md`
- `docs/sphinx/reference/backend_matrix.md`
- `docs/sphinx/amr/index.md`

Work:

- Parameterize some Python tests over `amr_backend`.
- Add a SAMRAI column to the backend matrix.
- Document limitations: 2D, ratio 2, CPU/MPI initial if Kokkos/SAMRAI
  device memory is not settled.

Exit criterion:

- Native CI unchanged.
- Opt-in SAMRAI CPU/MPI job.
- Documented parity.

## 8. Validation tests

### New unit tests

- `test_samrai_box_adapter`
- `test_samrai_cell_data_view`
- `test_samrai_hierarchy_adapter`
- `test_samrai_fill_boundary`
- `test_samrai_physical_bc`
- `test_samrai_refinement`
- `test_samrai_cf_interface`
- `test_samrai_regrid`
- `test_samrai_flux_register`
- `test_samrai_fac_poisson`

### Existing adc_cpp tests to replay on the SAMRAI backend

Mesh and transfers:

- `test_box2d`, `test_box_array` via adapters.
- `test_multifab` via the `SamraiMultiFabView` facade.
- `test_fill_boundary`, `test_physical_bc`, `test_refinement`.

AMR primitives:

- `test_amr_hierarchy`
- `test_cluster` as native reference only, plus structural
  comparison against `BergerRigoutsos`.
- `test_regrid`
- `test_flux_register`
- `test_coverage_mask`
- `test_patch_range`
- `test_cf_interface`
- `test_load_balance`

AMR runtime:

- `test_amr_system_coupler`
- `test_amr_source_covered_cells`
- `test_amr_composite_source_conservation`
- `test_amr_stride_cadence`
- `test_amr_layout_guard`
- `test_amr_spatial_parity`
- `test_amr_system_contract`
- `test_amr_system_twoblock`
- `test_amr_multiblock_regrid_union`
- `test_amr_multiblock_compiled`
- `test_amr_regrid_mpi_parity`

Elliptic:

- `test_geometric_mg`
- `test_poisson_convergence`
- `test_amr_potential`
- `test_composite_fac_poisson`
- `test_composite_fac_tensor`
- new SAMRAI FAC MMS.

MPI:

- `test_mpi_fillboundary`
- `test_mpi_redistribute`
- `test_mpi_coupler_inject`
- `test_mpi_mbox_parity`
- `test_mpi_hybrid_mbox_parity`
- `test_mpi_amr_compiled_parity`
- `test_mpi_amr_twoblock_parity`
- `test_mpi_amr_distributed_coarse`
- `test_mpi_poisson`

Python:

- `python/tests/test_bindings.py`
- `python/tests/test_amr_patch_boxes.py`
- `python/tests/test_amr_multiblock.py`
- `python/tests/test_dsl_hybrid_amr.py`
- `python/tests/test_dsl_production_amr.py`
- `python/tests/test_poisson_composite.py`
- `python/tests/test_dsl_elliptic.py`

### Numerical criteria

- Native backend: bit-identical relative to the current state.
- SAMRAI backend, levels/patches: exact equality of the exposed boxes when the
  scenario imposes the same layout; otherwise structural invariants and coverage.
- SAMRAI backend, fields: relative L_inf and L1 norm with explicit tolerance,
  not necessarily bit-identical because of the MPI/schedule ordering.
- Conservation: mass per block conserved to machine tolerance on tests without
  source; explicit source/reflux balance on tests with source.
- Regrid: no block loses its shared layout; `same_layout_or_throw` or
  equivalent after each regrid.

## 9. Technical risks

1. Device memory and execution
   - adc_cpp relies on Kokkos; SAMRAI offers RAJA/Umpire as an option.
   - Decision to make: SAMRAI CPU/MPI backend first, or device
     integration from the start.
   - Risk: host/device copies if `pdat::CellData` is not directly
     usable by the Kokkos kernels.

2. Memory layout
   - `CellData` is stored in Fortran-like order, with component depth.
   - The views must encapsulate strides and ghosts; no kernel must
     assume the native `Fab2D` layout.

3. Regrid parity
   - SAMRAI `BergerRigoutsos` may produce different boxes from the
     native clustering.
   - The tests must separate numerical parity, tag coverage and
     nesting invariants.

4. Reflux
   - This is the most sensitive area for conservation.
   - Phase 1 must keep the adc arithmetic; migration to `FaceData` /
     `SideData` only after parity.

5. Composite elliptic
   - The current default behavior is mostly coarse solve + injection,
     with a limited composite.
   - SAMRAI FAC is relevant but must stay a separate, opt-in milestone.

6. `distribute_coarse` semantics
   - The native backend supports a coarse-replicated mode.
   - SAMRAI naturally handles a distributed hierarchy; we must decide whether
     `distribute_coarse=false` is emulated, refused, or ignored with a warning on
     the SAMRAI backend.

7. Schedules invalid after regrid
   - `RefineSchedule` and `CoarsenSchedule` stay valid only as long as the
     patches do not change.
   - `resetHierarchyConfiguration` must rebuild everything.

8. Multi-block
   - adc_cpp imposes a hierarchy shared by union of tags.
   - SAMRAI must not introduce a per-species hierarchy; a single
     `PatchHierarchy`, several variables/data ids.

9. HPC build
   - SAMRAI may depend on MPI, HDF5, HYPRE, RAJA, Umpire depending on options.
   - The backend must stay optional so as not to weigh down the normal CI.

10. License and distribution
    - SAMRAI is distributed under an LLNL/LGPL license; check the implications
      of static/dynamic linking for the deliverables.

## 10. Decisions to make before heavy coding

- SAMRAI backend v1 CPU/MPI only, or immediate GPU goal?
- `distribute_coarse=false`: emulation, explicit error, or ignored option?
- Coarse/fine interpolation v1: constant for strict parity, or SAMRAI linear
  directly with a change of reference?
- Flux: keep the adc `FluxRegister` in v1, or switch to `FaceData` /
  `SideData` immediately?
- SAMRAI FAC: separate milestone with `poisson_backend="samrai_fac"`?
- Regrid tests: do we require the same boxes as the native backend, or only
  same coverage/nesting and same solution to tolerance?
- Checkpoint/restart: stays out of scope, or mapping to HDF5/SAMRAI restart?

## 11. Summary

The least-risky path is:

1. Add a backend selector without changing the native backend.
2. Build a SAMRAI adaptation layer that exposes views compatible
   with the adc_cpp kernels.
3. Port hierarchy/data/ghosts/transfers/regrid first.
4. Keep adc reflux and elliptic for the first parity.
5. Enable SAMRAI/HYPRE FAC only after validating the baseline AMR backend.

This way, SAMRAI takes the responsibilities it is built for
(`PatchHierarchy`, patch data, schedules, gridding, load balancing), while
adc_cpp keeps its main value: physics models, DSL/Python,
multi-block coupling, numerical kernels and public contracts.
