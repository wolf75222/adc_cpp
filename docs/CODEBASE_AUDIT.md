> **STATUS: historical.** A maintainability audit from 2026-06-06; its counts and findings reflect that snapshot, not the current code. Items worth acting on are tracked in Linear. Read it as a historical record (class D, see `DOC_QUALITY.md`).

# Maintainability audit of `adc_cpp`

Date: 2026-06-06.  
Base reviewed: `origin/master` / `9ba36f5` after PRs #118-#142, #135 and #141.
Open PR impacting the audit: #140 (AMR stride cadence).
Scope read: `include/adc/**/*.hpp`, `python/*.cpp`, `python/adc/*.py`, main architecture
docs. The GPU tests under `python/tests/gpu/` are classified as validation harness, not
as a production API.

Objective: verify that each class/file has a clear reason to exist, that responsibilities
do not mix, and identify the pieces to factor out, archive or harden. This document
is not a scientific roadmap: it is a code audit.

Associated comment convention: see
`docs/CODE_DOCUMENTATION_CONVENTION.md`. Each file must carry
a responsibility header, each non-trivial class must expose its contract and its constraints,
and internal comments must explain the numerical/MPI/GPU invariants rather than
paraphrase the lines.

## 1. Global reading

The library now has a readable architecture:

```text
PhysicalModel local
  -> operateurs numeriques
  -> donnees maillage/MultiFab
  -> backends execution/MPI/Kokkos
  -> runtime System/AmrSystem + bindings Python
```

The general direction is good: `adc_cpp` no longer contains an application solver named `DiocotronSolver`
in the public API; the cases live in `adc_cases`. The core provides generic bricks, a
DSL, compiled native paths, MPI and Kokkos.
Nuance: a few generic header comments still cite "diocotron" to explain
the origin of a polar/Schur work item. This is not a public application symbol, but the vocabulary
should be neutralized progressively to "polar ring", "ExB drift" or "test model".

The main risk is no longer "bad starting abstraction". The risk is the accumulation of
several generations of code that all remain present:

- `System` modern Python runtime.
- `AmrSystem` Python runtime, deliberately more limited.
- old header-only couplers (`Coupler`, `SystemCoupler`, `AmrCoupler`, `AmrCouplerMP`).
- multiple DSL paths: `dynamic`, `aot`, `production/native`.
- historical and multipatch AMR engines in one big header.

We must therefore keep the low-level bricks, but clarify which classes are the current API,
which classes are internal engines, and which classes are compatibility ones.

Recent status integrated into this audit:

- Schur is now a complete C++ stack on the uniform side: full tensor operator, brick
  `LorentzEliminator`, condensation builder, Krylov BiCGStab solver, stage
  `CondensedSchurSourceStepper`, Python binding `adc.Split` / `adc.CondensedSchur`.
- `System` gained per-block stride, `SourceImplicit`, mask `implicit_vars` / `implicit_roles`,
  set/get in primitive variables, explicit coupled DSL source, and an API that rejects several
  misleading paths instead of silently ignoring them.
- The polar geometry has a Phase 1 transport plus a standalone direct polar Poisson solver
  (FFT-theta + tridiag-r). The full wiring into `System.step` remains to be done.
- `AmrSystem` has progressed on WENO5/HLLC/Roe/primitive reconstruction, local source-only IMEX and
  mono-block DSL production. The multi-block AMR capstone is documented but not yet runtime.
- The CI was reorganized (#136) with Kokkos/ccache/ninja cache, Python auto-discovery and a split
  fast/full. The audit must therefore distinguish "test covered in CI" and "manual GH200 harness".
- #135 is now merged on `origin/master`: `Geometry` / `Box2D` are device-callable and the
  CFL goes through a global MPI reduction. This is a device/MPI validity fix, not a reason
  to remove the GH200 Schur/polar harness.
- #141 is now merged on `origin/master`: `AmrHierarchyLayout` and `same_layout_or_throw`
  exist as the first guard for shared AMR layout.

Open PRs not to be counted as "done" as long as they are not merged:

- #140 fixes the AMR `stride` cadence to hold-then-catch-up. As long as #140 is not merged,
  `AmrSystemCoupler` keeps the condition `macro_step_ % stride`, which advances a slow block on the first
  macro-step.

## 2. Architecture invariants to preserve

These invariants are the "rites" of the code: if a file breaks them, it becomes suspect.

| Invariant | Meaning |
|---|---|
| A `PhysicalModel` is local | No `MultiFab`, no MPI, no AMR, no global storage. |
| The geometry belongs to the mesh | `PolarMesh` / `CartesianMesh`, not `FiniteVolume(geometry=...)`. |
| The spatial scheme does not know the scenario | It composes reconstruction, Riemann, variables. |
| The Python runtime composes | Python chooses the bricks; the cell loops stay C++. |
| The GPU paths use named functors | Avoid fragile cross-TU device lambdas under nvcc. |
| `System` is multi-block | Species, global Poisson, coupled sources, stride/substeps. |
| `AmrSystem` is mono-block for now | Its public surface stays narrower than that of `System` as long as conservative multi-block AMR is not implemented. |
| Future conservative multi-block AMR | Common hierarchy, co-located cells for all evolved blocks; no species locally absent per patch. |
| The application names live outside the core | `diocotron`, `two_fluid_ap`, validations, presets: `adc_cases`. |
| Backend specified in the reports | Say `MPI CPU`, `Kokkos Cuda`, `MPI + Kokkos Cuda`, not just "MPI". |
| Global Schur separated from local sources | `SourceImplicit` = local implicit source; Schur = non-local condensed source stage. |

## 3. Map of responsibilities by module

| Module | Correct responsibility | Possible problem |
|---|---|---|
| `core/` | Concepts, states, variables, abstract blocks. | Stable. Independent of heavy numerics. |
| `physics/` | Local and composable physics bricks. | Keep generic; no named scenario. |
| `numerics/` | Reconstruction, flux, FV RHS, time, elliptic, Schur primitives. | Some headers are too big or too coupled. |
| `mesh/` | Box, MultiFab, halos, BC, cell execution. | Well separated, but `fill_boundary`/`physical_bc` must stay low level. |
| `amr/` | Tagging, clustering, abstract regrid. | Correct, small. |
| `coupling/` | Historical header-only and AMR coupling engines, coupled sources, Schur. | Most redundant zone. To classify public/internal/deprecated. |
| `runtime/` | C++ facades used by Python, DSL ABI, loaders. | Strong direction, but many paths. Needs strict naming. |
| `python/*.cpp` | Bindings and offline runtime implementation. | `python/bindings/system/base/system.cpp` is too big. |
| `python/adc/*.py` | Python user API and DSL. | Some docstrings remain old. |

## 4. Audit by file

### `include/adc/core`

| File | Classes / objects | Meaning | Boundary | Audit |
|---|---|---|---|---|
| `types.hpp` | `Real`, host/device macros | Minimal shared base. | Import Kokkos heavily. | OK. |
| `state.hpp` | `StateVec<N>`, `Aux`, canonical aux fields | Local device-callable state. | Becoming a grid container. | OK, but adding aux requires synchronization with `dsl.py`. |
| `variables.hpp` | `VariableRole`, `VariableSet`, ABI metadata | Give meaning to the components. | Drive the numerical computation directly. | Very useful for coupled sources and Schur. |
| `physical_model.hpp` | concepts `PhysicalModel`, `HyperbolicPhysicalModel`, `aux_comps` | Local contract of the model. | Carry a solver, a scheme or a mesh. | OK. It is the conceptual core. |
| `equation_block.hpp` | `EquationBlock` | Associates model + state + scheme + time. | Solve or schedule. | Good abstraction level, especially on the compiled C++ side. |
| `coupled_system.hpp` | `CoupledSystem`, `CoupledSystemLike` | Heterogeneous group of blocks. | Impose a physics or a time order. | OK. Keep simple. |
| `allocator.hpp` | `ManagedArena`, `ManagedAllocator` | Unified/device memory abstraction. | Hide the fences in the accessor. | Useful but to watch with Kokkos/CUDA. |
| `kokkos_env.hpp` | Kokkos init/finalize | Backend environment management. | Carry numerics. | OK. |

### `include/adc/physics`

| File | Classes / objects | Meaning | Boundary | Audit |
|---|---|---|---|---|
| `hyperbolic.hpp` | `ExBVelocity`, `ExBVelocityPolar`, `IsothermalFlux`, `CompressibleFlux` | Vars + flux + velocities. | Mix with source or Poisson. | Good. `ExBVelocityPolar` is a physical/local choice compatible with the polar mesh. |
| `euler.hpp` | `Euler` | Full compressible flux. | Becoming an Euler-Poisson solver. | OK. |
| `source.hpp` | `NoSource`, `PotentialForce`, `GravityForce` | Local per-cell sources. | Represent global implicit Schur. | OK. Coupled sources must stay out of this file. |
| `elliptic.hpp` | `ChargeDensity`, `BackgroundDensity`, `GravityCoupling` | Local per-block elliptic RHS. | Modify the elliptic operator. | OK. Important to clarify Schur: RHS only. |
| `composite.hpp` | `CompositeModel<H,S,E>` | Composes transport/source/elliptic. | Allow incoherent Vars/Flux combinations. | Good grain of abstraction. |
| `advection_diffusion.hpp` | `AdvectionDiffusion` | Reference transport-diffusion model. | Becoming the main user API. | TEST/VALIDATION brick (not used by adc_cases); used in `tests/test_weno5_ssprk3.cpp`. Classified example/validation, marked in the doc-comment. |
| `langmuir.hpp` | `LangmuirMode` | 0D IMEX Langmuir mode kernel. | Pollute the public API if the scenario is too specific. | TEST/VALIDATION brick (not used by adc_cases); `two_fluid_ap/` is standalone (TwoFluidAP2D). Marked in the doc-comment. |
| `two_fluid_isothermal.hpp` | `TwoFluidLinear` | Two-fluid isothermal IMEX kernel. | -- | TEST/VALIDATION brick (not used by adc_cases); same situation as LangmuirMode. Marked in the doc-comment. |
| `bricks.hpp` | aggregator include | Convenience header. | Carry logic. | OK. |

### `include/adc/numerics`

| File | Classes / objects | Meaning | Boundary | Audit |
|---|---|---|---|---|
| `reconstruction.hpp` | `NoSlope`, `Minmod`, `VanLeer`, `Weno5` | Point-wise reconstruction. | Loop over the grids. | OK. |
| `numerical_flux.hpp` | `RusanovFlux`, `HLLFlux`, `HLLCFlux`, `RoeFlux` | Local Riemann flux. | Know `MultiFab`. | OK. |
| `spatial_discretisation.hpp` | `SpatialDiscretisation` aliases | Name limiter + Riemann. | Carry the time. | OK. |
| `spatial_operator.hpp` | `assemble_rhs`, face/RHS kernels, `load_aux` | Cartesian FV operator. | Choose the model or the Python API. | Correct, but big and central. Keep as low-level engine. |
| `spatial_operator_polar.hpp` | `assemble_rhs_polar`, polar kernels | Polar FV divergence. | Be wired silently into `System`. | OK as Phase 1. An explicit runtime path is needed before user use. |
| `lorentz_eliminator.hpp` | `LorentzEliminator` | Local brick for Schur: analytic `B^{-1}`. | Solve the elliptic. | Good: small class with a clear responsibility. |

### `include/adc/numerics/elliptic`

| File | Classes / objects | Meaning | Boundary | Audit |
|---|---|---|---|---|
| `elliptic_solver.hpp` | concept `EllipticSolver` | Minimal solver contract. | Describe a complete problem. | OK. |
| `elliptic_problem.hpp` | `EllipticProblem`, `FieldPostProcess` | Problem description + phi/grad postprocess. | Replace `System::set_poisson`. | Good, but still not very central in the runtime. |
| `poisson_operator.hpp` | laplacian, residual, smoother kernels, conductor BC | Discrete elliptic operator. | Carry full solver policy. | Big but coherent. With Schur, extract a more formal `OperatorSpec` interface. |
| `geometric_mg.hpp` | `GeometricMG` | MG solver + levels + coefficients. | Carry the whole future elliptic family. | Works, but mixes solver, level storage, coefficients. To factor out before GMRES/Schur. |
| `krylov_solver.hpp` | `TensorKrylovSolver`, `KrylovResult` | Matrix-free BiCGStab solver for a non-symmetric tensor operator. | Replace MG everywhere. | C++ uniform OK; keep explicit GPU validation after the #135 fix. |
| `polar_poisson_solver.hpp` | `PolarPoissonSolver` | Direct Poisson on a polar ring: FFT in theta + tridiag in r. | Present itself as a distributed solver or a complete `System` runtime. | Good numerical choice. Single-rank/single box; Phase 2b runtime to do. |
| `poisson_fft.hpp` | `PoissonFFT` direct | Low-level single-rank FFT solver. | Pretend to be MPI `System`. | OK if the single-rank guard stays clear. |
| `poisson_fft_solver.hpp` | `PoissonFFTSolver`, `DistributedFFTSolver` | FFT wrappers for `MultiFab`. | Hide a layout incompatibility. | Clean MPI guard added. `DistributedFFTSolver` not routed in `System`. |

### `include/adc/numerics/time`

| File | Classes / objects | Meaning | Boundary | Audit |
|---|---|---|---|---|
| `time_integrator.hpp` | tags `SSPRK2`, `SSPRK3`, `TimePolicy`, `TimeTreatment` | Describe the compiled time choice. | Execute the step directly. | OK. |
| `time_steppers.hpp` | `ForwardEuler`, `SSPRK2Step`, `SSPRK3Step` | Integrators `take_step(rhs,U,dt)`. | Know Poisson. | OK, good seam. |
| `scheduler.hpp` | `advance_subcycled`, compiled stride | Per-block C++ scheduler. | Hide time effects in the model. | OK. The Python runtime has its own cadence. |
| `ssprk.hpp` | old SSPRK2 helper | Convenience/legacy. | Duplicate the modern driver. | Potentially to archive if no longer used. |
| `imex.hpp` | `imex_euler_step` | Small splitting brick. | Replace a real general IMEX policy. | OK but limited. |
| `implicit_stepper.hpp` | local source Newton, `ImplicitSourceStepper` | Local implicit source. | Sell the global Schur as a local source. | Useful but must be renamed/documented "local implicit". |
| `splitting.hpp` | Lie/Strang | Operator composition. | Know the physics. | OK. |
| `amr_multilevel.hpp` | old multi-level AMR | Simple historical AMR step. | Be the main path if `amr_reflux_mf` replaces it. | To classify legacy or validation. |
| `amr_reflux.hpp` | AMR Fab2D 2 levels | Pedagogical/low-level reflux. | Coexist as the main API. | Probably legacy. |
| `amr_reflux_mf.hpp` | `AmrLevelMF`, `AmrLevelMP`, `PatchRange`, `FluxRegister`, `CoverageMask`, `SubcyclingSchedule`, `CoarseFineInterface`, `advance_amr`, local IMEX source | Multipatch MultiFab AMR engine. | Stay a catch-all 1000-line file. | Biggest factoring point. Gap2 local IMEX merged, but file to split. |

### `include/adc/mesh`

| File | Classes / objects | Meaning | Boundary | Audit |
|---|---|---|---|---|
| `box2d.hpp` | `Box2D` | Rectangular discrete domain. | Know MPI. | OK. |
| `box_array.hpp` | `BoxArray` | List of boxes. | Store the data. | OK. |
| `distribution_mapping.hpp` | `DistributionMapping` | Boxes -> ranks mapping. | Do complex load balancing. | OK. |
| `fab2d.hpp` | `Fab2D`, `Array4`, `ConstArray4` | Local array + device views. | Know global AMR. | OK. |
| `multifab.hpp` | `MultiFab` | Distributed collection of `Fab2D`. | Carry the numerical schemes. | OK. |
| `fill_boundary.hpp` | local/MPI halo, periodicity | Ghost exchange. | Apply complex physical BC. | OK, device-sensitive. |
| `physical_bc.hpp` | `BCRec`, `fill_physical_bc` | Domain physical BC. | Exchange MPI. | OK. |
| `mf_arith.hpp` | saxpy, lincomb, norm, sum | Grid operations. | Carry solver logic. | OK. |
| `for_each.hpp` | execution/reductions/fence seams | Backend loops. | Contain numerical logic. | OK. |
| `geometry.hpp` | `Geometry`, `PolarGeometry` | Metric/domains. | Be chosen in `FiniteVolume`. | OK. |
| `refinement.hpp` | refinement/coarsen helpers | Coarse/fine indices. | Orchestrate a full AMR step. | OK, but check remaining device lambdas. |
| `box_hash.hpp` | `BoxHash` | Box lookup acceleration. | Carry AMR logic. | OK. |

### `include/adc/amr`

| File | Classes / objects | Meaning | Boundary | Audit |
|---|---|---|---|---|
| `tag_box.hpp` | `TagBox` | Mask of tagged cells. | Build the hierarchy. | OK. |
| `cluster.hpp` | Berger-Rigoutsos clustering | Boxes from tags. | Advance in time. | OK. |
| `regrid.hpp` | `RegridParams`, `regrid_level` | Abstract regridding. | Manage Poisson/coupling. | OK. |
| `amr_hierarchy.hpp` | `AmrHierarchy` | Hierarchy of levels. | Duplicate `AmrSystem`. | OK but to relate clearly to the main engine. |

### `include/adc/coupling`

| File | Classes / objects | Meaning | Boundary | Audit |
|---|---|---|---|---|
| `coupler.hpp` | `Coupler<Model>` | Historical mono-block Poisson + transport coupler. | Be the modern Python API. | Useful in tests/direct C++, but secondary API. |
| `system_coupler.hpp` | `SystemAssembler`, `SystemDriver`, alias `SystemCoupler` | Compiled multi-block coupling. | Redo the Python runtime. | Good tutor abstraction, but coexists with `runtime/System`. |
| `amr_system_coupler.hpp` | `AmrSystemCoupler`, `AmrSystemDriver` | CoupledSystem on compiled AMR. | Pretend to cover the `AmrSystem` runtime. | Useful but surface to clarify. |
| ~~`amr_coupler.hpp`~~ | ~~`AmrCoupler`~~ | Mono-box AMR REMOVED (C.2, June 2026). | Replaced by `AmrCouplerMP`. | No client #include; file deleted. |
| `amr_coupler_mp.hpp` | `AmrCouplerMP` + helpers | Coupled multipatch AMR engine. | Carry all the Python runtime logic. | Important, but big. Extract layout/read/write/inject helpers. |
| `amr_regrid_coupler.hpp` | regrid AMR coupler | Rebuild mono-block fine level. | Advance in time. | OK. For multi-block, the regrid by union of tags remains to be written. |
| `amr_level_storage.hpp` | `AmrLevelStack` | Level + aux storage. | Carry equations. | OK. |
| `amr_diagnostics.hpp` | AMR mass, velocity | Diagnostics. | Modify state. | OK. |
| `elliptic_rhs.hpp` | mono/multi-species RHS | RHS assembly from blocks. | Solve Poisson. | OK. |
| `coupled_source.hpp` | `NoCoupledSource`, coupled source concept | Source reading several blocks. | Replace Python runtime couplings. | Useful concept, now completed by the DSL bytecode. |
| `coupled_source_program.hpp` | `CoupledSourceKernel`, coupled source bytecode | Arbitrary DSL coupled source, evaluated in C++/device without Python callback. | Becoming an implicit solver. | Good P5 point. Conservation and MPI test to harden on composite AMR. |
| `coupling_policy.hpp` | Poisson cadence tags | Coupling policy. | Carry implementation. | OK. |
| `aux_fill.hpp` | aux drift, Bz fill | Auxiliary field helpers. | Becoming a solver. | OK. |
| `schur_condensation.hpp` | Schur operator/RHS builder | Assemble the condensed operator coefficients. | Solve or advance time. | Good split: builder only. |
| `condensed_schur_source_stepper.hpp` | `CondensedSchurSourceStepper` | Schur-condensed source stage. | Be a local `model.source`. | C++ uniform merged; GPU validation to track explicitly after #135. |
| `spectral_coupler.hpp` | mono-model FFT coupler | Spectral variant. | Be an MPI path without guard. | Probably secondary API. |

### `include/adc/runtime`

| File | Classes / objects | Meaning | Boundary | Audit |
|---|---|---|---|---|
| `system.hpp` | `SystemConfig`, `System` | C++ multi-block runtime facade exposed to Python. | Contain all the implementation. | Rich interface: stride, primitives, DSL source, Schur. Implementation too heavy in `python/bindings/system/base/system.cpp`. |
| `amr_system.hpp` | `AmrSystemConfig`, `AmrSystem`, hooks | Mono-block AMR runtime facade. | Promise multi-block runtime. | Mono-block but richer: WENO5/HLLC/Roe/primitive recon, DSL production, local IMEX. Multi-block capstone not merged. |
| `model_spec.hpp` | `ModelSpec` | Native brick tags from Python. | Becoming a named scenario. | OK. |
| `model_factory.hpp` | dispatch `ModelSpec` -> C++ types | Bridge runtime tags to templates. | Grow endlessly. | OK short term. To watch if bricks become numerous. |
| `block_builder.hpp` | builds compiled `advance/rhs` closures | Unifies `add_block` and native DSL. | Know Python. | Very important, good seam. |
| `grid_context.hpp` | `GridContext`, `BlockClosures` | Minimal data for closures. | Does not become owner of the mesh nor of the auxiliaries. | OK. |
| `dsl_block.hpp` | `add_compiled_model(System&)` | Zero-copy production path for DSL/C++ model. | Redo flat ABI. | Good. |
| `amr_dsl_block.hpp` | `add_compiled_model(AmrSystem&)` | AMR production path. | Hide the AMR limits. | OK but explicit mono-block; named multi-block = future capstone. |
| `compiled_block_abi.hpp` | flat AOT `.so` ABI | Runtime prototyping without shared C++ ABI. | Be presented as the main production. | To keep advanced/debug. |
| `dynamic_model.hpp` | `IModel`, `ModelAdapter` | Virtual host prototype. | GPU/MPI hot path. | To mark clearly experimental. |
| `abi_key.hpp` | ABI key headers/build | Avoids native loader UB. | Carry runtime logic. | OK. |
| `export.hpp` | symbol export macro | Loader boundary. | Business logic. | OK. |
| `wall_predicate.hpp` | common wall predicate | Simple conductor geometry. | Confuse with moving ring boundary. | OK. |

### `python/*.cpp`

| File | Classes / objects | Meaning | Boundary | Audit |
|---|---|---|---|---|
| `bindings.cpp` | pybind11 module `_adc` | Expose `System`, `AmrSystem`, configs. | Implement the simulation. | OK. |
| `amr_system.cpp` | `AmrSystem::Impl` | Offline AMR runtime. | Becoming multi-block without explicit registry. | Reasonable size. Still refuses the 2nd block; the capstone must add a type-erased registry, not just remove the throw. |
| `system.cpp` | `System::Impl`, loaders, Poisson, couplings, I/O, stride, Schur stage | Main multi-block runtime. | Stay a god class. | Biggest maintainability risk on the runtime side; the recent additions reinforce the need for extraction. |

### `python/adc/*.py`

| File | Meaning | Audit |
|---|---|---|
| `__init__.py` | Python user API: bricks, `System`, `AmrSystem`, `FiniteVolume`, time, couplings, Schur. | Good entry point. The explicit rejections #137 go in the right direction. Verify that each AMR/polar limit stays clear. |
| `dsl.py` | symbolic DSL, codegen, cache, backends `prototype/aot/production`, DSL couplings. | Functional but too big. The header docstring still says "we do NOT yet generate compiled code", which contradicts the current status. To fix as a priority. |
| `integrate.py` | Per-step Python integrators. | Useful for prototyping, off the hot path. |
| `elliptic.py` | Python elliptic facade. | Keep as a declaration, not a solver. |

### `python/tests/gpu`

These files are harness, not API bricks. Their purpose is to freeze device/MPI invariants:

| File | What it proves |
|---|---|
| `gpu_dsl_production_validate.cpp` | DSL production path on GPU. |
| `gpu_epm_validate.cpp` | advanced elliptic operators. |
| `gpu_amrsys_facade_validate.cpp` | AMR facade under nvcc. |
| `gpu_amr_bz_validate.cpp`, `gpu_amr_bz_mpi_validate.cpp` | AMR + Bz + device/MPI. |
| `phase1_transport.cpp` a `phase7_system.cpp` | historical progression of the GPU port. |
| `mpi6_fillboundary.cpp` | MPI/device halos. |
| `amrmpi_integrated.cpp` | integrated AMR MPI validation. |
| `diff_bin.cpp` | binary comparison. |

Audit: keep them, but separate `tests/gpu/` clearly from the API. Do not make it a runtime
dependency.

## 5. Classes/files to factor out as a priority

### P0 - Do not break, but clarify

1. **`python/bindings/system/base/system.cpp`**
   - Problem: it carries too many responsibilities: block registry, allocation, Poisson,
     couplings, native loaders, I/O state, stride, diagnostics, auxiliary fields.
   - Target refactor:
     - `SystemBlocks`: add/lookup blocks, ghosts, state I/O.
     - `SystemFields`: Poisson, aux, epsilon/kappa/Bz/Te.
     - `SystemCouplingsRuntime`: ionization/collision/thermal exchange.
     - `NativeLoader`: dlopen, ABI, symbols.
     - `SystemStepper`: `step`, `step_cfl`, stride/substeps.
   - Gain: make each future feature localized.

2. **`include/adc/numerics/time/amr_reflux_mf.hpp`**
   - Problem: 1016 lines, contains several abstraction levels.
   - Target split:
     - `amr_level.hpp`: `AmrLevelMF`, `AmrLevelMP`, `LevelHierarchy`.
     - `amr_patch_range.hpp`: `PatchRange`, coarse/fine interfaces.
     - `amr_flux_register.hpp`: flux/reflux.
     - `amr_subcycling.hpp`: schedules, reflux and `advance_amr`.
     - `amr_source_step.hpp`: local IMEX source stages on the hierarchy.
   - Gain: readable AMR, easier to port GPU/MPI.

3. **`coupling/` family**
   - `amr_coupler.hpp` (AmrCoupler mono-box) is REMOVED (C.2, June 2026): no client #include detected, replaced by AmrCouplerMP.
   - `Coupler` and `SpectralCoupler` are useful direct C++ facades, but secondary.
   - `SystemCoupler` is conceptually clean, but coexists with `runtime/System`.
   - Action: add a "public vs internal vs deprecated" page.

### P1 - Harden the contracts

4. **DSL and docstrings**
   - `python/adc/dsl.py` starts with an obsolete description of a CPU prototype without compiled codegen.
   - Action: align the header with the current status: `prototype`, `aot`, `production`, cache, GPU/MPI.

5. **Implicit time: local source vs global Schur**
   - `SourceImplicit` is now the right name for the local implicit source.
   - `adc.Split` / `adc.CondensedSchur` introduce the real non-local stage by Schur.
   - Action: keep this separation in all examples: do not present
     `SourceImplicit` as "total implicit", and do not hide Schur behind a local source.

6. **`AmrSystem`**
   - Honest limits: runtime still mono-block, even if the compile-time engine
     `AmrSystemCoupler` can already represent several co-located blocks.
   - Action: evolve `AmrSystem` toward an explicit multi-block registry, not just
     remove the `throw` on the second block.
   - Framing: conservative multi-block AMR must share one hierarchy and co-located cells
     for all evolved blocks; the flexibility is model/scheme/time, not the
     local spatial absence of a species on certain patches.
   - Open prerequisite: #140 to fix the `stride` cadence of `AmrSystemCoupler`.

### P2 - Stabilize Schur and elliptic

7. **Elliptic**
   - `GeometricMG` still does solver + storage + coefficients for scalar Poisson.
   - Schur now has the C++ bricks: full tensor operator, Krylov BiCGStab, builder,
     condensed stage and Python binding.
   - Action: formalize the common interfaces after the fact:
     - `EllipticOperator`: apply/residual/smooth/coefficients.
     - `LinearSolver`: MG, FFT, Krylov.
     - `FieldPostProcess`: phi -> aux.
   - Backend: #135 lifts the known device bug (`Geometry` / `Box2D` + CFL MPI). Continue to require
     a GH200 harness to declare a Schur/polar path device-clean.

8. **Variable roles**
   - `VariableRole` is a good base.
   - Action: enforce roles on DSL models used by coupled sources/Schur, raise an error if
     `Density/MomentumX/MomentumY` are missing.

### P3 - User API cleanup

9. **DSL paths**
   - `dynamic`, `aot`, `production` all have a meaning.
   - Action: in docs and API, recommend:
     - `production` by default.
     - `aot` for flat ABI debug.
     - `prototype/dynamic` for host CPU test only.

10. **Polar geometry**
    - `PolarMesh`, polar transport and `PolarPoissonSolver` exist.
    - `System.step` with `mesh=PolarMesh` and the complete runtime coupling remain to be wired.
    - Action: keep the explicit error as long as Phase 2b is not done. Do not promise
      paper reproduction before a complete annular run.

## 6. What seems useless or too specific

| Element | Diagnostic | Recommendation |
|---|---|---|
| ~~`amr_coupler.hpp`~~ | REMOVED (C.2, June 2026). | Deleted -- no client #include; replaced by AmrCouplerMP. |
| `amr_multilevel.hpp`, `amr_reflux.hpp` | Simple historical AMR engines. | Classify legacy/test or merge documentation with `amr_reflux_mf`. |
| `SpectralCoupler` | Mono-model variant useful but secondary. | Keep if tests, otherwise document as advanced C++ API. |
| `DynamicModel` | Host CPU prototype. | Keep, but do not put in the main path. |
| `AdvectionDiffusion` | TEST/VALIDATION brick: used in `tests/test_weno5_ssprk3.cpp` (WENO5/SSPRK3 reference model) but NOT used by `adc_cases` (verified 2026-06-06). | Keep as example and core test brick; marked in the doc-comment. |
| `LangmuirMode` | TEST/VALIDATION brick (not used by adc_cases as of 2026-06-06): `adc_cases/two_fluid_ap/` reimplements `TwoFluidAP2D` without including this file. | Keep as IMEX analytic reference; marked in the doc-comment. |
| `TwoFluidLinear` | TEST/VALIDATION brick (not used by adc_cases as of 2026-06-06): same situation as `LangmuirMode` -- `two_fluid_ap/` is standalone. | Keep as two-species analytic reference; marked in the doc-comment. |

## 7. What is well abstracted

- `CompositeModel<Hyperbolic, Source, Elliptic>`: good grain, not too purist.
- `VariableRole`: indispensable for coupled sources, Schur, DSL.
- `block_builder.hpp`: good seam to share `add_block` and DSL production.
- `GridContext`: keeps the compiled closures decoupled from the runtime.
- `FiniteVolume` on the Python side with `riemann`, not `flux`: good separation physical/numerical flux.
- `AmrSystem` documents its limits instead of pretending.
- FFT MPI guard: refusing cleanly is better than a segfault.
- `CoupledSourceKernel`: good direction for an arbitrary coupled source without Python callback.
- `Split` / `CondensedSchur`: good separation between splitting policy and condensed source
  stage.
- `PolarPoissonSolver`: good direct choice for the polar ring; avoids forcing MG into a
  geometry where it stagnated.

## 8. MPI + Kokkos validation and optimization plan

### 8.1 Vocabulary to freeze

In the repo, "MPI" and "Kokkos" are two different axes:

| Short name | Build | What it validates |
|---|---|---|
| MPI CPU | `ADC_USE_MPI=ON`, device Kokkos Serial/OpenMP | rank/process decomposition, halos, reductions, ranks without data. |
| Kokkos CPU Serial | `ADC_USE_KOKKOS=ON` with Serial device | only on-node backend in mono-thread; reference and CI guard. |
| Kokkos CPU OpenMP | `ADC_USE_KOKKOS=ON` with OpenMP device | local multi-thread CPU parallelism via Kokkos. |
| Kokkos GPU | `ADC_USE_KOKKOS=ON` with Cuda/HIP device | cell kernels on GPU, often `np=1`. |
| MPI + Kokkos CPU | `ADC_USE_MPI=ON`, `ADC_USE_KOKKOS=ON`, device Serial/OpenMP | MPI between ranks + local Kokkos CPU execution. |
| MPI + Kokkos GPU | `ADC_USE_MPI=ON`, `ADC_USE_KOKKOS=ON`, device Cuda/HIP | distributed production target: MPI between ranks/nodes, Kokkos on local GPU. |

Language rule: do not write "MPI validated" if the proof is only CPU. Write
explicitly `MPI CPU`, `Kokkos Cuda np=1`, or `MPI + Kokkos Cuda np=1/2/4`.

### 8.2 Current status to maintain

The base design is good: `parallel/comm.hpp` isolates MPI, `mesh/for_each.hpp` isolates Kokkos, and
`MultiFab` / `fill_boundary` make the junction between the two. The natural target is therefore:

```text
rang MPI 0 -> kernels Kokkos sur CPU/GPU local
rang MPI 1 -> kernels Kokkos sur CPU/GPU local
...
```

What seems already in place:

- optional MPI axis (`ADC_USE_MPI`) and required on-node Kokkos axis (`ADC_USE_KOKKOS` ON by
  default; configuring with OFF is a fatal CMake error);
- `for_each_cell` / Kokkos reductions for the local loops;
- `comm.hpp` for ranks, barriers, all-reduce;
- MPI halos in `fill_boundary`, with buffers in unified memory under Kokkos;
- GPU harness under `python/tests/gpu/`, including MPI + Kokkos Cuda cases;
- regular CI covering at least Kokkos Serial (PR gate) and, in full mode, MPI + Kokkos Serial
  and Kokkos OpenMP.

What remains to make systematic:

- a coverage table by subsystem, not just by PR;
- CTest labels by backend (`mpi`, `kokkos-serial`, `kokkos-openmp`,
  `kokkos-cuda`, `mpi-kokkos-cuda`);
- an explicit Kokkos OpenMP CPU validation, because its execution space is not strictly
  identical to Kokkos Serial (per-tile reduction reassociation);
- a regular device-MPI validation on ROMEO/GH200, at least nightly or standardized manual.

### 8.3 Validation matrix by layer

| Layer | MPI CPU | Kokkos CPU | Kokkos GPU | MPI + Kokkos GPU | Expected quality |
|---|---|---|---|---|---|
| `comm.hpp` | required | n/a | n/a | required | collectives called by all ranks, no deadlock. |
| `for_each.hpp` | indirect | required | required | required | no fragile cross-TU device lambda; reductions tolerated not bit-exact. |
| `Fab2D` / `MultiFab` | required | required | required | required | `local_size()==0` safe; no `fab(0)` outside guard. |
| `fill_boundary` | required | required | required | critical | fence before MPI; valid buffers for MPI CUDA-aware. |
| `spatial_operator` | useful | required | required | useful | same schemes; FP differences documented. |
| `GeometricMG` / elliptic | required | required | required | critical | convergence under all backends; clean refusal of incompatible solvers. |
| `System` runtime | required | required | required | critical | `step`, `step_cfl`, Poisson, sources, DSL production. |
| `AmrSystem` / `advance_amr` | required | required | required | critical | reflux, average_down, regrid, fine/coarse halos. |
| Schur / Krylov | required | required | to validate after #135 | to validate after #135 | collective dot/norm, preconditioner without unexpected affine effect. |
| Polar | useful | required | to validate after #135 | later | polar MMS, polar Poisson, no paper claim before complete runtime. |

### 8.4 Concrete plan

1. **Tests/backend inventory**
   - Add or verify CTest labels.
   - Produce a `docs/BACKEND_COVERAGE.md` table:
     `test -> Kokkos Serial/MPI CPU/Kokkos OpenMP/Kokkos Cuda/MPI+Cuda`.
   - Mark the tests that self-skip under Kokkos or MPI, to avoid false greens.

2. **Kokkos CPU OpenMP**
   - Add a Kokkos OpenMP build configuration.
   - Run the base tests: `for_each`, reductions, `MultiFab`, `fill_boundary`, `System`,
     `GeometricMG`, `advance_amr`.
   - Compare to the Kokkos Serial results with numerical tolerance, not necessarily bit-identical.

3. **MPI + Kokkos CPU**
   - Combined build `ADC_USE_MPI=ON` + `ADC_USE_KOKKOS=ON` with Serial then OpenMP device.
   - Verify that the collectives stay called by all ranks even when a rank has no
     local fab.
   - Priority tests: `fill_boundary`, `solve_fields`, `mf_arith::dot`, AMR reflux, Schur.

4. **Kokkos GPU np=1**
   - Standardize the GH200 harness: one script per group (`system`, `amr`, `elliptic`,
     `schur`, `polar`).
   - Require `compute-sanitizer` on the small cases.
   - Any new kernel must have a minimal Cuda test or be explicitly noted "non device".
   - Keep a regression test for the #135 fix: `Geometry` / `Box2D` device-callable and
     collective CFL reductions in MPI.

5. **MPI + Kokkos GPU np=2/4**
   - First validate the communication bricks: halos, reductions, `parallel_copy`,
     `refinement`.
   - Then the complete paths: `System` DSL production + `geometric_mg`, multi-patch AMR,
     Schur when it is device-clean.
   - Measure and document the MPI CUDA-aware limits: UVM, GPUDirect, fences, cost of the
     all-reduce and the exchanges.

6. **Optimization**
   - Replace the host-side paths that read large arrays with Kokkos kernels or
     reductions.
   - Avoid repeated global fences; group the fences before MPI or host read.
   - Reduce the temporary allocations in the AMR loops (`FluxRegister`, face `MultiFab`,
     regrid buffers).
   - Measure separately kernel time, MPI halos, Poisson, regrid, reflux.
   - For GPU, watch: occupancy, kernel size, stride accesses, UVM page faults,
     implicit synchronizations.

### 8.5 Quality criteria

A "validated" path must specify:

- exact backend (`MPI CPU`, `Kokkos OpenMP`, `Kokkos Cuda`, `MPI+Kokkos Cuda`);
- number of ranks (`np=1/2/4`);
- elliptic solver (`geometric_mg`, `fft`, `polar_poisson`, Schur);
- expected tolerance (`bit-identical`, `dmax < tol`, or MMS convergence);
- behavior of ranks without local data;
- compatibility or explicit refusal if the path is not supported.

Minimum quality to merge a backend feature:

1. reference Kokkos Serial test;
2. MPI CPU test if the code touches `MultiFab`, halos, reductions or AMR;
3. Cuda test or explicit justification if the code is called from a device path;
4. no per-cell Python callback in the hot path.

## 9. Documentation audit

The documentation has become a second codebase. It contains good sources of truth, but
it also keeps historical layers. The main risk is the same as in the code: several
documents say almost the same thing, with different dates and statuses.

### 9.1 Map of the documents

| Document | Current role | Status | Recommended action |
|---|---|---|---|
| `README.md` | Public GitHub page, pitch, short API, build, validation. | Useful but too long. It mixes introduction, architecture, DSL, limits, GPU validation, roadmap. Several statuses are already fragile. | Keep short: promise, installation, minimal example, links to specialized docs. Move the long tables to docs. |
| `todo.md` | Living session journal. | Very useful to follow the work items, but already stale: `master = #139`, #135 still "in flight" while merged on `origin/master`, references `docs/ROADMAP.md` while the living file is archived. | Treat it as a living backlog, not a public source of truth. Update it after each merge wave or rename it `DEVELOPMENT_STATUS.md`. |
| `docs/ARCHITECTURE.md` | Main source for the layers and boundaries. | Good conceptual base. Some counters/tests/backend statuses may diverge. | Keep as the architecture source of truth; add a short and dated "current runtime status" box. |
| `docs/ALGORITHMS.md` | Numerical catalog: formulas, code, validations. | Good format. Must stay stable and less "session". | Add Schur/polar only as algorithms, not as roadmap. Keep the cited tests up to date. |
| `docs/DSL_MODEL_DESIGN.md` | DSL spec + history. | Very rich but too long; mixes stable API, historical notes and statuses. Still mentions `AmrSystem.potential()` as in progress while the binding exists. | Split: short `DSL_API.md` for the user, `DSL_MODEL_DESIGN.md` for design, history in archive. |
| `docs/PAPER_ROADMAP.md` | Hoffart scientific status / paper reproduction. | Useful and prudent, but some infra statuses are stale (`AmrSystem.potential()`, explicit AMR) and must follow the merges. | Keep as a scientific doc. Do not put the details of all the infra PRs there; link to `BACKEND_COVERAGE.md`. |
| `docs/AMR_MULTIBLOCK_DESIGN.md` | Multi-block AMR capstone design. | Good source for the conservative invariant: common hierarchy, co-located blocks. | Keep. Update after #140/#141; do not duplicate in README. |
| `docs/COUPLER_HIERARCHY.md` | Classification of the couplers. | Very useful to clarify public/internal/deprecated. | Reference it from `ARCHITECTURE.md` and `CODEBASE_AUDIT.md`. |
| `docs/SCHUR_CONDENSATION_DESIGN.md` | Schur design. | Good design document, but still starts as a "target" while several PRs are delivered. | Add a status box at the top: delivered / remaining / GPU validation. |
| `docs/GPU_RUNTIME_PORT.md` | GPU/GH200 validation journal. | Indispensable for device proofs, but very historical. | Add a synthesis at the top by backend and keep the detail as a journal. |
| `docs/GPU_ROMEO.md` | Manual ROMEO recipe. | Useful. | Keep operational, verify commands/module load after each campaign. |
| `docs/PERFORMANCE.md` | M1/OpenMP/FFT/AMR measurements. | Old measurements tied to application drivers. Must not be read as current global perf. | Rename or stronger preamble: "historical measurements". Redo a Kokkos/MPI/GPU campaign before drawing conclusions. |
| `docs/CHOICES.md` | Design decisions. | Short and useful. | Keep stable; point structural decisions there, not PR statuses. |
| `docs/BIBLIOGRAPHY.md` | External references. | Stable. | Keep. Add the Schur/polar references if necessary. |
| `docs/CODE_DOCUMENTATION_CONVENTION.md` | Comment convention. | Good local source, aligned Doxygen / Google / C++ Core Guidelines / PEP 257. | Apply folder by folder; do not do a massive comment patch. |
| `docs/sphinx/*.md` | Published documentation. | Shorter, but risk of diverging from the README and the current API. | Make Sphinx the short user doc, generated/reviewed after each release. |
| `docs/archive/*` | History and old roadmaps. | Necessary to not lose the context, but must stay out of the main navigation. | Never cite an archive path as a current source of truth without saying so explicitly. |

### 9.2 Confirmed documentation inconsistencies

These points do not require an architectural debate; they are divergences between text and code or
between documents:

1. **#135 is merged on `origin/master`, but `todo.md` still lists it "in flight".**
   - Fix the current status and the Schur/polar device sections.
   - Keep #140 as the only structural open PR in this block; #141 is already merged.

2. **`AmrSystem.potential()` is documented "IN PROGRESS" in several documents, but the binding exists.**
   - Code read: `python/bindings/core/bindings.cpp` exposes `.def("potential", ...)` for `AmrSystem` and
     `python/bindings/amr/amr_system.cpp` implements `AmrSystem::potential()`.
   - Documents to fix: `README.md`, `docs/DSL_MODEL_DESIGN.md`, `docs/PAPER_ROADMAP.md`,
     probably `todo.md`.

3. **The AMR status "explicit only / no IMEX" is still present in public docs.**
   - The AMR engine now has the local source IMEX (#132).
   - The correct sentence is: `AmrSystem` runtime stays mono-block; the local implicit source exists;
     global Schur and multi-block AMR remain separate work items.

4. **`todo.md` references `docs/ROADMAP.md`, which no longer exists as an active doc.**
   - The file is `docs/archive/ROADMAP.md`.
   - Either fix the link, or assume that `todo.md` is no longer derived from this roadmap.

5. **`README.md` sometimes presents `DistributedFFTSolver` as a general MPI capability.**
   - The sentence must always specify: `DistributedFFTSolver` exists and is tested separately, but
     is not routed in `System` because of the band layout.

6. **The test figures and CI claims are scattered.**
   - `README.md`, `docs/sphinx/installation.md`, `docs/ARCHITECTURE.md` and `todo.md` do not always
     talk about the same CI after #136.
   - Action: create a single source `docs/BACKEND_COVERAGE.md` and link from the other docs.

7. **`docs/DSL_MODEL_DESIGN.md` deliberately keeps historical sections, but the reader can
   confuse target and current status.**
   - The document has a good `0bis`, but the size makes the reading risky.
   - Action: extract a stable doc `docs/DSL_API.md` or strengthen the Sphinx API doc.

8. **`docs/PERFORMANCE.md` is not up to date with the current Kokkos/MPI/GPU stack.**
   - The preamble says these are application measurements, but the name `PERFORMANCE.md` suggests
     a current performance status.
   - Action: add a campaign date, exact backend, and a "historical" status.

### 9.3 Proposed sources of truth

To avoid the divergences:

| Subject | Proposed source of truth | Other docs |
|---|---|---|
| Python user API | `docs/sphinx/api.md` + a future `docs/DSL_API.md` | README only shows a minimal example. |
| Layer architecture | `docs/ARCHITECTURE.md` | README points to this document. |
| Numerical algorithms | `docs/ALGORITHMS.md` | README only lists the families. |
| Schur | `docs/SCHUR_CONDENSATION_DESIGN.md` + tests | PAPER_ROADMAP only keeps the scientific impact. |
| Multi-block AMR | `docs/AMR_MULTIBLOCK_DESIGN.md` | README must not describe the capstone. |
| Backends / CI / GPU | future `docs/BACKEND_COVERAGE.md` + `docs/GPU_RUNTIME_PORT.md` | README only gives the summary. |
| Paper reproduction | `docs/PAPER_ROADMAP.md` | todo must not be the scientific source. |
| Work in progress | `todo.md` | Never cited as user proof without date/commit. |

### 9.4 Documentation cleanup plan

1. **Short truth patch**
   - Fix `todo.md` for #135 merged.
   - Fix `AmrSystem.potential()` in `README.md`, `DSL_MODEL_DESIGN.md`,
     `PAPER_ROADMAP.md`.
   - Fix the "AMR not IMEX" sentences to account for #132.

2. **Create `docs/BACKEND_COVERAGE.md`**
   - Table by test/group: Kokkos Serial, MPI CPU, Kokkos OpenMP, Kokkos Cuda,
     MPI + Kokkos Cuda.
   - Note which tests are CI, which tests are manual ROMEO, which tests self-skip.

3. **Shorten `README.md`**
   - Keep: promise, build, minimal Python example, very short matrix, links.
   - Move: long DSL/AMR/GPU limits to `DSL_MODEL_DESIGN.md`, `PAPER_ROADMAP.md`,
     `BACKEND_COVERAGE.md`.

4. **Split the DSL doc**
   - `DSL_API.md`: how to write a model, compile, wire `System`/`AmrSystem`.
   - `DSL_MODEL_DESIGN.md`: decisions and technical history.

5. **Bring Sphinx to the same level as the Markdown doc**
   - Fix the `Spatial` / `FiniteVolume` examples according to the recommended API.
   - Generate or re-read Sphinx after each user API change.

6. **Explicitly archive the historical documents**
   - `docs/archive/README.md` must say: "non-normative documents".
   - The active docs must not point to `archive/*` except for history.

## 10. Proposed cleanup plan

### Lot A - Documentation and API truth

1. Fix the header docstring of `python/adc/dsl.py`.
2. Add in `ARCHITECTURE.md` a "current API / internal / legacy" table.
3. Add a note: `SourceImplicit` = local implicit source, `CondensedSchur` = global stage.
4. Neutralize the application mentions in the comments of generic headers
   (`diocotron` -> `anneau polaire` / `derive ExB` when that is the real meaning).
5. Progressively apply `CODE_DOCUMENTATION_CONVENTION.md` folder by folder: file header,
   class header, MPI/GPU/conservation invariants, without automatic churn.

### Lot B - `System` runtime

1. Extract `NativeLoader` from `python/bindings/system/base/system.cpp`.
2. Extract `SystemFieldSolver` for `ensure_elliptic`, `solve_fields`, `apply_eps/kappa/Bz/Te`.
3. Extract `SystemBlockStore` for blocks, ghosts, get/set state.
4. Keep `System::Impl` as a thin orchestrator.

### Lot C - AMR

1. Split `amr_reflux_mf.hpp`.
2. `amr_coupler.hpp` REMOVED (C.2, June 2026) -- no client #include, deleted, replaced by AmrCouplerMP.
3. Rely on the strict layout guard #141 before any multi-block API.
4. Fix the AMR `stride` cadence per #140 before exposing slow steps from Python.
5. Decide explicitly: `AmrSystem` stays mono-block or becomes multi-block runtime.
6. If multi-block runtime: enforce a common AMR hierarchy, regrid by union of tags
   (`electrons OR ions OR neutres OR phi OR user`), prolongation/restriction and reflux block by
   block, coupled sources on co-located cells. Do not introduce separate hierarchies nor
   blocks absent from certain patches as long as the conservation is not formally handled.

### Lot D - Schur and elliptic

1. Extract a common `EllipticOperator` interface between scalar Poisson, full tensor and
   Schur operator.
2. Keep `TensorKrylovSolver` as a solver, not as owner of the physical problem.
3. Keep `CondensedSchurSourceStepper` as a time/coupling stage, not as
   `model.source(U,aux)`.
4. Validate Schur on `Kokkos Cuda` with a dedicated harness now that #135 is merged.

### Lot E - Validation

1. Stride non-regression tests: hold-then-catch-up AMR (#140), `step_cfl` with stride,
   Poisson with held block.
2. Required roles tests for DSL coupled sources.
3. Explicit error tests for `PolarMesh` as long as the complete polar runtime is absent.
4. Backend tests with precise names: `MPI CPU`, `Kokkos Serial`, `Kokkos Cuda`,
   `MPI + Kokkos Cuda`.
5. Auto-discovery CI coverage tests: verify that the #136 optimization does not mask an
   unregistered test.

## 11. Verdict

`adc_cpp` now has a real level of abstraction. The problem is not that the core would be badly
designed; the problem is that it grew very fast and keeps several generations of valid paths.
The merges #118-#142 added important bricks: Schur usable from Python,
DSL coupled sources, runtime primitives, substeps-aware cadence, polar transport/Poisson, and
many API guards. These additions make the project more powerful, but they also increase the
pressure on `python/bindings/system/base/system.cpp`, `amr_reflux_mf.hpp` and the backend matrix.

The code is maintainable if we now do two things:

1. **Classify the surfaces**: public / internal / legacy / test.
2. **Extract the god classes**: especially `python/bindings/system/base/system.cpp` and `amr_reflux_mf.hpp`.
3. **Name the backend proofs**: never replace `MPI CPU`, `Kokkos Cuda` or
   `MPI + Kokkos Cuda` by a simple "MPI".

The principle to preserve is simple: a class must either describe a local physics, or apply
a numerical operator, or store data, or orchestrate. When it starts to do two of
these things at once, it must be split.
