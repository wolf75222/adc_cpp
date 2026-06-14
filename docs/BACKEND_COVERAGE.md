# BACKEND_COVERAGE -- backend coverage matrix

> **SINGLE SOURCE OF TRUTH** for the backend coverage of the test suite.
> README and Sphinx must POINT HERE, not duplicate this table.
> Updated manually after each test or CI job addition.
> Last revision: 2026-06-07 (Lot E.4: matrix <-> disk resynchronization,
> +13 C++ tests and +10 Python tests that were missing from the table).

---

## Legend

| Symbole | Signification |
|---------|---------------|
| **ci-fast** | Runs in the REQUIRED gate **build-and-test** (ubuntu-latest, g++, Release, Kokkos Serial: `-DADC_USE_KOKKOS=ON`, Kokkos 4.4.01 `Kokkos_ENABLE_SERIAL=ON`, C++ + Python module). Trigger: any ordinary `pull_request`. |
| **ci-full** | Runs in full mode (push `master`, nightly cron, `workflow_dispatch`, or PR labeled `ci-full`). Adds the **MPI** (`-DADC_USE_MPI=ON` + Kokkos Serial) and **Kokkos-OpenMP** (`-DADC_USE_KOKKOS=ON`, Kokkos 4.4.01 `Kokkos_ENABLE_OPENMP=ON`, multi-thread CPU) jobs. |
| **ROMEO** | Validated manually on GH200 (`armgpu` node, Kokkos 4.4.01, `Kokkos_ARCH_HOPPER90`, `nvcc_wrapper`, CUDA-aware OpenMPI). Harness cited in parentheses. Evidence in `docs/GPU_ROMEO.md` and/or `docs/GPU_RUNTIME_PORT.md`. |
| **self-skip** | The test detects the absence of the backend and returns without error (exit 0). Note: in the **MPI CPU** column, a non-MPI test (sections 1a-1g) marked "self-skip" actually means "runs at np=1 in the MPI build (linked against MPI, mono-process)" -- it IS compiled and launched in the `mpi` job, outside the CMake `if(ADC_HAS_MPI)` block. This is NOT a real skip: the binary executes, it simply ignores the optional MPI calls. |
| **?** | Unknown / not exercised -- see the Gaps section. |

Columns:

- **Serial**: `build-and-test` gate, `-DADC_USE_KOKKOS=ON` with `Kokkos_ENABLE_SERIAL=ON`, mono-thread CPU, without MPI (g++). This is the Kokkos Serial path of the required gate (ci-fast).
- **MPI CPU**: build `-DADC_USE_MPI=ON -DADC_USE_KOKKOS=ON` (Kokkos Serial), CPU only (MPI job).
- **Kokkos Serial**: same Kokkos Serial backend as the Serial column. The `build-and-test` gate runs in ALL modes (fast as well as full), so Kokkos Serial is also covered every time ci-full executes.
- **Kokkos OpenMP**: build `-DADC_USE_KOKKOS=ON` with `Kokkos_ENABLE_OPENMP=ON`, CPU.
- **Kokkos Cuda (GH200)**: Kokkos build + `Kokkos_ARCH_HOPPER90`, one GPU per rank.
- **MPI + Kokkos Cuda**: same build + CUDA-aware OpenMPI, `srun -n {1,2,4} --gpus-per-task=1`.

> **Important note**: CI NEVER builds with `-DADC_USE_KOKKOS=ON -DKokkos_ENABLE_CUDA=ON`.
> All "Kokkos Cuda" and "MPI + Kokkos Cuda" cells are therefore either ROMEO or "?".
> Kokkos OpenMP is now enabled in CI via the **ci-full** job (job added #155,
> `Kokkos_ENABLE_OPENMP=ON`, 91/91 ctest, 0 failure / 0 skipped).

---

## 1. C++ tests (ctest -- source in `tests/CMakeLists.txt`)

### 1a. Mesh / containers group (adc_add_test)

| Test | Serial | MPI CPU | Kokkos Serial | Kokkos OpenMP | Kokkos Cuda | MPI+Kokkos Cuda |
|------|--------|---------|---------------|---------------|-------------|-----------------|
| test_box2d | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_fab2d | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_box_array | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_multifab | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_sync_residence | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_reduce | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_fill_boundary | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_physical_bc | ci-fast | self-skip | ci-full | ci-full | ROMEO (phase2_transport.cpp -- indirect, via non-periodic transport) | ? |
| test_geometry | ci-fast | self-skip | ci-full | ci-full | ? | ? |

### 1b. AMR primitives group

| Test | Serial | MPI CPU | Kokkos Serial | Kokkos OpenMP | Kokkos Cuda | MPI+Kokkos Cuda |
|------|--------|---------|---------------|---------------|-------------|-----------------|
| test_refinement | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_hierarchy | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_cluster | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_regrid | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_flux_register | ci-fast | self-skip | ci-full | ci-full | ROMEO (romeo_amr_build.sh -- phase 5, AMR device ops) | ? |
| test_coverage_mask | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_patch_range | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_cf_interface | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_diagnostics | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_load_balance | ci-fast | self-skip | ci-full | ci-full | ? | ? |

### 1c. Elliptic group

| Test | Serial | MPI CPU | Kokkos Serial | Kokkos OpenMP | Kokkos Cuda | MPI+Kokkos Cuda |
|------|--------|---------|---------------|---------------|-------------|-----------------|
| test_poisson_smoother | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_geometric_mg | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_poisson_convergence | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_poisson_fft | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_elliptic_operator | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_elliptic_problem | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_elliptic_interface (static_assert concepts) | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_elliptic_composite_rhs | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_poisson_disc | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_solve_robust | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_cut_cell | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_cut_cell_anisotropic | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_variable_epsilon | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_screened_poisson | ci-fast | self-skip | ci-full | ci-full | ROMEO (gpu_epm_validate.cpp via romeo_gpuval2_build.sh, dmax=0) | ? |
| test_anisotropic_epsilon | ci-fast | self-skip | ci-full | ci-full | ROMEO (gpu_epm_validate.cpp via romeo_gpuval2_build.sh, dmax=0) | ? |
| test_cut_cell_anisotropic_multibox | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_full_tensor_operator | ci-fast | self-skip | ci-full | ci-full | ROMEO pass (#150) | ? |
| test_polar_poisson_mms | ci-fast | self-skip | ci-full | ci-full | ROMEO pass (#150) | ? |
| test_krylov_solver (serie) | ci-fast | self-skip | ci-full | ci-full | ROMEO pass (#152) | ? |
| test_schur_condensation (serie) | ci-fast | self-skip | ci-full | ci-full | ROMEO pass (#158) | ? |
| test_condensed_schur_source_stepper (serie) | ci-fast | self-skip | ci-full | ci-full | ROMEO pass | ? |

### 1d. Reconstruction / generic integrations group

| Test | Serial | MPI CPU | Kokkos Serial | Kokkos OpenMP | Kokkos Cuda | MPI+Kokkos Cuda |
|------|--------|---------|---------------|---------------|-------------|-----------------|
| test_spatial_discretisation | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_primitive_recon | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_system_abstraction | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_system_coupler | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_two_species_minimal | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_coupled_source | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_system_two_explicit | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_assembler_driver | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_system_coupler | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_source_covered_cells | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_composite_source_conservation | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_stride_cadence | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_layout_guard | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_system_hardening | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_imex_partial | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_imex_transport | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_multirate_stride | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_cfl_dt | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_adaptive_multirate | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_user_time_integrator | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_diffusion | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_diffusion | ci-fast | self-skip | ci-full | ci-full | ROMEO (romeo_amr_build.sh -- phase 5, AMR device ops) | ? |
| test_amr_spatial_parity | ci-fast | self-skip | ci-full | ci-full | ? | ? |

### 1e. Extensible aux channel group (B_z, T_e)

| Test | Serial | MPI CPU | Kokkos Serial | Kokkos OpenMP | Kokkos Cuda | MPI+Kokkos Cuda |
|------|--------|---------|---------------|---------------|-------------|-----------------|
| test_aux_extra | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_aux_coupler_bz | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_aux_system_bz | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_aux_composite | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_aux_bz | ci-fast | self-skip | ci-full | ci-full | ROMEO (gpu_amr_bz_validate.cpp via romeo_gpuval2_build.sh, dmax=0) | ? |
| test_amr_system_bz_pop | ci-fast | self-skip | ci-full | ci-full | ROMEO (gpu_amr_bz_validate.cpp via romeo_gpuval2_build.sh -- 2-level AMR device, dmax=0) | ? |
| test_amr_system_bz_multibox (serie np=1) | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_system_bz_multibox (MPI np=2/4) | self-skip | ci-full | self-skip | ? | ? | ROMEO (gpu_amr_bz_mpi_validate.cpp via romeo_gpuval2_mpi_build.sh, bz_bad=0, dcmax=0) |
| test_aux_single_source | ci-fast | self-skip | ci-full | ci-full | ? | ? |

### 1f. WENO, splitting, IMEX, Roe, polar, DSL group

| Test | Serial | MPI CPU | Kokkos Serial | Kokkos OpenMP | Kokkos Cuda | MPI+Kokkos Cuda |
|------|--------|---------|---------------|---------------|-------------|-----------------|
| test_weno_convergence | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_splitting | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_imex_ap | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_ap_limit | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_lorentz_eliminator | ci-fast | self-skip | ci-full | ci-full | ROMEO pass | ? |
| test_polar_ring_advection | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_polar_transport_mms | ci-fast | self-skip | ci-full | ci-full | ROMEO pass | ? |
| test_polar_system_step | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_dynamic_model | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_block_builder | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_weno5_ssprk3 | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_variable_role | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_roe_flux | ci-fast | self-skip | ci-full | ci-full | ? | ? |

### 1g. System / AmrSystem runtime group (add_executable, system.cpp linkage)

| Test | Serial | MPI CPU | Kokkos Serial | Kokkos OpenMP | Kokkos Cuda | MPI+Kokkos Cuda |
|------|--------|---------|---------------|---------------|-------------|-----------------|
| test_compiled_model_parity | ci-fast | self-skip | ci-full | ci-full | ? (nvcc limit on cross-TU extended lambdas, documented GPU_RUNTIME_PORT.md phase 8) | ? |
| test_weno5_compiled_model | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_compiled_model | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_potential | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_native_loader | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_weno5_native | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_riemann_native | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_imex_native | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_system_contract | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_aux_runtime_bz | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_aux_te | ci-fast | self-skip | ci-full | ci-full | ROMEO (gpu_aux_validate.cpp via romeo_gpuval2_build.sh -- assemble_rhs path, dmax=0; NOTE: the add_compiled_model+T_e path remains outside the device scope, see phase 8) | ? |

### 1g-bis. Multi-block AMR group (runtime capstone, amr_system.cpp linkage)

`add_executable` tests linked to the `python/amr_system.cpp` runtime (AmrSystem facade -> AmrRuntime engine),
NAMED model / tag functors (no generic lambda) hence in principle nvcc-compatible. Same CPU
backends as section 1g; not yet exercised on device (Cuda columns = ?).

| Test | Serial | MPI CPU | Kokkos Serial | Kokkos OpenMP | Kokkos Cuda | MPI+Kokkos Cuda |
|------|--------|---------|---------------|---------------|-------------|-----------------|
| test_amr_system_twoblock | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_multiblock_compiled | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_multiblock_substeps | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_multiblock_coupled_source | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_coupled_source_role_strict | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_multiblock_imex | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_multiblock_regrid_union | ci-fast | self-skip | ci-full | ci-full | ? | ? |

### 1h. MPI core group (compiled only if ADC_HAS_MPI)

| Test | Serial | MPI CPU | Kokkos Serial | Kokkos OpenMP | Kokkos Cuda | MPI+Kokkos Cuda |
|------|--------|---------|---------------|---------------|-------------|-----------------|
| test_krylov_solver_np1/2/4 | self-skip | ci-full | self-skip | ? | ? | ROMEO pass np2/4 (#157) |
| test_schur_condensation_np1/2/4 | self-skip | ci-full | self-skip | ? | ? | ROMEO pass np2/4 (#157) |
| test_condensed_schur_source_stepper_np1/2/4 | self-skip | ci-full | self-skip | ? | ? | ROMEO pass np2/4 (#157) |
| test_mpi_smoke (np=4) | self-skip | ci-full | self-skip | ? | ? | ? |
| test_mpi_array_reduce (np=4) | self-skip | ci-full | self-skip | ? | ? | ? |
| test_mpi_fillboundary (np=4) | self-skip | ci-full | self-skip | ? | ? | ROMEO (mpi6_fillboundary.cpp via mpi6_romeo_build.sh, gfails=0 np=1/2/4) |
| test_mpi_poisson (np=4) | self-skip | ci-full | self-skip | ? | ? | ROMEO pass np2/4 (#157) |
| test_mpi_redistribute (np=4) | self-skip | ci-full | self-skip | ? | ? | ? |
| test_mpi_coupler_inject (np=4) | self-skip | ci-full | self-skip | ? | ? | ? |
| test_mpi_fft_distributed (np=4) | self-skip | ci-full | self-skip | ? | ? | ? |
| test_mpi_mbox_parity_np1/2/4 | self-skip | ci-full | self-skip | ? | ? | ROMEO (harness amrmpi_integrated.cpp + amrmpi_romeo_build.sh -- identical to test_mpi_mbox_parity, dmax=0 np=1/2/4) -- ROMEO pass np2/4 (#157) |
| test_mpi_hybrid_mbox_parity_np1/2/4 | self-skip | ci-full | self-skip | ? | ? | ? |
| test_mpi_cutcell_multibox_np1/2/4 | self-skip | ci-full | self-skip | ? | ? | ROMEO pass np2/4 (#157) |
| test_amr_system_bz_multibox_np2/4 | self-skip | ci-full | self-skip | ? | ? | ROMEO (gpu_amr_bz_mpi_validate.cpp via romeo_gpuval2_mpi_build.sh, bz_bad=0, dcmax=0) |
| test_mpi_amr_compiled_parity_np1/2/4 | self-skip | ci-full | self-skip | ? | ? | ROMEO (amrmpi_integrated.cpp via amrmpi_romeo_build.sh, dmax=0, masse=0, crossrank_spread=0) -- ROMEO pass np2/4 (#157) |
| test_mpi_amr_twoblock_parity_np1/2/4 | self-skip | ci-full | self-skip | ? | ? | ? |
| test_mpi_amr_distributed_coarse_np1/2/4 | self-skip | ci-full | self-skip | ? | ? | ROMEO (amrmpi_integrated.cpp via amrmpi_romeo_build.sh -- measured distributed mode, correct but does not scale) -- ROMEO pass np2/4 (#157) |
| test_mpi_system_solve_fields_np1/2/4 | self-skip | ci-full | self-skip | ? | ? | ROMEO pass np2/4 (#157) |
| test_mpi_system_fft_np1/2/4 | self-skip | ci-full | self-skip | ? | ? | ? |
| test_mpi_coupled_source_np1/2/4 | self-skip | ci-full | self-skip | ? | ? | ROMEO pass np2/4 (#157) |

---

## 2. Python tests (`python/tests/test_*.py`)

All exercise the `_adc` module (pybind11), built **with Kokkos** (the Python module is linked against the
Kokkos backend like everything that links `adc`; Kokkos is required). The COMPLETE suite runs in the
`build-and-test` gate, where the module is compiled in **Kokkos Serial**
(`-DADC_BUILD_PYTHON=ON -DADC_USE_KOKKOS=ON`, without MPI); ci-fast and ci-full are identical for
the Python suite (not in `mpi`). In ci-full, the `kokkos-openmp` job recompiles the module in **Kokkos
OpenMP** but only replays a targeted subset there (ABI std guard: `test_native_abi_std`,
`test_dsl_production`, `test_dsl_production_amr`).

| Test Python | Serial | MPI CPU | Kokkos Serial | Kokkos OpenMP | Kokkos Cuda | MPI+Kokkos Cuda |
|-------------|--------|---------|---------------|---------------|-------------|-----------------|
| test_amr_multiblock.py | ci-fast | ? | ? | ? | ? | ? |
| test_amr_production_stride_reject.py | ci-fast | ? | ? | ? | ? | ? |
| test_bindings.py | ci-fast | ? | ? | ? | ? | ? |
| test_block_names_dynamic.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_abi_metadata.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_aot.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_aot_bz.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_aot_te.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_aux_naux.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_block.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_brick.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_codegen.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_compile_cache.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_compile_facade.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_compose.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_coupled.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_coupled_role_error.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_coupled_source.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_coupled_source_conservation.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_cse.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_dynamic.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_elliptic.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_hybrid.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_hybrid_amr.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_hybrid_bz.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_hybrid_coupling.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_jit.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_jit_bz.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_jit_te.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_jitlib.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_phase_a.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_production.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_production_amr.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_recon.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_roles.py | ci-fast | ? | ? | ? | ? | ? |
| test_dsl_source.py | ci-fast | ? | ? | ? | ? | ? |
| test_implicit_vars.py | ci-fast | ? | ? | ? | ? | ? |
| test_magnetic_field.py | ci-fast | ? | ? | ? | ? | ? |
| test_poisson_composite.py | ci-fast | ? | ? | ? | ? | ? |
| test_poisson_eps.py | ci-fast | ? | ? | ? | ? | ? |
| test_poisson_eps_aniso.py | ci-fast | ? | ? | ? | ? | ? |
| test_poisson_screened.py | ci-fast | ? | ? | ? | ? | ? |
| test_polar_conservation_radial_flux.py | ci-fast | ? | ? | ? | ? | ? |
| test_polar_diocotron.py | ci-fast | ? | ? | ? | ? | ? |
| test_polar_rejections.py | ci-fast | ? | ? | ? | ? | ? |
| test_polar_system.py | ci-fast | ? | ? | ? | ? | ? |
| test_polar_teardown_stability.py | ci-fast | ? | ? | ? | ? | ? |
| test_primitive_state.py | ci-fast | ? | ? | ? | ? | ? |
| test_schur_split.py | ci-fast | ? | ? | ? | ? | ? |
| test_schur_via_system.py | ci-fast | ? | ? | ? | ? | ? |
| test_stride.py | ci-fast | ? | ? | ? | ? | ? |
| test_time_policy.py | ci-fast | ? | ? | ? | ? | ? |
| test_weno5_compiledmodel.py | ci-fast | ? | ? | ? | ? | ? |
| test_weno5_ssprk3.py | ci-fast | ? | ? | ? | ? | ? |

---

## 3. Manual GH200 harnesses (`python/tests/gpu/`)

These harnesses are NOT in the ctest graph; they are built and launched manually via SBATCH on
ROMEO. The table below indexes them to ease cross-referencing with section 1.

| Harness | Validated backend | Evidence |
|---------|---------------|----------|
| `romeo_run.sh` + raw CUDA harness (`gen_cuda_harness.py -> euler_gpu.cu`) | Kokkos Cuda (flux `EulerGen` vs `adc::Euler`, maxdiff=0) | `docs/GPU_ROMEO.md` "Recipe" section |
| `romeo_kokkos_build.sh` + `gen_kokkos_harness.py -> kokkos_euler.cpp` | Kokkos Cuda (`parallel_for`, exec=Cuda, diff=5.55e-17) | `docs/GPU_ROMEO.md` "Kokkos" section |
| `romeo_kokkos_sim_build.sh` + `gen_kokkos_sim.py -> kokkos_euler_sim.cpp` | Kokkos Cuda (80 Euler 2D steps, exact mass, maxdiff=8.9e-16) | `docs/GPU_ROMEO.md` "Complete case" section |
| `romeo_phase1_build.sh` + `phase1_transport.cpp` | Kokkos Cuda (full Euler transport, BIT-IDENTICAL to CPU) | `docs/GPU_RUNTIME_PORT.md` phase 1 |
| (phase2, same script) + `phase2_transport.cpp` | Kokkos Cuda (non-periodic BCs, BIT-IDENTICAL to CPU) | `docs/GPU_RUNTIME_PORT.md` phase 2 |
| (phase2) + `phase3_poisson.cpp` | Kokkos Cuda (Poisson Dirichlet n=128, BIT-IDENTICAL to CPU) | `docs/GPU_RUNTIME_PORT.md` phase 3 |
| (phase2) + `phase4_coupling.cpp` | Kokkos Cuda (3-species ionization couplings, BIT-IDENTICAL to CPU) | `docs/GPU_RUNTIME_PORT.md` phase 4 |
| `romeo_amr_build.sh` + `amr_CMakeLists.txt` (test_flux_register + test_amr_diffusion) | Kokkos Cuda (AMR flux_register + diffusion ops, PASS) | `docs/GPU_RUNTIME_PORT.md` phase 5 |
| `mpi6_romeo_build.sh` + `mpi6_fillboundary.cpp` | MPI + Kokkos Cuda (fill_boundary np=1/2/4, gfails=0) | `docs/GPU_RUNTIME_PORT.md` phase 6 |
| `romeo_phase7_build.sh` + `phase7_system.cpp` | Kokkos Cuda (full euler_poisson System, BIT-IDENTICAL to CPU) | `docs/GPU_RUNTIME_PORT.md` phase 7 |
| `romeo_gpuval2_build.sh` + `gpu_epm_validate.cpp` | Kokkos Cuda (EPM screened + aniso, dmax=0) | `docs/GPU_RUNTIME_PORT.md` "round 2" |
| `romeo_gpuval2_build.sh` + `gpu_aux_validate.cpp` | Kokkos Cuda (T_e via load_aux<5>, dmax=0) | `docs/GPU_RUNTIME_PORT.md` "round 2" |
| `romeo_gpuval2_build.sh` + `gpu_amr_bz_validate.cpp` | Kokkos Cuda (B_z per AMR level, dmax=0) | `docs/GPU_RUNTIME_PORT.md` "round 2" |
| `romeo_gpuval2_build.sh` + `gpu_amrsys_facade_validate.cpp` | Kokkos Cuda (entire AmrSystemCoupler facade, dmax=0) | `docs/GPU_RUNTIME_PORT.md` "round 2" |
| `romeo_gpuval2_build.sh` (optional MPI build) + `gpu_dsl_production_validate.cpp` | Kokkos Cuda + optional MPI (native add_compiled_model path, dmax < 1e-13 tolerated) | `python/tests/gpu/gpuval2_CMakeLists.txt` + source code |
| `romeo_gpuval2_mpi_build.sh` + `gpu_amr_bz_mpi_validate.cpp` | MPI + Kokkos Cuda (B_z AMR multi-box np=1/2/4, bz_bad=0, dcmax=0; additive sums not bit-exact across np -- FMA reduction order) | `docs/GPU_RUNTIME_PORT.md` "round 2" + script |
| `amrmpi_romeo_build.sh` + `amrmpi_integrated.cpp` | MPI + Kokkos Cuda (AmrSystem + MPI + GPU, dmax=0, masse=0 np=1/2/4; distributed mode correct, does not scale) | `docs/GPU_RUNTIME_PORT.md` phase 10 + 11 |

---

---

## 3b. MPI + Kokkos OpenMP (ROMEO x64cpu)

Validated on the ROMEO x64cpu node (`Kokkos_ENABLE_OPENMP=ON`, OpenMPI, cmake + g++ build).

- 52/57 runs rank-invariant (bit-identical np=1/2/4 on the parity/AMR/Krylov observables, dmax=0).
- 3 heavy distributed-MG tests (mpi_cutcell_multibox, mpi_amr_distributed_coarse,
  condensed_schur_source_stepper) too slow at np>1 (exceed 600s) -- a Kokkos-OpenMP PERFORMANCE
  pathology on small tiles + MPI halos (~5-7x slowdown at np>1), NOT a deadlock nor
  a correctness bug. All pass at np=1.


## 4. Quantified summary

Counting base (as of 2026-06-07, regenerable via `docs/gen_test_counts.py`): 109 ctest C++
targets outside the MPI block (91 `adc_add_test` + 18 `add_executable` runtime, including the 7 multi-block
AMR capstones of section 1g-bis), + 11 `add_executable` in the `ADC_HAS_MPI` block (each
replays np=1/2/4), + 60 Python tests.

| Status | Number of cells (approx.) |
|--------|------------------------------|
| **ci-fast** | ~169 (109 C++ outside-MPI x Kokkos Serial [gate] + 60 Python x Kokkos Serial [gate]) |
| **ci-full** | ~239 (109 C++ x Kokkos Serial + 109 x Kokkos OpenMP; ~21 MPI CPU entries) |
| **ROMEO** | ~55 (mono and multi-GPU GPU harnesses covering ~15 functional groups) |
| **self-skip** | ~350 (Kokkos-Serial tests on MPI columns, and MPI-only on columns without MPI) |
| **?** | Kokkos Cuda of the runtime tests not yet exercised on device (including section 1g-bis multi-block AMR) + MPI+Kokkos Cuda of the majority of MPI tests + the entire Python block outside the Kokkos Serial gate |

---

## 5. Notable gaps (priority gaps)

> **Closed gaps:**
> - Kokkos OpenMP CI: CLOSED (#155, ci-full job, 91/91 ctest).
> - MPI + Kokkos Cuda multi-GPU: CLOSED for the 10 Krylov/Schur/MPI-core tests (#157, rank-invariant dmax=0).
> - MPI + Kokkos OpenMP ROMEO: VALIDATED (52/57 rank-invariant; 3 heavy tests too slow at np>1, perf, not deadlock).
> - Doc <-> disk sync (Lot E.4): CLOSED. The audit confirmed that NO CPU backend-path was
>   without a test: every C++ test outside-MPI runs in the 4 CPU backends (Serial, Kokkos Serial, Kokkos
>   OpenMP, MPI np=1) and every MPI test in the MPI job. The only gap was DOCUMENTARY: 13 C++ tests
>   and 10 Python tests delivered by sister workstreams (multi-block AMR, polar, elliptic_interface,
>   DSL, schur_via_system) were missing from the table. They are now indexed (sections 1c/1d/1f/1g-bis,
>   1h for MPI, section 2). No renaming: the names were already explicit about their backend.

1. **Kokkos OpenMP (all suites)** -- CLOSED (#155): ci-full job `Kokkos_ENABLE_OPENMP=ON`, 91/91 ctest.
   Columns filled `ci-full` for all non-MPI tests (sections 1a-1g, including 1g-bis multi-block
   AMR). The MPI-only tests (section 1h, Serial=self-skip) remain `self-skip` in Kokkos OpenMP
   (MPI build not joined).

2. **MPI + Kokkos Cuda -- majority of MPI tests** -- CLOSED for 10 tests (#157, GH200,
   rank-invariant np=1/2/4, dmax=0): krylov_solver, condensed_schur_source_stepper, mpi_poisson,
   mpi_system_solve_fields, mpi_amr_compiled_parity, mpi_amr_distributed_coarse, mpi_coupled_source,
   mpi_mbox_parity, mpi_cutcell_multibox, test_schur_condensation. Still "?":
   `test_mpi_smoke`, `test_mpi_array_reduce`, `test_mpi_redistribute`, `test_mpi_coupler_inject`,
   `test_mpi_fft_distributed`, `test_mpi_system_fft`, `test_mpi_hybrid_mbox_parity`,
   `test_amr_system_bz_multibox` (MPI+Cuda).

3. **Python tests under Kokkos OpenMP (full suite) / MPI / Cuda** -- the `_adc` module is built
   with Kokkos (required): the COMPLETE suite runs under Kokkos Serial in the
   `build-and-test` gate. Under Kokkos OpenMP, only an ABI subset (`test_native_abi_std`,
   `test_dsl_production`, `test_dsl_production_amr`) is replayed (`kokkos-openmp` job). No Python
   test yet covers Kokkos OpenMP in full, nor Kokkos Cuda, nor MPI.

4. **`add_compiled_model` path on Kokkos Cuda** -- `test_compiled_model_parity` and its variants
   (`test_weno5_compiled_model`, `test_amr_compiled_model`, ...) are not validated on device. The
   nvcc limit (cross-TU `__host__ __device__` extended lambdas) is documented in
   `docs/GPU_RUNTIME_PORT.md` phase 8; the workaround (named functors) exists but has not yet
   been ported to the `test_compiled_model_parity` path itself.

5. **T_e via `add_compiled_model` on Kokkos Cuda** -- only the `assemble_rhs` path (named
   functors) is device-validated. The T_e marshaling of the `System::add_compiled_model` +
   `apply_te` path remains covered only in Serial CI.

6. **Multi-block AMR capstone (section 1g-bis) on Kokkos Cuda** -- the 7 multi-block AMR runtime
   tests (`test_amr_system_twoblock`, `test_amr_multiblock_compiled/_substeps/_coupled_source/
   _imex/_regrid_union`, `test_amr_coupled_source_role_strict`) are validated on the 4 CPU backends
   (Serial, Kokkos Serial, Kokkos OpenMP, MPI np=1). Their model and tag functors are NAMED
   (no generic lambda), hence in principle nvcc-compatible, but they do not yet have a dedicated
   ROMEO harness (Kokkos Cuda columns = "?"). Device leg of the multi-block capstone, to be ported when
   the `add_compiled_model` device path (gap #4) is closed. The MPI CPU parity is covered by
   `test_mpi_amr_twoblock_parity_np1/2/4` (section 1h).
