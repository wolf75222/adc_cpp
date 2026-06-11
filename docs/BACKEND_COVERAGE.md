# BACKEND_COVERAGE -- matrice de couverture des backends

> **SOURCE DE VERITE UNIQUE** pour la couverture backend de la suite de tests.
> README et Sphinx doivent POINTER ICI, pas dupliquer ce tableau.
> Mis a jour manuellement apres chaque ajout de test ou de job CI.
> Derniere revision : 2026-06-07 (Lot E.4 : resynchronisation matrice <-> disque,
> +13 tests C++ et +10 tests Python qui manquaient au tableau).

---

## Legende

| Symbole | Signification |
|---------|---------------|
| **ci-fast** | Tourne dans le gate OBLIGATOIRE **build-and-test** (ubuntu-latest, g++, Release, Kokkos Serial : `-DADC_USE_KOKKOS=ON`, Kokkos 4.4.01 `Kokkos_ENABLE_SERIAL=ON`, C++ + module Python). Declenchement : tout `pull_request` ordinaire. |
| **ci-full** | Tourne en mode plein (push `master`, nightly cron, `workflow_dispatch`, ou PR labellisee `ci-full`). Ajoute les jobs **MPI** (`-DADC_USE_MPI=ON` + Kokkos Serial) et **Kokkos-OpenMP** (`-DADC_USE_KOKKOS=ON`, Kokkos 4.4.01 `Kokkos_ENABLE_OPENMP=ON`, CPU multi-thread). |
| **ROMEO** | Valide manuellement sur GH200 (noeud `armgpu`, Kokkos 4.4.01, `Kokkos_ARCH_HOPPER90`, `nvcc_wrapper`, OpenMPI CUDA-aware). Harness cite entre parentheses. Evidence dans `docs/GPU_ROMEO.md` et/ou `docs/GPU_RUNTIME_PORT.md`. |
| **self-skip** | Le test detecte l'absence du backend et retourne sans erreur (exit 0). Note : dans la colonne **MPI CPU**, un test non-MPI (sections 1a-1g) marque "self-skip" signifie en realite "tourne a np=1 dans le build MPI (lie MPI, mono-process)" -- il EST compile et lance dans le job `mpi`, hors du bloc `if(ADC_HAS_MPI)` du CMake. Ce n'est PAS un vrai skip : le binaire s'execute, il ignore simplement les appels MPI facultatifs. |
| **?** | Inconnu / pas exerce -- voir section Gaps. |

Colonnes :

- **Serial** : gate `build-and-test`, `-DADC_USE_KOKKOS=ON` avec `Kokkos_ENABLE_SERIAL=ON`, CPU mono-thread, sans MPI (g++). C'est le chemin Kokkos Serial du gate obligatoire (ci-fast).
- **MPI CPU** : build `-DADC_USE_MPI=ON -DADC_USE_KOKKOS=ON` (Kokkos Serial), CPU uniquement (MPI job).
- **Kokkos Serial** : meme backend Kokkos Serial que la colonne Serial. Le gate `build-and-test` tourne dans TOUS les modes (fast comme full), donc Kokkos Serial est aussi couvert chaque fois que ci-full s'execute.
- **Kokkos OpenMP** : build `-DADC_USE_KOKKOS=ON` avec `Kokkos_ENABLE_OPENMP=ON`, CPU.
- **Kokkos Cuda (GH200)** : build Kokkos + `Kokkos_ARCH_HOPPER90`, un GPU par rang.
- **MPI + Kokkos Cuda** : meme build + OpenMPI CUDA-aware, `srun -n {1,2,4} --gpus-per-task=1`.

> **Note importante** : la CI ne construit JAMAIS avec `-DADC_USE_KOKKOS=ON -DKokkos_ENABLE_CUDA=ON`.
> Toutes les cellules "Kokkos Cuda" et "MPI + Kokkos Cuda" sont donc soit ROMEO, soit "?".
> Kokkos OpenMP est desormais active en CI via le job **ci-full** (job ajoute #155,
> `Kokkos_ENABLE_OPENMP=ON`, 91/91 ctest, 0 echec / 0 skipped).

---

## 1. Tests C++ (ctest -- source dans `tests/CMakeLists.txt`)

### 1a. Groupe maillage / conteneurs (adc_add_test)

| Test | Serial | MPI CPU | Kokkos Serial | Kokkos OpenMP | Kokkos Cuda | MPI+Kokkos Cuda |
|------|--------|---------|---------------|---------------|-------------|-----------------|
| test_box2d | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_fab2d | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_box_array | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_multifab | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_sync_residence | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_reduce | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_fill_boundary | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_physical_bc | ci-fast | self-skip | ci-full | ci-full | ROMEO (phase2_transport.cpp -- indirect, via transport non-periodique) | ? |
| test_geometry | ci-fast | self-skip | ci-full | ci-full | ? | ? |

### 1b. Groupe AMR primitives

| Test | Serial | MPI CPU | Kokkos Serial | Kokkos OpenMP | Kokkos Cuda | MPI+Kokkos Cuda |
|------|--------|---------|---------------|---------------|-------------|-----------------|
| test_refinement | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_hierarchy | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_cluster | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_regrid | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_flux_register | ci-fast | self-skip | ci-full | ci-full | ROMEO (romeo_amr_build.sh -- phase 5, ops AMR device) | ? |
| test_coverage_mask | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_patch_range | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_cf_interface | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_diagnostics | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_load_balance | ci-fast | self-skip | ci-full | ci-full | ? | ? |

### 1c. Groupe elliptique

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

### 1d. Groupe reconstruction / integrations generiques

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
| test_amr_diffusion | ci-fast | self-skip | ci-full | ci-full | ROMEO (romeo_amr_build.sh -- phase 5, ops AMR device) | ? |
| test_amr_spatial_parity | ci-fast | self-skip | ci-full | ci-full | ? | ? |

### 1e. Groupe canal aux extensible (B_z, T_e)

| Test | Serial | MPI CPU | Kokkos Serial | Kokkos OpenMP | Kokkos Cuda | MPI+Kokkos Cuda |
|------|--------|---------|---------------|---------------|-------------|-----------------|
| test_aux_extra | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_aux_coupler_bz | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_aux_system_bz | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_aux_composite | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_aux_bz | ci-fast | self-skip | ci-full | ci-full | ROMEO (gpu_amr_bz_validate.cpp via romeo_gpuval2_build.sh, dmax=0) | ? |
| test_amr_system_bz_pop | ci-fast | self-skip | ci-full | ci-full | ROMEO (gpu_amr_bz_validate.cpp via romeo_gpuval2_build.sh -- AMR 2 niveaux device, dmax=0) | ? |
| test_amr_system_bz_multibox (serie np=1) | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_system_bz_multibox (MPI np=2/4) | self-skip | ci-full | self-skip | ? | ? | ROMEO (gpu_amr_bz_mpi_validate.cpp via romeo_gpuval2_mpi_build.sh, bz_bad=0, dcmax=0) |
| test_aux_single_source | ci-fast | self-skip | ci-full | ci-full | ? | ? |

### 1f. Groupe WENO, splitting, IMEX, Roe, polaire, DSL

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

### 1g. Groupe runtime System / AmrSystem (add_executable, liaison system.cpp)

| Test | Serial | MPI CPU | Kokkos Serial | Kokkos OpenMP | Kokkos Cuda | MPI+Kokkos Cuda |
|------|--------|---------|---------------|---------------|-------------|-----------------|
| test_compiled_model_parity | ci-fast | self-skip | ci-full | ci-full | ? (limite nvcc lambdas etendues cross-TU, documente GPU_RUNTIME_PORT.md phase 8) | ? |
| test_weno5_compiled_model | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_compiled_model | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_potential | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_native_loader | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_weno5_native | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_riemann_native | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_imex_native | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_system_contract | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_aux_runtime_bz | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_aux_te | ci-fast | self-skip | ci-full | ci-full | ROMEO (gpu_aux_validate.cpp via romeo_gpuval2_build.sh -- chemin assemble_rhs, dmax=0 ; NOTE: le chemin add_compiled_model+T_e reste hors perimetre device, voir phase 8) | ? |

### 1g-bis. Groupe AMR multi-blocs (capstone runtime, liaison amr_system.cpp)

Tests `add_executable` lies au runtime `python/amr_system.cpp` (facade AmrSystem -> moteur AmrRuntime),
foncteurs de modele / de tag NOMMES (pas de lambda generique) donc nvcc-compatibles en principe. Memes
backends CPU que la section 1g ; non encore exerces sur device (colonnes Cuda = ?).

| Test | Serial | MPI CPU | Kokkos Serial | Kokkos OpenMP | Kokkos Cuda | MPI+Kokkos Cuda |
|------|--------|---------|---------------|---------------|-------------|-----------------|
| test_amr_system_twoblock | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_multiblock_compiled | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_multiblock_substeps | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_multiblock_coupled_source | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_coupled_source_role_strict | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_multiblock_imex | ci-fast | self-skip | ci-full | ci-full | ? | ? |
| test_amr_multiblock_regrid_union | ci-fast | self-skip | ci-full | ci-full | ? | ? |

### 1h. Groupe MPI coeur (compile seulement si ADC_HAS_MPI)

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
| test_mpi_mbox_parity_np1/2/4 | self-skip | ci-full | self-skip | ? | ? | ROMEO (harness amrmpi_integrated.cpp + amrmpi_romeo_build.sh -- identique a test_mpi_mbox_parity, dmax=0 np=1/2/4) -- ROMEO pass np2/4 (#157) |
| test_mpi_hybrid_mbox_parity_np1/2/4 | self-skip | ci-full | self-skip | ? | ? | ? |
| test_mpi_cutcell_multibox_np1/2/4 | self-skip | ci-full | self-skip | ? | ? | ROMEO pass np2/4 (#157) |
| test_amr_system_bz_multibox_np2/4 | self-skip | ci-full | self-skip | ? | ? | ROMEO (gpu_amr_bz_mpi_validate.cpp via romeo_gpuval2_mpi_build.sh, bz_bad=0, dcmax=0) |
| test_mpi_amr_compiled_parity_np1/2/4 | self-skip | ci-full | self-skip | ? | ? | ROMEO (amrmpi_integrated.cpp via amrmpi_romeo_build.sh, dmax=0, masse=0, crossrank_spread=0) -- ROMEO pass np2/4 (#157) |
| test_mpi_amr_twoblock_parity_np1/2/4 | self-skip | ci-full | self-skip | ? | ? | ? |
| test_mpi_amr_distributed_coarse_np1/2/4 | self-skip | ci-full | self-skip | ? | ? | ROMEO (amrmpi_integrated.cpp via amrmpi_romeo_build.sh -- mode reparti mesure, correct mais ne scale pas) -- ROMEO pass np2/4 (#157) |
| test_mpi_system_solve_fields_np1/2/4 | self-skip | ci-full | self-skip | ? | ? | ROMEO pass np2/4 (#157) |
| test_mpi_system_fft_np1/2/4 | self-skip | ci-full | self-skip | ? | ? | ? |
| test_mpi_coupled_source_np1/2/4 | self-skip | ci-full | self-skip | ? | ? | ROMEO pass np2/4 (#157) |

---

## 2. Tests Python (`python/tests/test_*.py`)

Tous exercent le module `_adc` (pybind11), construit **avec Kokkos** (le module Python est lie au
backend Kokkos comme tout ce qui lie `adc` ; Kokkos est obligatoire). La suite COMPLETE tourne dans
le gate `build-and-test`, ou le module est compile en **Kokkos Serial**
(`-DADC_BUILD_PYTHON=ON -DADC_USE_KOKKOS=ON`, sans MPI) ; ci-fast et ci-full y sont identiques pour
la suite Python (pas dans `mpi`). En ci-full, le job `kokkos-openmp` recompile le module en **Kokkos
OpenMP** mais n'y rejoue qu'un sous-ensemble cible (garde-fou ABI std : `test_native_abi_std`,
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

## 3. Harnesses GH200 manuels (`python/tests/gpu/`)

Ces harnesses ne sont PAS dans le graphe ctest ; ils sont builds et lances manuellement par SBATCH sur
ROMEO. Le tableau ci-dessous les indexe pour faciliter le croisement avec la section 1.

| Harness | Backend valide | Evidence |
|---------|---------------|----------|
| `romeo_run.sh` + harness CUDA brut (`gen_cuda_harness.py -> euler_gpu.cu`) | Kokkos Cuda (flux `EulerGen` vs `adc::Euler`, maxdiff=0) | `docs/GPU_ROMEO.md` section "Recette" |
| `romeo_kokkos_build.sh` + `gen_kokkos_harness.py -> kokkos_euler.cpp` | Kokkos Cuda (`parallel_for`, exec=Cuda, diff=5.55e-17) | `docs/GPU_ROMEO.md` section "Kokkos" |
| `romeo_kokkos_sim_build.sh` + `gen_kokkos_sim.py -> kokkos_euler_sim.cpp` | Kokkos Cuda (80 pas Euler 2D, masse exacte, maxdiff=8.9e-16) | `docs/GPU_ROMEO.md` section "Cas COMPLET" |
| `romeo_phase1_build.sh` + `phase1_transport.cpp` | Kokkos Cuda (transport Euler complet, BIT-IDENTIQUE CPU) | `docs/GPU_RUNTIME_PORT.md` phase 1 |
| (phase2, meme script) + `phase2_transport.cpp` | Kokkos Cuda (BCs non-periodiques, BIT-IDENTIQUE CPU) | `docs/GPU_RUNTIME_PORT.md` phase 2 |
| (phase2) + `phase3_poisson.cpp` | Kokkos Cuda (Poisson Dirichlet n=128, BIT-IDENTIQUE CPU) | `docs/GPU_RUNTIME_PORT.md` phase 3 |
| (phase2) + `phase4_coupling.cpp` | Kokkos Cuda (couplages ionisation 3 especes, BIT-IDENTIQUE CPU) | `docs/GPU_RUNTIME_PORT.md` phase 4 |
| `romeo_amr_build.sh` + `amr_CMakeLists.txt` (test_flux_register + test_amr_diffusion) | Kokkos Cuda (ops AMR flux_register + diffusion, PASS) | `docs/GPU_RUNTIME_PORT.md` phase 5 |
| `mpi6_romeo_build.sh` + `mpi6_fillboundary.cpp` | MPI + Kokkos Cuda (fill_boundary np=1/2/4, gfails=0) | `docs/GPU_RUNTIME_PORT.md` phase 6 |
| `romeo_phase7_build.sh` + `phase7_system.cpp` | Kokkos Cuda (System euler_poisson complet, BIT-IDENTIQUE CPU) | `docs/GPU_RUNTIME_PORT.md` phase 7 |
| `romeo_gpuval2_build.sh` + `gpu_epm_validate.cpp` | Kokkos Cuda (EPM screened + aniso, dmax=0) | `docs/GPU_RUNTIME_PORT.md` "round 2" |
| `romeo_gpuval2_build.sh` + `gpu_aux_validate.cpp` | Kokkos Cuda (T_e via load_aux<5>, dmax=0) | `docs/GPU_RUNTIME_PORT.md` "round 2" |
| `romeo_gpuval2_build.sh` + `gpu_amr_bz_validate.cpp` | Kokkos Cuda (B_z par niveau AMR, dmax=0) | `docs/GPU_RUNTIME_PORT.md` "round 2" |
| `romeo_gpuval2_build.sh` + `gpu_amrsys_facade_validate.cpp` | Kokkos Cuda (facade AmrSystemCoupler entiere, dmax=0) | `docs/GPU_RUNTIME_PORT.md` "round 2" |
| `romeo_gpuval2_build.sh` (build MPI optionnel) + `gpu_dsl_production_validate.cpp` | Kokkos Cuda + MPI optionnel (chemin natif add_compiled_model, dmax < 1e-13 tolere) | `python/tests/gpu/gpuval2_CMakeLists.txt` + code source |
| `romeo_gpuval2_mpi_build.sh` + `gpu_amr_bz_mpi_validate.cpp` | MPI + Kokkos Cuda (B_z AMR multi-box np=1/2/4, bz_bad=0, dcmax=0 ; sommes additives non bit-exactes entre np -- ordre reduction FMA) | `docs/GPU_RUNTIME_PORT.md` "round 2" + script |
| `amrmpi_romeo_build.sh` + `amrmpi_integrated.cpp` | MPI + Kokkos Cuda (AmrSystem + MPI + GPU, dmax=0, masse=0 np=1/2/4 ; mode reparti correct, ne scale pas) | `docs/GPU_RUNTIME_PORT.md` phase 10 + 11 |

---

---

## 3b. MPI + Kokkos OpenMP (ROMEO x64cpu)

Valide sur le noeud x64cpu de ROMEO (`Kokkos_ENABLE_OPENMP=ON`, OpenMPI, build cmake + g++).

- 52/57 runs rank-invariant (bit-identiques np=1/2/4 sur les observables parity/AMR/Krylov, dmax=0).
- 3 tests distribues-MG lourds (mpi_cutcell_multibox, mpi_amr_distributed_coarse,
  condensed_schur_source_stepper) trop lents a np>1 (depassent 600s) -- pathologie PERFORMANCE
  Kokkos-OpenMP sur petites tuiles + halos MPI (~5-7x ralentissement a np>1), PAS un deadlock ni
  un bug de correction. Tous passent a np=1.


## 4. Bilan chiffre

Base de comptage (au 2026-06-07, régénérable via `docs/gen_test_counts.py`) : 109 cibles ctest
C++ hors bloc MPI (91 `adc_add_test` + 18 `add_executable` runtime, dont les 7 capstones AMR
multi-blocs de la section 1g-bis), + 11 `add_executable` dans le bloc `ADC_HAS_MPI` (chacun
rejoue np=1/2/4), + 60 tests Python.

| Statut | Nombre de cellules (approx.) |
|--------|------------------------------|
| **ci-fast** | ~169 (109 C++ hors-MPI x Kokkos Serial [gate] + 60 Python x Kokkos Serial [gate]) |
| **ci-full** | ~239 (109 C++ x Kokkos Serial + 109 x Kokkos OpenMP ; ~21 entrees MPI CPU) |
| **ROMEO** | ~55 (harnesses GPU mono et multi-GPU couvrant ~15 groupes fonctionnels) |
| **self-skip** | ~350 (tests Kokkos-Serial sur colonnes MPI, et MPI-only sur colonnes sans MPI) |
| **?** | Kokkos Cuda des tests runtime non encore exerces sur device (dont la section 1g-bis AMR multi-blocs) + MPI+Kokkos Cuda de la majorite des tests MPI + tout le bloc Python hors gate Kokkos Serial |

---

## 5. Lacunes notables (gaps prioritaires)

> **Gaps fermes :**
> - Kokkos OpenMP CI : FERME (#155, job ci-full, 91/91 ctest).
> - MPI + Kokkos Cuda multi-GPU : FERME pour les 10 tests Krylov/Schur/MPI-noyau (#157, rank-invariant dmax=0).
> - MPI + Kokkos OpenMP ROMEO : VALIDE (52/57 rank-invariants ; 3 tests lourds trop lents a np>1, perf, non deadlock).
> - Synchro doc <-> disque (Lot E.4) : FERME. L'audit a confirme qu'AUCUN backend-path CPU n'etait
>   sans test : tout test C++ hors-MPI tourne dans les 4 backends CPU (Serial, Kokkos Serial, Kokkos
>   OpenMP, MPI np=1) et tout test MPI dans le job MPI. Le seul ecart etait DOCUMENTAIRE : 13 tests C++
>   et 10 tests Python livres par des chantiers soeurs (AMR multi-blocs, polaire, elliptic_interface,
>   DSL, schur_via_system) manquaient au tableau. Ils sont desormais indexes (sections 1c/1d/1f/1g-bis,
>   1h pour le MPI, section 2). Aucun renommage : les noms etaient deja explicites sur leur backend.

1. **Kokkos OpenMP (toutes suites)** -- FERME (#155) : job ci-full `Kokkos_ENABLE_OPENMP=ON`, 91/91 ctest.
   Colonnes renseignees `ci-full` pour tous les tests non-MPI (sections 1a-1g, y compris 1g-bis AMR
   multi-blocs). Les tests MPI-only (section 1h, Serial=self-skip) restent `self-skip` en Kokkos OpenMP
   (build MPI non joint).

2. **MPI + Kokkos Cuda -- majorite des tests MPI** -- FERME pour 10 tests (#157, GH200,
   rank-invariant np=1/2/4, dmax=0) : krylov_solver, condensed_schur_source_stepper, mpi_poisson,
   mpi_system_solve_fields, mpi_amr_compiled_parity, mpi_amr_distributed_coarse, mpi_coupled_source,
   mpi_mbox_parity, mpi_cutcell_multibox, test_schur_condensation. Restent "?" :
   `test_mpi_smoke`, `test_mpi_array_reduce`, `test_mpi_redistribute`, `test_mpi_coupler_inject`,
   `test_mpi_fft_distributed`, `test_mpi_system_fft`, `test_mpi_hybrid_mbox_parity`,
   `test_amr_system_bz_multibox` (MPI+Cuda).

3. **Tests Python sous Kokkos OpenMP (suite complete) / MPI / Cuda** -- le module `_adc` est construit
   avec Kokkos (obligatoire) : la suite COMPLETE tourne sous Kokkos Serial dans le gate
   `build-and-test`. Sous Kokkos OpenMP, seul un sous-ensemble ABI (`test_native_abi_std`,
   `test_dsl_production`, `test_dsl_production_amr`) est rejoue (job `kokkos-openmp`). Aucun test
   Python ne couvre encore Kokkos OpenMP en entier, ni Kokkos Cuda, ni MPI.

4. **Chemin `add_compiled_model` sur Kokkos Cuda** -- `test_compiled_model_parity` et ses variantes
   (`test_weno5_compiled_model`, `test_amr_compiled_model`, ...) ne sont pas valides sur device. La
   limite nvcc (lambdas etendues `__host__ __device__` cross-TU) est documentee dans
   `docs/GPU_RUNTIME_PORT.md` phase 8 ; le contournement (foncteurs nommes) existe mais n'a pas encore
   ete porte sur le chemin `test_compiled_model_parity` lui-meme.

5. **T_e via `add_compiled_model` sur Kokkos Cuda** -- seul le chemin `assemble_rhs` (fonceurs
   nommes) est valide device. Le marshaling T_e du chemin `System::add_compiled_model` +
   `apply_te` reste couvert uniquement en CI Serial.

6. **Capstone AMR multi-blocs (section 1g-bis) sur Kokkos Cuda** -- les 7 tests runtime AMR
   multi-blocs (`test_amr_system_twoblock`, `test_amr_multiblock_compiled/_substeps/_coupled_source/
   _imex/_regrid_union`, `test_amr_coupled_source_role_strict`) sont valides sur les 4 backends CPU
   (Serial, Kokkos Serial, Kokkos OpenMP, MPI np=1). Leurs foncteurs de modele et de tag sont NOMMES
   (pas de lambda generique), donc en principe nvcc-compatibles, mais ils n'ont pas encore de harness
   ROMEO dedie (colonnes Kokkos Cuda = "?"). Pendant device du capstone multi-blocs, a porter quand
   le chemin `add_compiled_model` device (gap #4) sera ferme. La parite MPI CPU est couverte par
   `test_mpi_amr_twoblock_parity_np1/2/4` (section 1h).
