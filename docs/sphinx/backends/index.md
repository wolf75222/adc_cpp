# Parallel backends

`adc_cpp` targets a single compute stack (mesh + transport + Poisson + AMR), but
this stack can run on six parallel configurations: from single-thread serial
up to multi-GPU Grace-Hopper distributed by MPI. The key design point is that no
operator changes from one configuration to another: all parallelism is confined to
two seams (dispatch seams). The backend is chosen at compile time, through
CMake options.

This page describes each configuration: what it is, the build command, how to
run it, and its validation status (tested in CI or validated manually on
ROMEO). For the test-by-test coverage matrix, see
[BACKEND_COVERAGE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md) (single source of truth); for the
phase-by-phase GPU port, see [GPU_RUNTIME_PORT.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/GPU_RUNTIME_PORT.md).

## The model: two seams, MPI + Kokkos

There are no "three layers" stacked. The architecture is MPI + Kokkos:

- **MPI** distributes subdomains across ranks (one GPU per rank in GPU mode). Everything goes
  through `my_rank()` / `n_ranks()` / `all_reduce_*` from
  [`include/adc/parallel/comm.hpp`](https://github.com/wolf75222/adc_cpp/blob/master/include/adc/parallel/comm.hpp). Without
  `ADC_HAS_MPI`, these functions return rank 0 / 1 rank: the code is serial by
  construction.
- **Kokkos** parallelizes the local compute and abstracts the hardware through its `ExecutionSpace`:
  `Cuda` backend for NVIDIA GPUs, `Serial`/`OpenMP` for CPU. Everything goes through `for_each_cell`
  (and `for_each_cell_reduce_*`) from
  [`include/adc/mesh/for_each.hpp`](https://github.com/wolf75222/adc_cpp/blob/master/include/adc/mesh/for_each.hpp), which switches
  CPU <-> GPU at compile time without touching the call sites.

No CUDA kernel is written by hand: the same `.cpp` targets CPU and GPU depending on the Kokkos
backend active at compile time. `nvcc_wrapper` is only the compiler required by the Kokkos
Cuda backend.

> **The Python `adc` module is serial by default.** The `_adc` extension (pybind11) is
> built in CI only in Kokkos Serial (without MPI). No Python test exercises the Kokkos
> OpenMP, Cuda or MPI paths. Multi-thread, GPU and distributed are driven from the
> C++ facade (`System` / `AmrSystem`), not from Python.

## The CMake options

Verified in [`CMakeLists.txt`](../../../CMakeLists.txt):

| CMake option | Effect | Default |
|--------------|-------|--------|
| `ADC_USE_KOKKOS` | Only on-node backend, required (CPU Serial/OpenMP + GPU Cuda/HIP). Configuring with `OFF` is a fatal CMake error. | `ON` |
| `ADC_USE_MPI` | Distributed comm seam (`comm.hpp` -> MPI collectives). | `OFF` |
| `ADC_BUILD_PYTHON` | Python `adc` module (pybind11), serial only. | `OFF` |

The Kokkos sub-backend (Serial / OpenMP / Cuda) is not an `adc_cpp` option: it is
chosen at the moment Kokkos is installed (`Kokkos_ENABLE_SERIAL`, `Kokkos_ENABLE_OPENMP`,
`Kokkos_ENABLE_CUDA` + `Kokkos_ARCH_HOPPER90`), then pointed at by `-DKokkos_ROOT=...`. That is what
distinguishes configurations 1/2/5 below.

Notes:

- Kokkos is the only on-node backend and it is required: configuring without it
  (`-DADC_USE_KOKKOS=OFF`) is a fatal CMake error, and the
  [`for_each.hpp`](https://github.com/wolf75222/adc_cpp/blob/master/include/adc/mesh/for_each.hpp)
  seam does not compile without `ADC_HAS_KOKKOS` (`#error`).
- **Kokkos does not need to be pre-installed**: CMake does `find_package(Kokkos)` then, as a fallback,
  fetches + builds it via FetchContent (version `ADC_KOKKOS_FETCH_VERSION`, default 4.4.01, tarball verified by SHA256). The
  `-DKokkos_ROOT=...` commands below reuse an install (faster); without them, Kokkos is fetched.
- The C++ standard is C++20 (nvcc CUDA 12.x does not offer `-std=c++23`).

---

## 1. Kokkos Serial

**What it is.** The reference build: Kokkos Serial (single-thread CPU via
`Kokkos::parallel_for`), without MPI. `comm.hpp` answers rank 0 / 1 rank. It is the oracle: all the
other backends are validated bit-for-bit (or to rounding) against it. Serial goes through
Kokkos Serial, not through a hand-written C++ loop. This is the job that had caught an
init regression (allocating a `Fab` before the lazy Kokkos init); the Kokkos
Serial gate now plays that role on EVERY PR.

**Build.** You need a Kokkos installed with `Kokkos_ENABLE_SERIAL=ON`, then:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DADC_USE_KOKKOS=ON -DKokkos_ROOT="$KOKKOS_PREFIX"
cmake --build build -j
```

**Run.**

```bash
ctest --test-dir build --output-on-failure
```

**Validation: CI (required gate of every PR).** This is the `build-and-test` job
(`ubuntu-latest`, Kokkos Serial), triggered on every `pull_request`. Covers the 109 non-MPI
ctest targets plus the 60 Python tests (built via `-DADC_BUILD_PYTHON=ON`, `_adc`
Kokkos Serial module). Status `ci-fast` in the matrix.

---

## 2. Kokkos OpenMP

**What it is.** Same Kokkos backend, `OpenMP` execution space: multi-thread parallelism
on CPU. `for_each_cell` becomes a multi-thread `parallel_for` (with a serial-switch threshold
for the small V-cycle grids, cf. `ADC_FOREACH_SERIAL_THRESHOLD`).

**Build.** Kokkos installed with `Kokkos_ENABLE_OPENMP=ON`. The `-DADC_USE_KOKKOS=ON` on the
`adc_cpp` side is identical to config 1; it is the Kokkos install that changes:

```bash
cmake -S . -B build-kokkos-omp -DCMAKE_BUILD_TYPE=Release \
  -DADC_USE_KOKKOS=ON -DKokkos_ROOT="$KOKKOS_OPENMP_PREFIX"
cmake --build build-kokkos-omp -j
```

**Run.** Bound the number of threads on small machines:

```bash
OMP_NUM_THREADS=4 ctest --test-dir build-kokkos-omp --output-on-failure
```

**Validation: CI (`ci-full` job).** Job `kokkos-openmp`
(`ubuntu-latest / Kokkos (OpenMP)`), Kokkos 4.4.01 OpenMP, `OMP_NUM_THREADS=2`. Full mode
only (unlike the Serial gate, which runs on every PR). Status `ci-full`,
91/91 ctest, 0 failure.

> **FP note.** The sum reduction (`Kokkos::Sum`) reassociates floating-point addition per tile:
> deterministic/idempotent (same data + same Kokkos space -> same bits) but not
> bit-identical to a hand-written lexicographic sum. Since there is only one
> Kokkos path, this holds for all spaces (Serial, OpenMP, Cuda). The max reduction
> (`Kokkos::Max`) is exact everywhere. Detail in the header of
> [`for_each.hpp`](https://github.com/wolf75222/adc_cpp/blob/master/include/adc/mesh/for_each.hpp).

---

## 3. MPI + Kokkos Serial (MPI CPU)

**What it is.** Distributed build with the Kokkos Serial execution space: `comm.hpp` goes
through `MPI_Comm_rank/size` + collectives on `MPI_COMM_WORLD`. The domain is split into boxes
distributed across ranks; the halos exchange via cross-rank `fill_boundary`, the reflux/mass
via `all_reduce_*`. Single-thread CPU per rank.

**Build.** Kokkos installed with `Kokkos_ENABLE_SERIAL=ON`, plus the MPI seam:

```bash
cmake -S . -B build-mpi -DCMAKE_BUILD_TYPE=Release \
  -DADC_USE_KOKKOS=ON -DADC_USE_MPI=ON -DKokkos_ROOT="$KOKKOS_PREFIX"
cmake --build build-mpi -j
```

**Run.** The MPI targets each replay np=1/2/4 under `mpirun`. For `-np 4` on a small
machine, allow oversubscribe:

```bash
OMPI_MCA_rmaps_base_oversubscribe=true \
  ctest --test-dir build-mpi --output-on-failure
```

**Validation: CI (`ci-full` job).** Job `mpi` (`ubuntu-latest / MPI`, OpenMPI). Full mode
only. Checks invariance to the number of ranks: the observables (parity, AMR,
Krylov, mass) are bit-identical at np=1/2/4. Status `ci-full` on the ~21 entries of the
`ADC_HAS_MPI` block; the non-MPI tests run at np=1 in this build (MPI-linked, single-process).

---

## 4. MPI + Kokkos OpenMP

**What it is.** Distributed CPU hybrid: MPI between nodes/ranks, Kokkos OpenMP for the
intra-rank multi-thread. This is the "full" CPU mode (all cores of all ranks).

**Build.** Both options at once, on a Kokkos OpenMP:

```bash
cmake -S . -B build-mpi-omp -DCMAKE_BUILD_TYPE=Release \
  -DADC_USE_MPI=ON -DADC_USE_KOKKOS=ON -DKokkos_ROOT="$KOKKOS_OPENMP_PREFIX"
cmake --build build-mpi-omp -j
```

**Run.**

```bash
OMP_NUM_THREADS=4 OMPI_MCA_rmaps_base_oversubscribe=true \
  ctest --test-dir build-mpi-omp --output-on-failure
```

**Validation: ROMEO-manual (`x64cpu` node).** This combination is not in CI (CI
joins MPI with Kokkos Serial, but never MPI with Kokkos OpenMP). Validated by hand on the `x64cpu` node of
ROMEO: 52/57 rank-invariant runs (bit-identical np=1/2/4, dmax=0 on the
parity/AMR/Krylov observables). Caveat: 3 heavy distributed-MG tests (`mpi_cutcell_multibox`,
`mpi_amr_distributed_coarse`, `condensed_schur_source_stepper`) are too slow at np>1
(exceed 600 s); performance pathology (small tiles + MPI halos, ~5-7x
slowdown), not a deadlock nor a correctness bug. All pass at np=1.

---

## 5. Kokkos CUDA (ROMEO / GH200 only)

**What it is.** Kokkos backend with the `Cuda` execution space: `for_each_cell` becomes a
`Kokkos::parallel_for` that runs on the GPU. The same code as configs 1/2; only the
Kokkos backend changes. The `Fab` live in unified memory (`Kokkos::SharedSpace`), so
device-accessible by construction; `for_each_cell` is async under Cuda, hence a
`device_fence()` (via `sync_host()`) before any host read.

**No local build.** There is no `nvcc` on the dev workstations; `nvcc` runs only
on the ROMEO GPU node (aarch64), not on the login (x86). CI never builds with
CUDA: all the "Kokkos Cuda" cells of the matrix are either ROMEO-manual, or `?`.

**Build (on ROMEO, `armgpu` node).** Kokkos installed with
`Kokkos_ENABLE_CUDA=ON -DKokkos_ARCH_HOPPER90=ON`, `nvcc_wrapper` compiler:

```bash
module load cuda/12.6
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DADC_USE_KOKKOS=ON \
  -DCMAKE_CXX_COMPILER="$KOKKOS_PREFIX/bin/nvcc_wrapper" \
  -DKokkos_ROOT="$KOKKOS_PREFIX"
cmake --build build-cuda -j
```

**Run (SLURM, one GPU).**

```bash
srun --account=<compte> -p instant --constraint=armgpu --gres=gpu:1 \
  ./build-cuda/bin/<harness>
```

**Validation: ROMEO-manual (never in CI).** The GPU harnesses live in
[`python/tests/gpu/*.cpp`](https://github.com/wolf75222/adc_cpp/tree/master/python/tests/gpu) (outside the ctest graph, launched by
sbatch/`srun`). Each compiles the same logic in `exec=Cuda` and in oracle `exec=Serial`,
then compares cell by cell (`dmax = max|cuda - serial|`). Results on GH200
(Kokkos 4.4.01, `Kokkos_ARCH_HOPPER90`):

- Full single-grid solver (transport + BCs + couplings + Poisson + time step,
  orchestrated by the `System`): bit-identical to CPU (phases 1-5, 7).
- Post-#48 elliptic bricks (T_e via `load_aux<5>`, screened/Helmholtz EPM, anisotropic EPM,
  B_z per AMR level): `dmax = 0` vs Serial, same MG cycles.

**Caveats.** The `System::add_compiled_model` path (zero-copy native DSL model)
hit an `nvcc` limit (extended `__host__ __device__` lambdas cross-TU), worked around by
named functors (the `assemble_rhs` / `advance_amr` device path), but
`test_compiled_model_parity` itself is not yet ported to device. The multi-block AMR
capstone (7 tests) stays `?` on Cuda (named functors, in principle nvcc-compatible, but
without a dedicated ROMEO harness).

---

## 6. MPI + Kokkos CUDA (ROMEO multi-GPU only)

**What it is.** The target production mode: MPI distributes the subdomains (one GPU per rank),
Kokkos Cuda computes on each GPU, CUDA-aware OpenMPI exchanges the halos device-to-device
(UCX). This is config 5 + config 3 in a single run.

**No local build** (same constraints as config 5: `nvcc` ROMEO-only, never in CI).

**Build (on ROMEO).** Both options, CUDA-aware OpenMPI:

```bash
module load cuda/12.6
cmake -S . -B build-mpicuda -DCMAKE_BUILD_TYPE=Release \
  -DADC_USE_KOKKOS=ON -DADC_USE_MPI=ON \
  -DCMAKE_CXX_COMPILER="$KOKKOS_PREFIX/bin/nvcc_wrapper" \
  -DKokkos_ROOT="$KOKKOS_PREFIX"
cmake --build build-mpicuda -j
```

**Run (SLURM, several GPUs, one per rank).**

```bash
srun -n 4 --gpus-per-task=1 --constraint=armgpu ./build-mpicuda/bin/<harness>
```

**Validation: ROMEO-manual multi-GPU (never in CI).** On a 4x GH200 node (OpenMPI 4.1.7
CUDA-aware), validated np=1/2/4 (np=1 = single-GPU oracle). Achieved:

- 10 tests of the elliptic / Schur(stepper) / Poisson / system-solve / AMR stack
  rank-invariant: cross-np `dmax` = 0 (krylov_solver, mpi_poisson,
  mpi_system_solve_fields, mpi_amr_compiled_parity, mpi_amr_distributed_coarse,
  mpi_coupled_source, mpi_mbox_parity, mpi_cutcell_multibox, condensed_schur_source_stepper,
  test_schur_condensation on the invariance side).
- Integrated AmrSystem + MPI + GPU validation in a single run (phase 10): coarse
  density bit-identical at np=1/2/4 (`dmax = 0`), mass conserved to 0.

**Caveats.** (a) The integrated run does not scale: the coarse level is replicated by
default (redundant compute); the distributed-coarse mode (`distribute_coarse`) is correct and
bit-identical but ~3.7-5x slower (the cross-rank halo traffic dominates the compute
saved), a negative result with numbers, documented. (b) `test_schur_condensation` fails on the
Cuda backend from np=1 (device assembly defect, independent of the number of ranks); it passes
in Serial / Kokkos Serial. (c) On distributed coarse, the global sums differ to rounding
between np (FMA reduction order, ~9e-13); the max stays bit-identical.

---

## Summary matrix

| # | Backend | CMake options (in addition to `-DCMAKE_BUILD_TYPE=Release`) | Local build? | CI? | Validated where |
|---|---------|---------------------------------------------------------|---------------|------|-----------|
| 1 | Kokkos Serial | `-DADC_USE_KOKKOS=ON` + Kokkos Serial | Yes | Yes (`ci-fast`, PR gate) | CI ubuntu (reference oracle) |
| 2 | Kokkos OpenMP | `-DADC_USE_KOKKOS=ON` + Kokkos OpenMP | Yes | Yes (`ci-full`) | CI job `kokkos-openmp`, 91/91 ctest |
| 3 | MPI + Kokkos Serial | `-DADC_USE_KOKKOS=ON -DADC_USE_MPI=ON` + Kokkos Serial | Yes | Yes (`ci-full`) | CI job `mpi`, rank-invariant np=1/2/4 |
| 4 | MPI + Kokkos OpenMP | `-DADC_USE_MPI=ON -DADC_USE_KOKKOS=ON` + Kokkos OpenMP | Yes | No (never MPI+Kokkos in CI) | ROMEO `x64cpu` manual (52/57 rank-invariant) |
| 5 | Kokkos CUDA | `-DADC_USE_KOKKOS=ON` + Kokkos Cuda + `nvcc_wrapper` | No (no local `nvcc`) | No (never CUDA in CI) | ROMEO GH200 manual (`python/tests/gpu/`, sbatch) |
| 6 | MPI + Kokkos CUDA | `-DADC_USE_MPI=ON -DADC_USE_KOKKOS=ON` + Kokkos Cuda + `nvcc_wrapper` | No | No | ROMEO multi-GPU manual (`srun --gpus-per-task=1`) |

**Quick read:**

- Configs 1-3 are covered by GitHub CI (1 on every PR; 2-3 in full mode
  `ci-full`: push to `master`, cron, dispatch, or `ci-full` label).
- Configs 4-6 are never in CI: CI joins MPI with Kokkos Serial but never with Kokkos
  OpenMP, and never builds CUDA. They are validated manually on ROMEO, by bit-for-bit
  comparison to the Serial oracle (`dmax`).
- Configs 5-6 have no local build: `nvcc` runs only on the aarch64 GPU node
  of ROMEO.

For the detailed coverage (each test x each backend column, with the status
`ci-fast` / `ci-full` / `ROMEO` / `self-skip` / `?`), the source of truth remains
[BACKEND_COVERAGE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md). For the GPU port phases and the
detailed validation results, see [GPU_RUNTIME_PORT.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/GPU_RUNTIME_PORT.md).
