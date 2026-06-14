# MPI + Kokkos CUDA


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
