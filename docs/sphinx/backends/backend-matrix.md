# Backend matrix


| # | Backend | CMake options (in addition to `-DCMAKE_BUILD_TYPE=Release`) | Local build? | CI? | Validated where |
|---|---------|---------------------------------------------------------|---------------|------|-----------|
| 1 | Kokkos Serial | `-DPOPS_USE_KOKKOS=ON` + Kokkos Serial | Yes | Yes (`ci-fast`, PR gate) | CI ubuntu (reference oracle) |
| 2 | Kokkos OpenMP | `-DPOPS_USE_KOKKOS=ON` + Kokkos OpenMP | Yes | Yes (`ci-full`) | CI job `kokkos-openmp`, 91/91 ctest |
| 3 | MPI + Kokkos Serial | `-DPOPS_USE_KOKKOS=ON -DPOPS_USE_MPI=ON` + Kokkos Serial | Yes | Yes (`ci-full`) | CI job `mpi`, rank-invariant np=1/2/4 |
| 4 | MPI + Kokkos OpenMP | `-DPOPS_USE_MPI=ON -DPOPS_USE_KOKKOS=ON` + Kokkos OpenMP | Yes | No (never MPI+Kokkos in CI) | ROMEO `x64cpu` manual (52/57 rank-invariant) |
| 5 | Kokkos CUDA | `-DPOPS_USE_KOKKOS=ON` + Kokkos Cuda + `nvcc_wrapper` | No (no local `nvcc`) | No (never CUDA in CI) | ROMEO GH200 manual (`python/tests/gpu/`, sbatch) |
| 6 | MPI + Kokkos CUDA | `-DPOPS_USE_MPI=ON -DPOPS_USE_KOKKOS=ON` + Kokkos Cuda + `nvcc_wrapper` | No | No | ROMEO multi-GPU manual (`srun --gpus-per-task=1`) |

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
