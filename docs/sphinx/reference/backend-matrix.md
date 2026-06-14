# Backend matrix

`adc_cpp` compiles the same numerical code to several backends through a single
dispatch seam. The per-backend test coverage is maintained in a single document, which is the
source of truth: [BACKEND_COVERAGE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md). This page summarizes its
structure; do not duplicate the detailed table here (it changes with every test added).

## The backends

| Backend | Build | Where it runs |
|---------|-------|--------------|
| **Kokkos Serial** | `-DADC_USE_KOKKOS=ON`, `Kokkos_ENABLE_SERIAL=ON` | CI gate `build-and-test` (every PR), C++ + Python module. |
| **MPI + Kokkos Serial** | `-DADC_USE_MPI=ON -DADC_USE_KOKKOS=ON` | CI job MPI (`ci-full`), `mpirun` np=1/2/4. |
| **Kokkos OpenMP** | `-DADC_USE_KOKKOS=ON`, `Kokkos_ENABLE_OPENMP=ON` | CI job `ci-full` (enabled since #155). |
| **Kokkos Cuda (GH200)** | Kokkos + `Kokkos_ARCH_HOPPER90`, one GPU per rank | ROMEO manual only (never in CI). |
| **MPI + Kokkos Cuda** | previous build + OpenMPI CUDA-aware | ROMEO manual (`srun -n {1,2,4}`). |

```{important}
CI never builds `-DADC_USE_KOKKOS=ON -DKokkos_ENABLE_CUDA=ON`. All GPU validation
(single-GPU Cuda or multi-GPU MPI) is done manually on the ROMEO supercomputer (GH200
node), not in CI. The GPU harnesses live in `python/tests/gpu/` and are launched by
SBATCH. See [GPU_RUNTIME_PORT.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/GPU_RUNTIME_PORT.md) and the
[limitations](known-limitations.md) page.
```

## How to read the matrix

The table in [BACKEND_COVERAGE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md) cross-references each test (C++ ctest and
Python) with the backends above. The legend distinguishes in particular:

- **build-and-test**: REQUIRED gate on every PR; compiles in Kokkos Serial (C++ + Python
  module).
- **ci-full**: runs in full mode (push `master`, nightly, labeled PR), adds MPI + Kokkos
  Serial and Kokkos OpenMP.
- **ROMEO**: validated manually on GH200; the exact harness is cited in parentheses, with
  the numerical evidence (e.g. `dmax=0`).
- **self-skip**: the test detects the absence of the backend and exits cleanly (exit 0).
- **?**: not exercised; these cells are listed in the "Notable gaps" section of the
  source document.

## Highlights

- The Python tests only exercise Kokkos Serial: the `_adc` module is built in CI in
  Kokkos Serial, without MPI or Cuda. No Python test covers MPI or Cuda.
- The FFT Poisson path (`PoissonFFTSolver`) is rejected under MPI (single-rank by design);
  a test locks this non-regression.
- The Kokkos Cuda and MPI + Kokkos Cuda columns are either `ROMEO` (validated by hand),
  or `?`. The detail of the remaining GPU gaps is in the "Notable gaps" section of the source
  document.

For the numerical counts (ctest targets, MPI entries, Python tests) and the exhaustive list of
gaps, refer directly to [BACKEND_COVERAGE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md).
