# Run with MPI

This guide builds `adc_cpp` with the distributed comm seam and runs the tests across
several MPI ranks. It assumes you can already build the reference Kokkos Serial backend
and that MPI and a Kokkos install are available on your machine. For the full backend
model, the validation status of each configuration and the coverage matrix, see the
[parallel backends page](../backends/index.md).

MPI distributes subdomains across ranks through the `comm.hpp` seam; the local compute
stays on the Kokkos Serial execution space, one process per rank. Enable it with the
`ADC_USE_MPI` CMake option.

## Build with MPI

This uses the same Kokkos Serial install as the reference build, plus the MPI seam.

1. Configure the build with both `ADC_USE_KOKKOS` and `ADC_USE_MPI` turned on.
   Replace `KOKKOS_PREFIX` with the path to your Kokkos Serial install.

   ```bash
   cmake -S . -B build-mpi -DCMAKE_BUILD_TYPE=Release -DADC_USE_KOKKOS=ON -DADC_USE_MPI=ON -DKokkos_ROOT=KOKKOS_PREFIX
   ```

2. Compile.

   ```bash
   cmake --build build-mpi -j
   ```

## Run the distributed tests

The MPI test targets each replay the run at np=1, np=2 and np=4 under `mpirun`, and
check that the observables (parity, AMR, Krylov, mass) are bit-identical across the
number of ranks.

1. Run the test suite. On a small machine, allow oversubscribe so that np=4 fits on
   fewer than four physical cores.

   ```bash
   OMPI_MCA_rmaps_base_oversubscribe=true ctest --test-dir build-mpi --output-on-failure
   ```

The non-MPI tests run at np=1 in this build (MPI-linked, single process); the MPI
block checks rank invariance at np=1, np=2 and np=4.

## Run on multiple CPU threads per rank

To use all cores of every rank, build the MPI plus Kokkos OpenMP hybrid against a
Kokkos OpenMP install. Replace `KOKKOS_OPENMP_PREFIX` with the path to that install.

```bash
cmake -S . -B build-mpi-omp -DCMAKE_BUILD_TYPE=Release -DADC_USE_MPI=ON -DADC_USE_KOKKOS=ON -DKokkos_ROOT=KOKKOS_OPENMP_PREFIX
```

Bound the thread count per rank with `OMP_NUM_THREADS` when you run the tests.

```bash
OMP_NUM_THREADS=4 OMPI_MCA_rmaps_base_oversubscribe=true ctest --test-dir build-mpi-omp --output-on-failure
```

## Next steps

- To run multi-GPU on ROMEO, see the [parallel backends page](../backends/index.md).
- For the Python `adc` module, which is serial only and does not exercise the MPI
  paths, see the [installation page](../getting_started/installation.md).
