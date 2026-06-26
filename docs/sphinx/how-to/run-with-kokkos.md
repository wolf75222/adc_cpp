# Run with Kokkos

This guide shows how to pick the Kokkos execution space for a run: `Serial`
(single-thread CPU), `OpenMP` (multi-thread CPU), or `Cuda` (NVIDIA GPU). The
execution space is fixed at compile time, not a runtime flag: you choose it by
pointing the build at a matching Kokkos install and reconfiguring. This guide
assumes you can build `adc_cpp` and run `ctest`; if not, start with
[Installation](../getting-started/installation.md). For the
configuration-by-configuration reference, see
[Parallel backends](../backends/index.md).

The space is decided when Kokkos is installed (`Kokkos_ENABLE_SERIAL`,
`Kokkos_ENABLE_OPENMP`, `Kokkos_ENABLE_CUDA`), then selected by the
`-DKokkos_ROOT` you point the build at. `KOKKOS_PREFIX` below is a Kokkos install
built with `Kokkos_ENABLE_SERIAL=ON`; `KOKKOS_OPENMP_PREFIX` is one built with
`Kokkos_ENABLE_OPENMP=ON`.

## Run on Kokkos Serial

This is the reference build, the oracle the other spaces are compared against.
Configure and build against a Serial Kokkos:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DADC_USE_KOKKOS=ON -DKokkos_ROOT="$KOKKOS_PREFIX"
```

```bash
cmake --build build -j
```

To run the tests:

```bash
ctest --test-dir build --output-on-failure
```

## Run on Kokkos OpenMP

Use a Kokkos install built with `Kokkos_ENABLE_OPENMP=ON`. The `adc_cpp` options
are identical to Serial; only the Kokkos install changes. Configure and build:

```bash
cmake -S . -B build-kokkos-omp -DCMAKE_BUILD_TYPE=Release -DADC_USE_KOKKOS=ON -DKokkos_ROOT="$KOKKOS_OPENMP_PREFIX"
```

```bash
cmake --build build-kokkos-omp -j
```

Bound the thread count with `OMP_NUM_THREADS` when you run:

```bash
OMP_NUM_THREADS=4 ctest --test-dir build-kokkos-omp --output-on-failure
```

## Run on Kokkos Cuda

The `Cuda` execution space targets the ROMEO GH200 node. There is no `nvcc` on
the dev workstations, so this build runs only on the ROMEO `armgpu` node, never
in CI. Use a Kokkos install built with
`Kokkos_ENABLE_CUDA=ON -DKokkos_ARCH_HOPPER90=ON` and the `nvcc_wrapper`
compiler:

```bash
module load cuda/12.6
```

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DADC_USE_KOKKOS=ON -DCMAKE_CXX_COMPILER="$KOKKOS_PREFIX/bin/nvcc_wrapper" -DKokkos_ROOT="$KOKKOS_PREFIX"
```

```bash
cmake --build build-cuda -j
```

Run one GPU harness under SLURM, replacing `HARNESS` with a binary from
`build-cuda/bin`:

```bash
srun -p instant --constraint=armgpu --gres=gpu:1 ./build-cuda/bin/HARNESS
```

A multi-thread (Kokkos OpenMP) Python module is available via the `python-parallel`
CMake preset; set the thread count with `pops.set_threads(n)` right after
`import adc`. There is no nvcc/CUDA Python module, so GPU runs stay C++-only, and
the Python module does not exercise the MPI code paths (MPI is validated through
the C++/ctest path). To distribute across ranks, add `-DADC_USE_MPI=ON` and see
[Parallel backends](../backends/index.md).
