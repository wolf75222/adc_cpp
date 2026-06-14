# Run on GH200

This guide builds and runs `adc_cpp` on the ROMEO GH200 GPU node with the Kokkos
Cuda execution space. The GPU harnesses live in `python/tests/gpu/`; they compile
the same logic in `exec=Cuda` and in the `exec=Serial` oracle and compare cell by
cell. For what each backend covers and where it is validated, see the
[parallel backends page](../backends/index.md).

`nvcc` runs only on the aarch64 GPU node, never on the x86 login and never in CI,
so build and run inside a SLURM allocation. Replace `ACCOUNT` with your ROMEO
account, `KOKKOS_PREFIX` with the path to a Kokkos install built with
`Kokkos_ENABLE_CUDA=ON` and `Kokkos_ARCH_HOPPER90=ON`, and `HARNESS` with the
harness binary you built.

## Build on the GPU node

1. Load the CUDA module.

   ```bash
   module load cuda/12.6
   ```

2. Configure with the Kokkos Cuda install and the `nvcc_wrapper` compiler.

   ```bash
   cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DADC_USE_KOKKOS=ON -DCMAKE_CXX_COMPILER="$KOKKOS_PREFIX/bin/nvcc_wrapper" -DKokkos_ROOT="$KOKKOS_PREFIX"
   ```

3. Build.

   ```bash
   cmake --build build-cuda -j
   ```

## Run one GPU

Launch the harness through SLURM on one Grace-Hopper GPU.

```bash
srun --account=ACCOUNT -p instant --constraint=armgpu --gres=gpu:1 ./build-cuda/bin/HARNESS
```

Each GPU harness reports `dmax`, the maximum absolute difference between the Cuda
result and the Serial oracle. The full single-grid solver and the elliptic bricks
reach `dmax = 0` on GH200; a flux-only harness deviates by one ULP from the FMA
contraction of `nvcc_wrapper`, which is not a bug.

## Run several GPUs with MPI

For the multi-GPU production mode, add `-DADC_USE_MPI=ON` and build against a
CUDA-aware OpenMPI.

```bash
cmake -S . -B build-mpicuda -DCMAKE_BUILD_TYPE=Release -DADC_USE_KOKKOS=ON -DADC_USE_MPI=ON -DCMAKE_CXX_COMPILER="$KOKKOS_PREFIX/bin/nvcc_wrapper" -DKokkos_ROOT="$KOKKOS_PREFIX"
```

Build it.

```bash
cmake --build build-mpicuda -j
```

Run with one GPU per rank.

```bash
srun -n 4 --gpus-per-task=1 --constraint=armgpu ./build-mpicuda/bin/HARNESS
```

The elliptic, Schur, Poisson and AMR harnesses are rank-invariant at np=1/2/4
(`dmax = 0`). The integrated AmrSystem run is correct but does not scale: the
distributed-coarse mode is bit-identical yet slower because cross-rank halo
traffic dominates the compute saved.

## Where to go next

- To pick a CPU or serial backend instead, read the [parallel backends page](../backends/index.md).
