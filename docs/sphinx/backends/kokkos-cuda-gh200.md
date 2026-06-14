# Kokkos CUDA on GH200


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
