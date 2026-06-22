# GPU verification of the generated brick (ROMEO, NVIDIA GH200)

The `adc.dsl` DSL generates a C++ hyperbolic brick (`emit_cpp_brick`) that is already device-ready (`ADC_HD` ->
`__host__ __device__` under nvcc, device-safe ops, `std::sqrt` like `adc::Euler`). This document records
the verification on a REAL GPU: the generated brick `EulerGen` runs in a CUDA kernel and gives the
same flux as the hand-written `adc::Euler`.

Not integrated into CI (no GPU on the runners). MANUAL test, reproducible on ROMEO.

## Harness

ROMEO: x86_64 login, aarch64 GPU nodes (Grace-Hopper, `armgpu`, 4x H100/GH200), SLURM, `cuda/12.6`.
nvcc must run ON the GPU node (aarch64 binary), so we compile and run inside the allocation.

```bash
# 1. generer le harnais CUDA (depuis la racine du repo, paquet adc construit dans build-py)
PYTHONPATH=$PWD/build-py/python python3 python/tests/gpu/gen_cuda_harness.py   # -> /tmp/euler_gpu.cu

# 2. envoyer en-tetes + harnais + script
rsync -az include /tmp/euler_gpu.cu python/tests/gpu/romeo_run.sh romeo:adc_dsl_gpu/
ssh romeo 'mv ~/adc_dsl_gpu/romeo_run.sh ~/adc_dsl_gpu/run.sh'

# 3. compiler (nvcc sm_90) + executer sur un noeud H100
ssh romeo 'cd ~/adc_dsl_gpu && srun --account=<compte> -p instant --constraint=armgpu \
           --gres=gpu:1 --mem=8G -c 4 -t 3 bash run.sh'
```

`romeo_run.sh`: `module load cuda/12.6` then `nvcc -std=c++20 -arch=sm_90 -I include euler_gpu.cu -o
euler_gpu && ./euler_gpu`. The `kflux` kernel instantiates `adc_generated::EulerGen` on the device, computes
the flux for a few states, and main compares against `adc::Euler` computed on the host.

## Result (obtained)

```
noeud=romeo-a057  arch=aarch64
BUILD_OK
device=NVIDIA GH200 120GB  maxdiff(GPU EulerGen vs hote adc::Euler)=0.000e+00
```

The brick GENERATED from Python formulas compiles with nvcc and runs on GH200 giving a
result BIT-IDENTICAL to the hand-written brick. The codegen is therefore correct all the way to the production
GPU.

## Kokkos (parallel_for dispatch, CUDA backend)

Beyond raw CUDA, we verify the generated brick through the REAL parallel dispatch that the solver
uses (`adc/mesh/execution/for_each.hpp` -> `Kokkos::parallel_for`). We build Kokkos from sources
(no module on ROMEO) then a `Kokkos::parallel_for(KOKKOS_LAMBDA ...)` harness that computes the flux
of `EulerGen` on the device and compares it against `adc::Euler` on the host.

```bash
# 1. generer le harnais Kokkos (depuis la racine, paquet adc construit dans build-py)
PYTHONPATH=$PWD/build-py/python python3 python/tests/gpu/gen_kokkos_harness.py   # -> /tmp/kokkos_euler.cpp

# 2. envoyer en-tetes + harnais (+ CMakeLists) + script, cloner Kokkos
rsync -az include /tmp/kokkos_euler.cpp python/tests/gpu/kokkos_CMakeLists.txt \
      python/tests/gpu/romeo_kokkos_build.sh romeo:adc_dsl_kk/
ssh romeo 'cd ~/adc_dsl_kk && mkdir -p harness && mv kokkos_euler.cpp harness/ \
           && mv kokkos_CMakeLists.txt harness/CMakeLists.txt && mv romeo_kokkos_build.sh kk_build.sh \
           && git clone --depth 1 -b 4.4.01 https://github.com/kokkos/kokkos.git'

# 3. build Kokkos (CUDA + Serial, HOPPER90) + harnais + run, sur un noeud GPU
ssh romeo 'cd ~/adc_dsl_kk && srun --account=<compte> -p instant --constraint=armgpu \
           --gres=gpu:1 --mem=16G -c 16 -t 25 bash kk_build.sh'
```

`kk_build.sh` configures Kokkos with `-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_HOPPER90=ON
-DKokkos_ENABLE_SERIAL=ON` and `nvcc_wrapper` as the compiler, installs, then compiles the harness
(`find_package(Kokkos)` + `Kokkos::kokkos`).

Result (obtained):
```
KOKKOS_OK
HARNESS_OK
exec_space=Cuda  maxdiff(Kokkos EulerGen vs hote adc::Euler)=5.551e-17
```

The default execution space is `Cuda` (the kernel does run on the GPU). The deviation is one ULP
(5.55e-17), due to the FMA contraction of nvcc_wrapper differing from the host, not a bug: the generated
brick is correct through the Kokkos dispatch on GH200.

## COMPLETE case (time-stepping) on GPU via adc's Kokkos seam

We go beyond an isolated flux: a complete 2D Euler case (80 steps, CFL, Rusanov order 1, periodic)
advances ENTIRELY on GPU through `adc::for_each_cell` / `for_each_cell_reduce_max|sum`
(`adc/mesh/execution/for_each.hpp` -> `Kokkos::parallel_for` / `parallel_reduce`). We simulate the SAME thing with
the generated brick `EulerGen` and with `adc::Euler`, and we compare the final fields + the mass.

```bash
PYTHONPATH=$PWD/build-py/python python3 python/tests/gpu/gen_kokkos_sim.py   # -> /tmp/kokkos_euler_sim.cpp
rsync -az include /tmp/kokkos_euler_sim.cpp romeo:adc_dsl_kk/sim/   # + kokkos_sim_CMakeLists.txt -> sim/CMakeLists.txt
rsync -az python/tests/gpu/romeo_kokkos_sim_build.sh romeo:adc_dsl_kk/kk_sim_build.sh
ssh romeo 'cd ~/adc_dsl_kk && srun --account=<compte> -p instant --constraint=armgpu \
           --gres=gpu:1 --mem=16G -c 16 -t 25 bash kk_sim_build.sh'
```

The harness defines `#define ADC_HAS_KOKKOS` then includes `adc/mesh/execution/for_each.hpp`: the cell
loops therefore go through the solver's REAL Kokkos dispatch (the same call site as on CPU).

Result (obtained):
```
exec=Cuda  n=64 steps=80  mass_drel=0.000e+00  rho[min,max]=[0.8912,1.0464]  maxdiff(EulerGen vs adc::Euler, GPU)=8.882e-16
```

80 time steps on GH200: mass EXACTLY conserved, non-trivial dynamics (the pressure bubble
makes the density vary), and the generated brick == `adc::Euler` to machine precision (8.9e-16 accumulated
over 80 steps). The complete case therefore runs on GPU through adc's Kokkos machinery.

## Limits / next steps

- The complete case goes through `adc/mesh/execution/for_each.hpp` (the solver's REAL Kokkos seam), but we do not
  rebuild the whole runtime stack (System / AMR / MPI) on GPU here: the cell loops
  use the same dispatch as production, which is enough to validate the device.
- Type-erased dispatch at runtime: DONE elsewhere (adc::IModel, see python/tests/test_dsl_dynamic.py).
  The GPU harnesses above, on the other hand, compile the brick STATICALLY (perf); the type-erased path
  (vtable) is a HOST complement, not for the hot GPU loop.
