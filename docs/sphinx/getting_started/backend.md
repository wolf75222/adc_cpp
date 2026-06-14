# Checking your backend

The same `adc_cpp` source code runs on the single on-node Kokkos backend, whose target is
chosen when Kokkos is installed: Kokkos Serial (sequential), Kokkos OpenMP (multi-thread CPU),
Kokkos CUDA/HIP (GPU). MPI is added as an option for the distributed case. Kokkos is required and fixed at
compile time. This page explains how to know which execution space is running, because the answer is not
always the one you think.

## The golden rule: the backend is fixed at compile time

The backend is not a runtime argument. It is a property of the `adc` target, attached to
the CMake interface: everything that links `adc` (the compiled facade, the tests) inherits it, and the
`for_each_cell` seam switches to the chosen execution space. To change backend, you
reconfigure and recompile (see [Installation](installation.md)). So there is nothing to
enable in a Python script.

## The Python module is serial

Often surprising: the `_adc` module shipped/tested in CI is built
only in Kokkos Serial (no MPI in the Python job). Any
Python script that does `import adc` therefore runs sequentially, whatever the hardware. This is
documented in the [`BACKEND_COVERAGE.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md) matrix (section 2:
"the Python binding only routes to Kokkos Serial").

The tutorial actually shows it at startup: it probes a few optional module attributes
(`backend`, `parallel_backend`, `build_info`) and, lacking such an attribute, explicitly falls
back to the "serial (default)" mention. The module does not lie about its parallelism; it does not
expose (yet) a backend flag on the Python side. The only ABI diagnostic exported is
`adc.abi_key()` (compiler + C++ standard + header signature), which serves the native DSL,
not to identify the dispatch backend.

For multi-thread or GPU parallelism, it is the C++ facade that must be compiled against a Kokkos
installed for the wanted execution space (OpenMP, CUDA), and the computation must be ported to the C++ side (tests,
executables, or an `adc_cases` case compiled natively).

## Python drives, compiled C++ computes

The split of roles does not depend on the hardware. Python builds the `System`, runs the
time loop and writes the outputs; the per-cell work goes through the `for_each_cell` seam, compiled
for the Kokkos execution space chosen at install time (Serial, OpenMP, or CUDA). The word "sequential"
describes the `_adc` module as CI builds it, in Kokkos Serial; it is not a limit of the
Python. The module links the same `adc::adc` as the tests (cf. `python/CMakeLists.txt`) and would inherit
a Kokkos OpenMP or CUDA if it were built against one.

On GPU, and therefore on ROMEO, the computation today runs through generated C++ harnesses, not through the
Python module. The flow is in [`GPU_ROMEO.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/GPU_ROMEO.md):
a host script (`python/tests/gpu/gen_kokkos_harness.py`, then `gen_kokkos_sim.py` for a full
simulation) emits a `.cpp` or `.cu` from the model; it is sent to ROMEO; `srun`
compiles it with `nvcc_wrapper` and runs it on the GH200 node. Python stays the author and
host-side orchestrator, the compiled binary does the computation on the device. The `_adc` module
is not built in CUDA on the compute nodes.

## Backend coverage

| Backend | How to get it | Where it is validated |
|---|---|---|
| Kokkos Serial | default build (`Kokkos_ENABLE_SERIAL=ON`) ; Python module | CI (gate `build-and-test`, C++ + Python) |
| MPI CPU | `-DADC_USE_MPI=ON`, launched via `mpirun -np {1,2,4}` | CI (ci-full) |
| Kokkos OpenMP | Kokkos installed with `Kokkos_ENABLE_OPENMP=ON` | CI (ci-full, 91/91 ctest) |
| Kokkos CUDA (GH200) | `-DADC_USE_KOKKOS=ON` + `Kokkos_ARCH_HOPPER90` + `nvcc_wrapper` | ROMEO, manual (never in CI) |
| MPI + Kokkos CUDA | same build + OpenMPI CUDA-aware, `srun --gpus-per-task=1` | ROMEO, manual |

CI never builds with CUDA enabled (the runners have no GPU): all the
"Kokkos CUDA" cells in the matrix are either ROMEO or not exercised. The up-to-date count and
the status of each test by backend live in [`BACKEND_COVERAGE.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md)
(do not duplicate it here).

## On GPU: ROMEO, manual

The GPU execution model is MPI + Kokkos, not hand-written CUDA: MPI distributes the
subdomains (one GPU per rank), Kokkos parallelizes the local computation and abstracts the hardware via
its `ExecutionSpace` (`Cuda` backend on NVIDIA). The same `.cpp` runs in `exec=Serial` /
`OpenMP` on CPU and in `exec=Cuda` on GPU depending on the backend chosen at compile time; `nvcc_wrapper`
is only the compiler required by the Kokkos Cuda backend. Detail:
[`GPU_RUNTIME_PORT.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/GPU_RUNTIME_PORT.md).

The GH200 validation is done by hand on ROMEO (node `armgpu`, Grace-Hopper, `cuda/12.6`,
Kokkos 4.4.01, `Kokkos_ARCH_HOPPER90`, OpenMPI CUDA-aware), via dedicated SLURM harnesses. State
(June 2026): the full single-grid solver, the AMR field ops, the multi-GPU halo exchange
(np=1/2/4 bit-identical) and the integrated AmrSystem + MPI + GPU validation are validated. The
typical `srun` harness and the proofs (`maxdiff=0` against the CPU) are in
[`GPU_ROMEO.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/GPU_ROMEO.md) and [`GPU_RUNTIME_PORT.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/GPU_RUNTIME_PORT.md).

## What to remember

- The `_adc` module as CI builds it = Kokkos Serial ; Python drives, compiled C++ computes.
- For multi-thread CPU / GPU: recompile the C++ facade against a Kokkos installed for the wanted space (OpenMP, CUDA) ; for the distributed case, add `-DADC_USE_MPI=ON`.
- The GPU is ROMEO-manual, via generated C++ harnesses ; CI is CPU only.
- The source of truth for "which test, which backend" is [`BACKEND_COVERAGE.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md).
