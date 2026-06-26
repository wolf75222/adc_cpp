# Presentation

`adc_cpp` is a C++23 solver for coupled hyperbolic-elliptic systems on an
AMR mesh stack written *from scratch*. Concretely: one or more densities are
transported by a finite-volume scheme (the hyperbolic part), while a system
Poisson supplies at each step the potential that drives this transport (the elliptic
part). The reference benchmark is the diocotron instability (ExB drift), but the core names
no scenario: it provides generic bricks that one composes.

## What `adc_cpp` is

- A header-only core, model-agnostic. The physics reduces to local laws
  (flux, sources, closures), device-callable, that see neither MPI, nor AMR, nor halos. A
  model is a brick composition (`pops.Model(state, transport, source, elliptic)`), not
  a hard-coded scenario.
- A `from scratch` mesh stack: `Box` / `BoxArray` / `MultiFab` /
  `Geometry` containers, and a block-structured multi-level, multi-patch AMR hierarchy
  (Berger-Rigoutsos clustering, coverage-aware reflux).
- A single parallelism seam. The same source code switches between the Kokkos
  execution spaces (sequential Serial / multi-threaded OpenMP CPU / Cuda-HIP GPU, chosen at the Kokkos
  install) and, optionally, MPI for the distributed case, through the `for_each_cell` seam and the
  `comm` layer. No CUDA kernel is written by hand: Kokkos abstracts the hardware.
- Two Poisson solvers: geometric multigrid (`GeometricMG`, red-black Gauss-Seidel
  V-cycle) and direct spectral FFT (`PoissonFFTSolver`).
- Python bindings via pybind11. The `adc` module is the composition facade: Python
  says what (which blocks, which scheme, which Poisson), the compiled C++ does the computation cell by
  cell. No numpy back-and-forth in the hot path.

The code is organized in five orthogonal layers, where a high layer expresses the problem and a
low layer executes it, without downward dependency:

| Layer | What |
|---|---|
| **Physics** | local laws: flux, equation of state, sources, closures (device-callable) |
| **Numerics** | reconstruction, Riemann flux, elliptic operator, boundary conditions |
| **Mesh / data** | `Box`, `BoxArray`, `MultiFab`, `Geometry`, AMR hierarchy |
| **Execution** | seams: `for_each_cell` (Kokkos: Serial / OpenMP / Cuda), `comm` (MPI), allocator |
| **Time / coupling** | SSPRK, IMEX, splitting, reflux / average-down / subcycling |

The key point: the physical model never depends on the parallel backend. Porting to GPU is
mostly a data-residence task, not a rewrite of the compute kernels.

## Honest scope

- Kokkos is the only on-node backend, and it is required (`-DADC_USE_KOKKOS=ON`, ON by
  default; configuring without Kokkos is a fatal CMake error). The target is chosen at the
  Kokkos installation: sequential (Serial), multi-threaded CPU (OpenMP) or GPU (Cuda/HIP).
  The standalone OpenMP backend (the `POPS_USE_OPENMP` option) has been removed; MPI remains optional
  for the distributed case.
- The GPU (NVIDIA GH200) is validated manually on ROMEO, not in CI: the runners have no
  GPU. The CI runs on CPU: the required PR gate (`build-and-test`) builds in
  Kokkos Serial (C++ + Python module), and the `ci-full` mode adds MPI + Kokkos Serial and
  Kokkos OpenMP. See [Check your backend](backend.md).
- The Python module (`_pops`) is built in Kokkos Serial: it does not route to MPI. The multi-thread
  CPU or the GPU are obtained by building the facade against a Kokkos OpenMP
  or Cuda/HIP install; the distributed case, by adding `-DADC_USE_MPI=ON`.
- The diocotron tutorial is a reduced model (one density advected by the ExB drift), the
  normalization benchmark, not a reproduction of the full Euler-Poisson system.

## Going further

- The repository split (lib vs scenarios): [Repository organization](repository-layout.md).
- The detailed architecture, the seams and the design decisions: the contributor reference
  [`ARCHITECTURE.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md) (section 1, "five orthogonal layers") and
  [`CHOICES.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/CHOICES.md).
- The algorithms (flux, MUSCL/WENO, multigrid, AMR reflux): [`ALGORITHMS.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md).
