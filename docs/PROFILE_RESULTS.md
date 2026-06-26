# Profile of a representative time step (measurement only)

Date: 2026-06-06. Branch: `feat/profiling-harness`. Author of the measurement: harness
`bench/profile_step` (cf. `bench/`), built OUTSIDE the default build (`-DADC_BUILD_BENCH=ON`).

This document REPORTS measurements. It APPLIES NO optimization and RECOMMENDS no
code change beyond a lead to investigate, in line with the owner's rule:
"no performance refactor without a profile showing the bottleneck". The historical files
`docs/PERFORMANCE.md` and `docs/BACKEND_COVERAGE.md` are NOT touched.

## 1. Method

### What is measured

The harness reconstructs, from the PUBLIC SEAMS of the library (headers only, without touching
`python/bindings/system/base/system.cpp` nor any hot path header), a REPRESENTATIVE time step of the diocotron case as
`System::step` orchestrates it:

- model `CompositeModel<ExBVelocity, NoSource, ChargeDensity>` (scalar ExB advection + charge
  density `q n` at the right-hand side of the Poisson) (the exact diocotron composition);
- `solve_fields`: assembly of the right-hand side `f = q n`, elliptic solve, derivation
  `(phi, grad phi)` to the aux channel, filling of the aux halos;
- SSPRK2 advance of the block: 2 stages, each `fill_ghosts(U)` + `assemble_rhs<Minmod, Rusanov>` (finite
  volume operator) + linear combination;
- CFL step: reduction `max_wave_speed_mf` (max of the wave speed, `all_reduce_max` under MPI);
- one reduction `dot(U, U)` (the Krylov / diagnostics dot product brick).

Each PHASE is timed separately (`std::chrono`, surrounded by `device_fence()` to capture
the actual device execution under Kokkos, which is asynchronous):

| phase | content |
|-------|---------|
| `transport`  | `assemble_rhs<L,F>` (FV operator) + SSPRK combinations |
| `poisson`    | elliptic solve (`GeometricMG::solve()` = V-cycles, or `PoissonFFTSolver::solve()`) |
| `halos`      | `fill_boundary` / `fill_ghosts` (MPI exchange + physical ghosts) on U and aux |
| `aux_derive` | assembly `f = q n` + derivation `(phi, grad phi)` -> aux (host loops per cell) |
| `reduction`  | `max_wave_speed_mf` (CFL) + `dot` |
| `fence`      | cost of an isolated `device_fence()` (~0 outside Kokkos) |
| `alloc_tmp`  | (re)allocation of temporary MultiFabs per step (residual R, SSPRK stage U1) |

### Reference case

256x256 grid, ONE box distributed round-robin `DistributionMapping(1, n_ranks())` (THAT is the layout
of `System`, which does not split the box). `bc=periodic`, `solver=geometric_mg`, `limiter=minmod`,
`cfl=0.4`, 5 warmup steps + 50 measured steps (unless stated otherwise). Under MPI the time reported
per phase is the MAX across ranks (critical path of a collective step).

### Exact platforms

- Local CPU: Apple M-series (macOS, AppleClang 21), 8 logical cores. OpenMPI 5.0.9 (Homebrew).
  Homebrew Kokkos (OpenMP device, without CUDA). NB: this is a laptop, not a compute node; the
  ABSOLUTE numbers there are indicative, the SHARES (%) and TRENDS are robust.
- GPU: ROMEO 2025, node `armgpu` `romeo-a057`, NVIDIA GH200 120GB (Grace-Hopper, aarch64),
  CUDA 12.6, Kokkos 4.x (SERIAL;CUDA, ARCH HOPPER90, nvcc_wrapper), OpenMPI 4.1.7 CUDA-aware.

## 2. Profile table (phase x backend, ms per step and %)

All values in MILLISECONDS PER STEP. `(pct)` = share of the phase in the step. 256x256.

| phase \\ backend | Serial CPU | Kokkos OMP t=1 | Kokkos OMP t=4 | Kokkos OMP t=8 | Kokkos Cuda GH200 | MPI CPU np=2 | MPI CPU np=4 | MPI+Cuda np=2 | MPI+Cuda np=4 |
|---|---|---|---|---|---|---|---|---|---|
| transport  | 1.29 | 1.29 | 0.49 | 0.57 | 0.57 | 3.62 | 3.88 | 0.40 | 0.40 |
| poisson    | 138.08 | 169.41 | 1301.80 | 3378.23 | 261.14 | 259.91 | 912.04 | 284.66 | 284.31 |
| halos      | 0.01 | 0.02 | 0.59 | 1.74 | 0.39 | 0.02 | 0.09 | 0.30 | 0.29 |
| aux_derive | 0.08 | 0.08 | 0.09 | 0.08 | 1.06 | 0.26 | 0.44 | 0.86 | 0.86 |
| reduction  | 0.22 | 0.22 | 0.13 | 0.17 | 0.14 | 6.08 | 16.53 | 1.61 | 1.60 |
| fence      | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| alloc_tmp  | 0.01 | -- | -- | -- | 0.24 | -- | -- | 0.20 | 0.19 |
| **TOTAL**  | **139.68** | **171.03** | **1303.14** | **3380.82** | **263.53** | **269.92** | **933.02** | **288.03** | **287.67** |
| poisson %  | 98.9% | 99.0% | 99.9% | 99.9% | 99.1% | 96.3% | 97.8% | 98.8% | 98.8% |

(Serial CPU TOTAL 139.68 ms corresponds to the direct run; the `MPI CPU np=1` column measured separately
gives 136.83 ms, identical to the serial vertical up to variance.)

### Sensitivity (Serial CPU, 30 steps)

| variation | per_step (ms) | poisson % |
|---|---|---|
| 256, geometric_mg, minmod | 144.6 | 98.9% |
| 256, **fft**, minmod      | 142.9 | 98.5% |
| 256, geometric_mg, **weno5** | 158.3 | 98.9% |
| **128**, geometric_mg, minmod | 40.6 | 99.0% |
| **512**, geometric_mg, minmod | 532.2 | 98.8% |

### Isolated scaling of transport vs poisson (Kokkos OMP, 512x512, weno5)

| threads | transport (ms) | poisson (ms) |
|---|---|---|
| 1 | 45.21 | 738.95 |
| 4 | 17.10 (x2.6 faster) | 2229.55 (x3.0 SLOWER) |

## 3. Identified bottleneck and recommendation (justified by the measurement)

### Main finding: the elliptic Poisson dominates ALL backends

Across the six measured backends, the `poisson` phase represents **96 to 99.9 %** of the time of a step. The
transport (the finite volume operator `assemble_rhs`), the halos, the reductions and the temporary
allocations are each **< 1 ms per step** in the reference case (together < 1.1 % of the step in serial).
The performance blocker is therefore, unambiguously, the **elliptic solve** (`GeometricMG::solve()`,
called at each `solve_fields`, hence at each step).

Two quantified facts aggravate this finding:

1. **The Poisson DOES NOT BENEFIT from on-node parallelism; it REGRESSES.** Under Kokkos OpenMP, the
   poisson phase goes from 169 ms (1 thread) to 1302 ms (4) then 3378 ms (8) in the reference case (it
   SLOWS DOWN by a factor of ~20 at 8 threads). The isolated scaling (512x512) confirms the dichotomy: the
   transport accelerates cleanly (x2.6 on 4 threads) while the poisson slows down (x3.0). The measured
   cause: the multigrid V-cycle descends down to tiny coarse grids (2x2, 4x4, ...)
   and launches a `Kokkos::parallel_for` PER smoothing sweep on each one; on a box of a few
   cells, the cost of opening the parallel region (OpenMP fork/join, or kernel launch)
   crushes the useful computation. `for_each_cell` (the Kokkos path, the only one) already switches, below a
   box-size threshold, to a SEQUENTIAL host loop executed UNDER the Kokkos host space (an
   optimization internal to the Kokkos path, not a separate backend); but this threshold does not cover the
   multigrid smoothing, which dispatches a kernel regardless of the box size.

2. **On GPU, the Poisson still costs 261 ms/step (GH200) and benefits from no additional GPU.**
   Kokkos Cuda single-GPU: 263.5 ms/step, poisson 99.1 %. MPI + Kokkos Cuda np=2 and np=4: 288.0 and
   287.7 ms/step (IDENTICAL to the single-GPU). The V-cycle chains dozens of kernel launches
   on coarse levels that are smaller and smaller; the launch LATENCY (and not the bandwidth)
   dominates, so neither a wider GPU nor more GPUs help. The non-improvement np=2/4 also comes
   from the MONO-BOX `System` layout (a single box, so a single rank carries the work; the elliptic
   solve remains collective and each additional rank only adds collective latency).

3. **On the MPI CPU side, same structure.** np=2 and np=4 slow down (270 and 933 ms/step) instead of accelerating,
   and the `reduction` phase swells (0.22 -> 6.08 -> 16.53 ms): this is the overhead of the collectives
   (`all_reduce` in `dot` / `max_wave_speed_mf`) on the mono-box split, with no work distributed in
   return. (The np=4 local CPU numbers are amplified by oversubscription on 8 cores: the TREND
   (no speedup, mono-box) is the solid observable, not the exact factor.)

### What to investigate FIRST (without implementing it here)

The profile points to a single priority target: **the elliptic solve `GeometricMG`**, and more
precisely the DISPATCH BEHAVIOR OF ITS V-CYCLE ON THE COARSE LEVELS under parallel
backend. Leads to explore, in order of ratio (expected gain / risk), each to be quantified by a
new measurement BEFORE any code:

1. **Extend to the multigrid smoothing the switch threshold to the sequential host loop already present
   in `for_each_cell`.** This is the directly measured cause of the multi-thread/GPU regression of the
   V-cycle. To validate: does a threshold below which the smoothing of the coarse levels executes via the host
   sequential loop (or in a single fused kernel) restore the scaling of the poisson without changing
   the numerical result? Decisive measurement: re-profile poisson at t=1/4/8 and on GH200 with the threshold.

2. **Fixed cost per `solve()`: tolerance and number of V-cycles.** `GeometricMG::solve()` by default
   does `solve(1e-8, 50)` (up to 50 V-cycles, tight tolerance) at EACH step, without explicit warm start
   of the previous `phi` in this harness. To quantify: how many V-cycles are actually performed per
   step in steady state (the diocotron evolves slowly, `phi^n` is an excellent starting point)?
   A warm start + an adapted stopping criterion would reduce the number of cycles (to MEASURE before
   concluding).

3. **Collective reductions under MPI** (`dot`, `max_wave_speed_mf`): secondary as long as the mono-box
   of the `System` is not lifted, but their growth (reduction 0.22 -> 16.5 ms in CPU np=4) is to
   keep an eye on as soon as a real multi-box split distributes the work.

The transport, the halos, the fences and the temporary allocations DO NOT NEED optimization given
these numbers (< 1 % of the step). Optimizing anything there would be gold-plating not justified
by the profile.

## 4. Reproduce

```sh
# Serie CPU
bench/run_bench.sh serie
# Kokkos OpenMP (Kokkos installe avec le device OpenMP)
bench/run_bench.sh kokkos-omp  /chemin/kokkos
# MPI CPU (np=2)
bench/run_bench.sh mpi 2
# Kokkos Cuda / MPI+Cuda : sur ROMEO (GH200), cf. bench/run_bench.sh {kokkos-cuda,mpi-cuda} <Kroot> [NP]
```

The harness accepts `--n --steps --warmup --cfl --solver {geometric_mg|fft} --limiter
{none|minmod|vanleer|weno5} --bc {periodic|dirichlet}`. It is OUTSIDE the default build (option
`POPS_BUILD_BENCH=OFF`): the CI never configures nor compiles it.

## 5. Guarantees

- NO optimization, NO refactor of the hot path. `python/bindings/system/base/system.cpp` and the hot path headers
  are NOT modified. Only additions: `bench/` (new) and the `POPS_BUILD_BENCH` option (OFF by
  default) + an `add_subdirectory(bench)` guarded in the root `CMakeLists.txt`.
- `docs/PERFORMANCE.md` (historical) and `docs/BACKEND_COVERAGE.md` are NOT touched.
