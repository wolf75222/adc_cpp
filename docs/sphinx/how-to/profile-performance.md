# Profile performance

Measure where a time step spends its time before you change any code. The repository
ships a profiling harness that rebuilds a representative diocotron step from the public
seams (without touching the hot path) and times each phase separately. Use it to confirm
the bottleneck; do not refactor for performance without a profile that shows it.

This task assumes you have a working build. For installing and a first run, see
[the installation guide](../getting-started/installation.md) and
[the tutorial](../getting-started/tutorial.md).

## What the harness measures

The harness times seven phases of one step, each bracketed by `device_fence()` to
capture the asynchronous device execution under Kokkos:

- `transport`: the finite volume operator `assemble_rhs` plus the SSPRK combinations.
- `poisson`: the elliptic solve (`GeometricMG::solve()` V-cycles, or `PoissonFFTSolver::solve()`).
- `halos`: `fill_boundary` / `fill_ghosts` on the state and the aux channel.
- `aux_derive`: assembly of `f = q n` and derivation of `(phi, grad phi)` to the aux.
- `reduction`: `max_wave_speed_mf` (CFL) and `dot`.
- `fence`: the cost of an isolated `device_fence()`.
- `alloc_tmp`: re-allocation of the temporary fields per step.

The reference case is a 256x256 grid, one box, `bc=periodic`, `solver=geometric_mg`,
`limiter=minmod`, `cfl=0.4`, with 5 warmup steps and 50 measured steps.

## Run the profile

The harness lives in `bench/profile_step.cpp` and is driven by `bench/run_bench.sh`. It
is out of the default build (`POPS_BUILD_BENCH=OFF`), so the build script enables and
compiles it for you. Run the serial CPU profile first, passing the Kokkos Serial install root
as `KROOT` (the build is Kokkos-only; you may instead export `$KOKKOS_ROOT` or
`$POPS_KOKKOS_ROOT` and omit the argument). Replace `KROOT` with the path to your Kokkos install:

```sh
bench/run_bench.sh serie KROOT
```

To profile the Kokkos OpenMP backend, pass the Kokkos install root. Replace `KROOT` with
the path to your Kokkos install:

```sh
bench/run_bench.sh kokkos-omp KROOT
```

To profile MPI on CPU, pass the Kokkos Serial install root as `KROOT` and the rank count as
`NP` (the third argument). You may instead export `$KOKKOS_ROOT` / `$POPS_KOKKOS_ROOT` and pass
`NP` as the third argument. Replace `KROOT` with your Kokkos install and `NP` with the rank
count:

```sh
bench/run_bench.sh mpi KROOT NP
```

The Kokkos Cuda and MPI+Cuda modes (`kokkos-cuda`, `mpi-cuda`) target the ROMEO GH200
node. Each takes the Kokkos root and, for `mpi-cuda`, the rank count.

## Vary the case

The harness accepts `--n`, `--steps`, `--warmup`, `--cfl`, `--solver`, `--limiter` and
`--bc`. The `--solver` value is `geometric_mg` or `fft`; the `--limiter` value is `none`,
`minmod`, `vanleer` or `weno5`; the `--bc` value is `periodic` or `dirichlet`. Use these
to confirm that the bottleneck holds across grid sizes, solvers and reconstructions.

## Read the result

Across the measured backends, the `poisson` phase takes 96 to 99.9 percent of the step.
The transport, the halos, the reductions and the temporary allocations are each below 1 ms
per step. The performance lock is the elliptic solve, called at each step. The Poisson
phase does not benefit from on-node parallelism: the V-cycle descends to tiny coarse grids
and the per-sweep kernel launch cost dominates, so more threads or more GPUs do not help.
Optimize the elliptic solve, not the phases already below 1 percent.

## Where to go next

- The full phase-by-backend table and the leads to investigate live in
  [PROFILE_RESULTS](https://github.com/wolf75222/adc_cpp/blob/master/docs/PROFILE_RESULTS.md).
- For the elliptic solvers themselves, see [the advanced topics](../advanced/index.md).
- To choose a backend for a run, see [the backends index](../backends/index.md).
