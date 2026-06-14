# Performance and profiling


The repository ships a profiling harness: `bench/profile_step.cpp`, driven by
`bench/run_bench.sh`. It rebuilds a time step representative of the diocotron from
the public seams (without touching the hot path) and times each phase
(`transport`, `poisson`, `halos`, `aux_derive`, `reduction`, `fence`, `alloc_tmp`)
bracketed by `device_fence()` to capture the device execution under Kokkos.

The harness is out of the default build (`ADC_BUILD_BENCH=OFF`): the CI never
configures or compiles it. It is enabled explicitly:

```sh
bench/run_bench.sh serie                  # Serie CPU
bench/run_bench.sh kokkos-omp  <Kroot>    # Kokkos OpenMP
bench/run_bench.sh mpi          2         # MPI CPU, 2 rangs
bench/run_bench.sh mpi-cuda    <Kroot> 4  # MPI + Kokkos Cuda (GH200), 4 rangs
```

It accepts `--n --steps --warmup --cfl --solver {geometric_mg|fft} --limiter
{none|minmod|vanleer|weno5} --bc {periodic|dirichlet}`.

Main finding of the profile: on the six measured backends, the `poisson` phase dominates
the step at 96 to 99.9 %. The transport, the halos, the reductions and the temporary
allocations are each `< 1 ms` per step (together `< 1.1 %` of the step in serial). The
performance lock is, unambiguously, the elliptic solve (`GeometricMG::solve()`, called at
each step). Two measured aggravating facts: Poisson does not benefit from the on-node
parallelism (it regresses, the V-cycle descends down to tiny grids 2x2, 4x4, and the
launch cost of each kernel crushes the useful computation), and on GPU the launch latency
dominates (neither a larger GPU nor more GPUs help, all the more so as the `System`
layout is single-box).

No performance refactor without a profile showing the bottleneck.
> `PROFILE_RESULTS.md` reports the measurements and points to a target (the V-cycle dispatch on
> the coarse levels); it applies no optimization.

## Going further

- Full profile (phase x backend table, exact platforms M-series + GH200, leads to
  investigate): [PROFILE_RESULTS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/PROFILE_RESULTS.md).
- `docs/PERFORMANCE.md` exists but carries historical measurements (old application
  drivers, M1): do not read it as the current performance.
