> **STATUS: historical.** This page records past run-time performance numbers; they reflect the runs at the time, not the current build. Read it as a historical record, not as a current source of truth (class D, see `DOC_QUALITY.md`).

# Performance (run-time)

> **STATUS: historical (non-normative).** Dated measurements (M1, old milestone). The current figures live in PROFILE_RESULTS.md; do not rely on these as the current perf status.

> **HISTORICAL MEASUREMENTS (application drivers, old). DO NOT read as the current perf.**
> An up-to-date Kokkos/MPI/GPU campaign is under way (see docs/PROFILE_RESULTS.md when available).


CS:APP methodology: measure first, identify the bottleneck, transform, re-measure.
Bench: Euler-Poisson coupled step, N=256, Apple M1 Pro (6 perf cores + 2 efficiency),
AppleClang -O3 -DNDEBUG.

> These measurements were taken with APPLICATION drivers and benches (named scenarios,
> figure harness) that live in [`adc_cases`](https://github.com/wolf75222/adc_cases),
> not in this repository (the core is model-agnostic). The figures remain representative
> of the core (elliptic operator, transport, AMR reflux); the file/type names cited
> below (`bench_amr`, `*Config`, figure scripts) are those of the application repository. The
> figures kept are under `docs/`. The trace of the GH200 device validations is in
> [GPU_RUNTIME_PORT.md](GPU_RUNTIME_PORT.md).

## Profile of a coupled step (where the time goes)

| phase | ms/step | share |
| --- | --- | --- |
| full coupled step (PerStage) | 53 | 100% |
| hyperbolic (2x assemble_rhs MUSCL + saxpy/lincomb) | 7.7 | 14% |
| **Poisson + aux (2 multigrid solves)** | **45.5** | **86%** |

The run-time IS the elliptic solver. The hyperbolic part is negligible. `-mcpu=native`
changes nothing (~2%): we are not compute-bound on the hyperbolic part.

## Measured levers

### 1. Coupling policy: `OncePerStepCoupling` -> x2.6

| policy | ms/step |
| --- | --- |
| PerStage (Poisson at each RK stage, 2 solves/step) | 52.3 |
| **OncePerStep (Poisson 1x/step)** | **19.8** |

x2.6 gain (better than the expected x2: the single solve starts from a warm-start closer
to convergence). Cost: 1st-order coupling instead of 2nd-order. Accessible through the
core `CouplingPolicy`: `OncePerStepCoupling` instead of `PerStage`
(`coupling/coupling_policy.hpp`).

### 2. Per-kernel multi-thread (Kokkos OpenMP space): LOSING here

| Kokkos space | 1 thread | 4 threads | 8 threads |
| --- | --- | --- | --- |
| Kokkos OpenMP (PerStage) | 100 | 91 | 119 |

To be compared with the **52 ms in Kokkos Serial**. Two reasons, confirmed by measurement:
- outlining each `for_each_cell` into a parallel region **breaks the inlining** of the hot
  loops (Kokkos OpenMP at 1 thread = 2x slower than Kokkos Serial);
- the multigrid is **memory-bound and latency-bound** (sequential traversal of the
  levels, red-black GS), a bad candidate for per-kernel fork/join.

Without a safeguard, it was **x47 slower** (fork/join of 8 threads on 2x2 boxes of the
coarse levels). Fixed: `for_each_cell` switches to a sequential host loop (under the host
Kokkos space) below a cell threshold -> no fork/join for the tiny boxes. This switch is
INTERNAL to the Kokkos path. Kokkos OpenMP is still not a gain here, but is no longer a
pitfall. The right grain would be to parallelize ABOVE the level loop (consolidated
region), not per kernel.

## Poisson via FFT for the periodic case: DONE, ~5x

The multigrid is ITERATIVE (several V-cycles x GS sweeps). For PERIODIC BCs,
`PoissonFFTSolver` (`numerics/elliptic/poisson_fft_solver.hpp`) is a DIRECT solver
(one transform), wrapping `PoissonFFT` at the MultiFab level and modeling the
`EllipticSolver` concept. The `Coupler` is generic over the backend:
`Coupler<Model, PoissonFFTSolver>` instead of `Coupler<Model>` (= MG).

Euler-Poisson coupled step, N=256 (M1 Pro, -O3, PerStage):

| elliptic backend | ms/step |
| --- | --- |
| GeometricMG (iterative) | 76 |
| **PoissonFFTSolver (direct)** | **16** |

That is **~4.8x** on the coupled step, at **bit-identical** physics: both invert
the SAME discrete 5-point Laplacian (cf. `test_elliptic_operator`: MG vs FFT, residuals
at the rounding `~1e-14`, solutions identical to `~1e-16`). This is the high-impact
run-time optimization on the 86% elliptic. Limit: periodic FFT, single-rank (the
MPI band-distributed one is `DistributedFFTSolver`, wraps `SpectralCoupler`). Stackable with
OncePerStep.

## `bench_amr` bench: AP two-fluid + multi-patch AMR coupler

`examples/bench_amr.cpp`, timed without I/O (M1 Pro, 8 cores = 6 perf + 2 efficiency,
Release -O3 -DNDEBUG, Kokkos OpenMP space). Run: `OMP_NUM_THREADS=k ./build-omp/bin/bench_amr n nsteps`.

**AP two-fluid single-grid** (2 species Rusanov + continuity + Poisson multigrid).
The OpenMP scaling DEPENDS ON THE SIZE:

Clean sweep (`scripts/plot_bench_scaling.py`, figure `docs/fig_openmp_scaling.png`):

| grid | 1 thread | 2 threads | 4 threads | 8 threads |
| --- | --- | --- | --- | --- |
| n=512 | 11.6 M cells/s | 19.8 M | **28.3 M (x2.4)** | 26.4 M (plateau) |
| n=768 | 6.4 M cells/s | 10.5 M | **15.7 M (x2.5)** | 15.6 M (plateau) |

![OpenMP scaling](fig_openmp_scaling.png)

At a VERY small grid (n=384, off-figure) we are overhead-bound: too many small kernels
(`tfap_mstar`, 2x `div_update`, `efield`, 2x `lorentz`) + coarse multigrid levels,
the fork/join per `for_each_cell` costs more than the work -> OpenMP loses. From n>=512 the
per-kernel grain amortizes the overhead: **~x2.4-2.5 from 4 threads** (an isolated run even
gave x3.6; thermal/scheduling variability), then a **clear plateau** beyond (and a
slight drop at 8). This plateau is NOT a core ceiling: the M1 Pro has **6 performance
cores**. It is the **saturation of the memory bandwidth**: the stencil + the
multigrid have low arithmetic intensity (bandwidth-bound), ~4 threads are enough to
saturate the memory bus, and adding cores (perf or efficiency) brings nothing more.
Mass conserved at `~3e-7` (CFL `dt = 0.4 dx`).

**FFT vs multigrid elliptic for the AP two-fluid: MG WINS (counter-intuitive).**
n=512, 60 steps, OMP=4: MG **9.66 ms/step**, FFT **23.1 ms/step** (FFT x0.42, that is 2.4x
SLOWER), at bit-identical physics (`|dev_MG - dev_FFT| = 6.7e-16`). This is the OPPOSITE of the
Euler-Poisson coupled step (where the FFT wins x4.8). Measured reason: the Euler-Poisson step is
Poisson-dominated (86%) and solves 2x/step (PerStage); the AP two-fluid step is
TRANSPORT-dominated (8 kernels: 2x mstar, 2x div, efield, 2x lorentz) and solves the Poisson
only 1x/step WITH warm-start (the multigrid starts from the phi of the previous step -> 1-2 V-cycles
are enough, cheaper than a pair of full FFT transforms). Conclusion: keep
`GeometricMG` by default for the two-fluid; the FFT advantage is specific to Poisson-dominated
and per-stage couplings. Lesson: measure, do not extrapolate from one solver to the other.

**Multi-patch AMR coupler** (`AmrCouplerMP`, n=256 + 1 fine level, Berger-Rigoutsos regrid):
~100 ms/step, 8 patches, **mass conserved at 6.4e-15 (machine rounding)**. Weak OpenMP gain
(6.5 -> 5.9 s): dominated by the coarse Poisson MG + the multi-box host reconciliation
(`mf_find_box`, `mf_average_down_mb` in a serial loop). Mass is conserved at machine
rounding under dynamic regrid.

The per-kernel dispatch helps the TRANSPORT at large grid (x3.6) but not the
MG-dominated loads (coupled, small): the right grain remains to parallelize above the level
loop.
