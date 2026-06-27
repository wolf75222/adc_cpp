> **STATUS: closed.** This is a closed TODO from the performance-scaling work; its items are done or tracked in Linear. Read it as a historical record, not as an active plan (class D, see `DOC_QUALITY.md`).

# TODO perf scaling and frontends

This TODO accompanies `docs/PERF_SCALING_FRONTENDS_AUDIT_2026-06-08.md`.
It is ordered to avoid launching a large campaign before having
comparable measurements and clean phases.

Closure status evening of 2026-06-08: all the campaign actions listed
below have been either completed, or replaced by a cleaner measurement,
or requalified as a documented limit/blocker in the report. A checked box
therefore does not always mean "satisfactory performance"; it means
"nothing left to do in this audit TODO without opening a dedicated code work item".

## T0 - Freeze and instrument

- [x] Synchronize the measurement checkout on explicit commits:
  `adc_cpp=075255b`, `adc_cases=6483e37` for the first `master` campaign.
- [x] Create a results directory out of source or ignored:
  `bench/results/`.
- [x] Define the common CSV format:
  `perf_scaling`, `perf_frontends`, `perf_phases`.
- [x] Add to each bench output: commit, backend, compiler,
  Kokkos, MPI, machine, case, `np`, threads, `n`, steps, warmup.
- [x] Check that the GPU timers bracket the measured zones with
  `Kokkos::fence()` or equivalent.
- [x] Parse the `bench/profile_step` outputs to a phase CSV, but mark this
  case as `profile-exb-systemlike`, not as a neutral Euler benchmark.
- [x] Parse `amrmpi_integrated` to a synthetic AMR scaling CSV.

## T1 - Safe benchmarks, outside Hoffart

- [x] Add or standardize a `frontend-euler-periodic` case:
  smooth 2D Euler, periodic, no disk, no Schur.
- [x] Use a fixed `dt` to compare the frontends, so that `step_cfl` does not
  mask a reduction cost or a CFL difference.
- [x] Measure pure transport without Poisson.
- [x] Measure controlled Euler-Poisson with periodic Poisson or MMS, separate from
  pure transport.
- [x] Measure ghosts/halos alone via the MPI fill-boundary harness.
- [x] Measure reductions alone or in a dedicated phase.
- [x] Use `amrmpi_integrated` as the primary synthetic AMR case.
- [x] Do not launch any main perf benchmark on Hoffart.

## T2 - CPU, GPU, MPI scaling

- [x] CPU Kokkos OpenMP strong scaling:
  threads `1,2,4,8,16,...`, fixed global size.
- [x] CPU Kokkos OpenMP weak scaling:
  fixed local size per thread, growing global size.
- [x] MPI + Kokkos CPU strong scaling:
  `ranks x threads_per_rank` matrix, without Kokkos Serial as a perf target.
- [x] MPI + Kokkos CPU weak scaling:
  fixed local size per rank or per rank*thread.
- [x] GPU Kokkos Cuda single-rank:
  `np=1` reference.
- [x] MPI + Kokkos Cuda strong scaling:
  `np=1,2,4,...`, one GPU per rank, max time over ranks.
- [x] MPI + Kokkos Cuda weak scaling:
  fixed local size per GPU.
- [x] Synthetic AMR:
  compare replicated coarse vs distributed, and publish the negative result if
  the MG/MPI latency dominates.

## T3 - Python frontends

- [x] Implement a `python-bricks` Python driver:
  `pops.Model(FluidState, CompressibleFlux, NoSource, ...)`.
- [x] Implement a `python-dsl-production` Python driver:
  `pops.physics.facade.Model(...).compile(backend="production")`.
- [x] Check explicitly:
  `compiled.backend == "production"` and `compiled.adder == "add_native_block"`.
- [x] Measure `T_import`.
- [x] Measure `T_setup` (`System`, `add_equation`, `set_state`).
- [x] Measure `T_compile_dsl` cold with empty cache.
- [x] Measure `T_compile_dsl` warm with cache hit.
- [x] Measure `advance(dt, nsteps)`.
- [x] Measure a Python loop `for _ in range(nsteps): step(dt)`.
- [x] Measure `extract_final` (`get_state` or `density`) separately.
- [x] Add the `aot` counter-example if the local build supports it.
- [x] Add the `python/pops/integrate.py` counter-example to quantify the
  misuse with full-array copies per stage.
- [x] Forbid any diagnostic in the main hot loop.

## T4 - Analysis, graphs, optimizations

- [x] Generate `strong_scaling_speedup.png`.
- [x] Generate `strong_scaling_efficiency.png`.
- [x] Generate `weak_scaling_efficiency.png`.
- [x] Generate `phase_breakdown_stacked.png`.
- [x] Generate `frontend_ratios.png`.
- [x] Generate `dsl_cold_warm.png`.
- [x] Generate `diagnostics_io_impact.png`.
- [x] Use `bench/plot_perf_campaign.py` to regenerate the PNGs from the
  official CSVs.
- [x] If `python-bricks / cpp-native > 1.05` in the hot loop, isolate:
  pybind `step`, diagnostics, allocation, setup/extraction copies.
- [x] If `dsl-production-warm / cpp-native > 1.05`, check that the path is
  indeed `add_native_block` and not `aot/prototype`.
- [x] If the GPU is slower than CPU on Poisson, profile MG:
  small kernels, fences, red-black smoother, bottom solve, halos.
- [x] If MPI+GPU does not scale, separate:
  halos, reductions, multi-box MG, regrid, host diagnostics.
- [x] Document each proposed optimization with a quantified before/after.

## Definition of done

- [x] The results are tied to exact commits.
- [x] The frontends are compared on the same case, same `dt`, same `nsteps`.
- [x] The curves are generated from CSV, not by hand.
- [x] The Python costs are separated into setup, compile, boundary pybind,
  extraction, diagnostics/I/O.
- [x] The report explicitly states whether a measurement is absent instead of
  inferring it.
- [x] No quantitative conclusion uses Hoffart as a perf benchmark.

## ROMEO update 2026-06-08

First campaign completed:

- [x] `adc_cpp=1f9fb4a`, `adc_cases=b8bccbe` frozen for jobs
  `647780` and `647781`.
- [x] CPU `x64cpu`: Kokkos OpenMP and MPI+Kokkos OpenMP measured.
- [x] GPU `armgpu`: Kokkos CUDA and MPI+Kokkos CUDA measured.
- [x] Synthetic AMR `amrmpi_integrated` measured on `np=1,2,4`.
- [x] Graphs generated from CSV in
  `docs/perf_figures_647780_647781_647815/`.
- [x] Python bricks and DSL `production` measured via job `647815`.
- [x] Cause of the initial Python build identified:
  static non-PIC Kokkos OpenMP, fixed by
  `/home/rmdraux/pops_perf_20260608/kinstall_omp_pic`.

Corrections required before final publication:

- [x] Update the reference commit if the next campaign targets the
  new `origin/master` (`adde23b` observed during the transport catch-up).
- [x] Re-run the native C++, Python bricks and Python DSL frontends in the
  same job, on the same node, with the same Kokkos PIC, to avoid the
  inter-job ratios.
- [x] Add a pure FV transport benchmark without Poisson: the current runs
  mostly measure Poisson/MG (`>95 %` of the step).
- [x] Redo the CPU weak scaling with `n_global = n_local*sqrt(np)` in 2D, not
  the `n = 128*np` smoke-test.
- [x] Redo the GPU weak scaling with `n_global = n_local*sqrt(np)` in 2D, not
  the `n = 128*np` smoke-test.
- [x] Investigate the DSL `production` warm: hot loop around `339 ms` and little
  sensitive to threads on the current harness.
- [x] Add a native C++ measurement with Kokkos PIC to compare exactly to the
  Python `_pops` module.
- [x] Separate build and frontend measurement: the isolated job `647848` shows that
  compiling the `frontend_cpp` C++ harness can dominate the campaign time.
  The next re-runs must reuse an already produced PIC build, or
  put the build in a preparatory job.
- [x] Do not conclude that Python is "faster" than C++ from the first
  table: the C++ and Python rows do not come from the same job/node.

Catch-up completed:

- [x] ROMEO job `647836`, commit `adc_cpp=adde23b`:
  `bench/profile_transport_mbox.cpp` measures periodic 2D Euler, pure transport,
  distributed multi-box, without Poisson/Hoffart.
- [x] Strong OpenMP pure transport: `n=1024`, threads `1,2,4,8,16`.
- [x] Strong MPI+OpenMP pure transport: `n=1024`, ranks `1,2,4,8`,
  `threads=4`.
- [x] Weak MPI+OpenMP pure transport 2D: `n_global ~= 384*sqrt(np)`, ranks
  `1,2,4,8`, `threads=4`.
- [x] Phase diagnostic of the catch-up: the MPI negative comes mostly from
  `fill_boundary`, then the global reductions, not Poisson.
- [x] ROMEO job `647848`, commit `adc_cpp=adde23b`:
  native PIC C++, Python bricks and DSL `production` frontends measured in the
  same job/node/Kokkos PIC.
- [x] Caught-up frontend verdict: `python-bricks` shows no measurable hot-loop
  penalty; DSL `production` warm stays around `341 ms` and does not
  follow the thread scaling of the native path.
- [x] Final graphs regenerated in
  `docs/perf_figures_647780_647781_647815_647836_647848/`.

Follow-up evening of 2026-06-08:

- [x] External ROMEO jobs `647857` and `647858` recovered as results from a
  separate branch `feat/perf-campaign-bench`, commit `0162d5f4a8`.
- [x] JSONL converted into CSV compatible with the plotter:
  `bench/romeo_results_matrix_647857_647858/perf_scaling_matrix_0162d5f4a8_647857_647858.csv`
  and
  `bench/romeo_results_matrix_647857_647858/perf_phases_matrix_0162d5f4a8_647857_647858.csv`.
- [x] Branch graphs generated in
  `docs/perf_figures_matrix_647857_647858/`.
- [x] DSL `production` warm diagnostic: the native loader was zero-copy but
  could be compiled without `POPS_HAS_KOKKOS`, hence with serial fallback in the
  inline templates.
- [x] Local fix added: `kokkos/mpi` features in the ABI key, Kokkos/OpenMP
  flags for `compile_native()` and `HybridModel.compile()`, DSL cache key
  dependent on the native backend.
- [x] ROMEO validation of the DSL/Kokkos fix: job `648034`, checkout
  `adde23b` + local patch `dsl.py/abi_key.hpp`, without re-running the C++
  frontend build.
- [x] Add the before/after table `647848 -> 648034` and regenerate the
  frontend graphs in `docs/perf_figures_frontends_dslkokkosfix_648034/`.
- [x] Explain the Kokkos shutdown of the dynamic DSL loader path:
  `648034` writes the measurements, then exits with `rc=134` because the loader links a
  second copy of `libkokkos*`.
- [x] Integrate the upstream Claude/`origin/feat/dsl-production-optflags` fix
  into the diagnostic: do not link `libkokkos*`, keep a single Kokkos runtime
  via `_pops`, and compile the production `.so` in `-O3 -DNDEBUG`.
- [x] Re-run a short validation after full alignment with
  `origin/feat/dsl-production-optflags`: expect a clean `exit 0` output and
  a warm DSL ratio close to `1.02-1.04x`. Validation taken into account via
  `adc_cases origin/feat/perf-campaign-harness`, report `perf/RAPPORT.md`:
  clean output and threaded DSL ratio close to parity (`1.02x` at 8 threads).
- [x] MPI+CUDA pure transport weak/strong still to be re-run, but the deadlock
  blocker has an upstream fix: `origin/master=f3e1bf9`, fix `#254`
  `MPI halos in pinned host memory`. Re-run v2 completed on `1d4cd25e25`
  (merge `origin/master` into `feat/perf-campaign-bench`): single-rank OK,
  multi-rank CPU/GPU still in timeout.
- [x] Build a measurement base that combines recent `origin/master`
  (`#254`) and the `feat/perf-campaign-bench` harnesses, then re-run pure
  multi-rank transport with one GPU per rank. Correct base observed:
  `1d4cd25e25d244cd7c4f6cfd4c0eb815cd997790`; results in
  `bench/romeo_results_mpi_v2_648114_648115/`.

## Remaining limits outside TODO

- The multi-rank `scaling_step` stays blocked/timed-out after `#254` on jobs
  `648114` CPU and `648115` GPU. A separate code work item must be opened on
  `fill_boundary`/MPI progress/halo scheduling, instead of continuing to
  launch perf campaigns.
- The DSL `production` is considered fixed upstream by
  `feat/dsl-production-optflags`; our local patch documents and reuses the
  principle, but a clean integration must be done by merge/cherry-pick
  rather than by manual stacking on this audit worktree.
- The Hoffart comparisons remain explicitly outside the perf benchmark.
