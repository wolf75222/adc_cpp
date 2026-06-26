# Design: distributed AMR hero-run (objective B)

Goal: run a dynamic-AMR diocotron at very large scale on ROMEO
(GH200, MPI + Kokkos/CUDA), to reach the analytic rate `0.911` at mode 4
with far fewer cells than the uniform run (M2b quantified the gain: same physics
for ~43 % of the cells at equal effective resolution). This document is the PLAN;
no code is written here.

Validation convention (inherited from the repo, non-negotiable): each step is
verified BIT-FOR-BIT identical `np=1/2/4` against a reference (rank 0 / replicated
state), mass conserved to roundoff, and the GPU port bit-identical to the CPU.

## Current distributed state (what ALREADY WORKS)

- **Distributed multi-patch reflux, 2 and N levels**: `amr_step_2level_multipatch`
  and `subcycle_level_mp` / `amr_step_multilevel_multipatch` actually run
  distributed (`test_mpi_amr_multipatch`, `test_mpi_amr_multipatch3`, `np=1/2/4`
  bit-identical). Level 0 replicated, levels > 0 distributed; `average_down` and
  reflux via global coarse buffer + `all_reduce_sum_inplace`; register
  RESTRICTED to the coarse-fine interface (`O(interface)`, not `O(NX*NY)`).
  (`include/pops/integrator/amr_reflux_mf.hpp`.)
- **Distributed AMR coupler**: `AmrCouplerMP` orchestrates everything; its aux injection
  `coupler_inject_aux_mb` goes through `parallel_copy` when the parent is distributed
  (`test_mpi_coupler_inject`). (`include/pops/coupling/amr/amr_coupler_mp.hpp`.)
- **SFC balancing**: `make_sfc_distribution` (Morton Z-order) WIRED IN and
  verified under distributed AMR (`maxdiff = 0` vs rank 0, `np=1/2/4`).
- **Execution seam**: `for_each_cell` picks ONE backend at compile time
  (serial / OpenMP / Kokkos). Under Kokkos the kernels launch on the GPU.
  (`include/pops/mesh/for_each.hpp`.)
- **Uniform GPU**: `examples/gpu/coupled_kokkos.cpp` does the FULL Euler-Poisson step
  on GPU (uniform, single-level), bit-identical to CPU.
- **Geometric MG**: `GeometricMG` applies stencils PER FAB, restriction and
  prolongation by box kernels, without level-by-level gather: structurally
  ready for a distributed multi-box coarse, BUT never exercised as such (the
  diocotron keeps the coarse replicated). (`include/pops/elliptic/geometric_mg.hpp`.)

## The three blockers for the hero-run

1. **Replicated coarse** (level 0 on every rank, `amr_coupler_mp.hpp:137`):
   memory `O(NX*NY*nranks)`. Tolerable for a moderate coarse on a few GH200,
   IMPOSSIBLE at high coarse or on the whole machine.
2. **Regrid of distributed levels (gather-tags)**: tagging is local per rank,
   the Berger-Rigoutsos clustering runs on a LOCAL `TagBox`; gathering the
   distributed tags before clustering is NOT implemented (`include/pops/amr/tag_box.hpp:11`).
   `comm.hpp` has no `gather`/`broadcast`, only `all_reduce`.
3. **GPU + AMR never combined**: the GPU only runs uniform. `parallel_copy`,
   the reflux gather, and the regrid (tagging + host clustering) have host-side
   logic to audit for the GPU (device memory, CUDA-aware MPI, device <-> host
   copies at the communication and clustering points).

## The insight that orders the plan

**A 2-LEVEL diocotron (replicated coarse + one distributed fine level) touches
NEITHER of the two hard blockers.** The regrid tags level 0, which is REPLICATED:
every rank sees the same tags, so the local clustering is consistent across ranks
without a gather. The gather-tags only bites at 3+ levels where one tags an
INTERMEDIATE distributed level. And coarse de-replication is required only when the
coarse becomes too large to be replicated.

So we can deliver a useful distributed AMR hero-run BEFORE touching
de-replication or gather-tags, building on the already-proven distributed reflux.
The risk is pushed to the end of the plan, when it becomes necessary.

## Staged plan

### Step 0: end-to-end distributed AMR diocotron (CPU), replicated coarse (DONE)

`test_mpi_diocotron_amr`: the 2-level diocotron on `AmrCouplerMP` (coarse Poisson
+ aux injection + multi-patch step + Berger-Rigoutsos regrid) runs distributed,
`np=1/2/4` BIT-FOR-BIT identical (`max|Uc_dist - Uc_ref| = 0`) and mass conserved
to roundoff (`2e-15`) under distributed dynamic regrid.

GAP FOUND AND FIXED along the way: the coupler's Poisson multigrid was NOT
replicated under MPI. `GeometricMG` distributed its single-box coarse in
round-robin (`DistributionMapping(1, n_ranks())` -> box 0 on rank 0 only),
while `compute_aux` reads `mg_.phi().fab(0)` and injects with
`replicated_parent=true` on EVERY rank: under MPI, ranks > 0 had no
coarse to read. No MPI test exercised the full `AmrCouplerMP.step()` (they
tested the bare reflux and the injection primitive alone), hence the hole. Fix:
`replicated` option on `GeometricMG` (each rank holds + solves the same coarse
Poisson, without communication: per-fab V-cycle, local `fill_boundary` on a box
covering the domain, residual via `norm_inf` = idempotent `all_reduce_MAX`); option
`replicated_coarse=true` on `AmrCouplerMP` that passes it through. In serial it is
bit-identical to round-robin (60/60 unchanged).

- **Deliverable**: `AmrCouplerMP` genuinely MPI-correct end to end (gate above).
- **Real risk**: this was NOT pure assembly: a real Poisson replication hole
  was revealed and fixed. Hence the value of the verification.

### Step 1: port step 0 to GPU (Kokkos), replicated coarse

Run that same driver under Kokkos/CUDA. Audit the host/device points of
distributed AMR: (a) `parallel_copy` between device fabs of different ranks
(CUDA-aware MPI, `openmpi +cuda`); (b) the reflux gather (`all_reduce` on a
buffer: device -> host -> all_reduce -> host -> device copy, with `device_fence`);
(c) the regrid (tagging reads device data; choice: device tag kernel +
host clustering on a copied-back `TagBox`, the clustering being cheap).

VALIDATED ON GH200 (DONE). `examples/gpu/diocotron_amr_kokkos.cpp` runs the whole
AMR step (coupler: coarse Poisson + aux injection + coverage-aware multi-patch
reflux + Berger-Rigoutsos regrid) UNDER THE KOKKOS SEAM. Launched on a GH200
of `armgpu` via `romeo/diocotron_amr_gpu.sbatch`: `Cuda` execution space, the AMR
step runs on the GPU and the result is BIT-FOR-BIT IDENTICAL to the CPU
(`checksum = 4394594.404318` identical to the serial and local Kokkos-OpenMP path;
4 patches; `derive_masse ~ 2.9e-15`, the `2.2e-15` gap to the CPU coming from the
mass sum reassociated by CUDA, not from the state). The uniform reference `coupled_kokkos`
also runs in `exec=Cuda`, `dmasse = 0`. The "GPU + AMR never combined" blocker
is thus lifted: the full AMR step runs on GH200, bit-identical to the CPU.

ROMEO note: Kokkos is NOT provided by spack/module (the `spack load
kokkos` recipe was wrong); the `.sbatch` compiles it from source (Serial + CUDA,
Hopper sm_90, nvcc_wrapper), once, cached on `/scratch_p`. Account: `r250127`.

MULTI-GPU VALIDATED (DONE). `examples/gpu/diocotron_amr_mpi_kokkos.cpp` launches the
distributed AMR diocotron on 4 GH200 of an `armgpu` node (1 MPI rank per GPU), via
`romeo/diocotron_amr_mpi_gpu.sbatch` (MPI + Kokkos/CUDA build, module
`openmpi/aarch64/4.1.7-cuda`). The `parallel_copy` and the reflux `all_reduce`
move data between DEVICE fabs of different ranks (CUDA-aware MPI). Distribution-invariance
gate: `exec=Cuda, np=4, max|Uc_dist - Uc_ref| = 0.000e+00`,
BIT-FOR-BIT identical between round-robin patches (DIST) and all-on-rank-0 (REF). The
distributed AMR step thus genuinely runs multi-GPU and the result does not depend on the
patch distribution.

- **Deliverable**: distributed AMR step validated bit-identical on 1 AND 4 real GH200. The
  "GPU + AMR never combined" blocker is fully lifted (single AND multi-GPU).
- **Remaining (perf, not correctness)**: optimal GPU binding, multi-node (Infiniband),
  and scale. The multi-GPU CORRECTNESS is secured.

PERFORMANCE (HONEST observation, measured). `concurrency()` confirms REAL parallelism:
`270336` threads on the GH200 (Cuda space), `1`/`8` on CPU OpenMP depending on
`OMP_NUM_THREADS` (so NOT disguised single-threading). BUT parallel != fast here: the
AMR step (80 steps) takes `29 s` at nc=64 and `174 s` at nc=1024 on GPU, against `1.85 s`
(nc=64) on ONE CPU thread. The GPU is thus slower than 1 CPU core at these sizes, and
the CPU OpenMP is itself slower at 8 threads than at 1 (`210 s` vs `1.85 s` at nc=64).
Cause: the AMR step launches a myriad of SMALL kernels (each V-cycle = dozens of
`gs_rb_sweep` x `fill_ghosts` x `device_fence`) and the Berger-Rigoutsos regrid is
HOST-SIDE (serial); at these sizes the launch latency + device<->host syncs dominate
the compute. The code is thus CORRECT in parallel (bit-identical, 270k threads)
but NOT optimized. Speedup would require kernel fusion, a GPU-resident regrid,
fewer fences: a PERF work item distinct from correctness, out of scope of the
hero-run AMR (which aims first at distributed correctness and memory scale).

### Step 2: coarse de-replication (objective B proper)

Required only when the coarse must grow beyond what replication allows.
State: the 2a/2b core WORKS, proven bit-identical np=1/2/4 (desync bug of the
a `parallel_copy` bug at np=4 and the gather-tags 2c.

- **2a. Distributed multi-box coarse (DONE, np=1/2/4).** `AmrCouplerMP` accepts a distributed
  multi-box coarse (`replicated_coarse=false`): `compute_aux` loops over the LOCAL coarse
  fabs (instead of `fab(0)`), `mass()` does an `all_reduce` when the coarse is
  distributed (local sum otherwise), `max_drift_speed()` an `all_reduce_max`, and
  the aux injection goes through `parallel_copy` (`replicated_parent=false` path). Everything
  stays BIT-FOR-BIT identical to the replicated path (serial 60/60, replicated AMR np=1/2/4
  `maxdiff=0`). `test_mpi_decoarse` proves that a distributed 2x2 multi-box coarse gives the
  SAME coarse as the replicated single-box, bit for bit, at np=1 and np=2.
- **2b. Poisson MG on distributed coarse (ALREADY THERE).** `GeometricMG` distributed multi-box
  solver works as is: `gs_rb_sweep` calls `fill_ghosts` -> `fill_boundary`
  (DISTRIBUTED inter-box halo exchange) between red/black sweeps, and the red-black GS
  is independent of the decomposition -> `phi` bit-identical to single-box. No separate bottom
  solver needed here (the bottom smoother also does `fill_ghosts`).
- **BUG FOUND AND FIXED (np=4).** De-replication is now bit-identical np=1/2/4
  (`test_mpi_decoarse` run at np=4, `maxdiff=0`). Root cause, found by dumping the per-rank
  `fill_boundary` call sequences: `GeometricMG::current_residual()` returned the `norm_inf`
  of the residual WITHOUT reducing it across ranks (LOCAL max). On a distributed multi-box
  coarse, each rank thus saw a different residual, and the V-cycle stopping criterion
  fired at a different iteration depending on the rank -> DIFFERENT number of V-cycles (so of
  `fill_boundary` calls) between ranks -> the tag-0 message streams desynchronized
  (one rank's aux nc=3 exchange paired with another's coarse nc=1 exchange) ->
  `MPI_ERR_TRUNCATE`. (The REPLICATED coarse worked because local max = global max.) The comment
  on `norm_inf` even noted the all-reduce as "later, not added here". FIX:
  `all_reduce_max` on the residual in `current_residual()` (one line) -> all ranks
  agree on the residual -> same number of V-cycles -> synchronized `fill_boundary`
  sequences. Idempotent under replication and identity in serial: serial 60/60 and the 13
  MPI tests stay green, bit-identical to the historical behavior.
- **SECOND BUG FOUND AND FIXED (np=4).** Distinct from the residual desync: a segfault
  as soon as a rank holds SEVERAL coarse boxes, or a fine footprint borders a DISTANT
  coarse box. Root cause: the multi-patch reflux `subcycle_level_mp` hard-coded
  `replicated_parent = (lev == 0)`, so at level 0 it sampled the bordering coarse flux
  via `mf_find_box`. Under de-replication level 0 is DISTRIBUTED: a bordering coarse
  cell can belong to a DISTANT rank, `mf_find_box` then returned -1 then
  `fx.fab(-1)` -> segfault (faulty address 0x40/0x80, null+offset). The 4-box case (1 per
  rank at np=4) survived by fortuitous round-robin alignment (each fine footprint fell
  into the coarse box of the same rank). FIX: we propagate `replicated_coarse` from the coupler
  down to `subcycle_level_mp`, `replicated_parent = (lev == 0) && coarse_replicated`. The
  de-replicated level 0 then takes the `parallel_copy` path (coarse footprint per
  child) already used for the fine levels, MPI-correct. `test_mpi_decoarse` regression:
  a CENTERED fine patch overlapping the 4 coarse boxes (3 distant at np=4) reproduces
  exactly the old segfault and now passes bit-identical (`maxdiff=0`, np=1/2/4).
  Idempotent under replication: serial 60/60 and MPI 73/73 stay green. NB: a 4x4 cut
  (16 boxes) degenerates the bottom of the geometric multigrid (16 boxes do not tile the
  coarsest 2x2 grid) and converges to a distinct point within tolerance, non-deterministic;
  the distant box is therefore tested by the patch centered on a clean 2x2 coarse, not by 16 boxes.
- **2c. Gather-tags for the regrid of a distributed level (REMAINING).** Add to `comm.hpp` a
  `gather` (or an `all_reduce` of the globally-indexed `TagBox`), gather the distributed tags
  before the Berger-Rigoutsos clustering (cf. `tag_box.hpp:11`). Not needed at 2 levels
  (level 0, even distributed, is tagged globally via a reduction) but required to
  refine a distributed intermediate level.

- **Risk**: the core (multi-box coarse + distributed MG) is secured bit-identical
  np=1/2/4. Remaining is 2c (gather-tags) to refine a distributed intermediate level.

### Step 3: production run + science

`romeo/diocotron_amr_hero.sbatch` (armgpu, MPI + Kokkos/CUDA), scaling up the base,
growth-rate measurement via `validate_diocotron_growth.py`, cells/time comparison
to the uniform hero-run (`diocotron_mpi`, the already-quantified reference). Scientific
target: `0.911` at mode 4, for ~43 % of the cells of the
equivalent uniform run (the M2b promise, at scale).

- **Gate**: rate converged toward `0.911`, mass conserved over the whole run, strong
  and weak scaling plotted.

- **Real results (GH200, ROMEO `armgpu`).** The pipeline runs end to end on H100:
  `diocotron_column_amr` (made Kokkos-compatible, `Kokkos::ScopeGuard` init) builds under
  Kokkos/CUDA (`nvcc_wrapper`, sm_90) and runs on the GPU. EXACT reproduction of the
  M2b-conv table on real hardware, rate extracted by `validate_diocotron_growth.py` (rhobar=0.9,
  omega_D=0.143):

  | case (1 GPU GH200 H100) | cells | gamma_norm | M2b table |
  |---|---|---|---|
  | uniform eff 448 (nc=448) | 200 704 | 0.577 | 0.577 |
  | AMR `ml` eff 448 (nc=224) | 82 808  | 0.592 | 0.592 |

- **Barrier toward `0.911` LIFTED: the multigrid diverged at the embedded boundary.** Beyond eff 448
  the sim went to `nan` from the first steps, in BOTH schemes (uniform and AMR `ml`). Full
  diagnosis, by instrumenting the MG residual and the localized `vmax`:
  - It is NEITHER the time step (already capped at `dt0`, the ring's E x B drift being
    ~constant), NOR the density floor: during divergence the density STAYS bounded in
    `[1e-3, 1]`, only `phi` blows up. The `vmax` started from the CONDUCTOR BOUNDARY (`r = 0.398 = Rwall`), not
    from the ring boundary.
  - REAL cause: the geometric V-cycle DIVERGES near the conducting wall on the fine grid. The
    coarsening is NON-Galerkin and the circle mask is re-evaluated at each level, so the
    coarse correction becomes inconsistent with the fine boundary; the `nu1=nu2=2` smoothing no longer
    dominates it and the cycle's spectral radius goes > 1. It is ERRATIC in resolution (depending on
    the circle's alignment on the hierarchy: eff 640/1280/2048 diverge, 512/896 only stagnate). The warm
    start propagates the divergence from one step to the next -> `phi` then the field to `nan`. Measurement: at
    nc=640 the MG residual rises (`ratio = r_fin / r_0 = 2.7e2`) where nc=224 converges (`5.7e-9`).
  - Fix: `GeometricMG::solve_robust` (`include/pops/elliptic/geometric_mg.hpp`). It runs the
    standard V-cycle, EXACTLY like `solve()` (so BIT-IDENTICAL when it converges or stagnates);
    ONLY if the final residual EXCEEDS the initial residual (true divergence, not a stagnation
    `ratio < 1` which we keep as is) does it harden the GS smoothing in a STICKY way (nu doubled,
    kept for the following steps) and restart cold (`phi = 0`) until it becomes contractant again.
    More smoothing makes the cycle contractant (the GS dominates the inconsistent coarse correction:
    `nu = 2` diverges at nc=640, `nu >= 4` converges). The example calls `solve_robust` for both
    Poisson (`mg` coarse and `fmg` multi-level).
  - **Verification.** The 8 recorded runs of the M2b-conv table (uniform eff 192-448, AMR `ml`
    eff 192-448) stay BIT-FOR-BIT identical (`diff` of the `ring_amp.csv`), because none diverged:
    `solve_robust` never triggers the hardening there. The elliptic suite stays green
    (`test_geometric_mg`, `test_poisson_convergence` order 2.00, `test_elliptic_operator` MG=FFT,
    `test_gauss_law` order 2.00). And the sweep now rises without `nan` up to eff 1024
    (uniform AND AMR `ml`, including the AMR `ml` base 320 case at 66 patches; mass conserved `~1e-14`).
  - **Science unblocked.** The M2b-conv table is extended up to eff 1024 (see `docs/ROADMAP.md`,
    M2b-conv-HR). In the linear phase (robust measurement `--window 5,14`) the rate CONTINUES its
    monotonic rise toward `0.911`: eff 448 -> 1024 gives `0.63 -> 0.65 -> 0.67 -> 0.70 -> 0.71` (uniform
    as AMR), the AMR `ml` following the uniform at ~40 % of the cells. The historical measurement relative
    to the peak, by contrast, plateaus at ~0.58 (window bias: the saturation rollover steepens with the
    resolution). Reaching `0.911` remains a matter of even higher resolution, now
    accessible without numerical blockage (the object of the distributed hero-run).
  - **Decisive lever toward `0.911`: second-order reconstruction.** The transport ran in `NoSlope`
    (first order, the most diffusive): IT is what capped the rate, not the resolution. Switched to
    MUSCL `VanLeer` (option `recon=1`, 2 ghosts), the rate jumps to `γ_norm ~ 0.86` from eff 256 (clean
    exp. window `--window 4,11`), ~95 % of the analytic `0.911`, against `0.56` in `NoSlope`; and it is
    nearly FLAT in resolution (already converged in reconstruction). Stable and conservative (`~1.9e-14`),
    uniform AND AMR `ml`. `recon=0` stays bit-identical to the recorded runs. Consequence for the
    hero-run: aiming for `0.911` does NOT necessarily require the extreme resolution, but second order (and a
    second-order time integrator for the last %); the distributed hero-run stays useful for the
    resolution margin and the demonstration at scale. Details in `docs/ROADMAP.md` (M2b-recon).

## Open decisions (to settle before coding)

1. **How far do we go?** Steps 0-1 (distributed GPU AMR with replicated coarse)
   already give an AMR hero-run that beats the uniform run at a moderate base. Step 2
   (de-replication) is needed only for a high coarse / the whole machine.
   We can stop after 1 and measure, or push through to 2.
2. **2 levels or more?** At 2 levels, the gather-tags (2c) is useless (level
   0 replicated carries the tags). Staying at 2 levels simplifies strongly; going to
   3+ distributed levels imposes 2c.
3. **Drift limit (Diocotron) or full system (M3) at the hero-run?** The
   Diocotron goes through `AmrCouplerMP` (aux via the flux); the full magnetic case
   (`MagneticEulerPoissonCoupler`) does NOT yet have a multi-patch AMR variant (it
   wraps `Coupler<EulerPoisson>`, single-level). Doing the magnetic AMR requires
   wrapping `AmrCouplerMP<EulerPoisson>` in the Strang splitting: feasible
   but it is a work item in itself, to place after step 1.

## Recommended path

Step 0 -> 1 first (low/medium risk, delivers a real GPU AMR hero-run with
replicated coarse, 2 levels, drift limit), measure the gain vs the uniform run at
scale; then decide on step 2 (de-replication) according to the reachable base and
on the magnetic AMR according to time. The hard risk (distributed bottom solver,
gather-tags) is engaged only if it becomes necessary, and each step stays
verifiable to the repo's bit-identical standard.
