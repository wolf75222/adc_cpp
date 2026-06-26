# Porting the runtime stack to GPU (roadmap)

The DSL and the compute core are verified up to the GH200 GPU (flux, generated brick, full Euler CASE
via the Kokkos seam `for_each_cell` ; cf. docs/GPU_ROMEO.md). What remains for an end-to-end GPU
PRODUCTION is porting the ENTIRE RUNTIME STACK (System / MultiFab / Poisson / AMR / MPI) onto device.
STATUS (June 2026) : the full SINGLE-GRID solver (transport + BCs + couplings + Poisson + time
step, orchestrated by the System) runs on GH200, verified == CPU (phases 1-5, 7) ; the AMR field ops
(reflux, transfers) run on device (phase 5) ; the MULTI-GPU halo exchange is validated
(phase 6, np=1/2/4 bit-identical) ; the AOT .so backend of a DSL-generated model runs on
device (phase 8, with host marshaling). The INTEGRATED VALIDATION AmrSystem + MPI + GPU (the three axes
together in ONE SINGLE run) is now DONE on GH200 (phase 10) : np=1/2/4 BIT-IDENTICAL (dmax=0)
and mass conserved to 0. What REMAINS : full-device perf (the integrated run does not scale, the coarse
being replicated, see phase 10) and zero-copy AOT parity on device. This document breaks it down into phases.

## Execution model : MPI + Kokkos (NO hand-written CUDA)

The architecture is **MPI + Kokkos**, not "three layers Kokkos+CUDA+MPI". MPI distributes the
subdomains across ranks (one GPU per rank) ; Kokkos parallelizes the LOCAL compute and abstracts the
hardware via its ExecutionSpace : **Cuda** backend for NVIDIA GPU, **Serial/OpenMP** for CPU. The SAME
source code (`for_each_cell`, `assemble_rhs`, the bricks) therefore targets CPU and GPU depending on the backend chosen
AT COMPILE TIME ; we write NO CUDA kernel by hand. In the outputs below, `exec=Cuda` is
simply the active Kokkos backend ; the same .cpp files switch to `exec=Serial`/`OpenMP` on CPU (that is what
the host-side MPI CI verifies). `nvcc_wrapper` is only the compiler required by the Kokkos Cuda backend.
The core is now **100% Kokkos** : the unified allocator uses
`Kokkos::kokkos_malloc<Kokkos::SharedSpace>` + `Kokkos::fence` (no longer `cudaMallocManaged` /
`cudaDeviceSynchronize`), and `POPS_HD` delegates to `KOKKOS_FUNCTION`. NO hand-written CUDA API
remains ; only a `__host__ __device__` fallback survives in the NON-Kokkos branch of `POPS_HD` (inert
in our case). `SharedSpace` being a portable alias (`CudaUVMSpace` / `HIPManagedSpace` /
`SYCLSharedUSMSpace` / `HostSpace`), the same core would also target AMD/Intel via the Kokkos backends.

## Design asset : the seam does not change the call sites

`adc/mesh/execution/for_each.hpp` (`for_each_cell`, `for_each_cell_reduce_*`) switches CPU <-> GPU at
COMPILE TIME without touching the operators. `POPS_HD` makes the whole core device-callable. So the GPU
port is mostly a job of data RESIDENCE on device + porting of the still-host steps,
not a rewrite of the compute kernels.

## Already device-ready (verified on GH200)

- `for_each_cell` / reductions -> `Kokkos::parallel_for` / `parallel_reduce` (`Cuda` space).
- `POPS_HD`, `StateVec`, `Aux` : device-callable ; flux + `eigenvalues` + bricks (Euler/iso/ExB).
- DSL-generated brick : compiles with nvcc `sm_90`, == `pops::Euler` to the bit (CUDA) / to 1 ULP (Kokkos).
- Full 2D Euler case (80 steps, CFL, Rusanov, periodic) on GH200 : mass conserved, == CPU.

## Runtime port phases (by increasing dependency)

1. **Device-resident MultiFab.** [x] DONE (verified GH200). Observation : `fab_allocator` under
   `POPS_HAS_KOKKOS + __CUDACC__` is `ManagedAllocator` (cudaMallocManaged) -> the Fabs are in UNIFIED
   MEMORY, hence already device-accessible, and `assemble_rhs` (via `for_each_cell` -> Kokkos) runs on
   the device by CONSTRUCTION. A FULL Euler transport (80 steps, fill_boundary + assemble_rhs +
   SSPRK/FE update on the REAL adc stack) gives on GH200 a result BIT-IDENTICAL to the CPU
   (`python/tests/gpu/phase1_transport.cpp`, mass 4096 / energy identical). Surfaced + fixed an nvcc
   bug in `numerics/spatial_operator.hpp` (capture of `dx`/`dy` in a `constexpr-if` context, forbidden
   for an extended `__host__ __device__` lambda). Remaining optimization : avoid the host round-trip
   of `fill_boundary` (phase 2) ; unified memory ensures correctness but not optimal perf.
2. **Boundary conditions on device.** [x] DONE (verified GH200). `copy_shifted` (PERIODIC
   ghosts, `mesh/fill_boundary.hpp`) was already `for_each` -> device. `fill_physical_bc`
   (Foextrap / Dirichlet, `mesh/physical_bc.hpp`) ports HOST loops to `for_each_cell` ;
   the internal `device_fence` removed (the BC kernels order after `copy_shifted`, and the y faces
   after the x faces, on the same stream). A NON-periodic transport (Foextrap outflow) on GH200 gives
   a result BIT-IDENTICAL to the CPU (`python/tests/gpu/phase2_transport.cpp` ; mass decreases as
   it should, free outflow). The core ctests (including test_physical_bc, poisson_disc, cut_cell) green.
3. **Poisson on device.** [x] DONE (verified GH200), and WITHOUT code modification. The whole
   V-cycle loop of `GeometricMG` was ALREADY `for_each` -> device : red-black GS smoother, residual, Laplacian
   (`poisson_operator.hpp`), restriction `average_down` + prolongation `interpolate` (`mesh/refinement.hpp`),
   norm via `for_each_cell_reduce_max` (`mf_arith.hpp`). Only the SETUP (mask + cut-cell coefs from
   `std::function`) remains host (one-shot, writes unified memory). A Dirichlet Poisson solve
   (n=128) on GH200 gives a result BIT-IDENTICAL to the CPU : cycles=9, sum/max(phi) identical
   (`python/tests/gpu/phase3_poisson.cpp`). The "big chunk" was already ported by the for_each seam.
   (The VARIABLE-COEFFICIENT solver eps(x) remains a separate numerical addition, not required here.)
4. **Inter-species couplings on device.** [x] DONE (verified GH200). The 3 couplings
   (ionization, collision, thermal exchange, `system.cpp`) ported from HOST loops to `for_each_cell`
   (device kernels reading/writing SEVERAL blocks at the same point) ; the prior `device_fence` of
   `apply_couplings` removed (kernels ordered after transport). The ionization kernel on GH200
   gives a result BIT-IDENTICAL to the CPU, n_i + n_g conserved (`python/tests/gpu/phase4_coupling.cpp`).
   Host : `test_bindings` (conservation of the 3 couplings) green. => milestone 1->2->4 reached : full
   MULTI-SPECIES TRANSPORT (transport + BCs + couplings) on GPU, without Poisson.
5. **AMR (field ops) on device.** [x] DONE (verified GH200), without code mod. The
   self-checking tests `test_flux_register` (flux register / 2-level reflux, conservation) and
   `test_amr_diffusion` (multi-level transport) PASS on GH200 : `average_down` /
   `interpolate` transfers + reflux + transport are `for_each` -> device. The CLUSTERING / the hierarchy
   (`cluster.hpp`, `regrid.hpp`) is host METADATA (box lists, `std::function` predicates) : stays
   host (correct, infrequent ; nvcc does not compile it, which is normal, it is not a kernel).
   The flux registers keep host loops (correct via unified memory ; full-device reflux =
   perf follow-up). python/tests/gpu/{amr_CMakeLists.txt, romeo_amr_build.sh}.
6. **Multi-GPU MPI (Kokkos Cuda backend + CUDA-aware OpenMPI).** [x] DONE (verified GH200). `fill_boundary`
   distributed (cross-rank halo exchange) runs on 1/2/4 GH200, one GPU per rank, with OpenMPI
   `+cuda` (UCX). The SAME `mpi6_fillboundary.cpp` (= `tests/test_mpi_fillboundary.cpp` + Kokkos init +
   `device_fence` before host read) gives `gfails=0` at np=1/2/4 (`exec=Cuda`) : remote ghosts
   bit-identical to the expected periodic value, hence device-to-device transfer correct. Unified
   memory + the `device_fence` suffice ; no modification of `fill_boundary`. `python/tests/gpu/mpi6_fillboundary.cpp`.
   REMAINS (perf) : device halos without host bounce (direct GPUDirect), distributed device FFT.
7. **End-to-end validation via the System.** [x] DONE (verified GH200), without code mod.
   The ENTIRE `system.cpp` (model dispatch, HLLC transport, gravity source, Poisson solve at EVERY
   step, CFL time step) compiles under nvcc, and the full `euler_poisson` case runs on GH200 : `max|phi|`
   and `sum(phi)` bit-identical to the CPU, mass at ~1.7e-15 relative (FMA in the CFL reduction). The
   phase 1/2 nvcc fixes + the for_each design suffice. `python/tests/gpu/phase7_system.cpp`.
   REMAINS for multi-GPU prod : phase 6 (CUDA-aware MPI) + perf (full-device reflux, avoiding the
   host syncs of the per-step reductions).
8. **AOT backend on device (DSL-generated model).** [x] DONE for the .so path with
   HOST MARSHALING (verified GH200). A `euler_poisson` model WRITTEN IN FORMULAS, compiled AOT
   (`compile_or_jit(mode="compile")`) into a .so via `compiled_block_abi.hpp` (`add_compiled_block`),
   runs the production path (`assemble_rhs<Minmod, HLLCFlux>` primitive recon + SSPRK2) on
   the GH200 : residual, mass, momentum and energy BIT-IDENTICAL to the host serial build
   (`sim_aot/`). Surfaced + fixed a real bug : `extract()` read unified memory BEFORE the end
   of the async kernel ; added a `device_fence()`. The flat-array marshaling crosses the
   dlopen without sharing a C++ object (clean ABI) ; this .so path carries neither AMR nor MPI.
   REMAINS (not done) : the NATIVE ZERO-COPY variant (`add_compiled_model` / `dsl_block.hpp`, model
   compiled in the same binary as the System, without marshaling) is validated BIT-IDENTICAL to
   `add_block` on CPU/Serial (`test_compiled_model_parity`), but on GPU (Cuda backend) the variant
   with EXTENDED LAMBDAS hit an nvcc limit : an extended `__host__ __device__` lambda instantiated
   in an EXTERNAL TU was not accepted. This limit has been lifted by the NAMED FUNCTOR path
   (cf. point 9).
9. **Multi-box MPI parity of the compiled path with named functors.** [x] DONE (verified GH200).
   The residual of the `make_block` closures (`block_builder.hpp` ; the exact machinery
   of `add_compiled_model`, instantiated from an EXTERNAL TRANSLATION UNIT via NAMED functors,
   the device-clean path that bypasses the nvcc limit of the extended lambdas of point 8) must be
   invariant to the splitting of the domain into boxes AND to the number of ranks. The test compares a
   16-box decomposition distributed by SFC to a single-box reference : `max|R|` BIT-IDENTICAL (`dmax = 0`), L2 to
   rounding of the summation order. Ported as a header-only regression test `tests/test_mpi_mbox_parity.cpp`
   (MPI CI job, np=1/2/4, Kokkos Serial on CPU) AND run on GH200 (Kokkos Cuda) : the SAME source,
   compiled by nvcc_wrapper and launched by `srun -n {1,2,4} --gpus-per-task=1` (OpenMPI 4.1.7 CUDA-aware,
   armgpu node), gives `dmax = 0.00e+00` at the three rank counts, `maxK`/`max1`/`L2` identical to the
   CPU run. A `Kokkos::fence()` (guarded by `POPS_HAS_KOKKOS`) precedes the HOST read of the residual (async
   `for_each_cell` kernels under Cuda). What this validates HONESTLY on device : the
   `make_block`/`add_compiled_model` path (named functors) + multi-box `fill_ghosts` intra-rank AND
   cross-rank MPI multi-GPU, for the residual of one step. What it does NOT validate : AMR integration
   in the same run, nor full-device perf (the test reads the residual host-side, hence a fence per step).
10. **INTEGRATED validation AmrSystem + MPI + GPU (the three axes in ONE SINGLE run).** [x] DONE (verified
   GH200). This was the last blocker : phases 5/6/9 validated the AMR, the multi-GPU MPI and the
   compiled path SEPARATELY, never together. A harness wires a `euler_poisson` model COMPILED via
   `add_compiled_model(AmrSystem, ...)` (`runtime/amr_dsl_block.hpp` path, PR #45) onto a real
   AMR hierarchy (`AmrSystem` : replicated coarse 128^2 + fine level 256^2 multi-patch tracked by
   Berger-Rigoutsos regrid, conservative reflux, coarse Poisson at every step), and DISTRIBUTES the
   fine patches across `n_ranks()` GH200 (one GPU per rank, cross-rank halos via `fill_boundary`, reflux
   and mass reduced by `all_reduce`). The SAME source runs in `srun -n {1,2,4} --gpus-per-task=1`
   (OpenMPI 4.1.7 CUDA-aware, armgpu node) under Kokkos Cuda (`exec=Cuda`), 4 fine patches, 40 steps
   after warmup. Result (`AMRMPI np=...`) :
   - **BIT-IDENTICAL to the rank count** : `mass`, `csum`, `csumsq`, `cmax` of the coarse density
     IDENTICAL to 17 digits at np=1 (single-GPU oracle), np=2 and np=4. `PARITE dmax = 0.00e+00`.
   - **coarse bit-identical cross-rank** : `crossrank_spread = 0.00e+00` (the replicated level 0 is
     the same field on each GPU, hence remote halos/reflux/injection correct).
   - **mass conserved to 0** : `dm = |mass - m0| = 0.00e+00` (exact conservative reflux on device).
   - perf : `per_step_ms` ~221 (np=1), ~266 (np=2), ~272 (np=4) on one GH200 node. The run DOES NOT
     SCALE : this is EXPECTED and HONEST. The coarse is REPLICATED (default `replicated_coarse=true`), so
     the coarse Poisson + the coarse transport are REDUNDANT on each GPU (compute O(NX*NY) x
     nranks, zero communication) ; only the fine patches get split (4 patches -> 2/GPU at np=2,
     1/GPU at np=4). At this size the coarse dominates -> adding GPUs does not accelerate, it just adds
     the cost of the cross-rank fine halos. The SCALABLE mode (`replicated_coarse=false`, distributed coarse)
     exists in `AmrCouplerMP` but degrades the geometric MG (cf. its comment) and is not
     wired in `AmrSystem` : that is the real remaining perf work item for AMR strong-scaling.
   A LATENT BUG was fixed along the way : `add_compiled_model(AmrSystem)` AND the native path
   `AmrSystem::build` built the single-box coarse in `DistributionMapping(1, n_ranks())`
   (round-robin) -> the box only lived on rank 0, and `coarse().fab(0)` segfaulted on the
   other ranks on the first write/inject/read under np>1. But `AmrCouplerMP` (and `GeometricMG`)
   expect a REPLICATED coarse (`DistributionMapping(vector<int>(ba.size(), my_rank()))`). The
   runtime AMR path had simply never been exercised under MPI. Fixed (replicated coarse) ;
   in serial `my_rank()=0` -> bit-for-bit identical to history. Ported as a regression test
   header-only `tests/test_mpi_amr_compiled_parity.cpp` (MPI CI job, np=1/2/4, Kokkos Serial on CPU,
   the same invariants : `crossrank_spread=0`, conservation, parity to the rank count) ; GPU harness
   `python/tests/gpu/{amrmpi_integrated.cpp, amrmpi_CMakeLists.txt, amrmpi_romeo_build.sh}`.

11. **AMR strong-scaling : distributed coarse wired into AmrSystem.** [!] DONE (wired + correct on
    GH200) but DOES NOT SCALE (NEGATIVE result, quantified, honest). The perf blocker of phase 10
    was that the REPLICATED coarse makes the coarse Poisson + transport redundant on each
    GPU. The SCALABLE mode (`replicated_coarse=false`, distributed multi-box coarse) existed in
    `AmrCouplerMP` but was not WIRED in `AmrSystem`. Wired here : `AmrSystemConfig::distribute_coarse`
    (+ `coarse_max_grid`) -> the two build paths (native `AmrSystem::Impl::build` and compiled
    `amr_dsl_block::build_amr_compiled`) build level 0 in `BoxArray::from_domain(dom, n/2)`
    (2x2) distributed round-robin and pass `replicated_coarse=false` to the coupler, the `GeometricMG` and
    `advance_amr`. Centralized coarse helpers (`detail::coupler_{make_coarse_layout,write_coarse,
    read_coarse,inject_coarse_to_fine_mb}`, amr_coupler_mp.hpp) multi-box + distribution-aware
    (read/write by GLOBAL boxes, reconstruction of `density()` n*n by `all_reduce_sum_inplace`
    over the disjoint boxes). The Berger-Rigoutsos regrid was made MPI-correct for a distributed
    coarse : `tag_cells` only sees the local boxes -> global OR of the tags (`all_reduce_or_inplace`,
    new collective) before the clustering, otherwise the fine BoxArray would differ per rank ; and the
    filling of the new patches from the parent goes through `parallel_copy` when the parent is
    distributed (instead of `mf_find_box`, which does not see the remote boxes).
    - **CORRECTNESS (host CI, Kokkos Serial) : distributed coarse == replicated BIT FOR BIT.** Regression
      test `tests/test_mpi_amr_distributed_coarse.cpp` (np=1/2/4) : same 4-bubble case
      euler_poisson, the distributed compared to the replicated oracle in the SAME binary ->
      `dist_vs_repl_dmax = 0.00e+00`, `cmax_crossrank_spread = 0`, mass conserved to ~1e-15. The MG
      CONVERGES on the 2x2 coarse (finite field, non-trivial, identical to single-box).
    - **MEASURED MG CONVERGENCE (single-box vs multi-box, MMS).** Local diagnostic (Dirichlet + off-center
      Gaussian pulse, criterion 1e-9) : 2x2 converges in AS MANY cycles as single-box (7-8 at
      n=64/128/256, identical residuals), 4x4 +0/+1 cycle, 8x8 degrades noticeably (~13-14 cycles,
      ~1.75x). SO the 2x2 (default, and the only useful split up to 4 ranks) does NOT degrade the
      multigrid ; the announced degeneracy only appears at aggressive splitting (>=8x8). That is
      why `coarse_max_grid` default = n/2.
    - **CORRECTNESS (GH200, Kokkos Cuda, srun -n 1/2/4 --gpus-per-task=1).** np=2 and np=4 distributed :
      `csum`/`csumsq`/`cmax` BIT-IDENTICAL to the replicated, mass conserved to 2.2e-16, `cmax` bit-identical
      cross-rank. A DEVICE BUG was found+fixed along the way : `parallel_copy` launches ASYNC copy
      kernels under Cuda and, at np=1, RETURNS WITHOUT a fence (the internal barrier is only on the
      np>1 path) ; the regrid parent filling then read `parloc` (fresh device memory)
      before the end of the copy -> NaN at np=1 distributed ONLY on Cuda (host Serial: `dmax=0`).
      Fixed by a `device_fence()` after this `parallel_copy` (amr_regrid_coupler.hpp).
    - **SCALING (per_step_ms, max over the ranks, n=128, 40 steps measured, 1 GH200 node), NEGATIVE :**

      | np | REPLICATED (default) | DISTRIBUTED (2x2) |
      |----|-------------------|---------------|
      | 1  | 222 ms            | 705 ms        |
      | 2  | 269 ms            | 999 ms        |
      | 4  | 278 ms            | 1403 ms       |

      Full run redone after the device-fence fix (np=1 distributed measured : no NaN, dm=2.2e-16,
      cmax bit-identical cross-rank at the 6 points). The distributed coarse is ~3.7x (np=2) to ~5.0x
      (np=4) SLOWER than the replicated, and WORSENS with the number of ranks (705 -> 999 -> 1403 ms).
      Strong-scaling is NOT reached. REASON, honest : at
      this size (coarse 128^2) the coarse compute is trivial, but the multi-box `GeometricMG`
      exchanges `fill_boundary` halos between coarse boxes at EVERY level of every V-cycle (~7
      levels), CROSS-RANK under MPI, and the AMR step adds `parallel_copy` (inject aux, reflux, regrid)
      device-to-device via UCX. This LATENCY traffic dominates by far the compute saved. Distributing
      an already small coarse TRADES cheap redundant compute for expensive communication.
    - **RECOMMENDATION (honest).** The wiring is CLEAN, correct and bit-identical to the replicated (mergeable as
      an OPTION, default unchanged), but AMR strong-scaling by distributed coarse is NOT worthwhile at
      this scale. Realistic paths for real scaling, not done here : (a) distributed coarse
      only when its MEMORY is the blocker (very large NX*NY), not its time ; (b) HYBRID MG :
      multigrid distributed on the fine levels of the coarse but bottom-solve on a GATHERED
      coarse (gather on 1 rank) instead of a multi-box cross-rank GS per level ; (c) reduce the
      traffic (GPUDirect halos, agglomeration of the `parallel_copy`). As it stands, KEEP the replicated by
      default (fast, bit-identical, validated phase 10) and only enable `distribute_coarse` as a
      memory escape hatch. Wired + CI regression test (`test_mpi_amr_distributed_coarse`, np=1/2/4
      Serial) + GH200 harness (`amrmpi_integrated` measures replicated AND distributed) ; perf result
      documented here as NEGATIVE quantified.

## Device validation of post-#48 features (round 2)

The CI only runs Release / Python / MPI / Kokkos SERIAL (CPU). Several bricks merged on
master AFTER #48 have a DEVICE PATH but had only been exercised on CPU. We confirmed them on
GH200 (`armgpu` node, `module load cuda/12.6`, Kokkos 4.4.01 `Kokkos_ARCH_HOPPER90`, `nvcc_wrapper`),
each by the SAME logic compiled in `exec=Cuda` (Kokkos Cuda backend, `srun -n 1 --gpus-per-task=1`)
AND in `exec=Serial` oracle (g++, `POPS_HAS_KOKKOS` off), with BIT-BY-BIT comparison cell by cell
(`diff_bin`, `dmax = max|cuda - serial|`). `for_each_cell` is ASYNC under Cuda : each harness does
`device_fence()` before the host read / the dump. Versioned harnesses (outside CI, guarded by `srun`/sbatch) :
`python/tests/gpu/{gpu_aux_validate,gpu_epm_validate,gpu_amr_bz_validate,diff_bin}.cpp`,
`gpuval2_CMakeLists.txt`, `romeo_gpuval2_build.sh`. REAL results (GH200 sbatch job) :

- **T_e read via `load_aux<5>` (aux component 4) (#50/#51).** [x] VALIDATED DEVICE. The previous
  port had only validated `load_aux<4>` (B_z, comp 3) ; we add comp 4 (T_e). A toy model `n_aux=5`
  (zero flux, source `S = T_e u`) reads `a.T_e = a(i,j,4)` in `assemble_rhs` -> `load_aux<5>` (named
  functor `AssembleRhsKernel`, `for_each_cell` POPS_HD) on device. NON-CONSTANT profile `T_e = 1 + x + 2y`.
  exec=Cuda : `R = T_e u` in [2.1875, 7.8125] (per-cell read), `max|R - T_e u| = 0`.
  **`dmax = 0.000e+00`** vs Serial (256 cells). Bit-identical.
  HONESTY NOTE : we validate HERE the REAL device path of the T_e read (`assemble_rhs`, named
  functors), NOT the `System::add_compiled_model` path. The latter instantiates extended lambdas
  `__host__ __device__` in the calling TU (known nvcc limit, documented in
  `runtime/dsl_block.hpp` and `tests/test_compiled_model_parity.cpp`) and SEGFAULTS at execution on Cuda
  , independently of T_e (a System+`add_compiled_model`+`eval_rhs` harness did crash on GH200,
  `compute-sanitizer memcheck` = 0 device error, so the crash is host-side/extended lambda). The T_e
  marshaling of the System path (`apply_te`, `copy_state` comp 4) thus stays covered only in Serial CI.

- **SCREENED EPM / Helmholtz `div(eps grad phi) - kappa phi = f` (#44, `GeometricMG::set_reaction`).**
  [x] VALIDATED DEVICE. The `kappa` term lives in the POPS_HD `for_each_cell` of the red-black smoother, the
  residual and the apply (`numerics/elliptic/poisson_operator.hpp`) -> device under Cuda. MMS `eps=1+0.5x`
  + `kappa=50`, exact Dirichlet, V-cycles with the same criterion as `tests/test_screened_poisson.cpp`.
  exec=Cuda : cycles 8/9/9 (IDENTICAL to Serial), order 2 convergence (Linf ratios 3.69 / 3.85).
  **`dmax = 0.000e+00`** vs Serial on phi (n=64, 4096 cells). Same cycles, same phi to the bit.

- **ANISOTROPIC EPM `div(diag(eps_x, eps_y) grad phi) = f` (#52/#56, `set_epsilon_anisotropic`).**
  [x] VALIDATED DEVICE. The second field `eps_y` (faces normal to y) is read in the same
  POPS_HD `for_each_cell` as above. MMS `eps_x=1+0.5x`, `eps_y=1+0.3y` (cf.
  `tests/test_anisotropic_epsilon.cpp`). exec=Cuda : cycles 9/10/11 (IDENTICAL to Serial), order 2
  (Linf ratios 4.00 / 4.00). **`dmax = 0.000e+00`** vs Serial on phi (n=64, 4096 cells).

- **Per-level B_z in the AMR path (#53, `AmrSystemCoupler::fill_bz`).** [x] VALIDATED DEVICE. B_z(x,y)
  is set at the centers OF EACH LEVEL (`geom.refine(1<<k)`, dx = dx_coarse / 2^k) on the
  `kAuxBaseComps` comp of the shared aux channel ; the model reads it `load_aux<4>` in the AMR source kernel
  (`for_each_cell` POPS_HD) level by level. NON-CONSTANT profile `B_z = 1 + sin(2 pi x) cos(2 pi y)`
  to distinguish the levels. exec=Cuda : `B_z` re-read = 0.80865828 at level 0 (center (4,4)),
  0.90245484 at level 1 (center (8,8)), DISTINCT VALUES, each == its level center ; the
  source consumes the right B_z per level (coarse and fine evolve with THEIR B_z). **`dmax = 0.000e+00`**
  vs Serial on coarse+fine U (512 cells, 2 levels), conservation respected.
  Initial validation by `advance_amr` (HEADER-ONLY, the engine that `AmrSystemCoupler` calls level
  by level). The ENTIRE `AmrSystemCoupler` FACADE is now validated under nvcc too (device limit
  (b) LIFTED) : the `CoupledSystemLike` concept probed `for_each_block` with a GENERIC LAMBDA
  in an unevaluated context (`requires s.for_each_block([](auto&){})`), which the nvcc/EDG frontend refused
  -> `CoupledSystemLike<CoupledSystem<...>>` false under Cuda -> coupler CTAD impossible. The probe
  switched to a NAMED FUNCTOR `detail::ForEachBlockProbe` (same recipe as the named functors,
  point 8). Harness `python/tests/gpu/gpu_amrsys_facade_validate.cpp` (instantiates the entire facade :
  CoupledSystem 2 blocks + system Poisson + `solve_fields` + `step` on AMR 2 levels) : GH200
  `CUDA_BUILD_OK`, `exec=Cuda` OK, U(coarse+fine, 2 blocks) **`dmax = 0.000e+00`** vs Serial,
  before/after confirmed (lambda probe put back -> nvcc fails on the CTAD).

- **Multi-box B_z AMR distributed across several GPUs (#59).** [!] FUNCTIONAL DEVICE multi-GPU, but NOT
  bit-identical in the strict sense on the global sums. #59 merged on master the multi-box
  coverage (single-rank + MPI np=2/4, CI Kokkos Serial). On GH200 (np=1/2/4, one GH200 per rank,
  exec=Cuda, CUDA-aware OpenMPI, coarse 2x2 boxes + 2 disjoint fine patches distributed SFC,
  `coarse_replicated=false`) : B_z is correctly sampled PER LEVEL and PER local BOX
  (`bz_bad = 0` at each np) and the source `S = B_z u` reads it per box on the device. `cmax`
  (max reduction, order-insensitive) is BIT-IDENTICAL at the three np (`dcmax = 0`). However the
  global ADDITIVE invariants (mass/csum/csumsq, `all_reduce_sum` over the local boxes) DIFFER at
  the rounding level between np : `dmass ~ 1e-15`, `dcsum ~ 3e-13`, `PARITE dmax = 9.1e-13` (np=2/4 vs
  np=1 oracle), an FMA REDUCTION ORDER effect (the multi-box coarse is genuinely distributed, so
  the partial sum changes order according to the per-rank split). This is not a bug : the device
  per-cell compute is correct and the max is exact ; only the float sums depend on the
  order. Contrast with phase 10 (`amrmpi_integrated`, dmax=0) where the coarse is REPLICATED -> each
  rank sums the SAME entire domain -> identical reductions. Honestly : multi-GPU FUNCTIONALLY
  CORRECT (B_z per box/level read on device, conservation to rounding) but strict bit-identity NOT
  reached on the summed quantities when the coarse is distributed. Harness
  `python/tests/gpu/{gpu_amr_bz_mpi_validate.cpp, gpuval2_mpi_CMakeLists.txt, romeo_gpuval2_mpi_build.sh}`.

Round 2 summary : the 4 single-GPU device-path features (T_e, EPM Helmholtz, EPM anisotropic, per-level
AMR B_z) are confirmed BIT-IDENTICAL (dmax=0) on GH200 in exec=Cuda vs Serial ; for each
elliptic, same MG cycles ; conservation respected. The distributed multi-box B_z multi-GPU (#59) is
functionally correct (B_z per box/level read on device, `bz_bad=0`, `dcmax=0`) but the global sums
are not bit-identical between np (reduction order, dmax ~ 9e-13). Honest reservation : the
`System::add_compiled_model` path on Cuda remains limited by the cross-TU extended lambdas (existing
follow-up phase 8), the read of the aux fields (B_z, T_e) was validated on device via `assemble_rhs` /
`advance_amr` (named functors), which are the real device paths.

## Suggested strategy

- Advance phase by phase, each validated CPU == GPU (the seam allows switching one step at a
  time). Start with 1 -> 2 -> 4 (pure multi-block transport on GPU, without Poisson) : already a
  useful and testable milestone. Device Poisson (3) next, then AMR (5) and MPI (6).
- ROMEO tooling : `module load cuda/12.6`, build Kokkos+CUDA (`Kokkos_ARCH_HOPPER90`), `srun
  --account=<account> -p instant --constraint=armgpu --gres=gpu:1` (cf. docs/GPU_ROMEO.md). nvcc only
  runs on the GPU node (aarch64), not on the login (x86).
- Orchestration : a multi-agent workflow per backend (Kokkos-Serial / OpenMP / CUDA) + adversarial
  verifiers comparing to the CPU oracle is the right tool for the validation phase (cf. DSL sessions).

## Status

The rest of the vision (symbolic DSL : interpreter, codegen flux/brick/source/elliptic, CSE, JIT
.so, type-erased dispatch in the System, AOT compiled block at native parity on CPU/Serial ; Roe flux ;
VariableRole present but not yet wired ; variable eps(x) core-side, System/Python wiring to do ;
physics/ numerics/ reorg) is COMPLETE at the PROTOTYPE level, tested (71 C++ core ctests, +21 MPI ctest
entries, 26 Python tests) and verified up to the GH200. The INTEGRATED VALIDATION AmrSystem + MPI + GPU (the three axes in one single run)
is DONE (phase 10, GH200, dmax=0, mass conserved to 0). The device-path bricks merged
AFTER #48 (T_e via `load_aux<5>`, screened EPM/Helmholtz #44, anisotropic EPM #52/#56, per-level B_z
AMR #53) are confirmed BIT-IDENTICAL on GH200 (round 2, dmax=0, same MG cycles, conservation).
What remains, on the PERF side and no longer correctness : AMR full-device strong-scaling (distributed coarse
`replicated_coarse=false` wired in `AmrSystem`, reflux without host bounce, device-direct GPUDirect
halos, distributed device FFT) and zero-copy AOT parity on device (nvcc limit, phase 8 ;
this is also why T_e was validated on device via `assemble_rhs` and not `add_compiled_model`).

## MPI + Kokkos Cuda multi-GPU validation (2026-06-06, GH200)

Closing the "MPI + Kokkos Cuda multi-GPU" gap of the `BACKEND_COVERAGE` matrix (column marked `?`).
The single-GPU Kokkos Cuda being device-clean (6/6 after #150+#152), we exercise here the RANK-COUNT
INVARIANCE of the ELLIPTIC / Schur / Poisson / system-solve stack under MPI + Kokkos Cuda on SEVERAL
GH200 (1 GPU per rank).

- Provenance : `origin/master` 2674d64 (archive, fresh rsync under `~/pops_gpu_p1/mpicuda/` on ROMEO).
- Node `romeo-a057` (`armgpu`), 4x GH200 visible. Kokkos SERIAL+CUDA, `Kokkos_ARCH_HOPPER90`,
  `nvcc_wrapper`. MPI = OpenMPI 4.1.7 CUDA-aware (`+cuda cuda_arch=90 fabrics=ucx schedulers=slurm`,
  hash `nkokjyt`). Launch `srun -n {1,2,4} --gpus-per-task=1` (1 GPU/rank). Single build
  `-DADC_USE_KOKKOS=ON -DADC_USE_MPI=ON`. np=1 = single-GPU Cuda oracle.
- Harness : script `mpicuda_run.sh` (single configure, build of the MPI targets, run np=1/2/4, parses
  the invariant printed by each test, computes the cross-np dmax).

Table per test (rc + cross-np invariance vs np=1, dmax on the printed invariants) :

| Test | np=1 | np=2 | np=4 | dmax np2-vs-np1 | dmax np4-vs-np1 | invariance |
|------|------|------|------|-----------------|-----------------|------------|
| test_krylov_solver (BiCGStab) | OK | OK | OK | 0 (iters=4) | 0 (iters=4) | BIT-IDENTICAL |
| test_condensed_schur_source_stepper | OK | OK | OK | 0 (all_reduce_max) | 0 | BIT-IDENTICAL |
| test_mpi_poisson | OK | OK | OK | 0 (res=1.832e-14) | 0 | BIT-IDENTICAL |
| test_mpi_system_solve_fields | OK | OK | OK | 0 (\|phi\|max=3.125e-2) | 0 | BIT-IDENTICAL |
| test_mpi_system_fft | OK | OK | OK | n/a (refuses fft if MPI) | n/a | expected behavior |
| test_mpi_amr_compiled_parity | OK | OK | OK | 0 (mass/csum/csumsq/cmax %.17e) | 0 | BIT-IDENTICAL |
| test_mpi_amr_distributed_coarse | OK | OK | OK | 0 (cmax=1.41585803814656397) | 0 | BIT-IDENTICAL |
| test_mpi_coupled_source | OK | OK | OK | 0 (n_e/n_i/n_g) | 0 | BIT-IDENTICAL |
| test_mpi_mbox_parity | OK | OK | OK | 0 (== single-box) | 0 | BIT-IDENTICAL |
| test_mpi_cutcell_multibox | OK | OK | OK | 0 (== single-box) | 0 | BIT-IDENTICAL |
| test_schur_condensation | FAIL | FAIL | FAIL | -- | -- | BACKEND failure (see reservation) |

All the %.17e invariants (AMR compiled parity, AMR distributed coarse) are BIT-FOR-BIT identical between
np=1/2/4 (dmax = 0.000e+00, not even an FMA reassociation drift) ; `crossrank_spread = 0` ;
mass conserved ; Krylov iterations invariant (min=max=4 at any np). `test_mpi_system_fft` REFUSES
`fft` under MPI (np=2/4) with a collective error, as designed (#93), so nothing to compare cross-np.
No deadlock, no unguarded empty rank.

HONEST RESERVATION, `test_schur_condensation` : fails on Cuda FROM np=1 (assembly eps = 1.026827
instead of ~1, RHS dmax = 4.892e+01, checks `B_rhs_analytique_interieur` / `C1_eps_*_diag` /
`C2_rhs_eq_neg_laplacien_bitident`). The failure is IDENTICAL at np=1/2/4 : it is a defect of the device
ASSEMBLY path (Cuda backend) of the Schur condenser, NOT an MPI invariance issue. It passes
on Serial / Kokkos Serial (ci-full). Consistent with the `Kokkos Cuda` column of this test already marked
`?` in `BACKEND_COVERAGE` : not device-clean, so its MPI variant cannot be declared
multi-GPU-clean. To be handled on the device assembler side (outside the scope of this MPI session).

compute-sanitizer (memcheck) :
- On the MPI-LINKED binaries, compute-sanitizer is UNUSABLE on this node : OpenMPI `mtl:ofi`
  does not open under the sanitizer wrapper (`Target application terminated before first instrumented
  API call`), even at `-n 1` ; and the CUDA-aware OpenMPI probes every pointer via `cuPointerGetAttribute`
  (which returns `INVALID_VALUE` for a HOST pointer by design), which the sanitizer counts as
  an "error" (708 API flags on `test_mpi_system_solve_fields`, but `0 bytes leaked`, no OOB,
  no invalid read/write, benign MPI noise, not a code defect).
- CLEAN memcheck of the device elliptic path (same checkout, Kokkos-Cuda build WITHOUT MPI) :
  `test_krylov_solver` and `test_polar_poisson_mms` -> `ERROR SUMMARY: 0 errors`, `LEAK SUMMARY:
  0 bytes leaked in 0 allocations`. The device kernels of the elliptic stack are thus memory-clean ;
  the "errors" of the MPI run were exclusively OpenMPI's CUDA-aware pointer probing.

VERDICT : multi-GPU rank-invariant = YES for the elliptic / Schur(stepper) / Poisson /
system-solve / AMR stack under MPI + Kokkos Cuda (10 tests OK at np=1/2/4, cross-np dmax = 0, clean kernel
memcheck), with ONE reservation : `test_schur_condensation` fails on the Cuda backend side from np=1 (device
assembly defect, independent of the number of ranks) and thus remains `?`/failure on the
Kokkos Cuda and MPI+Kokkos Cuda column. No deadlock observed.
