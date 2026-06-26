# GH200 device validation: dense_eig + hyqmom15 (.so), single + multi-GPU

ADC-181. Validates on GH200 (ROMEO, NVIDIA GH200 120GB, aarch64) the device path of the exact
eigenvalue-based wave speeds: `include/pops/numerics/linalg/dense_eig.hpp` (`real_eig_minmax`: Hessenberg
reduction + Francis double-shift QR iteration, named `POPS_HD` functors, on-stack buffers, zero
allocation) through the compiled hyqmom15 `.so` (DSL-emitted bricks, `exact_speeds=True`,
`riemann="hll"`), wired by the `pops::add_compiled_model` compilation seam (full native path: device
`assemble_rhs`, halos).

`dense_eig.hpp` was designed for nvcc but had never been EXECUTED on device. This note provides the
device-execution proof and the host/device parity on the same node (the md5s of the tested
sources/artifacts are in the Reproducibility section below).

Complement to `docs/GPU_ROMEO.md` (which validates the flow of one generated brick): here we validate
the EIGEN path (signed HLL wave-speed bounds) end to end in a run.

## Recipe (recap)

`armgpu` node (Grace-Hopper, aarch64), `module load cuda/12.6` + `romeo_load_armgpu_env`. Device
compiler = the `nvcc_wrapper` of the Kokkos install `~/pops_gpu_p1/kinstall` (SERIAL;CUDA, sm_90). The
driver `diocotron_gpu.cpp` (`Hyqmom15Hyp/Src/Ell` bricks + `pops::System` + Poisson `geometric_mg`)
reads the binary initial state `ic_128.raw` and advances the 15-moment diocotron.

## Reproducibility (versioned sources)

Everything that produces the figures below is in the repository, under `docs/validation/`:

* `make_brick_and_ic.py`: generator of the DSL brick `hyqmom15_brick.hpp` (emitted by
  `emit_cpp_{brick,source,elliptic}` from the validated hyqmom15 model) AND of the initial state
  `ic_<n>.raw` (`diocotron_state` of the validated python);
* `diocotron_gpu.cpp`: driver (reads the binary IC, assembles the composite via
  `add_compiled_model`, advances, dumps `snap_*.raw` + `growth.csv`);
* `CMakeLists.txt`: driver build (also compiles `python/bindings/system/base/system.cpp`);
* `compare_snap.cpp`: final-state comparator;
* `parity181.sbatch` (parity + timing, sections 2 and 4), `mpi181.sbatch` (multi-GPU substrate,
  section 3).

`hyqmom15_brick.hpp` and `ic_<n>.raw` are NOT versioned: they are derived artifacts, regenerated
identically by `make_brick_and_ic.py` on a machine with the adc python module (the C++ module is not
required for the DSL emission, pure-python):

```
POPS_CASES=/path/adc_cases PYTHONPATH=/path/adc_cpp/python \
  python docs/validation/make_brick_and_ic.py --ns 128 256 --out $WORK
```

Determinism verified (local regeneration vs the ROMEO run artifacts, bit-for-bit):

```
md5(ic_128.raw)          = da245ba8934546986508976a64156d2e   (regenerated == ROMEO)
md5(hyqmom15_brick.hpp)  = d785b13ac0da1dd349ff4775368c8ff2   (--ns 128 256, 1940 lines, == ROMEO)
md5(include/.../dense_eig.hpp) = 86fb1cbbec0e265cd255559434ce83c6   (cap QR 30 ; sections 1/2/4)
md5(include/.../dense_eig.hpp) = 0d5f0f3086814e5e2053af30bae8ae92   (cap QR 100 by default, ADC-195/#309 ; section 3 multi-GPU ; == ROMEO/current worktree)
```

The device binary is no longer staged by hand: `parity181.sbatch` recompiles BOTH variants (device
nvcc_wrapper / host g++ Serial) from the SAME versioned sources staged in `$WORK`, on the GH200 node.
Only the brick and the IC (regenerated) must be placed in `$WORK` beforehand (cf. the script header).

## 1. Device execution (acquired, job 654562)

The full driver ran on GH200 device (`exec=Cuda`), 24706 steps, the 15-moment model with `dense_eig`
/ exact HLL / electric sources / multigrid, `DT_COLLAPSE` guard:

```
noeud=romeo-a045  NVIDIA GH200 120GB
[fin] 24706 pas, t=0.97320, derive de masse 1.72e-13
[DT_COLLAPSE] pas 24706 ... etat au bord de realisabilite, projection requise (ADC-177). Sortie propre.
```

`real_eig_minmax` is called PER CELL at each step (HLL flux wave-speed bounds). 24706 steps with no
NaN nor destructive Gershgorin fallback, mass drift 1.7e-13: the per-cell QR loop stays bounded (no
stall nor iteration-count blow-up) on device. This is the first proof that the Hessenberg+QR `POPS_HD`
path executes on GH200.

## 2. Host (Serial) / device (Cuda) parity, SAME node

A==B protocol of the previous campaigns. The SAME `diocotron_gpu.cpp` + `hyqmom15_brick.hpp` + the
SAME `~/adc_cpp/include` header tree are compiled in two variants on the GH200 node, by
`parity181.sbatch`, from the SAME versioned sources staged in `$WORK` (no hand-staged binary
anymore):

* device: the `kinstall` GPU `nvcc_wrapper`, `DefaultExecutionSpace=Cuda`;
* host: `g++` against a Serial aarch64 Kokkos built on the node, `DefaultExecutionSpace=Serial`.

Under Serial, `DefaultExecutionSpace == DefaultHostExecutionSpace`: the `if constexpr` guard of
`for_each_cell` (#165) takes the sequential host loop on the small boxes (coarse V-cycle levels)
where the device keeps `parallel_for`; no inter-iteration dependency, so bit-identical expected
outside reductions. `real_eig_minmax` is the SAME `POPS_HD` source instantiated host on one side,
device on the other.

20 steps from the SAME initial state `ic_128.raw`, binary dump of the final state (15 moments + phi),
compared by `compare_snap`. Initial run job 654862; REPRODUCED byte-for-byte by the versioned
`parity181.sbatch` (job 654998, romeo-a057, BOTH variants recompiled from the repo sources, brick +
IC regenerated):

```
=== RUN DEVICE (Cuda) 20 pas ===  [fin] 20 pas, t=0.01594, derive de masse 1.63e-13
=== RUN HOST (Serial) 20 pas ===  [fin] 20 pas, t=0.01594, derive de masse 1.64e-13
=== COMPARE snap_000020.raw (device vs host) ===
n_a=128 t_a=0.015936559201338053 k_a=20
n_b=128 t_b=0.015936559201338029 k_b=20
dt_clock (|t_a-t_b|) = 2.429e-17
payload doubles      = 262144   (15*128^2 moments + 128^2 potentiel)
bit-identiques       = 27710 (10.5705%)
max |a-b|            = 3.450573e-13   (index 256884 -> maximum dans le potentiel phi)
max rel              = 6.456023e-10
```

The full `t` (17 digits) show the clock gap `...338053` vs `...338029`: this is the ~1 ULP evidence
cited below, invisible in the `%.3e` of `dt`.

Reading:

* **`dense_eig` path: agreement to the printed precision (~1 ULP), not a bit-exact proof.** The time
  step `dt` (column `dt` of `growth.csv`) comes from `step_cfl` = CFL on the MAX of the wave-speed
  bounds returned by `real_eig_minmax` cell by cell. The MAX reduction is exact regardless of order:
  at identical input, the `POPS_HD` QR returns identical bounds device/host (this is the algorithmic
  property of the eigen path, deterministic and from the SAME host/device source). But what we
  OBSERVE is only an agreement to the PRINTED precision: `growth.csv` writes `dt` only in `%.3e`
  (4 digits) and the `a_l` modes in `%.6e` (7 digits), not the 16 digits of a double. The cumulative
  clock `t` differs by 2.4e-17 after 20 steps (rel 1.5e-15, ~1 ULP): this is the DIRECT proof that
  the `dt` are NOT all strictly bit-identical (otherwise, summed in the same order, `t` would be
  equal to the bit). The origin is the phi -> source feedback (next bullet) which perturbs at noise
  level the state read by `real_eig_minmax` from step 2 on. On the diocotron physical mode, `a4`
  (l=4, ~5.94e-2) coincides on the 7 printed digits device/host; the small noise modes `a2/a3/a5/a6`
  (~1e-16) already differ in the printed CSV. Correct phrasing: ~1 ULP agreement on the observable,
  not a bit-for-bit equality.
* **DIFFUSE gaps at machine-noise level, MAXIMUM in the multigrid potential.** The comparator gives
  only 27710/262144 = 10.57% of bit-identical doubles: ~89% of the payload differs, moment fields
  INCLUDED (cf. `a2/a3/a5/a6` above). The gaps are therefore DIFFUSE, not confined to `phi`. The
  MAXIMUM `|a-b|` = 3.45e-13 falls, itself, at index 256884 (`phi` zone >= 245760). Common origin:
  the Poisson V-cycle has a residual criterion in a SUM reduction, whose reassociation `parallel_for`
  (device) vs serial loop (host) changes the last bit; the corrected potential then propagates
  through the electric source term to the WHOLE state (phi -> source -> moments coupling), hence the
  diffusion. Absolute gap 3e-13 on fields of order 1 to 1.7e3: machine noise. `max rel` 6.5e-10 is on
  an entry of near-zero magnitude (inflated relative). Expected behavior of parallel reductions: the
  wave-speed solver (the subject of ADC-181) is not the SOURCE of the divergence, it faithfully
  propagates the state it is given.

## 3. Multi-GPU MPI (halo substrate, jobs 654863 then 654999) and scope

The hyqmom15 driver uses `pops::System` (mono-box, mono-rank): it has NO MPI domain decomposition. The
multi-GPU correctness of the model factors into two independent bricks:

1. correctness of the model on one GPU, including `dense_eig` (section 1 + section 2);
2. multi-box halo + multi-GPU MPI machinery, which is PURE position/neighborhood and never touches
   the per-cell computation of `real_eig_minmax`.

`real_eig_minmax` being strictly cell-local (reads the local moment vector, forms the Jacobian block,
returns the spectrum min/max), its multi-GPU correctness REDUCES to (1) already validated here and
(2) already validated independently (CUDA-IPC halo fix #254; multi-box parity np=1/2/4 #59).

Corroboration on THIS node, post-#254, of the multi-GPU halo substrate (versioned harness
`python/tests/gpu/gpu_amr_bz_mpi_validate.cpp`, B_z per AMR level distributed multi-box, one GH200
per rank). Re-executed by `mpi181.sbatch` (job 654999, romeo-a057), figures identical to the initial
run 654863:

```
np=1 exec=Cuda : mass=2.10017927603615240 csum=537.645894665255014 csumsq=1129.79422430042723 cmax=2.19619397662556448 | bz_bad=0
np=2 exec=Cuda : mass=2.10017927603615151 csum=537.645894665254787 csumsq=1129.79422430042814 cmax=2.19619397662556448 | bz_bad=0
```

np=1 vs np=2 gap: `cmax` (max reduction, associative+commutative exact in floating point)
BIT-IDENTICAL; `mass`/`csum`/`csumsq` (sum reductions) at the last ulp (drel 4.2e-16 / 4.2e-16 /
8.0e-16) due to the reassociation of the sums across rank counts. `bz_bad=0` on both ranks. The
multi-GPU substrate is therefore bit-identical on the max and at the last ulp on the sums: expected
and documented behavior.

### Direct multi-rank hyqmom15 (System, mono-box round-robin) -- `diocotron_mpi.sbatch`

The SAME driver `diocotron_gpu.cpp` (`Hyqmom15Hyp/Src/Ell` bricks + Poisson `geometric_mg`), linked
to CUDA-aware OpenMPI by `-DADC_VALIDATION_MPI=ON` (a CMake option that defines `POPS_HAS_MPI` and
switches `comm.hpp` + the collectives of `system.cpp` onto the real MPI path), runs under
`srun -n {1,2,4} --gpus-per-task=1` (one GH200 per rank). Topology = MPI decomposition of `System`
mono-box round-robin (box 0 on rank 0; the other ranks have `local_size()==0` and contribute 0 to the
all-reduce), identical to the CPU topology already validated by `run_mpi.py`. The diagnostic and
snapshot reads go through `state_global` / `potential_global` (collective all-reduce,
`system.cpp:1808/1824`); the `step_cfl` / `solve_fields` step is collective; the `DT_COLLAPSE` vote
goes through `all_reduce_max` (all ranks exit together, never a deadlock). Only rank 0 writes files +
stdout. Without `POPS_HAS_MPI`, the stubs of `comm.hpp` (`my_rank()=0` / `n_ranks()=1`) keep the
serial path (`parity181.sbatch`) bit-unchanged.

Validation criterion (`diocotron_mpi.sbatch`, embedded python): mass conservation
`massdrift < 1e-12` per run (the mono-GPU run of section 1 already shows 1.7e-13 over 24706 steps),
AND global mass np=2/4 vs np=1 at the ulp level (rel < 1e-12; expected ulp ~4e-16, due to the
reassociation of the Poisson V-cycle sum -- NOT bit-exact, cf. section 3 / substrate above). The
machine-parsable line `HYQMOMPI np=.. mass=.. massdrift=..` of the driver feeds the gate directly.
ROMEO result (job 656726, romeo-a058, GH200 120GB, partition `instant`, 4 GPU/node; `dense_eig.hpp`
cap QR 100, md5 0d5f0f30):

```
HYQMOMPI np=1 n=128 t=0.12232851 steps=200 mass=1.71504800000000751e+03 massdrift=1.766e-13 l4=9.637906e-02
HYQMOMPI np=2 n=128 t=0.12232851 steps=200 mass=1.71504800000000751e+03 massdrift=1.766e-13 l4=9.637906e-02
HYQMOMPI np=4 n=128 t=0.12232851 steps=200 mass=1.71504800000000751e+03 massdrift=1.766e-13 l4=9.637906e-02
np=1 masscons=1.766e-13(OK) massrel_vs_np1=0.000e+00(OK)
np=2 masscons=1.766e-13(OK) massrel_vs_np1=0.000e+00(OK)
np=4 masscons=1.766e-13(OK) massrel_vs_np1=0.000e+00(OK)
MULTIGPU_MPI_PASS
```

The 200 steps run on device (Cuda, GH200) at np=1/2/4; the `DT_COLLAPSE` guard does not trigger
(t=0.122, mass conserved to 1.77e-13). The global mass is here BIT-IDENTICAL across np=1/2/4
(`massrel_vs_np1 = 0`), NOT only at the ulp level as the criterion announced: the mono-box
round-robin topology places the whole box on rank 0, the other ranks have `local_size()==0` and add
only zeros to the all-reduce -- so there is NO inter-rank sum reassociation to perturb the last bit.
The real inter-rank reassociation (sums that differ at the last ulp across rank counts) is, itself,
exercised by the B_z multi-box substrate (#59/#254, start of this section). This run also re-confirms
the device path of `real_eig_minmax` on the CURRENT revision of `dense_eig.hpp` (cap QR 100,
ADC-195/#309), later than sections 1/2/4 (cap 30): 200 per-cell steps with no NaN nor QR stall. The
OpenMPI `mtl/ofi` warnings in the log are benign transport probes (automatic BTL fallback); the three
ranks finish `COMPLETED 0:0`.

The distinct variant -- a hyqmom15 `AmrSystem` MULTI-BOX driver wiring the `Hyqmom15Hyp/Src/Ell`
composite + Poisson on a DISTRIBUTED coarse (real domain decomposition, inter-GPU halos) -- is now
DELIVERED by ADC-320: see section 5.

## 4. real_eig_minmax warp divergence (indicative)

`real_eig_minmax` iterates a DATA-DEPENDENT number of times per cell (QR deflation): threads of the
same warp may iterate a different number of times (warp divergence). Indicative time/step measurement
(device GH200 vs host Serial of the same node romeo-a057, n=128, 50 steps, job 654998, same run as
the parity above):

```
device(Cuda) : 50 pas en  2.664 s -> 0.0533 s/pas (18.8 pas/s)
host(Serial) : 50 pas en 35.212 s -> 0.7042 s/pas ( 1.4 pas/s)
```

The device is ~13.2x faster than a single Grace core at n=128 (throughput comparison, not an isolated
divergence cost: the device uses the whole GPU, the host one core). No pathological slowdown: the
per-cell QR loop does not make the device step diverge, and the 24706-step run (section 1) confirms
the stability.

The EISPACK iteration cap (30/block) and the Gershgorin fallback bound the per-cell cost: the
divergence stays the benign one of a tiny dense solver per thread.

## 5. hyqmom15 AmrSystem MULTI-BOX: real MPI domain decomposition (ADC-320)

Section 3 validates the hyqmom15 multi-GPU branch through the MONO-BOX round-robin MPI decomposition
of `System`: the whole box stays on rank 0, the other ranks have `local_size()==0` and add only zeros
to the all-reduce. There the inter-GPU halos NEVER transport the 15-moment state (the mass being
bit-identical across np was an artifact of that topology). ADC-320 records the DIRECT validation: a
real DOMAIN-DECOMPOSED run where the halos exchange the state between GPUs.

The driver `diocotron_amr_gpu.cpp` wires the SAME composite (`Hyqmom15Hyp/Src/Ell` bricks + Poisson
`geometric_mg`) onto `pops::AmrSystem` with `distribute_coarse=true`: the coarse level becomes a
MULTI-BOX `BoxArray` (2x2 tiles, `coarse_max_grid=n/2`) spread round-robin across `n_ranks()` GPUs.
At np>1 the coarse transport calls `fill_boundary` on this multi-box MultiFab (`amr_subcycling.hpp`),
a REAL cross-rank MPI exchange of the 15 conserved components over `SharedHostPinnedSpace` buffers
(the #254 CUDA-IPC fix); with refinement active (`set_refinement` + `regrid_every=8`) the fine
patches distribute round-robin TOO (conservative reflux). `distribute_coarse` has been effective
since ADC-319 (#140). The SINGLE compiled block routes through `AmrCouplerMP<Model>` (never the
`AmrSystemCoupler` facade, which is not instantiable under nvcc/EDG). The driver replays, in the SAME
binary and from the SAME initial state, the REPARTI mode (distributed coarse) and the REPLIQUE mode
(oracle: mono-box coarse replicated on every rank); dt is FIXED, so the step sequence is identical
across all np and in both modes.

### Recipe (versioned sources)

`docs/validation/diocotron_amr_gpu.cpp` + `CMakeLists.txt` (target `diocotron_amr_gpu`, compiles
`python/bindings/amr/amr_system.cpp` -- NOT `system.cpp`) + `diocotron_amr_mpi.sbatch`. Brick + IC: the SAME
generated artifacts as section 3 (`make_brick_and_ic.py --ns 128`, md5 `ic_128.raw`
da245ba8934546986508976a64156d2e, brick d785b13ac0da1dd349ff4775368c8ff2). Device build with the GPU
`kinstall` `nvcc_wrapper`, CUDA-aware OpenMPI (`-DADC_VALIDATION_MPI=ON`),
`srun -n {1,2,4} --gpus-per-task=1` (one GH200 per rank), `OMPI_MCA_btl_smcuda_use_cuda_ipc=0`.

### Result (nvcc build job 657120 `BUILD_OK`, gate job 657147 `PASS`, romeo-a057, GH200 120GB, partition instant)

80 steps, dt=4e-4, n=128, coarse 2x2 (`coarse_max_grid=64`), `regrid_every=8`; `mode=reparti` lines
(figures bit-for-bit identical between the two jobs: device determinism):

```
np=1 mass=1.04678222656249986e-01 massdrift=1.326e-16 csum=1.71504800000000023e+03 cmax=5.74956081822768139e-01 | patches=65 clocal=4 ctotal=4
np=2 mass=1.04678222656250014e-01 massdrift=1.326e-16 csum=1.71504799999999682e+03 cmax=5.68102576704108797e-01 | patches=47 clocal=2 ctotal=4
np=4 mass=1.04678222656250000e-01 massdrift=0.000e+00 csum=1.71504799999999409e+03 cmax=5.64622946171666751e-01 | patches=52 clocal=1 ctotal=4

np=1 masscons=1.326e-16(OK) massrel_vs_np1=0.000e+00(OK) dcmax_vs_np1=0.000e+00(info) distributed=yes(OK) dist_vs_repl_dmax=0.000e+00(OK) run_ok=OK
np=2 masscons=1.326e-16(OK) massrel_vs_np1=2.652e-16(OK) dcmax_vs_np1=6.854e-03(info) distributed=yes(OK) dist_vs_repl_dmax=0.000e+00(OK) run_ok=OK
np=4 masscons=0.000e+00(OK) massrel_vs_np1=1.326e-16(OK) dcmax_vs_np1=1.033e-02(info) distributed=yes(OK) dist_vs_repl_dmax=0.000e+00(OK) run_ok=OK
HYQMOMAMR_MULTIBOX_PASS
```

Reading:

* **REAL domain decomposition (ADC-319).** At np=2/4 the distributed coarse has
  `coarse_local_boxes < coarse_total_boxes` (2/4 then 1/4): each rank carries only a part of the 4
  tiles, so the coarse `fill_boundary` is a REAL inter-GPU exchange of the 15 moments (at np=1,
  local==total=4, intra-rank exchange). This is exactly what the section-3 mono-box round-robin lacked
  (where the other ranks had `local_size()==0`).
* **Bit-exact halos (the bit-exact criterion, max included).** At EACH np, the distributed coarse
  density is BIT-FOR-BIT equal to the replicated one (`dist_vs_repl_dmax = 0`, max and every cell): at
  fixed np both modes share the same trajectory (same tags, same patches -- np=2: 47/47, np=4: 52/52),
  so the inter-GPU multi-box exchange of the 15 moments is proven TRANSPARENT. This is the direct
  hyqmom15 equivalent of the bit-exact B_z result of section 3.
* **Conservation: mass at last ulp across np (REAL).** Mass conserved per run (massdrift <= 1.3e-16)
  and global mass np=2/4 vs np=1 agreeing at the last ulp (massrel 2.7e-16 / 1.3e-16). Unlike
  section 3 (bit-identical BY ARTIFACT of the round-robin), this is the REAL cross-rank FMA
  reassociation (the coarse being genuinely split), at the expected ulp level.
* **cmax (peak) and patch count diverge across np -- expected, not a bug.** cmax goes from 0.5750
  (np=1) to 0.5681 (np=2) / 0.5646 (np=4) (~1-2%) and the patches from 65 to 47/52. The diocotron is
  UNSTABLE: it exponentially amplifies the ulp difference introduced by the cross-rank Poisson
  reassociation, and the AMR regrid then builds different meshes across np (the peak is resolved
  differently). The CONSERVED invariant (mass) is unaffected. A bit-identical cmax across np is
  therefore NOT exigible on a refined unstable flow: the "bit-exact on the max" of section 3 was for a
  STATIC B_z field; here the bit-exact reading is `dist_vs_repl` at fixed np. phi stays finite (the
  geometric MG converges on the 2x2 multi-box coarse).
* **nvcc.** The SINGLE compiled block (`add_compiled_model` -> `AmrCouplerMP<Model>`, never the
  facade) compiles and runs on GH200 (`nvcc_wrapper`, `python/bindings/amr/amr_system.cpp`): job 657120 `BUILD_OK`.

## Conclusion

* `dense_eig` (`real_eig_minmax`, Hessenberg+QR `POPS_HD`) EXECUTED and CORRECT on GH200 device
  through the hyqmom15 `.so` (`exact_speeds`, `hll`): section 1 (24706 steps) + section 2 (host/device
  parity same node, ~1 ULP agreement on the observable, diffuse machine-noise-level gaps maximal in
  the multigrid potential).
* multi-GPU MPI substrate re-confirmed bit-identical (max) / last ulp (sums) on the node post-#254:
  section 3.
* direct multi-rank hyqmom15 harness EXECUTED on GH200 (section 3, `diocotron_mpi.sbatch`,
  `-DADC_VALIDATION_MPI=ON`): MPI decomposition of `System`, np=1/2/4 (job 656726, romeo-a058), gate
  `MULTIGPU_MPI_PASS` -- mass conserved to 1.77e-13 per run, global mass bit-identical across np=1/2/4
  (mono-box round-robin: only rank 0 carries the box).
* REAL hyqmom15 domain decomposition EXECUTED on GH200 (ADC-320, section 5,
  `diocotron_amr_mpi.sbatch`): `AmrSystem` distributed MULTI-BOX coarse (`distribute_coarse=true`),
  np=1/2/4 (build job 657120, gate job 657147, romeo-a057), gate `HYQMOMAMR_MULTIBOX_PASS` -- coarse
  genuinely distributed (`coarse_local < coarse_total`), bit-exact inter-GPU halos of the 15 moments
  (reparti==replique at fixed np), mass conserved + at last ulp across np.
* validation sources fully versioned (`docs/validation/`), brick + IC regenerated bit-for-bit by
  `make_brick_and_ic.py`, both variants (device/host) recompiled by `parity181.sbatch` from these
  sources.

ADC-181: the direct multi-GPU branch is COVERED by a dedicated hyqmom15 multi-rank run EXECUTED on
GH200 -- the MPI decomposition of `System` (`diocotron_mpi.sbatch`, section 3, job 656726,
`MULTIGPU_MPI_PASS`) -- in addition to the halo substrate (`gpu_amr_bz_mpi_validate`, B_z model,
#59/#254) and the cell-locality argument of `real_eig_minmax`. The `AmrSystem` MULTI-BOX variant
(real domain decomposition, each rank carrying a sub-box with inter-GPU halos of the 15 moments) is
now DELIVERED by ADC-320 (section 5, `HYQMOMAMR_MULTIBOX_PASS`).
