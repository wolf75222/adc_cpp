# TODO - adc_cpp

> **Source of truth — read first.** Active work-tracking lives in **Linear** (team ADC), not here.
> This file is a **historical method/worklog** : it records the reasoning behind work waves and is
> cross-referenced by `docs/PAPER_ROADMAP.md` (e.g. "section 6", Hoffart M1/M2/M3) and kept per
> `docs/DOC_REFONTE_AUDIT.md`. Open research items live in `docs/RESEARCH_BACKLOG.md`. Treat it as a
> backlog/journal, **not** a task tracker — to act on something, open or update a Linear issue.

> Living worklist. Synthesis of (1) the initial goal of the work item (extensible `aux`
> channel + AMR parity + runtime / Python / DSL wiring), (2) what `docs/archive/ROADMAP.md` (ARCHIVE) marks "queued",
> (3) what the agents explicitly noted as "still to do".
> Convention: `[x]` done and on `master`, `[~]` partial, `[ ]` to do.

## CURRENT STATUS (session June 2026, master = #154)

Synthesis to date after the Schur stack + System pipeline + polar + adversarial review + night run. Detail
in sections 10 to 15 (new).

**Night-run merge (#150-#166):** #150/#152/#158 TEST-SIDE device fix (host functors in kernels);
#151 NativeLoader extraction from system.cpp; #155 Kokkos OpenMP CI (91/91); #156 coupling surface;
#157 MPI+Kokkos Cuda multi-GPU rank-invariant validation; #159 neutralize diocotron mentions;
#160/#161/#162 doc corrections (BACKEND_COVERAGE 7/7, ARCHITECTURE A.2, findings A2/A3/A4);
#163 elliptic C++20 concepts (EllipticOperator/LinearSolver/FieldPostProcessor); #164 remove
deprecated amr_coupler.hpp; #165 profiling harness (Poisson MG dominates 96-99.9%); #166 refresh todo.

**Previous-session merges:** #130 polar Poisson (2a); #131 DSL coupled source (P5); #132 IMEX
AMR (Gap 2); #133 nvcc functors; #134 homogeneous BC krylov; #135 Geometry/Box2D+CFL device fix;
#136 fast/full CI split; #137 API honesty; #138 substeps-aware step_cfl; #139 multi-block AMR doc;
#141 AMR layout guard; #154 PR1 capstone (multi-block AmrSystem runtime facade, shared hierarchy).

**Current-session merge (post night-agent cutoff):** #167 A2 (conservative `add_pair` DSL helper +
opt-in verification); #170 C.1 (split `amr_reflux_mf.hpp` into 5 sub-headers + umbrella, verbatim, ctest 93/93);
#169 A3 (average_down trailing covered cells AMR + discriminating tests). #168 Polar 2b MERGED.

**PROCESS (new, integrator):** each PR wave passes an ADVERSARIAL REVIEW (multi-
lens workflow: correctness / bit-identity / test-rigor / hygiene-scope, each finding verified by an
independent skeptic) BEFORE merge, IN ADDITION to ci-full. Merge only if ci-full green AND zero confirmed
blocker. Measured gain: the review caught on #168 a blocker that the CI did NOT see (precondition
nr>=3 of `derive_aux_polar` not enforced -> out-of-bounds phi read for `PolarMesh(nr=2)`) -> fixed.

Review findings: **1-7 on master**; 8 deferred to the MPI port.

**GH200 device VERDICT: 7/7 device-clean Kokkos Cuda single-GPU + MPI+Kokkos Cuda multi-GPU
rank-invariant (10 tests, dmax=0, #157) + Kokkos OpenMP CI 91/91 (#155) + MPI+Kokkos OpenMP rank-invariant
(perf caveat: 3 heavy tests too slow at np>1). All initial device failures were TEST-SIDE
(#150/#152/#158); the elliptic/Schur/polar LIBRARY is device-correct.**

CI (since #136): routine PR = ci-fast (Release + Python). MPI + Kokkos via push master / nightly /
`workflow_dispatch` / label `ci-full`. RULE: set the `ci-full` label on any risky PR (MPI /
Kokkos / device / Schur / AMR) BEFORE merge for the full validation.

**Polar 2b MERGED (#168, master = 43a9160):** transport + polar Poisson in `System.step` (GLOBAL
ring, NO cart<->polar coupling; cartesian bit-identical). 2 bugs found+fixed: (1) `phi`
without ghost -> aux drift read phi out of allocation -> masked nan (test passed WRONGLY: `nan>tol`
false in C++); fix `derive_aux_polar` (radial UPWIND order 2 at walls, theta WRAPPED periodic).
(2) adversarial review: precondition nr>=3 not enforced -> OOB for `PolarMesh(nr=2)`; guard
(check_geometry C++ + PolarMesh Python) + rejection test. Validation: C++ mass conservation 3.4e-15;
Python `System(PolarMesh)` step+step_cfl <1e-11; cartesian 4/4 PASS.

**SCIENTIFIC MILESTONE (#174):** the polar path CAPTURES the diocotron instability. Local sanity
(hollow charge ring, mode l=4 seeded, WENO5/SSPRK3): the mode amplitude GROWS cleanly (x8.8 at
96x96/600 steps, fitted rate gamma~0.195), mass conserved to 2.2e-14, finite/positive density. Locked
in as a fast CI regression (`python/tests/test_polar_diocotron.py`, 48x48, x>2). The QUANTITATIVE measurement of the
rate vs theory (l=3/4/5, O5, n=384/512) remains the ROMEO run.

**MERGED parallel wave (adversarial review + ci-full):**
- [x] **AMR capstone (iv) #175**: runtime facade honors per-block substeps/stride + substeps-aware step_cfl
  (mirror of `AmrSystemCoupler::step`; mono-block bit-identical via AmrCouplerMP routing; multirate test
  with neutralize-and-fail). MERGE-SAFE (4-lens review, 0 blocker).
- [x] **Lot A.5 core/ #173**: comment convention on `include/pops/core/` (comments only,
  no-code-change verified, ctest 96/96).
- [x] **Doc honesty #172**: DSL docstring (CoupledSource/add_pair paragraph) + classification of the 3
  physics bricks (AdvectionDiffusion/LangmuirMode/TwoFluidLinear = TEST/VALIDATION, not used by
  adc_cases; correction of a false "deprecated" comment).
- [x] **Polar diocotron test #174** (cf. milestone above).

**MERGED wave 2 (adversarial review + ci-full):**
- [x] **Lot B SystemFieldSolver #176**: extraction from `system.cpp` (elliptic + field drift) to
  `system_field_solver.hpp`; 1470->1155 lines; md5 bit-identity (5 paths); ctest 133/133.
- [x] **AMR capstone (vi) #179**: inter-species coupled sources on the runtime facade (+k/-k same cell,
  average_down covered cells #169); per-cell+global conservation; disable-and-fail. MERGE-SAFE.
- [x] **Lot B SystemStepper #180**: extraction of step/advance/step_cfl/step_adaptive/stride_due/
  run_source_stage/apply_couplings; 1155->1044; md5 bit-identity EXACT MATCH; ctest 134/134.
- [x] **Lot E explicit rejections #178**: tests locking the rejections of the polar path (transport/riemann/
  imex/eps-variable) + DSL roles. REVEALED the role-fallback gap -> fixed #181.
- [x] **Fix role-fallback #181**: `add_coupled_source` (DSL resolve path) STRICT -> throw if the block
  does not expose the role (before: silent fallback comp=0). role_index + named couplings unchanged.
- [x] **Schur-via-System coverage #182**: native test `System -> run_source_stage -> Schur` (review gap #180).

**SCIENTIFIC MILESTONE 2 (local probe, PRELIMINARY):** polar gamma(l) rate (96x96, hollow ring):
l=3 gamma~0.233 (x7.0), l=4 ~0.210 (x6.4), l=5 ~0.168 (x4.7); clean exponential growth, positive,
l-dependent (decreases with l), mass conserved ~1e-14 everywhere. QUALITATIVE; vs theory = ROMEO run.

VERIFIED INFRA NOTE: the Python tests run in CI via a find-glob `python/tests/test_*.py` in the
Release job (ci.yml l.148-150); `python/CMakeLists.txt` (foreach) is the SEPARATE ctest registration (MPI/Kokkos
jobs). TRACKED DEBT: teardown segfault on UNSTABLE polar profiles (PRE-EXISTING, outside #176).

**MERGED wave 3:**
- [x] **Per-block runtime IMEX AMR #184**: the runtime facade honors per-block time=imex (local implicit
  source, mirror of the AmrImplicitSourceStepper callback); stiff stable, disable-and-fail (explicit ->
  non-finite); all-explicit/mono-block bit-identical. (3 real NON-blocking findings -> handled by #185.)
- [x] **IMEX-hardening #185**: IMEX DIVERGENCE substeps>1 documented and assumed: the runtime facade
  SUB-CYCLES the IMEX (substeps=K: K applications of dt/K, sound and more CFL-safe) whereas the
  compile-time engine `AmrSystemCoupler` IGNORES substeps on its IMEX branch (a single bdt step). The two
  trajectories DIFFER for substeps>1; this is NOT a bit-identical mirror at substeps>1 (the claim
  "faithful mirror" corrected in a comment). substeps>1 test (DIFFERS load-bearing), mx/my/E observed
  directly. ctest 99/99.

**AMR SCOPING (real status, do NOT overstate): the runtime multi-block AMR is PHASE 1 ON A FROZEN
HIERARCHY, complete**: substeps/stride (#175), coupled sources (#179), local IMEX (#184/#185), on ONE
common NON-adaptive hierarchy (runtime regrid is refused for multi-block). STILL ABSENT:
the **union-tags regrid** (adaptive hierarchy = Phase 2), the **native multi-block DSL**, the **GLOBAL
Schur on AMR**. Do NOT write "complete AMR capstone".

**In flight (parallel wave, isolated branches, June 2026):** P0 AMR role-fallback fix (mirror #181);
design-only union-tags regrid; A4 radial flux test; unstable polar teardown segfault fix; A.5
physics/ comments. (Plus: A.5 `amr/` comments already in local WIP not committed, to preserve.)

**=== STILL TO DO (todo synthesis + CODEBASE_AUDIT, June 2026) ===**

FULL AUDIT (June 2026): ALL the lots of the section 10 plan of docs/CODEBASE_AUDIT.md are done or are
maintained invariants. No remaining audit lot.
- Lot A (doc + API truth): A.1 dsl.py #172, A.2 ARCHITECTURE table #161, A.3 SourceImplicit/CondensedSchur #194,
  A.4 application mentions neutralized #159, A.5 comment convention ALL folders (#173 core, #189 physics,
  #193 numerics, #196 mesh+coupling, #198 runtime, #200 amr).
- Lot B (runtime System): B.1 NativeLoader #151, B.2 SystemFieldSolver #176, B.2bis SystemStepper #180,
  B.3 SystemBlockStore #197, B.4 System::Impl thin orchestrator = result of B.1-B.3 (god-class P0 closed).
- Lot C (AMR): C.1 amr_reflux_mf split #170, C.2 amr_coupler removed #164, C.3 layout guard #141, C.4 stride #140,
  C.5 decision = multi-block runtime, C.6 union-tags regrid #199 => COMPLETE AMR CAPSTONE (Phase 1 frozen + Phase 2 regrid).
- Lot D (Schur/elliptic): D.1 EllipticOperator #163, D.2 TensorKrylovSolver = PURE solver (takes GeometricMG& op
  as argument, does not own the physics: VERIFIED INVARIANT), D.3 CondensedSchurSourceStepper = time/coupling stage
  via Split/set_source_stage, NOT model.source (VERIFIED INVARIANT + doc #194), D.4 Schur device-clean 7/7.
- Lot E (validation): E.1 stride tests, E.2 roles #178/#181, E.3 PolarMesh errors #168/#178, E.4 backend names #201,
  E.5 CI auto-discovery verified (find-glob).
NB: D.2 and D.3 are INVARIANTS TO MAINTAIN (do not let TensorKrylovSolver own the physics nor
CondensedSchurSourceStepper become a model.source), not tasks to do.

REMAINING (audit):
- [x] **Lot B.3 SystemBlockStore: DONE #197**: `class SystemBlockStore` (include/pops/runtime/system_block_
  store.hpp, 165 l) OWNS the registry `std::vector<BlockState>` (ex-Species) + index/find/copy_comp0/copy_state/
  write_state; Impl delegates. CONSERVATIVE decision: alias `using Species=...` + `std::vector<Species>& sp=
  blocks_.blocks` to NOT churn the templates already extracted (SystemFieldSolver/SystemStepper/native_loader).
  Bit-identical (verbatim helpers, insertion order preserved, member order == aggregate init: verified by
  4-lens adversarial review: 0 finding). system.cpp 1065->1015. ctest 140/140 unchanged. CI full green.
- [x] **Lot C.6 / AMR (viii) union-tags regrid: DONE #199 (COMPLETE AMR CAPSTONE)**: `AmrRuntime::regrid()`
  driven by the UNION of the tags (per-block predicate D1 + |grad phi| D4) -> tag_union (OR) -> all_reduce_or (MPI) ->
  Berger-Rigoutsos -> ONE shared fine layout -> prolongation of ALL blocks (including stride-held D3) + aux ->
  same_layout_or_throw. regrid BEFORE step (D2), 2 levels (D5). amr_regrid_finest refactored into reusable
  bricks (mono-block bit-identical). Facade unlocked (multi-block+regrid_every>0 no longer raises).
  regrid_every=0 BIT-IDENTICAL. Test test_amr_multiblock_regrid_union (24 assertions a-e + mass conservation).
  CI full green (including MPI). 4-lens adversarial review: 2 major findings RAISED then REFUTED (phi opt-in
  = documented scope limit, and the density default already captures the ring edge = density discontinuity;
  all_reduce path exercised by the MPI CI job). => PHASE 1 + PHASE 2 = COMPLETE AMR CAPSTONE.
  FOLLOW-UP: DONE #205: (1) phi predicate |grad phi| wired from the Python facade (AmrSystem.set_phi_refinement,
  default <=0 = disabled/bit-identical); (2) MPI np=1/2/4 parity test of the regrid (test_amr_regrid_mpi_parity).
- [x] **AMR (v) multi-block production DSL: DONE #195**: `add_compiled_model(AmrSystem&)`/`set_compiled_block`
  no longer raise on the 2nd compiled block; spec queue + lazy build at `ensure_built` (mirror of the native
  path). 1 block ALWAYS routed by AmrCouplerMP (mono-block bit-identical, dmax==0); N blocks via AmrRuntime.
  Footgun fixed (adversarial review BLOCKER): `AmrSystem.add_equation` REJECTS stride>1 / IMEX mask on
  the compiled .so path (the flat ABI does not transport them) and FORWARDS them on the native ModelSpec path.
  Tests: test_amr_multiblock_compiled (case F = IMEX stride=2 + per-component mask), test_amr_production_
  stride_reject.py. ctest 140/140. => PHASE 1 MULTI-BLOCK ON A FROZEN HIERARCHY COMPLETE (substeps/stride #175,
  coupled sources #179/#191, IMEX #184/#185, multi-block DSL/compile #195). Remaining Phase 2 = regrid (C.6).
- [x] **Lot A.5 comment convention: COMPLETE**: core/ #173, physics/ #189, numerics/ #193,
  mesh/+coupling/ #196, runtime/ non-amr #198, amr/ primitives #200. (runtime/amr_*.hpp already commented by
  #185/#195/#199.) RESIDUAL NOTE: 2 pre-existing non-ASCII typos (abi_key.hpp "batis", system_field_
  solver.hpp "ind+ependants" = broken em-dash) to fix in a dedicated ASCII pass.
- [x] **Lot A.3: DONE #194**: SourceImplicit (local, per cell) vs CondensedSchur (global, Schur) note
  in the docstrings python/pops/__init__.py + docs/SCHUR_CONDENSATION_DESIGN.md (no examples/
  folder in adc_cpp; the runnable examples are in adc_cases). Doc-only, 28 insertions.
- [x] **Lot E.4: DONE #201**: per-backend coverage audit. VERDICT: per-backend-path coverage was
  ALREADY complete (every C++ test outside the MPI-block runs Serial/Kokkos-Serial/Kokkos-OpenMP/MPI-np1; MPI tests in
  the MPI job; names already explicit test_mpi_*/_npN). Only gap = STALE doc (13 C++ + 10 py missing in the
  table) -> docs/BACKEND_COVERAGE.md brought up to date (AMR capstone section + recomputed balance), 0 test churn.
  New gap #6 noted: the multi-block capstone validated on 4 CPU backends, no device ROMEO harness yet.
- [x] **Role-fallback fix on the AMR side: DONE #191**: `AmrRuntime::add_coupled_source` (include/pops/runtime/
  amr_runtime.hpp, resolve lambda) hardened strict (mirror #181): unknown block/non-canonical role/canonical role
  not exposed -> throw naming block+role, no more silent fallback comp=0. Test tests/test_amr_coupled_source_
  role_strict.cpp (ctest #94): valid/absent/unknown role. Build 214/214, ctest 100/100. (verified 06/06)

REMAINING (scientific / outside audit):
- [x] **High-resolution Hoffart run (ROMEO/GH200): n=384 AND n=512 DONE**: -fPIC build resolved (Kokkos PIC
  /scratch_p/rmdraux/kokkos-install-pic); _pops module aarch64+CUDA built (pops_gpu_hires/adc_cpp/build-gpu-py).
  O5 WENO5/SSPRK3 on 1 GH200 (Kokkos Cuda np=1):
    n=384 (job 644126): l=3 -2.3%, l=4 -4.9%, l=5 +11.2%
    n=512 (job 647010): l=3 -0.38%, l=4 -8.4%, l=5 +16.0%
  l=3 CONVERGES cleanly toward the paper (-0.38% at n=512) = strong confirmation of the 2pi/rhobar normalization.
  l=4/l=5 NON monotone (l=4 -4.9->-8.4, l=5 +11->+16): sensitivity to the FIT WINDOW for the fast modes
  (the auto window of sweep.py is long [~5,~38] and bites into the saturation for the high-growth modes).
  FOLLOW-UP (measurement, not physics): re-fit l=4/l=5 on a dedicated EARLY exponential window (probe_fit.py
  already exists on ROMEO). The scientific core (normalization, l=3 exact) is ESTABLISHED.
- [x] **Schur PR6: MEASUREMENT (adc_cases, branch feat/normalization-and-schur-measurement)**: TEMPORAL
  effect of the Schur on a CARTESIAN stiff magnetized fluid measured. case schur_magnetized_cartesian/: stable explicit dt
  = 3.16e-4 (dt*omega_c=0.32, cyclotron bound) vs CondensedSchur theta=0.5 = 1.78e-1 (562x gain), theta=1.0
  = 3.16e-1 (1000x gain). The Schur removes the dt*omega_c<O(1) bound; the step approaches the transport step.
  set_magnetic_field cartesian OK; condensed stage via set_source_stage (same C++ CondensedSchurSourceStepper
  #126). (the polar path remains explicit-only; "polar Schur" = later feature). TO BE REVIEWED by the owner.
- [x] **CONSOLIDATED diocotron normalization (adc_cases, same branch)**: NORMALIZATION.md + diag/diag_polar_omega.py
  (gamma_norm = gamma_raw*2pi/rhobar; l=4 exact n=128/192). Codex's hoffart_euler_poisson_dsl case included.
- [x] **Perf: Poisson MG small-box under Kokkos: DONE (PR #206)**: `for_each_cell` runs in SERIAL (host
  loop) if box < 4096 cells UNDER a Kokkos backend with HOST execution (Serial/OpenMP); the Cuda/GH200 device
  path is strictly UNCHANGED (guard `if constexpr DefaultExecutionSpace==DefaultHostExecutionSpace`). Threshold 4096
  (overridable POPS_FOREACH_SERIAL_THRESHOLD). ROMEO PROFILE (Kokkos OpenMP, poisson phase ms/step): n=128/16th
  -91%, n=256/16th -80%, n=512/16th -55%, and 1-thread at/below baseline everywhere (the MG WORSENED with threads
  before). BIT-IDENTITY PROVEN: FNV hash of the final phi identical threshold=0 vs 4096 on n=64..512 at 1/8/16 threads
  (for_each_cell with no inter-iteration dependence: red-black GS colored; reduce NOT touched). 7 elliptic tests OK.

DEBT / DEFERRED (do not forget):
- [x] teardown segfault on UNSTABLE polar profiles (PRE-EXISTING, outside #176) -> FIXED #192:
  get/set_primitive_state bug + accessors (to_2d/to_3d) assumed a SQUARE domain (n*n); changed to
  (ny,nx) -> polar nr!=ntheta correct, cartesian bit-identical (ny==nx==n), AMR keeps square. CI full green.
- [x] A4: DONE #202: test_polar_mms_vr.cpp, DEDICATED polar MMS v_r != 0 (field (v_r=0.35, v_theta=0.6),
  polar manufactured source (1/r)d_r(r rho v_r)+(1/r)d_theta(rho v_theta)), convergence order 2.00 (limited
  by the radial face weighting). + #209: 3-var fluid MMS (v_r != 0) order 2.00. (A4 DONE. A polar TRANSIENT
  MMS v_r!=0 remains an OPTIONAL hardening, NOT priority: must not delay polar Schur
  nor the scientific reproduction.) MMS AUDIT (June 2026 workflow): verdict SUFFICIENT COVERAGE: order-2
  GENUINE/structural (radial metric weighting r_{i+/-1/2}, breaks high-order telescoping; NOT a bug);
  transient gap already covered by the CARTESIAN temporal validation (same SSPRK3 integrator reused in
  polar); NO new test required (2 optional non-blocking doc notes).
- [x] **finding 8: DONE #204**: `fab(0)` without `local_size` guard fixed (6 rhs_into/max_speed/
  advance closures in native_loader.hpp; guard `if (U.local_size()==0) return`; collectives outside closures -> no
  deadlock) + MPI test np=1/2/4 (test_mpi_system_gather_scatter). CI MPI job green. REMAINING: `/(2*dx)->*cx`
  at the last bit (NOT bit-identical, deferred); P7
  (implicit-total + DSL runtime params).

**Next steps:** see the "STILL TO DO" section above (single source of truth). Locked
invariants NOT to violate:
- **Schur PR6 = CARTESIAN only** (measurement of the TEMPORAL effect on a stiff magnetized fluid). The
  POLAR path is EXPLICIT-ONLY: the Schur stage is NOT wired there. Do NOT claim that "polar
  Schur" works; wiring it would be a later feature. (The Hoffart PAPER, for its part, does the
  COMPLETE stiff Euler-Poisson system with its Schur complement; the adc Schur stack #118-128 is its
  FV analogue: reproducing the paper's METHOD = a separate work item, not Schur PR6.)
- **RING EDGE = TRANSPORTED density discontinuity, NOT a wall.** Do NOT put a transport
  "wall" on it (physically wrong); valid levers = polar / high order / AMR.
- **Perf: NO optimization without a BEFORE/AFTER profile.** Target identified #165 (Poisson MG small-box
  under Kokkos), one optim per PR.
- **finding 8 (`fab(0)`)**: to do WITH the gather/scatter of the MPI port, NOT as a half-fix (a half-fix
  turns a clean crash into a false-silent). `/(2*dx)->*cx`: NOT bit-identical, last bit. P7 deferred.

## 0. RECOMMENDED public API (user entry point)

Two ways to write a model, both executed in native C++ (zero cell-by-cell loop
in Python on the performant path):

- **Compose native bricks**: `pops.Model(state, transport, source, elliptic)` -> assembles a
  `ModelSpec` from state / transport / source / elliptic bricks already compiled in the lib.
- **Write the model in formulas**: `pops.dsl.Model(...)` (symbolic DSL) then `m.compile(...)`. The
  RECOMMENDED backend is **`production`** (zero-copy native loader `add_native_block`, target
  MPI/GPU/AMR); mark it as the advised default of the DSL.

ADVANCED / LEGACY / TEST paths (NOT the main user path):
- `m.compile(backend='prototype')` (NumPy/host JIT), `m.compile(backend='aot')` (.so flat ABI):
  development / portability paths, doubled by `production`.
- `System.add_dynamic_block` (JIT) and `System.add_compiled_block` (AOT): low-level adders; prefer
  `add_native_block` (production) via the DSL facade.
- `pops.PythonFlux`: PROTOTYPING backend, pure HOST path (numpy, order 1 Rusanov periodic), OUTSIDE
  the GPU/MPI hot path. To quickly test a novel flux without recompiling, NOT for production.

## 1. "Extensible aux" work item (auxiliary fields beyond phi / grad)

Goal: a model declares/reads ADDITIONAL aux fields (magnetic B_z, electronic T_e)
without breaking the existing one, in bit-exact backward compatibility (`n_aux` default = 3 -> strictly identical).

- [x] **Inc. 1 - Read**: `pops::Aux` + `B_z` (comp 3), `kAuxBaseComps=3`, `aux_comps<Model>()`,
      `load_aux<NComp>`. The named functors read `load_aux<aux_comps<Model>()>`. (#24)
- [x] **Inc. 2 - Coupler population** mono-block: `fill_bz`, aux allocated at `aux_comps<Model>()`. (#24)
- [x] **Inc. 3 - SystemAssembler population** multi-block (aux = max over the blocks). (#25)
- [x] **Inc. 4 - `CompositeModel::n_aux`** = max of the bricks; `aux_comps` moved to
      `physical_model.hpp` (contract header). (#26)
- [x] **Inc. 5 - runtime `System`**: `ensure_aux_width` + `set_magnetic_field` (Python binding
      modeled on `set_epsilon_field`). Native path `add_compiled_model` complete. (#29)
- [x] **Inc. 6 - DSL**: emits `n_aux` when a formula reads `aux('B_z')` (`AUX_CANONICAL`). (#30)
- [x] **Inc. 7 - JIT path** `add_dynamic_block`: virtual `IModel::n_aux()` + marshaling
      `aux_ncomp_` -> B_z transported, Python end-to-end. (#32)
- [x] **Inc. 8 - compiled AOT path** `add_compiled_block`: the ABI `compiled_block_abi.hpp`
      now transports the aux width (B_z/T_e), symmetric to inc. 7 on the `extern "C"` ABI side;
      DSL B_z model driven 100% from Python via `compile_aot`. (#46)
- [x] **T_e - 2nd DERIVED extra field**: T = p/rho computed by the `System` from a designated fluid
      block at each solve (comp 4, `set_electron_temperature_from`, recomputed in `solve_fields`,
      not user-supplied like B_z). Validates the generalization to 2 aux fields. (#35); read on the THREE
      dynamic paths: native (#35), AOT (test #50), JIT (marshaling completed #51, was at 0).
- [x] **AMR / implicit**: `load_aux` width-aware on `advance_amr` and the implicit stepper
      (extensible channel on the AMR path); bit-identical for a base model. (#42)
      + B_z populated PER LEVEL in the system AMR coupler (each level receives its B_z). (#53)

## 2. "Generic elliptic EPM" work item (composable elliptic operator)

- [x] Variable permittivity `eps(x)`: `GeometricMG::set_epsilon` + `System::set_epsilon_field`
      (Python binding) on `master`.
- [x] Named `EllipticProblem` / `FieldPostProcess` (coeff, BC, nullspace, convention `E = -grad phi`).
- [x] GENERIC system Poisson right-hand side (sum of the bricks' `elliptic_rhs` per block). (#43)
- [x] SCREENED / Helmholtz elliptic operator `div(eps grad phi) - kappa phi = f` (GeometricMG + binding). (#44)
- [x] ANISOTROPIC elliptic operator `div(diag(eps_x, eps_y) grad phi)`: GeometricMG core (#52),
      eps_x(x)/eps_y(x) exposed to the runtime System + Python (#56), cut-cell test + anisotropic MMS order 2 (#55).
- [x] cx (`/(2*dx)` -> `*cx`): CLOSED NOT WORTHWHILE (evaluated June 2026). The 3 remaining divisive sites
      (amr_coupler_mp:307-308, spectral_coupler:113-114, system_field_solver:429-430) are CPU loops
      OUTSIDE THE HOTSPOT (1x per solve, after the MG V-cycle which dominates 96-99% of the time #165/#206), memory-bound,
      and division by CONSTANT is already optimizable by the compiler. Breaking the IEEE-754 bit-identity for
      a sub-ms NON measurable gain is not justified. The GPU sites (field_postprocess, condensed_schur,
      schur_condensation) are ALREADY in `*cx`. Decision: leave the 3 sites as-is. No PR.

## 3. Architecture hardening (`docs/archive/ROADMAP.md`, ARCHIVE: no longer an active doc)

- [x] **Unified AMR engine**: `advance_amr(LevelHierarchy&)`, `FluxRegister`, `CoverageMask`,
      `RegridPolicy` (`amr_regrid_finest`) promoted to real types. `PatchRange` promoted (coarse
      footprint `[I0..I1]x[J0..J1]` of a fine patch; dedup of 6 inline footprint computations, arithmetic
      `(hi-1)/2` historical preserved hence bit-identical). (#80) REMAINING: the border routing of
      `CoarseFineInterface` and `SubcyclingSchedule` (still inline in the `subcycle_level_mp` recursion)
      and folding the `amr_step_*` family (`amr_step_2level_multipatch` replicate-coarse vs
      `amr_step_multilevel_multipatch` recursive: distinct invariants, not fused to keep the
      bit-identity; `advance_amr` already serves as the unified facade).
- [x] **Explicit memory API**: `for_each_cell_reduce_{sum,max}`, `sum`/`norm_inf` done.
      `sync_host()` / `sync_device()` explicit posed on the `for_each.hpp` seam + `MultiFab`
      methods: encode the residence intent. Under unified memory (`Kokkos::SharedSpace`)
      `sync_host()` is a targeted `device_fence()`, `sync_device()` a no-op (bit-identical).
      Scaffolding for the future NON unified path (separate buffers + deep_copy).
- [x] **Ghost families**: `fill_physical_bc` / `fill_boundary` / `mf_fill_fine_ghosts` separated;
      the coarse-fine raised into a first-level named helper `fill_cf_ghost_cell` (constant interp
      in space + linear in time, factors the 3 bodies of `mf_fill_fine_ghosts_t/_multi/_mb`). (#80)
- [x] **VariableRole**: inter-species couplings by role (#18) + the DSL-generated brick
      declares its VariableRole and the runtime resolves the variables by role with index fallback. (#40)
      `block_names()` now reads the C++ block registry (sees JIT/AOT, no longer only
      `add_block`) (#72); the `.so` blocks (JIT `add_dynamic_block` + AOT `add_compiled_block`)
      transport their names/roles/gamma via OPTIONAL ABI symbols, with fallback for old `.so`
      and a guard on the length of `names=`. (#75)
- [x] MPI distributed multi-patch AMR (2 and N levels), thin `CouplingPolicy`, core numerical
      validation suite, elliptic splitting (operator / solver / problem).

**`AmrSystem` -> `System` parity order** (close the AMR-path gaps in this order):
- [x] **Gap 1 - facade flip**: Python FACADE rejection of HLLC/Roe + primitive reconstruction raises on
  `AmrSystem.add_equation` (the C++ engine `add_compiled_model(AmrSystem&)` already supported them; rejection
  PURELY facade). DONE.
- [x] **Gap 2 - IMEX on AMR**: `mf_apply_source_treatment` selects forward-Euler vs
  `backward_euler_source` (named functor `BackwardEulerSourceKernel`) via a runtime bool; parity
  dmax=0, conservation 1e-15, explicit default bit-identical, stiff tests. (#132)
- [x] **Gap 3 - native multi-box** + **Gap 4 - multi-species (capstone)**: MERGED into the multi-block
  AMR capstone (section 15, doc #139). Observation: the multi-block engine ALREADY EXISTS (`AmrSystemCoupler`);
  remaining is to expose N blocks via the runtime facade + add `regrid` to the coupler. AWAITING GO.

## 4. GPU (GH200) - integration

- [x] Components validated SEPARATELY and bit-identical to the CPU on GH200: mono-grid System, AMR
      field ops, multi-GPU MPI halos, AOT backend of a DSL model, `load_aux<4>` (B_z device).
- [x] Multi-box MPI parity of the compiled path (`add_compiled_model` / `make_block`, np=1/2/4
      bit-identical) MADE PERMANENT as a regression test in the repo. (#39)
- [x] `add_compiled_model` wired on the `AmrSystem` side (multi-level counterpart of the compiled path). (#45)
- [x] **INTEGRATED validation** `AmrSystem` + MPI + GPU in a single run: DONE on GH200 (`exec=Cuda`,
      euler_poisson compiled on AMR 128->256 multi-patch, fine patches distributed 1 GPU/rank). np=1/2/4
      BIT-IDENTICAL (csum identical to 17 digits, dmax=0), `crossrank_spread=0`, mass conserved
      (dm=0). CPU/MPI regression test `test_mpi_amr_compiled_parity` (CI) + GPU harness
      `python/tests/gpu/amrmpi_integrated.cpp`. Doc `docs/GPU_RUNTIME_PORT.md` (phase 10). (#48)
- [x] **Device validation round 2** of the post-#48 features (GH200, `exec=Cuda` vs Serial oracle,
      bit-by-bit `dmax`): T_e `load_aux<5>`, EPM Helmholtz, EPM anisotropic, B_z per AMR level ->
      all **`dmax=0`**; harness `python/tests/gpu/gpu_{aux,epm,amr_bz}_validate.cpp`. (#61)
- [x] **B_z multi-box distributed multi-GPU (#59)**: FUNCTIONAL on device (B_z per box/level read,
      `bz_bad=0`, `cmax` bit-identical `dcmax=0`) but NOT bit-identical on the global sums
      (`dmass~1e-15`, `dcsum~3e-13`): effect of FMA REDUCTION ORDER (distributed coarse). Not a bug.
- [x] **Device limit (a) LIFTED**: `System::add_compiled_model` is device-clean on Cuda. The
      wrapper closures of `block_builder.hpp` (`BlockRhsEval`, `Advance*`, `RhsInto`, `MaxSpeed`,
      `PoissonRhs`) moved from extended lambdas to NAMED FUNCTORS (same recipe as the
      kernels) -> no more Cuda segfault, A==B bit-identical parity on GH200 (`dres=0`, exit 0;
      before: SIGSEGV exit 139). Compiled DSL models usable on GPU via the `System` facade. (#64)
- [x] **Device limit (b) LIFTED**: the `AmrSystemCoupler` facade now instantiates + compiles under
      nvcc. The probe of the `CoupledSystemLike` concept (`s.for_each_block([](auto&){})`, generic lambda
      in unevaluated context that the nvcc/EDG frontend refused -> coupler CTAD impossible) was
      moved to a NAMED FUNCTOR `detail::ForEachBlockProbe` (same recipe as (a) / #64). Validated GH200
      (job 637927): `CUDA_BUILD_OK`, `exec=Cuda` OK, U(coarse+fine, 2 blocks) bit-identical to Serial
      (`dmax=0`), before/after confirmed (lambda probe put back -> nvcc failure). Permanent harness
      `python/tests/gpu/gpu_amrsys_facade_validate.cpp` (+ gpuval2 CMake). Host unchanged (72/72 ctest).
- [ ] **Full-device perf: RESEARCH/ROMEO-NEEDED (June 2026 scope, cf docs/RESEARCH_BACKLOG.md)**: the
      coarse replication is a CHOICE (the distributed mode measures 3-5x SLOWER: MG halo latency-bound on
      small boxes). Next step = ROMEO multi-GPU profile per V-cycle level; DECISION GATE: latency
      >50% -> hybrid MG; otherwise replication = good tradeoff -> close. NOT auto-completable. [the integrated run DOES NOT SCALE (coarse REPLICATED -> redundant coarse Poisson/transport
      per GPU; only the fine patches distribute). Mode `replicated_coarse=false`
      (distributed coarse) exists in `AmrCouplerMP` but degrades the MG and is not wired in `AmrSystem`:
      real AMR strong-scaling work item. + zero-copy AOT parity on device (without host bounce).

## 5. Magnetized physics

- [x] Combined E+B Boris push (`tfap_boris`, exact cyclotron, ExB drift without secular growth).
- [ ] **Tensor AP reformulation under strong field: RESEARCH (June 2026 scope, cf docs/RESEARCH_BACKLOG.md)**
      : the condensed Schur ALREADY handles the strong field unconditionally (stable); the AP would be an
      EFFICIENCY gain only. Next step = math study (asymptotic expansion + 1D toy), no code. NO-GO
      if non-local operator incompatible with roles/DSL. NOT auto-completable.

## 6. Hoffart reproduction (arXiv:2510.11808) - APPLICATION, `adc_cases` side

- [x] **RESOLVED T2+T3 (June 2026): the "-95% deficit" was a METROLOGY ARTEFACT, not the solver.**
      The cartesian full system-schur REPRODUCES the paper once the measurement is done in the right units:
      l=3 -9.1%, l=4 -1.9%, l=5 +0.04% (n=96), and the error CONVERGES toward 0 (n=256: the 3 modes <1%).
      CAUSE: (1) clock: gamma_paper = gamma_raw_sim * 2pi/rhobar (cyclic->angular conversion of the
      drift clock, EXACT and mode-independent, verified <0.5% against the Petri eigenvalue;
      diag/petri_eigenvalue.py: Wd=2pi*omega_d reproduced, Wd=omega_d=1 gives targets/6.2832). (2) window:
      the paper windows (time T_d) were fitted on the raw SIM time (transient); they must be
      mapped t_sim=2pi/rhobar*t_paper. KEY: alpha=beta^2/rho_max and omega=beta^2 SIMPLIFY
      (alpha/omega=1) so the full advects the field of the normalized ExB reduced; full==reduced at ~2% in established
      window. l=3 decomposition closed: 0.0312 x 3.20(window) x 6.28(2pi) x 1.23(grid) = 0.772.
      DELIVERED: adc_cases #32 (T3 code run.py/results.py: paper_to_sim_time_window + gamma_to_paper_units,
      report gamma_raw_sim AND gamma_paper_units), #38/#39 (README = full tutorial install->run + figures
      + GIF + commented DSL, pedagogical model.py, TUTORIAL merged); adc_cpp #251+#256 (HOFFART_FIDELITY
      corrected: Strang not Lie, 2pi applies to the full; premise "no 2pi for full" REMOVED). Verified by
      4-lens adversarial workflow (confirmed-with-caveats). CAVEATS: PARTIAL metrology (residual
      ~0-9% = cart edge + resolution + window roll-off); l=5 sensitive to the window (+-27-29%). The [~]
      below (verdict "geometry/structural", "no 2pi for full", "Lie not Strang") are SUPERSEDED.
      BRANCHES: #238 (spatial playbook, mine) closed+deleted (superseded). Remain open, outside my
      scope (sibling-agent WIP): #253 (DSL optflags, CI in progress -> auto-merge if green), #229 (polar
      multibox, behind 31), #109 (transport-wall, behind 181), #232 (AMR-Schur design, draft). Merging them =
      rebases/finishing that belong to those work items, not to the Hoffart repro.

- [~] **(SUPERSEDED by RESOLVED T2+T3 above: kept for history)** **SESSION T5 + ABI + GEOMETRY (June 2026): critical path = GEOMETRY, signal in progress**:
      Phase 0 audit (5 agents, docs/HOFFART_FIDELITY.md + HOFFART_STEP_SEQUENCE.md): geometry = DOMINANT
      suspect of the complete model deficit (analytical rate = f(l, omega_d, r0, r1, R) only; the
      secondary suspects are negligible/minor). RESOLVED: |Omega|=beta^2=1e12 correct (paper l.1082),
      omega_d=1, raw slope of the complete DIRECTLY comparable to 0.772/0.911/0.683 WITHOUT 2pi (the 2pi/rhobar
      belongs only to the reduced ExB path). Analytical Petri eigenvalue (#12 adc_cases) confirms the
      targets to <0.13% in omega_d units, without 2pi.
      MEASURED cartesian BASELINE (n=256 AND n=384, raw, paper windows): l=3 -95%, l=4 -94/-93%, l=5 -82%,
      RESOLUTION-INDEPENDENT -> structural (geometry), not a lack of mesh. Validation table up to date
      on adc_cases/master (full-system-schur cart-square lines filled).
      T5 CUT-CELL STACK MERGED (adc_cpp master): #218 shared cut_fraction, #222 conservative EB transport
      (MMS order ~2, mass 3e-15, small-cell clamp), #224 facade wiring (staircase+cutcell, live routing
      verified, Lie+Strang). Before: T1 #217 generic Strang, T2 #216 mask contract, #214 polar Lorentz,
      #215 polar Schur facade.
      NATIVE ABI BUG found + fixed (#225 adc_cpp, in CI): compile_native froze c++23 whereas the loader
      _pops is c++20 under Kokkos (CUDA 12.x without c++23) -> add_native_block rejected -> NO native GH200 run
      possible (CI invisible: model compiled at runtime). Fix = the model follows loader_cxx_std() + CI test
      add_native_block under Kokkos (Kokkos -fPIC). Also fixed: ssprk3 not supported in native
      (add_native_block=explicit|imex) -> case in ssprk2 (aligned baseline); matplotlib rendering optional
      (DISC #14 adc_cases) -> Hoffart case EXECUTABLE in native GH200.
      MIGRATION: cases hoffart_euler_poisson_dsl + schur_magnetized_cartesian versioned on adc_cases/master
      (#13, reproduction-candidate), nothing left on the side branch. Docs honest (#10 diocotron, #221/#223 doc
      refonte ASCII-ified pending after #225).
      GEOMETRY SIGNAL LANDED (ROMEO 647507): VERDICT (docs/HOFFART_GEOMETRY_VERDICT.md): square = staircase
      = cutcell give the SAME rate (-95/-95/-82%, diff ~1e-11 = rounding). The cut-cell/disk mask at the
      R=16 EDGE DOES NOT CORRECT the rate (the ring lives at r0/r1, deep inside; the corners outside R are
      inert; the Poisson wall already imposes the disk). MOREOVER the deficit is RESOLUTION-INDEPENDENT (n=256 =
      n=384) -> it is NOT a mesh diffusion (which would decrease with n) NOR the outer edge -> suspect =
      STRUCTURAL (normalization/scale/coupling of the complete system-schur; raw rate ~0.037 = plateau). The
      reduced (resolution-DEPENDENT, recovered in polar l=4 exact) behaves differently. The T5 cut-cell
      remains a real capability (wrong tool for THIS blocker). NEXT STEP = normalization/structure diagnostic
      of the complete (compare complete system-schur vs reduced ExB on the same setup), NOT big GPU nor a new
      geometry. NO reproduction of the complete model claimed.

- [x] **M1 - NORMALIZATION FOUND (June 2026)**: `gamma_norm = gamma_raw * 2pi/rhobar` (project factor).
      Verified on the POLAR ExB path (paper scale 6:8:16, top-hat, WENO5/SSPRK3, n=128):
      l=4 EXACT (0.913 vs 0.911), l=3 +26% (0.97), l=5 -29% (0.48). The +-29% scatter is the sensitivity to
      the fit window (intrinsic to a numerical measurement of the diocotron rate, cf. project cartesian
      sweep). KEY DIAGNOSTIC: `gamma_raw(sim) ~ Im(omega)_eigenmode` directly (0.155 vs 0.123) -> the polar
      sim runs in ExB-natural time units, NO beta re-scaling needed. The rotation at the
      inner edge r0 is ~0 (no charge trapped in r<r0) -> a "local rotation" normalization
      does NOT work; it is indeed the GLOBAL factor 2pi/rhobar. WHY Codex (cartesian-Schur) gives 0.035
      and not 0.77: (a) its runner OMITS the 2pi/rhobar factor (x2pi -> 0.22) AND (b) its CARTESIAN grid
      diffuses the ring edge (polar gamma_raw 0.155 ~ 4.4x its 0.035 at comparable resolution). Polar
      + 2pi/rhobar reproduces the REDUCED diocotron rate (ExB-scalar, Petri benchmark); NOT the complete Hoffart model (cf. correction below). RESOLUTION-ROBUST: at n=192 the
      three modes fall in [0.87,0.97] (l=4 EXACT at both resolutions; l=5 goes from -29% n=128 to
      +27% n=192 as its window tightens -> the scatter is fit-window SENSITIVITY,
      NOT a physics deficit). Reproducible diag: `/tmp/diag_polar_omega.py`.
- [~] **(SUPERSEDED by RESOLVED T2+T3 at the top of the section: "Lie not Strang" and "no 2pi for full" corrected; kept for history)** **COMPLETE Hoffart MODEL: NOT yet validated at paper fidelity (honesty correction, Codex June 2026)**:
      WARNING: the "-0.38% l=3 n=512" cited above is the REDUCED ExB-SCALAR DIOCOTRON (adc_cases/diocotron/,
      models.diocotron, CARTESIAN + circular wall, ROMEO sweep ~/pops_gpu_hires/.../diocotron/sweep.py). It validates
      the NORMALIZATION 2pi/rhobar + the method on the REDUCED MODEL (standard Petri benchmark): NOT the complete
      Hoffart system. The COMPLETE MODEL (continuity + momentum + Lorentz + isothermal pressure + Gauss via Schur)
      EXISTS = adc_cases/hoffart_euler_poisson_dsl, on the branch feat/normalization-and-schur-measurement (NOT master).
      CODE AUDIT (workflow audit-hoffart-branch, June 2026): Codex's 5 statements CONFIRMED in the code:
        (1) SPLITTING = LIE not Strang (system_stepper.hpp:104-121: T(dt) then S(dt) full step; strang_step exists
            but dead code on the runtime side; no --time-scheme option). theta=0.5 = Crank-Nicolson of the source SOLVE, NOT
            a Strang half-step. => difference of splitting ORDER (Lie O(dt) vs Strang O(dt^2)), not just FV-vs-FE.
        (2) amr-imex EXPERIMENTAL: not CondensedSchur (cell-local IMEX), initial momentum NULL (build_amr calls
            only set_density, never drift_velocity_from_potential), cartesian AMR transport.
        (3) POLAR Schur strictly MONO-RANK: 3 HARD guards throw if n_ranks!=1 (#215 wires the facade only).
        (4) ALL paper PARAMS conformant (allMatch): r0=6,r1=8,R=16,rho_min=1e-6,rho_max=1,beta=1e6,alpha=1e12,
            delta=0.1,modes 3/4/5,tf=10. Geometry = disk emulated by a circular Poisson wall on a square box (corners
            at rho_min); theta_T=0 (cold limit, paper gives no value); dt=1e-3 fixed (paper adaptive).
        (5) docs OVERCLAIMS CORRECTED (commit 29deabe on the branch): NORMALIZATION.md/README.md/model.py no longer say
            "reproduces the paper" for the REDUCED; + mandatory validation table (PENDING for full-Hoffart,
            reduced-ExB lines labeled). diocotron/README.md title corrected via PR adc_cases #10 (to re-read).
      QUANTITATIVE RUN IN FLIGHT (ROMEO ~/hoffart_full, jobs 647393 n=256 / 647394 n=384, t_end=10, run.py paper
      geometry): preliminary SIZING l=3 = -45 to -67% vs paper (FAR, unlike the reduced -0.38%) BUT too short.
      VERDICT CLOSED at their return. Do NOT claim reproduction of the complete before. Capabilities there (Schur cart #118-128,
      polar Schur #210/#212, polar fluid #209); remaining is the validated ASSEMBLY + (if match) Strang + resolution increase.
      (Propagated error corrected: the frontier workflow had attributed the reduced -0.38% to the complete model: false.)
- [x] **Discrete conservation cartesian-fluid-Schur: DONE #207**: mass tests (machine, closed domain),
      momentum symmetry (machine), momentum impulse (physics O(dt) convergent), E>0/p>0 (3 limiters).
      Honest FV-vs-FE note. Discovery: Dirichlet leaks mass ~1e-2 via Foextrap (BC artefact, not scheme).
- [~] **Hoffart REMAINING: DECISIONS MADE + IN PROGRESS (June 2026)**: the owner has decided: target l=4/5
      to +-2% (=> re-fit + replay), sharp 2D figure REQUIRED (=> Path A pursued), all in parallel.
      IN FLIGHT:
        - #208 doc CONSERVATION_SUMMARY (FV vs FE): CI.
        - #209 Path A STEP 1 MERGED: polar isothermal fluid transport (IsothermalFluxPolar + geometric centrifugal
          source S_geom verified EXACT by independent sympy derivation; ExB-polar + cartesian
          bit-identical via concept-gate; rotational equilibrium order 1.99, MMS 2.00, mass 1.2e-15). 4-lens
          adversarial review = 0 real issue (6 "findings" = positive confirmations). -- CI green, MERGED (675e587).
        - early-window re-fit l=4/l=5 on ROMEO (job 647356) -- HONEST RESULT: the early window
          DOES NOT REACH +-2% and is NOT a pure fit-window. n=512: l=3 +4.86% (the LONG window was
          better: -0.38%), l=4 +5.09% (better than -8.4%), l=5 +18.10% (~ unchanged vs +16%). The modes
          prefer DIFFERENT windows; l=5 stays +16-18% whatever the window => REAL
          DEVIATION (insufficient resolution for the fast high-gradient mode, and/or delta=0.1
          perturbation slightly non-linear), NOT a measurement artefact. CONCLUSION: the 2pi/rhobar normalization is
          PROVEN by l=3 (-0.38% long window); +-2% on l=4/l=5 would require n=768/1024 (more GPU)
          and/or smaller delta (more linear regime). Data: /scratch_p/rmdraux/early_fit_647356/.
      HONEST STATUS OF THE RATES (do not oversell): l=3 VALIDATES the normalization (-0.38% n=512); l=4 SENSITIVE
      TO THE fit window; l=5 ROBUST DEVIATION 16-18% whatever the window; CAUSE NOT YET
      IDENTIFIED (non-linearity? resolution/diffusion? model/geometry/diagnostic?). NO +-2% promise
      before results.
      DISCRIMINATING CAMPAIGN DONE (ROMEO job 647366, 12 runs): VERDICT: NON-LINEARITY dominant. Robust
      trend: gamma_num rises toward the analytical when delta decreases (l=5 n=512: delta=0.10 -71%, 0.05 -37%,
      delta=0.025 -> +0.17% QUASI-EXACT; l=4 n=512: -64/-39/-22%). => the l=5 deviation +16-18% was a
      NON-LINEAR ARTEFACT (delta=0.10 too large). Lever = smaller delta, NOT n=768 (resolution not the
      blocker). METHOD CAVEAT: the delta=0.10 campaign (l=5 n=512=0.20, long window in the saturation) DOES NOT
      MATCH the original sweep (0.81) -> the campaign did not take the SAME window; the delta trend is
      valid but the campaign<->original absolute comparison is biased by the window. REMAINING: (1) reconcile
      the window (redo with the EXACT observable/window of the original); (2) investigate the l=4 residual (-22%
      at delta=0.025, inverted n trend). NO +-2% promise. Data: /scratch_p/rmdraux/647366/. [old note:] modes l=4,5 x n=256,512 x
      delta=0.10/0.05/0.025, SAME windows + SAME observable (no opportunistic adjustment). Reading:
      error DECREASES with delta -> non-linearity; DECREASES with n -> resolution/diffusion; STABLE ->
      model/geometry/diagnostic problem. n=768 NEXT, only for the configs that justify it.
      FOLLOW-UP: Path A STEP 2a = iterative TENSOR polar elliptic operator MERGED #210
      (BiCGStab + RadialLine precond, MMS order 2, clean review); STEP 2b = polar Schur source stage MERGED #212 (dt gain 2000x); STEP 2c = facade wiring -> PR6 (cross-tensor elliptic +
      polar Schur stencils); polar fluid diocotron demonstrator case (the sharp 2D figure); validation
      table final; (optional) Strang order 2. Detailed roadmap: docs/FULL_MODEL_VALIDATION_ROADMAP.md.
      Remaining open question: formal FE structure-preservation proof required, or do empirical O(dt^2)
      tests suffice (the scheme is FV, momentum not exact by construction)?
- [x] **M2 / M2b**: AMR on the ring edge (triples the rate at equal base) + multi-level Poisson.
- [~] Resolution increase / convergence toward the analytical rate: DONE (n=384/512 GH200, l=3 -0.38%).
      SAMRAI integration = EXTERNAL-BIG, DEFERRED (adc's homemade AMR is capstone-complete and covers the
      science path; cf docs/RESEARCH_BACKLOG.md for the reopening criterion). NOT auto-completable.

## 7. Cleanup / consolidation (wave P1/P2, June 2026)

Hardening pass AFTER the aux/EPM/GPU work item: no paper feature (SAMRAI, disk
domain, Schur EPM, multi-block AMR, Hoffart repro): all DEFERRED. One PR per block, CI green.

- [x] **ctest safeguard**: the 10 Python tests of the aux/EPM/roles work items registered in ctest
      before the cleanup (regression net). (#69)
- [x] **AmrSystem**: EXPLICIT rejection of un-wired parameters + dedup of the two AMR build paths. (#71)
- [x] **Honest docs**: README / ARCHITECTURE / ALGORITHMS aligned with the real status, off-nav notes
      archived (without touching the living backlog `todo.md` nor the `ROADMAP`). (#70)
- [x] **Dead headers**: 6 headers at 0 include / 0 test documented as un-wired API (0 deletion). (#73)
- [x] **Dedup aux helpers**: `derive_aux_bc`, `fill_bz`, `wall_active` factored
      (`coupling/aux_fill.hpp`, `runtime/wall_predicate.hpp`). (#74)
- [x] **`block_names()` from the C++ registry** (sees JIT/AOT, no longer only `add_block`). (#72)
- [x] **`.so` block ABI**: names / roles / gamma transported by OPTIONAL ABI symbols on the two
      paths (JIT + AOT), fallback for old `.so`, guard on the length of `names=`. (#75)
- [x] **adc_cases**: package + CI manifest, reinforced validations (no important validation
      masked by the manifest). (adc_cases #2, #3)

### Parallel wave (June 2026, 4 work items in separate PRs, disjoint write-sets)

- [x] **maintainable adc_cases package**: `recipes.py` (system recipes separated from the mono-species
      models), `common/native.py` (out-of-source JIT compile + ABI key + explicit symbol errors),
      guarded import preamble (no more unconditional `sys.path.insert`), `two_fluid_ap`
      builds in `out/<case>/build/` (git-ignored). (adc_cases #4)
- [x] **DSL facade `m.compile(backend=prototype|aot|production)`**: routing by intent coupled to
      the correct System adder; preserves names/roles/gamma/n_aux/B_z/T_e; `require_metadata=True` raises an
      explicit error instead of the silent fallback. PURE-PYTHON (no binding change). At #79, `production`
      was still an alias of `aot`; since #85 it is a DISTINCT zero-copy native backend
      (`add_native_block`), cf. "Identified follow-up (done)" below. (#79)
- [x] **Hardened AMR engine**: `PatchRange` + coarse-fine helper `fill_cf_ghost_cell` promoted,
      bit-identical (Serial 73/73, MPI 94/94 np=1/2/4). (#80) See section 3.
- [x] **Paper roadmap audit**: `docs/PAPER_ROADMAP.md` (4 baskets: current API / production DSL
      / disk domain FV / multi-block AMR + advanced EPM). Blocker = cartesian ring edge (cut-cell only serves
      Poisson, not the transport). No implementation. (#78)

### Identified follow-up (done)

- [x] **`CoarseFineInterface` + `SubcyclingSchedule` routing** extracted from `subcycle_level_mp` into named
      types, bit-identical (Serial 74/74, MPI 95/95 np=1/2/4). (#82)
- [x] **Distinct `production` DSL backend**: zero-copy native loader `add_native_block` (inline
      `add_compiled_model<ProdModel>`, ABI-key gate, `POPS_EXPORT` symbols), `production` no longer points
      to `aot`. ELF portability (promotion of `_pops` to global scope). CPU parity bit-identical to
      `add_block`. (#85)

## 8. Ideal ADC Plan: write the model in Python, run in native C++

Goal: the user writes the equations in Python (symbolic DSL), ADC generates/compiles a NATIVE
C++ brick wired into `adc_cpp` like a hand-written model; no cell-by-cell loop
in Python on the performant path. Three backends: `prototype` (NumPy/host), `aot` (.so flat ABI),
`production` (zero-copy native, target MPI/GPU/AMR).

- [x] **Step 1 - stable `dsl.Model` API** on top of `HyperbolicModel` (facade; `m.flux` declarator
      / `m.eval_flux` evaluator; `m.primitive_vars(**kwargs)`). (#89)
- [x] **Step 2 - `param` + `CompiledModel` + clean errors**: `Param` compile-time named (runtime =
      phase E); `CompiledModel` (backend/adder/so_path/names/roles/gamma/n_aux/params/ABI key);
      `add_equation` (dispatch `ModelSpec` vs `CompiledModel`), `FiniteVolume(riemann=)`, `run`. Explicit
      errors (role/param/backend/flux). (#89, #90 substeps ModelSpec fix)
- [x] **Step 3 - real `production` for `System`**: zero-copy native loader. (#85)
- [x] **Step 4 - `adc_cases` demonstrator cases**: `diocotron_dsl` (ExB in formulas, == `models.diocotron`
      BIT-IDENTICAL) + `two_species_dsl` (electrons+ions, per-block time, Poisson `sum q_s n_s`), `production`
      backend, light CI validation (adc_cases #7) + `magnetic_isothermal_dsl` (magnetized isothermal,
      B_z driven from Python via `set_magnetic_field`, Lorentz oracle, `aot`==`production` bit-identical in
      Linux CI, adc_cases #9). The 3 DSL demonstrators run in CI.
- [x] **Step 5 - `production` -> `AmrSystem`** (Phase D): `AmrSystem::add_native_block` +
      `target="amr_system"`; bit-identical parity to `add_compiled_model(AmrSystem&)`. VALIDATED CPU/CI
      (test_amr_native_loader dlopen, Release+MPI+Kokkos green). (#92) **WENO5/Rusanov/conservative** wired
      on the native AMR path (parity `add_native_block`==`add_compiled_model`==`add_block`, dmax=0, #105).
      Remaining limits (cf. AmrSystem parity order, section 3 + capstone section 15): mono-block AMR /
      no multi-species; **LOCAL IMEX source OK (Gap 2 #132)** but no GLOBAL Schur on AMR; native multi-box
      not wired on the facade side. (HLLC/Roe/primitive reconstruction: **Gap 1 LIFTED on the facade side**.)
- [x] **Step 6 - MPI/GPU validation of the `production` path**: **np=1 GPU VALIDATED on GH200.** The device
      crash in `solve_fields()` was due to inline extended `POPS_HD` lambdas (elliptic/mesh kernels
      `copy_shifted`/`fill_boundary`/MG, first cross-TU instantiation -> null nvcc kernel stub in
      Release without `-g`); converted into NAMED FUNCTORS (#97, same recipe as #64). GH200 proof (ROMEO job
      640236, Release without `-g`): `geometric_mg` AND `fft` Cuda np=1 exit 0 (were 139),
      compute-sanitizer 0 error, `dmax_abs` Cuda-vs-Serial = 5.0e-13 (MG) / 1.3e-15 (FFT): WITHIN
      TOLERANCE (FMA reassociation of the Kokkos reductions), CPU bit-identical. **MPI `solve_fields`
      np=1/2/4 VALIDATED CPU/CI**: the mono-box / `fab(0)` bug called without a `local_size()` test
      (rank without local box -> host segfault) is fixed by a `local_size()` guard (no-op on rank without box,
      np=1 bit-identical) + test `test_mpi_system_solve_fields_np{1,2,4}`. (#99) **Device-MPI production
      VALIDATED on GH200** (ROMEO job 641249, harness #93) for `geometric_mg`: np=1/2/4 exit 0, no
      deadlock, bit-identical cross-np, compute-sanitizer 0 error, `dmax_abs` Cuda-vs-Serial within
      tolerance; `fft` np=1 OK. **`fft` np>1 under System = cleanly REFUSED (#106)**: `set_poisson("fft")`
      raises if n_ranks()>1, and the `assert(n_ranks()==1)` compile-out becomes a HARD safeguard (active throw in
      Release) in `PoissonFFTSolver`. direct fft = np=1 only; `DistributedFFTSolver` exists (tested
      separately, `test_mpi_fft_distributed`) but not routed in System (band layout vs single box).
- [~] **Step 7 / Polar Phase 2b: ALMOST DONE, step 2c remaining**:
      DONE: transport + polar Poisson in System.step (#168); Path A polar fluid (#209); tensor polar
      elliptic (#210); polar Schur source stage (#212, dt gain 2000x, cartesian bit-identity);
      annular diocotron RUN (n=384/512 GH200 sweeps + demonstrator); 2D polar fluid demonstrator
      (adc_cases feat/diocotron-polar-fluid: l=4 grows x14.9, machine mass, SHARP RING EDGE vs cartesian).
      REMAINING:
        - step 2c = FACADE WIRING of the polar Schur (#212) into System.step/Python -- IN FLIGHT (afd1c6c5)
          -> unblocks PR6 (stiff polar diocotron-Schur measurement). This is the ONLY active remainder.
        - MINOR: fix the PolarMesh docstring ("ExB scalar only", stale since #209).
        - IN PROGRESS (workflow): native POLAR Lorentz v x B_z source brick (the demonstrator worked around via
          electrostatic drift + centrifugal) -> makes the polar diocotron physically EXPLICITLY complete
          (true magnetic force instead of the workaround). Source = (0, +B_z*m_theta, -B_z*m_r) in the local
          polar frame (invariant algebraic rotation, cf. review #212).
      OWNER INVARIANT (keep): ring edge = TRANSPORTED DENSITY DISCONTINUITY, NOT a wall.
      Do NOT put a fixed wall/mask/cut-cell on it (physically WRONG). Cut-cell relevant ONLY for
      the EXTERNAL conductor (#109 shows it does NOT correct the rate). Rate levers = polar / high-order
      (WENO5/SSPRK3) / AMR: all DONE.

**HONEST STATUS (do NOT present "Ideal Plan complete")**: System production CPU = OK; AmrSystem
production CPU = OK (WENO5/Rusanov/conservative, #105); DSL demonstrators = OK (diocotron_dsl,
two_species_dsl, magnetic_isothermal_dsl, all in CI); **production GPU np=1 = OK (GH200, #97)**;
**`solve_fields` MPI np=1/2/4 = OK on CPU/CI side (#99)**; **device-MPI production `geometric_mg` = VALIDATED
(GH200, #93, np=1/2/4)**; **`fft` np>1 under System = cleanly refused (#106), no more segfault** (DistributedFFTSolver not routed, layout); `set_density`/`get_state` multi-rank
= out of scope; WENO5 on `CompiledModel` (.so AOT AND production) = SUPPORTED (3 ghosts via `set_block_ghosts`, #102),
WENO5 also wired on the native AMR path (#105); `m.compile()` ergonomic (auto-detect include + `so_path` cache, #103);
`AmrSystem.potential()` = binding EXISTS and exposed (`python/bindings.cpp:272`); **Schur/polar device**:
6/6 device-clean on Kokkos Cuda single-GPU (cf. VERDICT at the top); the 3 initial failures of the GH200
validation were TEST-SIDE (host functors in device kernels), fixed #150+#152, device-correct library;
`PAPER_ROADMAP.md` = NOT to be rewritten automatically
(awaits human validation of the O5 sweep); **next scientific deliverable = COMPLETE POLAR PATH (Polar
Phase 2b). SCIENTIFIC DECISION: the ring edge is a transported density discontinuity, NOT a physical
wall; do NOT redo wall-transport on this edge (physically wrong). Cut-cell relevant for the external
conductor only (#109 did not correct the rate). Valid levers for the ring edge: polar geometry,
high order (WENO5/SSPRK3), AMR.**

## 9. High-order diocotron measurement (PR-0 + O5, `adc_cases` side)

- [x] Order x resolution sweep O1/O2 (PR-0) then O5 = WENO5-Z + SSPRK3 (reachable from Python
      via #88): `diocotron/SWEEP_RESULTS.md`. l=3 starts mostly diffuse; l=4 goes from ~-12% (O2)
      to ~-4% on two clean points (n=128/256) at O5 (n=192 = fit-window artefact, traced); l=5
      at the target. CAUTIOUS conclusion: the PR-0 hypothesis of a ~12% structural plateau is WEAKENED (without
      being refuted). (#5, #6)
- [x] **High-resolution confirmation n=384 / n=512 (incl. O5) on ROMEO/GH200** BEFORE any rewrite
      of the paper roadmap (`PAPER_ROADMAP.md`). Upon agreement.

## 10. Schur condensation (condensed implicit EPM, theta-scheme)

NUMERICAL algorithm (not a physics): eliminate the velocity to condense onto the potential ->
stable implicit step at large dt. Model = physics; SchurCondensation = algo. Condensed operator
`-Lap phi - theta^2 dt^2 alpha div(rho B^-1 grad phi)` (NON symmetric) -> Krylov.

- [x] **PR0 - doc** `docs/SCHUR_CONDENSATION_DESIGN.md`: sign convention locked
      `A_op = I + theta^2 dt^2 alpha rho B^-1`, A non symmetric -> Krylov needed. (#119)
- [x] **PR2 - LorentzEliminator**: `B`, analytical `B^-1` (`apply_Binv`, `binv_ij`), POD/`POPS_HD`. (#118)
- [x] **PR1 - TensorEllipticOperator**: `-div(A grad phi) + kappa phi`, cross terms (9-point stencil,
      named functor `cross_div`), `set_cross_terms`. A=I/diag bit-identical. (#120)
- [x] **PR3 - matrix-free BiCGStab Krylov** non symmetric, preconditioned by MG on the symmetric
      part (MG alone stagnates/diverges), MPI-safe collective dots. (#122)
- [x] **PR3 - builder** `schur_condensation.hpp`: `A_op` coefficients + condensed RHS, generic
      over the roles. (#124)
- [x] **PR4 - source stage** `CondensedSchurSourceStepper`: build -> Krylov -> reconstructs v ->
      energy -> extrapolates U^{n+1} -> ghosts. Implicit relation 1e-15, stable at 8x the explicit dt. (#126)
- [x] **PR5 - Python binding**: `pops.Split(hyperbolic, source=pops.CondensedSchur(...))` + `pops.Role` +
      `set_source_stage` routing (no-op if nullptr, after transport). (#128)
- [x] **Review fix (findings 1+2)**: precond AND matvec at HOMOGENEOUS BC under non-zero Dirichlet (the matvec
      of r0 keeps the affine; matvec in loop + precond linearized by subtracting their offset);
      `op_`!=`precond_` enforced. Bit-identical to zero Dirichlet. (#134)
- [x] **Device-clean GPU**: the Schur stack was SILENTLY WRONG on Cuda (Geometry/Box2D accessors
      not `POPS_HD` -> wrong RHS/reductions, BiCGStab "0 iters rel=0 then NaN"). Fix in #135 (in flight,
      awaits GH200 validation). Host = correct, CI green.
- [~] **PR6 - polar diocotron-Schur measurement: UNBLOCKED, IN PROGRESS**: the POLAR Schur capability is DONE
      (tensor elliptic #210 + source stage #212, dt gain 2000x); facade wiring step 2c IN FLIGHT
      (afd1c6c5) -> makes the polar Schur usable from System/Python -> the stiff polar diocotron-Schur
      measurement becomes runnable.
      CONCRETE REMAINING (almost nothing): (1) step 2c = facade wiring -- IN FLIGHT (afd1c6c5), SHARED with Step 7
      (not double-counted); (2) ONE specific STIFF polar diocotron run. NB: the dt gain 2000x (stable where
      the explicit blows up) is ALREADY PROVEN at the stepper level by the #212 test; the diocotron run is only a
      demonstration on the precise case. (Reminder: the Schur stabilizes TIME; the RATE gap is geometric/non-linear,
      addressed separately by the polar path + the delta campaign.)
- [x] **PR7/PR8 - GPU/MPI port + AMR**: after #135; bit-invariance to the number of ranks.

## 11. Ergonomic System pipeline (P1-P7)

- [x] **P1 - per-block stride**: `Explicit/IMEX(stride=M)` (hold-then-catch-up), CFL
      `dt=cfl*h*substeps/(stride*w)`. (#121) Wording honesty -> review finding 3 (#138).
- [x] **P2 - clarify Implicit/IMEX**: rename `SourceImplicit`, deprecation of `Implicit`. (#123)
- [x] **P3 - `implicit_vars` mask** on the time policy / the block (NOT on the model). (#125)
- [x] **P4 - `set/get_primitive_state`** (init/diagnostic in primitive variables). (#127)
- [x] **P5 - CoupledSource DSL**: `pops.dsl.CoupledSource` -> postfix bytecode interpreted by a
      named device functor in `apply_couplings`; generic inter-species; 3 species + conservation,
      MPI np=1/2/4 bit-identical. (#131)
- [x] **P6 - multi-block AMR** = capstone Gap 4, see section 15.
- [~] **P7**: P7-b DSL runtime params = DONE #213 (m.param(kind=runtime) + System.set_block_params; change a param without recompiling the .so; byte-identity of the const params confirmed); P7-a implicit-total =
      RESEARCH (fully implicit scheme, large work item; cf docs/RESEARCH_BACKLOG.md), deferred unless the
      order-1 Lie splitting becomes a measured limit (otherwise order-2 Strang suffices).

## 12. Polar / annular geometry (the real diocotron growth-rate blocker)

Scientific conclusion: the growth-rate blocker is the CARTESIAN advection of the ring's radial
gradient; the POLAR geometry preserves it (proto: ratio 73 vs 1.0 cartesian). The Schur
does NOT address this gap (it stabilizes time). So the lever = put in the polar geometry.

- [x] **Mesh abstraction**: `pops.PolarMesh` / `pops.CartesianMesh` -> `System(mesh=)` (NOT
      `FiniteVolume(geometry=)`). `PolarGeometry`, polar divergence `(1/r)d_r(r F_r)+(1/r)d_th F_th`,
      `ExBVelocityPolar` `v_r=-(1/(Br))d_th phi`, `v_th=(1/B)d_r phi`. (#116)
- [x] **Phase 2a - direct polar Poisson** on the ring: FFT-in-theta + tridiagonal-in-r (robust,
      avoids the MG-in-polar which stagnates). (#130)
- [x] **Device**: polar flux/metric SILENTLY WRONG on Cuda (same cause #135: `r_cell`/`theta_cell`
      not `POPS_HD`). Fix #135 (in flight).
- [x] **Phase 2b (#168, DONE)**: transport + polar Poisson wired into `System.step` on a GLOBAL
      ring (NO cartesian<->polar coupling in v1; cartesian default bit-identical). `derive_aux_polar`
      (aux in local base e_r/e_theta), solid radial wall (wall_radial), polar `mass`/`step_cfl`/`step_adaptive`,
      guard nr>=3. C++ tests (conservation 3.4e-15) + Python `System(PolarMesh)` (<1e-11). REMAINING
      after merge: the quantitative annular diocotron RUN (rate measurement) = next scientific deliverable.

## 13. Adversarial review + hardening (findings)

Exhaustive adversarial review of the merged work (Schur + System pipeline + polar): 8 findings
confirmed real out of 12. Disposition:

- [x] **F1 - non-homogeneous BC precond** (krylov): affine operator -> wrong BiCGStab for non-zero
      Dirichlet. Fixed #134.
- [x] **F2 - `op_`==`precond_` aliasing**: comment "always valid" false. Fixed #134.
- [x] **F3 - step_cfl substeps>1 not bit-identical**: DECISION = keep the substeps-aware CFL
      (correct), fix the wording + tests. In flight #138.
- [x] **F4 - `evolve=False` silently ignored on prototype/aot**: explicit rejection. DONE #137.
- [x] **F5 - `pops.Split` silently lost on AmrSystem**: explicit rejection. DONE #137.
- [x] **F6 - `CondensedSchur` descriptors never transmitted** (dead API code): non-default rejection. DONE #137.
- [x] **F7 - `max_wave_speed_mf` non-collective** (divergent dt per MPI rank): `all_reduce_max`. In #135.
- [x] **F8 - `fab(0)` without `local_size()` guard** in the marshaling: DONE #204 (local_size guard posed; no half-fix because a half-fix
      would make a false-silent instead of a clean crash; to do with gather/scatter).

## 14. CI acceleration (keep all the tests, go faster)

- [x] **#136 (DONE, on master)**: cache of the INSTALLED Kokkos (big gain: no more rebuild on each run)
      + ccache (`CMAKE_CXX_COMPILER_LAUNCHER`) + Ninja + `ctest --parallel 2` (NO more: AMR/Eigen TU
      1-2 GB) + `setup-python@v6` + auto-discovery `python/tests/test_*.py` +
      `concurrency: cancel-in-progress` + split **ci-fast** (PR) / **ci-full** (push master + nightly +
      `workflow_dispatch` + label `ci-full`). No test removed; `ci-full` label to require
      MPI/Kokkos on a risky PR.
      MEASURED RESULT: **~25 min -> ~5 min warm** (Release 4.8 min, MPI 2 min, Kokkos 1.7 min);
      **45 Python tests** auto-discovered (superset of the 40; +5 never wired before). The following agent
      PRs benefit from now on (faster CI = faster iteration).

## 15. Multi-block AMR on a shared hierarchy (capstone Gap 4 / P6)

Reference: `docs/AMR_MULTIBLOCK_DESIGN.md` (#139). AMReX/FLASH/SAMRAI model: ONE common AMR
hierarchy carrying N co-located blocks; regrid driven by the UNION of the criteria.

STRICT scoping (locked): same hierarchy, same cells, co-located fields; ALL the blocks
evolved on ALL the patches (never a spatial absence of a species); flexibility only per-block
(model/spatial/time/substeps/stride/evolve); Poisson `rhs[level]=sum of the elliptic_rhs` co-located;
same-cell coupled sources with EXACTLY opposite contributions; reflux per block; tags = e OR i OR n
OR phi OR user; future optimizations TEMPORAL only (stride / global evolve at the block), NEVER
local spatial. NO per-species hierarchy in v1 (Phase 3 only if real scientific need).

KEY OBSERVATION (confirmed by scoping June 2026 on master = #154): the multi-block engine ALREADY EXISTS =
`AmrSystemCoupler` (compile-time: shared hierarchy, Poisson sum, per-block scheme/substeps/stride,
IMEX-callback, level-by-level sources + average_down trailing #169, shared B_z; `same_layout_or_throw`).
The RUNTIME FACADE `AmrRuntime` (#154, `include/pops/runtime/amr_runtime.hpp`) exposes a type-erased
registry by name and ALREADY runs 2 explicit blocks, but does not yet TRANSMIT substeps/stride/IMEX/coupled
sources/regrid from the facade to the engine. So COMPLETION on the facade side, not creation. The `stride_due`
bug is ALREADY fixed (#140: cadence `(macro_step_+1) % stride`, hold-then-catch-up). Mono-block still goes
through `AmrCouplerMP` (never `AmrRuntime`) -> bit-identical baseline. Multi-block + regrid_every>0 REFUSED.

PR split (Phase 1, write-sets) - PRECISE STATUS (scoping):
- [x] **(i)** extract `AmrBlock` / `AmrHierarchyLayout` (mono-block bit-identical). DONE #141.
- [x] **(ii)** 2 explicit blocks, different schemes (AmrRuntime facade + co-located Poisson sum +
      per-block conservation + MPI np=1/2/4). DONE #154. Tests test_amr_system_twoblock + mpi_twoblock_parity.
- [x] **(iii)** co-located Poisson sum VALIDATION test [trivial, validation only; the engine already does it].
- [x] **(iv) #175** per-block substeps/stride + substeps-aware step_cfl: `AmrRuntime::step` honors
      substeps/stride (hold-then-catch-up, mirror `AmrSystemCoupler::step`); `AmrSystem::step_cfl` =
      `cfl*h*min_b(substeps_b/(stride_b*w_b))`. Mono-block bit-identical (AmrCouplerMP routing). MERGE.
- [x] **(v)** multi-block production DSL: `add_native_block`/`add_compiled_model(AmrSystem&)` must no longer
      raise on the 2nd block (queue + build at `ensure_built`). Write-set amr_dsl_block.hpp + amr_system.cpp.
- [x] **(vi) #179** same-cell / opposite AMR coupled sources: `coupled_source_step` via the runtime
      registry + average_down covered cells (#169); per-cell+global conservation, disable-and-fail. MERGE.
- [x] **(vii) DONE (#184/#185)** local AMR runtime IMEX: the facade honors `time="imex"` multi-block (the engine already has
      the AmrImplicitSourceStepper callback). Agent in progress.
- [x] **(viii)** Phase 2: union-tags regrid (unlocks multi-block + regrid_every>0); then Schur / global
      implicit / paper repro.

Acceptance tests: 2 explicit blocks different schemes; e- IMEX(substeps=10) + ions
Explicit(substeps=1) AND the reverse; neutrals stride=20 feeding sources/Poisson; evolve=False fixed
background in the elliptic RHS; regrid conserves the mass per block; 2-block production DSL; MPI np=1/2/4;
Kokkos Serial; 1 production multi-block case on GH200.

## 16. Conservation findings (review)

- [x] **A3 [conservation BUG] DONE #169**: `AmrSystemCoupler::coupled_source_step` and `AmrImplicitSourceStepper` applied the source to the COVERED coarse cells without average_down trailing. Fix: cascade `mf_average_down_mb` fine->coarse after the level loop (strict no-op mono-level, bit-identical). Tests `test_amr_source_covered_cells` + `test_amr_composite_source_conservation` (discriminating: fail without the fix). Adversarially reviewed (MERGE-SAFE).
- [x] **A2 [conservation RISK] DONE #167**: helper `add_pair(block_a, block_b, role, expr)` emits +expr / -expr from the SAME subtree (conservative by construction) + mode `compile(verify_conservation=True)` opt-in. Test `test_dsl_coupled_source_conservation`. Adversarially reviewed (MERGE-SAFE).
- [x] **A4 [test GAP]**: covered in practice by the #168 tests (polar mass conservation with non-zero INTERIOR radial flux, profile modulated in theta -> v_r != 0). REMAINING optional: a dedicated polar MMS case with analytical v_r != 0 (beyond the global conservation).
- Tests to add: `test_dsl_coupled_source_conservation`, `test_amr_composite_source_conservation`, `test_amr_source_covered_cells`, `test_polar_conservation_radial_flux`.
