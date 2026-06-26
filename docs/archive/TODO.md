# pops: Tasks: from `PhysicalModel` to the multi-species system

A single actionable list of what is **done** and what **remains**, based on the
tutor discussion and the increments. Not a production solver: we lay down a **testable
skeleton**. `[x]` = done and green; `[ ]` = to do.

---

## 0. Posture (tutor's framing)

- Skeleton first, not the final solver; test the abstraction level on simple cases.
- Abstraction and architecture **before** the data structure (it changes later).
- Performance **after** stabilization ("I will optimize once the code is clean and frozen").
- Validated **by a user** (can Sacha describe his case without touching AMR/MPI/GPU?), not by the compiler.
- `PhysicalModel` is validated: we **add** the level above it, we do not replace it.
- Diffusion = one more flux, not a new layer.
- Two repositories: `adc_cpp` = generic core, `adc_cases` = models / facades / Python / examples.

---

## 1. Done (current status, all green)

### Architecture / repositories
- [x] Core/applications split: `adc_cpp` (engine, zero model) <-> `adc_cases` (models, facades, Python) via FetchContent (`pops::pops`). Pushed.
- [x] `adc_cpp` HEAD contains **no** application (verified); README refocused on "core library".

### Per-block abstractions (the missing level: skeleton in place)
- [x] `core/equation_block.hpp`: `EquationBlock<Model, Spatial, Time>` = state + model + spatial discretization + time policy + **per-block BC**.
- [x] `core/coupled_system.hpp`: `CoupledSystem<Blocks...>` (variadic), `for_each_block`, `block<I>()`. **-> design data = `tuple<Blocks...>` (one MultiFab per block) kept for the skeleton.**
- [x] `integrator/time_integrator.hpp`: `TimePolicy<Method, Treatment, Substeps>` + `ExplicitTime` / `ImplicitTime` / `IMEXTime` / `PrescribedTime`; tags `SSPRK2`/`SSPRK3`; `UserTimeIntegrator`.
- [x] `integrator/scheduler.hpp`: `advance_subcycled(system, dt, advance_block)` reads the substeps per block.
- [x] `coupling/elliptic_rhs.hpp`: `SingleModelEllipticRhs` (mono) + `TwoFieldChargeDensityRhs` / `TwoBlockChargeDensityRhs` (RHS = q0*n0 + q1*n1, reads several blocks).
- [x] `coupling/system_coupler.hpp`: `SystemCoupler` single-level, global elliptic RHS, explicit blocks advanced by the core, **implicit/IMEX blocks delegated to a callback** (Newton/linear/collisions branch point), substeps per block.
- [x] Tests `test_system_abstraction`, `test_system_coupler` (delegated implicit electrons + SSPRK3 ions, substeps, system RHS) green.

### Selectable bricks
- [x] `SpatialDiscretisation<Limiter, NumericalFlux>` + `Coupler::step<Disc, TimePolicy>` (flux + integrator + subcycling) on a uniform mesh.
- [x] `SpatialDiscretisation` wired into **AMR** (`AmrCoupler/MP::step<Disc>`, conservative MUSCL with 2 ghosts).
- [x] Elliptic solver chosen at runtime (diocotron facade MG/FFT, `variant` pattern).

### Corrections
- [x] **AMR applies `model.source`** (the AMR path ignored it, bug) + test `test_amr_source`.
- [x] Diffusion = flux: trait `DiffusiveModel` (`+nu*Delta U` in `assemble_rhs`) + model `AdvectionDiffusion` (uniform mesh).
- [x] `SpectralCoupler` calls `model.elliptic_rhs` instead of hardcoding the diocotron's `alpha*(u-n_i0)`.

---

## 2. To do: complete the multi-species skeleton

**Suggested order**: (1) `ChargeDensityRhs` N-blocks [done] -> (2) `CoupledSource` [done] -> (3) implicit
interface `ImplicitBlockStepper` + default [done] -> (4) **minimal C++ example** (implicit
electrons + explicit ions, rhs = n_i - n_e, no Python) [done] -> (5) `AmrSystemCoupler` [done] ->
(6) Python API in `adc_cases` [done]. Each step validated by an integration test (all green:
adc_cpp 38/38, adc_cases 47/47 + Python bindings). **Sections 2 and 3 entirely done**; **section 4**: 4.1
(AMR diffusion) and 4.2 (`Aux` contract) done, 4.3 (max_drift_speed, changes the dt) and 4.4 (diagnostics
dedup) deferred with justification. Mostly **refinements** remain (generic Python composition,
user C++ model exposed to Python, perf).

### 2.1 Coupling and RHS
- [x] **`ChargeDensityRhs` with N blocks**: `f = Sum_s q_s n_s` (sum over `for_each_block`),
  **per-species configurable charge/component/sign** (`SpeciesCharge` + `add_scaled_component`),
  in `coupling/elliptic_rhs.hpp`. `TwoBlockChargeDensityRhs` kept (compat). Tested
  (`test_two_species_minimal`, `test_amr_system_coupler`).
- [x] **`CoupledSource` (inter-species sources)**: `coupling/coupled_source.hpp`, concept
  `CoupledSourceFor`, `NoCoupledSource`, and `SystemCoupler::coupled_source_step(src, dt)`
  (forward-Euler splitting) that reads **several blocks + aux**. Distinct from `model.source`
  (local). Tested (`test_coupled_source`: conservative linear exchange).
- [x] **Per-block BC actually applied**: `SystemCoupler::stage_rhs` fills the halos
  with `block.bc` (not a global BC). Verified by `test_system_two_explicit` (periodic vs
  outflow -> divergent fields, same initial data).

### 2.2 Time: implicit / IMEX actually executed
- [x] **Real implicit interface + default**: `integrator/implicit_stepper.hpp`, concept
  `ImplicitBlockStepper`, `backward_euler_source` (local Newton, Jacobian by finite
  differences: **unconditionally stable**, exact in 1 iteration for a linear relaxation,
  where Picard would diverge as soon as `dt*stiffness > 1`), and `ImplicitSourceStepper` (ready-to-use
  default, **no Newton on the user side**). Tested (`test_two_species_minimal`: `dt*k=100`).
- [x] **Partial IMEX**: trait `Model::is_implicit(c)` (concept `PartiallyImplicitModel`).
  `backward_euler_source` does forward-backward Euler: explicit variables in forward Euler,
  implicit variables by Newton on the **reduced subsystem** (`solve_dense` of size
  `m_impl <= n_vars`). Without the trait -> all implicit (compat). Tested (`test_imex_partial`:
  stiff implicit var bounded, soft explicit var, the mask does change the treatment).
- [x] **Per-species time subcycling + phi cadence**: `substeps` executed (single-level **and**
  AMR). The phi re-solve cadence is now **explicit**: `PoissonCadence`
  (`OncePerStep` default / `PerSubstep`) in `AmrSystemCoupler`; the single-level `SystemCoupler`
  already re-solves phi at each RK stage. Tested (`test_amr_system_coupler` part C: 1 vs 4 solves).

### 2.3 AMR for the system: `AmrSystemCoupler`
- [x] `coupling/amr_system_coupler.hpp`: ports `CoupledSystem` onto AMR. Each block has **its**
  hierarchy (`std::vector<AmrLevelMP>`), all species **share** the AMR mesh,
  the aux (phi, grad phi) and the **coarse system Poisson** (`f = Sum_s q_s n_s`). Orchestration:
  `sync_down` (per block) -> coarse Poisson -> aux + fine injection -> each block advanced by
  `advance_amr<Disc_block>` (Berger-Oliger subcycling + **reflux** + **average_down**), with
  its **species substeps**; implicit/IMEX delegated (default `AmrImplicitSourceStepper`,
  backward-Euler per level). Tested (`test_amr_system_coupler`: per-block conservation, system
  RHS, nonzero phi, coarse+fine implicit relaxation). Reuses the `advance_amr` engine
  and the `_mb` primitives; one box per level validated (like `AmrCoupler`).

### 2.4 Validation cases (testable skeleton)
- [x] **Minimal C++ example WITHOUT Python**: `test_two_species_minimal`, implicit electrons
  (stiff relaxation) + explicit ions + `rhs Poisson = n_i - n_e` via `ChargeDensityRhs` +
  `ImplicitSourceStepper`. The "can a user compose his case?" test: he
  assembles only bricks (model, scheme, time policy, charge), no solver written.
- [x] **Euler electrons + isothermal Euler ions + Poisson** (canonical two-species case),
  `adc_cases`: models `ChargedEuler` (4 var) + `ChargedEulerIsothermal` (3 var) in
  `model/charged_fluid.hpp`, composed into heterogeneous blocks (`test_two_species_euler`).
- [x] **diocotron with mobile ions**, `adc_cases/test_diocotron_mobile_ions`: reuses the
  `Diocotron` model for 2 blocks, the frozen `n_i0` becomes a transported block, Poisson `alpha(n_e - n_i)`.
- [x] Guard: mass conserved per block (`test_amr_system_coupler`), RHS = q_i n_i + q_e n_e
  correct (`test_two_species_minimal`, `test_amr_system_coupler`). Comparison to an analytic
  reference (exact backward-Euler, exact production).

### 2.5 Integration tests to add
- [x] `CoupledSystem` + **nonzero Poisson RHS** (`ChargeDensityRhs`), `test_two_species_minimal`.
- [x] `SystemCoupler` with **two different explicit blocks** (distinct schemes/substeps), `test_system_two_explicit`.
- [x] **Implicit block** wired and hardened (stiff relaxation `dt*k=100`, unconditional stability), `test_two_species_minimal`.
- [x] `AmrSystemCoupler`: per-block conservation, reflux, system RHS, multi-level implicit relaxation, `test_amr_system_coupler`.

---

## 3. To do: Python composition API (in `adc_cases`, AFTER the C++ example)

Goal: Python **composes**, does not compute cell by cell. The strings select
compiled C++ bricks. To be done only **after** the minimal C++ example (section 2.4) which validates
the user architecture.

- [x] **Physics models ready to compose** (in `adc_cases`): `ChargedEuler`,
  `ChargedEulerIsothermal`, `Diocotron`, `Euler`... serve as blocks (`EquationBlock`).
- [x] Compiled facade `MultiSpeciesSolver` (PIMPL) instantiating a `CoupledSystem` +
  `SystemCoupler` (`adc_cases/solver/multispecies_solver.{hpp,cpp}`), like the other facades.
  Concrete two-fluid case; the fully generic composition `vector<SpeciesConfig>`
  (arbitrary species) remains a later milestone (combinatorial explosion / type erasure).
- [x] pybind11 bindings: `pops.MultiSpeciesConfig`, `pops.MultiSpeciesSolver`
  (`step/advance/density_e/density_i/potential/mass_e/mass_i/max_charge`), fields as numpy
  (`python/bindings/core/bindings.cpp`, tested `python/test_bindings.py`).
- [x] **Runtime composition**: `pops.Simulation` (`solver/simulation.{hpp,cpp}`),
  `add_species(name, charge)` adds N species on the fly, sharing a system Poisson;
  `set_density` (numpy), `step/advance`, `density/potential/mass`. The `sim.add_equation(...)`
  spirit from the TODO, bounded to drift species (Diocotron, 1 var, simple IC); physics compiled,
  no Python callback in the hot path. Tested (`test_simulation`, `test_bindings.py`).
  Extending to Euler / IMEX = same pattern + one IC per model.
- [x] `model`/`flux`/`time` <-> C++ tags: physics is in compiled C++ (schemes and policies
  fixed at compile time), no Python callback in the hot path.
- [x] (advanced) expose a user C++ `PhysicalModel`, composable from Python: extension point
  = the tag dispatch of `Simulation::add_species(name, model, charge)`
  (`adc_cases/src/simulation.cpp`). The user writes his model (header), adds a tag +
  an advance closure, recompiles, composes from Python, compiled physics, no
  callback in the hot path. Runtime plugin without recompiling = design choice (future).
- [x] **Evolution (June 2026)**: this layer (`add_species`, named blocks `Diocotron`/`Euler`,
  tags `model=`) is replaced by the **generic brick composition**: `CompositeModel`
  in C++, `pops.Model(state, transport, source, elliptic)` in Python. `adc_cpp` no longer names
  any scenario; the named compositions live on the application side (`adc_cases/models.py`).

---

## 4. To do: core follow-ups (cleanups / consistency)

- [x] **Diffusion on AMR**: carried as a **diffusive face flux** `-nu(u_R-u_L)/h` in
  `compute_face_fluxes` (receives `dx/dy` from the level), guarded by `DiffusiveModel` (hyperbolic
  path strictly bit-identical). Seen by the reflux -> conservative at the
  coarse-fine interfaces. Tested (`test_amr_diffusion`: mass conserved to 1e-12 + smoothing).
- [x] **`Aux` contract**: decided, `Aux` is **fixed** (`pops::Aux`). The `PhysicalModel` concept
  now requires `std::same_as<typename M::Aux, Aux>`: it promises exactly what
  `load_aux` provides (generalizing to an arbitrary `Model::Aux` remains possible later).
- [x] **`SpectralCoupler::max_drift_speed`**: generalized via `model.max_wave_speed` (no more
  hardcoded `/model_.B0` -> spectral coupler not tied to the diocotron). [warning] **Changes the
  CFL `dt`** of the diocotron (`max(|gx|,|gy|)/B0` per direction instead of `hypot(gx,gy)/B0`): not
  bit-identical on the trajectory, **to re-validate on the physics side** (diocotron growth).
  The MPI tests (inter-rank identity) stay green.
- [x] **Diagnostics dedup**: `amr_diagnostics.hpp` now carries the **single** multi-box
  implementation (`amr_mass_mb`, `amr_max_drift_speed_mb`); the single-box variants reduce to it
  (degenerate 1-fab case, bit-identical) and `AmrCouplerMP::mass()`/`max_drift_speed()`
  call them (+ `all_reduce` depending on ownership) instead of reimplementing the loops. Verified
  bit-identical (adc_cpp 38/38, adc_cases 48/48 including AMR diocotron).
- [x] **Catch-all `Coupler`**: the orchestrator is extracted, `SystemAssembler` (assembles) +
  `SystemDriver` (advances), `SystemCoupler`/`SystemDriver` aliases. The legacy single-model
  `Coupler` keeps its historical name (diocotron validated); renaming it globally = pure churn,
  left as is (board decision).

---

## 5. Decision to enact at the board (with Sacha)

- **System data structure**: the skeleton kept `tuple<Blocks...>` (one
  `MultiFab` per block, `EquationBlock::state` = pointer). Alternative: `StateVec<N_total>`
  stacked (one contiguous memory block, offsets per species, better locality, more complex).
  To confirm/adjust: this is the decision that weighs on perf (point 0: data
  structure = later, but the interface decision is taken now).
  - **Feedback from filling it in**: `tuple<Blocks...>` held without friction for
    all delivered cases, including **heterogeneous** blocks (Euler 4 var + isothermal 3 var:
    a stacked `StateVec<N_total>` would impose species of the same size or an irregular AoS
    layout). The `for_each_block`/`block<I>()` stays the interface; the internal structure
    (one `MultiFab` per block) can switch to stacked later **without changing the API** of the
    couplers. Recommendation: **keep `tuple` per block** as the default, measure before
    stacking; only stack if a homogeneous case (same `n_vars`) becomes the perf bottleneck.

---

## 6. Glossary (for the slides: define before throwing the acronyms around)

| Term | Short meaning |
|---|---|
| `BoxArray` | partition of the domain into blocks (boxes) |
| `MultiFab` | the fields `U` stored on these blocks (distributed collection) |
| `BCRec` | boundary conditions of a field |
| `aux` | transported auxiliary variable (`phi, grad phi`) |
| seam | seam where the parallelism lives (`for_each_cell`, `comm`) |
| `EquationBlock` | state + model + spatial method + time policy + BC (one block) |
| `CoupledSystem` | several `EquationBlock` |
| `SystemCoupler` | single-level orchestrator: global elliptic RHS + advances each block |
| `AmrSystemCoupler` | the `SystemCoupler` ported onto AMR (coarse Poisson + per-block reflux) |
| `ChargeDensityRhs` | Poisson right-hand side with N species: `f = Sum_s q_s n_s` |
| `CoupledSource` | inter-species source (reads several blocks + phi), distinct from `model.source` |
| `ImplicitSourceStepper` | implicit default: backward-Euler (Newton) on the source, no user Newton |
| `is_implicit(c)` | partial IMEX trait: which variables of a block are implicit |
| `PoissonCadence` | re-solve frequency of phi between substeps (`OncePerStep`/`PerSubstep`) |
| `MultiSpeciesSolver` / `Simulation` | `adc_cases` facades: multi-species composition (compiled / at runtime) |

---

## 7. Synthesis (board sentence)

> `pops` knows how to take a **local physical law** (`PhysicalModel`) and run it on
> a mesh with Poisson, AMR, MPI and GPU. The **multi-block assembly** level is
> now **filled in** (testable skeleton, all tests green): `EquationBlock` (state +
> model + spatial scheme + time policy + BC), `CoupledSystem` (several blocks),
> single-level `SystemCoupler` **and** `AmrSystemCoupler` on AMR (system elliptic RHS
> `Sum_s q_s n_s`, explicit advanced by the core, implicit/IMEX delegated with a default
> backward-Euler without user Newton, inter-species coupling source, substeps per
> block, conservation by reflux). The skeleton is **filled in AND exercised by real
> physics**: canonical two-fluid models (Euler electrons + isothermal ions) and
> diocotron with mobile ions in `adc_cases`, `MultiSpeciesSolver` facade + Python bindings
> that compose a `CoupledSystem` from Python without a callback in the hot path; diffusion
> made conservative on AMR (face flux). The `PhysicalModel` describes an equation, the
> `CoupledSystem` a system, the `Scheduler`/coupler the execution order; the core guarantees
> AMR / MPI / GPU. Refinements remain (generic Python composition, perf, sections 4.3/4.4).

---

## 8. Tutor feedback (2026-06-01): assembly level: covered vs to extract

Discussion: couple ions / electrons / **neutrals**, each with its `PhysicalModel`; the
species interact in the **elliptic right-hand side** `f` AND in the **source `S`**
(never in the flux `F`). We want to choose **per species**: the model (isothermal / given
constant profile / solved or not / not at every step), the **spatial scheme**, the **time
step** (subcycling: 10 electron steps per 1 ion step), the implicit / explicit **treatment**,
and even **implicit on ONLY PART of the variables** (cost). On the architecture
side: extract the **time integrator** as an object (`take_step`) given to the coupler,
and **lighten the coupler** that "does too much" ("advancing a coupler" is philosophically
shaky: a coupler *assembles*, a *driver* advances).

### 8.1 Already covered (verified by test: answers to the "can we?")
- [x] **N heterogeneous species, each with its `PhysicalModel`** -> `CoupledSystem<Blocks...>`
  (Euler electrons 4 var + isothermal ions 3 var, `test_two_species_euler`). A **neutral**
  = one more block (charge 0 -> does not contribute to `f`, but interacts via `S`).
- [x] **Interaction in `f`** (elliptic): `f = Sum_s q_s n_s` -> `ChargeDensityRhs`.
- [x] **Interaction in `S`** (collisions, exchange; not in `F`) -> `CoupledSource`
  (`coupled_source.hpp`; `SystemCoupler::coupled_source_step` reads all blocks + phi).
- [x] **Different spatial scheme per species** (electrons MUSCL, ions order 1, distinct
  limiters) -> per-block `SpatialDiscretisation` (`test_system_two_explicit`).
- [x] **Subcycling per species** (10 electron steps: 1 ion step) -> `substeps` +
  scheduler (`test_system_abstraction`: ne=10, ni=1).
- [x] **Implicit electrons + explicit ions** -> per-block `TimeTreatment`
  (`test_two_species_minimal`, `test_amr_system_coupler` part B).
- [x] **Implicit on PART of the variables** (not the whole model) -> trait
  `Model::is_implicit(c)` (`test_imex_partial`) = "say which variables we step".
- [x] **Species in constant profile / not advanced** -> `PrescribedTime` (the scheduler skips
  the `Prescribed` blocks, which still contribute to `f`). *[to illustrate with a test]*
- [x] **Solve the elliptic every step or not** -> `PoissonCadence` (`OncePerStep`/`PerSubstep`).
- [x] "numerical method" is **already** named `SpatialDiscretisation` (!= time integrator).
- [x] **The "bigger `PhysicalModel`" that encompasses everything** = `EquationBlock`
  (`{Model, SpatialDiscretisation, TimePolicy, BC}`); `CoupledSystem` = several of these
  bundles. This is already the composable "state physical model" described (each fluid has its
  set of equations: density alone, or all the moments).

### 8.2 To extract (the real abstraction gaps): detailed plan

**A. `TimeIntegrator` as a first-class object (priority 1).** Today SSPRK is hardcoded
in `SystemCoupler::advance_explicit_ssprk2/3` AND copied into `ssprk.hpp` and `Coupler`;
`Time` is only a template *tag*, `UserTimeIntegrator` has no `take_step`. Goal:
give `{PhysicalModel, SpatialDiscretisation, TimeIntegrator}` to the coupler as a trio,
the `TimeIntegrator` being a core object (or supplied by the user).
- [x] **A1. Contract**: concept `TimeStepper` exposing `take_step(rhs_eval, U, dt)`, where
  `rhs_eval(U_stage, R)` fills `R = -div F + S`. The integrator sees ONLY `rhs_eval` + the
  MultiFab ops. (`integrator/time_steppers.hpp`.)
- [x] **A2. Core impls**: `ForwardEuler`, `SSPRK2Step`, `SSPRK3Step` as generic
  `take_step` objects. `ssprk.hpp::advance_ssprk2` now delegates to `SSPRK2Step`.
- [x] **A3. Wire the user integrator**: the `Method` of an explicit block can be
  a core tag (`SSPRK2`/`SSPRK3`) **or a `TimeStepper` type written by the user**,
  the coupler instantiates it and calls its `take_step` (`test_user_time_integrator`). From
  Python: selection by tag (no hot-path callback). *[BYO-stepper Python = future]*
- [x] **A4. Delegation**: `SystemCoupler::advance_explicit_block`, `ssprk.hpp`, **AND the
  legacy single-model `Coupler`** (`advance`/`advance_ssprk3`) now delegate to the
  `SSPRK2Step`/`SSPRK3Step` objects, no more inline SSPRK copy. The legacy kept a
  `recompute_aux` per stage (`true` at stage 0, `per` afterwards): reproduced by an `rhs_eval`
  that counts the stages. **Bit-identical** (diocotron: adc_cases 48/48).
- [x] **A5. Example**: `test_user_time_integrator` (user integrator) + all the rest
  all-core.
- [x] Guard: **bit-identical** to the current SSPRK (adc_cpp 41/41, adc_cases 48/48).

**B. Split the coupler, Assembler vs Driver (priority 2; section 9.6).** A coupler assembles
the elliptic + solves Poisson + derives aux + computes the spatial RHS + integrates + subcycles:
too much. "Advancing a coupler" is shaky, a coupler *assembles*, a *driver* *advances*.
- [x] **B1/B2. Extraction into TWO classes** (`system_coupler.hpp`): `SystemAssembler`
  (`solve_fields` + `block_residual` = the residual evaluator; phi/aux/geom/ba/dm; NO
  step) and `SystemDriver` (schedule, `advance_explicit_block` via `take_step`, IMEX,
  `coupled_source_step`) which **owns** a `SystemAssembler` (composition). Bit-identical
  (adc_cpp 42/42, adc_cases 48/48). Tested `test_assembler_driver` (assembler alone + driver).
- [x] **B3. The "that advances" name**: `SystemCoupler` is now the **alias** of `SystemDriver`
  (the "that advances" name); `AmrSystemDriver` alias for AMR. Compat preserved (tests, facades).
- [x] **B4.** The compiled facade `MultiSpeciesSolver` (`adc_cases`) already rests on
  `SystemDriver` (via the `SystemCoupler` alias). The runtime `Simulation` keeps its own
  orchestration (type-erased) by design (species composed at runtime).

**C. Multirate, proper `dt` per model (priority 3).** `substeps` covers the "more often"
(10 electron substeps / 1 ion).
- [x] **Species "not solved at every step"** (every-N): `TimePolicy` gains a `Stride`
  (`ExplicitTime<Method, Substeps, Stride>`). A block of cadence N advances only 1 macro-step
  out of N, then by an effective step `N*dt` (it catches up the time: total `M*dt` after M
  macro-steps, but computed N times less often, the "gas not solved every step"). Handled
  by `block_stride_v` + `advance_subcycled(system, dt, macro_step, ...)`; `SystemDriver` and
  `AmrSystemCoupler` carry `macro_step_`. `Stride=1` = historical bit-identical.
  Tested (`test_multirate_stride`: fast stride 1 + slow stride 3, synchronized at M=3).
- [x] **Macro step chosen by CFL**: `SystemDriver::step_cfl(cfl)` / `cfl_dt(cfl)` compute
  `dt = cfl*min(dx,dy) / w_max`, `w_max` = largest wave speed **over all the
  species** (`max_wave_speed_mf` reduction per species, via `model.max_wave_speed`), the fastest
  species constrains the step. Combined with the `Stride` of a slow species, this gives the
  **practical multirate** (macro step fixed by the fast ones, slow advanced once per stride).
  Tested `test_cfl_dt`.
- [x] **Fully adaptive multirate**: `SystemDriver::step_adaptive(cfl)`, macro step
  fixed by the fastest species (CFL), and the `stride` of EACH species derived **at runtime**
  from the ratio `w_max / w_s` (`stride_s = max(1, floor(w_max/w_s))`). A species N times slower advances
  automatically once per N, by a step N times larger (= its stable dt). The per-block dispatch
  is factored out (`advance_block_dispatch`) and shared with `step` (no duplication).
  Tested `test_adaptive_multirate` (fast a=4 -> stride 1; slow a=1 -> stride 4, derived alone).

---

## 9. Hardening: Codex review (2026-06-01): HANDLED (2026-06-02)

Independent review of `multispecies-fill`. Initial verdict: skeleton **functional + tested**
but points to harden. **The 6 points have been handled** (adc_cpp 41/41, adc_cases 48/48, refactors
bit-identical on the existing). Final state below.

- [x] **9.1 Real IMEX** *(overlaps with section 8.2 A)*. `SystemCoupler::step` AND `AmrSystemCoupler::step`
  do, for an `IMEX` block: **explicit transport by the core** (-div F via `SourceFreeModel`
  + forward Euler / source-free `advance_amr`), **then** implicit source by the callback.
  Pure implicit: no transport. (`test_imex_transport`; before: frozen field.) Known limit:
  a **diffusive** IMEX block loses the Fickian flux at the explicit half-step (`SourceFreeModel`
  does not expose `diffusivity()`), separate refinement.
- [x] **9.2 `ChargeDensityRhs`: charge default** -> `SpeciesCharge{}` is worth **`q = 0`** (a
  neutral / a forgotten block no longer pollutes Poisson) AND `operator()` requires `species.size() ==
  System::n_blocks` (throw otherwise: one charge per block, neutral declared explicitly `q=0`).
  (`test_system_hardening`.)
- [x] **9.3 `AmrSystemCoupler` construction safeguards** -> the ctor **throws** if
  `block_levels.size() != n_blocks`, if a block does not have the right `nlev`, or if the per-level
  grids differ (number of boxes != block 0). No more silent out-of-bounds.
  (`test_system_hardening`.)
- [x] **9.4 AMR `phi`/`aux` BC** -> `AmrSystemCoupler::solve_fields` derives aux by the **same
  path** as the single-level: `fill_ghosts(phi, bcPhi)` -> `field_postprocess` ->
  `fill_ghosts(aux, aux_bc)`, `aux_bc` derived from `bcPhi` (Periodic->Periodic, otherwise Foextrap).
  Handles the non-periodic instead of a hardcoded periodic `fill_boundary`.
- [x] **9.5 `CoupledSource` on AMR** -> `AmrSystemCoupler::coupled_source_step` (splitting
  **per level**: repoints each block to its level k, reads all blocks + aux[k]).
  (`test_amr_system_coupler` Part D: conservative exchange.)
- [x] **9.6 `Coupler` name** *(cosmetic)* -> alias `SystemDriver` / `AmrSystemDriver` (the name
  "that advances") + doc of the two roles (assembler = `solve_fields`; driver = `step`). The
  split into two classes remains deferred (section 8.2 B1/B2: cosmetic on validated code).

**Nuance (TimeIntegrator point)**: `ssprk.hpp` only carried a generic `advance_ssprk2`;
SSPRK3 was recoded in the couplers (real duplication). This is resolved (section 8.2 A):
`SystemCoupler` **delegates** SSPRK2/3 to the core `SSPRK2Step`/`SSPRK3Step` objects,
`ssprk.hpp` delegates too. The legacy single-model `Coupler` remains (not migrated, diocotron validated,
section 8.2 A4).
