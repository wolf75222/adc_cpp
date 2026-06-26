# Hyperbolic-elliptic coupler hierarchy

This document describes each coupling class present in `include/pops/coupling/`,
its responsibility, what it assembles or advances, and when to choose it.
It is a reference complement to `ARCHITECTURE.md` (sections 5 and 8) and does not
duplicate what is already described there.

---

## 1. Overview: tree of couplers

```
Coupler<Model, Elliptic>                  -- mono-modele, mono-niveau
    |
    +-- SystemAssembler<System, RhsAsm, Elliptic>   -- multi-especes, mono-niveau (ASSEMBLE)
    |        |
    |        +-- SystemDriver<System, RhsAsm, Elliptic>    -- idem, AVANCE (= SystemCoupler)
    |
    +-- AmrCouplerMP<Model, Elliptic>     -- mono-modele, AMR multi-box + regrid BR
    |
    +-- AmrSystemCoupler<System,RhsAsm,Elliptic>   -- multi-especes, AMR (= AmrSystemDriver)
```

The elliptic backend `Elliptic` is parameterized everywhere via the `EllipticSolver` concept
(file `numerics/elliptic/elliptic_solver.hpp`). The default value is
`GeometricMG` (geometric multigrid V-cycle).

---

## 2. Coupler -- single-model, single-level

**File:** `include/pops/coupling/single/coupler.hpp`

**Instantiation:** `pops::Coupler<Model, Elliptic = GeometricMG>`

### Role

Closes the Poisson -> aux -> advance loop for a single physical model on a unique
uniform grid level. It is the simplest coupler; it is used directly
in unit test cases and in single-species tutorials.

### What it assembles

At each stage of the integrator (SSPRK2 or SSPRK3):

1. Elliptic right-hand side: `f = SingleModelEllipticRhs(model, U)`, that is
   `model.elliptic_rhs(U)` cell by cell
   (`coupler.hpp:52-54`, delegated to `elliptic_rhs.hpp:27-40`).
2. Solving `lap(phi) = f` by the elliptic backend (`GeometricMG::solve`).
3. Derivation `aux = (phi, d_x phi, d_y phi)` by centered differences, stored with
   sign `+grad phi`; the `GradSign::Plus` convention is documented in
   `elliptic_problem.hpp` (`coupler.hpp:61-65`).
4. If the model declares `n_aux > 3` (extra `B_z` or `T_e` field), populating
   the extra component from the `bz` callback supplied to the constructor.

### What it advances

The methods `advance` (SSPRK2), `advance_ssprk3` (SSPRK3) and `step` (unified entry
point with subcycling) call `SSPRK2Step::take_step` /
`SSPRK3Step::take_step` passing them the residual evaluator
(`coupler.hpp:115-162`). The coupler contains no integrator logic of its own.

### Notable template parameters

| Parameter | Role |
|---|---|
| `Limiter` | Interface reconstruction (NoSlope / Minmod / VanLeer / Weno5) |
| `Policy` | Frequency of the elliptic solve: `PerStageCoupling` (phi recomputed at each RK stage) or `OncePerStepCoupling` (phi frozen during the step) |
| `NumericalFlux` | Riemann flux (Rusanov / HLL / HLLC / Roe) |

### When to use it

- A single physical model, a single field, a single level.
- Unit test case, tutorial, quick validation.
- The multi-species model must go through `SystemAssembler` / `SystemDriver`.

---

## 3. SystemAssembler / SystemDriver (alias SystemCoupler) -- multi-species, single-level

**File:** `include/pops/coupling/system/system_coupler.hpp`

**Instantiations:**
- `pops::SystemAssembler<System, RhsAssembler, Elliptic = GeometricMG>`
- `pops::SystemDriver<System, RhsAssembler, Elliptic = GeometricMG>`
- `pops::SystemCoupler` is an alias of `SystemDriver` (historical compatibility).

### Separation of responsibilities

The design rationale (introductory comment of `system_coupler.hpp:29-40`)
explains the split into two classes:

> "SystemAssembler: ASSEMBLE. [...] It does NO time step.
> SystemDriver: AVANCE. [...] Advancing an assembler makes no sense.
> The Driver OWNS an Assembler."

**SystemAssembler (lines 63-167):**

- Holds the `CoupledSystem` (tuple of `EquationBlock`), the elliptic backend, the shared
  `aux` channel (a `MultiFab` common to ALL blocks).
- `solve_fields()`: assembles `f = Sum_s q_s n_s` via the supplied `RhsAssembler`
  (typically `ChargeDensityRhs`), solves Poisson, derives `aux = (phi, grad phi)`.
- `block_residual<Limiter, NumericalFlux>(block, state, R, recompute_aux)`:
  block residual evaluator at one stage (called by the Driver via the TimeStepper).
- Width of the `aux` channel = maximum of `aux_comps<Model>` over all blocks
  (`system_coupler.hpp:136-143`): a block requesting `B_z` (n_aux = 4) sees the
  extra component, a base block (n_aux = 3) does not see it; bit-identical to
  the history if no block requests an extra.

**SystemDriver (lines 169-361):**

- Owns a `SystemAssembler`.
- `step(dt, [implicit_callback])`: per-block subcycling according to `block_stride_v<Block>`,
  advances each block according to its `TimeTreatment`:
  - `Explicit`: call of `SSPRK2Step` / `SSPRK3Step` (or a user `TimeStepper`)
    passing the residual evaluator of the assembler.
  - `Implicit` / `IMEX`: field solve, explicit transport if IMEX, then
    delegated to the callback `implicit_advance(*this, block, dt, ...)`.
- `step_adaptive(cfl)`: adaptive macro step; each block computes its own `stride`
  at runtime from the ratio of wave speeds (`system_coupler.hpp:208-232`).
- `step_cfl(cfl)`: computes `dt_b = cfl * min(dx,dy) * substeps_b / (stride_b * w_b)` for each
  evolving block, keeps the minimum, then advances by one `step(dt)`. Substeps-aware formula (post-#121):
  bit-identical with the history ONLY for `substeps=1`. With `substeps>1`, the returned dt is
  larger than the old formula `cfl*h/w_max` (each substep stays at the stability limit).
  To reproduce a run calibrated with the old formula, use `step(dt)` with the explicit dt.
- `coupled_source_step(src, dt)`: inter-species coupling source stage (forward-Euler
  splitting), reading all blocks + aux (`system_coupler.hpp:276-283`).

### What RhsAssembler must expose

The `RhsAssembler` is a callable `operator()(const System&, MultiFab& rhs)`.
The header `elliptic_rhs.hpp` provides:

- `SingleModelEllipticRhs<Model>`: `f = model.elliptic_rhs(U)` (single-block).
- `TwoBlockChargeDensityRhs`: `f = q0 * n0 + q1 * n1` (two blocks, compatibility).
- `ChargeDensityRhs` (recommended): `f = Sum_s q_s n_s` generic to N blocks, requires
  a `SpeciesCharge` entry per block (`elliptic_rhs.hpp:117-133`).

### When to use it

- Several species on a uniform mesh.
- Explicit transport (SSPRK) or IMEX, with or without per-species subcycling.
- Use `SystemAssembler` directly if you want to write your own scheduler.
- Use `SystemDriver` (or its alias `SystemCoupler`) in all standard cases.

---

## 4. AmrCoupler -- REMOVED (#164)

The old single-box AMR E x B coupler `pops::AmrCoupler<Model, Elliptic>`
(`include/pops/coupling/amr_coupler.hpp`) has been **removed (#164)**. Its role is
entirely taken over by `AmrCouplerMP` (section 5), whose single-box is the
bit-identical degenerate case (validation guard `test_amr_multilevel_multipatch`).

---

## 5. AmrCouplerMP -- single-model, AMR multi-patch

**File:** `include/pops/coupling/amr/amr_coupler_mp.hpp`

**Instantiation:** `pops::AmrCouplerMP<Model, Elliptic = GeometricMG>`

### Role

Single-model AMR E x B coupler, multi-patch at each level. The only single-model
AMR coupler in production (the old single-box `AmrCoupler` has been removed, #164).
The single-box is the bit-identical degenerate case (validation guard
`test_amr_multilevel_multipatch`).

### Sequence of a step (`step`)

1. `update()` = `sync_down()` + `compute_aux()`:
   - `sync_down`: conservative fine -> coarse averaging over the whole hierarchy
     (`mf_average_down_mb`).
   - Coarse Poisson: `model.elliptic_rhs(U)` on level 0, MG solve.
   - `aux(0)` = `(phi, d_x phi, d_y phi)` by centered differences on the coarse level.
   - Piecewise-constant injection `aux(k-1) -> aux(k)` for each fine level via
     `detail::coupler_inject_aux_mb` (`amr_coupler_mp.hpp:56-96`).
2. `advance_amr<Disc>(model, levels, domain, dt, ...)`: conservative AMR step
   (Berger-Oliger + coverage-aware reflux, per-level subcycling).

### Notable construction parameters

- `replicated_coarse` (default `true`): level 0 is replicated on each rank
  (one box, single-rank). Pass `false` for a distributed multi-box coarse level
  (memory scalability; penalizes the MG if too many boxes). Bit-for-bit equivalence
  proven (`test_mpi_decoarse`, `maxdiff = 0`).
- `active`: optional "conductor cell" predicate (circular wall of the diocotron).

### Regrid

`regrid(crit, grow, margin)` delegates to `amr_regrid_coupler.hpp::amr_regrid_finest`
(Berger-Rigoutsos). The caller supplies the refinement criterion; the coupler
resynchronizes `aux` after regrid.

### Diagnostics

`mass()` and `max_drift_speed()` / `max_wave_speed()` delegated to `amr_diagnostics.hpp`.

### When to use it

- A single model, adaptive AMR with or without multi-patch.
- For a multi-species system on AMR, use `AmrSystemCoupler`.

---

## 6. AmrSystemCoupler (alias AmrSystemDriver) -- multi-species, AMR

**File:** `include/pops/coupling/system/amr_system_coupler.hpp`

**Instantiation:** `pops::AmrSystemCoupler<System, RhsAssembler, Elliptic = GeometricMG>`

`AmrSystemDriver` is an alias of `AmrSystemCoupler` (design note
`amr_system_coupler.hpp:371-375`: cosmetic split deferred, unified class
validated bit-identical).

### Role

Carries a `CoupledSystem` on an AMR hierarchy. All species share the
SAME per-level AMR grid (structural assumption) and therefore the SAME `aux` channel.
Poisson is only solved on the coarse level.

### Internal structure

- `block_levels_[b][k]`: `AmrLevelMP` of block b at level k (2D array).
- `aux_[k]`: a `MultiFab` common to all blocks at level k
  (`amr_system_coupler.hpp:130-139`).
- Width of the shared `aux` channel = max of `aux_comps<Model>` over the blocks (exact
  mirror of `SystemAssembler::system_aux_comps`, `amr_system_coupler.hpp:308-315`).

### Sequence of `solve_fields()`

1. `sync_down` per block (fine -> coarse, `mf_average_down_mb`).
2. `rhs_assembler_(system_, mg_.rhs())`: system RHS (e.g. `ChargeDensityRhs`).
3. `mg_.solve()`: Poisson on the coarse level.
4. `aux_[0]` = `(phi, grad phi)` via the same path as `SystemAssembler::derive_aux`
   (ghosts `bcPhi_`, `field_postprocess`, ghosts `aux_bc_`):
   `amr_system_coupler.hpp:187-197`.
5. Injection `aux_[k-1] -> aux_[k]` by `coupler_inject_aux_mb`.
6. Possible re-populating of `B_z` per level if `bz_` supplied (after injection, to
   preserve the spatial resolution of the field: `amr_system_coupler.hpp:199-205`).

### Poisson cadence (`PoissonCadence`)

- `OncePerStep` (default): `phi` solved a single time at the head of the step, frozen during
  the advance of the blocks. The cheapest.
- `PerSubstep`: `phi` re-solved before each species substep. More faithful for a
  transport strongly driven by the field.

Note: the single-level `SystemDriver` re-solves phi at EACH RK stage
(`recompute_aux = true`) because that is the maximal cadence by construction.

### Block advance (`step`)

For each block (according to its compile-time `stride`):

- `Explicit`: `advance_amr<Disc::Limiter, Disc::NumericalFlux>(model, levels, dt)`
  for each substep; re-solve Poisson before each substep if `PerSubstep`.
- `IMEX`: AMR transport on `SourceFreeModel<Block::Model>` (operator -div F only)
  then callback `implicit_advance(*this, block, levels, dt)`.
- `Implicit`: everything to the callback.

### Inter-species coupling source

`coupled_source_step(src, dt)`: re-points each block to its level k for each
level, then calls `src.apply(system, aux[k], dt)`: exact parity with
`SystemDriver::coupled_source_step` (`amr_system_coupler.hpp:266-283`).

### Implicit default: `AmrImplicitSourceStepper`

Backward-Euler (Newton) on the source, applied level by level
(`amr_system_coupler.hpp:361-370`). No user-side solver required.

### When to use it

- Several species, adaptive AMR.
- Same AMR grid for all species (central assumption).

---

## 7. RhsAssemblers: `elliptic_rhs.hpp`

**File:** `include/pops/coupling/base/elliptic_rhs.hpp`

These types are passed as `RhsAssembler` to the system couplers. They are not
couplers themselves but composition bricks.

| Type | Signature | Usage |
|---|---|---|
| `SingleModelEllipticRhs<Model>` | `operator()(const MultiFab& state, MultiFab& rhs)` | Single-block: `f = model.elliptic_rhs(U)` |
| `TwoFieldChargeDensityRhs` | `operator()(U0, U1, rhs)` | Two decoupled fields, legacy compatibility |
| `TwoBlockChargeDensityRhs` | `operator()(system, rhs)` | Two blocks of a CoupledSystem |
| `ChargeDensityRhs` | `operator()(system, rhs)` | N blocks: `f = Sum_s q_s n_s`; **recommended** |

`ChargeDensityRhs` requires a `SpeciesCharge{.charge, .comp}` entry per block:
if a species is neutral, declaring it with `charge = 0` is required
(`elliptic_rhs.hpp:122-125`).

---

## 8. CoupledSource: `coupled_source.hpp`

**File:** `include/pops/coupling/source/coupled_source.hpp`

A `CoupledSource` models an inter-species source term that depends on several
blocks AND on the potential. The concept requires `apply(system, aux, dt)`.

- `NoCoupledSource`: no-op, zero cost, single-species case or coupling only through
  Poisson (`coupled_source.hpp:37-40`).
- The concrete sources (collisions, friction, charge exchange) live in
  `adc_cases` or the core tests, not in the core itself.

---

## 9. CondensedSchurSourceStepper -- implicit Schur source stage

**File:** `include/pops/coupling/schur/source/condensed_schur_source_stepper.hpp`

Added in PR #126 (branch `feat/schur-pr4-stepper`).

### Role

STANDALONE source stage (transport frozen) implicitly solving the
potential / velocity / Lorentz coupling of a magnetized fluid block (Hoffart et al.,
arXiv:2510.11808). It does NOT replace and does NOT yet integrate into the
`System::step` path -- the facade wiring is planned in PR5
(`condensed_schur_source_stepper.hpp:23`).

### What it composes

1. `ElectrostaticLorentzCondensation` (`schur_condensation.hpp`, PR #124): assembles
   the condensed operator `A_op = I + c rho B^{-1}` (eps_x, eps_y diagonal, a_xy/a_yx
   crossed) and the condensed right-hand side.
2. `TensorKrylovSolver` (`krylov_solver.hpp`, PR #122): matrix-free BiCGStab
   preconditioned by a `GeometricMG` V-cycle on the symmetric part (the operator
   is non self-adjoint as soon as `B_z != 0`).
3. `LorentzEliminator` (`lorentz_eliminator.hpp`, PR #118): closed 2x2 `B^{-1}`
   elimination for velocity reconstruction.

### Sequence of a step (`step`)

1. Freeze `phi^n`; extract `v^n = (mx, my) / rho`; copy `B_z` into an internal buffer.
2. Assemble `A_op` and `rhs_schur`.
3. Solve `L_int(phi) = -rhs_schur` by BiCGStab (sign convention documented
   in the header: `L_schur = -L_int`, hence the negation of the RHS).
4. Reconstruct `v^{n+theta} = B^{-1}(v^n - theta dt grad phi^{n+theta})`.
5. Extrapolate `phi` and `v` from the theta-stage to the full step.
6. Update the kinetic energy if the `Energy` role is present.
7. Fill the ghosts of the state and the potential.

### Characteristics

- Generic: reads the roles `Density / MomentumX / MomentumY` of a `VariableSet`.
  Any fluid block that exposes them is eligible without additional Schur code.
- Device-clean: all kernels are named functors (no extended lambda,
  nvcc/GH200 compatible).
- Buffers allocated a single time at the constructor, reused at each `step()`.
- MPI-clean: the loops iterate over `local_size()`, the Krylov solve is collective.

### When to use it

- Stiff source coupling potential / velocity / Lorentz, implicit treatment required.
- Requires a fluid block with roles `Density / MomentumX / MomentumY`.
- Not connected to the `System::step` facade for now (PR5 to come).

---

## 10. Shared aux channel: extensible channel

The `aux` channel carries at minimum three components: `phi`, `d_x phi`, `d_y phi`
(constant `kAuxBaseComps = 3`). The extensible bricks add:

- component 3: `B_z(x, y)` (out-of-plane magnetic field, static, supplied by `bz_`).
- component 4: `T_e` (electron temperature, derived from `p/rho`).

The width is determined by `aux_comps<Model>()`. In the system couplers, it
is the MAX over all blocks: a base block (n_aux = 3) stays bit-identical to
the history. The mechanism is identical in `SystemAssembler`, `AmrSystemCoupler`
and `AmrCouplerMP` (see respectively `system_coupler.hpp:136`, `amr_system_coupler.hpp:308`,
and the construction of `AmrLevelStack`).

---

## 11. Summary table

| Coupler | Models | Levels | ASSEMBLE | AVANCE | Status |
|---|---|---|---|---|---|
| `Coupler<M,E>` | 1 | 1 (uniform) | yes (in the coupler) | yes | stable |
| `SystemAssembler<Sys,Rhs,E>` | N | 1 (uniform) | yes (system Poisson + aux) | NO | stable |
| `SystemDriver<Sys,Rhs,E>` (= `SystemCoupler`) | N | 1 (uniform) | via owned `SystemAssembler` | yes | stable |
| `AmrCouplerMP<M,E>` | 1 | N (multi-box) | yes | yes | stable |
| `AmrSystemCoupler<Sys,Rhs,E>` (= `AmrSystemDriver`) | N | N (multi-box) | yes | yes | stable |
| `CondensedSchurSourceStepper` | 1 fluid block | 1 (uniform) | Schur operator | source stage only | experimental (PR #126) |

The ASSEMBLE / AVANCE distinction takes on full meaning at the level of `SystemAssembler` vs
`SystemDriver`: `SystemAssembler` can be reused in a specialized scheduler
(external Newton, AP integrator) without dragging in the internal scheduling
of `SystemDriver`. `AmrSystemCoupler` has not (yet) undergone this split; the design
note (`amr_system_coupler.hpp:371-375`) flags it as deferred.

---

## 12. References in the source code

| Symbol | File | Line(s) |
|---|---|---|
| `Coupler<Model,Elliptic>` | `coupling/coupler.hpp` | 68 |
| `detail::coupler_eval_rhs` | `coupling/coupler.hpp` | 51 |
| `detail::coupler_grad_phi` | `coupling/coupler.hpp` | 61 |
| `SystemAssembler` | `coupling/system_coupler.hpp` | 63 |
| `SystemDriver` / `SystemCoupler` | `coupling/system_coupler.hpp` | 169, 360 |
| `AmrCouplerMP` | `coupling/amr_coupler_mp.hpp` | 225 |
| `detail::coupler_inject_aux_mb` | `coupling/amr_coupler_mp.hpp` | 56 |
| `AmrSystemCoupler` / `AmrSystemDriver` | `coupling/amr_system_coupler.hpp` | 70, 374 |
| `PoissonCadence` | `coupling/amr_system_coupler.hpp` | 63 |
| `AmrImplicitSourceStepper` | `coupling/amr_system_coupler.hpp` | 361 |
| `SingleModelEllipticRhs` | `coupling/elliptic_rhs.hpp` | 26 |
| `ChargeDensityRhs` | `coupling/elliptic_rhs.hpp` | 117 |
| `SpeciesCharge` | `coupling/elliptic_rhs.hpp` | 87 |
| `CoupledSourceFor` (concept) | `coupling/coupled_source.hpp` | 29 |
| `NoCoupledSource` | `coupling/coupled_source.hpp` | 37 |
| `CondensedSchurSourceStepper` | `coupling/condensed_schur_source_stepper.hpp` | 189 |
| `ElectrostaticLorentzCondensation` | `coupling/schur_condensation.hpp` | -- |
