# Compiled time programs (`pops.time.Program`)

A **time program** is a restricted, *compiled* description of one time step. You write the
algorithm in Python with {py:mod}`pops.time`; Python only builds a typed intermediate
representation (IR). The IR is lowered to C++ and compiled into a single `problem.so` that runs
**entirely C++-side** during `sim.step(dt)`. No numerical stage ever re-enters Python or NumPy.

This is the companion of the *physical* DSL ({doc}`symbolic-dsl`): `pops.physics.facade.Model` describes the
local formulas (conservative variables, flux, sources); `pops.time.Program` describes the temporal
*algorithm* (stages, field solves, linear combinations, commit).

```{admonition} Design split
:class: note
The generated `problem.so` is **rich on specialization, poor on infrastructure**; `adc_cpp` is
**rich on infrastructure, poor on specific formulas**. Python *describes*, C++ *executes*. The IR is
interpreted by the codegen, **never** by the simulation -- the `.so` calls the stable
`ProgramContext` C++ API and does not reimplement MultiFab / `fill_boundary` / `assemble_rhs` /
`solve_fields` / the elliptic or Krylov solvers / MPI / Kokkos / CFL / substeps / write / checkpoint.
```

## Building a program

`Program` is a builder. State and RHS values are SSA nodes that support an affine algebra over the
symbolic time step `dt` (coefficients are polynomials in `dt`), so a whole family of schemes is
expressed without any per-scheme C++ class:

```python
import pops

P = pops.time.Program("forward_euler")
dt = P.dt

U = P.state("plasma")                       # current conservative state of the block
fields = P.solve_fields(U)                  # elliptic solve -> a FieldContext for THIS state
R = P.rhs(state=U, fields=fields, flux=True, sources=["default"])   # -div F(U) + named sources
U1 = P.linear_combine("U1", U + dt * R)     # fused linear combination
P.commit("plasma", U1)                       # replace the block state at end of step
```

Each `solve_fields` returns a *distinct* `FieldContext`; an RHS reads the fields solved from its own
state (never a stale global aux). Named sources are never summed implicitly -- `sources=[...]` lists
exactly the ones to include. Every advanced block must be committed exactly once.

The builder is validated strictly: a missing/duplicate commit, a use-before-definition, a runtime
scalar used as a Python `bool`, an unknown source, etc. all raise with an actionable message.

### Standard library

`pops.lib.time` provides ready schemes (called by their explicit names) that *lower to the same IR*
(they are not separate C++ steppers). A macro is an ordinary Python function that builds IR nodes -- it
never computes arrays. The schemes live in `pops.lib.time`; `pops.time` is the temporal language only
(`Program`, the scheduler, the optimizer passes).

| Macro | Scheme | Lowers to |
|---|---|---|
| `forward_euler(P, block)` | explicit Euler | `rhs` + `linear_combine` |
| `ssprk2` / `ssprk3` / `rk4` | SSP / classic Runge-Kutta | the stage chain (`rhs` + `linear_combine`) |
| `rk(P, block, tableau)` | generic explicit RK from a Butcher `tableau` | the data-driven stage chain (`RK4_TABLEAU`/`SSPRK2_TABLEAU` reproduce `rk4`/`ssprk2`) |
| `adams_bashforth(P, block, order)` | explicit AB, `order` in {1,2,3} | `history`/`store_history` ring (AB1 == `forward_euler`; `adams_bashforth2` aliases order 2) |
| `strang(P, block, H, S)` | Strang splitting H(dt/2);S(dt);H(dt/2) | three sub-flow stages |
| `lie(P, block, H, S)` | Lie (Godunov) splitting H(dt);S(dt) | two sub-flow stages |
| `imex_local(P, block, linear_source=)` | explicit flux/source + implicit cell-local linear source | `rhs` + `solve_local_linear` ((I - theta*dt*L)^{-1}) |
| `bdf(P, block, order)` | implicit-flux BDF1/BDF2 (Newton-Krylov) | `matrix_free_operator` + `rhs_jacvec` + GMRES `solve_linear` (+ history for BDF2) |
| `bdf(P, block, order, linear_source=)` | implicit BDF1/BDF2 (cell-local L fast path) | `solve_local_linear` (+ history for BDF2) |
| `condensed_schur(P, block, alpha=)` | implicit electrostatic-Lorentz source stage | the Schur assemble / Krylov / reconstruct chain |

`bdf` has two lowerings. The default (implicit FLUX) solves `F(U) = U - U^n - dt*rhs(U) = 0` (BDF2 adds
the `U^{n-1}` history term) by a matrix-free Newton-Krylov iteration: the Jacobian `J = I - c*dt*
d(rhs)/dU` is applied matrix-free by a finite-difference Jacobian-vector product (`P.rhs_jacvec`, which
calls `rhs` inside the apply) and each Newton step solves `J dU = -F` with GMRES. Naming a cell-local
`linear_source` instead selects the block-diagonal fast path (`solve_local_linear`, no Newton/Krylov).
The cell-local `imex_local` / `bdf` (linear-source path) / `condensed_schur` macros need the physical
model at codegen (they name an `m.linear_source` whose coefficients the per-cell kernel reads); the
implicit-flux `bdf` reuses the block's own compiled `rhs` closure, so it needs no extra model coupling.

A program body can also be recorded with the `@P.step` **decorator**: the decorated `build_fn(P)` runs
**once at build time** to populate the IR (never numerically during a step) and yields IR identical to
the inline builder body.

```python
P = pops.time.Program("fe")

@P.step
def _(P):
    pops.lib.time.forward_euler(P, "plasma")   # same IR as calling it inline
```

## Compiling and running

```python
m = pops.physics.facade.Model(...)                       # the physical model
compiled = pops.compile_problem(model=m, time=P, backend="production", target="system")

sim = pops.System(n=128, L=1.0, periodic=True)
sim.add_block("plasma", m, spatial=pops.FiniteVolume(...), time=pops.Explicit(method="euler"))
sim.install_program(compiled.so_path)        # dlopen + ABI-key check + install the program
sim.step(dt)                                  # the compiled program drives the step, C++-side
```

`compile_problem` lowers the IR (`Program.emit_cpp_program`), compiles it against the pops headers
with the **same Kokkos toolchain** as the loaded `_pops` module (so the `.so` is ABI-compatible and
loads in-process), and caches the `.so` out-of-source keyed by the generated source + the pops header
signature + the compiler + the C++ standard. A changed temporal coefficient changes the IR, hence
the cache key, hence forces a recompile -- a stale `.so` is never reused. `debug=True` writes the
generated `.cpp` next to the `.so` for inspection. `backend` must be `"production"` and `target`
`"system"` (other values raise a clear error).

`pops.CompiledTime(substeps=, stride=, cfl=)` records the compiled Program's macro-step cadence;
apply it with `sim.set_program_cadence(substeps, stride)` after `sim.install_program` (`substeps` and
`stride` orchestrate the program System-side, mirroring the native per-block loop -- `substeps>1` is
bit-exact vs native only for an uncoupled program, and `stride` is global, i.e. single-block exact).
The old time schemes (`pops.Explicit`, `pops.IMEX`, `pops.Strang`, `pops.CondensedSchur`) and the old
compile path (`m.compile`, `m.source`) keep working unchanged.

## Operator-first programs

The PDE-shaped builders above (`P.rhs`, `P.solve_fields`, `P.apply`, `P.source`) are sugar over a
more general layer: a program composes **typed operators by name**. After `P.bind_operators(model)`,
`P.call("name", *args)` resolves an operator against the model's typed registry, type-checks the
arguments against its signature, and lowers to the same IR as the matching shortcut, so the result
is identical:

```python
# PDE-shaped (spec 1)                       # operator-first (spec 2)
fields = P.solve_fields(U)            <=>    fields = P.call("fields_from_state", U)
R = P.rhs(state=U, fields=fields,     <=>    R = P.call("explicit_rhs", U, fields)
          flux=True, sources=["electric"])   #   (after m.rate_operator("explicit_rhs",
                                             #    flux=True, sources=["electric"]))
L = P.linear_source("lorentz")        <=>    L = P.call("lorentz", fields)
```

An operator-first program mentions no flux, source, Poisson or Lorentz; it composes operator calls,
linear combinations, solves and a commit. The same generic program then runs against any model that
provides operators with the expected signatures. The model-free type system, the operator kinds and
the `pops.lib.time` operator-first macros are documented in {doc}`operator-modules`; see
`examples/operator_modules/predictor_corrector_operator_first.py`.

## Optimization passes

The IR carries an **opt-in** pass pipeline. It never runs on the default `emit_cpp_program` path, so
it cannot change an already-compiled program; you optimize a copy explicitly. The hard requirement is
that **optimization must not change results**: each transform pass is proven to preserve the emitted
numerics and is byte-for-byte identical when it finds nothing to do, so a program with no optimizable
structure emits identical C++ with the pipeline on or off.

| Method | Kind | What it does |
|---|---|---|
| `P.eliminate_dead_nodes()` | transform | Drop unconsumed flat nodes whose op allocates a fresh result and has no side effect (an explicit allow-list); every buffer-writer / side-effecting / unknown op is kept. |
| `P.eliminate_common_subexpressions()` | transform | Compute a duplicated **pure** sub-IR (same op + inputs + attrs) once and alias the rest. Only `_PURE_OPS` are candidates; a reduce, a solve, a buffer-writer or a side-effecting op is never collapsed. The consumer's `axpy` structure is preserved, so the result is bit-identical. |
| `P.eliminate_redundant_field_solves()` | transform | Remove a second `solve_fields` over the same state when **no** state/aux mutation (a commit, `project`, `fill_boundary`, `store_history`, another field solve) intervenes; conservative -- kept otherwise. |
| `P.optimize()` | pipeline | Run the three proven-safe transforms in sequence; byte-identical on an already-optimal program. |
| `P.dump_passes()` | report | Trace each pass's node-count delta and whether it changed the IR hash (the original is never mutated). |
| `P.scratch_liveness()` | analysis | Per-scratch live ranges `[def, last use]` over the linear step-body order. |
| `P.buffer_reuse_report()` | analysis | Greedy buffer allocation over disjoint live ranges -- the minimum buffer count the codegen could reuse to. |
| `P.estimate()` / `P.estimate_report()` | analysis | Static, grid-relative memory-traffic (field-sized passes) + kernel-count estimate. |
| `P.gpu_detectors()` | analysis | Flag too-many-small-kernels / too-many-scratches / excessive memory traffic (host-side heuristic warnings, never a hard error). |

The analysis passes are reports: the static cost estimate is a host-side prediction (the **measured**
GPU kernel count / occupancy is a profiled run, not a build-time figure). The transform passes are the
generic-IR realization of Spec 3 section 28; a pass that cannot be made sound for the general IR stays
a report rather than a transform that could change numerics.

## What is implemented today

| Capability | Status |
|---|---|
| `pops.time.Program` builder IR + `pops.lib.time` macros | available |
| `Program.emit_cpp_program` codegen, **Forward Euler** (`forward_euler_program.py`) | available, runs end-to-end |
| `pops.compile_problem` + cache + `debug=` + `sim.install_program` + `pops.CompiledTime` | available |
| Named sources / linear sources on `pops.physics.facade.Model` (`m.source_term`, `m.linear_source`) | available |
| Multi-stage codegen (SSPRK2 / SSPRK3 / RK4) (`ssprk2_program.py`, `ssprk3_program.py`, `rk4_program.py`) | available, runs end-to-end |
| `P.source` / `P.apply` / `P.solve_local_linear` (split sources, Lorentz) + predictor-corrector (`predictor_corrector_poisson_lorentz.py`) | available, runs end-to-end |
| `strang` splitting combinator (compiled H(dt/2); S(dt); H(dt/2) == native `pops.Strang`) (`strang_program.py`) | available, runs end-to-end |
| Structured control flow (`P.range` / `P.static_range` / `P.while_` / `P.if_`) + reductions (`P.norm2` / `P.dot` / `P.norm_inf`) | available, runs end-to-end |
| Matrix-free operators (`P.matrix_free_operator` / `P.set_apply`) + Krylov (`P.solve_linear`: cg / bicgstab / richardson) (`matrix_free_solve.py`) | available, runs end-to-end |
| Differential primitives (`P.gradient` / `P.divergence` / `P.laplacian`) in a matrix-free apply (`divergence_solve.py`) | available, runs end-to-end |
| `P.solve_local_linear` (per-cell implicit local linear solve, e.g. Lorentz / relaxation) | available, runs end-to-end |
| `P.solve_local_nonlinear` (per-cell Newton solve of a local non-linear residual, e.g. an implicit reaction) (`test_time_local_newton.py`) | available, runs end-to-end |
| `P.solve_fields` / `solve_fields_from_state` (per-stage elliptic field solve from the stage state) | available, runs end-to-end |
| `step_cfl` routes through the compiled program (`sim.step_cfl` honors the installed Program) | available, runs end-to-end |
| `substeps` / `stride` cadence (`pops.CompiledTime`, `sim.set_program_cadence`) | available, runs end-to-end |
| Histories / multistep (Adams-Bashforth, `P.history` / `P.store_history`) (`adams_bashforth2_program.py`) | available, runs end-to-end |
| Checkpoint / restart of a compiled Program (history slices serialized) | available, runs end-to-end |
| Reductions `P.sum` / `P.max` / `P.min` / `P.sum_component`, `P.fill_boundary`, `P.project` (block positivity), `P.record_scalar` diagnostics (`sim.program_diagnostic`) | available, runs end-to-end |
| Anisotropic coefficient assembly (`P.schur_coeffs` / `P.apply_laplacian_coeff` / `P.schur_rhs` / `P.schur_reconstruct`) | available, runs end-to-end |
| `condensed_schur` as a Program macro (ADC-421) | available (`theta == 1` backward-Euler source stage from `phi^n = 0`); near-match to native `pops.CondensedSchur` (no preconditioner); cross-step phi carry / `theta < 1` extrapolation / energy update deferred |
| `std.rk` (generic Butcher tableau) / `std.lie` / `std.imex_local` / `std.adams_bashforth(order)` + `@P.step` decorator (ADC-423) | available (all lower to the existing IR) |
| `std.bdf` implicit-flux BDF1/BDF2 (matrix-free Newton-Krylov; `P.rhs_jacvec` FD Jacobian-vector + GMRES) (ADC-431) (`test_time_bdf.py`) | available, runs end-to-end (cell-local-`L` fast path unchanged) |
| Multi-block programs: N `P.state` / N `P.commit`, each op routed to its block index (ADC-426) | available, runs end-to-end (add the System blocks in `P.state` declaration order); per-block `P.solve_fields(state=)` is a coupled solve; the simultaneous multi-target `P.solve_fields_from_blocks` lowers to `ctx.solve_fields_from_blocks` (Spec 3 crit 24, ADC-457) |

Each lowered path is verified against an independent reference to machine precision: Forward Euler /
SSPRK2 / SSPRK3 reproduce the native `pops.Explicit` step bit-for-bit; the compiled `strang` macro
reproduces the native `pops.Strang` macro-step bit-for-bit on an uncoupled model; RK4 matches an offline
stage reference; Adams-Bashforth 2 (over the System-owned history, with the runtime cold start) matches
an offline AB2 reference to machine precision; the split-source predictor-corrector (Poisson/Lorentz)
and the matrix-free `solve_linear` (CG / BiCGStab solves of `(I - alpha*Lap) phi = b` and
`(I - alpha*div(grad)) phi = b`) match offline references to ~1e-15; the per-cell `solve_local_nonlinear`
Newton solve of an implicit reaction `r(U) = U - U0 - dt*S(U) = 0` matches the analytic closed form and
an offline numpy Newton on the identical residual to ~1e-10; the dynamic `while_` / `range` / `if_`
loops run a runtime-dependent number of iterations entirely C++-side. `compile_problem` raises a clear
`NotImplementedError` for any construct the codegen cannot yet lower, rather than mis-lowering it.

The `condensed_schur` macro (ADC-421) composes the anisotropic coefficient assembly
(`P.schur_coeffs` -> the native `A = I + c*rho*B^{-1}` tensor), the coefficiented matrix-free matvec
(`P.apply_laplacian_coeff` -> `pops::apply_laplacian`'s coefficient path), the fused RHS (`P.schur_rhs`
-> `-Lap(phi^n) - theta*dt*alpha*div(B^{-1}(mx,my))`) and the closed-B^{-1} velocity reconstruction
(`P.schur_reconstruct`) with a matrix-free `P.solve_linear` (BiCGStab), mirroring the native
`CondensedSchurSourceStepper` assemble / solve / reconstruct sequence. It is a **near-match** to the
native `pops.CondensedSchur`, which preconditions the BiCGStab solve with a GeometricMG V-cycle: the
operator and RHS are identical, the Krylov path differs (both converge to the same phi at tolerance), so
`test_time_condensed_schur.py` checks it against an offline reference of the identical
assemble / solve / reconstruct steps rather than bit-against-native. The macro lowers the `theta == 1`
(backward-Euler) source stage from `phi^n = 0`; cross-step phi carry (the System history ring is
state-sized, so a scalar phi cannot ride it safely yet), the `theta < 1` n+1 extrapolation (a
momentum-only update with no IR op yet) and the energy-role update are deferred -- the macro raises for
`theta != 1` rather than mis-lower. The native `pops.CondensedSchur` source stepper stays fully
supported and untouched.

See {doc}`symbolic-dsl` for the physical model DSL and `examples/time_programs/` for runnable
programs. The runnable time programs are `forward_euler_program.py`, `ssprk2_program.py`,
`ssprk3_program.py`, `rk4_program.py`, `strang_program.py`, `adams_bashforth2_program.py`,
`predictor_corrector_poisson_lorentz.py`, `matrix_free_solve.py`, `divergence_solve.py`, and
`condensed_schur_program.py`; each self-skips cleanly (exit 0) without a compiler / a visible Kokkos.
