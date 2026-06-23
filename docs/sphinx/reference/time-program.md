# Compiled time programs (`adc.time.Program`)

A **time program** is a restricted, *compiled* description of one time step. You write the
algorithm in Python with {py:mod}`adc.time`; Python only builds a typed intermediate
representation (IR). The IR is lowered to C++ and compiled into a single `problem.so` that runs
**entirely C++-side** during `sim.step(dt)`. No numerical stage ever re-enters Python or NumPy.

This is the companion of the *physical* DSL ({doc}`symbolic-dsl`): `adc.dsl.Model` describes the
local formulas (conservative variables, flux, sources); `adc.time.Program` describes the temporal
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
import adc

P = adc.time.Program("forward_euler")
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

`adc.time.std` provides macros that *lower to the same IR* (they are not separate C++ steppers):
`forward_euler`, `ssprk2`, `ssprk3`, `rk4`, and a `strang` splitting combinator. A macro is an
ordinary Python function that builds IR nodes -- it never computes arrays.

## Compiling and running

```python
m = adc.dsl.Model(...)                       # the physical model
compiled = adc.compile_problem(model=m, time=P, backend="production", target="system")

sim = adc.System(n=128, L=1.0, periodic=True)
sim.add_block("plasma", m, spatial=adc.FiniteVolume(...), time=adc.Explicit(method="euler"))
sim.install_program(compiled.so_path)        # dlopen + ABI-key check + install the program
sim.step(dt)                                  # the compiled program drives the step, C++-side
```

`compile_problem` lowers the IR (`Program.emit_cpp_program`), compiles it against the adc headers
with the **same Kokkos toolchain** as the loaded `_adc` module (so the `.so` is ABI-compatible and
loads in-process), and caches the `.so` out-of-source keyed by the generated source + the adc header
signature + the compiler + the C++ standard. A changed temporal coefficient changes the IR, hence
the cache key, hence forces a recompile -- a stale `.so` is never reused. `debug=True` writes the
generated `.cpp` next to the `.so` for inspection. `backend` must be `"production"` and `target`
`"system"` (other values raise a clear error).

`adc.CompiledTime(substeps=, stride=, cfl=)` records the compiled Program's macro-step cadence;
apply it with `sim.set_program_cadence(substeps, stride)` after `sim.install_program` (`substeps` and
`stride` orchestrate the program System-side, mirroring the native per-block loop -- `substeps>1` is
bit-exact vs native only for an uncoupled program, and `stride` is global, i.e. single-block exact).
The old time schemes (`adc.Explicit`, `adc.IMEX`, `adc.Strang`, `adc.CondensedSchur`) and the old
compile path (`m.compile`, `m.source`) keep working unchanged.

## What is implemented today

| Capability | Status |
|---|---|
| `adc.time.Program` builder IR + `adc.time.std` macros | available |
| `Program.emit_cpp_program` codegen, single-block **Forward Euler** | available, runs end-to-end |
| `adc.compile_problem` + cache + `debug=` + `sim.install_program` + `adc.CompiledTime` | available |
| Named sources / linear sources on `adc.dsl.Model` (`m.source_term`, `m.linear_source`) | available |
| Multi-stage codegen (SSPRK2 / SSPRK3 / RK4) | available, runs end-to-end |
| `P.source` / `P.apply` / `P.solve_local_linear` (split sources, Lorentz) + predictor-corrector | available, runs end-to-end |
| Structured control flow (`P.range` / `P.static_range` / `P.while_` / `P.if_`) + reductions (`P.norm2` / `P.dot` / `P.norm_inf`) | available, runs end-to-end |
| Matrix-free operators (`P.matrix_free_operator` / `P.set_apply`) + Krylov (`P.solve_linear`: cg / bicgstab / richardson) | available, runs end-to-end |
| Histories / multistep (Adams-Bashforth) + checkpoint/restart | in progress |
| `condensed_schur` as a Program macro | planned (the matrix-free + Krylov primitives are in place) |

Each lowered path is verified against an independent reference to machine precision: Forward Euler /
SSPRK2 reproduce the native `adc.Explicit` step bit-for-bit; RK4 matches an offline stage reference;
the split-source predictor-corrector (Poisson/Lorentz) and the matrix-free `solve_linear` (a CG solve
of `(I - alpha*Lap) phi = b`) match offline references to ~1e-15; the dynamic `while_` / `range` /
`if_` loops run a runtime-dependent number of iterations entirely C++-side. `compile_problem` raises a
clear `NotImplementedError` for any construct the codegen cannot yet lower, rather than mis-lowering it.

See {doc}`symbolic-dsl` for the physical model DSL and `examples/time_programs/` for runnable
programs.
