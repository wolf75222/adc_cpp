# API Python

Curated reference for the `pops` module (pybind11 bindings of `libadc` plus the object sugar of the
`pops/` package). Python composes a system block by block; all the cell-by-cell computation stays
in the compiled C++ lib (no numpy loop on the hot path, GPU/MPI preserved).

Only the public surface is documented here (internal symbols are not listed). For
annotated walkthroughs, see the [quickstart](../getting-started/first-run.md); named compositions
(scenarios) live in the [`adc_cases`](https://github.com/wolf75222/adc_cases) repository.

## Python sub-packages (Spec 4)

The `pops` package is organized into typed sub-packages with a strictly acyclic
import graph. The old flat modules (`pops.dsl`, the flat `pops.moments`, etc.) are
deleted; no backward-compatibility shims exist.

| Sub-package | Responsibility |
|-------------|---------------|
| `pops.ir` | Symbolic IR nodes, ops, values, lowering, visitors. Imported by every layer above it. |
| `pops.model` | Typed model core: `Module`, spaces, `Operator`, `OperatorRegistry`. Imports `pops.ir` only. |
| `pops.physics` | Math/physics authoring facade (`pops.physics.Model`). Lowers to a `pops.model.Module`. |
| `pops.time` | Temporal language: `Program`, schedules, equations. |
| `pops.lib` | Descriptors, time schemes, moment closures (`pops.lib.moments`), provided standard models. |
| `pops.codegen` | The only C++ emitter. Free functions (`emit_cpp_*`, `module_codegen`) that take a model. |

Key rule: `pops.physics`, `pops.time`, and `pops.lib` never import `pops.codegen` or `_pops`.
`pops.codegen` holds C++ emission as free functions so authoring packages stay decoupled from the
toolchain. See the full design: {doc}`../design/spec4-package-architecture`.

The top-level `pops` namespace re-exports the most-used symbols so existing simulation code
(`pops.System`, `pops.Model(...)`, `pops.compile_problem(...)`) continues to work unchanged.

```{note}
The `autoclass` / `autofunction` blocks below only render if the `pops` module has been
built (`-DPOPS_BUILD_PYTHON=ON`) and is importable at doc build time. See
[installation](../getting-started/installation.md) and the [quickstart](../getting-started/first-run.md); watch out for the interpreter
footgun (the `.so` is linked to a specific cpython), detailed in [limitations](known-limitations.md).
```

## System: compose, configure, advance

`pops.System` is the coupler: you add blocks (one model per block), you configure a
shared system Poisson, you set the initial conditions in numpy, you advance. `add_block`
takes a composed model `pops.Model(...)`; `add_equation` dispatches on the model type
(native `ModelSpec` or `CompiledModel` from the DSL). `set_poisson(rhs=..., solver=..., bc=...)`
configures the system elliptic; `set_density` / `step_cfl` / `advance` / `run` drive
the advance; `density` / `mass` / `time` read the state.

```{eval-rst}
.. autoclass:: pops.System
   :members:
```

## AMR: composition on a refined hierarchy

`pops.AmrSystem` is the refined counterpart of `System`: one or more blocks carried on a
block-structured AMR hierarchy (Berger-Rigoutsos regrid, conservative reflux). In multi-block,
the hierarchy is re-gridded on the union of the tags (per-block density and/or `|grad phi|`). Same
`add_block` / `add_equation` signatures as `System`; the regrid cadence is carried by
`AmrSystemConfig.regrid_every`.

```{eval-rst}
.. autoclass:: pops.AmrSystem
   :members:
```

## Geometry and mesh

`pops.System` runs in Cartesian geometry by default; the polar / disk geometry is
selected via the `geometry`, `nr`, `ntheta`, `r_min`, `r_max` fields of `SystemConfig` (cf.
[Advanced topics](../advanced/index.md), geometry section). The exposed mesh classes:

```{eval-rst}
.. autoclass:: pops.CartesianMesh
.. autoclass:: pops.PolarMesh
```

## Native models: brick composition

A native model is assembled by `pops.Model(state, transport, source, elliptic)` from
generic bricks. The C++ core only knows these bricks (no scenario name). The
function validates the state <-> transport consistency (Scalar with ExB; compressible FluidState
with CompressibleFlux; isothermal with IsothermalFlux).

```{eval-rst}
.. autofunction:: pops.Model

.. autoclass:: pops.Scalar
.. autoclass:: pops.FluidState
.. autoclass:: pops.ExB
.. autoclass:: pops.CompressibleFlux
.. autoclass:: pops.IsothermalFlux
.. autoclass:: pops.NoSource
.. autoclass:: pops.PotentialForce
.. autoclass:: pops.GravityForce
.. autoclass:: pops.MagneticLorentzForce
.. autoclass:: pops.PotentialMagneticForce
.. autoclass:: pops.ChargeDensity
.. autoclass:: pops.BackgroundDensity
.. autoclass:: pops.GravityCoupling
```

## Elliptic model: operator, right-hand side, output

The system elliptic is not a hard-coded Poisson special case: `pops.elliptic(...)` composes an
`EllipticModel` from an operator (`div_eps_grad`), a right-hand side (`composite_rhs`, the generic
sum of the per-block elliptic bricks, or its usual case `charge_density`) and an output
(`electric_field_from_potential`). `System.set_poisson(...)` is the shortcut for the Poisson
instance; `EllipticSolver` selects the linear solver. The `elliptic` field of `pops.Model(...)`
(see above) decides which brick each block contributes to the right-hand side.

```{eval-rst}
.. autofunction:: pops.elliptic
.. autoclass:: pops.EllipticModel
.. autoclass:: pops.EllipticSolver

.. autofunction:: pops.div_eps_grad
.. autofunction:: pops.charge_density
.. autofunction:: pops.composite_rhs
.. autofunction:: pops.electric_field_from_potential
```

## Named aux fields

Beyond the canonical aux channel (`phi`, `grad_x`, `grad_y`, `B_z`, `T_e`), a model can declare
named auxiliary fields with `m.aux_field(name)` (see
[aux vs aux_field](#aux-vs-aux-field) in the DSL reference). Each is set per block
via `System.set_aux_field(block, name, array, halo=...)`; `pops.AuxHalo` is the optional per-field
ghost boundary policy (`foextrap` zero-gradient, or `dirichlet` with a fixed `value`), applied to
the non-periodic faces only. Default (no `halo`) keeps the shared aux boundary, bit-identical.

```{eval-rst}
.. autoclass:: pops.AuxHalo
```

## Per-block spatial scheme

Each block independently chooses its reconstruction (limiter), its numerical Riemann
flux and the reconstructed variables. `pops.FiniteVolume(...)` is a shortcut (the Riemann
flux is called `riemann` there, so as not to collide with the physical flux `m.flux` of the DSL)
that remaps onto the `pops.Spatial(...)` object.

```{eval-rst}
.. autoclass:: pops.Spatial
   :members:

.. autofunction:: pops.FiniteVolume

.. autoclass:: pops.PythonFlux
   :members:
```

## Per-block time treatment

The time treatment is carried by the block (and not the model): the same model is reused
with distinct policies. `pops.Explicit` (SSPRK2/3, substeps, stride) is the default;
`pops.IMEX` / `pops.SourceImplicit` treat the stiff source implicitly (backward-Euler,
Newton local to the cell) while the transport stays explicit; this is not a global implicit
solver. `pops.IMEXRK` is the order-2 IMEX-RK family (ARS(2,2,2) scheme), Cartesian-System only.
`pops.Split` / `pops.Strang` are the opt-in for explicit/implicit splitting and
take a source stage `pops.CondensedSchur` (Schur condensation of the electrostatic Lorentz
coupling). `pops.Role` addresses a component by its physical meaning.

```{eval-rst}
.. autoclass:: pops.Explicit
   :members:

.. autoclass:: pops.IMEX
   :members:

.. autoclass:: pops.IMEXRK
   :members:

.. autoclass:: pops.SourceImplicit
   :members:

.. autoclass:: pops.Split
   :members:

.. autoclass:: pops.Strang
   :members:

.. autoclass:: pops.CondensedSchur
   :members:

.. autoclass:: pops.Role
   :members:
```

```{note}
`pops.Implicit(...)` still exists as an alias of `pops.IMEX` but is obsolete (the name wrongly
suggests a global implicit solver) and emits a `DeprecationWarning`: use
`pops.SourceImplicit(...)` or `pops.IMEX(...)`.
```

## Inter-species couplings

Operator-split couplings applied after the transport, passed to `sim.add_coupling(...)`:
ionization, inter-species friction, thermal exchange.

```{eval-rst}
.. autoclass:: pops.Ionization
.. autoclass:: pops.Collision
.. autoclass:: pops.ThermalExchange
```

## Symbolic model DSL

The `pops.physics` and `pops.ir` packages describe a model in symbolic formulas (conservative
variables, auxiliaries, flux, eigenvalues, primitives, elliptic right-hand side), check it, then
compiles it into a `.so` pluggable via `System.add_equation`. `pops.physics.facade.Model` is the facade;
`pops.codegen.loader.CompiledModel` is the result of `m.compile(...)` (it carries the `.so` and the dispatch
metadata); `pops.physics.hybrid.HybridModel` mixes native bricks and partial DSL bricks in a single
model (produced by `pops.CompositeModel(...)`).

```{eval-rst}
.. autoclass:: pops.physics.facade.Model
   :members:

.. autoclass:: pops.codegen.loader.CompiledModel
   :members:

.. autoclass:: pops.physics.hybrid.HybridModel
   :members:

.. autofunction:: pops.CompositeModel
```

```{note}
Compilation backends (`m.compile(..., backend=...)`, default `auto`, which auto-selects
`production` under toolchain parity with the installed `_pops`, otherwise `aot`; explicit
`prototype | aot | production` still available): `prototype` and `aot`
are CPU-only (no MPI/AMR/GPU); `production` is CPU + MPI + AMR. See
[backend matrix](backend-matrix.md) and [limitations](known-limitations.md). DSL design:
[DSL_API.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_API.md), [DSL_MODEL_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_MODEL_DESIGN.md).
```

## Moment models

The `pops.lib.moments` submodule generates a `pops.physics.facade.Model` for a 2D velocity-moment hierarchy from a
single closure: the central and standardized moments, the flux, and the signed wave speeds are
derived, so you write only the closure (and, optionally, the sources). For the concept see
[moments and closures](../concepts/moments-and-closures.md), for the worked example
[the HyQMOM tutorial](../tutorials/moment-model-hyqmom15.md), and for the prose reference
[moment models](moment-models.md).

```{eval-rst}
.. automodule:: pops.lib.moments
   :members: build_moment_model, gaussian_closure, lorentz_sources, moment_names, moment_indices
```

## Capabilities matrix

`pops.capabilities()` returns the support matrix by facade / geometry / backend. It is the single
source of truth for what each path actually wires (riemann, time, stability policy, poisson,
geometry, schur, DSL backends, io, AMR layout, regrid, aux) -- other pages key off it rather than
re-listing combinations that can drift. The returned keys are `dimension`, `riemann`, `time`,
`stability_policy`, `poisson`, `geometry`, `schur`, `backends_dsl`, `io`, `amr_layout`,
`regrid`, and `aux`. Combinations outside the matrix raise an explicit error on the C++ side rather than
being silently ignored.

```{eval-rst}
.. autofunction:: pops.capabilities
```

## Environment and diagnostics

Runtime introspection and the single thread knob. `pops.set_threads(n=None)` writes
`OMP_NUM_THREADS` and `KOKKOS_NUM_THREADS` and MUST be called before the first `System`
(Kokkos initializes lazily then); `pops.has_kokkos()` and `pops.parallel_info()` report the
compiled backend and current thread state; `pops.doctor()` is the one-command troubleshooting
entry point; `pops.abi_key()` returns the module ABI key used by the production DSL path; and the
module attribute `pops.__version__` carries the version baked into the extension (single source:
`project(VERSION)` in CMake).

```{eval-rst}
.. autofunction:: pops.set_threads
.. autofunction:: pops.has_kokkos
.. autofunction:: pops.parallel_info
.. autofunction:: pops.doctor
.. autofunction:: pops.abi_key
```
