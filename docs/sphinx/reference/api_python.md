# API Python

Curated reference for the `adc` module (pybind11 bindings of `libadc` plus the object sugar of the
`adc/` package). Python composes a system block by block; all the cell-by-cell computation stays
in the compiled C++ lib (no numpy loop on the hot path, GPU/MPI preserved).

Only the public surface is documented here (internal symbols are not listed). For
annotated walkthroughs, see the [quickstart](../getting_started/first_run.md); named compositions
(scenarios) live in the [`adc_cases`](https://github.com/wolf75222/adc_cases) repository.

```{note}
The `autoclass` / `autofunction` blocks below only render if the `adc` module has been
built (`-DADC_BUILD_PYTHON=ON`) and is importable at doc build time. See
[installation](../getting_started/installation.md) and the [quickstart](../getting_started/first_run.md); watch out for the interpreter
footgun (the `.so` is linked to a specific cpython), detailed in [limitations](limitations.md).
```

## System: compose, configure, advance

`adc.System` is the coupler: you add blocks (one model per block), you configure a
shared system Poisson, you set the initial conditions in numpy, you advance. `add_block`
takes a composed model `adc.Model(...)`; `add_equation` dispatches on the model type
(native `ModelSpec` or `CompiledModel` from the DSL). `set_poisson(rhs=..., solver=..., bc=...)`
configures the system elliptic; `set_density` / `step_cfl` / `advance` / `run` drive
the advance; `density` / `mass` / `time` read the state.

```{eval-rst}
.. autoclass:: adc.System
   :members:
```

## AMR: composition on a refined hierarchy

`adc.AmrSystem` is the refined counterpart of `System`: one or more blocks carried on a
block-structured AMR hierarchy (Berger-Rigoutsos regrid, conservative reflux). In multi-block,
the hierarchy is re-gridded on the union of the tags (per-block density and/or `|grad phi|`). Same
`add_block` / `add_equation` signatures as `System`; the regrid cadence is carried by
`AmrSystemConfig.regrid_every`.

```{eval-rst}
.. autoclass:: adc.AmrSystem
   :members:
```

## Geometry and mesh

`adc.System` runs in Cartesian geometry by default; the polar / disk geometry is
selected via the `geometry`, `nr`, `ntheta`, `r_min`, `r_max` fields of `SystemConfig` (cf.
[Advanced topics](../advanced/index.md), geometry section). The exposed mesh classes:

```{eval-rst}
.. autoclass:: adc.CartesianMesh
.. autoclass:: adc.PolarMesh
```

## Native models: brick composition

A native model is assembled by `adc.Model(state, transport, source, elliptic)` from
generic bricks. The C++ core only knows these bricks (no scenario name). The
function validates the state <-> transport consistency (Scalar with ExB; compressible FluidState
with CompressibleFlux; isothermal with IsothermalFlux).

```{eval-rst}
.. autofunction:: adc.Model

.. autoclass:: adc.Scalar
.. autoclass:: adc.FluidState
.. autoclass:: adc.ExB
.. autoclass:: adc.CompressibleFlux
.. autoclass:: adc.IsothermalFlux
.. autoclass:: adc.NoSource
.. autoclass:: adc.PotentialForce
.. autoclass:: adc.GravityForce
.. autoclass:: adc.ChargeDensity
.. autoclass:: adc.BackgroundDensity
.. autoclass:: adc.GravityCoupling
```

## Per-block spatial scheme

Each block independently chooses its reconstruction (limiter), its numerical Riemann
flux and the reconstructed variables. `adc.FiniteVolume(...)` is a shortcut (the Riemann
flux is called `riemann` there, so as not to collide with the physical flux `m.flux` of the DSL)
that remaps onto the `adc.Spatial(...)` object.

```{eval-rst}
.. autoclass:: adc.Spatial
   :members:

.. autofunction:: adc.FiniteVolume
```

## Per-block time treatment

The time treatment is carried by the block (and not the model): the same model is reused
with distinct policies. `adc.Explicit` (SSPRK2/3, substeps, stride) is the default;
`adc.IMEX` / `adc.SourceImplicit` treat the stiff source implicitly (backward-Euler,
Newton local to the cell) while the transport stays explicit; this is not a global implicit
solver. `adc.Split` / `adc.Strang` are the opt-in for explicit/implicit splitting and
take a source stage `adc.CondensedSchur` (Schur condensation of the electrostatic Lorentz
coupling). `adc.Role` addresses a component by its physical meaning.

```{eval-rst}
.. autoclass:: adc.Explicit
   :members:

.. autoclass:: adc.IMEX
   :members:

.. autoclass:: adc.SourceImplicit
   :members:

.. autoclass:: adc.Split
   :members:

.. autoclass:: adc.Strang
   :members:

.. autoclass:: adc.CondensedSchur
   :members:

.. autoclass:: adc.Role
   :members:
```

```{note}
`adc.Implicit(...)` still exists as an alias of `adc.IMEX` but is obsolete (the name wrongly
suggests a global implicit solver) and emits a `DeprecationWarning`: use
`adc.SourceImplicit(...)` or `adc.IMEX(...)`.
```

## Inter-species couplings

Operator-split couplings applied after the transport, passed to `sim.add_coupling(...)`:
ionization, inter-species friction, thermal exchange.

```{eval-rst}
.. autoclass:: adc.Ionization
.. autoclass:: adc.Collision
.. autoclass:: adc.ThermalExchange
```

## Symbolic model DSL

The `adc.dsl` submodule describes a model in symbolic formulas (conservative variables,
auxiliaries, flux, eigenvalues, primitives, elliptic right-hand side), checks it, then
compiles it into a `.so` pluggable via `System.add_equation`. `dsl.Model` is the facade;
`dsl.CompiledModel` is the result of `m.compile(...)` (it carries the `.so` and the dispatch
metadata); `dsl.HybridModel` mixes native bricks and partial DSL bricks in a single
model (produced by `adc.CompositeModel(...)`).

```{eval-rst}
.. automodule:: adc.dsl
   :members: Model, CompiledModel, HybridModel
```

```{note}
Compilation backends (`m.compile(..., backend=...)`, default `aot`): `prototype` and `aot`
are CPU-only (no MPI/AMR/GPU); `production` is CPU + MPI + AMR. See
[backend matrix](backend_matrix.md) and [limitations](limitations.md). DSL design:
[DSL_API.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_API.md), [DSL_MODEL_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_MODEL_DESIGN.md).
```
