# Add a new model

A *model* describes one equation: its flux, source, wave speeds and elliptic right-hand side.
This page covers the task of defining a model and running it in a simulation, and the choice
between the two writing fronts: native bricks and the symbolic DSL. For the concepts behind a
model, see [The physical model](../concepts/physical-model.md). For the full step-by-step
walkthrough of each front, follow the two write-a-model tutorials linked below.

This page assumes you have already built the `adc` Python module. If you have not, follow the
[installation guide](../getting_started/installation.md) first.

## Choose a writing front

Both fronts produce the same computational object on the C++ core and plug into an `adc.System`
the same way. Pick the front by whether the bricks you need already exist:

- Use **native bricks** when the bricks you need already exist. You compose four generic,
  pre-compiled bricks with `adc.Model(state, transport, source, elliptic)`. There is no
  just-in-time compilation, and you keep full production parity (MPI, AMR, GPU). Follow
  [Write a model with bricks](../tutorials/write-a-model-with-bricks.md).
- Use the **symbolic DSL** when the model you want does not exist as a native brick. You write
  the equation as formulas with `adc.dsl.Model`, then compile it into a `.so`. Follow
  [Write a model with the DSL](../tutorials/write-a-model-with-dsl.md).

The two fronts are interchangeable and produce an identical numerical kernel. The full brick
catalog is in the [brick reference](../reference/bricks_reference.md), and the formula declarators
are in the [DSL reference](../reference/dsl_reference.md).

## Define the model

For native bricks, compose the four slots and let `adc.Model` validate the
state-versus-transport consistency:

```python
model = adc.Model(state=adc.Scalar(), transport=adc.ExB(B0=1.0), source=adc.NoSource(), elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0))
```

For the DSL, declare the conservative variables, the auxiliary fields, the flux, the eigenvalues
and the elliptic right-hand side, then call `m.check()` to verify every referenced variable is
declared, and compile:

```python
compiled = diocotron_model(n_i0).compile(backend="production")
```

Replace `production` with `aot` for a marshaled, mono-rank `.so` for CPU debug or bench. The
default backend of `m.compile` is `aot`, so request `production` explicitly for the native
zero-copy path under MPI or AMR. For the backend trade-offs, see the
[backend matrix](../reference/backend_matrix.md).

## Run the model

Build a system, add the model as a block, set the Poisson coupling, set the initial density and
advance in time. The same `model` object plugs into `adc.System` (uniform grid) or `adc.AmrSystem`
(adaptive refinement) without change:

```python
sim = adc.System(n=96, L=1.0, periodic=True)
```

```python
sim.add_block("ne", model=model, spatial=adc.Spatial(minmod=True), time=adc.Explicit())
```

```python
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
```

```python
sim.set_density("ne", ne0)
```

```python
sim.step_cfl(0.4)
```

Replace `ne0` with your initial density: a contiguous 2D array indexed `ne[j, i]`.

## Next steps

- [Write a model with bricks](../tutorials/write-a-model-with-bricks.md) for the native front.
- [Write a model with the DSL](../tutorials/write-a-model-with-dsl.md) for the formula front.
- [Models overview](../models/index.md) for the hybrid front (`adc.CompositeModel`) and the
  `PhysicalModel` contract.
- [Configure a simulation](../simulation/index.md) to wire the system, Poisson and time integration.
