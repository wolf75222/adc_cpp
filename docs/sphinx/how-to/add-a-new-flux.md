# Add a numerical flux

This how-to selects the Riemann numerical flux that a block uses to combine the
reconstructed interface states: `rusanov`, `hll`, `hllc`, or `roe`. The numerical flux is
carried by the block (the spatial scheme), not by the model, so the same model runs with
different fluxes. For the math behind these solvers, see
[fluxes, sources and eigenvalues](../concepts/fluxes-sources-eigenvalues.md).

This page assumes you already have a model and a `System`. If not, start with the
[tutorial](../getting-started/tutorial.md).

## Choose a flux

Pass the flux through `adc.FiniteVolume`, where the numerical flux is named `riemann`:

```python
spatial = adc.FiniteVolume(limiter="minmod", riemann="rusanov")
```

Replace `riemann` with one of these values, matched to your model:

- `rusanov`: the generic minimal flux. It needs only `max_wave_speed`, so it works with any
  model. Use it as the default.
- `hll`: a generic flux with signed waves. It requires `model.wave_speeds` (a native
  isothermal or compressible model, or a DSL model that declares the primitive `p`). It is the
  path for a non-Euler model with signed waves; pair it with `minmod`.
- `hllc` and `roe`: 2D Euler only (4 variables and perfect-gas pressure). They require a
  compressible transport and a declared primitive `p`. Without `p`, the wiring raises a
  `ValueError`.

## Wire the flux to a block

Pass the spatial scheme as the `spatial=` argument of `add_block` or `add_equation`:

```python
sim.add_block("gas", model=model, spatial=adc.FiniteVolume(limiter="minmod", riemann="hll"))
```

For the full list of limiters, fluxes and reconstruction variables, see the
[native bricks reference](../reference/native-bricks.md).

## Declare a pressure for hllc or roe

`hllc` and `roe` read a pressure. A native `FluidState(kind="compressible")` carries it. A DSL
model must declare the primitive `p` and provide eigenvalues, which makes the generated brick
expose `pressure` and `wave_speeds`. See
[write a model with the DSL](../tutorials/write-a-model-with-dsl.md).

## Check backend support

The native `add_block` path supports every flux. On a compiled DSL model, the `prototype`
backend accepts `rusanov` only and rejects the other fluxes with a `ValueError`; the `aot` and
`production` backends accept all four. A wrong choice fails fast at plug time, never silently.
For the per-backend matrix, see the [backend matrix](../reference/backend-matrix.md).

## Where to go next

To add a flux that no native brick provides, write it as a hyperbolic brick in the DSL and
compile it: see [write a model with the DSL](../tutorials/write-a-model-with-dsl.md). To prototype
a flux in host Python without recompiling, see the `PythonFlux` entry in the
[native bricks reference](../reference/native-bricks.md).
