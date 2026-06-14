# Configure outputs and diagnostics

This guide shows how to read the fields and diagnostics a run produces from an
`adc.System` (or `adc.AmrSystem`). The solver does not write files by itself: it returns
its state as numpy arrays that you read and save from Python. This assumes you already
have a composed and initialized system. To build one, see
[the simulation guide](../simulation/index.md).

The fields come back row-major, shape `(NY, NX)` or `(NCOMP, NY, NX)`, where `NY` is the
number of rows (slow `y` axis) and `NX` the number of columns (fast `x` axis). A field is
indexed `field[j, i]` with `j` the row and `i` the column.

## Read the per-block fields

Read the density of a block by its name. Replace `NAME` with a block name from
`sim.block_names()`.

```python
rho = sim.density("ne")
```

Read the full conservative state of a fluid block, shape `(NCOMP, NY, NX)`, for example
`[rho, rho*u, rho*v, E]` for Euler.

```python
U = sim.get_state("electrons")
```

Read the same state in named primitive variables, returned as a dict keyed by the
primitive names of the model (for an Euler block, `rho`, `u`, `v`, `p`).

```python
P = sim.get_primitive_state("electrons")
```

## Read the shared diagnostics

Read the electrostatic potential `phi` solved on the shared system Poisson, shape `(NY, NX)`.

```python
phi = sim.potential()
```

Read the conservation invariant: the total mass of a block. Compare it against the initial
mass to track drift.

```python
mass = sim.mass("ne")
```

Read the current physical time of the run.

```python
t = sim.time()
```

## Refresh the field at a frozen state

Solve the system Poisson on the current state and repopulate the `phi` and `grad phi`
channels without advancing in time. Call this before `sim.potential()` when you want the
field at a state you did not reach with a time step.

```python
sim.solve_fields()
```

## Next steps

- Set the initial fields a run starts from with [the simulation guide](../simulation/index.md).
- Work an end-to-end run in [the tutorial](../getting-started/tutorial.md).
