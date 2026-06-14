# Define initial conditions

Set the initial density and fields of a block from numpy arrays, then advance the run. This
page assumes you already have a `System` with at least one block and a configured Poisson; see
[Configure a simulation](../simulation/index.md) for that. The layout convention is row-major
`(ny, nx)`: index a field as `arr[j, i]`, where `j` is the row (`y` axis) and `i` is the column
(`x` axis).

## Set a scalar density

`set_density(NAME, ARR)` sets the density (component 0) of block `NAME` and leaves the rest at
rest. `ARR` is an `(n, n)` array; `n` is the per-side cell count passed to `adc.System`. Use it
for a scalar transport block.

1. Build the cell-center coordinates and the field.

   ```python
   coord = (np.arange(n) + 0.5) / n * L
   ```

2. Mesh the coordinates with `indexing="xy"` so both arrays have shape `(ny, nx)`.

   ```python
   xx, yy = np.meshgrid(coord, coord, indexing="xy")
   ```

3. Fill the density array and apply it.

   ```python
   ne = np.full((n, n), 1e-3)
   ```

   ```python
   sim.set_density("ne", ne)
   ```

For a periodic Poisson, fix a neutralizing background equal to the mean (`n_i0 = ne.mean()`)
so the right-hand side is solvable.

## Set a fluid state from primitives

`set_primitive_state(NAME, **PRIMS)` initializes a fluid block from its named primitive
variables (`rho`, `u`, `v`, `p`). Each primitive is an `(n, n)` array, and the model converts
them to conservative variables. An unknown or missing name raises an error.

```python
sim.set_primitive_state("electrons", rho=rho0, u=u0, v=v0, p=p0)
```

For an explicit conservative state, `set_state(NAME, U)` takes the flattened `(ncomp, n, n)`
array.

## Advance and check the result

Set the density, then step the run. `step_cfl(CFL)` takes one step at the given CFL number and
returns the effective `dt`; `advance(DT, STEPS)` takes `STEPS` steps of fixed `DT`.

```python
sim.step_cfl(0.4)
```

Read `sim.density(NAME)` for the field, `sim.potential()` for `phi`, and `sim.mass(NAME)` for
the total mass. The mass is the conservation invariant; with periodic advective transport its
drift stays near machine roundoff.

## Next steps

- [Configure a simulation](../simulation/index.md) for the spatial scheme, time policy and Poisson.
- [A->Z tutorial](../getting-started/tutorial.md) for an end-to-end diocotron run.
