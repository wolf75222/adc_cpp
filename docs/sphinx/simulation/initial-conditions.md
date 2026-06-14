# Initial conditions


Two ways to set the initial state of a block, both in numpy `(n, n)`. The layout
convention is row-major `(ny, nx)`: the first index (rows) is the slow `y` axis, the second
(columns) is the fast `x` axis; a field is indexed `ne[j, i]` (`j` = row / `y`, `i` =
column / `x`).

`set_density(name, arr)` sets the density (component 0) and leaves the rest at rest (for a
fluid: zero velocity, consistent energy). It is the usual shortcut for a scalar transport.

```python
coord = (np.arange(n) + 0.5) / n * L
xx, yy = np.meshgrid(coord, coord, indexing="xy")          # xx, yy of shape (n, n) = (ny, nx)
r = np.hypot(xx - 0.5 * L, yy - 0.5 * L)
ne = np.full((n, n), 1e-3)
ne[(r > 0.15) & (r < 0.20)] = 1.0
sim.set_density("ne", ne)
```

`set_primitive_state(name, **prims)` initializes a fluid block from its primitive variables,
named (`rho`, `u`, `v`, `p`...). Each primitive is an `(n, n)` array; the block model
converts them into conservative variables (compressible: `E = p/(g-1) + 1/2 rho|v|^2`). The names
expected are those of the model; an unknown or missing name raises a clear error.

```python
sim.set_primitive_state("electrons", rho=rho0, u=u0, v=v0, p=p0)
```

For an explicit conservative state (diagnostic / advanced case), `set_state(name, u)` takes the
flattened `(ncomp, n, n)` array.
