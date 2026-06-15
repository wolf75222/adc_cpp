# Outputs and diagnostics


The system returns its fields in numpy `(ny, nx)` (or `(ncomp, ny, nx)`), same row-major convention
as on input.

- `sim.density(name)`: density of the block, `(n, n)` array;
- `sim.mass(name)`: total mass of the block (scalar), the conservation invariant to check;
- `sim.potential()`: electrostatic potential `phi`, `(n, n)` array;
- `sim.time()`: current physical time;
- `sim.block_names()`: names of the blocks, in order of addition;
- `sim.get_state(name)`: full conservative state, `(ncomp, n, n)` (e.g. `[rho, rho*u, rho*v, E]`
  for Euler);
- `sim.get_primitive_state(name)`: state returned in primitive variables, dict
  `{name: (n, n)}` (inverse of `set_primitive_state`, round-trip to machine precision).

```python
m0 = sim.mass("ne")
for _ in range(500):
    sim.step_cfl(0.4)
rho = sim.density("ne")             # ndarray (n, n)
phi = sim.potential()
print("mass drift:", abs(sim.mass("ne") - m0))   # ~ machine roundoff

U = sim.get_state("electrons")      # (4, n, n) = [rho, rho*u, rho*v, E]
P = sim.get_primitive_state("electrons")            # {"rho": ..., "u": ..., "v": ..., "p": ...}
```

Two primitives serve to drive the solver from Python (custom time integrator, field
oracle):

- `sim.solve_fields()`: solves the system Poisson on the current state and repopulates the
  `aux` channel (`phi`, `grad phi`) without advancing in time, useful to read `potential()` at a frozen state;
- `sim.eval_rhs(name)`: evaluates the right-hand side `R = -div F + S` of the block (the spatial `dU/dt`),
  `(ncomp, n, n)`, for a user-provided time integrator.

## Write to disk

The facade also writes files directly:

- `sim.write(path, format="vtk", step=None, fields=None, parallel=False)`: visualization output.
  `format="vtk"` writes a cartesian ImageData `.vti` (one CellData array per conservative variable
  of each block plus `phi`, openable in ParaView / VisIt); `format="npz"` writes a compressed
  `np.savez` archive (any backend, any geometry). `step` adds a numbered suffix; `fields` selects a
  subset of blocks (`None` = all).
- `sim.checkpoint(path, parallel=False)`: restartable npz checkpoint (full conservative state of
  every block plus the clock). Reload it with `sim.restart(path)` after replaying the same
  composition.

`adc.AmrSystem` exposes the same surface with slightly different signatures: `write(path,
format="npz", step=None)` (npz default, coarse per-block fields plus fine-patch footprints) and
`checkpoint(path)` (single-block, single-rank restartable npz).

```python
sim.write("out/state", format="vtk", step=42)
sim.checkpoint("out/run.ckpt")
```

Condensed API reference: [api](../reference/python-api.md). Complete recipes (figures, AMR):
[examples](../getting-started/repository-layout.md). A->Z tutorial: [tutoriels](../getting-started/tutorial.md).
