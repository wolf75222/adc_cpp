# Configure outputs and diagnostics

This guide shows how to read the fields and diagnostics a run produces from an
`pops.System` (or `pops.AmrSystem`), and how to write them to disk. The facade returns its
state as numpy arrays that you read and save from Python, and it also provides `sim.write(...)`
and `sim.checkpoint(...)` to write files directly. This assumes you already
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

## Write fields to disk

Write the current state to a visualization file with `sim.write(path, format="vtk", step=None,
fields=None, parallel=False)`. The `vtk` format writes a cartesian ImageData `.vti` (one CellData
array per conservative variable of each block plus the potential `phi`, openable in ParaView /
VisIt); `npz` writes a compressed `np.savez` archive (any backend, any geometry); `hdf5` writes one
group per block (HDF5 output via the Python h5py facade, an optional dependency: if `h5py` is absent
the call raises a `RuntimeError` pointing to `npz`). The `step` argument adds a numbered suffix, and
`fields` selects a subset of blocks (`None` writes all).

```python
sim.write("out/state", format="vtk", step=42)
```

To open a `.vti` (or an `.h5`) in ParaView, see
[Visualize with ParaView](visualize-with-paraview.md).

`parallel=True` is valid only with `format="hdf5"` (any other format raises a `ValueError`): each rank
writes its own boxes as hyperslabs into one file, which needs `h5py` built with MPI plus `mpi4py`
(a clear `RuntimeError` names the missing piece otherwise). A cartesian `System` is mono-box, so rank 0
writes it; true parallelism only appears for multi-box runs.

For which `format` / `parallel` combinations a backend supports, read
`pops.capabilities()["io"]` rather than a static table here.

Write a restartable checkpoint with `sim.checkpoint(path, parallel=False)`. It saves the full
conservative state of every block plus the clock; reload it with `sim.restart(path)` after you
have replayed the same composition (`add_block` / `set_poisson` / couplings).

```python
sim.checkpoint("out/run.ckpt")
# later, after replaying the composition:
sim.restart("out/run.ckpt.npz")
```

`pops.AmrSystem` exposes the same surface with slightly different signatures: `write(path,
format="npz", step=None)` (npz default; writes coarse per-block fields plus fine-patch
footprints) and `checkpoint(path)` (single-block, single-rank restartable npz).

## Next steps

- Set the initial fields a run starts from with [the simulation guide](../simulation/index.md).
- Work an end-to-end run in [the tutorial](../getting-started/tutorial.md).
