# Visualize with ParaView

This guide opens `adc_cpp` output in ParaView (or VisIt). The only format that opens
directly is the cartesian VTK ImageData `.vti` written by `sim.write(path, format="vtk")`;
everything else needs a re-export or a small loader. This assumes you already have a
composed and initialized system. To build one, see
[the simulation guide](../simulation/index.md). For the full write surface and the other
formats, see [configure outputs and diagnostics](configure-outputs-diagnostics.md) and
[outputs and diagnostics](../simulation/outputs-diagnostics.md).

## Write a .vti from a System run

`sim.write(path, format="vtk")` gathers the current state and writes one ASCII
`.vti` file. It holds one CellData array per conservative variable of each block, named
`<block>_<var>`, plus the potential `phi`. The grid is the unit domain in index space:
`WholeExtent` is `0..nx 0..ny` with `Spacing` `1/nx`, `1/ny`.

```python
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
for _ in range(500):
    sim.step_cfl(0.4)
sim.write("out/state", format="vtk")   # writes out/state.vti
```

## Open in ParaView and color by a field

1. Start ParaView, then `File > Open` and pick `out/state.vti`.
2. Click `Apply` in the Properties panel to load the data.
3. In the coloring dropdown of the toolbar, select a CellData array: a
   `<block>_<var>` array (for example `ne_rho`) or `phi`. Use the block names from
   `sim.block_names()` and the conservative variable names of each block.
4. Use `Rescale to Data Range` so the color map spans the field.

## Write a time series and animate

There is no native `.pvd`, `.vts`, or XDMF writer. To animate, write one numbered `.vti`
per frame with the `step` argument, which appends a six-digit suffix
(`name_000123.vti`). ParaView auto-groups `name_<NNNNNN>.vti` files in the same directory
into a single time series.

```python
import os
os.makedirs("frames", exist_ok=True)
for k in range(200):
    sim.step_cfl(0.4)
    sim.write("frames/run", format="vtk", step=k)   # frames/run_000000.vti, ...
```

In ParaView, `File > Open` and select the collapsed `run_*.vti` group (it shows as a single
entry), then `Apply`. The VCR controls in the toolbar play the series; export it with
`File > Save Animation`.

## Other formats

`format="npz"` (`np.savez_compressed`) and `format="hdf5"` (needs `h5py`) carry the same
fields but have no direct ParaView importer. The simplest path is to re-export the state you
want with `format="vtk"`. To read an `.npz` in place, add a ParaView `Programmable Source`
(output type `vtkImageData`) that loads the `state_<block>` and `phi` keys:

```python
# ParaView Programmable Source, Output DataSet Type = vtkImageData
import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk

d = np.load("out/state.npz")
nx, ny = int(d["nx"]), int(d["ny"])
out = self.GetImageDataOutput()
out.SetExtent(0, nx, 0, ny, 0, 0)
out.SetSpacing(1.0 / nx, 1.0 / ny, 1.0)

phi = d["phi"]                      # (ny, nx)
arr = numpy_to_vtk(phi.ravel(order="C"), deep=1)
arr.SetName("phi")
out.GetCellData().AddArray(arr)
```

The `.npz` keys are `t`, `macro_step`, `nx`, `ny`, `blocks`, `state_<block>` (shape
`(nv, ny, nx)`), `names_<block>`, `roles_<block>`, and `phi` (shape `(ny, nx)`).

## AMR and polar runs

`pops.AmrSystem.write(format="vtk")` writes the COARSE level only: per-block density plus
`phi`. The fine patches are not in the `.vti`; they are reported as
`patch_rectangles` (their footprints) in the `npz` output. Polar runs still emit a
cartesian `.vti` (there is no polar `.vts` writer).

## Under MPI

With more than one rank, `sim.write` gathers the fields collectively (every rank must call
it) and then rank 0 writes ONE `.vti` (the cartesian System is mono-box, so there is no
`.pvti` part file). The result is identical to the single-rank file. See
[run with MPI](run-with-mpi.md).

## Next steps

- For the full write surface and the npz/hdf5/checkpoint paths, see
  [configure outputs and diagnostics](configure-outputs-diagnostics.md).
- For the field and diagnostic accessors a run exposes, see
  [outputs and diagnostics](../simulation/outputs-diagnostics.md).
