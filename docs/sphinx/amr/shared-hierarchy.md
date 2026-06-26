# Shared hierarchy


All blocks live on a single AMR hierarchy: same boxes, same MPI distribution
(`DistributionMapping`), same space steps per level. This is the model "a common hierarchy
carrying several fields", never one hierarchy per species. The current version
carries two levels (refinement ratio 2: the fine level has a step `dx/2`).

- **Single-block** (a single `add_block`): historical path `AmrCouplerMP`, with dynamic regrid
  and conservative reflux. Bit-identical to what it has always produced.
- **Multi-block** (two or more `add_block`): N blocks co-located on the shared
  hierarchy (engine `AmrRuntime`). A single auxiliary channel per level (`phi`, `grad phi`)
  and a single coarse Poisson whose right-hand side is the co-located sum of the elliptic
  bricks of the blocks (`f = somme_b q_b n_b`, read at the same cells). Conservation is
  ensured per block (reflux + average-down). In multi-block, the block name indexes
  `set_density(name)`, `mass(name)` and `density(name)`.

A guard (`same_layout_or_throw`) verifies at construction that all blocks share
exactly the same layout per level (boxes, order, distribution, `dx`/`dy`): this is the
precondition of the single aux and the single Poisson. Detail: [ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md)
section 8, [AMR_MULTIBLOCK_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/AMR_MULTIBLOCK_DESIGN.md) sections 1-2, and the core
`include/pops/runtime/amr_system.hpp`.

```python
import numpy as np
import pops

n, L = 96, 1.0
ne0 = np.ones((n, n))                 # densite initiale (n, n), row-major

sim = pops.AmrSystem(n=n, L=L, periodic=True)
model = pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                  source=pops.NoSource(), elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0))
sim.add_block("ne", model=model, spatial=pops.Spatial(minmod=True), time=pops.Explicit())
sim.set_refinement(0.05)              # raffine la ou la densite depasse le seuil
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_density("ne", ne0)

for _ in range(60):
    sim.step_cfl(0.4)                 # CFL sur le pas du niveau grossier

print("patchs fins :", sim.n_patches(), "| masse :", sim.mass("ne"))
rho = sim.density("ne")               # densite grossiere (n, n)
```

`pops.AmrSystem(n=, L=, periodic=)` is a shortcut: you can also pass an
`pops.AmrSystemConfig` (fields `n`, `L`, `periodic`, `regrid_every`, `distribute_coarse`,
`coarse_max_grid`) if you want to tune the regrid cadence or the distribution of the coarse.
