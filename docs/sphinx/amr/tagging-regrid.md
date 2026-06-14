# Tagging and regrid


Refinement is driven by tag criteria evaluated on the parent level. The fine
grid is then rebuilt by Berger-Rigoutsos clustering: given the tagged cells,
the algorithm finds a small number of rectangular boxes that cover them without
too much waste (recursive cut on the signature of the marks). The cadence is carried by
`regrid_every` (re-refinement every N macro-steps; `0` = never after initialization).

Two criteria are exposed and compose (OR cell by cell, "union of tags"):

- `set_refinement(threshold)`: density per block. Refines where the density (component 0)
  of a block exceeds `threshold`. Base criterion, valid for single- and multi-block.
- `set_phi_refinement(grad_threshold)`: gradient of the potential `|grad phi|`. Refines where
  the norm of the gradient of the electrostatic potential exceeds `grad_threshold` (physical criterion
  of the diocotron: the ring edge follows the gradient of the potential, not the density alone).
  Multi-block only; disabled by default (`grad_threshold <= 0`). To be called before
  the first step.

In multi-block, the shared hierarchy is regridded from the union of the tags of all
the blocks (plus the tag of `phi` if it is active): a single collective criterion, a single clustering,
a single new shared layout, then prolongation / restriction / reflux per block on this
layout. The mass of each block is conserved across the regrid. With `regrid_every == 0` the
multi-block hierarchy stays frozen (regrid never called, bit-identical to a static
hierarchy). Under MPI, the union of the tags is reduced cross-rank (`all_reduce_or`) before the
clustering, otherwise the fine boxes would differ from one rank to another.

Detailed algorithm: [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md) section 15 (clustering + regrid) and
[AMR_REGRID_UNION_TAGS_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/AMR_REGRID_UNION_TAGS_DESIGN.md) (steps R0-R8 of
the union of tags).

```python
sim = adc.AmrSystem(n=128, L=1.0, periodic=True)
sim.add_block("electrons", model=elec, spatial=adc.Spatial(minmod=True), time=adc.Explicit())
sim.add_block("ions",      model=ions, spatial=adc.Spatial(minmod=True), time=adc.Explicit())
sim.set_refinement(0.05)         # union des tags de densite (electrons OU ions)
sim.set_phi_refinement(0.5)      # + |grad phi| (multi-blocs ; bord d'anneau)
```

> The regrid cadence (`regrid_every`) is tuned via the `AmrSystemConfig`:
> `sim = adc.AmrSystem(adc.AmrSystemConfig())` then `config.regrid_every = 20`, or by passing
> `regrid_every=20` to the constructor (config kwargs). With a density threshold at its default
> value (no tag), the grid stays unchanged.
