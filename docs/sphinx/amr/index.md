# AMR (adaptive refinement)

`adc.AmrSystem` is the refined counterpart of `adc.System`: one or more blocks (species)
carried on a block-structured AMR hierarchy (with rectangular boxes, AMReX /
FLASH / SAMRAI style). The mesh is refined where the solution requires it, and only there. This
page summarizes how to drive the AMR from Python; for design details see
[ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md) (section 8), [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md)
(sections 13-15) and the design notes
[AMR_MULTIBLOCK_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/AMR_MULTIBLOCK_DESIGN.md) /
[AMR_REGRID_UNION_TAGS_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/AMR_REGRID_UNION_TAGS_DESIGN.md).

The API is identical to that of `System` (same `add_block` / `add_equation` /
`set_poisson` / `set_density` / `step_cfl`): you refine an existing case by changing
`adc.System(...)` into `adc.AmrSystem(...)` and adding a refinement criterion. The
A->Z tutorial moreover compares the two paths on the same physics (cf.
[tutorials/diocotron_tutorial.py](https://github.com/wolf75222/adc_cpp/blob/master/docs/sphinx/tutorials/diocotron_tutorial.py), function
`uniform_vs_amr`).

## Shared hierarchy

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
`include/adc/runtime/amr_system.hpp`.

```python
import numpy as np
import adc

n, L = 96, 1.0
ne0 = np.ones((n, n))                 # densite initiale (n, n), row-major

sim = adc.AmrSystem(n=n, L=L, periodic=True)
model = adc.Model(state=adc.Scalar(), transport=adc.ExB(B0=1.0),
                  source=adc.NoSource(), elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0))
sim.add_block("ne", model=model, spatial=adc.Spatial(minmod=True), time=adc.Explicit())
sim.set_refinement(0.05)              # raffine la ou la densite depasse le seuil
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_density("ne", ne0)

for _ in range(60):
    sim.step_cfl(0.4)                 # CFL sur le pas du niveau grossier

print("patchs fins :", sim.n_patches(), "| masse :", sim.mass("ne"))
rho = sim.density("ne")               # densite grossiere (n, n)
```

`adc.AmrSystem(n=, L=, periodic=)` is a shortcut: you can also pass an
`adc.AmrSystemConfig` (fields `n`, `L`, `periodic`, `regrid_every`, `distribute_coarse`,
`coarse_max_grid`) if you want to tune the regrid cadence or the distribution of the coarse.

## Tagging / regrid

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

## Prolongation / restriction

The transfers between levels are the two classical conservative operators:

- **Restriction** (`average_down`, fine -> coarse): each coarse cell receives the
  average of the fine cells it covers. This is what keeps the coarse consistent under
  a fine patch (and what prevents a mass diagnostic on the coarse alone from counting a phantom
  value under the patch).
- **Prolongation / injection** (`interpolate`, coarse -> fine): a new fine patch is
  filled by interpolation from the parent where it did not yet exist, and by carrying over
  the existing fine data where the old patch covers the new one. This injection is
  conservative (the average of the reconstructed children equals the parent).

When advancing in time, the coarse takes one step `dt` while the fine level takes `r` substeps of
`dt/r` (Berger-Oliger subcycling), each respecting its own CFL. The coarse-fine ghosts of the
fine level are filled by space-time interpolation from the coarse. Core code:
`mesh/refinement.hpp` (`average_down` / `interpolate`), engine `numerics/time/amr_reflux_mf.hpp`
(`advance_amr`). Detail: [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md) section 13.

## Reflux

At the interface between a fine level and the coarse, the two levels compute different face fluxes:
conservation would be broken if no care were taken. The reflux corrects the
coarse cell by the difference between the fine flux (integrated over the `r` substeps) and the coarse
flux:

$$U_c \mathrel{-}= \frac{1}{\Delta x_c}\Big(\textstyle\sum_s \Delta t_f\,\bar F_f^{(s)} - \Delta t_c\,F_c\Big)$$

The reflux is coverage-aware: a coverage mask (`CoverageMask`) built on the global BoxArray
avoids the double correction at the joint between two neighboring fine patches (fine-fine interface, where
it must not reflux) and directs the correction to the right parent box when the coarse
is itself multi-box. In multi-block, each block has its own flux registers: conservation
is verified block by block. Under MPI, the flux register is gathered by
`all_reduce_sum` and the remote reflux goes through `parallel_copy`. Roles promoted to types:
`FluxRegister` (accumulation of face fluxes), `CoverageMask` (covered cells). Detail:
[ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md) sections 13-14, [ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md)
section 8.

## Multi-block

`AmrSystem` is a multi-block facade: you call `add_block` (native bricks) or
`add_equation` (compiled DSL model) once per species, exactly as on `System`. All the
blocks share the hierarchy, the aux and the coarse Poisson (co-located summed right-hand side),
but each keeps its own spatial scheme (limiter x flux x reconstruction), its temporal
treatment (`explicit` or `imex`), and its multirate (`substeps` / `stride`). The runtime engine is
`AmrRuntime` (type-erased registry by block name), refined counterpart of the multi-block
single-level engine of `System`.

```python
sim = adc.AmrSystem(n=96, L=1.0, periodic=True)

electrons = adc.Model(state=adc.FluidState("compressible", gamma=1.4),
                      transport=adc.CompressibleFlux(),
                      source=adc.PotentialForce(charge=-1.0),
                      elliptic=adc.ChargeDensity(charge=-1.0))
ions = adc.Model(state=adc.FluidState("isothermal", cs2=0.5),
                 transport=adc.IsothermalFlux(),
                 source=adc.PotentialForce(charge=+1.0),
                 elliptic=adc.ChargeDensity(charge=+1.0))

sim.add_block("electrons", model=electrons,
              spatial=adc.Spatial(vanleer=True, flux="hllc"), time=adc.IMEX(substeps=10))
sim.add_block("ions", model=ions,
              spatial=adc.Spatial(minmod=True), time=adc.Explicit())
sim.set_refinement(0.05)
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_density("electrons", ne0)
sim.set_density("ions", np.ones((96, 96)))

sim.advance(0.001, 8)
print("blocs :", sim.n_blocks(), "| masse e- :", sim.mass("electrons"))
```

Inter-species coupled sources (ionization, collisions) can be wired in via
`add_coupled_source`: they are read cell by cell (same `(i, j)`, no interpolation
between species, thanks to the shared hierarchy) and, built with exactly opposite
contributions, conserve the pair mass to machine precision. The multi-block is validated by a
battery of tests called "capstone": two blocks with different schemes (`test_amr_system_twoblock`),
production multi-block DSL (`test_amr_multiblock_compiled`), IMEX (`test_amr_multiblock_imex`),
coupled sources (`test_amr_multiblock_coupled_source`), substeps (`test_amr_multiblock_substeps`),
union regrid (`test_amr_multiblock_regrid_union`) and MPI parity np=1/2/4
(`test_mpi_amr_twoblock_parity`). Detail: [ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md) section 8 and
the banner "STATUT : implemente" at the top of [AMR_MULTIBLOCK_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/AMR_MULTIBLOCK_DESIGN.md).

## Current limits

What the AMR does not do yet.

- **Two levels only.** The hierarchy is coarse + one fine level (ratio 2). The regrid only
  rebuilds the finest level; beyond 2 levels, multi-level regrid does not exist
  yet, even in single-block.
- **Poisson "coarse + inject".** The Poisson is solved on the coarse then injected toward the
  fine, it is not a multi-level composite elliptic solve. This is sufficient for
  the diocotron observable (which lives on a median circle resolved by the coarse) but worth knowing.
- **No global Schur source stage on AMR.** The Schur-condensed source splitting (`adc.Split`,
  `CondensedSchur`) has no AMR counterpart: `AmrSystem.add_block` / `add_equation`
  reject it explicitly. For this stage, use a non-refined `System`.
- **Multirate via the compiled path: restricted.** On the "production" DSL path (`.so`),
  `add_equation` explicitly rejects `stride > 1` and the partial IMEX mask
  (`implicit_vars` / `implicit_roles`): the flat ABI of the loader does not carry them, and they
  would silently be taken at their default values. For a multirate or partial-IMEX-mask `.so`,
  go through native `add_block` (`adc.Model(...)`), which exposes them.
- **Elliptic solver.** On AMR, the solver is always the geometric multigrid
  (`geometric_mg`); no FFT. The right-hand side is the sum of the elliptic bricks of the blocks.
- **Validation: what is tested vs ROMEO only.** The multi-block AMR is covered by the
  CPU tests (Serial / OpenMP) and the MPI parity np=1/2/4 in this repository. The GPU validation
  (GH200) of the AMR paths is done manually on ROMEO (the path is device-clean by
  construction, named functors); see [BACKEND_COVERAGE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md) for the
  test / backend cross-reference line by line.
- **Cut-cell / polar out of AMR scope.** The cut-cell walls and the polar geometry are
  worksites of the `System` (single-level); they are not carried on the AMR hierarchy.

Design frontier (Phase 2 / Phase 3): per-block refinement criteria, multi-level composite
elliptic solve, and (much further out) distinct hierarchies per species
with conservative projections. Detail:
[AMR_MULTIBLOCK_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/AMR_MULTIBLOCK_DESIGN.md) section 7.
