# Adaptive mesh refinement

Adaptive mesh refinement (AMR) puts fine cells only where the solution is sharp and
leaves the rest of the domain coarse. This page explains what that buys you, how the
nested hierarchy is organized, and why coupling levels conservatively is the hard part.
For the API and the mechanics, see the [AMR section](../amr/index.md).

## Why refine adaptively

A uniform grid pays for its finest cell everywhere, even in regions where the solution
is smooth and a coarse cell would resolve it well enough. In a diocotron simulation the
interesting structure is the thin edge of the electron ring; the bulk inside and the
vacuum outside carry little information. Refining everywhere to capture that edge wastes
most of the cells.

AMR spends resolution where it is needed. You keep a coarse background grid and overlay
finer cells on the features that demand them, so the cost tracks the area of the sharp
structures rather than the area of the whole domain. The trade-off is bookkeeping: you
now manage several levels, transfer data between them, and keep the whole thing
conservative.

## A block-structured nested hierarchy

`AmrSystem` is the refined counterpart of `System`. Instead of one grid it carries a
hierarchy of levels in the block-structured style of AMReX, FLASH, and SAMRAI: each level
is a set of rectangular boxes, and a finer level is nested inside its parent.

Refinement does not move the physical mesh. With refinement ratio 2 a coarse cell becomes
a 2x2 block of fine cells over the same physical extent, so the fine level has step
`dx/2`. The current hierarchy has two levels (coarse plus one fine level).

The boxes are not fixed. Tag criteria, evaluated on the parent level, mark the cells that
need resolution. Berger-Rigoutsos clustering then covers the tagged cells with a small
number of rectangular boxes, cutting recursively so the fine patches hug the feature
without too much wasted area. As the solution moves, a periodic regrid rebuilds the fine
level to follow it.

When several blocks (species) are present, they share one hierarchy: the same boxes, the
same MPI distribution, the same steps per level. The grid is regridded from the union of
every block's tags, which keeps a single layout, a single auxiliary channel, and a single
coarse Poisson solve consistent across all blocks.

## Subcycling in time

Levels also advance at their own pace. In the Berger-Oliger scheme the coarse level takes
one step `dt`, and the fine level takes `r` substeps of `dt/r`, each respecting its own
CFL (Courant-Friedrichs-Lewy) limit. The fine level's coarse-fine ghost cells are filled
by space-time interpolation from the coarse data.

Moving data between levels uses two classical conservative transfers. Restriction
(average-down, fine to coarse) overwrites a coarse cell with the average of the fine cells
it covers, so the coarse stays consistent under a patch. Prolongation (coarse to fine)
interpolates a new fine patch from its parent while carrying over existing fine data, in a
way that preserves the parent average.

## Conservative coupling: reflux

Subcycling creates a conservation problem at the coarse-fine boundary. The coarse side and
the fine side compute different fluxes across the same physical interface, so a finite-
volume scheme that ran both as-is would leak mass at the seam.

Reflux fixes this. It corrects the bordering coarse cell by the difference between the
fine flux, integrated over the `r` substeps, and the coarse flux it already took in:

```{math}
U_c \mathrel{-}= \frac{1}{\Delta x_c}\Big(\sum_s \Delta t_f\,\bar F_f^{(s)} - \Delta t_c\,F_c\Big)
```

The correction is coverage-aware: a coverage mask built on the global box layout avoids
double-correcting a fine-fine joint and routes the fix to the right parent box. With
multiple blocks each block keeps its own flux register, so conservation is checked block
by block. This is what holds the total mass to round-off across a refined run; the design
is detailed in [ALGORITHMS](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md)
and [ARCHITECTURE](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md).

## What AMR here does not cover

The hierarchy is deliberately two levels with ratio 2; multi-level regrid does not exist
yet. The Poisson solve is "coarse plus inject", not a multi-level composite elliptic
solve, which is enough for an observable that lives on cells the coarse already resolves.
The Schur-condensed source stage has no AMR counterpart, so for that path you use a non-
refined `System`. See [limitations](../reference/known-limitations.md) for the full list.
