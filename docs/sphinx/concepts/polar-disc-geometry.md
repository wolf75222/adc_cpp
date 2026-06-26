# Polar and disc geometry

The diocotron problem lives on a disc, but `adc_cpp` discretizes it on a Cartesian
box by default. This page explains the two ways to represent a circular domain, why
the Cartesian box is the production path, and what the embedded wall costs you near
the ring edge.

## Why a disc is awkward on a square grid

The physics you study (the diocotron instability, an `ExB`-advected charge ring) is
naturally posed inside a circle. A finite-volume code, on the other hand, wants a
logically rectangular grid: uniform cells, simple neighbours, halos that exchange by
face. These two preferences pull in opposite directions, and the geometry you pick
decides which one you serve.

`adc_cpp` gives you two answers. Neither is strictly better; they trade fidelity at
the boundary against the production features (MPI, AMR, fluid models) you can keep.

## The Cartesian box with an embedded wall

The default path is a square `pops.System`. The disc is not the mesh; it is a circular
Poisson wall embedded inside the square. The transport runs on the full Cartesian
grid, and the elliptic solve imposes the potential boundary condition on a circle cut
through that grid.

This is the path that carries everything. It composes with the geometric multigrid
Poisson solver under MPI, with adaptive mesh refinement (AMR), and with the full fluid
and Schur-condensed source stages. It is the geometry that the diocotron tutorial and
the reproduction runs use, because it is the only one where every production feature is
available at once.

The cost is at the boundary. The circular wall does not align with the square cells, so
the ring edge is resolved by a staircase rather than a smooth curve. The Poisson wall
diffuses that edge, and the diffusion grows as the ring sits closer to the wall.

## The native polar grid

The alternative is a native polar mesh, `pops.PolarMesh`, a global ring in `(r, theta)`.
Here the circle *is* the grid: the radial and angular directions are the coordinate
axes, so the boundary is exact and the ring edge is resolved cleanly. The transport, the
polar Poisson solve, and the auxiliary fields all work in the local `e_r` / `e_theta`
basis.

Exact geometry comes at the price of reach. The polar path has sharp edges:

- transport is scalar `ExB` or the isothermal fluid (`IsothermalFluxPolar`): `rusanov` on any
  polar model and `hll` on the isothermal fluid (it declares `wave_speeds`), with a limiter
  (minmod / vanleer / weno5). Only `hllc` / `roe` stay unavailable on the polar side (no polar
  energy flux brick).
- the direct polar Poisson field solve (`PolarPoissonSolver`) is single-rank/single-box:
  it covers the ring with a single box and refuses MPI (`n_ranks > 1` is an error). The
  polar condensed Schur source stage, by contrast, is multi-rank/multi-box (theta
  decomposition).
- no Cartesian-to-polar coupling; the polar ring is a separate global domain, not a
  region embedded in a Cartesian hierarchy.

## Choosing between them

Reach for the polar mesh when boundary fidelity dominates and the model is a scalar
`ExB` ring small enough to fit one rank. Stay on the Cartesian box when you need MPI,
AMR, a fluid model, or the Schur coupling, and accept the ring-edge diffusion as the
price of those features.

The ring-edge diffusion is not a bug; it is the known cost of the embedded wall. It is
also the identified lock in the Hoffart full-model reproduction: the Cartesian square
plus circular Poisson wall diffuses the ring edge, which crushes the diocotron growth
rate in the full magnetized model. The lead under study is a conservative disc domain
built with cut-cells rather than a plain square, which would keep the production features
while sharpening the boundary.

For the precise wording of these edges and their status, see the
[known limitations](../reference/known-limitations.md). For how a model is composed and plugged
into a `System`, see [Models](../models/index.md).
