# Hyperbolic-elliptic systems

This page explains what a coupled hyperbolic-elliptic system is in adc, why the code
splits the problem in two, and how the two halves exchange information each time step.
It is the mental model behind every scenario the solver runs, from diocotron to
self-gravitating Euler.

## Two parts, two natures

The problems adc solves have two pieces with opposite mathematical character.

The hyperbolic part is a conservation law. A state `U` (a density, or a fluid with
momentum and energy) is advanced in time by a flux balance with a source,

```{math}
\partial_t U + \nabla \cdot F(U) = S(U).
```

Information travels at finite speed along characteristics. You discretize it with finite
volumes, a Riemann flux, and an explicit time integrator under a CFL bound.

The elliptic part is a Poisson problem. A potential `phi` is determined instantaneously
from the current state through its right-hand side `f(U)`,

```{math}
-\nabla \cdot (\nabla \phi) = f(U).
```

It has no time derivative. The field reacts to the whole domain at once, so it is solved,
not marched. Information is global.

## Why the split

The two natures want different solvers, and forcing them into one would compromise both.
The hyperbolic update is local and cheap per cell, ideal for a cell-by-cell kernel that
maps onto serial, OpenMP, Kokkos (GPU), and MPI through a single dispatch seam. The
elliptic solve is global and iterative, served by multigrid (`GeometricMG`) or a direct
spectral FFT.

Keeping them separate lets each layer stay orthogonal. A model author writes only
pointwise physics. The numerics, the mesh, and the time integrator never need to know
which scenario they serve. The Poisson solver never sees a flux formula. This is the
separation the [architecture](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md)
is built around.

## The aux channel: how the two halves talk

The two parts meet through one shared object, `adc::Aux`. Each time step the elliptic
solver writes the field into the aux, and the hyperbolic update reads it back. The aux is
the only contract between them.

The base aux carries three fields:

| field | meaning |
| --- | --- |
| `phi` | the potential from the Poisson solve |
| `grad_x` | x-derivative of the potential |
| `grad_y` | y-derivative of the potential |

Both the flux and the source receive the aux. That single design choice covers two
otherwise distinct couplings under one spatial operator. In ExB (E cross B) drift
transport, the flux reads `grad_phi` to build the drift velocity, so the field enters
through the flux. In a self-gravitating or electrostatic fluid, the source reads
`grad_phi` to apply a force on the momentum, so the field enters through the source. Same
channel, different slot.

The aux is extensible. A model may declare extra fields beyond the base three: a magnetic
field `B_z` or an electron temperature `T_e`. These ride the same channel and are filled
by the same field stage, so adding a coupling does not change the step skeleton.

## A model declares both halves

A model in adc names the pointwise laws of both parts at once. The C++ contract,
the `adc::PhysicalModel` concept, asks for four pure functions: `flux`, `source`,
`max_wave_speed` (for the CFL and the Riemann solver), and `elliptic_rhs`. The first three
describe the hyperbolic part; `elliptic_rhs` is the `f(U)` that feeds the Poisson solve.

The elliptic right-hand side is what makes the coupling self-consistent. A charge-density
brick contributes `f = q n`; a neutralizing background contributes `f = alpha (n - n0)`;
a gravity coupling contributes `f = sign * 4piG (rho - rho0)`. With several blocks, the
system Poisson sums their contributions, `f = sum_b f_b(U_b)`, so multiple species share
one field. See [Models](../models/index.md) for how to compose these bricks.

## One step, in order

The coupling fixes the order of operations inside a step. The field is solved first, the
aux is populated, then the transport and source run while reading that aux.

1. Solve the system Poisson from `sum_b elliptic_rhs(U_b)`, giving `phi`.
2. Derive the aux: `grad_x`, `grad_y`, and any declared `B_z` or `T_e`.
3. Advance each block's transport, the flux reading the aux.
4. Apply the source stage, reading the aux.
5. Advance time.

This grammar is identical on a single grid (`System`) and on an adaptive hierarchy
(`AmrSystem`); only the transport engine differs. The Strang-split variant re-solves the
field before each half-advance, because a potential from the step head would be stale for
the second one. See [Simulating](../simulation/index.md) and [AMR](../amr/index.md) for
the two engines.

## What this is not

The split is operator-based, not a structure-preserving weak form. Mass is conserved to
round-off by the finite-volume telescoping and the coarse-fine reflux. Momentum is exact
only when the net force vanishes by symmetry; under a real electrostatic or Lorentz force
it is not conserved by construction, which is the expected behavior of this class of
scheme. The AMR path solves Poisson at the coarse level and injects the field to the
fine levels rather than running a global Schur condensation across the hierarchy; see
[Limitations](../reference/known-limitations.md).
