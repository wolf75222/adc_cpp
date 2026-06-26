# Multi-block and multi-species systems

A system in `pops` is a set of blocks, one per model or species, that all live on a
single mesh and couple through one shared elliptic solve. This page explains what a
block is, why several blocks share the same hierarchy and auxiliary channel, and how
the coupling between species reduces to a summed right-hand side.

## One block per model or species

A block is a named carrier for one model on the mesh. You add it with `add_block`
(native bricks) or `add_equation` (a compiled DSL, domain-specific language, model),
once per species. The same call exists on `pops.System` (single level) and on
`pops.AmrSystem` (refined); the multi-block facade is identical on both.

Each block keeps its own numerics. A block fixes its spatial scheme (limiter, flux,
reconstruction), its time treatment (`explicit` or `imex`), and its multirate
(`substeps` / `stride`). Two species can therefore advance with different schemes
inside one run. For what a model is and how you write one, see
[the physical model](./physical-model.md).

The block name is the index for state access: `set_density(name)`, `mass(name)`,
and `density(name)` all take the name, so blocks never collide.

## A shared mesh and auxiliary channel

All blocks share one mesh: the same boxes, the same MPI distribution, and the same
space steps. There is never one mesh per species. On AMR this means a single
hierarchy carrying several fields, not a hierarchy per block. A construction guard
verifies that every block has exactly the same layout, because that identical layout
is the precondition for the single auxiliary channel and the single Poisson solve.

The blocks also share one auxiliary channel: the `pops::Aux` fields, the potential
`phi` and its gradient (`grad_x`, `grad_y`), plus optional extended fields such as
`B_z` or `T_e`. Each model reads `aux` where it needs the external state. A drift
transport reads `aux` in its flux; a self-gravitating fluid reads it in its source.
One channel feeds them all.

Sharing the layout is what lets coupled sources read cell by cell. Because every
block sits on the same cells, an inter-species term at `(i, j)` reads its partner at
the same `(i, j)`, with no interpolation between species.

## The shared system Poisson

The blocks share one elliptic solve, the system Poisson. Every model exposes an
elliptic contribution through `elliptic_rhs` (a charge density `q n`, a neutralizing
background `alpha (n - n0)`, a gravity coupling, and so on). The solver does not run
one Poisson per block. It builds a single right-hand side by summing the elliptic
contributions of all blocks, read at the same cells:

```{math}
f = \sum_b q_b\, n_b
```

It then solves once for `phi`, and writes `phi` and its gradient back into the shared
auxiliary channel. Every block reads the same self-consistent potential on the next
substep. This is what makes the species couple: they do not see each other directly;
they see the one field that their summed charge produces.

The mental model is two-layer. The hyperbolic update of each block is independent and
keeps its own scheme; the elliptic field is global and common. The coupling lives
entirely in the shared right-hand side and the shared potential, not in the transport.

## Coupled sources between species

Beyond the field coupling, you can wire an explicit inter-species source with
`add_coupled_source` for terms such as ionization or collisions. These read cell by
cell on the shared layout, so no interpolation is needed between species. When you
build the two contributions as exactly opposite terms, the pair mass is conserved to
machine precision.

A coupling resolves the component it targets by role, not by a literal index. It can
use a user-defined role label, resolved via `index_of(name)`, in addition to the
canonical `VariableRole` enum, so a block can expose a component whose role sits
outside the enum. Resolution is strict and fail-loud: if a block declares roles but
not the one a named coupling (collision, thermal exchange, ionization) requires, the
lookup raises rather than silently falling back to a default component. Declaring the
roles in the model is what makes this work; see
[the symbolic DSL reference](../reference/symbolic-dsl.md).

## Conservation across blocks

Conservation is checked per block, not only globally. On AMR each block owns its own
flux registers, so reflux and average-down keep every species conservative on its
own, and the mass of each block survives a regrid. The shared hierarchy is regridded
once from the union of all blocks' refinement tags, then transfers run per block on
the new common layout. For how the refinement and conservation operators work, see
[adaptive mesh refinement](./amr.md) and the
[ALGORITHMS guide](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md).

## What to keep in mind

The shared Poisson is the reason multi-block stays cheap: one elliptic solve serves
every species, and the summed right-hand side is the whole coupling. The price is the
identical-layout requirement, which the construction guard enforces up front. Distinct
hierarchies per species, with conservative projections between them, are a design
frontier and not available today; see the
[ARCHITECTURE guide](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md)
and [current limitations](../reference/known-limitations.md).
