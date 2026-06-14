# Elliptic right-hand side and the aux channel

This page explains how a hyperbolic block couples to the elliptic (Poisson)
solve in `adc`. The coupling is a contract with two halves: a model declares
what it pushes into the elliptic right-hand side, and it reads back what the
solve produces through a shared channel called `aux`.

## Why the coupling exists

The systems `adc` targets are not pure conservation laws. The transport of a
density depends on a field (a potential `phi` and its gradient) that the density
itself sources. A drift-advected plasma and a self-gravitating fluid are both
this shape: an evolving state `U` feeds an elliptic equation, the elliptic
equation produces a field, and the field drives the next transport step.

The architecture keeps these two responsibilities apart. A model (`PhysicalModel`)
describes only pointwise laws; it never solves the elliptic equation and never
owns storage. The elliptic solve and its field belong to the system. The aux
channel is the seam between them, so the model stays a set of pure functions.

## What the model declares: `elliptic_rhs(U)`

A model exposes a function `elliptic_rhs(U)` that returns its pointwise
contribution `f(U)` to the right-hand side of the elliptic equation. This is one
of the four functions of the `PhysicalModel` concept, alongside `flux`,
`source`, and `max_wave_speed`.

The contribution is a density: charge density `q n`, a neutralizing background
`alpha (n - n0)`, or a gravity coupling `sign * 4piG (rho - rho0)`. The model
names a physical role, not a solver step. It does not know which solver runs or
how the field is computed.

When several blocks (species) share a domain, the system Poisson sums their
contributions:

```{math}
f = \sum_b f_b(U_b)
```

Each `f_b` is the `elliptic_rhs` of one block. The solve is shared; the source
of charge is composed across species.

## What the solve produces: the aux channel

After assembling the summed right-hand side, the system solves the elliptic
equation for `phi`, then derives the field and stores everything in `aux` (the
`adc::Aux` channel). The channel carries:

- `phi`: the potential.
- `grad_x`, `grad_y`: the components of its gradient.
- `B_z`, `T_e` (optional): extended fields, present only when declared.

The aux channel is the single shared input that the rest of the step reads. A
block's transport reads `aux` to build its flux (a drift velocity is a function
of `grad phi`), and a source term reads `aux` too (a potential force
`-rho grad phi` reads `grad_x`/`grad_y`). This is the unification point: the same
spatial operator covers drift transport, where the field enters the flux, and a
self-gravitating fluid, where the field enters the source.

## The contract in one step

Within a macro-step, the order is fixed: the system solves the fields once at the
head, then advances each block. Reading the aux that the solve produced is what
closes the loop:

1. The system gathers `f = sum_b f_b(U_b)` from every block's `elliptic_rhs`.
2. It solves the elliptic equation for `phi` and fills `aux` with `phi`,
   `grad_x`, `grad_y` (and `B_z`, `T_e` if declared).
3. Each block advances its transport and source, reading `aux`.

On the adaptive hierarchy the shape is the same, with the elliptic solved at the
coarse level by geometric multigrid and the coarse aux injected toward the fine
levels before transport. The grammar (solve, populate aux, transport, source) is
deliberately identical between the single-level and AMR pipelines.

## Writing the contract two ways

You declare the same contract whichever way you author a model. With native
bricks you pick an elliptic brick in `adc.Model(..., elliptic=...)`, for example
`adc.ChargeDensity` or `adc.BackgroundDensity`. With the DSL you write
`m.elliptic_rhs(expr)` and reference the aux fields you need with `m.aux("phi")`,
`m.aux("grad_x")`, `m.aux("grad_y")`. Both produce the same C++ object and plug
into a `System` or `AmrSystem` the same way.

The `m.check()` validator enforces one half of the contract on the read side:
every aux field a formula references must be declared, or it raises a `ValueError`.

## Where to go next

To see the elliptic brick choices and the DSL declarators in context, read
[the models overview](../models/index.md). For how the field solve and the aux
channel sit inside a full time step, read
[ARCHITECTURE](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md);
for the discretization of the multigrid Poisson and the reconstruction, read
[ALGORITHMS](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md).
