# Fluxes, sources and eigenvalues

A model in `adc` is, at its core, three pointwise functions: a flux that transports
matter across cell faces, a source that adds or removes it in place, and a set of
eigenvalues that tells the time stepper how fast information travels. This page explains
what each piece means, why they are kept separate, and how the numerical flux turns the
physical flux into a face value.

## What the core solves

The hyperbolic part of every model has the same shape:

```{math}
\partial_t U + \nabla\cdot F(U, \mathrm{aux}) = S(U, \mathrm{aux})
```

Here `U` is the evolved state of a cell (a density, or a density and its momenta), `F` is
the flux, and `S` is the source. The `aux` channel carries the fields the elliptic solver
produces (the potential `phi` and its gradient, and optionally `B_z` or `T_e`). Both `F`
and `S` read `aux`, which is the single idea that lets one spatial operator serve both a
drift transport (where the coupling enters the flux) and a self-gravitating fluid (where it
enters the source).

## The flux: transport across a face

The flux answers "how much of `U` crosses a face, in which direction". For scalar advection
by the ExB drift it is the density times the drift velocity; for compressible Euler it is the
usual mass, momentum and energy flux. You write it once, as a pure function of the local state.

The flux is not evaluated naively at a face, because two cells meet there with two different
states. That meeting is a local Riemann problem: a small discontinuity whose evolution the
scheme must approximate. The physical flux `F` describes the equation; the numerical flux
decides how to resolve the discontinuity.

## The numerical flux as a policy

The numerical flux is a separable choice, not a property of the model. The same model can
run under any of four resolvers, ordered by how many waves they reconstruct:

| Policy | Resolves | Trade-off |
|---|---|---|
| Rusanov | one diffusion bump | most robust, most diffusive, needs only a wave speed |
| HLL | two signal speeds | one star region, still smears the contact |
| HLLC | the contact wave | sharper, assumes Euler 2D |
| Roe | all waves, linearized | sharpest, needs an entropy fix and a pressure |

Rusanov is the safe default: it reads only `max_wave_speed`, so it applies to any model,
including a plain scalar transport. The cost of that robustness is added numerical diffusion,
which can erode a growth rate you care about. HLLC and Roe target the four-variable Euler
state and need the model to expose a pressure; they sharpen the contact discontinuity at the
price of a narrower domain of safety. You pick the policy where you build the spatial operator,
independently of the reconstruction limiter (see [the models guide](../models/index.md)).

## The source: change in place

The source `S` adds to a cell without moving anything across a face: a potential force on the
momentum, a gravitational pull, a relaxation toward equilibrium. It reads the same `aux` as the
flux, so a force like `-rho grad phi` simply reads `grad_x` and `grad_y`.

A source can be stiff, relaxing far faster than the transport. Treating it explicitly would
force a tiny time step. That is why the source is its own function: the integrator can advance
it implicitly while transport stays explicit. The discretization of that split lives in the
[algorithms guide](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md); the
concept to retain is that flux and source are kept separate so they can be integrated differently.

## The eigenvalues: setting the time step

The eigenvalues are the characteristic speeds of the system per direction: the drift velocity
for a scalar, or `{u-c, u, u, u+c}` for Euler with sound speed `c`. They are what makes the
equation hyperbolic, and they set the largest speed at which information moves.

The core reduces them to `max_wave_speed` and from it derives the explicit time step through
the CFL (Courant-Friedrichs-Lewy) condition:

```{math}
\Delta t \le C\,\frac{\min(\Delta x, \Delta y)}{\max|\lambda|}
```

A wave must not cross more than one cell per step, or the scheme loses stability. The maximum
is taken over every local box and then reduced across MPI ranks, so all ranks pick the same
`dt` and stay in step. A model with no transport reports a zero wave speed and places no limit
on the step. The eigenvalues also feed the numerical flux: Rusanov bounds its diffusion by the
maximum speed, and HLL uses the signal speeds directly.

## How you declare the three

In a symbolic (DSL) model the three pieces map one to one onto declarators: `m.flux(x=..., y=...)`
for the physical flux, `m.eigenvalues(x=..., y=...)` for the characteristic speeds, and
`m.source(...)` for the source. A native model picks the same three from precompiled bricks
instead. Either way the result is one `PhysicalModel`: a flux, a set of eigenvalues, a source,
and an elliptic right-hand side, with nothing about the mesh or the integrator baked in.
