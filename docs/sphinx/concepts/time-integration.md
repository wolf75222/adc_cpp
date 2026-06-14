# Time integration

This page explains how `adc_cpp` advances a solution in time: the method of lines, the explicit
SSP Runge-Kutta schemes on the hyperbolic part, the CFL condition that bounds the step, and what it
means to march a coupled hyperbolic-elliptic system forward by one macro-step.

## The method of lines

The spatial discretization (finite volumes) turns the conservation law
$\partial_t U + \nabla\cdot F(U) = S(U)$ into a system of ordinary differential equations in time.
The space side assembles a residual, one value per cell:

```{math}
\frac{\mathrm{d}U}{\mathrm{d}t} = L(U), \qquad L(U) = -\nabla\cdot \hat F(U) + S(U).
```

Separating space from time this way is the method of lines. The space operator answers a single
question, "given the current state, what is its rate of change?", and the time integrator decides
how to use that answer to step forward. The two concerns stay independent: you can pair any
reconstruction and flux with any time scheme.

## Explicit SSP Runge-Kutta

The default integrators are Strong-Stability-Preserving Runge-Kutta (SSPRK) schemes. The idea is
that every stage is a convex combination of forward Euler steps. So any bound a single Euler step
respects under CFL (positivity of density, total variation, an admissible range) is inherited by the
whole multi-stage scheme. You gain accuracy without losing the robustness of the first-order update.

Two schemes are built in. SSPRK2 (two stages, order 2, equivalent to Heun) and SSPRK3 (three stages,
order 3). SSPRK3 is less dissipative and is the natural partner for WENO5 reconstruction, where a
low-order time scheme would waste the spatial accuracy.

Both schemes have an SSP coefficient of 1. That number is the practical catch: the strong-stability
guarantee holds only while the step obeys the same forward Euler CFL bound. A higher-order time
scheme buys you accuracy, not a larger stable step.

The integrator is a first-class object with a `take_step(rhs, U, dt)` contract. It sees only the
method-of-lines arrow `rhs(U, R)` and a couple of array operations; it knows nothing of the model or
the discretization. Because the contract is small, a case can supply its own integrator the same way
it supplies a model.

## The CFL condition

An explicit scheme can only transport information one cell per step at most. The
Courant-Friedrichs-Lewy (CFL) condition makes this precise: the step must be small enough that no
wave crosses a cell in a single update.

```{math}
\Delta t \le C\,\frac{\min(\Delta x, \Delta y)}{\max|\lambda|},
```

where $\lambda$ is the local wave speed and $C \le 1$. The core computes $\max|\lambda|$ by a
reduction over all cells, followed by an MPI `all_reduce_max` so every rank agrees on one global
step. Without that collective, each rank would pick its own $\Delta t$ and the ranks would diverge.

A model with no transport (a pure source, $\max|\lambda| = 0$) places no constraint on the step. You
choose the CFL number with `step_cfl(cfl)`, which returns the step it selected.

## Advancing a coupled system

A simulation rarely holds a single equation. `adc.System` advances several blocks (one per species)
that share one Poisson solve. Within a macro-step, the elliptic field is solved once, the `aux`
channel (`phi`, `grad phi`) is refreshed, and each block then advances its hyperbolic transport
explicitly reading that frozen field.

This once-per-step coupling caps the global accuracy. However high the order of the SSPRK scheme on
the hyperbolic side, solving the elliptic field once per step limits the coupled system to first
order in time. Recovering order requires resolving the field more carefully inside the step, which is
the role of operator splitting.

Blocks need not all march at the same rate. The time policy of each block carries two orthogonal
integers handled by the scheduler, not the integrator: `substeps`, which splits a macro-step into
several smaller steps for a fast species, and `stride`, which advances a slow species only once every
few macro-steps. With both set to one, you recover a plain uniform step.

## Stiff sources need more than CFL

The CFL bound governs transport. A stiff source (fast relaxation, the Lorentz force, Debye screening)
imposes its own, far smaller, time scale and would force the explicit step down to an impractical
value.

The remedy is not to shrink $\Delta t$ but to treat the source differently from the transport. You
keep transport explicit under its CFL and integrate the stiff source implicitly, separating the two
through operator splitting. For how the two operators are composed in sequence, and why a coupled
field must be re-solved between the half-steps, see the
[Strang, Schur, and IMEX](./strang-schur-imex.md) concept page.

For the formulas, pseudocode, and the C++ entry points, see the
[ALGORITHMS](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md) guide (time
integration, multirate). The Python side of composing blocks and time policies is in the
[simulation](../simulation/index.md) section.
