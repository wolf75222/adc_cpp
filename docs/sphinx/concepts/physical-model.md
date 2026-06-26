# PhysicalModel

A model in `adc` is the equation, not the run: it is the set of pointwise laws that
describe transport, sources, and the elliptic coupling, and it names no scenario. This page
explains what the `PhysicalModel` concept captures, why it is kept this narrow, and how a model
emerges as a composition of generic bricks.

## What a PhysicalModel is

`PhysicalModel` (`pops::PhysicalModel`, in `core/physical_model.hpp`) is the contract for "what to
compute". A type that satisfies it exposes a handful of pure functions of pointwise states:

- `flux(U, aux, dir)`: the physical flux in direction `dir` (0 = x, 1 = y);
- `max_wave_speed(U, aux, dir)`: the largest characteristic speed, for the CFL bound and the
  Riemann solver;
- `source(U, aux)`: the pointwise source term;
- `elliptic_rhs(U)`: the contribution to the right-hand side of the elliptic equation (charge or
  mass density, depending on the model).

That is the whole interface. A `PhysicalModel` sees a single cell at a time. It has no access to
storage, to the mesh, or to parallelism, and the methods called in kernels carry `POPS_HD` so they
remain callable on a GPU device. There is no allocation in hot loops, no `std::function`, no dynamic
polymorphism. This locality is why the same model runs unchanged on Serial, OpenMP, Kokkos Cuda,
MPI, and AMR.

A complete hyperbolic brick additionally satisfies `pops::HyperbolicPhysicalModel`: it carries its
conservative and primitive variables and the conversions `to_primitive` / `to_conservative`, because
the variable layout and the flux are physically linked. A flux is written for one specific layout.

## Why it stays this narrow

The core is model-agnostic on purpose. It defines bricks; it never names a physical case. Diocotron,
Euler-Poisson, and two-fluid are not types in the library. They are compositions assembled on the
application side.

Keeping `PhysicalModel` to local laws separates three concerns that change for different reasons:

- *what to compute*: the pointwise laws, the `PhysicalModel` axis;
- *where and how to iterate*: the mesh and the execution seams;
- *in what order*: the time integrator and the coupler.

A high layer expresses the problem; a low layer executes it. A high layer never depends on an
execution detail. Because the model knows nothing of boxes, ranks, or loop policy, you can refine the
mesh, switch the backend, or change the time scheme without touching a single flux formula.

## The three orthogonal axes

A model is built along three independent axes. You pick one choice per axis, and the choices compose
freely.

| axis | concept | example bricks |
| --- | --- | --- |
| transport | `PhysicalModel` flux | ExB drift, compressible Euler, isothermal Euler |
| numerical flux | `NumericalFlux` policy | Rusanov, HLL, HLLC, Roe |
| elliptic coupling | `EllipticSolver` concept | geometric multigrid, Poisson FFT |

The first axis is the physics: the exact flux `F(U, aux)` and its wave speeds. The second is a purely
numerical choice, the Riemann solver that turns two interface states into one flux; it lives in the
discretization layer (`numerics/numerical_flux.hpp`), not in the model. The third is the elliptic
solver that closes the Poisson coupling. None of the three knows the internals of the others.

The link between the two hyperbolic axes is the `aux` channel. `flux` and `source` both receive
`pops::Aux`, which carries the potential `phi`, its gradient `grad_x` / `grad_y`, and the optional
extended fields `B_z` and `T_e`. The same spatial operator can therefore host a drift transport, where
`aux` is read inside the flux, and a self-gravitating compressible fluid, where `aux` is read inside
the source.

## A model is a composition of bricks

You obtain a `PhysicalModel` by composing generic bricks, not by hand-writing a monolithic equation.
In C++ this is `CompositeModel<Hyperbolic, Source, Elliptic>` (`physics/composite.hpp`), a model
assembled on three slots: transport, source, and elliptic. Each slot is a small, reusable struct from
`physics/bricks.hpp`.

The reduced diocotron model is one such composition: a scalar density (`Scalar`) advected by the ExB
drift (`ExBVelocity`), with no source (`NoSource`) and a neutralizing background as the elliptic
right-hand side (`BackgroundDensity`). Swap the transport brick for `CompressibleFlux` and the
elliptic brick for `GravityCoupling`, and the same machinery describes a self-gravitating gas. The
scenario lives in the choice of bricks, never in the core.

From Python you assemble the same object three ways, all producing the same compiled C++ model: native
bricks (`pops.Model`), symbolic formulas compiled to a `.so` (`pops.dsl.Model`), or a mix of the two
within one model (`pops.CompositeModel`). The cell-by-cell computation stays compiled either way, so
MPI, AMR, and GPU are preserved.

## Where to go next

For the composition mechanics and the catalog of bricks, see the [models guide](../models/index.md).
For the numerical flux and reconstruction policies behind the second axis, and the multigrid Poisson
behind the third, see [ALGORITHMS](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md).
For the layered architecture that keeps the three axes orthogonal, see
[ARCHITECTURE](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md).
