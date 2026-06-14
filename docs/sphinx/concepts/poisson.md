# Poisson equation

The elliptic side of `adc` is a Poisson-type problem solved once per time step. Its solution, the
potential `phi`, is read back by the hyperbolic blocks through the `aux` channel, which is what closes
the coupling between transport and field. This page explains what the equation is, what its right-hand
side and boundary conditions mean, and where it sits in a step. It does not cover the solvers; for
geometric multigrid, the spectral FFT path, and the matrix-free Krylov path see the
[advanced section](../advanced/index.md).

## What the equation is

The core advances a hyperbolic part `U` and shares an elliptic part `phi`. In its most general form the
elliptic equation is a screened, variable-coefficient operator:

```{math}
\mathrm{div}(\varepsilon\,\nabla \phi) - \kappa\,\phi = f(U).
```

With `eps = 1` and `kappa = 0` this is the plain Poisson equation `Lap phi = f(U)`, the constant-coefficient
5-point Laplacian. The generalizations are opt-in and stay bit-identical to plain Poisson when their
coefficient is left unset. A `kappa >= 0` reaction term turns it into a screened (Helmholtz) operator,
for example Debye screening with `kappa = 1 / lambda_D^2`. A space-varying `eps(x)` models a dielectric
medium, and a diagonal `diag(eps_x, eps_y)` an anisotropic one. A full anisotropic tensor with cross
terms `A_xy`, `A_yx` also exists; it arises from Schur condensation of a stiff Lorentz source, not from a
physical permittivity.

The operator is shared across blocks: every block contributes to the same `phi`, and there is one
elliptic solve per step regardless of how many blocks couple to it.

## The right-hand side f(U)

The right-hand side `f(U)` is what each block contributes to the field. It is the `elliptic_rhs(U)`
function of a model: a pointwise formula, evaluated from the current state, that says how this block
sources the potential. The system Poisson sums the contributions of all blocks before solving.

Common choices express the physical role of the field:

| Contribution | Formula | Meaning |
|---|---|---|
| charge density | `f = q n` | electrostatic potential of a charge `q n` |
| neutralizing background | `f = alpha (n - n0)` | deviation from a fixed background `n0` |
| self-gravity / plasma | `f = sign * 4piG (rho - rho0)` | `sign = +1` gravity, `-1` plasma |

Because `f` is a function of the evolving state, the field is not static: it is recomputed each step from
the freshly advanced `U`. The sign convention matters. Self-gravity and electrostatics differ only by the
sign of the coupling, which is why one parameter selects between them.

## Boundary conditions

The boundary conditions decide what `phi` means at the edge of the domain, and they constrain which
solver applies.

Periodic boundaries make the operator translation-invariant. On a periodic constant-coefficient domain
the Laplacian is diagonal in Fourier, so the spectral path solves it exactly. Periodicity also imposes a
gauge: the `k = 0` mode is fixed, so `phi` has zero mean. The right-hand side must then also have zero
mean, otherwise the potential drifts with no steady solution.

Dirichlet boundaries pin `phi` to a prescribed value, for example a conductor held at a potential. A
non-aligned conductor wall (a circular electrode, a diocotron ring edge) is not a grid staircase: a
Shortley-Weller cut-cell places the Dirichlet condition at the true interface position rather than at the
nearest cell face, recovering second order where a 0/1 mask would fall to first order.

## Where it sits in a step

The elliptic equation is the coupling seam. The hyperbolic blocks read the field, never the field
equation directly: a block sees only `phi` and its gradient `grad_x`, `grad_y` through the `aux` channel.
Drift transport reads `aux` in its flux (the `ExB` velocity is built from `grad phi`); a self-gravitating
fluid reads `aux` in its source (the force is `-rho grad phi`). The same spatial operator serves both,
which is the point of routing the field through `aux`.

Within a step the field is solved once, then re-read by `aux`. That single solve has a consequence for
accuracy: a Poisson solved once per step caps the global time order at one for the coupled field, whatever
high-order Runge-Kutta runs on the hyperbolic side. A Strang-split scheme that wants order two has to
re-solve the elliptic between the source half-steps, or the second half-step reads a stale `phi`. This
trade-off, the cost of an extra solve against the order gained, is why the solve cadence is a deliberate
choice rather than a fixed rule.

## Related reading

- [Models](../models/index.md) for how a block declares its `elliptic_rhs(U)` contribution.
- The [advanced section](../advanced/index.md) for the multigrid, spectral, and Krylov solvers that
  invert this operator.
- [ALGORITHMS](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md) for the discrete
  stencils, the screened and anisotropic extensions, and the Schur-condensed full-tensor case.
