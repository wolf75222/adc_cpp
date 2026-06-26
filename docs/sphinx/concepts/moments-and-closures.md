# Moment models and closures

A moment model represents a kinetic distribution by a finite set of its velocity moments and
transports them as a hyperbolic system. This page explains what the moments are, why the truncated
system is not closed on its own, how a closure repairs that, and what the `pops.moments` generator
derives for you. The [HyQMOM tutorial](../tutorials/moment-model-hyqmom15.md) puts it to work; the
[moment models reference](../reference/moment-models.md) is the API.

## From a distribution to its moments

Kinetic theory describes a population by a distribution $f(x, v)$ over position and velocity.
Following the whole distribution is expensive, so a moment method keeps only a few low-order velocity
moments and evolves those instead:

$$ M_{pq}(x) = \int f(x, v)\, v_x^p\, v_y^q\, dv. $$

For the 2D hierarchy truncated at order four ($p + q \le 4$) that is 15 numbers per cell. `M00` is
the density, `M10` and `M01` the momentum, the order-2 moments the directional energies and the
correlation, and the higher moments the skewness and the kurtosis of the velocity distribution.

## The closure problem

The Vlasov transport produces one conservation law per moment, and the flux of an order-$k$ moment is
an order-$(k+1)$ moment:

$$ \partial_t M_{pq} + \partial_x M_{p+1,q} + \partial_y M_{p,q+1} = S_{pq}. $$

This is exact, but it does not close: the flux of the order-4 moments references order-5 moments that
are not in the state. A truncated moment system is always one order short of closing itself. A
*closure* repairs that by expressing the missing top moments as a function of the moments you keep.

## Standardization: closures live in scaled coordinates

A closure is cleaner to state in coordinates that remove the local density, drift, and temperature.
Two transforms get there. First the *central* moments subtract the mean velocity $(u, v) =
(M10/M00, M01/M00)$. Then the *standardized* moments divide by the per-direction scales:

$$ S_{pq} = \frac{C_{pq}}{s_x^p\, s_y^q}, \qquad s_x = \sqrt{C_{20}}, \quad s_y = \sqrt{C_{02}}. $$

By construction $S_{20} = S_{02} = 1$, so a closure is a relation among the remaining standardized
moments and is independent of the local state. The generator does both transforms, hands your closure
the standardized moments, and undoes the scaling afterward.

## The closure is the physics; everything else is mechanical

A closure is a callable that takes the standardized moments of order up to the truncation and returns
the standardized moments one order higher. Two are common:

- **Gaussian (Levermore)** closure: assume the distribution is a local Maxwellian. The standardized
  higher moments follow a fixed recurrence; odd orders vanish. It is entropy-stable and exact when the
  flow really is Gaussian. `pops.moments.gaussian_closure(order)` provides it.
- **HyQMOM** (hyperbolicity-preserving quadrature method of moments): a polynomial closure of the
  order-5 standardized moments chosen so the flux Jacobian has real eigenvalues, which keeps the
  transport hyperbolic away from the realizability boundary. The
  [HyQMOM tutorial](../tutorials/moment-model-hyqmom15.md) writes it out.

The contract is the same for both, and for any closure you write: a function `S -> dict` returning
exactly the keys `S{p}{q}` with `p + q = order + 1`. Because the body is pure arithmetic, the same
closure works on symbolic DSL expressions when the model is built and on plain numbers when you check
it against a reference.

## What the generator derives

From the closure alone, `pops.moments.build_moment_model` produces a full
[symbolic DSL model](../reference/symbolic-dsl.md): the mean velocities and central moments, the
standardization and its inverse, the reconstruction of the order-$(k+1)$ raw moments, the **flux** by
the order shift $F_x[M_{pq}] = M_{p+1,q}$, and -- by automatic differentiation of that flux plus a
per-cell eigenvalue solve -- the signed **wave speeds** that the HLL resolver needs. You add only what
is genuinely model-specific: optional Lorentz **sources** for a Vlasov-Lorentz coupling
(`lorentz_sources`), and an `elliptic_rhs` if the density couples to a self-consistent potential
(Vlasov-Poisson). See [fluxes, sources, and eigenvalues](fluxes-sources-eigenvalues.md) for what each
piece means to the core.

Beyond the Rusanov and HLL resolvers the wave speeds feed, `build_moment_model(roe=True)` also emits a
generic moment Roe dissipation so `riemann="roe"` is usable on the hierarchy. Realizability is not a
generator concern: it is a separate pointwise projection hook (`m.projection`) the system applies
after each step.

## Limits

- The generator is 2D. A different velocity-space dimension is not generated.
- A closure is a modeling choice, not a fact: the Gaussian closure is wrong for heavy-tailed or
  strongly non-equilibrium distributions, and a poor closure can lose hyperbolicity.
- Near the realizability boundary (a vanishing directional variance) the bare transforms divide by
  zero. Build with `robust=True` to insert smooth floors where they protect the divisions and square
  roots.

## See also

- [Build and simulate a moment model (HyQMOM)](../tutorials/moment-model-hyqmom15.md) -- the
  step-by-step tutorial.
- [Moment models reference](../reference/moment-models.md) -- the `pops.moments` API.
- [Conservative and primitive variables](conservative-primitive-variables.md) and
  [fluxes, sources, and eigenvalues](fluxes-sources-eigenvalues.md) -- the model contract the
  generated moment model fills in.
