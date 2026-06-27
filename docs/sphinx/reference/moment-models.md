# Moment models

Reference for `pops.lib.moments`, the generic 2D moment-hierarchy generator. It turns a single closure
into a full [symbolic DSL model](symbolic-dsl.md): the central and standardized moments, the flux,
and the signed wave speeds are all derived, so you write only the closure (and, optionally, the
sources). For the concept see [moments and closures](../concepts/moments-and-closures.md); for a
worked example see [the HyQMOM tutorial](../tutorials/moment-model-hyqmom15.md).

`pops.lib.moments` is a specialized generator and is imported explicitly:

```python
from pops.lib import moments as gmom
```

## `build_moment_model`

```python
gmom.build_moment_model(name, order, closure, blocks=None, exact_speeds=True,
                        robust=False, eps_m00=1e-12, eps_cov=1e-12, sources=None,
                        roe=False)
```

Builds and returns an `pops.physics.facade.Model` for the 2D velocity-moment hierarchy of the given `order`. The
returned model is ready to `compile`, or you may add an `elliptic_rhs` or extra `param` first.

| Parameter      | Meaning                                                                                       |
| -------------- | --------------------------------------------------------------------------------------------- |
| `name`         | Model name (used for the generated symbol names and the `.so`).                               |
| `order`        | Truncation order of the hierarchy. `order >= 2` (standardization needs `C20`, `C02`). Order 2 is 6 moments, 3 is 10, 4 is 15. |
| `closure`      | The closure callable. Its contract is in the "Closure contract" section below.                 |
| `exact_speeds` | `True` (default): signed wave speeds by automatic differentiation of the flux plus a per-cell eigenvalue solve, which enables `riemann="hll"`. `False`: you set `m.eigenvalues(...)` or `m.wave_speeds(...)` yourself. |
| `robust`       | `True`: smooth floors `max(x, eps) = ((x + eps) + abs(x - eps)) / 2` on `M00` (division) and `C20`/`C02` (square root), so the flux stays finite near the realizability boundary. `False` (default): the bare path, which may produce non-finite values on a degenerate state. |
| `blocks`       | Optional block structure of the flux Jacobian, passed through to the wave-speed solve (a dict `{"x": [...], "y": [...]}` of index lists). Default: the full matrix. Ignored when `exact_speeds=False`. |
| `eps_m00`, `eps_cov` | Floor thresholds used when `robust=True`.                                               |
| `sources`      | Optional callable `(model, M) -> list[Expr]`, one source per moment, wired with `m.source(...)`. `M` is a dict mapping `(p, q)` to the conservative `Var`. Use `lorentz_sources` (below). |
| `roe`          | `True`: also emit the generic moment Roe dissipation (`m.roe_from_jacobian`), with `|A|` via `pops::roe_abs_apply` and a spectral-radius Rusanov fallback. Additive to `exact_speeds`, so it enables `riemann="roe"` for a moment hierarchy. Needs the `aot` or `production` backend. `False` (default): no Roe path. |

You do not call `m.flux(...)` or `m.eigenvalues(...)` on the returned model when you use the
generator: both are derived from the closure. The flux is the order shift `Fx[M_pq] = M_{p+1,q}`,
`Fy[M_pq] = M_{p,q+1}`; the order-`(order+1)` moments in the top row come from your closure.

## Closure contract

A closure is a callable `closure(S) -> dict`.

- **Input** `S`: a dict of the standardized moments for `2 <= p + q <= order`, keyed `"S{p}{q}"`. By
  construction `S["S20"] == S["S02"] == 1.0`. The values are DSL expressions when the model is built,
  or plain numbers when you evaluate the closure directly; the body must use only arithmetic so both
  work.
- **Output**: a dict that contains exactly the keys `"S{p}{q}"` with `p + q == order + 1` (the
  standardized moments one order above the truncation), and no others. A missing or extra key is an
  error. A value of `0.0` removes that term from the generated flux.

```python
def my_closure(S):
    # order 4 -> must return S50, S05, S41, S14, S32, S23
    ...
    return {"S50": ..., "S05": ..., "S41": ..., "S14": ..., "S32": ..., "S23": ...}
```

## `gaussian_closure`

```python
gmom.gaussian_closure(order)   # -> closure callable
```

The Gaussian (Levermore) closure: the higher standardized moments of a local Maxwellian. Odd orders
vanish; even orders follow the standardized recurrence. Use it as a ready closure or as a reference
to check your own against (it is exact when the flow is Gaussian).

## `lorentz_sources`

```python
gmom.lorentz_sources(M, ex, ey, q_over_m, omega_c)   # -> list[Expr], one per moment
```

The Vlasov-Lorentz source for each moment, closure-independent:

$$ S[M_{pq}] = \frac{q}{m}\,(p\, e_x\, M_{p-1,q} + q\, e_y\, M_{p,q-1})
   + \Omega_c\,(p\, M_{p-1,q+1} - q\, M_{p+1,q-1}). $$

The electric term lowers the moment order (the referenced moments always exist); the magnetic term
conserves the total order. `S[M00] = 0` (mass has no source). `M` is a dict mapping `(p, q)` to the
moment value (a `Var` inside `build_moment_model`, or a number for a numeric check); `ex`/`ey` are the
electric field components (read the canonical `grad_x`/`grad_y` aux channels for the self-consistent
field, with `E = -grad phi`); `q_over_m` and `omega_c = q B / m` are scalars. Wire it through the
`sources=` argument of `build_moment_model`.

## `maxwellian_moments` and `bgk_source`

```python
gmom.maxwellian_moments(M)        # -> list[Expr], one per moment: the equilibrium M_eq
gmom.bgk_source(M, nu)            # -> list[Expr], one per moment: nu (M_eq - M)
```

`maxwellian_moments` returns the raw moments of the local Maxwellian (the Gaussian in velocity)
that matches the density, mean velocity, and covariance of `M`. The mean is `(M10/M00, M01/M00)`
and the covariance comes from the second central moments; every higher moment follows the Isserlis
(Wick) rule, so `maxwellian_moments` is the closure-free equilibrium of the hierarchy at any order.

`bgk_source` returns the BGK relaxation source toward that equilibrium:

$$ S[M_{pq}] = \nu\,(M^{\mathrm{eq}}_{pq} - M_{pq}). $$

Because the Maxwellian shares the mass, momentum, and covariance of `M`, the collisional invariants
`M00`, `M10`, `M01` are exact equilibria: those rows are identically zero, so BGK conserves mass and
momentum by construction. `M` is a dict mapping `(p, q)` to the moment value (a `Var` inside
`build_moment_model`, or a number for a numeric check); `nu` is the collision frequency (an
expression or a scalar).

BGK relaxation is wired as a source, not a new hook: pass `bgk_source` (alone, or summed with
`lorentz_sources`) through the `sources=` argument so it rides the existing source brick
(`m.source`, with `m.source_frequency(nu)` for the explicit CFL bound, or `time="imex"` for the
stiff path). It is independent of the pointwise realizability projection (`m.projection`): the
Maxwellian is itself realizable, so a model may run BGK relaxation and a projection together
(transport, then the BGK source, then the projection).

## `moment_names` and `moment_indices`

```python
gmom.moment_names(order)     # -> ['M00','M10', ..., 'M0{order}'], canonical order
gmom.moment_indices(order)   # -> [(0,0),(1,0), ...], the (p,q) exponents, aligned with the names
```

The canonical state ordering (q outer, p inner). Entry `k` of `moment_names` is `M{p}{q}` with
exponents `moment_indices[k]`. Use these to index a state array or to build name-to-index maps; they
are the order the generated kernel expects.

## Simulating a moment model

A compiled moment model is an ordinary `CompiledModel`, attached to an `pops.System` like any DSL
model:

```python
import pops
compiled = gmom.build_moment_model("mom", 4, my_closure).compile("mom.so", pops.pops_include())
sim = pops.System(n=64, L=1.0, periodic=True)
sim.add_equation("mom", model=compiled,
                 spatial=pops.FiniteVolume(limiter="none", riemann="hll"),
                 time=pops.Explicit())
```

`riemann="hll"` requires the signed wave speeds that `exact_speeds=True` generates; `riemann="rusanov"`
needs only the maximum speed and works either way. `riemann="roe"` is available once the model is built
with `build_moment_model(roe=True)`, which emits the generic moment Roe dissipation; only `riemann="hllc"`
remains rejected for a moment model, being specific to the four-variable Euler system. For the Riemann-solver choices
see [fluxes, sources, and eigenvalues](../concepts/fluxes-sources-eigenvalues.md); for the spatial and
time classes see the [Python API](python-api.md).

## See also

- [Moments and closures](../concepts/moments-and-closures.md) -- the concept.
- [Build and simulate a moment model (HyQMOM)](../tutorials/moment-model-hyqmom15.md) -- the tutorial.
- [Symbolic DSL](symbolic-dsl.md) -- the model the generator emits.
