# Build and simulate a moment model (HyQMOM, 15 moments)

Build the 2D fifteen-moment kinetic model the way the `hyqmom15` case does: declare the moment
state, write one closure, let `pops.moments` generate the fluxes and the wave speeds, compile to a
`.so`, and run the same model under three numerical methods. The point of this tutorial is the
workflow, not the golden-file validation: you will see how little physics you write (a single
closure) and how the generator derives the rest.

The snippets are faithful to the tested case at `adc_cases/hyqmom15` (`model.py`,
`runs/run_crossing.py`, `runs/run_diocotron.py`). That case is the canonical, CI-tested source; this
page explains how it is put together so you can write your own moment model the same way.

## Prerequisites

- A built `adc` Python module. If you have not built it yet, follow
  [Installation](../getting-started/installation.md); the Kokkos Serial backend is enough.
- `numpy` in the same Python environment.
- The repository headers on disk, because the model is compiled to a `.so` against them. Set
  `POPS_INCLUDE=$PWD/include`, or pass `include=` to `compile`.
- Background, if you want it: the [moments and closures concept](../concepts/moments-and-closures.md)
  explains the moment hierarchy and the closure problem; the
  [moment models reference](../reference/moment-models.md) is the API. The DSL mechanics this builds
  on (`flux`, `eigenvalues`, `compile`) are in
  [Write a model with the symbolic DSL](write-a-model-with-dsl.md).

## The model in one breath

A moment model transports the velocity moments of a kinetic distribution $f(x, v)$. For the 2D
fourth-order hierarchy there are 15 moments $M_{pq} = \int f\, v_x^p\, v_y^q\, dv$ with $p + q \le 4$,
in this order:

```text
U = [M00, M10, M20, M30, M40, M01, M11, M21, M31, M02, M12, M22, M03, M13, M04]
```

`M00` is the density, `M10` and `M01` the momentum, `M20` and `M02` the directional energies, and so
on. The Vlasov transport gives one conservation law per moment, and the flux of a moment is simply
the next moment up in that direction:

$$ \partial_t M_{pq} + \partial_x M_{p+1,q} + \partial_y M_{p,q+1} = S_{pq}. $$

The catch is at the top of the hierarchy: the flux of the order-4 moments references order-5 moments
that are not part of the state. Supplying those six order-5 moments is the *closure*, and it is the
only physics you write. Everything else -- the binomial transforms, the standardization, the flux
assembly, and the wave speeds -- is generated.

## Step 1: name the 15-moment state

The order above is the canonical order of the generic generator. You do not retype it; you ask the
generator for it, which also guarantees your indices match what the kernel expects.

```python
from pops import moments as gmom

names = gmom.moment_names(4)     # ['M00','M10','M20','M30','M40','M01', ... ,'M04'], 15 entries
pq    = gmom.moment_indices(4)   # [(0,0),(1,0),(2,0), ... ,(0,4)], the (p,q) exponents
```

`moment_names(order)` and `moment_indices(order)` are aligned: entry `k` is the moment `M{p}{q}` with
exponents `pq[k]`. Order 2 has 6 moments, order 3 has 10, order 4 has 15.

## Step 2: write the closure -- the only physics

The closure receives the *standardized* moments (the central moments divided by `sqrt(C20)^p
sqrt(C02)^q`, so `S20 = S02 = 1` by construction) and returns the six standardized order-5 moments.
This is the HyQMOM closure, a literal transcription of the reference `closureS5.m`:

```python
def hyqmom_closure(S):
    """HyQMOM order-5 closure. Input: a dict of standardized moments S11,S30,S21,S12,S03,
    S40,S31,S22,S13,S04 (DSL Expr or plain numbers). Output: the six closed order-5
    standardized moments S50,S05,S41,S14,S32,S23."""
    s11, s30, s21, s12, s03 = S["S11"], S["S30"], S["S21"], S["S12"], S["S03"]
    s40, s31, s22, s13, s04 = S["S40"], S["S31"], S["S22"], S["S13"], S["S04"]
    return {
        "S50": s30 * (5.0 * s40 - 3.0 * s30 * s30 - 1.0) / 2.0,
        "S05": s03 * (5.0 * s04 - 3.0 * s03 * s03 - 1.0) / 2.0,
        "S41": (-s30 * (8.0 * s40 - 9.0 * s30 * s30 - 4.0) * s11 / 4.0
                + (10.0 * s40 - 15.0 * s30 * s30 - 6.0) * s21 / 4.0 + 2.0 * s30 * s31),
        "S14": (-s03 * (8.0 * s04 - 9.0 * s03 * s03 - 4.0) * s11 / 4.0
                + (10.0 * s04 - 15.0 * s03 * s03 - 6.0) * s12 / 4.0 + 2.0 * s03 * s13),
        "S32": (2.0 * s40 - 3.0 * s30 * s30) * s12 / 2.0 + (3.0 * s22 - 1.0) * s30 / 2.0,
        "S23": (2.0 * s04 - 3.0 * s03 * s03) * s21 / 2.0 + (3.0 * s22 - 1.0) * s03 / 2.0,
    }
```

The contract is fixed by the generator: the closure is a callable `S -> dict`, and it must return
exactly the keys `S{p}{q}` with `p + q = order + 1` (here the six order-5 names). The body uses only
arithmetic, so the same function works on DSL expressions (when the model is built) and on plain
numpy arrays (when you check it). A different closure of the same contract -- for example the built-in
`gmom.gaussian_closure(4)` -- drops in without touching anything else.

## Step 3: generate the model -- fluxes and wave speeds for free

One call turns the closure into a full symbolic model:

```python
m = gmom.build_moment_model("hyqmom15", 4, hyqmom_closure)
```

`build_moment_model(name, order, closure, ...)` returns an `pops.dsl.Model`. From the closure alone it
generates, as symbolic formulas:

- the mean velocities `u = M10/M00`, `v = M01/M00`;
- the central moments `C_pq` (binomial transform) and the scales `sx = sqrt(C20)`, `sy = sqrt(C02)`;
- the standardized moments `S_pq` fed to your closure, then de-standardization back to raw order-5
  moments `M5`;
- the **fluxes** `Fx` and `Fy` by the order shift `Fx[M_pq] = M_{p+1,q}`, `Fy[M_pq] = M_{p,q+1}`
  (20 of the 30 entries are direct copies of the state; only the six order-5 reconstructions per
  direction carry your closure);
- the **signed wave speeds**, by automatic differentiation of the generated flux plus a per-cell
  eigenvalue solve (this is on by default, `exact_speeds=True`, and is what makes the sharper HLL
  resolver usable).

You do not call `m.flux(...)` or `m.eigenvalues(...)` yourself with the generator -- they are derived.
For a near-degenerate state (a vanishing `C20` or `M00`) pass `robust=True`, which inserts smooth
floors only where they protect the divisions and square roots.

## Step 4: compile to a `.so`

```python
import pops
compiled = m.compile("hyqmom15.so", pops.pops_include(), backend="aot")
```

`compile` runs the code generator and a C++ compiler once and returns a `CompiledModel`. `backend="aot"`
is the portable host path, fine for a first run; `backend="production"` is the native zero-copy path
(preferred under MPI and AMR) and needs the headers and compiler to match the build of `_pops`. The
default (`backend="auto"`) picks one for you. `pops.pops_include()` locates the headers; you can pass an
explicit path instead.

## Step 5: simulate -- one model, several methods

The same compiled model now runs under different numerical methods. You choose them when you attach
the model to an `pops.System`, not when you build it.

### 5a: Rusanov plus explicit time stepping (the safe start)

Rusanov reads only the maximum wave speed, so it works for any model and is the right place to start.
This is the jet-crossing run:

```python
import numpy as np
sim = pops.System(n=64, L=1.0, periodic=True)
sim.add_equation("mom", model=compiled,
                 spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                 time=pops.Explicit())
sim.set_state("mom", crossing_state(64, ma=2.0))   # initial 15-moment field, shape (15, 64, 64)
for _ in range(10):
    sim.step_cfl(0.4)
U = np.array(sim.get_state("mom"))                 # (15, 64, 64), finite, mass conserved
```

`pops.FiniteVolume(limiter="none", riemann="rusanov")` selects the finite-volume scheme, no slope
limiter, and the Rusanov numerical flux; `pops.Explicit()` is the default explicit integrator
(SSP-RK2). `set_state` loads the full moment field, `step_cfl(0.4)` advances one CFL-limited step.
`crossing_state` builds the two-jet initial condition (it lives in the case's `model.py`).

### 5b: exact HLL (sharper, signed wave speeds)

HLL keeps two signal speeds instead of one diffusion bump, so it smears the solution less than
Rusanov. It needs *signed* wave speeds, which the generator produces when you build with the exact
path:

```python
m_exact  = gmom.build_moment_model("hyqmom15_exact", 4, hyqmom_closure, exact_speeds=True)
compiled = m_exact.compile("hyqmom15_exact.so", pops.pops_include(), backend="aot")

sim = pops.System(n=64, L=1.0, periodic=True)
sim.add_equation("mom", model=compiled,
                 spatial=pops.FiniteVolume(limiter="none", riemann="hll"),
                 time=pops.Explicit())
```

`exact_speeds=True` is already the generator default; it is shown here to make the intent explicit.
Asking for `riemann="hll"` on a model that has no signed wave speeds raises an error, which is your
cue to rebuild with the exact path.

### 5b-roe: generic Roe (matches the reference ROE scheme)

For a smooth eigenmode the Roe flux is structurally less diffusive than HLL. Build with `roe=True`
and the generator also emits the generic moment Roe (`m.roe_from_jacobian()`): the full flux Jacobian
at the arithmetic-mean interface is eigendecomposed, `|A|(UR - UL)` is the dissipation, and `|A|` is
the matrix sign with a spectral-radius Rusanov fallback. This needs no fluid roles and no primitive
pressure, so it works for a moment hierarchy. It is additive to `exact_speeds` (which still provides
the maximum wave speed for the CFL step), and it needs the `aot` or `production` backend.

```python
m_roe   = gmom.build_moment_model("hyqmom15_roe", 4, hyqmom_closure, roe=True)
compiled = m_roe.compile("hyqmom15_roe.so", pops.pops_include(), backend="aot")

sim = pops.System(n=64, L=1.0, periodic=True)
sim.add_equation("mom", model=compiled,
                 spatial=pops.FiniteVolume(limiter="none", riemann="roe"),
                 time=pops.Explicit())
```

`riemann="roe"` matches the reference Matlab `space_scheme='ROE'`. The
`adc_cases/hyqmom15/runs/run_fluid_wave.py` case uses this path and pins a one-step ROE golden against
the reference `flux_ROE` to about `1e-17`; on a smooth eigenmode it measures `L2_roe < L2_hll`.

### 5c: Vlasov-Poisson coupling (sources plus multigrid)

To couple the moments to a self-consistent electric field, build the model with the Lorentz sources
and a Poisson right-hand side, then turn on the system Poisson solver. The sources read the canonical
`grad_x`/`grad_y` aux channels that the solver fills in:

```python
from pops import dsl

def lorentz(m_, M_):                      # E = -grad phi
    gx, gy = m_.aux("grad_x"), m_.aux("grad_y")
    qm = m_.param("q_over_m", 1.0)
    oc = m_.param("omega_c", 0.0)
    return gmom.lorentz_sources(M_, -1.0 * gx, -1.0 * gy, qm, oc)

m_vp = gmom.build_moment_model("hyqmom15_vp", 4, hyqmom_closure,
                               exact_speeds=True, sources=lorentz)
inv_l2 = m_vp.param("inv_debye2", 1.0 / lam ** 2)   # lam = the dimensionless Debye length
rho_bg = m_vp.param("rho_background", rho_mean)      # neutralizing background = mean density
M00 = dsl.Var("M00", "cons")
m_vp.elliptic_rhs(inv_l2 * (M00 - rho_bg))           # Delta(phi) = (M00 - rho_bg) / lam^2
m_vp.check()
compiled = m_vp.compile("hyqmom15_vp.so", pops.pops_include(), backend="aot")

sim = pops.System(n=64, L=1.0, periodic=True)
sim.add_equation("mom", model=compiled,
                 spatial=pops.FiniteVolume(limiter="none", riemann="hll"),
                 time=pops.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
```

`gmom.lorentz_sources(M, ex, ey, q_over_m, omega_c)` returns the Vlasov-Lorentz source for every
moment; it is closure-independent. `elliptic_rhs` declares how the block feeds the Poisson equation,
and `sim.set_poisson(...)` selects the charge-density right-hand side and the geometric-multigrid
solver. From here the run loop is the same `step_cfl` as in 5a, with `sim.solve_fields()` before the
first step if you want the initial potential.

### What to take away: the method matrix

The same model, defined once, runs under all of these by changing only the `build_moment_model` flags
and the `spatial=` argument:

| Goal                  | Build flag            | `spatial=`                                            |
| --------------------- | --------------------- | ----------------------------------------------------- |
| Robust start          | (defaults)            | `FiniteVolume(limiter="none", riemann="rusanov")`     |
| Sharper resolution    | `exact_speeds=True`   | `FiniteVolume(limiter="none", riemann="hll")`         |
| Least diffusive (reference ROE) | `roe=True`  | `FiniteVolume(limiter="none", riemann="roe")`         |
| Vlasov-Poisson        | `sources=...` + `elliptic_rhs` | `FiniteVolume(..., riemann="hll")` + `set_poisson(...)` |
| Near-degenerate state | `robust=True`         | unchanged                                             |
| Realizability (long/high-Ma) | `m.projection([...])` hook | unchanged; pointwise projector applied post-step |

### Realizability: the pointwise projection hook

On long or high-Mach runs the moment state can drift out of the realizable set; the reference applies
a per-cell `relaxation15` projection each step. Realizability is NOT a `build_moment_model` flag --
the generator has no `projection` parameter. It is a separate pointwise hook on the model:
`m.projection([...])` (ADC-177, the `HasPointwiseProjection` trait), one expression per conservative
component. The system then applies `U <- project(U, aux)` to the valid cells at the end of each whole
macro-step in C++, instead of a per-cell Python callback. It runs on the flat `pops.System`, under
MPI, and on `pops.AmrSystem` (per level after the reflux, ADC-312). The projection must be idempotent
and pointwise (no neighbor reads), and the clamps are written branchlessly with `dsl.abs_` / `sign`;
`dsl.eig_all_real` builds the realizable-cone masks (ADC-362). Without the hook the model is
unchanged.

The `adc_cases/hyqmom15/model.py` case wrapper bundles this: its own `build_moment_model(...,
projection=True)` -- a CASE-level wrapper, not the generic generator -- transcribes the reference
`relaxation15` projector into the `m.projection([...])` hook for you.

```python
m.projection([...])  # one expr per conservative component; relaxation15 in adc_cases
```

Why it matters: on a Ma=20 crossing run without the projection, non-realizable cells blow up the
flux eigenvalues and the CFL step collapses (the measured `dt` decays as `exp(-18.5 t)`); with the
projection applied at every step `dt` stays stable (around `1.2e-3`) over the full run. The
`adc_cases/hyqmom15` case measures this contrast.

## Troubleshooting

- The closure raises about missing keys: it must return exactly the order-5 names
  `S50, S05, S41, S14, S32, S23` (the keys with `p + q = order + 1`), no more and no fewer.
- `riemann="hll"` raises about wave speeds: the model has no signed speeds. Rebuild with
  `exact_speeds=True` (or use `riemann="rusanov"`, which only needs the maximum speed).
- The flux returns non-finite values on a degenerate state (a zero `C20`, `C02`, or `M00`): rebuild
  with `robust=True`. The default path is bare on purpose, to stay faithful to a reference that has no
  guards.
- `RuntimeError` about headers when compiling: set `POPS_INCLUDE` to the repository `include`
  directory, or pass `include=` to `compile`.

## Next

- The [moments and closures concept](../concepts/moments-and-closures.md) for why the standardization
  and the closure problem look the way they do.
- The [moment models reference](../reference/moment-models.md) for the full `pops.moments` API: the
  closure contract, `gaussian_closure`, `lorentz_sources`, and the `robust` / `exact_speeds` flags.
- The tested case at `adc_cases/hyqmom15` for the complete model, the realizable state generators,
  and the validation against the reference solution.
