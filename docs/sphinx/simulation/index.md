# Simulation

`adc.System` is the composition facade of the solver: you declare one block per physical model
(one species), you share one system Poisson across all blocks, you set initial conditions
in numpy, then you advance the whole thing step by step. The core names no scenario (diocotron,
Euler-Poisson... live on the `adc_cases` side); here you assemble generic bricks.

This page walks through the simulation mechanics on the Python side: composing a system, coupling
several species, choosing the spatial scheme and the time policy, handling the multirate,
initializing and reading the fields. The theoretical detail of the numerical methods is in
[ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md); the layered architecture (dispatch seam, frontier
lib/application) in [ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md).

## System

`adc.System` builds the coupler. The configuration passed as keywords describes only the
mesh (`n`, `L`, `periodic`): the domain is a `[0, L]^2` square of `n x n` cells.

```python
import numpy as np
import adc

sim = adc.System(n=128, L=1.0, periodic=False)
```

You then add one block per model. A model is a brick composition
(`adc.Model(state, transport, source, elliptic)`); the block additionally receives a spatial scheme
(`adc.Spatial` / `adc.FiniteVolume`) and a time policy (`adc.Explicit`, `adc.IMEX`...).

```python
model = adc.Model(state=adc.Scalar(), transport=adc.ExB(B0=1.0),
                  source=adc.NoSource(), elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0))
sim.add_block("ne", model=model, spatial=adc.Spatial(minmod=True), time=adc.Explicit())
```

The system Poisson is shared by all blocks: its right-hand side is the sum of the elliptic
contributions of each block (`f = sum_s elliptic_rhs_s(u_s)`). You configure it once,
after the blocks.

```python
sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet",
                wall="circle", wall_radius=0.40)
```

`rhs` is `"charge_density"` (usual case: all blocks carry a charge density
`f = sum_s q_s n_s`) or `"composite"` (generic sum of the elliptic bricks per block);
`solver` is `"geometric_mg"` (multigrid, any case) or `"fft"` (periodic, `n = 2^k`); `bc`
handles the boundary condition (`"auto"`, `"dirichlet"`, `"periodic"`), `wall`/`wall_radius`
materialize a circular conducting wall (cut-cell). `set_poisson` is the shortcut for
`add_elliptic_model` (the Poisson is an instance of a composable elliptic model).

You set the initial condition then you advance:

```python
sim.set_density("ne", ne0)          # (n, n) array
sim.step_cfl(0.4)                   # one step at CFL 0.4 (returns the effective dt)
sim.advance(0.01, 50)               # 50 steps of fixed dt = 0.01
```

`add_block`, `add_equation`, `set_poisson`, `set_density`, `step`, `step_cfl`, `advance` and the
diagnostics are forwarded to the compiled facade. On the C++ side the coupler lives in
`runtime/system.hpp` (`System`, multi-block single-level, shared Poisson) and is exposed to Python
by `python/bindings.cpp`. The backend (serial / OpenMP / Kokkos GPU / MPI) is the one with which
`libadc` was compiled; the physics never sees the backend.

> AMR variant. `adc.AmrSystem(n=, L=, periodic=)` is the refined counterpart of `System`:
> same `add_block` / `add_equation` / `set_poisson` / `set_density` / `step_cfl` signatures,
> plus `set_refinement(threshold)` (refines where the density exceeds a threshold) and
> `set_phi_refinement(grad_threshold)` (refines on `|grad phi|`). The regrid cadence is set
> via `AmrSystemConfig.regrid_every` (0 = frozen hierarchy). Detail:
> [AMR_MULTIBLOCK_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/AMR_MULTIBLOCK_DESIGN.md).

## Multi-block and multi-species

Several blocks co-exist in the same `System`, coupled only by the right-hand side of the
shared Poisson (`f = sum_s q_s n_s`) and, optionally, by inter-species sources; never
by the flux. Each block keeps its own model, its own spatial scheme and its own
time policy. In multi-block, the block name indexes `set_density(name)` / `density(name)`
/ `mass(name)`.

```python
n = 48
electrons = adc.Model(state=adc.FluidState("compressible", gamma=1.4),
                      transport=adc.CompressibleFlux(),
                      source=adc.PotentialForce(charge=-1.0),
                      elliptic=adc.ChargeDensity(charge=-1.0))
ions = adc.Model(state=adc.FluidState("isothermal", cs2=0.5),
                 transport=adc.IsothermalFlux(),
                 source=adc.PotentialForce(charge=+1.0),
                 elliptic=adc.ChargeDensity(charge=+1.0))

sim = adc.System(n=n, L=1.0, periodic=True)
sim.add_block("electrons", model=electrons,
              spatial=adc.Spatial(vanleer=True, flux="hllc"),
              time=adc.IMEX(substeps=10))           # stiff: implicit source, subcycled
sim.add_block("ions", model=ions, spatial=adc.Spatial(minmod=True), time=adc.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_density("electrons", ne0)
sim.set_density("ions", np.ones((n, n)))
sim.advance(0.001, 8)
print("blocks:", sim.block_names())
```

Inter-species coupled sources. In addition to the coupling by the field, inter-species sources
(operator-split, applied after the transport) transfer matter, momentum or
energy between blocks. Three fixed forms are added via `sim.add_coupling(...)` (or the direct
methods `add_ionization` / `add_collision` / `add_thermal_exchange`):

- `adc.Ionization(electron, ion, neutral, rate)`: ionization `n_g -> n_i + n_e` (rate
  `k n_e n_g`), mass transferred from the neutral to the ion;
- `adc.Collision(a, b, rate)`: inter-species friction (force `k (u_a - u_b)`), momentum
  conserved (fluid blocks, >= 3 variables);
- `adc.ThermalExchange(a, b, rate)`: thermal exchange `k (T_a - T_b)`, energy conserved
  (Euler blocks with 4 variables).

```python
sim.add_ionization(electron="ne", ion="ni", neutral="ng", rate=0.5)   # n_g decreases, n_i increases
sim.add_coupling(adc.Collision("a", "b", rate=1.0))                    # momentum transfer a -> b
sim.add_thermal_exchange("a", "b", rate=1.0)                           # energy hot -> cold
```

For a generic inter-species source (described in formulas rather than fixed), the DSL
`adc.dsl.CoupledSource(...).compile(...)` produces a descriptor that `sim.add_coupling(...)`
plugs in too (interpreted bytecode on the C++ side, no per-cell Python callback, MPI-safe). The
detail of the multi-species / plasma case is in
[ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md) (section 18, "composition runtime and multi-species
system") and [ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md). Exhaustive coupling surface:
[COUPLING_SURFACE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/COUPLING_SURFACE.md), [COUPLER_HIERARCHY.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/COUPLER_HIERARCHY.md).

## Spatial schemes

The spatial scheme = reconstruction (limiter) + numerical Riemann flux + reconstructed
variables. Two equivalent facades describe it.

`adc.Spatial(limiter=, flux=, recon=)` is the direct facade, with boolean shortcuts
(`minmod=True`, `vanleer=True`, `weno5=True`, `none=True`, `primitive=True`):

```python
adc.Spatial(minmod=True)                       # MUSCL minmod, Rusanov, conservative variables
adc.Spatial(vanleer=True, flux="hllc")         # MUSCL Van Leer, HLLC
adc.Spatial(weno5=True, primitive=True)        # WENO5-Z, primitive reconstruction
```

`adc.FiniteVolume(limiter=, riemann=, variables=)` is the same thing, but the numerical
Riemann flux is called `riemann` (and not `flux`, reserved for the physical flux of a DSL model):

```python
adc.FiniteVolume(limiter="minmod", riemann="rusanov", variables="conservative")
```

The possible values:

- limiter: `"none"` (first-order Godunov), `"minmod"`, `"vanleer"` (second-order MUSCL, 2 ghosts),
  `"weno5"` (WENO5-Z, order 5 in a smooth zone, 5-point stencil / 3 ghosts, oscillation-free
  capture near a front). `weno5` is exposed only by the native `add_block` path and the
  compiled `aot`/`production` backends (the `prototype` JIT path rejects it);
- Riemann flux: `"rusanov"` (the most stable, default for scalar transport), `"hllc"`,
  `"roe"`. HLLC and Roe require a compressible transport (4 variables + pressure);
- reconstruction: `"conservative"` or `"primitive"`. The primitive is more stable for Euler
  (positivity of `rho` and `p`).

On the C++ side, the limiters are policies in `numerics/reconstruction.hpp` (`NoSlope`,
`Minmod`, `VanLeer`, `Weno5`), the fluxes in `numerics/numerical_flux.hpp` (`RusanovFlux`,
`HLLFlux`, `HLLCFlux`, `RoeFlux`). Detail and formulas: [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md)
sections 2 and 3.

## Time schemes: explicit / IMEX / Strang / Schur

The time policy is per block (the object passed in `time=`). Four families.

Explicit: `adc.Explicit(substeps=, stride=, method=)` advances transport and source
explicitly by an SSP Runge-Kutta (`method="ssprk2"` by default, 2-stage Heun; `"ssprk3"`,
or shortcut `ssprk3=True`, 3-stage order 3, to pair with `weno5`).

```python
time=adc.Explicit()                       # SSPRK2, default
time=adc.Explicit(ssprk3=True)            # SSPRK3, less dissipative
```

IMEX: `adc.IMEX(substeps=, stride=, implicit_vars=, implicit_roles=)` (clear alias
`adc.SourceImplicit`) combines an explicit transport (SSPRK) and a stiff implicit source
(backward-Euler, cell-local Newton). The treatment is partial: only the source is
implicit, the transport stays explicit. It is not a global implicit PDE solver. The
`implicit_vars` / `implicit_roles` mask chooses which conserved variables are treated implicitly
(the others stay explicit); it is carried by the policy (the block), not by the model.

```python
time=adc.IMEX(substeps=10)                                  # stiff source, subcycled
time=adc.IMEX(implicit_roles=["MomentumX", "MomentumY", "Energy"])
```

Lie / Strang splitting + source stage condensed by Schur: `adc.Split` and `adc.Strang`
opt-in in the Schur effort, an explicit hyperbolic transport stage (`adc.Explicit`,
SSPRK) followed by a separate source stage `adc.CondensedSchur`. `adc.Split` chains `H(dt) ; S(dt)`
(Lie / Godunov, first order); `adc.Strang` plays `H(dt/2) ; S(dt) ; H(dt/2)` (symmetric, second
order). The `adc.CondensedSchur` stage handles the stiff coupled source potential / velocity / Lorentz
by assembling and solving a condensed tensorial elliptic operator (BiCGStab preconditioned
MG); it is a global implicit (it couples the whole domain). `adc.Split` / `adc.Strang` are
wired only by `add_equation` (which plugs in the source stage), not by `add_block`.

```python
sim.add_equation("ions", model=compiled,
                 spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
                 time=adc.Strang(hyperbolic=adc.Explicit(),
                                 source=adc.CondensedSchur(theta=0.5, alpha=3.0)))
```

> Local vs global. `adc.SourceImplicit` (IMEX) is local: it couples only the components
> of the same cell (relaxation, reactions, friction), without an elliptic solve. `adc.CondensedSchur`
> is global: for the stiff non-local Lorentz / electrostatic coupling. A purely local stiff
> source does not need Schur.

`adc.Implicit` is deprecated (alias of IMEX, emits a `DeprecationWarning`): its name wrongly suggests
a global implicit solver. Use `adc.SourceImplicit(...)` or `adc.IMEX(...)`.

Detail: [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md) sections 4 to 6,
[SCHUR_CONDENSATION_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/SCHUR_CONDENSATION_DESIGN.md), Hoffart step sequence
[HOFFART_STEP_SEQUENCE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/HOFFART_STEP_SEQUENCE.md). On the C++ side: `numerics/time/*.hpp`.

## Substeps, stride and multirate

Two orthogonal parameters, carried by every time policy, handle the multirate (not all
species require the same `dt`).

- `substeps=N`: the block advances `N` times per macro-step, each substep of length `dt/N`
  (fast species, e.g. electrons `substeps=10`). Default 1;
- `stride=M`: cadence of the block, hold-then-catch-up semantics (catch-up at the end of the window).
  The block is held (state unchanged) as long as `(macro_step + 1) % M != 0`, then advances by an
  effective step `M*dt` at the macro-step where `(macro_step + 1) % M == 0` (slow species, e.g. neutrals
  `stride=20`). It thus stays temporally consistent with the fast blocks (never advanced
  "into the future"). Default 1.

```python
sim.add_block("a", model=m, time=adc.Explicit(stride=1))   # every macro-step
sim.add_block("b", model=m, time=adc.Explicit(stride=3))   # advances once every 3 (end of window)
```

Between two catch-ups, the held block contributes to the right-hand side of the system Poisson with its
stale state (its last advanced density, frozen until the next catch-up). `step_cfl` honors
the cadence: the stable step includes the stride and substeps factor,
`dt <= cfl * h * substeps / (stride * w)`.

> Bit-parity warning. With `substeps=1` (whatever the stride), `step_cfl` is
> bit-identical to the history. With `substeps > 1` it advances a larger `dt` (each substep
> stays at the stability limit). To reproduce a calibrated run with the old formula, use
> `step(dt)` with the explicit historical `dt`.
>
> Backend note. The `aot` backend (`add_equation` on a `CompiledModel` `backend='aot'`) does not
> carry the cadence and rejects `stride > 1` (explicit route, no silent ignore);
> `add_block` (native) and `backend='production'` support the stride.

The multirate is obtained by setting `stride` (and `substeps`) per block. Detail:
[ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md) section 7.

## Initial conditions

Two ways to set the initial state of a block, both in numpy `(n, n)`. The layout
convention is row-major `(ny, nx)`: the first index (rows) is the slow `y` axis, the second
(columns) is the fast `x` axis; a field is indexed `ne[j, i]` (`j` = row / `y`, `i` =
column / `x`).

`set_density(name, arr)` sets the density (component 0) and leaves the rest at rest (for a
fluid: zero velocity, consistent energy). It is the usual shortcut for a scalar transport.

```python
coord = (np.arange(n) + 0.5) / n * L
xx, yy = np.meshgrid(coord, coord, indexing="xy")          # xx, yy of shape (n, n) = (ny, nx)
r = np.hypot(xx - 0.5 * L, yy - 0.5 * L)
ne = np.full((n, n), 1e-3)
ne[(r > 0.15) & (r < 0.20)] = 1.0
sim.set_density("ne", ne)
```

`set_primitive_state(name, **prims)` initializes a fluid block from its primitive variables,
named (`rho`, `u`, `v`, `p`...). Each primitive is an `(n, n)` array; the block model
converts them into conservative variables (compressible: `E = p/(g-1) + 1/2 rho|v|^2`). The names
expected are those of the model; an unknown or missing name raises a clear error.

```python
sim.set_primitive_state("electrons", rho=rho0, u=u0, v=v0, p=p0)
```

For an explicit conservative state (diagnostic / advanced case), `set_state(name, u)` takes the
flattened `(ncomp, n, n)` array.

## Outputs and diagnostics

The system returns its fields in numpy `(ny, nx)` (or `(ncomp, ny, nx)`), same row-major convention
as on input.

- `sim.density(name)`: density of the block, `(n, n)` array;
- `sim.mass(name)`: total mass of the block (scalar), the conservation invariant to check;
- `sim.potential()`: electrostatic potential `phi`, `(n, n)` array;
- `sim.time()`: current physical time;
- `sim.block_names()`: names of the blocks, in order of addition;
- `sim.get_state(name)`: full conservative state, `(ncomp, n, n)` (e.g. `[rho, rho*u, rho*v, E]`
  for Euler);
- `sim.get_primitive_state(name)`: state returned in primitive variables, dict
  `{name: (n, n)}` (inverse of `set_primitive_state`, round-trip to machine precision).

```python
m0 = sim.mass("ne")
for _ in range(500):
    sim.step_cfl(0.4)
rho = sim.density("ne")             # ndarray (n, n)
phi = sim.potential()
print("mass drift:", abs(sim.mass("ne") - m0))   # ~ machine roundoff

U = sim.get_state("electrons")      # (4, n, n) = [rho, rho*u, rho*v, E]
P = sim.get_primitive_state("electrons")            # {"rho": ..., "u": ..., "v": ..., "p": ...}
```

Two primitives serve to drive the solver from Python (custom time integrator, field
oracle):

- `sim.solve_fields()`: solves the system Poisson on the current state and repopulates the
  `aux` channel (`phi`, `grad phi`) without advancing in time, useful to read `potential()` at a frozen state;
- `sim.eval_rhs(name)`: evaluates the right-hand side `R = -div F + S` of the block (the spatial `dU/dt`),
  `(ncomp, n, n)`, for a user-provided time integrator.

Condensed API reference: [api](../reference/api_python.md). Complete recipes (figures, AMR):
[examples](../getting_started/organisation.md). A->Z tutorial: [tutoriels](../getting_started/tutorial.md).
