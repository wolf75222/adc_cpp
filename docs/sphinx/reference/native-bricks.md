# Reference : the native bricks (composition)

This page is the exhaustive registry of `pops`'s composable native bricks: each brick,
its signature, its parameters, what it declares or adds, and its constraints. It is the
detailed complement of the [models page](../models/index.md) (which presents the three ways
to write a model) and of the [Python API](python-api.md) (curated autodoc) ; here the detail at
the parameter level.

A native model is a composition of four role bricks via
`pops.Model(state=, transport=, source=, elliptic=)` : the cell-by-cell math stays compiled
C++ (no numpy loop on the hot path, GPU/MPI preserved), Python only assembles
objects. The core stays agnostic to the scenario : no physical name (diocotron,
Euler-Poisson, two-fluid) lives in `pops` ; the named compositions live in
[`adc_cases`](https://github.com/wolf75222/adc_cases). `pops.Model(...)` validates the
state <-> transport coherence and reports the parameters into a `ModelSpec` (tags read on the C++ side by the
model factory) ; an incoherent pairing raises an immediate `ValueError`. The builtin tags
(transport / source / elliptic) have a single C++ source of truth,
`include/pops/runtime/dynamic/model_registry.hpp` : the dispatch validates every tag against that registry
(an unknown tag is rejected explicitly) and the supported-vs-not-routed combinations
(e.g. `compressible` has no polar brick, a fluid force needs a transport with >= 3 variables) are
encoded there as data.

## State (state)

The state brick fixes the number of conservative variables, their names / primitives, and imposes
the compatible transport brick. It is the `state=` argument of `pops.Model(...)`.

| Brick | Signature | Declared variables | Required transport |
|---|---|---|---|
| `Scalar` | `pops.Scalar()` | 1 conservative variable `n` (transported density) ; primitive = conservative (`n`). | `ExB` |
| `FluidState` (compressible) | `pops.FluidState(kind="compressible", gamma=1.4)` | 4 variables `[rho, rho_u, rho_v, E]`, primitives `[rho, u, v, p]` ; carries `gamma` (reported into `spec.gamma`). | `CompressibleFlux` |
| `FluidState` (isothermal) | `pops.FluidState(kind="isothermal", cs2=0.5, vacuum_floor=0.0)` | 3 variables `[rho, rho_u, rho_v]`, primitives `[rho, u, v]` ; carries `cs2` (reported into `spec.cs2`) and `vacuum_floor` (reported into `spec.vacuum_floor`). | `IsothermalFlux` |

`FluidState(kind=...)` only accepts `"compressible"` or `"isothermal"` (any other value raises
a `ValueError`). The `gamma` / `cs2` arguments are stored even when the `kind` does not use them
; only the one of the chosen `kind` is reported into the spec.

`vacuum_floor` (isothermal only, native path only) bounds the velocity at quasi-vacuum : when
`> 0` the model reads `u = m / max(rho, vacuum_floor)`, capping the wave speed and the advective
flux where the flow evacuates the background (`rho -> ~0`). It leaves the conserved state
untouched and is `0` (inactive, bit-identical) by default. It is distinct from the spatial
Zhang-Shu `positivity_floor` (the reconstruction limiter) : the two address different failure
modes and are enabled separately.

## Transport

The transport brick writes the hyperbolic physical flux. It is the `transport=` argument of
`pops.Model(...)`. The physical parameters (`gamma`, `cs2`) come from the state, not from the transport.

| Brick | Signature | Physics | Required state |
|---|---|---|---|
| `ExB` | `pops.ExB(B0=1.0)` | Scalar advection by the E x B drift, `v = (-d_y phi, d_x phi) / B0`. Sets `spec.transport="exb"`, `spec.B0`. C++ struct `pops::ExBVelocity`. | `Scalar` |
| `CompressibleFlux` | `pops.CompressibleFlux()` | Compressible Euler flux (`gamma` comes from `FluidState`). Sets `spec.transport="compressible"`. C++ struct `pops::CompressibleFlux` (alias `pops::Euler`). | `FluidState(compressible)` |
| `IsothermalFlux` | `pops.IsothermalFlux()` | Isothermal Euler flux (`cs2` comes from `FluidState`). Sets `spec.transport="isothermal"`. C++ struct `pops::IsothermalFlux`. | `FluidState(isothermal)` |

There is no other native transport brick. For a novel hyperbolic flux, you go
through the DSL (`pops.physics.bricks.HyperbolicBrick`, cf. [models page](../models/index.md)).

## Source

The source brick adds the pointwise source term `S(U, aux)` to the block's RHS. It is the
`source=` argument of `pops.Model(...)`. It reads the external state through the `pops::Aux` channel (potential
`phi`, gradients `grad_x` / `grad_y`).

| Brick | Signature | Adds to the RHS | Min. variables |
|---|---|---|---|
| `NoSource` | `pops.NoSource()` | nothing. Sets `spec.source="none"`. C++ struct `pops::NoSource`. | 1 |
| `PotentialForce` | `pops.PotentialForce(charge=1.0)` | Potential force `(q/m) rho E` on the momentum (+ work term if 4 variables). Sets `spec.source="potential"`, `spec.qom=charge`. C++ struct `pops::PotentialForce`. | 3 |
| `GravityForce` | `pops.GravityForce()` | Gravitational force `rho g` (+ work if 4 variables). Sets `spec.source="gravity"`. C++ struct `pops::GravityForce`. | 3 |
| `MagneticLorentzForce` | `pops.MagneticLorentzForce(charge=1.0)` | Magnetic Lorentz force `q (v x B_z)` on the momentum (explicit regime, no work). Reads `B_z` (set via `System.set_magnetic_field`). Sets `spec.source="magnetic"`, `spec.qom=charge`. C++ struct `pops::MagneticLorentzForce`. | 3 |
| `PotentialMagneticForce` | `pops.PotentialMagneticForce(charge=1.0)` | Electrostatic + magnetic Lorentz summed `(q/m) rho E + q (v x B_z)`. Reads `B_z` (set via `System.set_magnetic_field`). Sets `spec.source="potential_magnetic"`, `spec.qom=charge`. C++ struct `pops::CompositeSource<pops::PotentialForce, pops::MagneticLorentzForce>`. | 3 |

Note : `PotentialForce(charge=...)` names the parameter `charge` on the Python side but reports it into
`spec.qom` (charge/mass ratio `q/m`) on the C++ side.

### Inter-species couplings (add_coupling)

Inter-species couplings are not `Model(source=)` sources : they are passed to
`System.add_coupling(...)`, applied in operator-split after the transport (not integrated into the
block's RHS). They link two blocks (or three for ionization) by their name.

| Brick | Signature | Effect | Target |
|---|---|---|---|
| `Ionization` | `pops.Ionization(electron, ion, neutral, rate)` | Ionization `n_g -> n_i + n_e`, rate `k n_e n_g` ; mass transferred from the neutral to the ion. Routes to `add_ionization`. | 3 blocks (electron, ion, neutral) |
| `Collision` | `pops.Collision(a, b, rate)` | Inter-species friction : force `k (u_a - u_b)`, momentum conserved. Routes to `add_collision`. | fluid blocks (>= 3 variables) |
| `ThermalExchange` | `pops.ThermalExchange(a, b, rate)` | Thermal exchange `k (T_a - T_b)`, energy conserved. Routes to `add_thermal_exchange`. | Euler blocks (4 variables) |

`add_coupling` also accepts a `pops.physics.multispecies.CompiledCoupledSource` (generic coupling described in
formulas, carried as bytecode and interpreted on the C++ side) ; cf. [models page](../models/index.md).

## Elliptic right-hand side (elliptic)

The elliptic brick fixes the block's contribution to the right-hand side of the system Poisson. It is
the `elliptic=` argument of `pops.Model(...)`. The system Poisson sums the contributions of
all the blocks.

| Brick | Signature | Contribution to the elliptic RHS |
|---|---|---|
| `ChargeDensity` | `pops.ChargeDensity(charge=1.0)` | Charge density `f = q n`. Sets `spec.elliptic="charge"`, `spec.q=charge`. C++ struct `pops::ChargeDensity`. |
| `BackgroundDensity` | `pops.BackgroundDensity(alpha=1.0, n0=0.0)` | Neutralizing background `f = alpha (n - n0)`. Sets `spec.elliptic="background"`, `spec.alpha`, `spec.n0`. C++ struct `pops::BackgroundDensity`. |
| `GravityCoupling` | `pops.GravityCoupling(sign=1.0, four_pi_G=1.0, rho0=1.0)` | Self-consistent coupling `f = sign 4piG (rho - rho0)` (`sign=+1` gravity, `sign=-1` plasma). Sets `spec.elliptic="gravity"`, `spec.sign`, `spec.four_pi_G`, `spec.rho0`. C++ struct `pops::GravityCoupling`. |

(briques-epm)=
### EPM bricks : the Poisson operator is itself composable

The elliptic model (EPM, EllipticPhysicalModel) is not a hard-coded case : it is a
composition of bricks (unknown + operator + right-hand side + output). The Poisson is its
current instance. These bricks compose via `pops.elliptic(...)` then plug in via
`System.add_elliptic_model(...)`.

| Brick / factory | Signature | Role |
|---|---|---|
| `DivEpsGrad` | `pops.DivEpsGrad(epsilon=1.0)` | Operator `D = div(eps grad .)`. `eps=1` -> Poisson ; `eps != 1` constant supported (`eps lap phi = f`). Variable `eps(x)` plugs in via `set_epsilon_field`. |
| `div_eps_grad` | `pops.div_eps_grad(epsilon=1.0)` | Factory : returns a `DivEpsGrad`. |
| `CompositeRhs` | `pops.CompositeRhs()` | Generic right-hand side `f = sum_s elliptic_rhs_s(u_s)` : the sum of the elliptic bricks carried by the blocks. Assumes no particular form. |
| `composite_rhs` | `pops.composite_rhs()` | Factory : returns a `CompositeRhs`. |
| `ChargeDensitySource` | `pops.ChargeDensitySource()` (subclass of `CompositeRhs`) | Usual case : all the blocks carry a charge density, so `f = sum_s q_s n_s`. Historical alias of `CompositeRhs` (same computation, the sum of the bricks). |
| `charge_density` | `pops.charge_density()` | Factory : returns a `ChargeDensitySource`. |
| `ElectricFieldFromPotential` | `pops.ElectricFieldFromPotential()` | Output / post-processing `E = -grad phi`, reinjected into the `aux` of the hyperbolic models. |
| `electric_field_from_potential` | `pops.electric_field_from_potential()` | Factory : returns an `ElectricFieldFromPotential`. |
| `EllipticModel` | `pops.EllipticModel(unknown, operator, rhs, output)` | Carries the 4 slots of the EPM (unknown + operator + right-hand side + output). |
| `elliptic` | `pops.elliptic(unknown="phi", operator=None, rhs=None, output=None)` | Composes an EPM. Defaults : `operator=DivEpsGrad()`, `rhs=CompositeRhs()`, `output=ElectricFieldFromPotential()`. |
| `EllipticSolver` | `pops.EllipticSolver(kind="geometric_mg")` | Solver choice : `"geometric_mg"` (any case, walls), `"fft"` (periodic, `n = 2^k`), or `"fft_spectral"` (periodic, continuous spectral symbol `-(kx^2+ky^2)`, reference fidelity). |

The canonical Poisson is thus written :

```python
poisson = pops.elliptic(
    operator=pops.div_eps_grad(1.0),         # D = lap
    rhs=pops.charge_density(),               # f = somme_s q_s n_s
    output=pops.electric_field_from_potential(),  # E = -grad phi
)
```

`System.add_elliptic_model(name, model, solver=None, bc="auto", wall="none", wall_radius=0.0)`
wires this EPM : it validates that `operator` is a `DivEpsGrad` (otherwise `NotImplementedError` :
only `div_eps_grad` is supported, diffusion / projection would require another solver) and that
`rhs` is a `CompositeRhs` (otherwise `NotImplementedError`), then forwards to `set_poisson(...)`.
The right-hand side token is `"charge_density"` when `rhs` is exactly a
`ChargeDensitySource`, otherwise `"composite"` (same C++ numerics : the sum of the elliptic
bricks per block). `add_elliptic_model(...)` is thus the explicit form of `set_poisson` :

```python
# ces deux appels sont equivalents (memes numeriques) :
sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet",
                wall="circle", wall_radius=0.40)

sim.add_elliptic_model(
    "poisson",
    pops.elliptic(operator=pops.div_eps_grad(1.0), rhs=pops.charge_density(),
                 output=pops.electric_field_from_potential()),
    solver=pops.EllipticSolver("geometric_mg"),
    bc="dirichlet", wall="circle", wall_radius=0.40,
)
```

## Composing a model

`pops.Model(state, transport, source, elliptic)` returns a `ModelSpec` (the 100 %
native model object, consumed by `add_block` / `add_equation`). The validation of the four roles :

- `state` must be `Scalar` or `FluidState(...)` (otherwise `ValueError`) ;
- the state <-> transport coherence is imposed : `Scalar` requires `ExB` ; `FluidState(compressible)`
  requires `CompressibleFlux` ; `FluidState(isothermal)` requires `IsothermalFlux` ;
- `source` must be `NoSource` / `PotentialForce` / `GravityForce` / `MagneticLorentzForce` /
  `PotentialMagneticForce` ;
- `elliptic` must be `ChargeDensity` / `BackgroundDensity` / `GravityCoupling`.

```python
model = pops.Model(
    state=pops.Scalar(),
    transport=pops.ExB(B0=1.0),
    source=pops.NoSource(),
    elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0),
)
```

### CompositeModel : hybrid native + DSL model

`pops.CompositeModel(transport, source, elliptic, name="hybrid")` mixes, in a single model,
native bricks (`pops.ExB`, `pops.PotentialForce`, `pops.ChargeDensity`...) and partial compiled DSL
bricks (`pops.physics.bricks.HyperbolicBrick(...).compile()`, `SourceBrick`, `EllipticBrick`).
Each slot accepts either a native brick or a compiled DSL brick.

- At least one slot must be a DSL brick : an all-native composition is written with
  `pops.Model(...)` (otherwise `CompositeModel` raises a `ValueError`).
- A brick placed in the wrong slot raises a `ValueError` (the slot is checked).
- `CompositeModel(...)` returns a `pops.physics.hybrid.HybridModel` ; you call `.compile(backend="aot")` for
  a `CompiledModel` pluggable via `System.add_equation`. Prototype : only `backend="aot"` is
  wired.

```python
m = pops.CompositeModel(
    transport=build_iso_transport(0.7).compile(),  # transport DSL
    source=pops.PotentialForce(charge=-1.0),        # source native
    elliptic=pops.ChargeDensity(charge=-1.0),       # elliptique native
)
compiled = m.compile(backend="aot")                # -> CompiledModel
sim.add_equation("gas", compiled,
                 spatial=pops.FiniteVolume(limiter="minmod", riemann="rusanov"),
                 names=["rho", "rho_u", "rho_v"])
```

The transport slot fixes the layout (`n_vars`, conservative names, primitives, gamma) ; a source / elliptic
DSL brick must declare the same `n_vars`. Detail : [models page](../models/index.md).

### PythonFlux : flux written in Python (host prototyping)

`pops.PythonFlux(flux, max_wave_speed)` is a prototyping backend : the user provides the physical flux
`flux(U, dir)` and the wave speed `max_wave_speed(U)` in numpy, and `PythonFlux` assembles the
residual `-div(F*)` by Rusanov flux (order 1, periodic domain) over the whole array. It is a
pure host path (never a Kokkos kernel), outside the GPU / MPI hot path ; it serves to iterate on a novel flux
without recompiling (pattern of the `custom_scheme` case, with `pops.System` as Poisson oracle). For
production, compose a compiled flux (`pops.CompressibleFlux`, `pops.ExB`, or a DSL model).

```python
import pops
pf = pops.PythonFlux(flux=mon_flux, max_wave_speed=ma_vitesse)
dUdt = pf.residual(U, dx)              # -div(F*) par Rusanov ordre 1, periodique
dt = pf.cfl_dt(U, h, cfl=0.4)          # dt = cfl * h / max_wave_speed(U)
```

## Spatial schemes (Spatial / FiniteVolume)

The spatial scheme is carried by the block (`spatial=` argument of `add_block` / `add_equation`),
not by the model. It combines reconstruction (limiter) + Riemann numerical flux + reconstructed
variables.

`pops.Spatial(limiter="minmod", flux="rusanov", recon="conservative", *, none=False,
minmod=False, vanleer=False, weno5=False, primitive=False, positivity_floor=None)` :

| Argument | Values | Detail |
|---|---|---|
| `limiter` | `"none"`, `"minmod"`, `"vanleer"`, `"weno5"` | MUSCL reconstruction (none / minmod / vanleer, 2 ghosts) or WENO5-Z. `weno5` = order 5 in smooth zone, 5-point stencil -> 3 ghosts ; only the native `add_block` path (and the `aot` / `production` / AMR backends) expose it ; the `prototype` backend (JIT) rejects it. Boolean shortcuts `none=` / `minmod=` / `vanleer=` / `weno5=`. |
| `flux` | `"rusanov"`, `"hll"`, `"hllc"`, `"roe"` | Riemann numerical flux. `rusanov` = minimal generic (only `max_wave_speed` required). `hll` = generic with signed waves : requires `model.wave_speeds` (native isothermal / compressible model, or DSL model with primitive `p` declared) ; it is the recommended path for a NON Euler model with signed waves (`hll` + `minmod`). `hllc` / `roe` = contact-resolving (HLLC) and Roe-linearized solvers. The canonical native path is 2D Euler (4 variables + perfect gas pressure : `FluidState(compressible)`) ; they are also GENERIC on a model that supplies the hooks `HasHLLCStructure` (`contact_speed` + `hllc_star_state`) or `HasRoeDissipation` (`roe_dissipation`), emitted in the DSL via `m.enable_hllc()` / `m.enable_roe()` (e.g. a 3-variable isothermal system). All paths read a pressure : declare a primitive `p` ; without `p` (and without the capability) the wiring raises a `ValueError`. |
| `recon` | `"conservative"`, `"primitive"` | Reconstructed variables. `primitive` is more stable for Euler (positivity of `rho` and `p`). Shortcut `primitive=`. |
| `positivity_floor` | `None` or a float | Zhang-Shu positivity floor on reconstructed face densities (default `None` = off). Clamps the reconstructed value to `>= floor`, for high-contrast initial conditions where WENO5 can reconstruct a negative density. |

`pops.FiniteVolume(limiter="minmod", riemann="rusanov", variables="conservative")` is the stable
surface factory : it remaps onto `pops.Spatial`. The numerical flux is named `riemann` there (and not
`flux`, reserved for the physical flux of the DSL model `m.flux`, so as not to collide the two senses) :

| `FiniteVolume(...)` | -> | `Spatial(...)` |
|---|---|---|
| `limiter` | -> | `Spatial.limiter` |
| `riemann` | -> | `Spatial.flux` |
| `variables` | -> | `Spatial.recon` |

`FiniteVolume(...)` returns a `Spatial` (consumed as is) ; `pops.Spatial` stays available
identically.

```{note}
The only existing limiters are none / minmod / vanleer / weno5 ; no other limiter is
exposed.
```

## Temporal treatment (time)

The temporal treatment is carried by the block (`time=` argument), not by the model : the same
model is reused with distinct policies.

| Brick | Signature | Detail |
|---|---|---|
| `Explicit` | `pops.Explicit(substeps=1, method="ssprk2", stride=1, *, ssprk3=False)` | Explicit integration. `method="ssprk2"` (Shu-Osher 2-stage order 2, bit-identical default), `"ssprk3"` (3-stage order 3, less dissipative, to pair with weno5 ; shortcut `ssprk3=True`), or `"euler"` (ForwardEuler order 1, first-order-reference fidelity / validation, never the default). Exposes `.kind` = `"explicit"`, `"ssprk3"`, or `"euler"`. |
| `IMEX` | `pops.IMEX(substeps=1, stride=1, implicit_vars=None, implicit_roles=None)` | Explicit transport (SSPRK) + implicit stiff source (backward-Euler, cell-local Newton). Not a global implicit PDE solver. `kind="imex"`. |
| `SourceImplicit` | `pops.SourceImplicit(substeps=1, stride=1, implicit_vars=None, implicit_roles=None)` | Clear name of the source-only IMEX scheme ; `kind="imex"` (same C++ path as `IMEX`, bit-identical). The doc contrasts local (this brick) vs global (`CondensedSchur`). |
| `Implicit` | `pops.Implicit(dt_ratio=1, substeps=None, stride=1)` | Obsolete : alias of `IMEX`. Emits a `DeprecationWarning` (the name wrongly suggests a global implicit solver) and returns an `IMEX(...)`. Use `SourceImplicit` / `IMEX`. |
| `Split` | `pops.Split(hyperbolic=None, source=None)` | Explicit / implicit splitting policy : transport stage `pops.Explicit` (default `Explicit()`) + separate source stage `pops.CondensedSchur` (required). `scheme="lie"` (Godunov, 1st order). Relays `kind` / `method` / `substeps` / `stride` of the hyperbolic stage. Wired only by `add_equation` (rejected by `add_block`). |
| `Strang` | `pops.Strang(hyperbolic=None, source=None)` | Subclass of `Split` : Strang splitting (symmetric, 2nd order) `H(dt/2); S(dt); H(dt/2)`. Sets `scheme="strang"` ; re-solves the fields between the stages. |
| `CondensedSchur` | see below | Schur-condensed source stage (global). It is the `source=` of a `Split` / `Strang`. |
| `Role` | constants | Physical roles : `Density`, `MomentumX`, `MomentumY`, `MomentumZ`, `Energy`, `VelocityX/Y/Z`, `Pressure`, `Temperature`, `Scalar` (stable snake_case keys). Serves in `CondensedSchur` and the IMEX masks. |

### CondensedSchur (global source stage)

`pops.CondensedSchur(kind="electrostatic_lorentz", theta=0.5, alpha=1.0, density=Role.Density,
momentum=(Role.MomentumX, Role.MomentumY), energy=None, magnetic_field="B_z", potential="phi")`
is the Schur-condensed source stage (Hoffart et al., arXiv:2510.11808). It assembles the condensed
elliptic operator `A = I + theta^2 dt^2 alpha rho B^{-1}`, solves it (preconditioned BiCGStab
MG) and reconstructs the velocity. Everything is C++ (no per-cell Python callback).

| Parameter | Constraint |
|---|---|
| `kind` | only `"electrostatic_lorentz"` (any other value raises a `ValueError`). |
| `theta` | theta-scheme in `(0, 1]` (`0.5` Crank-Nicolson, `1` backward Euler). |
| `alpha` | electrostatic coupling constant of the source subsystem. |
| `density` / `momentum` / `energy` / `magnetic_field` / `potential` | Role / field descriptors. All rejected if they depart from the default : the C++ source stage hard-fixes the roles `Density` / `MomentumX` / `MomentumY` (`Energy` optional) and the fields `B_z` / `phi`. The signature is kept for when the C++ will carry them, but a different descriptor raises a `ValueError` (rejection rather than silent ignore). |

`CondensedSchur` requires from the block the roles `Density` / `MomentumX` / `MomentumY` and a `B_z` field
(`set_magnetic_field`) ; a missing role / `B_z` raises an explicit error at `add_equation`. It
is wired in cartesian and in polar ; the polar counterpart is single-rank (`n_ranks > 1` raises).

### Multirate : substeps and stride

`substeps` and `stride` are orthogonal (valid on `Explicit` / `IMEX` / `SourceImplicit`) :

- `substeps=N` : the block advances N times per macro-step, each substep of length `dt/N`
  (fast electrons : `substeps=10`). Default 1 = bit-identical to the history.
- `stride=M` : hold-then-catch-up cadence (catch-up at window end). The block is held as long as
  `(macro_step + 1) % M != 0`, then advances by an effective step `M dt` when
  `(macro_step + 1) % M == 0` (slow block, e.g. neutrals : `stride=20`). Between two catch-ups, its
  stale state (last advanced density / charge, frozen) still contributes to the system Poisson and to
  the coupled sources. `step_cfl` honors the cadence : `dt <= cfl h substeps / (stride w)`.

### IMEX mask (implicit_vars / implicit_roles)

`implicit_vars` (conserved variable names) and `implicit_roles` (physical roles, normalized
into stable keys via `Role`) list the components treated implicitly in the source step ;
the rest stays explicit. The mask is carried by the temporal policy / the block, not by the
model -> the same model is reused with distinct implicit treatments. Default `[]`
(empty union) = model default, bit-identical. The names / roles -> indices resolution and the
validation (name / role absent from the block) are C++-side (single source of truth). A bare string
is tolerated (`implicit_vars="rho_u"` -> `["rho_u"]`).

### Per-backend safeguards

| Path | Stride > 1 | evolve=False | weno5 | non-rusanov flux | IMEX mask |
|---|---|---|---|---|---|
| native `add_block` (`ModelSpec`) | supported | supported | supported | supported | supported |
| `add_equation` `production` backend | supported | supported | supported | supported | rejected (native `.so`) |
| `add_equation` `aot` backend | rejected | rejected | supported | supported | rejected |
| `add_equation` `prototype` backend | (substeps only) | rejected | rejected | rejected (rusanov only) | rejected |

The rejections are explicit (`ValueError`), never a silent ignore : the `.so` ABI of these
backends does not carry the argument concerned (cadence, `evolve`, mask), so the block
would run at the default value without saying so. `Split` / `Strang` are rejected by `add_block`
(only `add_equation` wires the source stage `set_source_stage`).

## Mesh (mesh)

The choice of geometry lives in a mesh object passed as `mesh=` to `pops.System(...)`, not
in the scheme (`pops.FiniteVolume` stays reconstruction + Riemann + variables, without a geometry
argument). The mesh is applied after `**cfg_kw`, so `mesh=` prevails over the `n=` / `L=`
passed as keywords.

| Brick | Signature | Effect |
|---|---|---|
| `CartesianMesh` | `pops.CartesianMesh(n=64, L=1.0, periodic=True)` | Square domain `[0, L]^2`, `n x n` cells (implicit default). `pops.System(mesh=pops.CartesianMesh(n, L, periodic))` is strictly equivalent (bit-identical) to `pops.System(n=n, L=L, periodic=periodic)`. Sets `config.geometry="cartesian"`, `n`, `L`, `periodic`. |
| `PolarMesh` | `pops.PolarMesh(r_min, r_max, nr, ntheta)` | Global ring `r in [r_min, r_max] x theta in [0, 2pi)`, `nr x ntheta` cells. theta periodic, r carries a physical boundary condition (direction 0 = radial, 1 = azimutal). Sets `config.geometry="polar"`, `nr`, `ntheta`, `r_min`, `r_max`, and `config.n = nr` (default size of the diagnostics). |

`PolarMesh` validation : `r_max > r_min >= 0` (otherwise `ValueError`), `nr >= 3` (the order-2 upwind radial
stencil at the walls would otherwise read `phi` out of bounds), `ntheta >= 1`.

Limits of the polar path (wired in `System.step` : polar transport + polar Poisson + aux in local
basis `e_r` / `e_theta`) : transport is scalar `ExB` or the isothermal fluid (`IsothermalFluxPolar`),
with `rusanov` on any polar model and `hll` on the isothermal fluid (it declares `wave_speeds`),
limiters `minmod` / `vanleer` / `weno5` ; `hllc` / `roe` are raised (no polar energy flux brick). The
DIRECT polar Poisson is single-box / single-rank (refuses MPI and `theta_boxes > 1`), while the polar
transport and the tensorial Schur stage are multi-box by azimuthal (`theta_boxes`) split. No
cartesian <-> polar coupling (global ring). Cf. `pops.capabilities()['riemann']['system_polar']` and
`['geometry']['system_polar']`.

### Configuration fields

`SystemConfig` (readwrite fields) : `n`, `L`, `periodic`, `geometry`, `nr`, `ntheta`, `r_min`,
`r_max`.

`AmrSystemConfig` (readwrite fields) : `n`, `L`, `regrid_every`, `periodic`, `distribute_coarse`,
`coarse_max_grid`. `regrid_every == 0` -> frozen hierarchy (regrid never called, bit-identical).

## System / AmrSystem : methods

The Python wrapper defines a few methods (composition, primitives, EPM, disc) ; all the
rest is delegated to the compiled C++ facade via `__getattr__`. Compact reference.

### System

| Method | Signature | Role |
|---|---|---|
| `add_block` | `add_block(name, model, spatial=None, time=None, evolve=True)` | Adds a block from a `ModelSpec`. Defaults `Spatial()` / `Explicit()`. Rejects `Split` / `Strang` (use `add_equation`). |
| `add_equation` | `add_equation(name, model, spatial=None, time=None, substeps=None, names=None, evolve=True, stride=None)` | Switches on the type : `ModelSpec` -> `add_block` ; `CompiledModel` -> the backend's adder (`add_dynamic_block` prototype / `add_compiled_block` aot / `add_native_block` production) ; handles `Split` / `Strang` (hyperbolic stage then `set_source_stage` + `set_time_scheme`). Applies the backend safeguards. |
| `run` | `run(t_end, cfl=0.4, max_steps=1_000_000)` | Sugar `while time() < t_end: step_cfl(cfl)` ; returns the number of steps. |
| `add_background` | `add_background(name, model, density, spatial=None)` | Frozen species = `add_block(evolve=False)` + `set_density`. |
| `add_elliptic_model` | `add_elliptic_model(name, model, solver=None, bc="auto", wall="none", wall_radius=0.0)` | Wires an EPM (validates `DivEpsGrad` + `CompositeRhs`) ; forwards to `set_poisson`. |
| `set_disc_domain` | `set_disc_domain(cx, cy, R, mode="none")` | Disc transport domain ; `mode` : `"none"` (mask set, full cartesian transport, bit-identical) / `"staircase"` (conservative masked transport) / `"cutcell"` (cut-cell / embedded-boundary). Cartesian only. |
| `set_geometry_mode` | `set_geometry_mode(mode)` | Switches the disc transport mode without redefining the disc. |
| `disc_mask` | `disc_mask()` | Cell-centered 0/1 mask `(ny, nx)` (diagnostic). |
| `add_coupling` | `add_coupling(coupling)` | Routes `Ionization` / `Collision` / `ThermalExchange` or a `CompiledCoupledSource`. |
| `block_names` | `block_names()` | Block names in order (includes dynamic / compiled blocks). |
| `set_primitive_state` | `set_primitive_state(name, **prims)` | Initializes a block from its named primitives (`rho` / `u` / `v` / `p`), assembled `(ncomp, n, n)` in the model's order, converted to conservative on the C++ side. |
| `get_primitive_state` | `get_primitive_state(name)` | Inverse : returns a dict `{nom_primitive: (n, n)}`. |
| `abi_key` | `System.abi_key()` (static) | Module ABI key. |

C++ facade methods reached by `__getattr__` (with defaults) :

- `set_poisson(rhs="charge_density", solver="geometric_mg", bc="auto", wall="none", wall_radius=0.0, epsilon=1.0, abs_tol=0.0)` : `bc` e.g. `"dirichlet"` ; `wall` e.g. `"circle"` + `wall_radius` ; `abs_tol` = absolute floor for the GeometricMG V-cycle stopping criterion (`0` = relative criterion only).
- `set_density(name, rho)` : `rho` array `n x n`.
- `set_epsilon_field(eps)`, `set_epsilon_anisotropic_field(eps_x, eps_y)`, `set_reaction_field(kappa)`, `set_magnetic_field(bz)`, `set_electron_temperature_from(name)`.
- `set_source_stage(name, kind, theta, alpha)`, `set_time_scheme(scheme)` (`"lie"` / `"strang"`).
- `add_coupled_source(in_blocks, in_roles, consts, out_blocks, out_roles, prog_ops, prog_args, prog_lens)`.
- `add_dynamic_block` / `add_compiled_block` / `add_native_block` / `set_block_params` : backend adders (used internally by `add_equation`).
- `add_ionization(electron, ion, neutral, rate)` / `add_collision(a, b, rate)` / `add_thermal_exchange(a, b, rate)`.
- `variable_names(name, kind="conservative")` / `variable_roles(name, kind="conservative")` / `block_gamma(name)` / `n_vars(name)`.
- `solve_fields()` ; `step(dt)` ; `advance(dt, nsteps)` ; `step_cfl(cfl)` ; `step_adaptive(cfl)`.
- `eval_rhs(name)` / `get_state(name)` / `set_state(name, u)` : primitives of a custom Python integrator.
- `nx()` ; `ny()` ; `time()` ; `n_species()` ; `mass(name)` ; `density(name)` -> `(ny, nx)` ; `potential()` -> `(ny, nx)`.

### AmrSystem

`pops.AmrSystem` is the refined counterpart : one or several blocks carried on a shared AMR
hierarchy, with a system Poisson with summed right-hand side `sum_b q_b n_b` and per-block
conservation. In multi-blocks the block name indexes `set_density(name)` / `mass(name)` / `density(name)`.

| Method | Signature | Role |
|---|---|---|
| `add_block` | `add_block(name, model, spatial=None, time=None)` | Rejects `Split` ; threads substeps / stride + IMEX mask to the C++. |
| `add_equation` | `add_equation(name, model, spatial=None, time=None, substeps=None)` | `ModelSpec` -> `add_block` (forwards stride + mask) ; a `CompiledModel` must be `backend="production"`, `target="amr_system"` -> `add_native_block`. Rejects stride > 1 and IMEX mask on the production `.so` (flat ABI) ; requires `p` for hllc / roe. Primitive recon + roe / hllc flux + weno5 are wired on AMR (parity with `add_block`). |
| `set_refinement` | `set_refinement(threshold, variable="", role="")` | Tags where the selected variable of a block exceeds `threshold`. Default = component 0 (historical density), bit-identical ; `variable=` / `role=` select it per block by name or physical role (a block lacking it raises at build, no silent component-0 fallback). Non-default selector is multi-block only (mono-block / compiled `.so` : component 0 only, selector rejected at build). |
| `set_phi_refinement` | `set_phi_refinement(grad_threshold)` | Adds a tag based on `|grad phi|` to the regrid union (multi-blocks + `regrid_every > 0` ; `<= 0` disables, default). |
| `set_poisson` | `set_poisson(rhs="charge_density", solver="geometric_mg", bc="auto", wall="none", wall_radius=0.0)` | System Poisson (no `epsilon` argument on this path). |
| `set_density` | `set_density(name, rho)` | Initial density of a block. |
| `add_coupled_source` | `add_coupled_source(...)` | Generic coupled source (bytecode). |
| `step` / `advance` / `step_cfl` | `step(dt)` / `advance(dt, nsteps)` / `step_cfl(cfl)` | Advance. |
| `nx` / `time` / `n_blocks` / `n_patches` | (no argument) | Scalar diagnostics. `n_blocks()` = number of blocks ; `n_patches()` = number of fine patches. |
| `mass` / `density` | `mass()` / `mass(name)` ; `density()` / `density(name)` -> `(nx, nx)` | Empty name -> 1st block ; in multi-blocks the name indexes the block. |
| `potential` | `potential()` -> `(n, n)` | phi of the coarse level (shared system Poisson). |
| `patch_boxes` | `patch_boxes()` (recent) | Index-space footprints of the fine patches : list of `(level, ilo, jlo, ihi, jhi)`, inclusive corners, in the index space of the level (`n << level` cells/direction, ratio 2). Rank-independent (MPI-safe). |
| `patch_rectangles` | `patch_rectangles()` (recent) | Converts `patch_boxes()` into physical rectangles `(x0, y0, w, h)` in `[0, L]^2` (one per fine patch). Convenient for drawing the patches (e.g. `matplotlib.Rectangle`). |

```{note}
`patch_boxes()` / `patch_rectangles()` expose the geometry of the fine patches (recent
addition). If the `pops` module built on your branch is earlier than this addition, these methods
may not exist yet ; they are listed here for the complete reference of the API.
```

## Complete example : a diocotron from the bricks

A reduced diocotron = a scalar electron density advected by E x B, with a neutralizing
background, and a Poisson with a circular conducting wall. Built entirely from
native bricks (no `models.diocotron` helper) :

```python
import numpy as np
import pops

# --- maillage + systeme (carre cartesien, non periodique pour le mur conducteur) ---
n = 192
sim = pops.System(mesh=pops.CartesianMesh(n=n, L=1.0, periodic=False))

# --- le modele, compose des quatre briques de role ---
#   state    : Scalar          (une variable : la densite electronique n)
#   transport: ExB(B0=1.0)     (derive E x B)
#   source   : NoSource        (pas de source ponctuelle pour un scalaire)
#   elliptic : BackgroundDensity(alpha=1.0, n0=0.0)   (f = alpha (n - n0))
model = pops.Model(
    state=pops.Scalar(),
    transport=pops.ExB(B0=1.0),
    source=pops.NoSource(),
    elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0),
)

# --- branche le bloc : schema spatial + integrateur temporel ---
sim.add_block(
    "ne",
    model=model,
    spatial=pops.FiniteVolume(limiter="minmod", riemann="rusanov", variables="conservative"),
    time=pops.Explicit(),                  # SSPRK2, substeps=1, stride=1
)

# --- Poisson de systeme avec mur conducteur circulaire (Dirichlet) ---
sim.set_poisson(bc="dirichlet", wall="circle", wall_radius=0.40)

# --- densite annulaire initiale (anneau d'electrons creux) ---
xs = (np.arange(n) + 0.5) / n
X, Y = np.meshgrid(xs, xs, indexing="ij")
r = np.hypot(X - 0.5, Y - 0.5)
ne0 = np.where((r > 0.20) & (r < 0.30), 1.0, 0.0)
# germe azimutal pour declencher l'instabilite
theta = np.arctan2(Y - 0.5, X - 0.5)
ne0 = ne0 * (1.0 + 0.01 * np.cos(5.0 * theta))
sim.set_density("ne", ne0)

# --- avance quelques pas limites par le CFL ---
for _ in range(50):
    sim.step_cfl(0.4)

print("t        =", sim.time())
print("mass(ne) =", sim.mass("ne"))      # invariant conserve
phi = sim.potential()                    # (n, n)
ne = sim.density("ne")                    # (n, n)
print("phi range:", float(phi.min()), float(phi.max()))
```

The same `ModelSpec` plugs onto `pops.AmrSystem` (adaptive refinement) without changing the
model : `sa.add_block("ne", model=model, ...)`. For the explicit form of `set_poisson` (via a
composed EPM), see the section [EPM bricks](#briques-epm).
