# Models

A *model* in `pops` describes an equation: its pointwise formulas (flux, source,
wave speeds, elliptic right-hand side). There are three ways to write a model, which
all produce the same computational object on the C++ core side and plug into
an `pops.System` the same way:

1. **Model with bricks (native)**: you compose generic bricks that are already compiled
   (`pops.Model(state, transport, source, elliptic)`). This is the most direct way to assemble
   an existing model: no just-in-time compilation, full production parity (MPI/AMR/GPU).
2. **DSL model**: you write the model as symbolic formulas (`pops.physics.facade.Model`), then you
   compile it into a `.so`. This is the way when the model you want does not exist as a native brick.
3. **Hybrid model**: you mix, within a single model, native bricks and partial DSL bricks
   (`pops.CompositeModel`). This is the middle ground: reuse a native brick for one
   slot and write the other one as formulas.

These three objects are compositions of generic bricks. The core stays agnostic to the
scenario: it does not name any physical case (diocotron, Euler-Poisson, two-fluid...); those are
compositions on the application side. For the details of the numerical methods (MUSCL/WENO reconstruction,
Riemann flux, SSPRK/IMEX integrators, multigrid Poisson), see
[ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md). For the layered architecture (model / mesh / dispatch /
integrator), see [ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md).

For one specialized family of models -- velocity-moment hierarchies (kinetic and QMOM
closures) -- you do not write the formulas by hand: a generator builds the DSL model from a
single closure. See [moments and closures](../concepts/moments-and-closures.md), the
[moment models reference](../reference/moment-models.md), and the
[HyQMOM tutorial](../tutorials/moment-model-hyqmom15.md).

## PhysicalModel: the concept

All bricks satisfy the same C++ contract, the `pops::PhysicalModel` concept
([include/pops/core/model/physical_model.hpp](https://github.com/wolf75222/adc_cpp/blob/master/include/pops/core/model/physical_model.hpp)). A
`PhysicalModel` describes an equation as a set of pure functions of pointwise states, nothing
more. It is the only "what to compute" axis of the architecture, separate from the "where / how to iterate"
axis (mesh + dispatch) and from the "in what order" axis (integrator + coupler, cf.
[ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md)).

The minimal contract requires four functions:

- `flux(U, aux, dir)`: the physical flux in the direction `dir` (0 = x, 1 = y);
- `max_wave_speed(U, aux, dir)`: the largest wave speed (for the CFL and the Riemann
  solver);
- `source(U, aux)`: the pointwise source term;
- `elliptic_rhs(U)`: the right-hand side of the elliptic equation (charge / mass density depending on
  the model).

Unification point: `flux` and `source` receive `aux` (the `pops::Aux` channel: potential `phi`,
gradient `grad_x`/`grad_y`, and optional extended fields `B_z`, `T_e`). This is what places under a
single spatial operator the drift transport (the `aux` is read in the flux) and the self-gravitating
compressible fluid (the `aux` is read in the source).

A complete hyperbolic brick additionally satisfies `pops::HyperbolicPhysicalModel`: it carries
the variables (conservative and primitive) and the conversions `to_primitive` / `to_conservative`,
because variables, conversions and flux are physically linked (a flux is written for a given
variable layout). This is the brick that you write, native or DSL.

## Model with bricks (native composition)

`pops.Model(state, transport, source, elliptic)` composes a model from four generic
bricks that are already compiled and returns a `ModelSpec` (tags read on the C++ side by the model
factory). Python composes the objects; the cell-by-cell computation stays compiled C++ (no numpy,
GPU/MPI preserved). The available bricks, as exposed by `pops.*` (and their C++ structs in
[include/pops/physics/](https://github.com/wolf75222/adc_cpp/blob/master/include/pops/physics/)):

**State** (`state=`)
- `pops.Scalar()`: scalar state (1 variable, e.g. a transported density).
- `pops.FluidState(kind="compressible", gamma=1.4)`: compressible Euler (the index `gamma`).
- `pops.FluidState(kind="isothermal", cs2=0.5)`: isothermal Euler (the sound speed `cs2`).

**Transport** (`transport=`)
- `pops.ExB(B0=1.0)`: scalar advection by the ExB drift (magnetic field `B0`),
  `pops::ExBVelocity` in `physics/hyperbolic.hpp`.
- `pops.CompressibleFlux()`: compressible Euler flux (`gamma` comes from the state),
  `pops::CompressibleFlux` (alias of `pops::Euler`).
- `pops.IsothermalFlux()`: isothermal Euler flux (`cs2` comes from the state), `pops::IsothermalFlux`.

**Source** (`source=`)
- `pops.NoSource()`: no source, `pops::NoSource` in `physics/source.hpp`.
- `pops.PotentialForce(charge=1.0)`: potential force `(q/m) rho E` on the momentum
  (plus work if 4 variables), `pops::PotentialForce`.
- `pops.GravityForce()`: gravitational force `rho g`, `pops::GravityForce`.

**Elliptic right-hand side** (`elliptic=`)
- `pops.ChargeDensity(charge=1.0)`: charge density `f = q n`, `pops::ChargeDensity` in
  `physics/elliptic.hpp`.
- `pops.BackgroundDensity(alpha=1.0, n0=0.0)`: neutralizing background `f = alpha (n - n0)`,
  `pops::BackgroundDensity`.
- `pops.GravityCoupling(sign=1.0, four_pi_G=1.0, rho0=1.0)`: self-consistent coupling
  `f = sign * 4piG (rho - rho0)` (`sign = +1` gravity, `-1` plasma), `pops::GravityCoupling`.

`pops.Model(...)` validates the state <-> transport consistency (Scalar with ExB; compressible FluidState
with CompressibleFlux; isothermal with IsothermalFlux): an inconsistent pairing raises an
immediate `ValueError`.

Example, the reduced diocotron model (scalar density advected by ExB, neutralizing background), as
used in the tutorial for the uniform/AMR comparison:

```python
import pops

model = pops.Model(
    state=pops.Scalar(),
    transport=pops.ExB(B0=1.0),
    source=pops.NoSource(),
    elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0),
)

sim = pops.System(n=96, L=1.0, periodic=True)
sim.add_block("ne", model=model, spatial=pops.Spatial(minmod=True), time=pops.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_density("ne", ne0)          # ne0 : tableau 2D (densite initiale)
sim.step_cfl(0.4)
```

The same `ModelSpec` also plugs into `pops.AmrSystem` (adaptive refinement) without changing the
model: `sa.add_block("ne", model=model, ...)`.

## DSL model (written as formulas)

`pops.physics.facade.Model` lets you write a model as symbolic formulas: Python composes an expression
tree (the operators `+`, `-`, `*`, `/`, `**`, `pops.ir.ops.sqrt` build the tree, not a
function called per cell), which the DSL translates into compilable C++. You declare the conservative
variables, the primitives (via formulas), the flux, the eigenvalues, the source and the
elliptic contribution, then you compile.

Here is the reduced diocotron model of the canonical tutorial
([docs/sphinx/tutorials/diocotron_tutorial.py](https://github.com/wolf75222/adc_cpp/blob/master/docs/sphinx/tutorials/diocotron_tutorial.py)), written as
formulas; it reproduces exactly the native bricks `ExBVelocity` (transport) and
`BackgroundDensity` (elliptic):

```python
import pops
from pops.numerics.riemann import Rusanov
from pops.numerics.reconstruction.limiters import Minmod

B0 = 1.0      # champ magnetique de fond (porte la derive E x B)
ALPHA = 1.0   # facteur du second membre elliptique alpha (n - n_i0)

def diocotron_model(n_i0):
    m = pops.physics.facade.Model("diocotron_tutorial")

    (n,) = m.conservative_vars("n")     # unique variable conservative : la densite (role Density)
    m.aux("phi")                        # champs auxiliaires fournis par le solveur (canal pops::Aux)
    grad_x = m.aux("grad_x")
    grad_y = m.aux("grad_y")

    vx = (-grad_y) / B0                  # derive E x B : v = (-d_y phi / B0, d_x phi / B0)
    vy = grad_x / B0
    m.flux(x=[n * vx], y=[n * vy])       # flux d'advection f = n v(dir)
    m.eigenvalues(x=[vx], y=[vy])        # spectre : une onde, la vitesse de derive

    m.primitive_vars(n=n)                # scalaire transporte : primitif = conservatif
    m.conservative_from([n])
    m.elliptic_rhs(ALPHA * (n - n_i0))   # couple le bloc au Poisson : rhs = alpha (n - n_i0)

    m.check()                            # toute variable referencee doit etre declaree
    return m

compiled = diocotron_model(n_i0).compile(backend="production")   # -> CompiledModel

sim = pops.System(n=96, L=1.0, periodic=True)
sim.add_equation("ne", model=compiled,
                 spatial=pops.FiniteVolume(limiter=Minmod(), riemann=Rusanov()),
                 time=pops.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_density("ne", ne0)
sim.step_cfl(0.4)
```

DSL details and points to watch (parameters named `m.param`, physical roles,
`require_metadata`, `.so` cache): see the short reference [DSL_API.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_API.md) and the
design [DSL_MODEL_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_MODEL_DESIGN.md).

## Hybrid model (native brick + DSL within a single model)

`pops.Model(...)` composes 100% native bricks; `pops.physics.facade.Model(...)` generates a 100%
DSL model. `pops.CompositeModel(transport, source, elliptic)` fills the middle ground: mixing, within a single
model, native bricks (`pops.ExB`, `pops.PotentialForce`, `pops.ChargeDensity`...) and partial compiled
DSL bricks (`pops.physics.bricks.HyperbolicBrick`, `pops.physics.bricks.SourceBrick`, `pops.physics.bricks.EllipticBrick`
followed by `.compile()`).

Each slot accepts either a native brick or a partial compiled DSL brick. At least one
slot must be DSL: an all-native composition is written with `pops.Model(...)`, otherwise
`CompositeModel` raises a `ValueError`. The mix is compiled into a composite `.so` (prototype:
`aot` backend), on the same production path as a complete DSL model; the native numerics are
reused identically (a derived struct bakes the native parameters `qom`, `q`, `cs2`... into the
type; no re-derivation). The transport slot fixes the layout (`n_vars`, conservative names,
primitives, gamma); a DSL source / elliptic brick must declare the same `n_vars`.

Example, isothermal DSL transport + native source + native elliptic (excerpt from
`python/tests/test_dsl_hybrid.py`):

```python
import pops
from pops.numerics.riemann import Rusanov
from pops.numerics.reconstruction.limiters import Minmod

CS2, QOM, Q = 0.7, -1.0, -1.0

# Brique hyperbolique DSL repliquant pops::IsothermalFlux{cs2} (3 variables).
def build_iso_transport(cs2):
    b = pops.physics.bricks.HyperbolicBrick("iso")
    rho, rho_u, rho_v = b.conservative_vars("rho", "rho_u", "rho_v")
    u = b.primitive("u", rho_u / rho)
    v = b.primitive("v", rho_v / rho)
    c = pops.ir.ops.sqrt(cs2)
    b.flux(x=[rho_u, rho_u * u + cs2 * rho, rho_v * u],
           y=[rho_v, rho_u * v, rho_v * v + cs2 * rho])
    b.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    b.primitive_vars(rho, u, v)
    b.conservative_from([rho, rho * u, rho * v])
    return b

m = pops.CompositeModel(
    transport=build_iso_transport(CS2).compile(),  # transport DSL
    source=pops.PotentialForce(charge=QOM),         # source native
    elliptic=pops.ChargeDensity(charge=Q),          # elliptique native
)
compiled = m.compile(backend="aot")                # -> CompiledModel (adder add_compiled_block)

sim = pops.System(n=48, L=1.0, periodic=True)
sim.add_equation("gas", compiled,
                 spatial=pops.FiniteVolume(limiter=Minmod(), riemann=Rusanov()),
                 names=["rho", "rho_u", "rho_v"])
```

The mix works both ways (native transport + DSL source/elliptic too). Source:
[DSL_MODEL_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_MODEL_DESIGN.md) section "composition hybride" and
`python/tests/test_dsl_hybrid.py`.

## Conservative / primitive variables

A DSL model distinguishes two sets of variables, with physical roles that let the system
find a quantity by its meaning (and not by a literal index), essential to inter-species couplings.

- `m.conservative_vars("rho", "mx", "my", roles=["Density", "MomentumX", "MomentumY"])` declares the
  conservative variables (the evolved state `U`) and returns a tuple of `Var` to unpack. The `roles=`
  is optional; without it, a canonical name -> role mapping applies (`rho`/`n` -> `Density`,
  `rho_u` -> `MomentumX`, `E` -> `Energy`...). An unrecognized name stays `Custom`.
- `m.primitive(name, expr)` defines a primitive by its formula (as a function of the conservatives or
  of the previous primitives), e.g. `u = m.primitive("u", mx / rho)`.
- `m.primitive_vars(rho=rho, ux=mx/rho, ...)` (kwargs form) defines each primitive and fixes the
  ordered layout of `Prim` (the order of the kwargs). The positional form
  `m.primitive_vars(rho, u, v, p)` just fixes the layout from already-defined names.
- `m.conservative_from([rho, rho*u, rho*v])` gives the inverse `Prim -> U` (the DSL does not know how to invert
  the primitives symbolically; you provide the inverse explicitly). It generates `to_conservative`.

The spatial operator can then reconstruct in primitive variables (`rho`, `u`, `p`) rather than
conservative, more stable for Euler (positivity of `rho` and `p`); see the
`variables=Primitive()` choice of `pops.FiniteVolume` and the details in [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md).

## Flux, sources, eigenvalues, elliptic RHS

These four declarators are the core of the DSL model; they correspond one to one to the functions of the
`pops::PhysicalModel` concept read by the core.

- `m.flux(x=[...], y=[...])`: the **physical flux** `F(U, aux, dir)`, one expression per conservative
  component and per direction. The spatial operator evaluates it at the interfaces then passes it to the
  Riemann solver (Rusanov / HLLC / Roe according to `riemann=`). Not to be confused with
  `m.eval_flux(U, aux, dir)`, which is the numpy evaluator (debug / host proto), nor with the numerical
  flux `riemann=` of `pops.FiniteVolume`.
- `m.eigenvalues(x=[...], y=[...])`: the **eigenvalues** (characteristic speeds) per
  direction. The core derives `max_wave_speed` from it (Rusanov bound and CFL time step); if a
  primitive `p` (pressure) is declared, the generated brick also exposes `pressure` / `wave_speeds`,
  which makes it compatible with the HLLC / Roe fluxes (which require a pressure).
- `m.source([...])`: the **source term** `S(U, aux)`, one expression per component (optional). It
  reads the exterior state through the `pops::Aux` channel (e.g. `grad_x` / `grad_y` for a potential
  force `-rho grad phi`).
- `m.elliptic_rhs(expr)`: the **contribution to the elliptic right-hand side**, which couples the block to
  the system Poisson (charge density `q n`, background `alpha (n - n0)`, gravity...). The system
  Poisson sums the contributions of all the blocks.

`m.check()` verifies that every referenced variable (in the primitives, the flux, the eigenvalues,
the source, the elliptic) is indeed declared (conservative / primitive / aux), and raises a
factual `ValueError` otherwise. For the physical meaning and the discretization of each operator
(reconstruction, Riemann, multigrid), see [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md).

## Compilation: production / AOT / prototype

`m.compile(backend=..., target=...)` translates the symbolic model into a `.so` and returns a
`CompiledModel` (which carries `so_path`, `backend`, the `adder` to use, the names/roles/gamma/n_aux,
the `abi_key` and the `model_hash`). The `.so` is cached by `model_hash`: an unchanged model
is not recompiled. The default is `backend="auto"`, which auto-selects `production` under
toolchain parity with the installed `_pops`, otherwise falls back to `aot`. The explicit values
`prototype | aot | production` are still available and short-circuit this policy.

Three backends, materialized on the code side in `_BACKEND_CAPS` (`python/pops/dsl.py`):

| backend | CPU | MPI | AMR | GPU | role |
|---|---|---|---|---|---|
| `production` | yes | yes (np=1/2/4) | via `AmrSystem` | reports `False` on the Python side | recommended in MPI/AMR ; native zero-copy loader (`add_native_block`) |
| `aot` | yes | no | no | no | `auto` fallback ; `.so` with marshaling, mono-rank, CPU debug/bench. Carries the runtime params (`set_block_params`) |
| `prototype` | yes (Rusanov o1) | no | no | no | JIT prototype, host virtual dispatch ; do not use in production |

`_BACKEND_CAPS["production"]` declares `{cpu, mpi, amr} = True`. The native `production` path shares
the engine of `add_block` (`fill_boundary` halos, hence MPI-capable by construction) and has an AMR
counterpart (`m.compile(backend="production", target="amr_system")` -> `AmrSystem.add_native_block`). `gpu`
is reported `False` out of caution: the native path is device-clean in C++ (validated on GH200), but the
end-to-end validation from Python on a module built with Kokkos/CUDA remains a dedicated step; the
host module tested in CI is not built for GPU.

These capabilities are diagnostic flags, checked at plugging time (`add_equation`) or at
runtime, not fixed like a `device=` compilation argument (a `.so` can compile without
the host module being device-capable). The safeguards raise a `ValueError` as early as possible:

- unknown backend (other than `prototype`/`aot`/`production`);
- `target="amr_system"` with a backend other than `production` (no AMR `.so` path outside native);
- `compile(backend="prototype", require_metadata=True)` (the JIT does not carry the useful
  metadata);
- on the plugging side: `riemann` HLLC/Roe without a declared pressure `p`, `names=` on the native
  `production` path (the names come from the `.so` metadata).

To plug the `CompiledModel`, `System.add_equation` routes by type: a `ModelSpec`
(`pops.Model(...)`) -> `add_block` (native); a `CompiledModel` -> the backend's adder
(`add_dynamic_block` for `prototype`, `add_compiled_block` for `aot`, `add_native_block` for
`production`). Full detail: [DSL_API.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_API.md) and
[DSL_MODEL_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_MODEL_DESIGN.md). Coverage of the backends on GPU/MPI/AMR:
[BACKEND_COVERAGE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md).
