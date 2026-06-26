> **STATUS: IMPLEMENTED.** This document is the original design spec. The condensed Schur source stage is delivered (`CondensedSchur`, `schur_condensation.hpp`, exposed in Python as `pops.CondensedSchur`; tests test_schur_via_system / test_schur_conservation). Read this file as design history, not as current state.

# Design: implicit source condensed by Schur (reproduction of arXiv:2510.11808)

DESIGN-ONLY, no implementation. This document is a reasoned SPEC, honest about its
limits, of a new numerical ABSTRACTION for `adc_cpp`: the Schur condensation of the
coupled implicit source (potential / velocity / Lorentz) of the scheme of Hoffart, Maier, Shadid,
Tomas (arXiv:2510.11808). It describes the target, the constraints and the sequencing; it does NOT
promise the reproduction of the paper (see section 9, the roadmap there is deliberately
cautious).

The document relies on the architecture already in place (sources read):
- `docs/ARCHITECTURE.md` sections 1 (five orthogonal layers), 6 (generic physical model
  `CompositeModel`), 7 (elliptic stage: `EllipticProblem` / `EllipticOperator` /
  `LinearSolver` / `FieldPostProcess`), 8 (distributed AMR);
- `docs/ALGORITHMS.md` (FV bricks, elliptic operator, cut-cell Shortley-Weller);
- `docs/PAPER_ROADMAP.md` (Hoffart reproduction status, Cartesian ring-edge blocker);
- `docs/DSL_MODEL_DESIGN.md` (`dsl.Model` facade, physical roles, `add_native_block`);
- `docs/GPU_RUNTIME_PORT.md` (device-clean harness: named functors, no extended lambda);
- the MERGED polar Phase 1 (#116, commit `004efca`): the MESH abstraction
  (`pops.CartesianMesh` / `pops.PolarMesh` -> `System(mesh=)`), with `pops.FiniteVolume` = recon +
  Riemann + variables ONLY (no geometry argument);
- `include/pops/core/state/variables.hpp` (`VariableRole`: `Density`, `MomentumX`, `MomentumY`,
  `MomentumZ`, `Energy`, ...);
- `docs/BIBLIOGRAPHY.md` section 3 (Hoffart entry).

PRELIMINARY HONESTY NOTE. What follows describes a TARGET. Nothing is delivered. The scientific
blocker of the diocotron (Cartesian ring edge, `docs/PAPER_ROADMAP.md` basket 3) is
ORTHOGONAL to this work item: the Schur condensation is a TIME scheme for the coupled
source, it does NOT correct the spatial diffusion of the transport. The two can land
independently.


## 1. The splitting of the paper

Hoffart et al. integrate the drift two-fluid system by SEPARATING three operators, which
is exactly the style of the `TimeIntegrator` of layer 5 (`docs/ARCHITECTURE.md` section 1:
splitting / IMEX / AP composing operators without knowing their interior):

1. **Hyperbolic transport**: the conservative advection of the fluid fields (density, momentum,
   energy where applicable). This is exactly the existing FV path
   (`numerics/spatial_operator.hpp`, reconstruction + numerical flux), explicit, SSPRK2/SSPRK3.
2. **Coupled implicit source**: the potential / velocity / Lorentz-force subsystem, treated
   IMPLICITLY (theta-scheme) because the cyclotron rotation and the electrostatic restoring are
   stiff in the drift limit.
3. **Schur condensation**: the velocity of the implicit subsystem is ELIMINATED algebraically, which
   reduces the source stage to a single MODIFIED Poisson-type elliptic solve for the
   potential, followed by an explicit reconstruction of the velocity. This is the key to the cost: one elliptic
   solve per source stage instead of a dense coupled inversion.

The split (1) explicit + (2)-(3) implicit is an IMEX in the sense of the time layer. The
contribution of this document is the (2)-(3) stage: the condensation, and above all the MODIFIED
elliptic operator that it produces.


## 2. The equations of the source

Source subsystem (transport frozen), with `rho` the density, `v` the velocity, `phi` the potential,
`Omega` the cyclotron vector (carried by `B_z` out of plane), `alpha` a coupling constant:

```
d_t rho = 0
d_t v   = -grad(phi) + v x Omega
d_t(-Lap phi) = -alpha div(rho v)
```

The density is frozen in the source (all its transport is in stage (1)). The velocity undergoes
the electrostatic restoring `-grad(phi)` and the Lorentz rotation `v x Omega`. The last
equation is the Poisson constraint differentiated in time (the charge evolves through the divergence
of the current `rho v`).

### 2.1 Theta-scheme and condensation

We discretize in time by a theta-scheme (`theta in [0,1]`, `theta=1/2` = Crank-Nicolson). We
denote by `B` the LOCAL operator (point by point) of implicit rotation:

```
B v = v - theta dt (v x Omega)
```

`B` is a 2x2 matrix per cell (plane (x,y)), invertible as soon as `theta dt |Omega|` is finite;
its inverse `B^{-1}` is CLOSED (2x2 rotation-dilation). The implicit velocity then writes

```
v^{n+theta} = B^{-1} (v^n - theta dt grad phi^{n+theta})
```

By injecting this expression into the differentiated Poisson constraint, we ELIMINATE `v` (this is
the Schur complement of the velocity block) and we obtain an equation for the SOLE `phi^{n+theta}`:

```
-Lap phi^{n+theta}
  - theta^2 dt^2 alpha div( rho^n B^{-1} grad phi^{n+theta} )
  = -Lap phi^n
  - theta dt alpha div( rho^n B^{-1} v^n )
```

then we RECONSTRUCT the velocity:

```
v^{n+theta} = B^{-1} ( v^n - theta dt grad phi^{n+theta} )
```

and, where applicable, we extrapolate the full state `U^{n+theta}` (and the energy) before filling the
ghosts.


## 3. KEY POINT: why `source(U, aux)` does not suffice

The design mistake to avoid is to treat this stage as one more LOCAL source, of the type
`source(U, aux)` (the signature of the current source bricks, `docs/ARCHITECTURE.md` section 6:
a source sees only one cell and its `Aux` fields). This is IMPOSSIBLE here, for a reason
STRUCTURAL and not of ergonomics:

> The Schur condensation MODIFIES the elliptic operator itself.

The condensed equation is not `-Lap phi = f(rho, v)` (a standard Poisson with a recomputed
right-hand side, which the current coupling already knows how to do via `elliptic_rhs(U)`). It is

```
L_schur(phi) = -Lap phi - theta^2 dt^2 alpha div( A grad phi )    avec A = rho B^{-1}
```

that is an operator `-div(A grad phi) + (Laplacian term)` whose TENSORIAL COEFFICIENT `A`
2x2:
- depends on the state (`rho^n` and `Omega` per cell), so CHANGES at every time step;
- is in general NON SYMMETRIC: `B^{-1}` is a rotation-dilation (nonzero antisymmetric part as
  soon as `Omega != 0`), so `A = rho B^{-1}` has an antisymmetric part. The operator
  `-div(A grad .)` is therefore NOT self-adjoint in general.

A local source cannot express this: `source(U, aux)` produces a contribution PER
CELL to the residual, whereas `div(A grad phi)` is a GLOBAL OPERATOR (stencil coupling the
neighbors, to be inverted). The term belongs to the LEFT-hand side of the elliptic stage, not to the
right-hand side. Concretely:
- the current `elliptic_rhs(U)` (`physics/elliptic.hpp`, Poisson right-hand side, layer 5 side)
  touches only the RHS: it CANNOT inject a term into the OPERATOR;
- `poisson_operator.hpp` (the canonical 5-point Laplacian, `docs/ARCHITECTURE.md` section 7) is
  a SCALAR constant-coefficient Laplacian: it has no room for a tensorial coefficient
  `A(x)` per cell.

So the Schur source stage requires (a) a NEW tensorial elliptic operator, and (b) a
local-global scheme that BUILDS it from the state, SOLVES it, then reconstructs the velocity. This
is neither a local source nor a standard Poisson: it is a numerical abstraction in its own
right.


## 4. KEY POINT: do NOT code a `DiocotronSchurSolver` in the core

The symmetric temptation is to bake a solver named `DiocotronSchurSolver` into the core. This is
contrary to the guiding principle of `adc_cpp` (`docs/ARCHITECTURE.md`: "the core is AGNOSTIC to the
model, it names no scenario"). The correct separation is:

- **Model = PHYSICS.** A model declares its variables and their ROLES (`VariableRole`:
  `Density`, `MomentumX`, `MomentumY`, `Energy`, ...), its flux, its LOCAL source, its elliptic right-hand
  side, its parameters. It knows nothing of Schur or the theta-scheme.
- **SchurCondensation = local-global numerical ALGORITHM.** It is a policy of the numerical
  + time layer (layers 2 and 5): from a state that EXPOSES the right roles, it
  assembles the tensorial operator, solves it, reconstructs the velocity. It names no
  scenario.

First concrete implementation: **`ElectrostaticLorentzCondensation`**, GENERIC over any
fluid species that exposes the roles
- `Density`, `MomentumX`, `MomentumY` (and `Energy` optionally),
- plus access to `phi`, `grad phi`, the field `B_z` / `Omega`, and the constant `alpha`.

It works for a model written in C++ BRICKS (`CompositeModel`, `add_block`) OR in compiled DSL
(`dsl.Model` -> `add_native_block`), provided the ROLES are present. The drift diocotron
is only ONE client; a model with two magnetized species would be another, without an extra line of
Schur code. This is the exact inverse of a scenario-named solver.

MINIMAL CONTRACT required of the model by `ElectrostaticLorentzCondensation`:
1. a `Density` role (read of `rho^n`);
2. the `MomentumX`/`MomentumY` roles (or `VelocityX`/`VelocityY`) to read/reconstruct `v`;
3. an aux field `B_z` (or `Omega`) already populated (`Aux` channel, like `set_magnetic_field`
   today);
4. a potential `phi` and an access `grad phi` (already provided by the coupled elliptic stage);
5. the coupling constant `alpha`.
Any model that satisfies this contract is eligible, whether it comes from the bricks or from the DSL.


## 5. Target C++ architecture (descriptive, NO implementation)

Five levels, aligned on the five existing layers. Each level is device-callable where it
runs in the hot loop, with no allocation, no `std::function`, no dynamic polymorphism
(`docs/ARCHITECTURE.md` section 1; device-clean harness of `docs/GPU_RUNTIME_PORT.md`).

### Level 1: `TensorEllipticOperator` (numerical layer)

The discrete operator `L(phi) = -div(A grad phi) + kappa phi`, with `A` a 2x2 TENSORIAL coefficient
per cell (potentially non symmetric) and `kappa` an optional scalar mass term. It
GENERALIZES `poisson_operator.hpp` (scalar 5-point Laplacian): same role of shared `OperatorSpec`
(`apply`, `residual`, restriction/prolongation for the MG), but with a tensorial face-flux
stencil. The coefficients `A` live in a `MultiFab` (component by component of the 2x2,
or compact storage of the 4 entries), restricted by `average_down` for the coarse MG levels, like `eps(x)`
today (`set_epsilon(eps_fine)`, `docs/ARCHITECTURE.md` section 7). The case
`A = I`, `kappa = 0` must fall back EXACTLY (bit-identical) on the canonical Laplacian: this is the
non-regression guard.

### Level 2: the solver, and the QUESTION of non-symmetry (TECHNICAL NOTE TO DECIDE)

The linear solver depends on the symmetry of `A`, which depends on the PHYSICS:

- **`A` symmetric** when the antisymmetric part of `B^{-1}` vanishes, i.e. `Omega = 0`
  (no magnetic field: we fall back on a Poisson with coefficient `rho`, symmetric positive
  definite if `rho > 0`). The constant `B_z` case is NOT enough on its own: `B^{-1}` remains a
  rotation as soon as `Omega != 0`. The truly symmetric case is therefore `Omega = 0`. The case
  constant `B_z` AND constant `rho` gives a CONSTANT-coefficient operator but still NON
  symmetric (pure rotation): it is just easier to precondition, not symmetric.
- **`A` non symmetric** as soon as `Omega != 0` (magnetized diocotron case), and a fortiori when
  `rho` varies in space (variable coefficient AND non symmetric).

Consequence: the current `GeometricMG` (V-cycle, red-black Gauss-Seidel smoother) assumes a
symmetric positive definite operator. For `A` non symmetric, the MG V-cycle alone is NOT a
reliable solver. TARGET direction (to be CONFIRMED, this is the technical point to decide before
the implementation of level 2):

> A non-symmetric KRYLOV solver (GMRES or BiCGStab) PRECONDITIONED by a V-cycle MG
> applied to the SYMMETRIC PART of the operator (the `-div(rho B^{-1}_sym grad .)` symmetrized, or
> simply the `rho`-weighted scalar Poisson).

This lead is PLAUSIBLE (the antisymmetric term is in `theta^2 dt^2 alpha`, so small at a reasonable
source CFL: the symmetric preconditioner should capture the essentials). But it remains
TO VALIDATE: at this stage we do not know (a) the real number of Krylov iterations, (b) the robustness of
the MG-on-symmetric-part preconditioning when `theta dt |Omega|` grows, (c) whether a simple
2x2 block Jacobi preconditioner would suffice for the targeted regimes. No GMRES /
BiCGStab brick exists today in `numerics/elliptic/` (the `EllipticSolver` concept has only MG and
FFT). It is therefore an ADDITION, behind the existing `EllipticSolver` concept (`rhs`/`phi`/`solve`/
`residual`/`geom`), so as to break no caller. FLAG: decide (Krylov + preconditioner) in
PR3 before any facade wiring.

### Level 3: `LocalLorentzEliminator` (numerical layer, device-callable)

The LOCAL core: per cell, build `B = I - theta dt [Omega]_x` (the 2x2 matrix of implicit
rotation) and its inverse `B^{-1}` in closed form (no solve, no allocation). Device-callable
(`POPS_HD`), NAMED functor (no extended lambda, cf. `docs/GPU_RUNTIME_PORT.md`: the nvcc limit
on cross-TU first-instantiated extended lambdas is worked around by named functors, harness
already validated on GH200 for transport #64 and elliptic #97). It serves two places: (a) assemble
the coefficient `A = rho B^{-1}` of level 1, (b) reconstruct `v^{n+theta}` at level 4. No
matrix is materialized globally: `B^{-1}` is recomputed on the fly per cell.

### Level 4: `CondensedSchurSourceStepper` (time / coupling layer)

The orchestrator of the source stage, played ONCE per IMEX stage (frequency carried by
`CouplingPolicy`, `PerStage` or `OncePerStep`, `docs/ARCHITECTURE.md` section 7). Sequence:
1. **Build the operator**: fill the coefficient `MultiFab` `A = rho^n B^{-1}` (level 3
   per cell); configure `L_schur` (level 1) with `theta`, `dt`, `alpha`.
2. **Build the RHS**: `-Lap phi^n - theta dt alpha div( rho^n B^{-1} v^n )` (discrete
   divergence of a tensorial face flux; reuses the existing face assembly).
3. **Solve `phi^{n+theta}`**: call the level 2 solver behind the `EllipticSolver` concept.
4. **Reconstruct `U^{n+theta}`**: `v^{n+theta} = B^{-1}(v^n - theta dt grad phi^{n+theta})`
   (level 3), then recompose momentum/energy according to the model roles.
5. **Extrapolate**: pass from the state `n+theta` to `n+1` according to the theta-scheme (linear
   extrapolation; `phi^{n+1}`, `v^{n+1}` deduced).
6. **Energy update**: if the model carries an `Energy` role, recompute the energy coherent
   with the new velocity (work of the electrostatic force).
7. **Fill the ghosts** (`fill_boundary`, MPI halos) before handing back to the transport stage.

The stepper names no scenario: it reads the roles, calls levels 1-3, and delegates the
solve to the `EllipticSolver` concept. It is the one that materializes the local-global (local assembly of
the coefficient -> global solve -> local reconstruction).

### Level 5: DSL exposure (facade layer)

At the start, the DSL has NOTHING to add: the condensation is selected by the PRESENCE of the
required roles (`Density`/`MomentumX`/`MomentumY` + `phi`/`B_z`/`alpha`) and by the choice of the integrator
on the facade side (section 6). An existing DSL model (`dsl.Model`) that declares these roles is
directly eligible via `add_native_block` (native zero-copy path, GPU/MPI, `docs/
DSL_MODEL_DESIGN.md`). ONLY LATER, an explicit declarator `m.implicit_coupling(...)`
(naming `phi`, `v`, `Omega`, `alpha`, `theta`) could make the intention readable in the
formula and raise errors as early as possible; this is NOT required for the first client (roles
suffice) and stays deferred.


## 6. Target Python API

The entry point is an EXPLICIT/IMPLICIT explicit splitting, which composes a hyperbolic
integrator and a condensed source stage:

```python
sim.add_equation(
    "ions",
    model=model,                       # briques natives OU CompiledModel DSL, roles requis
    time=pops.Split(
        hyperbolic=pops.Explicit(ssprk3=True),
        source=pops.CondensedSchur(
            kind="electrostatic_lorentz",
            theta=0.5,
            density="rho",             # role Density
            momentum=("mx", "my"),     # roles MomentumX / MomentumY
            energy="E",                # role Energy (optionnel)
            magnetic_field="B_z",      # champ aux Omega / B_z
            potential="phi",
        ),
    ),
)
```

Principles of this API:
- `pops.Split(hyperbolic=, source=)` is a new integrator of the time layer: it plays the
  explicit hyperbolic stage then the `CondensedSchur` source stage (IMEX in the layer-5 sense).
- `pops.CondensedSchur(kind=, theta=, ...)` NAMES the algorithm and MAPS the fields onto the roles.
  `kind="electrostatic_lorentz"` selects `ElectrostaticLorentzCondensation` (level 4-3);
  other `kind` will be able to be added without touching the facade.
- **The default path is UNCHANGED.** Nothing breaks: `pops.Explicit`, `pops.IMEX`,
  `pops.Implicit`, `add_block`, `add_equation` continue to work identically. A model
  that does not use `pops.Split(... source=pops.CondensedSchur ...)` never sees the new stage.
  The selection is OPT-IN, like the polar grid (#116, Cartesian default bit-identical).

BOX -- `pops.SourceImplicit` (LOCAL) vs `pops.CondensedSchur` (GLOBAL). Two distinct mechanisms
treat a stiff source implicitly; do not confuse them.
- `pops.SourceImplicit` (= IMEX source-only) is LOCAL: the implicit couples only the components
  of the SAME cell (backward-Euler solved by Newton at the cell), NO spatial coupling.
  It is the right choice for purely local stiff terms (relaxation, reactions, friction):
  no elliptic solve, so much cheaper.
- `pops.CondensedSchur` (via `pops.Split`, cf. sections 2 to 5) is GLOBAL: it assembles the condensed
  tensorial elliptic operator and solves it by Krylov (BiCGStab), coupling the WHOLE domain.
  It is the right choice ONLY for a nonlocal stiff coupling (Lorentz / electrostatic, e.g.
  the magnetized Euler-Poisson of Hoffart). A local stiff source does not need Schur.


## 7. Geometry = MESH abstraction, not an option of `FiniteVolume`

STRUCTURAL CONSTRAINT, already laid by the MERGED polar Phase 1 (#116, commit `004efca`). The
CHOICE of geometry lives in a MESH object, never in the scheme:
- `pops.CartesianMesh(...)` / `pops.PolarMesh(...)` -> `pops.System(mesh=...)` carry the geometry
  (`SystemConfig` carries `geometry`, `nr`, `ntheta`, `r_min`, `r_max`).
- `pops.FiniteVolume(limiter=, riemann=, variables=)` stays reconstruction + numerical flux +
  variables ONLY. It HAS NO geometry argument, and never will.

For the Schur condensation, this requires that the `TensorEllipticOperator` (level 1) and its
face-flux assembly respect the GEOMETRY carried by the `Mesh` (Cartesian or
polar metrics). The divergence `div(A grad phi)` and the Lorentz inverse `B^{-1}` must be expressed
in the basis of the active geometry (in polar: the divergence weighted by the face radius
`(1/r) d_r(r F_r) + (1/r) d_theta F_theta`, like `assemble_rhs_polar` of #116; and the Lorentz
inverse in the local basis `(r, theta)`, like `ExBVelocityPolar`). The condensation is therefore
parameterized by the `Mesh`, not by `FiniteVolume`. This is coherent with the blocker of the
`docs/PAPER_ROADMAP.md` (basket 3): if one wants one day to combine Schur + polar grid for the
diocotron, the seam is already in the right place (the mesh), but these are two distinct work items.


## 8. GPU / MPI from the start, AMR afterwards

Non-negotiable constraint, aligned on the production state of the core
(`docs/DSL_MODEL_DESIGN.md` section 5, `docs/GPU_RUNTIME_PORT.md`):
- **NAMED functors**, no `__host__ __device__` cross-TU first-instantiated extended lambda
  (nvcc limit worked around, validated on GH200 for transport #64 and elliptic #97). Levels 1
  and 3 (tensorial operator, Lorentz eliminator) must be device-clean named functors
  from the writing.
- **Coefficients in `MultiFab`**: `A = rho B^{-1}` is a per-cell field (4 components of the
  2x2, or compact storage), not a scalar parameter; restricted by `average_down` for the MG,
  like `eps(x)`.
- **`local_size() == 0` safe, MPI-clean**: any per-cell post-processing keeps the ranks without
  a box safe (cf. the `solve_fields` MPI #99 fix: `fab(0)` without a guard segfaults on an empty
  rank). The coefficient assembly and the velocity reconstruction must be guarded.
- **No Python in the hot path**: the Python facade only CONFIGURES the stage (roles,
  `theta`, `alpha`); the loop is entirely C++ (`add_native_block` zero-copy path).
- **AMR only AFTER the uniform path**. The `TensorEllipticOperator` on an AMR hierarchy
  (restriction/prolongation of the tensorial coefficient, reflux coherent with the source stage) is a
  separate work item, NOT to be attempted before the uniform single-level is validated CPU + GPU + MPI.
  Known reservation: `AmrSystem` is not at parity with `System` (mono-block, explicit,
  `docs/ARCHITECTURE.md` section 8); wiring the condensation onto AMR will require this
  parity first.


## 9. PR sequencing (roadmap, HONEST about the horizon)

PR doc-only first, then a progressive ramp-up. This list is a TARGET, not a calendar
commitment; the late PRs are FAR and some depend on undecided questions (the
non-symmetric solver of level 2, the AMR). This document does NOT PROMISE the reproduction of the paper:
at best, it lays the numerical INFRASTRUCTURE of the condensed source stage, which is NECESSARY but
NOT SUFFICIENT (the Cartesian ring-edge blocker, `docs/PAPER_ROADMAP.md` basket 3, is
orthogonal and remains open).

- **PR0 (this document)**: the design SPEC. Doc-only.
- **PR1**: `TensorEllipticOperator` (level 1) in SCALAR generalized + non-regression bit-
  identical to the canonical Laplacian for `A = I`, `kappa = 0`. MMS order 2 on a constant SYMMETRIC
  tensorial coefficient (analytical test case, no physics). CPU serial.
- **PR2**: `LocalLorentzEliminator` (level 3), device-clean named functor, unit test
  `B B^{-1} = I` per cell. Assembly of the coefficient `A = rho B^{-1}` in `MultiFab`.
- **PR3**: the non-symmetric solver of level 2. This is the TECHNICAL POINT TO DECIDE (Krylov
  GMRES/BiCGStab + MG preconditioner on the symmetric part, or block Jacobi). Research
  PR: we measure the iterations and the robustness in `theta dt |Omega|` before freezing the
  choice. Highest risk of the sequence.
- **PR4**: `CondensedSchurSourceStepper` (level 4) + `pops.Split` / `pops.CondensedSchur` (Python
  API, section 6), native `add_native_block` path. Default unchanged. Validation: a manufactured
  case (MMS on the source subsystem alone, transport frozen); NOT yet the diocotron.
- **PR5**: GPU port (named functors already in place; validate Serial vs Cuda parity on GH200,
  like #97).
- **PR6**: MPI port (parity bit-invariant to the number of ranks, `local_size()==0` guarded, like
  #99/#93).
- **PR7**: optional DSL declarator `m.implicit_coupling(...)` (level 5, sugar + errors as early as
  possible). Non blocking.
- **PR8**: AMR (`TensorEllipticOperator` on a hierarchy, restriction of the coefficient, reflux).
  FAR, conditioned on the parity `AmrSystem` <-> `System` (`docs/ARCHITECTURE.md` section 8).

Known LIMITS to keep in mind (do NOT hide them):
- the non-symmetry of level 2 is a REAL unresolved numerical risk (PR3);
- none of these PRs treats the Cartesian ring edge: the QUANTITATIVE reproduction of the
  Hoffart diocotron rate will stay bounded by this blocker (`docs/PAPER_ROADMAP.md`) even with a perfect
  condensed source stage;
- the full MAGNETIZED two-fluid model (E x B + inhomogeneous diamagnetic coupling to the transport,
  beyond the drift limit) is still beyond this source stage (`docs/PAPER_ROADMAP.md`
  basket 4);
- the horizon is LONG: PR1-PR4 are the credible backbone in the medium term; PR5-PR8 depend on
  hardware validations (GH200) and transverse work items (AMR parity) that are not
  guaranteed on the horizon of the internship.
