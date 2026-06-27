# Spec 5 typed packages

This page is the reference for the typed-object packages added in Spec 5. Each is a small,
focused namespace of descriptors: values that name a brick, an algorithm, a layout, or a
policy and carry its metadata, and that lower to the IR the codegen and runtime consume. They
compute nothing in Python. For the principle behind them (Python describes, C++ executes; no
string algorithm selectors) see {doc}`../concepts/typed-api`.

```{note}
`pops.solvers` and `pops.linalg` are forthcoming; the linear solvers and preconditioners still
live under `pops.lib` (see {doc}`typed-bricks` and {doc}`custom-solvers`) pending their
relocation.
```

## `pops.numerics`: fluxes, reconstruction, limiters

`riemann` (Rusanov / HLL / HLLC / Roe / User), `reconstruction` (FirstOrder / MUSCL / WENO5 /
WENO5Z / User), the `limiters` it re-exports (Minmod / VanLeer / MC / Superbee), `terms` (Flux /
LocalTerm / SourceTerm), and `projections` (positivity / bound_preserving / ...). A Riemann
descriptor carries the native C++ id and the model capabilities it needs; see
{doc}`native-numerics` for the native-brick table.

```python
from pops.numerics.riemann import HLL, HLLC
from pops.numerics.reconstruction import MUSCL, WENO5Z, limiters

HLL().native_id              # 'pops::HLLFlux'
HLL().requirements           # {'capabilities': ['physical_flux', 'wave_speeds']}
rec = MUSCL(limiter=limiters.Minmod())   # typed limiter, not limiter="minmod"
WENO5Z().native_id           # 'pops::Weno5'
```

## `pops.fields`: elliptic field problems

A `FieldProblem` describes a field solved each step from typed parts: an `unknown` (a string
the user names), an `equation`, `inputs` (`rhs.ChargeDensity`), `coefficients`
(`ScalarCoefficient` / `ReactionCoefficient`), `bcs` (`Dirichlet` / `Neumann` / `Periodic`,
optionally pinned to a face `XMin` / `XMax` / `YMin` / `YMax`), and a `nullspace`
(`ConstantNullspace`). `PoissonProblem`, `ScreenedPoissonProblem`, and
`AnisotropicPoissonProblem` are the specialized aliases.

```python
import pops.fields as F
from pops.fields.bcs import Dirichlet, Periodic, XMin
from pops.fields.rhs import ChargeDensity
from pops.fields.coefficients import ScalarCoefficient
from pops.fields.nullspace import ConstantNullspace

pb = F.PoissonProblem(
    name="phi",
    unknown="phi",                              # a string: the user names the field
    coefficients=(ScalarCoefficient("eps"),),
    inputs=(ChargeDensity(),),
    bcs=(Periodic(),),
    nullspace=ConstantNullspace(),
)
pb.inspect()["category"]                         # 'poisson_problem'
```

A face Dirichlet is `Dirichlet(value=0.0, on=XMin)`. See {doc}`../concepts/elliptic-rhs` and
{doc}`../concepts/poisson` for the concepts.

## `pops.mesh`: meshes, layouts, AMR, geometry

Meshes (`CartesianMesh` / `PolarMesh`), `layouts` (`Uniform` / `AMR`), `amr` (the refinement
criteria and policies), `geometry` (`Disc` / `HalfPlane` / `EmbeddedBoundary` / `LevelSet`),
`masks`, and `boundaries`. Refinement is authored with the `Refine` builder
(`Refine.on(subject).above(threshold)` / `.below(...)`); the regrid cadence, proper nesting,
patch layout, and output are typed policies, not loose keyword strings.

```python
from pops.mesh import CartesianMesh
from pops.mesh.layouts import AMR
from pops.mesh.amr import Refine, RegridEvery, ProperNesting

mesh = CartesianMesh(n=128, L=1.0, periodic=True)
amr = AMR(
    mesh,
    max_levels=2,
    ratio=2,
    refine=Refine.on("density").above(0.5),      # typed criterion
    regrid=RegridEvery(8),
    nesting=ProperNesting(buffer=1),
)
```

`NATIVE_RATIOS` and `NATIVE_MAX_LEVELS` expose what the native AMR actually wires, so a layout
that asks for more is rejected rather than silently clamped. For the concept see
{doc}`../concepts/amr`.

## `pops.params`: runtime parameters with constrained domains

`RuntimeParam` is a value set at run time; `ConstParam` is frozen at compile time;
`DerivedParam` is computed from an expression; `Constant` carries an optional unit. A
`RuntimeParam` domain is a typed `Constraint` -- `Positive()`, `NonNegative()`, `Range(lo, hi)`,
`In(*allowed)` -- validated rather than left implicit.

```python
from pops.params import RuntimeParam, ConstParam, Positive, Range

nu = RuntimeParam("nu", default=0.1, domain=Positive())   # "nu" is the user's name
cfl = RuntimeParam("cfl", default=0.4, domain=Range(0.0, 1.0))
gamma = ConstParam("gamma", 1.4)
```

## `pops.output`: output and checkpoint policies

`OutputPolicy` selects a typed `format` (`HDF5(parallel=...)` / `Plotfile()`), a `levels`
selector (`AllLevels()` / `CoarseOnly()` / `SelectedLevels(*levels)`), the `fields` and
`diagnostics` to write. `CheckpointPolicy` describes restart behavior.

```python
from pops.output import OutputPolicy, CheckpointPolicy, HDF5, SelectedLevels
from pops.diagnostics import mass, energy

out = OutputPolicy(
    format=HDF5(parallel=True),                  # typed format, not format="hdf5"
    levels=SelectedLevels(0, 1),
    fields=("density", "phi"),
    diagnostics=(mass(), energy()),
)
ckpt = CheckpointPolicy(restartable=True)
```

## `pops.external`: compiled brick references

A user ships a brick in a standalone `.so` whose `pops_brick_manifest()` exports its ids,
categories, requirements, and capabilities. `CompiledManifest` is the inert parsed manifest
(it neither dlopens nor registers anything); `CompiledBrickRef` references one brick by
`native_id` and can assert its `expect_category`. `read_manifest` / `register` /
`load_cpp_library` read and register manifests. See {doc}`typed-bricks` for the C++ side and
the static-dispatch path.

```python
import pops
from pops.external import CompiledManifest, CompiledBrickRef

man = CompiledManifest(
    bricks=[{"id": "my_hllc", "category": "riemann",
             "requirements": "physical_flux,wave_speeds"}],
    abi_key=pops.abi_key(),
)
ref = CompiledBrickRef(man, "my_hllc", expect_category="riemann")
ref.native_id                                    # 'my_hllc'
```

## `pops.diagnostics`: invariants and norms

Typed diagnostic descriptors: `mass`, `energy`, `momentum`, `norm(kind=...)`, `integral`,
`invariant_error`, `residual`. They name what to measure; the measurement runs in C++.

```python
from pops.diagnostics import mass, norm

mass()                                           # BrickDescriptor('mass', 'macro', ...)
norm(kind="l2")
```

## `pops.descriptors`: the descriptor protocol and capability matrix

The shared base for every typed brick. `DescriptorProtocol` is the structural protocol all
descriptors satisfy (`available`, `capabilities`, `requirements`, `options`, `lower`,
`inspect`, `validate`); `Descriptor` / `BrickDescriptor` are the concrete carriers.
`Availability` is the explicit answer to "does this combination work" (`Availability.yes()`,
`Availability.no(reason)`, `Availability.partial(reason)`), naming what is missing and the
alternatives. `reject_string_selector` is the guard that turns a string algorithm selector
into a typed-replacement error. `load_cpp_library` and `external` surface external bricks.

```python
from pops.descriptors import Availability, reject_string_selector

Availability.no("not wired on GPU")              # Availability('no', reason='not wired on GPU')

reject_string_selector("hll", "riemann", "pops.numerics.riemann.HLL()")
# TypeError: String algorithm selector rejected: riemann='hll'.
#            Use pops.numerics.riemann.HLL().
```

Every descriptor answers `inspect()` (its id / category / native id / requirements /
capabilities / availability), which is how a page or a tool reads what a brick wires without
re-listing combinations that can drift. The native-side capability matrix (`pops.capabilities`,
see {doc}`python-api`) is the single source of truth for what each facade / geometry / backend
supports.

## `pops.codegen.Optimization`: codegen knobs

`Optimization` carries the IR optimization passes (CSE, dead-node elimination, redundant-solve
elimination, local-op fusion, reciprocal hoisting) and a typed `math` mode from
`pops.codegen.math_options`: `StrictMath()` (IEEE), `FastMath()`, `DebugMath()`,
`GpuRegisterAware()`. The math mode is a typed object, not a `math="strict"` string.

```python
from pops.codegen import Optimization
from pops.codegen.math_options import StrictMath

opt = Optimization(math=StrictMath())            # typed math mode
opt = Optimization(cse=True, fuse_local_ops=True, math=StrictMath())
```
