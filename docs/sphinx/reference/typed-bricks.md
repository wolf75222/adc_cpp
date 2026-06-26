# Typed bricks (`pops.lib`)

`pops.lib` is a catalog of typed brick descriptors and IR macros. It computes nothing in
Python: a descriptor names a brick and carries its metadata (a native C++ id, a runtime
scheme string, requirements, capabilities); the codegen and runtime consume it.

## The four brick kinds

| Kind | Meaning | Example |
| --- | --- | --- |
| NativeBrick | a C++ type already in `include/pops` | `pops.lib.riemann.HLLC()` -> `pops::HLLCFlux` |
| GeneratedBrick | a DSL-authored brick compiled to C++ | a custom source / local matrix |
| MacroBrick | a Python function that builds Program IR | `pops.lib.time.predictor_corrector` |
| ExternalCppBrick | a user C++ brick registered by id | `pops.lib.riemann.User("my_hllc")` |

A descriptor carries `brick_type` (`native` / `generated` / `macro` / `external_cpp`),
`native_id`, `scheme`, `requirements`, `capabilities`, `options`, and `available`. A
catalogued brick with no native symbol yet is emitted with `available=False` and an empty
`native_id` -- never a fabricated id.

## Namespaces

`riemann` (Rusanov/HLL/HLLC/Roe/User), `reconstruction` (FirstOrder/MUSCL/WENO5/WENO5Z/User),
`limiters` (Minmod/VanLeer; MC/Superbee are catalogued but not yet wired), `spatial`
(FiniteVolumeResidual/FluxDivergence/SourceAssembly), `fields` (GeometricMG native; Poisson/
Helmholtz/EllipticSolve catalogued), `solvers` (CG/BiCGStab/GMRES/Richardson native free
functions; Newton/FixedPoint catalogued; Schur), `preconditioners` (GeometricMG native;
identity/jacobi/block_jacobi catalogued), `diagnostics`, `projections` (positivity native),
`invariants`, and `time` (macros forwarding to `pops.time` `std`).

```python
import pops.lib as lib
d = lib.riemann.HLLC()
d.brick_type   # 'native'
d.native_id    # 'pops::HLLCFlux'
d.scheme       # 'hllc'
lib.solvers.GMRES().native_id   # 'pops::gmres_solve' (a native free function)
lib.fields.Poisson().available  # False (no standalone native type yet; solved via GeometricMG)
```

The native ids resolve to real symbols in `include/pops` (verified by the test suite); see
{doc}`native-numerics` for the FV solver/reconstruction bricks.

## Macros (time schemes)

`pops.lib.time.*` are MacroBricks: they build a Program from operator names and add no runtime
stepper of their own. They forward to the `pops.time` `std` library, so a library scheme and a
hand-written {doc}`time-program` produce the same IR.

```python
T = pops.lib.time.predictor_corrector(P, "plasma",
                                     fields_operator="fields_from_state",
                                     explicit_rate_operator="explicit_rate",
                                     implicit_operator="implicit_operator")
```

## External C++ bricks

A user ships a brick in a standalone `.so` that registers a manifest entry at static-init time
with the C++ macro `POPS_REGISTER_BRICK(id, category, requirements)`
(`include/pops/runtime/program/external_brick.hpp`) and exports a C function
`const char* pops_brick_manifest()` returning JSON over the process-global `BrickRegistry`.
`pops.lib.load_cpp_library(path)` dlopens the `.so`, reads that manifest, and registers the ids
in an in-process catalog; `pops.lib.riemann.User(id)` (and the category-agnostic
`pops.lib.external(id)`) then surface an `external_cpp` descriptor carrying the manifest's
requirements/capabilities. An id that was never loaded raises a clear `LookupError`
("external brick 'x' not loaded; call pops.lib.load_cpp_library(...)") rather than fabricate a
descriptor.

```python
n = pops.lib.load_cpp_library("./my_bricks.so")   # registers every brick the .so manifests
d = pops.lib.riemann.User("my_hllc")
d.brick_type     # 'external_cpp'
d.native_id      # 'my_hllc'
d.requirements   # {'capabilities': ['pressure', 'wave_speeds']}
```

### Static dispatch of an external Riemann brick

A `riemann` brick is not only catalogued, it is DISPATCHED into the finite-volume machinery in the
same type system as a native flux (`pops::HLLCFlux`), statically and with no per-cell string lookup.
The brick `.so` does the static instantiation itself: it defines a `NumericalFlux` functor and uses
`POPS_DEFINE_EXTERNAL_RIEMANN_BRICK(id, Flux, Model, requirements)`
(`include/pops/runtime/program/external_riemann_brick.hpp`), which registers the manifest and emits a
flat-array `pops_brick_residual` entry point that calls `build_block<Limiter, UserFlux>` -- the user
flux is a compile-time template argument, fully inlined, exactly the leaf the native string ladder
routes to. The host `ExternalBrickHandle` dlopens the `.so`, reads + registers its manifest, and
resolves that entry point ONCE; the only runtime string left is the limiter (a 4-way `if` resolved
once at install, never per cell).

```cpp
// my_riemann.cpp -- compiled to my_riemann.so against the pops headers
struct MyRiemann {
  template <class Model>
  POPS_HD typename Model::State operator()(const Model& m, const typename Model::State& UL,
                                          const pops::Aux& AL, const typename Model::State& UR,
                                          const pops::Aux& AR, int dir) const { /* ... */ }
};
namespace user { using Model = pops::CompositeModel<pops::Euler, pops::NoSource, pops::BackgroundDensity>; }
POPS_DEFINE_EXTERNAL_RIEMANN_BRICK("my_riemann", MyRiemann, user::Model, "pressure,wave_speeds");
POPS_DEFINE_BRICK_MANIFEST();  // exports pops_brick_manifest() once per .so
```

## Status

The descriptor catalog, the native ids, the time macros and external C++ bricks are in place:
`pops.lib.load_cpp_library` + the C++ `BrickRegistry` / `POPS_REGISTER_BRICK` / `POPS_DEFINE_BRICK_MANIFEST`
register and surface a brick, and `POPS_DEFINE_EXTERNAL_RIEMANN_BRICK` + `ExternalBrickHandle`
statically dispatch an external Riemann brick's flux (CPU/OpenMP Kokkos validated on macOS; the
GPU/GH200 dispatch is a ROMEO follow-up). `pops.compile_library(name, objects, emit=True)` compiles a
reusable brick library to a real `.so` (Kokkos toolchain) that exports its ABI key + brick metadata;
`pops.read_library_manifest` reads it back (with a hard ABI guard) and
`pops.compile_problem(..., libraries=[...])` consumes it. The `specialization` modes (native / library
/ specialized / auto) are follow-ups.
