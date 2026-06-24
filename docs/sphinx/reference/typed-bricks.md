# Typed bricks (`adc.lib`)

`adc.lib` is a catalog of typed brick descriptors and IR macros. It computes nothing in
Python: a descriptor names a brick and carries its metadata (a native C++ id, a runtime
scheme string, requirements, capabilities); the codegen and runtime consume it.

## The four brick kinds

| Kind | Meaning | Example |
| --- | --- | --- |
| NativeBrick | a C++ type already in `include/adc` | `adc.lib.riemann.HLLC()` -> `adc::HLLCFlux` |
| GeneratedBrick | a DSL-authored brick compiled to C++ | a custom source / local matrix |
| MacroBrick | a Python function that builds Program IR | `adc.lib.time.predictor_corrector` |
| ExternalCppBrick | a user C++ brick registered by id | `adc.lib.riemann.User("my_hllc")` |

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
`invariants`, and `time` (macros forwarding to `adc.time` `std`).

```python
import adc.lib as lib
d = lib.riemann.HLLC()
d.brick_type   # 'native'
d.native_id    # 'adc::HLLCFlux'
d.scheme       # 'hllc'
lib.solvers.GMRES().native_id   # 'adc::gmres_solve' (a native free function)
lib.fields.Poisson().available  # False (no standalone native type yet; solved via GeometricMG)
```

The native ids resolve to real symbols in `include/adc` (verified by the test suite); see
{doc}`native-numerics` for the FV solver/reconstruction bricks.

## Macros (time schemes)

`adc.lib.time.*` are MacroBricks: they build a Program from operator names and add no runtime
stepper of their own. They forward to the `adc.time` `std` library, so a library scheme and a
hand-written {doc}`time-program` produce the same IR.

```python
T = adc.lib.time.predictor_corrector(P, "plasma",
                                     fields_operator="fields_from_state",
                                     explicit_rate_operator="explicit_rate",
                                     implicit_operator="implicit_operator")
```

## Status

The descriptor catalog, the native ids and the time macros are in place. `compile_library`
(separately compiled brick libraries), the `specialization` modes (native / library /
specialized / auto) and external C++ brick registration (`adc.lib.load_cpp_library`) are
follow-ups.
