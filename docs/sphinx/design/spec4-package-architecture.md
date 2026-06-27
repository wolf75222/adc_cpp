# Spec 4: Python package architecture

This page describes the target layout of the `pops` Python package introduced in
Spec 4. The restructure replaces the previous flat module layout (`pops/dsl.py`,
`pops/physics.py`, `pops/time.py`, `pops/lib.py`, ...) with seven sub-packages that
form a strictly acyclic import graph.

```{note}
The old flat modules (`pops.dsl`, `pops.physics` at the top level, etc.) are deleted.
No backward-compatibility shims are shipped. Callers must update their imports to
the sub-packages listed here.
```

## The seven sub-packages

| Package | Responsibility |
|---------|---------------|
| `pops.ir` | Symbolic IR: expression nodes, ops, values, lowering passes, and visitors. Imports nothing inside `pops`. Used by every layer above it. |
| `pops.model` | Operator-first typed model core: `Module`, typed spaces (`StateSpace`, `FieldSpace`, `RateSpace`, `ParameterSpace`, `AuxSpace`), `Operator`, `OperatorRegistry`, `Signature`. Imports only `pops.ir`. |
| `pops.physics` | Math and physics authoring facade. `pops.physics.Model` is the high-level PDE description (conservative variables, flux, eigenvalues, sources, elliptic right-hand side). Lowers to a `pops.model.Module`. Imports `pops.ir` and `pops.model`. |
| `pops.time` | Temporal language: `Program`, schedules, equations, and the operator-first time IR. Imports `pops.ir` and `pops.model`. |
| `pops.lib` | Descriptors, time schemes, moment closures, and provided standard models. Imports `pops.ir`, `pops.model`, `pops.time`, and `pops.physics`. |
| `pops.codegen` | The only C++ emitter. Holds `module_codegen` and all `emit_cpp_*` functions as free functions that take a model object. Imports `pops.ir`, `pops.model`, `pops.time`, `pops.physics`, and `pops.lib`. Does not import `pops.runtime`, and never imports `_pops` at module scope (the only `_pops` touch is a lazy in-function `abi_key` hop in `toolchain`). |
| `pops.runtime` | Thin facade over the `_pops` native extension: `System`, `AmrSystem`, and their configuration objects. Imports only `_pops`. |

## Acyclic import graph

The graph has a single direction: lower packages never import upper ones.
`pops.codegen` is the exclusive C++ emission point; authoring packages
(`pops.physics`, `pops.time`, `pops.lib`) never import it or `_pops`.

| Package | May import (inside pops) |
|---------|--------------------------|
| `pops.ir` | _(nothing)_ |
| `pops.model` | `pops.ir` |
| `pops.physics` | `pops.ir`, `pops.model` |
| `pops.time` | `pops.ir`, `pops.model` |
| `pops.lib` | `pops.ir`, `pops.model`, `pops.time`, `pops.physics` |
| `pops.codegen` | `pops.ir`, `pops.model`, `pops.time`, `pops.physics`, `pops.lib` |
| `pops.runtime` | `_pops` only |

## Codegen as free functions: the key design decision

The `emit_cpp_*` helpers and `module_codegen` live in `pops.codegen` as ordinary
free functions that accept a model object. They do NOT live on `pops.physics.Model`
as methods (no `m.compile()`, no `m.emit_cpp_source()`).

Why: keeping C++ emission out of the authoring packages prevents a cycle. If
`pops.physics.Model.compile()` existed, `pops.physics` would have to import
`pops.codegen`, which imports `pops.lib`, which imports `pops.physics` -- a cycle.
As free functions, `pops.codegen` may import everything above it while authoring
packages import nothing from `pops.codegen`. Callers that need to compile call
`pops.compile_problem(model=m)` (the top-level convenience that delegates to
`pops.codegen`), while authoring workflows that only need to inspect or lower a
model never pay the cost of loading the C++ toolchain.

## Public API surface

The entries below are the stable public surface as of Spec 4. The top-level
`pops` namespace re-exports the most commonly used symbols.

| Symbol | Package |
|--------|---------|
| `pops.physics.Model` | `pops.physics` |
| `pops.time.Program` | `pops.time` |
| `pops.compile_problem(model=...)` | top-level (delegates to `pops.codegen`) |
| `pops.codegen.module_codegen` | `pops.codegen` |
| `pops.codegen.emit_cpp_*` | `pops.codegen` |
| `pops.runtime.System` | `pops.runtime` (also `pops.System`) |
| `pops.runtime.AmrSystem` | `pops.runtime` (also `pops.AmrSystem`) |

## Migration from the old flat API

The table below maps every old symbol to its new location. The old flat
modules are deleted; no shims exist.

| Old (flat) | New (Spec 4) |
|------------|-------------|
| `pops.dsl.Model` | `pops.physics.Model` |
| `pops.dsl.HyperbolicModel` | `pops.physics.Model` (or internal `pops.physics._HyperbolicModel`) |
| `m.compile(...)` | `pops.compile_problem(model=m, ...)` |
| `m.emit_cpp_source(...)` | `pops.codegen.emit_cpp_source(m, ...)` |
| `m.emit_cpp_header(...)` | `pops.codegen.emit_cpp_header(m, ...)` |
| `pops.dsl.CompiledModel` | `pops.codegen.CompiledModel` |
| `pops.dsl.HybridModel` | `pops.codegen.HybridModel` |
| `from pops.dsl import ...` | `from pops.physics import ...` (authoring); `from pops.codegen import ...` (C++ emission) |
| `pops.math.ddt`, `pops.math.div`, etc. | `pops.physics.math.ddt`, or `from pops.ir import ddt, div, ...` |
| `pops.physics.Model` (Spec 3 flat) | `pops.physics.Model` (unchanged, now in the `pops.physics` sub-package) |
| `pops.time.Program` (Spec 3 flat) | `pops.time.Program` (unchanged, now in the `pops.time` sub-package) |
| `pops.lib.*` (Spec 3 flat) | `pops.lib.*` (unchanged, now in the `pops.lib` sub-package) |
| `pops.model.*` (Spec 2 flat) | `pops.model.*` (unchanged, now in the `pops.model` sub-package) |
| `pops.System` | `pops.System` (top-level re-export of `pops.runtime.System`) |
| `pops.AmrSystem` | `pops.AmrSystem` (top-level re-export of `pops.runtime.AmrSystem`) |

## Module file-size rule

Each module file must stay at or below 500 lines. Files that would exceed this
limit are split into mixins or sub-modules within the package.

## Related documentation

- Operator-first module reference: {doc}`../reference/operator-modules`
- Blackboard DSL (physics authoring): {doc}`../reference/board-like-dsl`
- Time program reference: {doc}`../reference/time-program`
- Symbolic DSL reference: {doc}`../reference/symbolic-dsl`
- Backend matrix: {doc}`../reference/backend-matrix`
