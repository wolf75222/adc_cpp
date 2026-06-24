# Operator-first modules (`adc.model`)

A **module** is the *model-free* description of a model: typed spaces plus a registry of
typed operators. It is the generalization of the physical DSL ({doc}`symbolic-dsl`): where
`adc.dsl.Model` describes a PDE with flux, sources and elliptic fields, `adc.model.Module`
describes any model as **operators with signatures**, and a compiled time program
({doc}`time-program`) composes those operators *by signature*, never by a hardcoded PDE
category.

```{admonition} Module vs Simulation
:class: note
A **Module** owns the *rules*: the spaces, the parameters, the aux declarations and the typed
operators. A **Simulation** (the `System` / `AmrSystem`) owns the *data*: the grid, the arrays,
the solvers, the IO and the clock. The Module carries no numerical values; the Simulation
instantiates the operators on a grid at a current time.
```

## Spaces

A module declares the typed spaces its operators read and write:

- `StateSpace` -- a conservative state, e.g. the components `(rho, mx, my)` of `U`, with
  optional physical roles. A generic program must not depend on a specific role.
- `FieldSpace` -- an auxiliary or solved-field space, e.g. `(phi, grad_x, grad_y)`. It is not
  necessarily produced by Poisson; any operator may produce it.
- `RateSpace` -- the tangent of a state space, the values of `dU/dt`. Written `Rate(U)`.
- `ParameterSpace` -- a named scalar parameter (a default value, read at runtime, never frozen
  at codegen).
- `AuxSpace` -- a named auxiliary field imposed or updated by the Simulation (distinct from a
  `FieldSpace`, which an operator produces).

## Operators and signatures

An operator is a named, typed function with a `Signature` `(inputs) -> output` and a `kind`.
The `>>` sugar builds a signature:

```python
import adc.model as M

U = M.StateSpace("U", ("rho", "mx", "my"))
Fields = M.FieldSpace("fields", ("phi", "grad_x", "grad_y"))

sig_rhs    = (U, Fields) >> M.Rate(U)                 # an explicit rate operator
sig_solve  = (U,) >> Fields                           # a field operator
sig_linear = (Fields,) >> M.LocalLinearOperator(U, U) # a local linear operator
```

The operator kinds are: `local_rate`, `local_source`, `local_linear_operator`,
`field_operator`, `grid_operator`, `projection`, `diagnostic`, `matrix_free_operator`,
`local_nonlinear_residual`, `global_residual`. Each operator also declares **capabilities**
(e.g. `produces_rate`, `linear`, `solve_i_minus_a`, `matrix_free`) and **requirements** (e.g.
the aux fields it reads, the elliptic solver it needs); program validation and the Simulation
use them to check an operator can run.

A module registers operators with `Module.operator`, as a builder or a decorator:

```python
mod = M.Module("euler_poisson_lorentz")
U = mod.state_space("U", ("rho", "mx", "my"))
Fields = mod.field_space("fields", ("phi", "grad_x", "grad_y"))

@mod.operator(name="explicit_rhs", signature=(U, Fields) >> M.Rate(U), kind="local_rate")
def explicit_rhs(state, fields):
    ...  # builds the operator body (codegen IR), never executed by Python during step
```

## Calling operators from a program

`adc.time.Program` composes operators by name. Bind the module's registry, then call:

```python
P = adctime.Program("generic_predictor_corrector").bind_operators(mod)
U_n      = P.state("plasma")
fields_n = P.call("fields_from_state", U_n)
R_n      = P.call("explicit_rhs", U_n, fields_n)
L_n      = P.call("lorentz", fields_n)
```

`P.call` resolves the operator against the bound registry, type-checks the arguments against
its signature (a clear error on an unknown operator, a wrong arity, a wrong value type, or an
unbound program), and lowers to the matching runtime primitive. The program never mentions
flux, source, Poisson or Lorentz: it only composes typed operator calls, linear combinations,
local and global solves, and a commit. That is the model-free level.

The standard library provides model-free macros keyed on operator names:
`adc.time.std.predictor_corrector_local_linear`, `explicit_rk` and `imex_local_linear`. The
same macro runs against any module that provides operators with the expected signatures; see
`examples/operator_modules/predictor_corrector_operator_first.py`.

## Compatibility with `adc.dsl.Model`

`adc.dsl.Model` is the **PDE convenience facade** over a module: its `flux` / `source_term` /
`linear_source` / `elliptic_field` / `projection` lower into typed operators, exposed via
`m.operator_registry()` and `m.module`:

| dsl.Model API                | typed operator              | signature                                   |
| ---------------------------- | --------------------------- | ------------------------------------------- |
| `m.flux(...)`                | `flux_default` grid_operator| `(U) -> Rate(U)`                            |
| `m.source_term("electric", ...)` | `electric` local_source | `(U[, Fields]) -> Rate(U)`                  |
| `m.linear_source("lorentz", ...)`| `lorentz` local_linear_operator | `(Fields?) -> LocalLinearOperator(U, U)` |
| `m.elliptic_rhs(...)`        | `fields_from_state` field_operator | `(U) -> Fields`                      |
| `m.rate_operator("explicit_rhs", flux=True, sources=["electric"])` | `explicit_rhs` local_rate | `(U[, Fields]) -> Rate(U)` |

`m.rate_operator` names a composite `-div F + sources` rate so a program can call it by name
instead of spelling `P.rhs(flux=True, sources=[...])`. The alias lowers to the same RHS IR and
does not change the model hash or the codegen.

## Migrating from the PDE shortcuts

The PDE shortcuts (`P.rhs`, `P.solve_fields`, `P.apply`, `P.source`) remain valid; they are
sugar that lowers to the same IR as `P.call`. To move a model toward the operator-first style:

1. Keep your `dsl.Model`.
2. Add `m.rate_operator("explicit_rhs", flux=True, sources=[...])` for the rate the program uses.
3. Replace `P.rhs(state=U, fields=f, sources=[...])` with `P.call("explicit_rhs", U, f)` and
   `P.solve_fields(U)` with `P.call("fields_from_state", U)` (after `P.bind_operators(m)`).
4. Optionally express the whole step with a `std` operator-first macro.
5. When a model is no longer naturally a single PDE, define it as an `adc.model.Module` directly.

## Introspection

The compiled handle and the module expose the registry metadata without loading the `.so`:
`list_operators()`, `operator_signature(name)`, `operator_requirements(name)`,
`operator_capabilities(name)`, `list_state_spaces()`, `list_field_spaces()`. For example
`compiled.operator_signature("explicit_rhs").output == adc.model.Rate("U")`.
