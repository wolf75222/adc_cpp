# Operator-first builder layer (Spec 2 kernel)

The operator-first layer is the single logical kernel of the DSL. The board facade
({doc}`board-like-dsl`) is sugar on top of it and lowers to the same objects; this layer
is not kept only for compatibility -- it is what libraries, generic time-scheme macros,
custom solvers, precise tests, advanced operators and lowering inspection are written
against.

## Authoring a model (operator-first)

```python
import pops.model as model

mod = model.Module("euler_poisson_lorentz")
U = mod.state_space("U", components=["rho", "mx", "my"])
fields = mod.field_space("fields", components=["phi", "grad_x", "grad_y"])

mod.operator(name="fields_from_state", signature=(U,) >> fields, kind="field_operator",
             expr=...)
mod.operator(name="explicit_rate", signature=(U, fields) >> model.Rate(U),
             kind="local_rate", expr=...)
mod.operator(name="implicit_operator", signature=(fields,) >> model.LocalLinearOperator(U, U),
             kind="local_linear_operator", expr=...)
```

See {doc}`operator-modules` for the full operator-first model authoring guide and
{doc}`typed-ir` for the type reference.

## Composing a program (operator-first)

```python
import pops.time as time

P = time.Program("predictor_corrector")
P.bind_operators(mod)
dt = P.dt
U_n = P.state("plasma", space=U)
fields_n = P.call("fields_from_state", U_n, name="fields_n")
R_n = P.call("explicit_rate", U_n, fields_n, name="R_n")
L_n = P.call("implicit_operator", fields_n, name="L_n")
rhs = P.linear_combine("U_star_rhs", U_n + dt * R_n)
U_star = P.solve_local_linear("U_star", operator=P.I - dt * L_n, rhs=rhs)
P.commit("plasma", U_star)
```

`P.call`, `P.linear_combine`, `P.apply`, `P.solve_local_linear`, `P.solve_linear`,
`P.commit`, `P.commit_many` are first-class. A library time-scheme macro is just a
function that builds such a `Program`; `pops.lib.time.*` macros do exactly this and add no
runtime stepper of their own.

## Relationship to the board facade

The board facade builds or calls these same objects -- it has no registry, scheduler,
codegen or runtime of its own. `examples/spec3/board_vs_operator_ir_equivalence.py`
asserts a board program and the equivalent operator-first program have identical IR, and
`examples/spec3/operator_first_same_problem.py` drives a board-authored model purely
through `P.call`. On any ambiguity, the operator-first level is the source of truth.
