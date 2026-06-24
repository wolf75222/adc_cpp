# Typed operator-first IR (`adc.model`)

The typed IR is the single internal contract the board facade ({doc}`board-like-dsl`)
and the operator-first builder ({doc}`operator-modules`) both lower to. It carries no
numerics; it is the type layer the compiler checks and the codegen consumes.

| Type | Meaning |
| --- | --- |
| `StateSpace(name, components, roles)` | a conservative state `U` (its components and physical roles) |
| `FieldSpace(name, components)` | an auxiliary / solved-field space (e.g. `phi, grad_x, grad_y`) |
| `RateSpace` / `Rate(U)` | the tangent of a state, values of `dU/dt` |
| `LocalLinearOperator(U, U)` | a local linear operator type `U -> U` |
| `MatrixFreeOperator(V, V)` | a matrix-free operator type for a Krylov solve |
| `Signature(inputs, output)` | a typed `(inputs) -> output` contract |
| `Operator(name, kind, signature)` | a named, typed operator in the registry |
| `OperatorRegistry` | the ordered, integer-id'd registry of operators |
| `RateBundle` | a typed multi-output `block -> Rate(StateSpace)` (arbitrary arity) |

The operator kinds are `local_rate`, `local_source`, `local_linear_operator`,
`field_operator`, `grid_operator`, `projection`, `diagnostic`,
`matrix_free_operator`, `local_nonlinear_residual`, `global_residual`.

## Lowering

The board facade builds exactly these objects (see {doc}`board-like-dsl` for the full
map): `m.state -> StateSpace`, `m.solve_field -> field_operator`,
`m.rate -> local_rate (output Rate(U))`, `m.local_linear_operator -> a
LocalLinearOperatorExpr` math object that `m.operator(...)` registers as a
`local_linear_operator`. A time program's `T.define` / `T.solve` / `T.fields` lower to
`P.call` / `P.linear_combine` / `P.solve_local_linear` / `P.solve_fields`.

`Program.dump_operator_ir()` renders the operator-first IR a board program lowers to, and
`Model.dump_module_ir()` renders the typed Module a board model lowers to -- so the
equivalence is inspectable, not assumed.
