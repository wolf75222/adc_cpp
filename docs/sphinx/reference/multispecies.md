# Generic multi-species

No species is hardcoded. A species, fluid, population or evolved field is a named
**BlockInstance** of a {doc}`StateSpace <typed-ir>`; the core knows BlockInstances,
StateSpaces, operators, a Program and bindings, not "electrons". The same StateSpace can be
instantiated several times (`fluid_A`, `fluid_B`, ...).

## Spaces and instances

```python
import pops.model as model
e = model.StateSpace("electron_state", ["ne", "mex", "mey"],
                     roles={"ne": "Density", "mex": "MomentumX", "mey": "MomentumY"})
i = model.StateSpace("ion_state", ["ni", "mix", "miy"])
n = model.StateSpace("neutral_state", ["nn", "mnx", "mny"])
```

## Arbitrary-arity operators and typed multi-output

An operator takes an arbitrary number of states (1, 2, 3, N) and may return several typed
outputs. `pops.model.RateBundle` is the typed multi-output of a coupling: one
`Rate(StateSpace)` per participating block, of arbitrary arity. It is typed, so a wrong rate
on a wrong state is rejected:

```python
coll = model.RateBundle({"electrons": model.Rate(e), "ions": model.Rate(i),
                         "neutrals": model.Rate(n)})   # arity 3
coll["electrons"]                # RateSpace('Rate(electron_state)')
coll.require("electrons", e)     # ok
coll.require("electrons", i)     # TypeError: wrong rate on wrong state
```

A coupling (charge density, a field solve, collisions, ionization, radiation) is just an
ordinary multi-input / multi-output operator; the runtime has no special "coupling" category.

## Multi-block program and atomic commit

A Program references several blocks, solves coupled fields from their stage states, and
commits them atomically (no operator observes a partially committed coupled group):

```python
import pops.time as adctime
P = adctime.Program("three_fluids_step")
dt = P.dt
e_n, i_n, n_n = P.state("electrons"), P.state("ions"), P.state("neutrals")
fields = P.solve_fields_from_blocks([e_n, i_n, n_n], name="fields")   # coupled, arity 3
e1 = P.linear_combine("e1", e_n + dt * P.rhs(name="Re", state=e_n, fields=fields, flux=True))
# ... i1, n1 ...
P.commit_many({"electrons": e1, "ions": i1, "neutrals": n1})          # atomic
```

`pops.time.StageStateSet` (built by `P.state_set`) packages a coherent set of stage states so
a field solve reads an unambiguous version of each block (see
`examples/spec3/stage_state_set_field_solve.py`).

## Board facade

The blackboard facade ({doc}`board-like-dsl`) authors N species directly and LOWERS them to the
operator-first multi-block IR above; it owns no second runtime. Each `m.species(...)` is one
`StateSpace` (a named block instance); `m.coupled_rate(...)` is the `coupled_rate` operator of
arbitrary arity; `m.solve_fields_from_species(...)` is a multi-input field operator (the
`solve_fields_from_blocks` surface). A species handle indexes its cons vars by name (`e["ne"]`)
for the rate formulas.

```python
import pops.physics as physics
m = physics.Model("three_fluid")
e = m.species("electrons", state=["ne", "mex", "mey"],
              roles={"ne": "density", "mex": "momentum_x", "mey": "momentum_y"})
i = m.species("ions", state=["ni", "mix", "miy"])
n = m.species("neutrals", state=["nn", "mnx", "mny"])
m.coupled_rate("collision", inputs=[e, i, n],
               outputs={e: [i["ni"] - e["ne"], e["ne"], e["ne"]],
                        i: [e["ne"] - i["ni"], i["ni"], i["ni"]],
                        n: [n["nn"], n["nn"], n["nn"]]})
m.solve_fields_from_species("fields", inputs=[e, i, n],
                            outputs={"phi": None}, solver="geometric_mg")
```

`m.module` is then the same `pops.model.Module` a hand-written operator-first model builds: a
two-fluid board model and its hand-written counterpart share a `module_hash` and emit
byte-identical C++. A wrong-species rate in an affine combine is rejected (typed multi-output).
The single-species board path is byte-identical to `m.state`.

## Status

The operator-first multi-block kernel is in place at the IR level: multiple StateSpaces on a
`Module`, a multi-block `Program` (`P.state` per block, `solve_fields_from_blocks`),
`RateBundle`, `commit_many` (atomic, validated). `examples/spec3/multispecies_three_fluids.py`
builds a 3-species step. The coupled multi-block field solve lowers to C++:
`P.solve_fields_from_blocks([...])` emits `ctx.solve_fields_from_blocks` over a per-block stage-state
vector (Spec 3 criterion 24, ADC-457). The board sugar (`m.species` for N > 1, `m.coupled_rate` -> a
`RateBundle` operator, `m.solve_fields_from_species`) now LOWERS to this IR (ADC-457); the compiled
`.so` coupled-rate kernel + multi-block solve RUN is validated on ROMEO (Kokkos-only AOT).
