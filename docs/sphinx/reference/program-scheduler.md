# Program scheduler

Spec 3 unifies cadence into a single Program scheduler: any Program operation can carry a
schedule, replacing the scattered block stride / substeps / source frequency knobs.

```{admonition} Status
:class: note
The schedule AUTHORING surface (the vocabulary below, policy chaining, recording a schedule on a
Program node, the cacheable-capability validation) and the CODEGEN that lowers every kind/policy to
C++ are available in `adc.time` (ADC-458). A scheduled node lowers to a due-test guard plus its
policy branch around the statements it emits (the cache lives in the C++ `CacheManager`). Two cases
still refuse to lower, loudly: `on_end()` (a compiled `sim.step(dt)` loop carries no end-of-run
signal, so the `.so` cannot know the last step -- use an on_end host hook) and a `when(cond)` over a
bare Python callable (only a Program Bool predicate lowers). The cache cadence RUNTIME in a stepping
`.so` is exercised on ROMEO; the `CacheManager` is unit-tested by `tests/test_cache_manager.cpp`. The
checkpoint of the cache state is a follow-up (ADC-458 section 30).
```

## Authoring

```python
import adc.time as adctime
P = adctime.Program("step").bind_operators(mod)
U = P.state("plasma")
fields = P.call("fields_from_state", U, schedule=adctime.every(10).hold())  # refresh every 10
transport = P.call("flux", U, schedule=adctime.subcycle(4))                 # 4 inner steps
```

`P.dump_operator_ir()` shows the schedule on each node. A caching policy (`hold` /
`accumulate_dt`) requires the operator to be cacheable, declared on the model:

```python
mod.operator_capabilities("fields_from_state", cacheable=True)
```

otherwise the call is rejected: `operator 'flux' is not cacheable; cannot use schedule hold`.
See `examples/spec3/scheduled_fields_subcycled_transport.py`.

## Schedules

A schedule decides when a node is due:

| Schedule | Meaning |
| --- | --- |
| `always()` | due every step (the default) |
| `every(N)` | due every N macro-steps |
| `when(cond)` | due when a runtime condition holds |
| `on_start()` / `on_end()` | due at the first / last step |
| `subcycle(count, dt)` | structured sub-cycling of a block |

and a policy decides what happens when it is NOT due:

| Policy | Behaviour |
| --- | --- |
| `recompute` | always recompute (default) |
| `hold` | reuse the last cached value |
| `skip` | produce nothing (diagnostics only) |
| `zero` | return zero (optional contributions) |
| `accumulate_dt` | accumulate the skipped dt and apply with `eff_dt = sum(dt_skipped)` |
| `error` | error if not due |

With a variable `step_cfl`, `accumulate_dt` must use the real sum of the skipped dt, not
`N * dt_current`.

## Cacheable capabilities

`hold` is only valid on a cacheable operator. An operator declares this:

```python
m.operator_capabilities("fields_from_state", cacheable=True, stale_allowed=True)
m.operator_capabilities("explicit_rate", cacheable=False, requires_fresh_inputs=True)
```

so requesting `every(10).hold()` on a non-cacheable operator is an error:
`operator 'explicit_rate' is not cacheable; cannot use schedule hold`.

## Lowering

A scheduled node lowers around the C++ its operation emits. The due test depends on the kind:
`every(N)` -> `ctx.cache_should_update(id, N)`; `on_start()` -> `ctx.macro_step() == 0`; `when(cond)`
-> the lowered Program Bool predicate; `subcycle(count, dt)` -> a `for` sub-loop over the sub-dt. The
policy decides the not-due branch: `recompute` / `skip` run the body only when due (skip keeps the
stale value, the cacheable contract); `zero` sets the output to zero off-cadence; `hold` stores the
output when due and restores it otherwise; `accumulate_dt` accumulates the actual skipped dt and reads
`eff_dt = sum(dt_skipped)` on the due step (never `N * dt_current`); `error` fails loud if read
off-cadence. A field solve caches the System aux; any other node caches its own named scratch.

## Cache and checkpoint

The `CacheManager` (`include/adc/runtime/program/cache_manager.hpp`) stores per node the cached value
(aux or named scratch), the last update step, the accumulated dt and a validity flag. It is typed,
C++-allocated and Kokkos/MPI-safe. The checkpoint of this state (serialize/restore on restart,
invalidate on a Program-hash mismatch) extends the existing checkpoint path and is tracked separately
(ADC-458 section 30); the `CacheManager` already exposes the in-memory state and accessors the
serializer will read.

## Per-block step policy

The older per-block step policy still runs alongside the scheduler: a block advances `substeps`
sub-steps of `dt/substeps`, or `1` macro-step out of `stride`. See {doc}`time-program`.
