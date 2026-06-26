# Substeps, stride and multirate


Two orthogonal parameters, carried by every time policy, handle the multirate (not all
species require the same `dt`).

- `substeps=N`: the block advances `N` times per macro-step, each substep of length `dt/N`
  (fast species, e.g. electrons `substeps=10`). Default 1;
- `stride=M`: cadence of the block, hold-then-catch-up semantics (catch-up at the end of the window).
  The block is held (state unchanged) as long as `(macro_step + 1) % M != 0`, then advances by an
  effective step `M*dt` at the macro-step where `(macro_step + 1) % M == 0` (slow species, e.g. neutrals
  `stride=20`). It thus stays temporally consistent with the fast blocks (never advanced
  "into the future"). Default 1.

```python
sim.add_block("a", model=m, time=pops.Explicit(stride=1))   # every macro-step
sim.add_block("b", model=m, time=pops.Explicit(stride=3))   # advances once every 3 (end of window)
```

Between two catch-ups, the held block contributes to the right-hand side of the system Poisson with its
stale state (its last advanced density, frozen until the next catch-up). `step_cfl` honors
the cadence: the stable step includes the stride and substeps factor,
`dt <= cfl * h * substeps / (stride * w)`.

> Bit-parity warning. With `substeps=1` (whatever the stride), `step_cfl` is
> bit-identical to the history. With `substeps > 1` it advances a larger `dt` (each substep
> stays at the stability limit). To reproduce a calibrated run with the old formula, use
> `step(dt)` with the explicit historical `dt`.
>
> Backend note. The `aot` backend (`add_equation` on a `CompiledModel` `backend='aot'`) does not
> carry the cadence and rejects `stride > 1` (explicit route, no silent ignore);
> `add_block` (native) and `backend='production'` support the stride.

The multirate is obtained by setting `stride` (and `substeps`) per block. Detail:
[ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md) section 7.
