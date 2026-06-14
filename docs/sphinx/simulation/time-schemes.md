# Time schemes


The time policy is per block (the object passed in `time=`). Four families.

Explicit: `adc.Explicit(substeps=, stride=, method=)` advances transport and source
explicitly by an SSP Runge-Kutta (`method="ssprk2"` by default, 2-stage Heun; `"ssprk3"`,
or shortcut `ssprk3=True`, 3-stage order 3, to pair with `weno5`).

```python
time=adc.Explicit()                       # SSPRK2, default
time=adc.Explicit(ssprk3=True)            # SSPRK3, less dissipative
```

IMEX: `adc.IMEX(substeps=, stride=, implicit_vars=, implicit_roles=)` (clear alias
`adc.SourceImplicit`) combines an explicit transport (SSPRK) and a stiff implicit source
(backward-Euler, cell-local Newton). The treatment is partial: only the source is
implicit, the transport stays explicit. It is not a global implicit PDE solver. The
`implicit_vars` / `implicit_roles` mask chooses which conserved variables are treated implicitly
(the others stay explicit); it is carried by the policy (the block), not by the model.

```python
time=adc.IMEX(substeps=10)                                  # stiff source, subcycled
time=adc.IMEX(implicit_roles=["MomentumX", "MomentumY", "Energy"])
```

Lie / Strang splitting + source stage condensed by Schur: `adc.Split` and `adc.Strang`
opt-in in the Schur effort, an explicit hyperbolic transport stage (`adc.Explicit`,
SSPRK) followed by a separate source stage `adc.CondensedSchur`. `adc.Split` chains `H(dt) ; S(dt)`
(Lie / Godunov, first order); `adc.Strang` plays `H(dt/2) ; S(dt) ; H(dt/2)` (symmetric, second
order). The `adc.CondensedSchur` stage handles the stiff coupled source potential / velocity / Lorentz
by assembling and solving a condensed tensorial elliptic operator (BiCGStab preconditioned
MG); it is a global implicit (it couples the whole domain). `adc.Split` / `adc.Strang` are
wired only by `add_equation` (which plugs in the source stage), not by `add_block`.

```python
sim.add_equation("ions", model=compiled,
                 spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
                 time=adc.Strang(hyperbolic=adc.Explicit(),
                                 source=adc.CondensedSchur(theta=0.5, alpha=3.0)))
```

> Local vs global. `adc.SourceImplicit` (IMEX) is local: it couples only the components
> of the same cell (relaxation, reactions, friction), without an elliptic solve. `adc.CondensedSchur`
> is global: for the stiff non-local Lorentz / electrostatic coupling. A purely local stiff
> source does not need Schur.

`adc.Implicit` is deprecated (alias of IMEX, emits a `DeprecationWarning`): its name wrongly suggests
a global implicit solver. Use `adc.SourceImplicit(...)` or `adc.IMEX(...)`.

Detail: [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md) sections 4 to 6,
[SCHUR_CONDENSATION_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/SCHUR_CONDENSATION_DESIGN.md), Hoffart step sequence
[HOFFART_STEP_SEQUENCE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/HOFFART_STEP_SEQUENCE.md). On the C++ side: `numerics/time/*.hpp`.
