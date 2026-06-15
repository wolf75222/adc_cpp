# Condensed Schur source stage


The integrator `adc.CondensedSchur` reproduces the splitting of Hoffart et al.
(arXiv:2510.11808) for the stiff potential / velocity / Lorentz-force source of the
magnetized Euler-Poisson system. The key: Schur condensation algebraically eliminates the
velocity from the implicit subsystem, which reduces the source stage to an elliptic solve
(tensor operator `-div(A grad phi)` with `A = rho B^{-1}`, in general non-symmetric)
followed by an explicit reconstruction of the velocity.

It is composed with an explicit transport stage via `adc.Split`:

```python
import adc

time_policy = adc.Split(
    hyperbolic=adc.Explicit(),
    source=adc.CondensedSchur(
        kind="electrostatic_lorentz",   # seul kind supporte
        theta=1.0,                      # theta-schema : 0.5 = Crank-Nicolson, 1 = Euler retrograde
        alpha=3.0,                      # constante de couplage
    ),
)

sim.add_equation(
    "ions",
    model=model,                        # roles requis : Density / MomentumX / MomentumY (Energy optionnel)
    spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
    time=time_policy,
)
```

The model must expose the roles `Density`, `MomentumX`, `MomentumY` (a native isothermal
fluid `adc.FluidState(kind="isothermal") + adc.IsothermalFlux()` provides them). The stage is
entirely C++ (`CondensedSchurSourceStepper`, exposed as `adc.CondensedSchur`): no
per-cell Python callback.

> **Only `potential=` is non-configurable.** `adc.CondensedSchur(...)` accepts `kind`,
> `theta`, `alpha`, and the role descriptors `density=`, `momentum=`, `energy=`,
> `magnetic_field=`: these default to the canonical roles and accept an `adc.Role.*` value or
> a stable role / aux name. Only `potential=` is fixed: passing anything other than `'phi'`
> raises an error, because the source stage solves for the potential itself. This is
> intentional (the contract of `CondensedSchurSourceStepper` is frozen on the potential).

`adc.Strang` is the 2nd-order extension of `adc.Split` (transport / source /
transport sequence). The default is unchanged: a block in pure `adc.Explicit` never sees the
condensed source stage.

> **CondensedSchur (global) vs SourceImplicit (local).** Do not confuse them. `adc.CondensedSchur`
> assembles and solves an elliptic operator coupling the whole domain (for a stiff non-local
> coupling: Lorentz / electrostatic). `adc.SourceImplicit` (= IMEX source-only) is
> local: the implicit only couples the components of the same cell (relaxation, reactions,
> friction), without an elliptic solve, so much cheaper. A local stiff source does not
> need Schur.

## Going further

- Detailed design (the five levels, the non-symmetry of the tensor operator, the
  question of the Krylov solver): [SCHUR_CONDENSATION_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/SCHUR_CONDENSATION_DESIGN.md)
  (banner: `implemente`; the document is the original spec, read as a design
  history).
- Conservation properties of the Cartesian Schur path (measured values):
  [CONSERVATION_SUMMARY.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/CONSERVATION_SUMMARY.md).
- Tests: `python/tests/test_schur_via_system.py` (path `System -> run_source_stage`,
  native bricks, CI-safe), `test_schur_conservation.py`.
