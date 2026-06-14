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

> **Roles hardcoded on the C++ side.** The role / field descriptors are not configurable
> from Python. `adc.CondensedSchur(...)` accepts `kind`, `theta`, `alpha`, but passing
> `density=`, `momentum=`, `energy=`, `magnetic_field=` or `potential=` raises an error:
> the C++ source stage fixes these roles hard. This is intentional (the contract of
> `CondensedSchurSourceStepper` is frozen).

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
