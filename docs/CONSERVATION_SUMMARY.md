# Conservation properties -- magnetized FV scheme (cartesian Schur path)

State: June 2026. Primary source: `python/tests/test_schur_conservation.py` (PR #207,
branch `test/schur-conservation`). The cited figures are THE VALUES MEASURED by this
test; the asserted bounds sit just above the actual observed drift (not machine
equalities where physics legitimately moves the quantity).

Paper reference: Hoffart et al., arXiv:2510.11808 (magnetized isothermal Euler-Poisson).
Methodological note in section 2.


---

## 1. Table of measured properties

| Property | Test (function) | Setup | Asserted bound | Measured drift | Regime |
|-----------|----------------|-------|----------------|--------------|--------|
| **Mass -- closed domain** | `test_masse` | 64x64, periodic, axisymmetric ring, 40 steps | relative drift < 1e-12 | ~1.9e-16 (machine precision) | Quasi-exact (conservative FV + rho frozen by the source stage) |
| **Mass -- Dirichlet** | (witness in `test_masse`, docstring note) | Same setup with `bc=dirichlet`, Foextrap at the boundaries | not asserted | ~1e-2 | BC ARTIFACT (Dirichlet boundary condition / Foextrap, not the scheme) |
| **Momentum -- axisymmetric profile** | `test_momentum` (2a) | 64x64, periodic AND Dirichlet, centered ring, 20 steps | max|delta_mom|/m0 < 1e-12 | ~5e-18 (machine precision) | Quasi-exact (discrete symmetry: net force = 0 by symmetry, FV telescoping) |
| **Momentum -- asymmetric profile (physical impulse)** | `test_momentum` (2b) | 64x64, Dirichlet, off-center bump, 20 steps | > 1e-6 (source active) and < 1e-1 (no blow-up) | ~1.9e-3 (physical impulse of the force) | O(dt) -- PHYSICAL, not an error; converges at fixed T (ratio ~1.04 under dt->dt/2) |
| **Momentum -- convergence at fixed T** | `test_momentum` (2b, dt refinement) | T=0.06 fixed, N1/N2 = dt/dt*2, Dirichlet | impulse(dt)/impulse(dt/2) ratio in [0.5, 2] | ratio ~1.04 | Convergent O(dt) -- physical quantity continuous in dt |
| **Energy E > 0** | `test_energie_positivite` (3a, 3d) | 48x48, Dirichlet, compressible (gamma=1.4), 30 steps | E_min > 0; bounded growth 0 < delta_E/E0 < 0.5 | ~12% total growth (PHYSICAL work of the self-consistent field) | Healthy balance -- bounded physical growth, no instability |
| **Energy without source -- machine invariance** | `test_energie_positivite` (3c) | Same setup without Schur stage | rel delta_E < 1e-12 | < 1e-12 | Quasi-exact (pure FV at equilibrium, E invariant) |
| **SchurEnergyKernel active** | `test_energie_positivite` (3b) | Diff E_schur vs E_nosrc | max|delta_E| > 1e-6 | ~2e-1 | Witness that the kernel is indeed active |
| **Positivity rho (minmod)** | `test_positivite_densite` (4) | 64x64, steep ring (rho_fond=0.2, drho=2), recon_prim, 40 steps | rho_min > 0 | rho_min > 0 (rho_floor > 0 maintained) | Quasi-exact under primitive reconstruction + limiter |
| **Positivity rho (vanleer)** | `test_positivite_densite` (4) | Same, vanleer limiter | rho_min > 0 | rho_min > 0 | Quasi-exact |
| **Positivity rho (weno5)** | `test_positivite_densite` (4) | Same, weno5 limiter | rho_min > 0 | rho_min > 0 | Quasi-exact |
| **Positivity pressure p = cs2*rho** | `test_positivite_densite` (4) | Isothermal p = cs2*rho, same 3 limiters | p_min > 0 | p_min > 0 | Derived from the positivity of rho |

Legend for the "Regime" column:
- **Quasi-exact**: conserved to machine precision (rounding error only).
- **O(dt) / physical**: the quantity varies by a real physical amount, convergent as dt->0 at fixed T.
- **BC artifact**: the drift is due to the boundary condition (Foextrap on Dirichlet boundary), not the internal scheme.


---

## 2. Honest note: FV vs FE structure-preserving

### What pops guarantees (FV)

The spatial scheme of pops is **FINITE VOLUME** (FV), not finite element (FE) like the
reference paper Hoffart et al. (arXiv:2510.11808). The consequences are as follows:

**Mass conservation -- exact by construction (FV).**
The FV discretization of the continuity fluxes is telescopic: the fluxes leaving a cell
enter the neighboring cell, and the total mass is invariant in the sum (machine precision)
in a closed domain (periodic or reflecting boundaries). Moreover, the condensed source stage
(`CondensedSchurSourceStepper`) freezes rho during the source step and only modifies the velocity.
Measured result: relative drift ~1.9e-16 in periodic (cf. #207).

**Momentum conservation -- exact ONLY when the net force is zero.**
The FV transport of momentum is also telescopic: in the absence of a body force, the
total momentum would be conserved to the machine. However, the electrostatic/Lorentz force
(-rho*grad(phi) + rho*v x Omega) constitutes an explicit source of momentum, applied by
the Schur stage. This force is a real physical quantity:

- On a centered **axisymmetric** profile, the net force integrates to zero by symmetry, and the
  scheme preserves this symmetry: total momentum conserved to the machine (~5e-18, cf. #207 (2a)).
- On an **asymmetric** profile, the net force is nonzero and the total momentum varies by
  the physical impulse of this force -- this is physics, not an error (measured ~1.9e-3
  over 20 steps, convergent at fixed T under dt refinement, cf. #207 (2b)).

**In FV, momentum conservation IS NOT exact by construction** when body forces
are present. It is so only when the force balance is discrete-exact zero.

**Energy conservation -- healthy balance, physical growth.**
The total energy grows (~12% over 30 steps) because the self-consistent field does NET work
on the fluid initially at rest -- this is physically expected (diocotron instability =
energy transfer from the field to the fluid). The `SchurEnergyKernel` accounts for the increment
of kinetic energy of the electrostatic work at each step. In FV, the energy balance is
not closed to the machine (the source does explicit work); we bound the growth and we
check the absence of blow-up, not an equality.

### What the Hoffart paper guarantees (FE, structure-preserving in the strong sense)

The paper uses **structure-preserving finite elements** (discrete de Rham complex, compatible
H(curl)/H(div)/L2 spaces). In this framework:

- Momentum conservation follows from the **discrete weak form**: the exact discrete force
  balance is written in the space of discrete differential forms, guaranteeing the EXACT
  integral conservation of momentum up to the DISCRETE body force (without telescoping
  or quadrature error).
- The energy can be conserved or dissipated depending on the time scheme (discrete mirror of
  the continuous identity).
- The positivity of rho can be guaranteed by construction via specific FE limiters.

These properties are **strictly stronger** than what FV offers: they hold
INDEPENDENTLY of the symmetry of the initial condition and without requiring any
particular domain closure.

### Validity of the pops vs Hoffart paper comparison

The comparison of the **diocotron growth rate** between pops and the paper is valid and
significant: the rate is an observable of the linear dynamics (slope of log|a_l(t)|),
which depends on the dispersion of the advection-Poisson operator, not on the exact
structure-preservation of the scheme. The measurement on pops reproduces l=3 to -0.38% of the paper at n=512 (GH200, cf.
`docs/FULL_MODEL_VALIDATION_ROADMAP.md`).

On the other hand, **attributing FE-sense structure-preservation to pops would be inaccurate**: pops conserves
mass exactly (closed FV), preserves momentum when the balance is discrete-exact zero (symmetry),
but does not offer exact-by-construction momentum conservation in the sense of the discrete
weak form. This is an honest distinction, documented here and in test #207.

**Open question:** is a formal proof of momentum conservation at the FE level (discrete weak
form balance) required for the validation context presented in the report/paper?
See `docs/FULL_MODEL_VALIDATION_ROADMAP.md` (PR3/PR4) for the rest of the roadmap.


---

## 3. Pointers

- **Source test**: `python/tests/test_schur_conservation.py` (PR #207,
  branch `test/schur-conservation`). Measured values cited as the primary source.
- **Global roadmap**: `docs/FULL_MODEL_VALIDATION_ROADMAP.md` -- lists the remaining validation
  PRs (PR3 re-fit windows l=4/l=5, PR4 polar, etc.) and the open questions.
- **Reference paper**: Hoffart et al., arXiv:2510.11808.
- **Open question**: formal FE proof required? (cf. roadmap PR2/PR3 and discussion
  section 2 above).
