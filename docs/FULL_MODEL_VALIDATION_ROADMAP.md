# Roadmap: reproducing the COMPLETE Hoffart model (magnetized Euler-Poisson)

> SUPERSEDED (June 2026, Phase 0 audit, see docs/HOFFART_FIDELITY.md and docs/HOFFART_STEP_SEQUENCE.md).
> Two claims in this document are CORRECTED by the code audit + re-reading of the paper:
> 1. "the complete model ALREADY runs, -0.38%" is FALSE: the -0.38% is the REDUCED ExB-scalar DIOCOTRON
>    (diag_polar_omega.py, POLAR path). The complete model (run.py system-schur, CARTESIAN) is NOT
>    validated; its short runs give ~0.035 (-95% vs paper), CRUSHED growth.
> 2. "CARTESIAN PATH B recommended, Path A cosmetic" is INVERTED: the geometry is the PRIMARY suspect.
>    The cartesian square + circular Poisson wall diffuses the ring edge and crushes the instability. The
>    retained path (Agent D) is a CONSERVATIVE DISK DOMAIN (2a mask -> cut-cell/EB), not the square.
> ALSO RESOLVED: |Omega| = beta^2 = 1e12 is CORRECT (paper line 1082 "omega_c := beta^2"), so
> omega_d = 1 and the raw slope of the complete model is DIRECTLY comparable to 0.772/0.911/0.683, WITHOUT
> a 2pi factor (the 2pi only concerns the reduced ExB path). The rest of this file is kept for
> history, but read HOFFART_FIDELITY.md first.

## Major fact (OBSOLETE, see note above): the COMPLETE model ALREADY runs (cartesian path)

`adc_cases/hoffart_euler_poisson_dsl/run.py` builds the COMPLETE system from the paper:
- 3-variable model (rho, rho_u, rho_v), isothermal pressure p = theta*rho (model.py:90-122),
- Lorentz force m x Omega = (omega*my, -omega*mx),
- Gauss -alpha*rho (elliptic_rhs),
- solved by `pops.Split(hyperbolic=Explicit(ssprk3), source=pops.CondensedSchur(theta=0.5, alpha))`,
  that is, the Schur stack #118-128 (CondensedSchurSourceStepper, LorentzEliminator,
  TensorEllipticOperator/GeometricMG, BiCGStab) wired in via system_stepper.hpp:86-90.

Measurement: diocotron rate l=3 = -0.38% vs paper at n=512 (GH200). The rate observable is
`sample_circle(phi, ring_inner)` + FFT-theta (azimuthal mode of the POTENTIAL on a circle): already clean
in polar metric EVEN on a cartesian grid. The "ring edge diffusion" only affects the rendering of
raw DENSITY (schlieren), NOT the rate measurement.

## Path decision: PATH B (cartesian-fluid), recommended

Path A (POLAR fluid + polar Schur) would rebuild at high cost a capability that already exists, and
it is at the RESEARCH level:
- `dispatch_transport_polar` REJECTS the fluid (block_builder_polar.hpp:59-65);
- `PolarPoissonSolver` is a scalar DIRECT solver (FFT-theta + tridiag-r) structurally
  INCOMPATIBLE with the anisotropic crossed Schur operator A = I + c*rho*B^{-1} (the FFT no longer
  diagonalizes once the crossed terms a_xy/a_yx exist; risk of MG stagnation on the 1/r^2 anisotropy);
- the entire Schur stack is wired to cartesian Geometry/dx/dy/GeometricMG.

=> Path A = a later COSMETIC improvement (2D density figure without edge diffusion) IF a
visual need is confirmed, NOT a path toward fidelity to the paper. In the short term: harden/validate B.

## The remaining gap = VALIDATION + CONVERGENCE (no missing capability)

1. l=4/l=5 rates non-monotonic (l=4 -4.9->-8.4%, l=5 +11->+16% from n=384 to n=512) = artifact of the FIT
   WINDOW (PAPER_FIT_WINDOWS static model.py:58, calibrated ~n=128, bite into the saturation of the
   fast modes at high resolution). l=3 converges cleanly (-0.38%).
2. Discrete conservation: missing structure-preserving bricks (mass/momentum/energy/positivity).

## Roadmap

- [x] **PR1 discrete conservation (mass/momentum/energy/positivity)** -- DONE #207. MEASURED tolerances:
  mass conserved to machine precision (1.9e-16, closed domain); momentum symmetry to machine precision; momentum
  impulse = physical O(dt) convergent; E>0, p>0 under minmod/vanleer/weno5. Honest FV-vs-FE note.
  Finding: on Dirichlet the mass leaks ~1e-2 through Foextrap (BC artifact, not from the scheme).
- [ ] **PR2 doc CONSERVATION_SUMMARY**: table [property, test, assertion, bound] + FV note (momentum
  not exact by construction, unlike the paper's FE). (small)
- [ ] **PR3 re-fit early window l=4/l=5**: detect the onset of saturation (d2/dt2 log|a|), sweep
  windows, choose the flat regime. BLOCKER: only the final gamma (sweep_results.csv) is saved on
  ROMEO, NOT amplitude(t) -> sweep.py must be MODIFIED to save amplitude(t) then RE-RUN n=384/512
  (GH200 hours). (medium, ROMEO)
- [ ] **PR4 final high-resolution validation table**: n=384 + n=512 O5 with the re-fitted windows;
  table of l=3/4/5 rates vs paper + convergence figure. (medium, ROMEO; depends on PR3 + fidelity decision)
- [ ] **PR5 (optional) order-2 Strang splitting**: half-step source / transport step / half-step source
  (currently order-1 Lie). Opt-in, Lie default bit-identical. (medium; depends on decision)
- [ ] **PR6 (research, conditional) Path A polar-fluid**: ONLY if a clean 2D density figure
  (Fig 5.1 type) is judged indispensable. Polar fluid flux (1/r curvature) + ITERATIVE polar
  elliptic for the crossed tensor + polar Schur stencils. 2-3 research PRs. (research)

## OWNER DECISIONS (June 2026)

- Q1 (clean 2D figure): **YES required** => Path A (polar fluid) PURSUED (no longer optional). Step 1
  = #209 (polar fluid transport, in CI). Step 2 = polar Schur (to be done).
- Q2 (fidelity target): **l=4/5 to +-2%** => re-fit early window (ROMEO job 647356) + update PAPER_FIT_WINDOWS
  + final validation table (re-run if needed).
- Q3 (formal FE structure-preservation vs empirical O(dt^2) tests): OPEN.
- Q4 (Lie vs Strang): OPEN (Strang = optional PR PR5).

PR status: PR1 conservation = #207 (merged). doc CONSERVATION_SUMMARY = #208 (CI). Path A step 1 = #209 (CI).
