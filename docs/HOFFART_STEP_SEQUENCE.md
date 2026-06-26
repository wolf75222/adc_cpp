# PoPS macro-step sequence (real, as wired on master)

This is the ACTUAL order the engine runs, verified from `system_stepper.hpp`, `amr_runtime.hpp`, `condensed_schur_source_stepper.hpp`, and `system_field_solver.hpp`. Both paths are LIE (Godunov) splits, NOT Strang. `strang_step` (splitting.hpp:29) is dead code, never called by any orchestrator.

## Headline finding: the phi / Gauss-restart data flow

Question: does a post-Schur `solve_fields()` overwrite the Schur-produced phi within the same macro-step?

Answer: NO within the same step (there is exactly ONE solve_fields per step, at the top, BEFORE transport). BUT the NEXT macro-step opens with `solve_fields()`, whose `ell_solve()` (system_field_solver.hpp:438) re-solves the bare electrostatic Poisson on rho^{n+1} and re-derives aux comp0/1/2 = (phi, grad phi) at system_field_solver.hpp:456-458, OVERWRITING the Schur-evolved phi^{n+1}. The Schur phi survives only as the BiCGStab/MG warm-start seed.

Consequence: the Lorentz/source-coupled phi evolution of paper eq (3.2) (d_t(-Lap phi) = -alpha div m) is functionally INERT for the trajectory: the phi that next step's transport reads (grad phi in aux) is the re-solved electrostatic potential of rho^{n+1}, not the Schur phi. No transport runs between the Schur stage and that overwrite, so phi-hat never drives a transport sub-step. This is a once-per-step Poisson re-imposition (Gauss-law cadence OncePerStep), an architectural divergence from the paper's source-evolution of -Delta phi, though the paper's REFERENCE runs use "no restart" too. Verified: copy_comp0(phi, op_.phi()) at condensed_schur_source_stepper.hpp:266 writes phi^{n+theta} into ell_phi(); SchurExtrapolateScalarKernel at :286-289 extrapolates in place to phi^{n+1}; system_stepper.hpp:110 is the only solve_fields() call in step().

## System (system-schur) macro-step

```
sequenceDiagram
  participant Py as run.py sim.step(dt)
  participant St as SystemStepper::step (system_stepper.hpp:104)
  participant SF as solve_fields (system_field_solver.hpp:438 ell_solve)
  participant Ell as ell_phi() / aux comp0-2
  participant Adv as s.advance (SSPRK3, S=0)
  participant Sc as CondensedSchurSourceStepper::step (theta=0.5)
  Py->>St: step(dt)
  St->>SF: solve_fields()  [ONCE, top, system_stepper.hpp:110]
  SF->>Ell: ell_solve(): Lap phi = Sum elliptic_rhs(rho^n) -> phi^n; derive aux=(phi^n, grad phi^n) lines 456-458
  St->>Adv: advance(U, dt, substeps)  [transport rho^n,m^n -> rho-hat,m-hat; phi^n FROZEN in aux; paper (3.1) d_t Lap phi = 0; S=0 so source not double-advanced, model.py m.source([0,0,0])]
  St->>Sc: run_source_stage(U, ell_phi()=phi^n, dt)  [system_stepper.hpp:116]
  Sc->>Sc: freeze phi^n (copy_comp0 phi_n_, :237); assemble A=I+c rho B^-1 + condensed RHS; BiCGStab solve L_int phi = -rhs_schur (:265)
  Sc->>Ell: phi <- phi^{n+theta} (copy_comp0, :266)
  Sc->>Sc: reconstruct v^{n+theta} = B^-1(v^n - theta dt grad phi^{n+theta}) (:273-280); extrapolate phi,v -> n+1 (:286-296)
  Sc->>Ell: phi := phi^{n+1}; mom := rho^n v^{n+1}; (E update if role); fill ghosts (:308-311)
  St->>St: apply_couplings(dt) [no-op, single block]; t += dt; ++macro_step (system_stepper.hpp:118-120)
  Note over Ell: phi^{n+1} sits in aux but is OVERWRITTEN by next step solve_fields ell_solve
  Py->>St: step(dt) [n+1]
  St->>SF: solve_fields() RE-SOLVES bare Poisson on rho^{n+1} -> discards Schur phi^{n+1} (warm-start only)
```

SSPRK selected: system-schur runs SSPRK3 (run.py:147 ssprk3=True -> block_builder.hpp:141 SSPRK3Step, 3-stage Shu-Osher, time_steppers.hpp:57-75). Source advanced ONCE per macro-step: the transport carries S=0 (model.py schur variant) precisely so the electrostatic/Lorentz source is advanced exactly once in run_source_stage.

Polar path: PolarCondensedSchurSourceStepper is structurally identical (assemble A, PolarTensorKrylovSolver, reconstruct, extrapolate, copy_comp0 phi at :407, extrapolate :423-426), reached through the SAME run_source_stage (system_stepper.hpp:92-96). Mono-rank guarded (throws if n_ranks > 1, line ~312). Same once-per-step phi overwrite via solve_fields_polar.

## AmrSystem (amr-imex) macro-step

```
step(dt) [amr_runtime.hpp:589]:
  [opt-in regrid()  amr_runtime.hpp:598, only if regrid_every_>0 && macro_step_>0; off by default]
  solve_fields()    [amr_runtime.hpp:603, ONCE, top: mg_.solve -> phi^n, fill_ghosts, field_postprocess -> aux[0]=(phi^n,grad phi^n)]
  for each block:    [amr_runtime.hpp:604-624]
    substeps x imex_advance:
       mf_advance_faces      [single forward-Euler flux-divergence transport, amr_subcycling.hpp:375]
       mf_apply_source_treatment [cell-local backward-Euler stiff source = full -rho grad phi + omega m, amr_subcycling.hpp:376]
  coupled_source_step(dt)  [amr_runtime.hpp:628, no-op for hoffart]
  ++macro_step             [amr_runtime.hpp:629]
  phi re-solved only at next step top
```

amr-imex specifics: NO CondensedSchur, NO Poisson re-solve after the source, NO SSPRK (the AMR engine has no method parameter; transport is first-order forward-Euler regardless of the Python Explicit(method=) requested). Source advanced once per SUBSTEP (= substeps times per macro-step), a deliberate sub-cycled Lie split. amr-imex also uses ZERO initial momentum (run.py:456). It is the same PDE but not the paper Schur stage or initial drift.

## Half-step hooks (for any future Strang)

No half-step entry point is exposed by either runtime. SystemStepper::step and AmrRuntime::step advance only a FULL dt; AdvanceExplicit::operator()(U,dt,n) splits dt into n EQUAL substeps, not a single half-step. The building blocks support arbitrary dt/theta (CondensedSchurSourceStepper::step(...,theta,dt) takes any dt), so a Strang composition is constructible, but wiring it requires a NEW orchestration loop in SystemStepper AND a fix to the phi overwrite: for a Strang scheme that feeds the post-source phi-hat into a second transport half-step, the once-per-step solve_fields re-solve is fatal because no transport currently runs between the Schur stage and that overwrite.
