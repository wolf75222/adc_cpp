# Conformance Audit -- specifications & requirements (ADC-188)

Date: 2026-06-15. Scope: confront the design specs (ARCHITECTURE.md, CHOICES.md,
ALGORITHMS.md) and the C++ contract headers against the actual code in
`System`/`SystemStepper`, `AmrSystem`/`AmrRuntime`, `SystemFieldSolver`, the
elliptic solvers (GeometricMG, PoissonFFTSolver, PolarPoissonSolver,
TensorKrylovSolver), the reflux path, and the Python facade/DSL.

Method (per AUDIT.md): findings land here and as Linear issues; NO mass
correction without a documented decision; the core stays model-agnostic and
device-safe.

## Verdict

Most contracts hold exactly. The step-ordering invariant
(solve_fields -> advance -> run_source_stage -> apply_couplings -> t += dt ->
++macro_step) matches the SystemStepper contract header. The explicit guards
promised under "Limitations" all exist and throw clear errors: FFT refused under
MPI on all ranks; polar Poisson single-rank/single-box; FFT incompatible with
wall/eps/anisotropy/kappa; no AMR global Schur. The model-agnostic guarantee
holds: `include/pops/**` carries no scenario-keyed branch -- scenario names
("diocotron", "Hoffart") appear only in explanatory comments and doc cross-refs,
not in logic. `dsl.py` error handling is careful (explicit RuntimeErrors with
remedies, ABI-mismatch pre-dlopen guards).

Two genuine spec<->code divergences were found.

## Findings

| Sev | Where | Issue |
|-----|-------|-------|
| Medium | python/bindings/amr/amr_system.cpp:438-442; include/pops/runtime/amr_runtime.hpp:84; include/pops/runtime/amr_system.hpp (contract comment); docs/ARCHITECTURE.md:220; docs/ALGORITHMS.md:1936 | Design docs and header contracts still claim multi-block AMR + `regrid_every>0` is REFUSED, but it is implemented and tested. |
| Medium | include/pops/numerics/elliptic/geometric_mg.hpp:378-391 (solve), 429-457 (solve_robust); include/pops/runtime/system_field_solver.hpp:385-397 (ell_solve calls the void solve()); include/pops/runtime/amr_runtime.hpp:504 (mg_.solve()) | Elliptic Poisson non-convergence is a silent failure: solve() returns best-effort, every caller discards the status, no facade reporting. |

### F1 -- stale multi-block + regrid_every refusal claim

ARCHITECTURE.md:220 states the multi-block with `regrid_every > 0` is refused
("hierarchie figee") and ALGORITHMS.md:1936 says the same. Two C++ contract
comments repeat it (amr_runtime.hpp:84 "facade runtime (AmrSystem) REFUSE
explicitement multi-blocs + regrid_every > 0", and the amr_system.hpp header
contract). But python/bindings/amr/amr_system.cpp:438-442 explicitly UNLOCKED it ("capstone
Phase 2, C.6 ... EST DESORMAIS SUPPORTE ... L'ancien REFUS ... est leve"):
build_multi wires set_regrid plus per-block union tag predicates, and
python/tests/test_amr_multiblock.py asserts the path is accepted and runs
several regrids. A user reading the design docs would believe a working, tested
capability is unavailable; a maintainer reading the header contract is actively
misled.

Action: rewrite ARCHITECTURE.md:220 and ALGORITHMS.md:1936 to state the support
via tag-union regrid, and fix the now-false header comments at
amr_runtime.hpp:84 and the amr_system.hpp contract. Documentation-only;
trivial risk.

### F2 -- silent elliptic Poisson non-convergence

GeometricMG::solve(rel_tol, max_cycles) returns max_cycles when it never reaches
the stopping criterion (it does not throw); solve_robust() returns its total as
a silent best-effort even after exhausting smoothing escalation. Every
runtime/coupler caller discards the result -- ell_solve()
(system_field_solver.hpp) calls the void solve() overload; amr_runtime.hpp:504,
system_coupler.hpp, amr_system_coupler.hpp likewise. No poisson/field
convergence report is exposed on the System or AmrSystem facade. This
contradicts the ARCHITECTURE.md "Limitations" framing that guards "raise a clear
error rather than drift silently", and is inconsistent with the Newton path
(newton_report exposes converged/max_residual/n_failed) and the Krylov path
(KrylovResult.converged via last_solve()), which at least surface a diagnostic.
The GeometricMG comment itself documents that high-resolution embedded-boundary
runs can diverge to NaN through the warm-started phi -- exactly the case a user
would want signalled, yet there is no signal and no query.

Action: surface the elliptic convergence status -- record final residual + a
converged flag in solve()/solve_robust() and expose a queryable diagnostic on
the facade (mirror newton_report, e.g. poisson_report() =
{converged, final_residual, cycles_used}). Optionally add an opt-in strict mode
that throws on non-convergence (off by default to preserve bit-identical
behavior). At minimum, document the best-effort/silent behavior as an explicit
Limitation so the "raise rather than drift" claim is not overstated. Low risk.

## Conclusion

ADC-188 is satisfied as an audit: the conformance surface is now documented, the
model-agnostic and device-safe guarantees are verified intact, and the two
divergences are tracked as Linear issues for a documented decision. No mass
correction is applied here.