<!-- Title: PoPS-NN short imperative description. One Linear issue = one PR. -->

Fixes PoPS-NN

## What / why

<!-- What changed and, above all, WHY it was needed. State the assumptions. -->

## How

<!-- Key implementation points; what was preserved (defaults, API). -->

## Validation

<!-- Tick what was actually run; paste the commands, not "tests pass". -->

- [ ] CI preset(s) run locally: `serial` / `python` / `mpi` / `parallel` (state which)
- [ ] `ctest --preset <x>` green (or the relevant Python tests)
- [ ] Tests added or updated for this change (or say why none)
- [ ] No regression on the touched scope

## Numerical validation (solver / flux / Poisson / AMR / backend / DSL only)

<!-- Remove this block for non-numerical PRs. Without it a reviewer cannot tell a
     normal difference from a silent model change. -->

- Reference case:
- Observed quantity:
- Expected value:
- Tolerance (and reason):
- Measured difference:

## Not verifiable locally

<!-- GPU / GH200 / multi-node MPI: anything this machine cannot check. Say so. -->

## Docs and versioning

- [ ] User docs updated if behavior or API changed (README, docs/sphinx, docmap)
- [ ] `CHANGELOG.md` [Unreleased] entry for a notable change
- [ ] `docs/check_docs.py` green (ASCII strict, no em-dash) if docs were touched

## Suggested review order (multi-file PRs)

<!-- Guide the reviewer instead of leaving them to wander, e.g.:
     1. include/pops/numerics/flux.hpp   (the new flux)
     2. include/pops/runtime/system.cpp  (dispatch wiring)
     3. python/bindings.cpp             (Python surface)
     4. tests/test_flux.cpp             (regression)
     5. docs/sphinx/reference/bricks_reference.md -->

## Risks / attention

<!-- Numerical tolerances, expected drift, safeguards, follow-up. -->

<!-- Before squash-merge: no AI author/committer/co-author in master..branch
     (no-ai-authors guard); delete the source branch on merge. -->
