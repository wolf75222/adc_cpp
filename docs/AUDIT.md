# Review and audit method

This page is the frame for the code review and audit work on `adc_cpp`: how a review is run,
with which tools, and where the findings land. The per-axis reports listed below are the
output; this page is the method that produces them.

Linear tracking: milestone *Revue & audit qualite du code*, frame issue **ADC-187**.

## Tools (Claude Code skills)

Each axis has a dedicated skill. A reviewer picks the one that matches the concern rather than
eyeballing everything at once.

| Skill | Axis | Use for |
| --- | --- | --- |
| `/code-review` | correctness | bugs in the diff, plus reuse / simplification / efficiency cleanups (effort low to ultra; `--comment` to annotate a PR, `--fix` to apply) |
| `/cpp-review` | C++ idioms | conformance to the language idioms and best practices |
| `/csapp-cpp-perf` | performance | CPE, cache misses, the critical path, allocations |
| `/simplify` | volume | code reduction and factorization (quality, **not** a bug hunt) |

These complement, they do not replace, the automated layer (`ci.yml` gate, `quality.yml`,
`no-ai-authors.yml`, `docs/check_docs.py`): the skills are for what a tool cannot judge.

## Findings tracking

- Every non-trivial finding becomes a Linear issue (or sub-issue) with the `audit` label,
  plus `refactor` / `performance` / `validation` according to the axis.
- **No mass correction** (style, refactor, performance) without a documented decision. The
  conventions actually adopted are recorded in
  [CODING_STANDARDS_DECISIONS.md](CODING_STANDARDS_DECISIONS.md) (ADC-124, ratified by ADC-219).
- Cross-cutting guardrail: the `adc_cpp` core stays model-agnostic. No correction introduces
  model logic into `include/adc/**`, and numerical parity (device-safe / Kokkos) is preserved.

## Severity

`High` = a real bug, a broken invariant, or an undefined behavior; fix or explicitly dismiss
in the PR with the reason. `Medium` = a maintainability or consistency issue worth an issue.
`Low` = a note, no action required. A review that finds nothing of `Medium` or above on a
substantial change is itself a signal worth recording.

## Per-axis reports (index)

| Report | Axis | Linear |
| --- | --- | --- |
| [CODEBASE_AUDIT.md](CODEBASE_AUDIT.md) | maintainability of the codebase | ADC-189/190/191 |
| [STYLE_CONFORMANCE_AUDIT.md](STYLE_CONFORMANCE_AUDIT.md) | conformance to C++ standards | ADC-124 |
| [COMMENTS_AUDIT.md](COMMENTS_AUDIT.md) | accuracy of comments | ADC-125 |
| [PERF_SCALING_FRONTENDS_AUDIT_2026-06-08.md](PERF_SCALING_FRONTENDS_AUDIT_2026-06-08.md) | performance and scaling | ADC-193 |
| [TOOLCHAIN_ROBUSTESSE_AUDIT_2026-06-10.md](TOOLCHAIN_ROBUSTESSE_AUDIT_2026-06-10.md) | toolchain and install robustness | toolchain hardening |
| [CONFORMANCE_AUDIT.md](CONFORMANCE_AUDIT.md) | spec & requirement conformance | ADC-188 |
| [STL_BOOST_AUDIT.md](STL_BOOST_AUDIT.md) | STL idiom & Boost decision | ADC-192 |
| [DOC_REFONTE_AUDIT.md](DOC_REFONTE_AUDIT.md) | documentation truth matrix | ADC-146 |
| [CODING_STANDARDS_DECISIONS.md](CODING_STANDARDS_DECISIONS.md) | adopted conventions (D1-D15) | ADC-124/219 |

## Per-axis status (ADC-188..194)

- ADC-188 -- Conformance to specs & requirements: NEW report [CONFORMANCE_AUDIT.md](CONFORMANCE_AUDIT.md). Contracts mostly hold (step ordering, explicit guards, model-agnostic core); two real spec<->code divergences: stale "multi-block AMR + regrid_every>0 is REFUSED" claim (now implemented/tested) and silent elliptic-Poisson non-convergence.
- ADC-189 -- Flexible architecture: COVERED by [CODEBASE_AUDIT.md](CODEBASE_AUDIT.md) (+ GENERICITY_2026-06.md). Bricks are unit-testable; the prior P0 god-class is substantially resolved (Lot B). Two Low post-snapshot watch-items tracked as issues.
- ADC-190 -- Anti-patterns (duplication etc.): PARTIALLY COVERED by [CODEBASE_AUDIT.md](CODEBASE_AUDIT.md) (pre-dates the polar Schur stepper / multi-box theta path). Two net-new Medium duplications tracked as issues.
- ADC-191 -- Simplification / LOC: PARTIALLY COVERED by [CODEBASE_AUDIT.md](CODEBASE_AUDIT.md) (coarse extraction, now done). Two net-new Low behavior-preserving factor-outs tracked as issues.
- ADC-192 -- STL & Boost: NEW report [STL_BOOST_AUDIT.md](STL_BOOST_AUDIT.md). Boost = NO (0 hits). STL host code is clean; 3 Low sites tracked. TOOLCHAIN_ROBUSTESSE is build-robustness only, no STL/Boost content.
- ADC-193 -- Performance: COVERED at campaign altitude by [PERF_SCALING_FRONTENDS_AUDIT_2026-06-08.md](PERF_SCALING_FRONTENDS_AUDIT_2026-06-08.md). Four net-new code-level hotspots (need before/after measurement) tracked as issues.
- ADC-194 -- Test coverage gaps: tracked via Linear issues; BACKEND_COVERAGE.md is a backend x test matrix, not a functional-gap analysis. Highest risk: MPI IO/checkpoint gather has no multi-rank test (dangling cross-ref to a nonexistent C++ test, verified); positivity floor tested only single-grid serial; matrix stale.

## Adding an audit

Run the matching skill on the target, write the findings into the per-axis report (severity,
file, finding, status), open an issue per non-trivial finding, and link the report here. Keep
the report a record, not a backlog: once a finding has an issue, the issue is the source of
truth for its resolution.
