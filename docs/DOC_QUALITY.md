# Documentation quality policy

This document sets the documentation quality policy for adc_cpp: how the docs are
classified, which tooling decisions were taken and why, and how the docs are kept from
drifting away from the code.

Guiding principle: the documentation describes the real code. A page that promises an
option, a target or a behavior that does not exist is a bug, just like a false test. Every
mechanism below exists to keep the docs close to the code and to flag the gap when it
appears.

The user-facing docs are written in English (en-US); translation conventions and the
French-to-English glossary live in [TRANSLATION_GLOSSARY.md](TRANSLATION_GLOSSARY.md).

Linear tracking: the milestone "Documentation, freshness and automated quality" (anchor
issue ADC-146). The tooling bricks named here are summarized in the status table at the
bottom.

## Three documentation classes

Every page belongs to a class. The class sets the source of truth and the control mode.

For a reader, the canon is the published Sphinx site (`docs/sphinx`) and, for the API, the
generated reference (class A, extracted from the code). The root `docs/*.md` carry intent and
rationale (class C) or a past design state (class D); they are authoritative for *why*, not for
the current code's exact behavior, which class A always reflects.

### A. Generated reference

The source of truth is the code. The page is produced by extraction; it is not proofread by
hand, it is rebuilt.

- `docs/sphinx/reference/python-api.md`: autodoc of the `adc` module (signatures, docstrings).
- `docs/sphinx/reference/cpp-api.md` plus the Doxygen site published under `/cpp/`: the C++
  API extracted from the headers.

Control: freshness is guaranteed by rebuild. If a signature changes in the code, the page
changes at the next build; there is nothing to acknowledge by hand.

### B. Testable tutorials

The source of truth is a script run in continuous integration. The text never includes
copied code: it pulls the script in via `literalinclude`, so an example that breaks breaks
the build.

- `docs/sphinx/getting-started/installation.md`
- `docs/sphinx/getting-started/first-run.md`
- `docs/sphinx/getting-started/tutorial.md`
- `docs/sphinx/reference/symbolic-dsl.md`
- `docs/sphinx/reference/native-bricks.md`

Control: the associated script runs in CI (smoke mode, see below) and the displayed fragment
is included from that same script. The code on the page and the tested code are the same
file.

### C. Human guides

The source of truth is the author's intent: architecture, choices, algorithms, design notes.
These pages carry a judgment that the code does not hold and are not generated.

- `docs/ARCHITECTURE.md` (layers, modules, AMR)
- `docs/ALGORITHMS.md` (methods, formulas)
- `docs/CHOICES.md` (deliberate trade-offs)
- `docs/BACKEND_COVERAGE.md` (backend and test matrix)
- design notes for in-progress or recently changed work (architecture rationale). Once a
  design note's feature has shipped, it moves to class D below.

Control: human review plus a freshness check. Since nothing is regenerated here, freshness
is tracked by an index (docmap) that raises a warning when a class-C page has not been
reviewed since a given window.

### D. Historical and archived

Some pages describe a PAST state: a design note written before the feature landed, or a closed
audit. They are kept for provenance, not as current truth. The current behavior is the code and
its class-A reference; a class-D page is authoritative only for the intent it recorded at the
time.

- delivered design notes: `docs/AMR_MULTIBLOCK_DESIGN.md`, `docs/SCHUR_CONDENSATION_DESIGN.md`,
  the older `*_DESIGN.md`.
- closed audits and roadmaps: `docs/DOC_REFONTE_AUDIT.md`, `docs/CODEBASE_AUDIT.md`,
  `docs/PERF_SCALING_TODO.md`, `docs/PERFORMANCE.md` (explicitly historical).

Control: the page carries a banner at the top (for example `STATUS: ... read it as design
history`) and stays out of the active Sphinx toctree. When a design note's feature ships, move
it here rather than leaving a future-tense body that contradicts the code.

## Architecture decisions (ADR)

The choices below are settled. Each entry gives the decision, the reason, and the rejected
alternative.

### Doxysphinx rather than Breathe or Exhale

Decision: the C++ API is rendered by Doxygen, then embedded into the Sphinx site by
Doxysphinx. The raw Doxygen site stays published as-is under `/cpp/`.

Reason: adc_cpp is heavy in C++23 (concepts, templates, functors). Breathe and Exhale replay
the C++ parsing on the Python side and quickly drift on these constructs; Exhale is lightly
maintained. Doxysphinx starts from the HTML already produced by Doxygen, so the rendering
follows what Doxygen can read, with no second parser to keep up to date.

Rejected alternative: Breathe and Exhale (rot on C++23 concepts and templates, low
maintenance of Exhale).

### Central docmap rather than per-file frontmatter

Decision: the freshness metadata (class, review date, test script) lives in a single index,
`docs/docmap.toml`, and not in a header at the top of each page.

Reason: a single index is read, validated and audited in one place; it avoids editing every
page to change the policy, and it keeps the pages clean for the Sphinx and Doxygen renders.

Why TOML: the parser is `tomllib`, part of the Python 3.11+ standard library. The doc check
therefore runs without any dependency to install. The scoping issues mentioned YAML, but the
zero-dependency constraint wins; the deviation is deliberate.

Rejected alternative: per-file frontmatter (scattered metadata, page-by-page edits) and YAML
(external dependency to install just to read the index).

### Examples run in smoke mode

Decision: the tutorial scripts run in CI with a `--quick` flag; the exit code (zero) is
checked, with no assertion on the physics.

Reason: the goal is to prove that the example imports, compiles and runs end to end, not to
re-validate the solver (the core test suite covers that). Smoke mode reduces the resolution
and the number of steps to stay within the time budget of a doc CI.

Rejected alternative: numerical assertions in the tutorials (duplicate of the test suite,
fragile, slow).

### Linters kept: linkcheck, codespell, markdownlint

Decision: the documentation lint chain adds the Sphinx linkcheck, codespell and markdownlint,
on top of `docs/check_docs.py` already in place.

Reason: these three tools cover faults that `check_docs.py` does not target (dead links,
typos, Markdown structure). They are complementary, not redundant.

Rejected alternative: Vale. Its value (style and forbidden terms) overlaps what
`check_docs.py` already does for this repo (ASCII strict for Sphinx pages, em-dash, false
terms), with an extra configuration cost. Vale is rejected to avoid the duplicate.

### CI cadence: light PR lane and weekly heavy lane

Decision: two distinct paths.

- PR lane: light, triggered by a path filter (only when the docs change), with no
  compilation of the module or the C++. It runs the lint and the fast checks.
- Heavy lane: weekly cron on Sunday, plus manual dispatch, plus push on `master`. It builds
  the module for autodoc, Sphinx, Doxygen and the examples.

Reason: the vast majority of PRs do not touch the docs; forcing a full doc build on them
would be wasteful. The heavy work goes on a cadence where time is not critical.

Informative-first policy: at the start, these paths report the gaps (annotations, job
summary) without blocking the PRs. A check is switched to blocking only once the base is
clean.

## Updating the docs

### When to edit docmap.toml

Edit `docs/docmap.toml` when you:

- add, move or remove a doc page (update its entry and its class);
- have just reviewed a class-C page (update its review field, see below);
- attach a new test script to a class-B page (`tested_by` field).

Do not touch docmap for class-A pages: their freshness comes from the rebuild, not from the
index.

### Reading and acknowledging a freshness warning

The freshness check compares the review date of a class-C page to a window. Beyond it, it
emits a warning (non-blocking by default). To acknowledge it:

1. reread the page and fix what drifted from the code;
2. update the `reviewed` field of the corresponding entry in `docs/docmap.toml`;
3. rerun the check; the warning disappears.

Acknowledging means having reread, not just having changed the date. The `reviewed` field is
an attestation of review.

### Adding a testable tutorial

1. write a standalone script that accepts `--quick` (smoke pass, reduced resolution and
   steps) and `--outdir DIR` (where to write the figures), modeled on
   `docs/sphinx/tutorials/diocotron_tutorial.py`;
2. show the fragments on the page via `literalinclude` of the script, never by hand-copying;
3. declare the script in the `tested_by` field of the page's docmap entry, so the example
   harness picks it up;
4. check locally that the `--quick` pass exits zero.

### Local commands

```bash
python docs/check_docs.py                       # documentation lint (em-dash, ASCII, links, terms)
bash scripts/build_docs.sh --sphinx             # lint plus Sphinx only (fast iteration)
python docs/run_doc_examples.py                 # runs the tutorial scripts in smoke mode
cmake --build --preset serial --target docs     # full doc build via the CMake target
```

## Status

The doctrine requires stating what exists and what is delivered. This table is the real state.

| Capability | State today | Delivered by |
| --- | --- | --- |
| Documentation lint | present: `docs/check_docs.py` | already there |
| Unified doc build | present: `scripts/build_docs.sh` | already there |
| Opt-in Pages publication | present: `.github/workflows/docs.yml` | already there |
| literalinclude tutorial with `--quick` | present: `docs/sphinx/tutorials/diocotron_tutorial.py` | already there |
| Central docmap index | present: `docs/docmap.toml` | ADC-147 (merged) |
| `--freshness-warn-only` and `--selftest` flags | present in `check_docs.py` | ADC-147 (merged) |
| Example harness `run_doc_examples.py` and `tested_by` | present: `docs/run_doc_examples.py` | ADC-148 (merged) |
| Embedded Doxygen (Doxysphinx) under `docs/sphinx/doxygen/` | present | ADC-149 (merged) |
| Linters linkcheck, codespell, markdownlint | present | ADC-150 (merged) |
| Light PR lane and weekly cron lane (`docs-pr.yml`) | present | ADC-151 (merged) |

The tooling above is merged on `master`. This document defines the contract those issues
implemented.
