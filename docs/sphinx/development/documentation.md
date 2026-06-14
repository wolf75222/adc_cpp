# Documentation style guide

This page is how you write documentation for adc_cpp. Follow the project style first (this
page), then the [Google developer documentation style guide](https://developers.google.com/style)
for anything not covered here. The freshness and tooling policy (doc classes, the docmap, the CI
lanes) lives in [DOC_QUALITY](https://github.com/wolf75222/adc_cpp/blob/master/docs/DOC_QUALITY.md);
this page does not repeat it.

## Documentation is code

Treat a doc page like a source file. It lives in the repository, not in an external wiki. It
changes in the same pull request as the code it describes. It is reviewed, it has an owner and a
freshness entry in `docs/docmap.toml`, and it is fixed through Linear when it is wrong.
`docs/check_docs.py` is the lint and `sphinx -W` is the build test, both run in CI.

## Write for one audience

Identify the primary reader before you write: a first-time user, a scientific user, a model
developer, a backend or build developer, or a maintainer. Write for that reader. Keep user
documentation (getting started, tutorials, how-to, concepts, reference) apart from maintainer
documentation (development). A user who wants to run a case must not have to read internal design
notes first.

## One type per page

Each page has a single type. Do not mix them.

- Landing page: orientation and links only. A section index or the README is a traffic cop, not
  content.
- Tutorial: step-by-step learning that makes something real happen.
- How-to guide: one specific task, assuming prior knowledge.
- Concept: mathematical, physical or numerical understanding, not exhaustive syntax.
- Reference: exact syntax, defaults, options, APIs and constraints.
- Development: architecture, design decisions, invariants, testing and maintenance.

## Canonical sources

Each topic has one canonical page. Every other page links to it instead of copying it. The target
locations are:

| Topic | Canonical source |
|---|---|
| Backend support | `reference/backend-matrix.md` |
| CMake options | `backends/cmake-options.md` |
| Run interface | `reference/cli.md` |
| Case format | `reference/case-format.md` |
| Symbolic DSL | `reference/symbolic-dsl.md` |
| Native bricks | `reference/native-bricks.md` |
| C++ API | public headers and Doxygen |
| Python API | the pybind11 bindings and `reference/python-api.md` |
| Numerical validation | `development/numerical-validation.md` |
| Limitations | `reference/known-limitations.md` |

## Writing rules

- Be prescriptive. Give one standard path, not twelve equivalents. For a first local run, tell the
  reader to use the Kokkos Serial backend; put OpenMP, MPI and CUDA in the backend and how-to pages.
- Use `must`, `can`, `might`, or an imperative. Avoid `should`, which is ambiguous.
- Be timeless. Describe the current behavior, not history or promises. Avoid `currently`, `new`,
  `latest`, `soon`, `does not yet`. Write "The CUDA backend targets ROMEO GH200", not "is currently
  only supported on GH200".
- No vague claims. Avoid `fast`, `best`, `always`, `never`, `guarantee` unless a sourced number
  backs them. A performance claim names the case, the machine and the numbers and links to the
  numerical-validation page.
- Define jargon on first use. Keep the domain terms (`PhysicalModel`, native brick, symbolic DSL,
  dispatch seam, elliptic right-hand side, Schur stage, Strang splitting, reflux, prolongation,
  restriction, Berger-Rigoutsos); define each one the first time it appears, or in a concept page.
- Headings in sentence case. A task page starts with a verb ("Write a model", "Run a case",
  "Configure a simulation"); a concept page uses a noun phrase ("Time integration", "Poisson
  equation").
- Commands are copy-pasteable: one command per block, the fewest options, no `[]`, `{}`, `|` or
  `...` in a command the reader is meant to copy. Put variants in a reference table.

  ```bash
  cmake --preset serial
  cmake --build --preset serial
  ctest --preset serial
  ```

- Placeholders in `UPPER_CASE`, explained on first use. Write `cmake --preset PRESET`, then "Replace
  `PRESET` with one of serial, python, mpi, parallel".
- Introduce each code block with a sentence. Mark omitted code with a language comment, not `...`.
- Use code font for filenames, classes, methods, constants, types, CLI options and terminal output:
  `PhysicalModel`, `ADC_USE_KOKKOS`, `CMakePresets.json`.
- Use descriptive link text, never "here" or a bare URL. Link sparingly; do not duplicate a link.
- Notices are rare. Reserve a warning for a strong or irreversible risk (an experimental backend, an
  unsupported configuration, a non-reproducible result, a normalization change, overwriting an
  output). Do not annotate every detail.
- Accessibility: hierarchical headings, short sentences, alt text that describes a figure (the data
  flow, not "see the diagram"), never code or terminal output as an image, never information by color
  alone.
- Math and units: split a numerical page into continuous model, nondimensionalization,
  discretization, boundary conditions, numerical flux, time integration, and validation. Write
  quantities clearly (`n_e = 1e17 m^-3`, `B = 5.7 T`). Use MyST math (`$...$` or a math block); keep
  the source ASCII.

## API reference (Doxygen)

The public headers are the source of truth for the C++ API. `reference/cpp-api.md` points to the
Doxygen output and gives reading guidance; it does not re-list the API by hand. Document every
public class, method, parameter, return value and exception. The first sentence of a class or method
comment is short, unique and useful, because the generator extracts it as the summary. Separate API
comments (how to use it) from implementation comments (the technical why).

## Page checklists

A tutorial includes: prerequisites, the exact commands, the exact case path, the expected output
files, the expected terminal output, the approximate runtime if known, a troubleshooting note, and
the next page to read. Number only user actions, not system behavior.

A concept page includes: the problem, the main idea, the formulation, the numerical treatment, how
adc_cpp implements it, the limitations, and links to the related reference pages.

A reference page includes: syntax, fields or options, defaults, allowed values, constraints,
examples, and related pages. Keep design rationale out of reference; put it in a development page.

A development page includes: the purpose, the relevant files, the internal design, the invariants,
the alternatives considered, the testing and numerical-validation requirements, and a review
checklist.

## Reviewing documentation

A doc change benefits from three angles: technical (accuracy, by someone who knows the code),
audience (clarity, by a newcomer), and writing (consistency). At least one reviewer is required.
Before merging, confirm the change tests its commands, has one audience and one purpose, replaces
duplicated content with links to the canonical page, and sources any numerical claim.

## Freshness and deprecation

Every page has an entry in `docs/docmap.toml` (owner and freshness). When a page is obsolete, do not
leave it in place: delete it, or mark it clearly at the top ("This page is obsolete. Use the X page
instead.").

## Do not merge a doc page if

- it describes unsupported behavior;
- it duplicates reference information instead of linking to the canonical page;
- it mixes tutorial, concept, reference and development content;
- a command is not copy-pasteable;
- a tutorial has no expected output;
- a numerical claim is not sourced or validated;
- a limitation is hidden in prose instead of stated explicitly.
