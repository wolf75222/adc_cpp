# Contributing to adc_cpp

`adc_cpp` is the header-only C++23 core of the ADC solver, with its Python bindings
(pybind11), its DSL path and its CMake packaging. This guide summarizes the workflow; the
technical detail lives in the [README](README.md) and in
[docs/DOC_QUALITY.md](docs/DOC_QUALITY.md).

## Build and tests (CMake presets)

The build is driven by the presets in `CMakePresets.json`, not by ad-hoc `-D` flags. The
`adc` conda env (Python 3.12) must be active for the `python`, `parallel` and `mpi` presets.

```bash
bash scripts/setup_env.sh && conda activate adc   # env + pinned toolchain

cmake --preset serial   && cmake --build --preset serial   && ctest --preset serial
cmake --preset python   && cmake --build --preset python    # _adc module (bindings)
cmake --preset mpi      && cmake --build --preset mpi      && ctest --preset mpi
cmake --preset parallel && cmake --build --preset parallel && ctest --preset parallel
```

The CI presets (`ci-serial`, `ci-python`, `ci-mpi`, `ci-kokkos`, `ci-kokkos-python`,
`ci-bench`) mirror `.github/workflows/ci.yml`: match your flags to a CI job rather than
inventing new ones. The GPU / GH200 paths cannot be validated outside ROMEO: say so
explicitly in the PR.

## Documentation

`bash scripts/build_docs.sh` builds the whole site (lint + Sphinx + Doxygen + doxysphinx).
The freshness policy (docmap, CI lanes) is described in
[docs/DOC_QUALITY.md](docs/DOC_QUALITY.md). The light PR lane (`docs-pr.yml`) compiles
nothing; the full build runs on the weekly cron and on manual dispatch.

## Workflow

- **Linear** is the source of truth for tasks: one `ADC-NN` issue = one PR.
- Branch: `adc-<n>-short-description`. PR title: `ADC-<n> Description`. PR body:
  `Fixes ADC-<n>`.
- `master` is the default branch; never commit directly to it. Deliver through a branch or
  an isolated `git worktree` off `master`.
- Minimal diffs, scoped to the issue; no incidental reformatting.

## Guardrails

- **No AI author, committer or co-author** (Claude, Copilot, Anthropic, ...) anywhere in
  the history: the `no-ai-authors.yml` workflow rejects such commits at the source (the
  GitHub squash hoists `Co-authored-by` trailers). Use your default git identity.
- Documentation style: ASCII strict for `docs/sphinx/**.md`, no em-dash anywhere; these
  rules are checked by `docs/check_docs.py` (run by `build_docs.sh` and the PR lane).

## License

By contributing, you agree that your contributions are published under the BSD-3-Clause
license (see [LICENSE](LICENSE)).
