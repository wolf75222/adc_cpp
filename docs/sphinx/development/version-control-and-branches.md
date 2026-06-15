# Version control and branch management

How we manage source, branches, and versions in `adc_cpp`, and **why**. This page is the *policy*
layer (the reasoning and the rules); the *mechanism* lives elsewhere and is linked here, not
duplicated:

- [VERSIONING](https://github.com/wolf75222/adc_cpp/blob/master/docs/VERSIONING.md) -- the SemVer
  release mechanism (single `project(VERSION)` source, the tag, the CHANGELOG, `release.yml`).
- [CONTRIBUTING](https://github.com/wolf75222/adc_cpp/blob/master/CONTRIBUTING.md) -- the concrete
  build / test / pull-request workflow and the branch-naming convention.

The principles below are adapted from *Software Engineering at Google*, chapter 16 (Winters), to a
small, agent-assisted project hosted on GitHub.

## Single Source of Truth

`master` in `wolf75222/adc_cpp` is the one Source of Truth: a change is "done" only when it is on
`master`. Git is a distributed VCS, so this is a *policy*, not something the technology enforces by
itself (no clone is inherently "canonical"). We make it explicit and back it with branch protection
so the question "which version is current?" never arises.

## Trunk-based development

Work happens in **short-lived branches** cut from `master`, named `adc-<n>-description` after a
Linear issue, merged by a reviewed pull request, then deleted (`delete_branch_on_merge` is on). We do
**not** keep long-lived development branches:

- Stability comes from **tests, CI, and review**, not from a stabilization branch. The same commits
  reach `master` eventually; small, author-merged changes are easier to integrate and to bisect than
  one large batched merge.
- Long-lived branches reintroduce "which version do I depend on?" and breed merge-strategy overhead
  -- the opposite of scalable. (DORA / Accelerate find trunk-based development predictive of
  high-performing teams.)
- **Work in progress is itself a branch.** Uncommitted edits and per-issue worktrees are
  conceptually branches; keep them small and frequently synced to `master`.

The one sanctioned exception is a small, explicitly-frozen set of experiment branches
(`feat/perf-campaign-*`), documented as such -- not a default to imitate.

## The One-Version Rule

> For any dependency, a developer must never have a *choice* of which version to depend on.

This is the highest-leverage rule on this page. Allowing several versions of the same dependency in
one build invites diamond-dependency failures, silent version skew, and wasted work. In `adc_cpp`:

- **Third-party dependencies are pinned to one version.** Kokkos, Eigen, and pybind11 come from a
  single pinned revision (FetchContent pins / the conda env), never chosen per target.
- **One compiled core.** There is a single `_adc` module; the DSL backends (`aot` / `production`) and
  the native bricks all build against the same headers. The **`abi_key`** guard makes this concrete:
  it refuses to load a model `.so` built against a *different* header or toolchain revision rather
  than risk a silent ABI mismatch. That is the One-Version Rule, enforced at load time.
- **No forking without renaming.** Do not copy a core component to tweak it; fix it in place, or
  design the variants so they coexist behind distinct names.

## Release branches: rarely, and abandoned

We do not use Git Flow. A `release/*` branch is justified only when a release lives longer than a few
hours and we must know exactly what is in the field (we ship a versioned Python package). When one is
used: cherry-pick minimal fixes from `master`, and **abandon** the branch afterwards -- never merge
it back. In most cases the right move is simply "add the fix to `master` and re-release".

## `adc_cpp` specifics

- Commits are authored as `Romain Despoullains <...wolf75222...>`. **No `Co-Authored-By`, AI, or tool
  trailers anywhere.** GitHub's squash merge hoists branch-commit trailers into the squash commit, so
  verify that `git log master..<branch>` is trailer-clean **before every squash merge**.
- Every pull request must pass the required **gate** check (it runs on every PR, docs-only included).
- Use **one isolated worktree per issue**, branched from `origin/master`. The repository is worked by
  several agents under a shared identity, so isolation matters.
- CHANGELOG discipline: one `[Unreleased]` line per notable PR; bumping the version and tagging is a
  deliberate, owner-driven step (see VERSIONING).

## Two repositories, one direction

`adc_cpp` (the solver) and `adc_cases` (the application cases) are separate repositories. The
dependency flows one way -- `adc_cases` uses `adc_cpp` -- so keep that boundary one-directional and
apply the One-Version Rule across it: `adc_cases` tracks the single `adc_cpp` it builds against, it
does not fork it.
