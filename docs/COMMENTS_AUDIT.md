# Audit of `adc_cpp` comments

Date: 2026-06-12.
Reviewed base: `origin/master` / `ffb9022`.

Scope: the Comments section of the Google C++ Style Guide applied to all the source code of the repository:
`include/pops/**/*.hpp`, `python/*.cpp`, `python/pops/*.py`, `tests/**`, `python/tests/**`, `bench/**`,
plus the CMake scripts. That is ~16 subsystems split by layer (core/AMR/parallel/physics,
mesh, numerics, coupling, runtime, bindings, tests, bench) and two cross-cutting sweeps
(TODO and language consistency). The audit covers comment quality, not code correctness
of the code: a comment is judged faithful, stale, redundant or missing relative to the code it describes.

Method: full review per subsystem, then cross-checking of every non-cosmetic finding
on the spot (the comment is confronted line by line with the code it claims to describe, and with
its neighboring comments). Findings marked `blocking` are comments that are factually FALSE,
all re-verified a second time. The coverage figures (`@file`, documented types) are
direct counts on the read-only worktree.

Related docs:
- [`CODEBASE_AUDIT.md`](CODEBASE_AUDIT.md): maintainability audit, same header and same tone.
- [`CODE_DOCUMENTATION_CONVENTION.md`](CODE_DOCUMENTATION_CONVENTION.md): THE comment
  convention of the project (`///` blocks, `@file`/`@brief`, contracts, threading/MPI/device invariants).
  CODEBASE_AUDIT.md cites it as a normative reference (lines 15, 542, 643), `.clang-format` aligns
  with its spirit. PROBLEM: this file is present in the main working tree but IS NOT
  COMMITTED (`git status` reports it as `??`). It is therefore ABSENT from the `ffb9022` commit that the worktrees
  check out, and the entire audit below had to be conducted against the DE FACTO convention
  (dominant patterns + CODEBASE_AUDIT.md) for lack of being able to read the standard. Priority recommendation ADC-125:
  commit `docs/CODE_DOCUMENTATION_CONVENTION.md`, otherwise the reference framework of the whole documentation
  effort remains a dead link in the versioned tree.
- [`check_docs.py`](check_docs.py): only lints the `.md`, not the `.hpp` headers.

## 1. Quantitative synthesis

Coverage of the `@file`/`@brief` headers and findings by severity, per subsystem. The severity is
that after cross-checking (an initial `blocking` invalidated at verification is demoted).

| Subsystem | Files / lines | @file header | Block. | Import. | Cosm. |
|---|---|---|---|---|---|
| core-amr-parallel-physics | 23 / 2861 | 21/23 | 1 | 5 | 5 |
| mesh | 13 / 1912 | 12/13 | 1 | 1 | 6 |
| numerics-1 (elliptic) | 12 / 3252 | 4/12 Doxygen | 1 | 8 | 7 |
| numerics-2 (spatial/EB) | 10 / 3119 | 8/10 | 2 | 3 | 4 |
| numerics-3 (time/AMR) | 13 / 2198 | 0/13 Doxygen | 1 | 2 | 7 |
| coupling-1 | 10 / 2499 | 10/10 | 1 | 1 | 8 |
| coupling-2 | 7 / 1850 | 7/7 | 0 | 3 | 4 |
| runtime-1 | 5 / 3163 | 5/5 | 1 | 2 | 2 |
| runtime-2 | 13 / 2855 | 13/13 | 0 | 3 | 3 |
| runtime-3 | 4 / 1380 | 4/4 | 0 | 1 | 5 |
| python-bindings (.cpp) | 3 / 3654 | 0/3 | 2 | 3 | 4 |
| python adc (core) | 3 / 3162 | 3/3 docstring | 0 | 5 | 4 |
| python adc (dsl) | 1 / 4648 | 1/1 docstring | 0 | 1 | 9 |
| tests + bench (cross-cutting) | 246 / 27915 | 245/246 intent | 0 | 0 | 5 |
| TODO + language (cross-cutting) | whole repo | n/a | 0 | 1 | 5 |

Total: 10 blocking (false comments), ~38 important, ~68 cosmetic, ~116 findings across the 13
code subsystems plus two cross-cutting sweeps that re-aggregate some findings (TODO,
English island).

Overall verdict: documentation quality is high and homogeneous, clearly above the average
for a project of this size. IMPLEMENTATION comments are the recurring strong point: the
real pitfalls (CUDA-IPC deadlock of MPI buffers, async safety of the Kokkos arena, derivation of the
polar geometric term, nvcc/EDG workarounds via named functors, CFL stability bounds,
harmonic vs arithmetic mean of permittivity) are explained with their justification, without
paraphrasing the obvious. The debt does not come from a lack of documentation but from three axes:
(1) a hard core of 10 comments that have become FALSE (comment-rot), which is the dangerous debt;
(2) a Doxygen migration left halfway (headers and public APIs documented in non-extractable `//`,
double `//`+`///` headers that duplicate the source of truth); (3) fragile
documentation references (hard-coded line numbers, stale doc paths).

## 2. Status by Google Comments subsection

### Comment Style
Clear de facto convention, widely applied: `///` Doxygen for the API, `//` for the internal,
trailing `///<` for members, unanimous `#pragma once`, no stray `/* */` block. Three structural
gaps. (a) Unfinished Doxygen migration: `include/pops/numerics/elliptic` documents 8/12
files with a prose `//` header (faithful but invisible to Doxygen), `numerics/time` 0/13 in
Doxygen, and several public APIs are in `//` (cf. File/Function Comments). (b) Legacy double headers:
a `//` block paraphrases the `/// @file` in ~12 files of the core, 12/13 of the mesh, 7/10
of coupling-1, 5/7 of coupling-2 - two sources of truth to maintain together. Extreme case:
`compute_face_fluxes` documented THREE times in `include/pops/numerics/spatial_operator.hpp:548,632,641`.
(c) Island of `/** */` Javadoc style in 4 files `include/pops/physics/` (`euler.hpp:17`,
`advection_diffusion.hpp`, `langmuir.hpp`, `two_fluid_isothermal.hpp`) where everything else uses
`///`. Isolated anomaly: an emoji in `include/pops/runtime/amr_dsl_block.hpp:172`, the only non-ASCII
character of this kind in the repository.

### File Comments
Good overall coverage but uneven. Absent (start on `#include`/`namespace`):
`include/pops/parallel/comm.hpp:1` and `parallel/load_balance.hpp:1` (`//` header only),
`include/pops/mesh/patch_box.hpp:1`, `include/pops/numerics/time/amr_advance.hpp:1` and
`amr_flux_helpers.hpp:1`, and above all `python/bindings/system/base/system.cpp:1` + `python/bindings/amr/amr_system.cpp:1` (no
header at all, 1831 and 1192 lines). Non-uniform placement of the `/// @file` (before vs after the
includes) in `numerics-2` (3 files) and `coupling-1` (4 before / 6 after). Dead reference:
`docs/CODEBASE_AUDIT.md:15` points to `CODE_DOCUMENTATION_CONVENTION.md`, not committed (cf. header).

### Struct and Class Comments
Nearly all non-trivial public types carry a contract. Qualitative defects: the concept
`EquationBlockLike` is summarized FALSE (`include/pops/core/equation_block.hpp:76`, cf. comment rot);
`struct Aux`, the central coupling channel, is described by a `//` block attached to an X-macro and not
by a `/// @brief` on the type (`include/pops/core/state.hpp:102`), so Doxygen associates it
with no class doc whereas its neighbor `StateVec` has one; the constraints of
`DistributedFFTSolver` are over- and under-declared (`include/pops/numerics/elliptic/poisson_fft_solver.hpp:102`).

### Function Comments
Very good coverage (`@param`/`@return`/`@throws` systematic on the runtime API). Three families
of defects. (a) Stale parameter enumerations, the most frequent case: the `@param limiter`
and `@param time` of `include/pops/runtime/system.hpp:79,169,198,199` lag behind the actual validators
(weno5, ssprk3, euler missing), likewise `include/pops/runtime/amr_system.hpp:203` (`weno5 ;
rusanov` wrongly suggests a restriction). (b) False contracts: `set_conservative_state`
(`include/pops/runtime/amr_system.hpp:363`, cf. comment rot). (c) Undocumented parameters:
the ctor of `GeometricMG` covers `active/replicated/cut_cell/levelset` but omits
`min_coarse/nu1/nu2/nbottom` (`include/pops/numerics/elliptic/geometric_mg.hpp:76`). On the Python side,
`System.add_block`/`AmrSystem.add_block` (`python/pops/__init__.py:1365,2415`), primary entry points,
have no docstring whereas `add_equation` is detailed; several non-trivial pybind `.def`
are exposed without docstring (`python/bindings/core/bindings.cpp:362`: `step_cfl`, `step_adaptive`,
`dt_hotspot`, `set_poisson`, `variable_names/roles`).

### Variable Comments
Generally careful (non-obvious members annotated with units). Defects: `phi_n_` described with a
false life cycle and API (`include/pops/coupling/schur/condensed_schur_source_stepper.hpp:395`, cf.
comment rot); `mg_` announces a "homogeneous Dirichlet" invariant that the code does not guarantee
(`include/pops/numerics/elliptic/composite_fac_poisson.hpp:489`); members of `MGLevel` partially
bare (`geometric_mg.hpp:496`, `coef`/`mask` not decodable on the spot); Poisson configuration
tokens without per-field doc (`include/pops/runtime/system_field_solver.hpp:90`).

### Implementation Comments
Strong point of the repository, but it is also where the dangerous debt lives. The majority of substantial
verified comments (>120 in total) are EXACT. The defects are the comment-rot comments
(section 3) and the fragile documentation references: stale directory paths after a
rename (`integrator/` and `operator/` in `include/pops/numerics/time/ssprk.hpp:13`,
`implicit_stepper.hpp:31`, `amr_reflux.hpp:25`), `bibliographie sect. 3.3`/`sect. 4.3` cross-references that
resolve to no anchor (`include/pops/mesh/box_hash.hpp:4,20` + `fill_boundary.hpp:8,101,159`,
5 sites), and hard-coded inter-file line numbers that drift at each edit of the target
`.md` (`HOFFART_FIDELITY.md ligne 39` now empty, cited by `wall_predicate.hpp:41` and
`cut_fraction.hpp:12`). Cross-cutting recommendation: replace the line numbers with symbolic
anchors (section title, function name).

### Punctuation, Spelling, and Grammar
Near-total consistency (cf. section 5). Isolated typos: `referenceent`
(`include/pops/core/coupled_system.hpp:40`), `apellent` (`numerics/spatial_operator.hpp:387`),
`Exposes` instead of `Expose` (`numerics/spatial_discretisation.hpp:32`), `Cohrent`
(`runtime/native_loader.hpp:733`), `tolerree` (`python/pops/__init__.py:870`). Occasional ungrammatical
sentences: `for_each.hpp:113` (`acces hote sur sont sans course`), `limite de / Debye` line break
in `numerics/time/imex.hpp:11`, self-correction left in place `... non : ...` in
`python/bindings/system/base/system.cpp:1511`. `@returns` tag (English) instead of `@return`: 2 occurrences
(`amr_coupler_mp.hpp:558`, `amr_system.hpp:439`).

### TODO Comments
See section 4. In summary: 20 markers, none in the Google format, all roadmap labels
rather than tasks.

## 3. Comment rot: FALSE comments (blocking severity)

The most dangerous debt: a reader who follows these comments is actively misled.
Each one has been confronted with the code and re-verified.

| File:line | Comment claim | Code reality |
|---|---|---|
| `include/pops/core/equation_block.hpp:76` | concept requires a `State` member | the concept requires `B::Model`; no `State` alias exists (lines 80-87) |
| `include/pops/mesh/fill_boundary.hpp:242` | MPI receives UNIFIED pointers (GPUDirect) | `sbuf`/`rbuf` are in `SharedHostPinnedSpace` (pinned host), seen as HOST by MPI to avoid CUDA-IPC; contradicts lines 118-124 and `core/allocator.hpp` |
| `include/pops/numerics/elliptic/geometric_mg.hpp:425` | STICKY smoothing hardening between solves | the code saves `nu1_/nu2_` (446) and RESTORES them on return (454,457); the neighboring paragraph (442-444) says the opposite |
| `include/pops/numerics/elliptic/polar_tensor_operator.hpp:381` | `solve()` = Jacobi-preconditioned BiCGStab | default precond = RadialLine (radial Thomas), selectable via `precond_`; omits the gauge pinning |
| `include/pops/numerics/spatial_operator_eb.hpp:136` | fractional face apertures `alpha in [1e-3,1]` between active cells | `cut_distance` returns `h` for any active neighbor -> alpha = 1; BINARY apertures {0,1}, only `kappa` is fractional |
| `include/pops/numerics/time/amr_reflux.hpp:19` | "minimal version": subcycling to come, uniform aux | `amr_step_2level` ALREADY IMPLEMENTS Berger-Oliger subcycling (155-174) and reads a spatial aux; living file (included by `spectral_coupler.hpp`) |
| `include/pops/coupling/schur/condensed_schur_source_stepper.hpp:395` | `phi_n_` allocated at the first `advance_source` | allocated unconditionally at the ctor (234); the `advance_source` method DOES NOT EXIST (the API is `step()`, repo grep = 0 other occurrence) |
| `include/pops/runtime/amr_system.hpp:363` | `set_conservative_state` raises in multi-block | the `build_multi` facade threads the state to the NATIVE blocks (379); only the compiled path (.so) rejects (315) |
| `python/bindings/core/bindings.cpp:70` | serial: `my_rank=1`, `n_ranks=0` | `comm.hpp` returns `my_rank()=0`, `n_ranks()=1`; contradicts the adjacent docstrings (73-74) |
| `python/bindings/core/bindings.cpp:240` | `set_source_stage` descriptors: "Cartesian only (polar: reject)" | the polar stage builds a `PolarCondensedSchurSourceStepper` and honors `bz_aux_component` without rejection (`system.cpp:1410-1438`) |

All are vestiges of earlier refactors (API rename, pinned-host fix, multi-block
wave, mono-rank safeguard removal). 6/10 are auto-correctable in one line; the other 4
require a reformulation of the contract.

Second-rank note: four findings initially classified as blocking were demoted to important
at verification because the comment describes a correct intent or a path with no break in
behavior: `composite_fac_poisson.hpp:489` (homogeneous Dirichlet assumed), `elliptic_solver.hpp:12`
(Coupler already templated), `elliptic_problem.hpp:41` (`FieldPostProcess::apply` nonexistent, correct mechanism
documented further down), and the doc-only enumerations of `system.hpp` (weno5/ssprk3/euler).

## 4. TODO inventory

| Measure | Value |
|---|---|
| TODO markers | 20 |
| FIXME / XXX / HACK | 0 / 0 / 0 |
| Conforming to Google format `TODO(id):` | 0/20 |
| Distribution | `include/` 9, `tests/` 11, `python` + `bench` + `cmake` + `scripts` 0 |

No TODO signals incomplete work: these are 20 provenance labels referring to a
roadmap numbering (`TODO 2.3`, `TODO 4`, `TODO 2.2.3`, `TODO 4.3`, `TODO 2.1.1`). They
DESCRIBE behavior already implemented (`include/pops/coupling/static_system/amr_system_coupler.hpp:33,44,62`,
`tests/test_two_species_minimal.cpp`) or a future generalization
(`include/pops/numerics/spatial_operator.hpp:560,653`). Two problems: the numbering is not
resolvable from the `todo.md` at the root (orphan cross-references for a fresh reader), and any CI scan
`grep TODO` would surface these 20 hits as pending tasks (false positives). The only cross-reference to
a concrete file (`composite_fac_poisson.hpp:30` -> `amr_reflux.hpp:20`) is EXACT. Also note
REAL deferrals expressed in prose without a marker, hence invisible to tooling
(`amr_diagnostics.hpp:29`, `amr_coupler_mp.hpp:289`, several `= Phase 4b`).

Recommendation: reserve the `TODO` token for incomplete work, in the `TODO(ADC-xxx):` format;
replace the roadmap labels with `(jalon 2.3)` or a cross-reference to the design doc; and mark the
real deferrals in prose with a findable `TODO(ADC-xxx)`.

## 5. Language consistency

Verdict: French WRITTEN WITHOUT ACCENTS, applied at 99.99%. Ratio of FR/EN markers per directory:
`include` 6679/32, `python` 2309/9, `tests` 3202/8, `bench` 157/2. The rare "English" hits are
almost all false positives (code keywords in comments, titles of articles cited in
the bibliography, the Latin word `via`).

Three breaches, all occasional:
- Authentic English prose: a SINGLE island, the Doxygen block of `max_wave_speed`
  (`include/pops/coupling/amr/amr_coupler_mp.hpp:553-559`), surrounded by French members. To translate.
- Accents: 2 lines in the whole repository (`bench/scaling_step.cpp:329-330`). To strip.
- ASCII: 1 emoji (`include/pops/runtime/amr_dsl_block.hpp:172`). To replace with `ATTENTION :`.

Proposed rule (to be written into `CODE_DOCUMENTATION_CONVENTION.md`): comments in French without
accents, pure ASCII, Doxygen tags and identifiers in English; `@return` canonical (not `@returns`).

## 6. Prioritized correction plan

P0: false comments (section 3). This is the only debt that actively misleads. 10 corrections,
each local to a file. Treatment: either a single targeted PR "fix the comment-rot
(ADC-125)" that touches the 10 sites, or fold each correction into the next PR that modifies
the concerned file. 6/10 are one-liners; the contracts of `set_conservative_state`,
`polar_tensor_operator::solve`, `spatial_operator_eb::eb_face_aperture` and the header of `amr_reflux`
require a reformulation to review. To do first, independently of the rest.

P1: public API `include/pops/**` and `python/` non-extractable or undocumented. PRs targeted per
subsystem (mechanical conversion, content already present):
- Add `@file`/`@brief` where it is missing: `parallel/comm.hpp`, `parallel/load_balance.hpp`,
  `mesh/patch_box.hpp`, `numerics/time/amr_advance.hpp` + `amr_flux_helpers.hpp`, `python/bindings/system/base/system.cpp`,
  `python/bindings/amr/amr_system.cpp`.
- Migrate `//` -> `///` on the public APIs documented in prose: `numerics/time/*` (12 files,
  including the `mf_*` helpers and `advance_amr`), checkpoint/restart of `AmrCouplerMP`
  (`coupling/amr_coupler_mp.hpp:319+`), `runtime/amr_runtime.hpp:279,893`.
- Fill the contract gaps: `struct Aux` (`core/state.hpp:102`), parameters of the `GeometricMG`
  ctor, docstrings of `System.add_block`/`AmrSystem.add_block`, pybind docstrings of the non-trivial
  `.def`.

P2: TODO + language normalization (sections 4 and 5). A single hygiene PR: reclassify the 20
roadmap labels, translate the English island, remove the 2 accented lines and the emoji,
normalize `@returns` -> `@return`. Commit `CODE_DOCUMENTATION_CONVENTION.md` in the same PR to
anchor the rule.

P3: cosmetic, as you go. Remove the double `//`+`///` headers (one source of truth),
unify the `/** */` island of `physics/`, align the placement of the `@file`, and replace the fragile
documentation cross-references (stale directory paths, inter-file line numbers) with symbolic
anchors. These points do not mislead anyone today but are the breeding ground for the
next comment-rot: to handle when a file is open anyway.

Cross-cutting note: the missing link of the whole effort remains `CODE_DOCUMENTATION_CONVENTION.md`. As long
as it is not committed, the "fidelity to the project convention" audit relies on de facto patterns
and not on a written standard; committing it via ADC-125 is the prerequisite for a reproducible
header linting.
