> **STATUS: historical.** The Phase-1 documentation audit (truth matrix); it reflects the state at the time and several of its items have since been acted on. Read it as a historical record (class D, see `DOC_QUALITY.md`).

# Documentation audit adc_cpp / adc_cases -- truth matrix (Phase 1)

> Synthesis document assembled from the 8-agent inventory (adc_cpp prose, Sphinx, three canonical docs verified code-in-hand, real Python/DSL API, adc_cases cases, build/run, tests, assets) plus an adversarial verification pass. Verdicts marked CONFIRMED come from the independent adversarial pass; the others rest on the inventory alone (flagged where the data is thin).

---

## 1. Executive summary

- **The canonical backbone is reliable, the bookkeeping is not.** The background docs (ARCHITECTURE, ALGORITHMS, BACKEND_COVERAGE, CHOICES, DSL_API, DSL_MODEL_DESIGN, CONSERVATION_SUMMARY, BIBLIOGRAPHY, COUPLER_HIERARCHY, COUPLING_SURFACE, PERFORMANCE) correctly describe the layers, the C++20 concepts (`PhysicalModel`, `EllipticSolver`), the three-way parallel seam (`for_each.hpp` + `comm.hpp`), the 4-job CPU-only CI matrix. The risk is concentrated on (a) the "AmrSystem single-block" narrative and (b) the bookkeeping (test counters, phantom symbols, deleted files).

- **Risk #1 (high severity, CONFIRMED): "AmrSystem single-block" is OBSOLETE.** Repeated as an honest limitation in README, ARCHITECTURE Sec. 8, DSL_MODEL_DESIGN (Sec. 0bis/Sec. 5/Phase D) and the persistent memory, whereas `python/bindings/amr/amr_system.cpp` ships a real multi-block facade (`multi_block()`/`build_multi()`/`set_regrid`, plus no "single block only" throw), with 7 capstone tests and union-tags regrid (#195/#199/#205). A refactor that trusts the canonical docs **would re-assert a false limitation**.

- **Risk #2 (CONFIRMED): deleted file still documented.** `amr_coupler.hpp`/`AmrCoupler` is DELETE (#164) but described as "deprecated kept for reference" in THREE normative docs (ARCHITECTURE:33, COUPLING_SURFACE:118, COUPLER_HIERARCHY Sec. 4). Only CODEBASE_AUDIT records the deletion. Any autodoc/coupler reference built on these three docs would list a nonexistent file.

- **Phantom symbol: the "MC" limiter.** Documented twice (ALGORITHMS:97, ARCHITECTURE:288/509) but absent from `reconstruction.hpp` (only `NoSlope`/`Minmod`/`VanLeer`/`Weno5` exist). A user who follows the docs and passes `limiter='mc'` fails.

- **Test counters: all stale and mutually inconsistent.** Real disk ~ 90-94 `pops_add_test` (non-MPI), ~51 MPI entries, 60-61 Python files. The docs say 71/84/103 (C++) and 26/55 (Python) depending on the file; ARCHITECTURE and BACKEND_COVERAGE contradict each other. All values are **conservative underestimates** (the suite is bigger than announced). To be regenerated programmatically, not hard-coded.

- **Sphinx: dead gallery mechanism (CONFIRMED).** `_copy_tutorials.setup_gallery()` points to a deleted `tutorials/` (commit 194c63f, app layer moved to adc_cases) -> copies 0 files; and even when filled, `tutorials_index.md` is a stub without a toctree -> doubly orphaned. A clean `sphinx-build` produces an empty "Tutorials" section. Local `_build/`/`_generated/` are stale (May 30) but already gitignored/untracked.

- **Honestly runnable locally (darwin/arm64, no CUDA):** C++ core Serial + ctest, MPI (OpenMPI 5.0.9), standalone OpenMP, Kokkos device=OpenMP, and the Python module -- **but only with the interpreter that compiled it** (`/opt/homebrew/anaconda3/bin/python3.12`, which has numpy). Confirmed footgun: the `.so` is `cpython-312`; under system `python3` (3.9.6) -> `ModuleNotFoundError: pops._pops`; under a 3.12 without numpy -> dies on `import numpy` in `dsl.py`.

- **Requires ROMEO/GH200:** any CUDA path -- Kokkos Cuda single-GPU, MPI+Kokkos Cuda multi-GPU. CI **never** builds `Kokkos_ENABLE_CUDA=ON`. The `python/tests/gpu/*.cpp` harnesses are SLURM-only (`--constraint=armgpu`, nvcc_wrapper, Kokkos 4.4.01) and excluded from the CI glob (`-maxdepth 1`). Any "GPU-validated" mention must say "manually on ROMEO GH200".

- **DSL_API.md is materially stale (4 high-severity false claims, CONFIRMED).** Some snippets crash as written (`m.u[0]`/`m.a.grad_x` -> AttributeError; `set_poisson('geometric_mg')` -> unknown rhs; `AmrSystem(max_level=2)` -> unknown attribute; "runtime params = NotImplementedError" false: they work). The runtime-NotImplementedError claim also lives in the **source docstring** (`dsl.py:1623/1627`) -> autodoc would propagate it.

- **adc_cases: HONEST diocotron-vs-Hoffart labeling.** The reduced ExB diocotron is flagged everywhere as "reduced-model normalization benchmark, NOT a reproduction of the full Euler-Poisson system"; the full case `hoffart_euler_poisson_dsl` is classified "reproduction-candidate" with a PENDING/MEASURED table showing -82 to -95% error. No file lies. **But** the top-level adc_cases README omits 6 DSL cases; only 4/15 cases have a README; and asset provenance is weak (figures committed without SHA, produced elsewhere than their committed path).

---

## 2. Documentation file matrix

Statuses: **active** (living normative), **historical** (spec/plan partially executed), **archive** (under `docs/archive/`), **generated** (artifact). Decisions: keep / rewrite / merge / archive / delete.

### adc_cpp -- prose & docs/

| File | Role | Audience | Status | Source of truth | Stale/False | Duplicates | Decision |
|---|---|---|---|---|---|---|---|
| README.md | Project portal (pitch, GIF, build, validation) | new user | active | ARCHITECTURE + GPU_RUNTIME_PORT + DSL_MODEL_DESIGN | "AmrSystem single-block" (l.130-132) FALSE; `potential()` :272 stale (real :332); 208 l. vs target 120-150 | ARCHITECTURE Sec. 6/Sec. 13, DSL Sec. 5, GPU_RUNTIME_PORT | **rewrite** |
| docs/ARCHITECTURE.md | Canonical: 5 layers, concepts, module map, AMR Sec. 8, validation Sec. 11 | contributor | active | `include/pops/**` | `amr_coupler.hpp` (l.33) deleted; Sec. 8 single-block obsolete; Sec. 11 stale counters; MC limiter nonexistent | README module-map, COUPLER_HIERARCHY, GPU_RUNTIME_PORT | **rewrite** |
| docs/ALGORITHMS.md | Canonical: methods catalogue (formula/code/test x20) | contributor | active | `numerics/**` + `tests/` | MC limiter (l.97) nonexistent; the rest verified good | Sec. 18 DSL compose, Sec. 19 DSL Sec. 0/Sec. 7, Sec. 20 ARCH Sec. 4 | **keep** (purge MC) |
| docs/BACKEND_COVERAGE.md | Self-declared SoT backends/tests matrix | contributor | active | `tests/CMakeLists.txt` + CI | Stale header totals (84/19/55 vs disk 90-94/31/60-61); per-test lines OK | replaces README "4 paths" + DSL Sec. 5 | **rewrite** (totals) |
| docs/CHOICES.md | Canonical: rationale decisions D-1..D-7 | contributor | active | ARCHITECTURE + oracles | No false claims; only doc with accents (encoding inconsistency) | -- | **keep** |
| docs/DSL_API.md | Short user DSL reference | new user | active | `adc/dsl.py` + DSL_MODEL_DESIGN | **4 false claims** (m.u/m.a, set_poisson, max_level, runtime); GPU backend oversold | subset of DSL Sec. 5 | **rewrite** |
| docs/DSL_MODEL_DESIGN.md | Canonical: DSL/model design + status per phase | contributor | active | `dsl.py` + `system.cpp` + `runtime/**` | "AmrSystem single-block" (Sec. 0bis/Sec. 5/Phase D) obsolete; line numbers indicative | README, ALGORITHMS Sec. 18/19, PAPER_ROADMAP | **rewrite** |
| docs/CONSERVATION_SUMMARY.md | Canonical: measured FV/Schur conservation vs Hoffart FE | contributor | active | `adc_cases/.../test_schur_conservation.py` (#207) | None internal; cross-repo authority not verifiable from adc_cpp | FULL_MODEL_VALIDATION PR2, HOFFART_FIDELITY | **keep** |
| docs/BIBLIOGRAPHY.md | Canonical: references (Hoffart arXiv:2510.11808, etc.) | new user | active | external literature | None | citations anchored elsewhere | **keep** |
| docs/HOFFART_FIDELITY.md | Cross-check hoffart case vs paper (per-aspect) | contributor | active | `adc_cases` model.py/run.py + engine | Cross-repo anchors not verifiable from adc_cpp; not_verified labels honest | CONSERVATION_SUMMARY, HOFFART_STEP_SEQUENCE | **keep** |
| docs/CODEBASE_AUDIT.md | Per-file maintainability audit (2026-06-06) | internal-dev | active | `include/pops/**` + `python/*.cpp` | The most accurate inventory (correctly notes amr_coupler REMOVED); slightly behind on AmrSystem multi-block | COUPLING_SURFACE, ARCH module map | **keep** |
| docs/DOC_CLEANUP_PLAN.md | Internal plan: overlaps + SoT per doc | internal-dev | active | self (meta) | "BACKEND_COVERAGE to create" exists; README says 373 l. (real 208); partially executed | indexes the others | **archive** |
| docs/COUPLER_HIERARCHY.md | Reference: each coupler in `coupling/` | contributor | active | `coupling/*.hpp` | Sec. 4 describes DELETED `amr_coupler.hpp` as existing (l.152, table l.424) | ARCH Sec. 5/Sec. 8, COUPLING_SURFACE | **rewrite** |
| docs/COUPLING_SURFACE.md | SoT PUBLIC/INTERNAL/DEPRECATED coupler classification | contributor | active | `coupling/*.hpp` | l.118 lists `amr_coupler.hpp` as kept-historical: FALSE (deleted) | COUPLER_HIERARCHY | **rewrite** |
| docs/PERFORMANCE.md | Historical perf measurements (CS:APP, M1) | contributor | historical | `adc_cases` benches + `coupling_policy.hpp` | "historical" banner OK; cites `examples/bench_amr.cpp`, `scripts/plot_bench_scaling.py` ABSENT from adc_cpp | PROFILE_RESULTS, GPU_RUNTIME_PORT ph11 | **archive** |
| docs/PROFILE_RESULTS.md | Measured profiling (2026-06-06), MG dominates 96-99.9% | internal-dev | active | `bench/profile_step` | None; GH200/CUDA columns ROMEO-only (honest) | PERFORMANCE (old), RESEARCH_BACKLOG | **keep** |
| docs/GPU_RUNTIME_PORT.md | GH200 validation log phases 1-11 | internal-dev | active | `python/tests/gpu/` harnesses (outside ctest) | None material | README Validation, BACKEND_COVERAGE Sec. 3, GPU_ROMEO | **keep** |
| docs/GPU_ROMEO.md | GPU verification recipe DSL brick on GH200 (maxdiff=0) | internal-dev | active | `gen_cuda_harness.py` + `romeo_run.sh` | None; narrow recipe | GPU_RUNTIME_PORT, BACKEND_COVERAGE Sec. 3 | **keep** |
| docs/PAPER_ROADMAP.md | Science roadmap (Hoffart, O5, ring-edge lock) | internal-dev | active | diocotron case + todo Sec. 6 | OLD growth-rate figures (Minmod-o2 n=192) superseded | FULL_MODEL_VALIDATION, HOFFART_FIDELITY, todo Sec. 6 | **keep** |
| docs/FULL_MODEL_VALIDATION_ROADMAP.md | Full-model reproduction roadmap; SUPERSEDES PAPER_ROADMAP | internal-dev | active | HOFFART_FIDELITY + HOFFART_STEP_SEQUENCE + run.py | Self-corrects PAPER_ROADMAP; PR2 (#208) done; most up-to-date science doc | PAPER_ROADMAP, HOFFART_FIDELITY | **keep** |
| docs/HOFFART_STEP_SEQUENCE.md | Exact master macro-step order (Lie not Strang) | contributor | active | `system_stepper.hpp`, `amr_runtime.hpp`, etc. | None apparent; specific file:line anchors | HOFFART_FIDELITY, CONSERVATION_SUMMARY | **keep** |
| docs/CODE_DOCUMENTATION_CONVENTION.md | Code doc convention (Doxygen/PEP257) | internal-dev | active | self (style) | None (style guide) | -- | **keep** |
| docs/SYSTEM_CPP_EXTRACTION_PLAN.md | `system.cpp` split plan (NativeLoader) | internal-dev | historical | `python/bindings/system/base/system.cpp` | Partially executed (#151, native_loader shipped) | CODEBASE_AUDIT | **archive** |
| todo.md | Living internal TODO (session state, PR log) | internal-dev | active | git history + master | Partially stale (l.579 "6/6" vs header "7/7"); 65 KB | PAPER_ROADMAP Sec. 6, RESEARCH_BACKLOG | **keep** |
| docs/RESEARCH_BACKLOG.md | Non-auto-completable research backlog | internal-dev | active | `schur_condensation.hpp` + ROMEO | None material | PAPER_ROADMAP, FULL_MODEL_VALIDATION | **keep** |
| docs/archive/README.md | Archive index ("out-of-archive is authoritative") | internal-dev | archive | self | None (correct disclaimer) | -- | **keep** |
| docs/archive/ROADMAP.md | Archived roadmap | internal-dev | archive | superseded ARCH/todo | Historical (archive placement) | todo, PAPER_ROADMAP | **archive** |
| docs/archive/TODO.md | Archived multi-species work-item TODO | internal-dev | archive | superseded todo.md | Historical | todo.md | **archive** |
| docs/archive/ETAT_DES_LIEUX.md | Archived synthesis of 6 audits | internal-dev | archive | superseded CODEBASE_AUDIT | Historical, file:line stale | CODEBASE_AUDIT | **archive** |
| docs/archive/ARCHITECTURE_CIBLE.md | Archived north-star vision | internal-dev | archive | superseded ARCHITECTURE | Aspirational ("desired vs implemented" case) | ARCHITECTURE | **archive** |
| docs/archive/DESIGN_MULTISPECIES.md | Archived multi-species design | internal-dev | archive | superseded ARCH Sec. 5 | Multi-species shipped (SystemCoupler) | ARCH Sec. 5 | **archive** |
| docs/archive/PLAN_VARIABLES_EPM.md | Archived Variables + EPM plan | internal-dev | archive | superseded ALGORITHMS Sec. 11 | Plan executed (eps/Helmholtz/aniso) | ALGORITHMS Sec. 11 | **archive** |
| docs/archive/HERO_RUN_AMR.md | Archived distributed AMR hero-run design | internal-dev | archive | superseded GPU_RUNTIME_PORT ph10/11 | Historical | GPU_RUNTIME_PORT, archive/ROMEO | **archive** |
| docs/archive/ROMEO.md | Archived ROMEO runs log | internal-dev | archive | superseded GPU_ROMEO/GPU_RUNTIME_PORT | Historical log | GPU_ROMEO, GPU_RUNTIME_PORT | **archive** |
| docs/archive/DIOCOTRON_GROWTH_RATE.md | Archived diocotron growth-rate note | internal-dev | archive | adc_cases diocotron + HOFFART_FIDELITY | Pre-O5 figures; content lives in adc_cases | HOFFART_FIDELITY, PAPER_ROADMAP | **archive** |
| docs/archive/two_fluid_ap.md | Archived two-fluid AP method note | internal-dev | archive | `adc_cases/two_fluid_ap/` | Correctly archived (AP left the core) | BIBLIOGRAPHY, RESEARCH_BACKLOG | **archive** |

### adc_cpp -- Sphinx (docs/sphinx/)

| File | Role | Audience | Status | Source of truth | Stale/False | Duplicates | Decision |
|---|---|---|---|---|---|---|---|
| docs/sphinx/conf.py | Sphinx config (Furo, autodoc, inject sys.path, setup_gallery) | internal-dev | active | `_copy_tutorials.py`, `docs.yml` | l.21 `setup_gallery()` dead (tutorials/ deleted); l.59 excludes `_generated/tutorials/README.md` | -- | **rewrite** |
| docs/sphinx/_copy_tutorials.py | Gallery helper (copies tutorials/ -> _generated) | nobody | active | src `../../tutorials` (is_dir=False) | Entirely dead module (early-return 0) | -- | **delete** |
| docs/sphinx/index.md | master_doc, 3 toctrees | new user | active | installation/quickstart/examples/api.md | Tutorials toctree = dead-end stub; flat nav vs target 7-sections | -- | **rewrite** |
| docs/sphinx/tutorials_index.md | Stub "tutorials moved to adc_cases" | new user | active | git 194c63f~1 (old toctree) | Navigational dead-end, no toctree | examples.md | **merge** |
| docs/sphinx/api.md | Python API reference (autodoc, 20 symbols) | new user | active | `build-py/python/pops/__init__.py` | None (20 symbols resolved) | -- | **keep** |
| docs/sphinx/installation.md | C++/Python build + backends | new user | active | `CMakeLists.txt` | "71 ctests" (l.17) to recheck | BACKEND_COVERAGE (intro) | **keep** |
| docs/sphinx/quickstart.md | Annotated Python recipes | new user | active | `adc/__init__.py` | Not executed (no local env); API plausible | installation, api | **keep** |
| docs/sphinx/examples.md | Redirects to adc_cases | new user | active | repo (no examples/ or scripts/) | Cross-doc links = GitHub URLs, not Sphinx pages | tutorials_index | **merge** |
| docs/sphinx/requirements.txt | Doc deps (sphinx, furo, myst, autodoc-typehints) | contributor | active | `conf.py` extensions | `sphinx-autodoc-typehints` never enabled; numpy MISSING | -- | **keep** (fix) |
| docs/sphinx/_generated/tutorials/ | Generated gallery output (00..08 + README) | nobody | generated | `.gitignore` | Stale May 30, not regenerated, gitignored/untracked | -- | **delete** |
| docs/sphinx/__pycache__/ | Bytecode cache | nobody | generated | `.gitignore` | Stale, gitignored | -- | **delete** |
| docs/_build/ | Sphinx HTML output + Doxygen | nobody | generated | `.gitignore` | Stale May 30 (old misleading gallery), gitignored/untracked | -- | **delete** |

### adc_cpp -- build & root

| File | Role | Audience | Status | Source of truth | Stale/False | Duplicates | Decision |
|---|---|---|---|---|---|---|---|
| CMakeLists.txt | Build root (options, C++23/C++20 Kokkos) | contributor | active | self (authoritative options) | Does NOT declare `POPS_USE_EIGEN` despite README+CI | -- | **keep** |
| python/CMakeLists.txt | pybind11 `_pops` module build | contributor | active | self + `python/tests/` | None | -- | **keep** |

### adc_cases

| File | Role | Audience | Status | Source of truth | Stale/False | Duplicates | Decision |
|---|---|---|---|---|---|---|---|
| README.md (top-level) | Cases portal + "The cases" table | new user | active | `cases_manifest.toml` | Table l.109-119 INCOMPLETE: omits 6 DSL cases | cases_manifest.toml | **rewrite** |
| cases_manifest.toml | Manifest of 16 entries (category/ci/needs) | contributor | active | self | Up to date; SoT of the scope | README table | **keep** |
| diocotron/README.md | Reduced diocotron case (honest "normalization") | new user | active | run.py + NORMALIZATION.md | None (correct disclaimer) | manifest, NORMALIZATION | **keep** |
| hoffart_euler_poisson_dsl/README.md | Full-model case (reproduction-candidate) | new user | active | run.py + results.py + measurement_record | Outputs section implies modes 3/4/5 (only mode_3 on disk) | HOFFART_FIDELITY | **keep** |
| magnetic_isothermal_dsl/README.md | DSL case aux B_z (inter-backend parity) | new user | active | run.py | macOS platform caveat = aot only (honest) | -- | **keep** |
| schur_magnetized_cartesian/README.md | Schur vs explicit timing case | new user | active | run.py | Docstring cites `pops.Split` but code = private hook `sim._s.set_source_stage` | -- | **rewrite** |

> README coverage note: only **4/15** case folders have a README (diocotron, hoffart_euler_poisson_dsl, magnetic_isothermal_dsl, schur_magnetized_cartesian). The other 11 rely on module docstrings (detailed and accurate). Phase 2 decision to make: generate per-case READMEs or canonicalize docstrings + top-level table.

---

## 3. Confirmed false/stale claims

Merge of `claim_findings` + adversarial pass. Verdict column: inventory verdict; adversarial status in bold (CONFIRMED = independently verified, NOT-CONFIRMED = not re-verified).

| Claim | Source doc | Verdict | Evidence |
|---|---|---|---|
| AmrSystem is single-block / explicit-only / no Roe / no primitive recon / NOT at parity with System | ARCHITECTURE Sec. 8 (l.26,379-382,547), README:130-132, DSL_MODEL_DESIGN Sec. 0bis/Sec. 5/Phase D | **obsolete -- CONFIRMED (high)** | `amr_system.cpp:202 multi_block()`, `:208 build_multi()`, `:300 if(multi_block())`, parse `recon in {conservative,primitive}` `:343-345`, `riemann in {rusanov,hllc,roe}` `:135`, IMEX `:340-346`; 7 capstone tests (`test_amr_system_twoblock`, `test_amr_multiblock_*`); union regrid #199. No "single block only" throw. |
| `m.u[0]`/`m.a.grad_x` to access conservative/aux vars in dsl.Model | DSL_API.md Sec. 1 (l.21-22,29) | **false -- CONFIRMED (high)** | `dsl.Model` has neither `.u` nor `.a`; real API = `conservative_vars(...)->Vars` (`dsl.py:1543`), aux via `m.aux('grad_x')` (`:1585`). Crashes with AttributeError. |
| Runtime params: `param(kind='runtime')` raises NotImplementedError (Phase E) | DSL_API.md Sec. 5 (l.104-106) **AND** `dsl.py:1623/1627` (docstring+comment) | **false -- CONFIRMED (high)** | `Param.__init__` (`dsl.py:1420-1432`) implements kind='runtime' (`RuntimeParamRef`), raises ValueError only for unknown kinds. `test_dsl_runtime_params.py` compiles+runs+`set_block_params` end-to-end. **The source docstring would propagate the false claim via autodoc.** |
| `sim.set_poisson("geometric_mg")` (positional) | DSL_API.md Sec. 3 (l.70) | **false -- CONFIRMED (high)** | 1st positional arg = `rhs` (`bindings.cpp:153`), valid in {charge_density,composite} (`system_field_solver.hpp:197-199`) -> throw "unknown rhs". Correct: `set_poisson(solver='geometric_mg')`. |
| Tutorials gallery wired into a toctree (reachable pages) | sphinx/tutorials_index.md, index.md | **false -- CONFIRMED (high)** | No `.md` references `_generated/tutorials/`; `tutorials_index.md` = prose stub without toctree; `index.md` Tutorials toctree only lists `tutorials_index`. Pre-194c63f had a real toctree. Clean build -> empty section. |
| `setup_gallery()` copies `tutorials/*.md` -> `_generated/tutorials/` | sphinx/_copy_tutorials.py (+conf.py:21) | **obsolete -- CONFIRMED (high)** | `tutorials/` deleted in commit 194c63f; `_copy_tutorials.py:21-22` early-return 0 (is_dir=False). Copies 0 files. Clean build -> empty gallery. |
| Building the docs produces a populated tutorials gallery | sphinx/index.md | **false -- CONFIRMED (high)** | Consequence of the two above. `_build/` May 30 shows a gallery that no longer reflects the sources. (Adversarial correction: `_build/`/`_generated/` are NOT committed -- gitignored/untracked -- local artifacts only.) |
| Reconstruction limiter "MC" (monotonized-central) exists | ALGORITHMS:97, ARCHITECTURE:288/509 | **false -- CONFIRMED (medium)** | `reconstruction.hpp` defines only `NoSlope:35`/`Minmod:46`/`VanLeer:61`/`Weno5:124`. `grep 'struct MC|MonotonizedCentral'` -> none. `limiter='mc'` would fail. |
| `amr_coupler.hpp`/`AmrCoupler` = deprecated file kept | ARCHITECTURE:33, COUPLING_SURFACE:118, COUPLER_HIERARCHY Sec. 4 | **false -- CONFIRMED (medium)** | `ls include/pops/coupling/amr_coupler.hpp` -> absent (deleted #164). Only `AmrCouplerMP` exists. CODEBASE_AUDIT:210 records REMOVED; the 3 other docs over-describe a dead symbol. |
| `AmrSystem(n=128, max_level=2, periodic=True)` | DSL_API.md Sec. 3 (l.76) | **false -- CONFIRMED (medium)** | `AmrSystemConfig` has NO `max_level` field (`bindings.cpp:259-266`: n,L,regrid_every,periodic,distribute_coarse,coarse_max_grid). `setattr(config,'max_level',...)` fails. |
| For backend='production'/target='amr_system' the Python AMR facade rejects HLLC/Roe/primitive (single-block) | DSL_API.md Sec. 3 (l.86-88) | **obsolete -- CONFIRMED (medium)** | `AmrSystem.add_equation` (`__init__.py:1321-1334`) wires `recon='primitive'` and roe/hllc flux via `dispatch_amr_compiled`; `test_amr_riemann_native.cpp` exists. |
| CMake option `POPS_USE_EIGEN` (default ON, target `pops_eigen`) | README:167 (+ CI ci.yml:145, docs.yml:45) | **false -- CONFIRMED (medium)** | No `option(POPS_USE_EIGEN)`, no `pops_eigen` target, no `find_package(Eigen)`. `-DADC_USE_EIGEN` = silent no-op. |
| `python -c "...import pops..."` (and python3 fallback) imports the module | (prompt command) | **false -- CONFIRMED (high)** | `python` not found; `python3`=3.9.6 -> `ModuleNotFoundError: pops._pops` (`.so` cpython-312). Imports cleanly with `/opt/homebrew/anaconda3/bin/python3.12` + numpy. |
| ARCHITECTURE Sec. 11: 71 core ctests + 21 MPI + 26 Python | ARCHITECTURE:442-444 | **obsolete -- CONFIRMED (medium)** | Disk: ~90-94 `pops_add_test` (non-MPI, incl. elliptic_interface), ~51 MPI entries, 60 Python files. All underestimated. |
| BACKEND_COVERAGE base: 84 `pops_add_test` + 19 `add_executable` + 55 Python | BACKEND_COVERAGE:322-324 | **partially true -- CONFIRMED (medium)** | Real: 90 `pops_add_test` + 19 runtime = 109 non-MPI; MPI block 12 add_executable -> ~51 entries; 60 Python. Stale header totals, per-test lines OK. |
| FFT Poisson works under MPI | ARCHITECTURE Sec. 7 | **false -- CONFIRMED (medium)** | `test_mpi_system_fft.cpp` is a non-regression lock: the FFT path is REFUSED under MPI (n_ranks>1) with a collective error (SIGSEGV fix #93). FFT = single-rank by design. |
| `schur_magnetized_cartesian` uses `pops.Split(Explicit, CondensedSchur)` | schur_magnetized_cartesian/run.py:130-146 | **obsolete -- CONFIRMED (medium)** | The code calls the PRIVATE hook `sim._s.set_source_stage(...)` (`run.py:142`) because the AOT ABI does not carry SSPRK3 (README:68-71). Same underlying `CondensedSchurSourceStepper` C++. |
| Python suite run under pytest | (implicit docs) | **false -- CONFIRMED (medium)** | `ci.yml:149`: `find python/tests -maxdepth 1 -name 'test_*.py' | xargs python3`. No pytest; tests = standalone scripts with assert+exit. |
| adc_cases top-level README "The cases" = complete set | adc_cases/README.md:107-119 | **partially true -- CONFIRMED (medium)** | Table lists 10 cases, omits diocotron_dsl, two_species_dsl, magnetic_isothermal_dsl, dsl_euler, hoffart_euler_poisson_dsl, schur_magnetized_cartesian (present in manifest). |
| diocotron figures "produced in diocotron/figures/" | diocotron/run.py:13, README:36 | **false -- CONFIRMED (high)** | `run.py:207` writes to `out/diocotron/` (gitignore); all savefig (`:246,257,276,286`) go there. No script writes `figures/`. The 4 committed figures are stale manual copies. |
| Each run emits a record capturing adc_cpp SHA + adc_cases SHA | hoffart_euler_poisson_dsl/README:143 | **partially true -- CONFIRMED (high)** | `run.py:422-423` writes the SHAs, BUT the `metadata.json` on disk do NOT have the keys `pops_cpp_sha`/`pops_cases_sha` (artifacts of an earlier run); the 4 diocotron figures have no metadata. |
| "Reproduction of the Hoffart diocotron benchmark" (title) | diocotron/run.py:2-11 | **partially true -- CONFIRMED (low)** | The same docstring + README:8-10 qualifies "reduced ExB model, NOT a full Euler-Poisson reproduction". Reproduces the analytic oracle to 3 digits; FV rates under-predicted -22/-27/-5%. The word "Reproduction" is disambiguated everywhere. |
| AOT two-fluid AP validated by a core test | tests/test_ap_limit.cpp | **partially true -- CONFIRMED (medium)** | `test_ap_limit.cpp`/`test_imex_ap.cpp` validate the AP PROPERTY on a scalar TOY model (du/dt=(u_eq-u)/eps), not the real two-fluid. The real integrator lives in `adc_cases/two_fluid_ap/` (scenario, not brick). |
| CUDA single-GPU / multi-GPU validated by automatic test | BACKEND_COVERAGE | **partially true -- CONFIRMED (high)** | CI never builds `Kokkos_ENABLE_CUDA=ON`. Device validation = ROMEO-manual via `python/tests/gpu/*.cpp` (excluded from CI glob `-maxdepth 1`). Say "manually on GH200". |
| DSL production/hybrid validated on Kokkos Cuda device | GPU_RUNTIME_PORT | **partially true -- CONFIRMED (high)** | ROMEO-only; `test_compiled_model_parity` NOT device-validated (segfault nvcc extended-lambda cross-TU). Named-functor workaround not ported to this test. |
| All hoffart outputs written "for each mode" 3/4/5 | hoffart_euler_poisson_dsl/README:192-200 | **partially true -- CONFIRMED (low)** | On disk only `mode_3/` exists for both engines; mode_4/5 absent (table marked PENDING). |

**Not-confirmed / to review (thin data):** the adversarial verification covered only ~7 claims. For ALGORITHMS ("all paths/tests exist"), the EllipticSolver/PhysicalModel concepts, single-box FFT, WENO5-Z Borges, GeometricMG eps/reaction/aniso, the `schur_magnetized_cartesian` timing (562x/1000x) and the ROMEO multi-GPU "?" matrix: verdicts **true/non-verifiable** from the inventory alone, to confirm in Phase 2 by targeted execution/grep.

---

## 4. Real API to document (autodoc)

Real public surface from `api_entries`. Python symbols = `python/pops/__init__.py` + `dsl.py` + `integrate.py`, bindings in `python/bindings/core/bindings.cpp` (the only `.def()` file -- `system.cpp`/`amr_system.cpp` are pure C++ without pybind). C++ concepts/classes = `include/pops/**`. `doc_status`: documented / wrong-docstring / unverified.

### C++ -- core concepts & classes (seed for C++ Reference / Doxygen)

| Symbol | Module | doc_status | Test evidence |
|---|---|---|---|
| `PhysicalModel` (concept) | `core/physical_model.hpp` | documented (but ARCH:102 over-lists `wave_speeds` as required) | `test_spatial_discretisation.cpp` |
| `HasPrimitiveVars`/`HyperbolicPhysicalModel` | `core/physical_model.hpp` | documented | `test_primitive_recon.cpp` |
| `EllipticSolver` (concept) | `numerics/elliptic/elliptic_solver.hpp` | documented | `test_elliptic_interface.cpp` |
| `CompositeModel` | `physics/composite.hpp` | documented | `test_amr_composite_source_conservation.cpp` |
| `RusanovFlux`/`HLLFlux`/`HLLCFlux`/`RoeFlux` | `numerics/numerical_flux.hpp` | documented | `test_roe_flux.cpp` |
| `NoSlope`/`Minmod`/`VanLeer`/`Weno5` | `numerics/reconstruction.hpp` | **wrong-docstring** (no `MC`) | `test_weno_convergence.cpp` |
| `EquationBlock` | `core/equation_block.hpp` | documented | `test_system_abstraction.cpp` |
| `CoupledSystem` | `core/coupled_system.hpp` | documented | `test_two_species_minimal.cpp` |
| `GeometricMG` (set_epsilon/set_reaction/set_epsilon_anisotropic/solve_robust) | `numerics/elliptic/geometric_mg.hpp` | documented | `test_geometric_mg.cpp` |
| `PoissonFFTSolver` (single-rank/single-box, throw if n_ranks!=1) | `numerics/elliptic/poisson_fft_solver.hpp` | documented | `test_poisson_fft.cpp` |
| `DistributedFFTSolver` (band MPI_Alltoall) | `numerics/elliptic/poisson_fft_solver.hpp` | documented | `test_mpi_fft_distributed.cpp` |
| `Coupler<Model,Elliptic>` | `coupling/coupler.hpp` | documented | `test_mpi_coupler_inject.cpp` |
| `SystemCoupler`/`SystemDriver`/`SystemAssembler` | `coupling/system_coupler.hpp` | documented | `test_assembler_driver.cpp` |
| `AmrCouplerMP` (legacy `AmrCoupler` DELETED) | `coupling/amr_coupler_mp.hpp` | documented | `test_mpi_amr_compiled_parity.cpp` |
| `AmrSystemCoupler` | `coupling/amr_system_coupler.hpp` | documented | `test_amr_system_coupler.cpp` |
| `System` (pimpl SystemStepper/FieldSolver/BlockStore) | `runtime/system.hpp` | documented | `python/tests/test_bindings.py` |
| `AmrSystem` (single- OR multi-block, recon/riemann/imex) | `runtime/amr_system.hpp` | **wrong-docstring** (docs say single-block) | `test_amr_system_twoblock.cpp` |
| `advance_amr` (defined in `amr_advance.hpp`, included by `amr_reflux_mf.hpp`) | `numerics/time/amr_advance.hpp` | documented | `test_amr_compiled_model.cpp` |
| `for_each_cell` (Kokkos/OpenMP/serial seam) | `mesh/for_each.hpp` | documented | `test_reduce.cpp` |
| `comm` (my_rank/n_ranks/all_reduce_*/barrier) | `parallel/comm.hpp` | documented | `test_mpi_array_reduce.cpp` |
| `ModelSpec`/`model_factory` (dispatch_model) | `runtime/model_factory.hpp` | documented | `python/tests/test_dsl_compose.py` |
| Bricks `ExBVelocity`/`IsothermalFlux`/`PotentialForce`/`GravityForce`/`ChargeDensity`/`BackgroundDensity`/`NoSource` | `physics/{hyperbolic,source,elliptic}.hpp` | documented (ARCH says "ExB" instead of `ExBVelocity`) | `test_coupled_source.cpp` |

### Python -- `pops.System` (single-level composition facade)

| Symbol | Signature | doc_status | Test evidence |
|---|---|---|---|
| `System(config, mesh, **cfg_kw)` | constructor | documented | `test_bindings.py` |
| `add_block` / `add_equation` / `add_background` | block composition (add_equation type-dispatch ModelSpec vs CompiledModel) | documented | `test_bindings.py`, `test_dsl_production.py`, `test_dsl_dynamic.py` |
| `add_coupling` / `add_coupled_source` / `add_ionization` / `add_collision` / `add_thermal_exchange` | inter-species couplings | documented | `test_dsl_coupled_source.py` |
| `add_elliptic_model` | EPM (operator/rhs/output) | documented | `test_poisson_eps.py` |
| `set_poisson(rhs,solver,bc,wall,wall_radius,epsilon)` | Poisson config | documented | `test_bindings.py` |
| `set_density` / `set_primitive_state` / `get_primitive_state` / `set_block_params` | state I/O | documented | `test_bindings.py`, `test_primitive_state.py`, `test_dsl_runtime_params.py` |
| `set_magnetic_field` / `set_epsilon_field` / `set_epsilon_anisotropic_field` | extended aux fields | documented | `test_magnetic_field.py`, `test_poisson_eps_aniso.py` |
| `set_source_stage` / `set_time_scheme` | source stage (Schur) / time scheme | documented | `test_schur_via_system.py`, `test_strang_split.py` |
| `step` / `advance` / `step_cfl` / `step_adaptive` / `run` | advancement | documented | `test_bindings.py`, `test_stride.py` |
| `solve_fields` / `eval_rhs` / `get_state` / `set_state` | fields/RHS | documented | `test_weno5_ssprk3.py` |
| `set_disc_domain` / `disc_mask` | disc domain | documented | `test_disc_domain_mask.py` |
| `block_names`/`mass`/`density`/`potential`/`nx`/`ny`/`time`/`n_species`/`n_vars`/`variable_names`/`variable_roles`/`block_gamma`/`abi_key` | introspection | documented | `test_bindings.py`, `test_dsl_roles.py`, `test_dsl_abi_metadata.py` |

### Python -- `pops.AmrSystem` (AMR facade; real multi-block)

| Symbol | doc_status | Test evidence |
|---|---|---|
| `AmrSystem(config, **cfg_kw)`, `add_block`, `add_equation` | documented | `test_amr_multiblock.py`, `test_dsl_production_amr.py` |
| `set_refinement` / `set_phi_refinement` / `set_poisson` | documented | `test_amr_multiblock.py` |
| `AmrSystemConfig` (n,L,regrid_every,periodic,distribute_coarse,coarse_max_grid -- **no max_level**) | documented | `test_amr_multiblock.py` |
| `SystemConfig` (n,L,periodic,geometry,nr,ntheta,r_min,r_max) | documented | `test_polar_system.py` |

### Python -- composition bricks (`pops.Model(...)`)

| Symbol | doc_status | Test evidence |
|---|---|---|
| `Model(state,transport,source,elliptic)->ModelSpec`, `ModelSpec` | documented | `test_bindings.py` |
| `CompositeModel(...)->dsl.HybridModel` | documented | `test_dsl_hybrid.py` |
| `Scalar`/`FluidState`, `ExB`/`CompressibleFlux`/`IsothermalFlux` | documented | `test_bindings.py` |
| `NoSource`/`PotentialForce`/`GravityForce`, `ChargeDensity`/`BackgroundDensity`/`GravityCoupling` | documented | `test_bindings.py` |
| `Spatial(limiter,flux,recon)` / `FiniteVolume(...)` | documented | `test_weno5_compiledmodel.py`, `test_dsl_recon.py` |
| `Explicit` / `IMEX` / `SourceImplicit` / `Implicit` (**OBSOLETE, DeprecationWarning -> IMEX**) | documented | `test_stride.py`, `test_time_policy.py` |
| `Split` / `Strang` / `CondensedSchur` (role/field descriptors hardcoded C++) | documented | `test_strang_split.py`, `test_schur_via_system.py` |
| `Role`, `CartesianMesh`/`PolarMesh`, `Ionization`/`Collision`/`ThermalExchange` | documented | `test_dsl_roles.py`, `test_polar_system.py`, `test_dsl_coupled.py` |
| EPM: `elliptic`/`div_eps_grad`/`charge_density`/`composite_rhs`/`electric_field_from_potential`, `EllipticSolver`/`EllipticModel`/`DivEpsGrad`/`CompositeRhs`/`ChargeDensitySource` | documented | `test_poisson_eps.py`, `test_poisson_composite.py` |
| `PythonFlux` (interpreted host backend) | documented | `test_dsl.py` |

### Python -- `pops.dsl` (symbolic DSL) & `pops.integrate`

| Symbol | doc_status | Test evidence |
|---|---|---|
| `dsl.Model` (conservative_vars/primitive/aux/flux/eigenvalues/source/elliptic_rhs/param/compile) | documented (but `param` runtime docstring = **wrong**) | `test_dsl_phase_a.py`, `test_dsl_production.py` |
| `dsl.Model.compile(backend='aot'|'production'|'prototype', target='system'|'amr_system', ...)` | documented (real default = `aot`) | `test_dsl_compile_facade.py`, `test_dsl_compile_cache.py` |
| `dsl.CompiledModel` (backend/adder/so_path/caps/abi_key/runtime_param_*) | documented | `test_dsl_compile_facade.py` |
| `dsl.HyperbolicModel`, `dsl.Param`/`RuntimeParam`, `dsl.sqrt`/`Expr` | documented (Param: see wrong-docstring) | `test_dsl_codegen.py`, `test_dsl_runtime_params.py`, `test_dsl_cse.py` |
| `dsl.HyperbolicBrick`/`SourceBrick`/`EllipticBrick`, `dsl.HybridModel`, `dsl.NativeBrick`/`CompiledBrick` | documented | `test_dsl_brick.py`, `test_dsl_hybrid*.py` |
| `dsl.CoupledSource`/`CompiledCoupledSource` (bytecode couplings) | documented | `test_dsl_coupled_source*.py` |
| `integrate.euler_step` / `ssprk2_step` | documented | `test_dsl.py` |
| `abi_key` (module-level) | documented | `test_dsl_abi_metadata.py` |

**Backend capabilities matrix (load-bearing, lives in `_BACKEND_CAPS` `dsl.py:1382-1393`):** prototype/aot = CPU-only, no MPI/AMR/GPU; production = CPU+MPI+AMR, **gpu=False on the Python side** (out of caution, despite device-clean C++). Restrictions applied as explicit `ValueError` in `add_equation` (stride>1 on aot; evolve=False on prototype/aot; weno5 on prototype; implicit_vars on compiled .so; HLLC/Roe requires a primitive 'p'). Any "feature X works on backend Y" doc must match these guards.

---

## 5. adc_cases cases -- truth per case

15 case folders (the manifest has 16 entries because diocotron has 2 scripts: `run.py` + `band_instability.py`). Model: native (composed C++ bricks) / dsl (compiled symbolic formulas) / hybrid / interpreted / bespoke-C++. CI/category = `cases_manifest.toml`.

| Case | README? | Model | Time scheme | Space scheme | Poisson | Real backends | CI/category | Assets | Honest limits |
|---|---|---|---|---|---|---|---|---|---|
| composition | no | native (electron_euler + ion_isothermal + diocotron) | IMEX(10) e- / Explicit ions; Part D = SSPRK2 Python | per-block vanleer/hllc + minmod/rusanov | charge_density, geometric_mg | adc compile | true / tutorial | none (prints) | No physics claim; tests heterogeneous composition + bit determinism |
| custom_scheme | no | pure numpy (ExB+upwind); adc = Poisson oracle only | SSPRK2 Python, dt=CFL*dx/v | central + upwind numpy | charge_density, geometric_mg (only lib call) | adc (Poisson only) | true / tutorial | none | Entire scheme in Python; lib = elliptic solver |
| **diocotron** | **yes** | **REDUCED native scalar ExB** (Scalar+ExB+BackgroundDensity) -- **NOT full Euler-Poisson** | Explicit SSPRK2 (CFL=0.4) | MUSCL minmod + Rusanov | dirichlet, wall=circle R=0.40 | adc + matplotlib | run.py false/**reproduction**; band_instability true/validation | figures/ committed (dispersion/amplitude/snapshots/diocotron.gif) | **HONEST**: README+docstring say "reduced-model normalization, NOT full system repro". Petri oracle to 3 digits; FV under-predicts -22/-27/-5% |
| diocotron_amr | no | native diocotron on AmrSystem | Explicit (CFL=0.4) | NoSlope + Rusanov o1 | charge_density, geometric_mg periodic | adc (AmrSystem) | true / validation | none | Multi-patch Berger-Rigoutsos + reflux; tests real refinement vs control |
| diocotron_dsl | no | dsl (formulas == ExBVelocity+BackgroundDensity) | Explicit SSPRK2 (== native) | FiniteVolume(minmod, rusanov) | charge_density, geometric_mg | dsl production->aot fallback; needs cxx | true / validation (needs cxx) | out/*.so | Claim: state **bit-identical** to native (np.array_equal) |
| dsl_euler | no | dsl HyperbolicModel Euler 2D (no source/Poisson) | forward Euler, cfl_dt | Rusanov via `PythonFlux` (**interpreted numpy**) | none | interpreted host (PythonFlux) | false / **experimental** | none | INTERPRETED prototype, distinct from the compiled path of the other *_dsl |
| euler_poisson | no | native euler_poisson (GravityForce + GravityCoupling +/-) | Explicit SSPRK2, 20 steps | vanleer/hllc | charge_density, geometric_mg periodic | adc compile | true / validation | none | Coupling sign contrast (attractive dE<0 / repulsive dE>0); no paper claim |
| **hoffart_euler_poisson_dsl** | **yes** | **FULL magnetic Euler-Poisson dsl** (continuity + momentum + Lorentz, barotropic); schur/local variants | system-schur SSPRK3 + CondensedSchur(theta=0.5); amr-imex backward-Euler | FiniteVolume WENO5-Z + Rusanov | composite, dirichlet, wall=circle R=16 | dsl production (system/amr_system); amr-imex needs Kokkos/MPI | check_model true/validation; run.py false/**reproduction-candidate** | out/.../{amplitude,snapshots,growth_rates,gifs,metadata} | **HONEST**: banner "NO established quantitative repro"; table -82 to -95% (n=256/384), n=512 PENDING, amr-imex PENDING. Gaps: Lie not Strang, Poisson once-per-step, cart+wall geometry suspect |
| **magnetic_isothermal_dsl** | **yes** | dsl magnetized isothermal (aux B_z index 3); **no native oracle** | Explicit SSPRK2, 40 steps | FiniteVolume(minmod, rusanov) | charge_density, geometric_mg periodic; set_magnetic_field | dsl production+aot (macOS=aot only); needs cxx | true / validation | out/*.so | Validated WITHOUT native oracle: inter-backend parity + analytic Lorentz oracle (dmax==0) |
| multispecies | no | native (e- Euler 4-var + ions isothermal 3-var) coupled 1 Poisson | Explicit, dt=0.001 | minmod both | charge_density, geometric_mg periodic | adc compile | true / validation | none | Per-species mass conservation <1e-9 |
| plasma | no | native recipe 3-species (e- HLLC+primitive, ions+neutrals isothermal) + ionization+collision | Explicit step_cfl(0.3) | vanleer/hllc/primitive + minmod | f=q_e n_e+q_i n_i | adc compile | true / validation | none | Honest: momentum/energy transfer of created particles = simplification |
| **schur_magnetized_cartesian** | **yes** | dsl magnetized isothermal; local/schur variants (same equations as magnetic_isothermal) | transport SSPRK2; explicit source OR CondensedSchur via `sim._s.set_source_stage` (private hook) | FiniteVolume(minmod, rusanov, conservative) | charge_density, geometric_mg periodic; B_z=omega_c | dsl aot (production fails dlopen macOS arm64); needs cxx | false / **experimental** | out/dt_stable.csv | Timing study: explicit plateaus dt*omega_c~0.3, Schur 178-316 (562x/1000x gain). Uses private hook because `pops.Split` not wired on the AOT ABI |
| two_euler | no | native euler x2 independent gases (NOT coupled) | multirate step_adaptive(0.4) | vanleer/hllc/primitive | f=0 (just for solve_fields) | adc compile | true / validation | none | "2 Euler same code"; independent blocks |
| two_fluid_ap | no | **bespoke C++** (two_fluid_ap.hpp) isothermal AP; NOT a composable brick | IMEX/AP (stiff implicit term) | FV continuity, in the JIT | elliptic AP reformulated in the C++ scenario | JIT .so via ctypes (build_shared); needs cxx C++20 | true / validation (needs cxx) | out/.../_two_fluid_ap.{so,dylib} | Replaces the `_TwoFluidAP` escape hatch removed from the core; lives in adc_cases |
| two_species_dsl | no | dsl x2 (e- Euler 4-var + ions isothermal 3-var) + electrostatic source, 1 Poisson | Explicit SSPRK2 (== native) | FiniteVolume(minmod, rusanov) | charge_density, geometric_mg periodic | dsl production->aot; needs cxx | true / validation (needs cxx) | out/*.so | Ions bit-identical; e- differ ~machine-eps (<1e-24, float reassociation in shared Poisson RHS, documented) |

**diocotron-vs-Hoffart honesty (CONFIRMED):** two DIFFERENT physics models share the name "diocotron" and the ref arXiv:2510.11808: (a) the scalar REDUCED ExB (diocotron, custom_scheme, diocotron_amr, diocotron_dsl) and (b) the FULL magnetic Euler-Poisson (hoffart_euler_poisson_dsl). The manifest classifies (a)/run.py "reproduction" (of the analytic normalization only) and (b) "reproduction-candidate" (PENDING table). Repo-wide grep for "complete/full reproduction" returns only disclaimers. `results.py` (verify_paper_windows, engine_label) **mechanically refuses** to mix the 2pi/rho rates of the reduced with the raw numbers of the full. No file lies; the residual risk is in prose: the word "Reproduction" in the diocotron title, out of context, oversells (it reproduces the analytic oracle, not the full system, and under-predicts down to -27%). Recommendation: replace with "reduced-model normalization benchmark".

---

## 6. Assets & provenance

All image assets live under adc_cases. The adc_cpp submodule is NOT present on disk (no `.gitmodules`, no gitlink) -> no figure under `adc_cpp/docs/` can be inspected locally.

**Counters:**

| Metric | Value |
|---|---|
| Total image files | 11 (4 PNG + GIF tracked; 7 untracked) |
| Tracked (canonical) | 4 -- `diocotron/figures/{dispersion,amplitude,snapshots}.png` + `diocotron.gif` |
| Orphans (referenced only by generic phrase, not by name) | 3 -- `mode_3/snapshots.png` (x2 engines), `mode_3/diocotron_l3.gif` |
| No provenance (no SHA/backend/resolution) | 11/11 |
| Broken image links (embeds `![]()` / `<img>`) | **0** -- the whole repo contains NO markdown embed; figures referenced by name in prose/table only |

**Broken embeds:** none. The repro risk is not the dangling `![]()` but the **path mismatch** and the **lack of provenance**.

**Untracked assets (out/, gitignored) -> regenerate decision:**

| Asset | Producer | Referenced by | Decision | Note |
|---|---|---|---|---|
| out/hoffart_..._system_schur/growth_rates.png | run.py:469 | README:200 (by name) | regenerate | metadata.json without SHA (earlier run) |
| out/hoffart_..._system_schur/mode_3/amplitude.png | run.py:320 | README:197 | regenerate | only mode_3 exists (3/4/5 expected) |
| out/hoffart_..._system_schur/mode_3/snapshots.png | run.py:339 | **orphan** ("3x3 panel" generic) | regenerate | no SHA |
| out/hoffart_..._system_schur/mode_3/diocotron_l3.gif | run.py:360-361 | **orphan** ("an animated GIF") | regenerate | dynamic mode-indexed name |
| out/hoffart_..._amr_imex/growth_rates.png | run.py:469 | README:200 | regenerate | experimental engine, no SHA |
| out/hoffart_..._amr_imex/mode_3/amplitude.png | run.py:320 | README:197 | regenerate | experimental |
| out/hoffart_..._amr_imex/mode_3/snapshots.png | run.py:339 | **orphan** | regenerate | experimental |

**Tracked assets (canonical) -> keep decision, but provenance to reconstruct:**

| Asset | Real producer | Decision | Note |
|---|---|---|---|
| diocotron/figures/dispersion.png | run.py:246 (writes to `out/diocotron/`, NOT figures/) | keep | 806x559; no SHA/backend; manual copy |
| diocotron/figures/amplitude.png | run.py:257 (-> out/) | keep | same |
| diocotron/figures/snapshots.png | run.py:286 (-> out/) | keep | 1690x442; no provenance |
| diocotron/figures/diocotron.gif | run.py:276 (-> out/) | keep | 420x420 ~512KB; manual copy |

**Phase 5 provenance decisions:** (1) repoint the `savefig` of `run.py` to `figures/` OR explicitly document the copy step; (2) regenerate with the current `run.py` to populate `pops_cpp_sha`/`pops_cases_sha` then freeze; (3) decide which hoffart figures are promoted (committed) vs left ephemeral; (4) freeze the exact modes (3/4/5) + n + dt + engine + SHA; (5) add references by filename or an asset manifest so renames are detected by a link-check.

---

## 7. Ordered integration plan (Phases 2-6)

Execution legend: **[L]** entirely doable locally on darwin/arm64; **[ROMEO]** requires the cluster (CUDA/MPI multi-GPU) to regenerate assets or validate; **[TC]** blocked/conditioned by the doc build toolchain (3.12 interpreter + numpy + C++23 compiler).

### Phase 2 -- Hygiene & skeleton (prerequisites)

- **PR-01 (adc_cpp)** -- *Remove broken gallery mechanism + stale cruft.* Delete `_copy_tutorials.py`, remove the `setup_gallery()` call from `conf.py`, delete locally `_build/`/`_generated/`/`__pycache__` (already gitignored, no gitignore change required). depends-on: none. **[L]**. Prerequisite for everything else Sphinx.
- **PR-02 (adc_cpp)** -- *Target Sphinx skeleton.* 7-section toctree (Getting Started / Models / Simulation / AMR / Parallel Backends / Advanced / Reference); merge `tutorials_index.md` + `examples.md` into a single "Examples -> adc_cases" page; integrate the canonical docs (`ARCHITECTURE`/`ALGORITHMS`/`BACKEND_COVERAGE`/`AMR_*`) as Sphinx pages instead of raw GitHub URLs. depends-on: PR-01. **[L]**.

### Phase 3 -- Fixing the false claims (truth)

- **PR-03 (adc_cpp)** -- *Purge "AmrSystem single-block".* README:130-132, ARCHITECTURE:26/379-382/547, DSL_MODEL_DESIGN Sec. 0bis/Sec. 5/Phase D, DSL_API Sec. 3 l.86-88; also fix the stale comment in the `amr_system.hpp:28-31` header. depends-on: none (can parallelize PR-02). **[L]**. Risk #1.
- **PR-04 (adc_cpp)** -- *Remove dead file/symbol.* Remove `amr_coupler.hpp`/`AmrCoupler` from ARCHITECTURE:33, COUPLING_SURFACE:118, COUPLER_HIERARCHY Sec. 4; remove the phantom "MC" limiter from ALGORITHMS:97 and ARCHITECTURE:288/509; fix "ExB"->`ExBVelocity` and `wave_speeds` outside the concept (ARCH:102). depends-on: none. **[L]**.
- **PR-05 (adc_cpp)** -- *Rewrite DSL_API.md + source docstring.* Regenerate the 4 false snippets (conservative_vars/aux, set_poisson(solver=), no max_level, runtime params work); **fix the `dsl.py:1623/1627` docstring** (false NotImplementedError) so autodoc does not propagate; adjust compile default (`aot`), flux tokens ('hll' invalid), GPU production cap=False. depends-on: none. **[L]**.
- **PR-06 (adc_cpp)** -- *Regenerate test counters programmatically.* Script that counts from `tests/CMakeLists.txt` + glob `python/tests/test_*.py`, injects into BACKEND_COVERAGE (SoT); ARCHITECTURE Sec. 11 and installation.md reference instead of hard-coding; fix `POPS_USE_EIGEN` (README:167 + CI) -- remove the phantom line. Remove the local Finder-copy artifacts (`test_polar_system 2.py`, `test_polar_system_step 2.cpp`). depends-on: none. **[L]**.
- **PR-07 (adc_cases)** -- *Reconcile top-level README with the manifest.* Add the 6 missing DSL cases to the "The cases" table; fix `schur_magnetized_cartesian` docstring (private hook `sim._s.set_source_stage`, not `pops.Split`). depends-on: none. **[L]**.

### Phase 4 -- Autodoc & Reference

- **PR-08 (adc_cpp)** -- *Stabilize the autodoc env.* Add `numpy` to `requirements.txt`, remove the unused `sphinx-autodoc-typehints`, document/pin the exact interpreter (cpython-3.12 + numpy + `.so` on PYTHONPATH); add the ABI/interpreter caveat to the Python quickstart README. depends-on: PR-02. **[L] [TC]** (local doc build OK with anaconda 3.12).
- **PR-09 (adc_cpp)** -- *Reference section (Python autodoc).* Models/Simulation/AMR pages from the Sec. 4 surface; make explicit add_block(ModelSpec) vs add_equation(type-dispatch), the 3 authoring paths (native/dsl/hybrid), the `_BACKEND_CAPS` matrix and the ValueError guards. depends-on: PR-05, PR-08. **[L]**.
- **PR-10 (adc_cpp)** -- *C++ Reference section (Doxygen).* Sec. 4 concepts/classes; mark `AmrSystem`/reconstruction wrong-docstring fixed upstream (PR-03/04). depends-on: PR-04. **[L]**.

### Phase 5 -- Assets & provenance

- **PR-11 (adc_cases)** -- *Fix the diocotron path mismatch.* Repoint the `savefig` to `figures/` (or document the copy step); add SHA capture to the runs. depends-on: none. **[L]** (diocotron runs on CPU).
- **PR-12 (adc_cases)** -- *Asset manifest + frozen provenance.* Regenerate the diocotron figures with SHA, freeze n/dt/backend; add references by filename (kill the 3 orphans). depends-on: PR-11. **[L]** for diocotron.
- **PR-13 (adc_cases)** -- *Hoffart system-schur figures (modes 3/4/5).* Decide committed promotion vs ephemeral; regenerate with SHA. depends-on: PR-12. **[L]** for system-schur (cart-square CPU); **[ROMEO]** for the **amr-imex** variant (needs Kokkos/MPI build + `--acknowledge-amr-approximation`).

### Phase 6 -- Doc CI

- **PR-14 (adc_cpp)** -- *CI `sphinx-build -W` + doxygen.* Job that builds with warnings-as-errors (catches orphan-toctree, broken refs) on a runner with 3.12+numpy+`.so`; build Doxygen. depends-on: PR-01..PR-10 (otherwise `-W` fails on the dead gallery and broken refs). **[L] [TC]** (the CI runner must have the doc toolchain; no CUDA required).

**Local vs cluster synthesis:** only **PR-13 (amr-imex variant)** has a hard ROMEO dependency to regenerate an asset. Everything else -- hygiene, skeleton, truth corrections, autodoc, Doxygen, doc CI, diocotron + hoffart system-schur figures -- is **entirely doable locally on darwin/arm64** with the anaconda 3.12 + numpy interpreter. No claim fix requires the GPU; device validation stays out of refactor scope (ROMEO-manual).

---

## 8. Risks & blind spots

**Consolidated key risks (confirmed or inventory):**

- **The "AmrSystem single-block" narrative is accuracy risk #1 and it is anchored down to the persistent memory.** Multi-block/multi-species/summed Poisson/coupled-source/union-tags regrid have all been shipped (#195/#199/#205, 7 capstone tests). Honest nuance to preserve: "not at COMPLETE parity" remains partially true (no GLOBAL Schur on AMR), but the current wording is false on 4 axes.
- **References to a deleted file in 3 normative docs** (`amr_coupler.hpp`) + **phantom MC symbol**: any autodoc/Doxygen built on them would list nonexistent entities.
- **Test counters: no coherent value between docs and disk.** Choose disk as authority and generate programmatically (PR-06), never hard-code.
- **"DESIGN-ONLY" specs already implemented** (`SCHUR_CONDENSATION_DESIGN` "nothing shipped", `AMR_MULTIBLOCK_DESIGN` "facade stays single-block") describe a past state: to archive/stamp IMPLEMENTED, not to use as a normative source.
- **Cross-repo authority not verifiable from a single repo:** CONSERVATION_SUMMARY/HOFFART_FIDELITY/HOFFART_STEP_SEQUENCE anchor their claims to adc_cases files (`model.py`, `run.py`, `test_schur_conservation.py`) absent from adc_cpp; keep these claims clearly labeled "application-side validation".
- **Widespread line-number rot:** most canonical docs embed file:line anchors shifted by the #151 native_loader extraction and other refactors (e.g. `AmrSystem.potential()` is at `bindings.cpp:332`, not :272). Disclaimed "indicative" but misleading if read as exact.
- **Dead Sphinx gallery + pinned Python ABI:** a `sphinx-build -W` will fail until PR-01 is done; autodoc silently degrades to signature-less stubs outside the exact 3.12+numpy interpreter.
- **adc_cases:** incomplete top-level README (6 DSL cases omitted); weak figure provenance (committed without SHA, produced elsewhere than their path); brittle `schur` coupling (private hook); only 4/15 cases documented.

**What the audit COULD NOT verify (to check BEFORE publication):**

- **adc_cpp submodule absent from the adc_cases checkout** (no `.gitmodules`/gitlink): all `../../adc_cpp/docs/*.md` links are unresolvable here and would break for a standalone adc_cases clone. No figure/doc on the adc_cpp side could be opened from this checkout.
- **Adversarial pass coverage limited to ~7 claims.** The "true/non-verifiable" verdicts for the rest (ALGORITHMS test exhaustiveness, concepts, single-box FFT, WENO5-Z Borges, GeometricMG, 562x/1000x timing of schur_magnetized_cartesian, ROMEO multi-GPU "?" matrix) rest on the inventory alone.
- **No code executed for quickstart.md / the Python recipes** (no numpy+py3.12 env in the inventory pass): plausible but unexercised API surface.
- **All perf and fidelity numbers** (PERFORMANCE M1, PROFILE_RESULTS, hoffart -82/-95% table, Schur gains) are measured and not reproducible locally (M1/GH200); not re-confirmed.
- **DSL Phase F status "merged June 2026" (memory)** not verified in the inventory; to confirm before removing any "unmerged branch" mention.
- **Exact counters** (90 vs 94 `pops_add_test`, 60 vs 61 Python) diverge slightly between the two inventory agents depending on how duplicated macros/lines are handled: regenerate canonically before publishing a figure.
---

## 9. Coordinator corrections (post-synthesis)

Verifications done by the coordinator after the 8-agent pass, correcting scope errors of a finder.

### 9.1 Sec. 6 WRONG: adc_cpp IS on disk and carries 33 tracked image assets

The "assets" finder wrongly concluded that "the adc_cpp submodule is NOT present on disk" and that "all assets live under adc_cases" (cwd too narrow). **False.** `adc_cpp/docs/` contains **33 tracked images** (20 PNG + 13 GIF). The Phase 5 scope is therefore ~2x wider than announced in Sec. 6.

**Real reference map (adc_cpp/docs/, excluding `_build/`):**

| Referenced by | Assets | Refactor status |
|---|---|---|
| **Live README (1)** | `anim_romeo_diocotron_amr3.gif` (hero GIF, README:12) | **keep** -- the only asset of the living portal; ROMEO provenance to document |
| **DEAD gallery `_generated/tutorials/*` only (6)** | `fig_diocotron_growth.png`, `fig_diocotron_modes.png`, `anim_diocotron.gif`, `anim_diocotron_column.gif`, `anim_diocotron_amr3.gif`, `anim_diocotron_multipatch.gif` | **orphans after PR-01** (deleting the dead gallery makes them unreferenced) -> reassign to the new tutorials section OR document/delete |
| **`docs/archive/*` only (8)** | `fig_diocotron_amr_vs_uniforme.png`, `fig_diocotron_conv_modes.png`, `fig_diocotron_highorder.png`, `fig_diocotron_invariants.png`, `fig_diocotron_ml_convergence.png`, `fig_diocotron_reproduction.png`, `romeo_amr_efficiency.png`, `romeo_growth_mode4.png`, `romeo_highorder_convergence.png`, `anim_magnetic_diocotron.gif` | archive-only -> **keep with the archive** or move under `docs/archive/assets/` |
| **`docs/PERFORMANCE.md` (historical) (1)** | `fig_openmp_scaling.png` (also dead gallery) | follows the PERFORMANCE.md decision (archive) |
| **COMPLETE ORPHANS -- the `tut_*` set (10)** | `tut_diocotron_growth.png`, `tut_diocotron_sequence.png`, `tut_euler_poisson.png`, `tut_plasma.png`, `tut_poisson_backends.png`, `tut_tfap_ap.png`, `tut_diocotron_py.gif`, `tut_diocotron_ring.gif`, `tut_ep_collapse.gif`, `tut_tfap_field.gif` | **already orphans**: this was the asset pool of the Sphinx tutorials moved to adc_cases (`tutorials/` deletion 194c63f). Phase 5 decision: regenerate-with-provenance for the new gallery, OR delete. NONE has provenance. |

**Consequence for the plan:** PR-12/PR-13 (assets) must cover **two** repos, not just adc_cases. Deleting the dead gallery (PR-01) **increases** the count of adc_cpp orphans from 10 -> 16; an explicit "regenerate for the new gallery vs delete" decision is needed on the 16, with an asset manifest (producer script + command + adc_cpp/adc_cases SHA + backend + resolution) as Phase 5 requires.

### 9.2 Sec. 6 WRONG: "0 broken embed" underestimates -- there are real embeds on the adc_cpp side

The adc_cpp repo does contain markdown/HTML image embeds (`README.md:12` `<img src="docs/...">`, `docs/archive/*` and the old gallery in `![]()`). The Phase 6 link-check (`sphinx-build -W`) must therefore cover adc_cpp, not just verify the absence of embeds. Confirmed real risk: after PR-01, 6 embeds of the dead gallery will point to deleted pages -- to clean up in the same PR.

### 9.3 To reconfirm before publication (unchanged)

The Sec. 8 blind spots remain valid: limited adversarial coverage (~7 claims), no execution of quickstart.md, test counters to regenerate canonically, and the "AmrSystem single-block" status in the coordinator's **persistent memory** must be corrected (multi-block shipped #195/#199/#205).
