> **ARCHIVE, non-normative document.** Internal plan/spec kept for the historical record. The current source of truth is the code plus the canonical docs (ARCHITECTURE.md, ALGORITHMS.md, BACKEND_COVERAGE.md). Do not rely on it as the current status.

# Documentation cleanup plan

PLAN ONLY. This document does not modify any existing file. It lists the overlaps
between the current docs and proposes a clean split of the sources of truth, then a plan to
shorten the README.

---

## 1. Target sources of truth

After cleanup, each document has an exclusive scope:

| Document | Exclusive source of truth |
|---|---|
| `README.md` | Overview (what / why / GIF), links to the other docs, installation, CI status |
| `ARCHITECTURE.md` | Five layers, seams, C++ concepts, lib/application boundary, real status of the components |
| `ALGORITHMS.md` | Generic numerical methods: formulas, code, validation, tests by name |
| `DSL_MODEL_DESIGN.md` (or future Sphinx) | Python user API: facade `dsl.Model`, `CompositeModel`, `add_equation`, status by phase |
| `GPU_RUNTIME_PORT.md` | GH200 validation log: phases, numeric results, fixed bugs, perf caveats |
| `PAPER_ROADMAP.md` | Science: Hoffart target, O5 sweep, ring-edge blocker, baskets 1-4 |
| `COUPLER_HIERARCHY.md` | Exhaustive reference of the couplers in `include/adc/coupling/` |
| `SCHUR_CONDENSATION_DESIGN.md` | Spec of the Schur source stage (design only, PR0-PR8) |
| `AMR_MULTIBLOCK_DESIGN.md` | Spec of the Phase 1 multi-block AMR (design only, PR i-viii) |
| `BACKEND_COVERAGE.md` (to create) | Backend matrix: which path (add_block/aot/production/amr), which GPU/MPI/AMR, which limit |
| `PERFORMANCE.md` | ROMEO profile, bench_amr benchmark, measured numbers, levers (OncePerStep, FFT vs MG) |
| `CHOICES.md` | Justification of the architecture decisions (why, not what) |
| `BIBLIOGRAPHY.md` | Bibliographic references only |

---

## 2. Current state of the overlaps, doc by doc

### 2.1 README.md

The README is 373 lines and contains:

- (a) Overview + GIF (line 1-48): KEEP here.
- (b) Table "What the core provides" (lines 56-82): DUPLICATES ARCHITECTURE.md section 6
  (module map) and section 13 (detailed tree). TO MOVE to ARCHITECTURE.md,
  replaced in the README by a link.
- (c) Section "Symbolic DSL" with the large Python example (lines 87-148): API
  documentation content (DSL_MODEL_DESIGN.md scope). The README can keep 5-6 lines of
  presentation + a minimal example, the rest moves to or points at DSL_MODEL_DESIGN.md.
- (d) Table "Four model paths" (lines 129-141): DUPLICATES the capability matrix of
  DSL_MODEL_DESIGN.md section 5. To move to DSL_MODEL_DESIGN.md or BACKEND_COVERAGE.md,
  keep a reference in the README.
- (e) Section "Current limits" (lines 152-173): honest content, but DUPLICATES
  DSL_MODEL_DESIGN.md section 0bis (GAP/SHIPPED) and GPU_RUNTIME_PORT.md (caveats). To move,
  replaced by a link to both.
- (f) Section "Multi-species systems" (lines 174-196): partially DUPLICATES ARCHITECTURE.md
  section 5 (time/coupling layer, SystemCoupler). To shorten to 4-5 lines + link to
  ARCHITECTURE.md.
- (g) Section "Backends" (lines 198-213): ARCHITECTURE.md section 9 content. To reduce to
  2-3 lines + link.
- (h) Section "Using the core" (lines 215-228): CMake snippet + PhysicalModel contract. To
  keep in brief (it is the hook), but the detailed description (module table) is
  in ARCHITECTURE.md.
- (i) Section "Python module adc" (lines 230-302): very long. The essentials (add_block,
  add_equation, AmrSystem example) stay in the README; the detail of each adder and the
  advanced/legacy paths go in DSL_MODEL_DESIGN.md.
- (j) Section "Ecosystem" (lines 304-312): KEEP here (this is navigation).
- (k) "Build and tests" (lines 314-337): KEEP here (required entry point).
- (l) "Repository organization" (lines 339-354): KEEP, but can be shortened to the folder
  list without the descriptions (which are in ARCHITECTURE.md section 13).
- (m) "Validation (core)" (lines 356-373): DUPLICATES GPU_RUNTIME_PORT.md. To shorten to
  3-4 lines + link.

### 2.2 ARCHITECTURE.md

Already contains the right sections (layers 1-5, seams, elliptic, distributed AMR, backends, module
map, detailed tree). Overlaps to fix:

- Section 1 and section 6 reuse the module table that the README also presents (points
  (b) and (i) above). After shortening the README, these sections STAY in ARCHITECTURE
  and become the single source of truth.
- Section 8 (distributed AMR) mentions the regrid of an intermediate level as a target; this
  OVERLAPS AMR_MULTIBLOCK_DESIGN.md section 5 (regrid algorithm). AMR_MULTIBLOCK_DESIGN is
  the design spec; ARCHITECTURE.md gives its current status. Keep both but add
  a cross-link.
- Section 11 (Validation) lists the tests; partially DUPLICATES GPU_RUNTIME_PORT.md (the
  device validations). To reduce: keep the list of core ctests and the CI numbers; refer
  to GPU_RUNTIME_PORT.md for the GH200 validations.
- Section 13 (detailed tree) is exhaustive and does NOT really DUPLICATE the README
  (which is less detailed). OK as is after shortening the README.

### 2.3 ALGORITHMS.md

Contains the numerical methods with formulas, code, validation. Overlaps:

- Introduction (lines 1-29) reuses the core principle (equation + aux channel) already in
  ARCHITECTURE.md section 2. To shorten to 3-4 lines with a link to ARCHITECTURE.
- Section 18 (runtime composition and multi-species system) OVERLAPS DSL_MODEL_DESIGN.md
  section 0bis and ARCHITECTURE.md section 10 (lib/application boundary). To move the
  "how to compose in Python" aspects to DSL_MODEL_DESIGN.md; keep here only
  the algorithmic point of view (CoupledSystem, SystemCoupler, right-hand side).
- Section 19 (DSL JIT/AOT) DUPLICATES DSL_MODEL_DESIGN.md sections 0 and 7 (sequencing). The
  codegen + test aspects (test_dynamic_model, test_block_builder, test_compiled_model_parity)
  STAY in ALGORITHMS; the explanations about the Python API and the backends go to
  DSL_MODEL_DESIGN.md.
- Section 20 (seam dispatch): ARCHITECTURE.md section 4 content. To reduce to a link.

### 2.4 DSL_MODEL_DESIGN.md

Spec/API document (Python API, status by phase). Overlaps:

- Section 0 (current status) and 0bis (implementation status) DUPLICATE the "Four
  paths" table of the README (point (d) above). After removing the README table, this
  section becomes the single source.
- The capability matrix (section 5) DUPLICATES the "Four paths" table of the README. Single
  source: DSL_MODEL_DESIGN.md.
- The file references (python/bindings/system/base/system.cpp lines) go stale quickly. To mark
  "indicative" and not to sync on every PR.

### 2.5 GPU_RUNTIME_PORT.md

GH200 validation log by phase. Overlaps:

- The header (lines 1-13) reuses the global status already in the README "Validation (core)" section.
  After shortening the README, GPU_RUNTIME_PORT.md is the one that stays the source of truth.
- Phases 1-11 are unique results (numbers, bugs, harness). NO duplicate.
- The "Device validation of the post-#48 features (round 2)" section and the "Suggested
  strategy" section have no duplicate elsewhere.

### 2.6 PAPER_ROADMAP.md

Science document. Overlaps:

- Reuses the status of the GPU/MPI paths in production (section "Status of the execution paths"):
  DUPLICATES GPU_RUNTIME_PORT.md (phases 7, 9, 10) and DSL_MODEL_DESIGN.md (section 0bis, SHIPPED).
  To shorten to 4-5 lines + link.
- The "Recommended public API" section DUPLICATES DSL_MODEL_DESIGN.md (section 0 + settled
  decisions). To replace with a link.
- Baskets 1-4 and the ordered plan: UNIQUE content (science, ring-edge blocker). KEEP here.

### 2.7 COUPLER_HIERARCHY.md

Exhaustive reference of the couplers. Overlaps:

- Section 1 (coupler tree) OVERLAPS ARCHITECTURE.md section 5 (time and
  coupling layer). ARCHITECTURE.md gives the principle, COUPLER_HIERARCHY.md the details. OK
  as is: the two can coexist with a cross-reference.
- Sections 2-11 (per coupler) are unique (responsibilities, signatures, when to use).
  No duplicate.

### 2.8 SCHUR_CONDENSATION_DESIGN.md

Design spec (design-only). Overlaps:

- Introduction mentions the sources read (ARCHITECTURE, ALGORITHMS, PAPER_ROADMAP, etc.):
  OK, this is an intentional anchoring, not a duplicate.
- Section 9 (PR0-PR8 sequencing) OVERLAPS the "Next blocker" of PAPER_ROADMAP.md, but
  each document has its own angle (SCHUR = algorithmic design; PAPER_ROADMAP = science angle).
  Adding a cross-link is enough.

### 2.9 AMR_MULTIBLOCK_DESIGN.md

Phase 1 design spec. Overlaps:

- Section 0 (honesty note) and section 2.2 (exact roles of AmrSystemCoupler) reuse
  ARCHITECTURE.md section 8 (distributed AMR) and COUPLER_HIERARCHY.md section 6
  (AmrSystemCoupler). This is intentional (the spec builds on the existing code). OK with
  cross-references.
- Section 4 (split into PRs) is unique to this document.

### 2.10 PERFORMANCE.md

Perf measurements (CS:APP methodology). Overlaps:

- The header reuses that the measurements come from adc_cases: OK (honesty, not a duplicate).
- The numbers (OncePerStep, FFT vs MG, bench_amr) are UNIQUE here. No duplicate.
- The note on the negative GH200 scaling (final sentence) OVERLAPS GPU_RUNTIME_PORT.md phase 11.
  To refer to GPU_RUNTIME_PORT.md for the GPU numbers.

### 2.11 CHOICES.md

Justifications of the decisions (D-1 to D-7). NO real duplicate: ARCHITECTURE.md describes the status,
CHOICES.md explains the why. Keep both.

### 2.12 BIBLIOGRAPHY.md

References only. No duplicate (the citations appear in the other docs as
anchors, the full definition is here). OK as is.

---

## 3. Matrix "where each statement lives"

| Statement / content | Currently in | Target source of truth |
|---|---|---|
| Generic equation dU/dt + div F = S, D phi = f(U) | README, ARCHITECTURE, ALGORITHMS | ARCHITECTURE section 2 (principle); README 2-3 lines |
| Core module table (PhysicalModel, assemble_rhs, GeometricMG...) | README + ARCHITECTURE | ARCHITECTURE section 6 only |
| Four model paths (a)/(b)/(c)/(d) | README + DSL_MODEL_DESIGN | DSL_MODEL_DESIGN section 5 (capability matrix) only |
| Current limits (AmrSystem mono-block, fft refused with MPI, AmrSystem.potential() IN PROGRESS) | README + DSL_MODEL_DESIGN 0bis | DSL_MODEL_DESIGN 0bis only |
| Python add_block multi-species example | README + DSL_MODEL_DESIGN | README (brief) + DSL_MODEL_DESIGN (detail) |
| GH200 validation status (phases 1-11, numbers) | README "Validation" + GPU_RUNTIME_PORT | GPU_RUNTIME_PORT only |
| CMake backends (ADC_USE_KOKKOS, ADC_USE_MPI...) | README + ARCHITECTURE | README (CMake snippet, 6 lines); ARCHITECTURE section 9 (detail) |
| Seam for_each_cell + device_fence | README + ARCHITECTURE + ALGORITHMS | ARCHITECTURE section 4 + ALGORITHMS section 20 (1 link) |
| Backend capability matrix (GPU/MPI/AMR per path) | DSL_MODEL_DESIGN section 5 | DSL_MODEL_DESIGN section 5 AND BACKEND_COVERAGE.md (to create) |
| Status of the GPU/MPI production paths | README + PAPER_ROADMAP + DSL_MODEL_DESIGN + GPU_RUNTIME_PORT | GPU_RUNTIME_PORT only, with links in the others |
| Recommended public API (add_block vs dsl.Model) | README + PAPER_ROADMAP + DSL_MODEL_DESIGN | DSL_MODEL_DESIGN "settled decisions" only |
| Cartesian ring-edge blocker, O5 sweep | PAPER_ROADMAP | PAPER_ROADMAP only |
| SSPRK2/3, MUSCL, WENO5, multigrid formulas | ALGORITHMS | ALGORITHMS only |
| Justification from-scratch stack vs pde_core_cpp | CHOICES | CHOICES only |
| Perf: 86% elliptic, OncePerStep x2.6, FFT vs MG | PERFORMANCE | PERFORMANCE only |

---

## 4. README shortening plan

### Principle

The target README is a navigation PORTAL, not technical documentation. Target: 120-150
lines (vs 373 today), i.e. a reduction of 60-65%.

### What STAYS in the README (overview + portal)

1. Header (1-23): title, CI badge, GIF, caption. KEEP.
2. Introduction paragraph (lines 25-51): what adc_cpp is, generic equation, aux channel,
   model-agnostic. SHORTENED to 10-12 lines (keep the core, remove the repetitions
   of the extensible channel).
3. Table "What the core provides": REPLACED by 4-5 lines + link to ARCHITECTURE.md
   section 6.
4. DSL section: KEEP 8-10 lines (concept + minimal example of 6 lines) + link to
   DSL_MODEL_DESIGN.md for the detail and the advanced paths.
5. Multi-species section: SHORTENED to 5-6 lines + link to ARCHITECTURE.md section 5.
6. Backends section: SHORTENED to CMake snippet (4 lines) + link to ARCHITECTURE.md section 9.
7. "Using the core": FetchContent snippet (6 lines). KEEP.
8. "Python module adc": SHORTENED to the minimal example add_block/set_poisson/step_cfl (12 lines)
   + brief description of AmrSystem + link to DSL_MODEL_DESIGN.md and ARCHITECTURE.md.
9. "Ecosystem": KEEP (navigation, 6-line table).
10. "Build and tests": KEEP (snippet + CMake options table, ~15 lines).
11. "Repository organization": SHORTENED to the folder list (4-5 lines), without the
    descriptions (which are in ARCHITECTURE.md section 13).
12. "Validation": SHORTENED to 4-5 lines + link to GPU_RUNTIME_PORT.md.

### What LEAVES the README

- Detailed 20-line table "What the core provides" -> ARCHITECTURE.md section 6.
- Table "Four model paths" (10 lines) -> DSL_MODEL_DESIGN.md section 5.
- Section "Current limits" (22 lines) -> DSL_MODEL_DESIGN.md section 0bis.
- Multi-species table (12 lines) -> ARCHITECTURE.md section 5 + link.
- Adder detail subsection (add_dynamic_block, add_compiled_block, etc.) -> DSL_MODEL_DESIGN.
- Detailed device validation (18 lines) -> GPU_RUNTIME_PORT.md.

### Target README structure (section order, indicative lengths)

```
1. Header + GIF + legende         (15 lignes)
2. Presentation du coeur          (12 lignes, lien ARCHITECTURE)
3. Ce que fournit le coeur        (5 lignes + lien ARCHITECTURE section 6)
4. DSL symbolique (exemple bref)  (15 lignes, lien DSL_MODEL_DESIGN)
5. Systemes multi-especes         (6 lignes + lien ARCHITECTURE)
6. Backends CMake                 (6 lignes + lien ARCHITECTURE)
7. Utiliser le coeur (FetchContent) (8 lignes)
8. Module Python adc (exemple)    (15 lignes, lien DSL_MODEL_DESIGN)
9. Ecosysteme                     (10 lignes)
10. Build et tests                (15 lignes)
11. Organisation du depot         (8 lignes, lien ARCHITECTURE section 13)
12. Validation                    (5 lignes, lien GPU_RUNTIME_PORT)
---
Total : ~120 lignes
```

---

## 5. Document to create: BACKEND_COVERAGE.md

This document does not exist yet. It materializes the capability matrix in a canonical way,
today scattered between the README (table "Four paths") and DSL_MODEL_DESIGN.md
(section 5, matrix `_BACKEND_CAPS`).

Proposed content:

```
| Chemin       | Adder Python         | CPU serie | MPI | AMR | GPU | zero-copie | Limites |
|---|---|---|---|---|---|---|---|
| (a) natif    | add_block            | oui       | oui | oui | oui | oui        | aucune  |
| (d) production | add_native_block   | oui       | oui | oui | oui | oui        | AmrSystem: mono-bloc, explicite |
| (c) aot      | add_compiled_block   | oui       | non | non | non | non        | mono-rang, SSPRK2 uniquement |
| (b) prototype | add_dynamic_block   | oui       | non | non | non | non        | Rusanov ordre 1, hote |
```

Each cell anchors to the source (DSL_MODEL_DESIGN, GPU_RUNTIME_PORT) for the detail.

---

## 6. Priority order of the interventions

Ranking by value / effort:

1. (HIGH VALUE, LOW EFFORT) Shorten README: remove the 4 sections identified
   above, add links. Goal 120-150 lines. No content lost.
2. (HIGH VALUE, LOW EFFORT) Create BACKEND_COVERAGE.md: materialize the matrix
   (content already present in DSL_MODEL_DESIGN section 5).
3. (MEDIUM VALUE, LOW EFFORT) Clean up the introductions of ALGORITHMS.md and GPU_RUNTIME_PORT.md
   (remove the entry duplications with ARCHITECTURE and README).
4. (MEDIUM VALUE, LOW EFFORT) Add cross-links between SCHUR_CONDENSATION_DESIGN,
   AMR_MULTIBLOCK_DESIGN and PAPER_ROADMAP (sections that refer to each other).
5. (LOW VALUE, HIGH EFFORT) Migration to Sphinx/ReadTheDocs for DSL_MODEL_DESIGN.md:
   useful only if the API user audience grows (out of internship scope).

---

## 7. Future maintenance rules

To prevent the duplicates from reappearing:

- Any new GH200 validation -> GPU_RUNTIME_PORT.md only. README points.
- Any Python API change -> DSL_MODEL_DESIGN.md section 0bis (SHIPPED/GAP). README does not
  list the limits.
- Any new numerical method -> ALGORITHMS.md. No formula in the README.
- Any coupler schema change -> COUPLER_HIERARCHY.md. ARCHITECTURE.md describes
  the principle, not the signatures.
- The "Four paths" table (capability matrix) lives in BACKEND_COVERAGE.md. The two
  other docs that referenced it point to it.
