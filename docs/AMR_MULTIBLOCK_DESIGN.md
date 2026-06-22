> **STATUS: IMPLEMENTED.** The AMR multi-block is delivered (`AmrSystem.multi_block()`/`build_multi()`, AmrCouplerMP, regrid union-tags #195/#199/#205, 7 capstone tests). This document describes a past state ('mono-block facade'); read it as design history.

# Design: AMR multi-block on a shared hierarchy (multi-species capstone)

DESIGN-ONLY. No implementation. This document is a reasoned SPEC, honest about its
limits, of the migration of the `adc` AMR from ONE single explicit block toward SEVERAL blocks
(species) on the SAME shared AMR hierarchy, conservatively coupled, modeled on the
way the mono-level mesh already carries several blocks. It is the capstone: electrons,
ions and neutrals on the SAME levels/patches, a single coarse Poisson whose right-hand side
is the SUM of the co-located elliptic contributions, coupled sources read cell by
cell, a regrid driven by the UNION of the criteria.

The target model is that of AMReX / FLASH / SAMRAI: ONE common hierarchy carrying several
fields, refinement by the union of the criteria, never one hierarchy per species.


## 0. Preliminary note of honesty (real state of the code at this head)

The code has been read directly. Two facts structure everything that follows.

FACT 1: the multi-block AMR ENGINE ALREADY EXISTS, at the C++ template level, under the name
`AmrSystemCoupler` (`include/adc/coupling/system/amr_system_coupler.hpp`). It carries:
- a hierarchy SHARED per block (`std::vector<std::vector<AmrLevelMP>> block_levels_`).
  The ctor checks at assembly the layout consistency between blocks. CAUTION (owner correction):
  the old check compared ONLY the NUMBER of levels and the NUMBER of boxes per level
  (`.size()`), NOT the exact mesh. It therefore did NOT guarantee the same `BoxArray`. The first
  step of this capstone (cf. 2.2, point 1) introduces `detail::same_layout_or_throw` which compares
  EXACTLY, and which REPLACES this size check at the ctor: number of levels, then per level
  `BoxArray` (the boxes AND their order), `DistributionMapping` (rank per box) and `dx`/`dy`;
- an aux SHARED per level (`std::vector<MultiFab> aux_`, one per level, rewired to each
  block, lines 136-141);
- a SYSTEM Poisson with a SUMMED right-hand side (`rhs_assembler_(system_, mg_.rhs())` in
  `solve_fields`, line 183; the assembler is `ChargeDensityRhs`, cf. below);
- a spatial scheme, substeps and a cadence PER BLOCK (`block_time_treatment_v`,
  `block_substeps_v`, `block_stride_v`, `step`, lines 211-250);
- the IMEX / implicit by callback (`AmrImplicitSourceStepper`, lines 361-369);
- coupled sources LEVEL BY LEVEL (`coupled_source_step`, lines 266-283);
- a B_z shared per level (`fill_bz`, lines 329-340).

What the owner calls "extract a multi-block engine" is therefore in large part a work
of RECOGNITION and COMPLETION, not of creation ex nihilo. It is good news to say
frankly: the Phase 1 target is closer than it seems.

FACT 2: the RUNTIME FACADE (the one exposed to Python) remains MONO-BLOCK. `AmrSystem`
(`include/adc/runtime/amr_system.hpp`, impl `python/bindings/amr/amr_system.cpp`) explicitly REFUSES a
2nd block:
```
// python/bindings/amr/amr_system.cpp:129-130
if (p_->has_block || p_->has_compiled)
  throw std::runtime_error("AmrSystem : un seul bloc (AMR mono-modele)");
// idem set_compiled_block, ligne 152-153
```
It wraps a single `AmrCouplerMP<Model>` (`include/adc/coupling/amr/amr_coupler_mp.hpp`),
materialized by `detail::dispatch_amr_compiled` / `build_amr_compiled`
(`include/adc/runtime/builders/compiled/amr_dsl_block.hpp`). The multi-block `AmrSystemCoupler` is NOT wired
to this facade.

The TWO real gaps for the Phase 1 target are therefore:
1. the multi-block RUNTIME FACADE and its Python binding. CAUTION (owner correction): in Python,
   REMOVING the 2nd-block throw (`amr_system.cpp:129-130 / 152-153`) is NOT ENOUGH. It is the BIG
   piece, the real work of this step (and it is the NEXT PR, NOT the first layout step
   described in 2.2.1). It needs a TYPE-ERASED REGISTRY BY NAME that holds, PER BLOCK: its stack of
   levels (`std::vector<AmrLevelMP>` on the shared layout) AND its type-erased closures
   (`advance` / elliptic `rhs` / `source` / `max_speed` / `mass` / `density`), exactly like
   `System::Impl::sp` (the `Species` struct, `python/bindings/system/base/system.cpp:270-301`) holds the closures per
   species for the mono-level runtime path. It is the gap already crossed between the compile-time
   `SystemCoupler` and the runtime facade `System`: type-erased registry of blocks, repeated
   `add_block`, multirate `stride` / `evolve` / `substeps`, sum of the Poisson;
2. the multi-block REGRID: `AmrSystemCoupler` has NO `regrid` method (verified: grep
   `regrid` in `amr_system_coupler.hpp` -> nothing), unlike `AmrCouplerMP::regrid`
   (lines 321-325) which delegates to `amr_regrid_finest`. Its hierarchy is FROZEN at
   construction. The UNION-of-tags regrid which rebuilds the mesh ONCE and
   prolongs/restricts ALL the blocks is implemented (capstone Phase 2; `build_multi` ->
   set_regrid + set_block_tag_predicate in `python/bindings/amr/amr_system.cpp`).

HARD CONSTRAINT (owner correction, point 4). AS LONG AS the UNION-of-tags regrid (cf. 5) is
not implemented, the MULTI-BLOCK combination (`n_blocks >= 2`) AND `regrid_every > 0` MUST be
EXPLICITLY REFUSED (clear throw) by the runtime facade. The hierarchy of `AmrSystemCoupler`
is FROZEN at construction: silently allowing `regrid_every > 0` in multi-block would make
the API CLAIM it does dynamic AMR while the mesh never moves (dangerous illusion).
The mono-block keeps its regrid (`AmrCouplerMP` path untouched). This refusal is
implemented in the RUNTIME REGISTRY PR (the next one, point 3), NOT in the first layout
step (2.2.1).

All the rest of this document relies on these real names.


## 1. Architecture decision and why

DECISION. A single AMR hierarchy SHARED by all the species. Each species is an
`AmrBlock`: a stack of levels `std::vector<AmrLevelMP>` placed on the SAME `BoxArray` per
level and the SAME `DistributionMapping`, plus its model, its spatial scheme, its temporal
policy. ALL the blocks live on ALL the patches: a block is NEVER absent from a
patch, even if its criterion did not trigger it, because it is coupled to the others. They all
share the SAME aux per level (phi, grad phi, [B_z, T_e]) and the SAME coarse Poisson; the
elliptic right-hand side is the SUM of the contributions per block, assembled from
CO-LOCATED fields (same cells of the same hierarchy). The refinement is the UNION of the
criteria per block (plus the user tags and the phi tags).

WHY a shared hierarchy, conservatively coupled, and NOT one per species:

- POISSON. The electrostatic coupling is GLOBAL: `lap phi = Sum_s q_s n_s`. On a
  shared hierarchy, `solve_fields` reads all the densities at the SAME cell index and
  writes a single phi. This is exactly what `AmrSystemCoupler::solve_fields` already does
  (lines 177-206): `rhs_assembler_(system_, mg_.rhs())` reads all the blocks on the coarse,
  then a single coarse->fine injection of the aux. Per-species meshes would require
  interpolating each density to a common mesh AT EACH solve, and redistributing phi
  to each mesh: an assembly from NON-CONFORMING meshes, which the target
  proscribes.

- COUPLED SOURCES. Ionization / collision / thermal exchange read the states of SEVERAL
  species in the SAME cell (cf. the target formula `d_t n_e = +k n_e n_g`,
  `d_t n_i = +k n_e n_g`, `d_t n_g = -k n_e n_g`). On a shared hierarchy, the read is
  LOCAL (same `(i,j)`, same fab), NO inter-species interpolation. It is the contract of
  `AmrSystemCoupler::coupled_source_step` (lines 266-283): at each level k, each block is
  temporarily repointed to its level k and the source reads all the blocks + `aux_[k]`. Disjoint
  hierarchies would make this splitting impossible without conservative projection.
  CAUTION (owner correction, point 5): applying the source LEVEL BY LEVEL OVER-COUNTS a
  NON-zero source in the coarse cells COVERED by a fine level (applied to the
  coarse AND to the fine that will be averaged down). The conservation of the COMPOSITE mass requires either
  a LEAF-ONLY application (uncovered cells only), or an `average_down` / sync
  AFTER the source. To test explicitly (cf. PR (vi) and checklist).

- CONSERVATION AND REFLUX. The reflux is BLOCK BY BLOCK (each block has its own flux
  registers, `FluxRegister` in `amr_reflux_mf.hpp`). The source stays CELL-LOCAL, applied
  after the transport (`mf_apply_source_treatment`, lines 75-82), NEVER in the reflux
  registers: this is exactly the property of the Gap2 IMEX-on-AMR already merged. A source with
  exactly opposite contributions in the SAME cell conserves the pair mass to machine
  precision, which only holds if the species share the cell.

- MPI / KOKKOS. A single hierarchy = a single distribution plan
  (`DistributionMapping(ba.size(), n_ranks())`), a single set of halos, a single reflux
  reduction. Disjoint hierarchies would multiply the plans and the collectives.

WHAT WE DO NOT OPEN (and why to say it). No per-species hierarchy. No different levels
per species (invariant verified at the ctor of `AmrSystemCoupler` by
`detail::same_layout_or_throw`: same number of levels, and per level same `BoxArray` [boxes
AND order], same `DistributionMapping`, same `dx`/`dy`). No partial allocation "patch active
for a single species". These restrictions are deliberate: they preserve the unique aux and the
unique Poisson, the raison d'etre of sharing, and the cell-by-cell conservation of the sources.


## 2. The multi-block engine: concrete API (what exists, what is named)

The multi-block engine is built around the following CONCRETE types. When they already exist,
they are cited by their real file name; when they are to be extracted, the code whose role
they promote is indicated.

### 2.1 The compile-time block layer (exists)

- `EquationBlock<Model, Spatial, Time>` (`include/adc/core/model/equation_block.hpp`): carries
  `Model`, `Spatial` (limiter + flux), `Time` (policy), `MultiFab* state`, `BCRec bc`.
- `CoupledSystem<Blocks...>` (`include/adc/core/model/coupled_system.hpp`): tuple of blocks with
  `n_blocks`, `block<I>()`, `for_each_block(f)`. It is the compile-time "registry".
- `AmrLevelMP { MultiFab U; const MultiFab* aux; Real dx, dy; }`
  (`amr_reflux_mf.hpp`, lines 791-795): a level of the multi-patch hierarchy.
- `LevelHierarchy { std::vector<AmrLevelMP> levels; Box2D base_dom; Periodicity base_per;
  bool coarse_replicated; bool recon_prim; bool imex; }` (`amr_reflux_mf.hpp`, lines
  1012-1019): the hierarchy as a named OBJECT. It is the grain of the future `AmrHierarchyLayout`
  (cf. 2.2).
- `AmrLevelStack<Level>` (`include/adc/coupling/amr/amr_level_storage.hpp`): holds the stack of
  levels AND the parallel aux stack, with the ADDRESS INVARIANT (aux dimensioned once,
  `reattach_aux(k)` replaces in place). It is the storage brick that makes the rewiring
  `levels[k].aux = &aux_[k]` safe.

### 2.2 `AmrRuntime`: the multi-block engine (to extract, partially present)

The target engine groups seven roles. Five are ALREADY carried by `AmrSystemCoupler`; two are to be
added. C++ signatures that fit the existing types are given.

- `AmrHierarchyLayout` (DELIVERED, first step): per-level `BoxArray` + `DistributionMapping` +
  `dx`/`dy` + number of levels (`nlev() == ba.size()`). This mesh was until now implicit:
  each `AmrLevelMP` carries its `U.box_array()` / `U.dmap()` / `dx,dy`, and we relied on the
  size check of the ctor. The EXPLICIT type is now the single source of truth on the
  shared mesh, and it is exactly what the `same_layout_or_throw` safeguard compares. MINIMAL
  SHAPE ACTUALLY DELIVERED (point 2, first step):
  ```cpp
  struct AmrHierarchyLayout {
    std::vector<BoxArray>            ba;     // [niveau] : boites ET ordre
    std::vector<DistributionMapping> dm;     // [niveau], parallele a ba : rang par boite
    std::vector<Real>                dx, dy; // [niveau] = dx_coarse / 2^k
    int nlev() const { return static_cast<int>(ba.size()); }
    static AmrHierarchyLayout from_levels(const std::vector<AmrLevelMP>&);
  };
  ```
  MINIMAL FIRST STEP (owner correction, point 2). We extract ONLY what the safeguard needs
  (per-level `BoxArray` + `DistributionMapping` + `dx`/`dy` + number of levels) plus a
  layout validation and bit-identical tests. We do NOT introduce the big abstraction
  `AmrBlock` below if it DUPLICATES `EquationBlock` (which already carries `Model`/`Spatial`/`Time`/
  `state`/`bc`): `base_dom`, `base_per`, `coarse_replicated`, `ref_ratio` stay elsewhere
  (coupler fields, `SubcyclingSchedule(2)` hard-wired) as long as they are not required by the
  safeguard. Any wider extraction is a LATER step, and ONLY if necessary.

- `AmrBlock` (NOT delivered at the first step; to evaluate later, and ONLY if necessary): the
  stack of levels of ONE block on the shared layout, plus its numerical identity. CAUTION
  (owner correction, point 2): this type WOULD DUPLICATE `EquationBlock<Model,Spatial,Time>` which
  ALREADY carries `Model`/`Spatial`/`Time`/`state`/`bc`. We therefore do NOT introduce it at the first step. The
  coupling `(block_levels_[b], system_.block<b>())` stays as is: `EquationBlock` is REUSED
  without re-wrap. If a wider extraction proves necessary later, it will reuse
  `EquationBlock` rather than re-duplicate it. INDICATIVE form (not delivered):
  ```cpp
  template <class Model, class Spatial, class Time>
  struct AmrBlock {
    std::string_view name;
    Model model; BCRec bc;
    std::vector<AmrLevelMP> levels;   // U par niveau sur AmrHierarchyLayout
    VariableSet cons_vars, prim_vars; // noms + ROLES (cf. variables.hpp)
    bool recon_prim = false;
    // Spatial = {Limiter, NumericalFlux} ; Time = politique (treatment/substeps/stride/evolve)
  };
  ```
  Invariant: `levels[k].U.box_array() == layout.ba[k]` for all k (now verified EXACTLY
  at the ctor by `same_layout_or_throw`).

- `AmrBlockRegistry`: the collection of blocks. At compile-time it is `CoupledSystem<Blocks...>`
  (already there). At runtime it will be a type-erased `std::vector<AmrSpecies>` modeled on
  `System::Impl::sp` (`python/bindings/system/base/system.cpp:270-301`, the `Species` struct).

- `AmrFieldSolver`: the SYSTEM Poisson with a SUMMED right-hand side, co-located. It is
  `solve_fields` of `AmrSystemCoupler` (lines 177-206) wired on `ChargeDensityRhs`
  (`include/adc/coupling/base/elliptic_rhs.hpp`, lines 117-133), which REQUIRES one `SpeciesCharge` per
  block (`species.size() != System::n_blocks` -> error), sums `q_s n_s` via
  `add_scaled_component`, then injects the aux coarse->fine (`coupler_inject_aux_mb`) a single
  time. Co-located by construction: `rhs` is on `block_levels_[0][k].U.box_array()`, the
  shared mesh.

- `AmrScheduler`: honors `treatment` / `substeps` / `stride` / `evolve` per block. It is
  `AmrSystemCoupler::step` (lines 211-250) plus the stride semantics of
  `advance_subcycled` (`include/adc/numerics/time/schemes/scheduler.hpp`) and of runtime `System`
  (`stride_due`, `python/bindings/system/base/system.cpp:327`). The target contract (cf. 4.iv):
  - `Explicit` -> AMR transport by `advance_amr<Limiter,NumericalFlux>`;
  - `IMEX` -> explicit transport (`SourceFreeModel<Model>`) + implicit source by the
    callback (reuses `mf_apply_source_treatment` / `backward_euler_source` of the Gap2);
  - `Implicit` -> REJECT as long as a true global stepper does not exist;
  - `evolve=false` -> block FROZEN (not advanced, does not constrain the CFL) but ALWAYS read by the
    right-hand side of the Poisson as a fixed background;
  - `substeps=N` -> N substeps in the step of the block; `stride=M` -> held M-1 macro-steps then
    catch-up of an effective step M*dt.

- `AmrCouplingRegistry`: the inter-species coupled sources applied level by level,
  cell by cell, exactly opposite contributions. It is `coupled_source_step`
  (lines 266-283) wired on a `CoupledSource` (concept `CoupledSourceFor`,
  `include/adc/coupling/source/coupled_source.hpp`). The production device kernel is
  `CoupledSourceKernel` (`include/adc/coupling/source/coupled_source_program.hpp`, P5 #131): POD
  captured by value, ADDITIVE writes `out[t](i,j,c) += dt*S_t`; two opposite terms
  (`+k n_e n_g` on ion, `-k n_e n_g` on neutral) conserve the pair mass exactly there.

- `AmrRegridPolicy`: the union of the tags + rebuild + prolong/restrict of ALL the blocks. It is the
  MISSING PIECE (cf. 5). The mono-block brick exists: `amr_regrid_finest`
  (`include/adc/coupling/amr/amr_regrid_coupler.hpp`); a multi-block orchestrator is needed that
  tags the UNION then rebuilds the mesh ONCE for all.

### 2.2.1 Layout safeguard `same_layout_or_throw` (DELIVERED, MINIMAL first step)

Point 1. The aux being SHARED per level, all the blocks must live on EXACTLY the same
mesh at each level, otherwise the rewiring `levels[k].aux = &aux_[k]` and the advance read an
inconsistent mesh (silent out-of-bounds access). The ctor check until now compared only
the NUMBER of boxes (`.size()`): insufficient (two blocks with 2 boxes of different SHAPES
or ORDERS passed). The safeguard now compares EXACTLY, and REPLACES this size
check at the point where the layout consistency is asserted (ctor):

```cpp
namespace detail {
// compare DEUX niveaux : BoxArray (boites ET ordre), DistributionMapping, dx, dy
bool same_level_layout(const BoxArray&, const DistributionMapping&, Real dx, Real dy,
                       const BoxArray&, const DistributionMapping&, Real dx, Real dy);
// jette une std::runtime_error claire au PREMIER ecart entre le bloc 0 (reference) et tout
// autre bloc : nombre de niveaux, puis par niveau BoxArray (boites ET ordre),
// DistributionMapping (rang par boite) et dx/dy.
void same_layout_or_throw(const std::vector<std::vector<AmrLevelMP>>& block_levels);
}
```

What `same_layout_or_throw` compares EXACTLY: (i) the number of levels; and per level k:
(ii) the complete `BoxArray` (the boxes AND their order, via `ba.boxes() ==`), (iii) the
`DistributionMapping` (the rank vector `dm.ranks() ==`), (iv) `dx` and (v) `dy` (to the bit).
Equality via `Box2D::operator==` (default) and `std::vector::operator==`. MONO-BLOCK strictly
bit-identical: a single block trivially matches itself (the loop over the other blocks
is empty), so the mono-block path is not touched (verified by a `dmax == 0` test).

PERIMETER OF THIS FIRST STEP (owner correction, point 2). It is MINIMAL and limited to: the type
explicit `AmrHierarchyLayout`, the layout validation (`same_layout_or_throw` substituted for the
size check), and bit-identical tests. It introduces NO big abstraction
`AmrBlock` that would duplicate `EquationBlock`. The runtime registry (cf. 3) is a SEPARATE and
LATER step, NOT this one.

### 2.3 Why an engine and not yet another coupler

`AmrSystemCoupler` already mixes "assemble" (Poisson + aux) and "advance" (step + reflux), as
its own comment notes (lines 371-375: alias `AmrSystemDriver`). The Assembler/Driver
split is done on the mono-level side (`SystemAssembler` / `SystemDriver`,
`include/adc/coupling/system/system_coupler.hpp`). The `AmrRuntime` engine formalizes the same separation
on the AMR side, but it is a COSMETIC refinement and deferred: the unified class is already validated.
The priority remains the RUNTIME FACADE and the REGRID, not the pretty split.


## 3. COMPAT-FACADE strategy: preserve the BIT-IDENTICAL mono-block

The objective is that the current mono-block `AmrSystem` and its tests stay BIT-IDENTICAL. The
rule is to NOT tinker with the mono-block class in place, but to transform it into a FACADE.

PLAN. `AmrSystem` (runtime) becomes a facade above a runtime multi-block engine
`AmrRuntime`; the mono-block path goes through an `AmrSingleBlockSystem` which installs only ONE
block. Concretely:

1. The current `AmrSystem::Impl` builds a unique `AmrCouplerMP<Model>` via
   `build_amr_compiled`. The mono-block path MUST stay this path as long as the registry
   contains only one block: `AmrCouplerMP<Model>` and `AmrSystemCoupler<CoupledSystem<Block>, ...>`
   are NOT the same code. To guarantee the bit-identical, the facade routes:
   - `n_blocks == 1` (and no inter-species coupling, no frozen block) -> CURRENT
     `AmrCouplerMP<Model>` path, UNCHANGED -> mono-block tests bit-identical by construction
     (no line of the path is touched);
   - `n_blocks >= 2` OR coupling requested -> new `AmrSystemCoupler` path.
2. This internal routing does NOT change the public signature of `AmrSystem`. CAUTION (correction
   owner): removing the 2nd-block throw is only a SMALL part. The real work of this PR
   is the TYPE-ERASED REGISTRY BY NAME: `add_block` must REGISTER, per block, its stack of
   levels PLUS its type-erased closures (`advance` / elliptic `rhs` / `source` / `max_speed`
   / `mass` / `density`), an exact copy of the `Species` struct of `System::Impl::sp`
   (`python/bindings/system/base/system.cpp:270-301`). Without this registry, two blocks can be neither advanced, nor
   summed in the Poisson, nor coupled: the throw is only the symptom. The 1st `add_block` alone,
   followed by `step` / `step_cfl` / `mass` / `density` / `potential`, must produce EXACTLY the
   same bytes as today (mono-block `AmrCouplerMP` path untouched).
3. The bit-identity criterion is verified by a NON-REGRESSION test: a mono-block case
   (one of the `test_amr_*` or `test_dsl_production_amr.py`) must give `maxdiff == 0` between the
   refactored facade and the current binary. As long as this test is not green, PR (i) does not
   merge.

WHY not to route the mono-block through `AmrSystemCoupler` right away. Because
`AmrSystemCoupler` and `AmrCouplerMP` differ on details that BREAK the bit-identical:
- `AmrCouplerMP::compute_aux` writes grad phi inline (lines 283-293) and calls
  `coupler_eval_rhs` (`f = model.elliptic_rhs(U)`), whereas `AmrSystemCoupler::solve_fields`
  goes through `field_postprocess` + `ChargeDensityRhs` (sum). The two are mathematically the
  same computation for one charge species `q=1`, but the order of the floating-point operations differs.
- `AmrCouplerMP` carries the periodic regrid (via the closure `h.step`,
  `amr_dsl_block.hpp:99-104`); `AmrSystemCoupler` has none.

So: the mono-block stays on `AmrCouplerMP` (untouched), and `AmrSystemCoupler` receives the
regrid + the facade routing. When `AmrSystemCoupler` has proven a mono-block strictly
bit-identical to `AmrCouplerMP` (maxdiff=0 test), the two paths CAN be merged; until
then, they coexist. It is exactly the prudence of the `replicated_coarse` comment
of `AmrCouplerMP` (lines 234-246): the removal of one path is deferred as long as the other
is not strictly superior.


## 4. Phase 1 migration: breakdown into ordered PRs

Each PR lists its WRITE-SET (files), its acceptance test, and its bit-identity /
conservation blocker. The order is strict: each PR leaves the tree green.

### PR (i) -- Introduce `AmrBlock` + registry, NO change of physics
WRITE-SET:
- `include/adc/coupling/system/amr_system_coupler.hpp`: extract `AmrHierarchyLayout` (promotion of the
  `BoxArray`/`DistributionMapping`/`dx` already imposed identical), have each block carry an
  `AmrBlock` (name + levels + cons/prim VariableSet). No new behavior.
- `include/adc/coupling/amr/amr_level_storage.hpp`: reused as is (address invariant).
ACCEPTANCE: a 1 explicit block case (`test_amr_compiled_model.cpp` style) runs identical.
BIT-IDENTITY: the `step` does not change body -> `maxdiff == 0` vs current head on this case.

### PR (ii) -- Two explicit blocks, DIFFERENT schemes, without coupled source
WRITE-SET:
- `include/adc/coupling/system/amr_system_coupler.hpp`: already N-blocks; add a test instantiating
  `CoupledSystem<BlockA, BlockB>` where `BlockA::Spatial != BlockB::Spatial` (e.g. Minmod/Rusanov
  vs VanLeer/HLLC) on the SAME 2-level hierarchy.
- `tests/CMakeLists.txt` + `tests/test_amr_system_twoblock.cpp` (new test).
ACCEPTANCE: two AMR blocks with different schemes, stable over N steps; mass of each block
conserved at the reflux (`AmrSystemCoupler::mass(b)`, lines 287-297).
CONSERVATION: `mass(0)` and `mass(1)` constant to the reflux tolerance (reflux block by block,
each block has its `FluxRegister`).

### PR (iii) -- SYSTEM Poisson with a SUMMED right-hand side (co-located)
WRITE-SET:
- `include/adc/coupling/base/elliptic_rhs.hpp`: `ChargeDensityRhs` (already there); verify the guard
  `species.size() == n_blocks`.
- test: two species of opposite charges, `lap phi = q_i n_i + q_e n_e`, compared to a Poisson
  mono-level assembled by hand.
ACCEPTANCE: `solve_fields` reads the two co-located blocks and solves ONE phi. The sum of the RHS
equals (to machine precision) the sum assembled separately -> co-location proven.
CONSERVATION: the total charge integrated on the coarse stays the expected sum.

### PR (iv) -- substeps / stride / evolve + step_cfl substeps-aware
WRITE-SET:
- `include/adc/coupling/system/amr_system_coupler.hpp`: add `step_cfl(cfl)` modeled on
  `System::step_cfl` (`python/bindings/system/base/system.cpp:1663-1693`), substeps-aware:
  `dt <= cfl * h * substeps_b / (stride_b * w_b)`, min over the evolving blocks; a block
  `evolve=false` does NOT constrain the step but stays in the Poisson RHS.
- the stride semantics MUST be that of `System`: HOLD-THEN-CATCH-UP, that is to say
  `stride_due = (macro_step_ + 1) % stride == 0` (`python/bindings/system/base/system.cpp:320-327`), NOT that of
  `advance_subcycled` (`macro%M==0`, start of window). OWNER CORRECTION (point 6): the condition
  `macro_step_ % stride` written higher up in an earlier version of this plan was WRONG; the
  correct cadence is indeed `(macro_step_ + 1) % stride == 0` (catch-up at the END of window, not
  at the start). `AmrSystemCoupler::step` still uses today the form `macro_step_ % stride`
  (otherwise a slow block would be "in the future" at the 1st step, wrong coupling): its HARMONIZATION on
  `stride_due` is done by the CONCURRENT STRIDE PR (which owns this condition), NOT by the
  first layout step (2.2.1), which does NOT touch the cadence to avoid any conflict.
ACCEPTANCE: electrons IMEX(substeps=10) + ions Explicit(substeps=1) stable, AND the inverse;
neutrals stride=20 always read by the source and the Poisson between two catch-ups;
`evolve=False` present in the elliptic RHS as a fixed background.
BLOCKER: `step_cfl` must respect BOTH stride and substeps; a dt computed on `w_max`
alone then multiplied by M would violate the CFL by a factor M (explicit note of `System::step_cfl`).

### PR (v) -- Multi-block production DSL (INSTALL of a NAMED block)
WRITE-SET:
- `include/adc/runtime/builders/compiled/amr_dsl_block.hpp`: `add_compiled_model(AmrSystem&, name, Model{}, ...)`
  must INSTALL A NAMED BLOCK (and not replace the unique block) -> symmetric of
  `add_compiled_model(System&)` (`include/adc/runtime/builders/compiled/dsl_block.hpp` + `block_builder.hpp`).
- `include/adc/runtime/amr_system.hpp` + `python/bindings/amr/amr_system.cpp`: `set_compiled_block` /
  `add_native_block` stop throwing at the 2nd call (cf. 3) and stack a spec.
- `python/bindings/core/bindings.cpp`: expose the 2nd `add_block` (already wired, `bindings.cpp:239`), validate
  `dsl.Model(...).compile(target="amr_system", backend="production")` for TWO blocks.
ACCEPTANCE: `test_dsl_production_amr.py` extended to two native compiled blocks on the same
hierarchy; ABI key verified (`adc_native_abi_key`, `amr_system.cpp:218-235`).
BLOCKER: NAMED device-clean functors (cf. `BlockRhsEval`/`AdvanceExplicit` of
`block_builder.hpp` and `ForEachBlockProbe` of `coupled_system.hpp:59-63`); no extended
lambda cross-TU (harness #64/#97).

### PR (vi) -- Coupled sources on AMR (same cell, opposite contributions)
WRITE-SET:
- `include/adc/coupling/system/amr_system_coupler.hpp`: `coupled_source_step` (already there) wired on the
  named couplings (ionization/collision/exchange) AND on `CoupledSourceKernel`
  (`coupled_source_program.hpp`).
- `python/bindings/amr/amr_system.cpp` + `bindings.cpp`: `sim.add_coupling(adc.Ionization(...))` modeled on
  `System::add_ionization` / `add_coupled_source` (`include/adc/runtime/system.hpp:266-329`).
ACCEPTANCE: ionization `+k n_e n_g` / `-k n_e n_g` on 3 co-located blocks, level by
level, on all the patches.
DOUBLE-COUNTING PITFALL (owner correction, point 5). `coupled_source_step` applies the source
LEVEL BY LEVEL: a coarse cell COVERED by a fine level receives the source TWICE
(once on the coarse, once via the fine cell that will then be averaged down into
this same coarse volume). For a NON-zero source (creation/destruction of mass), this
OVER-COUNTS the term in the covered cells and breaks the conservation. Valid CORRECTIONS:
(a) LEAF-ONLY application (apply the source only to cells NOT covered by a finer
level, via the `CoverageMask`), OR (b) `average_down` / sync AFTER the source so that the fine value
overwrites the over-counted coarse value in the covered cells. The pair invariant
(`n_i + n_g`) stays exact PER CELL (opposite terms) even in case of double counting; what
breaks is the integrated TOTAL MASS. Per-cell conservation is therefore NOT sufficient as a test.
ACCEPTANCE (REQUIRED test, point 5): a COMPOSITE AMR CONSERVATION test on a NON-zero source,
where the integrated TOTAL mass on the composite hierarchy (uncovered coarse + covering fine)
is conserved to the tolerance; a 2-level case where the source zone OVERLAPS
the coarse-fine interface, to exercise specifically the double counting.
CONSERVATION: pair-creation invariant and heavy mass to MACHINE PRECISION
(`n_i + n_g` constant) PER CELL; AND COMPOSITE mass (integral on the hierarchy, leaf-only
or after sync) conserved to the tolerance, to rule out the double counting of the covered cells.

### PR (vii) -- Local IMEX on AMR
WRITE-SET:
- `include/adc/coupling/system/amr_system_coupler.hpp`: the IMEX callback reuses
  `mf_apply_source_treatment(m, U, aux, dt, /*imex=*/true)` (`amr_reflux_mf.hpp:75-82`) ->
  `backward_euler_source` (named device functor `BackwardEulerSourceKernel`). The source stays
  cell-local, snapshotted, OUTSIDE reflux.
ACCEPTANCE: a stiff IMEX block on AMR stable where the explicit diverges; mass conserved.
CONSERVATION: the implicit split does not touch the reflux registers (source out of flux) ->
conservation at the coarse-fine interfaces intact (Gap2 property).

### PR (viii) -- ONLY then: Schur / true global implicit / paper repro
WRITE-SET: out of scope for Phase 1. Relies on `CondensedSchurSourceStepper`
(`include/adc/coupling/schur/source/condensed_schur_source_stepper.hpp`) and `schur_condensation.hpp`. The
`treatment == Implicit` stays REJECTED by the `AmrScheduler` until a true global stepper
exists.


## 5. Regrid algorithm (union of the tags, rebuild once, prolong/restrict of ALL the blocks)

PRELIMINARY HARD CONSTRAINT (owner correction, point 4). As long as the algorithm below is
not implemented, `n_blocks >= 2` AND `regrid_every > 0` is EXPLICITLY REFUSED by the facade
(clear throw). The multi-block hierarchy is FROZEN at construction; accepting `regrid_every`
in multi-block would make one believe in a nonexistent dynamic AMR. The refusal is wired in the PR of the
runtime registry (point 3), not in the first layout step.

It is the central missing piece. The mono-block brick is `amr_regrid_finest`
(`include/adc/coupling/amr/amr_regrid_coupler.hpp`): tag of the parent -> `grow_tags` -> `all_reduce_or`
if coarse distributed -> `berger_rigoutsos` -> clamp nesting -> new fine `BoxArray` ->
carry-over of the fine data + parent interp -> realloc aux. It operates on ONE block.

MULTI-BLOCK ALGORITHM (to write, `AmrRegridPolicy` / `AmrSystemCoupler::regrid`):

1. `solve_fields()` once (aux up to date, for the phi gradient criterion).
2. UNION OF THE TAGS on the parent level: for each block, `tag_cells(block.levels[pk].U,
   pdom, crit_block)` (`include/adc/amr/regridding/regrid.hpp:30-42`), then logical OR of the `TagBox`:
   ```
   tags = tags_electrons OR tags_ions OR tags_neutrals OR tags_phi OR tags_user
   ```
   `tags_phi` tags on `aux_[pk]` (component 0 = phi, or its gradient). `TagBox` is a
   char grid; the OR is a `|=` cell by cell. `grow_tags` next (nesting + margin).
3. If coarse distributed: `all_reduce_or_inplace` on the UNITED tags (same MPI-safe guard as
   `amr_regrid_finest:59-60`) so that all the ranks start from the SAME tag grid ->
   `berger_rigoutsos` produces IDENTICAL patches per rank (otherwise the dmaps diverge).
4. REBUILD OF THE LAYOUT A SINGLE TIME: `berger_rigoutsos(grown)` -> clamp nesting -> new
   fine `BoxArray` + a single `DistributionMapping(nfine, n_ranks())`. This layout is SHARED by
   all the blocks (it is the golden rule: one rebuild, not one per species).
5. COHERENT PROLONG / RESTRICT OF ALL THE BLOCKS on this new layout: for each block,
   rebuild `levels[fk].U` on the new `BoxArray` (ghost width inherited from
   `n_grow()`, cf. `amr_regrid_finest:73`), fill by CARRY-OVER of the fine data where
   the old patch covers + INTERP from the parent elsewhere (exactly the body of
   `amr_regrid_finest:94-121`). ALL the blocks use the SAME `BoxArray`/`dmap`: no block
   absent from a patch.
6. REBUILD AUX / PHI / RHS: realloc the SHARED aux per level on the new layout
   (`AmrLevelStack::reattach_aux` preserves the address), rewire `levels[k].aux = &aux_[k]`,
   re-place B_z (`fill_bz`), re-`solve_fields`.
7. VERIFICATION PER BLOCK: the mass of each block (component 0 integrated) must be conserved
   by the regrid (the carry-over + conservative interp redistributes without creating/destroying). Test:
   `mass(b)` before == `mass(b)` after regrid, to the tolerance.

REAL DIFFICULTY FLAGGED. `amr_regrid_finest` is a free function that only regrids the
finest level (`L.back()`), not a hierarchy with N arbitrary levels. For Phase 1
(coarse + 1 fine, which `build_amr_compiled:71-78` materializes) it is sufficient. Beyond 2
levels, the multi-level regrid does not exist yet even in mono-block: to note as a limit,
not to solve here.


## 6. Risk registry (with de-risking)

- MIXED stride/substeps across the blocks in `step_cfl`. RISK: a dt computed on `w_max`
  alone, multiplied by stride, violates the CFL by a factor M. DE-RISK: copy LITERALLY the
  per-block formula of `System::step_cfl` (`python/bindings/system/base/system.cpp:1667-1680`),
  `dt = min_b cfl*h*substeps_b/(stride_b*w_b)`, frozen blocks excluded. Test: a block stride=4
  stays under the effective CFL.

- MULTI-LEVEL ELLIPTIC PRECISION. RISK: the Poisson of Phase 1 is solved on the COARSE
  then injected coarse->fine (`coupler_inject_aux_mb`), it is NOT a true multi-level solve.
  DE-RISK: Phase 1 assumes "coarse + inject" (the observable diocotron instability lives on a
  median circle resolved by the coarse, cf. `AmrSystem::potential()`,
  `amr_system.hpp:175-179`). The true multi-level (composite solve) is Phase 2, flagged as
  such; do not promise it in Phase 1.

- REGRID COHERENCE + CONSERVATION. RISK: rebuilding the blocks on slightly
  different layouts (box order, dmap) breaks the co-location and the conservation.
  DE-RISK: A SINGLE `BoxArray`/`DistributionMapping` computed (step 4), reused as is by
  all the blocks; test `mass(b)` invariant at the regrid, per block.

- COUPLED SOURCE: exactly opposite contributions, same cell. RISK: an index shift
  or a read of an already-modified state breaks the pair conservation. DE-RISK:
  `CoupledSourceKernel` (`coupled_source_program.hpp:98-108`) evaluates ALL the terms on the
  state FROZEN at the start of the step (`reg` frozen) before writing; the order of the `.add` does not matter at 1st
  order. On a shared hierarchy, no inter-species interpolation (same `(i,j)`). Test:
  `n_i + n_g` constant to machine precision, per level.

- COUPLED SOURCE: AMR DOUBLE COUNTING (owner correction, point 5). RISK: `coupled_source_step`
  applies the source LEVEL BY LEVEL, so a coarse cell COVERED by a fine level
  receives the term TWICE (coarse + fine averaged down). For a NON-zero source, the
  integrated COMPOSITE MASS is over-counted in the covered cells (the pair invariant PER
  CELL, for its part, stays exact: it is therefore not enough to detect this bug). DE-RISK: application
  LEAF-ONLY (uncovered cells only, via `CoverageMask`) OR `average_down` / sync AFTER
  the source. REQUIRED test: COMPOSITE AMR CONSERVATION on a non-zero source whose zone
  OVERLAPS the coarse-fine interface (total mass on the composite hierarchy conserved).

- INVARIANCE BY MPI RANK. RISK: `fab(0)` without `local_size()` guard, or a coarse distributed
  badly reduced. DE-RISK: iteration over `local_size()` everywhere (cf. `mass`,
  `amr_system_coupler.hpp:291`); `DistributionMapping(ba.size(), n_ranks())`;
  `all_reduce_or_inplace` on the united tags before clustering (distributed regrid);
  `FluxRegister::gather` = `all_reduce_sum_inplace` (identity in serial -> bit-identical np=1).
  Test: np=1/2/4 bit-identical (modeled on `test_mpi_amr_multipatch`).

- nvcc CONFORMANCE OF THE NAMED FUNCTORS. RISK: a generic lambda in an unevaluated context
  (concept) or an extended lambda cross-TU trips nvcc (device Heisenbug). DE-RISK: the
  harness is already applied (`ForEachBlockProbe`, `coupled_system.hpp:59-63`;
  `BlockRhsEval`/`AdvanceExplicit`/`MaxSpeed`/`PoissonRhs`, `block_builder.hpp`;
  `CoupledSourceKernel`, `BackwardEulerSourceKernel`). Any new first-instantiation
  from an external TU goes through a NAMED FUNCTOR, never a lambda (harness #64/#97).

- DEVICE CLEANLINESS GH200. RISK (recent lesson): a `Geometry`/`Box2D` accessor NOT `ADC_HD`
  used in a device kernel silently breaks the device numerics. DE-RISK: every
  new accessor read in a device kernel MUST be `ADC_HD` (cf.
  `for_each_cell_reduce_sum` in `mass`, `amr_system_coupler.hpp:293-294`). Validate once
  a multi-block production case on GH200 (modeled on `gpu_amrsys_facade_validate.cpp`, which
  already instantiates `CoupledSystem` 2 blocks + AMR 2 levels + Poisson + step).


## 7. Phase 2 / Phase 3 boundary (locked)

PHASE 1 (the v1 target): one common hierarchy; N `AmrBlock` each on ALL the patches with
its model / scheme / time / substeps / stride / evolve; co-located summed Poisson; coupled
sources same-cell with opposite contributions; regrid by union of the tags (incl. tags_phi)
with coherent prolong/restrict of ALL the blocks; reflux / conservation block by block;
substeps / stride / evolve honored by the scheduler AND by `step_cfl`.

PHASE 2 (later, STAYS CONSERVATIVE): refinement criteria PER BLOCK (the union stays, but
each block declares its criterion); cost weights PER BLOCK for load balancing;
exploitation of evolve/stride/substeps for a TEMPORAL jump ONLY, that is to say a jump
GLOBAL TO A BLOCK in time (stride / evolve=false at the block level), NEVER a local spatial
absence of a block on a patch. EXPLICITLY: do NOT skip the advance of a block on
certain patches; a block is always present and conservative everywhere. True multi-level elliptic
solve (composite) here.

PHASE 3 (only if a real scientific need emerges): DISTINCT hierarchies per species
WITH conservative projections between hierarchies. Much harder, explicitly NOT the
first objective.

CROSS-CUTTING INVARIANT: any future optimization stays CONSERVATIVE and TEMPORAL; never a local spatial
absence of a block on a patch.


## 8. Acceptance test checklist (Phase 1)

- [x] LAYOUT (first step, DELIVERED): `same_layout_or_throw` THROWS on any mismatch (boxes,
      ORDER of the boxes, dmap, dx/dy, number of levels) and PASSES on an identical layout;
      mono-block bit-identical (`dmax == 0`). Test `tests/test_amr_layout_guard.cpp`.
- [x] multi-block RUNTIME FACADE (DELIVERED, runtime registry PR): `AmrSystem` accepts N native blocks
      co-located on ONE shared hierarchy via the type-erased engine `AmrRuntime`
      (`include/adc/runtime/amr/amr_runtime.hpp`), registry by name of closures
      (advance / add_elliptic_rhs / max_speed / mass / density / potential). SYSTEM Poisson with a
      SUMMED right-hand side co-located (Sum_b elliptic_rhs_b(U_b) = q0 n0 + q1 n1 on the shared
      coarse). Tests `tests/test_amr_system_twoblock.cpp`, `python/tests/test_amr_multiblock.py`.
- [x] Two explicit AMR blocks with DIFFERENT schemes, stable over N steps (mass per block conserved).
- [ ] electrons IMEX(substeps=10) + ions Explicit(substeps=1), stable; AND the inverse. (later PR)
- [ ] neutrals stride=20 always read by the sources and the Poisson between catch-ups. (later PR)
- [ ] `evolve=False` present as a fixed background in the elliptic right-hand side. (later PR)
- [x] MULTI-BLOCK + `regrid_every > 0` now SUPPORTED via the union-tag regrid (capstone Phase 2);
      the old refusal at `ensure_built` is lifted.
- [ ] regrid conserves the mass of EACH block (`mass(b)` before == after, per block). (union regrid
      = later PR; the multi-block is FROZEN for now)
- [x] MULTI-BLOCK PRODUCTION DSL (capstone v, DELIVERED): add_compiled_model(AmrSystem&) installs a block
      NAMED (not a replacement) -> SEVERAL compiled blocks (and MIX compiled + native) co-located
      on the shared hierarchy via AmrRuntime. add_compiled_model freezes TWO builders (mono AmrCouplerMP
      + multi AmrRuntimeBlock); set_compiled_block STACKS instead of throwing; build_multi invokes the
      runtime builder of the compiled block on the SHARED layout. The 2nd compiled block NO LONGER THROWS.
      Test `tests/test_amr_multiblock_compiled.cpp`. A .so loader recompiled against the provided header supplies the
      runtime builder (flat ABI of the loader unchanged; an EARLIER loader throws clearly at build_multi).
      The flat ABI of the loader carries NEITHER the multirate (stride) NOR the partial IMEX mask
      (implicit_vars / implicit_roles): it is NOT an acceptable silent loss -> the facade
      Python `AmrSystem.add_equation` REJECTS them explicitly (ValueError) on the production path
      (.so). For these parameters: native `AmrSystem.add_block` (ModelSpec) or `add_compiled_model(
      AmrSystem&)` DIRECT (C++ header), which expose stride and the mask (the DIRECT IMEX
      multi-block path is exercised by `tests/test_amr_multiblock_compiled.cpp`).
- [ ] coupled source `+k n_e n_g` / `-k n_e n_g`: `n_i + n_g` constant to machine precision,
      per level, on all the patches. (later PR)
- [ ] COMPOSITE AMR CONSERVATION of the coupled source (point 5): on a NON-zero source whose
      zone OVERLAPS the coarse-fine interface, the integrated TOTAL mass on the composite hierarchy
      is conserved (leaf-only OR sync after the source) -> rules out the double counting of the covered
      cells. Per-cell conservation is NOT sufficient. (later PR)
- [x] MPI np=1/2/4 bit-identical (reflux per block; replicated coarse consistent cross-rank).
      Test `tests/test_mpi_amr_twoblock_parity.cpp` (fine patch DISTRIBUTED round-robin -> no
      over-counting of the reflux; np=1/2/4 bit-identical, cross-rank spread == 0).
- [x] Kokkos Serial / OpenMP green (two-block + mono-block).
- [ ] a multi-block production case validated on GH200 (full device instantiation). (dedicated
      device validation to do on ROMEO; the path is device-clean by construction, named functors)
- [x] NON-REGRESSION: a mono-block case stays BIT-IDENTICAL (maxdiff=0) through the facade
      (the mono-block ALWAYS routes through AmrCouplerMP untouched; never through AmrRuntime).


## 9. Code references (all verified at this head)

- `include/adc/coupling/system/amr_system_coupler.hpp`: AMR multi-block engine (exists); NO
  regrid.
- `include/adc/coupling/amr/amr_coupler_mp.hpp`: MONO-BLOCK AMR coupler + `regrid` (delegates to
  `amr_regrid_finest`).
- `include/adc/coupling/amr/amr_regrid_coupler.hpp`: `amr_regrid_finest` (Berger-Rigoutsos, finest
  level).
- `include/adc/coupling/base/elliptic_rhs.hpp`: `ChargeDensityRhs` (sum `Sum_s q_s n_s`,
  one `SpeciesCharge` per block).
- `include/adc/coupling/source/coupled_source.hpp`: concept `CoupledSourceFor`, `NoCoupledSource`.
- `include/adc/coupling/source/coupled_source_program.hpp`: `CoupledSourceKernel` (P5 #131, POD
  device-clean, opposite additive writes).
- `include/adc/coupling/amr/amr_level_storage.hpp`: `AmrLevelStack` (aux address invariant).
- `include/adc/core/model/coupled_system.hpp`: `CoupledSystem<Blocks...>`, `for_each_block`,
  `ForEachBlockProbe` (named device-clean functor).
- `include/adc/core/model/equation_block.hpp`: `EquationBlock<Model, Spatial, Time>`.
- `include/adc/numerics/time/amr/reflux/amr_reflux_mf.hpp`: `AmrLevelMP`, `LevelHierarchy`, `advance_amr`,
  `FluxRegister`, `CoverageMask`, `CoarseFineInterface`, `mf_apply_source_treatment`,
  `mf_average_down_mb`.
- `include/adc/numerics/time/schemes/scheduler.hpp`: `advance_subcycled`, `block_substeps_v`,
  `block_stride_v`, `block_time_treatment_v`.
- `include/adc/coupling/system/system_coupler.hpp`: `SystemAssembler` / `SystemDriver` (mono-level
  split), `step_cfl` substeps-aware (`cfl*h*substeps/(stride*w)`).
- `include/adc/runtime/system.hpp` + `python/bindings/system/base/system.cpp`: multi-block RUNTIME facade (model to
  imitate), `Species`, `stride_due` (HOLD-THEN-CATCH-UP), `step_cfl` (lines 1663-1693).
- `include/adc/runtime/amr_system.hpp` + `python/bindings/amr/amr_system.cpp`: MONO-BLOCK AMR RUNTIME facade
  (refusal of the 2nd block, lines 129-130 and 152-153).
- `include/adc/runtime/builders/compiled/amr_dsl_block.hpp`: `add_compiled_model(AmrSystem&)` (a single block).
- `include/adc/runtime/builders/block/block_builder.hpp`: `make_block` / `make_max_speed` /
  `make_poisson_rhs`, named device-clean functors.
- `docs/COUPLER_HIERARCHY.md`, `docs/SCHUR_CONDENSATION_DESIGN.md`, `docs/GPU_RUNTIME_PORT.md`
  (device-clean harness), `docs/PAPER_ROADMAP.md`.
