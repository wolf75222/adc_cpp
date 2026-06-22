# Design: regrid by TAG UNION on the shared multi-block hierarchy (Phase 2)

DESIGN VALIDATED + IMPLEMENTED (Phase 2 capstone, C.6). This document was first a reasoned SPEC
(DESIGN-ONLY); decisions D1-D5 were settled by the owner and the R0-R8 algorithm is now
IMPLEMENTED in the runtime multi-block engine. It describes the algorithm that makes the previously
FROZEN multi-block AMR hierarchy ADAPTIVE.

## PROGRESS STATUS (Lot C.6 implementation)

DELIVERED (runtime multi-block engine AmrRuntime):
- [x] (R3) helper `tag_union(span<TagBox>)` (cell-by-cell OR) -> `include/adc/amr/tagging/tag_box.hpp`.
- [x] REFACTOR (section 6): `amr_regrid_finest` split into `regrid_compute_fine_layout`
      (tags -> grow -> all_reduce_or -> berger_rigoutsos -> clamp -> (fb, dmap)) + `regrid_field_on_layout`
      (re-grids ONE field on an IMPOSED layout) -> `include/adc/coupling/amr/amr_regrid_coupler.hpp`.
      `amr_regrid_finest` stays the chaining of the two -> mono-block path BIT-IDENTICAL (verified).
- [x] (R0-R8) `AmrRuntime::regrid()`: (R0) solve_fields, (R1) per-block tags (predicate D1), (R2) phi
      tags on |grad phi| (D4), (R3) union + grow, (R4) all_reduce_or if coarse distributed, (R5) a single
      clustering -> shared layout, (R6) prolong/restrict of ALL blocks (including stride-held, D3)
      with ghost inherited per block, (R7) rebuild shared aux + re-wiring, (R8) re-solve (coverage
      cascade). (V3) `same_layout_or_throw` post-regrid. -> `include/adc/runtime/amr/amr_runtime.hpp`.
- [x] registration API: `set_regrid(every, grow, margin)` (D2: regrid BEFORE the step, macro-step
      cadence), `set_block_tag_predicate(b, crit)` (D1), `set_phi_tag_predicate(crit)` (D4).
- [x] cadence in `AmrRuntime::step`: regrid every `regrid_every` macro-steps, BEFORE the step;
      `regrid_every == 0` -> regrid never called -> BIT-IDENTICAL to the frozen hierarchy (verified).
- [x] FACADE UNLOCK (T7): `python/bindings/amr/amr_system.cpp` no longer REFUSES multi-block +
      `regrid_every > 0`. `build_multi` wires `runtime->set_regrid(cfg.regrid_every)` and sets the tag
      predicate PER BLOCK (D1: selected variable > `refine_threshold`; default = component 0 like the
      mono-block AmrCouplerMP, ADC-296 lets `set_refinement(threshold, variable=, role=)` pick it per
      block by name/role, resolved against the block's cons_vars, STRICT -- no silent component-0
      fallback). `refine_threshold == 1e30` (default) -> no tag -> mesh unchanged. `regrid_every == 0`
      -> FROZEN hierarchy, bit-identical to Phase 1.
- [x] acceptance tests (a-e + T7): `tests/test_amr_multiblock_regrid_union.cpp` (NAMED tag functors,
      nvcc-safe). (a) hierarchy that evolves, (b)+(c) union A OR B OR |grad phi|, (d) stride-held block
      re-gridded, (e) regrid_every==0 bit-identical, (V1) conservation per block, (T7) facade
      AmrSystem no longer throws + frozen bit-identical. Contract tests updated (test_amr_system_contract,
      test_amr_system_twoblock, python/tests/test_amr_multiblock.py: the old refusal becomes an
      acceptance).
- [x] non-regression: 102/102 ctest green (including all existing `test_amr_*` and `test_mpi_amr_*`).

OUT OF SCOPE (conforming to the boundaries):
- > 2 levels (D5: v1 at 2 levels), per-species criterion producing distinct meshes (Phase 3),
  true composite elliptic solve: all OUT of scope, unchanged.
- phi predicate (|grad phi|, D4) and user predicate: exposed at the AmrRuntime ENGINE level
  (`set_phi_tag_predicate`) and tested there, but NOT wired by default from the Python facade (the v1
  facade sets the per-block density predicate; phi stays opt-in on the engine side, mechanical to wire if needed).

PROJECT FRAME. The multi-block runtime delivered so far (`AmrRuntime`,
`include/adc/runtime/amr/amr_runtime.hpp`, runtime counterpart of `AmrSystemCoupler`,
`include/adc/coupling/system/amr_system_coupler.hpp`) is "Phase 1 multi-block with FROZEN hierarchy":
N blocks (electrons, ions, neutrals, ...) co-located on ONE shared AMR hierarchy, a single
coarse Poisson with SUMMED right-hand side, cell-by-cell coupled sources, multirate per block
(substeps / stride / evolve), but a mesh that NEVER MOVES after construction. Neither
`AmrSystemCoupler` nor `AmrRuntime` has a `regrid` method (verified: grep `regrid` in the
two files returns nothing). The Python facade therefore explicitly REFUSES the combination
multi-block + `regrid_every > 0` (`python/bindings/amr/amr_system.cpp:246-251`).

PURPOSE OF THIS DOCUMENT (Phase 2, the capstone finale). Specify the `regrid` by TAG UNION
that turns this frozen hierarchy into an ADAPTIVE hierarchy: a single collective criterion
(the union of all blocks), a single Berger-Rigoutsos clustering, a single new SHARED layout,
then prolongation / restriction / reflux PER BLOCK on this layout. This document UNLOCKS
explicitly multi-block + `regrid_every > 0`, refused today.

The target model stays that of AMReX / FLASH / SAMRAI already laid out by
`docs/AMR_MULTIBLOCK_DESIGN.md` (section 5): ONE common hierarchy refined by the UNION of the
criteria of all fields, NEVER a per-species hierarchy.


## 0. Preliminary honesty note (real state of the code at this head)

The code was re-read directly. Four facts structure everything that follows.

FACT 1: the MONO-BLOCK REGRID BRICK EXISTS and is proven. `amr_regrid_finest`
(`include/adc/coupling/amr/amr_regrid_coupler.hpp`) does, for ONE block: tag the parent
(`tag_cells`) -> `grow_tags` (nesting + margin) -> `all_reduce_or_inplace` if coarse distributed
-> `berger_rigoutsos` -> nesting clamp (margin) -> new fine `BoxArray` + a single
`DistributionMapping(nfine, n_ranks())` -> carry over existing fine data + interpolation
from the parent elsewhere -> realloc of the fine-level aux (address re-wired). This is exactly
the skeleton the multi-block version needs; only the "one single layout for all blocks"
orchestration is missing.

FACT 2: the MONO-BLOCK REGRID IS ALREADY WIRED TO THE FACADE, but ONLY for the mono-block path.
`AmrCouplerMP::regrid` (`include/adc/coupling/amr/amr_coupler_mp.hpp:321-325`) delegates to
`amr_regrid_finest`, and the `h.step` closure of the mono-block path calls it periodically
(`include/adc/runtime/builders/compiled/amr_dsl_block.hpp:101-104`:
`if (regrid_every > 0 && *step_state % regrid_every == 0) cpl->regrid(crit);`). The
multi-block path, in contrast, goes through `AmrRuntime` which has no `regrid`: its hierarchy is frozen.

FACT 3: the multi-block + `regrid_every > 0` REFUSAL IS ALREADY IN PLACE, by design (owner
correction, `docs/AMR_MULTIBLOCK_DESIGN.md` section 4). It is wired into `ensure_built` of the facade
(`python/bindings/amr/amr_system.cpp:246-251`): silently allowing `regrid_every > 0` in multi-block
would make the API CLAIM it does dynamic AMR while the mesh never moves
(dangerous illusion). LIFTING this refusal is precisely what THIS DESIGN authorizes, once
the algorithm below is implemented and tested.

FACT 4: the LAYOUT GUARD EXISTS and will be the natural safety net of the post-regrid.
`detail::same_layout_or_throw` (`include/adc/coupling/system/amr_system_coupler.hpp:122-140`, reused
by `AmrRuntime` at the ctor, `amr_runtime.hpp:213-218`) compares EXACTLY, across all blocks:
number of levels, then per level `BoxArray` (boxes AND order, via `ba.boxes() ==`),
`DistributionMapping` (rank per box, via `dm.ranks() ==`) and `dx`/`dy` (bit for bit). It is the
exact PRECONDITION of the shared aux: all blocks live on EXACTLY the same layout per
level. The regrid must RE-ESTABLISH this invariant after moving the mesh (cf. section 4,
verification (V3)). Reusing it as a post-regrid assertion costs one line and catches any
reconstruction inconsistency.

CONSEQUENCE. The Phase 2 work is NOT to invent a regrid, but to ORCHESTRATE the existing
mono-block brick around a COLLECTIVE criterion (tag union) and a UNIQUE layout applied
to ALL blocks, then lift the facade refusal and add the layout-change tests. Good
news to state plainly: the target is closer than it looks.


## 1. Architecture decision and why

DECISION (locked by the owner). ONE single SHARED AMR hierarchy for ALL blocks, NEVER
a per-species hierarchy in v1. The regrid is driven by the COLLECTIVE UNION of the tags of all
blocks (tags = electrons OR ions OR neutrals OR phi OR user), computed across all
blocks AND all MPI ranks, followed by A SINGLE Berger-Rigoutsos clustering that produces A SINGLE
new shared layout. Prolongation / restriction / reflux are applied PER BLOCK on this
new shared layout: each block is re-gridded on the union layout; ALL evolved blocks
are present on ALL patches, NEVER spatially absent.

WHY a UNION regrid on a shared hierarchy, and NOT a per-species regrid:

- SHARED AUX AND SINGLE POISSON. The aux (phi, grad phi, [B_z, T_e]) is SHARED per level
  (`AmrRuntime::aux_`, one `MultiFab` per level, `amr_runtime.hpp:233-238`; likewise
  `AmrSystemCoupler::aux_`). The aux pointer of each `AmrLevelMP` of each block points to this
  same shared `MultiFab`. This ONLY MAKES SENSE if all blocks live on EXACTLY the same
  layout per level. A regrid that produced per-species layouts would immediately break
  the single aux and the single Poisson: each density would have to be interpolated onto a mesh
  common to each solve. The tag union guarantees a single layout by construction.

- CO-LOCATION OF COUPLED SOURCES. The inter-species sources read several species in
  the SAME cell (`AmrRuntime::coupled_source_step`, `amr_runtime.hpp:372-407`: `kern.in[c]` and
  `kern.out[t]` index the SAME `(i,j)`, same `fab(li)`, per level). This cell-local read
  requires all blocks to share the layout. A union regrid preserves it; a per-species
  regrid would destroy it.

- CONSERVATION AND REFLUX PER BLOCK. The reflux is already block by block (each block has its own
  flux registers in `advance_amr`, closure `AmrRuntimeBlock::advance`). The regrid must
  conserve the mass of EACH block independently, which assumes the mesh is the same for
  all (otherwise the coarse-fine interfaces would differ per block and the reflux would no longer be
  comparable). Cf. section 3, step (R5) and verification (V1).

- MPI / KOKKOS: a single distribution plan. A shared hierarchy = a single
  `DistributionMapping` per level, hence a single collective reduction of the tags and a single set of
  halos. The tag-union must therefore be a cross-rank COLLECTIVE (cf. section 3, step (R3), and
  section 6): all ranks must start from the SAME tag grid so that Berger-Rigoutsos
  produces IDENTICAL patches per rank, otherwise the `DistributionMapping` diverge and MPI
  desynchronizes. This is exactly the MPI-safe guard already present in `amr_regrid_finest:59-60`,
  which must be hoisted to the union level.

WHAT WE DON'T OPEN (and why say it). No per-species hierarchy (Phase 3,
`docs/AMR_MULTIBLOCK_DESIGN.md` section 7). No per-block criterion that would produce distinct
meshes: in v1 the criterion stays the UNION, a single layout. No local spatial absence of a
block on a patch: a block is ALWAYS present and conservative everywhere, even if its own
criterion did not trigger it, because it is coupled to the others. These restrictions are
deliberate: they preserve the single aux, the single Poisson and the cell-by-cell
conservation of the sources, the reason for the sharing.


## 2. Scope: what Phase 2 delivers, and what it does not deliver

DELIVERED (v1 target of the union regrid):
- a regrid driven by the UNION of the tags of all blocks + phi tags + user tags;
- ONE clustering, ONE new shared layout applied to ALL blocks;
- prolong / restrict / reflux per block on the new layout;
- mass conservation PER BLOCK across the regrid (per-block invariant assertion);
- backend correctness (MPI + Kokkos CPU and GPU): cross-rank collective tag union, layout
  consistent on each rank;
- the lifting of the facade refusal: multi-block + `regrid_every > 0` becomes SUPPORTED.

NOT DELIVERED (stays conforming to the boundaries of `docs/AMR_MULTIBLOCK_DESIGN.md` section 7):
- the MULTI-LEVEL regrid (> 2 levels). Like `amr_regrid_finest`, the v1 union regrid only
  reconstructs the finest level (coarse + 1 fine), which the shared layout
  `make_shared_amr_layout` materializes (`amr_dsl_block.hpp`, coarse + 1 FIXED central fine patch). Beyond
  2 levels, the multi-level regrid does not exist yet even in mono-block: a limit to note, not to
  solve here.
- the PER-BLOCK CRITERIA (extended Phase 2 / Phase 3: each block declares its criterion, the union
  stays). In v1 the union criterion can be a single criterion applied to each field; the
  refinement stays the union.
- the true multi-level elliptic SOLVE (composite). The Poisson stays "coarse + inject"
  (`coupler_inject_aux_mb`), as in Phase 1.


## 3. Union regrid algorithm, step by step

Target: a method `AmrRuntime::regrid(...)` (and its compile-time counterpart
`AmrSystemCoupler::regrid(...)`), called periodically by the facade (cf. section 5). We denote
`pk` the parent level (coarse, `pk = 0` in v1 at 2 levels), `fk = pk + 1` the fine level to
reconstruct, `nlev_` the number of levels.

(R0) PRECONDITION. Up-to-date fields: call `solve_fields()` once (aux per level up to date, for
the phi-gradient criterion, exactly like `amr_runtime.hpp:413`). Snapshot of the masses per
block `mass(b)` BEFORE regrid (for verification (V1)).

(R1) PER-BLOCK TAGS ON THE PARENT. For each block `b`, compute a `TagBox` on the parent level
via `tag_cells((*blocks_[b].levels)[pk].U, pdom, crit_b)` (`include/adc/amr/regridding/regrid.hpp:36-47`).
`crit_b` is a predicate `(ConstArray4 a, int i, int j) -> bool` on the density of the block (component
0) or a gradient. In v1 the criterion can be common to all blocks; the UNION below stays
the contract. `TagBox` is a dense grid of `char` 0/1 on `pdom` (`tag_box.hpp`).

(R2) PHI TAGS AND USER TAGS. Compute a `TagBox` on the parent's aux channel `aux_[pk]`
(component 0 = phi, or its discrete gradient): `tag_cells(aux_[pk], pdom, crit_phi)`. Add the
optional user tags (predicate provided by the caller).

(R3) TAG UNION (cell by cell). Compose all the `TagBox` of (R1)+(R2) into ONE union `TagBox`
by cell-by-cell logical OR:

    tags_union = tags_e OR tags_i OR tags_n OR tags_phi OR tags_user

All `TagBox` share the same box `pdom`, so the union is a `|=` per index. NEW
CODE: a helper `tag_union(span<const TagBox>) -> TagBox` (a few lines, no physical
dependency). Then `grow_tags(tags_union, grow, pdom)` (`regrid.hpp:52-64`) for nesting and the
margin.

(R4) CROSS-RANK COLLECTIVE REDUCTION (MPI). If the coarse is DISTRIBUTED (`pk == 0 &&
!replicated_coarse_`), each rank has only tagged its LOCAL boxes (`tag_cells` only iterates over
`mf.local_size()`, `regrid.hpp:38`). Reduce the UNITED tags via `all_reduce_or_inplace` on the
`grown.t.data()` buffer (SAME MPI-safe guard as `amr_regrid_finest:59-60`, hoisted to the union
level) so that ALL ranks start from the SAME tag grid. Replicated: the tag grid
is already complete on each rank -> `all_reduce_or` would be the identity (no-op, we avoid it). This
reduction is INDISPENSABLE to cross-rank layout consistency: otherwise Berger-Rigoutsos
would produce different patches per rank and the `DistributionMapping` would diverge.

(R5) UNIQUE CLUSTERING -> SHARED LAYOUT. A SINGLE `berger_rigoutsos(grown, ClusterParams{})`
(`include/adc/amr/tagging/cluster.hpp:171-181`) on the reduced union tags. Apply the nesting clamp
(`margin`) and the parent coords -> fine coords conversion (parent x2) EXACTLY like
`amr_regrid_finest:62-68`. Build A SINGLE fine `BoxArray fb` and A SINGLE
`DistributionMapping((int)fb.size(), n_ranks())`. This is THE GOLDEN RULE: one rebuild, not one per
species. If `fb` is empty (nothing to refine), return without touching the mesh (no-op, like
`amr_regrid_finest:69`).

(R6) COHERENT PROLONG / RESTRICT OF ALL BLOCKS. For EACH block `b`, reconstruct
`(*blocks_[b].levels)[fk].U` on the SAME `BoxArray fb` and the SAME `dmap`, with the ghost width
INHERITED from `(*blocks_[b].levels)[fk].U.n_grow()` (a MUSCL order-2 block carries 2 ghosts, cf.
`amr_regrid_finest:73`) and the component width `U.ncomp()` of the block. Fill by:
  (a) INTERP from the block's parent where the new patch is not covered by the old fine
      (replicated path: `mf_find_box`; distributed path: `parallel_copy` toward a LOCAL
      child-coarsen grid then `device_fence`, EXACTLY `amr_regrid_finest:84-112`);
  (b) CARRY OVER existing fine data where the old patch covers the new (intersection
      `nb.intersect(old.box(ol))`, `amr_regrid_finest:113-120`).
This is, per block, the BODY of `amr_regrid_finest`, but on an `fb`/`dmap` layout IMPOSED from
the outside (the same for all blocks) instead of being recomputed per block. All blocks
use the SAME `fb`/`dmap`: no block absent from a patch.

(R7) REBUILD OF THE SHARED AUX + RE-WIRING. Reallocate the SHARED aux of the fine level on the new
layout: `aux_[fk] = MultiFab(fb, dmap, aux_ncomp_, 1)` (width `aux_ncomp_` = max of the aux_comps
of the blocks, `amr_runtime.hpp:226-228`). Re-wire the aux pointer of EACH block:
`(*blocks_[b].levels)[fk].aux = &aux_[fk]` (the address `&aux_[fk]` stays stable after in-place
reallocation of the `MultiFab` in the existing `std::vector`). Re-set B_z per level if a block reads it
(`fill_bz` on the `AmrSystemCoupler` side; the `AmrRuntime` does not populate multi-block B_z in v1, cf.
`amr_runtime.hpp:222-225`). Then re-`solve_fields()` so that phi / grad phi are coherent with
the new mesh.

(R8) RESTORATION OF THE COVERAGE INVARIANT. As after a source or a transport, restore
the coherence of the coarse cells covered by a fine -> coarse cascade
(`mf_average_down_mb`, already done in `solve_fields`, `amr_runtime.hpp:416-419`). The re-solve of
(R7) already triggers it; noting it explicitly avoids a mass diagnostic (sum of the coarse alone)
counting a ghost coarse value under the patch.

CENTRAL INVARIANT. Steps (R5) and (R6) guarantee that `(*blocks_[b].levels)[fk].U.box_array()
== fb` and `... .dmap().ranks() == dmap.ranks()` for EVERY block `b`. This is exactly what
`detail::same_layout_or_throw` verifies; calling it as a post-regrid assertion (verification (V3))
catches any incoherent reconstruction.


## 4. Post-regrid verifications (what we assert when the layout changes)

Three invariants must be verified at EACH regrid (in debug, and by the acceptance tests
of section 7):

(V1) CONSERVATION PER BLOCK. The mass of each block (component 0 integrated over the coarse,
`AmrRuntime::mass(b)`, `amr_runtime.hpp:251`) is conserved across the regrid:
`mass(b)` BEFORE == `mass(b)` AFTER, to the tolerance of the carry-over + conservative interp. The carry-over of
the fine data + the interp from the parent redistribute without creating or destroying mass. To test
FOR EACH block independently (the reflux and the conservation are block by block).

(V2) GHOST CONSISTENCY. After the re-`solve_fields()` of (R7), the ghosts of the new fine level
(of `U` per block and of the shared aux) are coherent: coarse->fine injection of the aux
(`coupler_inject_aux_mb`, `amr_runtime.hpp:437-439`) and ghost filling of transport by
`advance_amr`. A patch reconstructed in MUSCL order 2 must carry `n_grow() >= 2` (verification of
the inherited ghost width, (R6)) otherwise the reconstruction would read out of bounds at the next step.

(V3) COLLECTIVE LAYOUT CONSISTENCY (cross-block AND cross-rank). Call
`detail::same_layout_or_throw` on the reconstructed level stack: all blocks share
EXACTLY the same `BoxArray` (boxes AND order), `DistributionMapping` (rank per box) and `dx`/`dy`
per level. Cross-rank: the collective reduction (R4) guarantees that each rank computed the SAME
`fb`; an MPI np=1/2/4 test must give bit-identical trajectories (mirror of
`test_mpi_amr_twoblock_parity` cited in `docs/AMR_MULTIBLOCK_DESIGN.md` section 8).


## 5. Composition with the current path (cadence, multirate, frozen hierarchy)

CADENCE `regrid_every`. The facade already carries `regrid_every` (`amr_system.hpp:40,63`; default 20).
The mono-block path uses it via the `h.step` closure
(`amr_dsl_block.hpp:101-104`). The multi-block path must use it analogously: call
`runtime->regrid(...)` every `regrid_every` macro-steps, BEFORE (or after) the `step(dt)` of the
macro-step, according to the chosen convention (cf. open decision (D2)). The macro-step counter
already exists (`AmrRuntime::macro_step_`, `amr_runtime.hpp:583`).

INTERACTION WITH THE MULTIRATE (substeps / stride). The regrid acts at the granularity of the MACRO-STEP,
NOT the substep: it sits between two macro-steps, never in the middle of a substep loop
(`amr_runtime.hpp:488-489`) nor between two stride catch-ups. A block HELD by its stride (outside the end
of the window, `(macro_step_+1) % stride != 0`, `amr_runtime.hpp:475`) is NONETHELESS re-gridded like
the others: it lives on all patches and contributes to the Poisson with its FROZEN state; the regrid
moves its mesh but does not advance it. The union criterion must therefore tag ALSO the held
blocks (their frozen state stays physically present). Interaction to test explicitly (cf. (D3)).

COMPOSITION WITH THE FROZEN HIERARCHY (transition from Phase 1 to Phase 2). As long as `regrid_every ==
0`, the multi-block path stays STRICTLY that of today (frozen hierarchy, bit-identical):
the regrid is never called. The UNLOCK consists in REPLACING the refusal
`python/bindings/amr/amr_system.cpp:246-251` by the activation of the cadence: multi-block + `regrid_every > 0`
stops throwing and wires the periodic regrid closure. The mono-block keeps its path
(`AmrCouplerMP`, untouched). Non-regression criterion: a multi-block case with `regrid_every == 0`
must stay BIT-IDENTICAL to before this PR (the union regrid only activates for
`regrid_every > 0`).


## 6. REUSE vs NEW CODE map

For each step of section 3, what is REUSED as-is vs what is NEW.

| Step | Reused (existing, verified) | New code |
|-------|-------------------------------|--------------|
| (R0) solve + snapshot | `AmrRuntime::solve_fields` (`amr_runtime.hpp:413`), `mass(b)` (`:251`) | snapshot of the masses per block |
| (R1) per-block tags | `tag_cells` (`amr/regrid.hpp:36`) | predicate `crit_b` per block (common in v1) |
| (R2) phi / user tags | `tag_cells` on `aux_[pk]` | `crit_phi`, optional user predicate |
| (R3) tag union | `TagBox` (`amr/tag_box.hpp`), `grow_tags` (`amr/regrid.hpp:52`) | helper `tag_union(span<TagBox>)` (cell-by-cell OR) |
| (R4) MPI reduction | `all_reduce_or_inplace` (already in `amr_regrid_finest:59-60`) | hoist the guard to the UNION level (on `tags_union`, not per block) |
| (R5) unique clustering | `berger_rigoutsos` (`amr/cluster.hpp:171`), `ClusterParams`, nesting clamp (`amr_regrid_coupler.hpp:62-68`) | compute `fb`/`dmap` ONCE, share between blocks |
| (R6) per-block prolong/restrict | BODY of `amr_regrid_finest:73-121` (replicated/distributed parent interp, `parallel_copy`, `device_fence`, fine carry-over) | refactor: `amr_regrid_finest` takes an IMPOSED layout (`fb`/`dmap`) instead of recomputing it; loop over the blocks |
| (R7) rebuild aux + re-wiring | aux realloc + re-wiring (`amr_regrid_coupler.hpp:123-124`; shared aux pattern `amr_runtime.hpp:233-238`), `coupler_inject_aux_mb`, `fill_bz` (`AmrSystemCoupler` side) | orchestration: SHARED aux (single one) re-wired toward ALL blocks |
| (R8) coverage cascade | `mf_average_down_mb` (already in `solve_fields`) | none (triggered by the re-solve) |
| (V1)-(V3) verifs | `mass(b)`, `same_layout_or_throw` (`amr_system_coupler.hpp:122`) | post-regrid asserts + tests (section 7) |

KEY REFACTOR (the only non-trivial change). `amr_regrid_finest` TODAY computes its own
`BoxArray`/`dmap` from the tags of a single block (`amr_regrid_coupler.hpp:52-74`). For the union,
this function must be SPLIT into two responsibilities:
  1. LAYOUT COMPUTATION: tags -> `grow` -> `all_reduce_or` -> `berger_rigoutsos` -> clamp ->
     `(fb, dmap)`. This is what we do ONCE on the tag union (R3)-(R5).
  2. RE-GRID A FIELD ON A GIVEN LAYOUT: takes `(fb, dmap, ngf, ncomp)` as a parameter and
     reconstructs `U` (parent interp + fine carry-over). This is the body `amr_regrid_coupler.hpp:73-124`
     WITHOUT the layout computation. We call it PER BLOCK (R6).
The current mono-block function stays the chaining of the two (1 then 2 on a single block), so the
mono-block path `AmrCouplerMP::regrid` stays BIT-IDENTICAL (it calls the full chaining).
This split is internal to `amr_regrid_coupler.hpp`: no new file, and the mono-block
contract is preserved.

NEW FACADE-SIDE METHOD. `AmrRuntime::regrid(crit, grow, margin, ...)` (and the counterpart
`AmrSystemCoupler::regrid`) orchestrates (R0)-(R8). It is the SOLE multi-block entry point added.
The facade unlock (`python/bindings/amr/amr_system.cpp`) replaces the throw by the wiring of the cadence.


## 7. Acceptance tests

Each test leaves the tree green; the invariants (V1)-(V3) are verified at each regrid.

(T1) TAG UNION. Two blocks whose fine structures are spatially DISJOINT (block A
tags one region, block B another): the union layout COVERS both regions. Verify that the
`fb` contains patches over both zones (not only that of one block). Degenerate case: a single
block tags -> same `fb` as the mono-block regrid of that block (parity with `amr_regrid_finest`).

(T2) CONSERVATION PER BLOCK (V1). Over N successive regrids (a multi-block diocotron case where the
structure moves), `mass(b)` stays constant to tolerance FOR EACH block. A block whose
criterion never triggers also conserves its mass (it is re-gridded as background, present
everywhere).

(T3) LAYOUT CONSISTENCY (V3) + GHOSTS (V2). After regrid, `same_layout_or_throw` passes (all
blocks on the same `fb`/`dmap`); the ghost width of the new fine is >= that required by the
most demanding scheme (MUSCL order 2 -> 2 ghosts). A transport step after regrid does not read
out of bounds (sanitizer / assertion).

(T4) MPI np=1/2/4 BIT-IDENTICAL. Mirror of `test_mpi_amr_twoblock_parity`
(`docs/AMR_MULTIBLOCK_DESIGN.md` section 8): coarse distributed round-robin, union regrid with
`all_reduce_or_inplace` on the united tags -> `fb` IDENTICAL per rank, np=1/2/4 trajectories
bit-identical after several regrids. The cross-rank spread of the fine `BoxArray` is zero.

(T5) KOKKOS Serial / OpenMP green (multi-block + regrid active). Backend-correct: the
reconstruction loops (R6) and `coupler_inject_aux_mb` go through the `_mb` primitives already ported
device-side. A GPU case (GH200) validates once the full device instantiation with regrid is in place
(`parallel_copy` + `device_fence` of the distributed path, already in `amr_regrid_finest:89-92`).

(T6) FROZEN NON-REGRESSION. Multi-block with `regrid_every == 0` stays BIT-IDENTICAL to Phase 1
(the regrid is never called). Mono-block stays BIT-IDENTICAL (`AmrCouplerMP` path untouched,
full chaining of `amr_regrid_finest`).

(T7) UNLOCK. Multi-block + `regrid_every > 0` no longer throws (the old refusal
`python/bindings/amr/amr_system.cpp:246-251` is lifted) and produces a hierarchy that effectively MOVES between
two steps (the fine `BoxArray` changes when the structure moves).


## 8. Risks and open decisions (to be validated by the owner)

RISKS.

- (X1) CROSS-RANK TAG REDUCTION. A union computed locally then reduced by
  `all_reduce_or_inplace` must be done on the COMPLETE tag grid (size `pdom.num_cells()`),
  not per fab. If the coarse is replicated (`replicated_coarse_ == true`), the reduction is
  the identity and must be AVOIDED (otherwise useless MPI cost). De-risk: copy the exact guard of
  `amr_regrid_finest:59-60` (`if (pk == 0 && !coarse_replicated) all_reduce_or_inplace(...)`),
  hoisted onto `tags_union`.

- (X2) `fb` CONSISTENCY PER RANK. If two ranks start from different tag grids,
  `berger_rigoutsos` (deterministic but data-dependent) produces different `fb` -> incompatible
  dmaps -> MPI desynchronized (deadlock or silent corruption). De-risk: the reduction
  (R4) BEFORE the clustering is the NECESSARY and SUFFICIENT condition; test (T4) np=1/2/4
  bit-identical as a permanent guard.

- (X3) MASS NOT CONSERVED BY NON-CONSERVATIVE INTERP. The fine carry-over is exact; the interp from the
  parent must be CONSERVATIVE (the average of the reconstructed children equals the parent). The current
  path of `amr_regrid_finest` does a piecewise-constant injection (each child = parent),
  which CONSERVES the mass in the integral sense (4 children x parent value x dV_fine = parent value x
  dV_parent). De-risk: test (T2) per block; if a higher-order interp is introduced
  later, it will have to stay conservative (bounded slope limiter).

- (X4) INHERITED GHOST TOO NARROW. If the ghost width of the new patch does not inherit the
  `n_grow()` of the replaced level, an order-2 scheme reads out of bounds. De-risk: `ngf =
  (*blocks_[b].levels)[fk].U.n_grow()` PER BLOCK (a Minmod block and a VanLeer block can differ;
  the shared aux takes the max width). Verification (V2) + test (T3).

- (X5) COVERAGE DOUBLE-COUNTING AFTER REGRID. The re-`solve_fields()` (R7) triggers the cascade
  `mf_average_down_mb` which restores the covered coarse cells. If we omit this re-solve, the
  mass diagnostic (sum of the coarse alone) would count a ghost coarse value under the
  new patch. De-risk: (R8) explicit; test (T2) which sums the mass AFTER regrid AND re-solve.

OPEN DECISIONS (owner signature required).

- (D1) UNION CRITERION: COMMON or PER BLOCK in v1? The design assumes a UNION criterion (a single
  layout). It remains to settle whether each block provides its own predicate `crit_b` (then OR of the tags)
  or whether a single criterion is applied to each field. The structure (R1)-(R3) supports both;
  the difference is the criterion registration API. RECOMMENDATION: predicate per block from v1
  (the tag union is then natural), separate phi criterion.

- (D2) CADENCE: regrid BEFORE or AFTER the `step(dt)` of the macro-step? The mono-block regrids at the START of
  `h.step` (`amr_dsl_block.hpp:104`, before the advance). For coherence, the multi-block should
  do the same (regrid then step). To confirm (impacts the phase of the `macro_step_` counter).

- (D3) STRIDE / SUBSTEPS INTERACTION. Is a block HELD by its stride tagged and re-gridded at the
  regrid macro-step even if it does not advance? RECOMMENDATION: YES (its frozen state is present
  everywhere and contributes to the Poisson; not re-gridding it would leave it on the old layout and
  break `same_layout_or_throw`). To validate: the regrid sits OUTSIDE the substep loops and
  the stride windows (macro-step granularity only).

- (D4) PHI TAGS: on phi or on grad phi? The physical criterion of the diocotron follows the gradient of the
  potential (ring edge). RECOMMENDATION: tag on |grad phi| (components 1,2 of the aux), not
  on phi (component 0). To confirm according to the target observable.

- (D5) MULTI-LEVEL: v1 stays at 2 levels (coarse + 1 fine), like `amr_regrid_finest`.
  Confirm that > 2 levels stays OUT of Phase 2 scope (limit inherited from the mono-block, cf. section 2).


## 9. Code references (all verified at this head)

- `include/adc/coupling/system/amr_system_coupler.hpp`: compile-time multi-block AMR engine (FROZEN
  hierarchy, NO regrid); `AmrHierarchyLayout`, `detail::same_layout_or_throw` (layout guard).
- `include/adc/runtime/amr/amr_runtime.hpp`: RUNTIME multi-block engine (type-erased registry by name,
  shared aux, summed Poisson, coupled sources, multirate); NO regrid (target of this design).
- `include/adc/coupling/amr/amr_coupler_mp.hpp`: MONO-BLOCK AMR coupler + `regrid` (`:321-325`,
  delegates to `amr_regrid_finest`); mono-block path UNTOUCHED.
- `include/adc/coupling/amr/amr_regrid_coupler.hpp`: `amr_regrid_finest` (Berger-Rigoutsos, finest
  level); brick to SPLIT into "layout computation" + "re-grid a field on a given layout".
- `include/adc/amr/tagging/tag_box.hpp`: `TagBox` (dense grid of tags 0/1; union = cell-by-cell OR).
- `include/adc/amr/regridding/regrid.hpp`: `tag_cells`, `grow_tags`, `regrid_level` (generic bricks).
- `include/adc/amr/tagging/cluster.hpp`: `berger_rigoutsos`, `ClusterParams` (geometric clustering).
- `include/adc/amr/hierarchy/amr_hierarchy.hpp`: `AmrHierarchy` (level container; note: "the future conservative
  multi-block AMR will have to share a common hierarchy", l. 31-32).
- `include/adc/runtime/amr_system.hpp` + `python/bindings/amr/amr_system.cpp`: RUNTIME facade (`regrid_every`;
  multi-block + `regrid_every > 0` REFUSAL at `amr_system.cpp:246-251`, lifted by this design).
- `include/adc/runtime/builders/compiled/amr_dsl_block.hpp`: mono-block regrid wiring (`:101-104`), shared 2-level
  FROZEN layout (`make_shared_amr_layout`) and per-block allocation (`build_amr_block`).
- `include/adc/parallel/comm.hpp`: `all_reduce_or_inplace`, `n_ranks`, `my_rank` (MPI collectives).
- `docs/AMR_MULTIBLOCK_DESIGN.md`: Phase 1 capstone (multi-block engine, layout guard, Phase 2 /
  Phase 3 boundary); this document is its Phase 2.
