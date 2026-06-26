# ADR-0001: Genericity contracts (2D core, explicit model spec, named aux, roles, regrid)

- Status: Proposed (decision input; not yet implemented)
- Date: 2026-06-19
- Scope: ADC-294, ADC-290, ADC-291, ADC-292, ADC-296
- Verified against: `wolf75222/adc_cpp` master `288ddb9` (audit baseline `0ded184`).
  Re-confirmed at `fb11d41`: ADC-335 / ADC-342 split `system.cpp` / `amr_system.cpp` into
  parallel translation units and ADC-337 added `POPS_COLD_FN` annotations, which shift line
  numbers but change none of the genericity semantics below. References are anchored on stable
  symbols; the line numbers are as of `288ddb9`.

## Context

The 2026-06-16 genericity audit raised eight core findings. Five fall to the architecture /
generalisation track: the 2D dimensionality of the core, silent physical defaults in `ModelSpec`,
the bounded named-aux channel, the closed role vocabulary with silent canonical fallbacks, and the
component-0-only AMR regrid predicate.

Each finding was re-checked against the current master, file by file. The audit is substantially
still valid, but real progress has landed since the baseline and must not be redone:

- `pops.capabilities()` now exists as a single Python matrix (`capabilities` in `python/pops/__init__.py`).
- Named aux phase 1 (ADC-70) is complete end to end on cartesian `System` (X-macro `POPS_AUX_FIELDS`,
  `extra[kAuxMaxExtra=4]`, JIT / native marshaling); the code itself declares the remaining
  AMR / polar / halo work via a `named_followups` key.
- Role infrastructure landed: `VariableSet.roles`, `bind_variable_roles`, CSV roles across the `.so`
  ABI, and fail-loud role resolution on the explicit / user-targeted paths (`add_block`,
  `add_coupled_source`, Schur, positivity) per #181.
- The AMR engine seam is already generic: `TagPredicate = std::function<...>`, per-block
  `set_block_tag_predicate`, separate `set_phi_tag_predicate`, union in `regrid`.

The common theme: the generic seam usually already exists; what remains is to remove an implicit
default or declare a limit, not to rewrite an abstraction. Hence every decision is "Option A"
(declare / tighten the contract); the deeper "Option B" is deferred to a dedicated future issue.

## Decision 1 (ADC-294): the core is officially 2D; do NOT start BoxND now

2D is load-bearing, not cosmetic. It is baked into the data layout (`Fab2D operator()(i,j,c)`,
strides `nx_tot*ny_tot`), the paired hand-written `FaceFluxX` / `FaceFluxY` kernels, the 2-component
momentum (`euler.hpp` `{rho,rho_u,rho_v,E}`), the 5-point Poisson, and ~392 `Box2D` / ~152
`Geometry` sites. No ND scaffolding exists (zero hits for `BoxND` / `GeometryND` / `IndexSpace` /
`template<int Dim>`). `box2d.hpp` still literally defers the Dim-template "for later". The polar
path is a second geometry at the same dimension (`(r,theta)` is a 2-index `Box2D`), already shipped;
it does not make the core ND.

Decision: declare "2D core" an explicit, introspectable invariant (Option A). Add a structured
dimension key to `capabilities()` and a paragraph in the live limitations doc; cross-link the
existing `box2d.hpp` comment. A real ND trajectory (Option B: `IndexSpace<Dim>` / `BoxND<Dim>` with
2D adapters and bit-identity gates) is a milestone-sized bet with genuine ABI / bit-identity risk
and no current scenario demanding it. File it as a future milestone, not this ticket.

Boundary vs ADC-327: ADC-327 (EmbeddedBoundary / LevelSetDomain) is about domain shape inside the
2D plane (masking cells of a fixed `(i,j)` grid); it adds no third index. ADC-294 = number of axes;
ADC-327 = boundary geometry at fixed dimension.

## Decision 2 (ADC-290): make ModelSpec explicit; kill silent physical defaults

`ModelSpec` hard-codes physics-selecting string defaults `transport="compressible"`,
`elliptic="charge"`. A default-constructed spec is silently dispatched as compressible Euler +
charge-density Poisson. The dispatchers already throw on unknown tags but never on unset, and there
is no validation. The numeric params (`gamma=1.4`, `cs2=0.5`, `B0`, ...) are harmless: each is read
only after a tag has already chosen its brick.

Decision (Option A): replace the three physics-selecting defaults with an `unset` sentinel and add an
explicit completeness check in `dispatch_model` that fails loudly (mirroring the existing
`throw_registry_dispatch_mismatch` in `dispatch_tags.hpp`). Keep the numeric defaults. Keep the
historical shortcut only at the Python edge: `pops.Model(...)` already requires all four bricks and is
explicit-or-fail. In-tree blast radius is near zero (every C++ test sets the three tags;
`pops.Model` always overwrites them).

Boundary vs ADC-331: ADC-290 makes "no model chosen" an error; ADC-331 later makes "which models can
be chosen" an open registry. Doing ADC-290 first gives the registry a clean explicit-or-fail
contract. The per-capability sub-config restructure (Option B) leans into ADC-331 territory; defer.

## Decision 3 (ADC-291): finish the named-aux channel incrementally; keep pops::Aux fixed

Phase 1 is fully landed; the remaining scope is exactly the code's own `named_followups`: extend
named aux to AMR and polar, and lock the compile-time limits. `pops::Aux` is a fixed, device-clean POD
by deliberate design; `PhysicalModel` still requires `M::Aux == pops::Aux`.

Decision (Option A), two slices: (1) lock the limits - tests pinning `kAuxMaxExtra=4` /
`AUX_NAMED_MAX` parity and the implicit 1-ghost aux halo, surfaced in capabilities + limitations;
(2) carry the per-block name-to-component table through `AmrSystem` and re-impose named fields across
regrid (the aux MultiFab is rebuilt on regrid, so an un-re-filled named field would silently zero,
the #51 class of bug); add a polar path that at least validates / documents the `grad_x == grad_r`,
`grad_y == grad_theta` slot reuse in `ExBVelocityPolar`.

Defer Option B (per-model `Model::Aux` / `AuxLayout`): it breaks the `PhysicalModel` concept, the
compiled-block flat ABI, and `abi_key`. It is explicitly sequenced after the ADC-328
`spatial_operator.hpp` split, and is its own later issue.

## Decision 4 (ADC-292): string user-roles + tighten the residual silent fallbacks

The fail-loud half is already 80% delivered (#181). The residual is narrow: (a) `VariableRole` is a
closed 12-member enum with no string-keyed user-role layer (`Custom` = unknown, not a user label;
`index_of` matches the first occurrence, so two `Custom` components are ambiguous); (b) the legacy
named couplings still silently assume the canonical layout - `role_index(vs, role, fallback)` returns
the canonical fallback in `add_collision`, `add_thermal_exchange`, and the ionization triple.

Decision (Option A): (1) add a string-keyed user-role layer parallel to the enum (roles already
serialize as CSV across the `.so` ABI, so no flat-ABI break); (2) make the named-coupling fallbacks
fail loud when a roles-bearing block lacks a declared critical role, keeping the canonical fallback
only for genuinely roleless legacy blocks behind an explicit opt-in. All shipped native models
declare canonical roles, so they stay bit-identical.

Defer Option B (enum to string everywhere): a broad public-type rewrite (~15 call sites) that this
issue does not justify and that the CLAUDE.md "avoid broad type rewrites" rule discourages.

Boundary: ADC-292 changes how a chosen brick resolves its component indices; ADC-331 changes which
brick is chosen. `bind_variable_roles` runs after dispatch, so the two are orthogonal.

## Decision 5 (ADC-296): role/name-configurable regrid variable; keep comp-0 default

The engine seam is already generic (`TagPredicate = std::function`, per-block predicates, union).
Only the facade pins component 0: `build_multi` installs `a(i,j,0) > thr`, and `set_refinement(double)`
exposes only a threshold. The recent multi-block-union + optional `phi` tag (`set_phi_tag_predicate`,
aux comps 1,2) added a second separate criterion, not a configurable selector.

Decision (Option A): add an optional variable selector to `set_refinement` (new overload / kwarg with
default = component 0, preserving the bit-identical `1e30` no-op). Resolve it per block to a component
via the existing `resolve_implicit_components` helper (already supports name + role with loud errors)
and emit `a(i,j,c) > thr`. The engine needs no change. Declare the single-block `AmrCouplerMP` path
(`amr_dsl_block.hpp`) and the compiled `.so` flat-ABI loader as comp-0-only (out of scope), matching
how `set_phi_refinement` is already documented as multi-block-only. Add a test that refines on a
non-zero component / named role.

This is the smallest, most surgical of the five. Option B (full composable-criteria mechanism that
subsumes density + phi) is disproportionate and would touch the flat ABI; defer.

## Shared mechanism: capabilities() (coordinate with ADC-297)

All five declarations land in the one `pops.capabilities()` dict, which ADC-297 owns. Convention:
supported variants stay lists under the existing facade keys; hard limits become explicit structured
scalars, not prose (for example `dimension: 2`, `ref_ratio: 2`, an AMR level bound, a uniform
`mpi` / `single_rank` flag) rather than buried in `note` strings. ADC-297 should add a snapshot test
(`python/tests/test_capabilities.py`) so any future limit declaration forces a reviewed diff; today
there is no such test. This ADR does not implement ADC-297; each of the five issues appends its one
declaration through that surface.

## Recommended implementation order

1. ADC-290 - establishes the explicit-or-fail, no-silent-physics contract at the model boundary;
   foundational; near-zero in-tree blast radius; unblocks ADC-331 cleanly. (Can run in parallel with
   ADC-296.)
2. ADC-296 - independent quick win; engine seam already generic; reuses `resolve_implicit_components`;
   bit-identical default.
3. ADC-292 - tighten the residual silent fallbacks (convention set by #181) and add the string
   user-role layer; generalises the role vocabulary that ADC-296's role-selector and ADC-291's named
   aux both lean on.
4. ADC-291 - largest; extend named aux to AMR / polar with regrid-safe re-imposition; sequenced after
   ADC-296 because both touch `amr_system.cpp` / `amr_runtime.hpp`.
- ADC-294 - orthogonal; a small additive PR any time, best bundled with the ADC-297 capabilities
   consolidation so the `dimension` key uses the agreed schema.

## API-compatibility risks

- ADC-290: raw `ModelSpec()` default-construct behavior changes. In-tree safe; the only exposure is an
  out-of-tree consumer (adc_cases) that builds a bare `ModelSpec` and relies on compressible+charge.
  Mitigation: keep shortcuts at `pops.Model(...)`; CHANGELOG `Changed`; pre-1.0.
- ADC-291: Option A non-breaking (POD layout + concept unchanged). Option B breaks the concept, the
  compiled-block ABI, and `abi_key` (deferred).
- ADC-292: tightening fallbacks breaks only roleless-but-canonical third-party dynamic blocks (shipped
  native models declare roles, so they are unaffected). String user-role layer is additive, no
  flat-ABI break. Option B (enum to string) is broad public-type churn (deferred).
- ADC-296: add the selector via a new overload / optional kwarg with default = comp 0 to keep
  `set_refinement(double)` and the pybind signature backward-compatible.
- ADC-294: Option A purely additive (a capabilities key + docs). Zero ABI risk.

## Out of scope / owned elsewhere

ADC-293 (HLLC/Roe capabilities), ADC-297 (capabilities matrix + tests), ADC-298 (GENERICITY doc sync),
ADC-331 (builtin model registry), ADC-327 (EmbeddedBoundary), ADC-328 (`spatial_operator.hpp` split).
This ADR coordinates with them but does not modify them.
