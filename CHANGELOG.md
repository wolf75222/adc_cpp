# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
[SemVer](https://semver.org/lang/en/) (0.y.z while the public API still moves).

## Versioning policy

- **Single source**: `project(VERSION x.y.z)` in `CMakeLists.txt`. Everything derives from it:
  `pops.__version__` (bakes `POPS_VERSION` into `_pops`), the pip wheel (regex in `pyproject.toml`),
  `adcConfigVersion.cmake` (install/export). NEVER duplicate the number elsewhere; the docs build
  derives it too (`scripts/build_docs.sh` injects `PROJECT_NUMBER`, `docs/sphinx/conf.py` reads
  `project(VERSION)`), so nothing is bumped by hand outside `CMakeLists.txt`.
- **Bump**: PATCH = fixes with no API change; MINOR = backward-compatible API/brick additions;
  MAJOR (post-1.0) = API or ABI break of the DSL production path.
- **Tag**: set `git tag vX.Y.Z` on master when the bumping PR merges, then `git push --tags`.
- On every notable PR: one line in `[Unreleased]` below; on a bump, the section becomes
  `[x.y.z] - YYYY-MM-DD`.

## [Unreleased]

### Added
- **Spec 5 sec.14.2.4 typed `P.rhs(terms=[...])` RHS composition** (epic ADC-479, criterion 27): `pops.time.Program.rhs` now ADDITIONALLY accepts a single typed `terms=` list -- a `pops.numerics.terms.Flux` plus the source terms to fold in (`P.rhs(U, fields=f, terms=[Flux(), electric])`) -- as a pure front door over the legacy `flux=`/`sources=` path. A `Flux()` in the list sets `flux=True` (its absence sets `flux=False`); every source term contributes its NAME to `sources=` in list order, accepting a `pops.numerics.terms.SourceTerm` / `LocalTerm` descriptor (its name must be set), an `pops.model.OperatorHandle` from `m.source_term`, or a plain source name string. The lowering is unchanged, so a `terms=` call builds the BYTE-IDENTICAL IR (same `_ir_hash`) as the equivalent `flux=`/`sources=` call and the legacy path is untouched. `terms=` is MUTUALLY EXCLUSIVE with `flux=`/`sources=`/`fluxes=` (passing both is a clear `ValueError` naming the conflict); a non-term object in the list (e.g. a bare bool -- `Flux()` is a term, not a bool) is a clear `TypeError`. `pops.numerics.terms` is imported function-locally so `pops.time` stays codegen-free and the import graph acyclic. New `python/tests/test_rhs_terms.py`.
- **Spec 5 sec.14.2.3 typed operator handles** (ADC-464, epic ADC-479): the user-facing operator declarers `m.rate_operator` / `m.source_term` / `m.linear_source` now RETURN a lightweight, inert `pops.model.OperatorHandle` (carrying the operator `name` and `kind`) instead of `None`, and `P.call` accepts EITHER the string operator name OR that handle, coercing a handle to its `.name` through the IDENTICAL registry lookup + lowering, so `P.call(handle, ...)` builds the BYTE-IDENTICAL IR (same `_ir_hash`) as `P.call(name, ...)` and the legacy string path is unchanged; a non-str / non-handle selector is a clear `TypeError`. The handle holds no numerics, IR or Program reference, keeps `pops.time` codegen-free and the import graph acyclic, and is covered by `python/tests/test_operator_handles.py`.
- **Spec 5 sec.12.1 compiled/sim inspection completeness** (epic ADC-479, criterion #15): a `pops.compile_problem(...)` handle and a configured `System` / `AmrSystem` can now be fully inspected WITHOUT binding, compiling or allocating, building on the ADC-509 introspection and the ADC-463 validation. New `compiled.inspect()` returns a printable `pops.codegen.CompiledReport` aggregating the carried metadata into the Spec 5 sec.12.1 `print(compiled)` shape (name, backend, platform, layout, committed blocks + state / components, elliptic / Krylov fields + solver-to-supply, program op count + commits, the REQUIRED runtime inputs from `arguments()`, the on-disk artifacts so_path / abi_key / cache_key, and the `compiled, waiting for pops.bind(...)` status), with `to_dict()` / `to_json(path)`. New `compiled.requirements()` returns a `RequirementsReport` of the COMPILE-TIME constraints DISTINCT from the runtime bind inputs -- the model capabilities the lowered flux relies on (`wave_speeds` / `hllc_star_state` / `roe_dissipation`, read from the carried model flags), the layout / backend / ABI constraints, and an honest `unknown` list for pieces genuinely not in today's metadata (the spatial scheme is a bind input; the reconstruction stencil width is unrecorded) rather than a fabricated value. New `compiled.inspect_capabilities()` delegates to the top-level `pops.inspect_capabilities()` machinery scoped to the descriptor categories a compiled Program selects at bind. New `compiled.dump_ir(path=None)` writes the serialized Program IR JSON (the SAME `_serialize` the IR hash digests), `compiled.dump_cpp(target)` writes the generated C++ source by REUSING the existing `Program.emit_cpp_program` codegen, and `compiled.dump_schedule(path=None)` writes the block commit order; each raises a clear error naming what is missing if the handle carries no Program (never faking a file). New `System.explain_bind(compiled)` / `AmrSystem.explain_bind(compiled)` return a printable `BindReport` computing provided-vs-still-required per group (instances / params / aux / solvers) by reusing the ADC-463 `collect_missing_arguments`. All inert host-side metadata reads (no `_pops`, acyclic import graph -- `pops.time` / runtime imported lazily) except `dump_cpp`, which reuses the existing emit. New `python/tests/test_inspection_completeness.py`; the dump on a real `.so` stays Kokkos-gated.
- **Spec 5 sec.10 strict early install validation + `AmrSystem.install` parity** (ADC-463, epic ADC-479): `System.install` now reads the compiled artifact's DECLARED bind inputs (`compiled.arguments()`, ADC-509) and rejects, BEFORE any native call, an install that fails to supply a REQUIRED argument the artifact declares (a committed instance, a `kind="runtime"` param, a required aux field, a required solver), with one clear actionable error naming exactly what is missing and the matching `install` keyword to supply it (e.g. `runtime param 'cs2'; pass install(params={'cs2': <value>})`); multiple misses aggregate into one message. The check is inert (it reads `arguments()` metadata and compares dicts -- no compile, bind or allocation) and conservative: it enforces only the `required` flag, so an optional input (a const param, the default-Poisson solver) is never demanded and a previously valid install passes through unchanged; a handle whose `arguments()` is unavailable or raises is skipped rather than rejected (a missing check never breaks a working install), and a NATIVE install (`compiled=None`) carries no declared arguments and is skipped. `AmrSystem` gains an `install(compiled=None, *, instances=, params=, aux=, solvers=, cadence=)` with signature parity to `System.install` that runs the SAME early validation, then wires the NATIVE AMR path (`add_equation` / `set_poisson` / `set_magnetic_field` / `set_aux_field` / `set_density`) and raises a clear `NotImplementedError` for the genuinely un-wired parts (a compiled time Program -- no AMR `install_program` seam; `params=` -- no `set_block_params`; `cadence=` -- no `set_program_cadence`). The shared pure router `collect_missing_arguments` and `validate_install_arguments` live in `pops.runtime._system_unified_install`. New `python/tests/test_install_validation.py` (pure-Python; the Kokkos-gated end-to-end is covered by `test_unified_install.py` / `test_install_requirement_validation.py`).
- **Spec 5 typed-surface reference docs** (epic ADC-479, section 16): `docs/sphinx/reference/spec5-packages.md` gains concise reference sections for the now-merged typed surfaces: the typed compile backends `pops.codegen.Production` / `AOT` / `JIT` and the execution platforms `pops.runtime.platforms.KokkosSerial` / `KokkosOpenMP` / `KokkosCuda` / `KokkosHIP` / `MPI` (sec.8.15), the typed profiling front door `pops.Profile` (Basic / Advanced) + `pops.PerformanceSummary` + `sim.profile(...)` (off by default, `POPS_PROFILE`; sec.12.5), the array-free `print()` summary on `pops.Spatial` / `pops.time.Program` / `pops.System` / `pops.AmrSystem` (sec.12.1), and a cross-reference to the typed native-brick constructors documented in `native-bricks.md` (sec.14.2.5). Every code block is verified runnable against the current typed API. Docmap depends_on / reviewed updated.
- **Spec 5 typed `set_poisson(bc=)` boundary** (ADC-464, epic ADC-479, section 14.2.6): `System.set_poisson` (and `add_elliptic_model`, which forwards `bc=`) now accept the typed native boundary bricks `pops.Dirichlet()` / `pops.Neumann()` / `pops.Periodic()` directly for `bc=`, in addition to the legacy string. A new transparent `_lower_bc` coercion lowers a typed boundary to its token via `.bc` and passes any STRING (including `"auto"`) straight through to the native `set_poisson`, which validates an unknown token with its own error exactly as before -- the coercion never adds a stricter string rejection of its own. A typed `bc=pops.Dirichlet()` produces a byte-identical Poisson potential to `bc="dirichlet"`. New `python/tests/test_set_poisson_bc_typed.py`. Additive, inert; no numeric behavior changed.
- **Spec 5 typed native-brick constructors** (ADC-504, epic ADC-479, section 14.2.5): the native bricks can now be NAMED by typed constructors instead of magic `kind=` / `bc=` strings, additively (the string path keeps working, no consumer breaks). New `pops.FluidState.compressible(gamma=1.4)` / `pops.FluidState.isothermal(cs2=0.5, vacuum_floor=0.0)` classmethods build the SAME inert state object as the `kind=` form and produce a bit-identical `ModelSpec` through `pops.Model(...)`. New `pops.ElectrostaticLorentzSchur(theta=0.5, alpha=1.0, ...)` is a `pops.CondensedSchur` subclass that pins `kind="electrostatic_lorentz"` (the currently unique kind) and carries every other argument identically, so `pops.Split` / `pops.Strang` / `add_equation` accept it unchanged; passing `kind=` to it is a `TypeError`. New native elliptic boundary bricks `pops.Dirichlet()` / `pops.Neumann()` / `pops.Periodic()` lower (via `.bc` / `.lower()`) to the `set_poisson(bc=...)` tokens `"dirichlet"` / `"neumann"` / `"periodic"`; they are inert and distinct from the `pops.fields.bcs` field-VALUE descriptors and the `pops.mesh.boundaries` topology descriptors (this brick is the native Poisson-solver boundary token). New module `python/pops/runtime/_bricks_typed.py`; new `python/tests/test_native_brick_typed_ctors.py`. Inert and additive; no `_pops` / numpy / runtime / codegen compute.
- **Spec 5 sec.12.1 print-readability `__str__` for the headline objects** (ADC-505, epic ADC-479, criteria #40-41): `print(obj)` now gives a short, deterministic, array-free summary instead of the default `<...object at 0x...>` that leaked a memory address. New `__str__` on `pops.Spatial` (`Spatial(limiter=minmod, flux=rusanov, recon=conservative)`, appending `positivity_floor` / `wave_speed_cache` only when set), on the authoring `pops.time.Program` (name + op count + committed block names, never the full SSA IR), and on the runtime `pops.System` / `pops.AmrSystem` (the installed block-name registry, never a Fab dump). The inert typed descriptors already print well: the mesh layouts `Uniform` / `AMR`, the elliptic solver `GeometricMG`, the Krylov solvers, `codegen.Optimization`, `params.RuntimeParam`, `output.OutputPolicy`, and `diagnostics.Norm` / `Integral` inherit the inert `Descriptor.__str__`, and `fields.FieldProblem` / `PoissonProblem` carry their own. New `tests/architecture/test_print_readability.py` builds each inert headline object (now including `time.Program`) and asserts `str(obj)` is short (< 800 chars), deterministic, free of any raw `array(` / `ndarray` dump, and never the default object repr. `__repr__` is unchanged for debug; the change is additive and inert.
- **Spec 5 diagnostics factory/typed-class scheme dedup** (ADC-506, epic ADC-479, sections 5.13 / 10): `pops.diagnostics` no longer spells the native reduction scheme label twice. The legacy `diagnostics.norm` / `diagnostics.integral` factories and the `invariants.conservation_check` factory now SOURCE their `scheme` from the typed measure classes (`pops.diagnostics.measures.Norm.scheme` / `Integral.scheme` / `ConservationCheck.scheme`), so each shared scheme string lives in exactly one place (the typed class is the canonical authoring form). The factories keep their historical `BrickDescriptor` return (`category="diagnostic"` / `category="invariant"`) so the architecture consumers (`tests/architecture/test_public_imports.py`, `test_spec5_typed_api_negatives.py`, `test_spec5_physics_codegen_free.py`) are untouched; the reductions with no typed counterpart (`mass` / `momentum` / `energy` / `invariant_error` / `residual`, and the typed-only `MinMax`) keep their own self-named scheme. `python/tests/test_diagnostics_typed.py` adds the dedup assertions (factory and typed class lower to one inert scheme; neither computes). No `pops.moments` factory duplicated a typed descriptor, so none changed. Inert and additive.
- **Spec 5 typed profiling: `pops.Profile` + `pops.PerformanceSummary` + `System.profile(...)`** (ADC-507, epic ADC-479, section 12.5 / criteria #41-44): a typed front door over the native `enable_profiling` / `profile_report` profiler. New `python/pops/runtime/profile.py` adds the inert `Profile` level (`Profile.Basic()` / `Profile.Advanced()`, NOT `profile="advanced"`) and the printable `PerformanceSummary` wrapper, which parses the native report into a structured table and exposes `to_dict()` / `to_json(path)` plus typed views `by_program_node()` / `by_native_brick()` / `by_solver()` / `by_memory()` that read the native scopes / counters and DECLARE an unavailable measure honestly (the per-brick and Kokkos-gated scratch-memory counters) rather than faking a zero. `System.profile(Profile.Basic())` is a context manager that resets and enables the native profiler on entry, disables it on exit, and exposes `prof.summary()`; with no argument the level comes from `POPS_PROFILE` (unset / `off` -> Basic, `advanced` -> Advanced). The off-by-default contract holds: a plain run never enables profiling. `Profile` / `PerformanceSummary` are exported from `pops.runtime.profile`, `pops.runtime`, and the `pops` top level. The wrapper imports neither `_pops` nor numpy (the native extension is reached only through the System). New `python/tests/test_profile_api.py`; the heavy native counters are Kokkos/ROMEO. Inert and additive.
- **Spec 5 typed compile backend + execution platform descriptors** (epic ADC-479, sec.8.15 / criterion #22): the compile backend and the execution platform are now typed objects, not bare strings. New `pops.codegen.backends` adds the inert `Production` / `AOT` / `JIT` descriptors (category `backend`), each lowering to the legacy backend string (`"production"` / `"aot"` / `"prototype"`) the compile drivers already key on and reporting the honest cpu/mpi/amr/gpu capabilities from the native backend table; an optional `platform=` slot records the target device (a bare string is refused). New `pops.runtime.platforms` adds the inert `KokkosSerial` / `KokkosOpenMP` / `KokkosCuda` / `KokkosHIP` / `MPI` descriptors (category `platform`) declaring host/gpu/mpi capabilities and answering `available()` with an EXPLAINABLE `Availability` that names the missing build flag against the compiled `_pops` flags (`__has_kokkos__` / `__has_mpi__`) -- a Serial-only build reports the OpenMP device as a Kokkos-build miss, an MPI-less build reports the MPI transport as missing, and a GPU device on a non-GPU build is never a false "yes". The coercion is ADDITIVE: `compile_problem` / `compile_model` accept BOTH a backend string (the existing consumers keep working) AND a typed `Production()` via the new `lower_backend` helper, lowering the descriptor before the production-only guard so a typed `AOT()` hits the same message the string `"aot"` produces; `Production(platform=KokkosOpenMP())` records the platform without changing the lowered backend string. Inert -- nothing compiles or runs in Python; numeric behavior is unchanged. New `python/tests/test_backend_platform.py`. The `pops.compile(backend=Production())` integration is a follow-up when the `pops.compile` PR lands.
- **Spec 5 compiled-artifact introspection: `CompiledProblem.arguments()` / `estimate_memory()` + metadata attributes** (ADC-509, epic ADC-479, sections 12.2 / 12.3 / 12.4, criteria #44-49): a `pops.compile_problem(...)` handle can now be inspected WITHOUT binding, compiling or allocating. New `compiled.arguments()` returns an inert `pops.codegen.Arguments` listing the runtime inputs the artifact expects at `sim.install(...)` -- instances (state space / component count / required), params (type / kind / required, only `runtime` settable at bind), aux (layout / required), the elliptic / Krylov solvers the Program performs (problem / solver-to-supply), recorded outputs, and the runtime layout (layout / requires_mpi / ghost_depth) -- mirroring `install`'s keyword groups and sourced from the carried Program (committed blocks, field solves) and physical model; it is printable and serialises via `to_dict()` / `to_json(path)`, and is distinct from the compile-constraint introspection. New `compiled.estimate_memory(mesh=, platform=None, layout=None)` returns a `pops.codegen.MemoryEstimate`: state / field-output / aux / RHS-scratch / state-scratch / scalar-field / Krylov / multigrid / halo / MPI-buffer / AMR-patch byte budgets computed as a FORMULA over the mesh shape and the Program's grid-relative static cost (`Program.estimate`) + the model component counts -- it allocates no MultiFab, every assumption is inspectable (`assumptions`), it is conservative (`conservative=True`), slices via `by_block()` / `by_solver()` / `by_scratch()`, serialises via `to_dict()` / `to_json(path)`, accepts `platform="mpi"` for the halo-exchange buffer and `layout=AMR(...)` for a conservative full-refinement hierarchy estimate; a non-2D / non-positive mesh is rejected. `CompiledProblem` now also exposes the compiled-artifact metadata `codegen_dir` / `problem_hash` / `abi_key` / `cache_key` / `compile_command` (redacted: ephemeral temp source replaced, secret-looking tokens masked) / `generated_sources`, populated by `compile_problem` (compile_command / generated_sources are `None` / empty on a cache hit or a non-compiled handle -- a documented absence, never fabricated). All inert host-side metadata reads (no `_pops`, no codegen layering break -- `pops.mesh` is imported lazily); the real `.so` compile that populates `compile_command` stays Kokkos-gated. New `python/tests/test_compiled_introspection.py`.
- **Spec 5 typed disc-domain / cut-cell geometry wiring** (ADC-510, epic ADC-479, section 8.16): the disc transport domain and the Poisson conducting wall are now chosen with typed objects, not the `wall="circle"` / `mode="staircase"` strings. New `pops.mesh.geometry.DiscDomain(center=(cx, cy), radius=R, mode=CutCell())` carries the circle plus a typed `pops.mesh.masks` transport mode and lowers to the native `(cx, cy, R, mode_token)` `set_disc_domain` call; new `pops.mesh.geometry.NoWall` (lowers to `wall="none"`) joins `Disc` (which now lowers to `wall="circle"` + its radius) as a typed Poisson wall. The `pops.mesh.masks` masks (`NoMask` / `Staircase` / `CutCell`) gain a faithful `lower()` to the native disc-transport tokens (`none` / `staircase` / `cutcell`) plus a shared `lower_disc_mode` coercion (a string passes through, a typed mask lowers, any other type is a clear `TypeError`). The runtime accepts BOTH forms additively: `System.set_disc_domain` takes either the legacy `(cx, cy, R, mode=)` scalars or a single typed `DiscDomain`, `set_geometry_mode` takes a string or a typed mask, and the new `System.set_poisson` wrapper (and `add_elliptic_model` / the unified install through it) accepts `wall=` as the legacy string or a typed `Disc` / `NoWall`; every typed path lowers to the byte-identical native call (a circle-wall Poisson solve and the disc mask are bit-identical to the string form). All descriptors stay inert (no module-scope `_pops`; the mesh layer imports nothing else). New `python/tests/test_typed_disc_domain.py`. The native cut-cell / staircase transport physics is Kokkos-gated and unchanged; only the lowering and acceptance are added here.
- **Spec 5 solver catalog package `pops.solvers`** (ADC-500, epic ADC-479, section 5.7 / criterion #4 / sec.13.11.1): a new top-level central package that homes the linear / nonlinear / Schur / elliptic solver and preconditioner catalog as inert typed descriptors, moved out of the transitional `pops.lib.solvers`. `pops.solvers.krylov` (CG / BiCGStab / GMRES / Richardson native descriptors), `pops.solvers.nonlinear` (Newton / FixedPoint, planned: no native solver type yet), `pops.solvers.schur` (`Schur` / `CondensedSchur` naming `pops::SchurCondensationOperator`, distinct from the `pops.time` CondensedSchur splitting policy), `pops.solvers.elliptic` (the RICH `GeometricMG` configured by typed sub-descriptors -- a smoother `Chebyshev(degree=)` / `RedBlackGaussSeidel()`, a coarse solver `DirectSmallGrid(threshold=)`, a convergence tolerance `Relative(rel=, floor=AbsoluteFloor(...))` / `Absolute(...)`, and a `max_cycles` V-cycle cap -- declaring its uniform / amr / mpi / gpu / variable_epsilon capabilities; plus the planned `FFT` spectral Poisson solver whose `available()` explains its missing native symbol and points at `GeometricMG`), `pops.solvers.preconditioners` (Identity / Jacobi / BlockJacobi / GeometricMG / User), and the typed `pops.solvers.options` / `pops.solvers.tolerances` / `pops.solvers.requirements` sub-descriptor vocabulary. A bare string or number for any typed slot is rejected with an actionable TypeError (section 7). All build on `pops.descriptors`; nothing imports `_pops` / `numpy` / `pops.runtime` / `pops.codegen` at module scope (the C++ solvers execute), and the architecture gates add `solvers` to the import-graph / Spec-5-structure suites (the DAG stays acyclic). `pops.lib.solvers` becomes a thin re-export shim onto `pops.solvers` (the custom-solver authoring DSL stays there), so the in-flight install path and the legacy names (`pops.lib.solvers.solvers.GMRES()` / `.custom` / `.registered`, `pops.lib.preconditioners`) keep working unchanged; the orphaned `pops/lib/solvers/preconditioners.py` is removed (single source in `pops.solvers.preconditioners`). New `python/tests/test_solvers_descriptors.py`. Inert and additive.
- **Spec 5 field-solve cadence: `FieldProblem.solve(schedule=, policy=)` + `pops.fields.HoldPrevious` / `Recompute`** (ADC-501, epic ADC-479, section 13.11.2): a `pops.fields.FieldProblem` / `PoissonProblem` can now RECORD an inert solve cadence pairing a typed `pops.time.Schedule` (WHEN to solve, e.g. `every(4)` / `when(cond)`) with a typed field-solve policy deciding what happens on a step where the solve is not due. New `python/pops/fields/policies.py` adds the inert `FieldSolvePolicy` base and the `HoldPrevious` (reuse the cached field, so it requires a cacheable output) / `Recompute` (always recompute) descriptors, each declaring its cacheability through `capabilities()` / `requirements()` with a short `repr()`; both are exported from `pops.fields`. `FieldProblem.solve` validates that `schedule` is a typed `pops.time.Schedule` (a bare int or string raises a clear `TypeError`) and that `policy` is a `FieldSolvePolicy` (a string such as `"hold"` raises), then stores the pair as an inert `SolveCadence` surfaced by the problem's `inspect()`. It is authoring metadata only: it does NOT lower into the Program / codegen, and it imports `pops.time` lazily so `pops.fields` keeps its empty import-graph layer. New `python/tests/test_field_solve_cadence.py`. Wiring the cadence into the runtime (the cached-field carry and the residual gate) stays a codegen-adjacent follow-up.
- **Spec 5 sec.5.16 / sec.11 `pops.Problem` assembly + `pops.compile` / `pops.bind` + `PhysicsModel`** (ADC-492, epic ADC-479): a top-level compilable assembly plus its thin lowering, all inert in this PR. New `pops.Problem(layout=Uniform(CartesianMesh()), name=)` is a `pops.descriptors.Descriptor` (category "problem") with chaining setters `.block(name, physics, spatial=)` / `.field(FieldProblem)` / `.param(name, default=, kind=)` / `.aux(name, value=)` / `.output(policy)` / `.time(Program)`, a `.layout` property and a `.amr` handle (raises unless `layout=AMR(...)`, else exposes `.refine(...)`); it answers the full DescriptorProtocol (`options` / `requirements` / `capabilities` / `available` / `validate` / `inspect`) in pure Python, with `validate()` doing the structural checks (layout, at least one block each with physics, each FieldProblem valid, no block/field name collision) and raising a LOUD `NotImplementedError` for the deferred routes (more than one block, a non-Poisson field, a non-empty output policy). `pops.PhysicsModel` is added as an ALIAS of `pops.physics.Model` (the same class object, exported from both `pops` and `pops.physics`; the class `__name__` stays "Model" so existing consumers are unaffected). New `pops.codegen.orchestration` provides `pops.compile(problem, backend="production", time=None, **kw)` -- it validates, derives the target from the LAYOUT (`Uniform` -> "system", `AMR` -> "amr_system"; no user `target=` string), resolves the block physics to the model `compile_problem` wants, and calls the EXISTING `compile_problem` (carrying `_problem` / `_target` on the handle); a missing time scheme raises (NO silent default), and `layout=AMR` / multi-block raise `NotImplementedError`. `pops.bind(compiled, initial_state=, params=, aux=, solvers=, cadence=)` dispatches `System` vs `AmrSystem` from the carried target and calls the unified `sim.install(...)` -- the one install seam -- with no parallel runtime and no new codegen. The real `.so` compile + install + run path is Kokkos-gated and validated on CI / ROMEO; the assembly, the alias, and the dispatch wiring are host-tested here (`python/tests/test_problem_orchestration.py`, monkeypatching `compile_problem` / `System` / `AmrSystem`). AMR enablement and multi-block / output lowering are the coordinated PR-2 follow-ups.
- **Spec 5 sec.15 architecture negative-tests** (ADC-494, epic ADC-479): two new source-discovered checks under `tests/architecture/` lock the typed-object contract. `test_spec5_typed_api_negatives.py` asserts `pops.descriptors.reject_string_selector` raises an actionable `TypeError` (and leaves a real typed object untouched), that typed constructors reject a `kind="..."` string (`HLL(kind=)`, `RuntimeParam(..., kind=)`), that every catalogued `pops.numerics.riemann` / `reconstruction` / `pops.diagnostics` descriptor is `.inspect()`-able (dict) and `.validate()`-truthy, and that the compiled-brick load path refuses a missing manifest with a clear error (`read_manifest` / `CompiledBrickRef.resolve` / `.validate`). `test_spec5_physics_codegen_free.py` adds a focused source-only AST check that `pops.physics` never imports `pops.codegen` at module scope, plus a print-contract check that typed-descriptor `repr()` / `str()` is short and carries no raw `array(` dump. Both run in the existing architecture CI step; the `_pops`-dependent assertions skip when the extension is absent so the source-only checks still run bare. Test-only; no production code changed.
- **Spec 5 inert authoring entry points `pops.inspect_amr` + `physics.Model.field_problem`** (ADC-495, epic ADC-479, sections 5.1 / 5.11 / 8 / 9.6): two additive, inert authoring shortcuts. New top-level `pops.inspect_amr(layout_or_context=None)` returns a printable, deterministic `AmrReport` of an AMR hierarchy sourced from the `pops.mesh.layouts.AMR` / `pops.mesh.amr` descriptor chain -- the levels/ratio envelope, the regrid/patch/nesting/refine/checkpoint/output policies (the `TagUnion` expanded into its criteria), the runtime requirements (reflux/tag reduction), and the explainable route limitations; `None` reports the native envelope (`NATIVE_MAX_LEVELS`/`NATIVE_RATIOS`) and a `Uniform` layout reports single-level. It is the introspectable counterpart of `inspect_capabilities` and reads descriptor metadata only (never imports `_pops`/runtime/codegen). New `pops.physics.Model.field_problem(name, equation, outputs=, solver=, bcs=, coefficients=)` constructs and RETURNS an inspectable `pops.fields.PoissonProblem` (or `FieldProblem` when coefficients are present) from a `pops.math.Equation` `-laplacian(phi) == rhs` and records it on the model so the new `Model.inspect()` surfaces it; unlike `solve_field` it is inert (no dsl elliptic-rhs / operator-graph / codegen mutation). New `python/tests/test_inspect_amr.py` and `python/tests/test_board_field_problem.py`. Wiring the field problem into the operator graph stays deferred to the existing multi-elliptic runtime.
- **POPS_* environment variable reference** (ADC-496, criteria #46/#47): new `docs/sphinx/reference/environment-variables.md` documents every `POPS_*` variable the code actually reads (codegen/toolchain `POPS_INCLUDE` / `POPS_CXX` / `POPS_KOKKOS_ROOT` / `POPS_KOKKOS_CXX` / `POPS_KOKKOS_USE_NVCC_WRAPPER` / `POPS_DSL_OPTFLAGS` / `POPS_CACHE_DIR`, runtime diagnostics `POPS_FOREACH_SERIAL_THRESHOLD` / `POPS_TRACE_SOLVE_FIELDS`) with effect, default, and the source file that reads it; a "Planned / not yet implemented" section lists names that are not honored today (including `POPS_JIT_BACKDOOR`, which does not exist). Registered in the reference toctree and `docs/docmap.toml`.
- **POPS_THREADS default for `pops.set_threads()`** (epic ADC-479, Spec 5 sec.12.4): `pops.set_threads()` called with no argument now reads its default thread count from `POPS_THREADS` (a positive integer), additively -- an explicit `pops.set_threads(n)` always wins, and an unset / non-integer / `< 1` value is ignored and falls back to `os.cpu_count()` (no stricter rejection than passing the argument directly). New `_threads_from_env()` in `python/pops/runtime/threading.py`; the `POPS_PROFILE` default for `sim.profile()` (already wired in ADC-507) is moved out of the doc's "Planned" section into the new "Python runtime defaults" table in `docs/sphinx/reference/environment-variables.md`, which now also lists `POPS_THREADS`. New `python/tests/test_env_vars.py`.
- **Spec 5 inspectable route-choosing moments descriptors** (ADC-498, epic ADC-479, section 6 #6/#3): the `pops.moments` objects that CHOOSE a math algorithm now carry the typed descriptor contract, matching the `pops.numerics` catalog descriptors. `ExactSpeeds` (category `wave_speed`), `RealizabilityProjection` (category `realizability`), `MagneticMomentSource` (category `moment_source`) and `HyQMOM15Closure` (category `closure`) subclass `pops.descriptors.Descriptor` and declare their `options()` / `capabilities()`, so they expose the inert `requirements` / `available` / `validate` / `lower` / `inspect` surface and satisfy `DescriptorProtocol`. `HyQMOM15Closure` stays a closure callable and `MagneticMomentSource.as_sources()` still returns its builder callable. The builders / handles are unchanged and deliberately NOT descriptors -- `MomentModel` / `CartesianVelocityMoments` / `MomentHierarchy` / `MomentBasis` / `CenteredTransform` / `StandardizedTransform` / `VlasovSources`, and `MomentOrdering` (a single forced layout, no route choice) -- so the hot construction path stays lightweight. The bare `gaussian_closure` factory also stays a plain callable. The ADC-494 inspectability test (`python/tests/test_descriptor_protocol.py`) now loops the new moments descriptors (protocol surface + `inspect()` dict + `validate()`) and asserts the handles are not descriptors; new `python/tests/test_moments_descriptors.py`. Inert and additive.
- **Spec 5 typed pops.diagnostics measure descriptors** (ADC-499, epic ADC-479, sections 5.13 / 14.2.7): a diagnostic is now a typed object, not the string form `diagnostics.norm(kind="l2")`. New `pops.diagnostics.measures` adds the inert descriptors `Norm(norm, block=, role=, cadence=)` (the norm kind is a typed `pops.linalg.norms` object L1 / L2 / LInf; a string is rejected with a clear TypeError), `Integral(block=, role=)` (the domain-integral sum, e.g. `mass = Integral(role=Density)`), `MinMax(block=, role=)` (the min / max fold), and `ConservationCheck(quantity, tolerance=1e-12)` (a drift check whose `quantity` must be a diagnostic descriptor, e.g. an `Integral`; a string is rejected). Each builds on `pops.descriptors.Descriptor`, lowers to an inert native reduction scheme label (`norm` / `integral` / `min_max` / `conservation_check`; `norm` and `integral` reuse the same-name legacy `pops.diagnostics` factory scheme, `conservation_check` matches `pops.diagnostics.invariants`, and `min_max` is a new label -- no native symbol is fabricated), and declares its reduction kind, MPI-reduction need, cadence slot and AMR / multi-level compatibility as inspectable metadata (it computes nothing; the C++ / Kokkos / MPI runtime evaluates the reduction). Exported from `pops.diagnostics`; the legacy factory functions are kept. Nothing imports `_pops` / `numpy` / `pops.runtime` / `pops.codegen` at module scope. New `python/tests/test_diagnostics_typed.py`.
- **Spec 5 de-stringed FiniteVolume / Spatial** (ADC-497, epic ADC-479, section 7 / criterion #1): the runtime spatial builder now chooses the scheme with TYPED `pops.numerics` descriptors, never a bare string. `pops.FiniteVolume(...)` and `pops.Spatial(...)` accept a Riemann flux (`pops.numerics.riemann.Rusanov()/HLL()/HLLC()/Roe()`), a reconstruction/limiter (`pops.numerics.reconstruction.FirstOrder()/MUSCL(...)/WENO5()/WENO5Z()`, `...limiters.Minmod()/VanLeer()`) and a reconstructed-variable set (the new `pops.numerics.variables.Conservative()/Primitive()`), lower each to the canonical C++ token, and reject a bare-string `limiter`/`riemann`/`flux`/`recon`/`variables` through `reject_string_selector` with a `TypeError` that names the typed object; a wrong-category or unknown-scheme descriptor is also a clear error. The boolean shortcuts (`none=`/`minmod=`/`vanleer=`/`weno5=`/`primitive=`) stay as typed-flag sugar and are now forwarded identically through `FiniteVolume` (so `FiniteVolume(minmod=True)` / `FiniteVolume(primitive=True)` work like on `Spatial`); both `FiniteVolume` and `Spatial` also accept the Spec 5 sec.14.1 `reconstruction=` alias for the reconstruction/limiter slot (`limiter=` keeps working). The C++ ABI is unchanged (the descriptors lower to the same string tokens), and the Spec-3 `pops.lib.spatial.FiniteVolume(...)` catalog descriptor still carries string options (lowered internally). Every example/bench/test consumer and every live doc code block is migrated to the typed forms; the string API is removed (no back-compat). New `pops.numerics.variables` module and `python/tests/test_destring_finite_volume.py`.
- **Spec 5 typed-API reference docs** (ADC-487 docs half, epic ADC-479, section 16 #49): new Sphinx pages document the Spec-5 typed-object API. `docs/sphinx/concepts/typed-api.md` states the principle ("Python describes with typed objects, C++ executes; no YAML disguised as Python"; strings name user objects, typed objects choose algorithms/backends/layouts; string algorithm selectors are rejected). `docs/sphinx/reference/spec5-packages.md` is the per-package reference (`pops.numerics` / `fields` / `mesh` + layouts/amr / `params` / `output` / `external` / `diagnostics` / `descriptors` / `codegen.Optimization`) with a runnable example each (every example verified against the real API). `docs/sphinx/reference/migration-strings-to-typed.md` is the old-string -> typed-object conversion guide. Registered in the toctrees + `docs/docmap.toml`; `pops.solvers` / `pops.linalg` are noted as forthcoming (pending the lib relocation). The architecture-gate half of ADC-487 was merged earlier.
- **Spec 5 board-node elliptic-operator algebra** (ADC-491, epic ADC-479, section 9.2): the variable-coefficient field-equation forms Spec 5 itself uses are now authorable as inert, inspectable IR. New `pops.ir.elliptic` (re-exported via `pops.math`) adds the elliptic operator terms `Reaction` (`k*phi`, built by `coeff * unknown("phi")`), `CoeffGradient` (`eps*grad(phi)`), `DivCoeffGrad` (`div(eps*grad(phi))`) and their `EllipticSum`, with `+` / `-` / unary `-` composition and a `principal_kinds(lhs)` inspector; `Laplacian` becomes an elliptic term, `grad(phi)` / `unknown("phi")` gain coefficient multiply, and `div(grad(phi))` lowers to a Laplacian while `div(coeff*grad(phi))` lowers to a `DivCoeffGrad` (hyperbolic `div(flux)` is unchanged). The Poisson family now validates its distinguishing shape: `ScreenedPoissonProblem` requires a reaction term, `AnisotropicPoissonProblem` a `div(coeff*grad)` principal operator (so `-div(eps*grad(phi)) + kappa*phi == charge` is constructible and recognized). The board AST `pops.ir.expr` is split (the concrete elliptic terms move to `pops.ir.elliptic`) so it stays within the file-size budget. Inert IR only; lowering the variable-coefficient operators in the elliptic codegen is the coordinated follow-up. New `python/tests/test_field_equation_algebra.py`.
- **Spec 5 abstract algebra package `pops.linalg`** (ADC-493, epic ADC-479, section 5.6 / criterion #5): a new top-level package that NAMES the linear algebra `A x = b` as inert typed descriptors, distinct from the solvers that run it. `pops.linalg.operator` (`LinearOperator(name, native_id=)` assembled / `MatrixFreeOperator(name)` action-only), `pops.linalg.problem` (`LinearProblem(operator=, unknown=, rhs=)` validating that the operator is a linear operator and rejecting a string / None, plus `Residual(problem)` naming `b - A x`), `pops.linalg.norms` (typed `L1` / `L2` / `LInf`), and `pops.linalg.reductions` (`Dot` / `Norm2` + `dot(a, b)` / `norm2(x)` builders that NAME the reduction and compute nothing). All build on `pops.descriptors.Descriptor`; nothing imports `_pops` / `numpy` / `pops.runtime` / `pops.codegen` at module scope; the architecture gates add `linalg` to the import-graph / no-runtime-import / Spec-5-structure suites. The preconditioners stay under `pops.lib` until the Phase A2 relocation (`pops.linalg.preconditioners`). New `python/tests/test_linalg.py`. Inert and additive; the C++ / Kokkos solvers execute.
- **Spec 5 formal DescriptorProtocol + capability matrix + manifest reader** (ADC-483, epic ADC-479, sections 6 / 7 / 13.12.1): the descriptor contract is now a documented `typing.Protocol` `pops.descriptors.DescriptorProtocol` (name / category / native_id + requirements / capabilities / options / available / validate / lower / inspect); both descriptor families satisfy it (the `Descriptor` base gains an inert `lower(context)` returning a metadata-only dict that never runs a numeric loop, plus a `native_id` slot, and `BrickDescriptor` gains inert `lower` / `inspect` / `validate` keeping its attribute-based metadata identity). The duplicate mesh `Availability` is unified: `pops.mesh._descriptor` now re-exports `Availability` from `pops.descriptors` and `MeshDescriptor` subclasses the shared `Descriptor` base (both stay importable). New `pops.inspect_capabilities()` returns a descriptor-sourced, printable `CapabilityMatrix` (name / category / native_id / available / requirements per catalogued brick / layout / solver / field; pure, never imports `_pops`). New `pops.external.read_manifest(path)` reads a `.json` / `.so` brick manifest's metadata (ids / categories / requirements / capabilities / abi_key) into a `CompiledManifest` WITHOUT registering or executing it. New `pops.descriptors.reject_string_selector(value, param, suggestion)` raises a clear `TypeError` for a free-string algorithm selector (provided + tested; not yet wired into FiniteVolume). New `python/tests/test_descriptor_protocol.py`. All additive and inert.
- **Spec 5 typed field-problem authoring `pops.fields` + rhs terms `pops.numerics.terms`** (ADC-484, epic ADC-479, sections 9 / 5.5 / 5.4 / 14.2.4): a new top-level `pops.fields` authoring package declares an elliptic field solve as typed, inert descriptors -- `FieldProblem(unknown=, equation=, inputs=, coefficients=, bcs=, nullspace=, outputs=, solver=)` with the Poisson-family shortcuts `PoissonProblem` / `ScreenedPoissonProblem` / `AnisotropicPoissonProblem`; the equation is a real `pops.math.Equation` (`-laplacian(phi) == rhs`), and `validate()` rejects a Python bool / non-Equation and a missing solver before the runtime, with a short `print()` summary (sec.12.1). Supporting submodules: `fields.bcs` (field-value `Periodic` / `Dirichlet` / `Neumann` / `FirstOrderExtrapolation` + `FaceBC` and face selectors), `fields.rhs` (`ChargeDensity.from_blocks(...)`), `fields.coefficients` (`ScalarCoefficient` / `ReactionCoefficient`), `fields.nullspace` (`ConstantNullspace`), `fields.aux` (`StaticAux` / `DerivedAux` + the re-exported `pops.mesh.aux.AuxHalo`). New typed right-hand-side composition terms `pops.numerics.terms.Flux` / `SourceTerm` / `LocalTerm`. All build on `pops.descriptors.Descriptor`; the architecture gates add `fields` to the import-graph / no-runtime-import / Spec-5-structure suites. New `python/tests/test_fields_authoring.py`. Inert; nothing computes in Python and the runtime install path is unchanged.
- **Spec 5 typed temporal-version handles** (ADC-485, epic ADC-479, section 5.3.1): a time `Program` can name the temporal versions of a block state with typed handles instead of bare positional values. New `P.state("U", block="plasma")` returns a `pops.time.TimeState` exposing `.n` (the current state, read-only), `.stage(k)` (a per-stage version), `.next` (the end-of-step state), and `.prev` / `.prev(lag)` (the lagged history, after `P.keep_history(U, depth=..., cold_start=...)`). `P.define(U.stage(k), expr)` / `P.commit(U.next)` lower through the EXISTING `define` / `linear_combine` / `commit` / `history` / `store_history` ops, so a scheme written with handles produces a byte-identical IR (`_ir_hash`) to the legacy positional style (SSPRK3-via-handles == `pops.lib.time.ssprk3`); the handles carry no runtime data. The guards fail loud: `T.define(U.n, ...)` (read-only), `T.define(U.prev, ...)` (history is produced by the history policy), a stage used before `T.define`, a stage defined twice (SSA), and `U.prev` without `keep_history`. The positional `P.state` / `P.commit` / `P.define` paths are unchanged. New `python/tests/test_time_handles.py`; `pops.time.CopyCurrent` is the default cold-start policy.
- **Spec 5 typed codegen optimization options** (ADC-486, epic ADC-479, section 13.8): the codegen optimization policy and numeric math mode are typed objects, not a string `optimization="fast"`. New `pops.codegen.Optimization(cse=, eliminate_dead_nodes=, eliminate_redundant_solves=, fuse_local_ops=, hoist_reciprocals=, fuse=, math=)` with `to_emit_kwargs()` mapping onto the existing `emit_cpp*` knobs (non-breaking); typed numeric math modes `StrictMath` (default) / `FastMath` / `DebugMath` / `GpuRegisterAware` (a non-strict transform is never implicit -- it must be chosen explicitly, Spec 5 sec.13.10); typed fusion policies `ConservativeFusion` (never crosses an elliptic solve / global reduction / halo refresh) / `Disabled`. Built on the generic `pops.descriptors.Descriptor` base, so the `print()`-summary contract (sec.12.1) holds for these objects. New `python/tests/test_optimization_options.py`.
- **Spec 5 architecture gates for the central packages** (ADC-487, epic ADC-479, sections 15 / 16): the Spec-5 top-level packages are now locked into the architecture tests so their purity is enforced, not just asserted in docstrings. `tests/architecture/test_import_graph.py` adds `numerics` / `moments` / `diagnostics` / `params` / `output` / `external` to the layered DAG (`moments -> ir`, the rest import only the flat `pops.descriptors` / `pops.math`; `lib -> moments` for `lib.models`); `test_no_runtime_imports.py` adds them to the symbolic layers that must never import `_pops` / `pops.codegen` / `pops.runtime` at module scope. New `tests/architecture/test_spec5_structure.py` asserts the central packages are top-level, the shared `BrickDescriptor`/`Descriptor` base lives in `pops.descriptors`, the relocated catalogs are gone from `pops.lib`, and `pops.lib` holds only presets (+ the Phase A2 carve-out). (Reference-doc pages for the new packages are tracked separately under ADC-487.)
- **Spec 5 typed mesh package `pops.mesh`** (ADC-481, epic ADC-479, sections 5.9-5.11 / 8): the mesh / geometry / layout / AMR objects are now typed descriptors in a top-level `pops.mesh` package (no physics, no solver, no Python compute). The Cartesian / polar meshes + aux halo are lifted out of `pops.runtime.mesh` (which becomes a thin re-export shim, so `pops.System(mesh=...)` and `pops.CartesianMesh` are unchanged) into `pops.mesh.cartesian` / `polar` / `aux`, joined by `pops.mesh.boxes` (`PatchBox` / `BoxLayout`). New typed sub-packages: `pops.mesh.layouts` (`Uniform` / `AMR`, replacing `target="amr_system"`), `pops.mesh.amr` (`Refine` / `TagUnion` refinement criteria replacing `set_refinement(..., variable=...)`, plus `RegridEvery` / `FrozenRegrid` / `PatchLayout` / `ProperNesting` / `BufferCells` / `CheckpointPolicy` / `AMROutput` / level policies), `pops.mesh.geometry` (`Disc` / `HalfPlane` / `LevelSet` / `EmbeddedBoundary`), `pops.mesh.masks` (`NoMask` / `Staircase` / `CutCell`), and `pops.mesh.boundaries` (`Periodic` / `Physical` / `FaceBC` + face selectors). Every object carries the Spec-5 descriptor contract (`requirements` / `capabilities` / `options` / `available` / `validate` / `inspect`) via the lightweight `pops.mesh._descriptor` base + an explainable `Availability` status; the `AMR` layout declares the native route envelope (`max_levels<=2`, `ratio==2`) and refuses a request beyond it BEFORE the runtime with a clear message and alternatives. New `python/tests/test_mesh_descriptors.py`. No runtime path is changed; these objects describe a mesh the C++ runtime materialises after validation.
- **Spec 5 typed params / output / external packages** (ADC-482, epic ADC-479, sections 5.12 / 5.14 / 5.17): three new top-level descriptor packages. `pops.params` -- `RuntimeParam` / `ConstParam` / `DerivedParam` with a typed dtype (new `pops.math.Real` / `Integer` / `Bool` markers) and a typed domain (`pops.params.constraints.Positive` / `NonNegative` / `Range` / `In`), plus dimensioned `Constant`; replaces `Param(kind="runtime")` / `domain="positive"` (a runtime param surfaces at bind, a const param is in the cache key). `pops.output` -- `OutputPolicy` / `CheckpointPolicy` with typed formats (`pops.output.formats.HDF5` / `Plotfile`) and level policies (`AllLevels` / `CoarseOnly` / `SelectedLevels`); replaces `output(format="hdf5", every=20)` / `checkpoint(mode=...)`. `pops.external` -- `CompiledBrickRef` / `ExternalBrick` reference a compiled brick by manifest (`.json` or `.so`) + native id and resolve to the typed `external_cpp` descriptor with the manifest's requirements/capabilities (no free-string brick id; an unresolvable ref is an explainable error before runtime). A generic `Descriptor` + explainable `Availability` base lands in `pops.descriptors` (the Spec-5 sec.6 protocol home these and, in Phase D, the mesh descriptors share). New `python/tests/test_params_output_external.py`. All inert; nothing computes in Python.
- **Spec 4 architecture acceptance tests + examples + design docs** (ADC-474): new `tests/architecture/` checks gate the package restructure invariants (no flat root modules, every `python/pops/**/*.py` <= 500 lines with the `__init__.py` facade capped at 120, the symbolic layers `ir`/`model`/`physics`/`time`/`lib` never import `_pops`/`pops.codegen`/`pops.runtime` at module scope, the public import surface resolves, and the intra-`pops` import graph is acyclic and layered); they run as a dedicated source-only step in the `gate-python` CI job (not auto-discovered by the `python/tests` runner). New `examples/spec4/` walkthroughs (manual + library time programs, a custom-closure moment model, a model-only HyQMOM15, and an `adc_cases`-style driver). New design reference `docs/sphinx/design/spec4-package-architecture.md` (+ `design/index.md`, a Design toctree entry, a Spec-4 sub-packages section in `reference/python-api.md`, and the docmap registrations).
- **Spec 4 Python package restructure** (epic ADC-467): the flat `python/pops/*.py` modules are reorganized into per-responsibility sub-packages with an acyclic import graph -- `pops.ir` (symbolic IR), `pops.model` (operator-first core), `pops.physics` (math/physics facade + the PDE `PdeModel`), `pops.time` (the temporal language; ready-made schemes now live in `pops.lib.time`), `pops.codegen` (the sole C++ emitter: `toolchain`/`cache`/`abi`/`cpp_writer`/`module_codegen`/`program_codegen`/`library_codegen`/`compile`/`loader`), and `pops.lib` (typed brick descriptors split into `riemann`/`reconstruction`/`spatial`/`fields`/`operators`/`solvers`/`diagnostics`, the moment toolkit `pops.lib.moments`, and the provided models `pops.lib.models`). New public API: `pops.lib.moments.MomentModel` / `CartesianVelocityMoments` / `closures.HyQMOM15Closure` and `pops.lib.models.moments.HyQMOM15`; `pops.physics.Model` (board facade) and `pops.physics.PdeModel`; `pops.time.Program`. Each module file is <= 500 lines and no symbolic layer (`ir`/`model`/`physics`/`time`/`lib`) imports `pops.codegen` or `_pops` at module scope. Old flat imports keep working via transitional re-export shims, removed in the consumer-migration step (ADC-473).
- **Spec 3 scheduler-cache checkpoint/restart** (ADC-458, epic ADC-450, section 30): the held-node value cache (`every(N).hold` / `accumulate_dt`) now round-trips through the EXISTING checkpoint, so a `(run, checkpoint, restart, continue)` run is bit-for-bit identical to a continuous run (a held node resumes on its cadence instead of cold-starting). The `CacheManager` (`include/pops/runtime/program/cache_manager.hpp`) is now owned by the `System` (`System::program_cache()`), not the `.so` step closure, and `ProgramContext` forwards its `cache_*` seam ops to it (no ABI change); `CacheManager` gains the serialize/restore accessors (`node_ids`/`name_of`/`value_of`/`restore_slot` + a named `store`). `System` exposes `program_cache_nodes`/`program_cache_global`/`restore_program_cache` (+ per-slot bookkeeping), gathering each valid slot through the SAME `gather_global` / `write_state` machinery as the block state and the history rings (MPI-safe, bit-identical under `np>1`), bound to Python in `init_system.cpp`. `sim.checkpoint`/`sim.restart` (`python/pops/__init__.py`) serialize the cache alongside the histories and raise two verbatim guards at restart: `checkpoint was created with a different compiled Program hash` (the existing program-hash guard) and `checkpoint missing cached value for scheduled node '<name>'`. New `tests/test_checkpoint_cache.cpp` (the CacheManager serialize/restore round-trip + both verbatim messages, host-validated); a checkpoint with no cached node restores as before (back-compatible). The full compiled-`.so` held-schedule continuous == restart run is Kokkos-only AOT (ROMEO); carrying the operator NAME into the cached slot (today nameless, `node_<id>` fallback) is a small time.py codegen follow-up.
- **Spec 3 custom-solver C++ codegen** (ADC-462, epic ADC-450, section 20 / criterion 23 / 24.9): `pops.lib.generate_solver_cpp(solver)` now LOWERS a `@pops.lib.solver` IR to a self-contained C++ kernel that RUNS, instead of raising the deferred NotImplementedError. The new `_SolverCppLowering` walks the solver IR (zeros_like / norm2 / dot / apply / residual / combine / scalar_int / logical_and / while_) and emits a `template <class Op> pops::KrylovResult <name>_solve(const Op& A, pops::MultiFab& x, const pops::MultiFab& b)` whose convergence loop is a REAL C++ `for (;;)` (the predicate re-evaluates each pass), calling the SHARED matrix-free HPC primitives (`pops::dot` / `pops::saxpy` / `pops::lincomb`). The operator `A` is a value-typed template parameter (the shape the native Krylov loops take), so it inlines: no type-erased indirection in the kernel, no Python callback in the loop, no per-iteration heap (scratch fields allocated once, before the loop), no per-cell name dispatch (criterion 24.9); the authored `it < max_iter` cap is bound to the live loop counter so it bounds the generated loop for real. `scripts/gen_solver_kernel.py` emits a Richardson kernel and the new serial C++ test `tests/test_solver_codegen_generated.cpp` COMPILES it against the real runtime and RUNS it on an SPD Helmholtz system, recovering the manufactured solution and matching the native `pops::richardson_solve` bit-for-bit (same iters, max|x_gen - x_native| == 0). Updated `python/tests/test_solver_dsl.py` (emit + no-Python-callback proof + cap binding + scratch-once). The Program `solve_linear` over the native Krylov solvers stays the supported runtime path.
- **Spec 3 external C++ brick static dispatch** (ADC-463, epic ADC-450, sections 21-22, criterion 20): the deferred half of the external-brick path now dlopens a real brick `.so` and DISPATCHES its flux into the finite-volume machinery in the same type system as a native flux, statically and with no per-cell string lookup. `external_brick.hpp` gains `BrickRegistry::to_json()` + the `POPS_DEFINE_BRICK_MANIFEST()` macro (exports the `pops_brick_manifest()` C reader `pops.lib.load_cpp_library` already parses). New `external_riemann_brick.hpp` adds `POPS_DEFINE_EXTERNAL_RIEMANN_BRICK(id, Flux, Model, requirements)` -- the brick `.so` instantiates `build_block<Limiter, UserFlux>` at ITS OWN compile time (the user flux is a static template argument, fully inlined, the same leaf the native string ladder routes to), exporting a flat-array `pops_brick_residual` entry point with the same ABI as the AOT compiled block. The host `ExternalBrickHandle` dlopens the `.so`, reads + registers its manifest (id + requirements visible in the process `BrickRegistry`), and resolves the entry-point function pointer ONCE at construction; the only runtime string is the limiter (a 4-way `if` resolved once, never per cell). An unknown id or a `.so` not exporting `pops_brick_manifest`/`pops_brick_residual` is rejected with a clear error. New `tests/test_external_riemann_dispatch.cpp` compiles a real brick `.so` at run time (Kokkos-matched), dlopens it, and asserts its residual is BIT-IDENTICAL to the native rusanov path (the brick wraps `pops::RusanovFlux`); `python/tests/test_external_bricks.py` gains real-`.so` `load_cpp_library` coverage (compile + dlopen + descriptor surface, plus the non-brick `.so` rejection). The GPU/GH200 dispatch of an external brick is the ROMEO follow-up (only the on-Mac CPU/OpenMP Kokkos path is validated here).
- **Spec 3 compiled-runtime end-to-end test** (ADC-450, sections 17-19/29): new `python/tests/test_spec3_runtime_end_to_end.py` RUNTIME-proves the Spec 3 acceptance criteria the emit-only tests could only assert on generated source or defer to ROMEO. It compiles a real board-flavoured Program (a forward-Euler step whose Poisson field solve is held on an `every(N).hold()` schedule) to a `problem.so` via `pops.compile_problem`, installs it, and steps it on the loaded Kokkos toolchain, asserting: no Python call-frames are entered inside the compiled step (criterion 19, a `sys.settrace` count plus a raise-on-access proxy on the Python model/handle); the step advances the state finitely (a real compiled run); checkpoint at a due boundary then restart into a replayed composition resumes bit-identically over a held-schedule run (criterion 22/35); and `sim.profile_report()` surfaces the per-node timings, kernel count, and the scheduler `cache_hits`/`cache_misses` + `nodes_due`/`nodes_skipped` counters at runtime with consistent values (criterion 40). Self-skips cleanly without `_pops`/numpy/a compiler/a visible Kokkos; never fakes a pass.
- **Spec 3 profiling counters** (ADC-459, epic ADC-450, section 29): `sim.profile_report()` now surfaces the named integer counters next to the per-node timings -- a kernel count, the scratch peak memory, the scheduler cache hits/misses and the scheduled nodes due/skipped. They are counted in the C++ runtime, not via codegen: `Profiler` gains `count_max` (a PEAK counter for the scratch bytes); `ProgramContext` bumps `kernels` at each kernel-dispatching seam op (rhs/source/flux/matvec/solve/linear-combine), records the scratch allocation count + byte peak in `rhs_scratch_like`, and counts `cache_hits`/`cache_misses` + `nodes_due`/`nodes_skipped` in `cache_should_update`; `System::Impl::solve_fields` counts a `kernels` launch so the NATIVE step path (which routes its head solve there) reports a non-zero kernel count too. Every counter is a single predictable branch when profiling is off (no hot-path cost), and no SystemStepper edit (the Profiler stays an Impl member the stepper never reads). New `python/tests/test_profiling_counters.py` + extended `tests/test_program_runtime.cpp` (host path: kernels + scratch counters move, cache counters stay 0) and `tests/test_profiler.cpp` (`count_max` peak semantics). The cache hit/skip cadence moves only under a compiled `.so` held-schedule step body (validated on Kokkos/ROMEO, not buildable host-only).
- **Spec 3 unified install + install-time validation** (ADC-466, epic ADC-450): `sim.install(compiled, instances={...}, params={...}, aux={...}, solvers={...})` (`python/pops/__init__.py`) is the single Spec-3 (section 22) entry that wires a compiled handle, each named instance's initial state + spatial brick, the runtime params, the aux fields and the field solvers in ONE call, LOWERING to the existing lower-layer calls (add_equation / set_poisson / set_magnetic_field / set_aux_field / set_block_params / install_program) -- no parallel runtime; the lower-layer calls stay available. Install-time requirement validation (section 24) now checks, beyond the existing aux requirement, the BLOCK-INSTANCE and SOLVER requirements from the .so metadata (`module_metadata.hpp` gains `required_blocks` / `required_solver` / `requirement_string` over the per-operator requirements JSON; `System::install_program` validates them against `block_names()` and the new `System::poisson_solver()` accessor) and the RIEMANN CAPABILITY host-side, raising the verbatim spec messages (`operator '...' requires block instance '...'`, `field operator '...' requires solver '...'`, `riemann HLLC requires capability 'hllc_star_state'`). New `python/tests/test_unified_install.py` (API shape + lowering + capability/aux/solver validation host-tested); the full compiled-.so install RUN is Kokkos-only (ROMEO / CI-Kokkos). The schedule-hold-on-non-cacheable requirement stays validated at Program authoring (`_validate_schedule`); exporting it from the .so for an install-time check is a follow-up (needs the time.py codegen to emit the cacheable/schedule requirement).
- **Spec 3 unified install + install-time validation** (ADC-466, epic ADC-450): `sim.install(compiled, instances={...}, params={...}, aux={...}, solvers={...})` (`python/pops/__init__.py`) is the single Spec-3 (section 22) entry that wires a compiled handle, each named instance's initial state + spatial brick, the runtime params, the aux fields and the field solvers in ONE call, LOWERING to the existing lower-layer calls (add_equation / set_poisson / set_magnetic_field / set_aux_field / set_block_params / install_program) -- no parallel runtime; the lower-layer calls stay available. Install-time requirement validation (section 24) now checks, beyond the existing aux requirement, the BLOCK-INSTANCE and SOLVER requirements from the .so metadata (`module_metadata.hpp` gains `required_blocks` / `required_solver` / `requirement_string` over the per-operator requirements JSON; `System::install_program` validates them against `block_names()` and the new `System::poisson_solver()` accessor) and the RIEMANN CAPABILITY host-side, raising the verbatim spec messages (`operator '...' requires block instance '...'`, `field operator '...' requires solver '...'`, `riemann HLLC requires capability 'hllc_star_state'`). New `python/tests/test_unified_install.py` (API shape + lowering + capability/aux/solver validation host-tested); the full compiled-.so install RUN is Kokkos-only (ROMEO / CI-Kokkos). The schedule-hold-on-non-cacheable requirement stays validated at Program authoring (`_validate_schedule`); exporting it from the .so for an install-time check is a follow-up (needs the time.py codegen to emit the cacheable/schedule requirement). A runtime param declared by a dsl.Model instance is routed to its block via `set_block_params` (sorted by name) after install resolves that instance via AOT (the production/native backend freezes runtime params), so `install(params={...})` works on the headline `compile_problem(model=...)` path; such an instance must use an AOT-compatible time (the default `pops.Explicit()` == SSPRK2). The positive routing is covered host-side and Kokkos end-to-end in `test_unified_install.py`.
- **Spec 3 full scheduler codegen** (ADC-458, epic ADC-450, sections 17-18): every schedule kind and policy now LOWERS to C++, generalizing the held-field-solve cache branch to ANY schedulable node. `Program._emit_schedule_wrap` wraps the statements a node emits in a due-test guard plus its policy branch: `every(N)` -> `ctx.cache_should_update(id, N)`, `on_start()` -> `ctx.macro_step() == 0`, `when(cond)` -> the lowered Program Bool predicate, `subcycle(count, dt)` -> a `for` sub-loop over the sub-dt; `recompute`/`skip` run the body only when due (skip keeps the stale value), `zero` sets the output to zero off-cadence, `hold` stores/restores the cached value, `accumulate_dt` accumulates the actual skipped dt and reads `eff_dt = sum(dt_skipped)` on the due step (never `N*dt_current`), `error` fails loud if read off-cadence. A field solve caches the System aux; any other node caches its own named scratch (its output declaration is hoisted out of the guard). `CacheManager` gains `effective_dt` (the due-step read, resets the accumulator) and a cold-start-safe `accumulate_dt`; `ProgramContext` gains `cache_store_scratch`/`cache_restore_scratch`/`cache_accumulate_dt`/`cache_effective_dt`/`macro_step`/`scheduler_error`. `_check_schedules_lowerable` now only refuses `on_end()` (a compiled `sim.step(dt)` loop has no end-of-run signal) and a `when()` over a Python callable. New `python/tests/test_scheduler_codegen.py` (emit, script + pytest) + extended `tests/test_cache_manager.cpp` (effective_dt sum + named-scratch cache); the cache cadence RUNTIME in a stepping `.so` is validated on ROMEO (Kokkos-only AOT), and the checkpoint of the cache state is the ADC-458 section 30 follow-up.
- Spec 3 generic multi-species board facade (ADC-457, epic ADC-450, sections 12 + 16): `pops.physics.Model.species` now authors N >= 2 species instead of rejecting a second one with NotImplementedError. Each species LOWERS to one `pops.model.StateSpace` (a named block instance); `m.coupled_rate(name, inputs=[...], outputs={species: [per-component exprs]})` lowers to the existing operator-first `coupled_rate` operator (a `RateBundle` signature over the input species, arbitrary arity, no two-input limit); `m.solve_fields_from_species(name, inputs=[...])` lowers to a multi-input `field_operator` (the `solve_fields_from_blocks` surface). A species handle indexes its cons vars by name (`e["ne"]`). The board IR is EQUIVALENT to the hand-written `pops.model.Module`: a two-fluid board model shares its `module_hash` and emits byte-identical C++. The single-species path stays byte-identical to `m.state` (no multi-block Module created). New `python/tests/test_board_multispecies.py` (pure authoring/IR, script + pytest); the compiled multi-block .so RUN is validated on ROMEO (Kokkos-only AOT), not locally.
- Spec 3 coupled-rate multi-state kernel codegen (ADC-457, epic ADC-450): a `coupled_rate` operator (collisions / ionization, criterion 27) now LOWERS to ONE multi-state `pops::for_each_cell` kernel instead of raising. `_emit_op` allocates one rate scratch per block (shaped like that block's state), emits the shared kernel that binds each input state's Array4 + the cons vars it references (each from its own StateSpace component index) and writes every block's rate scratch, and aliases each `coupled_rate_out` projection to its block's scratch; `_check_lowerable` un-gates a cons-only coupled_rate and `dump_cpp_plan` shows the kernel. The cons-only MVP defers any formula referencing a prim/aux var with a clear ADC-457 NotImplementedError. New `python/tests/test_coupled_rate_codegen.py`; the compiled .so collision step (the RUNTIME that fills the rate and advances the species) is validated on ROMEO (Kokkos-only AOT), not locally.
- **Spec 3 coupled multi-block field-solve codegen** (ADC-457, epic ADC-450): `P.solve_fields_from_blocks([...])` now LOWERS instead of raising NotImplementedError. `SystemFieldSolver` gains `assemble_poisson_rhs_from_blocks` (the system Poisson RHS as Sum_s elliptic_rhs_s(U_s) reading every block's stage state at once, indexed by block index; nullptr uses the live state) and `solve_fields_from_blocks` (cartesian + polar, mirroring solve_fields); `System` and `ProgramContext` forward it; `_emit_op` emits `ctx.solve_fields_from_blocks(<vec>)` over a per-block pointer vector sized to `ctx.n_blocks()`, each listed block slotted at its index. New `python/tests/test_coupled_fieldsolve_codegen.py` + C++ `tests/test_coupled_fieldsolve.cpp` (the from-blocks RHS == the per-block sum, the per-block stage override honored); the compiled .so coupled solve is the ROMEO follow-up.
- **Spec 3 arbitrary-formula Riemann hooks** (ADC-456, epic ADC-450): `m.riemann("hllc", pressure=<pops.math expr>)` now codegen's that board formula into the generated brick's `pressure(U)` capability hook instead of the role-derived primitive `p`. `pops.physics.Model.riemann` wires the recorded formulas through to `pops.dsl.Model.set_riemann_hooks`, which records single-state Expr overrides and folds them into `_model_hash`; a capability-hook descriptor (`pops.lib.riemann.hllc.contact_speed.euler()`) or `None` keeps the role-derived default, and a formula referencing a quantity the model cannot provide raises the clear capability error at codegen. Two-state hooks (`contact_speed`/`star_state`) accept only the role-derived descriptor. Emit-level test `python/tests/test_riemann_formula_hooks.py`; the brick `.so` compile is the ROMEO follow-up.
- **Spec 3 scheduler cache codegen** (ADC-458, epic ADC-450): a `solve_fields` node carrying an `every(N).hold()` schedule now LOWERS to a held field solve -- `ProgramContext` gains `cache_should_update`/`cache_store_aux`/`cache_restore_aux` (a `mutable CacheManager` keyed by Program node id, storing/restoring the System aux via `GridContext`), and `_emit_op` emits `if (ctx.cache_should_update(id, N)) { solve; cache_store_aux } else { cache_restore_aux }` so the elliptic solve recomputes every N macro-steps and the cached aux is reused in between. `_check_schedules_lowerable` un-gates exactly the hold solve_fields (other ops/policies still refuse with a clear ADC-458 message). New `python/tests/test_cache_codegen.py`; the cache RUNTIME cadence + checkpoint run in a compiled .so (ROMEO; the CacheManager is unit-tested by `tests/test_cache_manager.cpp`).
- **Spec 3 external C++ bricks** (ADC-463, epic ADC-450): a process-global brick registry `pops::runtime::program::BrickRegistry` + `BrickManifestEntry` + the `POPS_REGISTER_BRICK(id, category, requirements)` static-init macro (`include/pops/runtime/program/external_brick.hpp`, header-only host registry); on the Python side `pops.lib.load_cpp_library(path)` dlopens a brick `.so`, reads its `pops_brick_manifest()` JSON and registers the ids, so `pops.lib.riemann.User(id)` and the generic `pops.lib.external(id)` surface an `external_cpp` descriptor carrying the manifest requirements/capabilities; an unloaded id raises a clear `LookupError`. Serial C++ test `tests/test_external_brick.cpp` + Python test `python/tests/test_external_bricks.py` (both passing locally) + runnable example `examples/spec3/external_cpp_riemann.py` (selects an external brick on an Euler board and lowers the IR; the compiled run is guarded to ROMEO). Wiring a loaded brick's kernel into a running codegen path is the follow-up.
- **Spec 3 compile_library brick .so** (ADC-464, epic ADC-450): `pops.compile_library(name, objects, backend="production")` (`python/pops/library.py`) collects `pops.lib` brick descriptors (native / generated / macro / external, e.g. `pops.lib.solvers.GMRES()`, `pops.lib.riemann.HLLC()`, an `@pops.lib.solver` brick) into a `LibraryManifest`: name, the loaded-module ABI key, the brick list (id, type, category, scheme, native_id, requirements, capabilities), the generated symbols, and a stable order-insensitive content hash (sha256, mirroring `_model_hash`). `compile_library(..., emit=True)` now EMITS the library C++ (`python/pops/library_codegen.py`) and compiles a REAL `.so` with the same Kokkos toolchain a problem `.so` uses (`pops_loader_build_flags`, `POPS_KOKKOS_ROOT`), cached out-of-source by content hash + ABI key; the `.so` exports `pops_library_abi_key` (`POPS_ABI_KEY_LITERAL`), the library identity, the per-brick metadata tables (id / type / category / scheme / native_id / requirements / capabilities / available), the generated symbols, and the `pops_brick_manifest()` JSON `pops.lib.load_cpp_library` reads (it registers each brick into `BrickRegistry` at static-init, so the library `.so` is also a self-describing external-brick `.so`). `pops.read_library_manifest(...)` now also accepts a compiled `.so` PATH: it dlopens it, reads the descriptor back, and rejects an ABI / Kokkos mismatch as a HARD `RuntimeError` (never a silent fallback). `pops.compile_problem(..., libraries=[...so])` reads + ABI-validates each library and carries the manifests on the `CompiledProblem`. New `python/pops/library_codegen.py`, extended `python/tests/test_compile_library.py` (manifest layer host-only + the real emit / compile / read-back / ABI-mismatch / consume gated on Kokkos) and `examples/spec3/compile_library_so.py`. The generated-brick KERNEL codegen (lowering an `@pops.lib.solver` body to a callable C++ symbol) remains the ADC-462 follow-up; the library carries it as a metadata-only generated symbol.
- **Spec 3 scheduler cache foundation (C++)** (ADC-458, epic ADC-450): `pops::runtime::program::CacheManager` + `CacheSlot` (`include/pops/runtime/program/cache_manager.hpp`) -- the per-node value cache the unified Program scheduler needs: `is_due(node, step, every_n)` (cold-start always due, then every N macro-steps), `store`/`retrieve` of a held `MultiFab`, and `accumulate_dt` that sums the skipped steps' dt (so a held result applies with `eff_dt = sum(dt_skipped)`, not `N*dt_current`). Header-only, serial C++ unit test `tests/test_cache_manager.cpp` (built + passing locally). The codegen that un-gates a scheduled node + the checkpoint of the slots are the ADC-458 follow-ups.

- **Spec 3 IR dead-node elimination pass** (ADC-465, epic ADC-450): `pops.time.eliminate_dead_nodes(P)`
  (also `P.eliminate_dead_nodes()`) returns a NEW Program with the unconsumed flat SSA nodes removed.
  An OPT-IN pass: it never runs on the default `emit_cpp_program` path, so it cannot change an existing
  compiled program. SAFE-BY-DEFAULT: a node is removable ONLY if its op is on an explicit allow-list of
  ops proven to allocate a FRESH result scratch and have no other side effect (rhs, source, apply,
  linear_combine, linear_source, solve_local_linear, cell_compare, where, reduce, scalar_op, compare)
  AND no live op consumes its result. Every other op -- the buffer-writers that alias a caller-allocated
  input buffer (schur_rhs, laplacian, gradient, divergence, the schur_* family), the side-effecting ops
  (solve_fields, project, fill_boundary, store_history, record_scalar), solve_linear, and the sub-block
  ops (while/if/range, matrix_free_operator, solve_local_nonlinear) -- is kept even with an unconsumed
  result, so a new/unknown op is never wrongly dropped. Surviving nodes are renumbered to contiguous ids
  in their original creation order, so the pass is a byte-for-byte no-op when nothing is dead and a
  program with a dead node emits C++ identical to the same program written without it. New
  `python/tests/test_ir_passes.py` pins removal, the parity byte-identities, the side-effect and
  buffer-writer (condensed_schur) exemptions and the IR-hash-changes-but-outputs-same contract.
- **Spec 3 remaining IR optimization passes** (ADC-465, epic ADC-450, section 28): the section-28
  pipeline beyond dead-node elimination, all on `pops.time.Program`. Two more proven-safe TRANSFORM
  passes (opt-in, never on the default `emit_cpp_program` path): `eliminate_common_subexpressions`
  computes a duplicated PURE sub-IR (same op + inputs + attrs, restricted to a `_PURE_OPS` allow-list)
  once and aliases the rest, preserving each consumer's `axpy` structure so the result is bit-identical
  (a reduce, a solve, a buffer-writer or a side-effecting op is never collapsed); and
  `eliminate_redundant_field_solves` removes a second `solve_fields` over the same state only when no
  state/aux mutation (a commit, `project`, `fill_boundary`, `store_history`, another field solve)
  intervenes. `P.optimize()` chains the three proven-safe transforms and is byte-for-byte identical on
  an already-optimal program; `P.dump_passes()` traces the pipeline. Plus the section-28 ANALYSIS
  reports (never rewrite the IR): `scratch_liveness` (per-scratch live ranges), `buffer_reuse_report`
  (minimum buffer count via disjoint-range allocation), `estimate` / `estimate_report` (static,
  grid-relative memory-traffic + kernel-count) and `gpu_detectors` (too-many-small-kernels /
  too-many-scratches / excessive-traffic warnings, never a hard error). The measured GPU kernel count
  is a ROMEO profile; the estimate is the host-side prediction. New `python/tests/test_ir_opt_passes.py`
  (script + pytest) pins the CSE byte-identity-to-handwritten, the never-collapse-side-effect /
  buffer-writer guards, the redundant-solve removal-and-keep contract, the report sanity, the GPU
  detector firing, and the CRITICAL pipeline byte-identity on a non-optimizable program.
- **Spec 3 profiling wired into System** (ADC-459, epic ADC-450): `sim.enable_profiling()` /
  `disable_profiling()` / `is_profiling()` / `reset_profiling()` / `profile_report()` drive a
  System-owned `pops::runtime::program::Profiler`; `System::step` and `solve_fields` wrap themselves
  in a `ProfileScope` (a "step" / "field_solve" phase + a "steps" counter). The profiler lives on
  `System::Impl` and is not referenced by `SystemStepper`, so the `MockImpl` is unaffected.
  During `step()` only the "step" phase + the "steps" counter are recorded; the "field_solve" phase
  is captured on a direct `sim.solve_fields()` call (the stepper drives the Impl's solve, not
  `System::solve_fields`). Disabled by default (no hot-path cost). New `python/tests/test_profiling.py`
  (binding surface + enable/disable state machine, and a stepped native end-to-end check that
  asserts the report carries the "step" phase + "steps=2").
  The per-Program-node / per-native-brick granularity (through the compiled-program ProgramContext)
  is the next ADC-459 step.
- **Spec 3 per-node Program profiling** (ADC-459, epic ADC-450): the compiled time Program now times
  each Program node. `System::profiler()` exposes the System-owned `Profiler` and
  `ProgramContext::profile_node` / `profile_record` time a node into it, so `sim.profile_report()`
  shows per-node scopes ("node:rhs2", "node:solve_fields1", ...) next to the coarse "step" /
  "field_solve" phases. `emit_cpp_program` brackets every work node's emitted C++ with a steady_clock
  pair (pure reference-binding nodes are skipped); additive and free-when-disabled, the numerics are
  unchanged. New `python/tests/test_pernode_profiling.py` pins the generated source.
- **Spec 3 custom solver DSL (IR-authoring slice)** (ADC-462, epic ADC-450): `@pops.lib.solver`
  registers a `generated` solver descriptor whose Python builder AUTHORS a solver IR with
  matrix-free Krylov primitives (`SolverContext.norm2` / `dot` / `scalar_int` / `logical_and` /
  `residual` / affine `x + omega*r` / `while_` context manager); the `while_` convergence test takes
  a condition BUILDER re-evaluated against the loop-updated iterate (recorded into its own
  `cond_block`, mirroring `pops.time.Program.while_`), never a pre-built Bool frozen on the initial
  iterate. `pops.lib.build_solver_ir` runs the builder to produce a `SolverIR` (no Python numerics) and
  the descriptor is selectable like a native solver (`pops.lib.solvers.GMRES()`). The generated-C++
  lowering + run is deferred: `generate_solver_cpp` raises a clear ADC-462 `NotImplementedError`
  rather than fake a Python solve. New `python/tests/test_solver_dsl.py` (11);
  `examples/spec3/custom_richardson_solver.py`.

- **Spec 3 Program profiler (C++ foundation)** (ADC-459, epic ADC-450): `pops::runtime::program::Profiler`
  (`include/pops/runtime/program/profiler.hpp`) -- a header-only per-node / per-brick wall-clock
  accumulator (count / total / mean / min / max per named scope) plus integer counters (kernels,
  cache hits/misses, scheduled nodes due/skipped) and the report `sim.profile_report()` will return,
  with an RAII `ProfileScope`. Disabled by default (no measurable hot-path cost when off). Serial C++
  unit test `tests/test_profiler.cpp` (built + passing locally). The `System` step instrumentation
  and the `sim.enable_profiling()` / `profile_report()` Python bindings are the ADC-459 follow-up.

- **Spec 3 coupled_rate operator + multi-output P.call** (ADC-457, epic ADC-450): a `coupled_rate`
  operator kind (`model.OPERATOR_KINDS`) whose `Signature` output is a `RateBundle` (now hashable/
  equatable so it can be a signature output), of arbitrary arity. `Program.call` on a coupled_rate
  returns a `_CoupledResult` whose `["electrons"]` is the per-block rate (an RHS Value) that composes
  like any other (`e_n + dt * C["electrons"]`) and feeds `commit_many`. The coupled-rate KERNEL
  codegen is deferred (`_check_lowerable` refuses a `coupled_rate` node with a clear ADC-457 message
  rather than faking it as independent single-block rates). New `python/tests/test_coupled_rate.py`
  (7); `examples/spec3/rate_bundle_collisions.py` extended to a full coupled step.

- **Spec 3 unified-scheduler authoring** (ADC-458, epic ADC-450): `pops.time` schedules
  (`always`/`every(N)`/`when`/`on_start`/`on_end`/`subcycle`) with policy chaining
  (`.hold`/`.skip`/`.zero`/`.accumulate_dt`/`.error`), a `schedule=` kwarg on `Program.call`
  recording the schedule on the IR node (shown by `dump_operator_ir`), and the cacheable
  validation: a caching policy (`hold`/`accumulate_dt`) on a non-cacheable operator is rejected,
  with `Module.operator_capabilities(name, cacheable=True)` to declare it. A non-`always`
  schedule refuses to lower (`NotImplementedError`, ADC-458) rather than silently no-op; the
  cache/checkpoint runtime is the C++ follow-up. New `python/tests/test_schedule_authoring.py`
  and `examples/spec3/scheduled_fields_subcycled_transport.py`.
- **Spec 3 native Riemann capability validation** (ADC-456, epic ADC-450): board `m.riemann(...)`
  and `m.finite_volume_rate(riemann=...)` now validate the model's capabilities for the chosen
  native solver and canonicalize board roles (`density` -> `Density`, `momentum_x` -> `MomentumX`,
  ...) so the role lookup recognizes them. HLLC/Roe require a pressure and the fluid roles;
  HLL requires wave speeds; Rusanov requires only a max wave speed. Missing capabilities are
  rejected early with a clear message ("riemann HLLC requires model capability 'pressure' for
  state 'U'"), and a valid selection drives the dsl `enable_hllc()`/`enable_roe()` role-derived
  `POPS_HD` hook generation. New `pops.lib.riemann.speeds.*` / `riemann.hllc.contact_speed.*` /
  `star_state.*` hook-selector descriptors. New `python/tests/test_riemann_capabilities.py` and
  `examples/spec3/hllc_capabilities_euler.py`. Generating hooks from arbitrary board formulas and
  the end-to-end board-model compile remain follow-ups.
- **Spec 3 reference docs (multispecies / scheduler / custom-solvers) + multi-species example**
  (ADC-455, epic ADC-450): `docs/sphinx/reference/multispecies.md` (species as BlockInstances,
  arbitrary-arity operators, `RateBundle` typed multi-output, `commit_many`, `StageStateSet`),
  `program-scheduler.md` (the unified-scheduler design: `every`/`hold`/`skip`/`accumulate_dt`,
  cacheable capabilities, checkpointed caches; current substeps/stride mechanism) and
  `custom-solvers.md` (native Krylov solvers + the generated solver-DSL design), plus
  `examples/spec3/multispecies_three_fluids.py` (a 3-species step over the operator-first
  multi-block kernel: multi-state `Module` + `RateBundle` + multi-block `Program` + atomic
  `commit_many`, IR-level). These complete the Spec 3 reference set; the not-yet-wired runtime
  pieces are flagged as ADC-457/458 follow-ups.
- **Spec 3 reference docs** (ADC-455, epic ADC-450): `docs/sphinx/reference/native-numerics.md`
  (the native C++ Riemann solvers + reconstruction bricks and the model capabilities they need)
  and `typed-bricks.md` (the `pops.lib` catalog: NativeBrick / GeneratedBrick / MacroBrick /
  ExternalCppBrick, the brick descriptors and the time macros). Document existing functionality;
  the not-yet-wired pieces (board `m.riemann` hook codegen, `compile_library`, specialization
  modes, external C++ registration) are clearly marked as ADC-456+ follow-ups.
- **Spec 3 inspection / debug API** (ADC-460, epic ADC-450, spec 3 section 33): show the lowering of
  a board-written model or program. `Program.dump_operator_ir()` renders the operator-first Program
  IR a board program lowers to, `Program.dump_board()` / `dump_cpp_plan()` the board view and the C++
  step plan; `physics.Model.dump_physics()` / `dump_module_ir()` / `dump_capabilities()` render the
  declared surface, the typed `pops.model.Module` and the operator requirements/capabilities. Pure
  introspection over the existing IR (no new IR, no parallel system). New
  `python/tests/test_inspection.py`; examples `examples/spec3/board_vs_operator_ir_equivalence.py`
  (asserts board IR == operator-first IR) and `operator_first_same_problem.py`; reference pages
  `typed-ir.md` + `spec2-builder-layer.md`.
- **Blackboard-style physics DSL** (ADC-451/452/453/454, epic ADC-450, spec 3): a layer-1 user API
  that reads like the blackboard and lowers to the spec-2 operator-first IR. `pops.math` adds the
  notation (`ddt`, `div`, `grad`, `laplacian`, `dx`/`dy`, `rate`, `unknown`, `integral`, `sqrt`,
  `==` equations, `@` operator application); `pops.physics.Model` authors a model from equations
  (`m.rate("explicit_rate", ddt(U) == -div(F) + S)`, `m.solve_field(-laplacian(phi) == rho)`,
  `m.flux`/`m.source`/`m.local_linear_operator`/`m.riemann`/`m.invariant`) and exposes the typed
  `pops.model.Module` via `m.module`. `pops.time.Program` gains board sugar (`T.fields`/`T.define`/
  `T.solve`/`T.commit_many`/`T.state_set`) that lowers to the SAME IR as the `P.call` /
  `linear_combine` / `solve_local_linear` / `commit` style. `pops.lib` is a catalog of typed brick
  descriptors and IR macros (riemann/reconstruction/limiters/fields/solvers/.../time) that compute
  nothing in Python and name native C++ ids. New types: `pops.model.RateBundle` (typed multi-output,
  arbitrary arity), `pops.time.StageStateSet`, generic `physics.Invariant`. Pure-Python, additive;
  the spec-1 `pops.dsl` and the `P` builder styles are unchanged. New `python/tests/test_physics_board.py`,
  `test_lib_descriptors.py`, `test_invariants.py`, `test_time_board.py`; examples under
  `examples/spec3/`. The native HLLC/Roe hook codegen, the generic multi-species runtime, the unified
  Program scheduler and profiling remain follow-ups (ADC-456/457/458/459).
- **Install-time operator-requirement validation** (ADC-446, epic ADC-436, spec 2 criterion 24):
  `System.install_program` now reads the compiled `problem.so`'s GeneratedModule descriptor and
  rejects, BEFORE installing the program, a simulation that does not provide an aux field an operator
  requires -- e.g. "operator 'lorentz' requires aux field 'B_z', but simulation did not provide it"
  -- instead of a cryptic failure mid-step. The user-supplied application fields `B_z`
  (`set_magnetic_field`) and `T_e` are the hard requirements; derived/lazy fields cannot block. A
  pre-Spec-2 `.so` carries no descriptor and is unaffected. New `required_aux` parser in
  `module_metadata.hpp` (covered by `tests/test_module_metadata.cpp`) and
  `python/tests/test_install_requirement_validation.py` (ROMEO).
- **Pure `pops.model.Module` compiles** (ADC-447, epic ADC-436, spec 2 "model-free"): a Module
  authored directly -- typed spaces, operators with IR (`dsl.Expr`) bodies (builder mode `expr=`),
  the Riemann wave speeds via `Module.eigenvalues`, and a composite rate via `Module.rate_operator`
  -- is a self-contained, compilable model. `Module.to_dsl()` lowers it to a `dsl.Model` (each typed
  operator mapped to the dsl method of its kind), reusing the dsl codegen engine (a translation, not
  a second backend); `pops.compile_problem(model=module, time=P)` accepts a Module. The
  `examples/operator_modules/predictor_corrector_operator_first.py` example is rewritten as a pure
  Module (no PDE method called on the model). New `python/tests/test_module_compile.py`.

- **Operator-first Program type diagnostics** (ADC-448, epic ADC-436, spec 2 "operator-first"): with
  `P.state(block, space=U)` tags and rates/operators flowing from `P.call`, a Program type-checks the
  composition -- an argument over the wrong StateSpace (`expects state 'U' but got a value over 'V'`),
  combining a `Rate(U)` with a `State(V)` (`cannot combine values over different state spaces`), or
  driving `solve_local_linear` / `apply` with an `L: U -> U` on a `State(V)` (`maps U -> U but was
  applied to a State over 'V'`) all raise clear errors. The space tag is build-time only (never
  serialized); untagged (legacy / Spec 1) programs skip the checks. New
  `python/tests/test_operator_validation.py`.

- **Operator-first perf benchmark** (ADC-445, epic ADC-436, spec 2 "operator-first"):
  `bench/operator_first_perf.py` times the SSPRK3 step loop for the same 2D Euler model two ways --
  the native stepper (`pops.Explicit("ssprk3")`) vs a compiled time Program (`pops.time.std.ssprk3`
  -> `compile_problem` -> `install_program`) -- and reports the generated/native ms-per-step ratio.
  The HPC contract is that the generated Program is a specialized scheduler over the same adc_cpp
  primitives (no alternative runtime, no per-step Python, no silent CPU fallback), so the ratio is
  near 1. Run on a Kokkos build (ROMEO); skips cleanly without a compiler/Kokkos.
- **GeneratedModule metadata in the compiled `.so`** (ADC-442, epic ADC-436, spec 2 "operator-first"):
  a combined model+program `problem.so` now carries, alongside `GeneratedProgram` (the installed
  step), a **GeneratedModule** descriptor -- `extern "C"` accessors (`pops_module_operator_count` /
  `_name` / `_kind` / `_signature` / `_requirements`, `_state_space_*`, `_field_space_*`) exposing the
  typed operator registry by integer `OperatorId` (the registration index). New
  `include/pops/runtime/program/module_metadata.hpp` reads the descriptor from a dlopen'd handle
  (`OperatorId` / `SpaceId`, `OperatorMetadata`, `ModuleMetadata`, `read_module_metadata`), degrading
  to `present=false` on a pre-Spec-2 `.so`. The descriptor is read once at install (introspection /
  requirement validation); the step body never references it, so operators stay inlined and there is
  no string lookup in any hot kernel. New `tests/test_module_metadata.cpp` and
  `python/tests/test_module_codegen.py`.

- **Operator-first example and docs** (ADC-444, epic ADC-436, spec 2):
  `examples/operator_modules/predictor_corrector_operator_first.py` builds the spec-1 Example 5
  physics with `pops.model` + the generic `predictor_corrector_local_linear` macro (no physics term in
  the program), validated against the same analytic reference. New reference page
  `docs/sphinx/reference/operator-modules.md` (Module vs Simulation, spaces, operators, signatures,
  capabilities / requirements, `P.call`, `dsl.Model` compatibility, migration from the PDE shortcuts);
  an "Operator-first programs" section in `time-program.md`.
- **Compiled operator introspection** (ADC-441, epic ADC-436, spec 2): `list_operators` /
  `operator_signature` / `operator_requirements` / `operator_capabilities` / `list_state_spaces` /
  `list_field_spaces` on `pops.model.Module`, `dsl.Model` and `CompiledProblem` (the compiled handle
  reads the carried model's metadata, no `.so` load). New `python/tests/test_operator_introspection.py`.
- **ModuleSpec hash** (ADC-443, epic ADC-436, spec 2): `pops.model.Module.module_hash` folds the
  spaces, parameters, aux and -- per operator -- name / kind / signature / capabilities / requirements
  and a body identity (callable source or repr), namespaced by a spec2 tag; deterministic and
  invalidated by any operator-spec change. The dsl codegen sensitivity stays with
  `dsl.Model._model_hash`; the module hash adds the operator-spec layer for a compiled module
  artifact. New `python/tests/test_module_hash.py`.
- **Operator-first standard macros** (ADC-440, epic ADC-436, spec 2): `pops.time.std`.
  `predictor_corrector_local_linear`, `explicit_rk` and `imex_local_linear` take typed operator NAMES
  (a field operator `U -> Fields`, an explicit rate `(U, Fields) -> Rate(U)`, a local linear operator
  `Fields -> LocalLinearOperator(U, U)`) and compose them with `P.call` against the bound Module. They
  are model-free (their source mentions no physics term) and reusable across any Module with matching
  signatures; an arity-aware helper calls each operator with exactly the inputs its signature declares.
  New `python/tests/test_operator_macros.py`.
- **Public `pops.model.Module`** (ADC-439, epic ADC-436, spec 2 "operator-first"): the model-free
  front-end. `Module` holds typed spaces and a registry of typed operators
  (`state_space` / `field_space` / `parameters` / `aux_fields` / `operator` as a builder or a
  decorator), with `(U, Fields) >> Rate(U)` signature sugar and `ParameterSpace` / `AuxSpace`.
  `dsl.Model` becomes the PDE convenience facade: `m.module` exposes the typed spaces and the
  derived OperatorRegistry that `source_term` / `linear_source` / `elliptic_field` / `flux`
  populate. `pops.model` is exported from the package; the same generic operator-first Program is
  reusable across any Module with matching signatures. New `python/tests/test_operator_module.py`.
- **Typed `P.call` and `m.rate_operator`** (ADC-438, epic ADC-436, spec 2 "operator-first"):
  `Program.bind_operators(model)` binds the typed registry; `P.call(name, *args, name=None)` resolves
  an operator by name, type-checks the arguments against its `Signature` (clear errors on unknown
  operator / arity / vtype / no-bind), and lowers to the matching PDE shortcut (`solve_fields` /
  `source` / `rhs` / `linear_source` / `project`) so the generated C++ is byte-identical to the Spec 1
  path. `m.rate_operator(name, flux=, sources=, fluxes=)` names a composite `-div F + sources` rate as
  a Program-side alias (lowers to the same `rhs` IR; no model-hash impact).
  `OperatorRegistry.default_of_kind` resolves a privileged default and raises a clear ambiguity error.
  New `python/tests/test_operator_call.py`.
- **Internal typed operator registry** (ADC-437, epic ADC-436, spec 2 "operator-first"):
  new `pops.model` type system (`StateSpace`, `FieldSpace`, `RateSpace`/`Rate(U)`,
  `LocalLinearOperator`, `MatrixFreeOperator`, `Signature`, `Operator`, `OperatorRegistry`)
  and a DERIVED, typed view of a `dsl.Model` via `m.operator_registry()` / `m.state_space()` /
  `m.field_space()`. The PDE shortcuts lower into typed operators -- `flux` to a `grid_operator`
  `(State) -> Rate(State)`, `source_term` to a `local_source` `(State[, Fields]) -> Rate(State)`,
  `linear_source` to a `local_linear_operator` `(Fields?) -> LocalLinearOperator(State, State)`,
  `elliptic_field` to a `field_operator` `(State) -> FieldSpace`, `projection` to a `projection`
  `(State) -> State` -- with stable integer operator ids for hot-path dispatch. Pure introspective
  view: no change to the public PDE API, the model hash, or the codegen. Foundation for `P.call`
  (operator-first Programs). New `python/tests/test_operator_registry.py`.
- **Multi-block compiled time Programs** (ADC-426, epic ADC-399, spec "Multi-blocs"):
  `Program.emit_cpp_program` lowers N `P.state("a")` / N `P.commit` -- the SSA walk allocates a base
  per block and routes every op (state, rhs, solve_fields, projection, max_wave_speed) to its block's
  runtime index, assigned in `P.state` declaration order (the System blocks must be added in that same
  order). A block declared but never committed is a read-only block (e.g. a passive field coupling the
  others through the shared Poisson); a commit of an undeclared block, and a double commit, are
  rejected. The per-block `P.solve_fields(state=Ub)` is already a coupled solve (block b at its stage
  state, every other block at its live state into the one shared phi/aux). New
  `Program.solve_fields_from_blocks([Ua, Ub])` records the simultaneous multi-target coupled solve in
  the IR but `emit_cpp_program` raises `NotImplementedError` for it (the native field solver overrides
  a single target block per solve; the coupled multi-target solve is deferred, never faked). New
  `python/tests/test_time_multiblock.py`.
- **`pops.time.std.condensed_schur` theta != 1** (ADC-427, epic ADC-399, completes ADC-421): the macro
  now lowers any `theta` in `(0, 1]`, not just backward Euler. The n+1 extrapolation by factor
  `1/theta` is expressed with the EXISTING affine algebra (no component-restricted IR op): because
  `schur_reconstruct` freezes the density, the plain state combine `U^n + (1/theta)(U^{n+theta} - U^n)`
  leaves rho untouched and matches the native momentum-only extrapolation. An OPTIONAL `c_E` energy
  component adds the native kinetic-energy increment `E += (1/2)rho(|v^{n+1}|^2 - |v^n|^2)` via a new
  `P.schur_energy` IR op (lowered to `ProgramContext::schur_energy`, reusing the native energy
  formula). `theta == 1` keeps its historical IR byte-identical (no copy / extrapolation / energy op).
  The cross-step persistent-phi carry stays deferred (it needs a 1-component history runtime path; the
  System history ring is block-ncomp); each step solves from phi^n = 0. Extended
  `python/tests/test_time_condensed_schur.py` (theta == 1 and theta == 0.5 compiled-vs-offline parity,
  the energy lowering, and a native `pops.CondensedSchur(theta=0.5)` diagnostic).
- **Named multi-elliptic-field runtime** (ADC-428, epic ADC-399; completes ADC-419): a SECOND elliptic
  solve beyond the default Poisson for a user-named field. `m.elliptic_field("phi2", rhs=, aux=[...])`
  now lowers end to end on the `production`/`system` backend -- the named field gets its own RHS brick
  (a function of the conservative state, like `m.elliptic_rhs`), a DEDICATED native elliptic solver
  instance (GeometricMG/FFT, reused, not reimplemented), and its OWN aux output channel (the model's
  named `aux_field` slots), distinct from the shared phi/grad. `P.solve_fields(field=name, state=U)`
  lowers to `ctx.solve_fields_from_state(field, block, U)` (a distinct FieldContext per field) instead
  of raising `NotImplementedError`; a source/flux reads the solved field via its named aux. The default
  Poisson path (`solve_fields` without `field=`) is byte-identical. New `System` seam
  (`solve_fields_from_state(field, ...)`, `register_elliptic_field`, `set_block_elliptic_field`) +
  `ProgramContext` overload. Deferred, rejected loud at the emit boundary (not silently dropped to a
  runtime error): the `aot` and `jit` flat-ABI backends (`emit_cpp_aot_source` /
  `emit_cpp_so_source` raise `NotImplementedError` when a named field is declared, since the loader
  macro / extern-C factory cannot register it), the `amr_system` target, and the polar (ring) named
  path. New `python/tests/test_time_multielliptic.py`.
- **Implicit-flux BDF via matrix-free Newton-Krylov** (ADC-431, epic ADC-399): `pops.time.std.bdf`
  completes the implicit-FLUX case (the globally coupled `-div F` stencil) as a pure-Python macro of
  existing primitives -- no new C++ stepper. It solves `F(U) = U - U^n - dt*rhs(U) = 0` (BDF2 adds the
  `U^{n-1}` history term) by Newton's method, each step solving `J dU = -F` with GMRES (J nonsymmetric).
  J is applied matrix-free by a new `Program.rhs_jacvec` finite-difference Jacobian-vector product, the
  codegen enabler that calls `rhs` INSIDE a `matrix_free_operator` apply sub-block (perturbing the
  frozen Newton iterate, frozen-Poisson). The final residual norm is recorded as
  `"<block>.bdf_residual"`. The cell-local linear-source fast path (`solve_local_linear`) is unchanged
  and still selected by naming a `linear_source`. New `python/tests/test_time_bdf.py` (BDF1/BDF2
  Burgers parity vs an offline Newton-Krylov on the engine's own `eval_rhs`).
- **`pops.time.std` library completion + `@P.step` decorator** (ADC-423, epic ADC-399): pure-Python
  macros that lower to the existing Program IR (no new C++ stepper) -- `std.rk` (generic explicit
  Butcher tableau; `RK4_TABLEAU` reproduces the `rk4` macro IR byte for byte, `SSPRK2_TABLEAU` gives
  Heun, the Butcher form of an SSPRK2 step), `std.lie` (sequential Lie splitting), `std.imex_local`
  (explicit flux/source + implicit
  cell-local linear source via `solve_local_linear`), `std.adams_bashforth(order in {1,2,3})`
  (`adams_bashforth2` kept as a thin alias), and `std.bdf` (cell-local-`L` BDF1/BDF2; the implicit-flux
  BDF is completed by ADC-431, see above). `Program.step` records a body from a decorated `build_fn(P)` run
  once at build time. New `python/tests/test_time_std_decorator.py`, `test_time_std_rk.py`,
  `test_time_std_imex_lie_ab.py`.
- **GMRES Krylov solver** (ADC-420, epic ADC-399): `pops.time.Program.solve_linear(method="gmres",
  restart=)` lowers to a new `pops::gmres_solve` (restarted GMRES(m), matrix-free `ApplyFn`, modified
  Gram-Schmidt Arnoldi + Givens rotations, identity preconditioner) in `generic_krylov.hpp`, alongside
  `cg_solve`/`bicgstab_solve`/`richardson_solve` and routing every inner product through the
  multi-component `krylov_dot`. `restart` is gmres-only (rejected for the others). On an SPD operator
  gmres matches the offline CG reference (~1e-15); on a non-symmetric operator (where CG stagnates) it
  converges to the same solution as BiCGStab. New `python/tests/test_time_gmres.py` + C++ coverage in
  `tests/test_generic_krylov.cpp`.
- **Per-cell non-linear local solve** (ADC-422, epic ADC-399 spec op 10): `pops.time.Program`'s
  `solve_local_nonlinear` is no longer a deferred stub. It now lowers a per-cell Newton iteration:
  the residual is built by an IR callable `residual_fn(P, U, U0)` from LOCAL ops only (named
  `P.source` / `P.apply`, the iterate / frozen guess, affine combines), and `emit_cpp_program`
  emits a device kernel that re-evaluates an inlined residual, forms an in-kernel finite-difference
  Jacobian, and solves the Newton step `J dU = -r` with the same stack dense inverse
  (`pops::detail::mat_inverse<N>`) `solve_local_linear` uses, iterating to `max_c |r_c| < tol` or the
  `max_iter` budget. The kernel is heap-free / `std::function`-free / Eigen-free; a non-local residual
  op and `n_cons > 8` are rejected loud. Reuses `pops::for_each_cell` and the `solve_local_linear`
  per-cell codegen (no flux / solver reimplementation).
- **Condensed-Schur implicit source stage as a compiled Program** (ADC-421, epic ADC-399 acceptance
  32): the anisotropic, position-dependent operator-coefficient assembly that the condensed-Schur
  operator needs. `pops.time.Program.schur_coeffs` assembles the per-cell tensor `A = I + c*rho*B^{-1}`
  (lowered to `ProgramContext::assemble_schur_coeffs`, reusing the native
  `detail::SchurOperatorCoeffKernel`), `P.apply_laplacian_coeff` applies `div(A grad phi)` matrix-free
  (`pops::apply_laplacian`'s coefficient path), `P.schur_rhs` assembles the fused
  `-Lap(phi^n) - theta*dt*alpha*div(B^{-1}(mx,my))` RHS (the native `assemble_rhs`), and
  `P.schur_reconstruct` reconstructs `v = B^{-1}(v^n - theta*dt*grad phi)` (the closed B^{-1}). The
  `pops.time.std.condensed_schur` macro composes them with `P.solve_linear` (matrix-free BiCGStab) into
  the native `CondensedSchurSourceStepper` assemble / solve / reconstruct sequence. Near-match to the
  native `pops.CondensedSchur` (which adds a GeometricMG preconditioner): same operator and RHS, a
  preconditioner-free Krylov path. The macro is the `theta == 1` (backward-Euler) source stage from
  `phi^n = 0`; cross-step phi carry, `theta < 1` extrapolation and the energy-role update are deferred
  (the macro raises for `theta != 1` rather than mis-lower). The native stepper is untouched.
- **Optional per-Program dt bound** (ADC-417, epic ADC-399 spec section 18): an
  `pops.time.Program` may declare a dt bound via `@P.dt_bound` or `P.set_dt_bound(expr)`, e.g.
  `cfl * P.hmin() / P.max_wave_speed(U)`. It builds a scalar IR sub-program (new `P.hmin` /
  `P.max_wave_speed` scalar ops plus scalar arithmetic) that the generated module exports as a second
  ABI pair, `pops_program_has_dt_bound` and `pops_program_dt_bound(ProgramContext*, cfl)`. `step_cfl`
  then uses `min(native CFL dt, program dt bound)` (a tighter bound wins, a looser one is ignored);
  without a bound the native CFL is unchanged. New `ProgramContext::hmin` / `max_wave_speed` forward
  to `System::cfl_min_dx` / `block_max_speed`, reusing the native CFL hmin and per-block wave-speed
  reduction (no reimplementation).
- **Multi-component matrix-free linear solve** (ADC-416, condensed-Schur foundation):
  `pops.time.Program.matrix_free_operator` now takes `domain` / `range_` in `{scalar, vector, state}`
  plus an `ncomp` (required, `>= 1`, for `vector` / `state`; the apply in/out buffers and the solution
  inherit it), and `P.solve_linear` validates the rhs / initial-guess component count and returns an
  `ncomp`-component solution; the codegen allocates the apply scratch / accumulator / solution with the
  operator `ncomp`. The runtime Krylov loop (`generic_krylov.hpp`) reduces its residual and CG /
  BiCGStab inner products over ALL components via a new full-component `pops::dot_all` (`mf_arith.hpp`)
  when `ncomp > 1`, so every component converges (a component-0-only norm would leave the others
  unsolved); the bare `pops::apply_laplacian` matvec now runs per component. The scalar (`ncomp == 1`)
  path is unchanged and bit-identical. New `python/tests/test_time_multicomp_solve.py`.
- **Per-cell conditional select** (ADC-418, spec section 17): the `pops.time.Program` builder gains
  `P.where(mask, a, b)` -- a PER-CELL select `out(i,j,c) = mask ? a(i,j,c) : b(i,j,c)` lowered to a
  ternary INSIDE a `for_each_cell` kernel (component-wise over the field; NOT the scalar runtime
  branch `P.if_`) -- plus `P.cell_ge` / `cell_gt` / `cell_lt` / `cell_le` (and `P.cell_compare`)
  building a per-cell 0/1 mask scalar_field from a threshold on a field's component 0. The select
  kernel is emitted purely into the generated `problem.so` (reusing the existing `ProgramContext`
  per-cell kernel pattern; no new `_pops` header). New `python/tests/test_time_where.py`.
- **Named fluxes and named elliptic fields** (ADC-419, epic ADC-399, spec "Flux nommes optionnels" +
  "Champs elliptiques nommes"): `pops.dsl.Model` gains `m.flux_term(name, x=, y=)` (an opt-in named
  physical flux, `n_cons` expressions per direction; `name='default'` aliases `m.flux(...)`) and
  `m.elliptic_field(name, rhs=, operator=, aux=)` (an opt-in named elliptic field). A compiled
  `pops.time.Program` selects fluxes via `ctx.rhs(..., fluxes=[name, ...])`: `fluxes=['default']` (or
  omitted) keeps the historical `-div F` (`ctx.rhs_into`) byte-identical, while a list of named fluxes
  assembles `-div` of their sum through the centered-FV `ProgramContext::neg_div_flux_into` (reusing
  `pops::apply_divergence` per component) -- so splitting the physical flux into named pieces that sum to
  it reproduces the same `-div F` to round-off. Both fold into the model hash only when declared (cache
  key of an existing model preserved); unknown names, dimension mismatches, `default`/named-flux mixing,
  and a default source dropped by the named-flux path are rejected with clear errors. The
  multi-elliptic-field RUNTIME (a second elliptic operator with its own aux channel) is deferred:
  `P.solve_fields(field=name)` validates + lowers to a clear `NotImplementedError`. New
  `python/tests/test_time_named_flux_elliptic.py`.
- **Program op-set completeness** (ADC-414, spec ops 10/16/21/22/23 + validation #18/#19): the
  `pops.time.Program` builder gains `P.sum` / `P.max` / `P.min` / `P.sum_component` (collective
  reductions lowered to new `pops::reduce_sum` / `reduce_max` / `reduce_min` in `mf_arith.hpp`, with
  matching `ProgramContext::sum_component` / `max_component` / `min_component`), `P.fill_boundary`
  (the shared ghost exchange, `ProgramContext::fill_boundary`), `P.project` (the block's own
  positivity projection, reusing the native `s.project` closure via `System::block_project` ->
  `ProgramContext::apply_projection`), and `P.record_scalar` (a named scalar stored in a System-side
  diagnostics map, retrievable after `sim.step` via `sim.program_diagnostic(name)` /
  `sim.program_diagnostics()`). `P.solve_local_nonlinear` is a clean `NotImplementedError` stub
  (per-cell Newton deferred; points at `solve_local_linear` for the linear case). Restart now fails
  loud when the checkpoint lacks a required Program history ("checkpoint does not contain required
  Program history '<name>'"), and `System::install_program` reports an ABI mismatch explicitly
  ("compiled program ABI mismatch: expected '<...>', got '<...>'"). New
  `python/tests/test_time_ops_polish.py`.
- **Time-program examples + docs batch** (ADC-415, epic ADC-399 acceptance 35): five new runnable
  programs in `examples/time_programs/` -- `ssprk3_program.py` (compiled SSPRK3 == native
  `pops.Explicit("ssprk3")` bit-for-bit), `strang_program.py` (compiled `std.strang` == native
  `pops.Strang` bit-for-bit on an uncoupled model), `predictor_corrector_poisson_lorentz.py` (the spec
  Example 5 named-source / Lorentz predictor-corrector vs an offline staged reference),
  `adams_bashforth2_program.py` (AB2 over the System-owned history vs an offline AB2 reference), and
  `condensed_schur_program.py` (the available divergence + matrix-free + Krylov primitives, plus the
  documented `std.condensed_schur` `NotImplementedError` gap). Each self-skips cleanly (exit 0) without
  a compiler / a visible Kokkos. `docs/sphinx/reference/time-program.md` refreshes the "What is
  implemented today" table (histories/multistep + checkpoint/restart now end-to-end; `condensed_schur`
  marked partial; rows added for substeps/stride, step_cfl routing, `solve_fields_from_state`, the
  differential primitives, reductions, and `solve_local_linear`) and `symbolic-dsl.md` gains
  `source_term` (named opt-in sources) and `linear_source` (named local linear operators, the Lorentz
  3x3) sections.
- **Divergence primitive + condensed-Schur partial** (ADC-412, acceptance 32): a centered
  finite-volume divergence is factored as `pops::apply_divergence` (next to `pops::apply_laplacian` in
  `poisson_operator.hpp`; the native Schur condensation keeps its bit-identical inline copy),
  `ProgramContext::divergence(out, fx, fy)` mirrors `laplacian`/`gradient`, and `P.divergence(out, fx,
  fy)` is a new IR op lowered to `ctx.divergence`. Since `div(grad phi) == Lap phi`, a Schur-like
  matrix-free operator `A(phi) = phi - alpha*div(grad phi)` equals `(I - alpha*Lap)` and is solved via
  `P.solve_linear` against the same offline CG reference (new `python/tests/test_time_divergence.py` +
  `examples/time_programs/divergence_solve.py`). `P.scalar_field(ncomp=)` now allocates a multi-component
  scratch (the 2-component gradient buffer the divergence consumes). The full Program rewrite of the
  condensed-Schur stage is BLOCKED on two deep IR features (multi-component `solve_linear`; anisotropic
  position-dependent operator-coefficient assembly), so `pops.time.std.condensed_schur` is a documented
  `NotImplementedError` stub naming both gaps and pointing at the still-supported native
  `pops.CondensedSchur` source stepper.
- **step_cfl routes through an installed compiled program** (ADC-413, epic ADC-399 criterion 7):
  `System::step_cfl(cfl)` now drives an installed compiled time Program. The CFL `dt` is still computed
  in adc_cpp on the native state (per-block transport / source-frequency / stability bounds + global
  bounds, UNCHANGED -- the CFL logic stays in the runtime), then the Program runs the macro-step at that
  `dt` through the SAME cadence helper as `step()` (the new `SystemStepper::run_program_cadence`, factored
  out of `step()` so both paths route a program identically: substeps + stride + clock tick, no implicit
  `solve_fields`/couplings/projections -- the Program expresses those). Previously `step_cfl` drove only
  the native per-block path and silently ignored an installed program. `step_adaptive` (multirate) still
  drives only the native path: a compiled program is one whole-system closure, so per-block subcycling
  does not apply. New test `python/tests/test_time_step_cfl.py`: `step_cfl` advances the state via the
  program, its `dt` equals the native CFL `dt`, `step_cfl(cfl) == step(dt_cfl)` bit-exact, the
  substeps/stride cadence is honored under `step_cfl`, and the clock stays coherent.
- **Substeps + stride in the compiled time program** (ADC-411): `pops.CompiledTime(substeps=, stride=)`
  no longer raises -- the macro-step cadence is wired as a SYSTEM-level orchestration AROUND the opaque
  compiled-program closure (`System::set_program_cadence`, a new `POPS_EXPORT` kept separate from
  `install_program` so the generated `.so` ABI is untouched), mirroring how the native path wraps the
  per-block advance (`stride_due` gate + substep subdivision). `substeps=n` calls the program closure
  `n` times over `eff_dt/n`; `stride=M` runs the whole program once per `M` macro-steps with
  `eff_dt = M*dt` (GLOBAL hold-then-catch-up) while the clock keeps ticking every macro-step. Default
  `1/1` is byte-identical to a single `program_step_(dt)` call. New test
  `python/tests/test_time_substeps_stride.py`: a compiled Forward-Euler program over an uncoupled
  transport-only model matches native `pops.Explicit(method="euler", substeps=2)` and a single-block
  `pops.Explicit(stride=2)` BIT-EXACTLY (the two limits where the GLOBAL/whole-program cadence equals the
  native per-block one are documented in `CompiledTime` and at the stepper seam).

- **Compiled Strang macro reproduces native pops.Strang** (ADC-410): a new test
  `python/tests/test_time_strang_parity.py` demonstrates that the compiled `pops.time.std.strang`
  combinator (the Program-IR Strang macro `H(dt/2); S(dt); H(dt/2)`) reproduces the native engine
  Strang macro-step (`SystemStepper::step_strang`) BIT-EXACTLY on a simple case: an uncoupled isothermal
  `NoSource` block advanced with Forward Euler, where native `H` is a single Euler transport step over
  `dt/2`, native `S` (`run_source_stage`) is a no-op, and the `solve_fields` fences are inert. The
  compiled `std.strang` program (installed via `sim.install_program`) and the native scheme
  (`set_time_scheme("strang")`) step `N` times and match to the last bit (`array_equal`, `max|d| = 0`),
  with an independent offline `H(dt/2); no-op; H(dt/2)` replay matching to machine precision. No new C++
  stepper and no `std.strang` change -- pure test coverage of the existing composition.

- **Per-stage elliptic field solve in the time program** (ADC-409, Phase 8): a compiled `problem.so`
  can now re-solve the Poisson fields from an arbitrary stage state, so a field-coupled multi-stage
  scheme is exact. New `System::solve_fields_from_state(block_idx, U_stage)` (`POPS_EXPORT`) assembles
  the target block's Poisson RHS from `U_stage` (the other blocks keep their live state), runs the same
  elliptic solve, and re-fills the shared aux with `phi(U_stage)`; `ProgramContext` forwards it. The
  `pops.time` codegen lowers every `solve_fields(state=...)` op to `ctx.solve_fields_from_state(0,
  <stage state>)`, so stage k's RHS reads phi solved from stage k's own state. This REMOVES the
  documented "solve from the block's current state only" limitation of `emit_cpp_program` (for an
  uncoupled model the field solve is inert, so the lowering stays bit-identical).

- **Multistep histories + Adams-Bashforth 2 in the time program** (ADC-406, Phase 7a): a compiled
  `problem.so` can declare, read, and write a history field carried across macro-steps. The history is
  SYSTEM-owned (a `HistoryManager` in `System::Impl`, ring buffer per name) rather than closure-captured,
  so a later checkpoint slice can serialize it. New `System` `POPS_EXPORT` seam:
  `register_history(name, lag)` (idempotent ring of depth `lag + 1`, co-distributed with block 0),
  `read_history(name, lag)` (throws `"history '<name>' with lag=<lag> was requested but not
  initialized"` on a read before the first store), `store_history(name, value)` (fills every slot on the
  FIRST store -- the cold start), `rotate_histories()` (shift the rings one step at end-of-step). New
  `ProgramContext` forwarders `history`/`store_history`/`rotate_histories` and `pops.time.Program` ops
  `P.history(name, lag=1)` (a State value) + `P.store_history(name, value)`; the codegen emits
  `ctx.history`/`ctx.store_history` and `ctx.rotate_histories()` last when any history is used. New std
  macro `pops.time.std.adams_bashforth2`: `U^{n+1} = U + dt*(3/2 R_n - 1/2 R_{n-1})`, with the cold start
  degenerating step 0 to Forward Euler (deterministic, machine-precision reproducible). Validated by
  `python/tests/test_time_history.py` (AB2 vs an offline recurrence to machine precision; an
  uninitialized-history read fails loud at `sim.step`).
- **Checkpoint/restart of compiled-Program histories + program-hash guard** (ADC-406, Phase 7b): a
  compiled `problem.so` with multistep histories (e.g. Adams-Bashforth 2) now checkpoints and restarts
  correctly -- the System-owned ring buffers survive the round-trip, so a continuous run is bit-for-bit
  identical to a (run, checkpoint, restart, continue) run. `sim.checkpoint` additionally records the
  installed Program's IR hash and each history ring (per name: depth, ncomp, every slot, the initialized
  flag); `sim.restart` rejects a restart against a DIFFERENT compiled Program with
  `"checkpoint was created with a different compiled Program hash"`, then restores the rings. The v1
  checkpoint format stays back-compatible (a checkpoint with no program/history keys restarts as before).
  New `System` `POPS_EXPORT` seam reusing the block-state gather/scatter machinery:
  `installed_program_hash()`, `history_names()`/`history_depth`/`history_ncomp`/`history_initialized`,
  `history_global(name, slot)` (collective gather, like `state_global`), `restore_history(name, slot,
  values)` (owner-rank scatter, like `set_state`; auto-registers the ring) and
  `set_history_initialized`. `rotate_histories()` now does O(1) `std::swap` handle rotations instead of a
  deep copy. Validated by `python/tests/test_time_history_checkpoint.py` (continuous == restart to
  machine precision; the hash mismatch fails loud).
- **Matrix-free operators + global `solve_linear` in the time program** (ADC-405, Phase 6b): a compiled
  `problem.so` can now run a matrix-free linear solve entirely C++-side via the runtime's Krylov loop
  (`pops::cg_solve`/`bicgstab_solve`/`richardson_solve`), Python only building the IR. New
  `ProgramContext` primitives (reuse, no reimplementation): `alloc_scalar_field` (a 1-component field
  co-distributed with the blocks; new `System` `POPS_EXPORT`), `geom`, `laplacian` (`fill_ghosts` +
  `pops::apply_laplacian`), `gradient` (`pops::field_postprocess`). New `pops.time.Program` ops:
  `P.matrix_free_operator(name)` + `P.set_apply(op, body_fn)` (the apply lowers to an install-time
  `pops::ApplyFn` lambda), `P.scalar_field`, `P.laplacian`/`P.gradient`, and
  `P.solve_linear(operator, rhs, method, preconditioner, tol, max_iter)` (method in
  `cg`/`bicgstab`/`richardson`; `max_iter` required and `> 0` -> `"dynamic solver loops require
  max_iter"`; `tol > 0`; non-identity preconditioner deferred). The codegen uses a two-phase template
  (persistent install-time scratch via `std::make_shared` + the step closure). The dynamic loop runs
  C++-side, invisible to the IR. Validated by `python/tests/test_time_solve_linear.py`: a compiled
  `(I - 0.1*Lap)phi = U` CG solve matches an offline numpy CG on the same discrete periodic system to
  1.78e-15. (`condensed_schur` as a Program macro is a later slice built on these primitives.)
- **Structured for-loops + if + norm_inf in the time-program codegen** (ADC-404, Phase 5b):
  `pops.time.Program` gains `P.static_range(state, count, body)` (a COMPILE-TIME unrolled loop -- `count`
  copies of the body inline, no C++ loop), `P.range(state, count, body)` (a C++ `for` over a fixed
  count, the body emitted once and re-run each pass), `P.if_(state, cond, body)` (a C++ `if` branch on
  a runtime `Bool`), and the `P.norm_inf(state)` reduction (`-> pops::norm_inf`). A runtime `Scalar`
  count is rejected loud (`static_range` -> TypeError, `range` -> NotImplementedError); the IR hash
  distinguishes loop counts and unrolled bodies (distinct `.so` cache keys). Validated by
  `python/tests/test_time_control_flow_b.py` (the unrolled-vs-loop codegen + a dt-free contraction run
  matching the offline closed form). `ctx.where` (per-cell mask) is a later slice.
- **Control flow + reductions in the time-program codegen** (ADC-404, Phase 5): `pops.time.Program`
  gains `Scalar`/`Bool` IR value types, the reductions `P.norm2(state)` and `P.dot(a, b)` (lowered to
  the collective `pops::dot`, an MPI all-reduce), scalar comparisons (`>`, `<`, `>=`, `<=` -> a `Bool`),
  and `P.while_(state, cond, body)` which lowers to a C++ convergence loop in the generated closure
  (an infinite loop with a break that RE-EVALUATES the condition each pass and mutates the loop-variable
  state in place; the condition and body ops are recorded in a separate sub-block so the top-level SSA
  invariants hold). A runtime `Scalar`/`Bool` can no longer silently collapse to a Python value:
  `bool(scalar)` and `range(scalar)` raise loud (`use P.while_ / P.if_ ...`). The convergence loop
  runs entirely C++-side with a runtime-dependent iteration count and matches an offline geometric
  reference to machine precision (`python/tests/test_time_control_flow.py`). `ctx.range` (static/dynamic
  for) and `ctx.if_` are a follow-up (ADC-404b).
- **Named-source RHS in the time-program codegen + predictor-corrector** (ADC-403, Phase 4): a
  `P.rhs(..., sources=["electric", ...])` now lowers to `-div F` plus the requested named
  `source_term`s (each assembled by the same per-cell kernel as `P.source`), so a compiled Program can
  build `-div F + S_named` without folding sources implicitly. Validated: an unknown source name
  raises `"unknown source_term '<name>'"`, and a named-source RHS on a model that also has a default
  source is rejected (deferred; avoids double-counting). The full predictor-corrector Poisson/Lorentz
  program (spec example 5) now compiles + runs and matches an offline replay of the same steps to
  machine precision (`python/tests/test_predictor_corrector.py`). Closes ADC-403. (Per-stage field
  re-solve from a stage state still awaits `solve_fields_from_state`, a later phase.)
- **`pops.time.Program` split-source / local-linear-operator IR ops** (ADC-403, Phase 4): `P.source`
  (a single named model source as an RHS-like value), `P.linear_source` (reference a model
  `m.linear_source` operator), `P.apply` (`LU = L U`), and `P.solve_local_linear(operator=P.I - a*L,
  rhs=, fields=)` for a cell-local implicit solve, with an operator algebra (`P.I`, `a * L`,
  `I +/- a*L`). These build typed IR (the predictor-corrector Poisson/Lorentz program from the spec
  validates); a non-local or non-linear operator is rejected (`"solve_local_linear currently supports
  local linear operators only"`). Codegen lowering of these ops is a follow-up, so
  `emit_cpp_program` still refuses them with a clear `NotImplementedError`.
  `python/tests/test_time_local_solve.py`.
- **Multi-stage compiled time programs (SSPRK2, SSPRK3, RK4)** (ADC-399 / ADC-407):
  `Program.emit_cpp_program` now lowers any single-block multi-stage scheme by a topological SSA walk
  -- intermediate stage states become zero-initialized scratch MultiFabs accumulated with `axpy`, and
  the committed stage writes the block state via `lincomb` -- so SSPRK2/SSPRK3/RK4 compile and run
  C++-side with no per-scheme class. `ProgramContext` gains `scratch_state_like` + `lincomb` (both
  forward to existing primitives; no reimplementation). Verified: a compiled SSPRK2 Program reproduces
  native `pops.Explicit("ssprk2")` bit-for-bit, and compiled SSPRK2/RK4 match an offline stage-by-stage
  reference to machine precision (`python/tests/test_time_multistage.py` + the `examples/time_programs/`
  ssprk2/rk4 scripts). Field-coupled multi-stage (re-solving the elliptic fields from each stage state)
  still needs `solve_fields_from_state` (a later phase) and is documented as such; uncoupled models are
  exact. Constructs not yet lowerable (multi-block, named sources beyond `default`) raise a clear error.
- **Compiled time-program reference doc + runnable example** (ADC-407, Phase 8b):
  `docs/sphinx/reference/time-program.md` explains the Model-vs-Program split, the builder API,
  `compile_problem` / `install_program` / `CompiledTime`, and a status table (Forward Euler runs
  end to end; multi-stage / split-sources / control-flow / Krylov / histories are tracked as
  follow-ups). `examples/time_programs/forward_euler_program.py` compiles + installs + runs a
  Forward-Euler `Program` and checks parity with the native `pops.Explicit("euler")` step.
- **`pops.compile_problem` + `sim.install_program` + `pops.CompiledTime`: run a compiled time Program
  end to end** (ADC-401, Phase 2c-ii): `compile_problem(model=, time=)` lowers an `pops.time.Program`
  to C++ (`emit_cpp_program`) and compiles it into a `problem.so` against the pops headers with the
  SAME Kokkos toolchain as the loaded `_pops` (reusing `pops.dsl.pops_loader_build_flags`), so the `.so`
  is ABI-compatible and loads in-process; it returns a `CompiledProblem` handle and caches the `.so`
  out-of-source keyed by [program source + header signature + compiler + std]. `sim.install_program`
  is now exposed to Python (it wraps the C++ loader; ABI mismatch / dlopen failure surface as
  `RuntimeError`). `pops.CompiledTime` is the compiled-Program time policy (MVP: single Forward-Euler
  step; `substeps`/`stride` > 1 and a non-default `cfl` raise `NotImplementedError`, deferred to Phase
  2c). `compile_problem` rejects `backend != "production"` and `target != "system"`.
  `python/tests/test_compile_problem.py` checks the rejections and the parity of a compiled
  Forward-Euler Program against the reference one-step `U0 + dt*eval_rhs`.
- **`pops.time.Program.emit_cpp_program`: lower the IR to a `problem.so` source** (ADC-401, Phase
  2c-ii codegen): generates the C++ of a compiled time Program -- the stable `.so` ABI
  (`pops_program_abi_key` via the `POPS_ABI_KEY_LITERAL` literal, `pops_program_name`,
  `pops_program_hash`, `pops_install_program`) plus the step expressed purely through `ProgramContext`
  primitives (`solve_fields` + `rhs_into` + `axpy`), the source `System::install_program` compiles
  and runs. MVP lowers single-block Forward Euler; multi-stage (needs scratch states) and named
  sources beyond `default` (need source masks) raise `NotImplementedError`, never a silent
  mis-lowering. `python/tests/test_time_codegen.py` pins the generated ABI + algorithm and the refusals.
- **`System::install_program`: load a compiled time-program `.so`** (ADC-401, Phase 2c-i): the runtime
  path that drives a compiled time Program from a generated `problem.so`. `install_program(so_path)`
  dlopens the `.so`, checks its `pops_program_abi_key` against the module (fail-loud on mismatch), and
  calls its `pops_install_program(this)`, which wraps the System in a `ProgramContext` and installs the
  macro-step closure (mirroring `add_native_block`: self-promote to the global scope so the `.so`
  resolves the `POPS_EXPORT` seam accessors). `tests/test_program_loader.cpp` compiles a stub Forward-Euler
  `problem.so` at runtime, loads it, runs `sim.step(dt)`, and asserts bit-parity vs the reference -- the
  full dlopen + ABI guard + symbol-resolution chain. Auto-skips under Kokkos / without a compiler. The
  Python `compile_problem` codegen that GENERATES the `.so` from a `Program` IR is Phase 2c-ii.
- **`pops.time.std`: standard library of time-stepping macros that lower to the Program IR** (ADC-407,
  Phase 8a): `forward_euler`, `ssprk2`, `ssprk3`, `rk4` and a `strang` splitting combinator are Python
  functions that BUILD `pops.time.Program` IR via the same builder ops and the affine algebra over `dt`
  -- a scheme is expressed once with no scheme-specific class (RK4 has no special RK4 class). They
  reproduce `pops.Explicit(method="euler"/"ssprk2"/"ssprk3")` at the IR level; a generated `problem.so`
  (`compile_problem`, Phase 2c) will execute the lowered IR. Tested via IR-structure assertions on the
  committed state's per-input coefficient polynomials; parity vs the old C++ steppers needs
  `compile_problem` and is deferred.
- **C++ `ProgramContext` runtime seam for compiled time programs** (ADC-401, Phase 2b): a new
  `pops::runtime::program::ProgramContext` facade (`include/pops/runtime/program/program_context.hpp`)
  lets a generated `problem.so` run a compiled time Program entirely C++-side during `sim.step(dt)`,
  reusing the existing runtime primitives and reimplementing nothing. `System` gains public seam
  accessors `install_program_step` / `n_blocks` / `block_state` / `block_rhs_into` (plus an
  `Impl::program_step_` member), and `SystemStepper::step` dispatches to the installed Program when
  present while keeping `t` / `macro_step` coherent -- the historical path is byte-for-byte unchanged
  when no program is installed. `tests/test_program_runtime.cpp` installs a Forward-Euler Program via
  `ProgramContext` and runs it through `sim.step(dt)`, asserting bit-identity with a reference step
  computed from the same primitives (`solve_fields` + `eval_rhs` + `U + dt*R`). `step_cfl` /
  `step_adaptive` with a Program, codegen, and the Python `compile_problem`/`install_program` wiring
  follow in Phase 2c.
- **`pops.time.Program`: builder-mode IR for compiled time programs** (ADC-401): a new `pops.time`
  module exposing `Program`, the central abstraction of the compiled time-program DSL (ADC-399).
  Python builds a typed SSA IR -- `state`, `solve_fields`, `rhs(flux=, sources=)`, `linear_combine`,
  `commit` -- with an affine algebra over the time step (`U + dt*R`, `0.5*U0 + 0.5*(U1 + dt*k1)`,
  `dt/6.0*k1`) that records per-input coefficient polynomials in `dt`, so Forward Euler, SSPRK2/SSPRK3
  and RK4 are all expressed without any scheme-specific class. Structural validation (each block
  committed at most once, at least one commit), a deterministic coefficient-sensitive IR hash (future
  cache key), and a guard that an IR value cannot be used as a Python bool. IR construction only: no
  codegen and no C++ runtime yet (those are later ADC-399 phases). First slice of Phase 2 (ADC-401).
- **Named physical-model sources: `m.source_term` and `m.linear_source`** (ADC-400): the DSL `Model`
  gains opt-in named local sources. `source_term(name, exprs)` declares a named `S_name(U, prim, aux,
  params)` with `n_cons` components; `linear_source(name, matrix)` declares a named `n_cons x n_cons`
  linear operator `L_name(aux, params) U` whose coefficients stay independent of U/primitives. They
  are the physical building blocks the compiled time-program DSL (ADC-399) will consume and are never
  summed implicitly. `m.source([...])` stays backward compatible and equals `source_term("default",
  ...)`. Validation rejects wrong dimensions, empty/duplicate/colliding names, and U/primitive
  dependent linear-source coefficients; an old stepper asking for the total source of a model that
  only declares named sources is rejected rather than silently summing them. The model hash folds the
  named sources in ONLY when present, so a model that never declares one keeps a byte-identical `.so`
  cache key, while changing a named source or a linear-source coefficient invalidates the cache.
  Python-only declaration/validation/hash (no codegen yet); first slice of the ADC-399 epic.
- **`.github/CODEOWNERS` routes review by zone** (ADC-252): the owner of a touched directory is
  auto-requested as reviewer. Solo today, `@wolf75222` owns every zone; the per-zone split documents
  the responsible owner per area and prepares reviewer routing at scale. This is the SWE-at-Google ch.10
  "Ownership" bit of the review process; turning it into a merge gate ("require review from Code
  Owners") is the branch-protection decision (ADC-166). CONTRIBUTING "Code review" links to it.
- **Regression guard: the red-black GS smoother reuses the cached halo schedule** (ADC-262):
  `tests/test_poisson_smoother_cache.cpp` asserts that a multi-level GeometricMG V-cycle builds the
  halo schedule once per MG-level layout and adds ZERO further builds across subsequent cycles, so the
  RB-GS smoother (the 86%-dominant Poisson path) does not re-enumerate the exchange jobs per sweep.
  This locks the lever-(a) win that ADC-260 (the `fill_boundary` halo-schedule cache) already shipped.
  A `bench/profile_step` measurement (serial, n=256, geometric_mg) confirmed the remaining Poisson cost
  is algorithmic (GS V-cycle iterations: 164 ms/step vs 10 ms/step with the FFT solver), not the
  per-sweep `fill_ghosts`, so on the SERIAL path the proposed face-subset/fused-exchange lever (b) is
  not worth it (sub-1% saving, and it would break bit-identity). Note: the ADC-260 cache hoists only
  the schedule ENUMERATION; under MPI the per-sweep pack/exchange still runs every sweep (the bottom
  level alone is ~100 exchanges per V-cycle), so the distributed-exchange lever (b) is NOT addressed
  here and is left as a follow-up. Test only; no behavior change.
- **`roe_abs_apply`: device-clean matrix-absolute-value for a generic moment Roe** (ADC-368): a new
  `pops::roe_abs_apply<N>(A, dU, out)` in `include/pops/numerics/dense_eig.hpp` returns the Roe
  dissipation `|A| dU = R |Lambda| R^-1 dU` of a small dense flux Jacobian via the
  infinity-norm-scaled Newton matrix-sign iteration (`|A| = A sign(A)`); `POPS_HD`, no allocation,
  `N <= 16`. For a real-diagonalizable `A` this equals the reference `flux_ROE_local.m` dissipation
  exactly (the Harten floor is inactive at O(1) wave speeds), and it returns `false` -- so the caller
  falls back to a spectral-radius (Rusanov) bound -- when the spectrum is not real or `A` is singular.
  This is the numerical kernel for the upcoming generic moment-system Roe flux (HyQMOM15); no public
  behavior changes yet. Covered by `tests/test_dense_eig.cpp` (parity to `R |Lambda| R^T` up to N=15,
  plus the complex and singular fallbacks).
- **Generic moment Roe flux: `m.roe_from_jacobian()` + `build_moment_model(roe=True)`** (ADC-368):
  a DSL emitter that makes `riemann='roe'` available for a moment hierarchy (HyQMOM) with NO fluid
  roles and NO primitive `p` (unlike `m.enable_roe`). It emits the `roe_dissipation` hook =
  `|A| (UR - UL)` with `A = dF_dir/dU` the autodiff flux Jacobian evaluated at the arithmetic-mean
  interface state, `|A|` via `pops::roe_abs_apply` (spectral-radius Rusanov fallback on a
  complex/singular spectrum) -- the reference `flux_ROE` dissipation. It is the third (mutually
  exclusive) provider of the `roe_dissipation` hook and folds into the model cache key; without a
  call nothing is emitted (bit-identical). `pops.moments.build_moment_model` gains a `roe=False`
  option (additive to `exact_speeds`, which still supplies `max_wave_speed`). This also fixes a
  latent core gap: `SourceFreeModel` (the IMEX explicit-half-step wrapper) now forwards the Roe/HLLC
  capability hooks (`roe_dissipation` / `contact_speed` / `hllc_star_state`), which it previously
  dropped (it forwarded only `pressure` / `wave_speeds`) -- so a non-Euler model on `riemann='roe'`
  (or `'hllc'`) lost the hook through the IMEX path and fell back to the Euler-4var branch, a compile
  error for `n_vars != 4`. Covered by `python/tests/test_dsl_roe_from_jacobian.py` (codegen + AOT
  compile + `System` `riemann='roe'` 10-step mass conservation + rejection without the capability).
- **Configure-time guard: Kokkos CUDA is rejected on native Windows** (ADC-168): `cmake` now fails
  fast with a clear message (use WSL2 for the GPU; native Windows is CPU Serial/OpenMP only) when
  `Kokkos_ENABLE_CUDA=ON` is requested on a native Windows configuration, instead of letting the
  unsupported Kokkos CUDA build break deep inside configure. The supported GPU path stays WSL2
  (ADC-96); Linux, macOS, and WSL2 are unaffected.
- **Per-field configurable aux halos for named aux** (ADC-369, follow-up of ADC-291): a model-declared
  named aux field can now carry its OWN ghost boundary policy via `pops.AuxHalo("foextrap")` /
  `pops.AuxHalo("dirichlet", value=v)` passed to `set_aux_field(block, name, field, halo=...)`, instead
  of inheriting the single shared phi-derived aux BC. The policy is applied to the NON-PERIODIC faces
  only (periodic faces -- a periodic domain, the polar theta direction -- keep their wrap, so a per-field
  policy never breaks the domain periodicity). Implemented by a component-scoped `fill_physical_bc`
  overload + `aux_halo_override` (`include/pops/mesh/physical_bc.hpp`), applied after the shared aux fill
  in `SystemFieldSolver` (cartesian + polar), `AmrRuntime` and `AmrCouplerMP` (coarse level). For the
  cartesian System COMPILED-block path the policy is carried to the `.so` via an append-only tail on the
  marshaled aux array (read by `compiled_block_abi.hpp` `make_grid`); the header signature (`abi_key`)
  bumps automatically so a stale `.so` is never mixed. `pops.capabilities()['aux']['named']['halo_policy']`
  reports it. Default (no halo) is strictly bit-identical; canonical fields (`B_z`, `T_e`) unchanged.
  Per-face asymmetric BC and the JIT host path remain follow-ups.
- **Named aux phase 2: AMR, polar, and a declarative `kAuxMaxExtra`** (ADC-291): model-declared named
  auxiliary fields (`m.aux_field("name")`, component `kAuxNamedBase + k`, read via `aux.extra_field(k)`)
  now work beyond the cartesian `System`. The polar `System::add_block` path widens the shared aux
  channel (`ensure_aux_width`), fixing a silent out-of-bounds read for any polar model with `n_aux > 3`
  (e.g. the polar Lorentz block) and unlocking `set_aux_field` / `aux_field` on `PolarMesh`.
  `AmrSystem.set_aux_field(block, name, array)` carries a static named field through both AMR engines
  (single-block `AmrCouplerMP`, multi-block `AmrRuntime`); the field is re-applied each `compute_aux` /
  `solve_fields` so it persists across a regrid and is injected to every level. The JIT host residual
  (`native_loader::host_residual`) now marshals named fields too (previously read as `0`). A new
  host-side C++ canonical name table (`include/pops/core/aux_names.hpp`) mirrors the Python
  `AUX_CANONICAL`, and `pops.capabilities()['aux']` is restructured (backends list, introspectable
  `limit`, `halo_radius`) and reads `kAuxMaxExtra` from the single C++ source so the Python mirror
  cannot silently drift. The only remaining compile-time aux limit (`kAuxMaxExtra`) is now declarative
  (`kAuxMaxComps` + static_asserts) and tested (`test_aux_names`, `test_native_aux_named`,
  `test_capabilities`, polar/AMR end-to-end in `test_aux_named.py`). Default paths stay bit-identical
  (empty named-aux map). The per-field configurable aux halo radius remains a documented follow-up.
- **Generic embedded-boundary / level-set domain contract** (ADC-327): the cut-cell / mask transport
  geometry is now expressed through a named, device-clean POD contract in
  `include/pops/numerics/embedded_boundary.hpp` (`level_set(x, y) < 0` inside, callable `operator()`,
  `cell_active`; built-in instances `DiscDomain` and the non-disc `HalfPlaneDomain`; a
  diagnostics-only `LevelSetDomain` concept that never constrains the hot-path templates).
  `numerics/spatial_operator_eb.hpp` no longer depends on the runtime `wall_predicate.hpp`. The
  Python helpers `set_disc_domain` / `set_geometry_mode` / `disc_mask` keep their name, signature and
  behavior (the disc is now one instance of the contract); the internal System fields are renamed
  `disc_ -> eb_domain_`, `disc_set_ -> eb_set_`, `disc_mask_ -> domain_mask_`, and runtime errors now
  speak of an embedded boundary / level-set domain. Default path strictly bit-identical;
  `tests/test_embedded_boundary_generic.cpp` locks the non-disc path end to end.
- **Generic real/complex spectrum predicate in `dense_eig.hpp`** (ADC-276): `pops::real_spectrum<N>()`
  classifies a small dense block as `Spectrum::kReal` / `kComplexPair` / `kUnknown`, with
  `EigBounds::all_real()` / `has_complex_pair()` accessors. The imaginary tolerance is relative
  (`im_tol * max(|lmin|, |lmax|, 1)`, default `1e-5`, covering a real multiplicity up to 3 -- the 3x3
  target -- since eps^(1/3) ~ 6e-6), so a quasi-degenerate real spectrum is not mislabeled complex, and
  non-convergence maps to `kUnknown` (never read as real) -- letting a native device projector test
  realizability on, e.g., a 3x3 HyQMOM15 block without NumPy or MATLAB. Header-only and additive:
  `real_eig_minmax` / `EigBounds` layout and the DSL eig path are unchanged.
- **DSL surface for the real/complex spectrum predicate** (ADC-362): `dsl.eig_all_real(rows,
  im_tol=1e-5)` exposes ADC-276's `EigBounds::all_real()` as a branchless DSL value (`1.0` if the small
  dense block's spectrum is real and the block converged, `0.0` otherwise -- complex pair or
  non-convergence), so a `m.projection` mask can ask "is this block's spectrum real?" without a branch.
  It lowers to a named device-clean functor returning `pops::Real(pops::real_eig_minmax(M).all_real(
  im_tol))`; gating on `converged` keeps a Gershgorin fallback at `0.0` (never read as real), unlike a
  raw `eig_max_im <= tol` whose `max_im` is `0` under fallback. The host NumPy mirror (LAPACK always
  converges, so no `kUnknown`) coincides with the generated brick on healthy matrices. Additive to
  `dsl.py`; the scalar `eig_max_im` / `eig_lmin` / `eig_lmax` path is bit-identical
  (`python/tests/test_projection_eig_predicate.py`).
- **Public `System.set_source_stage` on the Python facade** (ADC-308): the Schur-condensed source
  stage, already wired internally by `add_equation(time=pops.Split(source=pops.CondensedSchur(...)))`, is
  now reachable as a public `pops.System.set_source_stage(name, kind, theta, alpha, ...)` method (a thin
  pass-through to the binding with the same flat signature and defaults), so cases configure a post-hoc
  source stage without reaching into the private `_s`. Purely additive: the public call is
  bit-identical to the historical `_s.set_source_stage` path
  (`python/tests/test_set_source_stage_facade.py`).
- **One-command Python build** (ADC-358): `scripts/build_python.sh` activates the `pops` env, sizes the
  heavy-TU Ninja pool (`POPS_HEAVY_TU_POOL`) from cores capped by RAM, exports the Kokkos/CMake
  discovery vars (`Kokkos_ROOT`, `POPS_KOKKOS_ROOT`, `CMAKE_PREFIX_PATH`) and a stable cross-worktree
  ccache, runs `pip install . --no-build-isolation`, and ends on `pops.doctor()`; `--clean` drops the
  wheel cache, `--fresh` also clears ccache for a cold build. `scikit-build-core` is now pinned in
  `environment.yml` (so `--no-build-isolation` reuses the pinned stack) and `setup_env.sh` persists
  `CMAKE_PREFIX_PATH`. No change to `-O3` or generated code.
- **Prebuilt macOS arm64 wheel** (ADC-360): a `Wheels` workflow (`.github/workflows/wheels.yml`) builds a
  self-contained macOS arm64 / CPython 3.12 `_pops` wheel with `cibuildwheel` (Kokkos Serial built
  static + PIC and linked in; `delocate` bundles the rest), uploads it as a build artifact on
  build-relevant PRs, and attaches it to the GitHub Release on a `vX.Y.Z` tag. End users can then
  `pip install adc_cpp-*.whl` with no local toolchain. Separate from the required CI `gate`; no
  source or runtime behavior change. Linux/Windows wheels and PyPI publishing are follow-ups.
- **Configurable AMR regrid variable by name or role** (ADC-296): `AmrSystem.set_refinement` gains
  optional `variable=` / `role=` selectors so the multi-block union-of-tags regrid can refine on any
  conserved variable, not just component 0. Each block resolves the selector against its own conserved
  variables (`detail::resolve_selected_component`, STRICT: an absent name/role raises at build, no
  silent component-0 fallback), so a model whose refinement variable is not at component 0 refines
  correctly. The default (empty selector) stays component 0 and bit-identical; a non-default selector is
  multi-block only (mono-block `AmrCouplerMP` and the compiled `.so` loader refine on component 0 only
  and reject it). Surfaced under a new `regrid` key in `pops.capabilities()`. Per ADR-0001 Decision 5
  (Option A): the engine seam (`TagPredicate`, per-block predicates, union) is unchanged.
- **2D-core invariant published in `pops.capabilities()`** (ADC-294): `capabilities()` now exposes a
  structured `dimension` scalar (`== 2`) declaring the core's two-dimensional scope as an
  introspectable invariant, with a matching "Spatial dimension" section in
  `docs/sphinx/reference/known-limitations.md` and a cross-link in `include/pops/mesh/box2d.hpp`.
  Per ADR-0001 Decision 1 (Option A): purely additive, no API or ABI change;
  `python/tests/test_capabilities.py` pins the key. The ND core (`BoxND` / `GeometryND`) stays
  deferred to a future milestone.
- **Varying-kappa coverage for the screened-Poisson reaction term** (ADC-251):
  `tests/test_screened_poisson.cpp` gains an MMS case with a spatially varying `kappa(x,y)`, run
  through both `GeometricMG::set_reaction` overloads (`fn` and `MultiFab`). The pre-existing cases
  used only a constant kappa, so the per-cell diagonal read was unverified; order-2 convergence here
  proves the `(i,j)`-only read with 0 ghost cells is correct and sufficient, locking the deliberate
  no-ghost-fill invariant against any future stencil that would read kappa on its unfilled ghosts.
  Tests and docs only; the kappa wiring and the hot path are unchanged.
- **Multi-rank test for the collective IO gather** (ADC-257): `tests/test_mpi_system_io_gather.cpp`
  exercises `System::density_global` / `state_global` / `potential_global` (the `all_reduce_sum`
  gather behind `sim.write` / `sim.checkpoint`) under `mpirun -np {1,2,4}` -- previously covered only
  mono-rank by `python/tests/test_io_multirank.py`, which deferred the `np>1` case to a C++ file that
  did not exist. The test pins gather == known reference (bit-identical, np-invariant, catching any
  double-count), gather == local accessor on the owning rank after collective steps, and a
  checkpoint/restart round-trip that is bit-identical at np=2/4. Tests only; no change to the library,
  the API, or the hot path.
- **hyqmom15 AmrSystem multi-box GH200 validation** (ADC-320): a domain-decomposed driver
  (`docs/validation/diocotron_amr_gpu.cpp` + `diocotron_amr_mpi.sbatch`) wiring the compiled hyqmom15
  composite (`Hyqmom15Hyp/Src/Ell` + Poisson `geometric_mg`) on `AmrSystem` with
  `distribute_coarse=True`, so at `np > 1` the coarse `fill_boundary` exchanges the 15 conserved
  moments between GPUs (the real inter-GPU halo path that ADC-181's mono-box round-robin never
  exercised). Records the np=1/2/4 parity in `docs/validation/GH200_HYQMOM15.md` (cmax bit-identical,
  sums at last ulp, `coarse_local_boxes < coarse_total_boxes` proving the coarse distributed).
  Validation only; no change to the library, the API, or the hot path.
- **Coarse-level MPI ownership diagnostic** (ADC-319): `AmrSystem.coarse_local_boxes()` returns the
  number of base (level-0) boxes owned by the calling rank and `AmrSystem.coarse_total_boxes()` the
  total across all ranks. With `distribute_coarse=True` a distributed base gives `local < total` per
  rank; a replicated or single-box base gives `local == total` everywhere. Wired through the mono-block
  (`AmrCouplerMP`) and multi-block (`AmrRuntime`) paths, exposed in the pybind bindings and the Python
  facade. A general MPI strong-scaling diagnostic; no change to the numerics or the hot path.
- **Positivity floor on the AMR transport** (ADC-259): `AmrSystem.add_block` / `add_equation` now
  honor `spatial.positivity_floor > 0` (Zhang-Shu), previously rejected on the AMR path. The floor is
  threaded through both engines (single-block `AmrCouplerMP`, multi-block `AmrRuntime`) into
  `compute_face_fluxes` (Density-role face states) and adds a Density clamp on the coarse-fine fine
  ghost means (`fill_cf_ghost_cell`), the refined-patch interface the diocotron Hoffart failure
  exercised. Guarantee = face / C/F-ghost-mean Density positivity only (order-1 fallback), NOT
  updated-mean nor pressure positivity (parity with `System`). `positivity_floor == 0` is bit-identical
  to before; a model without a Density role rejects `positivity_floor > 0` explicitly.
- **Positivity floor on the compiled AMR `.so` path** (ADC-322): the production DSL loader
  (`pops_install_native_amr`) now marshals `positivity_floor` as a trailing flat argument, threaded
  through `add_compiled_model` / `set_compiled_block` into the same `compute_face_fluxes` leaf the native
  path uses (mono via `AmrBuildParams::pos_floor`, multi via a new `AmrCompiledBlockBuilder` slot). A
  `CompiledModel(target='amr_system')` block built with `positivity_floor > 0` now floors instead of
  raising (`AmrSystem.add_equation` / `add_native_block`); `positivity_floor == 0` stays bit-identical. A
  loader regenerated against pre-floor headers is rejected at load by the ABI key (the header signature
  changed), so the 9-argument call never reaches a stale 8-argument `.so`. Follow-up to ADC-259.
- **Distributed FFT Poisson under MPI** (ADC-287): `System.set_poisson(..., "fft"|"fft_spectral")` now
  runs with `n_ranks() > 1` via a box-slab remap (`RemappedFFTSolver`), replacing the previous explicit
  rejection. The new solver presents the System single round-robin box outward (so the field-solve path
  is unchanged) and hides a scatter/gather around `PoissonFFT` inside `solve()`. Periodic-only, constant
  coefficient, requires `Ny % n_ranks() == 0`; the potential matches `geometric_mg` to FP tolerance.
  `geometric_mg` stays the MPI default and the only option for walls, variable/anisotropic eps, or kappa.
  Ratified by the ADC-273 multi-agent design vote (correctness sound on every load-bearing axis; the
  elliptic `ell_` variant is not serialized, so no public-ABI break).
- **BGK collision helpers** (ADC-277): `pops.moments.maxwellian_moments` builds the local
  Maxwellian equilibrium moments of a 2D moment hierarchy (Isserlis closure, generic in the
  order and closure-free), and `pops.moments.bgk_source` returns the relaxation source
  `nu (M_eq - M)` toward it. Both work as DSL expressions or as a numeric oracle, and conserve
  mass and momentum exactly (the M00/M10/M01 rows are identically zero). BGK is meant to be
  wired through the existing source brick (`m.source` / `m.source_frequency`, explicit split or
  IMEX), so it adds no core trait, kernel, or stepper path. Strictly additive: the
  `build_moment_model` signature is unchanged. `Model.eval_source` (numpy source evaluator,
  parity with `eval_flux`) lets a host test check the emitted source without compiling.
- **Multi-GPU + MPI hyqmom15 diocotron validation harness** (ADC-181): the `docs/validation`
  diocotron driver gains an optional MPI bootstrap (`comm_init`/`comm_finalize` + rank-0 I/O
  guards) behind the new `POPS_VALIDATION_MPI` CMake option, so the same `diocotron_gpu.cpp`
  runs serial (unchanged at np=1) and under `srun -n N`. New `diocotron_mpi.sbatch` is the ROMEO
  GH200 recipe: build with CUDA-aware OpenMPI, run np=1/2/4 (one GH200 per rank), gate on per-run
  mass conservation (< 1e-12) and ulp-level global-mass parity vs np=1. Closes the System-MPI
  branch named in `GH200_HYQMOM15.md` section 3.
- **Coverage for the `step_cfl` zero / NaN wave-speed guard** (ADC-267): `tests/test_cfl_dt.cpp`
  gains two in-file cases (no new target) for the `std::max(w_max, 1e-30)` clamp that ADC-194 left
  untested. A quiescent state (`w_max == 0`) asserts `dt` is finite and equals `cfl*h/1e-30` (without
  the floor it would be `+inf`), and a model with `max_wave_speed() == NaN` asserts `cfl_dt` stays
  finite and positive (`system_max_wave_speed` does `max(0, NaN) == 0`, swallowing the NaN). Test-only;
  no change to the library, the API, or the hot path.

### Changed
- **Spec 4 s23: user-facing examples migrated to install()** (ADC-489): the `examples/time_programs/` walkthroughs and `examples/operator_modules/predictor_corrector_operator_first.py` now wire their compiled-program simulation through the unified `sim.install(compiled, instances={...}, aux=, solvers=)` headline entry instead of the manual `add_block`/`add_equation` + `set_poisson`/`set_magnetic_field` + `set_state` + `install_program` sequence; the native reference systems keep the lower-level API (still the documented building blocks). Behavior is unchanged -- each example reproduces its prior result to machine precision.
- **Spec 4 s23: install() serves native sims** (ADC-489): `sim.install(compiled=None, instances={name: {model, time, initial, spatial}}, aux=, solvers=)` now builds a NATIVE simulation -- it skips `install_program` (there is no compiled Program) and drives the blocks via the native per-block advance loop, so install() is the ONE entry for BOTH the compiled-program and native modes. Each native instance carries its own model + time policy; `cadence=` is rejected in native mode (set substeps/stride on `pops.Explicit` instead). The native install() is bit-identical to the manual add_equation/set_state sequence.
- **Spec 4 s23: install(cadence=)** (ADC-488): `sim.install(compiled, ...)` gains a program-level `cadence=pops.CompiledTime(substeps=, stride=)` kwarg, applied with `set_program_cadence` after `install_program`. This completes the unified install() coverage of every amendment setter (a compiled Program is one whole-system closure, so cadence is GLOBAL -- a single kwarg, not per-instance). A bad type or a non-default `cfl` is rejected up front.
- **Spec 4 s7: drop the `std` time-scheme bundle** (ADC-467): `pops.lib.time.std` is removed; the ready schemes are called by their explicit names (`pops.lib.time.ssprk3`, `pops.lib.time.strang`, ...). Section 7 bans the `std` name outright (not just the `std/` directory), enforced by `tests/architecture/test_no_forbidden_paths.py::test_no_std_name_anywhere`. The internal Spec-3 macro catalog (`pops.lib.time.macros`) now forwards to those names directly.

- **Spec 5 central catalogs leave `pops.lib`** (ADC-480, epic ADC-479, sections 4 / 5 / 5.15): the generic discretisation / moment / diagnostic building blocks move out of `pops.lib` (which Spec 5 reserves for ready-to-use presets) into top-level central packages. New homes: `pops.numerics` (`riemann` / `reconstruction` + `limiters` / `projections`), `pops.moments` (the moment-model toolkit), `pops.diagnostics` (the reduction catalog), and `pops.descriptors` (the shared `BrickDescriptor` + factories + external-brick loader). Each catalog now also exports its entries at module scope, so the Spec-5 form works (`from pops.numerics.riemann import HLL`, `from pops.numerics.reconstruction.limiters import Minmod`, `from pops.diagnostics import norm`), in addition to the namespace objects. `pops.lib` keeps only the presets (`pops.lib.time`, `pops.lib.models`) plus the transitional `spatial` / `fields` / `solvers` / `preconditioners` catalogs (relocated to `pops.numerics` / `pops.fields` / `pops.solvers` / `pops.linalg` in the Phase A2 follow-up, held back so the in-flight unified-install path is undisturbed). Consumers (`codegen.library`, the moment / external-brick / riemann-hook tests, `examples/spec4`) are migrated to the new homes; no back-compat re-export of the moved catalogs from `pops.lib` (a thin `pops.lib.descriptors` alias remains only for the still-parked `lib` catalogs). The descriptors are unchanged inert metadata; nothing computes in Python.
- **Spec 4 ready time schemes home to `pops.lib.time`** (ADC-477, epic ADC-467, sections 6 / 14): the ready time-stepping schemes (forward_euler / ssprk2 / ssprk3 / rk4 / rk / adams_bashforth / strang / lie / condensed_schur / imex_local / bdf / explicit_rk / imex_local_linear / predictor_corrector_local_linear), their `std` bundle and the `_stage_rhs` helper now live in `pops.lib.time` (the scheme library); `pops.time` keeps ONLY the temporal language (`Program`, `CompiledTime`, the `Value` / `StageStateSet` values, the `Schedule` scheduler, and the `eliminate_*` / `optimize` IR-optimizer wrappers). This inverts the Spec-3 leftover where `pops.time` was canonical and `pops.lib.time` forwarded to it (the `pops.lib.time` scheme files no longer take the lazy `pops.time` hop), and keeps `pops.time` free of any `pops.lib.time` import (the acyclic layering time -> {ir, model}). `pops.time.std` is removed; the schemes are called by their explicit names (`pops.lib.time.ssprk3`, ...; no `std` bundle, Spec 4 s7). The IR each scheme builds is unchanged (`_ir_hash` byte-identical before and after).
- **Spec 4 runtime layer extracted to `pops.runtime`** (ADC-468, epic ADC-467): the
  ~3585-line `python/pops/__init__.py` is now a minimal hub (under 120 lines). The
  runtime objects (System / AmrSystem, the composable bricks and scheme/time policies,
  the mesh objects, threading, doctor / capabilities, host PythonFlux) live under
  `pops.runtime`, the only layer that imports the `_pops` extension; the extension load
  moves to `pops._bootstrap` and the version to `pops._version`. `pops.integrate` moves
  to `pops.runtime.integrate` (the `pops.integrate` name is preserved). No public API or
  hot-path change.
- **Spec 4 consumer migration off the flat shims** (ADC-473, epic ADC-467): every
  in-repo consumer now imports the model-authoring symbols from their package homes
  instead of the transitional `pops.dsl` / `pops.moments` re-export shims. The
  symbolic-IR builders move to `pops.ir` (`pops.ir.expr.Var`/`Const`,
  `pops.ir.ops.sqrt`/`eig_all_real`, `pops.ir.lowering.diff`), the PDE facade to
  `pops.physics.facade.Model`, the lower-level authoring objects to
  `pops.physics.model`/`bricks`/`hybrid`/`multispecies`/`aux`, the build/compile
  engine to `pops.codegen.compile`/`loader`/`toolchain`/`cache`/`abi`, and the moment
  toolkit to `pops.lib.moments`. Migrated 100 Python tests, 8 examples, 2 benches and
  the tutorial script plus the affected docs (`pops.dsl.X` -> the fully-qualified
  home, `pops.moments` -> `pops.lib.moments`). The public top-level re-exports
  (`pops.compile_problem` / `pops.CompiledProblem` / `pops.compile_library` /
  `pops.read_library_manifest` / `pops.integrate.*`) and every runtime sim method are
  unchanged. The `pops.dsl` / `pops.moments` shims are deleted in the final clean-break
  step.

- **Spec 4 clean break** (ADC-473, epic ADC-467): removed the flat `pops.dsl` and
  `pops.moments` modules; consumers now import from `pops.physics` / `pops.ir` /
  `pops.codegen` / `pops.lib.moments`, and the library module moved to
  `pops.codegen.library`.

- **pops.model package extraction** (ADC-469, epic ADC-450): the flat
  `python/pops/model.py` operator-first core is split into a
  `python/pops/model/` package (`spaces`, `signatures`, `operators`,
  `registry`, `module`, `bundles`), with `pops.model` re-exporting the
  same public surface (no API or behavior change). The package
  intra-imports are relative; the `dsl.py` standalone-import fallback
  and the five `docs/docmap.toml` `depends_on` entries are repointed
  at `model/__init__.py`.

- Bind compiled-Program blocks to System blocks by name, not add-order (ADC-457 criterion 23). The
  block names join the Program IR identity (`block_order` in the hash), so this invalidates every
  pre-existing problem-`.so` cache (a different block binding is correctly a different program).

- **`restore_history` scatter uses the multi-box `write_state`** (ADC-406b follow-up): the history
  checkpoint restore now scatters through `Impl::write_state` (the multi-box dispatcher `set_state`
  uses, the true inverse of the multi-box `gather_global`/`history_global`) instead of the mono-box
  `blocks_.write_state`. The mono-box / MPI mono-box round-trip is unchanged; `theta_boxes > 1` now
  restores correctly. The inline comment that overclaimed "the EXACT inverse of gather_global /
  state_global" was corrected.

- **Matrix-free apply allocates nothing per Krylov iteration** (ADC-408, follow-up of ADC-405
  Phase 6b): the time-program codegen for a `matrix_free_operator` no longer allocates the
  affine-combine accumulator inside the `pops::ApplyFn` lambda (it ran
  `pops::MultiFab acc = ctx.scratch_state_like(...)` on every matvec). It is now a persistent
  install-time scratch (`std::make_shared<pops::MultiFab>(ctx.alloc_scalar_field(1, 1))`, captured by
  the lambda), zeroed with `set_val(0)` and reused each iteration, matching the alloc-once runtime
  Krylov scratch (`generic_krylov.hpp`). The emitted result is unchanged (bit-identical over the
  valid cells the axpy / lincomb touch); only the allocation is hoisted.
- **`include/pops` deep re-nest, phase 5 (final): runtime split + coupling families finished**
  (ADC-396, follow-up of ADC-395): `runtime/` keeps only the public facade at top (system,
  amr_system, facade_options, export); `detail/` splits into `config/` (runtime_params,
  dispatch_tags, model_spec), `context/` (grid_context, wall_predicate) and `dynamic/` (abi_key,
  dynlib, dynamic_model, model_registry); `builders/` splits into `block/`, `compiled/` and
  `factory/`. `coupling/static_system/` is renamed `coupling/system/`, and `coupling/schur/` splits
  into `core/`, `source/` and `amr/`. Every internal `#include <pops/...>` and the DSL emit (runtime
  config/dynamic/builders paths) are repointed. Public include-path break (pre-1.0). Completes the
  include/pops family layout.
- **`include/pops` deep re-nest, phase 4: numerics split into sub-families** (ADC-395, follow-up of
  ADC-394): top-level numerics headers move to `linalg/` (dense_eig, lorentz_eliminator) and `fv/`
  (numerical_flux, reconstruction, spatial_discretisation); `spatial/` splits into `primitives/`
  (state_access, positivity, face_flux, wave_speed), `operators/` (cartesian_operator,
  masked_operator, polar_operator) and `embedded_boundary/` (domain, operator); `time/` splits into
  `integrators/` (time_integrator, time_steppers, ssprk, implicit_stepper), `schemes/` (imex,
  splitting, scheduler) and `amr/` (levels, advance, reflux). The `numerics/spatial_operator.hpp`
  umbrella (ADC-328) is kept and repointed. Every internal `#include <pops/...>` and the DSL emit
  (`numerics/dense_eig.hpp`) are repointed. Public include-path break (pre-1.0).
- **`include/pops` deep re-nest, phase 3: mesh split into sub-families** (ADC-394, follow-up of
  ADC-393): `mesh/` now groups `index/` (box2d, box_hash), `layout/` (box_array, patch_box,
  distribution_mapping, refinement), `storage/` (fab2d, multifab, mf_arith), `geometry/`
  (geometry), `boundary/` (physical_bc, halo_schedule, fill_boundary) and `execution/` (for_each).
  Every internal `#include <pops/...>` is repointed. Public include-path break (pre-1.0); the
  adc_cases two_fluid_ap mesh includes are updated in a companion PR.
- **`include/pops` deep re-nest, phase 2: core / physics / amr split into sub-families** (ADC-393,
  follow-up of ADC-392): `core/` now groups `foundation/` (types, cold, allocator, kokkos_env),
  `state/` (state, variables, aux_names) and `model/` (physical_model, equation_block,
  coupled_system); `physics/` groups `bricks/` (bricks, hyperbolic, elliptic, source),
  `composition/` (composite) and `fluids/` (euler); `amr/` groups `hierarchy/`, `tagging/` and
  `regridding/`. Every internal `#include <pops/...>` and the DSL emit (`physics/bricks.hpp`,
  `core/variables.hpp`) are repointed. Public include-path break (pre-1.0). Also retires the
  deprecated physics compat forwarders
  `physics/{advection_diffusion,langmuir,two_fluid_isothermal}.hpp` (the bricks live at
  `validation/physics/` under `pops::validation` since ADC-329; only the dedicated
  `test_physics_validation_compat` used the old `pops::` aliases, removed with them).
- **`quality.yml` pins clang-format to 19.1.7** (ADC-381): the now-blocking `format` job installed
  whatever clang-format the `ubuntu-latest` runner shipped (~v14-18) via `apt-get`, so Google-style
  output drift between major versions could fail the gate on an otherwise-clean tree. It now installs
  the ADC-118 sweep version (`pipx install clang-format==19.1.7`) for a deterministic gate; the
  canonical version is documented in `.clang-format` and `CONTRIBUTING.md`. CI install step only; no
  source reformatting.
- **`quality.yml` clang-format check is now blocking** (ADC-219): with the tree conforming to
  `.clang-format` since the ADC-118 sweep, the `format` job's `clang-format --dry-run` step fails the
  job on any new style deviation (a regression) instead of only warning. `ruff` (Python) stays
  informational (`if: always()`, so its signal survives a format failure). This is the D15 hardening;
  the corresponding clang-tidy-family checks stay informative until their existing findings are fixed.
  Note: `quality.yml` runs on schedule, manual dispatch, or a `quality`-labeled PR, not on every PR.
- **Repo-wide clang-format sweep** (ADC-118): `include/`, `python/` and `tests/` reformatted to
  `.clang-format` in one mechanical commit (`78816b0`, no functional change), bringing the
  `format`-scanned trees to zero non-conforming. Its SHA is recorded in `.git-blame-ignore-revs` so
  `git blame` skips it (`git config blame.ignoreRevsFile .git-blame-ignore-revs`, wired into
  `scripts/setup_env.sh`; GitHub's blame UI honors the file automatically). The four tests carrying a
  generated-C++ raw string are fenced with `// clang-format off` to keep the emitted source verbatim.
- **`pops/coupling` headers grouped by family** (ADC-326): the 18 flat coupling headers are moved into
  abstraction families (`base/`, `source/`, `single/`, `static_system/`, `amr/`, `schur/`,
  `deprecated/`) so the API surface is legible; internal cross-includes now use the canonical family
  paths and a new `include/pops/coupling/README.md` documents each family's stability tier (stable,
  reference/static, AMR, Schur, deprecated). Every historical `#include <pops/coupling/<name>.hpp>`
  keeps compiling through a forwarding stub, so the move is source-compatible with no behavior change.
- **Cartesian spatial operator split into focused modules** (ADC-328): the ~975-line
  `include/pops/numerics/spatial_operator.hpp` is split into a one-way module DAG under
  `include/pops/numerics/spatial/` (`state_access`, `positivity`, `face_flux`, `wave_speed`,
  `cartesian_operator`, `masked_operator`), each documenting its own contract (state/aux access,
  reconstruction + positivity, face flux, wave-speed reductions, residual assembly, masked path).
  `spatial_operator.hpp` becomes an umbrella header that re-exports every symbol, so existing
  `#include <pops/numerics/spatial_operator.hpp>` users are unaffected and the numerics are
  bit-identical (pure code move, no behavior change).
- **Reference and deprecated AMR headers moved out of the production surface** (ADC-332): the
  single-box Fab2D/MultiFab oracle headers `numerics/time/amr_reflux.hpp` and
  `numerics/time/amr_level.hpp` move to `numerics/time/reference/`, and the deprecated N-level
  engine `numerics/time/amr_multilevel.hpp` moves to `numerics/time/deprecated/`, so the tree makes
  their status explicit. Forwarding stubs keep every historical
  `#include <pops/numerics/time/...>` path compiling (the live `amr_reflux_mf.hpp` aggregator still
  reaches `amr_level.hpp` through the stub), so there is no behavior change. Audit recorded: none of
  the three headers, nor the deprecated `coupling/spectral_coupler.hpp`, has any live includer in the
  core, the tests, the bindings, or `adc_cases` (`amr_reflux.hpp` is pulled only by the deprecated
  `amr_multilevel.hpp` and `coupling/spectral_coupler.hpp`); `spectral_coupler.hpp` is relocated to
  `coupling/deprecated/` by ADC-326.
- **Generalize runtime/system header comments** (ADC-370): follow-up to ADC-333 on the headers that
  were then in flight with ADC-369 (`runtime/system.hpp`, `runtime/amr_system.hpp`,
  `runtime/system/system_field_solver.hpp`, `coupling/amr_coupler_mp.hpp`). Diocotron/Hoffart/
  ROMEO-GH200 narration reworded to the generic contract; the provenance is recorded in
  `docs/validation/HEADER_PROVENANCE.md`. `system_stepper.hpp` is already generic (its
  `HOFFART_STEP_SEQUENCE.md` references are sanctioned validation-doc pointers). Comments and docs
  only; no behavior change.
- **Hoisted the SSPRK RK scratch out of the per-substep hot loop** (ADC-261): the explicit steppers
  (`ForwardEuler`, `SSPRK2Step`, `SSPRK3Step`) re-allocated and zero-initialized fresh full-state
  scratch (the residual `R` plus the stage copies `U1`/`U2`/`U3`) on every `take_step`, so an
  n-substep explicit advance churned malloc/zero/free n times per macro-step. Each stepper now has a
  reusable `Scratch` struct and a `take_step(rhs, U, dt, Scratch&)` overload; a `run_explicit_substeps`
  helper allocates the scratch once per advance and reuses it across substeps (with a one-shot
  fallback for custom user steppers). The one-arg `take_step` is kept for the existing callers. The
  same hoist already lived in `AdvanceImexRkArs222`; behavior is bit-identical (the buffers are fully
  overwritten each substep and stage ghosts are re-derived by the residual fill_ghosts), locked by
  `test_weno5_ssprk3` (1e-14 parity).
- **Runtime headers organized by layer** (ADC-330): the flat `include/pops/runtime/` directory is
  split into layers so the structure shows the API surface. The public facades stay at the root
  (`system.hpp`, `amr_system.hpp`, `model_spec.hpp`, `facade_options.hpp`, `export.hpp`); the ABI /
  dispatch / internal helpers move to `detail/`, the DSL and native model builders to `builders/`,
  the extracted `System` internals to `system/`, and the AMR engine to `amr/`. Every historical
  include path (`<pops/runtime/*.hpp>`) keeps compiling through a forwarding header, so the runtime
  consumers, the bindings and the tests are unaffected. Source-tree layout only; no behavior or
  ABI change.
- **Factored the geometry-free Schur source kernels into one header** (ADC-263): the five device
  functors carried byte-identical by the Cartesian and polar Schur source steppers
  (extrapolate-scalar, extrapolate-velocity, energy, extract-velocity, copy-Bz) plus the
  `set_krylov` validation now live in `include/pops/coupling/schur_source_kernels.hpp`, which depends
  only on the lightweight `Array4`/`ConstArray4` handles (not on the elliptic solver stack), so the
  polar path can share them instead of re-declaring its own copies. Only the metric-bearing kernels
  (reconstruct, operator-coeff, explicit-flux, RHS-assemble) stay local to each stepper. Behavior is
  bit-identical; covered by the existing Cartesian/polar/AMR Schur tests (serial + MPI np2/np4).
- **Centralized native AMR refinement ratio** (ADC-295): the refinement ratio of the in-house AMR
  hierarchy is now a single named invariant `kAmrRefRatio` (with a `require_supported_ref_ratio`
  guard) in `include/pops/amr/refinement_ratio.hpp`, instead of the literal `2` scattered across the
  coarse/fine paths. `AmrHierarchy` defaults to and validates that ratio, rejecting any other value
  at construction with a clear error rather than silently mis-coarsening; the parametric call sites
  (`coarsen_index`/`coarsen_grown`/`refine`, composite-FAC `CompositeFacPoisson`/`fac_bilerp_coarse`,
  the subcycling `dxc / r` spacings and `SubcyclingSchedule`) read the constant, and the
  ratio-2-structural kernels (the NxN average-down and coarse-fine flux unrolls in `numerics/time`)
  carry a `static_assert` so a future non-2 ratio fails loudly at exactly those sites. `capabilities()`
  now states `refinement_ratio = 2 only`. Behavior is bit-identical (the constant is 2);
  `test_ref_ratio` locks the invariant and the rejection.
- **Share the Newton-options range validation** (ADC-213): the four-check defensive range validation
  of the `NewtonOptions` POD (duplicated verbatim in `System::add_block` and `AmrSystem::add_block`)
  is factored into `validate_newton_options(newton, where)` next to `NewtonOptions`
  (`implicit_stepper.hpp`). The `time='imex'` gate and `newton_diagnostics` handling, which differ
  between the two callers, stay inline. The AMR `newton_fail_policy` error text is harmonized to the
  System wording (no test depends on it; the binding's string parser rejects bad policies first, so
  the integer-range check is unreachable from Python). No behavior change.
- **Elliptic solver headers organized by family** (ADC-334): `include/pops/numerics/elliptic/` is
  split into `interface/`, `poisson/`, `mg/`, `eb/`, `polar/`, and `linear/` subdirectories so the
  numeric surface shows its solver families instead of one flat directory. Every historical include
  path (`<pops/numerics/elliptic/*.hpp>`) keeps compiling through a forwarding header, so downstream
  consumers and docs are unaffected; only the source-tree layout and the headers' internal
  cross-includes change. No numerical behavior, public API, or capability surface changes.
- **Generalize core public-header comments** (ADC-333): the headers under `include/pops/**` now read
  as the generic math contract (invariants, preconditions, expected errors, maintainer warnings)
  rather than the implementation of one scenario. The diocotron/Hoffart reproduction, the
  ROMEO/GH200 machine names, and the cross-repo `adc_cases` ticket trails are reworded generically
  and the provenance is relocated to the validation docs (new `docs/validation/HEADER_PROVENANCE.md`,
  plus the existing `HOFFART_*`/`SCHUR_CONDENSATION_DESIGN`/`DIOCOTRON_GROWTH_RATE` docs). Comments
  and docs only; no behavior change.
- **Factor the stepper global-dt-bound and grid-step duplication** (ADC-213): `step_cfl` and
  `step_adaptive` shared, verbatim, the `add_dt_bound` `all_reduce_min` loop and the
  polar/Cartesian minimum-grid-step expression. Both are now private `SystemStepper` helpers
  (`apply_global_dt_bounds(dt, reason*)`, mirroring the existing `apply_coupled_freq_expr_bounds`
  idiom, and `cfl_grid_h()`). Trajectory is bit-identical and the MPI collective count/order is
  unchanged (no behavior change).
- **Factor the stepper due-block advance loop** (ADC-213): `step` (Lie) and `step_cfl` shared,
  verbatim, the per-block loop that advances every DUE block by its catch-up step (transport plus the
  opt-in Schur source stage). It is now a private `SystemStepper::advance_due_blocks(dt)` helper. The
  two other macro-steps keep their own loops (`step_strang` interleaves a re-solved source stage
  between two half advances; `step_adaptive` subcycles each block `n_b` times). The helper introduces
  no new collective and iterates the same blocks in the same order, so each rank issues the same
  sequence of transport/source halo and `all_reduce` calls as before: the trajectory is bit-identical
  and the MPI collective count/order is unchanged (no behavior change).
- **Builtin model bricks behind a single registry** (ADC-331): the transport / source / elliptic
  tag lists are centralized in `include/pops/runtime/model_registry.hpp` (constexpr `kTransports` /
  `kSources` / `kElliptics` tables plus CSV / choices / validator helpers), the model-axis
  counterpart of `dispatch_tags.hpp`. The sites that re-listed `exb|compressible|isothermal` inline
  (`model_factory.hpp`, `block_builder_polar.hpp`, `python/system.cpp`, `python/amr_system.cpp`) and
  the completeness messages of `validate_model_spec` now derive from that single table, so adding a
  builtin brick is one table row plus its compile-time dispatch case. Rejection messages stay
  byte-identical; a non-drift `static_assert` ties the registry `n_vars` to the brick types, the
  dispatch keeps an explicit registry/dispatch-consistency guard, and `test_model_registry.cpp` plus
  a routing-completeness check lock the table to generic brick tags (never an application scenario
  name). The `ModelSpec` / capabilities surface and the compiled native backend are unchanged.
- **Named user roles and strict named-coupling fallbacks** (ADC-292): `VariableSet` gains a
  string-keyed `user_roles` layer parallel to the canonical `VariableRole` enum, and a new
  `index_of(string)` resolves a component by a canonical role name OR a user-defined role label
  (removing the first-occurrence ambiguity of several `Custom` components). Coupled-source role
  resolution (`System` / `AmrRuntime::add_coupled_source`), implicit-role masks and the AMR regrid
  selector now accept a user-role label, and the label round-trips through the compiled-block `.so` roles CSV
  (`roles_csv` / new `parse_roles_into`; the flat ABI and `abi_key` are unchanged). The legacy named
  couplings (`add_collision`, `add_thermal_exchange`, `add_ionization`) now fail loud (new
  `coupling_role_index`) when a ROLES-BEARING block omits a required role instead of silently coupling
  component 0; a genuinely ROLELESS block keeps the canonical fallback. Shipped models declare full
  canonical roles, so they stay bit-identical. Per ADR-0001 Decision 4 (Option A).
- **Cache and harden the macOS wheel build** (ADC-367): the `Wheels` workflow now caches the Serial
  static+PIC Kokkos install (deterministic key on the pinned 4.4.01 commit + flags, so warm runs skip
  the Kokkos build) and wires `ccache` into the `_pops` compile (`CMAKE_CXX_COMPILER_LAUNCHER` +
  persisted `CCACHE_DIR` + `CCACHE_BASEDIR`/`NOHASHDIR`/`COMPILERCHECK=content`; a `ccache -s` step
  reports the warm hit-rate). Hardening: the Kokkos clone is retried and its checkout asserted against
  the pinned commit (re-pointed-tag / supply-chain guard), `-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0` is
  passed explicitly to the Kokkos build (no min-version skew vs `_pops`), and the tag-time release-attach
  is best-effort (a missing Release no longer red-flags a successful wheel build). CI-only.
- **Shard + instrument the gate-python CI** (ADC-366): the PR-blocking `gate-python` job is split into 3
  `matrix.shard` legs (round-robin over the sorted test list, ~1/3 of the ~18-19 min Python suite each);
  the `gate` aggregator still `needs: gate-python` so the required check name is unchanged (branch
  protection intact) and `fail-fast: false` keeps a full red/green signal. The opaque test runner is
  replaced by a per-file timing harness (slowest-first log + a TSV/JSON timings artifact) that preserves
  the exact fail semantics. ccache hit-rate is lifted on both gates (`CCACHE_BASEDIR`, `CCACHE_NOHASHDIR`,
  `CCACHE_SLOPPINESS` + a codegen-neutral `-ffile-prefix-map`; cache keys unchanged, abi_key untouched),
  and `setup-kokkos` bumps `actions/cache@v4 -> v5`. CI-only; no change to `-O` levels, the shipped
  library, or test coverage (the 3 shards are a complete, disjoint partition of all 110 test files).
- **Trim push-master CI scope and the MPI job** (ADC-366): the full suite (MPI + Kokkos OpenMP + bench)
  now runs on a push to `master` only when a build/backend path changed (a new conservative `full`
  paths-filter covering sources, tests, build files, `python/**`, `bench/`, `scripts/`, and the CI
  action/workflow); a push touching only meta files (other workflows, linter configs, LICENSE) skips
  them, with the nightly cron as the unconditional backstop and the `ci-full` label as the manual
  override (schedule/dispatch stay full). The MPI job runs `ctest --preset ci-mpi -L mpi` (the serial
  tests already run in gate-cpp), backed by a `tests/CMakeLists.txt` backfill that labels every test in
  the `POPS_HAS_MPI` block `mpi` (was ~8 of 60) plus a CI floor that fails loudly if the selection
  collapses. CI-only; no change to `-O` levels or global test coverage.
- **Split `bindings.cpp` into per-area pybind TUs** (ADC-365): the `py::class_` / `.def` surface moves
  from the single `PYBIND11_MODULE` into `init_core` (module attrs + `SystemConfig` + `ModelSpec`),
  `init_system` (`System`), and `init_amr` (`AmrSystemConfig` + `AmrSystem`), each its own TU declared in
  `python/bindings_detail.hpp` (which also holds the shared `to_2d`/`to_3d`/`flat`/`newton_fail_policy`
  helpers). `bindings.cpp` is now a thin module that calls them in order (init_core first, so the configs
  register before `System`/`AmrSystem` reference them). The bound API is byte-identical (bodies moved
  verbatim); the win is parallel compilation + lower peak pybind memory per TU, and better incrementals.
- **Memoize the fill_boundary halo schedule** (ADC-260): `fill_boundary_begin` no longer rebuilds the
  `BoxHash` and re-enumerates the full local + global (cross-rank send/recv) halo job list on every
  call. That schedule is a pure function of the layout (`BoxArray`, `DistributionMapping`, `n_grow`)
  and the per-call (`Periodicity`, domain), so it is now built once and memoized per distinct
  (`Periodicity`, domain) on the `MultiFab` (new `include/pops/mesh/halo_schedule.hpp`); only the local
  copies and the pack/MPI/unpack of the live data run per call. The plan is replayed in the SAME
  deterministic order, so packed buffers stay bit-identical and the per-rank send/recv lists stay
  aligned (`tests/test_mpi_mbox_parity`, `test_mpi_amr_compiled_parity` unchanged). The cache lives on
  the `MultiFab` and is dropped when the object is reassigned (AMR regrid builds a fresh `MultiFab`),
  so it cannot go stale. New `tests/test_fill_boundary_cache.cpp` (serial + `np=1/2/4`) proves cache-on
  equals rebuild bit-for-bit, the schedule is built once across K calls, and it is rebuilt when the
  periodicity, domain, `n_grow`, or layout changes. No API or behavior change; biggest win in the
  MG-dominated and multi-rank halo-dominated regimes.
- **Validation bricks out of the production physics surface** (ADC-329): the validation/reference
  bricks `AdvectionDiffusion`, `LangmuirMode` and `TwoFluidLinear` move from `include/pops/physics/` to
  `include/pops/validation/physics/` under the new `namespace pops::validation`, so the production brick
  surface (`physics/{hyperbolic,source,elliptic,composite}.hpp`, aggregated by `physics/bricks.hpp`)
  stays free of test-only models. The old `include/pops/physics/<name>.hpp` paths remain as deprecated
  compat forwarders that alias the type back into `pops::` (e.g. `pops::AdvectionDiffusion`), so existing
  and external includes keep compiling unchanged; `tests/test_physics_validation_compat.cpp` pins that
  both the new and legacy paths build and name the same types. No numerical behavior change.
- **Flux-subdivide the AMR compressible runtime TUs** (ADC-359, follow-up to ADC-335/342): the
  compressible (Euler 4-var) AMR seam was the heaviest remaining module TU because
  `python/amr_block_compressible.cpp` and `python/amr_compiled_compressible.cpp` each instantiated all
  four fluxes (the riemann dispatch lived inside `dispatch_amr_block` / `dispatch_amr_compiled`, whose
  hllc/roe capability guards pass for Euler). Each per-flux branch of the two dispatchers is factored
  into `dispatch_amr_block_<flux>` / `dispatch_amr_compiled_<flux>` (bodies moved verbatim), and the two
  TUs become a thin riemann dispatcher plus one per-flux TU each (`amr_{block,compiled}_compressible_{rusanov,hll,hllc,roe}.cpp`),
  so every flux compiles in parallel. `dispatch_amr_block` / `dispatch_amr_compiled` are unchanged and
  still serve the exb/isothermal seam. The reachable `build_amr_block` / `build_amr_compiled` leaf set,
  the validation, and the error messages are unchanged, so the numerics are byte-identical (guarded by
  the `dmax==0` parity suite). The 8 new TUs are added to the module, tests, bench, and docs/validation
  source lists.
- **Factor the multi-box global-gather idiom** (ADC-264): the five copy-pasted collective gather sites
  in `python/system.cpp` (`Impl::copy_comp0` / `copy_state` multi-box branches and
  `System::density_global` / `state_global` / `potential_global`) now route through a single
  anonymous-namespace `gather_global(mf, ncomp, gnx, gny)` helper (zero-init buffer, local-box write at
  global indices, `all_reduce_sum_inplace`, component-major). The loop bodies are moved verbatim, so
  every site stays bit-identical (`ncomp == 1` collapses the layout to `j*gnx + i`); the caller keeps
  the `device_fence` and the single-box fast path delegating to `SystemBlockStore`. Net -33 lines, no
  API, ABI, or behavior change. Locked by `tests/test_mpi_system_io_gather.cpp` (np=1/2/4, ADC-257) and
  `python/tests/test_polar_theta_boxes.py` (theta_boxes=2/4).
- **Explicit `ModelSpec`, no silent physics defaults** (ADC-290): `ModelSpec` no longer hard-codes the
  physics-selecting defaults `transport="compressible"` and `elliptic="charge"`; both tags are now unset
  by default and a new `detail::validate_model_spec` (called at `dispatch_model` and at the top of
  `System::add_block` / `AmrSystem::add_block`) rejects an unset `transport`/`elliptic` with a clear,
  field-naming message instead of silently composing Euler + Poisson-charge. `source="none"` (the
  explicit, neutral no-source choice) and all numeric defaults are unchanged; each numeric is read only
  once its brick is chosen, so it cannot inject physics on its own. The historical shortcuts stay at the
  Python edge (`pops.Model(...)` always sets the three tags). API note: a bare native `ModelSpec()` that
  relied on `compressible`+`charge` now raises (pre-1.0, see ADR-0001 Decision 2). Anti-regression tests
  (ADC-300): `tests/test_config_model_validation.cpp` and the `test_bindings.py` garde-fous assert the
  incomplete-spec rejection and message, so a silent Euler/charge fallback cannot return.
- **Validate `SystemConfig` / `AmrSystemConfig` before building the runtime** (ADC-299): `System` now
  validates the config (`n >= 1`, `L > 0`, plus the existing geometry/polar invariants) BEFORE
  constructing its `Impl`, which already derived the geometry, box array, distribution mapping and the
  aux `MultiFab` (all sized from `n`) ahead of the old post-construction `check_geometry`. An invalid
  `n`/`L` previously built a silent degenerate grid (`dx = L/0 = +inf` or negative `dx`); it is now
  rejected upstream with a message naming the cause. `AmrSystem` gains the same upstream guard with
  `L > 0`, `regrid_every >= 0` and `coarse_max_grid >= 0` (only `n >= 1` was checked, and after `Impl`).
  Covered by `tests/test_config_model_validation.cpp` and `test_bindings.py` (ADC-300). C++ and Python
  error messages stay aligned; no change to any valid run (bit-identical).
- **No-optimize the cold model/block factories** (ADC-337, P1-B): the host string->closure wiring
  (`dispatch_transport/_source/_elliptic/_model/_model_for`, `bind_variable_roles`,
  `resolve_implicit_components`, `make_implicit_mask`, `build_block`, `make_block`/`make_block_*`) is
  marked `POPS_COLD_FN` (clang `optnone` / gcc `optimize("O0")`, new `include/pops/core/cold.hpp`), so the
  backend stops inlining and `-O3`-optimizing the entire CompositeModel instantiation tree into one
  giant factory function -- the dominant slice of the heavy TUs' `-O3` cost (cf. `docs/BUILD_PROFILING.md`
  P1-B). The HOT kernels (`BlockRhsEval` / `Advance*` / `take_step` / Kokkos `for_each_cell`) are separate
  functions reached through `std::function` closures and stay `-O3`; the small closure-returning helpers
  (`make_max_speed` etc.) are left untouched. No `-ffast-math` and `-O0` vs `-O3` never changes IEEE
  results, so the numerics are byte-identical (guarded by the `dmax==0` parity suite). Stacks on ADC-335.
- **Flux-subdivide the isothermal runtime TU** (ADC-342, follow-up to ADC-335): `python/system_isothermal.cpp`
  instantiated both reachable fluxes (rusanov + hll) x 4 limiters x 15 models in one TU -- the post-split
  long pole (~120 `-O3` leaves). It is now split one `.cpp` per flux (`system_isothermal_rusanov.cpp` +
  `system_isothermal_hll.cpp`) via the existing per-flux seam, halving the leaves per TU so `-j` compiles
  them in parallel. System dispatches the riemann string to the matching seam (after the same
  `validate_riemann`/`validate_limiter` as `make_block`); hllc/roe (unreachable on a 3-var transport) hit
  the explicit registry throw. Byte-identical codegen for the kept combos (verbatim `make_block_<flux>`;
  guarded by the `dmax==0` parity suite), mirroring the ADC-335 compressible subdivision.
- **Pin the conda build toolchain and surface the heavy-TU pool** (ADC-338): `environment.yml` pins
  `pybind11>=2.13,<3` (the conservative/validated 2.x line; 3.x still compiles, drop `<3` to opt in) and
  documents the local-vs-validated Kokkos gap (conda ships a Serial CPU-dev `kokkos`, default per-platform
  -- dry-run verified osx-arm64 ~4.7.01 / linux-64 ~4.3.00 -- a separate artifact from the source-built
  GPU Kokkos 4.4.01 used on ROMEO/CI, so it is intentionally not hard-pinned). `scripts/setup_env.sh`
  keeps AppleClang the macOS default and installs the conda `cxx-compiler` (gcc 14.2 via cxx-compiler
  1.11.0) as the pinned Linux default -- the fix for the slow `-j40` Linux build (wrong floating host gcc)
  -- and now prints how to widen `POPS_HEAVY_TU_POOL` on a high-RAM host so `-j` parallelizes the
  (post-ADC-335) heavy sub-TUs, while CI/constrained machines keep the size-1 OOM guard. Pins verified to
  resolve by `conda create --dry-run` on osx-arm64 and (`--platform`) linux-64.
- **Parallelize the `_pops` build by splitting the heavy runtime TUs** (ADC-335): `python/system.cpp`
  and `python/amr_system.cpp` instantiated the full transport x source x elliptic x flux x limiter x
  integrator product (~1700 `-O3` leaves) in two giant TUs, so `_pops` had only 3 TUs and `-j` capped at
  3 (a colleague's `-j40` build took 2h+). The runtime dispatch is now split, by a verbatim move behind
  fixed-signature type-erased seams, into ~16 TUs: System by transport (`system_{exb,isothermal,polar}`)
  with the compressible/Euler transport further by flux (`system_compressible_{rusanov,hll,hllc,roe}`),
  and AmrSystem by transport x {single-block `AmrCouplerMP`, multi-block `AmrRuntime`}
  (`amr_{block,compiled}_{exb,isothermal,compressible}`). A new `POPS_HEAVY_TU_POOL` CMake cache var
  (default 1, anti-OOM) lets the Ninja heavy-TU pool widen so `-j` actually compiles the now-smaller
  sub-TUs in parallel. Byte-identical: the inner make_block / dispatch_amr ladders move unchanged, so the
  set of instantiated kernel symbols is identical (verified: `nm -g` exported table unchanged, 0 hot
  kernel leaves added/removed) and the production-parity suite stays `dmax==0`. Measured on an 8-core
  Mac: clean `-O3` `_pops` build 1112 s (pool=1) -> 284 s (pool=8, ~16 TUs), 3.9x. No numerics change.
- **Test build deduplicates the heavy runtime TUs** (ADC-336): `python/system.cpp` and
  `python/amr_system.cpp` are now compiled once per test configuration into two `OBJECT` libraries
  (`pops_runtime_system`, `pops_runtime_amr`) in `tests/CMakeLists.txt`, instead of being re-listed as a
  source in ~23 test executables (each a multi-GB cc1plus compile of the full dispatch product). The 23
  plain and MPI heavy-source targets link the matching object library; the 4 `ENABLE_EXPORTS`
  native-loader tests keep their own copy so the dlopen/-rdynamic resolution of the `POPS_EXPORT` runtime
  symbols is unchanged. Byte-identical: the object libraries carry exactly the flags those targets used
  before (`pops::pops` only, no extra defines), and the `-O0` RAM cap is propagated PUBLIC so each
  consumer's own `.cpp` stays at the same `-O` level, preserving the `add_compiled_model` vs `add_block`
  bit-parity (`dmax==0`) against FMA contraction. `_pops` (`python/CMakeLists.txt`) is untouched: it
  already compiled each TU once, and no preset co-builds tests and the Python module. Serial tree:
  `system.cpp` 6 -> 1 compile, `amr_system.cpp` 14 -> 5 (1 library + 4 retained loaders). No change to
  the numerics or the public API.
- **Reliable Linux/Ubuntu user install** (ADC-321): `scripts/setup_env.sh` now bootstraps a fresh
  machine end to end -- it guides the Miniforge install when `conda` is absent, configures conda-forge
  to survive HTTP 429, forces a CPU Kokkos by default via `CONDA_OVERRIDE_CUDA=""` (so `pip install .`
  no longer fails `Could not find nvcc` on a CPU host with an NVIDIA driver; `--cuda` opts in),
  persists `POPS_INCLUDE`/`POPS_KOKKOS_ROOT`/`Kokkos_ROOT`/`POPS_CACHE_DIR` in the env, and ends on
  `pops.doctor()`. `pops.doctor()` gains a `kokkos_root` check (the tutorial's "no DSL backend" blocker)
  and a CUDA-Kokkos-without-nvcc check, each with a copy-paste fix. `installation.md` gains a
  "Linux and Ubuntu: fresh install" section and a troubleshooting table; the diocotron tutorial routes
  a both-backends-failure to `pops.doctor()`. `setup_env.sh`, `environment.yml`, the diocotron tutorial
  and `pyproject.toml` are translated to English.
- **Distributed FFT Poisson hardening** (ADC-316, fast-follow to ADC-287): `RemappedFFTSolver::solve()`
  adds an owner-only `device_fence()` after the periodic-ghost wrap (PR #254 managed-buffer ordering
  discipline; the caller's post-`ell_solve` fence already brackets the read, so this is belt-and-
  suspenders that self-documents the seam, and a no-op on CPU). `test_mpi_system_fft` now asserts the
  remapped solver's `residual()` is machine-zero and covers a non-power-of-two grid (n=12), exercising
  PoissonFFT's O(n^2) DFT fallback under the box-slab remap. The System Cartesian single-box invariant
  is documented at its source (`python/system.cpp`).
- **Build parallelism derived from cores, not hardcoded** (ADC-339): the README no longer prints a
  literal `-j8`; it states the Ninja default already uses every core and gives the dynamic cap form
  (`-j$(nproc)` on Linux, `-j$(sysctl -n hw.ncpu)` on macOS). The ROMEO validation `.sbatch` builds
  that matched their `--cpus-per-task` allocation now read `-j "${SLURM_CPUS_PER_TASK:-N}"` so they
  self-adjust to the allocation. The CI `--parallel`, the WSL2 `-j 6` (RAM bound), and the parity181
  half-allocation nvcc cap stay explicit and documented (memory-bounded environments, intentional).

### Removed

- **`include/pops` forwarding shims deleted; one canonical family path per header** (ADC-392): the 54
  transition stubs left at the old flat include paths by the M2 generalisation reorgs (ADC-326
  coupling, ADC-329 physics, ADC-330 runtime, ADC-332 reference/deprecated, ADC-334 elliptic) are
  gone, so `install(DIRECTORY include/pops)` no longer ships duplicate header paths. Every internal
  `#include <pops/...>` (~290 references across include/python/tests/bench/docs) and the DSL emitted
  includes in `python/pops/dsl.py` now point at the canonical family path, e.g.
  `<pops/runtime/builders/block_builder.hpp>`, `<pops/coupling/single/coupler.hpp>`,
  `<pops/numerics/elliptic/mg/geometric_mg.hpp>`. Public include-path break, acceptable while the
  public API still moves (pre-1.0; One-Version rule: one pinned `_pops`, adc_cases regenerates its
  bricks from the DSL). Phase 1 of the include/pops reorganization; the per-layer deep re-nest follows.

### Fixed

- **Stale `theta != 1` assertion in the condensed_schur example** (ADC-427): `examples/time_programs/condensed_schur_program.py` `show_macro()` still asserted that `condensed_schur(theta != 1)` raises `NotImplementedError`, but ADC-427 made `0 < theta <= 1` lower the n+1 extrapolation, so the example printed `UNEXPECTED ... did not raise` and exited MISMATCH. `show_macro()` (and the prose) now demonstrate that `theta=0.5` lowers the full assemble / solve / reconstruct chain plus the `schur_extrap` n+1 extrapolation, and that a theta outside `(0, 1]` raises `ValueError`.
- **Symbolic IR imports without numpy** (epic ADC-467, Spec 4): `pops.ir.expr` / `pops.ir.values` imported numpy at module scope, but the build-time codegen tool `scripts/gen_solver_kernel.py` runs under the C++ job's bare interpreter (no numpy) and imports `pops.lib` -> `pops.ir` to emit the custom-solver kernel header. The `ir` restructure thus reddened the `gate-cpp` / MPI / OpenMP build at the gen-solver step (`ModuleNotFoundError: No module named 'numpy'`); the python PRs route to `gate-python` and skip the C++ build, so it only surfaced on the master push. numpy is now imported lazily inside the host `.eval()` interpreter (the only place it is used; IR construction and C++ emission never touch it), so the symbolic IR imports in a bare interpreter. A new `tests/architecture/test_codegen_without_numpy.py` re-runs the real codegen tool with numpy forced absent and runs in `gate-python`, so the regression is caught before the master push.
- **Scheduled program `.so` now dlopens** (ADC-458, epic ADC-450): `pops::System::macro_step()` is now
  `POPS_EXPORT`. A scheduled (`every(N)` / `hold` / `accumulate_dt`) program `.so` calls it for the
  cadence decision, so without the export the held-schedule `.so` failed to load with `symbol not
  found in flat namespace '__ZNK3adc6System10macro_stepEv'` -- the scheduler runtime was never reachable
  end to end. The new `test_spec3_runtime_end_to_end.py` (above) exposed and now guards it.

- **Source-only RHS no longer leaks the flux** (ADC-430, epic ADC-399, sibling of ADC-425):
  `pops.time.Program.rhs(flux=False, ...)` used to STILL emit the `-div F` base -- the codegen routed on
  `sources` but ignored the `flux` flag, so a source-only stage of a Lie/Strang/IMEX split double-added
  the flux on any model with a non-zero flux (masked because split source stages were tested only on
  zero-flux models, where `-div F == 0`). A new runtime primitive `System::block_source_into` (and
  `ProgramContext::source_default_into`) assembles the model's default source `S(U, aux)` WITHOUT the
  flux divergence -- the exact mirror of ADC-425's `block_neg_div_flux_into` -- via a per-cell
  `SourceInto<Model>` kernel (the SAME `m.source` `assemble_rhs` adds, no numerical-flux dispatch, so it
  is flux-template agnostic and bit-identical to the source half of `rhs_into`). The codegen now branches
  on `flux`: `flux=False` emits a zeroed base + `ctx.source_default_into` iff `"default"` is requested
  (so `flux=False,sources=["default"]` is the default source only, `flux=False,sources=["s"]` is just
  `s`, `flux=False,sources=[]` is the zero RHS); `flux=True` is unchanged. `flux=False` with named
  fluxes is rejected. New `python/tests/test_time_rhs_flux_false.py`.
- **Flux-only RHS no longer leaks the default source** (ADC-425, epic ADC-399, spec criterion 17):
  `pops.time.Program.rhs(flux=True, sources=[])` on a model with a default `m.source` used to STILL add
  that source (the codegen always lowered the default-flux RHS to `ctx.rhs_into` = `-div F` + the
  default source), breaking the hyperbolic stage of a Lie/Strang/IMEX split (which needs "flux but no
  source"). A new runtime primitive `System::block_neg_div_flux_into` (and
  `ProgramContext::neg_div_flux_default_into`) assembles `-div F(U)` WITHOUT the default source by
  reusing the block's existing transport path on `SourceFreeModel<Model>` (bit-identical flux / ghost /
  geometry handling, only the source dropped). The codegen now routes on whether `"default"` is among
  the requested sources: `sources=None` (legacy) / `["default"]` keep `ctx.rhs_into` (unchanged);
  `sources=[]` is flux only; `sources=["a","b"]` is flux + a + b (named sources axpy'd on top, no
  double-count). New `python/tests/test_time_rhs_sources.py`.
- **Embedded C++ API page no longer inherits the doxygen-awesome theme** (ADC-388): the in-site
  `/doxygen/` pages (doxysphinx) inline the raw Doxygen `<head>`, so the doxygen-awesome global
  selectors (`body`, `a:link`, root variables, dozens of `!important` rules) were leaking into the
  furo Sphinx theme and breaking the embedded layout. `scripts/build_docs.sh` now clears
  `HTML_EXTRA_STYLESHEET` for the embed-only build; the standalone `/cpp/` site keeps the theme.
- **Build robustness on bare / FetchContent / HPC paths the conda CI never exercises** (ADC-387,
  same bug class as ADC-386, found by a build-system audit):
  - The FetchContent Kokkos fallback now sets `CMAKE_POSITION_INDEPENDENT_CODE ON` before
    `FetchContent_MakeAvailable(Kokkos)`, so the fetched (static) Kokkos links into the shared `_pops`
    extension on Linux instead of failing with `relocation R_X86_64_PC32 ... recompile with -fPIC`.
    Previously only `scripts/build_docs.sh` and `.github/workflows/wheels.yml` worked around it by
    hand; the documented `pip install .` / `cmake -B build` paths were unguarded.
  - `POPS_USE_HDF5=ON` no longer aborts configure when HDF5 is absent: `find_package(HDF5)` dropped
    `REQUIRED` and links/defines only when found. No C/C++ TU calls the HDF5 API (parallel output is
    the Python h5py facade), so a missing HDF5 must not be fatal (hit the HPC Spack recipe).
  - Removed the dead `@POPS_USE_OPENMP@` gate from `cmake/adcConfig.cmake.in` (the standalone OpenMP
    backend was removed in ADC-263; the variable expanded to an always-false `if()`). Kokkos-OpenMP
    installs get OpenMP transitively via `find_dependency(Kokkos)`.
  - Raised the `find_package(pybind11)` floor from 2.11 to 2.13 to match `environment.yml` and the
    FetchContent tag.
- **`_pops` builds on machines without pybind11 in the environment** (ADC-386): the FetchContent
  fallback for pybind11 now sets `PYBIND11_FINDPYTHON ON` before fetching, so the downloaded pybind11
  reuses the modern Python found via `find_package(... Development.Module)`. Without it pybind11 fell
  back to classic discovery, which did not propagate the Python include, and the `_pops` compile failed
  with `Python.h: No such file or directory` on bare/HPC/spack toolchains (the conda CI path, which
  finds pybind11 via `find_package`, was unaffected). Verified on ROMEO (x64cpu, Kokkos OpenMP).
- **Quasi-vacuum velocity bound for the isothermal model** (ADC-77, third stability barrier of the
  WENO5 rollup): when the diocotron rollup evacuates the background (rho -> ~1e-7) the Schur source
  stage writes O(1) momentum onto those cells, so the raw u = m/rho exploded and collapsed the CFL.
  `IsothermalFlux` (and the inherited `IsothermalFluxPolar`) now computes the velocity as
  u = m/max(rho, vacuum_floor), which bounds both the wave speed and the advective flux at evacuated
  cells; mass and momentum are untouched (only the velocity ESTIMATE is bounded, unlike a cell density
  clamp). It is an independent opt-in knob set per model via
  `pops.FluidState("isothermal", cs2=..., vacuum_floor=...)` (carried on `ModelSpec::vacuum_floor`);
  it is deliberately SEPARATE from the spatial `positivity_floor` (the Zhang-Shu reconstruction
  limiter), which addresses a different failure mode -- coupling them would change the CFL dt of
  existing positivity_floor transport runs. With `vacuum_floor == 0` (default) the path is
  bit-identical. Covered by `tests/test_isothermal_vacuum_floor.cpp` and
  `python/tests/test_isothermal_vacuum_floor_system.py`. The energy (Euler) model is out of scope:
  a velocity bound alone does not stabilize it (the sound speed c = sqrt(gamma p / rho) also diverges
  at vacuum, a coupled pressure-positivity concern).
- **GPU validation drivers broken by the TU split** (ADC-346): `docs/validation/diocotron_gpu.cpp` and
  `diocotron_amr_gpu.cpp` compile `python/system.cpp` / `amr_system.cpp` standalone, but after the
  ADC-335 split those TUs delegate to the `build_block_*` / `build_amr_*` seams now living in per-transport
  sub-TUs -- so the drivers failed to LINK on a GH200 build (`undefined reference to
  pops::detail::build_block_compressible_rusanov`, ...). CI never builds these drivers, so it went unseen
  until a ROMEO nvcc run (the split + the `optnone` factories COMPILE cleanly under nvcc; only the link
  was missing the sub-TUs). `docs/validation/CMakeLists.txt` now compiles the seam sub-TUs into both
  drivers (same list as `_pops` / `pops_runtime_*`), and `parity181.sbatch` / `diocotron_mpi.sbatch` stage
  `diocotron_amr_gpu.cpp` (referenced by the shared CMakeLists since ADC-320). Validation-only; no library
  or hot-path change.
- **AMR seed fine patch persisted without refinement** (ADC-324): the compiled/native mono-block AMR
  builder (`build_amr_compiled`) always allocated a central seed fine patch on the explicit/imex path,
  even when `set_refinement` was never called. With the `1e30` "no refinement" threshold the build-time
  regrid tags nothing and the zero-tag regrid is a deliberate no-op, so the seed survived as a single
  un-chopped fine box pinned to rank 0 of the coarse dmap (`n_patches() == 1`), dead weight that starved
  MPI strong-scaling (rank 0 carried its coarse boxes plus the whole fine patch). The seed is now
  allocated only when refinement is configured (`refine_threshold < 1e30`): no refinement gives a
  mono-level hierarchy (`n_patches() == 0`, coarse distributes cleanly), and the refined path is
  unchanged. Regression test: `test_amr_seed_no_refine`. Follow-up to ADC-319.
- **Stale negative control in the compiled positivity-floor test** (ADC-341): ADC-324 made
  `set_refinement(1e30)` a mono-level hierarchy, so the `python/tests/test_amr_compiled_positivity_floor.py`
  section (1) assertion that the UNFLOORED `.so` run blows up on the spike became vacuous (the coarse-only
  grid diverges in neither branch), reddening the `gate-python` CI job. Mirrors the ADC-324 fix already
  applied to the native sibling test: drop the `unfloored-blows-up` control and keep the compiled-facade
  contract (floor accepted + floored run finite); the load-bearing property stays covered by
  `tests/test_positivity_floor.cpp` and the native test's refined C/F interface, and that the floor rides
  the loader by the same test's dmax==0 marshalling check and multi-block routing. Test-only; no library
  or ABI change.
- **Backend-blind DSL compile cache** (ADC-186): recompiling a `production` model onto an explicit
  `so_path` where an `aot` artifact was already loaded via dlopen in the same process re-served the
  stale aot handle (`add_native_block: pops_native_abi_key missing`), since the dynamic loader caches
  handles by path. `compile()` now keeps an in-process registry of the backend written to each path
  and redirects an explicit `so_path` already held by another backend to a distinct
  `<base>.<backend>.so` sibling, so dlopen reloads a fresh handle. The out-of-source cache was
  already keyed by backend; the three compile facades (`HyperbolicModel`, `Model`, `HybridModel`)
  share the redirect. Regression test: `test_compile_cache_backend`.
- **DSL production/AOT loaders now compile with MPI** (ADC-319): the `backend="production"` and
  `backend="aot"` model `.so` were compiled without `-DPOPS_HAS_MPI` even when `_pops` is built with
  MPI, so `comm.hpp` fell back to its serial stubs (`n_ranks()=1`, `my_rank()=0`) inside the loader.
  Any distributed layout built in the loader then collapsed to a single owner on every rank: an
  `AmrSystem(distribute_coarse=True)` replicated the whole coarse transport on all ranks (no MPI
  strong-scaling). `dsl.py` now re-bakes `-DPOPS_HAS_MPI` plus the module's MPI include dir (exposed as
  `_pops.__has_mpi__` / `__mpi_include__`), leaving the MPI symbols undefined to resolve at load against
  the libmpi already loaded by `_pops`/mpi4py (no second libmpi linked, like the Kokkos runtime). The
  MPI seam enters the loader cache key (`mpi=on|off`). Measured on ROMEO (hyqmom15 diocotron, N=256,
  cmg=64, 16 boxes): per-rank coarse box count drops from 16 to 4 at np=4 (the base now distributes),
  and ms/step falls from a flat 2554 to 1962. Serial builds are unaffected (no flag, bit-identical).
- **`bench scaling_amr` broken by the AMR TU split** (ADC-347): `bench/scaling_amr` compiles
  `python/amr_system.cpp` (which calls the `build_amr_block_*` / `build_amr_compiled_*` factories) but,
  after ADC-335/336 moved those into per-transport seam TUs, was never updated to link them, so
  `bin/scaling_amr` failed to LINK (`undefined reference to pops::detail::build_amr_block_exb`, ...) and
  the non-required `bench` job had been red since ADC-335. The six `amr_{block,compiled}_*.cpp` seam
  sources are now linked into `scaling_amr`, mirroring the `pops_runtime_amr` object library. Build-graph
  only; no behavior, API, or numerics change. Bench-side parallel to ADC-346.

## [0.2.0] - 2026-06-16

### Added

- **Pointwise projection on AMR** (ADC-312): `add_compiled_model(AmrSystem)` now accepts a model
  declaring `m.projection` (ADC-177, e.g. HyQMOM relaxation15). The projection `U <- project(U, aux)`
  is applied per level at the end of each macro-step, after the reflux and cascade (cell-local and
  idempotent, so conservation is preserved). Opt-in: a model without the projection trait is
  bit-identical to the historical trajectory. Previously the AMR path rejected such models.
- **AMR checkpoint/restart** (ADC-65): `AmrSystem.checkpoint(path)` / `restart(path)` write and read
  a bit-identical npz (per-level conservative state with fine patches, per-level phi as the multigrid
  warm-start, the fine hierarchy and the clock), replacing the old `NotImplementedError`. Restart
  imposes the saved fine layout via the new `set_hierarchy` (no re-clustering, no re-prolongation).
  Mono-block mono-rank for now: MPI np>1, multi-block and `regrid_every>0` are rejected explicitly.
  Adds the append-only level accessors `n_levels` / `n_vars` / `level_state` / `level_potential` /
  `set_hierarchy`.
- **Model-declared named aux fields** (ADC-70, phase 1, Cartesian `System`): a model declares
  persistent auxiliary fields with `m.aux_field("name")` (read in a formula via `aux.extra_field(k)`);
  `sim.set_aux_field(block, name, array)` and `sim.aux_field(block, name)` set and read them (up to
  four). The `Aux` POD stays trivially copyable and device-clean, named components are static (never
  rewritten by `solve_fields`), and a model with no named field is cache/hash-identical to before;
  `B_z` / `T_e` stay on their dedicated setters.
- **Generic 2D moment-hierarchy generator** (ADC-164): the new `pops.moments` module derives the full
  M->C->S->closure->C'->M' algebra (binomial transform plus standardization) over the DSL AST, so a
  user supplies only the closure (S -> standardized moments of order N+1). Ships `build_moment_model`,
  `gaussian_closure` (Levermore), `lorentz_sources` (Vlasov-Lorentz) and `moment_indices` /
  `moment_names`; `robust=True` adds differentiable smooth floors and `exact_speeds=True` wires HLL
  speeds via autodiff plus numeric eig.
- **SSPRK3 time integration on AMR** (ADC-64): `AmrSystem.add_block(..., time="ssprk3")` (or
  `pops.Explicit(ssprk3=True)`) runs a 3-stage Shu-Osher SSPRK3 per subcycled level, staying exactly
  conservative across coarse-fine boundaries by refluxing the effective flux
  `Feff = 1/6 F0 + 1/6 F1 + 2/3 F2`. The default stays forward Euler (bit-identical); rejected
  explicitly on the compiled `.so` paths and with `imex`.
- **First-order forward Euler explicit method** (ADC-174): `pops.Explicit(method="euler")` selects the
  `ForwardEuler` stepper on `System.add_block` (native and `backend="production"`), for
  first-order-reference fidelity; `ssprk2` stays the default. The frozen-ABI AOT path rejects `euler`
  (pointing to the production/native backends) rather than silently ignoring it.
- **Explicit signed wave speeds in the DSL** (ADC-83): `m.wave_speeds(x=(smin,smax), y=(smin,smax))`
  declares signed face speeds directly, unlocking `riemann="hll"` for models with no pressure
  primitive (moment systems, isothermal). Without `set_eigenvalues` the Rusanov/CFL bound derives
  from `max(|smin|,|smax|)`; an `add_equation` guard rejects `riemann="hll"` with no emitted speeds.
- **HLL wave speeds from the flux Jacobian** (ADC-86/87): `m.wave_speeds_from_jacobian(x=, y=,
  eig="numeric"|"fd", blocks=)` emits exact HLL speeds as the spectrum extremes of the flux Jacobian
  (new device-clean header `pops::real_eig_minmax`: closed form for N<=2, Hessenberg plus Francis QR
  otherwise, Gershgorin outer-bound fallback). `x/y=None` autodiffs the declared flux; the `Abs`
  autodiff node is now differentiable, so smooth `max(x,eps)` floors are Jacobian-able.
- **Positivity floor limiter** (ADC-76): `pops.FiniteVolume(positivity_floor=...)` imposes a density
  floor on reconstructed face states, falling back locally to the source-cell average (conservative,
  first-order) on a violating face. `floor<=0` is bit-identical to before; it is threaded through the
  Cartesian, polar and cut-cell kernels and the compiled-block ABI, and rejected explicitly on the
  paths that do not support it (prototype JIT, AOT/production, AMR).
- **Opt-in HLL wave-speed cache** (ADC-199): `pops.FiniteVolume(wave_speed_cache=True)` evaluates
  `model.wave_speeds` once per cell and direction and reuses it as the face bound instead of
  recomputing on every face and RK stage (measured speedup on moment hierarchies). Off by default,
  bit-identical to OFF in the no-slope path, and rejected outside `riemann="hll"` plus explicit time.
- **Spectral Poisson FFT variant** (ADC-175): a new elliptic kind `solver="fft_spectral"` (from
  `System.set_poisson`, `pops.EllipticSolver`, listed in `pops.capabilities()`) reuses the periodic
  single-rank FFT plumbing of `"fft"` but with the continuous Laplacian symbol `-(kx^2+ky^2)`, exact
  on sinusoids. The discrete `"fft"` and `"geometric_mg"` solvers are unchanged.
- **Native Windows (MSVC) support** (ADC-99/100/136/144): `adc_cpp` compiles and imports natively on
  Windows without WSL2. A portable dynamic-loading layer `pops::dynlib` (`LoadLibraryW` /
  `GetProcAddress`), an `POPS_EXPORT` macro (`__declspec`), an MSVC-aware ABI key, and
  `std::numbers::pi` for `M_PI` cover the runtime; the DSL `production` backend compiles a model to a
  `.dll` with `cl` and runs bit-identical to the brick path (shared Kokkos, `_pops.lib`). All `_WIN32`
  branches are dead off Windows.
- **`System.dt_hotspot(name)` CFL diagnostic** (ADC-182): returns `{w, i, j}` for the global cell
  that dominates a block's transport CFL bound, to locate a collapsing dt without an external scan.
  On-demand and off the hot path (`step` / `step_cfl` stay bit-identical), a two-pass device-clean
  reduction with an MPI all-reduce.
- **Find-or-fetch Kokkos** (ADC-263): if no Kokkos install is found, CMake downloads and builds a
  release tarball verified by SHA256, so a plain `cmake -B build` works without a pre-installed
  Kokkos. Overridable via `POPS_KOKKOS_FETCH_VERSION` (default 4.4.01) and `POPS_KOKKOS_FETCH_SHA256`.
- **Eigenvalue witness in the projection DSL** (ADC-289): `dsl.eig_max_im(rows)`, `dsl.eig_lmin(rows)`
  and `dsl.eig_lmax(rows)` build a small dense matrix from moment expressions and return a scalar
  Expr from its spectrum via `pops::real_eig_minmax` (max imaginary part as a complex-eigenvalue
  witness, or the real-part bounds). The codegen emits a named device-clean functor (no extended
  lambda), so `m.projection` can express a branchless "if a moment matrix has a complex eigenvalue,
  correct it" rule (unblocks the native relaxation15 projector, ADC-275). Additive: the existing
  expression set and `m.projection` (ADC-177) are unchanged.

- **Documentation rebuilt around a Diataxis navigation** (ADC-248): Getting started, Tutorials,
  Concepts (11 pages, ADC-249), How-to (12 task pages), Simulation, AMR, Running, Advanced topics,
  Reference and Development sections, plus a Quickstart page and an internal documentation style
  guide (ADC-250). Pages are kebab-case.
- **Embedded C++ reference** in the Sphinx site via doxysphinx (ADC-149), with a modern Doxygen
  theme and full dot diagrams (ADC-239).
- **Documentation-as-code tooling**: per-page `docmap.toml` with owner and freshness plus an
  example harness (ADC-147), a doc taxonomy and a stack ADR (ADC-146), and CI doc lanes (a light PR
  lane and a weekly heavy lane with lld and ccache, ADC-151/225).
- **Quality tooling / static analysis** (ADC-105): dedicated CI workflow `.github/workflows/quality.yml`,
  off the PR critical path (weekly Sunday cron + `workflow_dispatch` + `quality` label). Five
  *informative* (non-blocking) jobs: `clang-format` (`.clang-format`), strict warnings
  (`POPS_ENABLE_WARNINGS`), `clang-tidy` (`.clang-tidy`), ASan+UBSan sanitizers (`POPS_ENABLE_SANITIZERS`,
  `ci-warnings`/`ci-asan` presets) and CodeQL. CMake options OFF by default (empty `pops_dev_options`
  target), so `ci.yml`, local builds and `adc_cases` are unchanged. See `docs/QUALITY_TOOLING.md`.
- **Fuzzing, coverage and developer automation** (ADC-113): invariant-checked libFuzzer harnesses in
  `fuzz/` (Box2D, Berger-Rigoutsos clustering, `real_eig_minmax`; option `POPS_BUILD_FUZZING` +
  `ci-fuzz` preset, clang), gcov/gcovr coverage (`POPS_ENABLE_COVERAGE` + `ci-coverage` preset),
  Python ruff lint (`[tool.ruff]`), opt-in pre-commit hooks (`.pre-commit-config.yaml`: clang-format
  and ruff at commit), and two more `quality.yml` jobs (`fuzz`, `coverage`) on the same weekly
  informative cadence; the default build stays bit-unchanged.
- **Repository health and GitHub hygiene**: BSD-3 `LICENSE` and license declaration, `CONTRIBUTING`,
  `SECURITY`, `.gitattributes`, PR and issue templates (ADC-223/224/244/246), a root-hygiene guard
  (ADC-169), Dependabot weekly Actions bumps (ADC-117), a release workflow that turns a `v*` tag
  into a GitHub Release from this changelog (ADC-234), and `docs/VERSIONING.md` with the bump rules
  (ADC-232/237).
- **Shared C++ test harness** (`test_harness.hpp`, `bench/common.hpp`, ADC-215); a Schur-split test
  with an AMR guard on `add_block` and auto-skip without Kokkos (ADC-207).
- **CI**: `ci.yml` split into parallel `gate-cpp` / `gate-python` lanes with `mold` and path
  routing (ADC-171).
- **Moment-model documentation**: a step-by-step HyQMOM tutorial, a moments-and-closures concept
  page, a moment-models reference, and `pops.moments` (`build_moment_model`, `gaussian_closure`,
  `lorentz_sources`) added to the Python API reference (autodoc).
- **Generic pointwise post-step PROJECTION hook** (ADC-177): `m.projection([...])` on the DSL side
  (C++ trait `HasPointwiseProjection`, compiled like the flux/source), applied by `System` at the
  end of every whole macro-step (never per RK stage) on the valid cells of each block -- replaces
  the per-cell Python callback (`aot` and `production` backends; `prototype` and
  `target="amr_system"` reject it explicitly). Adds `dsl.sign(x)` (branch-free mask selections,
  differentiable). See `docs/DSL_API.md` section 5.

### Changed

- **Kokkos is the only on-node backend** (ADC-263): the standalone OpenMP path is removed (the
  `POPS_USE_OPENMP` option, `find_package(OpenMP)`, the mutual-exclusion check and every `#pragma omp`
  or manual host loop); `POPS_USE_KOKKOS` defaults ON and is fatal if disabled. Serial runs through
  Kokkos Serial, threads through Kokkos OpenMP, GPU through Kokkos Cuda/HIP, all chosen at Kokkos
  install time (`Kokkos_ENABLE_*`). The `for_each_cell` seam `#error`s without `POPS_HAS_KOKKOS`; the
  standard is pinned to C++20. The DSL loaders (`compile_aot` / `compile_native`) build against Kokkos
  and the `.so` cache is keyed on Kokkos state.
- **AOT DSL backend compiles at native optimization flags** (ADC-201): `compile_aot` (and the hybrid
  AOT path) dropped the hardcoded `-O2` (about 1.48x slower) for the native `_dsl_optflags()`
  (default `-O3 -DNDEBUG`, overridable via `$POPS_DSL_OPTFLAGS`); only the prototype JIT stays at
  `-O2`. An `aot-optflags` marker in the `.so` cache key prevents serving a stale `-O2` binary.
- **Hybrid AOT models build with the Kokkos toolchain** (ADC-103): since the Kokkos-only switch,
  `HybridModel.compile(backend="aot")` uses the native Kokkos compiler and flags (with macOS
  `-undefined dynamic_lookup`) and raises a clear `POPS_KOKKOS_ROOT` error when no Kokkos is visible;
  the prototype JIT stays pure-host.
- **Leaner DSL codegen** (ADC-200): `emit_cpp_*` now emit only the primitives transitively live in
  each method (no dead closure or its `sqrt`), values bit-identical; an opt-in `hoist_reciprocals=True`
  hoists `1/x` once for a recurring conservative denominator and turns those divisions into products
  (off by default, since it changes rounding).
- **Mixed relative/absolute stop criterion for GeometricMG** (ADC-202): `System.set_poisson(...,
  abs_tol=0.0)` adds an absolute residual floor, so the stop test becomes
  `residual <= max(rel_tol*r0, abs_tol)` with an early exit when `r0` is already under the floor
  (avoids over-solving an already-converged off-step `solve_fields`). `abs_tol=0` keeps the historical
  relative behavior bit-identical; inert for the FFT solver.
- **Higher default QR cap in `dense_eig`** (ADC-195): the default iteration cap of `real_eig_minmax`
  rises from 30 to 100 so near-degenerate companion blocks (HyQMOM 5x5, about 42 iterations) converge
  instead of falling back to Gershgorin (which over-estimated the wave speed about 9x); a new optional
  `fallback` out-parameter reports when the fallback fires.
- **Overflow-safe index arithmetic in PoissonFFT** (ADC-286): the hot indexing (row offsets,
  eigenvalue scaling, transpose buffers) widens to `size_t` before multiplying, removing an int*int
  overflow past INT_MAX on large multi-rank grids (CodeQL). Numerically neutral, bit-identical under
  INT_MAX.
- **Documentation translated to English**: the whole Sphinx site (ADC-228), the root design guides
  and this changelog (ADC-241), the remaining French docs (ADC-245), `CONTRIBUTING` / `SECURITY` /
  `DOC_QUALITY` (ADC-227), and a restructured English README with a translation glossary
  (ADC-119/236/218).
- **BREAKING (C++)**: facade parameters regrouped into a POD struct, source-incompatible for C++
  callers; the Python API is unchanged (ADC-214).
- **Docs version single-sourced** (ADC-233): the docs build derives the version from
  `project(VERSION)` in `CMakeLists.txt` (`scripts/build_docs.sh` injects the Doxygen
  `PROJECT_NUMBER`, `docs/sphinx/conf.py` reads it), so Doxygen and Sphinx no longer drift.
- **Portability (LLP64 / Windows data model)**: `long` to `int64_t` in `mesh/` and the `box_hash`
  key, removing undefined `<< 32` shifts (ADC-209/216).
- **Internal C++ cleanups**: rule of five on owning types (ADC-212), bit-exact factorizations of
  duplicated FV/MG and Newton-Jacobian code (ADC-213), residual extended device lambdas converted
  to named functors (ADC-210), hardened runtime guards (ADC-211), and a coding-standards and
  comments audit (ADC-124/125).
- **Code comments and Python docstrings translated to English** (ADC-272): all of `include/pops/**`,
  `python/pops/**` (including the pybind bindings) and the `CODE_DOCUMENTATION_CONVENTION` guide; the
  published `/cpp/` Doxygen reference and the Python autodoc now render in English. Code structure is
  byte-identical; codegen-template strings and cross-TU dispatch tokens are kept verbatim.

### Fixed

- **GPU device-clean EB path**: `aux_comps()` is now `POPS_HD` so it can be evaluated inside the
  embedded-boundary device kernels (`load_aux<aux_comps<Model>()>` in `spatial_operator_eb.hpp`).
  nvcc rejected the constexpr `__host__` call from a `__host__ __device__` kernel (#20013-D), which
  broke the CUDA build of the magnetized EB diocotron on GH200 (ADC-306). Host builds unchanged
  (`POPS_HD` is empty off nvcc).
- **Periodic theta ghosts in the polar Schur source coupling**: the polar condensed-Schur stepper
  and the polar Krylov solver now force the azimuthal (theta) ghosts of phi periodic instead of
  honoring the caller's four-face Dirichlet BCRec. The theta=0/2pi seam was filled by odd reflection
  (ghost = -phi), injecting a dipole radial-momentum kick that grew like O(1/h) and diverged the
  polar Hoffart run near t~0.01. A step with the System-style Dirichlet BCRec is now bit-identical to
  the canonical theta-periodic step.
- **Stale phi ghosts on the direct FFT Poisson path** (ADC-175): `PoissonFFTSolver::solve()` now does
  a periodic `fill_boundary` on the phi ghosts after the solve; the direct solver wrote only valid
  cells, so a centered grad-phi for an electric source read stale domain-edge ghosts (wrong Ex on the
  boundary ring with `solver="fft"`).
- **Ghost-width guard on the EB and polar FV operators** (ADC-221): the structural
  `require_reconstruction_ghosts<Limiter>` check (ADC-163) now also runs at the entry of
  `assemble_rhs_eb` and `assemble_rhs_polar`, which reused the same reconstruction stencil with no
  input validation (bound `Limiter::n_ghost`, host-only; no sane production caller triggers it).
- **Non-circular spectral check in `check_model`**: `check_model` now bounds `max_wave_speed` by the
  spectral radius of the full dense flux Jacobian (central differences, independent of any `blocks=`
  partition), catching a `set_wave_speeds_from_jacobian` partition that silently fails to bound the
  spectrum (under-estimated CFL); tunable via `jac_rtol` / `jac_atol`. Also corrects a misleading
  duplicate-index message.
- clang portability: `make_system_coupler` factory replaces CTAD on the `SystemCoupler` alias
  template, which GCC accepts but clang rejects (P1814) -- this broke every clang build of the
  coupling tests, surfaced by the new TSan job (ADC-302).
- Heap-buffer-overflow masked on disc geometry, caught by a ghost-width guard at FV operator entry
  (ADC-163).
- Comment rot: 10 inaccurate comments corrected (ADC-208); broken sister-solver links in the docs
  index.
- **Docs (Pages) build**: `suppress_warnings = ["docutils"]` set unconditionally in
  `docs/sphinx/conf.py` so `sphinx -W` no longer fails on autodoc-rendered Doxygen `@param`
  docstrings (the meaningful broken-reference and toctree checks stay strict).
- **`master` gate unblocked** (ADC-281): `test_dispatch_tags` greps the dispatch-registry error
  fragments, which ADC-272 (#105) had translated to English in `dispatch_tags.hpp` without updating
  the test -- reconciled the grepped fragments (`unknown limiter` / `unknown Riemann flux` /
  `unsupported` / `polar`).
- **Python test message-assertions reconciled with ADC-272 translation** (ADC-283): ~26 assertions
  across 21 `python/tests/` files grepped now-translated French error fragments (masked by the
  gate-python swallow, ADC-112). Reconciled to assert language-stable substrings (echoed values,
  quoted tokens) so they survive the ongoing translation; also reconciled the order-sensitive
  `ABI incompatible` greps with the source's `incompatible ABI` -- the `test_dsl_production{,_amr}.py`
  asserts and the C++ `test_amr_native_loader.cpp` guard (`test_native_abi_std.py` rides along under
  this umbrella) -- plus the stale forward-order wording in comments and hints (`bindings.cpp`,
  `__init__.py`, `dsl.py`, `python/CMakeLists.txt`).
- **MPI FFT rejection test** (ADC-282): `test_mpi_system_fft` greps the translated
  `fft solver unsupported under MPI` message, completing the post-ADC-272 test fixups alongside
  ADC-281/283.
- **CI hardening**: the weekly quality jobs are bounded against cold-cache OOM (serial instrumented
  `ci-asan` / `ci-coverage` builds, `timeout-minutes` caps, and a single Ninja pool for the heavy
  Kokkos translation units; ADC-284/290), CodeQL is scoped to the pops sources to drop 187
  vendored-Kokkos findings (ADC-285), seven orphan DSL tests are registered in ctest with a
  self-contained `POPS_KOKKOS_ROOT` (ADC-104), and a `no-ai-authors` guard rejects AI authorship or
  co-author trailers.

## [0.1.0] - 2026-06-10

First numbered release (previously `0.0.1`, never exposed; Doxygen/Sphinx already announced
0.1.0, and this bump aligns the CMake single source with it).

### Added
- `pip install .` via scikit-build-core: module in site-packages, no PYTHONPATH; backends via
  environment variables (`POPS_USE_KOKKOS=ON Kokkos_ROOT=... pip install .`).
- `find_package(pops)`: install/export rules for the header-only core (`POPS_INSTALL`).
- `pops.__version__`, `pops.doctor()` (full diagnostic), `pops.set_threads()` /
  `pops.parallel_info()` / `pops.has_kokkos()`, `_pops.kokkos_is_initialized()`.
- CMake presets (`python`, `python-parallel`, `serial`, `parallel`, `mpi` plus the `ci-*` series
  used by CI: single source of the flags).
- Conda environment (`environment.yml`) plus `scripts/setup_env.sh` (per-platform toolchain
  pinned in the env) plus `pixi.toml` (reproducible cross-platform lockfile).
- `scripts/kokkos_openmp_conda.sh` (Kokkos Serial+OpenMP in the conda env, ~2 min);
  `scripts/build_docs.sh` (lint + Sphinx + Doxygen + site in one command); machine profile
  `Tools/machines/romeo/romeo_adc.profile.example`.
- Extended ABI key of the DSL production path: `kokkos=` and `stdlib=` tokens (divergences
  previously undetected), as a preprocessor literal (insensitive to ELF interposition).
- DSL runtime toolchain guards: build compiler baked in (`__cxx_compiler__`) and preferred over
  PATH, standard probe (`c++23`->`c++2b`), pre-dlopen module/header guard (including on cache HIT),
  compilation errors surfaced with the compiler output.

### Changed
- `POPS_BUILD_TESTS` follows `PROJECT_IS_TOP_LEVEL`: a FetchContent consumer no longer builds the
  test suite.
- `import pops` works without numpy (`pops.dsl` is lazy, with a targeted error at use).
- DSL `.so` cache: machine-aware key (arch + optflags) and fingerprint of the Kokkos install
  (`KokkosCore_config.h`); a different Kokkos invalidates the cache.
- Tests: ctest labels (`core`/`mpi`) plus timeouts; memory guard `-O0` plus the ninja pool
  extended automatically to any target compiling `system.cpp`/`amr_system.cpp` (39 objects).
- `pybind11` taken from the environment before any FetchContent; ccache auto-detected;
  `POPS_PY_LTO` option (OFF by default).

### Fixed
- Three real user bugs of the DSL production path: conda PATH compiler rejecting `-std=c++23`,
  stale module leading to a cryptic dlopen `symbol not found`, `CalledProcessError` without the
  compiler output.
- `find_package(Kokkos)` failed on macOS against a Kokkos OpenMP (libomp hints set before
  KokkosConfig's `find_dependency(OpenMP)`).
- Docs: pip/PYTHONPATH contradiction, phantom Catch2 mention, `$KOKKOS_ROOT` undefined in the
  tutorial, stale numpy claim, inconsistent versions (0.0.1 vs 0.1.0).
