# Design of the Python DSL model API (`dsl.Model`)

> Short user API: see [docs/DSL_API.md](DSL_API.md) ; this document = design + history.


Design + STATUS. This document was written as a target SPEC (API described and mapped
line by line onto the existing code). Almost all of it has since been shipped: Phase A,
native `production` backend (`System` AND `AmrSystem`), WENO5 on all paths, GPU/MPI
validated. The sections below keep the design reasoning, and section 0bis
(`Implementation status`) marks what is SHIPPED and what remains a GAP.
Each statement is anchored in a file that was read (cited `path:symbol`). The line
references are indicative (they date from the writing; the code has since moved).

RECOMMENDED public API (main user entry point):
- `adc.Model(state, transport, source, elliptic)` (`__init__.py:Model`): COMPOSE a model from
  already-compiled NATIVE bricks (`add_block` path, full production parity). This is
  the default route to assemble an existing model.
- `adc.dsl.Model(...)` (`dsl.py:Model`): WRITE a model as symbolic FORMULAS, then
  compile it. RECOMMENDED default: `backend="production"` (zero-copy native path, GPU/MPI validated).
- ADVANCED / LEGACY / TEST paths (NOT the main user route): `backend="prototype"`
  (JIT proto), `backend="aot"` (`.so` single-rank with marshaling), `add_dynamic_block`,
  `add_compiled_block`, and `adc.PythonFlux` (numpy HOST path to TEST a flux, outside the GPU/MPI
  hot path, never production). To use only for prototyping or numerical debugging.

## 0bis. Implementation status (up to date with `origin/master`)

Quick recap; the per-section detail follows (the SHIPPED/GAP tags are repeated there).

SHIPPED (no longer read as "target"):
- **Phase A** (#89/#90): `dsl.Model` facade (`dsl.py:Model`), named `Param` (`dsl.py:Param`,
  `const` mode inline; `runtime` mode SHIPPED on `aot`, P7-b, cf. bullet "Runtime params"),
  `CompiledModel` (`dsl.py:CompiledModel`,
  carries `abi_key`/`model_hash`/`cxx`/`std`), `System.add_equation` (`__init__.py:add_equation`,
  dispatch `ModelSpec`->`add_block` vs `CompiledModel`->backend adder), `adc.FiniteVolume(limiter=,
  riemann=, variables=)` (`__init__.py:FiniteVolume`), `System.run(t_end, cfl)` (`__init__.py:run`).
  `m.flux`/`m.eval_flux` (declarator vs evaluator, distinct names), `m.primitive_vars(**kwargs)`.
- **Native `production` backend** (#85): `production` is NO LONGER an alias of `aot`. `_BACKENDS`
  (`dsl.py`) maps `production -> ("native", "add_native_block")`; `compile_native` emits a native
  LOADER (`emit_cpp_native_loader`) that inlines `add_compiled_model<ProdModel>` on the REAL
  `grid_context()` of the `System` (zero-copy, `add_block` parity), with an ABI-key safeguard
  (`add_native_block`, `system.cpp:902`).
- **`AmrSystem` in native production** (#92): `adc.AmrSystem.add_native_block` and
  `m.compile(target="amr_system")` SHIPPED; the native counterpart `add_compiled_model(AmrSystem&)`
  (`amr_dsl_block.hpp`) now has its Python binding. No longer read 0bis/section 7 as
  "raises `NotImplementedError`" or "no Python binding" for this path.
- **WENO5 everywhere**: SHIPPED on the `.so`/`CompiledModel` paths (`aot` AND `production`, #102;
  the `.so` allocates its ghosts according to the limiter, hence 3 ghosts for WENO5) AND on the native
  AMR path (#105: WENO5 + Rusanov + conservative reconstruction; parity
  `add_native_block` == `add_compiled_model` == `add_block`, `dmax=0`). No more "weno5 /
  2 ghosts" rejection for the `.so`. (The native `add_block` path via `adc.Model` already carried
  WENO5/SSPRK3 since #88.)
- **GPU / MPI validated**: production `System` GPU `np=1` VALIDATED GH200 (#97); `solve_fields` MPI
  `np=1/2/4` VALIDATED CPU/CI (#99); device-MPI production `geometric_mg` VALIDATED GH200 `np=1/2/4`
  (#93). The device/MPI validation of the native `production` path is therefore ACQUIRED (no longer say
  "device crash in `solve_fields`" nor "GPU validation not acquired").
- **`compile()` ergonomics + cache** (#103): `m.compile()` no longer requires the output path nor
  the header directory (defaults are derived); the `.so` is cached by `so_path` (key =
  `model_hash`), so an unchanged model is not recompiled.
- **DSL demonstrators**: `diocotron_dsl`, `two_species_dsl`, `magnetic_isothermal_dsl` are
  complete and all marked `ci=true` in `adc_cases` (CI coverage).

GAP (still target / deferred):
- **`fft` under MPI `System` `np>1`**: REFUSED cleanly (#106, hard safeguard, no more segfault).
  A `DistributedFFTSolver` exists and is tested separately, but it is NOT routed in `System` (its
  band layout does not match the single-box layout expected by `System`). MPI `np>1` must therefore
  use `geometric_mg`, not `fft`.
- **`AmrSystem.potential()`**: binding SHIPPED (python/bindings.cpp:272, `#135`). Acquired.
- **Wall-transport Phase 1**: EXPERIMENTAL, closed WITHOUT merge (#109). It masks the external
  CONDUCTOR (wrong boundary); the scientific lock remains the RING BOUNDARY. DO NOT read as shipped.
- **Runtime params**: `m.param(kind="runtime")` is SHIPPED on the "aot" backend (P7-b): the param
  emits `params.get(<index>)` (member `adc::RuntimeParams` of the brick, include/adc/runtime/
  runtime_params.hpp), the AOT ABI carries a block of values (symbols `_p`), and
  `System.set_block_params(name, values)` changes the value at RUNTIME without recompiling. The
  "prototype"/"production" backends compile a runtime param as its DECLARATION value (frozen: the model
  is default-constructed there, no injection). `const` params stay INLINED (bit-identical).
- **`AmrSystem` single- AND multi-block**: multi-block is SHIPPED (N co-located blocks on ONE
  shared AMR hierarchy via the `AmrRuntime` engine, system Poisson with SUMMED right-hand side, regrid
  of the union of tags, multi-block production DSL; cf. `AmrSystem` `__init__.py` and `amr_system.cpp`
  `build_multi`). `AmrSystem` LIMITS STILL REAL (to keep honest): IMEX local
  source OK (Gap 2 #132, backward_euler_source / mf_apply_source_treatment) but global Schur on
  AMR remains to be done, and
  HLLC/Roe/`primitive` REJECTED on the Python AMR facade side. This rejection is PURELY FACADE: the C++
  engine (`amr_dsl_block.hpp`/`make_block`) already supports them; only the Python facade does not expose
  them yet on AMR.

HISTORICAL NOTE. The text below still uses the future tense ("TARGET", "to add") where the
thing is now shipped; rely on 0bis and the SHIPPED/GAP tags. The design content is
kept on purpose (justifications, anchors, error taxonomy).

Sources read for this design:
- `python/adc/dsl.py`: `HyperbolicModel` (all methods), the codegen
  (`emit_cpp_brick`/`emit_cpp_source`/`emit_cpp_elliptic`/`emit_cpp_so_source`/
  `emit_cpp_aot_source`), the `compile(backend=)` facade + `_BACKENDS` + `adder_for`.
- `python/adc/__init__.py`: runtime sugar `Model`/`Spatial`/`Explicit`/`IMEX`/
  `Implicit`/`DivEpsGrad`/`System`/`AmrSystem`/`PythonFlux`.
- `python/system.cpp` + `python/bindings.cpp`: adders `add_block`,
  `add_dynamic_block`, `add_compiled_block`, and metadata (`read_block_meta`,
  `variable_names`/`variable_roles`/`block_gamma`), `set_poisson`, `step_cfl`/
  `step_adaptive`.
- `include/adc/core/variables.hpp`: `VariableRole`/`VariableSet`/`role_from_name`/
  `ADC_EXPORT_BLOCK_METADATA`/`ADC_EXPORT_BLOCK_GAMMA`.
- `include/adc/core/physical_model.hpp`: contract `PhysicalModel`/
  `HyperbolicPhysicalModel`/`aux_comps`.
- `include/adc/runtime/compiled_block_abi.hpp`: AOT ABI `ADC_DEFINE_COMPILED_BLOCK`.
- `include/adc/runtime/dynamic_model.hpp`: `IModel`/`ModelAdapter` (JIT).
- `include/adc/runtime/dsl_block.hpp`: `add_compiled_model` (native, template).
- `include/adc/runtime/amr_dsl_block.hpp`: `add_compiled_model` on the `AmrSystem` side.
- `docs/PAPER_ROADMAP.md` (basket 2), `docs/ARCHITECTURE.md` (runtime/DSL section).

ENVIRONMENT NOTE. A sibling agent adds in parallel a real native `production`
path (native loader, `System` adder, ABI-key safeguard). This spec is
written AROUND this direction (production = zero-copy native `add_compiled_model`,
NOT the host `add_compiled_block` with marshaling) without presuming its exact
symbol names: we describe the CONTRACT, not the sibling's implementation.


## Settled decisions (review)

API points fixed after review, to avoid ambiguity at implementation:
1. `m.flux(x=, y=)` = symbolic DECLARATOR; the numpy evaluator is `m.eval_flux(...)` (distinct names). (section 1)
2. `m.primitive_vars(rho=expr, ...)` accepts KWARGS (order = `Prim` layout); positional form too. (section 1)
3. The NUMERICAL flux is `adc.FiniteVolume(limiter=, riemann=, variables=)` -- `riemann`, NOT `flux` (which is the physical flux). (section 6)
4. `m.param(name, value)` returns a NAMED `Param` object (`name`/`value`/`kind`), not an anonymous `Const`. (section 2)
5. `CompiledModel` carries `abi_key` + `model_hash` + build flags, not just `so_path`/backend/names. (section 3)
6. Native GPU status ACQUIRED: the device-clean path with named functors is VALIDATED GH200, Python binding included (System GPU np=1 #97, device-MPI geometric_mg np=1/2/4 #93). Production GPU is NOT broken. (section 5)
7. `m.compile(backend, target)` WITHOUT `device=`: GPU/MPI/AMR capabilities verified at attach/execution, not frozen at compilation. (sections 1, 5, 7)


## 0. Current state (factual, starting point)

`HyperbolicModel` (`dsl.py:266`) is the ONLY model object. It carries the formulas
(`Expr` tree), three groups of generators (`emit_cpp_brick`, `emit_cpp_source`,
`emit_cpp_elliptic`), three wrappers (`emit_cpp_so_source` JIT, `emit_cpp_aot_source`
AOT, and the shared `_emit_bricks`/`_emit_metadata` factory), and the
`compile(backend=)` facade. Three backends are declared (`dsl.py:_BACKENDS`):
`prototype` -> (`jit`, `add_dynamic_block`), `aot` -> (`compile`, `add_compiled_block`),
`production` -> (`native`, `add_native_block`).

> SHIPPED (#85). At the time of writing, `production` was an ALIAS of `aot`. That is NO LONGER the
> case: `_BACKENDS["production"] = ("native", "add_native_block")`, and `compile_native`
> (`emit_cpp_native_loader`) emits the zero-copy native LOADER described in point 3 below. The rest
> of this section 0 keeps the analysis of the 3 C++ paths, still exact.

Three execution paths exist on the C++ side:
1. JIT: `.so` exposes `adc_make_model`/`adc_model_nvars`/`adc_destroy_model`
   (`dsl.py:emit_cpp_so_source`), loaded by `System::add_dynamic_block`
   (`system.cpp:706`) as `IModel<NV>` with VIRTUAL dispatch, order-1 Rusanov HOST residual
   (`system.cpp:host_residual`). Outside the GPU/MPI hot path (`dynamic_model.hpp:18`).
2. AOT: `.so` exposes the `ADC_DEFINE_COMPILED_BLOCK` ABI (`compiled_block_abi.hpp:156`),
   loaded by `System::add_compiled_block` (`system.cpp:738`). Runs the production
   path (`make_block`<Limiter,Flux>, SSPRK2/IMEX) BUT on a local grid
   reconstructed in the `.so` with flat-array marshaling (`copy_state`/
   `write_state`), hence NOT zero-copy, single-rank (`compiled_block_abi.hpp:56
   DistributionMapping dm(ba.size(), 1)`, comment `:24-26`: "without AMR/MPI").
3. NATIVE: `add_compiled_model(System&, ...)` (`dsl_block.hpp:36`), C++ template
   that builds the closures on the REAL `grid_context()` of the `System` via
   `make_block` (bit-identical parity to `add_block`, validated CPU and GH200,
   `dsl_block.hpp:18-29`). AMR counterpart: `add_compiled_model(AmrSystem&, ...)`
   (`amr_dsl_block.hpp`). This is the target of the `production` backend, now REACHED for
   `System` (#85) AND `AmrSystem` (#92) (see SHIPPED below).

> SHIPPED (#85 for `System`, #92 for `AmrSystem`). The `add_compiled_model(System&, ...)` template has
> a Python path: `compile_native` (`dsl.py:emit_cpp_native_loader`) emits a `.so` LOADER (two
> `extern "C"` symbols: `adc_native_abi_key` + `adc_install_native`) that inlines
> `add_compiled_model<ProdModel>` on the REAL `grid_context()` of the `System`. `System.add_native_block`
> (`system.cpp:902`) `dlopen`s it, compares the ABI key (explicit rejection if headers/compiler/std
> diverge) then calls the installer: zero-copy native block, `add_block` parity. The counterpart
> `add_compiled_model(AmrSystem&)` (`amr_dsl_block.hpp`) now also has a Python binding
> (#92): `adc.AmrSystem.add_native_block` + `m.compile(target="amr_system")`; parity
> `add_native_block` == `add_compiled_model` == `add_block` (#105, `dmax=0`).

> SHIPPED (#89). `dsl.Model` now EXISTS (`dsl.py:Model`, module `adc.dsl`). It is the facade
> that COMPOSES a private `HyperbolicModel` (`_m`) and delegates each call. The name `adc.Model`
> (`__init__.py:Model`) remains the distinct function that composes a `ModelSpec` of NATIVE bricks
> (path (a)); the two coexist as planned (one in `adc.dsl`, the other in `adc`).


## 1. Stable `dsl.Model` facade

> SHIPPED (#89). This section is DELIVERED: `dsl.Model`, `m.flux`/`m.eval_flux`,
> `m.primitive_vars(**kwargs)`, `m.param`, `m.compile(backend, target)` exist
> (`dsl.py:Model`). The mapping table below describes the actual implementation.

TARGET: `dsl.Model` is the stable SURFACE; `HyperbolicModel` remains the unchanged
internal BACKEND. `dsl.Model` delegates each call to an existing method of
`HyperbolicModel` (composition, not inheritance: `dsl.Model` holds a private
`HyperbolicModel` `_m`). No numerics is touched.

Construction: `m = dsl.Model("euler")` creates the internal `HyperbolicModel("euler")`.

Mapping of target method -> backing `HyperbolicModel`:

| `dsl.Model` (target) | backed by `HyperbolicModel` (`dsl.py`) | note |
|---|---|---|
| `m.conservative_vars(*names, roles=)` | `conservative_vars(*names, roles=)` `:291` | identical (forwards `roles=`) |
| `m.primitive_vars(rho=expr, u=expr, ...)` (kwargs) or `(*vars, roles=)` | `set_primitive_state(*vars_or_names, roles=)` `:323` + `primitive(name, expr)` `:301` | TARGET STYLE = KWARGS `name=expr`: each kwarg defines a primitive (`_m.primitive(name, expr)`) AND fixes the ORDERED layout of `Prim` in kwarg order (Python 3.7+: insertion order guaranteed). The positional form `(*vars, roles=)` remains accepted. The `roles=` (kwarg or list) remain supported for the role->index mapping |
| `m.primitive(name, expr)` | `primitive(name, expr)` `:301` | definition of a primitive by formula |
| `m.aux(name)` | `aux(name)` `:306` | auxiliary field (must be a key of `AUX_CANONICAL` `:35`) |
| `m.conservative_from(exprs)` | `set_conservative_from(exprs)` `:334` | inverse prim->cons (the DSL cannot invert) |
| `m.flux(x=, y=)` | `set_flux(x, y)` `:311` | symbolic DECLARATOR of the physical flux (settled decision, see below) |
| `m.eval_flux(U, aux, dir)` | `flux(U, aux, dir)` `:354` | numpy EVALUATOR (debug / host proto); name DISTINCT from the declarator `m.flux` |
| `m.source(s)` | `set_source(s)` `:313` | optional |
| `m.eigenvalues(x=, y=)` | `set_eigenvalues(x, y)` `:312` | |
| `m.elliptic_rhs(e)` | `set_elliptic_rhs(e)` `:314` | optional (Poisson coupling) |
| `m.gamma(value)` or `m.set_gamma(value)` | `set_gamma(gamma)` `:316` | EOS, carried by `ADC_EXPORT_BLOCK_GAMMA` |
| `m.param(name, value)` | `Param` + `Model.param` (#89) | SHIPPED in `const` mode (NAMED constant inlined at codegen, stored in `m.params`); `kind="runtime"` raises `NotImplementedError` (section 2b, GAP). Case `name=="gamma"` also calls `set_gamma` |
| `m.check()` | `check()` `:382` | checks dependencies |
| `m.compile(backend, target)` | `Model.compile(...)` (#89, delegates to `HyperbolicModel.compile`) | SHIPPED: returns a `CompiledModel` (section 3). `target="system"` AND `target="amr_system"` wired (#92). Ergonomics #103: `so_path`/`include` have defaults, the `.so` is cached by `model_hash`. NO `device` argument: capabilities (GPU/MPI/AMR) are verified AT ATTACH/EXECUTION (`add_equation`/`run`), not frozen as a compilation flag (this would avoid a false guarantee if the module is not built with Kokkos/CUDA). See sections 5 and 7 |

NAME COLLISION: SETTLED. In `HyperbolicModel`, `flux(U, aux, dir)` `:354` is
the numpy EVALUATOR (CPU interpreter) and `set_flux` `:311` is the DECLARATOR. The target
plan names the declarator `m.flux`. DECISION: on `dsl.Model`, `m.flux(x=, y=)` is the
symbolic DECLARATOR (delegates to `set_flux`); the numpy evaluator is exposed under the
DISTINCT name `m.eval_flux(U, aux, dir)` (delegates to `_m.flux`). The declarative surface
prevails; no name carries both senses. (No `m.set_flux` alias on the facade: a
single name per intent.)

State of these target points (initially GAPs; see 0bis):
- `m.param(name, value)`: SHIPPED in `const` mode (#89, `Param` class + `Model.param`,
  `dsl.py`). `runtime` mode remains GAP (section 2b, `NotImplementedError`).
- `m.compile(target=)`: SHIPPED (#89). `Model.compile` takes `backend`, `target`, `name`, `cxx`,
  `std`, `require_metadata`. `target="system"` AND `target="amr_system"` wired (#92). Ergonomics
  #103: `so_path`/`include` have defaults, cache by `model_hash`. NO `device=` (point 7):
  capabilities verified at attach (`add_equation`) / at execution (`run`), not frozen as a
  compilation flag (a `device=True` at compilation would give a false guarantee: a
  `.so` can compile without the host module being device-capable).
- `m.compile()` now returns a `CompiledModel` (#89, section 3), not a `str so_path`.


## 2. `m.param(name, value)`: two modes

> STATUS. Mode (a) SHIPPED (#89, `Param` class, `dsl.py`). Mode (b) GAP (raises
> `NotImplementedError`; ABI/codegen change, Phase E). The reasoning below
> documents why (b) is not deliverable without an engine change.

### Mode (a): constant frozen at compilation (FEASIBLE today)

The codegen already INLINES any constant. A Python scalar passed in a formula is
promoted to `Const(float(o))` (`dsl.py:_wrap :110`), and `Const.to_cpp()` returns
`repr(self.value)` (`dsl.py:117`): the value is written HARD into the `.so`. Real
example: `dsl_euler/run.py` writes `GAMMA = 1.4` then `(GAMMA - 1.0) * (...)`; the `1.4`
is inlined. So `m.param("gamma", 1.4)` in constant mode needs NO new
engine: it is Python sugar that returns a `Const` (or a scalar) reusable
in several formulas, recompiled at each value change.

Proposed form: `g = m.param("gamma", 1.4)` returns a NAMED `Param` object (not an
anonymous `Const`), carrier of its IDENTITY: `name`, `value`, `kind` (`"const"` at the start,
`"runtime"` reserved, see mode b). `Param` behaves like an `Expr` in the formulas
(it INLINES to `Const(value)` at codegen, hence zero-risk on the generated brick
side), but KEEPS name/value/kind for: introspection (`m.params`), logs/diagnostics, the
reproducibility (a run traces its parameters), and the future transition toward runtime
params (mode b) WITHOUT changing the user surface. Changing the value requires
`m.compile(...)` again (everything is recompiled today).

SPECIAL CASE already wired: `gamma` has a dedicated channel outside the formula via `set_gamma`
`:316` -> `ADC_EXPORT_BLOCK_GAMMA` (`variables.hpp:153`), read by `read_block_meta`
(`system.cpp:179`) for inter-species couplings. `m.param("gamma", ...)` should
therefore, besides inlining it in the formulas, call `set_gamma` so that the
ABI metadata is coherent (otherwise the `System` falls back to 1.4, `system.cpp:629`/`:844`).

### Mode (b): runtime parameter (modifiable WITHOUT recompiling) -> LATER PHASE

UNFEASIBLE with the current codegen without an engine change. Anchored justification:
- The codegen has NO concept of uniform/member. The generated bricks read
  only the conservative variables (`U[i]`, `cons_locals` `:468`), the
  derived primitives (`prim_locals` `:471`), and the `Aux` fields (`a.<name>`,
  `aux_locals` `:474`). Any other value is an inline `Const`.
- The AOT ABI (`compiled_block_abi.hpp:156`) carries no parameter: the
  signatures `adc_compiled_residual`/`advance`/`max_speed` (`compiled_block_abi.hpp:159-176`)
  take `U`, `aux`, geometry, scheme, no parameter block. `Model model{}`
  is default-constructed (`compiled_block_abi.hpp:103`,`:118`,`:130`,`:143`), without state.
- On the JIT side, `IModel`/`ModelAdapter` (`dynamic_model.hpp`) is also without
  parametric state: `M model{}` default-constructed (`dynamic_model.hpp:51`).

Two possible routes for a REAL runtime parameter, both PHASE 2:
1. Pass the parameters via the `Aux` channel: declare a constant aux field per
   cell and populate it from Python (as `set_magnetic_field` populates `B_z`,
   `system.cpp:911`). Feasible WITHOUT changing the codegen (the param becomes an `m.aux`),
   but limited to the 5 canonical components (`AUX_CANONICAL` `:35`) and at the cost of an
   n*n field for a scalar (abuse of the aux channel).
2. Extend the ABI: add a parameter/block of doubles to the signatures
   `adc_compiled_*`, generate a `Model` constructed with these parameters (members), and
   propagate via `set_density`-like. This is an ABI CHANGE (header
   `compiled_block_abi.hpp` + reading `system.cpp:add_compiled_block`) AND a codegen change
   (struct with members + constructor). To mark explicitly PHASE 2.

VERDICT. `m.param(name, value)` is deliverable in mode (a) (frozen constant) immediately
and without risk. Mode (b) (runtime, without recompilation) requires an engine change
(ABI + codegen) and remains a later phase; `m.param` must therefore either support
only (a) at the start, or accept a `kind="const"|"runtime"` where `kind="runtime"` RAISES
explicitly `NotImplementedError` (see taxonomy section 4).


## 3. Python `CompiledModel` object

> SHIPPED (#89). `CompiledModel` exists (`dsl.py:CompiledModel`) and is produced by
> `Model.compile`. It carries all the fields below (`abi_key`, `model_hash`, `cxx`,
> `std`, `caps`, `params`...). `System.add_equation` (#89/#90) consumes it and routes
> to the backend adder. The table and the consumption subsection describe the actual implementation;
> see the inline notes for the `production` case (now native, no longer an alias of `aot`).

At the time of writing, `compile()` returned a `str` (`so_path`) and exposed separately
`adder_for(backend)` to know which `System` adder to use. The `CompiledModel` replaces
this `(str, classmethod)` pair with an object that CARRIES the path and everything needed to
wire it correctly.

### Fields

| field | source (anchor) | role |
|---|---|---|
| `so_path` | return of `compile_so`/`compile_aot` (`dsl.py:705`/`:743`) | path of the `.so` |
| `backend` | `backend=` passed to `compile` | `prototype`/`aot`/`production` |
| `adder` | `_BACKENDS[backend][1]` (`dsl.py:792`) | name of the `System` method to use |
| `cons_names` | `HyperbolicModel.cons_names` | conservative names (override `names=`) |
| `cons_roles` | `roles_for(cons_names, cons_roles)` (`dsl.py:77`) | physical roles |
| `prim_names` | `HyperbolicModel.prim_state` | primitive layout |
| `n_vars` | `HyperbolicModel.n_vars` (`dsl.py:340`) | number of components |
| `gamma` | `HyperbolicModel.gamma` (`dsl.py:316`) | EOS (None = default 1.4) |
| `n_aux` | `aux_n_aux(aux_names)` (`dsl.py:39`) | required aux channel width |
| `params` | dict `{name: Param}` (named `Param` objects, section 2) | introspection / reproducibility |
| `caps` | derived from the backend (section 5) | CPU/MPI/AMR/GPU flags |
| `abi_key` | baked ABI key (compiler + std + header signature, cf. `abi_key.hpp` / `adc_cases/common/native.py`) | refuse an incompatible `.so` AT LOAD rather than a silent UB |
| `model_hash` | stable hash of the model (formulas + roles + n_aux + params) | identify/reuse an already-compiled `.so`; trace the run |
| `cxx`, `std`, `cxx_flags` | compiler, standard, flags passed to compilation | reproducibility + ABI incompatibility diagnostic |

`CompiledModel` is PRODUCED by `m.compile(...)`: the facade compiles the `.so` (via
`compile_so`/`compile_aot` unchanged), then packages the path with the
metadata already known to the `HyperbolicModel` (no re-reading of the `.so`: Python
already holds names/roles/gamma/n_aux). The same metadata is ALREADY emitted in the
`.so` by `_emit_metadata` (`dsl.py:675`) and re-read on the C++ side by `read_block_meta`
(`system.cpp:179`); `CompiledModel` just exposes them on the Python side for dispatch and
diagnostics, without a new source of truth.

A `CompiledModel` is therefore NOT a simple `str so_path`: it also carries the ABI KEY
and the build flags (`abi_key`, `model_hash`, `cxx`/`std`/`cxx_flags`). This is what
allows refusing at LOAD a `.so` compiled with an incompatible state (compiler,
standard, divergent headers) instead of a silent undefined behavior: the native
`production` path already compares this key on the C++ side (`abi_key.hpp`, loader safeguard),
and `CompiledModel.abi_key` makes the check + the diagnostic available on the Python side.

### Consumption: `System.add_equation(model=compiled, ...)`

TARGET: `sim.add_equation(name, model=compiled, spatial=, time=, substeps=, names=)`
dispatches to the right adder according to `compiled.backend`/`compiled.adder`:

- `backend="prototype"` (`adder="add_dynamic_block"`) ->
  `self._s.add_dynamic_block(name, compiled.so_path, substeps, names, recon)`
  (`bindings.cpp:78`, `system.cpp:706`). NB: `add_dynamic_block` does NOT take
  `limiter`/`riemann`/`time` (order-1 Rusanov host path); it takes `recon`
  (`none`/`minmod`/`vanleer`) and `substeps`. The facade must therefore IGNORE (or refuse,
  section 4) a `spatial.riemann != "rusanov"` for this backend (`riemann` = NUMERICAL flux
  of `FiniteVolume`, cf. section 6; `flux` remains the PHYSICAL flux of the model).
- `backend="aot"` (`adder="add_compiled_block"`) ->
  `self._s.add_compiled_block(name, compiled.so_path, spatial.limiter, spatial.riemann,
  spatial.variables, time.kind, substeps, names)` (`system.cpp:776`; the C++ arg of the numerical
  flux is already called `riemann`, `variables` maps `recon`).
- `backend="production"` (`adder="add_native_block"`) SHIPPED (#85) -> `self._s.add_native_block(name,
  compiled.so_path, spatial.limiter, spatial.riemann, spatial.variables, time.kind, gamma, substeps,
  evolve)` (`system.cpp:902`). This is NO LONGER the alias of `aot`: the zero-copy native adder EXISTS
  (`.so` loader -> `add_compiled_model<ProdModel>` on the real context; ABI key verified).
  NB: this path does NOT take `names=` (names/roles come from the metadata of the `.so`); `add_equation`
  raises if `names=` is provided (`__init__.py:add_equation`).

`add_equation` is SHIPPED (#89/#90, `__init__.py:add_equation`): it is the method of the Python
`System` facade that routes according to the type of `model`: a `ModelSpec` (`adc.Model(...)`) ->
`add_block`; a `CompiledModel` -> the compiled/dynamic/native block adder fixed by the backend.
`System.add_block` (`__init__.py:add_block`) takes a `ModelSpec` of native bricks, NOT a `.so`.
`add_equation` centralizes the backend<->adder coupling that `dsl.py` documents as a safety
rule ("do not wire an AOT `.so` onto `add_dynamic_block`").


## 4. Error taxonomy

All errors are `ValueError` (or `NotImplementedError` for the deferred),
raised AS EARLY AS POSSIBLE (at declaration or at `compile`/`add_equation`, not at execution),
with a FACTUAL message naming the cause and the corrective action. Modeled on the existing
messages of `dsl.py` (e.g. `:296`, `:822`, `:845`).

| error | when | message form (template) | anchor of the existing guard |
|---|---|---|---|
| required role missing | `compile(require_metadata=True)` without role or canonical name | "model '<name>' does not provide physical roles (...); the .so would fall back to the fallback (roles 'custom')" | ALREADY implemented `dsl.py:837-847` |
| required gamma missing | idem, `gamma is None` | "(...) does not provide gamma (set_gamma(...)) (...)" | ALREADY implemented `dsl.py:842` |
| undefined param | a formula references a `param` never set | "param '<name>' referenced but not defined (m.param('<name>', value))" | extends `check()` `:382` (which already raises on undeclared variable, `:397`) |
| param runtime mode | `m.param(name, value, kind="runtime")` | "runtime param not supported (ABI/codegen change required, later phase); use a constant param or an aux field" | NEW, `NotImplementedError` (section 2 mode b) |
| unknown backend | `compile(backend=x)` outside `_BACKENDS` | "compile: backend <x> unknown (expected ['aot','production','prototype'])" | ALREADY `dsl.py:821-823` |
| flux incompatible variables | `spatial.riemann in {hllc,roe}` without declared primitive `p` | "riemann 'hllc'/'roe' requires a pressure: declare m.primitive('p', ...)" | physical ANCHOR: the brick only emits `pressure`/`wave_speeds` if `'p' in prim_defs` (`dsl.py:558`); `make_block` requires `m.pressure`/`m.wave_speeds` for HLLC/Roe (cf. `amr_dsl_block.hpp:135`) |
| GPU/MPI/AMR incompatible backend | `add_equation` with `device="gpu"`/MPI/AMR on a non-capable backend | "backend '<b>' is not device-clean / multi-rank: use backend='production' (native path)" | section 5; `compiled_block_abi.hpp:24-26` ("without AMR/MPI"), `dynamic_model.hpp:18-21` (outside the GPU hot path) |
| fft under MPI System np>1 | `run`/`solve_fields` with an `fft` Poisson solver and np>1 | "fft not supported under System in multi-rank MPI (band layout vs single box); use geometric_mg" | OBSOLETE the old "production->aot" case (#85: no more alias). Hard safeguard SHIPPED (#106): clean refusal, no more segfault. `DistributedFFTSolver` exists but not routed in System |
| require_metadata on prototype | `compile(backend="prototype", require_metadata=True)` | "backend 'prototype' (...) incompatible with require_metadata=True; use 'aot' or 'production'" | ALREADY `dsl.py:830-834` |
| names= wrong length | `add_equation(names=)` of length != n_vars | "names= has K names but block '<n>' has NV variables" | ALREADY on the C++ side `system.cpp:618`/`:832` |

CURRENT PRIORITY of name/role resolution (to document for the user, NOT to
change): explicit `names=` > ABI metadata of the `.so` (`read_block_meta`) > fallback
`u0..` (`system.cpp:826-844` and `:614-628`). The ROLES and the PRIMITIVE come ONLY
from the ABI (the `names=` API does not provide them, `system.cpp:828`).


## 5. Backend capability matrix

HONEST state. Rows = `compile()` backend, columns = capability. The matrix is also
materialized on the code side in `_BACKEND_CAPS` (`dsl.py`), read by `CompiledModel.caps`.

| backend | serial CPU | MPI | AMR | GPU/Kokkos | device zero-copy | Python callbacks in hot path |
|---|---|---|---|---|---|---|
| `prototype` (JIT, `add_dynamic_block`) | yes (host Rusanov o1 residual) | no | no | no | no | no (C++ virtual dispatch, without GIL) |
| `aot` (`add_compiled_block`) | yes (production o2, HLLC/Roe, WENO5 #102) | no | no | no | no | no |
| `production` (native `add_native_block`, #85/#92) | yes | yes (np=1/2/4 #99/#93) | yes via `AmrSystem` (#92, single- AND multi-block) | yes (GH200 np=1 #97; device-MPI np=1/2/4 #93) | yes | no |

> SHIPPED (#85 System, #92 AmrSystem). `production` is NO LONGER the alias of `aot`.
> `_BACKEND_CAPS["production"]` (`dsl.py`) declares `{cpu, mpi, amr}=True`; `gpu` is reported `False`
> on the Python side (the host module tested in CI is not built Kokkos/CUDA -- the GPU capability of the
> native path is validated on the C++ side GH200, not exposed as `True` from this build). Production
> System GPU `np=1` VALIDATED GH200 (#97), `solve_fields` MPI `np=1/2/4` VALIDATED CPU/CI (#99),
> device-MPI production `geometric_mg` VALIDATED GH200 `np=1/2/4` (#93). AMR is SHIPPED via
> `AmrSystem.add_native_block` (#92): single- AND multi-block (shared hierarchy `AmrRuntime`,
> multi-block production DSL). Remains BOUNDED (explicit, HLLC/Roe/`primitive` rejected on the Python
> AMR facade side while the C++ engine carries them). These capabilities are diagnostic flags verified at
> attach/execution, not frozen at compilation (point 7).

> MPI REMINDER. `fft` is NOT supported under `System` in MPI `np>1`: hard refusal SHIPPED (#106, no
> more segfault), use `geometric_mg`. A `DistributedFFTSolver` exists (tested separately) but is
> not routed in `System` (band layout vs single box).

Anchors:
- `prototype`/JIT: `dynamic_model.hpp:18-21` ("HOST PATH / PROTOTYPING only;
  the virtual calls do NOT go through a GPU kernel (...) do not use in the
  hot loop"). Order-1 Rusanov host residual: `system.cpp:host_residual`, MUSCL
  host recon 0/minmod/vanleer (`system.cpp:41`). No HLLC/Roe flux (Rusanov frozen).
- `aot`: runs the production path `make_block`<Limiter,Flux> / SSPRK2 / IMEX
  (`compiled_block_abi.hpp:21-26`), hence HLLC/Roe, order 2 and WENO5 available (#102: the `.so`
  allocates its ghosts according to the limiter, 3 ghosts for WENO5). BUT on a single-rank LOCAL grid
  reconstructed in the `.so` (`compiled_block_abi.hpp:56` `DistributionMapping dm(ba.size(), 1)`)
  with `copy_state`/`write_state` marshaling (`system.cpp:794-824`): not zero-copy, no
  MPI halos, no AMR (`ARCHITECTURE.md:499` "without AMR/MPI").
- `production` SHIPPED (#85 System, #92 AmrSystem): `add_compiled_model` (`dsl_block.hpp`) builds
  the closures on the REAL `grid_context()`, residual via `make_block` with `fill_boundary`
  (MPI halos) + `assemble_rhs` Kokkos on the real `MultiFab`, WITHOUT recopy; bit-identical
  `add_block` parity validated CPU/Serial AND GH200 (named functors). The `production`
  backend is RE-ROUTED from Python: `compile_native` emits a `.so` loader,
  `System.add_native_block` (`system.cpp:902`) `dlopen`s it, verifies the ABI key, then calls
  `add_compiled_model<ProdModel>`. AMR: `add_compiled_model(AmrSystem&)` (`amr_dsl_block.hpp`) now
  has its Python binding (`adc.AmrSystem.add_native_block`, #92); WENO5 + Rusanov +
  conservative on this native AMR path (#105; parity `add_native_block` == `add_compiled_model`
  == `add_block`, `dmax=0`). GPU STATUS: the OLD path with `__host__ __device__` EXTENDED LAMBDAS
  segfaulted on Cuda (nvcc limit); it was replaced by the device-clean NAMED FUNCTORS of
  `block_builder.hpp`, and it is THAT path that `add_compiled_model` takes. VALIDATED end to end
  FROM Python: production System GPU `np=1` (#97), device-MPI production `geometric_mg`
  `np=1/2/4` (#93), `solve_fields` MPI `np=1/2/4` CPU/CI (#99). The device/MPI validation is therefore
  no longer a GAP.
- No backend executes a Python callback per cell: even `prototype` is a
  C++ model (from the codegen), not an `adc.PythonFlux`. `adc.PythonFlux`
  (`__init__.py:420`) is a SEPARATE HOST numpy path, outside the compiled DSL, OUTSIDE the GPU/MPI hot path:
  it is a TEST tool (verify a flux), never production.


## 6. Runtime sugar of the plan -> existing

Mapping of the plan's target runtime objects/methods onto what EXISTS
(`__init__.py`, `bindings.cpp`, `system.cpp`).

| target (plan) | existing | status |
|---|---|---|
| `adc.FiniteVolume(limiter=, riemann=, variables=)` | `FiniteVolume` (`__init__.py:FiniteVolume`), remapped onto `adc.Spatial(limiter=, flux=, recon=)` | SHIPPED (#89). DECISION (point 3): the NUMERICAL flux (Rusanov/HLL/HLLC/Roe) is called `riemann=`, NOT `flux=`, so as not to collide with the PHYSICAL flux `m.flux` of the model. `limiter=` -> `Spatial.limiter` (none/minmod/vanleer/weno5), `riemann=` -> `Spatial.flux`, `variables=` (`"conservative"`/`"primitive"`) -> `Spatial.recon`. |
| `adc.IMEX(substeps=)` | `adc.IMEX(substeps=)` (`__init__.py:IMEX`) | EXISTS identically. `Explicit(substeps=, method=)` (`ssprk2`/`ssprk3`, #88) and `Implicit(dt_ratio=, substeps=)` (alias of IMEX) too. |
| `adc.DivEpsGrad` | `adc.DivEpsGrad(epsilon=)` (`__init__.py:DivEpsGrad`) | EXISTS. Elliptic operator `div(eps grad)`. eps(x) variable via `set_epsilon_field`. |
| `adc.DirichletWall.circle(radius=)` | `set_poisson(wall="circle", wall_radius=)` (`system.cpp`) | sugar GAP: no `DirichletWall` object; today it is a string argument of `set_poisson`/`add_elliptic_model`. `DirichletWall.circle(r)` would be a constructor returning `(wall="circle", wall_radius=r)`. |
| `sim.add_equation(model=, ...)` | `System.add_equation` (`__init__.py:add_equation`, #89/#90) | SHIPPED: dispatcher of section 3 (routes `ModelSpec` -> `add_block` vs `CompiledModel` -> backend adder). |
| `sim.run(...)` | `System.run(t_end, cfl, max_steps)` (`__init__.py:run`, #89) | SHIPPED: loop `while time()<t_end: step_cfl(cfl)`. `step`/`advance`/`step_cfl`/`step_adaptive` remain exposed. |
| `adc.System(n=, periodic=)` | EXISTS (`__init__.py:System`) | identical. |

All these items are pure-facade Python SUGAR: none touches the C++ numerics.
`add_equation` and `run` are the only ones carrying real logic (dispatch + loop),
the rest is renaming/remapping of arguments. Remaining sugar GAP: `adc.DirichletWall`.


## 7. Implementation sequencing

> Global STATUS. Phase A SHIPPED (#89/#90). Phase B SHIPPED (#85 `System`, #92 `AmrSystem`).
> Phase C (Python device/MPI validation) SHIPPED (#97/#99/#93). Phase D (`AmrSystem` in production)
> SHIPPED (#92/#105), single- AND multi-block, but BOUNDED (explicit/facade limits, see 0bis). Phase E
> (runtime params) is SHIPPED on the `aot` backend (P7-b; `kind="runtime"` + `set_block_params`). The
> sequencing below is kept for traceability; the per-phase tags indicate the real state.

Grouped by dependency. Each step notes its file WRITE-SET.

### Phase A: pure-Python facade, NO engine change -- SHIPPED (#89/#90)

Delivered. Edits only Python; touches neither `include/**` nor `python/system.cpp`/`bindings.cpp`.
Items 1-6 below: all SHIPPED.

1. `dsl.Model` (section 1): delegation to `HyperbolicModel`. WRITE-SET:
   `python/adc/dsl.py` (adding a class; do not modify `HyperbolicModel`).
2. `m.param` mode (a) constant (section 2a) + `gamma` case -> `set_gamma`. WRITE-SET:
   `python/adc/dsl.py`. Relies on `_wrap`/`Const` unchanged.
3. `CompiledModel` (section 3) produced by `m.compile`, packaging `so_path` +
   metadata already known. WRITE-SET: `python/adc/dsl.py`.
4. Non-dispatch runtime sugar (section 6): `adc.FiniteVolume` (remap of `Spatial`),
   `adc.DirichletWall.circle`, `sim.run`. WRITE-SET: `python/adc/__init__.py`.
5. `sim.add_equation` (section 3): dispatch `ModelSpec` -> `add_block`;
   `CompiledModel(prototype)` -> `add_dynamic_block`; `CompiledModel(aot/production)`
   -> `add_compiled_block`. WRITE-SET: `python/adc/__init__.py` (method of the `System`
   facade). Requires NO new binding (the adders exist, `bindings.cpp:78`/`:81`).
6. Error taxonomy (section 4) on the Python side: extend `check()` for the unset
   params, flux/`p` guard, backend/device guard. WRITE-SET: `python/adc/dsl.py`
   (+ messages in `add_equation`, `__init__.py`).

Phase A is testable against the `prototype` and `aot` backends (serial CPU), without GPU/MPI.

### Phase B: native `production` backend -- SHIPPED (#85 `System`, #92 `AmrSystem`)

Delivered. It is dispatch wiring (no new numerics).

7. SHIPPED. `_BACKENDS["production"]` (`dsl.py`) equals `("native", "add_native_block")`; `compile`
   routes to `compile_native` (`emit_cpp_native_loader`); `add_equation` wires onto
   `System.add_native_block`. The `AmrSystem` counterpart is also wired (#92, see Phase D).
8. SHIPPED (obsolete). Since `production` is no longer the alias of `aot`, the gap to make explicit has disappeared.
   WENO5 is no longer rejected on the `.so` (#102: ghosts allocated according to the limiter). Remains:
   `compile(backend="prototype", require_metadata=True)` raises (`dsl.py:Model.compile`), and
   `add_equation` rejects HLLC-Roe-without-`p` / `names=` on the native path (section 4).

### Phase C: Python device/MPI validation -- SHIPPED (#97/#99/#93)

9. SHIPPED. Device/MPI validation of the native `production` path FROM Python: production System GPU
   `np=1` VALIDATED GH200 (#97), `solve_fields` MPI `np=1/2/4` VALIDATED CPU/CI (#99), device-MPI
   production `geometric_mg` `np=1/2/4` VALIDATED GH200 (#93). The A==B parity (`dres=0`) device and the
   multi-rank bit-identity via `add_native_block` are acquired end to end. NO new Python write-set
   (tests). REMINDER: `fft` remains refused under System MPI `np>1` (#106), use
   `geometric_mg`.

### Phase D: `AmrSystem` in production -- SHIPPED (#92/#105), single- AND multi-block, BOUNDED

10. SHIPPED. `add_equation`/`add_native_block` on the `adc.AmrSystem` side (`__init__.py:AmrSystem`) route
    to the native AMR counterpart `add_compiled_model(AmrSystem&)` (`amr_dsl_block.hpp`);
    `m.compile(target="amr_system")` is wired (#92). WENO5 + Rusanov + conservative reconstruction
    validated on this path (#105; parity `add_native_block` == `add_compiled_model` == `add_block`,
    `dmax=0`). `AmrSystem` is now single- AND multi-block: multi-block (N blocks co-located on
    ONE shared hierarchy `AmrRuntime`, system Poisson with SUMMED right-hand side, regrid of the union of
    tags, multi-block production DSL) is SHIPPED (`amr_system.cpp` `build_multi`). LIMITS STILL
    REAL (to keep honest): `AmrSystem` is NOT at full parity with `System` (IMEX local
    source OK Gap 2 #132 but global Schur on AMR remains to be done);
    HLLC/Roe/`primitive` are REJECTED on the Python AMR facade side while the C++ engine supports them
    already (purely facade rejection). `AmrSystem.potential()`: binding SHIPPED (bindings.cpp:272,
    `#135`). WRITE-SET: `python/adc/__init__.py` (`AmrSystem` facade).

### Phase E: `m.param` runtime -- GAP (phase 2, engine change)

11. Mode (b) (section 2b): ABI with parameters + codegen with members, OR dedicated aux channel.
    HEAVY WRITE-SET: `include/adc/runtime/compiled_block_abi.hpp`,
    `python/system.cpp` (`add_compiled_block`), `python/adc/dsl.py` (codegen). Outside the
    critical path of the Hoffart reproduction (`PAPER_ROADMAP.md:147-150`, basket 2
    transverse, optional).

### Phase F: HYBRID composition native + DSL within ONE model -- prototype

12. `adc.Model(...)` composes 100% NATIVE bricks (ModelSpec, C++ tags); `dsl.Model(...)` generates
    a 100% DSL model. Phase F fills the in-between: MIXING, within a SINGLE model, NATIVE
    bricks and PARTIAL DSL bricks.

    **API**: `dsl.HyperbolicBrick` / `dsl.SourceBrick` / `dsl.EllipticBrick` -> `.compile()` ->
    `CompiledHyperbolicBrick` / `...Source...` / `...Elliptic...` (the C++ of ONE brick + metadata,
    no .so per brick). Then `adc.CompositeModel(transport=, source=, elliptic=)` accepts, per slot,
    EITHER a native brick (`adc.ExB`/`PotentialForce`/`ChargeDensity`...), OR a compiled DSL brick
    (>= 1 DSL slot required; otherwise `adc.Model`). `.compile(backend=, target=)` -> `CompiledModel` attachable
    via `add_equation`.

    **Design (option B)**: the mixing is generated at the compilation of the FINAL `adc::CompositeModel<H,S,E>`
    = ONE single .so, on the PRODUCTION path (no .so per brick nor partial virtual ABI =
    option A, set aside: host virtual dispatch, without GPU inline). The C++ core already composes
    heterogeneous slot types; `physics/bricks.hpp` is included by all backends. The PARAMETERS of a native
    brick are baked into the TYPE via a generated derived struct
    (`struct NatSrc : adc::PotentialForce { NatSrc(){ qom = adc::Real(-1.0); } };`): native numerics
    reused identically, zero re-derivation.

    **Covered (prototype)**: backends `aot` (add_compiled_block), `production` (add_native_block,
    zero-copy, MPI by construction) AND `prototype` (virtual JIT); target `system` AND `amr_system`
    (production only, loader `adc_install_native_amr`); extensible aux (B_z `n_aux=4`, T_e `n_aux=5`)
    propagated through the composite; inter-species couplings BY ROLE (collision `index_of(MomentumX/Y)`)
    on a hybrid block. Tests: `test_dsl_hybrid` (2 directions x aot/production/jit bit-identical to the native
    oracle), `test_dsl_hybrid_bz` (B_z e2e + T_e), `test_dsl_hybrid_amr` (AMR parity), `test_dsl_hybrid_coupling`
    (collision by role hybrid == native), `test_mpi_hybrid_mbox_parity` (bit-identical residual np=1/2/4,
    MPI job). WRITE-SET: `python/adc/dsl.py` (partial bricks +
    `HybridModel`), `python/adc/__init__.py` (`adc.CompositeModel`). NO C++ change (the
    `CompositeModel` already composes the mixing; the roles come from `ADC_EXPORT_BLOCK_METADATA`).

    **Remaining**: T_e end-to-end on the Python side (the marshaling is shared with B_z, already proven);
    GPU validation from Python (ROMEO PR, like the native).

### Dependency summary

- A (1-6): SHIPPED (#89/#90). The bulk of the value (`dsl.Model` stable, `CompiledModel`,
  `add_equation`, runtime sugar, errors). `compile()` ergonomics + `model_hash` cache (#103).
- B (7-8): SHIPPED (#85 `System`, #92 `AmrSystem`). Native `production` backend -> `add_native_block`;
  WENO5 now on all paths (.so #102, native AMR #105).
- C (9): SHIPPED (#97/#99/#93). Device/MPI validation FROM Python (System GPU np=1, solve_fields
  MPI np=1/2/4, device-MPI geometric_mg np=1/2/4). `fft` refused under System MPI np>1 (#106).
- D (10): SHIPPED (#92/#105), single- AND multi-block (shared hierarchy `AmrRuntime`, multi-block
  production DSL, regrid of the union of tags), but BOUNDED: explicit, IMEX local source OK (#132) but
  global Schur on AMR remains to be done, HLLC/Roe/primitive rejected on the facade (C++ engine OK).
  `AmrSystem.potential()`: SHIPPED (#135).
- E (11): SHIPPED (P7-b) on the `aot` backend: `m.param(kind="runtime")` emits `params.get(<index>)`
  (member `adc::RuntimeParams`) and `System.set_block_params` changes the value at runtime WITHOUT
  recompiling. The `prototype`/`production` backends freeze a runtime param to its declaration value.
- F (12): PROTOTYPE (branch, not merged). HYBRID composition native + DSL within a model
  (`adc.CompositeModel`); backends aot/production/jit, target system/amr_system, aux B_z/T_e, inter-species
  coupling by role, MPI parity np=1/2/4. Remaining: GPU validation. NO C++ change (option B).
