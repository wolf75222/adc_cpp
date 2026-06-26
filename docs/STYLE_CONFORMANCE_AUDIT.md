# Audit of `adc_cpp` conformance to C++ standards

Date: 2026-06-12.
Reviewed base: `origin/master` / `ffb9022`.
Scope: the entire repository, that is 283 files and ~56,700 lines, split into 14 audit
units: core/AMR/parallel/physics, mesh, numerics (3 batches), coupling (2 batches), runtime
(3 batches), Python bindings, tests (2 batches) and bench/CMake/scripts. Reviewed are all the
`include/pops/**/*.hpp`, `python/*.cpp`, `tests/**/*.cpp`, `python/tests/**/*.cpp`, `bench/*.cpp`,
`CMakeLists.txt`, `CMakePresets.json`, `.clang-format`, `.clang-tidy` and the build scripts.

Method: review by subsystem, one unit at a time. Each non-cosmetic finding was
verified on the spot (reading the cited `file:line`, the twin site and the
de facto convention of the repository before classification). Findings that turned out to be unfounded or
oversold on verification were either removed, or downgraded in severity: 6 findings were
rejected outright (cf. closing note at the end of section 6), and about twenty initial "important" ones were brought back to
"cosmetic" for lack of a trigger or real impact. The document reports only the deviations INTERNAL to the
repository conventions or the substantive rules actually broken; the guide-rules
deliberately not followed (snake_case, `#pragma once`, no `[[nodiscard]]`...) are listed once
in section 3 and are NOT counted as deviations.

This document is a style and conformance audit, not a scientific roadmap nor a maintainability
audit. It complements:
- [`CODEBASE_AUDIT.md`](CODEBASE_AUDIT.md): maintainability and responsibilities audit.
- `CODE_DOCUMENTATION_CONVENTION.md`: comments/Doxygen convention. WARNING: this file
  is referenced by `CODEBASE_AUDIT.md` but has never been committed: it exists only in the main
  working tree, so the link is dead for any fresh clone of `master`. To be committed
  (tracked via ADC-125).
- `CODING_STANDARDS_DECISIONS.md`: register of the arbitrations between Google / Core Guidelines / LLVM
  for the target profile. This is where the choices of section 3 must be recorded; this audit
  assumes its content and designates it as the source of truth of the conventions.

## 1. Global reading

The repository is of very good quality and, above all, COHERENT with itself. It does not follow a single
external guide: it applies a stable in-house convention (snake_case, header-only `#pragma once`,
device-copyable POD structs, `POPS_HD` idiom, named functors rather than extended lambdas under
`nvcc`, prefixed `std::runtime_error` exceptions, always explicit casts). The global conformance
verdict is therefore not "follows Google" or "follows Core Guidelines", but: the repository follows
mostly and faithfully a coherent internal convention, and the deviations found are
deviations from THIS convention, plus a few substantive rules (security/UB/portability) that the three
guides share.

Only two findings are blocking, both conditional on the Windows port work item
(epic ADC-90) or on the device path:
- `include/pops/mesh/box_hash.hpp:71-73`: `static_cast<long>(bx) << 32` assumes `long >= 64` bits;
  on the LLP64 ABI (native Windows) `long` is 32 bits, the shift is undefined behavior and the
  key loses `bx` -> spatial hash broken. Verified on the spot.
- `include/pops/numerics/elliptic/elliptic_problem.hpp:104`: `field_postprocess` dispatches via an
  extended lambda `[=] POPS_HD(int i, int j)` on a device path, exactly the pattern that the rest
  of the elliptic directory bans (`nvcc` cross-TU segfault, doctrine #93). Verified: it is the only
  POPS_HD-lambda in all of `elliptic/`.

No other certain UB nor hard bug was found. The other risks (uninitialized POD members,
member pointers/references that alias stores without the rule of five, `assert` guards lost
under NDEBUG) are LATENT: neutralized by current usage but to be hardened.

## 2. Quantified summary

180 findings retained after verification (6 rejected in addition, cf. closing note at the end of section 6).
The table below is authoritative for the count; sections 4.x enumerate the representative findings
per subsystem and group the repetitive deviations, so that the literal count
of bullets there drifts by one to two items per section.

| Subsystem | Files | Lines | Blocking | Important | Cosmetic | Total |
|---|---|---|---|---|---|---|
| core/AMR/parallel/physics | 23 | 2861 | 0 | 2 | 13 | 15 |
| mesh | 13 | 1912 | 1 | 5 | 7 | 13 |
| numerics-1 (elliptic) | 12 | 3252 | 1 | 5 | 14 | 20 |
| numerics-2 (flux/operators) | 10 | 3119 | 0 | 2 | 12 | 14 |
| numerics-3 (time/AMR) | 13 | 2198 | 0 | 1 | 14 | 15 |
| coupling-1 | 10 | 2499 | 0 | 0 | 14 | 14 |
| coupling-2 | 7 | 1850 | 0 | 3 | 4 | 7 |
| runtime-1 (DSL/AMR system) | 5 | 3163 | 0 | 1 | 10 | 11 |
| runtime-2 (loader/ABI) | 13 | 2855 | 0 | 1 | 15 | 16 |
| runtime-3 (stepper/store) | 4 | 1380 | 0 | 1 | 7 | 8 |
| Python bindings | 3 | 3654 | 0 | 4 | 14 | 18 |
| tests-a | 72 | 12888 | 0 | 1 | 7 | 8 |
| tests-b | 86 | 13293 | 0 | 2 | 5 | 7 |
| bench/CMake/scripts | 12 | 1734 | 0 | 2 | 12 | 14 |
| TOTAL | 283 | 56658 | 2 | 30 | 148 | 180 |

Breakdown by dimension (important and blocking): organization/DRY dominates (duplication of
dispatch cascades, of stencils, of pack/unpack, of test harnesses), ahead of security (latent UB,
`assert` guards not hardened, unguarded modulo by zero), types (incomplete rule of five, uninitialized
members) and device idioms (residual extended lambdas).

Global verdict: conforms to the internal convention, with a structural duplication debt
concentrated on the AMR/Schur/DSL paths and one real portability point (`long` 32 bits on
LLP64) that becomes blocking as soon as the Windows port advances.

## 3. Framing: why the three guides are not superposable

The target is a header-only C++23 library, HPC, Kokkos/CUDA (`nvcc`, device path) with
pybind11 bindings. On this profile, Google C++ Style, C++ Core Guidelines and LLVM Coding Standards
diverge on mutually exclusive axes: one cannot satisfy all three at once.

- Function case: `UpperCamelCase` (Google) vs `camelBack` (LLVM) vs `snake_case` (CG):
  three incompatible styles. The repository settles on `snake_case` (CG).
- Variable/member and constant case: `snake_case` + `kCamelCase` (Google/CG) vs
  `UpperCamelCase` (LLVM). The repository follows `snake_case` for locals, `_` suffix for private
  members, `kCamelCase` for magic numbers.
- Include guard: `#ifndef` imposed (Google/LLVM) vs `#pragma once` tolerated (CG SF.8). In header-only
  the repository takes `#pragma once`, ergonomic.
- Include order: std early (Google) vs system last (LLVM), opposite orders. The repository
  imposes its own order: `<pops/...>` block first, blank line, then STL (with `SortIncludes:false`).
- Exceptions and RTTI: forbidden (Google/LLVM) vs recommended E.2/E.3 and `dynamic_cast` C.146 (CG).
  `__device__` code can neither `throw` nor `typeid` -> the no-except/no-RTTI position is imposed
  de facto on the device path; exceptions live on the host side only.
- Contracts: `assert`/`CHECK` (Google/LLVM, device with reservations) vs `Expects()/Ensures()` GSL (CG,
  not device-callable). The repository uses host `assert` + modeling `static_assert` by concepts.
- `[[nodiscard]]`: under-specified by the three. The repository DISABLES it deliberately
  (`modernize-use-nodiscard` removed in `.clang-tidy`).

Methodological consequence: these choices are coherent throughout the repository and stem from an
arbitration, not a defect. They are NOT counted as deviations. The arbitrations themselves must
be recorded in `CODING_STANDARDS_DECISIONS.md`; the present audit judges only conformance to the
internal convention thus fixed, and the substantive rules common to the three guides (object init,
no division by zero, rule of five, no shift UB).

Guide-rules deliberately not followed, valid throughout the repository (single reminder, NOT reported
as findings): `snake_case` (vs Google CamelCase); `struct` with public members for the
POD/policies/functors; `#pragma once`; absence of `[[nodiscard]]`; comments in French without
accents; idiomatic `Real(literal)`; C arrays `Real v[N]` (device-clean); out-params by
reference for multiple device returns (F.21); model capture BY VALUE in the functors;
`using namespace pops;` at file scope in the test/bench/bindings TUs (leaves, not a header).

## 4. Findings by subsystem

### 4.1 core/AMR/parallel/physics (23 files)

Very good quality: concept-driven architecture (`PhysicalModel`/`HyperbolicPhysicalModel`/
`EquationBlockLike`), rigorous device-clean invariant, solid Berger-Rigoutsos AMR, dense
documentation. No hard bug; the two real risks are latent.

Important:
- `include/pops/core/variables.hpp:32-44` (C.48/ES.20): `struct Variable` leaves `role`/`component`
  and `struct VariableSet` leaves `kind`/`size` without an in-class initializer, whereas `ArenaStats`/
  `ClusterParams`/`Aux` set them. Public default-constructible type -> read = UB. Trivial fix
  without breaking the aggregate `{kind, names, size}`.
- `include/pops/physics/hyperbolic.hpp:81-114` (DRY): `ExBVelocityPolar` duplicates byte for byte
  `ExBVelocity` (l.27-59); only the semantics (Cartesian vs polar local basis) differ, carried
  by the comments. The same file resolves the twin case by inheritance (`IsothermalFluxPolar :
  IsothermalFlux`). Alias or factor out.

Cosmetic (13): `equation_block.hpp:63` (`string_view` member potentially dangling, latent);
pervasive use of `long` for cell counts -> 32 bits on LLP64; `regrid_level` tags
only the local fabs without the documented `all_reduce_or` (correct in serial); `cluster_rec`
nested 3-4 levels; `double(x)` conversions; capitalized locals `Sx`/`Sy`/`D`; non-uniform
placement of `@file`/`#pragma once`; `comm.hpp`/`load_balance.hpp` without a Doxygen header; double
Doxygen+prose documentation; `/** */` island in `physics/`; obsolete "16 bits" comment in
`part1by1`; `ManagedArena::stats()` non-const; `VariableSet::size` denormalized vs `names.size()`.

### 4.2 mesh (13 files)

Clean and coherent: unanimous `#pragma once`, device-copyable POD structs, rule-of-0 respected,
correct const-correctness and `explicit`.

Blocking:
- `include/pops/mesh/box_hash.hpp:71-73` (Integer Types/ES.101): `BoxHash::key()` does
  `static_cast<long>(bx) << 32` and stores into `unordered_map<long, ...>`. On LLP64 the `<< 32` is
  a shift >= width of the type = UB, and `bx` is lost -> massive collisions, broken hash. Move
  to `std::int64_t`.

Important:
- `fab2d.hpp:45,95,123-127` + `box2d.hpp:54` + `box_array.hpp:52`: native `long` for
  memory strides/sizes/offsets; silent overflow > 2^31 on LLP64, compounds the blocker.
- `refinement.hpp:44-47` / `box2d.hpp:111-114` / `box_hash.hpp:68`: integer floor division
  (negative handling) is WRITTEN THREE TIMES, identical bodies. Factor out a `floor_div` POPS_HD.
- `refinement.hpp:179,209` + `multifab.hpp:137`: extended lambdas `[=] POPS_HD` on the hot
  V-cycle MG path, whereas `fill_boundary`/`mf_arith`/`physical_bc` have migrated to named functors
  (same documented `nvcc` cross-TU failure mode).
- `patch_box.hpp:16-20` (C.48/C.49): 5 members `level/ilo/jlo/ihi/jhi` without an initializer whereas
  all the sibling PODs (`Box2D`, `Array4`, `Geometry`, `BCRec`) initialize; type exported to Python.
- `refinement.hpp:121-157`: `parallel_copy` packs/unpacks the MPI messages in host loops
  4-nested duplicated send/receive with `std::vector` `std::allocator`, whereas
  `fill_boundary` does the same pack/unpack in `for_each` device + pinned `comm_allocator`.

Cosmetic (7): inverted include order + unused `<cstdlib>/<cstdio>` in `refinement.hpp`;
`fill_boundary_begin`/`parallel_copy` too long; POD methods `Box2D`/`Geometry` not `constexpr`
(174 `constexpr` elsewhere); camelCase locals; double Doxygen+prose headers; `patch_box.hpp`
without `@file`; divisions by `r` not guarded in `coarsen`/`refine`.

### 4.3 numerics-1 elliptic (12 files)

Healthy and homogeneous: C++23, header-only, RAII, concept-driven design, no C cast, guarded
divisions, real device care (named functors, stack buffers O(N^2)).

Blocking:
- `elliptic_problem.hpp:104`: only kernel of the unit dispatching via an extended lambda `[=] POPS_HD`
  on a device path (phi -> aux grad consumed by the coupler), pattern banned by
  `poisson_operator.hpp:127-132`. Extract a functor `detail::FieldPostprocessKernel`.

Important:
- `poisson_fft_solver.hpp:114`: guards `Ny % n_ranks() == 0` by `assert` (lost under NDEBUG) whereas
  `nyl_` is already truncated in the init list -> mis-sized slabs, OOB in Release. The
  twin `PoissonFFTSolver` has precisely migrated to `throw`. Harden to `throw`.
- `poisson_operator.hpp:150-166 / 203-219 / 332-349`: the harmonic face permittivity block
  (+ cut-cell branch) copied identically into the three functors. Factor out `face_weights(...)`.
- `polar_poisson_solver.hpp:260-316`: `residual()` rebuilds by hand the operator already assembled
  by `solve()` (radial coeffs, azimuthal eigenvalue, BC fallback, FFT per line). Risk of
  solve/residual drift.
- `geometric_mg.hpp:189-360`: 6 `set_*` overloads repeating host-init + restriction + ghosts; the
  `kappa` fallback omits `fill_ghosts` (risk of silent bug). A private template `sample_per_level`.
- `composite_fac_poisson.hpp:377-480`: `composite_coarse_residual()` ~103 lines, base residual + 4
  quasi-symmetric C-F blocks (4-5 level nesting) + inf norm. Split by direction.

Cosmetic (14): implicit converting ctor `DistributedFFTSolver`; `TensorKrylovSolver` has two
member-references (lifetime); `hqr_minmax()` ~133 lines (verbatim EISPACK port); 8 files
without `@file`; inverted include order `dense_eig.hpp`; UPPERCASE matrix locals; `kPi_` literal
vs `std::numbers::pi`; `std::function` by value without `move`; accessors `op_*()` non-const;
`x,y,z,w,s` uninitialized; fragile positional init of `MGLevel`; FFT helpers in `pops::` instead
of `detail::`; magic numeric thresholds; heterogeneous exception prefixes.

### 4.4 numerics-2 flux/operators (10 files)

Mature and very documented, strong device discipline (named functors, `if constexpr` guards,
protected divisions). No outright UB in the kernels.

Important:
- `amr_flux_helpers.hpp:85-90`: `mf_apply_source` (Model-template, default Euler path) launches a
  kernel via an extended lambda `[=] POPS_HD` capturing the model and calling `m.source(...)`, whereas
  `AmrSspRhsKernel` just above and the whole unit impose the named functor.
- `polar_tensor_operator.hpp:726-727`: `PolarTensorKrylovSolver` carries pointers `a_rr_`/`a_tt_`
  that alias its own stores `a_rr_store_`/`a_tt_store_`, without copy/move/dtor (rule of five);
  `MultiFab`/`Fab2D` being copyable AND movable, a copy/move makes the pointers dangle (UB).
  `= delete` the copy/move or re-point in a hand-written move.

Cosmetic (12): `std::numeric_limits` without `#include <limits>` (compiles by transitivity);
`assert(a_rr && a_tt)` instead of `throw` (nullptr deref in NDEBUG); stencil helpers with 9-11
interchangeable homogeneous parameters; `else` under-indentation (auto-fixable); 2 files without
`@file`; `@file` after includes; documentation duplicated 3x; `__is_trivially_copyable`
(reserved identifier); out-params `Real&` (device idiom); math-notation locals;
structural duplication of the 4 RHS assemblers.

### 4.5 numerics-3 time/AMR (13 files)

Two families: short and clean foundation files (steppers, splitting, imex, ssprk); AMR engines and
implicit Newton, correct but heavy. Security generally good (`static_cast<size_t>` indexing,
bounded cell encoding).

Important:
- `implicit_stepper.hpp:234-257 / 297-320` (F.3/DRY): the assembly of the Newton jacobian
  (`if constexpr (HasSourceJacobian) {...} else {fd...}`) duplicates ~24 lines word for word between the
  path 2a and the path 2b, supposed to be bit-identical -> risk of silent divergence.
  Extract `assemble_newton_jacobian(...)`.

Cosmetic (14): `fab(bL/bR/bB/bT)` not guarded on `mf_find_box==-1` (protected by the
parent-replica invariant); encoding constants `1048576`/`16` duplicated encode/decode; `fail_policy` in
`int`+`kFail*` vs `enum class`; `subcycle_level_mp` ~205 lines; local `struct Reg` == `RegMP`;
4-face pattern copied ~8 times (including deliberate oracles); 13 files without `@file`; raw `double`
instead of `Real` in the oracles; `TimeTreatment::Explicit` bare vs `k` prefix;
`else` under-indentation; non-owning raw pointers `aux`/`pOld`/`pNew`; callables by value vs
`Step&&`; reuse of ambiguous `r`/`L`/`m`; `char msg[256]` snprintf instead of `std::string`.

### 4.6 coupling-1 (10 files)

Solid and coherent: `#pragma once`, include order respected, `POPS_HD` + named functors,
concepts/`requires`, `enum class`, exceptions, explicit casts, sink by value+`move`. No blocking
deviation, no important one after verification (the 6 important candidates were downgraded).

Cosmetic (14): triple duplication of the mapping `BCRec->Foextrap` (`coeff_bc` x2 +
`detail::derive_aux_bc`) whereas `aux_fill.hpp` already centralizes it; `shared_ptr` where `unique_ptr`
suffices (inconsistent with the sister class); `CsProgram::eval` pops without a stack guard (protected
only on the Python side); `n*n` validation in overflowable `int` (the sister avoids it in `size_t`);
`max_wave_speed`/`level_state`/`level_potential` non-const; `AmrCouplerMP` 626 lines
orchestration+marshaling; `step_multilevel` dense; SCREAMING locals `PNX`/`PNY`; double Doxygen+prose
headers; heterogeneous exception prefixes; `same_box` reimplements `operator==`; C array
`MultiFab* saved[]`; `(void)dom` vs `[[maybe_unused]]`; member `bcPhi_` camelCase.

### 4.7 coupling-2 (7 files)

Carefully written code, device-clean, no blocking finding.

Important:
- `system_coupler.hpp:61-70` (C.21): `ScopedBlockState` scope-guard has an active mutating dtor without
  `= delete` copy/move -> possible double restoration if copied.
- `system_coupler.hpp:244,292` (ES.45): hardcoded `Real(1e-30)` whereas `kCflSpeedFloor` exists
  (`core/types.hpp:49`) and is documented as the replacement of the "scattered 1e-30".
- `system_coupler.hpp:143-147` (DRY): `SystemAssembler::derive_aux` re-encodes the convention
  `FieldPostProcess{Plus,true}` inline instead of the helper `detail::coupler_grad_phi` already used by
  `Coupler::derive_aux` -> risk of convention drift.

Cosmetic (4): polar `step()` ~118 lines; precondition `Ny/np_` not checked (DEPRECATED class,
dead code); UPPERCASE locals + useless `cmath` + `@file` before `#pragma once`; double
header + inconsistent callback forwarding + polar phases mis-numbered.

### 4.8 runtime-1 DSL/AMR system (5 files)

Good standing: `@file`/`@brief` everywhere, `#pragma once`, modern C++ (concepts, `if constexpr`,
forward-declared PIMPL), ABI safety (`abi_key.hpp`) and device-clean exemplary.

Important:
- `amr_dsl_block.hpp:596-703 / 707-778` + `block_builder.hpp:443-528` (DRY): the dispatch cascade
  riemann x limiter (rusanov/hll/hllc/roe x none/minmod/vanleer/weno5) with its `if constexpr` guards
  is replicated THREE times. The comments (l.665-667, 750-752) record a table divergence
  already occurred (hllc AMR without weno5). Extract a dispatch template parameterized by the
  terminal action.

Cosmetic (10): `add_block` ~18 params including 6 flat Newton scalars whereas `NewtonOptions`
exists (the internal functions already use it); UPPERCASE locals `I0/I1/J0/J1`/`PNX/PNY`;
`SourceNewtonReport` 8 uninitialized members; `abi_key` helper macros not `#undef`; code lines
>100 col not reformatted (auto-fixable); `build_amr_compiled` ~250 lines; non-const refs to
the internal state; class-qualified exception prefix vs `adc (...)`; superfluous `mutable` on
`solve_count_` (auto-fixable); copy not `= delete`-d explicitly on move-only `AmrSystem`.

### 4.9 runtime-2 loader/ABI (13 files)

Healthy and very documented, no outright bug/UB, prudent host/device marshaling (`static_cast`
systematic, guards `local_size()==0`, `device_fence()` before host read). `dynlib.hpp`/
`export.hpp`/`runtime_params.hpp`/`model_spec.hpp` are clean layers.

Important:
- `system.hpp:122-134` (I.23): `add_block` ~19 params (8 `newton_*`), `set_source_stage` ~11,
  `add_coupled_source` ~12; on the ABI side `residual` 13 and `advance` 15. Parameters of the same
  adjacent types = ordering footgun. Group into POD (`NewtonOptions`, `SourceStageOptions`) like
  `ModelSpec` already present.

Cosmetic (15): functions extracted VERBATIM from `system.cpp` without re-splitting
(`add_compiled_block` ~222 l.); magic `3` vs `kAuxBaseComps`; `IModel<NV>` polymorphic base without
copy suppression (abstract, null practical risk); message prefixes `System::` vs bare;
include `adc` interleaved in the STL block; lines >100 col (auto-fixable); sinks by value without
`move`; dead variable `nn` (auto-fixable); SFINAE `void_t` vs `requires`; single-letter locals +
hand-written max; integers stored in `double` in `SourceNewtonReport`; MUSCL recon in magic
`int` 0/1/2; `reinterpret_cast` dlsym (~20 sites, to confine); const-correctness `potential()`.

### 4.10 runtime-3 stepper/store (4 files)

Healthy and of good quality: mastered modern C++, systematic MPI-safe iteration on the local
fabs, `@file`/`@brief` everywhere, `#pragma once`, coherent error messages.

Important:
- `system_stepper.hpp:241-517` (DRY): the 4 macro-steps (`step`/`step_strang`/`step_cfl`/
  `step_adaptive`) replicate verbatim the block-advance loop, the computation of the physical step `h`, the
  loop over the global bounds `dt_bounds_` (which carries the MPI semantics `all_reduce_min` whose
  desync = deadlock) and the `n_b` computation. Extract `cfl_grid_h()`, `apply_global_dt_bounds()`,
  `for_each_due_block()`.

Cosmetic (7): `BlockState` members `ncomp/substeps/evolve/gamma` uninitialized (hypothetical UB,
all sites construct by aggregate); sentinel `1e30` instead of `numeric_limits`; `reg[]`
not zero-init (auto-fixable); `poisson_bc`/`wall_active` non-const; `find()` overloads duplicated;
public data exposed via `class` + raw back-pointer (deliberate, documented); STL block split
in two.

### 4.11 Python bindings (3 files)

Healthy and very defensive (size guards before memcpy, prefixed errors, device lambdas capturing
by value, sink `std::function`/`BlockClosures` by value+`move`, no C cast, `reinterpret_cast`
confined to the dlopen). No certain blocking bug.

Important:
- `amr_system.cpp:849-852` (ES.105): `set_conservative_state` computes `nn = cfg.n * cfg.n` then
  `U.size() % nn`; `cfg.n==0` (settable from Python, ctor does not validate) -> modulo by zero = UB
  reachable. Validate `cfg.n >= 1` at construction.
- `system.cpp:57-93` + `amr_system.cpp:29-61` (DRY): `resolve_implicit_components` copies almost
  identically between the two TUs (copy acknowledged in the comment). Extract into a shared header.
- `system.cpp` (~12 sites, l.392/407/427/905/1038/1492/1716/1744/1759/1783/1823...): the marshaling
  loop `out[(c*gny+j)*gnx+i]` is repeated ~12 times almost identically. Helpers `gather_fab`/
  `scatter_fab`.
- `system.cpp:502-558` + `amr_system.cpp:498-518` (DRY): Newton options validation duplicated, +
  triple Newton report (lambda `py::dict` x2 in `bindings.cpp`, conversion `SourceNewtonReport`
  x2). A shared `parse_newton_options(...)` + a single factory of the `py::dict`.

Cosmetic (14): `using namespace pops;` at file scope in `bindings.cpp`; very long functions
(`add_block`, `add_coupled_source`, `build_multi`, `add_native_block`); `PYBIND11_MODULE`
monolithic ~570 lines; under-indented `else` (auto-fixable); include `adc` interleaved in the STL;
exact float equalities of defect detection; `static` helpers vs `namespace {}`; `_` suffix
of pimpl members inconsistent; `size_t->int` narrowing; `time_method` in `int` vs `enum class`;
non-const accessors; `(void)ncomp` vs `[[maybe_unused]]`; `reinterpret_cast` dlsym (to confine);
misleading comment on the inclusion order.

### 4.12 tests-a (72 files)

High and homogeneous quality: each test autonomous (`int main` 72/72), device-clean, without new/delete,
without C cast, clean negative tests (catch type). No UB triggered, clean printf scan.

Important:
- `test_condensed_schur_source_stepper.cpp:490,530` (ES.50/Type.3): `const_cast<Setup&>(S)` removes
  the const from a genuinely `const Setup S` object to pass it to a member `Setup& S` of
  `RefIntegrator` that only reads. No UB today, fragile. Declare `const Setup& S`.

Cosmetic (7): test harness triplicated (lambda `chk`+`fails` redeclared, `raises`/`close_rel`/
`checksum` copied, no `test_support.hpp` header); `reinterpret_cast` void*->T* instead of
`static_cast` in 4 native loaders (auto-fixable); `std::system()` to compile a `.so` (4
files, build-controlled inputs); `main()` 180-290 lines; `double()` casts (28, auto-fixable);
ALL_CAPS constants `NC`/`KAPPA`; mutable global counter `static int fails`.

### 4.13 tests-b (86 files)

Healthy and aware of the device constraints (`POPS_HD` helpers, named functors). Coherent includes,
`comm_init` always paired with `comm_finalize`, `int` size accessors (no format bug).

Important:
- `test_geometry.cpp:12` (+ ~118 files, DRY): no test framework; each `.cpp`
  reimplements the same harness (lambda `chk` x50+, `fails` counter, inline variant). A change
  of report format touches 86 files. Extract a `tests/test_harness.hpp`.
- `test_polar_condensed_schur_source_stepper.cpp:449,485` (ES.50/Type.3): `const_cast<Setup&>` on
  a `const Setup S` only read (the polar solvers already take `const&`). Eliminable.

Cosmetic (5): manual Kokkos teardown `initialize/finalize` not exception-safe and inconsistent with
`ScopeGuard`; C-style/function casts `(double)x`/`double(x)` (~50, auto-fixable); constant `kPi`
re-duplicated (16 declarations + 28 literals); polar `main()` 150-200 lines; `std::exit(2)` in
a `load()` helper with a `FILE*` leak.

### 4.14 bench/CMake/scripts (12 files)

Carefully written: clear structure, modern root `CMakeLists` (isolated INTERFACE target `adc`, FetchContent
Kokkos with SHA256 check, `pops_dev_options` in PRIVATE), `set -euo pipefail` scripts + quoting. Tools
outside CI executed, not the API.

Important:
- `frontend_cpp.cpp:251` + bench sites (DRY): `percentile`/`timed`/`PhaseTimers`/`eat` byte-identical
  on 4-5 files, without `bench/common.hpp`. A fix divergence (interpolation, fence semantics)
  breaks the consistency of the measurements.
- `profile_step.cpp:188-228` + `frontend_cpp.cpp` + `scaling_step.cpp` (device): 3 benches write
  the `Array4` by raw HOST loops without `sync_host`/`sync_device`, whereas
  `profile_transport_mbox.cpp` shows the right idiom (`for_each_cell`+`POPS_HD`+sync). Portable
  only by host-accessible memory (UVM); would break on a non-UVM device backend, and `bench/`
  being only compile-tested the deviation would not be detected.

Cosmetic (12): owning `new`/`delete` `fft_storage`; `atoi`/`atof` without error reporting;
signatures 11-14 params with out-params `double&`; unknown argument silently ignored;
`volatile Real s` as an anti-elision barrier (CP.200); `double()` casts (auto-fixable); `using
namespace pops;` global; `namespace {}` vs `static` not uniform; C++ standard set twice in
`CMakeLists.txt`; outdated "target 3.20" comment (3.21); dead `kPi` + `wall` not emitted; `printf`
~34 args outside `-Wformat=2` coverage.

## 5. Auto-fixable vs judgment

Auto-fixable (tooling, to be routed to the "Code quality & hardened CI" milestone). The repository
already has `.clang-format` (`BasedOnStyle: Google`, `ReflowComments:false`, `SortIncludes:false`) and
`.clang-tidy` (broad families, `modernize-use-nodiscard`/`magic-numbers`/`identifier-length`
disabled), both INFORMATIVE (the `format`/`tidy` job only signals):
- `clang-format` would fix: `else` under-indentation of `if constexpr`
  (`numerical_flux.hpp:207`, `implicit_stepper.hpp:243/306`, `system.cpp:605`), CODE lines >100
  col (`amr_dsl_block.hpp:615+`, `block_builder_polar.hpp:299`, `compiled_block_abi`).
- `clang-tidy google-readability-casting`: all the `double(x)`/`(double)x`/`(int)b` (tests-a ~28,
  tests-b ~50, bench ~20, core, python). Distinguish from `Real(literal)`, idiom kept.
- Trivial edits: `/** */` -> `///` (`physics/`), missing in-class initializers
  (`patch_box.hpp`, `Variable`/`VariableSet`, `BlockState`, `SourceNewtonReport`), `reg[] = {}`,
  removal of `mutable solve_count_`, dead variable `nn`, `reinterpret_cast` void*->T* -> `static_cast`
  in the test loaders.

These cosmetic corrections could become blocking in CI once the convention is recorded in
`CODING_STANDARDS_DECISIONS.md` (move `format`/some `tidy` families from informative to `WarningsAsErrors`).

Judgment / refactor (not to be entrusted to a tool): the blockers (`box_hash` int64, `field_postprocess`
functor), all the structural DRY (dispatch cascades x3, stencils x3, `set_*` x6, marshaling x12,
Newton jacobian x2, macro-steps x4, test harness, `bench/common.hpp`), the rule of five
(`PolarTensorKrylovSolver`, `ScopedBlockState`), the guards to harden (`assert`->`throw`,
`cfg.n>=1`, `CsProgram::eval` stack guard), the `long`->`int64_t` move for the Windows port, and
the parameter groupings into POD.

## 6. Follow-ups (prioritized)

Blockers first, then high-leverage structural debt, then tooled cosmetic. These
follow-ups are tracked in the *Code review & quality audit* milestone: ADC-209 (1), ADC-210 (2 and 3),
ADC-211 (4 and 6), ADC-212 (5), ADC-213 (7), ADC-214 (8), ADC-215 (9), ADC-216 (10), ADC-217 (11),
ADC-219 (12); the acceptance criterion #5 (blocking deviations fixed or tracked) is thus closed.

1. `box_hash.hpp:71-73`: move the key and `bins_` to `std::int64_t`, cast `bx` before the shift.
   Gate of the native Windows port. To be tracked in an issue (linked to epic ADC-90).
2. `elliptic_problem.hpp:104`: extract `FieldPostprocessKernel` (device-clean named functor).
   `nvcc` segfault risk in Release. To be tracked in an issue.
3. `amr_flux_helpers.hpp:85` and `refinement.hpp:179/209` + `multifab.hpp:137`: convert the residual
   extended lambdas to named functors (same risk class as 2). To be tracked in an issue.
4. `amr_system.cpp:852` (`cfg.n==0` modulo by zero): validate `n>=1` at the ctor of `System`/`AmrSystem`.
5. Rule of five: `= delete` or implement copy/move of `PolarTensorKrylovSolver`
   (`polar_tensor_operator.hpp:726`) and `ScopedBlockState` (`system_coupler.hpp:61`).
6. Harden the guards: `Ny%np` `assert`->`throw` (`poisson_fft_solver.hpp:114`), stack guard
   `CsProgram::eval` (debug mode), `mf_find_box==-1` (`amr_subcycling.hpp:523`).
7. High-leverage DRY (each to be tracked in an issue): dispatch cascade x3 (runtime-1), stencil
   eps/cut-cell x3 and `set_*` x6 (numerics-1), Newton jacobian x2 (numerics-3), 4 macro-steps of the
   stepper (runtime-3), marshaling x12 + Newton + `resolve_implicit_components` (bindings),
   `floor_div` x3 + `parallel_copy` (mesh).
8. Group the long parameter lists into POD (`NewtonOptions`/`SourceStageOptions`):
   `system.hpp:122`, `amr_system.hpp:232`, ABI `residual`/`advance`.
9. Shared test harness `tests/test_harness.hpp` (`chk`/`raises`/`close_rel`/`kPi`) and
   `bench/common.hpp` (`timed`/`percentile`/`PhaseTimers`/`eat`). To be tracked in an issue.
10. Portability `long`->`std::int64_t`/`std::size_t`/`std::ptrdiff_t` for memory sizes/offsets
    (mesh, core). Large scope, to be tracked in an issue in the Windows epic.
11. Missing Doxygen headers (`comm.hpp`, `load_balance.hpp`, `patch_box.hpp`, 13 numerics-3 files,
    8 numerics-1, 2 numerics-2) and `CODE_DOCUMENTATION_CONVENTION.md` never
    committed, to be committed to repair the link from `CODEBASE_AUDIT.md` (tracked via ADC-125).
12. Single tooled pass (section 5) once the convention is recorded, then harden the `quality.yml` job.

Methodological honesty note: 6 findings were rejected outright on verification (one per
unit: core, coupling-1, coupling-2, runtime-1, runtime-3, bench), and about twenty
"important" candidates were downgraded for lack of a real trigger or impact (notably `string_view` member
without a triggering site, `long` usage bounded by the box, `regrid_level` serial-only documented,
`cluster_rec`/`hqr_minmax` cohesive and ported). The count of 180 findings reflects this state
after cross-verification, not the initial raw list.
