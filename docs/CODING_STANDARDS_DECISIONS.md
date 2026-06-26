# Decision note: coding conventions of `adc_cpp`

Date: 2026-06-12.
Reviewed base: `origin/master` / `ffb9022`.
Scope: C++ coding conventions of the core (`include/pops/**/*.hpp`, 110 files, 25,089 lines)
and of the Python binding layer (`python/{bindings,system,amr_system}.cpp`). Out of scope: pure
Python code (`python/pops/*.py`) and `adc_cases`, which have their own conventions.

Method: review by subsystem (naming, headers, comments, error handling, formatting,
idioms), confronted with three public references (Google C++ Style Guide, C++ Core Guidelines
hereafter CG, LLVM Coding Standards). For each axis where the three guides diverge, or where the practice
of the repository diverges from the three, a single rule is set. Every non-cosmetic finding is verified on
the spot (`file:line` cited). Read-only, no source file modified.

Related docs: [`CODEBASE_AUDIT.md`](CODEBASE_AUDIT.md) (maintainability audit, June 6, 2026),
[`QUALITY_TOOLING.md`](QUALITY_TOOLING.md) (static analysis, milestone *Code quality & hardened
CI*, epic ADC-105). The `CODEBASE_AUDIT.md` refers to
`CODE_DOCUMENTATION_CONVENTION.md`, **present in the working tree but not committed**
(therefore outside version control, to be committed; see D9).

Guiding principle: consistency with the existing code prevails. The base is healthy; we do not impose a
massive rename. For each axis we keep the rule closest to the dominant practice, except
if that practice poses a real problem (security, readability, tooling). The choices imposed by the
device constraint (`nvcc`, `__device__` code, header-only Kokkos) are flagged: latitude there is
zero, not just low.

Status: decisions D1-D15 are now **adopted** (ratified on 2026-06-15 by the maintainer). They
apply as the coding conventions of `adc_cpp` and remain revisable should practice or tooling warrant
it.

## D1 Naming

Divergence: this is the most fragmented axis. On the case of functions the three guides are
mutually exclusive (Google `UpperCamelCase`, LLVM `camelBack`, CG `snake_case` NL.10); on the
variables LLVM alone imposes `UpperCamelCase`; on the constants Google imposes the `k` prefix,
LLVM `UpperCamelCase`, CG nothing. Macros are the only point of convergence (NL.9, `ALL_CAPS`).

Repository practice and proposed rule:

| Element | Practice (counts, `file:line`) | Proposed rule | Closest guide |
|---|---|---|---|
| Types, classes, enums | `PascalCase`; 219 `struct` / 41 `class` (`mesh/box2d.hpp:35`) | `PascalCase` | Google, LLVM |
| Type aliases | 71 `PascalCase` / 22 STL-like | `PascalCase`, except STL imitation (`value_type`) in `snake_case` | Google |
| Functions, methods | `snake_case` (`physics/euler.hpp:49`, `:54`) | `snake_case` | CG (NL.10) |
| Local variables | short `snake_case`, often `const` | `snake_case` | Google, CG |
| Non-public members | `_` suffix; 506 occ. (`mesh/multifab.hpp:49-53`) | `_` suffix required | Google |
| Public POD members | without suffix (`Box2D::lo/hi`, `Euler::gamma` `physics/euler.hpp:40`) | without suffix | all three |
| Literal constants | `kCamelCase` (`kTwoPi` `mesh/geometry.hpp:72`, `kMaxRuntimeParams` `runtime/runtime_params.hpp:34`) | `kCamelCase` | Google |
| Concept-trait constants | `snake_case` (`n_vars` `physics/euler.hpp:38`) | `snake_case`, name imposed by `requires` | constraint |
| Macros | `POPS_` + `SCREAMING_SNAKE`; `POPS_HD` 338x, `POPS_EXPORT` 24x (excluding its definition) | `POPS_` + `SCREAMING_SNAKE` | all three (NL.9) |
| Files | `.hpp` `snake_case`; 110/110, zero `.h`/`.cc` | `.hpp` `snake_case`; `.cpp` for the TUs | Google (adapted) |
| Namespaces | lowercase; `pops`, `pops::detail` 45 occ. (`amr/cluster.hpp:49`) | `pops` public, `pops::detail` internal | all three |
| Template parameters | descriptive `PascalCase` (`Model` 127x); letter for arithmetic (`M`, `N`) | same | none |

Justification: practice is already homogeneous on all these axes and admits, for each line, only
one of the three positions without a massive rename. Three points deserve a note. (1) The case of
functions keeps `snake_case` (CG NL.10) because switching to Google or LLVM would rename the whole
public API for zero gain. (2) Constants have two assumed registers: free literal in
`kCamelCase`, name imposed by a concept in `snake_case`; this is not drift but a
structural distinction, to be documented so as not to pass for an inconsistency. (3) The `_` suffix
on non-public members removes the member-name / local-name ambiguity in init lists.

Status: adopted (2026-06-15).

## D2 Include guard

Divergence: Google and LLVM impose `#ifndef ..._H_` and proscribe `#pragma once`; CG (SF.8)
asks for a guard without imposing the form and tolerates `#pragma once`.

Repository practice: `#pragma once` in 110/110 headers, zero `#ifndef` as include guard (the
only `#ifndef` serve conditional compilation: `POPS_HAS_KOKKOS`, `POPS_HEADER_SIG`,
`NOMINMAX`/`WIN32_LEAN_AND_MEAN`). Unanimous.

Proposed decision: `#pragma once` on the first line of each header.

Justification: in deep header-only, `#pragma once` removes the maintenance of guard macros and
name collisions. The Google/LLVM position answers porting constraints that the
target compilers (gcc, clang, nvcc, MSVC) no longer impose. CG allows it.

Status: adopted (2026-06-15).

## D3 Include order

Divergence: Google places C++ std early (before third-party libs); LLVM places the system header
last; opposite orders. CG does not decide.

Repository practice: `<pops/...>` block first, blank line, then STL block `<...>`; angle brackets everywhere;
manual sort (`SortIncludes: false`).

Proposed decision: two groups, `<pops/...>` then STL, separated by a blank line; angle brackets for
everything; order maintained by hand, `SortIncludes` stays `false`.

Justification: neither Google nor LLVM matches the usage; an automatic sort would mix the two
blocks and break the intended order dependencies. The in-house order (own module first) is stable
and readable.

Status: adopted (2026-06-15).

## D4 Exceptions and error handling

Divergence: Google and LLVM forbid exceptions (LLVM compiles `-fno-exceptions`); CG
recommends them to signal failure (E.2, E.3).

Repository practice: host exceptions dominant (`throw std::runtime_error`: 134 in `include/pops`,
305 including Python bindings; 3 `std::invalid_argument`, zero `std::logic_error`); message prefixed
by a context then ` : `
(`runtime/native_loader.hpp:210`, `runtime/wall_predicate.hpp:33`). Zero `std::expected`,
`std::optional` marginal (3).

Proposed decision: exceptions allowed and idiomatic on the host path (config validation,
loading of `.so`, API errors); `std::runtime_error` by default, `std::invalid_argument` for
an invalid argument; message always prefixed by a context then ` : `. The device path (code
under `POPS_HD`) never `throw`s.

Justification: the device constraint already imposes the Google/LLVM ban on the hot part, but the host
has no reason to deprive itself of it and the usage there is massive and homogeneous. We split according to the execution path
rather than imposing a global ban that does not reflect the code.

Status: adopted (2026-06-15).

## D5 Asserts and contracts

Divergence: LLVM "assert liberally" plus `llvm_unreachable`; Google `assert()` plus in-house macros
`CHECK`/`DCHECK`; CG `Expects()`/`Ensures()` (I.6, I.8) via GSL.

Repository practice: 45 `static_assert` (compile-time constraints), 8 raw `assert(` only,
no `POPS_ASSERT` macro (does not exist).

Proposed decision: strong preference for `static_assert`. Runtime `assert` on the host path
only, sparingly. No GSL `Expects`/`Ensures`. No `CHECK`/`POPS_ASSERT` macro as long
as a recurring need is not established.

Justification: `Expects`/`Ensures` are not callable on device and `assert` on device is
costly or disabled; compile-time (concepts, `static_assert`) already covers the essential of the
contracts of this base.

Status: adopted (2026-06-15).

## D6 RTTI

Divergence: Google and LLVM forbid RTTI (LLVM has in-house `isa<>`/`cast<>`/`dyn_cast<>`); CG
(C.146) allows `dynamic_cast` for an unavoidable hierarchy navigation.

Repository practice: static design by templates/concepts, "device-clean" invariant (no vtable
in the kernels); no significant use of `dynamic_cast`/`typeid`.

Proposed decision: no RTTI on the numerical path; polymorphism goes through templates and
policies.

Justification: `typeid`/`dynamic_cast` are not usable on device; the constraint decides
for Google/LLVM and the architecture does not need it.

Status: adopted (2026-06-15).

## D7 `auto`

Divergence: CG (ES.11) permissive (avoid repeating type names); LLVM and Google restrictive
(only if readability increases).

Repository practice: 290 uses, including 168 trailing-return `) -> ` and 10 structured bindings. Pragmatic
usage.

Proposed decision: `auto` when it clarifies or avoids a heavy repetition (template returns,
iterators, structured bindings); explicit type when it carries useful information (units,
numerical semantics). Trailing-return-type accepted as a repository idiom.

Justification: the practice sits between CG and Google, healthy and readable in a
template-heavy base. We keep a pragmatic rule rather than a formal ban untenable here.

Status: adopted (2026-06-15).

## D8 `struct` vs `class`

Divergence: convergence in spirit. Google `struct` for passive data; CG (C.2, C.8) `class` if
invariant or non-public member; LLVM same informally.

Repository practice: 219 `struct` (POD, functors, policies without invariant) against 41 `class` carrying
the private members suffixed `_`.

Proposed decision: `struct` for an aggregate without invariant (POD, device functor, policy); `class`
as soon as there is an invariant to maintain or a non-public member, which implies the `_` suffix.

Justification: the three guides converge and the practice already follows this line.

Status: adopted (2026-06-15).

## D9 Documentation (Doxygen `///`)

Divergence: LLVM imposes Doxygen `///`; Google favors `//` without an imposed system; CG (NL.1-NL.4)
wants intent comments without an imposed system.

Repository practice: `///` for the API (5,057 lines), `//` for the internal (5,079), `///<` trailing
for the members (312). `/// @file` + `/// @brief` on 84/110 files (76%); `@param` 137x,
`@return` 17x (under-used). `/** */` blocks isolated to 4 files of `physics/`. Language: French without
accents, almost exclusively. There remain 26 files without a Doxygen header and one redundant double header
(prose `//` block paraphrasing the `/// @brief`).

Proposed decision: Doxygen `///` (`@` tags) for the API, `//` for the internal, `///<` trailing for
the members; `/// @file` + `/// @brief` on each file; proscribe `/** */`; `@param` for the
parameters and `@return` as soon as a function returns a meaningful value; French without accents
(ASCII). Remove the prose double header as edits go. **Commit
`CODE_DOCUMENTATION_CONVENTION.md`** (target of the link from `CODEBASE_AUDIT.md`, present in the working
tree but not committed, therefore outside version control) to fix the link, or redirect this link
to the present note.

Justification: the repository has already chosen Doxygen `///` (LLVM position) and ASCII French; it remains to
fill the gaps (24% of the headers, `@return`, `/** */` island) and to fix the links by committing
the convention file already present but not tracked. No change of system, only a
conformance update.

Status: adopted (2026-06-15).

## D10 TODO format

Divergence: Google imposes `// TODO: <context>` (historically `// TODO(user):`); LLVM and CG do not
prescribe anything.

Repository practice: no in-code TODO markers currently exist in `include/pops/` or `python/`
(a `grep -ri todo` over the sources returns none; only the top-level `todo.md` backlog file). The
rule below is therefore preventive: it fixes the format for future TODOs rather than normalizing an
existing inconsistent set.

Proposed decision: `// TODO(<context>) : <description>` where `<context>` identifies the work item or
the issue (e.g. `// TODO(ADC-124) : ...`). Keep the existing work-item numbers as context.

Justification: only Google decides; a single format makes TODOs greppable and traceable towards
Linear, at a low cost.

Status: adopted (2026-06-15).

## D11 `[[nodiscard]]`

Divergence: no guide has a numbered rule; LLVM had `LLVM_NODISCARD`; CG encourages it by
spirit for functions returning a status.

Repository practice: 0 `[[nodiscard]]`, deliberate (`-modernize-use-nodiscard` disabled in
`.clang-tidy`).

Proposed decision: no systematic introduction. Reserve it, case by case and with
justification, for the rare functions where ignoring the return is a certain bug (handle to release).

Justification: error handling goes through exception (D4), not through return code; the interest of
`[[nodiscard]]` is marginal here. Decision to reopen if error-code routines appear.

Status: adopted (2026-06-15).

## D12 `explicit`

Divergence: Google imposes `explicit` on single-argument constructors and conversions; CG
(C.46) "by default, declare single-argument constructors explicit"; LLVM silent.

Repository practice: ~10 `explicit` on single-argument constructors in `include/pops`, ~12 across the
reviewed scope (`include/pops` + `python/{bindings,system,amr_system}.cpp`) (`mesh/box_array.hpp:30`,
`runtime/system.hpp:66`).

Proposed decision: `explicit` required on any constructor callable with a single argument and
on the conversion operators, except explicit intent of implicit conversion (rare, to be justified).

Justification: Google/CG convergence, practice already installed, prevents the silent implicit conversions.
Zero cost.

Status: adopted (2026-06-15).

## D13 Parameter passing

Divergence: CG detailed (F.16 by value if copy is cheap otherwise `const&`; F.17 in-out `&`; F.18
`X&&` + `std::move`; F.20 prefer the return; F.21 struct for multiple outputs); Google "prefer
return values over output parameters", outputs historically by pointer; LLVM silent.

Repository practice: inputs `const&` or by value for small numerical PODs (`Real`, `State`);
return value preferred; `std::move` in init lists.

Proposed decision: input by value if the copy is cheap (scalars, small PODs passed to the
kernels), otherwise `const&`; in-out by non-const reference; prefer the return to output
parameters; dedicated struct for multiple related outputs; `X&&` + `std::move` for transferred
resources. No output parameter by pointer (CG, not the legacy Google).

Justification: aligns CG/Google on the dominant usage; passing small PODs by value is moreover
required by the device path.

Status: adopted (2026-06-15).

## D14 Early exits and nesting

Divergence: LLVM strongly prescribes early exits (`return`/`continue`, no `else` after
a `return`); Google and CG have no equivalent dedicated rule.

Repository practice: no written rule; style globally flat.

Proposed decision: encourage early exits to reduce nesting, without making it a
blocking rule.

Justification: only LLVM decides; the recommendation improves readability without imposing a
rewrite. Guide, not a verified constraint. Status: adopted (2026-06-15).

## D15 Automatic formatting

Divergence: each guide has its reference `.clang-format`; the repository already has its own.

Repository practice: `.clang-format` present (Google base, C++20, `IndentWidth 2`, `ColumnLimit 100`,
`PointerAlignment`/`ReferenceAlignment Left`, `SortIncludes: false`, `ReflowComments: false`,
`FixNamespaceComments: true`), plus `.editorconfig` (LF, 2 spaces C++, 4 spaces Python) and
`.clang-tidy` informative (`WarningsAsErrors` empty). `ReflowComments: false` leaves ~9% of lines
beyond 100 columns (long FR Doxygen comments). The `format`/`tidy` jobs of `quality.yml` only
report.

Proposed decision: the existing `.clang-format` is the formatting reference; decisions
D1-D14 complete it on what it does not cover. The switch of the `format`/`tidy` jobs from informative
to blocking is **deferred to the milestone *Code quality & hardened CI* (epic ADC-105)**, which
will decide which checks become a gate once the base is cleaned up.

Justification: the mechanical formatting is already tooled and stable; nothing to decide here, only
to refer the hardening to the milestone that carries it. Status: adopted (2026-06-15).

## Follow-ups

Once validated, these decisions feed `CODE_DOCUMENTATION_CONVENTION.md` (already present, to be committed, D9) and the
progressive hardening of `quality.yml` (D15, milestone ADC-105). The inconsistencies noted that do not
require a style arbitration (`CODE_DOCUMENTATION_CONVENTION.md` present but not committed,
Doxygen headers missing on 26 files, redundant double
header, `/** */` island of `physics/`) are conformance updates to the present note, not
open choices.
