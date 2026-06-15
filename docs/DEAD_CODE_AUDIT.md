# Dead-code audit (cppcheck `unusedFunction`)

One-shot dead-code pass for `adc_cpp`, in the spirit of "remove all unnecessary code" adapted to a
**young** repository : a tooled pass, not a manual hunt. Linear tracking : **ADC-123** (milestone
*Code quality and hardened CI*). This is **not** a permanent CI job -- CodeQL and clang-tidy already
occupy the static-analysis niche ; the `unusedFunction` whole-program check is the one thing neither
`-Wunused` (already on via `ADC_ENABLE_WARNINGS`) nor clang-tidy sees, so it is run on demand and
recorded here.

## Command

```bash
cppcheck --enable=unusedFunction --project=build/compile_commands.json \
  --suppress=missingIncludeSystem -q 2> cppcheck-dead.txt
```

`unusedFunction` is a whole-program check, so it must run single-threaded (cppcheck disables it under
`-j > 1`). The compile database is the tests build (`ADC_BUILD_TESTS=ON`).

## Result : zero confirmed-dead functions

cppcheck flagged ~26 `unusedFunction` candidates. A whole-tree usage sweep (`grep -w` across
`include/ python/ tests/ bench/ fuzz/` **and** the sibling `adc_cases` repo) confirms **every one is
either used or deliberately kept** -- so **nothing is removed**. This is the expected outcome of a
`unusedFunction` run on a **header-only** library : cppcheck only sees the translation units in the
compile database (the tests), never `adc_cases`, the pybind11 bindings, or the cross-header and
codegen/`dlsym` use sites. The article's rule applies throughout : **when in doubt, do not delete.**

### Used (false positives from partial TU coverage)

21 of the candidates are referenced well outside their defining header -- heavily so. A few examples
from the sweep (number of referencing files) :

| Function | Files | Note |
| --- | --- | --- |
| `set_val` (`mesh/fab2d`, `mesh/multifab`) | 107 | core MultiFab fill, used everywhere |
| `ncomp` (`mesh/fab2d`) | 53 | component-count accessor |
| `boxes` (`mesh/box_array`) | 49 | box-list accessor |
| `n_ghost` (`mesh/fab2d`) | 26 | ghost-width accessor |
| `r_cell` (`mesh/geometry`) | 18 | polar radial centre |
| `sync_host` (`mesh/for_each`) | 19 | device→host fence wrapper |
| `for_each_cell_reduce_sum` / `reduce_max_cell` … | 3–8 each | template reductions (templates evade `unusedFunction`) |
| `role_from_name` (`core/variables`) | 10 | also bound in `python/` |
| `tag_union` (`amr/tag_box`), `extra_field` (`core/state`), `count` (`amr/tag_box`) … | 2–3 each | used in tests / `adc_cases` / runtime |

cppcheck's `unusedFunction` is structurally unreliable on inline header accessors and templates ; the
sweep is the authority here.

### No real caller -- kept, with rationale

Five functions have no real call site (only their defining header, or matched elsewhere solely by a
same-named symbol). None is confirmed-dead internal code :

| Function | Header | Why kept |
| --- | --- | --- |
| `var_names_meta<Model>` | `core/variables.hpp` | **dlsym / codegen entry point.** Emitted into the optional `extern "C" adc_compiled_var_names` symbol (macro at `variables.hpp:143`) and read by `runtime/native_loader.hpp:183` via `dynlib::sym(...)` (test `python/tests/test_dsl_abi_metadata.py`). Invisible to a static call-graph -- exactly the "DSL entry points resolved by dlsym" false-positive class. |
| `roles_meta<Model>` | `core/variables.hpp` | Same compiled-model ABI path (`adc_compiled_roles`). |
| `theta_face` | `mesh/geometry.hpp` | Completes the `r_cell` / `r_face` / `theta_cell` / `theta_face` accessor set (the other three are used) ; removing one breaks a coherent, documented API surface. |
| `sound_speed` | `physics/euler.hpp` | Lone uncalled accessor on the otherwise-live public `Euler` model (its sibling `pressure` is used widely). Standard physics-model API completeness -- same class as `theta_face`. The only tree matches are a Python DSL local variable and the `c = sqrt(theta)` concept in docs, not C++ callers. |
| `arena_stats` | `core/allocator.hpp` | Public diagnostic accessor over `ManagedArena::instance().stats()` ; introspection API, kept by the "when in doubt" rule. |

## Repository hygiene (root artifacts)

The issue also asked to check for stray root artifacts. `.gitignore` already covers them and **none is
tracked** (`git ls-files`) :

- `build/`, `build*/`, `cmake-build-*/` -- ignored (`.gitignore:1-3`) ;
- `.DS_Store` -- ignored (`.gitignore:8`) ;
- root `*.csv` (e.g. a local `diocotron_theory.csv`) -- untracked working-tree artifact ; the
  `quality.yml` `format` job's *root allowlist* step already warns if such a file is ever committed.

The only tracked root document of note is `todo.md` (see below).

## `todo.md` : kept as a worklog, not gutted

The issue suggested reducing `todo.md` (~70 KB) to a Linear pointer "so it does not compete with
Linear as the source of truth". On inspection that reduction is **deliberately not performed**, because
curated docs already verdict the file **keep** and depend on its content :

- `docs/DOC_REFONTE_AUDIT.md` classifies `todo.md` as **keep / active** ;
- `docs/PAPER_ROADMAP.md` cross-references **`todo.md` section 6** (Hoffart M1/M2/M3 method notes) in
  three places ;
- `docs/CODEBASE_AUDIT.md` recommends treating it as a **living backlog**.

Gutting it would break those cross-references and lose method notes. Instead, the source-of-truth
concern is addressed **non-destructively** by a header banner on `todo.md` stating that **Linear owns
active work-tracking** and that the file is a historical method/worklog. Its open items are not lost :
the 2 open `[ ]` entries are research items already in `docs/RESEARCH_BACKLOG.md`, and the 9 `[~]`
partials are superseded, done, or tracked in Linear.
