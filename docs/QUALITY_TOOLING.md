# Quality tooling and static analysis

`adc_cpp` has a static-analysis / code-quality chain that is **deliberately kept off the
critical path of PRs**. The `ci.yml` gate remains the authority on compilability and correctness
(tests) ; quality runs separately, without slowing down the development cycle.

Linear tracking : epic **ADC-105** (milestone *Code quality and hardened CI*).

## Triggers (`.github/workflows/quality.yml`)

The `Quality` workflow **never runs on a push or on an ordinary PR**. It is triggered on :

| Trigger | When |
| --- | --- |
| `schedule` (cron `0 4 * * 0`) | every **Sunday** ~04:00 UTC |
| `workflow_dispatch` | manually (*Actions* tab -> *Quality* -> *Run workflow*) |
| `quality` label on a PR | full **opt-in** pass on that PR (risky PR) |

> The `quality` label must exist in the repository :
> `gh label create quality --description "Run quality.yml on this PR" --color FBCA04`.
> The workflow only takes effect (cron, dispatch, label) once it is present on the default branch
> (`master`).

## Policy : informative first

At startup, **almost nothing is blocking** : findings appear as GitHub annotations, in the job
summary and as artifacts, but they do not make the run fail (no `-Werror`, `clang-format` in
`--dry-run`, `clang-tidy` without `WarningsAsErrors`). **The one exception already in effect** : the
`fuzz` job is blocking (a libFuzzer crash is always a real bug ; see the first switch below). All
CMake options are **OFF by default** -> `ci.yml`, local builds and `adc_cases` are unchanged. The
other jobs switch to blocking according to the criteria in the dedicated section.

## The eight jobs

| Job | Tool | Config | Preset / option |
| --- | --- | --- | --- |
| `format` | clang-format + ruff | `.clang-format`, `[tool.ruff]` | -- (no build) |
| `warnings` | gcc `-Wall -Wextra ...` | `cmake/PopsDevTooling.cmake` | preset `ci-warnings` (`POPS_ENABLE_WARNINGS`) |
| `tidy` | clang-tidy | `.clang-tidy` | preset `ci-kokkos` (compile DB) |
| `sanitizers` | ASan + UBSan | `cmake/PopsDevTooling.cmake` | preset `ci-asan` (`POPS_ENABLE_SANITIZERS`) |
| `tsan` | ThreadSanitizer (races) | `cmake/PopsDevTooling.cmake`, `tsan-suppressions.txt` | preset `ci-tsan` (`POPS_ENABLE_TSAN`, clang + Kokkos OpenMP) |
| `fuzz` | libFuzzer (90 s/target) | `fuzz/` (invariant harnesses) | preset `ci-fuzz` (`POPS_BUILD_FUZZING`, clang) |
| `coverage` | gcov + gcovr | `cmake/PopsDevTooling.cmake` | preset `ci-coverage` (`POPS_ENABLE_COVERAGE`) |
| `codeql` | CodeQL C++ | suite `security-and-quality` | preset `ci-kokkos` (traced build) |

Warnings, sanitizers, TSan and coverage are carried by an **`INTERFACE pops_dev_options`** target that
**only** the internal targets link in `PRIVATE` (the ~140 tests via `pops_add_test`, the `_pops`
module). The public core `pops::pops` is never touched -> no flag leaks to consumers.

CodeQL is free here because the repository is **public** ; the results show up in
**Security > Code scanning**.

## Fuzzing (`fuzz/`)

Three **invariant** libFuzzer harnesses over the pure core components (no Kokkos at runtime),
compiled `-fsanitize=fuzzer,address,undefined`, so three oracles at once : crash, memory, UB, plus
the explicit invariants (`abort()` -> reproducible artifact).

| Harness | Target | Invariants |
| --- | --- | --- |
| `fuzz_box2d` | `Box2D` arithmetic | symmetric / contained intersect, `refine(coarsen(b)) = b`, `coarsen(refine(b))` contains `b` (floor division on negatives), `grow` / `shift` inverses, `contains` iff `intersect` |
| `fuzz_cluster` | `berger_rigoutsos` | every tagged cell covered ; boxes non-empty, within the domain, dims <= `max_box_size` |
| `fuzz_dense_eig` | `real_eig_minmax<N>` (N=2,3,4,8) | if `converged` : `lmin <= lmax`, finite, within Gershgorin (tolerance) ; NaN/Inf inputs : no-crash / no-UB |

Reproduce a CI crash : download the `fuzz-crashes` artifact, then
`./build-fuzz/bin/<target> <crash-file>` (single, deterministic run).

## ThreadSanitizer (`tsan`)

The `tsan` job runs the whole `ctest` suite under **ThreadSanitizer** on the **Kokkos OpenMP**
backend -- the angle the other sanitizers miss. The gate's `ctest` runs **Serial** (zero threads, so
no race can ever appear), yet OpenMP is the on-node production backend : `for_each_cell`, the
reductions and the `fill_boundary` halo packing are exactly where a race stays latent. Two
mandatory choices come from how the runtimes behave under TSan :

- **clang + libomp LLVM, never gcc/libgomp.** `libgomp` is not TSan-aware -> a storm of false
  positives. The `ci-tsan` preset compiles with `clang++`, and `setup-kokkos` (`exec-space: openmp`,
  `compiler: clang`) installs `libomp-dev` and a Kokkos OpenMP install (separate cache key).
- **`tsan-suppressions.txt` (repo root) is seeded, not final.** Kokkos and libomp are linked but
  uninstrumented, so their internals report benign races ; each suppression is justified there, and a
  race inside `include/pops/` must never be suppressed. The first weekly run triages signal vs noise
  (`OMP_NUM_THREADS=2`, `TSAN_OPTIONS` points at the file) ; a confirmed race becomes a dedicated
  High-priority issue.

ASan and TSan are **mutually exclusive** (one memory/thread runtime per binary) : enabling both
`POPS_ENABLE_SANITIZERS` and `POPS_ENABLE_TSAN` is a hard CMake error -- use `ci-tsan` **or** `ci-asan`.

## pre-commit (auto-format at commit, opt-in, local)

```bash
pipx install pre-commit && pre-commit install   # once per clone
```

After that, each commit auto-formats the **touched** files (clang-format, ruff, file hygiene) ;
see `.pre-commit-config.yaml`. Nothing is imposed on anyone who does not install it.

## Dependabot

`.github/dependabot.yml` : weekly grouped bumps of the pinned GitHub Actions (triggered by the
Node 20 deprecation of `actions/cache@v4` / `upload-artifact@v4` seen on the first run).

## Reproduce locally

```bash
# Style
clang-format --dry-run --Werror include/pops/**/*.hpp     # reports; -i to apply

# Strict warnings (Kokkos required: conda env 'pops' active, or KOKKOS_PREFIX pointing at an install)
cmake --preset parallel -DPOPS_ENABLE_WARNINGS=ON
cmake --build --preset parallel

# ASan+UBSan sanitizers
cmake --preset parallel -DPOPS_ENABLE_SANITIZERS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build --preset parallel
ASAN_OPTIONS=detect_leaks=0:detect_container_overflow=0 ctest --preset parallel --output-on-failure

# TSan races (clang + Kokkos OpenMP required; mutually exclusive with ASan). KOKKOS_PREFIX must point
# at an OpenMP-enabled, clang-built Kokkos install (the CI builds one; see setup-kokkos exec-space).
KOKKOS_PREFIX=<openmp-clang-kokkos> cmake --preset ci-tsan \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++   # or your LLVM clang++
cmake --build --preset ci-tsan
OMP_NUM_THREADS=2 TSAN_OPTIONS=suppressions=$PWD/tsan-suppressions.txt ctest --preset ci-tsan --output-on-failure

# clang-tidy (after a configure that exports compile_commands.json)
run-clang-tidy -p build-kokkos 'tests/.*\.cpp'   # build-kokkos = binaryDir of the parallel/ci-kokkos preset

# Fuzzing (LLVM clang required; on macOS, the Homebrew one -- AppleClang has no libFuzzer).
# macOS: -DPOPS_FUZZ_SANITIZERS=undefined is MANDATORY -- the Homebrew LLVM ASan deadlocks BEFORE
# main on recent macOS (re-entrant init via dyld, confirmed with `sample`: 95% CPU, zero output,
# MallocNanoZone=0 is not enough). The Linux CI keeps the full address,undefined default.
KOKKOS_PREFIX=$CONDA_PREFIX cmake --preset ci-fuzz \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ -DPOPS_FUZZ_SANITIZERS=undefined
cmake --build --preset ci-fuzz
./build-fuzz/bin/fuzz_cluster -max_total_time=60

# Coverage (GCC required -> in practice: the CI job)
cmake --preset ci-coverage && cmake --build --preset ci-coverage && ctest --preset ci-coverage
gcovr --root . build-kokkos-coverage --filter 'include/pops/' --print-summary

# Python lint
ruff check python          # see [tool.ruff] in pyproject.toml; --fix to apply
```

> In CI, the **gcc** jobs (warnings, tidy, sanitizers, coverage, codeql) reuse the Kokkos Serial
> install **cached** by the gate (`ci.yml`), via the composite action `.github/actions/setup-kokkos`
> (same `...-serial-cxx20-gcc-pic` key). The `fuzz` job compiles with **clang** -> separate Kokkos
> install (`...-serial-cxx20-clang-pic` key) ; the `tsan` job uses **clang + OpenMP** -> its own
> (`...-openmp-cxx20-clang-pic` key). Each is rebuilt once then cached in turn.

## From informative to blocking : switch criteria

The "informative first" policy is not a final state : each job has an **objective criterion** that
triggers its switch to blocking. Until the criterion is met, the job stays an observation. Once met,
we harden.

| Job | Switch criterion | Action |
| --- | --- | --- |
| `warnings` | 0 `warning:` on `master` (weekly report) | `-Werror` in the `ci-warnings` preset |
| `format` | sweep **ADC-118** merged (cleaned base) | the job **fails** on any style deviation |
| `sanitizers` | 4 **consecutive green** weekly runs | drop the tolerance (`ctest` fatal on a sanitizer report) |
| `tsan` | suppressions triaged + 4 **consecutive green** weekly runs | drop the tolerance (`ctest` fatal on a TSan report) |
| `tidy` | one check family clean, family by family | `WarningsAsErrors` on the cleaned families |
| `fuzz` | from the **first green run** | **BLOCKING** (applied: a fuzz crash is always real) |
| `coverage` | never | an **observation** metric, not a gate |

Each switch = a small **dedicated PR**, tracked in **ADC-120**, that changes a single job
(traceability of the hardening).

## Out of scope (future extensions)

- `clang-format` sweep of the whole base (massive reformat) -- **ADC-118** (a quiet window is
  required: it would conflict with all open branches).
- Switch to `-Werror` / blocking PRs -- after cleanup.
- include-what-you-use, cppcheck, OSS-Fuzz / persistent corpus, Python DSL fuzzing.
