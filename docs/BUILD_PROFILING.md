# Report - Compilation profiling of `adc_cpp`

**Audited worktree:** `/Users/romaindespoulain/.codex/worktrees/8a24/pops_cpp_audit`
**Date:** 2026-06-10 - **Measured toolchain:** AppleClang (Xcode), macOS arm64, `-O3 -std=c++2b`, Release
**Goal:** reduce compilation time **without changing the numerics or the scientific behavior** (the `-O` level does not change the IEEE results as long as `-ffast-math` is not used, which is the case here).

---

## Outcome - milestone "Compilation" (2026-06-19)

The leads identified below were implemented and merged. The diagnostic that follows is kept as the
analysis that motivated them. No numerical change (the `-O` level and the verbatim TU moves keep the
IEEE results bit-identical; guarded by the `dmax==0` parity suite).

| Lever | Issue | Result |
|---|---|---|
| Split the heavy TUs by transport x flux | ADC-335 | `_pops` went from **3 TUs to 16**; the critical path dropped from the single ~469 s `system.cpp` to the slowest sub-TU. Measured on this Mac: a full module build **1112 s -> 284 s (~3.9x)**. |
| Flux-subdivide the isothermal TU | ADC-342 | the post-split long pole (`system_isothermal`, ~120 `-O3` leaves) is split into rusanov + hll sub-TUs (~60 leaves each). |
| Compile the heavy TUs once | ADC-336 | the ~20x test recompile collapses to one `pops_runtime_{system,amr}` object library, linked by the tests (size-1 Ninja pool as the OOM guard). |
| Derive `-j` from the core count | ADC-339 | no hardcoded `-j8`; the build saturates all cores by default. |
| Pinned per-OS conda toolchain + heavy-TU pool | ADC-338 | AppleClang (macOS) / conda gcc 14.2 (Linux) pinned; `POPS_HEAVY_TU_POOL` lets `-j` parallelize the now-small sub-TUs on a high-RAM host (CI keeps the size-1 OOM guard). |
| `optnone` on the cold factories | ADC-337 | the once-per-block string->closure wiring is no longer `-O3`-optimized; the hot kernels stay `-O3`. |
| Gate unused flux x limiter x integrator combos (P1-C) | ADC-340 | **NO-GO**: the `if constexpr` capability guards already prune the impossible combos (recoverable residual ~ 0); further gating would be a forbidden scenario allowlist. Documented and closed. |
| Targeted PCH for the module (P2) | ADC-361 | **NO-GO**: a precompiled header for pybind11/Kokkos made a COLD `_pops` build **slower** (`361 s -> 416 s`, +15%) on AppleClang/macOS. The frontend is small relative to the `-O3` backend (cf. 3.2), so the PCH build + per-TU PCH load cost more than the parse they save. Cold AppleClang only (GCC-Linux and the warm/ccache path not re-measured -- both could only worsen it). Default OFF, not added. See the PCH note in section 6 (P2). |

The CI `--parallel 2` (7 GB-runner memory bound) and the WSL2 `-j 6` (RAM bound) are intentionally kept.

---

## TL;DR

| Finding | Measured evidence | Lever |
|---|---|---|
| **`system.cpp` = 469 s, `amr_system.cpp` = 218 s** per compilation (single-threaded) | `/usr/bin/time -p` on the isolated TU | - |
| The cost is **93 % in the LLVM `-O3` backend**, not in parsing nor the frontend | `-ftime-trace`: Backend 435 s / 468 s; `-O0` drops back to **41 s** | lower `-O`, reduce the number of instantiations |
| `_pops` compiles only **3 files** -> `-j` caps at 3 cores, critical path = `system.cpp` (~469 s) | `python/CMakeLists.txt:9` | **split the TU** |
| `system.cpp`/`amr_system.cpp` recompiled **20x** in the test build (6 + 14) | `grep` on `build.ninja` | **single object library** |
| Root cause = **combinatorial explosion** (~36 CompositeModels x 4 fluxes x 4 limiters x ~3 integrators ~ 1700 paths) all optimized at `-O3` in **one** TU | `model_factory.hpp` + top OptFunction of the trace | reduce the spread / split |
| **Linking is NOT a problem** (< 1 s per executable) | `.ninja_log`: top link = 0.85 s | - |

**Sensitivity to the optimization level (`system.cpp`, single-threaded, measured):**

| `-O` | time | gain vs `-O3` |
|------|-------|---------------|
| `-O3` (current) | **469 s** | reference |
| `-O2` | 363 s | **-23 %** |
| `-O1` | 284 s | **-39 %** |
| `-O0` | 41 s | **-91 %** |

The ~41 s at `-O0` are the incompressible cost (parsing + frontend + unoptimized codegen). Everything else (~428 s) is the optimizer working on the ~1700 instantiated leaf functions.

---

## 1. Scope & method

- **Build files read:** [`CMakeLists.txt`](../CMakeLists.txt), [`python/CMakeLists.txt`](../python/CMakeLists.txt), [`tests/CMakeLists.txt`](../tests/CMakeLists.txt), [`.github/workflows/ci.yml`](../.github/workflows/ci.yml), [`.github/workflows/docs.yml`](../.github/workflows/docs.yml).
- **Reused builds** (main folder `Documents/Stage_Romain/adc_cpp/`, same code base, Release/AppleClang/serial): `build-docs`, `build-audit` (Ninja, usable `.ninja_log`), `build-master`, `build-py` (Makefiles, stale). **No file deleted, no clean build.**
- **Clean measurements:** isolated compilation, **single-threaded**, output to `/tmp`, flags taken verbatim from the test build (`-O3 -DNDEBUG -std=c++2b -arch arm64 -isysroot ... -I include`), with `-ftime-trace` for the internal breakdown.

> **Caveat 1.** The reused builds live in the main repository (`adc_cpp/`), not in this worktree; the `CMakeLists` are identical, so the `.ninja_log` are representative. The measured sources (`system.cpp`/`amr_system.cpp`) come **from the audited worktree**.
> **Caveat 2.** The durations in the `.ninja_log` of `build-docs` (600-934 s per TU) are **inflated by contention** (parallel build + concurrent docs). The authoritative numbers in this report are the **isolated single-threaded** compilations (469 s / 218 s).

---

## 2. Build architecture (what actually compiles)

- **The `pops` core is `INTERFACE` (header-only)** - `CMakeLists.txt:57`. It compiles nothing on its own; all the cost is carried by the TUs that *instantiate* it.
- **Python module `_pops` = 3 TUs only**: `bindings.cpp`, `system.cpp`, `amr_system.cpp` - `python/CMakeLists.txt:9`. This is where your pain "even `_pops` is slow" comes from: there is nothing to parallelize beyond 3 cores, and the critical path is the slowest TU (`system.cpp`).
- **Tests = ~113 executables** (`grep -c CXX_EXECUTABLE_LINKER build.ninja` -> 113; 133 `.cpp.o` objects). Most are small TUs, **but 20 of them re-list `python/bindings/system/base/system.cpp` or `python/bindings/amr/amr_system.cpp` as an extra source** (e.g. `tests/CMakeLists.txt:204, 304, 334-336, 349, 374, 393, 411, 426, 435...`), so they fully recompile the heavy TU.

---

## 3. Detailed measurements

### 3.1 Cost per translation unit (single-threaded, clean)

| TU | `-O3` real | user | Backend % |
|---|---|---|---|
| `python/bindings/system/base/system.cpp` | **469.4 s** | 379 s | 93 % |
| `python/bindings/amr/amr_system.cpp` | **217.6 s** | 192 s | 93 % |

### 3.2 Where the time goes - `-ftime-trace` (`system.cpp`)

```
ExecuteCompiler                 468,1 s
  Frontend                       31,0 s   (parsing + instanciation côté front)
  Backend                       435,6 s   ← 93 %
    OptModule / OptFunction     ~221 s    (optimiseur LLVM -O3)
  Source (parsing en-têtes)       1,2 s   ← négligeable : ce n'est PAS un problème de gros headers
  InstantiateFunction            19,1 s
  PerformPendingInstantiations   18,6 s
```

**Reading:** neither the headers (1.2 s) nor the frontend instantiation (31 s) are the bottleneck. It is the **`-O3` backend optimizer** that dominates. And it is not any single hot function: the top OptFunction caps at **0.25 s** per function - the cost is the **sum of thousands** of small instantiated functions:

```
0,25s  SSPRK3Step::take_step<BlockRhsEvalEb<VanLeer, RusanovFlux, …>>
0,24s  SSPRK3Step::take_step<BlockRhsEvalMasked<Minmod, RusanovFlux, …>>
0,23s  SSPRK3Step::take_step<BlockRhsEval<Minmod, HLLFlux, CompositeModel<…>>>
0,21s  build_block<NoSlope, RusanovFlux, CompositeModel<Euler, PotentialForce…>>
…  (× ~1700 combinaisons)
```

### 3.3 Root cause - combinatorial explosion of the dispatch factories

`include/pops/runtime/model_factory.hpp` nests three dispatchers:

```
dispatch_transport : exb | compressible | isothermal                         (3)
  × dispatch_source : none | potential | gravity | magnetic | potential+mag  (~4)
    × dispatch_elliptic : charge | background | gravity                       (3)
```

~ **36 `CompositeModel`**; each then goes through `make_block`/`build_block` which dispatches on **4 fluxes** (rusanov/hll/hllc/roe) x **4 limiters** (noslope/minmod/vanleer/weno5) x **~3 integrators** (ssprk2/ssprk3/imex) -> **~1700 leaf paths**, all **fully instantiated and optimized at `-O3` in a single TU**. This is exactly the pattern already described in the comment in `tests/CMakeLists.txt:10-25` ("~33 CompositeModels x flux families x limiters").

### 3.4 Duplication of the heavy TUs (test build, serial)

```
python/bindings/system/base/system.cpp      compilé  6×   (grep -c "python/bindings/system/base/system.cpp.o:" build.ninja)
python/bindings/amr/amr_system.cpp  compilé 14×
```

- **Duplicated CPU cost (serial):** 6 x 469 + 14 x 218 ~ **5866 CPU-s**.
- **If compiled only once:** 469 + 218 = **687 s**. -> **~88 % of work wasted** (~5200 CPU-s recoverable).
- With `POPS_USE_MPI=ON`, ~8 additional test variants recompile these TUs (`test_mpi_amr_*`, `test_mpi_system_*`) -> the duplication gets worse.

### 3.5 CI impact

`ci.yml` builds **6 configurations** (`build`, `build-py`, `build-mpi`, `build-kokkos`, `build-kokkos-omp`, `build-kokkos-py`), **all at `--parallel 2`** (runner memory constraint ~7 GB, `ci.yml:111-119`). Each config recompiles these heavy TUs from scratch:

- `_pops` in CI (`--parallel 2`): ~469 s + the rest ~ **8 min minimum**, bounded by `system.cpp`.
- The full test build (`build`, 20 heavy TUs + 113 small ones at `--parallel 2`) ~ **~1 h** on the runner (slower than this Mac).
- The **Kokkos** configs are even heavier (device instantiation via `nvcc_wrapper`).

---

## 4. Likely causes, ranked by impact

1. **[Dominant] `-O3` backend optimizer on a combinatorial explosion of instantiations.** 93 % of the time; ~1700 leaf functions. Evidence: `-ftime-trace` (Backend 435/468 s), scaling `-O0`->41 s.
2. **[Structural] `_pops` has only 3 TUs** -> impossible to parallelize; critical path = `system.cpp` (469 s). Evidence: `python/CMakeLists.txt:9`.
3. **[Waste] `system.cpp`/`amr_system.cpp` recompiled 20x in the tests** (x6 in CI). Evidence: `grep` on `build.ninja`.
4. **[Secondary] Frontend / instantiation 31 s** per TU - real but 7x smaller than the backend.
5. **Non-causes (ruled out by the measurements):** large headers (Source = 1.2 s), linking (< 1 s/exe), LTO (disabled - leave it OFF), Kokkos/pybind in `system.cpp` (the measured TU has neither and already costs 469 s).

---

## 5. Most expensive targets

| Rank | Target | `-O3` single-threaded cost | Multiplicity |
|---|---|---|---|
| 1 | `python/bindings/system/base/system.cpp` | **469 s** | x1 in `_pops`, x6 in tests |
| 2 | `python/bindings/amr/amr_system.cpp` | **218 s** | x1 in `_pops`, x14 in tests |
| 3 | `bindings.cpp` | not isolated (includes pybind11 + facade) | x1 in `_pops` |
| 4 | small test TUs instantiating `make_block` (`test_block_builder`, `test_compiled_model_parity`...) | 15-35 s each | x113 |

---

## 6. Roadmap P0 / P1 / P2

> All the leads below preserve the numerical results: either they do not touch the generated code (TU partition, object library), or they only change the optimization level (identical IEEE results without `-ffast-math`).

### P0 - high impact, ~zero risk

- **P0-A. Compile `system.cpp` + `amr_system.cpp` ONCE into a shared object library.**
  `add_library(pops_runtime STATIC python/bindings/system/base/system.cpp python/bindings/amr/amr_system.cpp)` linked by `_pops` **and** by the 20 test executables (replace the `add_executable(test_x test_x.cpp .../system.cpp)` with a link to `pops_runtime`).
  -> removes ~18 recompilations (serial) / ~26 (MPI) per config, **x6 in CI**. Test build gain: **~88 %** of the heavy TU cost.
  *Caution:* some tests pass specific `-D` (`POPS_TEST_CXX`, `ENABLE_EXPORTS`) - those stay on their own test `.cpp`; only the `system.cpp`/`amr_system.cpp` object is shared. `pops_cap_opt_for_kokkos_ram` (the targeted `-O0`) would then apply once to the lib, not N times.

- **P0-B. Split each heavy TU by dispatch axis to parallelize.**
  Partition the instantiations of `system.cpp` (e.g. one `.cpp` per flux family, or per transport) with explicit instantiation. `_pops` then goes from 3 to ~8-12 TUs -> `-j` becomes useful again, critical path `469 s -> ~469/N`.
  -> this is **THE** fix for your pain "`_pops` alone is slow". Identical behavior (same instantiations, simply spread out).

### P1 - high impact, to be validated

- **P1-A. Lower `-O` on these 2 TUs.** `-O2` = **-23 %** for free, `-O1` = **-39 %**. The vast majority of the ~1700 instantiated functions are **cold dispatch wiring** (run once at setup), not the hot `for_each_cell` loop. -> measure the runtime impact of `-O2` on a representative case; if < 5 %, switch these TUs (and their test duplicates) to `-O2`.
- **P1-B. Surgical `-O` partitioning.** Mark **only** the cold factories (`dispatch_*`, `make_block` glue) with `__attribute__((optnone))` / `#pragma clang optimize off`, keeping the kernels (`take_step`, `for_each_cell`) at `-O3`. -> big compile gain, **zero** impact on the hot path. More invasive (the cold code must be isolated).
- **P1-C. Reduce the combinatorial spread.** Gate at compile time the combinations (flux x limiter x model) never exercised at runtime. -> fewer leaves to optimize. Requires knowing which combos are actually used (application decision, not numerical).

### P2 - convenience / non-regression

- **P2-A. `ccache`** (`-DCMAKE_CXX_COMPILER_LAUNCHER=ccache`) for local iteration: no gain on the 1st build nor when a header changes (the `POPS_HEADER_SIG` signature invalidates), but useful when only a test `.cpp` changes.
- **P2-B. CI `-ftime-trace` safeguard**: per-TU compilation time budget to detect a regression (a new flux/limiter that adds a dimension to the combinatorial product).
- **P2-C. Leave LTO OFF** (enabling it would clearly worsen the backend). Linking is not an issue (< 1 s/exe) - no need to invest in mold/lld.
- **PCH: NO-GO, measured (ADC-361).** The prediction above (frontend ~31 s vs ~435 s backend per heavy TU) was confirmed by measurement: a precompiled header of `<pybind11/pybind11.h>` (+ `<Kokkos_Core.hpp>`) on the `_pops` module made a COLD build **slower**, 361 s -> 416 s (+15%, isolated empty ccache, `POPS_HEAVY_TU_POOL=4`, AppleClang macOS arm64). The extra PCH-build translation unit plus the per-TU cost of loading a large PCH exceed the parse time it removes, because the `-O3` backend pass -- not the frontend -- dominates this build. The cold number is the measured result; the warm/ccache path was reasoned, not benchmarked (it would need `CCACHE_SLOPPINESS=pch_defines,time_macros` to not defeat ccache on warm rebuilds, and PCH invalidation can only make it worse, never rescue it). Closed NO-GO; the opt-in flag was NOT merged (no dead config). Re-measure only if a future toolchain profile shows the frontend dominating; reproduce by adding `target_precompile_headers(_pops PRIVATE <pybind11/pybind11.h> <Kokkos_Core.hpp>)` and timing a cold `cmake --build --target _pops` with an isolated `CCACHE_DIR`, with vs without.

**Recommended order:** P0-A (CI/tests, safe and immediate) -> P0-B (unblocks your `_pops` pain) -> P1-A/P1-B (shave the backend) -> P1-C (reduce the spread).

---

## 7. Commands used (reproducibility)

```bash
# Inventaire des builds et options
grep -E '^(POPS_USE_|POPS_BUILD_|CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER):' <build>/CMakeCache.txt

# Durées par cible depuis les .ninja_log (format: start_ms end_ms mtime output hash ; durée = end-start)
#   parsées en Python (dédup par output, tri décroissant) -- cf. §3

# Flags exacts d'une TU lourde (depuis build.ninja du build de tests Ninja)
sed -n '2550,2556p' <build-audit>/build.ninja      # → -O3 -DNDEBUG -std=c++2b -arch arm64 -isysroot … -I include

# Mesure propre, mono-thread, avec ventilation interne
SDK=/Applications/Xcode.app/.../MacOSX26.5.sdk
CXX=/Applications/Xcode.app/.../usr/bin/c++
/usr/bin/time -p "$CXX" -O3 -DNDEBUG -std=c++2b -arch arm64 -isysroot "$SDK" -I include \
  -ftime-trace -c python/bindings/system/base/system.cpp -o /tmp/pops_system.o      # → real 469 s, trace JSON
# idem amr_system.cpp (218 s) ; idem en -O2/-O1/-O0 pour le scaling

# Comptage de la duplication
grep -c "python/bindings/system/base/system.cpp.o:"     <build-audit>/build.ninja  # → 6
grep -c "python/bindings/amr/amr_system.cpp.o:" <build-audit>/build.ninja  # → 14
```

> The temporary artifacts (`/tmp/pops_*.o`, JSON traces ~150 MB) were removed after analysis. No project file was modified or deleted; no clean build was launched.
