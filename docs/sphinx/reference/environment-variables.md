# Environment variables

`adc_cpp` reads a small set of `POPS_*` environment variables. They are all
optional: with none of them set the toolchain auto-detects everything and the
runtime stays inert. Each variable below is documented with its effect, its
default, and the exact file that reads it, so this page never describes a knob
that the code does not actually honor.

There are three families:

- **Codegen / toolchain** knobs, read by the Python DSL compilation path when it
  builds a problem `.so` (`pops.codegen`). They override auto-detection of the
  compiler, the headers, the Kokkos install, the optimization flags, and the
  cache directory.
- **Python runtime defaults**, read by `pops.runtime` to supply a default for a
  Python call that takes no argument. An explicit Python argument always wins.
- **Runtime diagnostics**, read by the C++ core. They are off by default and have
  no effect on numerics or outputs when unset.

`adc_cpp` also *writes* a few standard (non-`POPS_`) variables to steer the
threading backend; those are listed at the end.

## Codegen and toolchain

These are read inside `python/pops/codegen/toolchain.py` and
`python/pops/codegen/cache.py`. They only affect the out-of-CMake compilation of
the DSL loader `.so`; they change neither the ABI key nor the numerics.

| Variable | Effect | Default | Read in |
|----------|--------|---------|---------|
| `POPS_INCLUDE` | Path to the `adc_cpp` `include/` tree used when compiling a DSL `.so`. Highest priority; checked before deducing the headers from the installed `pops` package or the neighboring repo. | Auto-detected (validity probe: `pops/mesh/storage/multifab.hpp` exists). | `pops_include()` in `python/pops/codegen/toolchain.py` |
| `POPS_CXX` | Forces the C++ compiler used for every DSL `.so` backend. Conscious override of the build compiler that produced `_pops`. | The compiler baked into `_pops`, else `c++`/`g++`/`clang++` from `PATH`. | `_default_cxx()` in `python/pops/codegen/toolchain.py` |
| `POPS_KOKKOS_ROOT` | Kokkos install root used to compile the DSL loader with the same backend as `_pops` (header `Kokkos_Core.hpp` must be present under `include/`). `Kokkos_ROOT` and `KOKKOS_ROOT` are accepted as fallbacks, in that order. Required for the `production`/`aot` backends, which are Kokkos-only. | Unset (no Kokkos root found -> the toolchain raises an explicit error, or warns about a serial fallback). | `_native_kokkos_root()` in `python/pops/codegen/toolchain.py` |
| `POPS_KOKKOS_CXX` | Explicit compiler for the Kokkos DSL loader (for example an `nvcc_wrapper` for a CUDA build). Used instead of the default host compiler. | Unset (falls back to `POPS_CXX`, then the `_pops` build compiler). | `_native_kokkos_compiler()` in `python/pops/codegen/toolchain.py` |
| `POPS_KOKKOS_USE_NVCC_WRAPPER` | Opt-in switch to select the `nvcc_wrapper` found under `<kokkos>/bin`. Truthy values: `1`, `on`, `true`, `yes`, `y`. CUDA selection is never implicit: without this flag (or `POPS_KOKKOS_CXX`) the host compiler is used, so a CPU job on a Kokkos install that ships an `nvcc_wrapper` is not silently broken. | Unset / false (host compiler). | `_native_kokkos_compiler()` in `python/pops/codegen/toolchain.py` |
| `POPS_DSL_OPTFLAGS` | Overrides the optimization flags of the `production`/`aot` DSL `.so` (whitespace-split). Enters the cache key, so changing it produces a distinct cached `.so`. Affects neither the ABI nor portability. | `-O3 -DNDEBUG`. | `_dsl_optflags()` in `python/pops/codegen/cache.py` |
| `POPS_CACHE_DIR` | Directory for the out-of-source `.so` cache written by `m.compile()` when no explicit `so_path` is given. Also recorded in `compiled.inspect()`. | `$XDG_CACHE_HOME/pops/dsl`, else `~/.cache/pops/dsl`. | `pops_cache_dir()` in `python/pops/codegen/cache.py` |

## Codegen logging, dumps, and autotune

These are read by `compile_problem` through the resolver in
`python/pops/codegen/env.py`. They supply DEFAULTS for the codegen step; an
explicit argument to `compile_problem` always wins (for example
`compile_problem(debug=True)` forces keep-generated regardless of
`POPS_KEEP_GENERATED`). Coercion is lenient: an unrecognised value falls back to
the safe default instead of raising. Every one of them is surfaced in
`compiled.inspect()` (criterion #47), so the env state that governed a compile is
inspectable rather than hidden.

| Variable | Effect | Default | Read in |
|----------|--------|---------|---------|
| `POPS_LOG` | Log level for the compile/codegen path: `info` / `1` / `on` -> info traces, `debug` / `verbose` / `2` -> verbose traces, anything falsey / unset -> quiet. `POPS_CODEGEN_LOG` (below) wins over this broader name. | Unset (quiet). | `resolve_log_level()` in `python/pops/codegen/env.py` |
| `POPS_CODEGEN_LOG` | Codegen-specific log level (same level vocabulary as `POPS_LOG`). When both are set, this one wins. | Unset (falls back to `POPS_LOG`, else quiet). | `resolve_log_level()` in `python/pops/codegen/env.py` |
| `POPS_CODEGEN_DIR` | Directory the compiled `.so` (and any kept generated source / IR / C++ dump) is written to. Redirects the out-of-source cache file (keeping its collision-free name); created on demand. An explicit `so_path=` on `compile_problem` bypasses it. | Unset (the `POPS_CACHE_DIR` cache directory). | `compile_problem` in `python/pops/codegen/compile_drivers.py` |
| `POPS_KEEP_GENERATED` | Keep the generated `.cpp` next to the `.so` instead of discarding the temporary build directory. Truthy values: `1`, `on`, `true`, `yes`, `y`. `compile_problem(debug=True)` has the same effect and always wins. | Unset / false (source kept only in a temporary directory). | `compile_problem` in `python/pops/codegen/compile_drivers.py` |
| `POPS_DUMP_IR` | After a successful compile (or a cache hit), dump the serialized Program IR (JSON) into `POPS_CODEGEN_DIR` via `compiled.dump_ir()`. Truthy values as above. | Unset / false (no dump). | `compile_problem` in `python/pops/codegen/compile_drivers.py` |
| `POPS_DUMP_CPP` | After a successful compile (or a cache hit), dump the generated C++ source into `POPS_CODEGEN_DIR` via `compiled.dump_cpp()`. Truthy values as above. | Unset / false (no dump). | `compile_problem` in `python/pops/codegen/compile_drivers.py` |
| `POPS_AUTOTUNE` | Requested autotune level: `off` (default), `basic`, `aggressive`. There is no autotune engine today, so any non-`off` value is an HONEST no-op stub: it is recorded and surfaced in `compiled.inspect()` but changes nothing in the emitted code, and therefore does NOT enter the cache key. If a future tuner ever changes codegen, it must then enter the cache key. | Unset (`off`). | `resolve_autotune()` in `python/pops/codegen/env.py` |

## Debug / unsafe gate

| Variable | Effect | Default | Read in |
|----------|--------|---------|---------|
| `POPS_JIT_BACKDOOR` | A DEBUG / UNSAFE gate. **Disabled by default** and **never enabled implicitly** by any other option. No backdoor behavior is wired today; this variable's only effect is the GUARD: if it is set truthy (`1` / `on` / `true` / `yes` / `y`), `compile_problem` emits a LOUD warning (plus a stderr line) and the flag is surfaced in `compiled.inspect()` and flagged by `pops.doctor()`, so it can never be silently honored. Never set this in production. | Unset / false (disabled). | `jit_backdoor_enabled()` / `CodegenEnv.from_env` in `python/pops/codegen/env.py`; surfaced by `doctor()` in `python/pops/runtime/doctor.py` |

## Python runtime defaults

These are read by `pops.runtime` to default a no-argument Python call. They only
supply a default: an explicit Python argument always overrides the environment,
and an unset or unparseable value falls back to the built-in default (no stricter
rejection than passing the argument directly).

| Variable | Effect | Default | Read in |
|----------|--------|---------|---------|
| `POPS_THREADS` | Default thread count for `pops.set_threads()` called with no argument (a positive integer). An explicit `pops.set_threads(n)` always wins; an unset / non-integer / `< 1` value is ignored. | Unset (falls back to `os.cpu_count()`). | `_threads_from_env()` in `python/pops/runtime/threading.py` |
| `POPS_PROFILE` | Default level for `sim.profile()` called with no argument: `advanced` / `full` -> `Profile.Advanced()`, `off` / `0` / `false` / `no` / `none` -> the call's own default (`Profile.Basic()`), anything else -> `Profile.Basic()`. An explicit `sim.profile(Profile.Advanced())` always wins. Also recorded in `compiled.inspect()`. | Unset (the call's default level). | `Profile.from_env()` in `python/pops/runtime/profile.py` |

## Runtime diagnostics

These are read by the C++ core (header-only). They are inert by default: setting
them changes no output and no numerics, only optional behavior or stderr traces.

| Variable | Effect | Default | Read in |
|----------|--------|---------|---------|
| `POPS_FOREACH_SERIAL_THRESHOLD` | Cell-count threshold under which `for_each_cell` runs serially instead of dispatching a Kokkos parallel loop, so small grids avoid fork/join overhead. Read once (the first call), parsed as a non-negative integer; an unparseable value is ignored. | `4096`. | `pops::detail::foreach_serial_threshold()` in `include/pops/mesh/execution/for_each.hpp` |
| `POPS_TRACE_SOLVE_FIELDS` | When set to any value, enables a diagnostic trace of the `solve_fields` / multigrid V-cycle path: markers are written to stderr with an immediate flush (added to locate the last marker before a device crash). Presence alone enables it; the value is not parsed. | Unset (no trace). | `pops::detail::mg_trace_mark()` in `include/pops/numerics/elliptic/mg/geometric_mg.hpp` and `pops::field_solver::pops_trace_sf()` in `include/pops/runtime/system/system_field_solver.hpp` |

## Standard variables that pops writes

`pops.set_threads(n)` (in `python/pops/runtime/threading.py`) does not define any
`POPS_*` variable; it writes the standard Kokkos / OpenMP knobs so the chosen
backend picks up the thread count:

- `OMP_NUM_THREADS` and `KOKKOS_NUM_THREADS` are both set to `n` (agnostic to
  whether Kokkos was built on the OpenMP or the Threads device);
- `OMP_PROC_BIND` is set to `false` only on macOS (via `setdefault`, so a value
  you already exported wins), to avoid libomp oversubscription warnings on dev
  Macs.

The cache directory resolution also reads the standard `XDG_CACHE_HOME` (see
`POPS_CACHE_DIR` above).

## Test-only knobs

Two further names exist only inside the GPU validation harness
(`python/tests/gpu/gpu_dsl_production_validate.cpp`): `POPS_DSLPROD_SOLVER` and
`POPS_DSLPROD_TRACE`. They are test-only knobs, not part of the public runtime,
and are mentioned here only to avoid confusion with the variables above.
