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
| `POPS_CACHE_DIR` | Directory for the out-of-source `.so` cache written by `m.compile()` when no explicit `so_path` is given. | `$XDG_CACHE_HOME/pops/dsl`, else `~/.cache/pops/dsl`. | `pops_cache_dir()` in `python/pops/codegen/cache.py` |

## Python runtime defaults

These are read by `pops.runtime` to default a no-argument Python call. They only
supply a default: an explicit Python argument always overrides the environment,
and an unset or unparseable value falls back to the built-in default (no stricter
rejection than passing the argument directly).

| Variable | Effect | Default | Read in |
|----------|--------|---------|---------|
| `POPS_THREADS` | Default thread count for `pops.set_threads()` called with no argument (a positive integer). An explicit `pops.set_threads(n)` always wins; an unset / non-integer / `< 1` value is ignored. | Unset (falls back to `os.cpu_count()`). | `_threads_from_env()` in `python/pops/runtime/threading.py` |
| `POPS_PROFILE` | Default level for `sim.profile()` called with no argument: `advanced` / `full` -> `Profile.Advanced()`, `off` / `0` / `false` / `no` / `none` -> the call's own default (`Profile.Basic()`), anything else -> `Profile.Basic()`. An explicit `sim.profile(Profile.Advanced())` always wins. | Unset (the call's default level). | `Profile.from_env()` in `python/pops/runtime/profile.py` |

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

## Planned / not yet implemented

The variable names below appear in design discussions but are **not** read by any
current code path. They are listed here so that nobody assumes they work; setting
them today has no effect.

- `POPS_LOG`, `POPS_CODEGEN_LOG` -- no logging-verbosity variable is honored
  today.
- `POPS_CODEGEN_DIR`, `POPS_KEEP_GENERATED` -- the generated `.cpp` is written to
  a temporary path and not retained; there is no env switch to redirect or keep
  it. Use `so_path=` on `m.compile(...)` to control the `.so` output instead.
- `POPS_DUMP_IR`, `POPS_DUMP_CPP` -- no env-gated dump of the IR or the emitted
  C++ exists.
- `POPS_AUTOTUNE` -- there is no environment-driven autotuning hook.
- `POPS_JIT_BACKDOOR` -- absent. No backdoor into the JIT path exists, implicit or
  otherwise. If such a debug-only escape hatch is ever added, it must be explicit
  and opt-in, never triggered implicitly.

Two further names exist only inside the GPU validation harness
(`python/tests/gpu/gpu_dsl_production_validate.cpp`): `POPS_DSLPROD_SOLVER` and
`POPS_DSLPROD_TRACE`. They are test-only knobs, not part of the public runtime,
and are mentioned here only to avoid confusion with the variables above.
