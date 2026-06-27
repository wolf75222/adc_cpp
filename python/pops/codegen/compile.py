"""pops.codegen.compile : compile / .so-loader layer for HyperbolicModel.

Free functions that receive a ``HyperbolicModel`` instance (or other objects)
as their first argument and drive the C++ compilation pipeline (source
emission -> compiler invocation -> .so on disk).

Does NOT import pops.physics at module level to avoid import cycles; the
physics facade and aux helpers are imported lazily inside the functions that
need them.

Public free functions (model compile/emit layer)
-------------------------------------------------
emit_cpp_so_source(model, ...)       -- JIT .so source (pops::ModelAdapter)
emit_cpp_aot_source(model, ...)      -- AOT .so source (POPS_DEFINE_COMPILED_BLOCK)
emit_cpp_native_loader(model, ...)   -- production native loader source
compile_so(model, so_path, ...)      -- JIT: emit + compile
compile_aot(model, so_path, ...)     -- AOT: emit + compile (Kokkos)
compile_native(model, so_path, ...) -- production: emit + compile (Kokkos)
compile_or_jit(model, so_path, ...) -- unified backend dispatcher
compile_model(model, ...)            -- full facade (HyperbolicModel.compile logic)
model_hash(model, params=None)       -- stable hash of the model formulas
adder_for(backend)                   -- name of the System adder for a backend

Module-level symbols also exported
-----------------------------------
_BACKEND_CAPS    -- per-backend capability table
_BACKENDS        -- per-backend (mode, adder) table
compile_problem  -- compile a pops.time.Program into a problem.so
_module_to_model -- lower a pops.model.Module -> Model (used by compile_problem)

This is the THIN public module of the model compile pipeline.  The lowering
machinery is split across two sibling modules so each file fits the Spec-4 size
budget, and every name re-imported below so the public surface of
``pops.codegen.compile`` is unchanged:

  - ``compile_emit``    -- the backend tables, ``model_hash`` / ``adder_for`` and
                           the three ``emit_cpp_*`` source emitters;
  - ``compile_drivers`` -- the per-backend compiler runners, ``compile_or_jit`` /
                           ``compile_model`` facade, ``_module_to_model`` and
                           ``compile_problem``.

``pops_cache_dir`` is re-exported for callers that read it off this module.
"""

# Re-export every moved name so the public surface of this module is unchanged.
# The original compile.py imported these toolchain/cache/abi helpers at module
# scope, so they were attributes of ``pops.codegen.compile``; preserve that surface.
from pops.codegen.toolchain import (  # noqa: F401
    pops_include,
    loader_cxx_std,
    _default_cxx,
    _probe_cxx_std,
    _check_headers_match_module,
    _warn_kokkos_parity,
    _native_kokkos_root,
    _native_kokkos_compiler,
    _native_kokkos_flags,
    _native_feature_key,
    _run_compile,
    _pops_import_lib,
    pops_header_signature,
    pops_loader_build_flags,
)
from pops.codegen.cache import (  # noqa: F401
    pops_cache_dir,
    _cache_so_path,
    _backend_distinct_so_path,
    _record_so_backend,
    _native_mpi_flags,
    _dsl_optflags,
)
from pops.codegen.abi import _abi_key_python  # noqa: F401
from pops.codegen.compile_emit import (  # noqa: F401
    _BACKEND_CAPS,
    _BACKENDS,
    model_hash,
    adder_for,
    emit_cpp_so_source,
    emit_cpp_aot_source,
    emit_cpp_native_loader,
)
from pops.codegen.compile_drivers import (  # noqa: F401
    compile_so,
    compile_aot,
    compile_native,
    compile_or_jit,
    compile_model,
    _module_to_model,
    compile_problem,
)
