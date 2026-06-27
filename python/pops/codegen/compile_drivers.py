"""pops.codegen.compile_drivers : the compiler-invocation + facade layer of the pipeline.

Extracted verbatim from ``pops.codegen.compile`` so the model compile pipeline fits the
Spec-4 file-size budget.  These drivers receive a ``HyperbolicModel`` (or a Program /
Module) and invoke the C++ compiler on the source the ``compile_emit`` emitters produce:
``compile_so`` / ``compile_aot`` / ``compile_native`` (one per backend), the
``compile_or_jit`` mode dispatcher, the ``compile_model`` facade, ``_module_to_model``
(lower a ``pops.model.Module`` to a dsl ``Model``) and ``compile_problem`` (compile a
``pops.time.Program`` into a ``problem.so``).  ``pops.codegen.compile`` re-imports every
name so its public surface is unchanged.

Does NOT import pops.physics at module level to avoid import cycles; the physics facade and
aux helpers are imported lazily inside the functions that need them.
"""

import os
import sys

from pops.codegen.toolchain import (
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
from pops.codegen.cache import (
    _cache_so_path,
    _backend_distinct_so_path,
    _record_so_backend,
    _native_mpi_flags,
    _dsl_optflags,
)
from pops.codegen.abi import _abi_key_python
from pops.codegen.compile_emit import (
    _BACKENDS,
    model_hash,
    emit_cpp_so_source,
    emit_cpp_aot_source,
    emit_cpp_native_loader,
)


# ---------------------------------------------------------------------------
# Compiler runners
# ---------------------------------------------------------------------------

def compile_so(model, so_path, include=None, name=None, cxx=None, std="c++20",
               hoist_reciprocals=False):
    """JIT: generate the FULL MODEL (emit_cpp_so_source) and compile a shared
    library loadable by System.add_dynamic_block (dlopen). Returns so_path.
    """
    import tempfile
    if include is None:
        include = pops_include()
    src = emit_cpp_so_source(model, name=name, hoist_reciprocals=hoist_reciprocals)
    cc = _default_cxx(cxx)
    if not cc:
        raise RuntimeError("compile_so: no C++ compiler found")
    std = _probe_cxx_std(cc, std)
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "model.cpp")
        with open(cpp, "w") as f:
            f.write(src)
        _run_compile([cc, "-shared", "-fPIC", "-std=" + std, "-O2", "-I", include, cpp,
                      "-o", so_path], "backend jit, compile_so")
    return so_path


def compile_aot(model, so_path, include=None, name=None, cxx=None, std="c++20",
                hoist_reciprocals=False):
    """Backend "compile" (AOT): generate the FULL MODEL (emit_cpp_aot_source)
    and compile a .so loadable by System.add_compiled_block. Returns so_path.

    KOKKOS-ONLY: the AOT model includes the pops headers (multifab/for_each),
    which do NOT compile without POPS_HAS_KOKKOS. So we compile the .so WITH
    Kokkos (same flags as the native loader), which also aligns its ABI with
    the _pops module.
    """
    import tempfile
    if include is None:
        include = pops_include()
    src = emit_cpp_aot_source(model, name=name, hoist_reciprocals=hoist_reciprocals)
    if _native_kokkos_root() is None:
        raise RuntimeError(
            "compile_aot: adc_cpp is Kokkos-only -- the AOT model includes the pops headers which "
            "require Kokkos. Point at an installed Kokkos via POPS_KOKKOS_ROOT (or Kokkos_ROOT), e.g. "
            "`export POPS_KOKKOS_ROOT=/path/to/kokkos` (Serial is enough on CPU). "
            "Run `python -c \"import pops; pops.doctor()\"` for a full diagnosis and copy-paste fixes.")
    cc = _native_kokkos_compiler(cxx)
    if not cc:
        raise RuntimeError("compile_aot: no C++ compiler found")
    std = _probe_cxx_std(cc, std)
    kokkos_compile_flags, kokkos_link_flags = _native_kokkos_flags()
    mpi_compile_flags = _native_mpi_flags()
    link_extra = ["-undefined", "dynamic_lookup"] if sys.platform == "darwin" else []
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "model_aot.cpp")
        with open(cpp, "w") as f:
            f.write(src)
        _run_compile([cc, "-shared", "-fPIC", "-std=" + std, *_dsl_optflags(), "-I", include]
                     + kokkos_compile_flags + mpi_compile_flags + link_extra
                     + [cpp, "-o", so_path] + kokkos_link_flags,
                     "backend aot, compile_aot")
    return so_path


def compile_native(model, so_path, include=None, name=None, cxx=None, std="c++23", target="system",
                   hoist_reciprocals=False):
    """Backend "production": generate the NATIVE LOADER (emit_cpp_native_loader)
    and compile it into a .so loadable by System.add_native_block
    (target="system") or AmrSystem.add_native_block (target="amr_system").
    Returns so_path.
    """
    import tempfile
    if include is None:
        include = pops_include()
    sig = _check_headers_match_module(include)
    _warn_kokkos_parity()
    src = emit_cpp_native_loader(model, name=name, target=target,
                                 hoist_reciprocals=hoist_reciprocals)
    cc = _native_kokkos_compiler(cxx)
    if not cc:
        raise RuntimeError("compile_native: no C++ compiler found")
    std = _probe_cxx_std(cc, std)
    kokkos_compile_flags, kokkos_link_flags = _native_kokkos_flags()
    mpi_compile_flags = _native_mpi_flags()
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "model_native.cpp")
        src_eff = ('#define POPS_HEADER_SIG "%s"\n' % sig + src) if sys.platform == "win32" else src
        with open(cpp, "w") as f:
            f.write(src_eff)
        if sys.platform == "win32":
            pops_lib = _pops_import_lib()
            if not pops_lib:
                raise RuntimeError(
                    "compile_native: _pops.lib not found next to the _pops module (required to "
                    "link the DSL .dll; rebuild _pops with POPS_EXPORT_BUILDING_MODULE).")
            cl_flags = (["/nologo", "/LD", "/std:" + std, "/O2", "/DNDEBUG", "/EHsc",
                         "/permissive-", "/Zc:preprocessor", "/DNOMINMAX", "/bigobj"]
                        + kokkos_compile_flags + mpi_compile_flags)
            cmd = ([cc] + cl_flags + ["-I", include, cpp,
                    "/Fe:" + so_path, "/Fo" + tmp + os.sep,
                    "/link"] + kokkos_link_flags + [pops_lib])
        else:
            optflags = _dsl_optflags()
            flags = ["-shared", "-fPIC", "-std=" + std, *optflags,
                     "-DPOPS_HEADER_SIG=\"%s\"" % sig, *kokkos_compile_flags, *mpi_compile_flags]
            if sys.platform == "darwin":
                flags += ["-undefined", "dynamic_lookup"]
            cmd = [cc, *flags, "-I", include, cpp, "-o", so_path, *kokkos_link_flags]
        _run_compile(cmd, "backend production, compile_native")
    return so_path


def compile_or_jit(model, so_path, include=None, mode="jit", name=None, cxx=None, std="c++20",
                   target="system", hoist_reciprocals=False):
    """Unified API selecting the backend by mode:

    - mode="jit"     -> compile_so (IModel, virtual dispatch: host prototyping);
    - mode="compile" -> compile_aot (AOT production path, numerically identical to native);
    - mode="native"  -> compile_native (native zero-copy loader; target consumed here).

    @p target: "system" (default) | "amr_system". ONLY consumed by mode="native".
    """
    if mode == "jit":
        if target != "system":
            raise ValueError("compile_or_jit: target='amr_system' not supported in mode 'jit' "
                             "(the AMR path exists only for mode='native')")
        return compile_so(model, so_path, include, name=name, cxx=cxx, std=std,
                          hoist_reciprocals=hoist_reciprocals)
    if mode == "compile":
        if target != "system":
            raise ValueError("compile_or_jit: target='amr_system' not supported in mode 'compile' "
                             "(the AMR path exists only for mode='native')")
        return compile_aot(model, so_path, include, name=name, cxx=cxx, std=std,
                           hoist_reciprocals=hoist_reciprocals)
    if mode == "native":
        return compile_native(model, so_path, include, name=name, cxx=cxx, std=std, target=target,
                              hoist_reciprocals=hoist_reciprocals)
    raise ValueError("compile_or_jit: mode 'jit' | 'compile' | 'native' (received %r)" % mode)


# ---------------------------------------------------------------------------
# compile_model -- full facade (mirrors HyperbolicModel.compile logic)
# ---------------------------------------------------------------------------

def compile_model(model, so_path=None, include=None, backend="auto", name=None, cxx=None,
                  std=None, require_metadata=False, target="system", hoist_reciprocals=False):
    """Compilation facade by INTENTION: compiles *model* (a ``HyperbolicModel``)
    into a .so via the engine designated by *backend* and returns its path.

    This is the free-function equivalent of ``HyperbolicModel.compile``.
    ``dsl.HyperbolicModel.compile`` is a thin wrapper that calls this.

    @p backend: "prototype" | "aot" | "production" | "auto".
    @p target:  "system" (default) | "amr_system".
    @p require_metadata: if True, requires physical roles AND explicit gamma.
    Returns so_path.
    """
    from pops.codegen.toolchain import resolve_auto_backend

    m = model
    if backend == "auto":
        backend, _auto_reason = resolve_auto_backend(include)
    if backend not in _BACKENDS:
        raise ValueError("compile: backend %r unknown (expected %s + 'auto')"
                         % (backend, sorted(_BACKENDS)))
    if target not in ("system", "amr_system"):
        raise ValueError("compile: target 'system' | 'amr_system' (received %r)" % (target,))
    mode, adder = _BACKENDS[backend]
    if target == "amr_system" and mode != "native":
        raise ValueError("compile: target='amr_system' exists only for backend='production' "
                         "(native AMR path); received backend=%r" % (backend,))
    if std is None:
        std = loader_cxx_std() if mode == "native" else "c++20"
    if include is None:
        include = pops_include()

    # Metadata guard rails (before any cache).
    # _check_require_metadata lives on the HyperbolicModel: call it via the model.
    m._check_require_metadata(require_metadata, backend)

    # Out-of-source CACHE when so_path is omitted.
    if so_path is None:
        kokkos_like = backend in ("production", "aot")
        eff_cxx = _native_kokkos_compiler(cxx) if kokkos_like else _default_cxx(cxx)
        abi_key = _abi_key_python(include, eff_cxx, std)
        cache_backend = (backend + ";" + _native_feature_key()) if kokkos_like else backend
        if hoist_reciprocals:
            cache_backend += ";hoist"
        so_path = _cache_so_path(model_hash(m), abi_key, cache_backend, target, name)
        if os.path.exists(so_path):
            _record_so_backend(so_path, backend)
            return so_path
    else:
        so_path = _backend_distinct_so_path(so_path, backend)

    out_path = compile_or_jit(m, so_path, include, mode=mode, name=name, cxx=cxx, std=std,
                              target=target, hoist_reciprocals=hoist_reciprocals)
    _record_so_backend(out_path, backend)
    return out_path


# ---------------------------------------------------------------------------
# _module_to_model -- lower a pops.model.Module to a dsl.Model
# ---------------------------------------------------------------------------

def _module_to_model(module):
    """Lower a :class:`pops.model.Module` to a :class:`pops.dsl.Model`
    (Spec 2, S2-11), reusing the dsl codegen engine -- a translation, NOT a
    second backend.  The Module's typed operators carry dsl ``Expr`` bodies;
    each is mapped to the dsl method of its kind.

    Imported lazily by compile_problem to avoid a top-level physics import.
    """
    # Import the model facade + aux constants lazily here (called only at
    # compile_problem time, not at import time).
    from pops.physics.facade import Model  # noqa: PLC0415
    from pops.physics.aux import AUX_CANONICAL  # noqa: PLC0415
    states = module.state_spaces()
    if len(states) != 1:
        raise ValueError("compile_problem: a Module must declare exactly one StateSpace to compile "
                         "(got %s)" % sorted(states))
    state = next(iter(states.values()))
    m = Model(module.name)
    _spec_role = {"density": "Density", "momentum_x": "MomentumX", "momentum_y": "MomentumY",
                  "momentum_z": "MomentumZ", "energy": "Energy", "pressure": "Pressure",
                  "velocity_x": "VelocityX", "velocity_y": "VelocityY", "velocity_z": "VelocityZ",
                  "temperature": "Temperature"}
    roles = None
    if state.roles:
        roles = [_spec_role.get(state.roles.get(c)) for c in state.components]
        if all(r is None for r in roles):
            roles = None
    cvars = m.conservative_vars(*state.components, roles=roles)
    m.primitive_vars(*cvars)
    m.conservative_from(list(cvars))
    for p in module.params().values():
        m.param(p.name, p.default, kind="const")
    declared = set()

    def _declare_aux(nm):
        if nm in declared:
            return
        declared.add(nm)
        if nm in AUX_CANONICAL:
            m.aux(nm)
        else:
            m.aux_field(nm)

    for fs in module.field_spaces().values():
        for comp in fs.components:
            _declare_aux(comp)
    for a in module.aux().values():
        _declare_aux(a.name)
    if module._eigenvalues is not None:
        m.eigenvalues(x=module._eigenvalues["x"], y=module._eigenvalues["y"])
    _CODEGEN_KINDS = ("grid_operator", "local_source", "local_linear_operator", "field_operator",
                      "projection")
    n_field_ops = 0
    for op in module.operator_registry():
        body = op.body
        if op.kind in _CODEGEN_KINDS and (body is None or callable(body)):
            raise ValueError(
                "compile_problem: operator %r (%s) has no IR body; a compilable Module operator "
                "needs an expression body (Module.operator(..., expr=...))" % (op.name, op.kind))
        if op.kind == "grid_operator":
            if op.name in ("flux", "flux_default"):
                m.flux(x=body["x"], y=body["y"])
            else:
                m.flux_term(op.name, x=body["x"], y=body["y"])
        elif op.kind == "local_source":
            m.source_term(op.name, body)
        elif op.kind == "local_linear_operator":
            m.linear_source(op.name, body)
        elif op.kind == "field_operator":
            n_field_ops += 1
            if n_field_ops > 1:
                raise ValueError(
                    "compile_problem: a Module currently supports one field_operator (the default "
                    "elliptic solve); multiple solved fields are deferred (operator %r)" % op.name)
            m.elliptic_rhs(body)
        elif op.kind == "local_rate":
            low = op.lowering
            m.rate_operator(op.name, flux=low.get("flux", True),
                            sources=low.get("sources"), fluxes=low.get("fluxes"))
        elif op.kind == "projection":
            m.projection(body)
    return m


# ---------------------------------------------------------------------------
# compile_problem -- compile a pops.time.Program into a problem.so
# ---------------------------------------------------------------------------

def compile_problem(so_path=None, *, model=None, time=None, backend="production", target="system",
                    force=False, cxx=None, include=None, std=None, debug=False, libraries=None):
    """Compile a ``pops.time.Program`` into a ``problem.so`` the runtime loads
    via ``sim.install_program``.

    Lowers the Program IR to C++ (``Program.emit_cpp_program``) and compiles
    it against the pops headers with the SAME Kokkos toolchain as the loaded
    _pops module (``pops_loader_build_flags``), so the ``.so`` is ABI-compatible
    and runs in-process. Returns a ``CompiledProblem`` (``.so_path`` + metadata).

    The physical ``model`` is validated here (fail-loud) and carried on the
    handle, but in this MVP it is added as a normal block
    (``sim.add_equation``) while the Program drives the step via
    ``ProgramContext`` (``ctx.rhs_into`` uses the block RHS); a single combined
    model+program ``.so`` is a later phase. MVP constraints (spec): ``backend``
    must be "production", ``target`` "system". Without an explicit ``so_path``
    the ``.so`` is cached out-of-source keyed by [program source + header
    signature + compiler + std]; ``force=True`` recompiles. ``debug=True`` also
    writes the generated ``.cpp`` next to the ``.so`` for inspection.
    """
    import hashlib
    import tempfile
    from pops.codegen.loader import CompiledProblem

    if backend != "production":
        raise ValueError("compiled time programs require backend='production'")
    if target != "system":
        raise ValueError("compiled time programs currently support target='system' only")

    library_manifests = []
    if libraries:
        # Lazy import to avoid a top-level library chain at import time.
        from pops.codegen.library import read_library_manifest  # type: ignore[attr-defined]
        for lib_obj in libraries:
            library_manifests.append(read_library_manifest(lib_obj))

    # A pure operator-first Module lowers to a dsl.Model via the shared codegen.
    if model is not None:
        try:
            from pops import model as _model_pkg
            if isinstance(model, _model_pkg.Module):
                model = _module_to_model(model)
        except ImportError:
            pass  # pops.model unavailable; carry model as-is

    if time is None or not hasattr(time, "emit_cpp_program"):
        raise ValueError("compile_problem: time must be an pops.time.Program (got %r)" % (time,))
    if model is not None and hasattr(model, "check"):
        model.check()

    src = time.emit_cpp_program(model=model)

    include = include or pops_include()
    sig = pops_header_signature(include)
    cc, cflags, lflags = pops_loader_build_flags(cxx)
    eff_std = _probe_cxx_std(cc, std or loader_cxx_std())
    abi_key = "%s|%s|%s" % (sig, cc, eff_std)

    if so_path is None:
        program_hash = hashlib.sha256(src.encode()).hexdigest()
        so_path = _cache_so_path(program_hash, abi_key, "program-production", target,
                                 getattr(time, "name", "problem"))
        if not force and os.path.isfile(so_path):
            return CompiledProblem(so_path, time, model, abi_key, cc, eff_std,
                                   libraries=library_manifests)

    optflags = _dsl_optflags()
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "problem.cpp")
        with open(cpp, "w") as f:
            f.write(src)
        if debug:
            try:
                with open(os.path.splitext(so_path)[0] + ".cpp", "w") as f:
                    f.write(src)
            except OSError:
                pass
        flags = ["-shared", "-fPIC", "-std=" + eff_std, *optflags,
                 "-DPOPS_HEADER_SIG=\"%s\"" % sig, *cflags]
        cmd = [cc, *flags, "-I", include, cpp, "-o", so_path, *lflags]
        _run_compile(cmd, "compile_problem (backend production)")
    return CompiledProblem(so_path, time, model, abi_key, cc, eff_std,
                           libraries=library_manifests)
