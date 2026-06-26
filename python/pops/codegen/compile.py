"""pops.codegen.compile : compile / .so-loader layer for HyperbolicModel.

Free functions that receive a ``HyperbolicModel`` instance (or other objects)
as their first argument and drive the C++ compilation pipeline (source
emission -> compiler invocation -> .so on disk).

Does NOT import pops.dsl or pops.physics at module level to avoid import
cycles.  ``pops.dsl`` imports from this module and re-exports each symbol so
that ``dsl.compile_so(...)`` etc. keep working.

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
    pops_cache_dir,
    _cache_so_path,
    _backend_distinct_so_path,
    _record_so_backend,
    _native_mpi_flags,
    _dsl_optflags,
)
from pops.codegen.abi import _abi_key_python


# ---------------------------------------------------------------------------
# Backend / capability tables (single source of truth in this module)
# ---------------------------------------------------------------------------

# HONEST characteristics per backend (cf. DSL_MODEL_DESIGN.md section 5).
# Serves diagnostics and the device/MPI/AMR guard rails.
_BACKEND_CAPS = {
    # backend: (cpu, mpi, amr, gpu)  -- True/False according to what the path SUPPORTS today
    "prototype": {"cpu": True, "mpi": False, "amr": False, "gpu": False},
    "aot": {"cpu": True, "mpi": False, "amr": False, "gpu": False},
    # production = NATIVE path (add_native_block, #85): same engine as add_block, hence MPI-capable
    # by construction (halos fill_boundary). amr=True: the native loader now has an AMR counterpart
    # (m.compile(backend='production', target='amr_system') -> AmrSystem.add_native_block, DSL Phase D)
    # which inlines add_compiled_model(AmrSystem&) -> SAME AMR hierarchy as AmrSystem.add_block (reflux,
    # regrid). gpu=False out of CAUTION: the native path is device-clean in C++ (GH200) but the
    # end-to-end validation from Python (add_native_block on device) is a dedicated PR (DSL sect. 5).
    "production": {"cpu": True, "mpi": True, "amr": True, "gpu": False},
}

# Maps backend name -> (compile mode, System adder method name).
# "prototype"  -> compile_so  (JIT, IModel, virtual dispatch, host first-order Rusanov)
# "aot"        -> compile_aot (AOT, host-marshaled PRODUCTION path)
# "production" -> compile_native (NATIVE LOADER: add_compiled_model<ProdModel>)
_BACKENDS = {
    "prototype": ("jit", "add_dynamic_block"),
    "aot": ("compile", "add_compiled_block"),
    "production": ("native", "add_native_block"),
}


# ---------------------------------------------------------------------------
# model_hash -- stable hash of a HyperbolicModel
# ---------------------------------------------------------------------------

def model_hash(model, params=None):
    """Stable hash of *model* (a ``HyperbolicModel``): formulas
    (flux/eig/source/elliptic/primitives/cons_from) + roles + n_aux + gamma
    (+ any NAMED params). Single source of the hash, reused by
    ``Model._model_hash`` (which passes its Params). Serves to identify/reuse
    an already compiled .so (cache key) and to trace the run. Relies on
    ``repr(Expr)`` (stable, structural); insensitive to dict ordering (sorted).
    """
    import hashlib
    # Import the helper lazily to avoid pulling pops.dsl at import time.
    # aux_total_n_aux and roles_for live in dsl; we read them from the model
    # package which is stdlib-only (no C extension).
    from pops.ir.values import _EIG_FIELDS  # noqa: F401 -- confirm ir is importable

    # --- lazy helpers: resolve at call time, not at import time ---
    def _aux_total_n_aux(aux_names, aux_extra_names):
        # Mirrors pops.dsl.aux_total_n_aux without importing dsl.
        _AUX_CANONICAL = {"phi": 0, "grad_x": 1, "grad_y": 2, "B_z": 3, "T_e": 4}
        _AUX_BASE_COMPS = 3
        _AUX_NAMED_BASE = 5
        w = _AUX_BASE_COMPS
        for nm in aux_names:
            if nm not in _AUX_CANONICAL:
                raise ValueError("unknown aux field %r" % (nm,))
            w = max(w, _AUX_CANONICAL[nm] + 1)
        if aux_extra_names:
            w = max(w, _AUX_NAMED_BASE + len(aux_extra_names))
        return w

    def _role_of(name):
        _CANONICAL_ROLES = {
            "rho": "Density", "n": "Density", "density": "Density",
            "rho_u": "MomentumX", "rhou": "MomentumX", "mom_x": "MomentumX", "mx": "MomentumX",
            "rho_v": "MomentumY", "rhov": "MomentumY", "mom_y": "MomentumY", "my": "MomentumY",
            "rho_w": "MomentumZ", "rhow": "MomentumZ", "mom_z": "MomentumZ", "mz": "MomentumZ",
            "E": "Energy", "rho_E": "Energy", "ener": "Energy", "energy": "Energy",
            "u": "VelocityX", "v": "VelocityY", "w": "VelocityZ",
            "vx": "VelocityX", "vy": "VelocityY", "vz": "VelocityZ",
            "p": "Pressure", "pressure": "Pressure",
            "T": "Temperature", "temperature": "Temperature",
        }
        return _CANONICAL_ROLES.get(name, "Custom")

    def _roles_for(names, override=None):
        if override is None:
            return [_role_of(nm) for nm in names]
        if len(override) != len(names):
            raise ValueError("roles: %d roles for %d variables" % (len(override), len(names)))
        return [(r if r is not None else _role_of(nm)) for nm, r in zip(names, override, strict=True)]

    m = model
    parts = []
    parts.append("name=%s" % m.name)
    parts.append("cons=%s" % ",".join(m.cons_names))
    parts.append("croles=%s" % ",".join(_roles_for(m.cons_names, m.cons_roles)))
    parts.append("prim_state=%s" % ",".join(m.prim_state))
    parts.append("proles=%s" % ",".join(_roles_for(m.prim_state, m.prim_roles)))
    parts.append("prim=%s" % ";".join("%s=%r" % (k, m.prim_defs[k]) for k in m.prim_defs))
    for d in ("x", "y"):
        parts.append("flux_%s=%s" % (d, ";".join(repr(e) for e in m._flux.get(d, []))))
        parts.append("eig_%s=%s" % (d, ";".join(repr(e) for e in m._eig.get(d, []))))
    parts.append("source=%s" % (";".join(repr(e) for e in m._source) if m._source else ""))
    if getattr(m, "_source_terms", None):
        parts.append("source_terms=%s" % ";".join(
            "%s:[%s]" % (k, ",".join(repr(e) for e in m._source_terms[k]))
            for k in sorted(m._source_terms)))
    if getattr(m, "_linear_sources", None):
        parts.append("linear_sources=%s" % ";".join(
            "%s:[%s]" % (k, ";".join(repr(e) for row in m._linear_sources[k] for e in row))
            for k in sorted(m._linear_sources)))
    if getattr(m, "_flux_terms", None):
        parts.append("flux_terms=%s" % ";".join(
            "%s:x[%s]:y[%s]" % (k,
                                ",".join(repr(e) for e in m._flux_terms[k]["x"]),
                                ",".join(repr(e) for e in m._flux_terms[k]["y"]))
            for k in sorted(m._flux_terms)))
    parts.append("cons_from=%s" % (";".join(repr(e) for e in m.cons_from) if m.cons_from else ""))
    parts.append("elliptic=%s" % (repr(m._elliptic) if m._elliptic is not None else ""))
    if getattr(m, "_elliptic_fields", None):
        parts.append("elliptic_fields=%s" % ";".join(
            "%s:%s:%s:[%s]" % (k, m._elliptic_fields[k]["operator"],
                               repr(m._elliptic_fields[k]["rhs"]),
                               ",".join(m._elliptic_fields[k]["aux"]))
            for k in sorted(m._elliptic_fields)))
    parts.append("stab_speed=%s" % (repr(m._stab_speed) if m._stab_speed is not None else ""))
    parts.append("stab_dt=%s" % (repr(m._stab_dt) if m._stab_dt is not None else ""))
    parts.append("src_freq=%s" % (repr(m._src_freq) if m._src_freq is not None else ""))
    parts.append("src_jac=%s" % (";".join(repr(e) for row in m._src_jac for e in row)
                                 if m._src_jac is not None else ""))
    if getattr(m, "_proj", None) is not None:
        parts.append("proj=%s" % ";".join(repr(e) for e in m._proj))
    parts.append("hllc=%d" % (1 if m._hllc else 0))
    forms = getattr(m, "_riemann_hook_forms", None)
    if forms:
        parts.append("riemann_hooks=%s" % ";".join(
            "%s=%r" % (k, forms[k]) for k in sorted(forms)))
    parts.append("roe=%d" % (1 if getattr(m, "_roe", False) else 0))
    if getattr(m, "_roe_rows", None) is not None:
        parts.append("roe_rows=%s" % ";".join(repr(e) for k in ("x", "y")
                                              for e in m._roe_rows[k]))
    if getattr(m, "_roe_jacobian", None) is not None:
        parts.append("roe_jac=%s" % ";".join(repr(e) for k in ("x", "y")
                                             for row in m._roe_jacobian[k] for e in row))
    if getattr(m, "_wave_speeds", None) is not None:
        parts.append("wave_speeds=%s" % ";".join(repr(e) for k in ("x", "y")
                                                 for e in m._wave_speeds[k]))
    if getattr(m, "_ws_jacobian", None) is not None:
        ws = m._ws_jacobian
        parts.append("ws_jac=%s|%s|%s" % (
            ws["eig"],
            "//".join(";".join(",".join(str(i) for i in b) for b in ws["blocks"][k])
                      for k in ("x", "y")),
            ";".join(repr(e) for k in ("x", "y") for row in ws["rows"][k] for e in row)
            if ws["rows"] is not None else ""))
    parts.append("n_aux=%d" % _aux_total_n_aux(m.aux_names, m.aux_extra_names))
    if m.aux_extra_names:
        parts.append("aux_extra=%s" % ",".join(m.aux_extra_names))
    parts.append("gamma=%r" % m.gamma)
    params = params or {}
    parts.append("params=%s" % ";".join("%s=%r:%s" % (k, params[k].value, params[k].kind)
                                         for k in sorted(params)))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


# ---------------------------------------------------------------------------
# adder_for -- System adder name for a backend
# ---------------------------------------------------------------------------

def adder_for(backend):
    """Name of the System method to use to wire the .so produced by
    ``compile_model(backend=...)``:
    ``'add_dynamic_block'`` (prototype/JIT), ``'add_compiled_block'`` (aot) or
    ``'add_native_block'`` (production/native). ValueError if unknown.
    """
    if backend not in _BACKENDS:
        raise ValueError("adder_for: backend %r unknown (expected %s)"
                         % (backend, sorted(_BACKENDS)))
    return _BACKENDS[backend][1]


# ---------------------------------------------------------------------------
# Source emitters (emit_cpp_so_source, emit_cpp_aot_source, emit_cpp_native_loader)
# ---------------------------------------------------------------------------

def emit_cpp_so_source(model, name=None, hoist_reciprocals=False):
    """Source of the JIT library (backend "jit"): the FULL MODEL as
    CompositeModel<GenHyp, GenSrc, GenEll> behind an extern "C" factory
    (pops_model_nvars / pops_make_model / pops_destroy_model via
    pops::ModelAdapter). This is what compile_so compiles and what
    System.add_dynamic_block loads as a coupled block with VIRTUAL DISPATCH
    (host prototyping).
    """
    from pops.codegen.module_codegen import _emit_bricks, _emit_metadata
    m = model
    if m._proj is not None:
        raise ValueError("backend 'prototype' (JIT, IModel) : projection ponctuelle "
                         "(m.projection) non transportee par ce chemin ; utiliser "
                         "backend='aot' ou 'production'")
    if m._elliptic_fields:
        raise NotImplementedError(
            "elliptic_field (named multi-elliptic, ADC-428) on backend='jit' is not supported "
            "yet; the JIT extern-C factory has no hook to register named elliptic fields on the "
            "System. Use backend='production'. Declared: %s" % sorted(m._elliptic_fields))
    nv, bricks, composite = _emit_bricks(m, name, hoist_reciprocals=hoist_reciprocals)
    return ('#include <pops/runtime/dynamic/dynamic_model.hpp>\n'
            '#include <pops/physics/bricks/bricks.hpp>\n'
            '#include <pops/core/state/variables.hpp>\n'
            + bricks
            + '\nnamespace pops_generated { using JitModel = %s; }\n' % composite
            + 'extern "C" int pops_model_nvars() { return %d; }\n' % nv
            + 'extern "C" void* pops_make_model() { return new pops::ModelAdapter<pops_generated::JitModel>(); }\n'
            + 'extern "C" void pops_destroy_model(void* p) { delete static_cast<pops::IModel<%d>*>(p); }\n' % nv
            + _emit_metadata(m, "pops_generated::JitModel"))


def emit_cpp_aot_source(model, name=None, hoist_reciprocals=False):
    """Source of the AOT library (backend "compile"): the FULL MODEL as
    CompositeModel<...> behind the extern "C" ABI of compiled_block_abi.hpp.
    The .so RUNS the PRODUCTION path (assemble_rhs<Limiter, Flux>, the core's
    SSPRK2/IMEX) on the generated model: inlined numerics, identical to a
    native add_block block. As opposed to the "jit" backend (IModel, virtual
    dispatch).
    """
    from pops.codegen.module_codegen import _emit_bricks, _emit_metadata
    m = model
    if m._elliptic_fields:
        raise NotImplementedError(
            "elliptic_field (named multi-elliptic, ADC-428) on backend='aot' is not supported yet; "
            "the flat-ABI compiled block cannot register named elliptic fields on the System. Use "
            "backend='production'. Declared: %s" % sorted(m._elliptic_fields))
    nv, bricks, composite = _emit_bricks(m, name, hoist_reciprocals=hoist_reciprocals)
    return ('#include <pops/runtime/builders/compiled/compiled_block_abi.hpp>\n'
            '#include <pops/physics/bricks/bricks.hpp>\n'
            '#include <pops/core/state/variables.hpp>\n'
            + bricks
            + '\nnamespace pops_generated { using AotModel = %s; }\n' % composite
            + 'POPS_DEFINE_COMPILED_BLOCK(pops_generated::AotModel)\n'
            + _emit_metadata(m, "pops_generated::AotModel"))


def emit_cpp_native_loader(model, name=None, target="system", hoist_reciprocals=False):
    """Source of the NATIVE LOADER (backend "production"): the FULL MODEL as
    CompositeModel<...> behind a THIN extern "C" ABI.

    Unlike the "aot" backend (emit_cpp_aot_source: flat array ABI, where the .so
    recomputes everything on a local grid and marshals the arrays), the native loader
    does NOT carry the numerics: it merely INSTALLS the generated model as a NATIVE
    block of the already-built facade, via the header template
    pops::add_compiled_model<ProdModel>.

    @p target: "system" (default) | "amr_system". Selects the targeted facade and
    thus the add_compiled_model OVERLOAD called.
    """
    from pops.codegen.module_codegen import _emit_bricks, _emit_metadata, _elliptic_field_registrations
    m = model
    if target not in ("system", "amr_system"):
        raise ValueError("emit_cpp_native_loader: target 'system' | 'amr_system' (got %r)"
                         % (target,))
    nv, bricks, composite = _emit_bricks(m, name, hoist_reciprocals=hoist_reciprocals)
    nm = name or (m.name.capitalize() + "Gen")
    ell_field_regs = _elliptic_field_registrations(m, nm)
    head = ('#include <cmath>\n'
            '#include <vector>\n'
            '#include <array>\n'
            '#include <cstddef>\n'
            '#include <string>\n'
            '#include <pops/runtime/dynamic/abi_key.hpp>\n'
            '#include <pops/physics/bricks/bricks.hpp>\n'
            '#include <pops/core/state/variables.hpp>\n')
    head += ('#include <pops/runtime/builders/compiled/dsl_block.hpp>\n' if target == "system"
             else '#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>\n')
    key = ('#if defined(_WIN32)\n'
           '#define POPS_LOADER_API extern "C" __declspec(dllexport)\n'
           '#else\n'
           '#define POPS_LOADER_API extern "C"\n'
           '#endif\n'
           'POPS_LOADER_API const char* pops_native_abi_key() {\n'
           '  return POPS_ABI_KEY_LITERAL;\n'
           '}\n')
    if target == "system":
        ell_field_lines = "".join(
            '  s->register_elliptic_field("%s", %d, %d, %d);\n'
            '  s->set_block_elliptic_field(name, "%s", pops::make_poisson_rhs(%s{}));\n'
            % (fld, phi_c, gx_c, gy_c, fld, brick)
            for (fld, brick, phi_c, gx_c, gy_c) in ell_field_regs)
        install = ('POPS_LOADER_API void pops_install_native(void* sys, const char* name, const char* limiter,\n'
                   '                                    const char* riemann, const char* recon,\n'
                   '                                    const char* time, double gamma, int substeps,\n'
                   '                                    int evolve, int stride, double pos_floor) {\n'
                   '  pops::System* s = reinterpret_cast<pops::System*>(sys);\n'
                   '  pops::add_compiled_model<pops_generated::ProdModel>(*s, name, pops_generated::ProdModel{},\n'
                   '                                                    limiter, riemann, recon, time, gamma,\n'
                   '                                                    substeps, evolve != 0, stride,\n'
                   '                                                    pos_floor);\n'
                   + ell_field_lines +
                   '}\n')
    else:
        if ell_field_regs:
            raise NotImplementedError(
                "elliptic_field (named multi-elliptic, ADC-428) on target='amr_system' is not "
                "supported yet; it is available on target='system' (cartesian). Declared: %s"
                % sorted(f for (f, *_rest) in ell_field_regs))
        install = ('POPS_LOADER_API void pops_install_native_amr(void* sys, const char* name,\n'
                   '                                        const char* limiter, const char* riemann,\n'
                   '                                        const char* recon, const char* time,\n'
                   '                                        double gamma, int substeps, double pos_floor) {\n'
                   '  pops::AmrSystem* s = reinterpret_cast<pops::AmrSystem*>(sys);\n'
                   '  pops::add_compiled_model<pops_generated::ProdModel>(*s, name, pops_generated::ProdModel{},\n'
                   '                                                    limiter, riemann, recon, time, gamma,\n'
                   '                                                    substeps, /*stride=*/1,\n'
                   '                                                    /*implicit_vars=*/{},\n'
                   '                                                    /*implicit_roles=*/{}, pos_floor);\n'
                   '}\n')
    return (head
            + bricks
            + '\nnamespace pops_generated { using ProdModel = %s; }\n' % composite
            + key
            + install
            + _emit_metadata(m, "pops_generated::ProdModel"))


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

    Imported lazily by compile_problem to avoid a top-level pops.dsl import.
    """
    # Import dsl lazily here (called only at compile_problem time, not at import time).
    from pops.dsl import Model, AUX_CANONICAL  # noqa: PLC0415
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
        # Lazy import to avoid a top-level pops.dsl chain at import time.
        from pops.library import read_library_manifest  # type: ignore[attr-defined]
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
