"""pops.codegen.compile_emit : the C++ source-emission layer of the compile pipeline.

Extracted verbatim from ``pops.codegen.compile`` so the model compile pipeline fits the
Spec-4 file-size budget.  This is the LEAF module: the per-backend capability / adder
tables, ``model_hash`` (the stable cache key over a model's formulas), ``adder_for`` (the
System adder name for a backend), and the three ``emit_cpp_*`` source emitters
(JIT / AOT / native-loader).  The brick / metadata emitters in ``module_codegen`` are
imported LAZILY inside each ``emit_cpp_*`` function.  ``pops.codegen.compile`` re-imports
every name so its public surface is unchanged.

The compiler-invocation drivers + the ``compile_problem`` facade live in
``pops.codegen.compile_drivers`` (they consume ``_BACKENDS`` / ``model_hash`` /
``emit_cpp_*`` from here).
"""

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
