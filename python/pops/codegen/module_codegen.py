"""pops.codegen.module_codegen : C++ emitter for HyperbolicModel.

Free functions that receive a ``HyperbolicModel`` instance as their first
argument (``model``) and return C++ source strings.  Extracted from the
``HyperbolicModel`` class in ``pops.dsl`` so the emit logic lives in the
codegen package (no circular import: this module never imports pops.dsl or
pops.physics at module level).

This is the THIN public module of the emitter.  ``emit_cpp_brick`` (the large
concept brick) lives in ``module_emit_brick`` and its OPTIONAL Riemann
capabilities in ``module_emit_riemann``; the shared codegen helpers and the
aux/role mirror copies live in ``module_emit_helpers``.  Every name that used
to be defined here is re-imported below so the public surface is unchanged.

Public emit functions
---------------------
emit_cpp                  -- standalone flux function (template <class Real>)
emit_cpp_brick            -- full HyperbolicModel concept brick (struct)
emit_cpp_source           -- composable source brick (struct)
emit_cpp_elliptic         -- composable elliptic RHS brick (struct)
emit_cpp_elliptic_field   -- named-field elliptic RHS brick (ADC-428)

Internal helpers (emit-only)
-----------------------------
_codegen_exprs, _live_prims, _prim_block, _jac_entries
_emit_bricks, _elliptic_field_registrations, _emit_metadata
"""

# Re-export the moved helpers + the brick emitter so the public surface of
# ``pops.codegen.module_codegen`` is unchanged (every name resolves here).
from pops.codegen.module_emit_helpers import (  # noqa: F401
    _AUX_BASE_COMPS,
    _AUX_CANONICAL,
    _AUX_NAMED_BASE,
    _CANONICAL_ROLES,
    _codegen_exprs,
    _jac_entries,
    _live_prims,
    _prim_block,
    _role_of,
    _roles_for,
)
from pops.codegen.module_emit_brick import emit_cpp_brick  # noqa: F401


# ---------------------------------------------------------------------------
# emit_cpp
# ---------------------------------------------------------------------------

def emit_cpp(model, func=None, cse=True):
    """Generates a compilable C++ function computing the physical flux from the symbolic
    tree (each Expr node knows how to write itself in C++ via to_cpp).

    Produced signature : template <class Real> void <func>_flux(const Real* U, Real* F, int dir).
    Constants inlined ; each primitive becomes a local variable. cse=True (default) factors
    the common subexpressions (H, c...) into ``cseK_`` locals ; cse=False recomputes them inline.

    Step (2) of the DSL (see docs/ARCHITECTURE_CIBLE.md sect. 3) : HOST C++ (templatable on Real)."""
    name = func or model.name
    if not model._flux:
        raise ValueError("emit_cpp : call set_flux(...) first")
    if len(model._flux.get("x", [])) != model.n_vars or len(model._flux.get("y", [])) != model.n_vars:
        raise ValueError("emit_cpp : flux expected with %d components per direction" % model.n_vars)
    nc = model.n_vars
    out = [
        "// genere depuis le modele symbolique '%s' (pops.dsl.emit_cpp)" % model.name,
        "// flux physique F = flux(U, dir) ; dir 0=x, 1=y ; U et F de taille %d." % nc,
        "#include <cmath>",
        "template <class Real>",
        "inline void %s_flux(const Real* U, Real* F, int dir) {" % name,
    ]
    out += ["  const Real %s = U[%d];" % (c, i) for i, c in enumerate(model.cons_names)]
    out += ["  const Real %s = %s;" % (p, e.to_cpp()) for p, e in model.prim_defs.items()]
    tl, cpps = _codegen_exprs(model, model._flux["x"] + model._flux["y"], cse, real="Real", indent="  ")
    out += tl
    out.append("  if (dir == 0) {")
    out += ["    F[%d] = %s;" % (i, cpps[i]) for i in range(nc)]
    out.append("  } else {")
    out += ["    F[%d] = %s;" % (i, cpps[nc + i]) for i in range(nc)]
    out += ["  }", "}"]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# emit_cpp_source
# ---------------------------------------------------------------------------

def emit_cpp_source(model, name=None, namespace="pops_generated", cse=True,
                    hoist_reciprocals=False):
    """Generate a composable C++ SOURCE BRICK (in the pops sense) from model._source.

    The produced struct exposes apply(U, a) returning the source term S(U, aux), with one line per
    conservative component (S[i] = model._source[i].to_cpp()). It has the same form as the source
    bricks written by hand (NoSource, PotentialForce in pops/model/bricks.hpp) and can therefore
    enter as the Source parameter of a CompositeModel.

    CONVENTION: the auxiliary names (set via aux(...)) must be FIELDS of pops::Aux,
    because they are read directly as a.<name> (e.g. aux('grad_x') -> a.grad_x, aux('grad_y') ->
    a.grad_y). This convention is the same as that of the manual bricks, where the source reads
    the outer state only through the pops::Aux channel (potential and its gradient).

    Style identical to emit_cpp_brick (inlined constants, cons -> locals, primitives -> locals;
    plus, aux -> locals); cse=True factors the common sub-expressions. Raises ValueError if
    set_source(...) has not been called."""
    from pops.ir.expr import _wrap as _ir_wrap
    if model._source is None:
        if model._source_terms:
            raise ValueError("model has multiple named sources; use pops.compile_problem(...) "
                             "or define m.source(...) explicitly")
        raise ValueError("emit_cpp_source: call set_source([...]) first")
    nm = name or (model.name.capitalize() + "Source")
    nc = model.n_vars

    def cons_locals():
        return ["    const pops::Real %s = U[%d];" % (c, i) for i, c in enumerate(model.cons_names)]

    def prim_locals(live=None):
        # FILTER on the live primitives (live) + OPT-IN hoist; without live, full output.
        return _prim_block(model, live, hoist_reciprocals)

    def aux_locals():
        return model._aux_locals_lines()  # canonical (a.<n>) + named (a.extra_field(k)), ADC-70

    na = model._total_n_aux()  # required aux width (B_z / T_e / named fields -> > 3)
    rt_member = model._runtime_params_member()  # P7-b: runtime indices BEFORE any to_cpp()
    S = [
        "#include <cmath>",  # self-sufficient for std::sqrt / std::pow
        "// brique de SOURCE generee depuis le modele symbolique '%s' (pops.dsl.emit_cpp_source)."
        % model.name,
        "// apply(U, a) -> terme source S(U, aux) ; noms aux = champs de pops::Aux (grad_x, grad_y).",
    ]
    if rt_member:  # RuntimeParams header only if a formula reads a runtime param
        S.append("#include <pops/runtime/config/runtime_params.hpp>")
    if model._ws_jacobian is not None:  # dense-block eigenvalues (exact wave_speeds)
        S.append("#include <pops/numerics/linalg/dense_eig.hpp>")
    S += [
        "namespace %s {" % namespace,
        "struct %s {" % nm,
    ]
    if rt_member:  # pops::RuntimeParams params{count, {defaults}} member (P7-b)
        S.append(rt_member.rstrip("\n"))
    # If a formula reads an EXTRA aux field (B_z...), declare n_aux: CompositeModel
    # propagates it (max over the bricks) and the system sizes/populates the shared aux channel.
    # Without an extra field -> no n_aux emitted -> brick strictly identical to the historical one.
    if na > _AUX_BASE_COMPS:
        S.append("  static constexpr int n_aux = %d;" % na)
    S.append("  POPS_HD pops::StateVec<%d> apply(const pops::StateVec<%d>& U, const pops::Aux& a) const {"
             % (nc, nc))
    src_exprs = [_ir_wrap(e) for e in model._source]
    S += cons_locals() + prim_locals(_live_prims(model, src_exprs)) + aux_locals()
    # _wrap: a component may be a Python literal (e.g. 0.0), promoted to Const.
    stl, scpps = _codegen_exprs(model, src_exprs, cse)
    S += stl
    S.append("    pops::StateVec<%d> S{};" % nc)
    S += ["    S[%d] = %s;" % (i, c) for i, c in enumerate(scpps)]
    S += ["    return S;", "  }"]
    # Source FREQUENCY (m.source_frequency, audit wave 2): emitted as frequency(U, a)
    # -- the OPTIONAL contract of the source bricks (cf. physics/source.hpp), forwarded by
    # CompositeModel (HasSourceFrequency) and aggregated by step_cfl. Without a call: nothing emitted.
    if model._src_freq is not None:
        S.append("")
        S.append("  POPS_HD pops::Real frequency(const pops::StateVec<%d>& U, const pops::Aux& a) "
                 "const {" % nc)
        S += cons_locals() + prim_locals(_live_prims(model, [model._src_freq])) + aux_locals()
        ftl, fcpps = _codegen_exprs(model, [model._src_freq], cse)
        S += ftl
        S += ["    return %s;" % fcpps[0], "  }"]
    # ANALYTIC JACOBIAN (m.source_jacobian, audit wave 3): emitted as jacobian(U, a, J),
    # forwarded by CompositeModel (HasSourceJacobian) -> the implicit Newton replaces the
    # finite differences. Without a call: nothing emitted (historical FD, bit-identical).
    if model._src_jac is not None:
        if len(model._src_jac) != nc or any(len(r) != nc for r in model._src_jac):
            raise ValueError("source_jacobian: expected %dx%d matrix (dS_r/dU_c)" % (nc, nc))
        S.append("")
        S.append("  POPS_HD void jacobian(const pops::StateVec<%d>& U, const pops::Aux& a, "
                 "pops::Real (&J)[%d][%d]) const {" % (nc, nc, nc))
        flat = [e for row in model._src_jac for e in row]
        S += cons_locals() + prim_locals(_live_prims(model, flat)) + aux_locals()
        jtl, jcpps = _codegen_exprs(model, flat, cse)
        S += jtl
        for r in range(nc):
            for c in range(nc):
                S.append("    J[%d][%d] = %s;" % (r, c, jcpps[r * nc + c]))
        S += ["  }"]
    S += ["};", "}  // namespace %s" % namespace]
    return "\n".join(S) + "\n"


# ---------------------------------------------------------------------------
# _emit_bricks
# ---------------------------------------------------------------------------

def _emit_bricks(model, name=None, hoist_reciprocals=False):
    """Generate the bricks (hyperbolic + source + elliptic) and the CompositeModel<...> type
    shared by BOTH backends (JIT IModel and AOT). Source / elliptic OPTIONAL: without
    set_source -> pops::NoSource; without set_elliptic_rhs -> zero rhs (no Poisson coupling).
    @p hoist_reciprocals: codegen option propagated to the bricks (cf. emit_cpp_brick).
    Returns (nv, bricks_code, composite_type)."""
    nm = name or (model.name.capitalize() + "Gen")
    nv = model.n_vars
    # CODEGEN guard (not only check(), which compile() does not call): a source
    # frequency or jacobian without m.source(...) would be silently PURGED by the
    # NoSource branch below -- explicit rejection (rule: never an ignored option).
    if model._source is None:
        if model._src_freq is not None:
            raise ValueError("source_frequency(...) declared without source: call "
                             "m.source([...]) (the frequency is emitted on the source brick)")
        if model._src_jac is not None:
            raise ValueError("source_jacobian(...) declared without source: call "
                             "m.source([...]) (the jacobian is emitted on the source brick)")
    parts = [emit_cpp_brick(model, name=nm + "Hyp", hoist_reciprocals=hoist_reciprocals)]
    if model._source is not None:  # source brick generated, otherwise NoSource
        parts.append(emit_cpp_source(model, name=nm + "Src", hoist_reciprocals=hoist_reciprocals))
        src_type = "pops_generated::%sSrc" % nm
    else:
        src_type = "pops::NoSource"
    if model._elliptic is not None:  # elliptic brick generated, otherwise zero rhs (no coupling)
        parts.append(emit_cpp_elliptic(model, name=nm + "Ell", hoist_reciprocals=hoist_reciprocals))
    else:
        parts.append(
            "namespace pops_generated { struct %sEll {\n"
            "  template <class State> POPS_HD pops::Real rhs(const State&) const { return pops::Real(0); }\n"
            "}; }\n" % nm)
    # NAMED elliptic fields (ADC-428): one SELF-CONTAINED brick per m.elliptic_field, paired with
    # make_poisson_rhs by the native loader and routed to a SECOND elliptic solve. Emitted only when
    # the model declares one -> backward-compatible (no named field => no extra struct, byte-identical
    # to the historical brick set).
    for fld in sorted(model._elliptic_fields):
        parts.append(emit_cpp_elliptic_field(
            model, fld, "%sEll_%s" % (nm, fld), hoist_reciprocals=hoist_reciprocals))
    composite = ("pops::CompositeModel<pops_generated::%sHyp, %s, pops_generated::%sEll>"
                 % (nm, src_type, nm))
    return nv, "".join(parts), composite


# ---------------------------------------------------------------------------
# _elliptic_field_registrations
# ---------------------------------------------------------------------------

def _elliptic_field_registrations(model, nm):
    """Per named elliptic field (ADC-428): (field, brick_struct, phi_comp, gx_comp, gy_comp) for the
    native loader. The aux component of each output name is its channel index: a CANONICAL name
    (phi/grad_x/...) maps via AUX_CANONICAL; a model-named aux (aux_field) maps to
    AUX_NAMED_BASE + its position in aux_extra_names. A name the model never declared as an aux is
    rejected (the solve would write a component no source can read). gx/gy default to -1 (phi only)
    when the field lists fewer than 3 aux names."""
    def comp(name):
        if name in _AUX_CANONICAL:
            return _AUX_CANONICAL[name]
        if name in model.aux_extra_names:
            return _AUX_NAMED_BASE + model.aux_extra_names.index(name)
        raise ValueError(
            "elliptic_field: aux output '%s' is not a declared aux field; declare it with "
            "m.aux_field('%s') (so it gets an aux-channel slot a source can read)" % (name, name))
    regs = []
    for fld in sorted(model._elliptic_fields):
        aux = model._elliptic_fields[fld]["aux"]
        phi_c = comp(aux[0])
        gx_c = comp(aux[1]) if len(aux) > 1 else -1
        gy_c = comp(aux[2]) if len(aux) > 2 else -1
        regs.append((fld, "pops_generated::%sEll_%s" % (nm, fld), phi_c, gx_c, gy_c))
    return regs


# ---------------------------------------------------------------------------
# _emit_metadata
# ---------------------------------------------------------------------------

def _emit_metadata(model, model_alias):
    """OPTIONAL metadata symbols of the .so block, read by dlsym on the System side. SHARED by both
    backends (JIT and AOT). The NAMES + ROLES are always emitted (POPS_EXPORT_BLOCK_METADATA):
    they come from the model's VariableSet (single source of truth), the System reads them instead of
    the u0.. fallback / no roles. The GAMMA is emitted (POPS_EXPORT_BLOCK_GAMMA) only if set_gamma(...)
    has been called; otherwise no gamma symbol -> the System keeps its default 1.4 (backward-compat).

    @p model_alias must be an alias WITHOUT a top-level comma (the preprocessor splits
    macro arguments on commas): callers pass a `using ... = CompositeModel<...>`."""
    out = "\nPOPS_EXPORT_BLOCK_METADATA(%s)\n" % model_alias
    if model.gamma is not None:
        out += "POPS_EXPORT_BLOCK_GAMMA(%r)\n" % model.gamma
    # Table of NAMED aux names (aux_field, ADC-70), ordered CSV (order = AUX_NAMED_BASE +
    # k index). OPTIONAL symbol, names/roles pattern: makes the .so SELF-DESCRIBING (a C++ loader
    # could resolve name -> component; on the Python side the table already lives in CompiledModel).
    # Emitted ONLY if the model declares named fields -> backward-compatible (.so without a named
    # field unchanged, symbol absent).
    if model.aux_extra_names:
        # Names = valid C++ identifiers (validated in aux_field) -> CSV without quotes, safe C
        # literal (only [A-Za-z0-9_,]).
        out += ('extern "C" const char* pops_compiled_aux_extra_names() { return "%s"; }\n'
                % ",".join(model.aux_extra_names))
    return out


# ---------------------------------------------------------------------------
# emit_cpp_elliptic
# ---------------------------------------------------------------------------

def emit_cpp_elliptic(model, name=None, namespace="pops_generated", cse=True,
                      hoist_reciprocals=False):
    """Generates a composable elliptic RIGHT-HAND SIDE BRICK from model._elliptic.

    The produced struct exposes rhs(U) -> Real (charge density, background, gravity...), same shape as
    the manual bricks (ChargeDensity, BackgroundDensity in pops/model/bricks.hpp): it enters
    as the Elliptic parameter of a CompositeModel. Inlined constants, cons/primitives -> locals,
    cse=True factors out common sub-expressions. ValueError if set_elliptic_rhs(...) is missing."""
    if model._elliptic is None:
        raise ValueError("emit_cpp_elliptic: call set_elliptic_rhs(...) first")
    nm = name or (model.name.capitalize() + "Elliptic")
    rt_member = model._runtime_params_member()  # P7-b: runtime indices BEFORE any to_cpp()
    out = [
        "#include <cmath>",  # self-sufficient for std::sqrt / std::pow
        "// brique de SECOND MEMBRE elliptique generee depuis '%s' (pops.dsl.emit_cpp_elliptic)."
        % model.name,
        "// rhs(U) -> Real : second membre f(U) de l'operateur elliptique (p.ex. densite de charge).",
    ]
    if rt_member:  # RuntimeParams header only if a formula reads a runtime param
        out.append("#include <pops/runtime/config/runtime_params.hpp>")
    out += [
        "namespace %s {" % namespace,
        "struct %s {" % nm,
    ]
    if rt_member:  # member pops::RuntimeParams params{count, {defaults}} (P7-b)
        out.append(rt_member.rstrip("\n"))
    out += [
        "  template <class State>",
        "  POPS_HD pops::Real rhs(const State& U) const {",
    ]
    out += ["    const pops::Real %s = U[%d];" % (c, i) for i, c in enumerate(model.cons_names)]
    out += _prim_block(model, _live_prims(model, [model._elliptic]), hoist_reciprocals)
    tl, cpps = _codegen_exprs(model, [model._elliptic], cse)
    out += tl
    out += ["    return %s;" % cpps[0], "  }", "};", "}  // namespace %s" % namespace]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# emit_cpp_elliptic_field
# ---------------------------------------------------------------------------

def emit_cpp_elliptic_field(model, field, struct_name, namespace="pops_generated",
                            hoist_reciprocals=False, cse=True):
    """Generates a SELF-CONTAINED elliptic RHS brick for the NAMED field @p field (ADC-428).

    Unlike emit_cpp_elliptic (which emits only ``rhs(U)``, consumed by CompositeModel), this brick
    is shaped like a minimal Model so the runtime can pair it with pops::make_poisson_rhs directly:
    it declares ``n_vars`` + ``State`` (so load_state<Brick> reads the conservative state) and
    exposes ``elliptic_rhs(State)`` (what detail::PoissonRhs<Brick> calls per cell). The native
    loader builds one std::function per named field via make_poisson_rhs(Brick{}) and attaches it to
    the block (System::set_block_elliptic_field). The RHS reads ONLY the conservative state (+
    primitives), never the aux (enforced at declaration). Reuses _codegen_exprs / _prim_block so the
    formula lowers IDENTICALLY to the default elliptic brick."""
    spec = model._elliptic_fields[field]
    rt_member = model._runtime_params_member()  # runtime indices BEFORE any to_cpp()
    out = ["#include <cmath>",
           "#include <pops/numerics/spatial/primitives/state_access.hpp>  // StateVec",
           "// brique de SECOND MEMBRE elliptique NOMMEE '%s' (champ '%s', pops.dsl.elliptic_field)."
           % (struct_name, field)]
    if rt_member:
        out.append("#include <pops/runtime/config/runtime_params.hpp>")
    out += ["namespace %s {" % namespace, "struct %s {" % struct_name,
            "  static constexpr int n_vars = %d;" % model.n_vars,
            "  using State = pops::StateVec<%d>;" % model.n_vars]
    if rt_member:
        out.append(rt_member.rstrip("\n"))
    out += ["  POPS_HD pops::Real elliptic_rhs(const State& U) const {"]
    out += ["    const pops::Real %s = U[%d];" % (c, i) for i, c in enumerate(model.cons_names)]
    out += _prim_block(model, _live_prims(model, [spec["rhs"]]), hoist_reciprocals)
    tl, cpps = _codegen_exprs(model, [spec["rhs"]], cse)
    out += tl
    out += ["    return %s;" % cpps[0], "  }", "};", "}  // namespace %s" % namespace]
    return "\n".join(out) + "\n"
