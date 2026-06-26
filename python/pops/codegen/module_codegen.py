"""pops.codegen.module_codegen : C++ emitter for HyperbolicModel.

Free functions that receive a ``HyperbolicModel`` instance as their first
argument (``model``) and return C++ source strings.  Extracted from the
``HyperbolicModel`` class in ``pops.dsl`` so the emit logic lives in the
codegen package (no circular import: this module never imports pops.dsl or
pops.physics at module level).

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

from pops.codegen.cpp_writer import (
    _cse_emit,
    _cpp_roe,
    _collect_eig_witnesses,
    _eig_witness_helpers,
    _count_cons_denoms,
    _recip_rewrite,
)
from pops.ir.expr import Const

# --- Aux channel constants (mirrors of pops.dsl module-level constants) -----
# These duplicate the values from pops.dsl intentionally to avoid an import
# cycle.  They MUST stay in sync with AUX_BASE_COMPS / AUX_CANONICAL in
# pops.dsl (which themselves mirror the C++ POPS_AUX_FIELDS table).
_AUX_BASE_COMPS = 3
_AUX_CANONICAL = {"phi": 0, "grad_x": 1, "grad_y": 2, "B_z": 3, "T_e": 4}
_AUX_NAMED_BASE = 5


# ---------------------------------------------------------------------------
# roles_for -- local copy; avoids importing pops.dsl at module level.
# Logic is identical to dsl.roles_for / dsl.role_of / dsl.CANONICAL_ROLES.
# ---------------------------------------------------------------------------
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


def _role_of(name):
    return _CANONICAL_ROLES.get(name, "Custom")


def _roles_for(names, override=None):
    """Roles list parallel to names -- local copy of dsl.roles_for."""
    if override is None:
        return [_role_of(nm) for nm in names]
    if len(override) != len(names):
        raise ValueError("roles: %d roles for %d variables" % (len(override), len(names)))
    return [(r if r is not None else _role_of(nm)) for nm, r in zip(names, override, strict=True)]


# ---------------------------------------------------------------------------
# Codegen-only helpers (used solely by the emit* functions below)
# ---------------------------------------------------------------------------

def _codegen_exprs(model, exprs, cse, real="pops::Real", indent="    "):
    """(CSE local lines, [C++ per expr]). If cse, factor the common subexpressions
    (H, c...) into ``cseK_`` locals ; otherwise inline each expression via to_cpp."""
    if cse:
        return _cse_emit(list(exprs), real, indent)
    return [], [e.to_cpp() for e in exprs]


def _live_prims(model, exprs, seed=()):
    """Names of the primitives transitively referenced by @p exprs (and the @p seed names).
    Closure over prim_defs: a live primitive pulls in its own primitive dependencies.
    Used to emit in a method only the primitives actually used (dead-code elimination):
    the live expressions stay identical, so the values are bit-identical."""
    prim = model.prim_defs
    live = set()
    stack = [n for n in seed if n in prim]
    for e in exprs:
        stack.extend(d for d in e.deps() if d in prim)
    while stack:
        nm = stack.pop()
        if nm in live:
            continue
        live.add(nm)
        stack.extend(d for d in prim[nm].deps() if d in prim)
    return live


def _prim_block(model, live=None, hoist=False):
    """``const pops::Real <prim> = ...;`` lines of a method. @p live (default None = all):
    declares only the live primitives. @p hoist: hoists at the top the reciprocal of the
    recurring conservative denominators (>= 2 uses) and replaces those divisions by
    products (OPT-IN, changes the rounding). Without @p hoist and with live=None, historical output."""
    items = [(p, e) for p, e in model.prim_defs.items() if live is None or p in live]
    if not hoist:
        return ["    const pops::Real %s = %s;" % (p, e.to_cpp()) for p, e in items]
    cons_set = set(model.cons_names)
    counts = {}
    for _, e in items:
        _count_cons_denoms(e, cons_set, counts)
    inv = [n for n in model.cons_names if counts.get(n, 0) >= 2]  # stable cons order
    inv_set = set(inv)
    lines = ["    const pops::Real inv_%s = pops::Real(1) / %s;" % (n, n) for n in inv]
    lines += ["    const pops::Real %s = %s;" % (p, _recip_rewrite(e, inv_set).to_cpp())
              for p, e in items]
    return lines


def _jac_entries(model):
    """Entries (Expr) of the Jacobian sub-blocks of both directions (wave_speeds 'numeric'
    path). Drives the dead-code elimination of max_wave_speed / wave_speeds."""
    ws = model._ws_jacobian
    out = []
    for key in ("x", "y"):
        rows = ws["rows"][key]
        for b in ws["blocks"][key]:
            for gi in b:
                for gj in b:
                    out.append(rows[gi][gj])
    return out


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
# emit_cpp_brick
# ---------------------------------------------------------------------------

def emit_cpp_brick(model, name=None, namespace="pops_generated", cse=True,
                   hoist_reciprocals=False):
    """Generates a C++ BRICK satisfying the pops::HyperbolicModel concept (wrapping : step
    2bis). The produced struct uses StateVec / Aux / POPS_HD / Variables and exposes flux,
    max_wave_speed, to_primitive, to_conservative, conservative_vars, primitive_vars : it can
    therefore enter a CompositeModel and run in the compiled solver.

    Requires set_primitive_state(...) (Prim layout) and set_conservative_from([...]) (to_conservative,
    which the DSL cannot invert on its own). cse=True (default) factors the common
    subexpressions (H, c...) into ``cseK_`` locals. Still to do (see ARCHITECTURE_CIBLE.md sect. 3) :
    Kokkos/CUDA codegen, JIT."""
    if not model.prim_state:
        raise ValueError("emit_cpp_brick : call set_primitive_state(...) first")
    if model.cons_from is None or len(model.cons_from) != model.n_vars:
        raise ValueError("emit_cpp_brick : set_conservative_from([...]) expected (%d expressions)"
                         % model.n_vars)
    if not model._flux:
        raise ValueError("emit_cpp_brick : call set_flux(...) first")
    if len(model._flux.get("x", [])) != model.n_vars or len(model._flux.get("y", [])) != model.n_vars:
        raise ValueError("emit_cpp_brick : flux expected with %d components per direction"
                         % model.n_vars)
    if not model._eig and model._wave_speeds is None and model._ws_jacobian is None:
        raise ValueError("emit_cpp_brick : call set_eigenvalues(...), set_wave_speeds(...) "
                         "or set_wave_speeds_from_jacobian(...) first (source of "
                         "max_wave_speed / CFL)")
    nm = name or (model.name.capitalize() + "Gen")
    nc, npr = model.n_vars, len(model.prim_state)

    def cons_locals():
        return ["    const pops::Real %s = U[%d];" % (c, i) for i, c in enumerate(model.cons_names)]

    def prim_locals(live=None):
        # FILTER on the live primitives (live) + OPT-IN hoist; without live, full output.
        return _prim_block(model, live, hoist_reciprocals)

    def aux_locals():
        return model._aux_locals_lines()  # canonical (a.<n>) + named (a.extra_field(k)), ADC-70

    # Aux parameter named 'a' only if a formula reads an auxiliary field (canonical OR
    # named ; otherwise anonymous, so as not to trigger an unused-parameter warning).
    aux_param = "const Aux& a" if model._reads_aux() else "const Aux&"

    def eig_reduce(cpps, ind):
        # cpps : C++ already generated (possibly CSE) for the eigenvalues. Internal names suffixed
        # '_' : they shadow neither a user variable nor the Aux parameter 'a' (see adversarial review).
        lines = ["%sconst pops::Real lam%d_ = %s;" % (ind, k, c) for k, c in enumerate(cpps)]
        lines.append("%spops::Real mws_ = lam0_ < 0 ? -lam0_ : lam0_;" % ind)
        for k in range(1, len(cpps)):
            lines.append("%s{ const pops::Real cand_ = lam%d_ < 0 ? -lam%d_ : lam%d_;"
                         " if (cand_ > mws_) mws_ = cand_; }" % (ind, k, k, k))
        lines.append("%sreturn mws_;" % ind)
        return lines

    def eig_minmax(cpps, ind):
        # signed wave speeds : smin = smallest, smax = largest eigenvalue (for
        # HLLC / Roe). Same internal names suffixed '_' as eig_reduce.
        lines = ["%sconst pops::Real lam%d_ = %s;" % (ind, k, c) for k, c in enumerate(cpps)]
        lines.append("%ssmin = lam0_; smax = lam0_;" % ind)
        for k in range(1, len(cpps)):
            lines.append("%sif (lam%d_ < smin) smin = lam%d_;" % (ind, k, k))
            lines.append("%sif (lam%d_ > smax) smax = lam%d_;" % (ind, k, k))
        return lines

    def ws_jac_pieces(key):
        # 'numeric' jacobian path : CSE of the NON-ZERO entries of the sub-blocks of
        # direction @p key ; the structural zeros (10 identity rows of a moment system,
        # arbitrary sparsity) are emitted as literals without going through the CSE.
        ws = model._ws_jacobian
        rows = ws["rows"][key]
        entries, zeros = [], []
        for bi, b in enumerate(ws["blocks"][key]):
            for r, gi in enumerate(b):
                for c, gj in enumerate(b):
                    e = rows[gi][gj]
                    if isinstance(e, Const) and e.value == 0.0:
                        zeros.append((bi, r, c))
                    else:
                        entries.append((bi, r, c, e))
        tl, cpps = _codegen_exprs(model, [e for (_, _, _, e) in entries], cse)
        fill = {}
        for (bi, r, c, _), cpp in zip(entries, cpps, strict=True):
            fill.setdefault(bi, []).append((r, c, cpp))
        for (bi, r, c) in zeros:
            fill.setdefault(bi, []).append((r, c, "pops::Real(0)"))
        return tl, fill

    def ws_jac_body(ind, lo, hi, key="x", fill=None):
        # body of the jacobian computation -> extremes (@p lo/@p hi : destination names).
        # eig='fd' : column-wise jacobian by finite differences of the COMPILED flux ;
        # eig='numeric' : fill of the sub-blocks from @p fill. @p key : direction
        # (chooses the block partition).
        ws = model._ws_jacobian
        nv = model.n_vars
        L = []
        if ws["eig"] == "fd":
            L.append("%sconst State F0_ = flux(U, a, dir);" % ind)
            L.append("%spops::Real Jf_[%d][%d];" % (ind, nv, nv))
            L.append("%sconst pops::Real eps_ = pops::Real(1e-6) * (U[0] < 0 ? -U[0] : U[0])"
                     " + pops::Real(1e-30);" % ind)
            L.append("%sfor (int k_ = 0; k_ < %d; ++k_) {" % (ind, nv))
            L.append("%s  State Up_ = U;" % ind)
            L.append("%s  Up_[k_] += eps_;" % ind)
            L.append("%s  const State Fk_ = flux(Up_, a, dir);" % ind)
            L.append("%s  for (int i_ = 0; i_ < %d; ++i_) Jf_[i_][k_] = (Fk_[i_] - F0_[i_])"
                     " / eps_;" % (ind, nv))
            L.append("%s}" % ind)
        for bi, b in enumerate(ws["blocks"][key]):
            nb = len(b)
            L.append("%s{" % ind)
            L.append("%s  pops::Real Jb_[%d][%d];" % (ind, nb, nb))
            if ws["eig"] == "fd":
                for r, gi in enumerate(b):
                    for c, gj in enumerate(b):
                        L.append("%s  Jb_[%d][%d] = Jf_[%d][%d];" % (ind, r, c, gi, gj))
            else:
                for (r, c, cpp) in sorted(fill.get(bi, [])):
                    L.append("%s  Jb_[%d][%d] = %s;" % (ind, r, c, cpp))
            L.append("%s  const pops::EigBounds eb_ = pops::real_eig_minmax(Jb_);" % ind)
            if bi == 0:
                L.append("%s  %s = eb_.lmin; %s = eb_.lmax;" % (ind, lo, hi))
            else:
                L.append("%s  if (eb_.lmin < %s) %s = eb_.lmin;" % (ind, lo, lo))
                L.append("%s  if (eb_.lmax > %s) %s = eb_.lmax;" % (ind, hi, hi))
            L.append("%s}" % ind)
        return L

    cnames = ", ".join('"%s"' % c for c in model.cons_names)
    pnames = ", ".join('"%s"' % p for p in model.prim_state)
    # Physical roles parallel to the names : C++ initializer of pops::VariableSet::roles. Emitted IF at
    # least one component has a recognized role (otherwise empty roles -> brick identical to history,
    # couplings fall back on the fallback indices). The roles let System
    # resolve inter-species couplings by index_of(role) instead of a literal index.
    def roles_init(roles):
        if all(r == "Custom" for r in roles):
            return None  # no useful role : we do not emit the 4th field (strict back-compat)
        return ", ".join("pops::VariableRole::%s" % r for r in roles)

    croles = roles_init(_roles_for(model.cons_names, model.cons_roles))
    proles = roles_init(_roles_for(model.prim_state, model.prim_roles))
    # P7-b : assign the runtime indices BEFORE any to_cpp() (a RuntimeParamRef raises otherwise).
    rt_member = model._runtime_params_member()
    S = [
        "#include <cmath>",  # std::sqrt / std::pow : self-sufficient brick (g++ does not pull cmath)
        "// brique HYPERBOLIQUE generee depuis le modele symbolique '%s' (pops.dsl.emit_cpp_brick)."
        % model.name,
        "// Satisfait pops::HyperbolicModel : flux + max_wave_speed + conversions + descripteurs.",
    ]
    if rt_member:  # RuntimeParams header only if a formula reads a runtime param
        S.append("#include <pops/runtime/config/runtime_params.hpp>")
    # dense_eig.hpp : eigenvalues of dense blocks (exact wave_speeds) OU temoin de VP dans la
    # projection (m.projection + dsl.eig_max_im, ADC-289). Sans l'un ou l'autre : non inclus.
    eig_pairs = _collect_eig_witnesses(model._proj or [])
    if model._ws_jacobian is not None or eig_pairs or model._roe_jacobian is not None:
        S.append("#include <pops/numerics/linalg/dense_eig.hpp>")
    S += [
        "namespace %s {" % namespace,
        "struct %s {" % nm,
        "  using State = pops::StateVec<%d>;" % nc,
        "  using Prim  = pops::StateVec<%d>;" % npr,
        "  using Aux   = pops::Aux;",
        "  static constexpr int n_vars = %d;" % nc,
    ]
    if rt_member:  # member pops::RuntimeParams params{count, {defaults}} (P7-b)
        S.append(rt_member.rstrip("\n"))
    # Foncteurs nommes des temoins de VP (EigWitness) : methodes statiques POPS_HD remplissant
    # M[k][k] + real_eig_minmax, declarees une fois par couple (field, k). Device-clean (ADC-289).
    S += _eig_witness_helpers(eig_pairs)
    # n_aux if a formula (flux / eigenvalues) reads an extra aux field : canonical
    # (B_z...) OR named (aux_field -> kAuxNamedBase + k). Without an extra field -> no n_aux emitted,
    # brick strictly bit-identical to history.
    if model._total_n_aux() > _AUX_BASE_COMPS:
        S.append("  static constexpr int n_aux = %d;" % model._total_n_aux())
    S += [
        "",
        "  POPS_HD State flux(const State& U, %s, int dir) const {" % aux_param,
    ]
    S += cons_locals() + prim_locals(_live_prims(model, model._flux["x"] + model._flux["y"])) \
        + aux_locals()
    ftl, fcpps = _codegen_exprs(model, model._flux["x"] + model._flux["y"], cse)
    S += ftl
    S.append("    State F{};")
    S.append("    if (dir == 0) {")
    S += ["      F[%d] = %s;" % (i, fcpps[i]) for i in range(nc)]
    S.append("    } else {")
    S += ["      F[%d] = %s;" % (i, fcpps[nc + i]) for i in range(nc)]
    S += ["    }", "    return F;", "  }", ""]

    # in 'fd' jacobian mode WITHOUT eigenvalues, max_wave_speed calls flux(U, a, dir) : the
    # Aux parameter must be named even if no formula reads an aux.
    jac_fd = model._ws_jacobian is not None and model._ws_jacobian["eig"] == "fd"
    mws_aux_param = "const Aux& a" if (jac_fd and not model._eig) else aux_param
    S.append("  POPS_HD pops::Real max_wave_speed(const State& U, %s, int dir) const {"
             % mws_aux_param)
    if model._eig:
        mws_drv = model._eig["x"] + model._eig["y"]
    elif model._wave_speeds is not None:
        mws_drv = list(model._wave_speeds["x"]) + list(model._wave_speeds["y"])
    elif model._ws_jacobian["eig"] == "fd":
        mws_drv = []  # fd path: max_wave_speed calls flux(), no direct primitive
    else:
        mws_drv = _jac_entries(model)
    S += cons_locals() + prim_locals(_live_prims(model, mws_drv)) + aux_locals()
    if model._eig:
        # historical source : max(|eigenvalues|), bit-identical.
        nx = len(model._eig["x"])
        etl, ecpps = _codegen_exprs(model, model._eig["x"] + model._eig["y"], cse)
        S += etl
        S.append("    if (dir == 0) {")
        S += eig_reduce(ecpps[:nx], "      ")
        S.append("    } else {")
        S += eig_reduce(ecpps[nx:], "      ")
        S += ["    }", "  }", ""]
    elif model._wave_speeds is not None:
        # WITHOUT eigenvalues : Rusanov / CFL bound derived from the explicit SIGNED wave speeds,
        # max(|smin|, |smax|) -- the pair bounds the spectrum by set_wave_speeds contract.
        ws = model._wave_speeds
        wtl, wcpps = _codegen_exprs(model, list(ws["x"]) + list(ws["y"]), cse)
        S += wtl
        S.append("    if (dir == 0) {")
        S += eig_reduce(wcpps[:2], "      ")
        S.append("    } else {")
        S += eig_reduce(wcpps[2:], "      ")
        S += ["    }", "  }", ""]
    else:
        # WITHOUT eigenvalues : Rusanov / CFL bound = max(|smin|, |smax|) of the jacobian
        # spectrum extremes (same blocks as wave_speeds : Rusanov and HLL share the
        # same truth).
        S.append("    pops::Real lo_ = pops::Real(0), hi_ = pops::Real(0);")
        jac_same_blocks = model._ws_jacobian["blocks"]["x"] == model._ws_jacobian["blocks"]["y"]
        if model._ws_jacobian["eig"] == "fd" and jac_same_blocks:
            S += ws_jac_body("    ", "lo_", "hi_")
        elif model._ws_jacobian["eig"] == "fd":
            S.append("    if (dir == 0) {")
            S += ws_jac_body("      ", "lo_", "hi_", "x")
            S.append("    } else {")
            S += ws_jac_body("      ", "lo_", "hi_", "y")
            S.append("    }")
        else:
            ptx, pty = ws_jac_pieces("x"), ws_jac_pieces("y")
            S.append("    if (dir == 0) {")
            S += ptx[0]
            S += ws_jac_body("      ", "lo_", "hi_", "x", ptx[1])
            S.append("    } else {")
            S += pty[0]
            S += ws_jac_body("      ", "lo_", "hi_", "y", pty[1])
            S.append("    }")
        S.append("    const pops::Real alo_ = lo_ < 0 ? -lo_ : lo_;")
        S.append("    const pops::Real ahi_ = hi_ < 0 ? -hi_ : hi_;")
        S += ["    return alo_ > ahi_ ? alo_ : ahi_;", "  }", ""]

    # pressure : emitted IF a primitive 'p' (pressure) is declared (compressible convention) ;
    # required by the canonical HLLC / Roe fluxes (make_block : requires { m.pressure(s); }).
    # ADC-456: an ARBITRARY-formula override (m.riemann(..., pressure=<expr>) -> set_riemann_hooks)
    # codegen's THAT formula as the pressure(U) body instead of the role-derived primitive 'p'.
    p_form = model._riemann_hook_forms.get("pressure")
    if p_form is not None:
        model._validate_hook_form("pressure", p_form, allow_aux=False)
        S.append("  // pressure(U) hook from an ARBITRARY board formula (m.riemann(pressure=...))")
        S.append("  POPS_HD pops::Real pressure(const State& U) const {")
        S += cons_locals() + prim_locals(_live_prims(model, [p_form]))
        ptl, pcpps = _codegen_exprs(model, [p_form], cse)
        S += ptl
        S += ["    return %s;" % pcpps[0], "  }", ""]
    elif "p" in model.prim_defs:
        S.append("  POPS_HD pops::Real pressure(const State& U) const {")
        S += cons_locals() + prim_locals(_live_prims(model, [], seed=["p"]))
        S += ["    return p;", "  }", ""]

    # SIGNED wave speeds wave_speeds(U, aux, dir, smin, smax) : HLL gate of the core
    # (block_builder.hpp requires { m.wave_speeds(...) }). Two sources, by priority :
    #   1. EXPLICIT pair set_wave_speeds (smin, smax per direction) -- INDEPENDENT of 'p' :
    #      a model without pressure (moments, isothermal...) gets access to riemann='hll' ;
    #   2. historical : min/max of the eigenvalues, emitted ONLY if 'p' is declared
    #      (compressible HLLC / Roe convention, bit-identical to the existing one).
    # Without either of the two (e.g. ExB scalar transport) : nothing emitted, Rusanov alone, unchanged.
    if model._wave_speeds is not None:
        ws = model._wave_speeds
        S.append("  POPS_HD void wave_speeds(const State& U, %s, int dir, pops::Real& smin, "
                 "pops::Real& smax) const {" % aux_param)
        S += cons_locals() \
            + prim_locals(_live_prims(model, list(ws["x"]) + list(ws["y"]))) + aux_locals()
        wtl, wcpps = _codegen_exprs(model, list(ws["x"]) + list(ws["y"]), cse)
        S += wtl
        S.append("    if (dir == 0) {")
        S.append("      smin = %s; smax = %s;" % (wcpps[0], wcpps[1]))
        S.append("    } else {")
        S.append("      smin = %s; smax = %s;" % (wcpps[2], wcpps[3]))
        S += ["    }", "  }", ""]
    elif model._ws_jacobian is not None:
        # EXACT speeds via jacobian eigenvalues (see set_wave_speeds_from_jacobian :
        # 'numeric' = entries as formulas, 'fd' = columns by finite differences of the compiled
        # flux ; extremes per sub-block via pops::real_eig_minmax, safe Gershgorin fallback).
        ws_aux = aux_param if model._ws_jacobian["eig"] != "fd" else "const Aux& a"
        S.append("  POPS_HD void wave_speeds(const State& U, %s, int dir, pops::Real& smin, "
                 "pops::Real& smax) const {" % ws_aux)
        ws_drv = [] if model._ws_jacobian["eig"] == "fd" else _jac_entries(model)
        S += cons_locals() + prim_locals(_live_prims(model, ws_drv)) + aux_locals()
        ws_same_blocks = model._ws_jacobian["blocks"]["x"] == model._ws_jacobian["blocks"]["y"]
        if model._ws_jacobian["eig"] == "fd" and ws_same_blocks:
            S += ws_jac_body("    ", "smin", "smax")
        elif model._ws_jacobian["eig"] == "fd":
            S.append("    if (dir == 0) {")
            S += ws_jac_body("      ", "smin", "smax", "x")
            S.append("    } else {")
            S += ws_jac_body("      ", "smin", "smax", "y")
            S.append("    }")
        else:
            ptx, pty = ws_jac_pieces("x"), ws_jac_pieces("y")
            S.append("    if (dir == 0) {")
            S += ptx[0]
            S += ws_jac_body("      ", "smin", "smax", "x", ptx[1])
            S.append("    } else {")
            S += pty[0]
            S += ws_jac_body("      ", "smin", "smax", "y", pty[1])
            S.append("    }")
        S += ["  }", ""]
    elif "p" in model.prim_defs:
        nx = len(model._eig["x"])
        S.append("  POPS_HD void wave_speeds(const State& U, %s, int dir, pops::Real& smin, "
                 "pops::Real& smax) const {" % aux_param)
        S += cons_locals() \
            + prim_locals(_live_prims(model, model._eig["x"] + model._eig["y"])) + aux_locals()
        wtl, wcpps = _codegen_exprs(model, model._eig["x"] + model._eig["y"], cse)
        S += wtl
        S.append("    if (dir == 0) {")
        S += eig_minmax(wcpps[:nx], "      ")
        S.append("    } else {")
        S += eig_minmax(wcpps[nx:], "      ")
        S += ["    }", "  }", ""]

    # CAPABILITY HLLC (m.enable_hllc, audit wave 3): contact_speed (Toro) + hllc_star_state
    # GENERATED from the block ROLES (no literal index: Density/MomentumX/MomentumY
    # resolved, Energy optional, any other component advected passively Us[c]=fac*U[c]/r).
    # The core (HasHLLCStructure) then applies the generic HLLC algorithm -- including on a
    # NON Euler model (isothermal 3-var, moments + passive scalars). Without a call: nothing emitted.
    if model._hllc:
        roles_l = _roles_for(model.cons_names, model.cons_roles)
        if "p" not in model.prim_defs:
            raise ValueError("enable_hllc: the primitive 'p' (pressure) must be declared "
                             "(m.primitive('p', ...)) -- contact_speed/star state depend on it")
        try:
            iD = roles_l.index("Density")
            iX = roles_l.index("MomentumX")
            iY = roles_l.index("MomentumY")
        except ValueError:
            raise ValueError("enable_hllc: roles Density / MomentumX / MomentumY required "
                             "(declare conservative_vars(..., roles=[...])); current roles %r"
                             % (roles_l,)) from None
        iE = roles_l.index("Energy") if "Energy" in roles_l else -1
        S.append("  // CAPABILITY HLLC generee depuis les ROLES (enable_hllc) : algorithme")
        S.append("  // contact-resolving generique du coeur (HasHLLCStructure), aucun layout fige.")
        S.append("  POPS_HD pops::Real contact_speed(const State& UL, const State& UR, "
                 "pops::Real pL, pops::Real pR, pops::Real sL, pops::Real sR, int dir) const {")
        S.append("    const int in_ = dir == 0 ? %d : %d;" % (iX, iY))
        S.append("    const pops::Real rL = UL[%d], rR = UR[%d];" % (iD, iD))
        S.append("    const pops::Real unL = UL[in_] / rL, unR = UR[in_] / rR;")
        S.append("    return (pR - pL + rL * unL * (sL - unL) - rR * unR * (sR - unR)) /")
        S.append("           (rL * (sL - unL) - rR * (sR - unR));")
        S += ["  }", ""]
        S.append("  POPS_HD State hllc_star_state(const State& U, pops::Real p, pops::Real s, "
                 "pops::Real sStar, int dir) const {")
        S.append("    const int in_ = dir == 0 ? %d : %d;" % (iX, iY))
        S.append("    const pops::Real r = U[%d];" % iD)
        S.append("    const pops::Real un = U[in_] / r;")
        S.append("    const pops::Real fac = r * (s - un) / (s - sStar);")
        S.append("    State Us{};")
        S.append("    for (int c = 0; c < %d; ++c) Us[c] = fac * (U[c] / r);  "
                 "// defaut : advection passive" % nc)
        S.append("    Us[%d] = fac;" % iD)
        S.append("    Us[in_] = fac * sStar;")
        if iE >= 0:
            S.append("    Us[%d] = fac * (U[%d] / r + (sStar - un) * (sStar + p / "
                     "(r * (s - un))));" % (iE, iE))
        S += ["    return Us;", "  }", ""]

    # CAPABILITY ROE (m.enable_roe, audit balance): roe_dissipation = |A_roe| (UR - UL)
    # GENERATED from the ROLES. With Energy: exact TRANSCRIPTION of the core canonical Euler
    # algebra (numerical_flux.hpp), gamma-1 deduced from p/(E - 1/2 rho |v|^2) -- numerical
    # parity expected with the historical path. Without Energy: same decomposition without the
    # E line, c = sqrt(p/rho) per side then Roe average (standard generalization). The
    # components OUTSIDE the fluid roles are passive scalars carried by the entropy wave
    # (tangential line, phi = q/rho). The core (HasRoeDissipation) does F = 1/2(FL+FR) - d/2.
    if model._roe:
        roles_l = _roles_for(model.cons_names, model.cons_roles)
        if "p" not in model.prim_defs:
            raise ValueError("enable_roe: the primitive 'p' (pressure) must be declared "
                             "(m.primitive('p', ...)) -- the Roe linearization depends on it")
        try:
            iD = roles_l.index("Density")
            iX = roles_l.index("MomentumX")
            iY = roles_l.index("MomentumY")
        except ValueError:
            raise ValueError("enable_roe: roles Density / MomentumX / MomentumY required "
                             "(declare conservative_vars(..., roles=[...])); current roles %r"
                             % (roles_l,)) from None
        iE = roles_l.index("Energy") if "Energy" in roles_l else -1
        passives = [c for c in range(nc) if c not in (iD, iX, iY, iE)]
        S.append("  // CAPABILITY ROE generee depuis les ROLES (enable_roe) : dissipation")
        S.append("  // |A_roe| dU du coeur generique (HasRoeDissipation), aucun layout fige.")
        S.append("  POPS_HD State roe_dissipation(const State& UL, const pops::Aux&, "
                 "const State& UR, const pops::Aux&, int dir) const {")
        S.append("    const int in_ = dir == 0 ? %d : %d;" % (iX, iY))
        S.append("    const int it_ = dir == 0 ? %d : %d;" % (iY, iX))
        S.append("    const pops::Real rL = UL[%d], rR = UR[%d];" % (iD, iD))
        S.append("    const pops::Real unL = UL[in_] / rL, unR = UR[in_] / rR;")
        S.append("    const pops::Real utL = UL[it_] / rL, utR = UR[it_] / rR;")
        S.append("    const pops::Real pL = pressure(UL), pR = pressure(UR);")
        S.append("    const pops::Real sqL = std::sqrt(rL), sqR = std::sqrt(rR), "
                 "den = sqL + sqR;")
        S.append("    const pops::Real un = (sqL * unL + sqR * unR) / den;")
        S.append("    const pops::Real ut = (sqL * utL + sqR * utR) / den;")
        S.append("    const pops::Real rho = sqL * sqR;")
        if iE >= 0:
            S.append("    // gaz parfait : H de Roe + gamma-1 deduit (algebre canonique du coeur)")
            S.append("    const pops::Real HL = (UL[%d] + pL) / rL, HR = (UR[%d] + pR) / rR;"
                     % (iE, iE))
            S.append("    const pops::Real H = (sqL * HL + sqR * HR) / den;")
            S.append("    const pops::Real q2 = un * un + ut * ut;")
            S.append("    const pops::Real gm1 = pL / (UL[%d] - pops::Real(0.5) * rL * "
                     "(unL * unL + utL * utL));" % iE)
            S.append("    const pops::Real c2 = gm1 * (H - pops::Real(0.5) * q2);")
            S.append("    const pops::Real c = std::sqrt(c2);")
        else:
            S.append("    // sans Energy : c LOCAL = sqrt(p/rho) par cote, moyenne de Roe")
            S.append("    const pops::Real c = (sqL * std::sqrt(pL / rL) + sqR * "
                     "std::sqrt(pR / rR)) / den;")
            S.append("    const pops::Real c2 = c * c;")
        S.append("    const pops::Real dr = rR - rL, dp = pR - pL, dun = unR - unL, "
                 "dut = utR - utL;")
        S.append("    const pops::Real a1 = (dp - rho * c * dun) / (pops::Real(2) * c2);")
        S.append("    const pops::Real a2 = dr - dp / c2;")
        S.append("    const pops::Real a3 = rho * dut;")
        S.append("    const pops::Real a5 = (dp + rho * c * dun) / (pops::Real(2) * c2);")
        S.append("    // correction d'entropie de Harten, MEME politique que le chemin")
        S.append("    // canonique : eps = pops::kRoeEntropyFixFraction * c (0.1, Euler/Roe).")
        S.append("    const pops::Real eps = pops::Real(0.1) * c;")
        S.append("    const pops::Real l1r = un - c, l5r = un + c;")
        S.append("    const pops::Real al1 = (l1r < 0 ? -l1r : l1r) < eps ? "
                 "pops::Real(0.5) * (l1r * l1r / eps + eps) : (l1r < 0 ? -l1r : l1r);")
        S.append("    const pops::Real al2 = un < 0 ? -un : un;")
        S.append("    const pops::Real al5 = (l5r < 0 ? -l5r : l5r) < eps ? "
                 "pops::Real(0.5) * (l5r * l5r / eps + eps) : (l5r < 0 ? -l5r : l5r);")
        S.append("    State d{};")
        S.append("    d[%d] = al1 * a1 + al2 * a2 + al5 * a5;" % iD)
        S.append("    d[in_] = al1 * a1 * (un - c) + al2 * a2 * un + al5 * a5 * (un + c);")
        S.append("    d[it_] = al1 * a1 * ut + al2 * (a2 * ut + a3) + al5 * a5 * ut;")
        if iE >= 0:
            S.append("    d[%d] = al1 * a1 * (H - un * c) + al2 * (a2 * pops::Real(0.5) * q2 "
                     "+ a3 * ut) + al5 * a5 * (H + un * c);" % iE)
        for cpa in passives:
            S.append("    {  // scalaire passif [%d] : porte par l'onde entropique (phi = q/rho)"
                     % cpa)
            S.append("      const pops::Real fL = UL[%d] / rL, fR = UR[%d] / rR;" % (cpa, cpa))
            S.append("      const pops::Real ft = (sqL * fL + sqR * fR) / den;")
            S.append("      d[%d] = al1 * a1 * ft + al2 * (a2 * ft + rho * (fR - fL)) "
                     "+ al5 * a5 * ft;" % cpa)
            S.append("    }")
        S += ["    return d;", "  }", ""]

    # CAPABILITY ROE PROVIDED (m.roe_dissipation): 'user' counterpart of enable_roe. The d_i
    # rows come from the user (their eigenstructure), written with left()/right() of both states.
    # We emit the SAME hook roe_dissipation(UL, AL, UR, AR, dir) as the roles path (trait
    # HasRoeDissipation, the core does F = 1/2(FL+FR) - 1/2 d). left(e) -> e on the L_ locals
    # (computed from UL), right(e) -> R_ locals (from UR). _roe_rows and _roe are exclusive
    # (guard at declaration and in check()).
    if model._roe_rows is not None:
        has_aux = bool(model.aux_names)  # Aux parameters named aL/aR only if some aux exist
        aL = "const pops::Aux& aL" if has_aux else "const pops::Aux&"
        aR = "const pops::Aux& aR" if has_aux else "const pops::Aux&"
        S.append("  // CAPABILITY ROE FOURNIE (m.roe_dissipation) : dissipation d ecrite par")
        S.append("  // l'utilisateur via left()/right() des deux etats ; hook HasRoeDissipation.")
        S.append("  POPS_HD State roe_dissipation(const State& UL, %s, const State& UR, %s, "
                 "int dir) const {" % (aL, aR))
        # locals of BOTH states: conservatives, primitives (def with prefix), then aux read.
        for side, U, av in (("L_", "UL", "aL"), ("R_", "UR", "aR")):
            S += ["    const pops::Real %s%s = %s[%d];" % (side, c, U, i)
                  for i, c in enumerate(model.cons_names)]
            S += ["    const pops::Real %s%s = %s;" % (side, p, _cpp_roe(e, side))
                  for p, e in model.prim_defs.items()]
            if has_aux:
                S += ["    const pops::Real %s%s = %s.%s;" % (side, n, av, n)
                      for n in model.aux_names]
        S.append("    State d{};")
        S.append("    if (dir == 0) {")
        S += ["      d[%d] = %s;" % (i, _cpp_roe(model._roe_rows["x"][i], None)) for i in range(nc)]
        S.append("    } else {")
        S += ["      d[%d] = %s;" % (i, _cpp_roe(model._roe_rows["y"][i], None)) for i in range(nc)]
        S += ["    }", "    return d;", "  }", ""]

    # CAPABILITY ROE FROM THE FLUX JACOBIAN (m.roe_from_jacobian): generic moment Roe. The hook
    # builds A = dF_dir/dU at Uavg = 1/2(UL+UR) (cons locals bound to the mean, like the
    # wave_speeds-from-jacobian path binds them to U), then d = |A| (UR-UL) via pops::roe_abs_apply
    # (matrix-sign |A| = A sign(A); for a real-diagonalizable A this is R|Lambda|R^-1 exactly,
    # the reference flux_ROE dissipation). On a complex/singular spectrum the kernel returns false
    # -> spectral-radius (Rusanov) fallback rho (UR-UL), rho = max(|lmin|,|lmax|) of
    # pops::real_eig_minmax(A). Roles-free (no 'p', no Density/Momentum): the generic provider for a
    # moment hierarchy. The core (HasRoeDissipation) does F = 1/2(FL+FR) - 1/2 d.
    if model._roe_jacobian is not None:
        Jx = model._roe_jacobian["x"]
        Jy = model._roe_jacobian["y"]
        live = _live_prims(model, [e for row in (Jx + Jy) for e in row])
        S.append("  // CAPABILITY ROE depuis la JACOBIENNE (roe_from_jacobian) : d = |A| (UR-UL),")
        S.append("  // A = dF/dU a l'etat moyen Uavg = 1/2(UL+UR) ; |A| via pops::roe_abs_apply")
        S.append("  // (matrix-sign), repli rayon spectral (real_eig_minmax) si complexe/singulier.")
        S.append("  POPS_HD State roe_dissipation(const State& UL, const pops::Aux&, "
                 "const State& UR, const pops::Aux&, int dir) const {")
        # conservatives at the ARITHMETIC-MEAN interface state Uavg = 1/2 (UL + UR)
        S += ["    const pops::Real %s = pops::Real(0.5) * (UL[%d] + UR[%d]);" % (c, i, i)
              for i, c in enumerate(model.cons_names)]
        S += _prim_block(model, live)  # live primitives, evaluated at Uavg
        S.append("    pops::Real A[%d][%d];" % (nc, nc))
        S.append("    if (dir == 0) {")
        tlx, cppx = _codegen_exprs(model, [Jx[i][j] for i in range(nc) for j in range(nc)],
                                   cse, indent="      ")
        S += tlx
        for i in range(nc):
            S += ["      A[%d][%d] = %s;" % (i, j, cppx[i * nc + j]) for j in range(nc)]
        S.append("    } else {")
        tly, cppy = _codegen_exprs(model, [Jy[i][j] for i in range(nc) for j in range(nc)],
                                   cse, indent="      ")
        S += tly
        for i in range(nc):
            S += ["      A[%d][%d] = %s;" % (i, j, cppy[i * nc + j]) for j in range(nc)]
        S.append("    }")
        S.append("    pops::Real dU[%d], out[%d];" % (nc, nc))
        S += ["    dU[%d] = UR[%d] - UL[%d];" % (i, i, i) for i in range(nc)]
        S.append("    State d{};")
        S.append("    if (pops::roe_abs_apply(A, dU, out)) {")
        S += ["      d[%d] = out[%d];" % (i, i) for i in range(nc)]
        S.append("    } else {  // spectre complexe/singulier : repli rayon spectral (Rusanov)")
        S.append("      const pops::EigBounds eb_ = pops::real_eig_minmax(A);")
        S.append("      const pops::Real al_ = eb_.lmin < pops::Real(0) ? -eb_.lmin : eb_.lmin;")
        S.append("      const pops::Real ah_ = eb_.lmax < pops::Real(0) ? -eb_.lmax : eb_.lmax;")
        S.append("      const pops::Real rho_ = al_ > ah_ ? al_ : ah_;")
        S += ["      d[%d] = rho_ * dU[%d];" % (i, i) for i in range(nc)]
        S.append("    }")
        S += ["    return d;", "  }", ""]

    # OPTIONAL step bounds (m.stability_speed / m.stability_dt): emitted like the C++
    # traits HasStabilitySpeed / HasStabilityDt (cf. pops/core/physical_model.hpp). A single
    # expression (isotropic): dir is ignored. WITHOUT a call, nothing emitted -> strict fallback
    # max_wave_speed (historical step policy).
    if model._stab_speed is not None:
        S.append("  POPS_HD pops::Real stability_speed(const State& U, %s, int dir) const {"
                 % aux_param)
        S.append("    (void)dir;  // borne isotrope : une seule expression pour les deux directions")
        S += cons_locals() + prim_locals(_live_prims(model, [model._stab_speed])) + aux_locals()
        stl, scpps = _codegen_exprs(model, [model._stab_speed], cse)
        S += stl
        S += ["    return %s;" % scpps[0], "  }", ""]
    if model._stab_dt is not None:
        S.append("  POPS_HD pops::Real stability_dt(const State& U, %s) const {" % aux_param)
        S += cons_locals() + prim_locals(_live_prims(model, [model._stab_dt])) + aux_locals()
        dtl, dcpps = _codegen_exprs(model, [model._stab_dt], cse)
        S += dtl
        S += ["    return %s;" % dcpps[0], "  }", ""]

    # PROJECTION PONCTUELLE post-pas (m.projection, ADC-177) : emise comme le trait C++
    # HasPointwiseProjection (project(U, aux) -> State), appliquee par le stepper a la FIN de
    # chaque macro-pas entier. SANS appel, rien d'emis -> aucun hook (chemin bit-identique).
    if model._proj is not None:
        S.append("  POPS_HD State project(const State& U, %s) const {" % aux_param)
        S += cons_locals() + prim_locals() + aux_locals()
        ptl, pcpps = _codegen_exprs(model, model._proj, cse)
        S += ptl
        S.append("    State Up{};")
        S += ["    Up[%d] = %s;" % (i, c) for i, c in enumerate(pcpps)]
        S += ["    return Up;", "  }", ""]

    S.append("  POPS_HD Prim to_primitive(const State& U) const {")
    S += cons_locals() + prim_locals(_live_prims(model, [], seed=model.prim_state))
    S.append("    Prim P{};")
    S += ["    P[%d] = %s;" % (i, p) for i, p in enumerate(model.prim_state)]
    S += ["    return P;", "  }", ""]

    S.append("  POPS_HD State to_conservative(const Prim& P) const {")
    S += ["    const pops::Real %s = P[%d];" % (p, i) for i, p in enumerate(model.prim_state)]
    ctl, ccpps = _codegen_exprs(model, model.cons_from, cse)
    S += ctl
    S.append("    State U{};")
    S += ["    U[%d] = %s;" % (i, c) for i, c in enumerate(ccpps)]
    S += ["    return U;", "  }", ""]

    cons_set = "{pops::VariableKind::Conservative, {%s}, %d%s}" % (
        cnames, nc, (", {%s}" % croles) if croles is not None else "")
    prim_set = "{pops::VariableKind::Primitive, {%s}, %d%s}" % (
        pnames, npr, (", {%s}" % proles) if proles is not None else "")
    S.append('  static pops::VariableSet conservative_vars() { return %s; }' % cons_set)
    S.append('  static pops::VariableSet primitive_vars() { return %s; }' % prim_set)
    S += ["};", "}  // namespace %s" % namespace]
    return "\n".join(S) + "\n"


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
