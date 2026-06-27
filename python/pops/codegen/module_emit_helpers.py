"""pops.codegen.module_emit_helpers : shared codegen-only helpers for the module emitter.

Extracted verbatim from ``pops.codegen.module_codegen`` so the emit logic fits the
Spec-4 file-size budget; the emit functions in ``module_codegen`` and the Riemann
capability helpers in ``module_emit_riemann`` import these.  No circular import: this
module never imports pops.dsl or pops.physics at module level.

Contents
--------
_AUX_BASE_COMPS, _AUX_CANONICAL, _AUX_NAMED_BASE   -- aux channel constants
_CANONICAL_ROLES, _role_of, _roles_for             -- role mirror (dsl.roles_for)
_codegen_exprs, _live_prims, _prim_block, _jac_entries
"""

from pops.codegen.cpp_writer import (
    _cse_emit,
    _count_cons_denoms,
    _recip_rewrite,
)

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
# Codegen-only helpers (used solely by the emit* functions)
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
