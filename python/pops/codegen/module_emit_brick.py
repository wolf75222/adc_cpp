"""pops.codegen.module_emit_brick : the HyperbolicModel concept brick emitter.

Extracted verbatim from ``pops.codegen.module_codegen`` so the brick emitter fits the
Spec-4 file-size budget.  ``emit_cpp_brick`` lowers a ``HyperbolicModel`` to a C++ struct
satisfying the ``pops::HyperbolicModel`` concept; the OPTIONAL Riemann capabilities
(HLLC / Roe) live in ``module_emit_riemann``.  The emitted text is byte-identical to the
historical single-module form.
"""

from pops.codegen.cpp_writer import (
    _collect_eig_witnesses,
    _eig_witness_helpers,
)
from pops.codegen.module_emit_helpers import (
    _AUX_BASE_COMPS,
    _codegen_exprs,
    _jac_entries,
    _live_prims,
    _prim_block,
    _roles_for,
)
from pops.codegen.module_emit_riemann import (
    _emit_hllc,
    _emit_roe_jacobian,
    _emit_roe_provided,
    _emit_roe_roles,
)
from pops.ir.expr import Const


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

    if model._hllc:
        S += _emit_hllc(model, nc)

    if model._roe:
        S += _emit_roe_roles(model, nc)

    if model._roe_rows is not None:
        S += _emit_roe_provided(model, nc)

    if model._roe_jacobian is not None:
        S += _emit_roe_jacobian(model, nc, cse)

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
