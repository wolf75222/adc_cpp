"""pops.codegen.module_emit_riemann : Riemann-capability blocks of emit_cpp_brick.

Extracted verbatim from ``pops.codegen.module_codegen.emit_cpp_brick`` so the brick
emitter fits the Spec-4 file-size budget.  Each helper builds the C++ lines for one
OPTIONAL Riemann capability and returns them as a list; ``emit_cpp_brick`` extends its
accumulator with the result.  The emitted text is byte-identical to the inlined form.

Helpers
-------
_emit_hllc           -- m.enable_hllc : contact_speed + hllc_star_state from roles
_emit_roe_roles      -- m.enable_roe : roe_dissipation |A_roe| dU from roles
_emit_roe_provided   -- m.roe_dissipation : user rows via left()/right()
_emit_roe_jacobian   -- m.roe_from_jacobian : d = |A| (UR-UL), A = dF/dU at Uavg
"""

from pops.codegen.cpp_writer import _cpp_roe
from pops.codegen.module_emit_helpers import (
    _codegen_exprs,
    _live_prims,
    _prim_block,
    _roles_for,
)


def _emit_hllc(model, nc):
    """CAPABILITY HLLC (m.enable_hllc, audit wave 3): contact_speed (Toro) + hllc_star_state
    GENERATED from the block ROLES (no literal index: Density/MomentumX/MomentumY
    resolved, Energy optional, any other component advected passively Us[c]=fac*U[c]/r).
    The core (HasHLLCStructure) then applies the generic HLLC algorithm -- including on a
    NON Euler model (isothermal 3-var, moments + passive scalars). Without a call: nothing emitted."""
    out = []
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
    out.append("  // CAPABILITY HLLC generee depuis les ROLES (enable_hllc) : algorithme")
    out.append("  // contact-resolving generique du coeur (HasHLLCStructure), aucun layout fige.")
    out.append("  POPS_HD pops::Real contact_speed(const State& UL, const State& UR, "
               "pops::Real pL, pops::Real pR, pops::Real sL, pops::Real sR, int dir) const {")
    out.append("    const int in_ = dir == 0 ? %d : %d;" % (iX, iY))
    out.append("    const pops::Real rL = UL[%d], rR = UR[%d];" % (iD, iD))
    out.append("    const pops::Real unL = UL[in_] / rL, unR = UR[in_] / rR;")
    out.append("    return (pR - pL + rL * unL * (sL - unL) - rR * unR * (sR - unR)) /")
    out.append("           (rL * (sL - unL) - rR * (sR - unR));")
    out += ["  }", ""]
    out.append("  POPS_HD State hllc_star_state(const State& U, pops::Real p, pops::Real s, "
               "pops::Real sStar, int dir) const {")
    out.append("    const int in_ = dir == 0 ? %d : %d;" % (iX, iY))
    out.append("    const pops::Real r = U[%d];" % iD)
    out.append("    const pops::Real un = U[in_] / r;")
    out.append("    const pops::Real fac = r * (s - un) / (s - sStar);")
    out.append("    State Us{};")
    out.append("    for (int c = 0; c < %d; ++c) Us[c] = fac * (U[c] / r);  "
               "// defaut : advection passive" % nc)
    out.append("    Us[%d] = fac;" % iD)
    out.append("    Us[in_] = fac * sStar;")
    if iE >= 0:
        out.append("    Us[%d] = fac * (U[%d] / r + (sStar - un) * (sStar + p / "
                   "(r * (s - un))));" % (iE, iE))
    out += ["    return Us;", "  }", ""]
    return out


def _emit_roe_roles(model, nc):
    """CAPABILITY ROE (m.enable_roe, audit balance): roe_dissipation = |A_roe| (UR - UL)
    GENERATED from the ROLES. With Energy: exact TRANSCRIPTION of the core canonical Euler
    algebra (numerical_flux.hpp), gamma-1 deduced from p/(E - 1/2 rho |v|^2) -- numerical
    parity expected with the historical path. Without Energy: same decomposition without the
    E line, c = sqrt(p/rho) per side then Roe average (standard generalization). The
    components OUTSIDE the fluid roles are passive scalars carried by the entropy wave
    (tangential line, phi = q/rho). The core (HasRoeDissipation) does F = 1/2(FL+FR) - d/2."""
    out = []
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
    out.append("  // CAPABILITY ROE generee depuis les ROLES (enable_roe) : dissipation")
    out.append("  // |A_roe| dU du coeur generique (HasRoeDissipation), aucun layout fige.")
    out.append("  POPS_HD State roe_dissipation(const State& UL, const pops::Aux&, "
               "const State& UR, const pops::Aux&, int dir) const {")
    out.append("    const int in_ = dir == 0 ? %d : %d;" % (iX, iY))
    out.append("    const int it_ = dir == 0 ? %d : %d;" % (iY, iX))
    out.append("    const pops::Real rL = UL[%d], rR = UR[%d];" % (iD, iD))
    out.append("    const pops::Real unL = UL[in_] / rL, unR = UR[in_] / rR;")
    out.append("    const pops::Real utL = UL[it_] / rL, utR = UR[it_] / rR;")
    out.append("    const pops::Real pL = pressure(UL), pR = pressure(UR);")
    out.append("    const pops::Real sqL = std::sqrt(rL), sqR = std::sqrt(rR), "
               "den = sqL + sqR;")
    out.append("    const pops::Real un = (sqL * unL + sqR * unR) / den;")
    out.append("    const pops::Real ut = (sqL * utL + sqR * utR) / den;")
    out.append("    const pops::Real rho = sqL * sqR;")
    if iE >= 0:
        out.append("    // gaz parfait : H de Roe + gamma-1 deduit (algebre canonique du coeur)")
        out.append("    const pops::Real HL = (UL[%d] + pL) / rL, HR = (UR[%d] + pR) / rR;"
                   % (iE, iE))
        out.append("    const pops::Real H = (sqL * HL + sqR * HR) / den;")
        out.append("    const pops::Real q2 = un * un + ut * ut;")
        out.append("    const pops::Real gm1 = pL / (UL[%d] - pops::Real(0.5) * rL * "
                   "(unL * unL + utL * utL));" % iE)
        out.append("    const pops::Real c2 = gm1 * (H - pops::Real(0.5) * q2);")
        out.append("    const pops::Real c = std::sqrt(c2);")
    else:
        out.append("    // sans Energy : c LOCAL = sqrt(p/rho) par cote, moyenne de Roe")
        out.append("    const pops::Real c = (sqL * std::sqrt(pL / rL) + sqR * "
                   "std::sqrt(pR / rR)) / den;")
        out.append("    const pops::Real c2 = c * c;")
    out.append("    const pops::Real dr = rR - rL, dp = pR - pL, dun = unR - unL, "
               "dut = utR - utL;")
    out.append("    const pops::Real a1 = (dp - rho * c * dun) / (pops::Real(2) * c2);")
    out.append("    const pops::Real a2 = dr - dp / c2;")
    out.append("    const pops::Real a3 = rho * dut;")
    out.append("    const pops::Real a5 = (dp + rho * c * dun) / (pops::Real(2) * c2);")
    out.append("    // correction d'entropie de Harten, MEME politique que le chemin")
    out.append("    // canonique : eps = pops::kRoeEntropyFixFraction * c (0.1, Euler/Roe).")
    out.append("    const pops::Real eps = pops::Real(0.1) * c;")
    out.append("    const pops::Real l1r = un - c, l5r = un + c;")
    out.append("    const pops::Real al1 = (l1r < 0 ? -l1r : l1r) < eps ? "
               "pops::Real(0.5) * (l1r * l1r / eps + eps) : (l1r < 0 ? -l1r : l1r);")
    out.append("    const pops::Real al2 = un < 0 ? -un : un;")
    out.append("    const pops::Real al5 = (l5r < 0 ? -l5r : l5r) < eps ? "
               "pops::Real(0.5) * (l5r * l5r / eps + eps) : (l5r < 0 ? -l5r : l5r);")
    out.append("    State d{};")
    out.append("    d[%d] = al1 * a1 + al2 * a2 + al5 * a5;" % iD)
    out.append("    d[in_] = al1 * a1 * (un - c) + al2 * a2 * un + al5 * a5 * (un + c);")
    out.append("    d[it_] = al1 * a1 * ut + al2 * (a2 * ut + a3) + al5 * a5 * ut;")
    if iE >= 0:
        out.append("    d[%d] = al1 * a1 * (H - un * c) + al2 * (a2 * pops::Real(0.5) * q2 "
                   "+ a3 * ut) + al5 * a5 * (H + un * c);" % iE)
    for cpa in passives:
        out.append("    {  // scalaire passif [%d] : porte par l'onde entropique (phi = q/rho)"
                   % cpa)
        out.append("      const pops::Real fL = UL[%d] / rL, fR = UR[%d] / rR;" % (cpa, cpa))
        out.append("      const pops::Real ft = (sqL * fL + sqR * fR) / den;")
        out.append("      d[%d] = al1 * a1 * ft + al2 * (a2 * ft + rho * (fR - fL)) "
                   "+ al5 * a5 * ft;" % cpa)
        out.append("    }")
    out += ["    return d;", "  }", ""]
    return out


def _emit_roe_provided(model, nc):
    """CAPABILITY ROE PROVIDED (m.roe_dissipation): 'user' counterpart of enable_roe. The d_i
    rows come from the user (their eigenstructure), written with left()/right() of both states.
    We emit the SAME hook roe_dissipation(UL, AL, UR, AR, dir) as the roles path (trait
    HasRoeDissipation, the core does F = 1/2(FL+FR) - 1/2 d). left(e) -> e on the L_ locals
    (computed from UL), right(e) -> R_ locals (from UR). _roe_rows and _roe are exclusive
    (guard at declaration and in check())."""
    out = []
    has_aux = bool(model.aux_names)  # Aux parameters named aL/aR only if some aux exist
    aL = "const pops::Aux& aL" if has_aux else "const pops::Aux&"
    aR = "const pops::Aux& aR" if has_aux else "const pops::Aux&"
    out.append("  // CAPABILITY ROE FOURNIE (m.roe_dissipation) : dissipation d ecrite par")
    out.append("  // l'utilisateur via left()/right() des deux etats ; hook HasRoeDissipation.")
    out.append("  POPS_HD State roe_dissipation(const State& UL, %s, const State& UR, %s, "
               "int dir) const {" % (aL, aR))
    # locals of BOTH states: conservatives, primitives (def with prefix), then aux read.
    for side, U, av in (("L_", "UL", "aL"), ("R_", "UR", "aR")):
        out += ["    const pops::Real %s%s = %s[%d];" % (side, c, U, i)
                for i, c in enumerate(model.cons_names)]
        out += ["    const pops::Real %s%s = %s;" % (side, p, _cpp_roe(e, side))
                for p, e in model.prim_defs.items()]
        if has_aux:
            out += ["    const pops::Real %s%s = %s.%s;" % (side, n, av, n)
                    for n in model.aux_names]
    out.append("    State d{};")
    out.append("    if (dir == 0) {")
    out += ["      d[%d] = %s;" % (i, _cpp_roe(model._roe_rows["x"][i], None)) for i in range(nc)]
    out.append("    } else {")
    out += ["      d[%d] = %s;" % (i, _cpp_roe(model._roe_rows["y"][i], None)) for i in range(nc)]
    out += ["    }", "    return d;", "  }", ""]
    return out


def _emit_roe_jacobian(model, nc, cse):
    """CAPABILITY ROE FROM THE FLUX JACOBIAN (m.roe_from_jacobian): generic moment Roe. The hook
    builds A = dF_dir/dU at Uavg = 1/2(UL+UR) (cons locals bound to the mean, like the
    wave_speeds-from-jacobian path binds them to U), then d = |A| (UR-UL) via pops::roe_abs_apply
    (matrix-sign |A| = A sign(A); for a real-diagonalizable A this is R|Lambda|R^-1 exactly,
    the reference flux_ROE dissipation). On a complex/singular spectrum the kernel returns false
    -> spectral-radius (Rusanov) fallback rho (UR-UL), rho = max(|lmin|,|lmax|) of
    pops::real_eig_minmax(A). Roles-free (no 'p', no Density/Momentum): the generic provider for a
    moment hierarchy. The core (HasRoeDissipation) does F = 1/2(FL+FR) - 1/2 d."""
    out = []
    Jx = model._roe_jacobian["x"]
    Jy = model._roe_jacobian["y"]
    live = _live_prims(model, [e for row in (Jx + Jy) for e in row])
    out.append("  // CAPABILITY ROE depuis la JACOBIENNE (roe_from_jacobian) : d = |A| (UR-UL),")
    out.append("  // A = dF/dU a l'etat moyen Uavg = 1/2(UL+UR) ; |A| via pops::roe_abs_apply")
    out.append("  // (matrix-sign), repli rayon spectral (real_eig_minmax) si complexe/singulier.")
    out.append("  POPS_HD State roe_dissipation(const State& UL, const pops::Aux&, "
               "const State& UR, const pops::Aux&, int dir) const {")
    # conservatives at the ARITHMETIC-MEAN interface state Uavg = 1/2 (UL + UR)
    out += ["    const pops::Real %s = pops::Real(0.5) * (UL[%d] + UR[%d]);" % (c, i, i)
            for i, c in enumerate(model.cons_names)]
    out += _prim_block(model, live)  # live primitives, evaluated at Uavg
    out.append("    pops::Real A[%d][%d];" % (nc, nc))
    out.append("    if (dir == 0) {")
    tlx, cppx = _codegen_exprs(model, [Jx[i][j] for i in range(nc) for j in range(nc)],
                               cse, indent="      ")
    out += tlx
    for i in range(nc):
        out += ["      A[%d][%d] = %s;" % (i, j, cppx[i * nc + j]) for j in range(nc)]
    out.append("    } else {")
    tly, cppy = _codegen_exprs(model, [Jy[i][j] for i in range(nc) for j in range(nc)],
                               cse, indent="      ")
    out += tly
    for i in range(nc):
        out += ["      A[%d][%d] = %s;" % (i, j, cppy[i * nc + j]) for j in range(nc)]
    out.append("    }")
    out.append("    pops::Real dU[%d], out[%d];" % (nc, nc))
    out += ["    dU[%d] = UR[%d] - UL[%d];" % (i, i, i) for i in range(nc)]
    out.append("    State d{};")
    out.append("    if (pops::roe_abs_apply(A, dU, out)) {")
    out += ["      d[%d] = out[%d];" % (i, i) for i in range(nc)]
    out.append("    } else {  // spectre complexe/singulier : repli rayon spectral (Rusanov)")
    out.append("      const pops::EigBounds eb_ = pops::real_eig_minmax(A);")
    out.append("      const pops::Real al_ = eb_.lmin < pops::Real(0) ? -eb_.lmin : eb_.lmin;")
    out.append("      const pops::Real ah_ = eb_.lmax < pops::Real(0) ? -eb_.lmax : eb_.lmax;")
    out.append("      const pops::Real rho_ = al_ > ah_ ? al_ : ah_;")
    out += ["      d[%d] = rho_ * dU[%d];" % (i, i) for i in range(nc)]
    out.append("    }")
    out += ["    return d;", "  }", ""]
    return out
