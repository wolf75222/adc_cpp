"""Generic generator of 2D moment models (Vlasov hierarchies / QMOM methods).

For a moment system of order <= N in 2D, the WHOLE chain M -> C -> S -> closure -> C' ->
M' is systematic binomial algebra EXCEPT the closure (the physics). This module derives
that algebra IN LOOPS over the DSL AST (adc.dsl): the equivalent of a symbolic toolbox
(MATLAB Symbolic / SymPy) generating the fluxes offline, but replayed at EVERY model
construction from the single source -- change the closure and the fluxes, the Jacobian
(autodiff) and the wave speeds follow, with no manual re-generation step and no risk of
desynchronization.

The user supplies only:
  - the order N of the transported moments (N=2 -> 6 variables, N=4 -> 15 variables);
  - the closure: a callable S -> dict of the STANDARDIZED moments of order N+1
    (keys 'S{p}{q}'), the only physics of the model. `gaussian_closure(order)` is provided
    (Levermore closure: Gaussian cumulants, generic in the order);
  - optionally sources (`lorentz_sources`: Vlasov-Lorentz hierarchy, generic);
  - optionally a BGK collision (`maxwellian_moments` / `bgk_source`: relax the hierarchy
    toward the local Maxwellian, generic in the order, closure-free).

Variable ordering convention (IDENTICAL to the hyqmom15 case of adc_cases for N=4):
q outer increasing then p increasing, i.e. [M00..M40, M01..M31, M02..M22, M03 M13, M04].

No named physics here: closures and parameters stay on the user side (the generic-core
doctrine). Cross-validation at the reference user (adc_cases hyqmom15): the generated flux
== the MATLAB goldens (Flux_closure15_2D.m) to 7.7e-13 and == the hand-written model to
2.6e-13 on the 10 golden states.
"""
from math import comb

from . import dsl


def moment_indices(order):
    """Canonical list of (p, q) with p + q <= order: q outer, p inner, increasing."""
    if order < 1:
        raise ValueError("moments: order >= 1 required (order %r)" % (order,))
    return [(p, q) for q in range(order + 1) for p in range(order + 1 - q)]


def moment_names(order):
    """Canonical names 'M{p}{q}' aligned with moment_indices(order)."""
    return ["M%d%d" % pq for pq in moment_indices(order)]


def _pow(e, k):
    """e**k by repeated multiplication (k >= 0; e a DSL Expr or a number)."""
    if k == 0:
        return 1.0
    r = e
    for _ in range(k - 1):
        r = r * e
    return r


def _is_zero(e):
    # NUMERIC zero (int/float) or SYMBOLIC zero (dsl.Const(0.0)): a closure may return
    # either one; both drop the term from the generated flux (dead primitive not emitted).
    if isinstance(e, (int, float)):
        return float(e) == 0.0
    return isinstance(e, dsl.Const) and e.value == 0.0


def gaussian_closure(order):
    """Generic Gaussian (Levermore) closure: the standardized moments of order order+1 are
    those of a standardized Gaussian with correlation s11 = S['S11'].

    ODD order+1: all zero (the odd central moments of a Gaussian vanish). EVEN order+1: the
    standardized Stein recurrence m_pq = (p-1) m_{p-2,q} + q s11 m_{p-1,q-1}.
    @return callable S -> dict 'S{p}{q}' (p+q = order+1)."""
    top = order + 1

    def closure(S):
        if top % 2 == 1:
            return {"S%d%d" % (p, top - p): 0.0 for p in range(top + 1)}
        s11 = S["S11"]
        memo = {(0, 0): 1.0, (1, 0): 0.0, (0, 1): 0.0}

        def m(p, q):
            if p < 0 or q < 0:
                return 0.0
            if (p, q) not in memo:
                if p >= 1:
                    memo[(p, q)] = (p - 1) * m(p - 2, q) + (q * s11 * m(p - 1, q - 1)
                                                            if q >= 1 else 0.0)
                else:
                    memo[(p, q)] = (q - 1) * m(p, q - 2)
            return memo[(p, q)]

        return {"S%d%d" % (p, top - p): m(p, top - p) for p in range(top + 1)}

    return closure


def lorentz_sources(M, ex, ey, q_over_m, omega_c):
    """Sources of the moment hierarchy under the Lorentz force (Vlasov), generic in the
    order and INDEPENDENT of the closure (no higher-order moment referenced: the electric
    term LOWERS the order, the magnetic term CONSERVES it):

        S[M_pq] = q_over_m (p ex M_{p-1,q} + q ey M_{p,q-1}) + omega_c (p M_{p-1,q+1} - q M_{p+1,q-1})

    @p M: dict (p, q) -> Expr/value of the transported moments (keys = moment_indices).
    @p ex, ey: electric field (aux Expr or values). @p q_over_m, omega_c: param Expr or
    values. @return list aligned with moment_indices(order). Accepts plain numbers
    everywhere (usable as a numeric oracle)."""
    order = max(p + q for (p, q) in M)
    out = []
    for (p, q) in moment_indices(order):
        expr = None
        if p >= 1:
            t = q_over_m * (float(p) * ex * M[(p - 1, q)])
            expr = t if expr is None else expr + t
            t = omega_c * (float(p) * M[(p - 1, q + 1)])
            expr = expr + t
        if q >= 1:
            t = q_over_m * (float(q) * ey * M[(p, q - 1)])
            expr = t if expr is None else expr + t
            t = omega_c * (-float(q) * M[(p + 1, q - 1)])
            expr = expr + t
        out.append(0.0 if expr is None else expr)
    return out


def maxwellian_moments(M):
    """Raw moments of the LOCAL Maxwellian (Gaussian in velocity) matching the lower moments
    of M: density M00, mean (u, v) = M10/M00, M01/M00, and covariance [[C20, C11], [C11, C02]]
    from the second central moments. The Maxwellian is its own closure, so this is INDEPENDENT
    of the model closure.

    All odd central moments of a Gaussian vanish; the even ones follow Isserlis (Wick):
    C40 = 3 C20^2, C22 = C20 C02 + 2 C11^2, C04 = 3 C02^2, C31 = 3 C20 C11, C13 = 3 C02 C11,
    and every order-3 and order-5 central moment is 0. The Gaussian central moments are
    tabulated up to order 4, so this supports moment hierarchies up to order 4 (6, 10 or 15
    variables); an order-6-and-higher even central moment is not tabulated.

    @p M: dict (p, q) -> Expr/value of the transported moments (keys = moment_indices(order));
       the order is inferred as max(p + q) and must be at most 4. Accepts plain numbers
       (usable as a numeric oracle).
    @return list aligned with moment_indices(order): the equilibrium raw moments M_eq[p, q].
    """
    order = max(p + q for (p, q) in M)
    M00 = M[(0, 0)]
    u = M[(1, 0)] / M00
    v = M[(0, 1)] / M00
    # second central moments of M -> covariance of the matched Gaussian.
    C20 = M[(2, 0)] / M00 - u * u
    C11 = M[(1, 1)] / M00 - u * v
    C02 = M[(0, 2)] / M00 - v * v
    # Gaussian central moments up to order 4 (Isserlis); everything else (odd, incl. order 5) = 0.
    cg = {(0, 0): 1.0, (1, 0): 0.0, (0, 1): 0.0,
          (2, 0): C20, (1, 1): C11, (0, 2): C02,
          (3, 0): 0.0, (2, 1): 0.0, (1, 2): 0.0, (0, 3): 0.0,
          (4, 0): 3.0 * C20 * C20, (3, 1): 3.0 * C20 * C11,
          (2, 2): C20 * C02 + 2.0 * C11 * C11,
          (1, 3): 3.0 * C02 * C11, (0, 4): 3.0 * C02 * C02}
    out = []
    for (p, q) in moment_indices(order):
        # de-standardization / reconstruction: M_eq[p, q] = M00 * sum_ij C(p,i) C(q,j)
        # u^(p-i) v^(q-j) Cg(i, j); a numeric-zero Cg term drops out of the generated flux.
        acc = None
        for i in range(p + 1):
            for j in range(q + 1):
                cij = cg.get((i, j), 0.0)
                if isinstance(cij, (int, float)) and cij == 0.0:
                    continue
                t = float(comb(p, i) * comb(q, j)) * _pow(u, p - i) * _pow(v, q - j)
                if not (isinstance(cij, float) and cij == 1.0):
                    t = t * cij
                acc = t if acc is None else acc + t
        out.append(M00 * acc)
    return out


def bgk_source(M, nu):
    """BGK relaxation source S[M_pq] = nu (M_eq[p, q] - M[p, q]) toward the local Maxwellian.

    @p M: dict (p, q) -> Expr/value of the transported (conservative) moments.
    @p nu: collision frequency (Expr or value).
    @return list aligned with moment_indices(order). The collisional invariants M00, M10, M01
       are exact equilibria (M_eq == M there), so those rows are identically 0 (no term emitted)
       and mass and momentum are conserved by construction. Accepts plain numbers everywhere
       (usable as a numeric oracle).
    """
    meq = maxwellian_moments(M)
    out = []
    for k, (p, q) in enumerate(moment_indices(max(p + q for (p, q) in M))):
        if (p, q) in ((0, 0), (1, 0), (0, 1)):
            out.append(0.0)  # collisional invariant: M_eq == M, exact, no term emitted.
        else:
            out.append(nu * (meq[k] - M[(p, q)]))
    return out


def build_moment_model(name, order, closure, blocks=None, exact_speeds=True,
                       robust=False, eps_m00=1e-12, eps_cov=1e-12, sources=None, roe=False):
    """2D moment model with an arbitrary closure: flux and intermediates GENERATED.

    @p order: max order of the transported moments (order=2 -> 6 variables, order=4 -> 15).
    @p closure: callable S -> dict 'S{p}{q}' of the standardized moments of order order+1
       (ALL keys p+q = order+1 required; values DSL Expr or numbers -- a numeric zero
       removes the term from the generated flux). S holds the let-bound standardized moments
       for 2 <= p+q <= order, with S20 = S02 = 1.0 exact (standardization identities).
    @p blocks: block structure of the Jacobian for the eigenvalue solve (pass-through to
       m.wave_speeds_from_jacobian; default full matrix). Ignored if exact_speeds=False.
    @p exact_speeds: True = exact wave speeds by autodiff of the flux + per-cell numeric
       eigenvalues (faithful riemann='hll'). False = the caller sets m.eigenvalues /
       m.wave_speeds itself (e.g. a bring-up bound).
    @p robust: True = smooth floors max(x, eps) = ((x+eps)+|x-eps|)/2 on M00 (division) and
       C20/C02 (sqrt) -- differentiable (diff(Abs)), so compatible with exact_speeds. False =
       the bare path, faithful to the guard-free references (may produce NaN on a degenerate
       state).
    @p sources: callable (m, M) -> list of Expr (aligned with moment_indices), wired through
       m.source; M = dict (p, q) -> conservative variable. See lorentz_sources.
    @p roe: True = also emit the generic Roe dissipation (m.roe_from_jacobian): the FULL flux
       Jacobian at the arithmetic-mean interface state is eigendecomposed (|A| via the matrix-sign
       kernel adc::roe_abs_apply, spectral-radius Rusanov fallback), making riemann='roe' available
       for the moment system (no fluid roles / pressure needed). Additive to exact_speeds (which
       still provides max_wave_speed for the CFL dt). Needs the 'aot' or 'production' backend.
    @return adc.dsl.Model ready to compile (the caller may still add elliptic_rhs, params,
       aux... before m.compile)."""
    if order < 2:
        raise ValueError("build_moment_model: order >= 2 required (standardization relies "
                         "on C20/C02; order %r)" % (order,))
    idx = moment_indices(order)
    m = dsl.Model(name)
    cons = m.conservative_vars(*moment_names(order))
    M = dict(zip(idx, cons))

    def floor(nm, x, eps):
        # max(x, eps) = ((x + eps) + |x - eps|) / 2: smooth floor, expressible in the AST.
        return m.primitive(nm, ((x + eps) + dsl.abs_(x - eps)) / 2.0)

    M00 = floor("M00f", M[(0, 0)], eps_m00) if robust else M[(0, 0)]
    u = m.primitive("u", M[(1, 0)] / M00)
    v = m.primitive("v", M[(0, 1)] / M00)

    # normalized raw moments m_pq = M_pq / M00 (no let: each used once)
    mn = {pq: (1.0 if pq == (0, 0) else M[pq] / M00) for pq in idx}

    # --- central moments: binomial transform, derived in a loop ---
    # C_pq = sum_{i<=p, j<=q} comb(p,i) comb(q,j) (-u)^(p-i) (-v)^(q-j) m_ij
    C = {(0, 0): 1.0, (1, 0): 0.0, (0, 1): 0.0}
    for s in range(2, order + 1):
        for q in range(s + 1):
            p = s - q
            expr = None
            for i in range(p + 1):
                for j in range(q + 1):
                    coef = float(comb(p, i) * comb(q, j) * (-1) ** (p - i + q - j))
                    t = coef * _pow(u, p - i) * _pow(v, q - j)
                    if (i, j) != (0, 0):
                        t = t * mn[(i, j)]
                    expr = t if expr is None else expr + t
            C[(p, q)] = m.primitive("C%d%d" % (p, q), expr)

    # --- standardization: S_pq = C_pq / (sx^p sy^q); S20 = S02 = 1 by construction ---
    C20 = floor("C20f", C[(2, 0)], eps_cov) if robust else C[(2, 0)]
    C02 = floor("C02f", C[(0, 2)], eps_cov) if robust else C[(0, 2)]
    sx = m.primitive("sx", dsl.sqrt(C20))
    sy = m.primitive("sy", dsl.sqrt(C02))
    S = {"S20": 1.0, "S02": 1.0}
    for (p, q), c in C.items():
        if p + q >= 2 and (p, q) not in ((2, 0), (0, 2)):
            S["S%d%d" % (p, q)] = m.primitive("S%d%d" % (p, q),
                                              c / (_pow(sx, p) * _pow(sy, q)))

    # --- closure (the ONLY physics) then de-standardization C'_pq = S'_pq sx^p sy^q ---
    top = closure(S)
    want = {"S%d%d" % (p, order + 1 - p) for p in range(order + 2)}
    if set(top) != want:
        raise ValueError("moments: the closure must return exactly the keys %s "
                         "(got %s)" % (sorted(want), sorted(top)))
    Call = dict(C)
    for key, e in top.items():
        p, q = int(key[1]), int(key[2])
        Call[(p, q)] = (0.0 if _is_zero(e)
                        else m.primitive("C%d%d" % (p, q), e * _pow(sx, p) * _pow(sy, q)))

    # --- reconstruction of the order order+1 raw moments: inverse binomial ---
    # m_pq = sum_{i<=p, j<=q} comb(p,i) comb(q,j) u^(p-i) v^(q-j) C_ij
    Mtop = {}
    for q in range(order + 2):
        p = order + 1 - q
        expr = None
        for i in range(p + 1):
            for j in range(q + 1):
                cij = Call.get((i, j))
                if cij is None or _is_zero(cij):
                    continue
                t = float(comb(p, i) * comb(q, j)) * _pow(u, p - i) * _pow(v, q - j)
                if not (isinstance(cij, float) and cij == 1.0):
                    t = t * cij
                expr = t if expr is None else expr + t
        Mtop[(p, q)] = m.primitive("M%d%d" % (p, q), M00 * expr)

    # --- flux: order shift F_x[M_pq] = M_{p+1,q}, F_y[M_pq] = M_{p,q+1} ---
    def raw(pq):
        return M[pq] if pq in M else Mtop[pq]

    m.flux(x=[raw((p + 1, q)) for (p, q) in idx],
           y=[raw((p, q + 1)) for (p, q) in idx])

    if exact_speeds:
        m.wave_speeds_from_jacobian(blocks=blocks)
    if roe:
        m.roe_from_jacobian()
    if sources is not None:
        m.source(sources(m, M))
    m.primitive_vars(*cons)
    m.conservative_from(list(cons))
    return m
