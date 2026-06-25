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

VERBATIM extraction (Spec 4) of the model-builder symbols from `adc/moments.py`; the
function bodies are preserved exactly. The flux-DSL primitives are imported lazily inside
the functions that use them (see the SPEC4-TODO markers).
"""
from math import comb


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
    # SPEC4-TODO: repoint to adc.ir / adc.physics once extracted
    from adc import dsl
    # NUMERIC zero (int/float) or SYMBOLIC zero (dsl.Const(0.0)): a closure may return
    # either one; both drop the term from the generated flux (dead primitive not emitted).
    if isinstance(e, (int, float)):
        return float(e) == 0.0
    return isinstance(e, dsl.Const) and e.value == 0.0


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
    # SPEC4-TODO: repoint to adc.ir / adc.physics once extracted
    from adc import dsl
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
