"""Moment-hierarchy source terms: Lorentz forcing, Maxwellian matching, BGK collision.

Symbols are re-exported via python/pops/lib/moments/__init__.py.

These functions are pure Python / DSL-expression arithmetic and do NOT require
the DSL compiler; no lazy import is needed here.
"""
from math import comb

from pops.descriptors import Descriptor

from .model_builder import moment_indices, _pow


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


# --- thin facades over the free functions (Spec 4 moments NEW API) ---------
# These carry no math: they forward to the closure-free free functions above so the
# arithmetic stays in one place and lowers to the same generated flux.
class VlasovSources:
    """Namespace of moment-hierarchy source builders (pure forwarders).

    Every method forwards to the corresponding closure-free free function in this module;
    no arithmetic lives here. ``M`` is the dict ``(p, q) -> conservative-moment Expr``.
    """

    @staticmethod
    def lorentz(M, ex, ey, q_over_m, omega_c):
        """The Vlasov-Lorentz hierarchy source (forwards to :func:`lorentz_sources`)."""
        return lorentz_sources(M, ex, ey, q_over_m, omega_c)

    @staticmethod
    def maxwellian_eq(M):
        """The local-Maxwellian raw moments (forwards to :func:`maxwellian_moments`)."""
        return maxwellian_moments(M)

    @staticmethod
    def bgk(M, nu):
        """The BGK relaxation source (forwards to :func:`bgk_source`)."""
        return bgk_source(M, nu)


class MagneticMomentSource(Descriptor):
    """Descriptor of a pure-magnetic moment source: ``omega_c(B) = q_over_m * B``.

    It CHOOSES the magnetic-source route, binding which ``q_over_m`` param name and
    ``b_field`` aux name the Lorentz magnetic branch reads, so it is a typed
    :class:`pops.descriptors.Descriptor` (Spec 5 sec.6): it declares its options /
    capabilities and is inspectable. :meth:`as_sources` returns a sources callable
    forwarding to the magnetic branch of :func:`lorentz_sources` (zero electric field).
    It carries no arithmetic.
    """

    category = "moment_source"

    def __init__(self, q_over_m="q_over_m", b_field="B_z"):
        self.q_over_m = str(q_over_m)
        self.b_field = str(b_field)

    def options(self):
        return {"q_over_m": self.q_over_m, "b_field": self.b_field}

    def capabilities(self):
        return {"provides": "magnetic_lorentz"}

    def as_sources(self, q_over_m_value=1.0):
        """A ``(m, M) -> list`` sources callable: ``omega_c = q_over_m * B`` (electric field 0).

        @p q_over_m_value: the default value of the declared ``q_over_m`` param.
        """
        qom_name, b_name = self.q_over_m, self.b_field

        def sources(m, M):
            qom = m.param(qom_name, q_over_m_value)
            b_z = m.aux(b_name)
            omega_c = qom * b_z
            return lorentz_sources(M, 0.0, 0.0, qom, omega_c)

        return sources

    def __repr__(self):
        return "MagneticMomentSource(q_over_m=%r, b_field=%r)" % (self.q_over_m, self.b_field)
