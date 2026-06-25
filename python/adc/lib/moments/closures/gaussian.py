"""Generic Gaussian (Levermore) closure for 2D moment models.

VERBATIM extraction (Spec 4) of `gaussian_closure` from `adc/moments.py`. The body is
preserved exactly; only the imports are repointed to a lazy form for the re-organization.
"""


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
