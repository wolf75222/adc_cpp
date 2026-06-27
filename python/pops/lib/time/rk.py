"""pops.lib.time.rk -- Classic explicit Runge-Kutta schemes (RK4, generic rk) and Butcher tableaux.

Exports: rk4, rk, explicit_rk, ButcherTableau, RK4_TABLEAU, SSPRK2_TABLEAU.
"""

from ._helpers import _opcall, _stage_rhs


def rk4(P, block, *, sources=("default",), flux=True):
    """Classic RK4, expressed with NO special RK4 class (spec acceptance 29):
    U^{n+1} = U0 + dt/6 (k1 + 2 k2 + 2 k3 + k4)."""
    U0 = P.state(block)
    k1 = _stage_rhs(P, U0, sources, flux)
    U1 = P.linear_combine("rk4_U1", U0 + 0.5 * P.dt * k1)
    k2 = _stage_rhs(P, U1, sources, flux)
    U2 = P.linear_combine("rk4_U2", U0 + 0.5 * P.dt * k2)
    k3 = _stage_rhs(P, U2, sources, flux)
    U3 = P.linear_combine("rk4_U3", U0 + P.dt * k3)
    k4 = _stage_rhs(P, U3, sources, flux)
    P.commit(block, P.linear_combine(
        "rk4_step", U0 + P.dt / 6.0 * k1 + P.dt / 3.0 * k2 + P.dt / 3.0 * k3 + P.dt / 6.0 * k4))


# Classic explicit Butcher tableaux (A lower-triangular, b weights, c nodes) for `rk` (ADC-423).
class ButcherTableau:
    """An explicit Butcher tableau ``(A, b, c)`` for `rk`: ``A`` is strictly lower-triangular (stage i
    depends only on stages j < i), ``b`` the final weights, ``c`` the (unused-by-the-lowering) nodes.
    Validated as explicit and consistent (``len(A) == len(b)``, row i has i entries, ``sum(b) == 1``)."""

    def __init__(self, A, b, c=None, name=None):
        self.A = [list(row) for row in A]
        self.b = list(b)
        self.c = list(c) if c is not None else [sum(row) for row in self.A]
        self.name = name
        s = len(self.b)
        if len(self.A) != s or len(self.c) != s:
            raise ValueError("ButcherTableau: A, b, c must share the stage count")
        for i, row in enumerate(self.A):
            if len(row) > i and any(row[j] != 0.0 for j in range(i, len(row))):
                raise ValueError(
                    "ButcherTableau: A must be strictly lower-triangular (stage %d reads stage >= %d); "
                    "rk lowers EXPLICIT tableaux only" % (i, i))
        if abs(sum(self.b) - 1.0) > 1e-12:
            raise ValueError("ButcherTableau: weights b must sum to 1 (got %r)" % (sum(self.b),))

    @property
    def stages(self):
        return len(self.b)


# RK4 (classic): the same tableau the rk4 macro hard-codes, written data-driven.
RK4_TABLEAU = ButcherTableau(
    A=[[],
       [0.5],
       [0.0, 0.5],
       [0.0, 0.0, 1.0]],
    b=[1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0],
    c=[0.0, 0.5, 0.5, 1.0],
    name="rk4")

# SSPRK2 (Heun) in NON-Shu-Osher Butcher form: k1 at U, k2 at U+dt*k1, U^{n+1}=U+dt(1/2 k1+1/2 k2).
SSPRK2_TABLEAU = ButcherTableau(
    A=[[],
       [1.0]],
    b=[0.5, 0.5],
    c=[0.0, 1.0],
    name="ssprk2")


def rk(P, block, tableau, *, sources=("default",), flux=True):
    """Generic explicit Runge-Kutta from a Butcher @p tableau (ADC-423), lowered to the SAME stage chain
    the hard-coded `rk4` macro emits -- ``solve_fields`` + ``rhs`` + ``linear_combine``, no RK class:

        k_i      = R( U + dt * sum_{j<i} A[i][j] * k_j )       (the i-th stage RHS)
        U^{n+1}  = U + dt * sum_i b[i] * k_i

    @p tableau is a `ButcherTableau` (or a raw ``(A, b, c)`` triple); ``A`` must be strictly
    lower-triangular (explicit). ``RK4_TABLEAU`` and ``SSPRK2_TABLEAU`` are provided as the classic
    constants: ``rk(P, blk, RK4_TABLEAU)`` builds the identical final affine combination as
    ``rk4(P, blk)`` (a permutation of the same ``U0 + dt(1/6 k1 + 1/3 k2 + 1/3 k3 + 1/6 k4)`` inputs),
    and ``rk(P, blk, SSPRK2_TABLEAU)`` matches Heun's ``U + dt(1/2 k1 + 1/2 k2)``."""
    if not isinstance(tableau, ButcherTableau):
        A, b, c = tableau if len(tableau) == 3 else (tableau[0], tableau[1], None)
        tableau = ButcherTableau(A, b, c)
    tag = (tableau.name + "_") if tableau.name else "rk_"
    U0 = P.state(block)
    ks = []
    for i in range(tableau.stages):
        if i == 0:
            Ui = U0  # the first stage reads U^n directly (no scratch combine, like rk4)
        else:
            expr = U0
            for j in range(i):
                aij = tableau.A[i][j]
                if aij != 0.0:
                    expr = expr + (P.dt * aij) * ks[j]
            Ui = P.linear_combine("%sU%d" % (tag, i), expr)
        ks.append(_stage_rhs(P, Ui, sources, flux))
    final = U0
    for i in range(tableau.stages):
        bi = tableau.b[i]
        if bi != 0.0:
            final = final + (P.dt * bi) * ks[i]
    P.commit(block, P.linear_combine("%sstep" % tag, final))


def explicit_rk(P, block, *, rhs_operator, fields_operator=None, tableau=None, A=None, b=None,
                c=None, state_space="U"):
    """Generic explicit Runge-Kutta over a typed rate operator (Spec 2, operator-first).

    Each stage is ``k_i = rhs_operator(U_i[, fields_operator(U_i)])``; the tableau lowers to the same
    affine stage chain as :func:`rk`. Pass a ``ButcherTableau`` / ``(A, b, c)`` via ``tableau`` or the
    raw ``A`` / ``b`` / ``c``. ``fields_operator`` is optional (a pure-flux rate needs no fields).
    """
    if tableau is None:
        if A is None or b is None:
            raise ValueError("explicit_rk: provide a tableau or A and b")
        tableau = ButcherTableau(A, b, c)
    elif not isinstance(tableau, ButcherTableau):
        ta, tb, tc = tableau if len(tableau) == 3 else (tableau[0], tableau[1], None)
        tableau = ButcherTableau(ta, tb, tc)
    tag = (tableau.name + "_") if tableau.name else "rk_"
    u0 = P.state(block)
    ks = []
    for i in range(tableau.stages):
        if i == 0:
            u_i = u0
        else:
            expr = u0
            for j in range(i):
                aij = tableau.A[i][j]
                if aij != 0.0:
                    expr = expr + (P.dt * aij) * ks[j]
            u_i = P.linear_combine("%sU%d" % (tag, i), expr)
        if fields_operator is not None:
            f_i = _opcall(P, fields_operator, u_i)
            ks.append(_opcall(P, rhs_operator, u_i, f_i, value_name="%sk%d" % (tag, i)))
        else:
            ks.append(_opcall(P, rhs_operator, u_i, value_name="%sk%d" % (tag, i)))
    final = u0
    for i in range(tableau.stages):
        bi = tableau.b[i]
        if bi != 0.0:
            final = final + (P.dt * bi) * ks[i]
    P.commit(block, P.linear_combine("%sstep" % tag, final))
