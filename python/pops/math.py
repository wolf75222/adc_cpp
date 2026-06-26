"""Blackboard math operators for the Spec 3 physics DSL.

This module is the user-facing notation layer: ``ddt``, ``div``, ``grad``,
``laplacian``, ``dx``/``dy``, ``rate``, ``unknown``, ``integral`` and ``sqrt``.
It builds a tiny, numerics-free IR of *board nodes* that :mod:`pops.physics`
(model authoring) and :mod:`pops.time` (time programs) destructure when they
lower to the Spec 2 operator-first IR. A board node carries structure only; the
actual expressions over a state are :mod:`pops.dsl` ``Expr`` trees.

Design note -- why a separate node type. :mod:`pops.dsl` ``Expr`` deliberately
does not define ``__eq__`` (it relies on identity hashing for CSE). The board
layer needs ``lhs == rhs`` to build an :class:`Equation`; so the board nodes are
a distinct, small hierarchy that owns ``==`` and the rate algebra, and convert to
``Expr`` only when a model resolves them in context (where the aux names of a
field are known). Pure standard library: it imports :mod:`pops.dsl` lazily, only
for ``sqrt`` and friends.
"""

__all__ = [
    "sqrt", "grad", "div", "laplacian", "dx", "dy", "ddt", "rate", "unknown",
    "integral",
    # IR node types (advanced / introspection)
    "Equation", "Gradient", "Partial", "Laplacian", "Divergence",
    "TimeDerivative", "Unknown", "OpApply", "Integral", "RateTerm", "RateExpr",
]


# --- equation node -----------------------------------------------------------
class Equation:
    """A board equation ``lhs == rhs``.

    Produced by ``ddt(U) == ...``, ``-laplacian(phi) == rhs`` or
    ``(I - dt*C) @ unknown("x") == rhs``. Carries the two sides verbatim; the
    consuming API (``Model.rate`` / ``Model.solve_field`` / ``Program.solve``)
    decides how to lower it.
    """

    __slots__ = ("lhs", "rhs")

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return "Equation(%r == %r)" % (self.lhs, self.rhs)


class _BoardNode:
    """Base of every board node: owns ``==`` (build an :class:`Equation`)."""

    def __eq__(self, other):  # noqa: D105 -- equation builder, not a comparison
        return Equation(self, other)

    # board nodes are not hashable (they define a non-identity ``__eq__``)
    __hash__ = None


# --- spatial operators -------------------------------------------------------
class Partial(_BoardNode):
    """A first partial derivative ``scale * d(field)/dx_axis`` (axis 0=x, 1=y).

    ``grad(phi).x`` and ``dx(phi)`` build this. A model resolves it to the field's
    canonical gradient aux (``grad_x`` / ``grad_y``). Negation/scaling track the
    leading coefficient so ``-grad(phi).x`` resolves to ``-grad_x``.
    """

    def __init__(self, field, axis, scale=1.0):
        self.field = field
        self.axis = int(axis)
        self.scale = float(scale)

    def __neg__(self):
        return Partial(self.field, self.axis, -self.scale)

    def __mul__(self, k):
        return Partial(self.field, self.axis, self.scale * float(k))

    __rmul__ = __mul__

    def __repr__(self):
        d = "x" if self.axis == 0 else "y"
        return "Partial(%s%r.d%s)" % (
            ("" if self.scale == 1.0 else "%g*" % self.scale), self.field, d)


class Gradient(_BoardNode):
    """The gradient of a scalar field; ``grad(phi).x`` / ``.y`` are :class:`Partial`."""

    def __init__(self, field, scale=1.0):
        self.field = field
        self.scale = float(scale)

    @property
    def x(self):
        return Partial(self.field, 0, self.scale)

    @property
    def y(self):
        return Partial(self.field, 1, self.scale)

    def __neg__(self):
        return Gradient(self.field, -self.scale)

    def __repr__(self):
        return "Gradient(%r)" % (self.field,)


class Laplacian(_BoardNode):
    """``scale * Laplacian(field)`` -- the elliptic operator of a field solve."""

    def __init__(self, field, scale=1.0):
        self.field = field
        self.scale = float(scale)

    def __neg__(self):
        return Laplacian(self.field, -self.scale)

    def __repr__(self):
        return "Laplacian(%s%r)" % (
            ("" if self.scale == 1.0 else "%g*" % self.scale), self.field)


# --- rate algebra ------------------------------------------------------------
class RateTerm(_BoardNode):
    """A summand of a rate equation right-hand side.

    Both :class:`Divergence` (a ``-div F`` flux contribution) and a model's
    source handle are :class:`RateTerm`; ``+`` / ``-`` / unary ``-`` build a
    :class:`RateExpr` that the model destructures into ``flux`` and ``sources``.
    """

    def _rate_terms(self):
        """Return ``[(kind, payload, sign)]`` -- one entry per primitive summand."""
        raise NotImplementedError

    def __neg__(self):
        return RateExpr([(k, p, -s) for (k, p, s) in self._rate_terms()])

    def __add__(self, other):
        return RateExpr(self._rate_terms() + _as_rate(other)._rate_terms())

    def __radd__(self, other):
        return RateExpr(_as_rate(other)._rate_terms() + self._rate_terms())

    def __sub__(self, other):
        return self + (-_as_rate(other))


def _as_rate(x):
    """Coerce ``x`` to a :class:`RateTerm` or raise a clear error."""
    if isinstance(x, RateTerm):
        return x
    raise TypeError(
        "a rate equation right-hand side must be a sum of -div(flux) and source "
        "terms; got %r" % (x,))


class RateExpr(RateTerm):
    """An accumulated sum of rate terms (flux contributions and source handles)."""

    def __init__(self, terms):
        self.terms = list(terms)

    def _rate_terms(self):
        return list(self.terms)

    def __repr__(self):
        return "RateExpr(%r)" % (self.terms,)


class Divergence(RateTerm):
    """``scale * div(flux)`` -- a flux contribution to a rate equation.

    Written ``-div(F)`` for the standard hyperbolic ``-div F``. ``flux`` is the
    model's flux handle (or ``None`` for the model's default flux).
    """

    def __init__(self, flux, scale=1.0):
        self.flux = flux
        self.scale = float(scale)

    def _rate_terms(self):
        return [("flux", self.flux, self.scale)]

    def __repr__(self):
        return "Divergence(%s%r)" % (
            ("" if self.scale == 1.0 else "%g*" % self.scale), self.flux)


class TimeDerivative(_BoardNode):
    """``ddt(U)`` / ``rate(U)`` -- the left-hand side of a rate equation."""

    def __init__(self, state):
        self.state = state

    def __repr__(self):
        return "ddt(%r)" % (self.state,)


class Unknown(_BoardNode):
    """A solve unknown: ``unknown("U*")`` in ``(I - dt*C) @ unknown("U*") == rhs``."""

    def __init__(self, name):
        self.name = str(name)

    def __rmatmul__(self, operator):
        """``operator @ unknown("U*")`` -- the left-hand side of an implicit solve."""
        return OpApply(operator, self)

    def __repr__(self):
        return "unknown(%r)" % (self.name,)


class OpApply(_BoardNode):
    """``operator @ unknown`` -- a board solve left-hand side, completed by ``== rhs``.

    Carries the operator (a Program ``_Operator`` / linear-source value) and the
    :class:`Unknown`; :meth:`pops.time.Program.solve` destructures it.
    """

    def __init__(self, operator, unknown):
        self.operator = operator
        self.unknown = unknown

    def __repr__(self):
        return "OpApply(%r @ %r)" % (self.operator, self.unknown)


class Integral(_BoardNode):
    """A spatial integral of an expression -- the value of a generic invariant."""

    def __init__(self, expr, over=None):
        self.expr = expr
        self.over = over

    def __repr__(self):
        return "integral(%r)" % (self.expr,)


# --- public constructors -----------------------------------------------------
def grad(field):
    """The gradient of a scalar field; use ``grad(phi).x`` / ``.y`` for components."""
    return Gradient(field)


def dx(field):
    """The x partial derivative of a field (``grad(field).x``)."""
    return Partial(field, 0)


def dy(field):
    """The y partial derivative of a field (``grad(field).y``)."""
    return Partial(field, 1)


def laplacian(field):
    """The Laplacian of a field -- the elliptic operator of a Poisson solve."""
    return Laplacian(field)


def div(flux):
    """The divergence of a flux; write ``-div(F)`` for the hyperbolic ``-div F``."""
    return Divergence(flux)


def ddt(state):
    """The time derivative of a state -- the left-hand side of a rate equation."""
    return TimeDerivative(state)


def rate(state):
    """The rate (time derivative) of a state; synonym of :func:`ddt` for time programs."""
    return TimeDerivative(state)


def unknown(name):
    """A named solve unknown, e.g. ``unknown("U*")``."""
    return Unknown(name)


def integral(expr, over=None):
    """The spatial integral of ``expr`` -- the value of a generic invariant."""
    return Integral(expr, over=over)


def sqrt(x):
    """Square root, lifted to the board: delegates to :func:`pops.dsl.sqrt` on an
    expression / parameter, falls back to :func:`math.sqrt` on a plain number."""
    try:
        from . import dsl as _dsl
        return _dsl.sqrt(x)
    except Exception:
        import math as _pymath
        return _pymath.sqrt(x)
