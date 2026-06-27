"""pops.ir.expr -- symbolic node classes for the flux-DSL and board AST.

Two clearly delimited sections:

  1. FLUX-DSL NODES  (originally in pops.dsl)
     Expr, _wrap, Const, Var, _Bin, Add, Sub, Mul, Div, Pow, Neg, Sqrt, Abs, Sign.

  2. BOARD AST NODES  (originally in pops.math)
     _BoardNode, Equation, Partial, Gradient, Laplacian, RateTerm, _as_rate,
     RateExpr, Divergence, TimeDerivative, Unknown, OpApply, Integral.

No C++ emission lives here: the to_cpp() methods on the flux-DSL nodes are
the canonical symbolic representation; helpers that *emit* C++ source strings
(_cpp_expand, _cse_emit, _cpp_roe, etc.) remain in pops.dsl.
"""

# numpy backs only the .eval() host interpreter (Sqrt/Abs/Sign below); IR construction and the
# to_cpp() codegen never touch it. It is imported lazily inside eval() so pops.ir stays importable
# in a bare interpreter -- e.g. the build-time scripts/gen_solver_kernel.py codegen, which emits
# C++ from the IR and never evaluates it (so numpy need not be installed for the C++ build).


# =============================================================================
# SECTION 1 -- FLUX-DSL NODES  (from pops.dsl)
# =============================================================================

class Expr:
    """Symbolic expression node. Operators build the tree; eval(env) applies it to
    numpy arrays (env: name -> array or scalar)."""

    def __add__(self, o): return Add(self, _wrap(o))
    def __radd__(self, o): return Add(_wrap(o), self)
    def __sub__(self, o): return Sub(self, _wrap(o))
    def __rsub__(self, o): return Sub(_wrap(o), self)
    def __mul__(self, o): return Mul(self, _wrap(o))
    def __rmul__(self, o): return Mul(_wrap(o), self)
    def __truediv__(self, o): return Div(self, _wrap(o))
    def __rtruediv__(self, o): return Div(_wrap(o), self)
    def __neg__(self): return Neg(self)
    def __pos__(self): return self  # +expr = identity (the CoupledSource API writes +k*ne*ng)
    def __abs__(self): return Abs(self)  # abs(expr) -> |expr| (absolute value, e.g. |lambda| of Roe)
    def __pow__(self, o): return Pow(self, _wrap(o))

    def eval(self, env): raise NotImplementedError
    def deps(self): return set()
    def __repr__(self): return self._str()
    def _str(self): return "?"


def _wrap(o):
    if isinstance(o, Expr):
        return o
    # A Param exposes its internal tree NODE (_node: Const for 'const', RuntimeParamRef for
    # 'runtime'). We promote it via that node, NOT via float(o): otherwise sqrt(param_runtime) /
    # dsl.sqrt(param) would inline the declaration value (Const) instead of emitting params.get(...).
    node = getattr(o, "_node", None)
    if isinstance(node, Expr):
        return node
    return Const(float(o))


class Const(Expr):
    def __init__(self, value): self.value = float(value)
    def eval(self, env): return self.value
    def to_cpp(self): return repr(self.value)
    def _str(self): return repr(self.value)


class Var(Expr):
    """Named variable: conservative, primitive, auxiliary (field) or constant."""

    def __init__(self, name, kind):
        self.name = name
        self.kind = kind
    def eval(self, env):
        if self.name not in env:
            raise KeyError("variable '%s' (%s) missing from the environment" % (self.name, self.kind))
        return env[self.name]
    def deps(self): return {self.name}
    def to_cpp(self): return self.name
    def _str(self): return self.name


class _Bin(Expr):
    op = "?"
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def deps(self): return self.a.deps() | self.b.deps()
    def to_cpp(self): return "(%s %s %s)" % (self.a.to_cpp(), self.op, self.b.to_cpp())
    def _str(self): return "(%s %s %s)" % (self.a, self.op, self.b)


class Add(_Bin):
    op = "+"
    def eval(self, env): return self.a.eval(env) + self.b.eval(env)


class Sub(_Bin):
    op = "-"
    def eval(self, env): return self.a.eval(env) - self.b.eval(env)


class Mul(_Bin):
    op = "*"
    def eval(self, env): return self.a.eval(env) * self.b.eval(env)


class Div(_Bin):
    op = "/"
    def eval(self, env): return self.a.eval(env) / self.b.eval(env)


class Pow(_Bin):
    op = "**"
    def eval(self, env): return self.a.eval(env) ** self.b.eval(env)
    def to_cpp(self): return "std::pow(%s, %s)" % (self.a.to_cpp(), self.b.to_cpp())


class Neg(Expr):
    def __init__(self, a): self.a = a
    def eval(self, env): return -self.a.eval(env)
    def deps(self): return self.a.deps()
    def to_cpp(self): return "(-%s)" % self.a.to_cpp()
    def _str(self): return "(-%s)" % self.a


class Sqrt(Expr):
    def __init__(self, a): self.a = a
    def eval(self, env):
        import numpy as np
        return np.sqrt(self.a.eval(env))
    def deps(self): return self.a.deps()
    def to_cpp(self): return "std::sqrt(%s)" % self.a.to_cpp()
    def _str(self): return "sqrt(%s)" % self.a


class Abs(Expr):
    """Absolute value ``|a|`` (e.g. ``|lambda_k|`` of a Roe dissipation). Emitted as std::fabs at codegen
    (equal to the ternary a<0?-a:a outside -0.0). Not differentiable by dsl.diff (no sign node)."""
    def __init__(self, a): self.a = a
    def eval(self, env):
        import numpy as np
        return np.abs(self.a.eval(env))
    def deps(self): return self.a.deps()
    def to_cpp(self): return "std::fabs(%s)" % self.a.to_cpp()
    def _str(self): return "abs(%s)" % self.a


class Sign(Expr):
    """Signe de a : -1, 0 ou 1 (np.sign cote interprete). Emis au codegen comme le ternaire SANS
    branche (a > 0) - (a < 0) (exact en pops::Real). Sert aux selections par masques des branches par
    cellule (ADC-177 : clamps de projection en max/min via abs/sign, sans if). Derivee nulle presque
    partout (saut en 0, mesure nulle), cf. dsl.diff."""
    def __init__(self, a): self.a = a
    def eval(self, env):
        import numpy as np
        return np.sign(self.a.eval(env))
    def deps(self): return self.a.deps()
    def to_cpp(self):
        s = self.a.to_cpp()
        return "(pops::Real(%s > 0) - pops::Real(%s < 0))" % (s, s)
    def _str(self): return "sign(%s)" % self.a


# =============================================================================
# SECTION 2 -- BOARD AST NODES  (from pops.math)
# =============================================================================

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


# --- elliptic field-operator algebra base (Spec 5 sec.9.2, ADC-491) -----------------------
# Laplacian inherits this base; the concrete terms (Reaction / CoeffGradient / DivCoeffGrad /
# EllipticSum) and the helpers (_as_elliptic / principal_kinds) live in pops.ir.elliptic to
# keep this file within the size budget. Inert IR only; the C++ elliptic solver executes.
class _EllipticTerm(_BoardNode):
    """A summand of an elliptic field-operator left-hand side.

    Elliptic terms compose with ``+`` / ``-`` / unary ``-`` into a
    ``pops.ir.elliptic.EllipticSum``; ``==`` builds the field :class:`Equation`.
    """

    def _elliptic_terms(self):
        return [self]

    def _principal_kinds(self):
        return {self._kind()}

    def _kind(self):
        raise NotImplementedError

    def __neg__(self):  # concrete terms (Laplacian / Reaction / ...) override this
        raise NotImplementedError

    def __add__(self, other):
        from .elliptic import EllipticSum, _as_elliptic
        return EllipticSum(self._elliptic_terms() + _as_elliptic(other)._elliptic_terms())

    def __radd__(self, other):
        from .elliptic import EllipticSum, _as_elliptic
        return EllipticSum(_as_elliptic(other)._elliptic_terms() + self._elliptic_terms())

    def __sub__(self, other):
        from .elliptic import _as_elliptic
        return self.__add__(-_as_elliptic(other))


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

    def __mul__(self, coeff):
        # coeff * grad(phi) -- a coefficient-scaled gradient (the flux of div(coeff*grad)).
        from .elliptic import CoeffGradient
        return CoeffGradient(self.field, coeff, self.scale)

    __rmul__ = __mul__

    def __repr__(self):
        return "Gradient(%r)" % (self.field,)


class Laplacian(_EllipticTerm):
    """``scale * Laplacian(field)`` -- the elliptic operator of a field solve."""

    def __init__(self, field, scale=1.0):
        self.field = field
        self.scale = float(scale)

    def _kind(self):
        return "laplacian"

    def __neg__(self):
        return Laplacian(self.field, -self.scale)

    def __repr__(self):
        return "Laplacian(%s%r)" % (
            ("" if self.scale == 1.0 else "%g*" % self.scale), self.field)


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

    def __mul__(self, coeff):
        # coeff * phi -- a zeroth-order reaction term (the k*phi of a screened Poisson).
        from .elliptic import Reaction
        return Reaction(self, coeff)

    __rmul__ = __mul__

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
