"""Pure-symbolic differentiation and algebra for the ADC IR (Spec 4).

Moved verbatim from :mod:`adc.dsl`: :func:`diff` (symbolic autodiff of the
:class:`~adc.ir.expr.Expr` tree) and its minimal-simplification helpers
(:func:`_is_const`, :func:`_s_add`, :func:`_s_neg`, :func:`_s_sub`,
:func:`_s_mul`, :func:`_s_div`, :func:`_s_pow`). NO C++ emission: these rebuild
``Expr`` nodes only.
"""
from .expr import (
    Abs,
    Add,
    Const,
    Div,
    Mul,
    Neg,
    Pow,
    Sign,
    Sqrt,
    Sub,
    Var,
    _wrap,
)
from .values import RuntimeParamRef


# --- Symbolic differentiation (autodiff of the Expr tree) -------------------
# dsl.diff(expr, var) differentiates the tree node by node: linearity (+, -), product (a*b)' = a'b + ab',
# quotient (a/b)' = (a'b - ab')/b^2, power (a^n)' = n a^(n-1) a' (constant exponent), root
# sqrt(a)' = a'/(2 sqrt(a)), negation. Used to build the flux Jacobian A = dF/dU (flux_jacobian)
# that the user employs to write its Roe dissipation (m.roe_dissipation). A DEFINED primitive
# is differentiated BY ITS DEFINITION (chain rule); a NON differentiated occurrence stays a
# symbol (readable emission), only the DERIVATIVE descends to the conservatives. Unknown node ->
# NotImplementedError (never a silent zero). Minimal simplifications (0*x, 1*x, x+0).
def _is_const(e, val=None):
    """True if e is a numeric constant (Const); if @p val is given, equal to val."""
    return isinstance(e, Const) and (val is None or e.value == val)


def _s_add(a, b):
    if _is_const(a, 0.0):
        return b
    if _is_const(b, 0.0):
        return a
    if isinstance(a, Const) and isinstance(b, Const):
        return Const(a.value + b.value)
    return Add(a, b)


def _s_neg(a):
    if _is_const(a, 0.0):
        return Const(0.0)
    if isinstance(a, Const):
        return Const(-a.value)
    if isinstance(a, Neg):
        return a.a
    return Neg(a)


def _s_sub(a, b):
    if _is_const(b, 0.0):
        return a
    if _is_const(a, 0.0):
        return _s_neg(b)
    if isinstance(a, Const) and isinstance(b, Const):
        return Const(a.value - b.value)
    return Sub(a, b)


def _s_mul(a, b):
    if _is_const(a, 0.0) or _is_const(b, 0.0):
        return Const(0.0)
    if _is_const(a, 1.0):
        return b
    if _is_const(b, 1.0):
        return a
    if isinstance(a, Const) and isinstance(b, Const):
        return Const(a.value * b.value)
    return Mul(a, b)


def _s_div(a, b):
    if _is_const(a, 0.0):
        return Const(0.0)
    if _is_const(b, 1.0):
        return a
    return Div(a, b)


def _s_pow(a, b):
    # b: exponent (Expr), here assumed INDEPENDENT of the differentiation variable.
    if _is_const(b, 0.0):
        return Const(1.0)
    if _is_const(b, 1.0):
        return a
    return Pow(a, b)


def diff(expr, var, defs=None):
    """Symbolic derivative of @p expr with respect to @p var (variable name or Var).

    @p defs (optional): dictionary {primitive name: definition Expr}. When the differentiation
    meets a DEFINED primitive, it differentiates its DEFINITION (chain rule) -- the primitives
    are expanded down to the conservatives without manual substitution. A primitive with the same name
    as @p var is treated as the independent variable (derivative 1). Without defs, any variable
    other than @p var is independent (derivative 0).

    @return an Expr minimally simplified (0*x, 1*x, x+0, ... removed for a readable emission).
    Raises NotImplementedError on a non differentiable node (naming its type) or a power whose
    exponent depends on @p var (would need a logarithm, a node absent from the DSL)."""
    target = var.name if isinstance(var, Var) else str(var)
    d = defs or {}

    def go(e):
        if isinstance(e, Const):
            return Const(0.0)
        if isinstance(e, RuntimeParamRef):
            return Const(0.0)  # runtime parameter: constant with respect to the conservative state
        if isinstance(e, Var):
            if e.name == target:
                return Const(1.0)
            if e.name in d:
                return go(d[e.name])  # defined primitive -> derivative of its definition (chain)
            return Const(0.0)         # another variable, independent of var
        if isinstance(e, Add):
            return _s_add(go(e.a), go(e.b))
        if isinstance(e, Sub):
            return _s_sub(go(e.a), go(e.b))
        if isinstance(e, Mul):
            return _s_add(_s_mul(go(e.a), e.b), _s_mul(e.a, go(e.b)))
        if isinstance(e, Div):
            num = _s_sub(_s_mul(go(e.a), e.b), _s_mul(e.a, go(e.b)))
            return _s_div(num, _s_mul(e.b, e.b))
        if isinstance(e, Neg):
            return _s_neg(go(e.a))
        if isinstance(e, Sqrt):
            return _s_div(go(e.a), _s_mul(Const(2.0), Sqrt(e.a)))
        if isinstance(e, Abs):
            # d|u| = (u / |u|) u' -- exact derivative away from the fold u = 0 (the smooth floors
            # max(x, eps) = ((x+eps) + |x-eps|)/2 of the 'robust' models give there exactly
            # the expected indicator); AT the fold, u/|u| is NaN: a zero-measure singularity,
            # documented (like the division of quotients).
            return _s_mul(_s_div(e.a, Abs(e.a)), go(e.a))
        if isinstance(e, Sign):
            # d sign(u) = 0 presque partout (saut en u = 0, mesure nulle -- meme convention que le
            # pli de Abs : singularite documentee, jamais rencontree par les clamps sur un ouvert).
            return Const(0.0)
        if isinstance(e, Pow):
            if not _is_const(go(e.b), 0.0):
                raise NotImplementedError(
                    "dsl.diff: derivative of a**b with exponent depending on '%s' (needs a "
                    "logarithm, a node absent from the DSL)" % target)
            # constant exponent with respect to var: (a^b)' = b a^(b-1) a'
            return _s_mul(_s_mul(e.b, _s_pow(e.a, _s_sub(e.b, Const(1.0)))), go(e.a))
        raise NotImplementedError("dsl.diff: non differentiable node %s (%r)" % (type(e).__name__, e))

    return go(_wrap(expr))
