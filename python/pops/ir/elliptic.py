"""pops.ir.elliptic -- the board-node elliptic field-operator algebra (Spec 5 sec.9.2).

The left-hand side of a field equation is a sum of elliptic operator terms. These nodes make
the variable-coefficient forms the Poisson family uses AUTHORABLE and inspectable:

  -laplacian(phi) + k*phi == rhs            (screened Poisson)
  -div(eps*grad(phi)) + k*phi == rhs        (anisotropic / variable-coefficient, sec.9.2)

They are INERT: IR construction only; the C++ elliptic solver executes. The base
:class:`pops.ir.expr._EllipticTerm` (which :class:`Laplacian` inherits) lives in
:mod:`pops.ir.expr`; the concrete terms + helpers live here to keep ``expr.py`` small.
Lowering the variable-coefficient operators in the elliptic codegen is the coordinated
follow-up.
"""
from .expr import _BoardNode, _EllipticTerm


def _as_elliptic(x):
    """Coerce ``x`` to an :class:`pops.ir.expr._EllipticTerm`, else a clear error."""
    if isinstance(x, _EllipticTerm):
        return x
    raise TypeError(
        "an elliptic field-equation left-hand side must be a sum of elliptic operator terms "
        "(laplacian(phi) / div(coeff*grad(phi)) / a reaction coeff*phi); got %r" % (x,))


def principal_kinds(node):
    """The set of elliptic principal-operator kinds in a field-equation LHS (empty if none)."""
    return node._principal_kinds() if isinstance(node, _EllipticTerm) else set()


class Reaction(_EllipticTerm):
    """A zeroth-order reaction term ``coeff * phi`` (built by ``coeff * unknown("phi")``)."""

    def __init__(self, field, coeff, scale=1.0):
        self.field = field
        self.coeff = coeff
        self.scale = float(scale)

    def _kind(self):
        return "reaction"

    def __neg__(self):
        return Reaction(self.field, self.coeff, -self.scale)

    def __repr__(self):
        lead = "" if self.scale == 1.0 else "%g*" % self.scale
        return "Reaction(%s%r*%r)" % (lead, self.coeff, self.field)


class CoeffGradient(_BoardNode):
    """``coeff * grad(phi)`` -- consumed by ``div(...)`` to build a :class:`DivCoeffGrad`."""

    def __init__(self, field, coeff, scale=1.0):
        self.field = field
        self.coeff = coeff
        self.scale = float(scale)

    def __neg__(self):
        return CoeffGradient(self.field, self.coeff, -self.scale)

    def __repr__(self):
        return "CoeffGradient(%r*grad(%r))" % (self.coeff, self.field)


class DivCoeffGrad(_EllipticTerm):
    """``scale * div(coeff * grad(phi))`` -- a variable / anisotropic principal operator."""

    def __init__(self, field, coeff, scale=1.0):
        self.field = field
        self.coeff = coeff
        self.scale = float(scale)

    def _kind(self):
        return "div_coeff_grad"

    def __neg__(self):
        return DivCoeffGrad(self.field, self.coeff, -self.scale)

    def __repr__(self):
        lead = "" if self.scale == 1.0 else "%g*" % self.scale
        return "DivCoeffGrad(%sdiv(%r*grad(%r)))" % (lead, self.coeff, self.field)


class EllipticSum(_EllipticTerm):
    """An accumulated sum of elliptic operator terms (laplacian / div-coeff-grad / reaction)."""

    def __init__(self, terms):
        self.terms = list(terms)

    def _elliptic_terms(self):
        return list(self.terms)

    def _principal_kinds(self):
        kinds = set()
        for term in self.terms:
            kinds |= term._principal_kinds()
        return kinds

    def _kind(self):
        return "sum"

    def __neg__(self):
        return EllipticSum([-term for term in self.terms])

    def __repr__(self):
        return "EllipticSum(%r)" % (self.terms,)


__all__ = ["Reaction", "CoeffGradient", "DivCoeffGrad", "EllipticSum", "principal_kinds"]
