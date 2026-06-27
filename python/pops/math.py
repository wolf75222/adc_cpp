"""Blackboard math operators for the Spec 3 physics DSL.

Thin facade: all node classes and constructors now live in :mod:`pops.ir`
(single source of truth). This module re-exports them so every existing
``from pops.math import ...`` or ``pops.math.X`` call keeps working unchanged.
"""

__all__ = [
    "sqrt", "grad", "div", "laplacian", "dx", "dy", "ddt", "rate", "unknown",
    "integral",
    # IR node types (advanced / introspection)
    "Equation", "Gradient", "Partial", "Laplacian", "Divergence",
    "TimeDerivative", "Unknown", "OpApply", "Integral", "RateTerm", "RateExpr",
    # elliptic field-operator algebra (Spec 5 sec.9.2)
    "Reaction", "CoeffGradient", "DivCoeffGrad", "EllipticSum", "principal_kinds",
]

from pops.ir.expr import (  # noqa: F401
    Equation,
    _BoardNode,
    Partial,
    Gradient,
    Laplacian,
    RateTerm,
    RateExpr,
    Divergence,
    TimeDerivative,
    Unknown,
    OpApply,
    Integral,
)
from pops.ir.elliptic import (  # noqa: F401  (Spec 5 sec.9.2 elliptic field-operator algebra)
    Reaction,
    CoeffGradient,
    DivCoeffGrad,
    EllipticSum,
    principal_kinds,
)
from pops.ir.expr import _as_rate  # noqa: F401  (used internally by RateTerm)
from pops.ir.ops import (  # noqa: F401
    grad,
    dx,
    dy,
    laplacian,
    div,
    ddt,
    rate,
    unknown,
    integral,
)
from pops.ir.ops import board_sqrt as sqrt  # noqa: F401


# --- scalar dtypes (Spec 5 sec.5.12: a typed param declares its dtype) -------------------
class _DType:
    """An inert scalar dtype marker (``Real`` / ``Integer`` / ``Bool``).

    Used by :mod:`pops.params` so a parameter declares a typed dtype instead of a string;
    the codegen / runtime consume it. It computes nothing.
    """

    def __init__(self, name):
        self._name = str(name)

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return self._name

    __str__ = __repr__


Real = _DType("Real")
Integer = _DType("Integer")
Bool = _DType("Bool")

__all__ += ["Real", "Integer", "Bool"]
