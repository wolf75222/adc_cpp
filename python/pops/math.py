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
