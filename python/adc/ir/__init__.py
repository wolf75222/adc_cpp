"""adc.ir : the internal SYMBOLIC intermediate representation (Spec 4).

This package collects the pure-symbolic layers of the DSL that carry NO C++
source emission:

  - :mod:`adc.ir.expr`    -- the flux-DSL expression node classes (``Expr`` and
                             its subclasses) and the board AST node classes
                             (``Equation``, ``Gradient``, ...);
  - :mod:`adc.ir.ops`     -- the user-facing symbolic operation constructors
                             (``grad``, ``div``, ``laplacian``, ``ddt``, ...);
  - :mod:`adc.ir.values`  -- reference / witness value objects (``EigWitness``,
                             ``StateRef``, ``RuntimeParamRef``);
  - :mod:`adc.ir.lowering`-- pure-symbolic differentiation / simplification
                             (``diff``, ``_s_*``);
  - :mod:`adc.ir.visitors`-- AST traversal helpers (``_children``, ``_key``).

C++ source emission stays in :mod:`adc.dsl` (the codegen layer); it imports the
symbolic nodes and transforms from here.
"""

from .expr import (
    Abs,
    Add,
    Const,
    Div,
    Divergence,
    Equation,
    Gradient,
    Integral,
    Laplacian,
    Mul,
    Neg,
    OpApply,
    Partial,
    Pow,
    RateExpr,
    RateTerm,
    Sign,
    Sqrt,
    Sub,
    TimeDerivative,
    Unknown,
    Var,
    _BoardNode,
    _Bin,
    Expr,
)
from .ops import (
    abs_,
    ddt,
    div,
    dx,
    dy,
    eig_all_real,
    eig_lmax,
    eig_lmin,
    eig_max_im,
    grad,
    integral,
    laplacian,
    left,
    rate,
    right,
    sign,
    sqrt,
    unknown,
)
from .values import EigWitness, RuntimeParamRef, StateRef

__all__ = [
    # public ops
    "grad", "dx", "dy", "div", "laplacian", "ddt", "rate", "unknown",
    "integral", "sqrt",
    # node base classes
    "Expr", "_BoardNode",
    # board AST node types (advanced / introspection)
    "Equation", "Gradient", "Partial", "Laplacian", "Divergence",
    "TimeDerivative", "Unknown", "OpApply", "Integral", "RateTerm", "RateExpr",
]
