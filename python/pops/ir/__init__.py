"""pops.ir -- internal symbolic intermediate representation.

Public re-exports for users and for pops.dsl / pops.math rewiring.

Node base classes
  Expr, _BoardNode

Flux-DSL node classes
  Const, Var, _Bin, Add, Sub, Mul, Div, Pow, Neg, Sqrt, Abs, Sign

Board AST node classes
  Equation, Partial, Gradient, Laplacian,
  RateTerm, RateExpr, Divergence, TimeDerivative, Unknown, OpApply, Integral

Reference / witness value classes
  EigWitness, StateRef, RuntimeParamRef

Public ops (board + DSL)
  grad, dx, dy, div, laplacian, ddt, rate, unknown, integral,
  sqrt (flux-DSL canonical), board_sqrt (board delegate to pops.dsl.sqrt),
  abs_, sign,
  eig_max_im, eig_lmin, eig_lmax, eig_all_real,
  left, right

Pure-symbolic helpers
  _children, _expr_uses_cons_or_prim, _key   (visitors)
  _is_const, _s_add, _s_neg, _s_sub, _s_mul, _s_div, _s_pow, diff  (lowering)
"""

# -- node classes ---------------------------------------------------------
from .expr import (
    Expr, _wrap,
    Const, Var, _Bin, Add, Sub, Mul, Div, Pow, Neg, Sqrt, Abs, Sign,
    # board nodes
    Equation, _BoardNode,
    Partial, Gradient, Laplacian,
    RateTerm, _as_rate, RateExpr, Divergence,
    TimeDerivative, Unknown, OpApply, Integral,
)

# -- reference / witness values -------------------------------------------
from .values import (
    _EIG_FIELDS, _EIG_PREDICATES,
    EigWitness, StateRef, RuntimeParamRef,
)

# -- free-function ops ----------------------------------------------------
from .ops import (
    # flux-DSL
    sqrt, abs_, sign,
    eig_max_im, eig_lmin, eig_lmax, eig_all_real,
    left, right,
    # board
    grad, dx, dy, laplacian, div, ddt, rate, unknown, integral,
    board_sqrt,
)

# -- pure-symbolic helpers ------------------------------------------------
from .visitors import _children, _expr_uses_cons_or_prim, _key
from .lowering import (
    _is_const, _s_add, _s_neg, _s_sub, _s_mul, _s_div, _s_pow,
    diff,
)

__all__ = [
    # node classes
    "Expr", "Const", "Var", "_Bin", "Add", "Sub", "Mul", "Div", "Pow",
    "Neg", "Sqrt", "Abs", "Sign",
    # board nodes
    "Equation", "_BoardNode",
    "Partial", "Gradient", "Laplacian",
    "RateTerm", "RateExpr", "Divergence",
    "TimeDerivative", "Unknown", "OpApply", "Integral",
    # values
    "EigWitness", "StateRef", "RuntimeParamRef",
    # ops
    "sqrt", "abs_", "sign",
    "eig_max_im", "eig_lmin", "eig_lmax", "eig_all_real",
    "left", "right",
    "grad", "dx", "dy", "laplacian", "div", "ddt", "rate", "unknown", "integral",
    "board_sqrt",
    # helpers
    "diff",
]
