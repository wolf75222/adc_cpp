"""pops.params -- typed scalar / small-object parameters (Spec 5 sec.5.12).

A parameter declares whether it is compile-time (:class:`ConstParam`) or runtime
(:class:`RuntimeParam`), its typed dtype (:mod:`pops.math`), an optional default and an
optional typed :mod:`~pops.params.constraints` domain -- instead of the string form
``Param(kind="runtime")`` / ``domain="positive"``. Constants with units live in
:mod:`pops.params.constants`. Everything here is inert; the codegen / runtime consume the
descriptors (a runtime param appears in ``compiled.arguments()``; a const param is in the
cache key).
"""
from .runtime import RuntimeParam, ConstParam, DerivedParam
from .constraints import Constraint, Positive, NonNegative, Range, In
from .constants import Constant
from . import constraints, constants

__all__ = [
    "RuntimeParam", "ConstParam", "DerivedParam",
    "Constraint", "Positive", "NonNegative", "Range", "In",
    "Constant", "constraints", "constants",
]
