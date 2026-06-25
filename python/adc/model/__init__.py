"""Operator-first type system (Spec 2, phase S2-1).

This package defines the abstract spaces and typed operators that a model-free
``adc.time.Program`` composes:

* ``StateSpace`` -- a conservative/primitive state space (the components of ``U``);
* ``FieldSpace`` -- an auxiliary or solved-field space (e.g. ``phi, grad_x, grad_y``);
* ``RateSpace`` / ``Rate(U)`` -- the tangent of a ``StateSpace`` (``dU/dt``);
* ``LocalLinearOperator(U, U)`` / ``MatrixFreeOperator`` -- operator-valued types;
* ``Signature`` -- a typed ``(inputs) -> output`` contract;
* ``Operator`` and ``OperatorRegistry`` -- a named, typed, integer-id'd registry.

These types are a TYPED VIEW: they carry no numerics and no array data. In phase
S2-1 the registry is DERIVED from an existing :class:`adc.dsl` model -- the PDE
shortcuts ``source_term`` / ``linear_source`` / ``elliptic_field`` / ``flux`` lower
into typed operators without changing the public PDE API. The public
``adc.model.Module`` front-end (S2-3), the typed ``P.call`` (S2-2) and the C++
codegen consumption (S2-6) build on these primitives in later phases.

The package imports only the standard library so it can be exercised without the
compiled ``_adc`` extension.
"""
from .bundles import RateBundle
from .module import Module
from .operators import (
    OPERATOR_KINDS,
    LocalLinearOperator,
    MatrixFreeOperator,
    Operator,
)
from .registry import OperatorRegistry
from .signatures import Signature
from .spaces import (
    AuxSpace,
    FieldSpace,
    ParameterSpace,
    Rate,
    RateSpace,
    Space,
    StateSpace,
)

__all__ = [
    "Space",
    "StateSpace",
    "FieldSpace",
    "RateSpace",
    "Rate",
    "LocalLinearOperator",
    "MatrixFreeOperator",
    "Signature",
    "Operator",
    "OperatorRegistry",
    "ParameterSpace",
    "AuxSpace",
    "Module",
    "RateBundle",
    "OPERATOR_KINDS",
]
