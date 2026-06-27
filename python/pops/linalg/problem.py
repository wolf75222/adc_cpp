"""pops.linalg.problem -- the typed linear system ``A x = b`` and its residual (Spec 5 sec.5.6).

:class:`LinearProblem` NAMES the algebraic system: an operator ``A`` (a
:class:`~pops.linalg.operator.LinearOperator` / :class:`~pops.linalg.operator.MatrixFreeOperator`),
the unknown handle ``x`` and the right-hand side ``b``. :class:`Residual` names ``b - A x`` for
a problem. Both are inert descriptors: they declare the algebra, they do NOT solve it (the
solvers live in :mod:`pops.solvers`, forthcoming) and they compute nothing.
"""
from pops.descriptors import Availability, Descriptor
from .operator import LinearOperator, MatrixFreeOperator

#: The operator types a :class:`LinearProblem` accepts for ``A``.
_OPERATOR_TYPES = (LinearOperator, MatrixFreeOperator)


def _handle_name(handle):
    """A short, stable name for an unknown / rhs handle (its ``name`` attr, else its repr)."""
    if handle is None:
        return None
    return getattr(handle, "name", repr(handle))


class LinearProblem(Descriptor):
    """The typed linear system ``A x = b`` (Spec 5 sec.5.6).

    ``LinearProblem(operator=A, unknown=x, rhs=b)`` names the algebra: the linear operator
    ``A`` (a :class:`~pops.linalg.operator.LinearOperator` or
    :class:`~pops.linalg.operator.MatrixFreeOperator`), the unknown handle ``x`` and the
    right-hand-side handle ``b``. The unknown / rhs are stored and surfaced, not interpreted,
    here. :meth:`validate` rejects an ``operator`` that is not a linear operator descriptor.
    It is inert; it does NOT solve (that is :mod:`pops.solvers`, forthcoming).
    """

    category = "linear_problem"

    def __init__(self, operator=None, unknown=None, rhs=None, name=None):
        self.operator = operator
        self.unknown = unknown
        self.rhs = rhs
        self._name = None if name is None else str(name)

    @property
    def name(self):
        return self._name if self._name is not None else type(self).__name__

    def options(self):
        return {"name": self._name,
                "operator": _handle_name(self.operator),
                "unknown": _handle_name(self.unknown),
                "rhs": _handle_name(self.rhs)}

    def requirements(self):
        return {"operator": True, "unknown": True, "rhs": True}

    def capabilities(self):
        matrix_free = bool(getattr(self.operator, "capabilities", dict)().get("matrix_free")) \
            if isinstance(self.operator, _OPERATOR_TYPES) else False
        return {"linear": True, "matrix_free": matrix_free}

    def available(self, context=None):
        if not isinstance(self.operator, _OPERATOR_TYPES):
            got = type(self.operator).__name__
            return Availability.no(
                "%s needs a LinearOperator/MatrixFreeOperator; got %r" % (self.name, got),
                missing=["operator"])
        return Availability.yes()

    def validate(self, context=None):
        if not isinstance(self.operator, _OPERATOR_TYPES):
            raise TypeError(
                "%s: operator must be a pops.linalg.LinearOperator or MatrixFreeOperator; "
                "got %r" % (self.name, type(self.operator).__name__))
        return True

    def inspect(self):
        info = super().inspect()
        info["operator"] = _handle_name(self.operator)
        info["unknown"] = _handle_name(self.unknown)
        info["rhs"] = _handle_name(self.rhs)
        return info


class Residual(Descriptor):
    """The residual ``b - A x`` of a :class:`LinearProblem` (Spec 5 sec.5.6).

    ``Residual(problem)`` names the residual vector of an ``A x = b`` system; it is the quantity
    a norm / reduction is taken of to measure convergence. It is inert: it references the problem
    and computes nothing (the runtime forms ``b - A x``).
    """

    category = "residual"

    def __init__(self, problem):
        self.problem = problem

    def options(self):
        return {"problem": _handle_name(self.problem)}

    def requirements(self):
        return {"problem": True}

    def available(self, context=None):
        if not isinstance(self.problem, LinearProblem):
            got = type(self.problem).__name__
            return Availability.no(
                "%s needs a LinearProblem; got %r" % (self.name, got), missing=["problem"])
        return Availability.yes()

    def validate(self, context=None):
        if not isinstance(self.problem, LinearProblem):
            raise TypeError(
                "%s: problem must be a pops.linalg.LinearProblem; got %r"
                % (self.name, type(self.problem).__name__))
        return True


__all__ = ["LinearProblem", "Residual"]
