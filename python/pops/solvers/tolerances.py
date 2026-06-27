"""pops.solvers.tolerances -- typed convergence-tolerance descriptors (Spec 5 sec.5.7).

A solver stops when its residual meets a TYPED criterion, not the bare keyword
``tol=1e-6``. :class:`Relative` names the relative test ``||r_k|| <= rel * ||r_0||`` with an
optional :class:`AbsoluteFloor`; :class:`Absolute` names ``||r_k|| <= abs``. They are inert
descriptors: they record the chosen bound and compute nothing (the C++ solver evaluates the
residual and the test). They are the sub-descriptors a solver's ``tolerance=`` keyword takes
(e.g. :class:`pops.solvers.elliptic.GeometricMG`).
"""
from pops.descriptors import Descriptor


class AbsoluteFloor(Descriptor):
    """An absolute floor on a relative tolerance: stop also when ``||r_k|| <= abs``.

    It is the optional ``floor=`` of :class:`Relative` -- a guard so a relative test does not
    chase an unreachable target on an already-tiny residual. Inert; computes nothing.
    """

    category = "tolerance_floor"
    native_id = None

    def __init__(self, abs_floor=1e-12):
        self.abs_floor = float(abs_floor)

    @property
    def name(self):
        return "absolute_floor"

    def options(self):
        return {"abs_floor": self.abs_floor}

    def lower(self, context=None):
        return {"kind": "absolute_floor", "abs_floor": self.abs_floor}


class Relative(Descriptor):
    """A relative convergence test ``||r_k|| <= rel * ||r_0||`` with an optional absolute floor.

    ``Relative(rel=1e-6, floor=AbsoluteFloor(1e-12))`` names the criterion; the C++ solver
    forms the residual norms and applies the test. The floor, when given, must be an
    :class:`AbsoluteFloor` -- a bare number / string is rejected loud (Spec 5 sec.7). Inert.
    """

    category = "tolerance"
    native_id = None

    def __init__(self, rel=1e-6, floor=None):
        self.rel = float(rel)
        if floor is not None and not isinstance(floor, AbsoluteFloor):
            raise TypeError(
                "Relative(floor=) must be a pops.solvers.tolerances.AbsoluteFloor(...) "
                "descriptor, not %r; use AbsoluteFloor(1e-12)." % (floor,))
        self.floor = floor

    @property
    def name(self):
        return "relative"

    def options(self):
        opt = {"rel": self.rel}
        if self.floor is not None:
            opt["abs_floor"] = self.floor.abs_floor
        return opt

    def lower(self, context=None):
        rec = {"kind": "relative", "rel": self.rel}
        if self.floor is not None:
            rec["floor"] = self.floor.lower(context)
        return rec


class Absolute(Descriptor):
    """An absolute convergence test ``||r_k|| <= abs``, named (not computed)."""

    category = "tolerance"
    native_id = None

    def __init__(self, abs_tol=1e-6):
        self.abs_tol = float(abs_tol)

    @property
    def name(self):
        return "absolute"

    def options(self):
        return {"abs_tol": self.abs_tol}

    def lower(self, context=None):
        return {"kind": "absolute", "abs_tol": self.abs_tol}


__all__ = ["Relative", "Absolute", "AbsoluteFloor"]
