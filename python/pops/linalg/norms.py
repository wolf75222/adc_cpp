"""pops.linalg.norms -- typed vector-norm descriptors (Spec 5 sec.5.6).

Spec 5 names a norm with a TYPED object, not the string ``norm="l2"``. :class:`L1` /
:class:`L2` / :class:`LInf` are those objects -- tiny inert descriptors that a consumer (a
solver tolerance, ``pops.diagnostics.Norm(L2(), ...)``, a convergence test) references to say
WHICH norm to take. They compute nothing; the runtime evaluates the norm.
"""
from pops.descriptors import Descriptor


class _Norm(Descriptor):
    """Base of the typed norm descriptors: an inert object naming one vector norm."""

    category = "norm"
    #: The native / IR tag for this norm (``"l1"`` / ``"l2"`` / ``"linf"``).
    kind = None

    def options(self):
        return {"kind": self.kind}


class L1(_Norm):
    """The 1-norm ``sum |x_i|`` (typed; inert)."""

    kind = "l1"


class L2(_Norm):
    """The Euclidean 2-norm ``sqrt(sum x_i^2)`` (typed; inert)."""

    kind = "l2"


class LInf(_Norm):
    """The maximum norm ``max |x_i|`` (typed; inert)."""

    kind = "linf"


__all__ = ["L1", "L2", "LInf"]
