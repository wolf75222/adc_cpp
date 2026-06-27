"""pops.linalg.reductions -- typed reduction descriptors (Spec 5 sec.5.6).

A reduction collapses one or two vectors to a scalar: the inner product ``a . b``
(:class:`Dot`) and the 2-norm ``||x||_2`` (:class:`Norm2`). These are inert descriptors that
NAME the reduction and reference their operands; they do NOT compute (no numpy, no loop). The
helpers :func:`dot` / :func:`norm2` build the descriptor referencing the operands -- they are
authoring sugar, not a numeric kernel. The C++ runtime performs the reduction.
"""
from pops.descriptors import Descriptor


def _operand_name(operand):
    """A short, stable name for a reduction operand (its ``name`` attr, else its repr)."""
    if operand is None:
        return None
    return getattr(operand, "name", repr(operand))


class Dot(Descriptor):
    """The inner product ``a . b`` of two vectors, named (not computed).

    ``Dot(a, b)`` references the two operand handles; it is inert and computes nothing (the
    runtime forms the reduction). Build it with :func:`dot`.
    """

    category = "reduction"

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def options(self):
        return {"op": "dot", "a": _operand_name(self.a), "b": _operand_name(self.b)}

    def requirements(self):
        return {"operands": 2}


class Norm2(Descriptor):
    """The 2-norm reduction ``||x||_2`` of one vector, named (not computed).

    ``Norm2(x)`` references the operand handle; it is the reduction (``sqrt(x . x)``), distinct
    from the typed norm SELECTOR :class:`pops.linalg.norms.L2` (which only names WHICH norm). It
    is inert and computes nothing. Build it with :func:`norm2`.
    """

    category = "reduction"

    def __init__(self, x):
        self.x = x

    def options(self):
        return {"op": "norm2", "x": _operand_name(self.x)}

    def requirements(self):
        return {"operands": 1}


def dot(a, b):
    """Build the :class:`Dot` reduction descriptor naming the inner product ``a . b``.

    This is inert authoring sugar -- it constructs a descriptor referencing @p a and @p b; it
    does NOT compute the inner product (the runtime does).
    """
    return Dot(a, b)


def norm2(x):
    """Build the :class:`Norm2` reduction descriptor naming the 2-norm ``||x||_2``.

    This is inert authoring sugar -- it constructs a descriptor referencing @p x; it does NOT
    compute the norm (the runtime does).
    """
    return Norm2(x)


__all__ = ["Dot", "Norm2", "dot", "norm2"]
