"""pops.fields.nullspace -- typed nullspace declarations for a field solve (Spec 5 sec.5.5).

A pure-Neumann / fully periodic elliptic operator has a non-trivial nullspace (the
constant function): the solution is defined up to an additive constant and the solver must
project it out for the system to be consistent. :class:`ConstantNullspace` declares that
constant nullspace; the runtime pins the mean.

Inert descriptors; they compute nothing.
"""
from pops.descriptors import Descriptor


class ConstantNullspace(Descriptor):
    """The constant-function nullspace of a pure-Neumann / periodic elliptic operator.

    Declaring it tells the solver to project the constant mode out (pin the solution mean)
    so the singular system is solved consistently.
    """

    category = "nullspace"

    def options(self):
        return {"nullspace": "constant"}

    def capabilities(self):
        return {"removes_constant": True}


__all__ = ["ConstantNullspace"]
