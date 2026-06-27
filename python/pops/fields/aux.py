"""pops.fields.aux -- field-side aux declarations (Spec 5 sec.14.2.4).

A field problem can declare auxiliary fields it reads or derives: a :class:`StaticAux`
holds a fixed value supplied once, a :class:`DerivedAux` is computed from a PoPS
expression (in C++, not Python). The per-field aux halo / ghost policy
:class:`pops.mesh.aux.AuxHalo` is re-exported here under the name Spec 5 sec.14.2.4 uses
(``pops.fields.aux.AuxHalo``); it is the SAME mesh descriptor, not a copy.

Inert descriptors; they compute nothing.
"""
from pops.descriptors import Descriptor


class StaticAux(Descriptor):
    """A static auxiliary field named :paramref:`name`, holding a fixed :paramref:`value`."""

    category = "aux"

    def __init__(self, name, value=None):
        self._name = str(name)
        self.value = value

    @property
    def name(self):
        return self._name

    def options(self):
        return {"name": self._name, "kind": "static", "value": self.value}


class DerivedAux(Descriptor):
    """An auxiliary field named :paramref:`name`, derived from a PoPS :paramref:`expression`.

    The expression is stored verbatim and evaluated in C++ by the runtime, not in Python.
    """

    category = "aux"

    def __init__(self, name, expression=None):
        self._name = str(name)
        self.expression = expression

    @property
    def name(self):
        return self._name

    def options(self):
        return {"name": self._name, "kind": "derived",
                "expression": getattr(self.expression, "name", repr(self.expression))
                if self.expression is not None else None}


# Re-export the per-field aux halo descriptor under the Spec 5 sec.14.2.4 name
# ``pops.fields.aux.AuxHalo``. It is the SAME mesh descriptor (a thin re-export, not a
# copy). Resolved lazily via module __getattr__ so ``pops.fields`` keeps importing only
# pops.descriptors at module scope (the mesh package is not pulled in until AuxHalo is
# actually referenced).
def __getattr__(name):
    if name == "AuxHalo":
        from pops.mesh.aux import AuxHalo
        return AuxHalo
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


__all__ = ["StaticAux", "DerivedAux", "AuxHalo"]
