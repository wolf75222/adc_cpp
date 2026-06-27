"""pops.fields.bcs -- field-VALUE boundary conditions (Spec 5 sec.5.5 / sec.8.16.4).

These describe the boundary condition imposed on a FIELD value: periodic wrap, a
Dirichlet value, a Neumann flux, or first-order extrapolation. The mesh-side
:mod:`pops.mesh.boundaries` owns the domain TOPOLOGY (which faces are periodic vs
physical); this module owns what value/flux a field takes on a face.

:class:`FaceBC` binds one of these conditions to a named domain face. The face selectors
(:class:`XMin` / :class:`XMax` / :class:`YMin` / :class:`YMax`) live here too so the
``fields`` package stays self-contained (it imports only :mod:`pops.descriptors`).

Inert descriptors; the runtime materialises the actual ghost fills.
"""
from pops.descriptors import Descriptor


class _Face(Descriptor):
    """A named domain face selector (used to attach a per-face field condition)."""

    category = "face"
    axis = -1
    side = ""

    def options(self):
        return {"axis": self.axis, "side": self.side}


class XMin(_Face):
    axis, side = 0, "lo"


class XMax(_Face):
    axis, side = 0, "hi"


class YMin(_Face):
    axis, side = 1, "lo"


class YMax(_Face):
    axis, side = 1, "hi"


class Periodic(Descriptor):
    """A periodic field boundary: the field wraps across the boundary."""

    category = "field_bc"

    def options(self):
        return {"bc": "periodic"}

    def capabilities(self):
        return {"periodic": True}


class Dirichlet(Descriptor):
    """A Dirichlet field boundary: the field value is fixed to :paramref:`value`.

    ``on`` optionally restricts the condition to a face selector (an instance of a
    :class:`_Face` subclass); ``None`` means all physical faces.
    """

    category = "field_bc"

    def __init__(self, value=0.0, on=None):
        self.value = float(value)
        self.on = on

    def options(self):
        return {"bc": "dirichlet", "value": self.value,
                "on": getattr(self.on, "name", self.on)}


class Neumann(Descriptor):
    """A Neumann field boundary: the normal flux/derivative is fixed to :paramref:`value`.

    ``on`` optionally restricts the condition to a face selector; ``None`` means all
    physical faces.
    """

    category = "field_bc"

    def __init__(self, value=0.0, on=None):
        self.value = float(value)
        self.on = on

    def options(self):
        return {"bc": "neumann", "value": self.value,
                "on": getattr(self.on, "name", self.on)}


class FirstOrderExtrapolation(Descriptor):
    """A zero-gradient field boundary (ghost cell mirrors the interior cell)."""

    category = "field_bc"

    def options(self):
        return {"bc": "foextrap"}


class FaceBC(Descriptor):
    """Bind a field boundary :paramref:`condition` to a specific :paramref:`face` selector."""

    category = "face_bc"

    def __init__(self, face, condition):
        if not isinstance(face, _Face):
            raise TypeError(
                "FaceBC: face must be a face selector (XMin/XMax/YMin/YMax), got %r" % (face,))
        self.face = face
        self.condition = condition

    def options(self):
        return {"face": self.face.name,
                "condition": getattr(self.condition, "name", repr(self.condition))}


__all__ = ["Periodic", "Dirichlet", "Neumann", "FirstOrderExtrapolation", "FaceBC",
           "XMin", "XMax", "YMin", "YMax"]
