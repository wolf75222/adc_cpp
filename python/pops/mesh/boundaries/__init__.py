"""pops.mesh.boundaries -- domain boundary topology descriptors (Spec 5 sec.5.9 / sec.8).

These describe the *topology* of the domain boundary -- periodic vs physical, and which
face a per-face condition applies to. The field-VALUE boundary conditions (Dirichlet /
Neumann / first-order extrapolation) are a fields concern and land in ``pops.fields.bcs``
(Spec 5 Phase E); here we only name the faces and the periodic/physical split.

Inert descriptors; the runtime materialises the actual ghost fills.
"""
from .._descriptor import MeshDescriptor


class _Face(MeshDescriptor):
    """A named domain face selector (used to attach a per-face condition)."""

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


class Periodic(MeshDescriptor):
    """Periodic boundary topology (the domain wraps on the given axes, all by default)."""

    category = "boundary"

    def __init__(self, axes=None):
        self.axes = tuple(axes) if axes is not None else None

    def options(self):
        return {"axes": self.axes if self.axes is not None else "all"}

    def capabilities(self):
        return {"periodic": True}

    def requirements(self):
        return {"mesh_topology": "periodic"}


class Physical(MeshDescriptor):
    """A physical (non-periodic) boundary: a wall / outlet on the named faces."""

    category = "boundary"

    def __init__(self, kind="wall"):
        if kind not in ("wall", "outlet"):
            raise ValueError("Physical: kind must be 'wall' or 'outlet' (got %r)" % (kind,))
        self.kind = kind

    def options(self):
        return {"kind": self.kind}

    def capabilities(self):
        return {"physical": True, "kind": self.kind}


class FaceBC(MeshDescriptor):
    """Bind a boundary condition to a specific :class:`_Face` (Spec 5 sec.8.16.4)."""

    category = "face_bc"

    def __init__(self, face, condition):
        if not isinstance(face, _Face):
            raise TypeError("FaceBC: face must be a face selector (XMin/XMax/YMin/YMax), got %r"
                            % (face,))
        self.face = face
        self.condition = condition

    def options(self):
        return {"face": self.face.name, "condition": getattr(self.condition, "name",
                                                              repr(self.condition))}


__all__ = ["Periodic", "Physical", "FaceBC", "XMin", "XMax", "YMin", "YMax"]
