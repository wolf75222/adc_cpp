"""pops.mesh.geometry -- embedded-geometry descriptors (Spec 5 sec.5.9 / sec.8.16.1).

Typed replacements for the string ``set_disc_domain(..., mode=...)`` form: a geometry
object (Disc / HalfPlane / LevelSet) provides a boundary predicate / level set, and
:class:`EmbeddedBoundary` pairs it with a transport mask. Inert descriptors; the runtime
builds the actual cut-cell / staircase geometry after validation.
"""
from .._descriptor import MeshDescriptor


class _Boundary(MeshDescriptor):
    """A handle to the boundary of a geometry (target of a field boundary condition)."""

    category = "geometry_boundary"

    def __init__(self, geometry):
        self.geometry = geometry

    def options(self):
        return {"of": self.geometry.name}


class _Geometry(MeshDescriptor):
    category = "geometry"

    def boundary(self):
        """The boundary of this geometry (e.g. ``Dirichlet(on=wall.boundary())``)."""
        return _Boundary(self)

    def capabilities(self):
        return {"provides": "level_set"}


class Disc(_Geometry):
    """A disc wall: center + radius (the embedded boundary is the circle)."""

    def __init__(self, center=(0.0, 0.0), radius=0.5):
        self.center = tuple(float(c) for c in center)
        self.radius = float(radius)
        if self.radius <= 0.0:
            raise ValueError("Disc: radius must be > 0 (got %r)" % (self.radius,))

    def options(self):
        return {"center": self.center, "radius": self.radius}


class HalfPlane(_Geometry):
    """A half-plane wall: a point on the plane + an outward normal."""

    def __init__(self, point=(0.0, 0.0), normal=(1.0, 0.0)):
        self.point = tuple(float(c) for c in point)
        self.normal = tuple(float(c) for c in normal)

    def options(self):
        return {"point": self.point, "normal": self.normal}


class LevelSet(_Geometry):
    """A generic level-set geometry (the wall is {phi(x) == 0})."""

    def __init__(self, expression):
        self.expression = expression

    def options(self):
        return {"expression": getattr(self.expression, "name", repr(self.expression))}


class EmbeddedBoundary(MeshDescriptor):
    """An embedded boundary = a geometry + a transport mask (Spec 5 sec.8.16.1).

    Passed to a layout: ``Uniform(mesh, embedded_boundary=EmbeddedBoundary(wall, CutCell()))``.
    Declares it needs embedded-boundary support in the spatial scheme + a compatible
    field/boundary route; the runtime materialises the masked transport.
    """

    category = "mesh_feature"

    def __init__(self, domain, transport):
        self.domain = domain
        self.transport = transport

    def options(self):
        return {"domain": self.domain.name, "transport": self.transport.name}

    def requirements(self):
        return {"embedded_boundary_support": True,
                "geometry": self.domain.name, "transport_mask": self.transport.name}


__all__ = ["Disc", "HalfPlane", "LevelSet", "EmbeddedBoundary"]
