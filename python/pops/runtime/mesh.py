"""pops.runtime.mesh -- transitional re-export of the mesh objects (Spec 5 sec.5.9).

Spec 5 lifts the mesh descriptors out of the runtime layer into the top-level
:mod:`pops.mesh` package (so ``from pops.mesh import CartesianMesh`` is the canonical
form). This module re-exports them so the runtime facade (``pops.System``,
``pops.CartesianMesh``) keeps working unchanged while consumers migrate to ``pops.mesh``.
"""
from pops.mesh.cartesian import CartesianMesh  # noqa: F401
from pops.mesh.polar import PolarMesh  # noqa: F401
from pops.mesh.aux import AuxHalo  # noqa: F401

__all__ = ["CartesianMesh", "PolarMesh", "AuxHalo"]
