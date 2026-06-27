"""pops.mesh.masks -- transport-mask descriptors for embedded boundaries (Spec 5 sec.8.16.1).

The typed replacement for the ``set_disc_domain(..., mode="none"|"staircase"|"cutcell")``
string. A mask says HOW transport is masked at an embedded boundary; the runtime applies
it. Inert descriptors.
"""
from .._descriptor import MeshDescriptor


class NoMask(MeshDescriptor):
    """No masking: the embedded geometry is ignored by transport (mode='none')."""

    category = "transport_mask"

    def capabilities(self):
        return {"masked_transport": False}


class Staircase(MeshDescriptor):
    """Staircase masking: cells fully inside the wall are excluded (mode='staircase')."""

    category = "transport_mask"

    def capabilities(self):
        return {"masked_transport": True, "conservative": False}


class CutCell(MeshDescriptor):
    """Cut-cell masking: conservative masked transport on cut cells (mode='cutcell')."""

    category = "transport_mask"

    def requirements(self):
        return {"embedded_boundary_support": True}

    def capabilities(self):
        return {"masked_transport": True, "conservative": True}


__all__ = ["NoMask", "Staircase", "CutCell"]
