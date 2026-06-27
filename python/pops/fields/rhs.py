"""pops.fields.rhs -- typed right-hand-side sources for a field solve (Spec 5 sec.5.5).

The right-hand side of an elliptic field solve (the source of a Poisson problem) is a
typed descriptor rather than a raw array. :class:`ChargeDensity` describes the charge
density assembled from the contributing physics blocks; :meth:`ChargeDensity.from_blocks`
builds it from the named blocks that deposit charge.

Inert descriptors; the runtime assembles the actual density field.
"""
from pops.descriptors import Descriptor


class ChargeDensity(Descriptor):
    """The charge-density right-hand side of a Poisson-type field solve.

    Built with :meth:`from_blocks` from the names of the physics blocks that deposit
    charge; the runtime assembles the density from those blocks' conserved state.
    """

    category = "rhs"

    def __init__(self, blocks=()):
        self.blocks = tuple(str(b) for b in blocks)

    @classmethod
    def from_blocks(cls, *blocks):
        """A :class:`ChargeDensity` summed over the named contributing :paramref:`blocks`.

        Accepts the block names either as varargs (``from_blocks("ions", "electrons")``)
        or as a single iterable (``from_blocks(["ions", "electrons"])``).
        """
        if len(blocks) == 1 and not isinstance(blocks[0], str) and hasattr(blocks[0], "__iter__"):
            blocks = tuple(blocks[0])
        return cls(blocks=blocks)

    def options(self):
        return {"rhs": "charge_density", "blocks": self.blocks}

    def requirements(self):
        return {"blocks": list(self.blocks)}


__all__ = ["ChargeDensity"]
