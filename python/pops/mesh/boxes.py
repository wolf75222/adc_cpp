"""pops.mesh.boxes -- patch-box / box-layout descriptors (Spec 5 sec.5.9).

A ``PatchBox`` names a rectangular index region [lo, hi] (inclusive integer corners); a
``BoxLayout`` names a set of boxes. They describe how a level is tiled; the runtime
materialises the actual ``BoxArray`` / ``DistributionMapping``. Inert descriptors.
"""
from ._descriptor import MeshDescriptor


class PatchBox(MeshDescriptor):
    """A rectangular index box ``[lo, hi]`` (inclusive integer corners, per dimension)."""

    category = "patch_box"

    def __init__(self, lo, hi):
        lo = tuple(int(x) for x in lo)
        hi = tuple(int(x) for x in hi)
        if len(lo) != len(hi):
            raise ValueError("PatchBox: lo and hi must have the same dimension")
        if any(h < l for l, h in zip(lo, hi)):
            raise ValueError("PatchBox: hi must be >= lo per dimension (got lo=%r hi=%r)"
                             % (lo, hi))
        self.lo = lo
        self.hi = hi

    @property
    def shape(self):
        return tuple(h - l + 1 for l, h in zip(self.lo, self.hi))

    def options(self):
        return {"lo": self.lo, "hi": self.hi, "shape": self.shape}


class BoxLayout(MeshDescriptor):
    """A set of :class:`PatchBox` covering (part of) a level."""

    category = "box_layout"

    def __init__(self, boxes=()):
        boxes = list(boxes)
        for b in boxes:
            if not isinstance(b, PatchBox):
                raise TypeError("BoxLayout: every entry must be a PatchBox (got %r)" % (b,))
        self.boxes = boxes

    def options(self):
        return {"n_boxes": len(self.boxes)}

    def __iter__(self):
        return iter(self.boxes)

    def __len__(self):
        return len(self.boxes)
