"""pops.output.formats -- typed output-format descriptors (Spec 5 sec.5.14).

A format is a typed object (``HDF5()`` / ``Plotfile()``), not a string ``format="hdf5"``.
Inert; the runtime writes the actual files.
"""
from pops.descriptors import Descriptor


class HDF5(Descriptor):
    """HDF5 output. ``parallel=True`` requests the parallel-HDF5 path (build-dependent)."""

    category = "output_format"

    def __init__(self, parallel=False):
        self.parallel = bool(parallel)

    def options(self):
        return {"parallel": self.parallel}

    def requirements(self):
        return {"parallel_io": True} if self.parallel else {}


class Plotfile(Descriptor):
    """AMReX-style plotfile output (per-level directories)."""

    category = "output_format"

    def capabilities(self):
        return {"per_level": True}


__all__ = ["HDF5", "Plotfile"]
