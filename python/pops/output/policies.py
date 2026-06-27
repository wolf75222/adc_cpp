"""pops.output.policies -- output / checkpoint / level descriptors (Spec 5 sec.5.14 / 8.11).

Typed replacements for ``output(format="hdf5", every=20)`` / ``checkpoint(mode=...)``. An
output or checkpoint policy declares its format, cadence, fields, diagnostics and level
selection; the runtime performs the I/O. Inert descriptors.
"""
from pops.descriptors import Descriptor


class _LevelPolicy(Descriptor):
    category = "level_policy"


class AllLevels(_LevelPolicy):
    def options(self):
        return {"levels": "all"}


class CoarseOnly(_LevelPolicy):
    def options(self):
        return {"levels": "coarse"}


class SelectedLevels(_LevelPolicy):
    def __init__(self, *levels):
        self.levels = tuple(int(l) for l in levels)

    def options(self):
        return {"levels": self.levels}


class OutputPolicy(Descriptor):
    """An output policy: a format, a cadence, the fields/diagnostics, and the level selection.

    ``OutputPolicy(format=HDF5(), cadence=every(20), fields=[phi, E], levels=AllLevels())``.
    ``cadence`` is any inert schedule object (e.g. ``pops.time.schedule.every(20)``) or an int
    step interval; it is stored, not interpreted, here.
    """

    category = "output_policy"

    def __init__(self, format=None, cadence=None, fields=(), diagnostics=(),
                 levels=None, require_parallel=False):
        self.format = format
        self.cadence = cadence
        self.fields = list(fields)
        self.diagnostics = list(diagnostics)
        self.levels = levels if levels is not None else AllLevels()
        self.require_parallel = bool(require_parallel)

    def options(self):
        return {"format": getattr(self.format, "name", self.format),
                "cadence": getattr(self.cadence, "name", self.cadence),
                "n_fields": len(self.fields), "n_diagnostics": len(self.diagnostics),
                "levels": self.levels.options().get("levels"),
                "require_parallel": self.require_parallel}

    def requirements(self):
        req = {}
        if self.require_parallel:
            req["parallel_io"] = True
        # Union the chosen format's own requirements (e.g. HDF5(parallel=True) -> parallel_io).
        if self.format is not None and hasattr(self.format, "requirements"):
            req.update(self.format.requirements())
        return req


class CheckpointPolicy(Descriptor):
    """The general checkpoint / restart policy (Spec 5 sec.8.4 / 8.11).

    ``CheckpointPolicy(cadence=every(100), restartable=True)``. This is the single general
    policy; :class:`pops.mesh.amr.CheckpointPolicy` is the AMR-compatible specialisation
    (Spec 5 Phase D reconciles them so there is one semantics, not two divergent APIs).
    """

    category = "checkpoint_policy"

    def __init__(self, cadence=None, restartable=False, require_bit_identical=False):
        self.cadence = cadence
        self.restartable = bool(restartable)
        self.require_bit_identical = bool(require_bit_identical)

    def options(self):
        return {"cadence": getattr(self.cadence, "name", self.cadence),
                "restartable": self.restartable,
                "require_bit_identical": self.require_bit_identical}


__all__ = ["OutputPolicy", "CheckpointPolicy", "AllLevels", "CoarseOnly", "SelectedLevels"]
