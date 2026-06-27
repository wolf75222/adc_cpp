"""pops.mesh.amr -- typed AMR policy descriptors (Spec 5 sec.5.11 / sec.8).

AMR is mesh / runtime infrastructure, not physics. These descriptors declare the
refinement criteria, patch clustering, regrid cadence, proper nesting, and the
checkpoint / output policy that :class:`pops.mesh.layouts.AMR` consumes and the C++
runtime executes after validation. Spec 5 (sec.8.6) replaces the string forms
(``set_refinement(0.05, variable="rho")`` / ``set_phi_refinement(0.5)``) with typed
criteria: ``Refine.on(block.role(Density)).above(0.05)`` and ``TagUnion(...)``.

Everything here is an inert descriptor; nothing tags a cell in Python.
"""
from .._descriptor import Availability, MeshDescriptor

# Current native AMR capability envelope (Spec 5 sec.8.7): the production AMR route
# supports 2 levels at refinement ratio 2. A request beyond this is refused BEFORE the
# runtime, with a clear message, rather than silently clamped.
NATIVE_MAX_LEVELS = 2
NATIVE_RATIOS = (2,)


class Refine(MeshDescriptor):
    """A typed refinement criterion (Spec 5 sec.8.6).

    Build with the fluent form ``Refine.on(subject).above(threshold)``. ``subject`` is a
    state component, a role handle, or a field expression (e.g. ``phi.gradient_norm()``);
    it is carried opaquely here and checked against the model / route at validation time.
    """

    category = "refinement_criterion"

    def __init__(self, subject, predicate=None, threshold=None):
        self.subject = subject
        self.predicate = predicate
        self.threshold = threshold

    @classmethod
    def on(cls, subject):
        return cls(subject)

    def _with(self, predicate, threshold):
        return Refine(self.subject, predicate, float(threshold))

    def above(self, threshold):
        return self._with("above", threshold)

    def below(self, threshold):
        return self._with("below", threshold)

    def gradient_above(self, threshold):
        return self._with("gradient_above", threshold)

    def magnitude_above(self, threshold):
        return self._with("magnitude_above", threshold)

    def options(self):
        subj = getattr(self.subject, "name", None) or repr(self.subject)
        return {"subject": subj, "predicate": self.predicate, "threshold": self.threshold}

    def validate(self, context=None):
        if self.predicate is None or self.threshold is None:
            raise ValueError(
                "Refine criterion is incomplete: use Refine.on(subject).above(value) "
                "(or .below / .gradient_above / .magnitude_above)")
        return True


class TagUnion(MeshDescriptor):
    """The union of several refinement criteria (a cell is tagged if ANY fires)."""

    category = "refinement_criterion"

    def __init__(self, *criteria):
        flat = []
        for c in criteria:
            if not isinstance(c, (Refine, TagUnion)):
                raise TypeError("TagUnion: every entry must be a Refine / TagUnion (got %r)" % (c,))
            flat.append(c)
        self.criteria = flat

    def options(self):
        return {"n_criteria": len(self.criteria)}

    def validate(self, context=None):
        for c in self.criteria:
            c.validate(context)
        return True


class RegridEvery(MeshDescriptor):
    """Regrid the hierarchy every N macro-steps."""

    category = "regrid_policy"

    def __init__(self, steps):
        self.steps = int(steps)
        if self.steps <= 0:
            raise ValueError("RegridEvery: steps must be > 0 (use FrozenRegrid for no regrid)")

    def options(self):
        return {"steps": self.steps}


class FrozenRegrid(MeshDescriptor):
    """No dynamic regrid: the hierarchy is built once and frozen."""

    category = "regrid_policy"

    def options(self):
        return {"frozen": True}


class PatchLayout(MeshDescriptor):
    """Patch-clustering policy: coarse distribution + max grid size."""

    category = "patch_layout"

    def __init__(self, distribute_coarse=False, coarse_max_grid=32):
        self.distribute_coarse = bool(distribute_coarse)
        self.coarse_max_grid = int(coarse_max_grid)

    def options(self):
        return {"distribute_coarse": self.distribute_coarse,
                "coarse_max_grid": self.coarse_max_grid}


class ProperNesting(MeshDescriptor):
    """Proper-nesting policy with a buffer (cells of guaranteed coarse padding)."""

    category = "nesting_policy"

    def __init__(self, buffer=1):
        self.buffer = int(buffer)
        if self.buffer < 0:
            raise ValueError("ProperNesting: buffer must be >= 0")

    def options(self):
        return {"buffer": self.buffer}


class BufferCells(MeshDescriptor):
    """Tag-buffer width: extra cells tagged around each flagged cell."""

    category = "tag_policy"

    def __init__(self, cells=1):
        self.cells = int(cells)

    def options(self):
        return {"cells": self.cells}


# --- level selection policies (shared semantics with pops.output, Spec 5 sec.5.14) -----
class AllLevels(MeshDescriptor):
    category = "level_policy"

    def options(self):
        return {"levels": "all"}


class CoarseOnly(MeshDescriptor):
    category = "level_policy"

    def options(self):
        return {"levels": "coarse"}


class SelectedLevels(MeshDescriptor):
    category = "level_policy"

    def __init__(self, *levels):
        self.levels = tuple(int(l) for l in levels)

    def options(self):
        return {"levels": self.levels}


class CheckpointPolicy(MeshDescriptor):
    """AMR checkpoint / restart policy (Spec 5 sec.8.11).

    Spec 5 keeps a single checkpoint semantics: :class:`pops.output.CheckpointPolicy` is
    the general policy and this one is the AMR-compatible specialisation. ``restartable``
    requests a bit-identical restart; the route validates whether the current native AMR
    supports it (single block / single rank / frozen regrid) before runtime.
    """

    category = "checkpoint_policy"

    def __init__(self, restartable=False, require_bit_identical=False):
        self.restartable = bool(restartable)
        self.require_bit_identical = bool(require_bit_identical)

    def options(self):
        return {"restartable": self.restartable,
                "require_bit_identical": self.require_bit_identical}


class AMROutput(MeshDescriptor):
    """AMR output policy: which fields on which levels, with patch metadata (sec.8.11)."""

    category = "amr_output"

    def __init__(self, fields=(), levels=None, include_patch_boxes=False):
        self.fields = list(fields)
        self.levels = levels if levels is not None else AllLevels()
        self.include_patch_boxes = bool(include_patch_boxes)

    def options(self):
        return {"n_fields": len(self.fields),
                "levels": self.levels.options().get("levels"),
                "include_patch_boxes": self.include_patch_boxes}


__all__ = [
    "Refine", "TagUnion", "RegridEvery", "FrozenRegrid", "PatchLayout",
    "ProperNesting", "BufferCells", "AllLevels", "CoarseOnly", "SelectedLevels",
    "CheckpointPolicy", "AMROutput", "NATIVE_MAX_LEVELS", "NATIVE_RATIOS",
    "Availability",
]
