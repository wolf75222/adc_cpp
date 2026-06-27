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


def _subject_name(subject):
    """The plain string name of a Refine subject (a string, or an object carrying ``.name``)."""
    if isinstance(subject, str):
        return subject
    name = getattr(subject, "name", None)
    return name if isinstance(name, str) else None


def _declared_subjects(model_or_context):
    """The set of subject names a model / context legitimately declares (duck-typed).

    A Refine subject is a state component, a physical role, or a named aux / field. This reads
    those names off whatever ``model_or_context`` is passed WITHOUT importing the physics / model
    layers (the mesh layer imports nothing else in pops): it tries the documented surfaces in
    turn -- conservative + primitive component names, explicit / canonical roles, named aux
    fields, and a typed ``state_space()`` view's components / role values. Returns ``None`` when
    the context exposes NONE of these surfaces, so the caller can DEFER (not falsely reject) a
    context that simply does not advertise its names.
    """
    names = set()
    found_surface = False

    def add_iter(value):
        nonlocal found_surface
        if value is None:
            return
        try:
            items = list(value)
        except TypeError:
            return
        found_surface = True
        for item in items:
            text = item if isinstance(item, str) else getattr(item, "name", None)
            if isinstance(text, str):
                names.add(text)

    for attr in ("cons_names", "prim_names", "aux_extra_names", "aux_names"):
        if hasattr(model_or_context, attr):
            add_iter(getattr(model_or_context, attr))

    # Explicit role overrides (cons_roles / prim_roles) name roles directly.
    for attr in ("cons_roles", "prim_roles"):
        value = getattr(model_or_context, attr, None)
        if value is not None:
            add_iter(value)

    # A typed StateSpace view: its components AND its role values are both legal subjects.
    space = getattr(model_or_context, "state_space", None)
    if callable(space):
        try:
            view = space()
        except Exception:  # pragma: no cover - a view that needs args is just skipped.
            view = None
        if view is not None:
            add_iter(getattr(view, "components", None))
            roles = getattr(view, "roles", None)
            if isinstance(roles, dict):
                found_surface = True
                names.update(k for k in roles if isinstance(k, str))
                names.update(v for v in roles.values() if isinstance(v, str))

    # A bare mapping / collection of declared subject names (a lightweight context).
    if isinstance(model_or_context, dict):
        for key in ("roles", "subjects", "variables", "components"):
            if key in model_or_context:
                add_iter(model_or_context[key])

    return names if found_surface else None


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
        """Validate the criterion shape and -- when a model context is given -- its subject.

        With no @p context this self-validates the predicate / threshold only and DEFERS the
        subject check to where the model is available (``problem.amr.refine`` / compile); the
        role-existence check is wired there so it happens SOMEWHERE before runtime, never never.

        When @p context is a model / context that advertises its declared subjects (state
        components, roles, named aux), a subject that is NOT among them raises a clear error
        listing the declared names. The discipline is NO FALSE POSITIVE: a context that exposes
        no subject surface (``_declared_subjects`` returns ``None``) is NOT rejected, and a
        non-string subject (a field-expression handle) is carried opaquely and skipped.
        """
        if self.predicate is None or self.threshold is None:
            raise ValueError(
                "Refine criterion is incomplete: use Refine.on(subject).above(value) "
                "(or .below / .gradient_above / .magnitude_above)")
        if context is not None:
            declared = _declared_subjects(context)
            subject = _subject_name(self.subject)
            if declared is not None and subject is not None and subject not in declared:
                raise ValueError(
                    "Refine.on(%r): %r is not a declared subject of the model; declared "
                    "subjects are %s (a role / state component / named aux). Refine on one of "
                    "those." % (subject, subject, sorted(declared) or "none"))
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
