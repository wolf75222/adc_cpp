"""Lightweight typed-descriptor base for the mesh / layout / AMR objects (Spec 5).

Spec 5 (sec.6) requires every object that *chooses a route* -- a layout, an AMR
policy, a geometry, a boundary, a mask -- to be a typed descriptor that can declare
its ``requirements`` / ``capabilities`` / ``options`` and answer ``available(context)``
with an *explainable* status (not just a bool). These mesh descriptors are inert: they
describe a choice the C++ runtime will materialise after validation; nothing here
computes a cell, a face or a patch.

This is a deliberately small base. The shared ``BrickDescriptor`` (Spec 5 Phase A, in
``pops.descriptors``) covers native C++ *bricks* (a flux, a limiter); the mesh objects are
a different family (a mesh, a layout, a refinement criterion), so they carry their own
minimal contract here. Spec 5 Phase D aligns both under the documented
``DescriptorProtocol``.
"""


class Availability:
    """An explainable availability status (Spec 5 sec.6: not just True/False).

    ``status`` is one of ``"yes"`` / ``"no"`` / ``"partial"``. The truthiness of an
    :class:`Availability` is ``status == "yes"`` so it still reads naturally in a
    boolean test, but the object carries the *reason* and the suggested alternatives so
    a rejection can be reported before the runtime is ever touched.
    """

    _STATUSES = ("yes", "no", "partial")

    def __init__(self, status, reason="", *, missing=None, alternatives=None):
        if status not in self._STATUSES:
            raise ValueError("Availability status must be one of %s (got %r)"
                             % (", ".join(self._STATUSES), status))
        self.status = status
        self.reason = str(reason)
        self.missing = list(missing or [])
        self.alternatives = list(alternatives or [])

    @classmethod
    def yes(cls, reason=""):
        return cls("yes", reason)

    @classmethod
    def no(cls, reason, *, missing=None, alternatives=None):
        return cls("no", reason, missing=missing, alternatives=alternatives)

    @classmethod
    def partial(cls, reason, *, missing=None, alternatives=None):
        return cls("partial", reason, missing=missing, alternatives=alternatives)

    @property
    def ok(self):
        return self.status == "yes"

    def __bool__(self):
        return self.status == "yes"

    def __repr__(self):
        return "Availability(%r, reason=%r)" % (self.status, self.reason)

    def __str__(self):
        lines = ["available: %s" % self.status]
        if self.reason:
            lines.append("  reason: %s" % self.reason)
        if self.missing:
            lines.append("  missing: %s" % ", ".join(map(str, self.missing)))
        if self.alternatives:
            lines.append("  alternatives: %s" % ", ".join(map(str, self.alternatives)))
        return "\n".join(lines)


class MeshDescriptor:
    """Base of the inert mesh / layout / AMR descriptors (Spec 5 sec.6).

    Subclasses set :attr:`category` and override :meth:`options` (and, where a route can
    be refused, :meth:`available` / :meth:`validate`). The default contract reports an
    empty requirements/capabilities set and an unconditionally-available status, so a
    plain description object is valid out of the box. :meth:`inspect` returns a plain
    dict and :meth:`__str__` a short, deterministic summary (Spec 5 sec.12.1) -- never a
    dump of runtime data.
    """

    category = "mesh"

    @property
    def name(self):
        return type(self).__name__

    def requirements(self):
        """Capabilities / build flags this object needs from the runtime route (dict)."""
        return {}

    def capabilities(self):
        """What this object provides / supports (dict)."""
        return {}

    def options(self):
        """The configurable knobs and their chosen values (dict)."""
        return {}

    def available(self, context=None):
        """An :class:`Availability` for the given route context (default: yes)."""
        return Availability.yes()

    def validate(self, context=None):
        """Raise a clear error if this object cannot be used in @p context.

        The default lifts a non-``yes`` :meth:`available` into a ``ValueError`` carrying
        the reason and alternatives, so validation is consistent with the explainable
        availability. Subclasses may add structural checks before calling ``super()``.
        """
        status = self.available(context)
        if not status.ok:
            raise ValueError("%s is not available for this route:\n%s"
                             % (self.name, status))
        return True

    def inspect(self):
        """A plain-dict view of the descriptor (name / category / options / ...)."""
        return {
            "name": self.name,
            "category": self.category,
            "options": self.options(),
            "requirements": self.requirements(),
            "capabilities": self.capabilities(),
        }

    def _summary(self):
        """Short option summary for ``str`` (subclasses may override)."""
        opts = self.options()
        return ", ".join("%s=%r" % (k, v) for k, v in opts.items())

    def __repr__(self):
        return "%s(%s)" % (self.name, self._summary())

    def __str__(self):
        head = "%s [%s]" % (self.name, self.category)
        body = self._summary()
        return "%s(%s)" % (head, body) if body else head
