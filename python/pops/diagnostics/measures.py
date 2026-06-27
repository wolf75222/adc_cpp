"""pops.diagnostics.measures -- typed diagnostic-measure descriptors (Spec 5 sec.5.13 / 14.2.7).

Spec 5 names a diagnostic with a TYPED object, not the string form
``diagnostics.norm(kind="l2")``. :class:`Norm` / :class:`Integral` / :class:`MinMax` /
:class:`ConservationCheck` are those objects -- inert descriptors that DESCRIBE a scalar
reduction over a block (and an optional model role): the reduction kind, whether it needs an
MPI reduction, its cadence slot and its AMR / multi-level compatibility, all carried as
METADATA. They compute nothing; the C++ / Kokkos / MPI runtime evaluates the reduction.

Each typed measure lowers to an inert native reduction scheme label (``norm`` / ``integral`` /
``min_max`` / ``conservation_check``). ``norm`` and ``integral`` reuse the same-name
:mod:`pops.diagnostics` factory scheme; ``conservation_check`` matches the
:mod:`pops.diagnostics.invariants` reduction; ``min_max`` is a new inert label. No native symbol
is fabricated -- the labels name the reduction the C++ runtime evaluates. The :class:`Norm`
measure takes a typed norm kind from :mod:`pops.linalg.norms` (``L1`` / ``L2`` / ``LInf``); a
bare string is rejected.
"""
from pops.descriptors import Availability, Descriptor
from pops.linalg.norms import _Norm


def _ref_name(value):
    """The stable display name for a block / role reference (its ``name`` or its repr).

    A block is named by a string and a role by a typed role object (carrying a ``name``); the
    measure references it WITHOUT interpreting it. ``None`` stays ``None`` so an unscoped
    measure (whole-domain, default role) reads cleanly.
    """
    if value is None:
        return None
    return getattr(value, "name", None) or (value if isinstance(value, str) else repr(value))


class _Measure(Descriptor):
    """Base of the typed diagnostic measures: a scalar reduction over a block / role.

    A measure stores its (opaque) ``block`` and ``role`` references and surfaces them by name;
    it never reads a cell. Subclasses set :attr:`category` + :attr:`scheme` (the native
    reduction identity shared with the legacy factory) and declare their reduction metadata via
    :meth:`capabilities` / :meth:`requirements`. ``cadence`` is any inert schedule object (e.g.
    ``pops.time.schedule.every(20)``) or an int step interval; it is stored, not interpreted.
    """

    #: The native diagnostic reduction scheme this measure lowers to (an inert authoring label;
    #: ``norm`` / ``integral`` reuse the same-name legacy factory scheme, ``min_max`` is new).
    #: Subclasses set it.
    scheme = None
    #: The reduction kind this measure performs ("sum" / "norm" / "min_max" / "check").
    reduction = None

    def __init__(self, block=None, role=None, cadence=None):
        self.block = block
        self.role = role
        self.cadence = cadence

    def options(self):
        return {"scheme": self.scheme, "block": _ref_name(self.block),
                "role": _ref_name(self.role), "cadence": _ref_name(self.cadence)}

    def requirements(self):
        # A scalar reduction over a distributed mesh needs an MPI all-reduce to be correct.
        return {"mpi_reduction": True}

    def capabilities(self):
        # Reduction metadata, declared not computed: a single scalar reduction is AMR /
        # multi-level safe (it sums / folds across levels) and runs on the diagnostic cadence.
        return {"reduction": self.reduction, "mpi_reduction": True,
                "amr_compatible": True, "multi_level": True, "cadence_slot": "diagnostic"}

    def lower(self, context=None):
        rec = super().lower(context)
        rec["scheme"] = self.scheme
        return rec

    def inspect(self):
        info = super().inspect()
        info["scheme"] = self.scheme
        return info


class Norm(_Measure):
    """A typed norm reduction over a block: ``Norm(L2(), block=..., role=...)``.

    The norm kind is a typed :mod:`pops.linalg.norms` object (``L1`` / ``L2`` / ``LInf``), NOT
    the string ``kind="l2"`` (Spec 5 sec.7 rejects a free-string selector). The measure lowers
    to the native ``norm`` reduction the legacy ``diagnostics.norm`` factory already names.
    """

    category = "diagnostic_norm"
    scheme = "norm"
    reduction = "norm"

    def __init__(self, norm, block=None, role=None, cadence=None):
        if not isinstance(norm, _Norm):
            from pops.descriptors import reject_string_selector
            if isinstance(norm, str):
                reject_string_selector(norm, "norm", "pops.linalg.norms.L2()")
            raise TypeError(
                "Norm(norm=...) takes a typed pops.linalg.norms object (L1 / L2 / LInf), "
                "got %r. Spec 5 forbids a string norm selector." % (norm,))
        super().__init__(block=block, role=role, cadence=cadence)
        self.norm = norm

    def options(self):
        opts = super().options()
        opts["norm"] = self.norm.kind
        return opts

    def capabilities(self):
        caps = super().capabilities()
        caps["norm_kind"] = self.norm.kind
        return caps


class Integral(_Measure):
    """A typed domain-integral reduction over a block: ``Integral(role=Density)``.

    Sums the (role-selected) quantity over the block volume; ``mass`` is
    ``Integral(role=Density)``. Lowers to the native ``integral`` reduction the legacy
    ``diagnostics.integral`` factory already names.
    """

    category = "diagnostic_integral"
    scheme = "integral"
    reduction = "sum"


class MinMax(_Measure):
    """A typed min / max reduction over a block: ``MinMax(block=..., role=...)``.

    Reports the (role-selected) extrema over the block. Lowers to the native ``min_max``
    reduction. Unlike a sum / norm it is a fold, but it is equally MPI- and AMR-safe.
    """

    category = "diagnostic_minmax"
    scheme = "min_max"
    reduction = "min_max"


class ConservationCheck(Descriptor):
    """A typed conservation check on a diagnostic quantity: ``ConservationCheck(Integral(...))``.

    Names a tolerance check that the runtime applies to a measured ``quantity`` (a diagnostic
    measure descriptor, e.g. an :class:`Integral`) -- the drift of that quantity must stay
    within ``tolerance``. ``quantity`` MUST be a diagnostic descriptor (Spec 5 sec.5.13); a
    string or anything else is rejected. The check itself computes nothing; the runtime
    measures the quantity and compares the drift.
    """

    category = "conservation_check"
    scheme = "conservation_check"

    def __init__(self, quantity, tolerance=1e-12):
        self.quantity = quantity
        self.tolerance = float(tolerance)

    def options(self):
        return {"scheme": self.scheme, "quantity": _ref_name(self.quantity),
                "tolerance": self.tolerance}

    def requirements(self):
        # The checked quantity is itself a reduction, so the check inherits its MPI need.
        return {"mpi_reduction": True, "quantity": True}

    def capabilities(self):
        return {"reduction": "check", "mpi_reduction": True, "amr_compatible": True,
                "multi_level": True, "cadence_slot": "diagnostic", "tolerance": self.tolerance}

    def available(self, context=None):
        if not isinstance(self.quantity, Descriptor):
            return Availability.no(
                "ConservationCheck(quantity=...) needs a diagnostic descriptor "
                "(e.g. Integral(role=Density)), got %r" % (self.quantity,),
                missing=["quantity"],
                alternatives=["Integral(...)", "Norm(L2(), ...)", "MinMax(...)"])
        return Availability.yes()

    def validate(self, context=None):
        status = self.available(context)
        if not status.ok:
            raise TypeError("%s is not valid:\n%s" % (self.name, status))
        return True

    def lower(self, context=None):
        rec = super().lower(context)
        rec["scheme"] = self.scheme
        return rec

    def inspect(self):
        info = super().inspect()
        info["scheme"] = self.scheme
        return info


__all__ = ["Norm", "Integral", "MinMax", "ConservationCheck"]
