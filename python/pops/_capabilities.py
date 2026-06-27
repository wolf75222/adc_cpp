"""pops._capabilities -- the descriptor-sourced capability matrix (Spec 5 sec.6 / sec.13.12.1).

:func:`inspect_capabilities` walks the inert descriptor catalogs (the Riemann / reconstruction
/ limiter / projection bricks, the mesh layouts, the solver / field catalogs) and reports, per
entry, its name / category / native id / availability / requirements. It is PURE: it imports
only the pure-stdlib authoring packages, never ``_pops``, and runs nothing -- it instantiates
each catalogued descriptor and reads its declared metadata.

This is the introspectable counterpart of the hand-written ``pops.capabilities()`` (the runtime
doctor's dispatch table): that one mirrors what the compiled runtime can dispatch, this one is
sourced straight from the typed descriptors, so the two cannot silently disagree about which
bricks exist.
"""


class CapabilityEntry:
    """One row of the capability matrix: a catalogued descriptor's declared metadata.

    A plain value -- name / category / native_id / available (an ``Availability`` status string)
    / requirements / source -- read from an inert descriptor. It computes nothing. ``source`` is
    ``"descriptor"`` for a row read from the Python catalog and ``"native"`` for a row sourced
    from the C++ ``_pops.module_capabilities()`` authoritative facts (Spec 5 sec.13.12).
    """

    def __init__(self, name, category, native_id, available, requirements, source="descriptor"):
        self.name = name
        self.category = category
        self.native_id = native_id
        self.available = available
        self.requirements = dict(requirements or {})
        self.source = source

    def to_dict(self):
        return {"name": self.name, "category": self.category, "native_id": self.native_id,
                "available": self.available, "requirements": self.requirements,
                "source": self.source}

    def __repr__(self):
        return ("CapabilityEntry(name=%r, category=%r, native_id=%r, available=%r, source=%r)"
                % (self.name, self.category, self.native_id, self.available, self.source))


class CapabilityMatrix:
    """The structured, printable result of :func:`inspect_capabilities`.

    Holds the :class:`CapabilityEntry` rows grouped by category; :meth:`to_dict` returns a
    plain nested dict and :meth:`__str__` a short, deterministic table. It is inert.
    """

    def __init__(self, entries):
        self.entries = list(entries)

    def categories(self):
        return sorted({e.category for e in self.entries})

    def by_category(self, category):
        return [e for e in self.entries if e.category == category]

    def to_dict(self):
        out = {}
        for entry in self.entries:
            out.setdefault(entry.category, []).append(entry.to_dict())
        return out

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return "CapabilityMatrix(%d entries, %d categories)" % (
            len(self.entries), len(self.categories()))

    def __str__(self):
        lines = ["capability matrix (%d entries):" % len(self.entries)]
        for category in self.categories():
            lines.append("  [%s]" % category)
            for entry in self.by_category(category):
                native = entry.native_id or "-"
                lines.append("    %-18s available=%-7s source=%-10s native_id=%s"
                             % (entry.name, entry.available, entry.source, native))
        return "\n".join(lines)


def _availability_status(descriptor):
    """The Availability status string of a descriptor (always defined; no context needed)."""
    try:
        return descriptor.available().status
    except Exception:  # a descriptor whose availability needs a context is reported as unknown.
        return "unknown"


def _entry_from_brick(descriptor):
    """A :class:`CapabilityEntry` from a :class:`pops.descriptors.BrickDescriptor`."""
    status = "yes" if descriptor.available else "no"
    return CapabilityEntry(descriptor.name, descriptor.category,
                           descriptor.native_id or None, status, descriptor.requirements)


def _walk_brick_catalog(namespace):
    """Yield brick-catalog entries from a SimpleNamespace of zero-arg descriptor factories.

    A factory that requires an argument (e.g. ``User(brick_id)``) is skipped: it names a slot
    that is only realisable with user input, not a standing catalog entry.
    """
    for attr_name in sorted(vars(namespace)):
        factory = getattr(namespace, attr_name)
        if not callable(factory):
            continue
        try:
            descriptor = factory()
        except TypeError:
            continue  # needs an argument (User selectors); not a standing entry.
        if hasattr(descriptor, "brick_type"):  # a BrickDescriptor
            yield _entry_from_brick(descriptor)


def _walk_class_catalog(category, classes):
    """Yield entries for descriptor CLASSES that need constructor args (e.g. mesh layouts).

    These cannot be instantiated without a mesh; we report the slot from the class itself
    (name / category) with an ``unknown`` availability that depends on the route context.
    """
    for cls in classes:
        native = getattr(cls, "native_id", None)
        yield CapabilityEntry(cls.__name__, category, native, "context", {})


# Spec 5 sec.13.12: the descriptor layout entries whose availability the C++ source ADJUDICATES.
# A native ``supports_uniform`` / ``supports_amr`` of ``False`` would make a layout descriptor that
# reports itself available a SILENT lie; the cross-check forbids that. Keyed by descriptor name.
_LAYOUT_NATIVE_FLAG = {"Uniform": "supports_uniform", "AMR": "supports_amr"}


class CapabilityMismatchError(RuntimeError):
    """A descriptor's declared availability disagrees with the C++ authoritative source (#36/#37).

    Raised by :func:`inspect_capabilities` when the native ``_pops.module_capabilities()`` reports a
    transport as UNAVAILABLE while the Python descriptor catalog still advertises it available. It
    closes the Spec 5 sec.13.12 "Python-derived, not authoritative" gap: a descriptor can no longer
    silently claim a capability the built module does not provide.
    """


def _module_capabilities():
    """The C++ authoritative capability dict (``_pops.module_capabilities()``) or ``None``.

    Lazily imports ``_pops`` (top-level then ``pops._pops``, mirroring the codegen toolchain) so the
    module import graph stays acyclic and the catalog walk works with no compiled module present.
    Returns ``None`` when ``_pops`` is unavailable or predates ``module_capabilities`` (old build):
    the descriptor walk then proceeds WITHOUT the cross-check rather than failing -- a missing native
    source is a graceful degradation, never a fabricated capability.
    """
    try:
        import _pops as mod  # noqa: PLC0415 -- lazy: keeps this module's import graph acyclic
    except Exception:
        try:
            from pops import _pops as mod  # noqa: PLC0415
        except Exception:
            return None
    fn = getattr(mod, "module_capabilities", None)
    if fn is None:  # an _pops built before this work: no authoritative source to cross-check against.
        return None
    try:
        return dict(fn())
    except Exception:
        return None


def _native_rows(native_caps):
    """Native-sourced :class:`CapabilityEntry` rows from the C++ ``module_capabilities()`` dict.

    One ``transport`` row per ``supports_*`` flag, ``source="native"``, ``available`` = ``"yes"`` /
    ``"no"`` straight from the C++ boolean (no Python computation). These are the AUTHORITATIVE facts
    the descriptor walk is checked against.
    """
    rows = []
    for key in sorted(native_caps):
        if not key.startswith("supports_"):
            continue
        status = "yes" if native_caps[key] else "no"
        rows.append(CapabilityEntry(key, "transport", None, status, {}, source="native"))
    return rows


def _cross_check(entries, native_caps):
    """Raise :class:`CapabilityMismatchError` if a descriptor disagrees with the C++ source (#36).

    For each descriptor whose capability the native facts adjudicate (the layout entries, via
    :data:`_LAYOUT_NATIVE_FLAG`), a descriptor that reports itself available while the C++ flag is
    ``False`` is a silent lie -- FAIL LOUD. A descriptor reported unavailable / context-dependent is
    never escalated (the no-false-positive discipline): the C++ source can only DEMOTE, never promote.
    """
    for entry in entries:
        flag = _LAYOUT_NATIVE_FLAG.get(entry.name)
        if flag is None or flag not in native_caps:
            continue
        descriptor_claims_available = entry.available in ("yes", "context")
        if descriptor_claims_available and not native_caps[flag]:
            raise CapabilityMismatchError(
                "descriptor %r (category %r) reports available=%r but the C++ source "
                "%s=False; the built module does not provide it (Spec 5 sec.13.12)"
                % (entry.name, entry.category, entry.available, flag))


def inspect_capabilities():
    """Return the capability :class:`CapabilityMatrix`, cross-checked against the C++ source (sec.6).

    Walks the available descriptor catalogs and reports one row per catalogued entry (``source =
    "descriptor"``), THEN cross-checks the route-deciding entries against the authoritative C++
    ``_pops.module_capabilities()`` facts and APPENDS those as ``source="native"`` rows (Spec 5
    sec.13.12, #36): the capability VALUES now come from the C++ core, not a Python walk. A descriptor
    that declares itself available while the C++ source reports the transport unavailable raises a
    :class:`CapabilityMismatchError` (closing the silent-default-fallback gap).

    The descriptor walk is PURE (only the inert authoring packages, no numeric loop). ``_pops`` is
    imported LAZILY (inside the function) and ONLY for the cross-check, so the module import graph
    stays acyclic; when ``_pops`` is absent or predates ``module_capabilities`` the walk proceeds
    WITHOUT the native rows / cross-check rather than failing (graceful degradation).
    """
    from pops.numerics.riemann import riemann
    from pops.numerics.reconstruction import reconstruction
    from pops.numerics.reconstruction.limiters import limiters
    from pops.numerics.projections import projections
    from pops.mesh.layouts import Uniform, AMR

    entries = []
    for namespace in (riemann, reconstruction, limiters, projections):
        entries.extend(_walk_brick_catalog(namespace))
    entries.extend(_walk_class_catalog("layout", (Uniform, AMR)))

    # The solver / field catalogs live under pops.lib when present (optional layer).
    try:
        from pops.lib.solvers import solvers
        entries.extend(_walk_brick_catalog(solvers))
    except ImportError:
        pass
    try:
        from pops.lib.fields import fields
        entries.extend(_walk_brick_catalog(fields))
    except ImportError:
        pass

    native_caps = _module_capabilities()
    if native_caps is not None:
        _cross_check(entries, native_caps)
        entries.extend(_native_rows(native_caps))

    return CapabilityMatrix(entries)


class AmrReport:
    """The structured, printable result of :func:`inspect_amr` (Spec 5 sec.5.11 / sec.8).

    A plain record of an AMR hierarchy's declared metadata -- the level / ratio envelope, the
    regrid / patch / nesting / refinement / checkpoint / output policies, the runtime
    requirements (reflux, tag reduction), and the explainable route limitations. :meth:`to_dict`
    returns a plain nested dict and :meth:`__str__` a short, deterministic report. It is inert:
    it holds metadata read from the descriptors, it computes nothing.
    """

    def __init__(self, *, layout, max_levels, ratio, native_max_levels, native_ratios,
                 available, limitations, requirements, policies):
        self.layout = layout
        self.max_levels = max_levels
        self.ratio = ratio
        self.native_max_levels = native_max_levels
        self.native_ratios = tuple(native_ratios)
        self.available = available
        self.limitations = list(limitations)
        self.requirements = dict(requirements or {})
        # policies: ordered list of (slot, name, options-dict) for the attached policies.
        self.policies = list(policies)

    def to_dict(self):
        return {
            "layout": self.layout,
            "max_levels": self.max_levels,
            "ratio": self.ratio,
            "native_max_levels": self.native_max_levels,
            "native_ratios": list(self.native_ratios),
            "available": self.available,
            "limitations": list(self.limitations),
            "requirements": dict(self.requirements),
            "policies": [{"slot": slot, "name": name, "options": dict(options)}
                         for slot, name, options in self.policies],
        }

    def __repr__(self):
        return ("AmrReport(layout=%r, max_levels=%r, ratio=%r, available=%r)"
                % (self.layout, self.max_levels, self.ratio, self.available))

    def __str__(self):
        lines = ["AMR hierarchy report (%s):" % self.layout]
        lines.append("  levels: max_levels=%s ratio=%s (native envelope: max_levels<=%s, "
                     "ratios=%s)" % (self.max_levels, self.ratio, self.native_max_levels,
                                     ", ".join(map(str, self.native_ratios))))
        lines.append("  available: %s" % self.available)
        if self.requirements:
            req = ", ".join("%s=%s" % (k, v) for k, v in sorted(self.requirements.items()))
            lines.append("  requirements: %s" % req)
        if self.policies:
            lines.append("  policies:")
            for slot, name, options in self.policies:
                body = ", ".join("%s=%r" % (k, v) for k, v in options.items())
                lines.append("    %-11s %s(%s)" % (slot + ":", name, body))
        if self.limitations:
            lines.append("  limitations:")
            for note in self.limitations:
                lines.append("    - %s" % note)
        return "\n".join(lines)


def _amr_policy_rows(layout):
    """Ordered (slot, name, options) rows for the policies attached to an AMR layout.

    A deterministic, stable walk of the descriptor chain (refine / regrid / patches / nesting /
    checkpoint / output); a slot left as ``None`` on the layout is skipped. The refinement
    criterion is expanded into its sub-criteria when it is a ``TagUnion`` so the report names
    each tagged subject / predicate / threshold, not just the union count.
    """
    rows = []
    for slot in ("refine", "regrid", "patches", "nesting", "checkpoint", "output"):
        policy = getattr(layout, slot, None)
        if policy is None:
            continue
        rows.append((slot, policy.name, policy.options()))
        criteria = getattr(policy, "criteria", None)
        if criteria is not None:
            for sub in criteria:
                rows.append((slot + ".criterion", sub.name, sub.options()))
    return rows


def inspect_amr(layout_or_context=None):
    """Return a printable :class:`AmrReport` of an AMR hierarchy (Spec 5 sec.5.11 / sec.8).

    The introspectable counterpart of :func:`inspect_capabilities` for the adaptive-mesh
    route. PURE: it imports only the inert :mod:`pops.mesh` authoring descriptors and reads
    their declared metadata (levels / ratio, the regrid / patch / nesting / refine / checkpoint
    / output policies, the runtime requirements such as reflux / tag reduction, and the
    explainable route limitations); it NEVER imports ``_pops`` / the runtime / codegen and runs
    no numeric loop.

    Args:
        layout_or_context: an :class:`pops.mesh.layouts.AMR` (or :class:`Uniform`) descriptor to
            report on, or ``None`` to report the current native AMR envelope (the
            :data:`pops.mesh.amr.NATIVE_MAX_LEVELS` / ``NATIVE_RATIOS`` capability limits).
    """
    from pops.mesh.amr import NATIVE_MAX_LEVELS, NATIVE_RATIOS
    from pops.mesh.layouts import AMR, Uniform

    native_note = ("the current native AMR route supports max_levels<=%d at ratio %s; a request "
                   "beyond that is refused before the runtime, not silently clamped"
                   % (NATIVE_MAX_LEVELS, ", ".join(map(str, NATIVE_RATIOS))))

    if layout_or_context is None:
        return AmrReport(
            layout="native-envelope", max_levels=NATIVE_MAX_LEVELS, ratio=NATIVE_RATIOS[0],
            native_max_levels=NATIVE_MAX_LEVELS, native_ratios=NATIVE_RATIOS,
            available="yes", limitations=[native_note], requirements={}, policies=[])

    if isinstance(layout_or_context, Uniform):
        caps = layout_or_context.capabilities()
        return AmrReport(
            layout="uniform", max_levels=caps.get("levels", 1), ratio=1,
            native_max_levels=NATIVE_MAX_LEVELS, native_ratios=NATIVE_RATIOS,
            available="yes",
            limitations=["a Uniform layout is single-level: no refinement, regrid or reflux"],
            requirements={}, policies=[])

    if not isinstance(layout_or_context, AMR):
        raise TypeError(
            "inspect_amr expects a pops.mesh.layouts.AMR / Uniform descriptor (or None for the "
            "native envelope); got %r" % (type(layout_or_context).__name__,))

    layout = layout_or_context
    status = layout.available()
    limitations = [native_note]
    if not status.ok and status.reason:
        limitations.append(status.reason)
    return AmrReport(
        layout="amr", max_levels=layout.max_levels, ratio=layout.ratio,
        native_max_levels=NATIVE_MAX_LEVELS, native_ratios=NATIVE_RATIOS,
        available=status.status, limitations=limitations,
        requirements=layout.requirements(), policies=_amr_policy_rows(layout))


__all__ = ["inspect_capabilities", "CapabilityMatrix", "CapabilityEntry",
           "CapabilityMismatchError", "inspect_amr", "AmrReport"]
