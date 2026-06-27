"""pops.descriptors -- the typed brick descriptor and the external-brick catalog.

This is the canonical home of :class:`BrickDescriptor` (the inert, numerics-free
metadata record of a numerical brick) and of the EXTERNAL C++ brick catalog
(:func:`load_cpp_library` / :func:`external` / :func:`_external_descriptor`).

It also owns the shared descriptor factories ``_native`` / ``_planned`` that the
catalog namespace modules (riemann, reconstruction, ...) import, so those
factories are defined ONCE here instead of in six files.

The hybrid/native brick CLASSES (``NativeBrick`` / ``HybridModel`` / the partial
DSL bricks) are NOT here: they live permanently in :mod:`pops.physics.bricks` and
:mod:`pops.physics.hybrid`. ``lib.descriptors`` is the Spec-3 catalog descriptor
only.
"""
import json

BRICK_TYPES = ("native", "generated", "macro", "external_cpp")


class BrickDescriptor:
    """A typed, numerics-free descriptor of a numerical brick.

    Identity is by all metadata fields so two descriptors of the same brick
    compare equal (used to detect a re-selected brick and to key the artifact
    hash). It is intentionally inert: it has no ``eval`` / ``compile`` / call.
    """

    def __init__(self, name, brick_type, *, category="brick", native_id="",
                 scheme=None, requirements=None, capabilities=None, options=None,
                 available=True, expression=None, builder=None):
        if brick_type not in BRICK_TYPES:
            raise ValueError("brick_type %r must be one of %s"
                             % (brick_type, ", ".join(BRICK_TYPES)))
        self.name = str(name)
        self.brick_type = str(brick_type)
        self.category = str(category)
        self.native_id = str(native_id)
        self.scheme = scheme
        self.requirements = dict(requirements or {})
        self.capabilities = dict(capabilities or {})
        self.options = dict(options or {})
        self.available = bool(available)
        # Optional board value carried by a generated/macro brick; kept OFF the
        # identity key (it may be an unhashable board node).
        self.expression = expression
        # Optional Python builder of a GENERATED-brick solver (``@pops.lib.solver``):
        # the function that AUTHORS the solver IR. Like ``expression`` it is kept OFF
        # the identity key (a callable is not part of the brick's value identity).
        self.builder = builder

    def _key(self):
        return (self.category, self.name, self.brick_type, self.native_id,
                self.scheme, tuple(sorted(self.options.items())))

    def __eq__(self, other):
        return isinstance(other, BrickDescriptor) and self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        return "BrickDescriptor(%r, %r, scheme=%r)" % (
            self.name, self.brick_type, self.scheme)


# --- shared descriptor factories (imported by every catalog namespace) ------
# Native ids below (in the catalog modules) are the REAL C++ symbols in include/pops
# (verified): the FV bricks live at top level in ``namespace pops`` (e.g. pops::HLLCFlux),
# not under a numerics/fv namespace. Some catalogued bricks have no native type yet --
# they are emitted with ``available=False`` and an EMPTY native_id rather than a
# fabricated symbol.
def _native(name, native_id, scheme, *, category, caps=None, **options):
    """A native-brick descriptor; ``caps`` lists required model capabilities."""
    req = {"capabilities": list(caps)} if caps is not None else {}
    return BrickDescriptor(name, "native", category=category, native_id=native_id,
                           scheme=scheme, requirements=req, options=options or None)


def _planned(name, scheme, *, category, **options):
    """A catalogued brick with NO native C++ symbol yet (available=False, no id).

    It names the slot in the catalog without overclaiming a symbol; wiring a native
    type for it is tracked as a follow-up.
    """
    return BrickDescriptor(name, "native", category=category, native_id="",
                           scheme=scheme, options=options or None, available=False)


# --- external C++ bricks (Spec 3 section 21-22 / criterion 20) -------------
# A user ships a brick in a standalone ``.so`` that registers a manifest entry at
# static-init time (the C++ ``POPS_REGISTER_BRICK`` macro -> ``BrickRegistry``) and exports
# a C ``pops_brick_manifest()`` returning JSON. ``load_cpp_library`` dlopens it, parses that
# JSON and registers the ids in this in-process catalog; ``riemann.User(id)`` /
# ``external(id)`` then surface an ``external_cpp`` descriptor carrying the manifest's
# requirements/capabilities. An id that was never loaded raises a clear error -- a
# descriptor is NEVER fabricated for an unregistered brick.
_EXTERNAL_BRICKS = {}


def _clear_external_catalog():
    """Drop every loaded external brick (test isolation; not part of the public API)."""
    _EXTERNAL_BRICKS.clear()


def _split_csv(value):
    """Split a manifest CSV field into a stripped, non-empty token list ([] when absent)."""
    if value is None:
        return []
    if not isinstance(value, str):
        raise ValueError("manifest requirements/capabilities must be a CSV string; got %r"
                         % (value,))
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _register_manifest(manifest_json):
    """Parse a brick manifest (the JSON ``pops_brick_manifest()`` returns) and register it.

    The manifest is ``{"bricks": [{"id", "category", "requirements", "capabilities"}, ...]}``
    where ``requirements``/``capabilities`` are optional CSV strings. Each entry's id is
    registered in the in-process catalog (last load wins on a repeated id). Returns the number
    of bricks registered. A malformed manifest or an entry missing its id raises ``ValueError``
    rather than silently drop a brick. This is the seam ``load_cpp_library`` calls after dlopen;
    it is also usable directly (a test does not need a compiled ``.so``).
    """
    try:
        doc = json.loads(manifest_json)
    except (json.JSONDecodeError, TypeError) as err:
        raise ValueError("external brick manifest is not valid JSON: %s" % (err,)) from err
    bricks = doc.get("bricks") if isinstance(doc, dict) else None
    if not isinstance(bricks, list):
        raise ValueError("external brick manifest must be {\"bricks\": [...]}; got %r"
                         % (manifest_json,))
    count = 0
    for entry in bricks:
        if not isinstance(entry, dict) or not entry.get("id"):
            raise ValueError("external brick manifest entry must carry a non-empty 'id'; "
                             "got %r" % (entry,))
        brick_id = str(entry["id"])
        _EXTERNAL_BRICKS[brick_id] = {
            "id": brick_id,
            "category": str(entry.get("category") or "brick"),
            "requirements": _split_csv(entry.get("requirements")),
            "capabilities": _split_csv(entry.get("capabilities")),
        }
        count += 1
    return count


def load_cpp_library(path):
    """Load an external C++ brick ``.so`` and register the bricks it manifests (criterion 20).

    Opens @p path with :func:`ctypes.CDLL` (its static initializers run the
    ``POPS_REGISTER_BRICK`` registrations), calls the exported C function
    ``const char* pops_brick_manifest()`` to read the registered bricks as JSON, and registers
    the ids in the in-process catalog so ``riemann.User(id)`` / :func:`external` resolve. The
    ``.so`` must export ``pops_brick_manifest`` (a missing symbol is a clear ``ValueError``).
    Returns the number of bricks registered.
    """
    import ctypes
    handle = ctypes.CDLL(str(path))  # raises OSError if the path is not a loadable library
    try:
        manifest_fn = handle.pops_brick_manifest
    except AttributeError as err:
        raise ValueError("external brick library %r does not export pops_brick_manifest(); it "
                         "is not an pops brick .so" % (path,)) from err
    manifest_fn.restype = ctypes.c_char_p
    raw = manifest_fn()
    if raw is None:
        raise ValueError("external brick library %r: pops_brick_manifest() returned NULL"
                         % (path,))
    return _register_manifest(raw.decode("utf-8"))


def _external_descriptor(brick_id, *, expect_category=None):
    """The ``external_cpp`` descriptor for a loaded brick @p brick_id (raise if not loaded).

    An unloaded id raises :class:`LookupError` naming the id and :func:`load_cpp_library`; a
    category mismatch (selecting via ``riemann.User`` a brick registered as a preconditioner)
    raises :class:`ValueError`. The manifest requirements/capabilities become list metadata on
    the descriptor (mirroring the native bricks' ``requirements={"capabilities": [...]}``).
    """
    entry = _EXTERNAL_BRICKS.get(str(brick_id))
    if entry is None:
        raise LookupError(
            "external brick %r not loaded; call pops.lib.load_cpp_library(...) on the brick "
            ".so first (loaded: %s)" % (brick_id, sorted(_EXTERNAL_BRICKS) or "none"))
    if expect_category is not None and entry["category"] != expect_category:
        raise ValueError("external brick %r is registered as category %r, not %r"
                         % (brick_id, entry["category"], expect_category))
    req = {"capabilities": list(entry["requirements"])} if entry["requirements"] else {}
    caps = {"provides": list(entry["capabilities"])} if entry["capabilities"] else {}
    return BrickDescriptor(entry["id"], "external_cpp", category=entry["category"],
                           native_id=entry["id"], scheme="user",
                           requirements=req or None, capabilities=caps or None)


def external(brick_id):
    """An ``external_cpp`` descriptor for a loaded brick of ANY category (criterion 20).

    The category-agnostic counterpart of ``riemann.User`` / ``preconditioner.User``: it surfaces
    whatever category the manifest registered. An unloaded id raises a clear :class:`LookupError`.
    """
    return _external_descriptor(brick_id)


# --- generic typed-descriptor protocol (Spec 5 sec.6) -----------------------------------
# Spec 5 stabilizes "every object that chooses a route is a typed descriptor that declares
# its requirements/capabilities/options and answers available(context) with an EXPLAINABLE
# status". The native-brick :class:`BrickDescriptor` above is one family; the params / output /
# external (and, after Phase D, the mesh) descriptors are another. They share this small base.
class Availability:
    """An explainable availability status (Spec 5 sec.6: not just True/False).

    ``status`` is ``"yes"`` / ``"no"`` / ``"partial"``; truthiness is ``status == "yes"`` so
    it reads naturally in a boolean test while still carrying the reason + alternatives, so a
    rejection can be reported before the runtime is ever touched.
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


class Descriptor:
    """Base of the inert typed descriptors (Spec 5 sec.6).

    Subclasses set :attr:`category` and override :meth:`options` (and :meth:`available` /
    :meth:`validate` where a route can be refused). The default contract reports an empty
    requirements/capabilities set and an unconditionally-available status. :meth:`inspect`
    returns a plain dict and :meth:`__str__` a short, deterministic summary (Spec 5 sec.12.1)
    -- never a dump of runtime data. A descriptor computes nothing.
    """

    category = "descriptor"

    @property
    def name(self):
        return type(self).__name__

    def requirements(self):
        return {}

    def capabilities(self):
        return {}

    def options(self):
        return {}

    def available(self, context=None):
        return Availability.yes()

    def validate(self, context=None):
        status = self.available(context)
        if not status.ok:
            raise ValueError("%s is not available for this route:\n%s" % (self.name, status))
        return True

    def inspect(self):
        return {"name": self.name, "category": self.category, "options": self.options(),
                "requirements": self.requirements(), "capabilities": self.capabilities()}

    def _summary(self):
        return ", ".join("%s=%r" % (k, v) for k, v in self.options().items())

    def __repr__(self):
        return "%s(%s)" % (self.name, self._summary())

    def __str__(self):
        body = self._summary()
        head = "%s [%s]" % (self.name, self.category)
        return "%s(%s)" % (head, body) if body else head
