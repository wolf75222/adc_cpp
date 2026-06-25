"""adc.compile_library -- the Spec 3 brick-library manifest / ABI layer.

``adc.compile_library("my_numerics.so", objects=[...], backend="production")``
collects generated / macro / native brick descriptors (from :mod:`adc.lib`, the
``@adc.lib.solver`` registry, IR macros) into a reusable-library MANIFEST: the
library name, the loaded-module ABI key, the brick list (id, type, category,
scheme, native id, requirements, capabilities), the generated symbols a future
``.so`` would export, and a stable content hash. The manifest is consumed by the
library-descriptor reader (:func:`read_library_manifest`) and by
``adc.compile_problem(..., libraries=[...])``.

This module is the LOCALLY-validatable slice: the manifest, the ABI key and the
content hash. It is numerics-free (no Python solve) and, crucially, it does NOT
emit or compile the C++ brick ``.so`` -- that reuses the ``_adc`` / problem.so ABI
machinery and is the deferred ADC-464 C++ follow-up. ``compile_library(..., emit=True)``
raises a clear :class:`NotImplementedError` rather than fake a compiled artifact.
"""
import hashlib

from .lib import BrickDescriptor

__all__ = ["LibraryManifest", "compile_library", "read_library_manifest"]

# Manifest schema version: bumped if the serialized shape changes, so a stale
# round-trip is rejected loud rather than silently mis-read.
_MANIFEST_VERSION = 1

_REQUIRED_KEYS = ("manifest_version", "name", "backend", "abi_key", "bricks",
                  "generated_symbols", "content_hash")


def _brick_entry(obj):
    """The serializable manifest entry for one brick descriptor.

    Folds the descriptor's identity metadata (id, type, category, scheme, native
    id) and its requirements / capabilities. It carries NO numerics and no Python
    callable (the ``@adc.lib.solver`` builder is kept off the manifest, mirroring
    how it is kept off the descriptor identity key).
    """
    if not isinstance(obj, BrickDescriptor):
        raise TypeError(
            "compile_library objects must be adc.lib brick descriptors "
            "(e.g. adc.lib.solvers.GMRES(), adc.lib.riemann.HLLC(), an "
            "@adc.lib.solver generated brick); got %r" % (obj,))
    return {
        "id": obj.name,
        "brick_type": obj.brick_type,
        "category": obj.category,
        "scheme": obj.scheme,
        "native_id": obj.native_id,
        "available": obj.available,
        "requirements": dict(obj.requirements),
        "capabilities": dict(obj.capabilities),
        "options": dict(obj.options),
    }


def _generated_symbols(bricks):
    """The sorted ids of the GENERATED bricks -- the symbols the compiled ``.so``
    would export. Native bricks reference EXISTING ``adc::`` symbols (already in the
    loaded module) and external bricks reference a user ``.so``, so neither adds a
    generated symbol; only a generated brick (e.g. an ``@adc.lib.solver`` solver)
    contributes one."""
    return sorted({b["id"] for b in bricks if b["brick_type"] == "generated"})


def _content_hash(name, backend, abi_key, bricks):
    """Stable content hash of the manifest: sha256 over the name, backend, ABI key
    and the SORTED brick entries.

    Mirrors the ``_model_hash`` / ``module_hash`` idiom (sha256 of a structured,
    sort-stable text blob). Sorting the brick entries by id makes the hash
    order-insensitive (a library is a SET of bricks, not a sequence); it is
    sensitive to the brick set, the name, the backend and the ABI key. The
    ``available`` flag and native id fold in so a planned brick gaining a real
    symbol re-keys the library.
    """
    parts = ["adc-library-v%d" % _MANIFEST_VERSION, "name=%s" % name,
             "backend=%s" % backend, "abi_key=%s" % abi_key]
    for b in sorted(bricks, key=lambda e: e["id"]):
        parts.append(
            "brick:%s:%s:%s:%s:%s:avail=%d:reqs=%r:caps=%r:opts=%r" % (
                b["id"], b["brick_type"], b["category"], b["scheme"],
                b["native_id"], 1 if b["available"] else 0,
                sorted(b["requirements"].items()),
                sorted(b["capabilities"].items()),
                sorted(b["options"].items())))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


class LibraryManifest:
    """The descriptor of a compiled brick library (Spec 3 section 21).

    An inert metadata record: the library ``name``, the ``backend``, the loaded-module
    ``abi_key`` (header signature + compiler + std), the ``bricks`` (serialized brick
    entries), the ``generated_symbols`` a future ``.so`` would export, and a stable
    ``content_hash``. It computes nothing; the codegen / runtime and the library reader
    consume it. :func:`compile_library` builds it; :func:`read_library_manifest`
    reconstructs it from :meth:`to_dict`.
    """

    def __init__(self, name, backend, abi_key, bricks, generated_symbols,
                 content_hash):
        self.name = str(name)
        self.backend = str(backend)
        self.abi_key = str(abi_key)
        self.bricks = list(bricks)
        self.generated_symbols = list(generated_symbols)
        self.content_hash = str(content_hash)

    def to_dict(self):
        """The serialized manifest (round-trips through :func:`read_library_manifest`)."""
        return {
            "manifest_version": _MANIFEST_VERSION,
            "name": self.name,
            "backend": self.backend,
            "abi_key": self.abi_key,
            "bricks": [dict(b) for b in self.bricks],
            "generated_symbols": list(self.generated_symbols),
            "content_hash": self.content_hash,
        }

    def __eq__(self, other):
        return isinstance(other, LibraryManifest) and self.to_dict() == other.to_dict()

    def __repr__(self):
        return "LibraryManifest(%r, bricks=%d, hash=%s)" % (
            self.name, len(self.bricks), self.content_hash[:12])


def compile_library(name, objects, *, backend="production", emit=False):
    """Build a reusable brick-library MANIFEST from a set of brick descriptors.

    @p name is the library ``.so`` name; @p objects is a non-empty list of
    :class:`adc.lib.BrickDescriptor` (native / generated / macro / external bricks,
    e.g. ``adc.lib.solvers.GMRES()``, ``adc.lib.riemann.HLLC()``, an
    ``@adc.lib.solver`` generated brick). Returns a :class:`LibraryManifest`
    carrying the brick metadata, the loaded-module ABI key and a stable content hash.

    The manifest layer is locally validatable and numerics-free. Emitting and
    compiling the C++ brick ``.so`` reuses the ``_adc`` / problem.so ABI machinery and
    is the deferred ADC-464 C++ follow-up: ``emit=True`` raises a clear
    :class:`NotImplementedError` rather than fake a compiled artifact.
    """
    if backend != "production":
        raise ValueError(
            "compile_library currently supports backend='production' only; got %r"
            % (backend,))
    if not objects:
        raise ValueError("compile_library requires a non-empty objects= list of "
                         "adc.lib brick descriptors")
    bricks = [_brick_entry(obj) for obj in objects]
    abi_key = _abi_key()
    manifest = LibraryManifest(
        name=name, backend=backend, abi_key=abi_key, bricks=bricks,
        generated_symbols=_generated_symbols(bricks),
        content_hash=_content_hash(name, backend, abi_key, bricks))
    if emit:
        raise NotImplementedError(
            "ADC-464: compile_library builds the library MANIFEST (bricks, ABI key, "
            "content hash); emitting and compiling the C++ brick .so (reusing the "
            "_adc / problem.so ABI machinery) is the deferred C++ follow-up. The "
            "manifest for %r is available via compile_library(..., emit=False)." % (name,))
    return manifest


def read_library_manifest(manifest):
    """Reconstruct a :class:`LibraryManifest` from its serialized dict (the round-trip
    reader for :meth:`LibraryManifest.to_dict`).

    Accepts a :class:`LibraryManifest` unchanged (idempotent) or a dict produced by
    :meth:`to_dict`. A dict missing a required key, or carrying an unknown manifest
    version, is rejected loud (a corrupt / stale manifest must never be silently
    half-read). ``adc.compile_problem(..., libraries=[...])`` uses this to accept a
    serialized library descriptor.
    """
    if isinstance(manifest, LibraryManifest):
        return manifest
    if not isinstance(manifest, dict):
        raise TypeError("read_library_manifest expects a manifest dict or a "
                        "LibraryManifest; got %r" % (manifest,))
    missing = [k for k in _REQUIRED_KEYS if k not in manifest]
    if missing:
        raise KeyError("library manifest is missing required keys: %s"
                       % ", ".join(missing))
    version = manifest["manifest_version"]
    if version != _MANIFEST_VERSION:
        raise ValueError("unsupported library manifest version %r (expected %d)"
                         % (version, _MANIFEST_VERSION))
    return LibraryManifest(
        name=manifest["name"], backend=manifest["backend"],
        abi_key=manifest["abi_key"], bricks=manifest["bricks"],
        generated_symbols=manifest["generated_symbols"],
        content_hash=manifest["content_hash"])


def _abi_key():
    """The loaded ``_adc`` module ABI key (header signature + compiler + std), or a
    stable placeholder when ``_adc`` is unavailable (a numpy-free / module-free
    interpreter exercising the pure-Python manifest layer). The key namespaces a
    library to the exact toolchain that will dlopen its bricks."""
    try:
        from . import abi_key as _key  # adc.abi_key delegates to _adc.abi_key()
        return _key()
    except Exception:
        return "abi_key=unavailable"
