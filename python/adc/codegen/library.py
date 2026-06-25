"""adc.compile_library -- the Spec 3 brick-library manifest / ABI layer.

``adc.compile_library("my_numerics.so", objects=[...], backend="production")``
collects generated / macro / native brick descriptors (from :mod:`adc.lib`, the
``@adc.lib.solver`` registry, IR macros) into a reusable-library MANIFEST: the
library name, the loaded-module ABI key, the brick list (id, type, category,
scheme, native id, requirements, capabilities), the generated symbols a future
``.so`` would export, and a stable content hash. The manifest is consumed by the
library-descriptor reader (:func:`read_library_manifest`) and by
``adc.compile_problem(..., libraries=[...])``.

The manifest, the ABI key and the content hash are numerics-free (no Python solve).
``compile_library(..., emit=True)`` ALSO emits the C++ of the library's bricks
(:mod:`adc.library_codegen`) and compiles a real ``.so`` with the same Kokkos toolchain a
problem ``.so`` uses (:func:`adc.dsl.adc_loader_build_flags`, ``ADC_KOKKOS_ROOT``), exporting
the metadata, the ABI key, the brick list / signatures / requirements / capabilities and the
generated symbols. :func:`read_library_manifest` reads that descriptor back from the ``.so``
(dlopen) and rejects an ABI / Kokkos mismatch as a HARD error. ``adc.compile_problem(...,
libraries=[...])`` reads + validates the compiled ``.so`` (the consume path).
"""
import hashlib
import json

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
    # SPEC4-TODO: repoint to adc.lib.descriptors / adc.codegen
    from adc.lib import BrickDescriptor
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
    entries), the ``generated_symbols`` the ``.so`` exports, a stable ``content_hash``, and
    -- once ``compile_library(..., emit=True)`` has compiled it -- the ``so_path`` of the
    real artifact (``None`` for a manifest-only build). It computes nothing; the codegen /
    runtime and the library reader consume it. :func:`compile_library` builds it;
    :func:`read_library_manifest` reconstructs it from :meth:`to_dict` OR from a compiled
    ``.so`` path.
    """

    def __init__(self, name, backend, abi_key, bricks, generated_symbols,
                 content_hash, so_path=None):
        self.name = str(name)
        self.backend = str(backend)
        self.abi_key = str(abi_key)
        self.bricks = list(bricks)
        self.generated_symbols = list(generated_symbols)
        self.content_hash = str(content_hash)
        # Path of the compiled .so, or None for a manifest-only (emit=False) build. It is
        # provenance, NOT identity: it stays OUT of __eq__ / the content hash (the same
        # library compiled to two paths is the same library) but IS carried on to_dict.
        self.so_path = None if so_path is None else str(so_path)

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
            "so_path": self.so_path,
        }

    def _identity(self):
        """The manifest dict WITHOUT the so_path (the artifact path is provenance, not
        identity: the same library compiled to two paths compares equal)."""
        d = self.to_dict()
        d.pop("so_path", None)
        return d

    def __eq__(self, other):
        return isinstance(other, LibraryManifest) and self._identity() == other._identity()

    def __repr__(self):
        return "LibraryManifest(%r, bricks=%d, hash=%s)" % (
            self.name, len(self.bricks), self.content_hash[:12])


def compile_library(name, objects, *, backend="production", emit=False, so_path=None,
                    cxx=None, force=False):
    """Build a reusable brick library from a set of brick descriptors.

    @p name is the library ``.so`` name; @p objects is a non-empty list of
    :class:`adc.lib.BrickDescriptor` (native / generated / macro / external bricks,
    e.g. ``adc.lib.solvers.GMRES()``, ``adc.lib.riemann.HLLC()``, an
    ``@adc.lib.solver`` generated brick). Returns a :class:`LibraryManifest`
    carrying the brick metadata, the loaded-module ABI key and a stable content hash.

    With ``emit=False`` (default) it returns the MANIFEST only (numerics-free, no
    compiler needed). With ``emit=True`` it ALSO emits the library C++
    (:func:`adc.library_codegen.emit_library_cpp`) and compiles a REAL ``.so`` with the
    same Kokkos toolchain a problem ``.so`` uses (:func:`adc.dsl.adc_loader_build_flags`,
    ``ADC_KOKKOS_ROOT``); the returned manifest carries the artifact ``so_path``. Without
    an explicit ``so_path`` the ``.so`` is cached out-of-source keyed by the content hash +
    ABI key (``force=True`` recompiles). The ``.so`` exports the metadata, the ABI key, the
    brick list / requirements / capabilities and the generated symbols; an ABI / Kokkos
    mismatch when it is later read back is a HARD error -- never a silent fallback.
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
        manifest.so_path = _emit_and_compile(manifest, so_path=so_path, cxx=cxx, force=force)
    return manifest


def _emit_and_compile(manifest, *, so_path=None, cxx=None, force=False):
    """Emit @p manifest's C++ and compile the library ``.so``; return its path.

    Reuses the production toolchain helpers (:mod:`adc.dsl`): the same Kokkos compiler,
    flags and ABI-keyed cache path a problem ``.so`` uses, so the library ``.so`` is
    ABI-compatible with the loaded ``_adc`` module. Kokkos is mandatory (adc_cpp is
    Kokkos-only); a missing ``ADC_KOKKOS_ROOT`` is a clear error from
    :func:`adc.dsl.adc_loader_build_flags`.
    """
    import os
    import tempfile

    # SPEC4-TODO: repoint to adc.lib.descriptors / adc.codegen
    from adc import dsl
    from adc.codegen.library_codegen import emit_library_cpp

    src = emit_library_cpp(manifest)
    include = dsl.adc_include()
    sig = dsl.adc_header_signature(include)
    cc, cflags, lflags = dsl.adc_loader_build_flags(cxx)
    eff_std = dsl._probe_cxx_std(cc, dsl.loader_cxx_std())

    if so_path is None:
        key = "%s|%s|%s" % (sig, cc, eff_std)
        so_path = dsl._cache_so_path(manifest.content_hash, key, "library-production",
                                     "library", manifest.name)
        if not force and os.path.isfile(so_path):
            return so_path

    optflags = dsl._dsl_optflags()
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "library.cpp")
        with open(cpp, "w") as f:
            f.write(src)
        flags = ["-shared", "-fPIC", "-std=" + eff_std, *optflags,
                 "-DADC_HEADER_SIG=\"%s\"" % sig, *cflags]
        cmd = [cc, *flags, "-I", include, cpp, "-o", so_path, *lflags]
        dsl._run_compile(cmd, "compile_library (backend production)")
    return so_path


def read_library_manifest(manifest):
    """Reconstruct a :class:`LibraryManifest` from a serialized dict, a compiled ``.so`` path,
    or a :class:`LibraryManifest` (idempotent).

    * a :class:`LibraryManifest` is returned unchanged;
    * a dict produced by :meth:`to_dict` round-trips; a dict missing a required key, or
      carrying an unknown manifest version, is rejected loud (a corrupt / stale manifest is
      never silently half-read);
    * a ``str`` / ``os.PathLike`` is treated as a compiled library ``.so`` path: it is
      dlopen'd (:func:`_read_so_manifest`), its exported descriptor is read back, and its ABI
      key is compared against the loaded ``_adc`` module -- an ABI / Kokkos mismatch is a HARD
      error (the bricks would otherwise crash the loader with a cryptic symbol failure).

    ``adc.compile_problem(..., libraries=[...])`` uses this to accept a manifest, a serialized
    descriptor, OR a compiled ``.so`` path.
    """
    import os

    if isinstance(manifest, LibraryManifest):
        return manifest
    if isinstance(manifest, (str, os.PathLike)):
        return _read_so_manifest(os.fspath(manifest))
    if not isinstance(manifest, dict):
        raise TypeError("read_library_manifest expects a manifest dict, a compiled .so path, "
                        "or a LibraryManifest; got %r" % (manifest,))
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
        content_hash=manifest["content_hash"], so_path=manifest.get("so_path"))


def _read_so_manifest(so_path):
    """Read a compiled library ``.so`` descriptor back into a :class:`LibraryManifest` (dlopen).

    Opens @p so_path with :func:`ctypes.CDLL` and reads the ``adc_library_*`` exports the
    codegen emitted (name / backend / content hash / ABI key + the per-brick string tables).
    Enforces the ABI / Kokkos guard FIRST: the ``.so``'s ``adc_library_abi_key()`` is compared
    against the loaded ``_adc`` module's ABI key, and a mismatch raises a HARD :class:`RuntimeError`
    (the bricks were compiled against a different toolchain -- dlopen-ing them into a problem would
    fail with a cryptic symbol error or, worse, silent UB). A ``.so`` lacking ``adc_library_*``
    exports is not an adc library (clear error). The static-init ``ADC_REGISTER_BRICK`` calls also
    populate the in-process external-brick catalog as a side effect of the load.
    """
    import ctypes

    handle = ctypes.CDLL(str(so_path))  # raises OSError if the path is not a loadable library

    def cstr(symbol):
        try:
            fn = getattr(handle, symbol)
        except AttributeError as err:
            raise ValueError(
                "library %r does not export %s(); it is not an adc compiled brick library "
                "(adc.compile_library(..., emit=True))" % (so_path, symbol)) from err
        fn.restype = ctypes.c_char_p
        raw = fn()
        return "" if raw is None else raw.decode("utf-8")

    def cint(symbol):
        fn = getattr(handle, symbol)
        fn.restype = ctypes.c_int
        return int(fn())

    def cstr_i(symbol, i):
        fn = getattr(handle, symbol)
        fn.restype = ctypes.c_char_p
        fn.argtypes = [ctypes.c_int]
        raw = fn(i)
        return "" if raw is None else raw.decode("utf-8")

    so_abi = cstr("adc_library_abi_key")
    module_abi = _abi_key()
    # HARD ABI / Kokkos guard: never silently load bricks compiled against a different toolchain.
    if module_abi not in ("", "abi_key=unavailable") and so_abi != module_abi:
        raise RuntimeError(
            "adc.read_library_manifest: library %r was compiled with an ABI key DIFFERENT "
            "from the loaded _adc module (library %r vs module %r). The bricks were built "
            "against another compiler / C++ standard / header tree / Kokkos build; dlopen-ing "
            "them into a problem would fail with a cryptic symbol error or undefined behavior. "
            "Recompile the library with adc.compile_library(..., emit=True) using the SAME "
            "toolchain (ADC_KOKKOS_ROOT) that built _adc." % (so_path, so_abi, module_abi))

    n = cint("adc_library_brick_count")
    bricks = []
    for i in range(n):
        scheme = cstr_i("adc_library_brick_scheme", i)
        bricks.append({
            "id": cstr_i("adc_library_brick_id", i),
            "brick_type": cstr_i("adc_library_brick_type", i),
            "category": cstr_i("adc_library_brick_category", i),
            "scheme": scheme or None,
            "native_id": cstr_i("adc_library_brick_native_id", i),
            "available": cstr_i("adc_library_brick_available", i) == "1",
            "requirements": json.loads(cstr_i("adc_library_brick_requirements", i) or "{}"),
            "capabilities": json.loads(cstr_i("adc_library_brick_capabilities", i) or "{}"),
            "options": {},
        })
    gen = [cstr_i("adc_library_generated_symbol", i)
           for i in range(cint("adc_library_generated_symbol_count"))]
    return LibraryManifest(
        name=cstr("adc_library_name"), backend=cstr("adc_library_backend"),
        abi_key=so_abi, bricks=bricks, generated_symbols=gen,
        content_hash=cstr("adc_library_content_hash"), so_path=so_path)


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
