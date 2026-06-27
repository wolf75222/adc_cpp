"""pops.external -- references to compiled bricks outside the standard core (Spec 5 sec.5.17).

A compiled brick shipped in a ``.so`` / manifest, compatible with the PoPS ABI manifests, is
referenced by a typed :class:`CompiledBrickRef` (manifest + native id), never a free string.
The reference resolves to the typed ``external_cpp`` descriptor with the manifest's
requirements / capabilities, so PoPS can validate compatibility before runtime. The
in-process catalog + the low-level loader live in :mod:`pops.descriptors`; this package is the
typed user surface over them.
"""
from .bricks import CompiledBrickRef, ExternalBrick
from .manifests import register, register_manifest_file
from pops.descriptors import load_cpp_library, external
from . import bricks, manifests

__all__ = [
    "CompiledBrickRef", "ExternalBrick",
    "register", "register_manifest_file",
    "load_cpp_library", "external",
    "bricks", "manifests",
]
