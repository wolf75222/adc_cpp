"""pops.external -- references to compiled bricks outside the standard core (Spec 5 sec.5.17).

A compiled brick shipped in a ``.so`` / manifest, compatible with the PoPS ABI manifests, is
referenced by a typed :class:`CompiledBrickRef` (manifest + native id), never a free string.
The reference resolves to the typed ``external_cpp`` descriptor with the manifest's
requirements / capabilities, so PoPS can validate compatibility before runtime. The
in-process catalog + the low-level loader live in :mod:`pops.descriptors`; this package is the
typed user surface over them.
"""
from .bricks import CompiledBrickRef, ExternalBrick
from .manifests import (register, register_manifest_file, read_manifest,
                        CompiledManifest)
from .artifact_manifest import (CompiledArtifactManifest, build_compiled_manifest,
                                 check_layout_supported)
from pops.descriptors import load_cpp_library, external

from . import bricks, manifests, artifact_manifest

__all__ = [
    "CompiledBrickRef", "ExternalBrick",
    "register", "register_manifest_file", "read_manifest", "CompiledManifest",
    "CompiledArtifactManifest", "build_compiled_manifest", "check_layout_supported",
    "load_cpp_library", "external",
    "bricks", "manifests", "artifact_manifest",
]
