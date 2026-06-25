"""adc.codegen -- the C++-emitter layer (Spec 4).

The only layer allowed to emit C++ and to touch ``_adc``. It owns the brick-library
manifest / ABI layer (:mod:`adc.codegen.library`) and the C++ source emitter
(:mod:`adc.codegen.library_codegen`).
"""
from adc.codegen.library import (
    LibraryManifest,
    compile_library,
    read_library_manifest,
)
from adc.codegen.library_codegen import emit_library_cpp

__all__ = [
    "compile_library",
    "LibraryManifest",
    "read_library_manifest",
    "emit_library_cpp",
]
