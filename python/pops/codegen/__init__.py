"""pops.codegen : build-infra / toolchain helpers for DSL compilation.

Re-exports the public symbols of the sub-modules so callers can write
``from pops.codegen import pops_header_signature`` etc.
"""

from .toolchain import (  # noqa: F401
    pops_header_signature,
    pops_include,
    loader_cxx_std,
    resolve_auto_backend,
    pops_loader_build_flags,
)
from .cache import pops_cache_dir  # noqa: F401
from .abi import check_compiled_matches_module  # noqa: F401
from .compile import compile_problem  # noqa: F401
from .loader import CompiledProblem  # noqa: F401
from .library_codegen import emit_library_cpp  # noqa: F401
# Spec 5 (sec.13.8): typed codegen optimization policy + numeric math modes.
from .optimization import Optimization, ConservativeFusion, Disabled  # noqa: F401
from .math_options import StrictMath, FastMath, DebugMath, GpuRegisterAware  # noqa: F401
from . import optimization, math_options  # noqa: F401
# Spec 5 (sec.8.15 / criterion 22): typed compile-backend descriptors (Production/AOT/JIT)
# that lower to the legacy backend string the drivers consume.
from .backends import Production, AOT, JIT, lower_backend  # noqa: F401
from . import backends  # noqa: F401

__all__ = [
    "pops_header_signature",
    "pops_include",
    "loader_cxx_std",
    "resolve_auto_backend",
    "pops_loader_build_flags",
    "pops_cache_dir",
    "check_compiled_matches_module",
    "compile_problem",
    "CompiledProblem",
    "emit_library_cpp",
    "Optimization", "ConservativeFusion", "Disabled",
    "StrictMath", "FastMath", "DebugMath", "GpuRegisterAware",
    "optimization", "math_options",
    "Production", "AOT", "JIT", "lower_backend", "backends",
]
