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

__all__ = [
    "pops_header_signature",
    "pops_include",
    "loader_cxx_std",
    "resolve_auto_backend",
    "pops_loader_build_flags",
    "pops_cache_dir",
    "check_compiled_matches_module",
]
