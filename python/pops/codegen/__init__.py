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
# Spec 5 (sec.12.2 / 12.3): inert introspection of a compiled artifact (bind arguments + memory).
from .inspect_compiled import Arguments, MemoryEstimate  # noqa: F401
# Spec 5 (sec.13.11.3 / criterion 38): inert scratch-liveness plan of a compiled time Program.
from .scratch_plan import ScratchPlan, build_scratch_plan  # noqa: F401
# Spec 5 (sec.12.1): inert printable reports of a compiled artifact (inspect / requirements / bind).
from .inspect_report import CompiledReport, RequirementsReport, BindReport  # noqa: F401
# Spec 5 (sec.13.8): typed codegen optimization policy + numeric math modes.
from .optimization import Optimization, ConservativeFusion, Disabled  # noqa: F401
from .math_options import StrictMath, FastMath, DebugMath, GpuRegisterAware  # noqa: F401
from . import optimization, math_options  # noqa: F401
# Spec 5 (sec.8.15 / criterion 22): typed compile-backend descriptors (Production/AOT/JIT)
# that lower to the legacy backend string the drivers consume.
from .backends import Production, AOT, JIT, lower_backend  # noqa: F401
from . import backends  # noqa: F401
# Spec 5 (sec.12.4, #47-48): the codegen POPS_* environment resolver (CodegenEnv) + the JIT-backdoor
# guard predicate. Stdlib-only, so this adds no numpy / _pops weight to the codegen surface.
from .env import CodegenEnv, jit_backdoor_enabled  # noqa: F401

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
    "Arguments", "MemoryEstimate",
    "ScratchPlan", "build_scratch_plan",
    "CompiledReport", "RequirementsReport", "BindReport",
    "emit_library_cpp",
    "Optimization", "ConservativeFusion", "Disabled",
    "StrictMath", "FastMath", "DebugMath", "GpuRegisterAware",
    "optimization", "math_options",
    "Production", "AOT", "JIT", "lower_backend", "backends",
    "CodegenEnv", "jit_backdoor_enabled",
]
