"""pops.time -- compiled time-program DSL (the temporal LANGUAGE, builder-mode IR).

A ``Program`` is a restricted, COMPILED numerical program describing one time step. Python
only BUILDS a typed IR; it never executes a numerical stage. The C++ lowering lives in
``pops.codegen.program_codegen`` and is reached through ``Program.emit_cpp_program`` (a lazy
delegator), so this package imports only ``pops.ir`` / ``pops.model`` -- never ``pops.codegen``,
``_pops``, or ``pops.lib`` at module scope (Spec 4 acyclic graph: time -> {ir, model}).

This package is the time LANGUAGE only: ``Program``, ``CompiledTime``, the ``Value`` /
``StageStateSet`` values, the ``Schedule`` scheduler (``always`` / ``every`` / ``when`` /
``on_start`` / ``on_end`` / ``subcycle``) and the IR-optimizer wrappers (``eliminate_*`` /
``optimize``). The READY time-stepping schemes live in ``pops.lib.time`` (Spec 4 s6 / s14),
called by their explicit names (no ``std`` bundle, Spec 4 s7); import them from there.

cf. docs/sphinx/reference/time-program.md (Phase 8) and the ADC-399 epic.
"""
from pops.time.handles import TimeState, _Version  # noqa: F401
from pops.time.history import CopyCurrent  # noqa: F401
from pops.time.passes_facade import (  # noqa: F401
    eliminate_common_subexpressions,
    eliminate_dead_nodes,
    eliminate_redundant_field_solves,
    optimize,
)
from pops.time.program import CompiledTime, Program
from pops.time.schedule import (  # noqa: F401
    Schedule, always, every, on_end, on_start, subcycle, when,
)
from pops.time.values import StageStateSet, Value  # noqa: F401

__all__ = ["Program", "CompiledTime", "StageStateSet", "Schedule",
           "TimeState", "CopyCurrent",
           "always", "every", "when", "on_start", "on_end", "subcycle",
           "eliminate_dead_nodes", "eliminate_common_subexpressions",
           "eliminate_redundant_field_solves", "optimize"]
