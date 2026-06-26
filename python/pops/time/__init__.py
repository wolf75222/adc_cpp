"""pops.time -- compiled time-program DSL (builder-mode IR).

A ``Program`` is a restricted, COMPILED numerical program describing one time step. Python
only BUILDS a typed IR; it never executes a numerical stage. The C++ lowering lives in
``pops.codegen.program_codegen`` and is reached through ``Program.emit_cpp_program`` (a lazy
delegator), so this package imports only ``pops.ir`` / ``pops.model`` -- never ``pops.codegen``
or ``_pops`` at module scope (Spec 4 acyclic graph).

Public surface (unchanged from the flat module): ``Program``, ``std``, ``CompiledTime``,
``StageStateSet``, ``Schedule`` and the scheduler helpers ``always`` / ``every`` / ``when`` /
``on_start`` / ``on_end`` / ``subcycle``.

cf. docs/sphinx/reference/time-program.md (Phase 8) and the ADC-399 epic.
"""
from pops.time.equations import (  # noqa: F401  (re-export the std builder surface)
    ButcherTableau,
    adams_bashforth,
    adams_bashforth2,
    condensed_schur,
    forward_euler,
    imex_local,
    lie,
    rk,
    rk4,
    ssprk2,
    ssprk3,
    strang,
)
from pops.time.equations_implicit import (  # noqa: F401
    bdf,
    eliminate_common_subexpressions,
    eliminate_dead_nodes,
    eliminate_redundant_field_solves,
    explicit_rk,
    imex_local_linear,
    optimize,
    predictor_corrector_local_linear,
)
from pops.time.program import CompiledTime, Program, std
from pops.time.schedule import (  # noqa: F401
    Schedule, always, every, on_end, on_start, subcycle, when,
)
from pops.time.values import StageStateSet, Value  # noqa: F401

__all__ = ["Program", "std", "CompiledTime", "StageStateSet", "Schedule",
           "always", "every", "when", "on_start", "on_end", "subcycle"]
