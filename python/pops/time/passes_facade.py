"""pops.time.passes_facade -- free-function wrappers over the Program IR-optimizer passes.

These are the temporal-LANGUAGE optimization entry points (Spec 3 s28 / ADC-465): each
delegates to the matching ``Program`` method, so a caller can write
``pops.time.optimize(prog)`` instead of ``prog.optimize()``. They belong to ``pops.time``
(the time language); the ready schemes themselves live in ``pops.lib.time`` (Spec 4 s6 / s14).

Authoring only; every pass returns a NEW Program and is OPT-IN (byte-identical when nothing
is optimizable). They never touch the default ``emit_cpp_program`` path.
"""


def eliminate_dead_nodes(program):
    """Return a NEW Program with dead flat-list nodes removed (free-function form of
    :meth:`Program.eliminate_dead_nodes`, Spec 3 s28 / ADC-465). OPT-IN: it optimizes a copy and never
    touches the default ``emit_cpp_program`` path. See the method for the dead-node rule."""
    return program.eliminate_dead_nodes()


def eliminate_common_subexpressions(program):
    """Return a NEW Program with duplicated PURE sub-IR computed once and aliased (free-function form
    of :meth:`Program.eliminate_common_subexpressions`, Spec 3 s28 / ADC-465). OPT-IN, proven-safe."""
    return program.eliminate_common_subexpressions()


def eliminate_redundant_field_solves(program):
    """Return a NEW Program with a provably-redundant second ``solve_fields`` removed (free-function
    form of :meth:`Program.eliminate_redundant_field_solves`, Spec 3 s28 / ADC-465). OPT-IN,
    conservative: only when no state/aux mutation intervenes between the two solves."""
    return program.eliminate_redundant_field_solves()


def optimize(program):
    """Return a NEW Program with the proven-safe Spec 3 s28 transform passes applied (free-function
    form of :meth:`Program.optimize`, ADC-465). OPT-IN: byte-identical when nothing is optimizable."""
    return program.optimize()
