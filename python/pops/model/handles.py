"""Typed operator handles (Spec 5 sec.14.2.3).

An :class:`OperatorHandle` is a lightweight, INERT reference to a declared operator: it
carries the operator ``name`` (and, when the declarer knows it cheaply, the operator
``kind``) and nothing else -- no numerics, no IR, no Program. A user-facing operator
declarer (``m.rate_operator`` / ``m.source_term`` / ``m.linear_source``) returns one so a
named operator can be passed around as a typed object instead of a bare string::

    R = m.rate_operator("explicit_rhs", flux=True, sources=["electric"])
    ...
    rate = P.call(R, U, fields)        # same as P.call("explicit_rhs", U, fields)

The handle is a transparent alias for its name: :meth:`pops.time.Program.call` accepts
EITHER the string operator name OR an ``OperatorHandle`` and resolves both through the
exact same registry lookup + lowering, so ``P.call(handle, ...)`` builds the byte-identical
IR (same ``_ir_hash``) as ``P.call(name, ...)``. The handle holds no Program reference, so
the same handle works in any Program bound to a registry that declares its name.

This module imports only the standard library, so it stays codegen-free and ``_pops``-free
and keeps the ``pops.time`` import graph acyclic (``pops.time`` already imports
``pops.model``; ``pops.model`` never imports ``pops.time``).
"""


class OperatorHandle:
    """An inert, typed reference to a declared operator (Spec 5 sec.14.2.3).

    Carries the operator ``name`` and an optional ``kind`` (the operator-first kind, e.g.
    ``"local_rate"`` / ``"local_source"`` / ``"local_linear_operator"``, when the declarer
    supplies it). It is value-like: two handles compare equal iff their ``(name, kind)``
    match, so a handle can be used as a dict key or compared in tests. It carries no
    Program, no registry and no IR; ``P.call`` resolves ``handle.name`` against the
    Program's bound registry exactly like a string.
    """

    __slots__ = ("name", "kind")

    def __init__(self, name, kind=None):
        if not isinstance(name, str) or not name:
            raise ValueError("OperatorHandle: name must be a non-empty string")
        self.name = name
        self.kind = kind

    def __eq__(self, other):
        return (isinstance(other, OperatorHandle)
                and self.name == other.name and self.kind == other.kind)

    def __hash__(self):
        return hash((self.name, self.kind))

    def __repr__(self):
        if self.kind is None:
            return "OperatorHandle(%r)" % (self.name,)
        return "OperatorHandle(%r, kind=%r)" % (self.name, self.kind)
