"""adc.lib.solvers -- custom solver DSL registry and IR authoring context (Spec 4 split).

Verbatim extraction from python/adc/lib.py lines 330-587 and 1007-1020.
No logic changed. Cross-references resolved:
- BrickDescriptor -> adc.lib.descriptors (was in the same file as lib.py)
- generate_solver_cpp (now in adc.codegen.solvers) is CALLED BY code that remains in
  lib.py; that direction is not imported here.
- adc.time is imported lazily (``from . import time``) exactly as lib.py does it.
"""
import json  # noqa: F401 -- kept to mirror lib.py; used by _SolverCppLowering (now in codegen)

from adc.lib.descriptors import BrickDescriptor  # SPEC4: was same-file in lib.py


# --- custom solver DSL (Spec 3 section 20 / criterion 23) -------------------
# ``@adc.lib.solver`` registers a GENERATED-brick solver whose body is a Python
# builder that AUTHORS a solver IR (matrix-free Krylov primitives). The builder
# computes nothing in Python; the IR lowers to C++ -- that lowering is the
# deferred follow-up (``generate_solver_cpp`` raises a clear ADC-462 error).
_CUSTOM_SOLVERS = {}


def solver(name=None, signature=None):
    """Register a custom solver written in the IR-authoring DSL (criterion 23).

    Decorates a builder ``f(ctx, *args)`` that AUTHORS a solver IR using the
    matrix-free Krylov primitives of :class:`SolverContext` (``ctx.norm2`` /
    ``ctx.dot`` / ``ctx.residual`` / affine ``x + omega*r`` / ``ctx.while_``).
    The builder builds IR ONLY -- it never performs Python numerics. Returns a
    ``generated`` :class:`BrickDescriptor` in the ``solver`` category, carrying the
    builder off its identity key, selectable wherever a native solver is (its
    ``scheme`` mirrors ``adc.lib.solvers.GMRES()``).

    The generated C++ lowering + run is the deferred C++ follow-up: see
    :func:`generate_solver_cpp` (it raises a clear ADC-462 ``NotImplementedError``;
    it is never faked as a Python solve).
    """
    if not isinstance(name, str) or not name:
        raise ValueError("@adc.lib.solver requires a non-empty name=")
    if signature is not None and not isinstance(signature, str):
        raise TypeError("@adc.lib.solver signature= must be a string (e.g. '(A, b)')")

    def decorate(builder):
        if not callable(builder):
            raise TypeError("@adc.lib.solver must decorate a callable builder; got %r"
                            % (builder,))
        opts = {"signature": signature} if signature is not None else None
        desc = BrickDescriptor(name, "generated", category="solver", scheme=name,
                               options=opts, builder=builder)
        _CUSTOM_SOLVERS[name] = desc
        return desc

    return decorate


def _custom_solver(name):
    """The registered custom-solver descriptor named @p name (KeyError if absent)."""
    return _CUSTOM_SOLVERS[name]


def _registered_solvers():
    """The names of the registered custom solvers (registration order)."""
    return list(_CUSTOM_SOLVERS)


def _as_descriptor(solver_brick):
    """Coerce a ``@adc.lib.solver`` argument to its descriptor: accept the descriptor
    itself or a registered name. A non-generated/non-solver brick is rejected loud."""
    if isinstance(solver_brick, BrickDescriptor):
        desc = solver_brick
    elif isinstance(solver_brick, str):
        desc = _CUSTOM_SOLVERS.get(solver_brick)
        if desc is None:
            raise KeyError("no custom solver named %r is registered" % (solver_brick,))
    else:
        raise TypeError("expected a custom solver descriptor or its name; got %r"
                        % (solver_brick,))
    if desc.brick_type != "generated" or desc.category != "solver" or desc.builder is None:
        raise ValueError("%r is not a custom (@adc.lib.solver) solver descriptor"
                         % (desc.name,))
    return desc


class SolverIR:
    """The IR authored by a custom-solver builder: an inert graph of typed ops.

    It is a thin view over the building :class:`adc.time.Program` -- it records the
    flat op list and the returned solution value. It holds NO numeric data: every
    node is a typed SSA record (see :class:`adc.time.Value`). The C++ lowering of
    this IR is deferred (ADC-462); :func:`generate_solver_cpp` raises rather than
    fake a Python solve.
    """

    def __init__(self, descriptor, program, result):
        self.descriptor = descriptor
        self.program = program
        self.result = result

    def nodes(self):
        """The IR value nodes the builder authored, including control-flow body ops.

        Walks the flat SSA list AND the recorded ``cond``/``body`` sub-blocks of ``while``
        nodes (those blocks are owned by the op, not the top-level list), in build order."""
        out = []
        _walk_nodes(self.program._values, out)
        return out

    def op_kinds(self):
        """The set of op kinds present in the IR (e.g. ``norm2`` / ``apply`` / ``while``)."""
        kinds = set()
        for node in self.nodes():
            kinds.add(node.attrs.get("kind", node.op))
        return kinds

    def __repr__(self):
        return "SolverIR(%r, nodes=%d)" % (self.descriptor.name, len(self.program._values))


class SolverContext:
    """The matrix-free Krylov authoring context handed to a custom-solver builder.

    It wraps an :class:`adc.time.Program` and exposes the primitives a solver needs
    -- ``norm2`` / ``dot`` / ``scalar_int`` / ``logical_and`` / ``while_`` (a context
    manager) / operator apply / affine ``x + omega*r``. Every primitive BUILDS an IR
    node and returns an IR value; NOTHING is computed in Python. The unknown ``x``,
    the residual ``r`` and the operator apply ``A(x)`` are IR values, not arrays.
    """

    def __init__(self, program, block="solve"):
        self._p = program
        self._block = block

    # --- operands -----------------------------------------------------------
    def unknown(self, name=None):
        """A fresh solver unknown (the iterate ``x`` / the rhs ``b``): a State IR value."""
        return self._p.state(self._block)

    def zeros_like(self, value):
        """A zero-initialized iterate over the same block as @p value (the warm start)."""
        _require_field(value, "zeros_like")
        return self._p.state(value.block or self._block)

    def scalar_int(self, n):
        """A COMPILE-TIME integer literal as a Scalar IR value (a loop count / index). It
        is an IR node, never a live Python counter the loop mutates."""
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError("scalar_int expects a Python int; got %r" % (n,))
        return self._p._scalar_binop(float(n), 0.0, "add")

    # --- reductions ---------------------------------------------------------
    def norm2(self, x):
        """The Euclidean norm ``||x||_2`` as a Scalar IR value (a collective reduction)."""
        return self._p.norm2(x)

    def dot(self, a, b):
        """The inner product ``<a, b>`` as a Scalar IR value (a collective reduction)."""
        return self._p.dot(a, b)

    # --- operator apply / residual ------------------------------------------
    def apply(self, operator, x):
        """Apply the matrix-free operator ``A(x)`` as an IR node (an RHS-like value)."""
        if not (hasattr(x, "vtype") and x.vtype == "state"):
            raise TypeError("apply: x must be a State IR value")
        return self._p.apply(operator=_operator_name(operator), state=x)

    def residual(self, operator, x, b):
        """The residual ``r = b - A(x)`` as an affine IR combine (no Python math)."""
        ax = self.apply(operator, x)
        return self._p.linear_combine(expr=b - ax)

    def combine(self, expr):
        """Materialize an affine IR expression (e.g. ``x + omega*r``) into a State IR node.

        The affine ``x + omega*r`` is a deferred IR expression; this records it as one
        ``linear_combine`` node (the next iterate). ``omega`` is an IR literal coefficient,
        never multiplied against data in Python."""
        return self._p.linear_combine(expr=expr)

    # --- predicates ---------------------------------------------------------
    def logical_and(self, a, b):
        """The conjunction of two Bool predicates as a Bool IR node (re-evaluated each
        loop pass). Builds an ``and`` node; it never short-circuits in Python."""
        for nm, val in (("a", a), ("b", b)):
            if not (hasattr(val, "vtype") and val.vtype == "bool"):
                raise TypeError("logical_and: %s must be a Bool IR value" % nm)
        return self._p._new("bool", "logical_and", (a, b), {}, None, a.block)

    # --- control flow -------------------------------------------------------
    def while_(self, cond_fn):
        """A convergence loop as a context manager: ``with ctx.while_(cond_fn):`` records the
        loop body, then RE-EVALUATES the convergence predicate against the loop-updated
        iterate and emits one IR ``while`` node owning both blocks.

        @p cond_fn is a zero-argument builder that BUILDS a Bool IR value each time it is
        called (e.g. ``lambda: ctx.norm2(ctx.residual(A, x, b)) > tol``). It is recorded
        into a SEPARATE ``cond_block`` after the body, so the predicate references the
        mutated iterate -- not the pre-loop ``x`` -- and re-runs every pass (mirroring
        :meth:`adc.time.Program.while_`). The loop is DYNAMIC (C++-side); it never iterates
        in Python.

        Wiring the predicate to a pre-built Bool value would freeze it on the initial
        iterate (a constant convergence test), so a bare Bool value is rejected loud."""
        if not callable(cond_fn):
            raise TypeError(
                "while_: condition must be a zero-argument builder that BUILDS the Bool "
                "predicate against the loop-updated iterate (e.g. "
                "lambda: ctx.norm2(ctx.residual(A, x, b)) > tol), not a pre-built Bool value "
                "(that would freeze the test on the initial iterate)")
        return _SolverWhile(self._p, cond_fn, self._block)


class _SolverWhile:
    """The context manager :meth:`SolverContext.while_` returns: it records the loop body
    ops into a sub-block and, on exit, RE-RECORDS the convergence predicate into a separate
    ``cond_block`` so it references the loop-updated iterate (mirroring
    :meth:`adc.time.Program.while_`). It then emits one ``while`` IR node owning both blocks.
    The blocks are re-emitted inside the generated C++ loop; they are never replayed in
    Python."""

    def __init__(self, program, cond_fn, block):
        self._p = program
        self._cond_fn = cond_fn
        self._block = block
        self._body = None

    def __enter__(self):
        self._body = []
        self._p._recording.append(self._body)
        return self

    def __exit__(self, exc_type, exc, tb):
        self._p._recording.pop()
        if exc_type is not None:
            return False
        # Record the predicate AFTER the body so it builds against the loop-updated iterate.
        # Its ops live in a separate cond_block (re-run each pass), not the body block.
        cond_block = []
        self._p._recording.append(cond_block)
        try:
            cond = self._cond_fn()
        finally:
            self._p._recording.pop()
        if not (hasattr(cond, "vtype") and cond.vtype == "bool"):
            raise TypeError("while_: the condition builder must return a Bool IR value "
                            "(e.g. ctx.norm2(r) > tol); got %r" % (cond,))
        self._p._new("state", "while", (),
                     {"cond_block": cond_block, "cond": cond, "body_block": self._body},
                     None, self._block)
        return False


def build_solver_ir(solver_brick):
    """Run a custom-solver builder to AUTHOR its IR (no Python numerics).

    @p solver_brick is a ``@adc.lib.solver`` descriptor (or its registered name). The
    builder receives a :class:`SolverContext` and two unknowns (the operator ``A`` and
    the rhs ``b``) and returns the solution IR value. Returns a :class:`SolverIR`.
    """
    from adc import time as _time  # SPEC4-TODO: repoint -- lazy import as in lib.py
    desc = _as_descriptor(solver_brick)
    program = _time.Program("solver_" + desc.name)
    ctx = SolverContext(program)
    a_op = program.linear_source("A")    # the matrix-free operator A, an IR operator value
    b_rhs = ctx.unknown("b")             # the right-hand side b, an IR State value
    result = desc.builder(ctx, a_op, b_rhs)
    # A builder may return an affine expression (``x + omega*r``); materialize it into a
    # State IR node so the solution is always a recorded value, never a deferred Python expr.
    if not (hasattr(result, "vtype")) and result is not None:
        result = ctx.combine(result)
    if not (hasattr(result, "vtype") and result.vtype == "state"):
        raise ValueError("a custom solver builder must return the solution State IR value; "
                         "got %r" % (result,))
    return SolverIR(desc, program, result)


# A hard upper bound on a custom solver's convergence loop: a generated kernel MUST terminate even if
# the authored predicate never goes false (a stalled / diverging custom solver). The authored
# ``it < max_iter`` cap normally stops the loop first; this is the backstop.
_SOLVER_MAX_ITERS = 1000000


def _walk_nodes(values, out):
    """Append @p values and any ops recorded in their ``while`` cond/body sub-blocks to @p
    out, depth-first in build order (a loop's cond and body blocks are owned by its op, not
    the flat list). The cond block is walked too so the re-evaluated convergence predicate's
    ops (its ``residual`` / ``apply`` over the loop-updated iterate) are visible."""
    for node in values:
        out.append(node)
        attrs = node.attrs if hasattr(node, "attrs") else {}
        for key in ("cond_block", "body_block"):
            block = attrs.get(key)
            if isinstance(block, list):
                _walk_nodes(block, out)


def _require_field(value, where):
    if not (hasattr(value, "is_field") and value.is_field()):
        raise TypeError("%s: a State/RHS IR value is required; got %r" % (where, value))


def _operator_name(operator):
    """The linear-source name of a matrix-free operator IR value (or a bare string)."""
    if isinstance(operator, str):
        return operator
    name = getattr(operator, "attrs", {}).get("linear_source") if hasattr(operator, "attrs") else None
    if name is None:
        raise TypeError("apply: operator must be a linear-source IR value or a name; got %r"
                        % (operator,))
    return name


def _iter_all_nodes(values):
    """Yield @p values and the ops in any ``while`` cond/body sub-blocks, depth-first (build order)."""
    for v in values:
        yield v
        for key in ("cond_block", "body_block"):
            blk = v.attrs.get(key) if hasattr(v, "attrs") else None
            if isinstance(blk, list):
                yield from _iter_all_nodes(blk)
