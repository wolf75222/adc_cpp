"""adc.time -- compiled time-program DSL (builder-mode IR).

A `Program` is a restricted, COMPILED numerical program describing one time step. Python only BUILDS
a typed intermediate representation (IR); it never executes a numerical stage. Later ADC-399 phases
lower this IR to C++ in a combined `problem.so` that calls the adc_cpp runtime (`ProgramContext`)
during `sim.step(dt)`. This module is the IR construction layer only: no codegen, no runtime.

Builder example (Forward Euler)::

    P = adc.time.Program("forward_euler")
    dt = P.dt
    U = P.state("plasma")
    fields = P.solve_fields(U)
    R = P.rhs(state=U, fields=fields, flux=True, sources=["default"])
    U1 = P.linear_combine("U1", U + dt * R)
    P.commit("plasma", U1)

Values (`state`, `rhs`, `fields`, ...) are SSA nodes. State and RHS values support an affine algebra
(`U + dt * R`, `0.5 * U0 + 0.5 * (U1 + dt * k1)`, `dt / 6.0 * k1`): coefficients are polynomials in
`dt`, recorded per linear-combination input so the C++ codegen can emit a single fused kernel.

cf. docs/sphinx/reference/time-program.md (Phase 8) and the ADC-399 epic.
"""
import hashlib
import json
import types

__all__ = ["Program", "std", "CompiledTime", "StageStateSet", "Schedule",
           "always", "every", "when", "on_start", "on_end", "subcycle"]


class Schedule:
    """When a Program node is due, and what to do when it is not (Spec 3 unified scheduler).

    A Schedule is an inert IR annotation recorded on a node (``Value.attrs['schedule']``). The
    ``kind`` decides WHEN the node is due (``always`` every step, ``every(N)``, ``when(cond)``,
    ``on_start`` / ``on_end``, ``subcycle``); the ``policy`` decides what happens when it is NOT
    due (``recompute`` the default, ``hold`` the cached value, ``skip``, ``zero``,
    ``accumulate_dt``, or ``error``). Build a kind with the module helpers and set the policy by
    chaining: ``every(10).hold()``.

    Only ``always()`` runs at ``sim.step`` today: the runtime that honors a non-trivial schedule
    (the typed cache, ``accumulate_dt``, the checkpoint) is the C++ part of ADC-458, so a node
    carrying a non-always schedule is recorded and inspectable but refuses to lower (it is never
    silently ignored). See ``docs/sphinx/reference/program-scheduler.md``.
    """

    _KINDS = ("always", "every", "when", "on_start", "on_end", "subcycle")
    _POLICIES = ("recompute", "hold", "skip", "zero", "accumulate_dt", "error")
    # policies that reuse a stored value, so the operator must be cacheable
    _CACHING = ("hold", "accumulate_dt")

    def __init__(self, kind, policy="recompute", **params):
        if kind not in Schedule._KINDS:
            raise ValueError("schedule kind %r must be one of %s"
                             % (kind, ", ".join(Schedule._KINDS)))
        if policy not in Schedule._POLICIES:
            raise ValueError("schedule policy %r must be one of %s"
                             % (policy, ", ".join(Schedule._POLICIES)))
        self.kind = kind
        self.policy = policy
        self.params = dict(params)

    def is_always(self):
        """True for the default cadence (every step, recompute) -- the only schedule that lowers."""
        return self.kind == "always" and self.policy == "recompute"

    def needs_cache(self):
        """True if the policy reuses a stored value (so the operator must be cacheable)."""
        return self.policy in Schedule._CACHING

    def _with_policy(self, policy):
        return Schedule(self.kind, policy=policy, **self.params)

    def recompute(self):
        return self._with_policy("recompute")

    def hold(self):
        return self._with_policy("hold")

    def skip(self):
        return self._with_policy("skip")

    def zero(self):
        return self._with_policy("zero")

    def accumulate_dt(self):
        return self._with_policy("accumulate_dt")

    def error(self):
        return self._with_policy("error")

    def __repr__(self):
        if self.kind == "every":
            base = "every(%r)" % (self.params.get("n"),)
        elif self.kind == "subcycle":
            base = "subcycle(%r)" % (self.params.get("count"),)
        elif self.kind == "when":
            base = "when(...)"
        else:
            base = "%s()" % self.kind
        return base if self.policy == "recompute" else "%s.%s()" % (base, self.policy)


def always():
    """Due every step, recomputed -- the default cadence (the only schedule that runs today)."""
    return Schedule("always")


def every(n):
    """Due every ``n`` macro-steps (``n`` a positive int)."""
    if not (isinstance(n, int) and n > 0):
        raise ValueError("every(n): n must be a positive int, got %r" % (n,))
    return Schedule("every", n=n)


def when(cond):
    """Due when the runtime condition ``cond`` holds (a Program Bool value or a callable)."""
    return Schedule("when", cond=cond)


def on_start():
    """Due only at the first step."""
    return Schedule("on_start")


def on_end():
    """Due only at the last step."""
    return Schedule("on_end")


def subcycle(count, dt=None):
    """Structured sub-cycling: ``count`` inner steps (of ``dt`` each, default ``macro_dt/count``)."""
    if not (isinstance(count, int) and count > 0):
        raise ValueError("subcycle(count): count must be a positive int, got %r" % (count,))
    return Schedule("subcycle", count=count, dt=dt)


class _Coeff:
    """Scalar coefficient: a polynomial in the time step ``dt`` (dict ``power -> float``).

    ``dt`` is ``_Coeff({1: 1.0})``; a plain number is ``_Coeff({0: c})``. Multiplying a coefficient by
    a State/RHS value yields an `_Affine` (one weighted term)."""

    def __init__(self, powers):
        # drop exact zeros so {0: 0.0} and {} hash identically
        self.powers = {int(p): float(c) for p, c in powers.items() if c != 0.0}

    def _binop_number(self, x):
        return _Coeff({0: float(x)}) if isinstance(x, (int, float)) else None

    def __add__(self, other):
        o = self._binop_number(other) if not isinstance(other, _Coeff) else other
        if o is None:
            return NotImplemented
        out = dict(self.powers)
        for p, c in o.powers.items():
            out[p] = out.get(p, 0.0) + c
        return _Coeff(out)

    __radd__ = __add__

    def __neg__(self):
        return _Coeff({p: -c for p, c in self.powers.items()})

    def __sub__(self, other):
        return self.__add__(-(other if isinstance(other, _Coeff) else _Coeff({0: float(other)})))

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return _Coeff({p: c * other for p, c in self.powers.items()})
        if isinstance(other, _Coeff):
            out = {}
            for p1, c1 in self.powers.items():
                for p2, c2 in other.powers.items():
                    out[p1 + p2] = out.get(p1 + p2, 0.0) + c1 * c2
            return _Coeff(out)
        if isinstance(other, Value) and other.is_field():
            return _Affine([(other, self)])
        if isinstance(other, _Affine):
            return other.__mul__(self)
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return _Coeff({p: c / other for p, c in self.powers.items()})
        return NotImplemented

    def as_dict(self):
        return dict(self.powers)

    def _key(self):
        return tuple(sorted(self.powers.items()))


def _to_affine(x):
    if isinstance(x, _Affine):
        return x
    if isinstance(x, Value) and x.is_field():
        return _Affine([(x, _Coeff({0: 1.0}))])
    raise TypeError("expected a State/RHS value or an affine combination, got %r" % (x,))


def _is_field_value(x):
    """True for a grid-field Value (State / RHS / scalar_field) -- the values that carry an
    adc::MultiFab and support the affine algebra."""
    return isinstance(x, Value) and x.is_field()


def _affine_ids(aff):
    """Stable JSON-able form of an _Affine (the apply result of a matrix_free_operator): an ordered
    list of ``[value_id, sorted-coeff-powers]``."""
    return [[v.id, sorted((int(p), c) for p, c in coeff.as_dict().items())]
            for v, coeff in aff._merge()]


def _residual_wants_guess(fn):
    """True if a `solve_local_nonlinear` residual callable takes the frozen guess as a third positional
    arg (``residual_fn(P, U, U0)``); False for the two-arg form (``residual_fn(P, U)``). A ``*args``
    callable is treated as wanting the guess (it can accept it)."""
    import inspect
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):  # builtins / C callables: pass the guess and let the call decide
        return True
    positional = [p for p in params
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if any(p.kind == p.VAR_POSITIONAL for p in params):
        return True
    return len(positional) >= 3


class _Affine:
    """Affine combination of State/RHS values: ordered ``[(value, _Coeff)]`` terms. Built by the
    operator overloads on field values; consumed by `Program.linear_combine`."""

    def __init__(self, terms):
        self.terms = list(terms)

    def _merge(self):
        # coalesce repeated values (sum their coefficient polynomials), preserve first-seen order
        order, acc = [], {}
        for v, c in self.terms:
            if id(v) not in acc:
                order.append(v)
                acc[id(v)] = (v, c)
            else:
                acc[id(v)] = (v, acc[id(v)][1] + c)
        return [acc[id(v)] for v in order]

    def __add__(self, other):
        return _Affine(self.terms + _to_affine(other).terms)

    __radd__ = __add__

    def __neg__(self):
        return _Affine([(v, -c) for v, c in self.terms])

    def __sub__(self, other):
        return _Affine(self.terms + (-_to_affine(other)).terms)

    def __rsub__(self, other):
        return _Affine((-self).terms + _to_affine(other).terms)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = _Coeff({0: float(other)})
        if isinstance(other, _Coeff):
            return _Affine([(v, c * other) for v, c in self.terms])
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return _Affine([(v, c / other) for v, c in self.terms])
        return NotImplemented

    def __bool__(self):
        raise TypeError("runtime affine combination cannot be used as a Python bool; "
                        "use Program control flow")


class _Operator:
    """A LOCAL linear operator expression ``c_I * I + sum_k c_k * L_k`` (coefficients are `_Coeff`,
    polynomials in dt). Built by ``Program.I`` and ``a * Program.linear_source(name)``; consumed by
    `Program.solve_local_linear` (operator ``I +/- a*L``). It is NOT a runtime field value -- it names
    the model linear source(s) and the scalar(s) that form the operator."""

    def __init__(self, identity, terms):
        self.identity = identity   # _Coeff: coefficient of the identity I
        self.terms = list(terms)   # [(Value(op='linear_source'), _Coeff)]

    def __add__(self, other):
        if not isinstance(other, _Operator):
            return NotImplemented
        return _Operator(self.identity + other.identity, self.terms + other.terms)

    def __sub__(self, other):
        if not isinstance(other, _Operator):
            return NotImplemented
        return _Operator(self.identity - other.identity,
                         self.terms + [(v, -c) for v, c in other.terms])

    def __neg__(self):
        return _Operator(-self.identity, [(v, -c) for v, c in self.terms])

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = _Coeff({0: float(other)})
        if isinstance(other, _Coeff):
            return _Operator(self.identity * other, [(v, c * other) for v, c in self.terms])
        return NotImplemented

    __rmul__ = __mul__


class Value:
    """A typed SSA node in a Program IR. Field-like values (State, RHS, scalar_field) support affine
    arithmetic. A ``scalar_field`` is a single-component grid field (the unknown / residual of a
    matrix-free linear solve), DISTINCT from the n_cons conservative ``state`` even though both lower
    to an adc::MultiFab; a ``matrix_free_op`` names a matrix-free operator A whose apply sub-block is
    recorded by ``set_apply`` and lowered to a C++ lambda the runtime Krylov loop calls."""

    _FIELD = ("state", "rhs", "scalar_field")
    _SCALAR = ("scalar", "bool")  # runtime scalars / predicates: never a Python bool / index

    def __init__(self, prog, vid, vtype, op, inputs, attrs, name, block):
        self.prog = prog
        self.id = vid
        self.vtype = vtype
        self.op = op
        self.inputs = tuple(inputs)
        self.attrs = dict(attrs)
        self.name = name
        self.block = block
        # Operator-first type tag (Spec 2): the adc.model space/operator-type this value lives over
        # (a StateSpace / RateSpace / FieldSpace / LocalLinearOperator), set by P.state(space=) and
        # P.call. Used only for build-time type checks; NEVER serialized into the IR. None = untyped
        # (legacy), and all the space checks are skipped (backward compatible).
        self.space = None

    def is_field(self):
        return self.vtype in Value._FIELD

    def __bool__(self):
        # A runtime Scalar/Bool is decided at step time, not at IR-build time: it must NEVER silently
        # collapse to a Python bool (which would make `while P.norm2(d) > tol:` loop in Python). Fields
        # (State/RHS) keep their own loud refusal too.
        if self.vtype in Value._SCALAR:
            raise TypeError(
                "a Program %s (%r) cannot be used as a Python bool; use P.while_ / P.if_ for runtime "
                "control flow" % (self.vtype, self.name))
        raise TypeError(
            "a Program %s value (%r) cannot be used as a Python bool; it is a runtime field, not a "
            "compile-time condition" % (self.vtype, self.name))

    def __index__(self):
        # range(scalar) / using a Scalar as a Python index is just as loud: the value is unknown until
        # the step runs.
        raise TypeError(
            "a Program %s (%r) cannot be used as a Python index; use P.while_ / P.if_ for runtime "
            "control flow" % (self.vtype, self.name))

    # --- scalar comparisons (scalar values only): build a Bool predicate, do not compare in Python ---
    def _compare(self, other, cmp):
        if self.vtype != "scalar":
            raise TypeError("%s value %r is not a scalar; only P.norm2 / P.dot results compare"
                            % (self.vtype, self.name))
        return self.prog._compare(self, other, cmp)

    def __gt__(self, other):
        return self._compare(other, ">")

    def __lt__(self, other):
        return self._compare(other, "<")

    def __ge__(self, other):
        return self._compare(other, ">=")

    def __le__(self, other):
        return self._compare(other, "<=")

    # --- affine algebra (field values only) ---
    def _affine(self):
        if not self.is_field():
            raise TypeError("%s value %r is not a field; only State/RHS support arithmetic"
                            % (self.vtype, self.name))
        return _to_affine(self)

    # --- scalar arithmetic (scalar values only): build a scalar_op node, NOT a Python float ---
    # A runtime Scalar (a reduction, max_wave_speed, hmin, the dt_bound's cfl) composes into a new
    # Scalar via + - * / so a dt_bound can express e.g. cfl * P.hmin() / P.max_wave_speed(U) (spec s18);
    # the value is unknown until the step runs, so the arithmetic builds IR, it is never evaluated here.
    def _scalar_op(self, other, fn, swap=False):
        if self.vtype != "scalar":
            raise NotImplementedError("scalar arithmetic is only defined for Scalar values")
        a, b = (other, self) if swap else (self, other)
        return self.prog._scalar_binop(a, b, fn)

    def __add__(self, other):
        if self.vtype == "scalar":
            return self._scalar_op(other, "add")
        return self._affine() + _to_affine(other)

    def __radd__(self, other):
        if self.vtype == "scalar":
            return self._scalar_op(other, "add", swap=True)
        return self._affine() + _to_affine(other)

    def __neg__(self):
        if self.vtype == "scalar":
            return self._scalar_op(-1.0, "mul")
        return -self._affine()

    def __sub__(self, other):
        if self.vtype == "scalar":
            return self._scalar_op(other, "sub")
        return self._affine() - _to_affine(other)

    def __rsub__(self, other):
        if self.vtype == "scalar":
            return self._scalar_op(other, "sub", swap=True)
        return _to_affine(other) - self._affine()

    def __mul__(self, other):
        if self.vtype == "scalar":
            return self._scalar_op(other, "mul")
        if self.vtype == "operator":  # a linear-source operator: scalar/dt * L -> an _Operator term
            if isinstance(other, (int, float)):
                other = _Coeff({0: float(other)})
            if isinstance(other, _Coeff):
                return _Operator(_Coeff({}), [(self, other)])
            return NotImplemented
        if isinstance(other, (int, float, _Coeff)):
            return self._affine() * other
        return NotImplemented

    def __rmul__(self, other):
        if self.vtype == "scalar":
            return self._scalar_op(other, "mul", swap=True)
        return self.__mul__(other)

    def __truediv__(self, other):
        if self.vtype == "scalar":
            return self._scalar_op(other, "div")
        if isinstance(other, (int, float)):
            return self._affine() * _Coeff({0: 1.0 / other})
        return NotImplemented

    def __rtruediv__(self, other):
        if self.vtype == "scalar":
            return self._scalar_op(other, "div", swap=True)
        return NotImplemented

    # --- operator application (Spec 3 board notation): operator @ state -> apply ---
    def __matmul__(self, other):
        """``L @ U`` -- apply a linear-source operator value ``L`` to a state ``U``.

        Returns the RHS-like value ``P.apply(operator=L, state=U)``. For
        ``operator @ unknown(name)`` (a board solve), this returns ``NotImplemented``
        so :meth:`adc.math.Unknown.__rmatmul__` builds the solve left-hand side.
        """
        if self.vtype == "operator" and isinstance(other, Value) and other.vtype == "state":
            return self.prog.apply(operator=self, state=other)
        return NotImplemented

    def __repr__(self):
        return "<%s %s #%d>" % (self.vtype, self.name, self.id)


def _state_base_name(space):
    """The StateSpace name a value lives over, for operator-first type checks: a StateSpace -> its
    name; a Rate(state) -> the base state name; anything else (FieldSpace / operator-type / None) ->
    None (not a state-like value)."""
    kind = getattr(space, "kind", None)
    if kind == "state":
        return space.name
    if kind == "rate":
        return getattr(space, "base_name", None)
    return None


class StageStateSet:
    """A coherent set of stage states (Spec 3): ``{block -> State value}``.

    Built by :meth:`Program.state_set`; consumed by :meth:`Program.fields` to solve
    fields from a chosen stage of several blocks at once, without ambiguity about
    which version of each block is read. Every entry must be a State value.
    """

    def __init__(self, name, mapping):
        self.name = str(name)
        self._states = {}
        for block, st in dict(mapping).items():
            if not (isinstance(st, Value) and st.vtype == "state"):
                raise ValueError("StageStateSet[%r] must be a State value" % (block,))
            self._states[str(block)] = st

    def states(self):
        """The stage states in insertion order."""
        return list(self._states.values())

    def __getitem__(self, block):
        return self._states[str(block)]

    def __contains__(self, block):
        return str(block) in self._states

    def keys(self):
        return list(self._states)

    def items(self):
        return list(self._states.items())

    def __len__(self):
        return len(self._states)

    def __repr__(self):
        return "StageStateSet(%r, blocks=%s)" % (self.name, list(self._states))


class _CoupledResult:
    """The typed multi-output of a coupled_rate ``P.call``: a mapping ``block -> per-block rate``.

    ``C = P.call("collision", e_n, i_n)`` returns this; ``C["electrons"]`` is the per-block rate
    (an RHS Value) that composes like any other (``e_n + dt * C["electrons"]``). It is not itself
    a Value: a coupled operator has no single output, so it cannot be combined as one.
    """

    def __init__(self, outs):
        self._outs = dict(outs)

    def __getitem__(self, block):
        return self._outs[block]

    def __contains__(self, block):
        return block in self._outs

    def keys(self):
        return list(self._outs)

    def items(self):
        return list(self._outs.items())

    def __len__(self):
        return len(self._outs)

    def __repr__(self):
        return "_CoupledResult(blocks=%s)" % list(self._outs)


class Program:
    """A compiled time program (builder mode). Holds the SSA value list and the committed blocks.

    The Python object only BUILDS the IR; it is never executed numerically during ``sim.step``."""

    def __init__(self, name):
        self.name = name
        self._values = []
        self._next_id = 0
        self._commits = {}      # block -> State value
        self._recording = []    # stack of sub-block lists (a control-flow body); see _new / while_
        self._histories = {}    # name -> max declared lag (multistep histories; ADC-406a)
        # OPTIONAL dt bound (spec s18 / ADC-417): a recorded scalar sub-program (cfl -> Scalar) the
        # generated .so exports as adc_program_dt_bound; None = no bound (the native CFL is used).
        self._dt_bound = None        # (block, scalar_value) once set; the block is the scalar sub-block
        self.dt = _Coeff({1: 1.0})   # symbolic time step; participates in coefficient arithmetic
        # OPTIONAL bound operator registry (Spec 2, operator-first): set by bind_operators so P.call
        # can resolve and type-check operators at build time. None = legacy PDE-shortcut-only Program.
        self._registry = None
        # Per-emit scratch names of coupled_rate blocks, keyed by (coupled node id, block): the
        # coupled_rate kernel fills them and each coupled_rate_out projection aliases its block's
        # scratch (ADC-457). Populated during _emit_op; harmless to keep across emits (keys are unique
        # per node id).
        self._coupled_scratch = {}

    # --- node construction ---
    def _new(self, vtype, op, inputs, attrs, name, block):
        for v in inputs:
            if isinstance(v, Value) and v.prog is not self:
                raise ValueError("IR value %r belongs to a different Program" % (v,))
        vid = self._next_id
        self._next_id += 1
        v = Value(self, vid, vtype, op, [i for i in inputs if isinstance(i, Value)],
                  attrs, name or ("%s%d" % (op, vid)), block)
        # Inside a control-flow recording scope (cond_fn / body_fn of a while_), ops go into the active
        # sub-block, NOT the flat self._values: a while body must RE-EXECUTE each iteration, so its ops
        # are owned by the while op and re-emitted in the loop, never walked once at the top level.
        if self._recording:
            self._recording[-1].append(v)
        else:
            self._values.append(v)
        return v

    def state(self, block, space=None):
        """Reference the current conservative state of ``block`` at the start of the step.

        @p space (Spec 2): the operator-first :class:`adc.model.StateSpace` this block instantiates.
        It is recorded for type checking (a State tagged with space U cannot be combined with a
        Rate(V), and an operator expecting state U cannot be called on it) and is NOT serialized into
        the IR. ``None`` keeps the legacy untyped state (no space checks)."""
        if not isinstance(block, str) or not block:
            raise ValueError("state: block must be a non-empty string")
        v = self._new("state", "state", (), {}, block, block)
        v.space = space
        return v

    def solve_fields(self, name=None, state=None, field=None):
        """Solve the elliptic fields from ``state`` and return a FieldContext. Accepts
        ``solve_fields(state)`` or ``solve_fields(name, state)``. Each call is a DISTINCT
        FieldContext (a stage's RHS must read the fields solved from its own state, never a stale
        global). @p field (ADC-419) names a NAMED elliptic field (m.elliptic_field) to solve instead of
        the default Poisson coupling; its derived aux populate that field's named aux channel. The
        multi-field RUNTIME is DEFERRED: a non-None @p field lowers to a clear NotImplementedError
        (the IR records it so a program reads cleanly when the runtime lands)."""
        if isinstance(name, Value) and state is None:
            name, state = None, name
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("solve_fields: a State value is required")
        if field is not None and not (isinstance(field, str) and field):
            raise ValueError("solve_fields: field must be a non-empty named elliptic field")
        # The attr is added ONLY for a named field so a default solve_fields keeps its historical IR
        # (empty attrs) -> the .so cache key of an existing time program is byte-identical (no spurious
        # invalidation from this feature).
        attrs = {"field": field} if field is not None else {}
        return self._new("fields", "solve_fields", (state,), attrs, name, state.block)

    def solve_fields_from_blocks(self, states, name=None):
        """Solve the elliptic fields from the SIMULTANEOUS stage states of MULTIPLE blocks (spec
        \"Multi-blocs\"): a coupled Poisson where each listed block reads its own @p states[k] override
        at once, returning a FieldContext.

        RUNTIME (Spec 3 criterion 24, ADC-457): this is the multi-target coupled solve. It lowers to
        ``ctx.solve_fields_from_blocks(u_stages)``, a per-block pointer vector the native field solver
        (``System::solve_fields_from_blocks`` ->
        ``SystemFieldSolver::assemble_poisson_rhs_from_blocks``) assembles the system Poisson RHS from as
        ``Sum_s elliptic_rhs_s(U_s)`` reading EVERY listed block's stage state AT ONCE (a true
        simultaneous override, not a sequence of single-target solves). A block NOT listed contributes
        its live state. The listed states slot at their block index (the P.state declaration order), so
        the runtime sees each coupled block at its stage state into the one shared phi/aux.

        A per-block ``P.solve_fields(state=Ub)`` remains the right choice when the blocks advance in
        sequence (block b at its stage state, every other block at its live state); this op is for the
        SIMULTANEOUS case where multiple coupled blocks must each contribute their stage state at once."""
        if not (isinstance(states, (list, tuple)) and states):
            raise ValueError("solve_fields_from_blocks: a non-empty list of State values is required")
        seen = set()
        for s in states:
            if not (isinstance(s, Value) and s.vtype == "state"):
                raise ValueError("solve_fields_from_blocks: every entry must be a State value")
            if s.block in seen:
                raise ValueError("solve_fields_from_blocks: block '%s' listed twice" % s.block)
            seen.add(s.block)
        # The FieldContext is attached to the first listed block (an arbitrary but stable owner).
        return self._new("fields", "solve_fields_from_blocks", tuple(states), {}, name, states[0].block)

    # --- operator-first calls (Spec 2) -------------------------------------------
    def bind_operators(self, source):
        """Bind a typed operator registry so ``P.call`` can resolve and type-check operators.

        ``source`` is an ``adc.model.OperatorRegistry`` or any object exposing
        ``operator_registry()`` (a ``dsl.Model`` / ``adc.model.Module``). Returns ``self`` for
        chaining. The bound registry is build-time TYPE information only -- the codegen still reads
        the model passed to ``compile_problem``; operator-first Programs and the ``adc.time.std``
        macros bind the module's operators here.
        """
        reg = source.operator_registry() if hasattr(source, "operator_registry") else source
        if not (hasattr(reg, "get") and hasattr(reg, "names")):
            raise TypeError("bind_operators: expected an OperatorRegistry or an object exposing "
                            "operator_registry(); got %r" % (source,))
        self._registry = reg
        return self

    def call(self, operator_name, *args, name=None, schedule=None):
        """Call a typed operator by name (the operator-first level).

        Resolves ``operator_name`` against the bound operator registry (see :meth:`bind_operators`),
        type-checks the arguments against the operator's ``Signature``, then lowers to the equivalent
        primitive op so the result is IDENTICAL to the matching PDE shortcut: a ``field_operator`` to
        ``solve_fields``, a ``local_source`` to ``source``, a ``grid_operator`` / ``local_rate`` to
        ``rhs``, a ``local_linear_operator`` to ``linear_source``, a ``projection`` to ``project``.
        A Program composes operators by signature, never by a hardcoded PDE category.
        """
        if self._registry is None:
            raise ValueError("P.call(%r): no operators bound; call P.bind_operators(model) first"
                             % (operator_name,))
        op = self._registry.get(operator_name)  # clear KeyError on an unknown operator
        self._check_call_args(op, args)
        if schedule is not None:
            self._validate_schedule(op, schedule)
        result = self._lower_call(op, operator_name, args, name)
        # A coupled_rate has no single output Value (it returns a _CoupledResult): its per-block
        # spaces are tagged inside _lower_coupled_rate, and a schedule on the whole bundle is not
        # meaningful yet -- reject it with a clear message rather than leaking an AttributeError.
        if isinstance(result, _CoupledResult):
            if schedule is not None:
                raise ValueError(
                    "schedule= is not supported on a coupled_rate operator (%r) yet; schedule its "
                    "per-block consumers instead (ADC-457/458)" % (operator_name,))
            return result
        # Tag the result with the operator's declared output type (a Rate / FieldSpace /
        # LocalLinearOperator / StateSpace) so downstream ops can type-check the composition
        # (a Rate(U) cannot be combined with a State(V); an L: U -> U cannot drive a State(V)).
        result.space = op.signature.output
        if schedule is not None:
            result.attrs["schedule"] = schedule
        return result

    def _validate_schedule(self, op, schedule):
        """A schedule= on P.call must be a Schedule; a caching policy (hold / accumulate_dt)
        requires the operator to be cacheable (Spec 3 criterion 27)."""
        if not isinstance(schedule, Schedule):
            raise TypeError(
                "schedule= expects an adc.time Schedule (always()/every(n)/...), got %r"
                % (schedule,))
        if schedule.needs_cache() and not op.capabilities.get("cacheable"):
            raise ValueError(
                "operator %r is not cacheable; cannot use schedule %s -- declare it with "
                "m.operator_capabilities(%r, cacheable=True)"
                % (op.name, schedule.policy, op.name))

    def _lower_call(self, op, operator_name, args, name):
        kind = op.kind
        if kind == "field_operator":
            # A multi-input field operator (e.g. fields_from_species over N species) is the COUPLED
            # multi-block field solve: every input species contributes to the one shared elliptic RHS
            # (Sum_s elliptic_rhs_s, the default phi). Route it to solve_fields_from_blocks so no
            # species is dropped -- a single-input field operator stays the historical single-block
            # solve_fields (named-field routing via the operator name as before).
            state_args = [a for a in args if getattr(a, "vtype", None) == "state"]
            if len(state_args) > 1:
                return self.solve_fields_from_blocks(state_args, name=name)
            field = None if operator_name == "fields_from_state" else operator_name
            return self.solve_fields(name=name, state=args[0], field=field)
        if kind == "local_source":
            fields = args[1] if len(args) > 1 else None
            if operator_name == "source_default":
                # The default source lives in m._source, not as a named source_term; reach it
                # through the source-only RHS path (byte-identical to P.rhs(flux=False,
                # sources=["default"])), since ctx.source(name) only resolves named source_terms.
                return self.rhs(name=name, state=args[0], fields=fields, flux=False,
                                sources=["default"])
            return self.source(operator_name, state=args[0], fields=fields)
        if kind in ("grid_operator", "local_rate"):
            fields = args[1] if len(args) > 1 else None
            if kind == "grid_operator":
                # Flux divergence only (no source): the default flux or a named flux_term.
                fluxes = None if operator_name == "flux_default" else [operator_name]
                return self.rhs(name=name, state=args[0], fields=fields, flux=True,
                                sources=[], fluxes=fluxes)
            low = op.lowering
            return self.rhs(name=name, state=args[0], fields=fields,
                            flux=low.get("flux", True), sources=low.get("sources"),
                            fluxes=low.get("fluxes"))
        if kind == "local_linear_operator":
            return self.linear_source(operator_name)
        if kind == "projection":
            return self.project(name=name, state=args[0])
        if kind == "coupled_rate":
            return self._lower_coupled_rate(op, operator_name, args, name)
        raise NotImplementedError(
            "P.call: operator kind %r is not yet lowerable (operator %r)" % (kind, operator_name))

    def _lower_coupled_rate(self, op, operator_name, args, name):
        """Lower a coupled_rate operator to a coupled node plus one per-block rate projection.

        A coupled operator (collisions, ionization, ...) of arbitrary arity returns a typed
        ``RateBundle``; ``P.call`` returns a :class:`_CoupledResult` whose ``["electrons"]`` is the
        per-block rate (an RHS Value over that block) so it composes like any other RHS. The
        coupled-rate KERNEL codegen has landed (ADC-457): ``_emit_coupled_rate_kernel`` lowers the
        ``coupled_rate`` node to one multi-state ``for_each_cell`` and each ``coupled_rate_out``
        projects its block's rate scratch.
        """
        bundle = op.signature.output                 # a model.RateBundle: block -> RateSpace
        blocks = bundle.keys()
        base = name or operator_name
        coupled = self._new("rhs", "coupled_rate", tuple(args),
                            {"operator": operator_name, "blocks": list(blocks)},
                            base, args[0].block)
        outs = {}
        for blk in blocks:
            out = self._new("rhs", "coupled_rate_out", (coupled,),
                            {"operator": operator_name, "out_block": blk},
                            "%s_%s" % (base, blk), blk)
            out.space = bundle[blk]                   # the per-block RateSpace, for type checks
            outs[blk] = out
        return _CoupledResult(outs)

    def _check_call_args(self, op, args):
        """Type-check ``P.call`` arguments against an operator's Signature: arity plus the vtype of
        each space-typed input (a StateSpace input wants a 'state' value, a FieldSpace input a
        'fields' value). Operator-valued inputs are not passed positionally. Clear error on mismatch.
        """
        expected = [t for t in op.signature.inputs
                    if getattr(t, "kind", None) in ("state", "field")]
        if len(args) != len(expected):
            raise ValueError(
                "operator %r expects %d argument(s) %s, got %d"
                % (op.name, len(expected), tuple(t.name for t in expected), len(args)))
        want_of = {"state": "state", "field": "fields"}
        for t, a in zip(expected, args, strict=True):
            want = want_of[t.kind]
            if not (isinstance(a, Value) and a.vtype == want):
                got = a.vtype if isinstance(a, Value) else type(a).__name__
                raise ValueError(
                    "operator %r argument for %s %r expects a %s value, got %s"
                    % (op.name, t.kind, t.name, want, got))
            # If the argument carries an operator-first space tag, its name must match the
            # operator's declared input space (a value over 'V' cannot feed an input typed 'U').
            arg_space = getattr(a, "space", None)
            arg_name = getattr(arg_space, "name", None)
            if arg_name is not None and arg_name != t.name:
                raise ValueError(
                    "operator %r expects %s %r but got a value over %r"
                    % (op.name, t.kind, t.name, arg_name))

    def rhs(self, name=None, state=None, fields=None, flux=True, sources=None, fluxes=None):
        """Build R = -div F(U) + sum of the requested named ``sources``. ``fields`` is the explicit
        FieldContext any field-dependent source reads (no implicit global aux). Named sources are
        never summed implicitly: ``sources`` lists exactly the ones to include.

        ``sources`` selects which sources are folded in (ADC-425, spec criterion 17): ``None`` (the
        argument default) keeps the legacy behavior (``-div F`` + the model's default/composite source,
        byte-identical to before); ``["default"]`` is the same explicitly; ``[]`` (an EMPTY list) is
        FLUX ONLY (``-div F`` with NO default source) -- the hyperbolic stage of a Lie/Strang/IMEX
        split; a list of named ``m.source_term`` names adds exactly those (plus the default iff
        ``"default"`` is in the list). So ``None``/``["default"]`` -> flux + default source, ``[]`` ->
        flux only, ``["a","b"]`` -> flux + a + b. ``None`` and ``[]`` are recorded DISTINCTLY in the IR.

        ``flux`` (ADC-430) toggles the ``-div F`` base. ``flux=True`` (the default) is the above.
        ``flux=False`` is SOURCE-ONLY: no flux divergence at all -- the RHS is just the requested
        ``sources`` (the source stage of a Lie/Strang/IMEX split). So ``flux=False, sources=["default"]``
        (or the bare ``flux=False``) is the model's default source only, ``flux=False, sources=["s"]`` is
        just the named ``s``, and ``flux=False, sources=[]`` is the zero RHS. Named ``fluxes`` with
        ``flux=False`` are rejected (a source-only stage has no flux to divide). Before ADC-430 the
        codegen ignored ``flux=False`` and still emitted ``-div F``, double-adding the flux on a
        non-zero-flux model in a split (it was masked only because split source stages were tested on
        zero-flux models).

        ``fluxes`` (ADC-419) selects the PHYSICAL flux: ``None`` or ``["default"]`` uses the model's
        historical -div F (the compiled Riemann/FV path, byte-identical to before). A list of NAMED
        fluxes (``m.flux_term``) assembles -div of their SUM via centered FV differencing of the
        evaluated flux fields: so splitting the physical flux into named pieces that sum to it
        reproduces the same -div F to round-off. Mixing ``"default"`` with named fluxes is rejected
        (the two divergence stencils differ); request either the default or a set of named fluxes."""
        if isinstance(name, Value):
            raise ValueError("rhs: pass state=/fields= by keyword (first arg is the debug name)")
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("rhs: a State value is required (state=...)")
        if fields is not None and not (isinstance(fields, Value) and fields.vtype == "fields"):
            raise ValueError("rhs: fields must be a FieldContext from solve_fields")
        # Preserve None (legacy default = flux + default source) DISTINCT from [] (flux only): the
        # codegen routes on whether "default" is requested, and None is the legacy "default included".
        src = list(sources) if sources is not None else None
        attrs = {"flux": bool(flux), "sources": src, "fluxes": list(fluxes) if fluxes else None}
        inputs = (state, fields) if fields is not None else (state,)
        return self._new("rhs", "rhs", inputs, attrs, name, state.block)

    def linear_combine(self, name=None, expr=None):
        """Materialize an affine combination of State/RHS values into a new State. Accepts
        ``linear_combine(name, expr)`` or ``linear_combine(expr)``. The per-input coefficient
        polynomials in ``dt`` are recorded in ``attrs['coeffs']`` (aligned with ``inputs``)."""
        if expr is None and not isinstance(name, str):
            name, expr = None, name
        aff = _to_affine(expr)._merge()
        if not aff:
            raise ValueError("linear_combine: empty combination")
        block = None
        state_space = None
        for v, _ in aff:
            if v.vtype == "state":
                block = v.block
                if state_space is None:
                    state_space = v.space
                break
        if block is None:
            block = aff[0][0].block
        # Operator-first type check (Spec 2): every State/Rate term must live over ONE StateSpace.
        # Combining a Rate(U) with a State(V) (V != U) is a type error; untyped (legacy) terms skip.
        spaces = {nm for nm in (_state_base_name(v.space) for v, _ in aff) if nm is not None}
        if len(spaces) > 1:
            raise ValueError(
                "cannot combine values over different state spaces %s; a State and the Rate(state) "
                "added to it must share one StateSpace" % sorted(spaces))
        inputs = tuple(v for v, _ in aff)
        coeffs = [c.as_dict() for _, c in aff]
        out = self._new("state", "linear_combine", inputs, {"coeffs": coeffs}, name, block)
        out.space = state_space  # the combine result is a State over the same space
        return out

    # --- named sources / local linear operators (Phase 4 / ADC-403) ---
    @property
    def I(self):  # noqa: E743  -- the mathematical identity operator (matches the spec's P.I)
        """The identity operator, for building a local linear operator ``self.I - a * L`` (L a
        linear source). Consumed by `solve_local_linear`."""
        return _Operator(_Coeff({0: 1.0}), [])

    def linear_source(self, name):
        """Reference a model linear-source operator ``L_name`` (declared via ``m.linear_source``).
        Use it in operator algebra (``self.I - a * P.linear_source('lorentz')``) or `apply`. The
        coefficients of L are the model's; the Program only names it (resolved at compile time)."""
        if not isinstance(name, str) or not name:
            raise ValueError("linear_source: name must be a non-empty string")
        return self._new("operator", "linear_source", (), {"linear_source": name}, name, None)

    def source(self, name, state=None, fields=None):
        """Evaluate a single named model source ``S_name(U, fields)`` (``m.source_term``) on its own.
        Returns an RHS-like value (a dU/dt contribution) usable in linear combinations. Named sources
        are never summed implicitly; this requests exactly one."""
        if not isinstance(name, str) or not name:
            raise ValueError("source: a non-empty source name is required")
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("source: a State value is required (state=...)")
        if fields is not None and not (isinstance(fields, Value) and fields.vtype == "fields"):
            raise ValueError("source: fields must be a FieldContext from solve_fields")
        inputs = (state, fields) if fields is not None else (state,)
        return self._new("rhs", "source", inputs, {"source": name}, name, state.block)

    def _check_operator_state(self, l_value, state_value, where):
        """Operator-first type check (Spec 2): a LocalLinearOperator L: U -> U may only act on a State
        over U. Fires only when both carry space tags (P.call / P.state(space=)); legacy skips."""
        lop = getattr(l_value, "space", None) if isinstance(l_value, Value) else None
        dom = getattr(lop, "domain_name", None)
        st = _state_base_name(getattr(state_value, "space", None))
        if dom is not None and st is not None and dom != st:
            raise ValueError(
                "%s: operator maps %s -> %s but was applied to a State over %r"
                % (where, dom, getattr(lop, "range_name", dom), st))

    def apply(self, operator=None, state=None, fields=None, name=None):
        """Apply a linear-source operator to a state: ``LU = L_name(aux, params) U``. ``operator`` is
        a `linear_source` value (or its name). Returns an RHS-like value."""
        lname = self._linear_source_name(operator, "apply")
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("apply: a State value is required (state=...)")
        if fields is not None and not (isinstance(fields, Value) and fields.vtype == "fields"):
            raise ValueError("apply: fields must be a FieldContext from solve_fields")
        self._check_operator_state(operator, state, "apply")
        inputs = (state, fields) if fields is not None else (state,)
        return self._new("rhs", "apply", inputs, {"linear_source": lname},
                         name or ("apply_" + lname), state.block)

    def solve_local_linear(self, name=None, operator=None, rhs=None, fields=None):
        """Solve a LOCAL linear system ``operator U = rhs`` cell by cell, where
        ``operator = self.I +/- a*L`` for a single model linear source ``L`` (``a`` may depend on dt
        / constants). Returns the solution State. A non-local or non-linear operator is rejected; the
        per-cell dense fallback bound (n_cons <= 8) is enforced by the codegen (a later phase)."""
        if not isinstance(operator, _Operator) or operator.identity.as_dict() != {0: 1.0}:
            raise ValueError("solve_local_linear currently supports local linear operators only")
        if len(operator.terms) != 1:
            raise NotImplementedError(
                "solve_local_linear currently supports a single linear source (I +/- a*L); got %d "
                "term(s)" % len(operator.terms))
        if not (isinstance(rhs, Value) and rhs.vtype == "state"):
            raise ValueError("solve_local_linear: rhs must be a State value (rhs=...)")
        if fields is not None and not (isinstance(fields, Value) and fields.vtype == "fields"):
            raise ValueError("solve_local_linear: fields must be a FieldContext from solve_fields")
        op_value, l_coeff = operator.terms[0]
        self._check_operator_state(op_value, rhs, "solve_local_linear")
        lname = op_value.attrs["linear_source"]
        a = (-l_coeff).as_dict()  # operator = I - a*L, so the L term carries the coefficient -a
        inputs = (rhs, op_value, fields) if fields is not None else (rhs, op_value)
        out = self._new("state", "solve_local_linear", inputs,
                        {"linear_source": lname, "a_coeff": a}, name, rhs.block)
        out.space = rhs.space  # the solution is a State over the same space as the rhs
        return out

    # The LOCAL per-cell ops a solve_local_nonlinear residual sub-block may use: the iterate / guess
    # State placeholders, named per-cell sources / linear-source applies, and the affine combine of
    # them. All lower to a per-cell scalar expression in the cell-local conservative stack -- NO
    # non-local op (rhs / divergence / solve_fields / a nested solve) is allowed (it would need a halo
    # / global solve, which a per-cell Newton kernel cannot evaluate at a perturbed stack state).
    _RESIDUAL_LOCAL_OPS = frozenset({"state", "source", "apply", "linear_combine"})

    def solve_local_nonlinear(self, name=None, residual=None, initial_guess=None, method="newton",
                              tol=1e-12, max_iter=20):
        """Solve a LOCAL non-linear system ``residual(U) = 0`` cell by cell with a per-cell Newton
        iteration (spec op 10). Returns the converged solution State.

        @p residual is an IR-building callable ``residual_fn(P, U, U0) -> State``: given the Newton
        iterate State @p U and the frozen initial-guess State @p U0 it BUILDS the residual ``r(U)`` (a
        State value) from LOCAL per-cell ops only -- ``P.source`` (a named ``m.source_term``),
        ``P.apply`` (a named ``m.linear_source``), the iterate / initial-guess States, and the affine
        algebra over them (e.g. an implicit reaction ``r(U) = U - U0 - dt*S(U)``). A non-local op
        (``P.rhs`` / ``P.divergence`` / ``P.solve_fields`` / a nested solve) is rejected: the residual
        must be re-evaluable at a PERTURBED cell-local stack state, which a halo / global solve cannot.
        The sub-block (like a ``set_apply`` body) lowers to a device-inlinable per-cell residual the
        kernel re-evaluates at ``U`` and at the finite-difference perturbations ``U + eps*e_j``. A
        two-argument ``residual_fn(P, U)`` (ignoring the guess) is also accepted.

        @p initial_guess is the start State ``U0`` (typically ``U^n``); it seeds the Newton iterate and
        the residual reads it as a frozen per-cell constant. @p method is ``"newton"`` (the only
        method). @p tol is the convergence threshold on ``max_c |r_c|`` (per cell) and @p max_iter the
        iteration budget (the kernel runs a fixed C++ ``for`` bounded by @p max_iter, breaking early
        once ``|r| < tol``).

        The Jacobian is formed in-kernel by finite differences (``J_ij = (r_i(U+eps e_j) - r_i(U))/eps``)
        and the Newton step ``J dU = -r`` is solved with the SAME stack-only dense inverse
        (``adc::detail::mat_inverse<N>``) `solve_local_linear` uses -- so the kernel is heap-free
        / allocation-free / dispatch-free (no ``std::function`` / Eigen / ``std::vector``). The dense
        fallback bound ``n_cons <= 8`` is enforced by the codegen (same as `solve_local_linear`)."""
        if not callable(residual):
            raise ValueError(
                "solve_local_nonlinear: residual must be an IR-building callable "
                "residual_fn(P, U, U0) returning the residual State r(U)")
        if not (isinstance(initial_guess, Value) and initial_guess.vtype == "state"):
            raise ValueError(
                "solve_local_nonlinear: initial_guess must be a State value (initial_guess=...)")
        if method != "newton":
            raise NotImplementedError(
                "solve_local_nonlinear: only method='newton' is supported (got %r)" % (method,))
        if not isinstance(tol, (int, float)) or tol <= 0:
            raise ValueError("solve_local_nonlinear: tol must be a positive number (got %r)" % (tol,))
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError(
                "solve_local_nonlinear: max_iter must be a positive int (got %r)" % (max_iter,))
        if self._recording:
            raise NotImplementedError(
                "solve_local_nonlinear: recording a residual inside another sub-block (apply / while "
                "body) is a later phase")
        block = initial_guess.block
        # Record the residual sub-block (like set_apply / a while body): the iterate U and the frozen
        # initial-guess U0 are State placeholders local to the sub-block; residual_fn builds r(U) from
        # them with LOCAL per-cell ops. The placeholders are NOT appended to self._values (they belong
        # to this op) -- the kernel binds the iterate to the cell stack and U0 to the frozen guess.
        wants_guess = _residual_wants_guess(residual)
        sub = []
        self._recording.append(sub)
        try:
            iterate = self._new("state", "state", (), {}, "newton_iterate", block)
            guess_ph = self._new("state", "state", (), {}, "newton_guess", block)
            # residual_fn(P, U, U0); a two-arg residual_fn(P, U) (ignoring the guess) is also accepted.
            r = residual(self, iterate, guess_ph) if wants_guess else residual(self, iterate)
        finally:
            self._recording.pop()
        if not (isinstance(r, Value) and r.vtype == "state"):
            raise ValueError(
                "solve_local_nonlinear: residual_fn must return the residual State r(U) (got %r)" % (r,))
        for w in sub:
            if w.op not in Program._RESIDUAL_LOCAL_OPS:
                raise ValueError(
                    "solve_local_nonlinear: residual op '%s' is not LOCAL; a per-cell Newton residual "
                    "may use only %s (the iterate / guess State, P.source, P.apply, affine combines). "
                    "Use a non-local op (P.rhs / P.divergence / P.solve_fields) outside the residual."
                    % (w.op, sorted(Program._RESIDUAL_LOCAL_OPS)))
        return self._new(
            "state", "solve_local_nonlinear", (initial_guess,),
            {"residual_block": sub, "residual": r, "iterate": iterate, "guess": guess_ph,
             "tol": float(tol), "max_iter": int(max_iter), "method": method}, name, block)

    def _linear_source_name(self, operator, where):
        """Resolve `operator` (a `linear_source` value, its name, or a single unit-coefficient
        ``_Operator`` term) to the linear-source name."""
        if isinstance(operator, str) and operator:
            return operator
        if isinstance(operator, Value) and operator.op == "linear_source":
            return operator.attrs["linear_source"]
        if (isinstance(operator, _Operator) and not operator.identity.as_dict()
                and len(operator.terms) == 1 and operator.terms[0][1].as_dict() == {0: 1.0}):
            return operator.terms[0][0].attrs["linear_source"]
        raise ValueError(
            "%s: operator must be a linear source (P.linear_source(name) or its name)" % where)

    # --- matrix-free operators / dynamic linear solve (ADC-405 Phase 6b) ----------------------------
    # A ``matrix_free_op`` names a GLOBAL matrix-free operator A : scalar_field -> scalar_field whose
    # apply ``out <- A(in)`` is an IR sub-block recorded by ``set_apply``. ``solve_linear`` lowers to a
    # call into the runtime's Krylov loop (adc::cg_solve / bicgstab_solve / richardson_solve /
    # gmres_solve): the iteration is DYNAMIC and lives C++-side (inside the loop), invisible to the IR --
    # the Program only supplies the apply (a C++ lambda) + the rhs / tolerance / iteration budget.
    _KRYLOV_METHODS = frozenset({"cg", "bicgstab", "richardson", "gmres"})
    _GMRES_RESTART_DEFAULT = 30  # GMRES(m) restart length when the caller does not override it

    def scalar_field(self, name=None, ncomp=1):
        """A fresh, zero-initialized scalar field: scratch the apply sub-block uses (e.g. the Laplacian
        output, or a 2-component gradient buffer). @p ncomp is the component count (1 by default; 2 for a
        gradient field consumed by ``P.divergence``). Lowered to ``ctx.alloc_scalar_field(ncomp, 1)``."""
        if not isinstance(ncomp, int) or ncomp < 1:
            raise ValueError("scalar_field: ncomp must be a positive integer (got %r)" % (ncomp,))
        return self._new("scalar_field", "scalar_field", (), {"ncomp": int(ncomp)}, name, None)

    _OPERATOR_KINDS = frozenset({"scalar", "vector", "state"})

    def matrix_free_operator(self, name, domain="scalar", range_="scalar", ncomp=None):
        """Declare a matrix-free operator ``A : domain -> range_``. @p domain / @p range_ are the field
        kind on each side and MUST match (a square operator: the Krylov iterate, residual and solution
        share one layout): ``"scalar"`` (a 1-component scalar field, the default), or ``"vector"`` /
        ``"state"`` (a multi-component field, e.g. the condensed-Schur block unknown). For a
        ``vector`` / ``state`` operator @p ncomp (an int >= 1) is REQUIRED -- the component count of the
        apply's in/out buffers and of the solution; for a ``scalar`` operator @p ncomp must be omitted
        (or 1). Supply the apply via ``P.set_apply(A, body_fn)`` before using it in ``P.solve_linear``."""
        if domain not in Program._OPERATOR_KINDS or range_ not in Program._OPERATOR_KINDS:
            raise ValueError(
                "matrix_free_operator: domain / range_ must be one of %s; got domain=%r range_=%r"
                % (sorted(Program._OPERATOR_KINDS), domain, range_))
        if domain != range_:
            raise ValueError(
                "matrix_free_operator: domain and range_ must match (a square operator); got "
                "domain=%r range_=%r" % (domain, range_))
        if domain == "scalar":
            if ncomp not in (None, 1):
                raise ValueError(
                    "matrix_free_operator: a scalar operator has ncomp=1 (omit ncomp); got ncomp=%r"
                    % (ncomp,))
            ncomp = 1
        else:  # vector / state: an explicit positive component count is required
            if isinstance(ncomp, bool) or not isinstance(ncomp, int) or ncomp < 1:
                raise ValueError(
                    "matrix_free_operator: a %r operator requires ncomp (an int >= 1); got ncomp=%r"
                    % (domain, ncomp))
        return self._new("matrix_free_op", "matrix_free_operator", (),
                         {"domain": domain, "range": range_, "ncomp": int(ncomp), "apply_block": None,
                          "apply_result": None, "apply_in": None, "apply_out": None}, name, None)

    def set_apply(self, operator, body_fn):
        """Record the apply ``out <- A(in)`` of a ``matrix_free_operator``. @p body_fn(P, out, in) is an
        IR-building callable: @p in and @p out are scalar_field values (the operator's argument and
        result); the body builds @p out from @p in (e.g. ``P.laplacian(tmp, in); ...``) using
        ``P.laplacian`` + the affine algebra and RETURNS the result scalar_field (the value written into
        @p out). The ops are captured into a separate sub-block (like a while body) and re-emitted as a
        C++ lambda the Krylov loop calls."""
        if not (isinstance(operator, Value) and operator.vtype == "matrix_free_op"):
            raise ValueError("set_apply: operator must be a matrix_free_operator value")
        if operator.attrs["apply_block"] is not None:
            raise ValueError("set_apply: operator '%s' already has an apply" % operator.name)
        if self._recording:
            raise NotImplementedError(
                "set_apply: recording an apply inside another sub-block (apply / while body) is a "
                "later phase")
        # The apply ops (the in/out placeholders + the body) live in the operator's OWN sub-block, NOT
        # the flat SSA list: they are re-emitted as the C++ apply lambda, never walked at the top level.
        sub = []
        self._recording.append(sub)
        # The in/out buffers carry the operator's component count: a vector / state operator applies on
        # an ncomp buffer (scalar -> ncomp == 1). The apply body sees ncomp-component in / out fields.
        op_ncomp = int(operator.attrs["ncomp"])
        try:
            out_sf = self._new("scalar_field", "apply_out", (), {"ncomp": op_ncomp}, "apply_out", None)
            in_sf = self._new("scalar_field", "apply_in", (), {"ncomp": op_ncomp}, "apply_in", None)
            result = body_fn(self, out_sf, in_sf)
        finally:
            self._recording.pop()
        block = sub
        result = result if result is not None else out_sf
        if not (isinstance(result, (Value, _Affine)) or _is_field_value(result)):
            raise ValueError("set_apply: body_fn must return the result scalar_field (out <- A(in))")
        operator.attrs["apply_block"] = block
        operator.attrs["apply_result"] = result
        operator.attrs["apply_in"] = in_sf
        operator.attrs["apply_out"] = out_sf
        return operator

    def laplacian(self, out, in_):
        """Record ``out = Lap(in_)`` (the shared discrete 5-point Laplacian). @p out and @p in_ are
        scalar_field values. Lowered to ``ctx.laplacian(out, in_)``. Used inside an apply sub-block to
        form a Helmholtz operator ``A(in) = in - alpha*Lap(in)`` via the affine algebra."""
        if not (isinstance(out, Value) and out.vtype == "scalar_field"):
            raise ValueError("laplacian: out must be a scalar_field value")
        if not (isinstance(in_, Value) and in_.vtype == "scalar_field"):
            raise ValueError("laplacian: in must be a scalar_field value")
        return self._new("scalar_field", "laplacian", (out, in_), {}, out.name, None)

    def gradient(self, out, phi):
        """Record ``out = grad(phi)`` (centered differences; @p out has >= 2 components). @p out and
        @p phi are scalar_field values. Lowered to ``ctx.gradient(out, phi)``."""
        if not (isinstance(out, Value) and out.vtype == "scalar_field"):
            raise ValueError("gradient: out must be a scalar_field value")
        if not (isinstance(phi, Value) and phi.vtype == "scalar_field"):
            raise ValueError("gradient: phi must be a scalar_field value")
        return self._new("scalar_field", "gradient", (out, phi), {}, out.name, None)

    def divergence(self, out, fx, fy):
        """Record ``out = div(fx, fy)`` (centered FV divergence d fx/dx + d fy/dy, component 0). @p out,
        @p fx and @p fy are scalar_field values. Lowered to ``ctx.divergence(out, fx, fy)``. The exact
        inverse of @ref gradient: chaining ``P.gradient(g, phi); P.divergence(d, gx, gy)`` recovers the
        5-point Laplacian, so a matrix-free apply ``phi - alpha*div(grad phi)`` is the Schur-like flux
        operator ``phi - alpha*Lap(phi)``."""
        for nm, val in (("out", out), ("fx", fx), ("fy", fy)):
            if not (isinstance(val, Value) and val.vtype == "scalar_field"):
                raise ValueError("divergence: %s must be a scalar_field value" % nm)
        return self._new("scalar_field", "divergence", (out, fx, fy), {}, out.name, None)

    # --- finite-difference Jacobian-vector product (ADC-431: implicit-flux BDF Newton-Krylov) --------
    def rhs_jacvec(self, out, in_, *, iterate, r0, c_dt, eps=1e-7, flux=True, sources=("default",)):
        """Record the finite-difference Jacobian-vector product of an implicit-flux residual, INSIDE a
        matrix_free_operator apply sub-block (ADC-431). It lowers to ``out <- J(@p iterate) @p in`` where
        the Newton-system Jacobian is ``J = I - c*dt * d(rhs)/dU`` and the matvec is formed matrix-free by
        a directional finite difference::

            out = in - (c*dt/eps) * (rhs(U^k + eps*in) - rhs(U^k))

        @p out / @p in_ are the apply sub-block's out / in scalar_field buffers (carrying the operator's
        component count). @p iterate is the FROZEN Newton iterate ``U^k`` (a State, defined OUTSIDE the
        apply, captured into the apply lambda); @p r0 is the precomputed ``rhs(U^k)`` (a State/RHS value,
        also captured) so the perturbation cost is one ``rhs`` per matvec. @p c_dt is the BDF coefficient
        ``c*dt`` (a number or a dt-polynomial: ``c == 1`` for BDF1, ``c == 2/3`` for BDF2). @p eps is the
        relative FD step (scaled by ``||U^k|| / ||in||`` inside the kernel). @p flux / @p sources select
        the same residual the outer ``rhs`` uses (so the linearized operator is consistent with the
        residual). The op may ONLY appear inside ``set_apply`` (it captures the apply's in/out buffers).

        Unlike the cell-local FD Jacobian of `solve_local_nonlinear` (a per-cell dense inverse), this is a
        GLOBAL operator: ``rhs`` couples the cells through the flux stencil, so the matvec is dense over
        the coupled stencil and the Newton step ``J dU = -F`` is solved by `solve_linear` (GMRES)."""
        if not self._recording:
            raise ValueError("rhs_jacvec may only be recorded inside a matrix_free_operator apply "
                             "(call it from the set_apply body_fn)")
        if not (isinstance(out, Value) and out.vtype == "scalar_field"):
            raise ValueError("rhs_jacvec: out must be the apply sub-block's out scalar_field value")
        if not (isinstance(in_, Value) and in_.vtype == "scalar_field"):
            raise ValueError("rhs_jacvec: in_ must be the apply sub-block's in scalar_field value")
        if not (isinstance(iterate, Value) and iterate.vtype == "state"):
            raise ValueError("rhs_jacvec: iterate must be the frozen Newton-iterate State (iterate=...)")
        if not (isinstance(r0, Value) and r0.is_field()):
            raise ValueError("rhs_jacvec: r0 must be the precomputed rhs(U^k) State/RHS value (r0=...)")
        if not isinstance(c_dt, (int, float, _Coeff)):
            raise ValueError("rhs_jacvec: c_dt must be a number or a dt-polynomial (got %r)" % (c_dt,))
        if not isinstance(eps, (int, float)) or eps <= 0:
            raise ValueError("rhs_jacvec: eps must be a positive number (got %r)" % (eps,))
        c_d = (c_dt if isinstance(c_dt, _Coeff) else _Coeff({0: float(c_dt)})).as_dict()
        src = list(sources) if sources is not None else None
        return self._new("scalar_field", "rhs_jacvec", (out, in_, iterate, r0),
                         {"c_dt": c_d, "eps": float(eps), "flux": bool(flux), "sources": src},
                         out.name, None)

    # --- anisotropic condensed-Schur coefficient assembly + coefficiented apply (ADC-399 / ADC-421) ---
    def schur_coeffs(self, name=None, state=None, c=None, th_dt=None, c_rho=0, c_bz=3):
        """Assemble the per-cell tensor coefficient ``A = I + c*rho*B^{-1}`` of the condensed-Schur
        operator from a State (rho at component @p c_rho) and the B_z aux field (component @p c_bz,
        canonical B_z=3). Returns a ``schur_coeffs`` bundle value carrying the four coefficient fields
        (eps_x, eps_y, a_xy, a_yx) -- pass it to ``P.apply_laplacian_coeff`` inside a matrix-free apply.

        @p c = theta^2 * dt^2 * alpha and @p th_dt = theta*dt are scalar coefficients (numbers or
        dt-polynomials via the affine ``P.dt`` algebra; ``B^{-1}`` depends only on ``w = th_dt*B_z``).
        The assembly runs ONCE per step (rho / B_z frozen in the source) and the bundle is reused across
        every Krylov iteration of the phi solve. Lowered to ``ctx.assemble_schur_coeffs`` -- the SAME
        native detail::SchurOperatorCoeffKernel + apply_laplacian coefficient path, no reimplementation.
        """
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("schur_coeffs: a State value is required (state=...)")
        for nm, sc in (("c", c), ("th_dt", th_dt)):
            if not isinstance(sc, (int, float, _Coeff)):
                raise ValueError("schur_coeffs: %s must be a number or a dt-polynomial (got %r)"
                                 % (nm, sc))
        for nm, ci in (("c_rho", c_rho), ("c_bz", c_bz)):
            if isinstance(ci, bool) or not isinstance(ci, int) or ci < 0:
                raise ValueError("schur_coeffs: %s must be a Python int >= 0 (got %r)" % (nm, ci))
        c_d = (c if isinstance(c, _Coeff) else _Coeff({0: float(c)})).as_dict()
        th_d = (th_dt if isinstance(th_dt, _Coeff) else _Coeff({0: float(th_dt)})).as_dict()
        return self._new("schur_coeffs", "schur_coeffs", (state,),
                         {"c": c_d, "th_dt": th_d, "c_rho": int(c_rho), "c_bz": int(c_bz)}, name,
                         state.block)

    def apply_laplacian_coeff(self, out, in_, coeffs):
        """Record ``out = div(A grad in_)`` with the tensor ``A`` of a @ref schur_coeffs bundle (the
        coefficiented matrix-free matvec of the condensed-Schur operator, ``adc::apply_laplacian``'s
        coefficient path). @p out and @p in_ are scalar_field values; @p coeffs is a ``schur_coeffs``
        value. Used inside a matrix-free apply: the condensed operator ``L_schur(phi) = -div(A grad
        phi) = -out``, so build it as ``-1 * P.apply_laplacian_coeff(out, in_, A)`` via the affine
        algebra. Lowered to ``ctx.apply_laplacian_coeff(out, in_, eps_x, eps_y, a_xy, a_yx)``."""
        if not (isinstance(out, Value) and out.vtype == "scalar_field"):
            raise ValueError("apply_laplacian_coeff: out must be a scalar_field value")
        if not (isinstance(in_, Value) and in_.vtype == "scalar_field"):
            raise ValueError("apply_laplacian_coeff: in_ must be a scalar_field value")
        if not (isinstance(coeffs, Value) and coeffs.vtype == "schur_coeffs"):
            raise ValueError("apply_laplacian_coeff: coeffs must be a schur_coeffs bundle "
                             "(P.schur_coeffs(...))")
        return self._new("scalar_field", "apply_laplacian_coeff", (out, in_, coeffs), {}, out.name,
                         None)

    def schur_explicit_flux(self, out, state, th_dt, c_mx=1, c_my=2, c_bz=3):
        """Record ``out = B^{-1} (mx, my)`` per cell -- the explicit condensed-Schur flux
        ``F = rho*B^{-1}*v^n`` (Fx in component 0, Fy in component 1). @p out is a scalar_field (>= 2
        components), @p state a State (mx / my at @p c_mx / @p c_my), B_z the aux field at @p c_bz.
        @p th_dt = theta*dt. Chain ``P.divergence(d, out, out)`` for the centered divergence of F.
        Lowered to ``ctx.schur_explicit_flux`` (native detail::SchurExplicitFluxKernel)."""
        if not (isinstance(out, Value) and out.vtype == "scalar_field"):
            raise ValueError("schur_explicit_flux: out must be a scalar_field value (ncomp >= 2)")
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("schur_explicit_flux: a State value is required")
        th_d = (th_dt if isinstance(th_dt, _Coeff) else _Coeff({0: float(th_dt)})).as_dict()
        return self._new("scalar_field", "schur_explicit_flux", (out, state),
                         {"th_dt": th_d, "c_mx": int(c_mx), "c_my": int(c_my), "c_bz": int(c_bz)},
                         out.name, None)

    def schur_rhs(self, out, phi_n, state, th_dt, g, c_mx=1, c_my=2, c_bz=3):
        """Record the FUSED condensed-Schur right-hand side ``out = -Lap(phi_n) - g*div(F)`` with
        ``F = B^{-1}(mx, my)`` -- the native ElectrostaticLorentzCondensation::assemble_rhs in one op.
        @p out is a 1-component scalar_field, @p phi_n a scalar_field (phi^n warm start; its ghosts are
        filled for the Laplacian), @p state a State (mx / my at @p c_mx / @p c_my). @p th_dt = theta*dt
        and @p g = theta*dt*alpha are scalar coefficients (numbers or dt-polynomials). Lowered to
        ``ctx.assemble_schur_rhs``. A single op because there is no scalar-field affine combine at the
        IR level -- the fused C++ assembler mirrors the native one (bare Lap + explicit flux + the
        SchurRhsAssemble divergence)."""
        if not (isinstance(out, Value) and out.vtype == "scalar_field"):
            raise ValueError("schur_rhs: out must be a scalar_field value")
        if not (isinstance(phi_n, Value) and phi_n.vtype == "scalar_field"):
            raise ValueError("schur_rhs: phi_n must be a scalar_field value")
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("schur_rhs: a State value is required (state=...)")
        for nm, sc in (("th_dt", th_dt), ("g", g)):
            if not isinstance(sc, (int, float, _Coeff)):
                raise ValueError("schur_rhs: %s must be a number or a dt-polynomial (got %r)"
                                 % (nm, sc))
        th_d = (th_dt if isinstance(th_dt, _Coeff) else _Coeff({0: float(th_dt)})).as_dict()
        g_d = (g if isinstance(g, _Coeff) else _Coeff({0: float(g)})).as_dict()
        return self._new("scalar_field", "schur_rhs", (out, phi_n, state),
                         {"th_dt": th_d, "g": g_d, "c_mx": int(c_mx), "c_my": int(c_my),
                          "c_bz": int(c_bz)}, out.name, None)

    def schur_reconstruct(self, name=None, state=None, phi=None, th_dt=None, c_rho=0, c_mx=1, c_my=2,
                          c_bz=3):
        """Record the condensed-Schur velocity reconstruction ``v^{n+theta} = B^{-1}(v^n - theta*dt*
        grad phi)`` IN PLACE on @p state (rho frozen; mom = rho*v written back). @p phi is the solved
        potential (a scalar_field or 1-component State), @p th_dt = theta*dt; B_z the aux at @p c_bz.
        Returns the updated State. Lowered to ``ctx.schur_reconstruct`` (the native centered gradient +
        closed B^{-1}). The final n+1 extrapolation (factor 1/theta) is the caller's affine algebra."""
        if isinstance(name, Value) and state is None:
            name, state = None, name
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("schur_reconstruct: a State value is required (state=...)")
        if not _is_field_value(phi):
            raise ValueError("schur_reconstruct: phi must be a scalar_field or State value (phi=...)")
        if not isinstance(th_dt, (int, float, _Coeff)):
            raise ValueError("schur_reconstruct: th_dt must be a number or a dt-polynomial (got %r)"
                             % (th_dt,))
        th_d = (th_dt if isinstance(th_dt, _Coeff) else _Coeff({0: float(th_dt)})).as_dict()
        return self._new("state", "schur_reconstruct", (state, phi),
                         {"th_dt": th_d, "c_rho": int(c_rho), "c_mx": int(c_mx), "c_my": int(c_my),
                          "c_bz": int(c_bz)}, name, state.block)

    def schur_energy(self, name=None, state=None, state_old=None, c_rho=0, c_mx=1, c_my=2, c_E=3):
        """Record the condensed-Schur kinetic-energy increment IN PLACE on @p state (ADC-427):
        ``E^{n+1} = E^n + (1/2)*rho*(|v^{n+1}|^2 - |v^n|^2)``, ``v = (mx, my)/rho`` (the native
        SchurEnergyKernel). @p state carries ``rho`` / ``mx`` / ``my`` / ``E`` at @p c_rho / @p c_mx /
        @p c_my / @p c_E AFTER the velocity update (mom = rho*v^{n+1}); @p state_old is U^n (read for
        v^n = mom^n/rho^n and the base energy E^n). rho is frozen, so the same rho is read from both.
        Returns @p state (E overwritten in place). Lowered to ``ctx.schur_energy``."""
        if isinstance(name, Value) and state is None:
            name, state = None, name
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("schur_energy: a State value is required (state=...)")
        if not (isinstance(state_old, Value) and state_old.vtype == "state"):
            raise ValueError("schur_energy: a State value is required (state_old=U^n)")
        if state_old.block != state.block:
            raise ValueError("schur_energy: state and state_old must belong to the same block")
        for nm, ci in (("c_rho", c_rho), ("c_mx", c_mx), ("c_my", c_my), ("c_E", c_E)):
            if isinstance(ci, bool) or not isinstance(ci, int) or ci < 0:
                raise ValueError("schur_energy: %s must be a Python int >= 0 (got %r)" % (nm, ci))
        return self._new("state", "schur_energy", (state, state_old),
                         {"c_rho": int(c_rho), "c_mx": int(c_mx), "c_my": int(c_my), "c_E": int(c_E)},
                         name, state.block)

    def solve_linear(self, name=None, operator=None, rhs=None, initial_guess=None, method="cg",
                     preconditioner="identity", tol=1e-8, max_iter=None, restart=None):
        """Solve the matrix-free linear system ``operator x = rhs`` with the runtime's Krylov loop and
        return the solution as a scalar_field. The iteration is DYNAMIC (C++-side, inside the loop):
        the IR only carries the operator (its apply lambda), the rhs, the initial guess, and the
        method / tolerance / iteration budget.

          - @p operator: a ``matrix_free_operator`` value (with a ``set_apply`` body);
          - @p rhs: the right-hand side -- a scalar_field, or (MVP) a 1-component State value;
          - @p initial_guess: warm start (defaults to zero);
          - @p method: ``"cg"`` (SPD), ``"bicgstab"`` (general), ``"richardson"``, or ``"gmres"``
            (restarted GMRES(m), the robust choice for a NON-symmetric operator);
          - @p preconditioner: ``"identity"`` only for now;
          - @p tol: relative L2 residual stop (> 0);
          - @p max_iter: iteration budget (REQUIRED, > 0: a dynamic solver loop with no budget is a
            configuration error -- ``adc::*_solve`` itself throws on a non-positive budget);
          - @p restart: GMRES restart length m (a positive int; defaults to 30). Ignored by the other
            methods; passing it to a non-gmres solve is rejected."""
        if not (isinstance(operator, Value) and operator.vtype == "matrix_free_op"):
            raise ValueError("solve_linear: operator must be a matrix_free_operator value")
        if operator.attrs["apply_block"] is None:
            raise ValueError("solve_linear: operator '%s' has no apply; call P.set_apply first"
                             % operator.name)
        if not _is_field_value(rhs):
            raise ValueError("solve_linear: rhs must be a scalar_field or State value (rhs=...)")
        if initial_guess is not None and not _is_field_value(initial_guess):
            raise ValueError("solve_linear: initial_guess must be a scalar_field or State value")
        op_ncomp = int(operator.attrs["ncomp"])
        # The rhs / initial guess must carry at least the operator's component count: the solve runs on
        # an op_ncomp buffer. A scalar_field exposes its ncomp here; a State's n_cons is only known at
        # compile (against the model), so a State is accepted now and checked there.
        for label, fld in (("rhs", rhs), ("initial_guess", initial_guess)):
            if fld is None or fld.vtype != "scalar_field":
                continue
            fld_ncomp = int(fld.attrs.get("ncomp", 1))
            if fld_ncomp < op_ncomp:
                raise ValueError(
                    "solve_linear: %s has %d component(s) but the operator needs %d (a scalar_field "
                    "with ncomp >= the operator ncomp, or a State)" % (label, fld_ncomp, op_ncomp))
        if method not in Program._KRYLOV_METHODS:
            raise ValueError("solve_linear: method must be one of %s; got %r"
                             % (sorted(Program._KRYLOV_METHODS), method))
        if preconditioner != "identity":
            raise NotImplementedError(
                "solve_linear: only preconditioner='identity' is supported yet (got %r)"
                % preconditioner)
        if not isinstance(tol, (int, float)) or tol <= 0:
            raise ValueError("solve_linear: tol must be a positive number (got %r)" % (tol,))
        if max_iter is None or not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("dynamic solver loops require max_iter")
        # restart is a gmres-only knob; the GMRES(m) basis size. Other methods have no restart concept,
        # so passing one to them is a config error (fail loud rather than silently ignore it).
        if method == "gmres":
            if restart is None:
                restart = Program._GMRES_RESTART_DEFAULT
            elif isinstance(restart, bool) or not isinstance(restart, int) or restart <= 0:
                raise ValueError("solve_linear: restart must be a positive integer for gmres (got %r)"
                                 % (restart,))
        elif restart is not None:
            raise ValueError("solve_linear: restart only applies to method='gmres' (got method=%r)"
                             % (method,))
        inputs = (operator, rhs) if initial_guess is None else (operator, rhs, initial_guess)
        return self._new("scalar_field", "solve_linear", inputs,
                         {"method": method, "preconditioner": preconditioner, "tol": float(tol),
                          "max_iter": int(max_iter), "has_guess": initial_guess is not None,
                          "ncomp": op_ncomp,
                          "restart": int(restart) if method == "gmres" else None}, name, rhs.block)

    # --- multistep histories (ADC-406a) ---
    def history(self, name, lag=1):
        """Read a SYSTEM-OWNED history field carried across macro-steps: the value stored @p lag steps
        back (e.g. ``P.history("plasma.R", lag=1)`` is R_{n-1} for Adams-Bashforth). Returns a
        State-typed value usable in the affine algebra. The history is owned by the System (a
        HistoryManager), not the Program, so a later checkpoint slice can serialize it; reading it
        before it has ever been stored is a fail-loud runtime error (it must be written by
        `store_history` every step). @p lag must be a Python int >= 1."""
        if not isinstance(name, str) or not name:
            raise ValueError("history: name must be a non-empty string")
        if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
            raise ValueError("history: lag must be a Python int >= 1 (got %r)" % (lag,))
        self._histories[name] = max(self._histories.get(name, 0), lag)
        return self._new("state", "history", (), {"history": name, "lag": int(lag)}, name, None)

    def store_history(self, name, value):
        """Store @p value (a State/RHS field) into the CURRENT slot of history @p name at the end of the
        step (rotated to lag 1 on the next step). A multistep scheme stores its current RHS so the next
        step can read it back via `history`. The history is System-owned; this is a side-effecting op
        (no value). @p value must be a State/RHS field of the Program."""
        if not isinstance(name, str) or not name:
            raise ValueError("store_history: name must be a non-empty string")
        if not _is_field_value(value):
            raise ValueError("store_history: value must be a State/RHS field (got %r)" % (value,))
        if value.prog is not self:
            raise ValueError("store_history: the value belongs to a different Program")
        self._histories.setdefault(name, 1)
        return self._new("state", "store_history", (value,), {"history": name}, name, value.block)

    def commit(self, block, state):
        """Replace the current state of ``block`` with ``state`` at the end of the step. Each block
        is committed AT MOST once; read-only blocks need no commit.

        @p state is normally a State value; a 1-component model's conservative state doubles as a
        scalar field, so a ``scalar_field`` (e.g. a ``solve_linear`` solution) is also accepted and
        copied back into the block state at commit (the final ``ctx.lincomb`` in the lowered body)."""
        if not (isinstance(state, Value) and state.vtype in ("state", "scalar_field")):
            raise ValueError("commit: a State (or scalar_field) value is required")
        if state.prog is not self:
            raise ValueError("commit: the State value belongs to a different Program")
        if block in self._commits:
            raise ValueError("block '%s' committed more than once" % block)
        self._commits[block] = state

    def commits(self):
        """Map of committed block -> committed State value (copy)."""
        return dict(self._commits)

    # --- board-like sugar (Spec 3): T.define / T.fields / T.solve / T.commit_many ---
    # These lower to the SAME primitive ops as the P.call / linear_combine /
    # solve_local_linear / commit style; they are blackboard notation, not a new IR.
    def op(self, name):
        """Return a callable board handle for a bound operator: ``expl = P.op("explicit_rate")``
        then ``expl(U, fields)`` builds ``P.call("explicit_rate", U, fields)``."""
        def _handle(*args, value_name=None):
            return self.call(name, *args, name=value_name)
        _handle.__name__ = str(name)
        return _handle

    def fields(self, name, from_state=None, from_states=None, from_state_set=None, operator=None):
        """Board sugar for a field solve. Lowers to ``P.call(operator, ...)`` when a named operator
        is bound, else to ``P.solve_fields`` (single state) or ``P.solve_fields_from_blocks``."""
        if from_state_set is not None:
            states = from_state_set.states()
        elif from_states is not None:
            states = list(from_states)
        elif from_state is not None:
            states = [from_state]
        else:
            raise ValueError("fields: provide from_state=, from_states= or from_state_set=")
        named = operator is not None and operator != "fields_from_state"
        if len(states) == 1:
            if named and self._registry is not None:
                return self.call(operator, states[0], name=name)
            return self.solve_fields(name, states[0])
        if named and self._registry is not None:
            return self.call(operator, *states, name=name)
        return self.solve_fields_from_blocks(states, name=name)

    def define(self, name, value):
        """Board sugar to name a value. An affine combination of states materializes via
        ``linear_combine``; a ``rate(U) == <expr>`` equation keeps its right-hand side; any other
        Value is named in place."""
        from . import math as _bm
        if isinstance(value, _bm.Equation):
            if not isinstance(value.lhs, _bm.TimeDerivative):
                raise ValueError("define(%r): an equation must read 'rate(U) == <rate expression>'"
                                 % (name,))
            value = value.rhs
        if isinstance(value, _Affine):
            return self.linear_combine(name, value)
        if isinstance(value, Value):
            value.name = name
            return value
        raise TypeError(
            "define(%r): expected a Value, an affine combination, or a rate equation; got %r"
            % (name, value))

    def solve(self, name, equation):
        """Board sugar for an implicit local solve ``(I -/+ a*L) @ unknown("x") == rhs``.

        Lowers to ``linear_combine`` (if the rhs is an affine combination) then
        ``solve_local_linear``; identical IR to writing those two calls by hand.
        """
        from . import math as _bm
        if not isinstance(equation, _bm.Equation):
            raise TypeError("solve(%r): expected '(I - dt*C) @ unknown(\"x\") == rhs'" % (name,))
        lhs, rhs = equation.lhs, equation.rhs
        if not isinstance(lhs, _bm.OpApply):
            raise ValueError("solve(%r): left-hand side must be 'operator @ unknown(name)'" % (name,))
        if isinstance(rhs, _Affine):
            rhs = self.linear_combine(name + "_rhs", rhs)
        elif not (isinstance(rhs, Value) and rhs.vtype == "state"):
            raise ValueError("solve(%r): right-hand side must be a State or an affine of States"
                             % (name,))
        return self.solve_local_linear(name=name, operator=lhs.operator, rhs=rhs)

    def commit_many(self, mapping, fields=None):
        """Atomically commit several coupled blocks (Spec 3). ALL entries are validated before any
        commit, so a partial or double commit of a coupled group is rejected as a unit and no block
        is left half-committed. ``fields`` (optional) is validated as a coherent FieldContext but is
        RESERVED: the IR commit has no fields slot yet (the runtime association lands with ADC-457)."""
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("commit_many: a non-empty {block: State} mapping is required")
        if fields is not None and not (isinstance(fields, Value) and fields.vtype == "fields"):
            raise ValueError("commit_many: fields must be a FieldContext from solve_fields")
        for block, state in mapping.items():
            if not (isinstance(state, Value) and state.vtype in ("state", "scalar_field")):
                raise ValueError("commit_many: block %r needs a State value" % (block,))
            if state.prog is not self:
                raise ValueError("commit_many: the State for %r belongs to a different Program"
                                 % (block,))
            if block in self._commits:
                raise ValueError("block '%s' committed more than once" % (block,))
        for block, state in mapping.items():
            self._commits[block] = state

    def state_set(self, name, mapping):
        """Build a :class:`StageStateSet` -- a coherent set of stage states for a field solve."""
        return StageStateSet(name, mapping)

    def record(self, name, value):
        """Record a scalar diagnostic (board sugar over :meth:`record_scalar`).

        ``value`` is a Program scalar -- a reduction result such as ``P.sum(U)`` or
        ``P.norm2(U)`` (the runtime value of a generic invariant). The automatic
        reduction of an arbitrary ``integral(expr)`` over a per-cell expression is a
        follow-up (it needs the scheduler / a generated reduction kernel, ADC-458)."""
        if not (isinstance(value, Value) and value.vtype == "scalar"):
            raise ValueError(
                "record(%r): value must be a Program scalar (e.g. P.sum / P.norm2); got %r"
                % (name, value))
        return self.record_scalar(name, value)

    def check_invariant(self, name, before=None, after=None, tolerance=1e-10):
        """Record the drift of a generic invariant between two stages (board diagnostic).

        ``before`` / ``after`` are Program scalars (reduction results); the recorded
        diagnostic ``"<name>_drift"`` is ``after - before``. ``tolerance`` is carried as
        metadata for a later assertion stage (the scheduled runtime check is ADC-458)."""
        if not (isinstance(before, Value) and before.vtype == "scalar"
                and isinstance(after, Value) and after.vtype == "scalar"):
            raise ValueError(
                "check_invariant(%r): before/after must be Program scalars" % (name,))
        drift = after - before
        out = self.record_scalar(name + "_drift", drift)
        out.attrs["tolerance"] = float(tolerance)
        return out

    # --- inspection / debug (Spec 3 section 33): show the lowering ---
    @staticmethod
    def _render_node(v):
        """Render one IR value as an operator-first line (introspection, not codegen)."""
        ins = ", ".join(i.name for i in v.inputs)
        extra = ""
        keys = {k: val for k, val in v.attrs.items() if k != "coeffs"}
        if "coeffs" in v.attrs:
            extra = "  # coeffs=%s" % (v.attrs["coeffs"],)
        elif keys:
            extra = "  # %s" % (keys,)
        return "%-16s = P.%s(%s)%s" % (v.name, v.op, ins, extra)

    def dump_operator_ir(self):
        """The operator-first Program IR (one line per node): P.call/linear_combine/
        solve_local_linear/commit. The board sugar lowers to exactly this -- the dump
        proves the board and the operator-first writings share one IR."""
        lines = ["# operator-first Program IR: %s" % self.name]
        for v in self._values:
            lines.append("  " + self._render_node(v))
        for block, st in self._commits.items():
            lines.append("  P.commit(%r, %s)" % (block, st.name))
        return "\n".join(lines)

    def dump_board(self):
        """The board-level view; board notation lowers to the operator-first IR below."""
        return ("# board program %s lowers to the operator-first IR (board == operator-first):\n%s"
                % (self.name, self.dump_operator_ir()))

    def dump_cpp_plan(self):
        """A textual C++ plan of the generated step (ProgramContext calls), NOT the exact
        codegen -- it shows which ctx / GeneratedModule call each node lowers to."""
        lines = ["// C++ plan for GeneratedProgram step of %s" % self.name]
        for v in self._values:
            ins = ", ".join(i.name for i in v.inputs)
            if v.op == "coupled_rate":
                # the coupled-rate kernel lowers to ONE multi-state for_each_cell filling every block's
                # rate scratch at once (ADC-457); there is no single ctx.coupled_rate(...) call.
                blks = ", ".join(v.attrs.get("blocks", []))
                lines.append("  // %s: multi-state for_each_cell rate kernel over (%s) for blocks "
                             "[%s];  // ADC-457" % (v.name, ins, blks))
            elif v.op == "coupled_rate_out":
                # a pure projection: it aliases its block's rate scratch (no code of its own).
                lines.append("  // %s = %s.rate[%r];  // ADC-457 (block projection, no ctx call)"
                             % (v.name, ins, v.attrs.get("out_block")))
            else:
                lines.append("  ctx.%s(%s);  // -> %s" % (v.op, ins, v.name))
        for block, st in self._commits.items():
            lines.append("  ctx.commit(%r, %s);" % (block, st.name))
        return "\n".join(lines)

    # --- decorator mode (ADC-423): record the step body from a function ---
    def step(self, fn):
        """Record this Program's IR by calling @p fn(self) ONCE, at build time (decorator mode).

        ``@P.step`` is sugar for an inline builder body: the decorated function receives the Program
        and builds the IR exactly as if its statements had been written at module scope. It is a
        BUILD-TIME callback -- it runs once, here, to populate the SSA value list; it is NEVER executed
        numerically during ``sim.step`` (the compiled ``.so`` owns the runtime step). So

            P = adc.time.Program("fe")

            @P.step
            def _(P):
                adc.time.std.forward_euler(P, "plasma")

        produces byte-identical IR (same ``_ir_hash``) to calling ``std.forward_euler(P, "plasma")``
        inline. Returns the Program so a one-liner ``P = adc.time.Program("p").step(build)`` also reads
        cleanly. @p fn must be callable; it is invoked with the Program as its single argument."""
        if not callable(fn):
            raise TypeError("Program.step expects a callable build_fn(P); got %r" % (fn,))
        fn(self)
        return self

    # --- ghost fill / positivity projection (spec ops 22 / 21) ---
    def fill_boundary(self, x):
        """Fill the ghost cells (halos) of a State/scalar_field @p x in place: the transport BC
        (periodic by default), the same exchange `laplacian` / `gradient` / `divergence` do internally
        before differencing (spec op 22). Returns @p x (the SAME value -- a side effect on its ghosts,
        the valid cells are untouched). Lowered to ``ctx.fill_boundary(x)``."""
        if not _is_field_value(x):
            raise ValueError("fill_boundary: a State/RHS/scalar_field value is required (got %r)"
                             % (x,))
        return self._new(x.vtype, "fill_boundary", (x,), {}, x.name, x.block)

    def project(self, name=None, state=None, projection="block"):
        """Apply the block's post-step positivity projection to @p state in place (spec op 21):
        ``U <- project(U, aux)`` over the valid cells, the SAME Zhang-Shu / floor projection the native
        per-step path runs (ADC-177). Returns a State value (the projected state). @p projection selects
        the projection primitive; only ``"block"`` (the block's own, set at add_block time) is supported
        -- a custom projection is a later phase. Lowered to ``ctx.apply_projection(idx, state)`` for the
        state's own block (ADC-426)."""
        if isinstance(name, Value) and state is None:
            name, state = None, name
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("project: a State value is required (state=...)")
        if projection != "block":
            raise NotImplementedError(
                "project: only projection='block' (the block's own positivity projection) is "
                "supported; a custom projection is a later phase (got %r)" % (projection,))
        return self._new("state", "project", (state,), {"projection": projection}, name,
                         state.block)

    # --- per-cell conditional select (spec op 17, ADC-418) ---
    _CELL_CMPS = {">": "cell_gt", ">=": "cell_ge", "<": "cell_lt", "<=": "cell_le"}

    def cell_compare(self, field, value, cmp, name=None):
        """A PER-CELL comparison ``field <cmp> value`` -> a fresh 1-component 0/1 mask scalar_field (1.0
        where the comparison holds, 0.0 otherwise), evaluated cell by cell on component 0 of @p field
        (its sole / first conserved component). @p field is a State/RHS/scalar_field value; @p value is a
        Python float threshold (a per-cell field threshold is a later phase); @p cmp is one of
        ``'>' '>=' '<' '<='``. The mask is the input the per-cell `where` selects on. Lowered to a
        for_each_cell kernel ``maskA(i,j,0) = fieldA(i,j,0) <cmp> value ? 1 : 0``. Convenience wrappers:
        `cell_gt` / `cell_ge` / `cell_lt` / `cell_le`."""
        if not _is_field_value(field):
            raise ValueError("cell_compare: a State/RHS/scalar_field value is required (got %r)"
                             % (field,))
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("cell_compare: value must be a Python float threshold (a per-cell field "
                            "threshold is a later phase); got %r" % (value,))
        if cmp not in Program._CELL_CMPS:
            raise ValueError("cell_compare: cmp must be one of %s; got %r"
                             % (sorted(Program._CELL_CMPS), cmp))
        return self._new("scalar_field", "cell_compare", (field,),
                         {"cmp": cmp, "value": float(value)}, name, field.block)

    def cell_gt(self, field, value, name=None):
        """Per-cell ``field > value`` mask (1.0 / 0.0). See `cell_compare`."""
        return self.cell_compare(field, value, ">", name)

    def cell_ge(self, field, value, name=None):
        """Per-cell ``field >= value`` mask (1.0 / 0.0). See `cell_compare`."""
        return self.cell_compare(field, value, ">=", name)

    def cell_lt(self, field, value, name=None):
        """Per-cell ``field < value`` mask (1.0 / 0.0). See `cell_compare`."""
        return self.cell_compare(field, value, "<", name)

    def cell_le(self, field, value, name=None):
        """Per-cell ``field <= value`` mask (1.0 / 0.0). See `cell_compare`."""
        return self.cell_compare(field, value, "<=", name)

    def where(self, mask, a, b, name=None):
        """A PER-CELL conditional select (spec op 17): ``out(i,j,c) = mask(i,j,*) != 0 ? a(i,j,c) :
        b(i,j,c)`` COMPONENT-WISE over the field. This is NOT the scalar runtime branch `if_` -- the
        condition is decided per cell INSIDE a Kokkos kernel.

          - @p mask: a 0/1 (or any nonzero/zero) mask field. Either 1-component (one mask shared by all
            components -- read at component 0) or with the SAME ncomp as @p a / @p b (a per-component
            mask). Built with `cell_ge` / `cell_gt` / ... (a threshold) or any scalar_field.
          - @p a, @p b: the two field values to choose between, on the SAME grid and ncomp (a State or a
            scalar_field). The result has @p a's vtype / block / ncomp.

        Lowered to a for_each_cell select kernel (a ternary, no branch divergence concern at MVP)."""
        for nm, fv in (("mask", mask), ("a", a), ("b", b)):
            if not _is_field_value(fv):
                raise ValueError("where: %s must be a State/RHS/scalar_field value (got %r)"
                                 % (nm, fv))
        if a.vtype != b.vtype:
            raise ValueError("where: a and b must have the same value type (a is %s, b is %s)"
                             % (a.vtype, b.vtype))
        if a.block != b.block:
            raise ValueError("where: a and b must belong to the same block (a is %r, b is %r)"
                             % (a.block, b.block))
        na, nb, nm_ = self._ncomp(a), self._ncomp(b), self._ncomp(mask)
        if na is not None and nb is not None and na != nb:
            raise ValueError("where: a and b must have the same ncomp (a has %d, b has %d)" % (na, nb))
        ncomp = na if na is not None else nb
        if nm_ is not None and ncomp is not None and nm_ not in (1, ncomp):
            raise ValueError("where: mask must be 1-component or match a/b's ncomp (mask has %d, "
                             "a/b have %d)" % (nm_, ncomp))
        attrs = {"ncomp": ncomp} if ncomp is not None else {}
        return self._new(a.vtype, "where", (mask, a, b), attrs, name, a.block)

    @staticmethod
    def _ncomp(value):
        """The statically-known component count of a field value, or None when it is not pinned in the
        IR (a State / RHS ncomp is the model's n_cons, known only at codegen): a scalar_field carries its
        own ``ncomp`` attr. Used by `where` for the static a/b/mask ncomp consistency check."""
        if value.vtype == "scalar_field":
            return int(value.attrs.get("ncomp", 1))
        return None

    # --- reductions / comparisons / control flow (ADC-404a) ---
    def norm2(self, state):
        """The Euclidean norm ``||u||_2`` of a State (a collective all_reduce). Returns a Scalar value.
        Lowered as ``sqrt(adc::dot(u, u))`` -- the same collective reduction every rank must run. NOTE:
        ``adc::dot`` reduces COMPONENT 0 only, so for a multi-component State this is the L2 norm of the
        first conserved variable, not the full state norm; a full multi-component reduction is a later
        phase (the convergence loops this enables are single-residual-component for now)."""
        if not (isinstance(state, Value) and state.is_field()):
            raise ValueError("norm2: a State/RHS value is required")
        return self._new("scalar", "reduce", (state,), {"kind": "norm2"}, None, state.block)

    def dot(self, a, b):
        """The inner product ``<a, b>`` of two State values (a collective all_reduce). Returns a Scalar.
        Lowered as ``adc::dot(a, b)`` -- COLLECTIVE, called on every rank (empty ranks included)."""
        if not (isinstance(a, Value) and a.is_field() and isinstance(b, Value) and b.is_field()):
            raise ValueError("dot: two State/RHS values are required")
        return self._new("scalar", "reduce", (a, b), {"kind": "dot"}, None, a.block)

    def norm_inf(self, state):
        """The infinity norm ``max|u|`` of a State (a collective all_reduce). Returns a Scalar value.
        Lowered as ``adc::norm_inf(u)``. Like norm2/dot it reduces COMPONENT 0 only (a multi-component
        reduction is a later phase) and MUST run on every rank (it goes through the collective seam)."""
        if not (isinstance(state, Value) and state.is_field()):
            raise ValueError("norm_inf: a State/RHS value is required")
        return self._new("scalar", "reduce", (state,), {"kind": "norm_inf"}, None, state.block)

    def sum(self, state):
        """The sum ``sum_cells u`` of a State over component 0 (a collective all_reduce). Returns a
        Scalar value. Lowered as ``adc::reduce_sum(u, 0)`` -- COLLECTIVE, called on every rank (empty
        ranks included), the same seam adc::dot uses. Like norm2/dot it reduces COMPONENT 0 only (a
        full multi-component reduction is a later phase). For a specific component use
        `sum_component`."""
        if not (isinstance(state, Value) and state.is_field()):
            raise ValueError("sum: a State/RHS value is required")
        return self._new("scalar", "reduce", (state,), {"kind": "sum", "comp": 0}, None, state.block)

    def max(self, state):
        """The maximum ``max_cells u`` of a State over component 0 (a collective all_reduce). Returns a
        Scalar value. Lowered as ``adc::reduce_max(u, 0)`` (the SIGNED max, not the magnitude -- use
        `norm_inf` for max|u|). COLLECTIVE: called on every rank. Component 0 only."""
        if not (isinstance(state, Value) and state.is_field()):
            raise ValueError("max: a State/RHS value is required")
        return self._new("scalar", "reduce", (state,), {"kind": "max", "comp": 0}, None, state.block)

    def min(self, state):
        """The minimum ``min_cells u`` of a State over component 0 (a collective all_reduce). Returns a
        Scalar value. Lowered as ``adc::reduce_min(u, 0)``. COLLECTIVE: called on every rank.
        Component 0 only."""
        if not (isinstance(state, Value) and state.is_field()):
            raise ValueError("min: a State/RHS value is required")
        return self._new("scalar", "reduce", (state,), {"kind": "min", "comp": 0}, None, state.block)

    def sum_component(self, state, comp):
        """The sum ``sum_cells u(.,comp)`` of a State over conservative component @p comp (a collective
        all_reduce). Returns a Scalar value. Lowered as ``adc::reduce_sum(u, comp)``. COLLECTIVE:
        called on every rank. @p comp must be a Python int >= 0 (a runtime component is meaningless)."""
        if not (isinstance(state, Value) and state.is_field()):
            raise ValueError("sum_component: a State/RHS value is required")
        if isinstance(comp, bool) or not isinstance(comp, int) or comp < 0:
            raise ValueError("sum_component: comp must be a Python int >= 0 (got %r)" % (comp,))
        return self._new("scalar", "reduce", (state,), {"kind": "sum", "comp": int(comp)}, None,
                         state.block)

    def record_scalar(self, name, value):
        """Record a runtime Scalar @p value (e.g. ``P.norm2(R)``) into the System diagnostics map under
        @p name, retrievable after the step via ``sim.program_diagnostic(name)`` /
        ``sim.program_diagnostics()`` (spec op 23). A side-effecting op (no value): it stores the scalar
        for inspection / logging, it does not feed the numerics. @p name must be a non-empty string;
        @p value must be a Scalar value (a P.norm2 / P.dot / P.sum / ... result), not a field. Lowered
        to ``ctx.record_scalar("<name>", <scalar>)``."""
        if not isinstance(name, str) or not name:
            raise ValueError("record_scalar: name must be a non-empty string")
        if not (isinstance(value, Value) and value.vtype == "scalar"):
            raise ValueError("record_scalar: value must be a Scalar value (e.g. P.norm2(R)); got %r"
                             % (value,))
        return self._new("scalar", "record_scalar", (value,), {"diagnostic": name}, name,
                         value.block)

    def _scalar_binop(self, a, b, fn):
        """Build a Scalar arithmetic node ``a <fn> b`` (fn in add/sub/mul/div). Each operand is a Scalar
        Value or a Python number (a literal constant, stored in attrs). Used by the Value scalar dunders
        so a dt_bound can express cfl * hmin / max_wave_speed (spec s18); never evaluated in Python."""
        inputs = []
        operands = []  # per operand: ("v", index-into-inputs) or ("c", literal float)
        for x in (a, b):
            if isinstance(x, Value):
                if x.vtype != "scalar":
                    raise TypeError("scalar arithmetic operands must be Scalar values or numbers; got "
                                    "a %s value %r" % (x.vtype, x.name))
                operands.append(("v", len(inputs)))
                inputs.append(x)
            elif isinstance(x, (int, float)) and not isinstance(x, bool):
                operands.append(("c", float(x)))
            else:
                raise TypeError("scalar arithmetic operands must be Scalar values or numbers; got %r"
                                % (x,))
        block = next((x.block for x in (a, b) if isinstance(x, Value)), None)
        return self._new("scalar", "scalar_op", tuple(inputs), {"fn": fn, "operands": operands}, None,
                         block)

    def max_wave_speed(self, state):
        """The maximum |wave speed| of @p state's block (a collective reduction): the SAME per-block
        wave speed the native CFL uses (BlockState::max_speed). Returns a Scalar value. Lowered to
        ``ctx.max_wave_speed(idx, u)`` for @p state's own block (ADC-426). The denominator of a
        CFL-style dt bound cfl * hmin / w (spec s18). REUSES the block's wave-speed closure -- it does
        not recompute the speed."""
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("max_wave_speed: a State value is required")
        return self._new("scalar", "max_wave_speed", (state,), {}, None, state.block)

    def hmin(self):
        """The MIN physical cell size of the grid (Cartesian min(dx, dy); polar min(dr, r_min*dtheta)):
        the SAME hmin the native CFL uses. Returns a Scalar value. Lowered to ``ctx.hmin()``. The
        numerator factor of a CFL-style dt bound cfl * hmin / max_wave_speed (spec s18)."""
        return self._new("scalar", "hmin", (), {}, None, None)

    def _compare(self, lhs, rhs, cmp):
        """Build a Bool predicate ``s_lhs <cmp> rhs`` (re-evaluated each loop pass). @p rhs is a Python
        float tolerance (stored as a literal) or another Scalar value (compared at runtime). Inputs are
        the Scalar operand(s); the float bound lives in attrs['rhs']."""
        if isinstance(rhs, (int, float)):
            return self._new("bool", "compare", (lhs,), {"cmp": cmp, "rhs": float(rhs)}, None,
                             lhs.block)
        if isinstance(rhs, Value) and rhs.vtype == "scalar":
            return self._new("bool", "compare", (lhs, rhs), {"cmp": cmp}, None, lhs.block)
        raise TypeError("compare: the right-hand side must be a float tolerance or a Scalar value, "
                        "got %r" % (rhs,))

    def while_(self, state, cond_fn, body_fn):
        """A convergence loop: starting from @p state, while ``cond_fn(self, x)`` holds, replace x by
        ``body_fn(self, x)``; return the final State.

        The condition and body are RE-EVALUATED each iteration, so the ops they build are captured into
        a separate recording sub-block (NOT the flat SSA list) and re-emitted inside a C++ loop. The
        loop variable is the SAME C++ MultiFab across passes (mutated in place).

          - ``cond_fn(self, x)`` must return a Bool value (e.g. ``self.norm2(diff) > tol``);
          - ``body_fn(self, x)`` must return the next-iteration State (e.g. a linear_combine)."""
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("while_: the loop variable must be a State value")
        if self._recording:
            raise NotImplementedError(
                "while_: nested control flow is a later phase; a while_ body cannot itself open a "
                "while_ yet")
        cond_block, cond_val = self._record(cond_fn, state)
        if not (isinstance(cond_val, Value) and cond_val.vtype == "bool"):
            raise ValueError("while_: cond_fn must return a Bool value (e.g. P.norm2(d) > tol)")
        body_block, next_state = self._record(body_fn, state)
        if not (isinstance(next_state, Value) and next_state.vtype == "state"):
            raise ValueError("while_: body_fn must return the next-iteration State value")
        if next_state.block != state.block:
            raise ValueError("while_: body_fn must return a State of the same block as the loop "
                             "variable")
        return self._new("state", "while", (state,),
                         {"cond_block": cond_block, "cond": cond_val,
                          "body_block": body_block, "body": next_state},
                         None, state.block)

    def static_range(self, state, count, body_fn):
        """A COMPILE-TIME (unrolled) loop: apply ``body_fn(self, x)`` to the State @p count times,
        threading the result, and return the final State. @p count must be a Python int known at IR
        build time -- the loop is unrolled HERE (no IR control-flow op, no C++ loop): it simply builds
        @p count copies of the body ops in order. Use `range` for a C++ ``for`` over a fixed count."""
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("static_range count must be a Python int (compile-time); use P.range for "
                            "a runtime / C++-loop count")
        if count < 0:
            raise ValueError("static_range count must be non-negative")
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("static_range: the loop variable must be a State value")
        x = state
        for _ in range(count):
            x = body_fn(self, x)
            if not (isinstance(x, Value) and x.vtype == "state" and x.block == state.block):
                raise ValueError("static_range: body_fn must return a State of the loop variable's "
                                 "block")
        return x

    def range(self, state, count, body_fn):
        """A C++ ``for`` loop over a FIXED count: from @p state, apply ``body_fn(self, x)`` @p count
        times, threading the loop-variable State in place, and return the final State. @p count must be
        a Python int (a runtime/Scalar count is a later phase). The body is RE-EXECUTED each pass, so
        its ops are captured into a recording sub-block (NOT the flat SSA list) and emitted ONCE inside
        the loop. Use `static_range` to unroll instead."""
        if isinstance(count, Value):
            if count.vtype == "scalar":
                raise NotImplementedError("range with a runtime Scalar count is deferred; use a "
                                          "Python int")
            raise TypeError("range count must be a Python int")
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("range count must be a Python int")
        if count < 0:
            raise ValueError("range count must be non-negative")
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("range: the loop variable must be a State value")
        if self._recording:
            raise NotImplementedError("range: nested control flow is a later phase; a control-flow "
                                      "body cannot itself open a range yet")
        body_block, next_state = self._record(body_fn, state)
        if not (isinstance(next_state, Value) and next_state.vtype == "state"
                and next_state.block == state.block):
            raise ValueError("range: body_fn must return the next-iteration State of the same block")
        return self._new("state", "range", (state,),
                         {"count": int(count), "body_block": body_block, "body": next_state},
                         None, state.block)

    def if_(self, state, cond, body_fn):
        """A C++ ``if`` branch: from @p state, if the runtime Bool @p cond holds, replace the state by
        ``body_fn(self, x)``; otherwise leave it unchanged. Returns the (possibly updated) State. @p
        cond is a Bool value built BEFORE if_ (e.g. ``P.norm2(d) > tol``), evaluated ONCE. The body ops
        are captured into a recording sub-block and emitted inside the branch."""
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("if_: the state must be a State value")
        if not (isinstance(cond, Value) and cond.vtype == "bool"):
            raise ValueError("if_: cond must be a Bool value (e.g. P.norm2(d) > tol)")
        if self._recording:
            raise NotImplementedError("if_: nested control flow is a later phase; a control-flow body "
                                      "cannot itself open an if_ yet")
        body_block, next_state = self._record(body_fn, state)
        if not (isinstance(next_state, Value) and next_state.vtype == "state"
                and next_state.block == state.block):
            raise ValueError("if_: body_fn must return a State of the same block as the input state")
        return self._new("state", "if", (state, cond),
                         {"body_block": body_block, "body": next_state}, None, state.block)

    def _record(self, fn, x):
        """Run a control-flow callable ``fn(self, x)`` with a fresh recording scope active, capturing the
        ops it builds into a sub-block (returned with the value fn produced). The sub-block ops are NOT
        appended to self._values (they belong to the owning control-flow op)."""
        sub = []
        self._recording.append(sub)
        try:
            out = fn(self, x)
        finally:
            self._recording.pop()
        return sub, out

    # --- optional dt bound (spec s18 / ADC-417) ----------------------------------------------------
    def set_dt_bound(self, expr_or_fn):
        """Set an OPTIONAL dt bound for this Program (spec s18). The generated .so exports it as a
        SECOND ABI function (``adc_program_dt_bound``) alongside the macro step; ``step_cfl`` then uses
        ``min(native CFL dt, program dt bound)``. Without a dt bound the native CFL is UNCHANGED.

        @p expr_or_fn is either a callable ``f(P, cfl)`` returning a Scalar (the common form -- it reads
        the runtime ``cfl`` to build e.g. ``cfl * P.hmin() / P.max_wave_speed(U)``), or a Scalar Value
        already built (a fixed bound, cfl-independent). The body is recorded as a scalar sub-program; it
        is NOT run in Python during ``sim.step_cfl`` -- it lowers to C++ that reads the live state /
        reductions. The dt_bound body may read state / fields / scalars only (no commit, no field write):
        a State / RHS / field op in it is rejected."""
        if self._dt_bound is not None:
            raise ValueError("set_dt_bound: a dt bound is already set (set it at most once)")
        if self._recording:
            raise NotImplementedError("set_dt_bound: a dt bound cannot be set inside a control-flow body")
        sub = []
        self._recording.append(sub)
        try:
            if callable(expr_or_fn):
                # The cfl placeholder is a runtime Scalar the body composes with (cfl * hmin / w).
                cfl = self._new("scalar", "cfl", (), {}, "cfl", None)
                result = expr_or_fn(self, cfl)
            else:
                result = expr_or_fn
        finally:
            self._recording.pop()
        if not (isinstance(result, Value) and result.vtype == "scalar"):
            raise ValueError("set_dt_bound: the body must return a Scalar value (e.g. "
                             "cfl * P.hmin() / P.max_wave_speed(U)); got %r" % (result,))
        # The dt_bound is a READ-ONLY scalar program: it may READ state / fields (P.state, P.solve_fields)
        # and produce scalars (reductions, hmin, max_wave_speed, cfl, scalar arithmetic, comparisons),
        # but it must NOT write a field or commit -- a flux / linear_combine / source / projection in it
        # is rejected fail-loud (it would mutate the state during the CFL dt evaluation).
        _DT_BOUND_OPS = frozenset({"state", "solve_fields", "reduce", "compare", "cfl", "hmin",
                                   "max_wave_speed", "scalar_op"})
        for v in sub:
            if v.op not in _DT_BOUND_OPS:
                raise ValueError("set_dt_bound: the dt bound body may read state / fields and compute "
                                 "scalars only (no field write / commit); op '%s' (value '%s') is not "
                                 "allowed" % (v.op, v.name))
        self._dt_bound = (sub, result)
        return result

    def dt_bound(self, fn):
        """Decorator form of `set_dt_bound`: ``@P.dt_bound`` over ``def f(P, cfl): return ...`` records
        the dt bound (spec s18) and returns the function unchanged (so the name stays usable)."""
        self.set_dt_bound(fn)
        return fn

    def has_dt_bound(self):
        """True iff an optional dt bound was set (spec s18)."""
        return self._dt_bound is not None

    # --- IR optimization passes (Spec 3 s28, ADC-465) --------------------------------------------
    # SAFE-BY-DEFAULT ALLOW-LIST. A flat node is removable ONLY if its op is enumerated here; EVERY
    # other op (known side-effecting, buffer-writing, sub-block-owning, OR new/unknown) is treated as
    # live even when its result looks unconsumed. This is the inverse of a blacklist: a buffer-writing
    # op whose ``_emit_op`` lowering ALIASES a caller-allocated input buffer (``var[v.id] =
    # var[out_in.id]``) is side-effecting on that buffer even though it has no dataflow output edge --
    # e.g. ``schur_rhs`` fills the ``rhs`` scratch that ``solve_linear`` then reads by BUFFER IDENTITY,
    # not via an input edge. A blacklist silently drops such an op (corrupting the codegen while
    # ``validate()`` stays True); a whitelist cannot, because the op is not listed.
    #
    # An op qualifies for this list ONLY if its ``_emit_op`` branch was verified to (a) ALLOCATE A
    # FRESH result scratch (``ctx.rhs_scratch_like`` / ``ctx.scratch_state_like`` / ``ctx.alloc_*`` /
    # a fresh ``s%d`` scalar local), NOT alias an input, AND (b) have no other observable side effect.
    #   rhs / source / apply  -> ``r%d = ctx.rhs_scratch_like(state)`` then a pure compute fill
    #   linear_combine        -> ``u%d = ctx.scratch_state_like(base)`` (the NON-commit branch; a
    #                            committed linear_combine is a commit root, never a dead-node candidate)
    #   linear_source         -> a pure operator DECLARATION node (no inputs, no emitted statement)
    #   solve_local_linear    -> ``u%d = ctx.scratch_state_like(base)`` (fresh; per-cell dense solve)
    #   cell_compare / where  -> a fresh ``ctx.alloc_scalar_field`` / ``ctx.scratch_state_like`` mask
    #   reduce / scalar_op    -> a fresh ``s%d`` scalar local (a collective reduce is recomputed if
    #                            re-added later; dropping an UNCONSUMED one removes only dead arithmetic)
    #   compare               -> an inline boolean expression, no statement of its own
    # Deliberately EXCLUDED (kept live): the buffer-writers schur_rhs / schur_explicit_flux / laplacian
    # / gradient / divergence / apply_laplacian_coeff / schur_coeffs / schur_reconstruct / schur_energy
    # (alias an input buffer); the side-effecting solve_fields[_from_blocks] / project / fill_boundary /
    # store_history / record_scalar; solve_linear (reads its rhs by buffer identity); scalar_field /
    # state / history (scratch/state bindings other ops fill or alias); and the sub-block ops below.
    _REMOVABLE_OPS = frozenset({
        "rhs", "source", "apply", "linear_combine", "linear_source", "solve_local_linear",
        "cell_compare", "where", "reduce", "scalar_op", "compare",
    })
    # Ops that own a recorded sub-block (a while / if / range body, a matrix-free apply, a Newton
    # residual). v1 does NOT descend into sub-blocks, so these are treated as always-live roots and
    # every value they (or their sub-blocks) read is conservatively kept. They are simply absent from
    # the allow-list above (hence live); listed here only to drive the sub-block reference walk.
    _SUBBLOCK_OPS = frozenset({
        "while", "if", "range", "matrix_free_operator", "solve_local_nonlinear",
    })
    # Ops PROVEN PURE for common-subexpression elimination (Spec 3 s28, ADC-465): each allocates a
    # FRESH result scratch from its inputs ALONE, reads nothing through a buffer-identity side channel,
    # and has no side effect (no aux/Poisson/history/diagnostic write, no in-place mutation). Two such
    # nodes with the same op + vtype + block + inputs + attrs compute the SAME value, so the second can
    # alias the first with no change to the result. This is a STRICT subset of ``_REMOVABLE_OPS``:
    #   - ``reduce`` is EXCLUDED (a collective reduction is a global communication; conservatively never
    #     deduplicated even though two identical reduces give the same scalar -- CSE soundness, not perf,
    #     drives the list, and a duplicated reduce is dead-node territory, not CSE);
    #   - ``solve_local_linear`` is EXCLUDED (it reads its rhs State by buffer; a later op may overwrite
    #     that rhs buffer in place between two solves, so two same-input solves are NOT provably equal);
    #   - ``where`` / ``cell_compare`` ARE pure (fresh scratch, component-wise from inputs only).
    #   - ``rhs`` / ``source`` / ``apply`` are EXCLUDED: their per-cell kernels read the SHARED System
    #     aux (``ctx.aux()``) BY BUFFER IDENTITY, not through a dataflow input edge (see
    #     ``_emit_source_kernel`` / ``_emit_apply_kernel`` / ``_cell_locals`` -> ``auxA(i,j,...)``). A
    #     ``solve_fields`` mutates that aux in place, so two same-(state,fields)-input source/apply/rhs
    #     nodes straddling a field solve are NOT equal (the second reads the freshly solved field). CSE
    #     keys only on dataflow inputs, so collapsing them would silently read the stale pre-solve aux.
    # Every op OUTSIDE this set is treated as non-CSE-able, so a buffer-writer / side-effecting / aux-
    # reading / unknown op is never collapsed (safe-by-default, mirroring the dead-node allow-list).
    _PURE_OPS = frozenset({
        "linear_combine", "linear_source", "cell_compare", "where", "scalar_op", "compare",
    })
    # Ops that WRITE a block state or otherwise change what a subsequent ``solve_fields`` over the same
    # state would see (the Poisson RHS reads every block's live state + the shared aux). A redundant
    # ``solve_fields`` may be eliminated ONLY when none of these appears between the two solves over the
    # same state input -- conservatively, ANY commit, in-place state mutation (project), boundary fill,
    # history store, or a second field solve into the shared aux counts as a state/aux barrier.
    _STATE_BARRIER_OPS = frozenset({
        "project", "fill_boundary", "store_history",
        "solve_fields", "solve_fields_from_blocks",
    })

    @staticmethod
    def _subblock_value_refs(v):
        """Yield every Value an op references THROUGH its attrs (sub-block result pointers + the ops
        nested in its recorded sub-blocks). Used to keep alive anything a control-flow / matrix-free
        node closes over from the enclosing scope -- v1 never rewrites a sub-block, so it is all live.
        The flat ``v.inputs`` already covers the directly-passed values; this adds the attr-borne ones."""
        attrs = v.attrs
        for key in ("cond", "body", "residual", "iterate", "guess",
                    "apply_result", "apply_in", "apply_out"):
            ref = attrs.get(key)
            if isinstance(ref, Value):
                yield ref
            elif isinstance(ref, _Affine):
                for term, _ in ref.terms:
                    yield term
        sched = attrs.get("schedule")
        if sched is not None:
            cond = getattr(sched, "params", {}).get("cond")
            if isinstance(cond, Value):
                yield cond  # a when(cond) predicate is live (kept off the dead-node / CSE-drop path)
        for key in ("cond_block", "body_block", "apply_block", "residual_block"):
            block = attrs.get(key)
            if block:
                for w in block:
                    yield w
                    yield from w.inputs
                    yield from Program._subblock_value_refs(w)

    def _live_value_ids(self):
        """The set of value ids reverse-reachable from the live roots: the commits plus every flat node
        whose op is NOT on the ``_REMOVABLE_OPS`` allow-list (safe-by-default -- a buffer-writing,
        side-effecting, sub-block-owning, or unknown op is a live root). A flat node is DEAD only if its
        op IS allow-listed AND no live op consumes its result. Sub-block-internal values are included so
        a dead-node clone keeps a self-consistent IR."""
        by_id = {v.id: v for v in self._values}
        # Sub-block ops are not in self._values (they belong to their owning op); index them too so a
        # reference from one sub-block op to another resolves during the walk.
        for v in self._values:
            for w in Program._subblock_value_refs(v):
                by_id.setdefault(w.id, w)
        roots = [s.id for s in self._commits.values()]
        for v in self._values:
            if v.op not in Program._REMOVABLE_OPS:
                roots.append(v.id)
        live, stack = set(), list(roots)
        while stack:
            vid = stack.pop()
            if vid in live or vid not in by_id:
                continue
            live.add(vid)
            v = by_id[vid]
            for inp in v.inputs:
                stack.append(inp.id)
            for w in Program._subblock_value_refs(v):
                stack.append(w.id)
        return live

    def eliminate_dead_nodes(self):
        """Return a NEW Program with the dead flat-list nodes removed (Spec 3 s28 dead-node
        elimination, ADC-465). An OPT-IN pass: call it explicitly to optimize a copy -- it NEVER runs
        on the default ``emit_cpp_program`` path, so it cannot change an existing compiled program.

        The pass is SAFE-BY-DEFAULT: a flat node is DEAD only if its op is on the ``_REMOVABLE_OPS``
        allow-list (ops verified to allocate a FRESH result scratch and have no other side effect: rhs,
        source, apply, linear_combine, linear_source, solve_local_linear, cell_compare, where, reduce,
        scalar_op, compare) AND no live op consumes its result. EVERY other op -- the buffer-writers
        that alias a caller-allocated input buffer (schur_rhs, laplacian, gradient, divergence,
        schur_*), the side-effecting ops (solve_fields, project, fill_boundary, store_history,
        record_scalar), solve_linear, and the sub-block-owning ops (while/if/range,
        matrix_free_operator, solve_local_nonlinear) -- is treated as LIVE even when its result looks
        unconsumed, so an unknown/new op is NEVER wrongly dropped. The live set is reverse-reachability
        from the commits plus those non-removable nodes. The surviving nodes are renumbered to
        contiguous ids in their original order, so a program with no dead node round-trips byte-for-byte
        (same ``_ir_hash`` and emitted C++) and one with a dead node matches the same program written
        without it. The histories, optional dt bound and bound operator registry carry over unchanged."""
        live = self._live_value_ids()
        return self._rebuild(lambda v: v.id in live)

    def _rebuild(self, keep, alias=None):
        """Clone this Program into a fresh one keeping the flat nodes for which ``keep(v)`` is true,
        renumbering surviving ids to a contiguous 0.. range in original order. Sub-blocks are cloned
        wholesale (never filtered). The clone reproduces the IR identity of an equivalent hand-built
        Program (same serialization), so it is byte-identical when nothing was dropped.

        @p alias (optional) maps a DROPPED node id -> the kept representative node id it should be
        replaced by (the CSE / redundant-solve passes use it to rewire every use of a duplicate onto its
        survivor). Every reference -- a flat input, an attr-borne Value / affine ref, a commit target --
        is resolved THROUGH this alias before id lookup, so a dropped node never leaves a dangling
        reference. A dropped node MUST have an alias entry (the passes guarantee a representative whose
        id < the duplicate's, hence already cloned); a kept node maps to itself. Without an alias map
        the behavior is the historical drop-only rebuild."""
        out = Program(self.name)
        out.dt = self.dt
        out._histories = dict(self._histories)
        out._registry = self._registry
        idmap = {}  # old Value -> new Value
        by_id = {v.id: v for v in self._values}
        for v in self._values:  # sub-block ops too, so an alias to a sub-block-internal id resolves
            for w in Program._subblock_value_refs(v):
                by_id.setdefault(w.id, w)
        alias = alias or {}

        def rep(v):
            """Follow @p v through the alias chain to the surviving representative Value (identity for a
            kept node). The passes only alias onto an EARLIER, kept node, so the chain terminates."""
            seen = set()
            while v.id in alias and alias[v.id] != v.id:
                if v.id in seen:  # defensive: never loop on a malformed alias map
                    break
                seen.add(v.id)
                v = by_id[alias[v.id]]
            return v

        def clone_block(block):
            return [clone(w) for w in block]

        def remap(ref):
            if isinstance(ref, Value):
                return idmap[rep(ref)]
            if isinstance(ref, _Affine):
                return _Affine([(idmap[rep(v)], c) for v, c in ref.terms])
            return ref

        def clone_attrs(v):
            attrs = {}
            for key, val in v.attrs.items():
                if key in ("cond_block", "body_block", "apply_block", "residual_block"):
                    attrs[key] = clone_block(val) if val else val
                elif key in ("cond", "body", "residual", "iterate", "guess",
                             "apply_result", "apply_in", "apply_out"):
                    attrs[key] = remap(val)
                elif key == "schedule" and val is not None and isinstance(
                        getattr(val, "params", {}).get("cond"), Value):
                    # a when(cond) schedule embeds a predicate Value in params["cond"]; remap it onto
                    # the survivor so a CSE-collapsed/renumbered predicate is not left dangling.
                    attrs[key] = Schedule(val.kind, val.policy,
                                          **{**val.params, "cond": remap(val.params["cond"])})
                else:
                    attrs[key] = val
            return attrs

        def deps(v):
            """The values v depends on that must be cloned (hence id-assigned) BEFORE v, in their
            ORIGINAL creation order. A fresh build records the inputs and most sub-blocks before the
            owning node, BUT a matrix_free_operator is created (its node id assigned) BEFORE
            ``set_apply`` records its apply sub-block -- the node id precedes the sub-block ids. Ordering
            every dependency by its original id (ascending) reproduces the build order verbatim for both
            shapes, so a no-drop clone is byte-identical (same renumbering) rather than reordering the
            matrix_free_operator node after its own sub-block. Each input is resolved THROUGH the alias
            map, so a dropped duplicate is replaced by its (already-earlier) representative."""
            seen = []
            for inp in v.inputs:
                seen.append(rep(inp))
            for key in ("cond_block", "body_block", "apply_block", "residual_block"):
                block = v.attrs.get(key)
                if block:
                    seen.extend(block)
            # A matrix_free_operator's sub-block ops are created AFTER the node, so they must NOT be
            # forced ahead of it; an input / control-flow body is created BEFORE. Keep only the deps
            # whose original id precedes v's (the genuine predecessors) and visit them id-ascending.
            return sorted((w for w in seen if w.id < v.id), key=lambda w: w.id)

        def clone(v):
            if v in idmap:
                return idmap[v]
            # Assign new ids in ORIGINAL creation order: clone every predecessor (id < v.id) first,
            # id-ascending, then v, then any sub-block op created AFTER v (e.g. a matrix_free_operator's
            # apply ops, whose original ids exceed the operator node's). Inputs / attr refs are remapped
            # through idmap after alias resolution (every referenced surviving value is mappable on its
            # own clone).
            for w in deps(v):
                clone(w)
            vid = out._next_id
            out._next_id += 1
            nv = Value(out, vid, v.vtype, v.op, [idmap[rep(i)] for i in v.inputs],
                       clone_attrs(v), v.name, v.block)
            nv.space = v.space
            idmap[v] = nv
            return nv

        # Clone all surviving flat nodes (and, transitively, their sub-block ops and any later-created
        # sub-block ops) in ascending original id, so the contiguous renumbering matches the original
        # build order exactly -- a no-op clone is byte-for-byte identical.
        kept = sorted((v for v in self._values if keep(v)), key=lambda v: v.id)
        for v in kept:
            clone(v)
        out._values = [idmap[v] for v in kept]
        out._commits = {b: idmap[rep(s)] for b, s in self._commits.items()}
        if self._dt_bound is not None:
            sub, result = self._dt_bound
            cloned_sub = [clone(w) for w in sub]
            out._dt_bound = (cloned_sub, idmap[result])
        return out

    # --- common-subexpression elimination (Spec 3 s28, ADC-465) ---
    @staticmethod
    def _cse_key(v, canon):
        """A canonical, alias-aware fingerprint of a PURE node: its op, vtype, block, the attrs the IR
        hash uses, and its inputs MAPPED THROUGH @p canon (each input id replaced by the id of the
        representative node it was deduplicated to). Two pure nodes with the same key compute the SAME
        value, so the later one can alias the earlier. Built on the same ``_serialize_node`` the IR hash
        uses (so attr equality is exactly the hash's notion of equality), with the node id stripped (it
        is position, not identity) and the input ids canonicalized."""
        node = Program._serialize_node(v)
        node.pop("id", None)
        node["inputs"] = tuple(canon.get(i, i) for i in node["inputs"])
        # JSON-serialize the attrs dict to a stable string so the whole key is hashable / comparable
        # exactly as the IR hash compares it.
        return (node["op"], node["vtype"], node["block"], node["inputs"],
                json.dumps(node["attrs"], sort_keys=True, separators=(",", ":")))

    def eliminate_common_subexpressions(self):
        """Return a NEW Program with duplicated PURE sub-IR computed once and aliased (Spec 3 s28
        common-subexpression elimination, ADC-465).

        PROVEN SOUND, not heuristic: a node is a CSE candidate ONLY if its op is on the
        ``_PURE_OPS`` allow-list -- ops that allocate a fresh result from their inputs alone, read no
        buffer through a side channel, and have no side effect. For each such node, a canonical key (op,
        vtype, block, attrs, and inputs mapped to their already-chosen representatives) is computed in
        creation order; the FIRST node with a given key is the representative, and every later node with
        the same key is dropped and its uses rewired to the representative. Because the key is exactly
        the IR hash's notion of equality (same ``_serialize_node`` attrs) over identical
        (canonicalized) inputs, the representative computes a bit-identical result -- so aliasing the
        duplicate CANNOT change the emitted numerics. Every NON-pure op (a reduce, a solve, a
        buffer-writer, a side-effecting op, an unknown op) is NEVER a representative target and is never
        dropped, so it is always recomputed -- safe-by-default.

        The pass is OPT-IN: it never runs on the default ``emit_cpp_program`` path. Sub-blocks are not
        descended into (v1), so a value consumed only inside a control-flow body is left untouched. A
        program with no duplicated pure sub-IR rebuilds BYTE-FOR-BYTE identically (same ``_ir_hash`` and
        emitted C++); a program with a duplicate emits, after the pass, C++ identical to the same
        program written with the value computed once."""
        canon = {}        # duplicate node id -> representative node id (the survivor it aliases)
        reps = {}         # cse key -> representative node id
        drop = set()      # ids of dropped duplicates
        for v in self._values:
            if v.op not in Program._PURE_OPS:
                continue
            # An input that was itself dropped resolves to its representative (so chained duplicates
            # collapse onto one representative chain, not a dangling id).
            if any(i.id in drop for i in v.inputs):
                # canon already maps every dropped input to its survivor; the key uses canon, so this
                # node keys against the survivors -- no special handling needed beyond canon lookup.
                pass
            key = Program._cse_key(v, canon)
            if key in reps:
                canon[v.id] = reps[key]
                drop.add(v.id)
            else:
                reps[key] = v.id
                canon[v.id] = v.id
        if not drop:
            return self._rebuild(lambda v: True)  # byte-identical no-op clone
        return self._rebuild(lambda v: v.id not in drop, alias=canon)

    # --- redundant field-solve elimination (Spec 3 s28, ADC-465) ---
    def eliminate_redundant_field_solves(self):
        """Return a NEW Program with a provably-redundant second ``solve_fields`` removed and aliased
        (Spec 3 s28 redundant-solve elimination, ADC-465).

        ``solve_fields`` is side-effecting (it fills the shared phi/aux and returns a FieldContext), so
        it is NEVER touched by CSE or dead-node elimination. But two ``solve_fields`` over the SAME
        state input with NO intervening STATE OR AUX MUTATION recompute the identical fields: the
        second is redundant and its FieldContext can alias the first. This is the ONLY field solve this
        pass removes, and ONLY when redundancy is PROVABLE.

        CONSERVATIVE soundness rule: walking the flat list in order, a ``solve_fields(state=U,
        field=f)`` is redundant iff an EARLIER ``solve_fields`` with the SAME state input AND the same
        ``field`` attr exists AND, between the two, NO op on the ``_STATE_BARRIER_OPS`` set (a commit
        target write, ``project``, ``fill_boundary``, ``store_history``, or ANY other field solve --
        which would re-fill the shared aux) appears. The Poisson RHS reads every block's LIVE state, so
        a write to ANY block's state -- not just U's -- is a barrier; a ``linear_combine`` that is a
        commit target is therefore a barrier too. If anything between the two solves could have changed
        what the elliptic solve sees, the second is kept. The single-block, no-mutation case (e.g. a
        macro accidentally solving twice from U^n before the first stage) is the one provably-redundant
        shape eliminated; everything else is left as written.

        OPT-IN: never on the default emit path. Byte-identical no-op when no redundant solve exists."""
        commit_ids = {s.id for s in self._commits.values()}
        active = {}       # (state input id, field attr) -> the live representative solve_fields id
        canon = {}
        drop = set()
        for v in self._values:
            if v.op == "solve_fields":
                (state_in,) = v.inputs
                sig = (state_in.id, v.attrs.get("field"))
                prior = active.get(sig)
                if prior is not None:
                    # A redundant re-solve over the same state with no barrier since `prior`.
                    canon[v.id] = prior
                    drop.add(v.id)
                    continue
                active[sig] = v.id
                # This solve_fields is itself a barrier for OTHER signatures (it re-fills the shared
                # aux), so any pending solve over a different state is no longer safe to reuse.
                active = {sig: v.id}
                continue
            # A barrier op invalidates every pending reuse: a state write (commit target, project,
            # fill_boundary), a history store, or anything that mutates what the elliptic solve reads.
            if v.op in Program._STATE_BARRIER_OPS or v.id in commit_ids:
                active = {}
        if not drop:
            return self._rebuild(lambda v: True)
        return self._rebuild(lambda v: v.id not in drop, alias=canon)

    # --- proven-safe optimization pipeline (Spec 3 s28, ADC-465) ---
    # The TRANSFORM passes, in the order ``optimize`` runs them. Each is PROVEN to preserve the emitted
    # numerics (see its docstring) and is a byte-identical no-op when it finds nothing to do, so the
    # whole pipeline is a no-op on an already-optimal Program. Analysis passes (liveness / estimate /
    # GPU detector) are reports, NOT in this list -- they never rewrite the IR.
    _OPTIMIZE_PASSES = (
        ("eliminate_dead_nodes", "dead-node elimination"),
        ("eliminate_common_subexpressions", "common-subexpression elimination"),
        ("eliminate_redundant_field_solves", "redundant field-solve elimination"),
    )

    def optimize(self):
        """Return a NEW Program with the proven-safe Spec 3 s28 transform passes applied in sequence
        (ADC-465): dead-node elimination, common-subexpression elimination, redundant field-solve
        elimination. OPT-IN -- the default ``emit_cpp_program`` path never optimizes. Each pass is
        proven to preserve the emitted numerics and is byte-identical when it changes nothing, so a
        Program with no optimizable structure round-trips byte-for-byte (same ``_ir_hash`` and C++)
        with the pipeline on or off (the spec's hard requirement: optimization must not change
        results). Use :meth:`dump_passes` to inspect what each pass did."""
        prog = self
        for name, _ in Program._OPTIMIZE_PASSES:
            prog = getattr(prog, name)()
        return prog

    def dump_passes(self):
        """Inspect the optimization pipeline: for each proven-safe transform pass, the number of flat
        nodes before / after and whether it changed the IR hash. A report only -- it RUNS the pipeline
        on a copy (``self`` is never mutated) and returns a human-readable trace, so a reviewer can see
        which pass fired and that an all-no-op pipeline leaves the hash unchanged (Spec 3 s28,
        ADC-465)."""
        lines = ["# optimization pipeline for Program %r" % self.name]
        prog = self
        for name, label in Program._OPTIMIZE_PASSES:
            before_n, before_h = len(prog._values), prog._ir_hash()
            nxt = getattr(prog, name)()
            after_n, after_h = len(nxt._values), nxt._ir_hash()
            changed = after_h != before_h
            lines.append("  %-34s %3d -> %3d nodes  %s"
                         % (label, before_n, after_n, "CHANGED" if changed else "no-op"))
            prog = nxt
        lines.append("  final: %d nodes, hash %s" % (len(prog._values), prog._ir_hash()[:12]))
        return "\n".join(lines)

    # --- analysis passes: liveness / buffer reuse / cost estimate / GPU detectors (Spec 3 s28) ---
    # Ops that allocate ONE step-body scratch buffer (a MultiFab the size of the block state / a
    # scalar field). The number of buffers each op writes per "kernel" + how many for_each_cell kernels
    # it launches drive the memory-traffic and kernel-count estimates and the per-scratch live ranges.
    # These counts are STRUCTURAL (read off the IR / the lowering above), not a measured profile -- the
    # measured GPU kernel count is a ROMEO run; this is the host-side static estimate.
    _SCRATCH_OPS = frozenset({
        "rhs", "source", "apply", "linear_combine", "linear_source", "solve_local_linear",
        "solve_local_nonlinear", "cell_compare", "where", "coupled_rate",
    })
    # Ops that launch at least one per-cell (for_each_cell) kernel when lowered -- the small-kernel
    # count a GPU launch-overhead detector flags. A linear_combine lowers to axpy/lincomb (vectorized,
    # counted as one kernel); a rhs to a divergence/flux kernel; a source/apply/where/cell_compare/
    # coupled_rate to an explicit for_each_cell. solve_fields / solve_linear launch many internal
    # kernels (an elliptic / Krylov solve) -- counted as a HEAVY kernel, not a small one.
    _PERCELL_KERNEL_OPS = frozenset({
        "rhs", "source", "apply", "linear_combine", "linear_source", "solve_local_linear",
        "solve_local_nonlinear", "cell_compare", "where", "coupled_rate", "project", "fill_boundary",
    })
    _HEAVY_KERNEL_OPS = frozenset({
        "solve_fields", "solve_fields_from_blocks", "solve_linear",
    })

    def scratch_liveness(self):
        """Per-scratch LIVE RANGES over the linear step-body order (Spec 3 s28 scratch-liveness
        analysis, ADC-465). A REPORT, not a transform: it never rewrites the IR.

        Each scratch-allocating flat node (rhs / source / apply / linear_combine / ... -- the
        ``_SCRATCH_OPS`` set, the ops that allocate a step-body MultiFab) owns one buffer. Its live
        range is ``[def_index, last_use_index]``: the node's own position in the flat list, to the
        position of the LAST flat node that reads it (through a flat input OR an affine combine term).
        A scratch read only inside a control-flow / matrix-free sub-block (v1 does not descend) is
        conservatively live to the END of the body. Returns a list of dicts (one per scratch) with the
        node name, op, def/last-use indices and the live span -- the raw material for buffer reuse."""
        order = {v.id: i for i, v in enumerate(self._values)}
        last_use = {}  # scratch node id -> last flat index that reads it
        end = len(self._values) - 1
        for i, v in enumerate(self._values):
            # Flat inputs + affine-term refs + sub-block closed-over refs are all "reads".
            reads = set(inp.id for inp in v.inputs)
            for key in ("cond", "body", "residual", "iterate", "guess",
                        "apply_result", "apply_in", "apply_out"):
                ref = v.attrs.get(key)
                if isinstance(ref, Value):
                    reads.add(ref.id)
                elif isinstance(ref, _Affine):
                    reads.update(term.id for term, _ in ref.terms)
            for rid in reads:
                if rid in order:
                    last_use[rid] = max(last_use.get(rid, order[rid]), i)
            # A sub-block-owning op may read a scratch inside its body (v1 never inspects it): keep every
            # such closed-over scratch live to the END (conservative, never under-estimates the span).
            if v.op in Program._SUBBLOCK_OPS:
                for w in Program._subblock_value_refs(v):
                    if w.id in order:
                        last_use[w.id] = end
        # A committed scratch is read by the final commit copy (after the last flat node).
        for state in self._commits.values():
            if state.id in order:
                last_use[state.id] = max(last_use.get(state.id, order[state.id]), end + 1)
        out = []
        for v in self._values:
            if v.op not in Program._SCRATCH_OPS:
                continue
            d = order[v.id]
            lu = last_use.get(v.id, d)  # an unused scratch is live only at its own def
            out.append({"name": v.name, "op": v.op, "block": v.block,
                        "def_index": d, "last_use_index": lu, "live_span": lu - d})
        return out

    def buffer_reuse_report(self):
        """Buffer-reuse opportunities from the scratch live ranges (Spec 3 s28 buffer reuse, ADC-465).
        A REPORT, not a transform: the codegen may keep separate buffers, but the memory ESTIMATE
        reflects reuse.

        Greedy left-to-right register-allocation over the live ranges from :meth:`scratch_liveness`:
        scratches sorted by def index; a free buffer (one whose last use precedes the current
        scratch's def) is reused, else a new buffer is allocated. Two scratches share a buffer only
        when their live ranges are DISJOINT (the earlier one's last read strictly precedes the later
        one's def), so reuse can never alias two simultaneously-live values -- this is why it is a
        sound ESTIMATE of the minimum buffer count, independent of whether the codegen actually
        reuses. Returns ``{"scratch_count", "buffer_count", "reused", "assignment"}`` where
        ``reused`` is how many scratches landed on a recycled buffer and ``assignment`` maps each
        scratch name to its buffer index."""
        ranges = sorted(self.scratch_liveness(), key=lambda r: r["def_index"])
        free = []          # buffer index -> last_use_index of its current occupant
        assignment = {}
        reused = 0
        n_buffers = 0
        for r in ranges:
            slot = None
            for b in range(n_buffers):
                if free[b] < r["def_index"]:  # this buffer's occupant is dead before r is defined
                    slot = b
                    break
            if slot is None:
                slot = n_buffers
                n_buffers += 1
                free.append(r["last_use_index"])
            else:
                free[slot] = r["last_use_index"]
                reused += 1
            assignment[r["name"]] = slot
        return {"scratch_count": len(ranges), "buffer_count": n_buffers,
                "reused": reused, "assignment": assignment}

    def estimate(self):
        """Static memory-traffic + kernel-count estimate over the lowered IR (Spec 3 s28, ADC-465).
        A REPORT, not a transform. All figures are STRUCTURAL (counted off the IR / the lowering),
        in UNITS of one block-state field traversal (n_cons * n_cells * 8 bytes) -- the absolute byte
        count needs the runtime grid, so the estimate is grid-relative: a "field" is one full
        state-sized buffer pass. The measured GPU kernel count / wall time is a ROMEO profile; this is
        the host-side static prediction the GPU detectors threshold on.

        Returns a dict:
          - ``kernel_count``: total per-cell + heavy kernel launches (one per scratch-writing op, plus
            the elliptic / Krylov solves);
          - ``small_kernels``: per-cell kernels that touch only a handful of buffers (launch-overhead
            bound on a GPU);
          - ``heavy_kernels``: elliptic / Krylov solves (each many internal kernels);
          - ``scratch_count`` / ``buffer_count`` / ``buffers_saved``: from the buffer-reuse report;
          - ``field_reads`` / ``field_writes`` / ``traffic_fields``: field-sized buffer passes
            (reads + writes), the proxy for memory traffic."""
        reuse = self.buffer_reuse_report()
        kernel_count = small = heavy = 0
        reads = writes = 0
        for v in self._values:
            if v.op in Program._HEAVY_KERNEL_OPS:
                heavy += 1
                kernel_count += 1
                # An elliptic / Krylov solve reads the state and writes the shared aux/field: count one
                # read + one write as a coarse field-traffic proxy (the internal V-cycle traffic is
                # solver-dependent and out of scope for a structural estimate).
                reads += 1
                writes += 1
                continue
            if v.op in Program._PERCELL_KERNEL_OPS:
                kernel_count += 1
                # One write (its result scratch) + one read per distinct field input it consumes.
                in_fields = len({i.id for i in v.inputs if i.is_field()})
                # An affine combine also reads each of its terms.
                aff = v.attrs.get("coeffs")
                if aff is not None:
                    in_fields = max(in_fields, len(aff))
                writes += 1 if v.op in Program._SCRATCH_OPS else 0
                reads += in_fields
                # A "small" kernel touches few buffers (the GPU launch-overhead regime): a per-cell
                # source / where / cell_compare / a 1-2 term combine.
                if in_fields <= 2:
                    small += 1
        traffic = reads + writes
        return {"kernel_count": kernel_count, "small_kernels": small, "heavy_kernels": heavy,
                "scratch_count": reuse["scratch_count"], "buffer_count": reuse["buffer_count"],
                "buffers_saved": reuse["scratch_count"] - reuse["buffer_count"],
                "field_reads": reads, "field_writes": writes, "traffic_fields": traffic}

    # GPU heuristic thresholds (Spec 3 s28 detectors, ADC-465). A warning report, never a hard error:
    # a small step legitimately trips none; a pathological IR (a long chain of tiny per-cell kernels, a
    # buffer explosion, or heavy field traffic) trips one and the report names it. The numbers are
    # launch-overhead / occupancy rules of thumb, not a measured limit -- the MEASURED GPU kernel count
    # is a ROMEO profile (this is the host-side static prediction).
    _GPU_MAX_SMALL_KERNELS = 12
    _GPU_MAX_SCRATCHES = 16
    _GPU_MAX_TRAFFIC_FIELDS = 40

    def gpu_detectors(self):
        """Flag GPU anti-patterns from the static :meth:`estimate` (Spec 3 s28, ADC-465). Returns a
        list of warning dicts ``{"detector", "value", "threshold", "message"}`` -- NEVER raises (an
        analysis report, not a hard error). Detects too-many-small-kernels (launch overhead),
        too-many-scratches (buffer pressure / allocator churn) and excessive memory traffic
        (bandwidth bound). An empty list means the IR trips no host-side GPU heuristic; the measured
        kernel count / occupancy is validated on ROMEO."""
        est = self.estimate()
        warnings = []
        checks = (
            ("too_many_small_kernels", est["small_kernels"], Program._GPU_MAX_SMALL_KERNELS,
             "many small per-cell kernels: launch overhead may dominate on a GPU; consider fusing"),
            ("too_many_scratches", est["buffer_count"], Program._GPU_MAX_SCRATCHES,
             "many live scratch buffers: GPU memory pressure / allocator churn; consider reuse"),
            ("excessive_memory_traffic", est["traffic_fields"], Program._GPU_MAX_TRAFFIC_FIELDS,
             "high field-sized memory traffic: likely bandwidth bound on a GPU"),
        )
        for name, value, thresh, msg in checks:
            if value > thresh:
                warnings.append({"detector": name, "value": value, "threshold": thresh,
                                 "message": msg})
        return warnings

    def estimate_report(self):
        """A human-readable cost report: the static memory-traffic / kernel-count :meth:`estimate`,
        the buffer-reuse summary and any GPU detector warnings (Spec 3 s28 inspection surface,
        ADC-465). A report only -- ``self`` is never mutated."""
        est = self.estimate()
        lines = ["# cost estimate for Program %r (static, grid-relative)" % self.name,
                 "  kernels        : %d (%d small, %d heavy elliptic/Krylov)"
                 % (est["kernel_count"], est["small_kernels"], est["heavy_kernels"]),
                 "  scratch buffers: %d allocated, %d after reuse (%d saved)"
                 % (est["scratch_count"], est["buffer_count"], est["buffers_saved"]),
                 "  memory traffic : %d field-passes (%d read, %d write)"
                 % (est["traffic_fields"], est["field_reads"], est["field_writes"])]
        warnings = self.gpu_detectors()
        if warnings:
            lines.append("  GPU detectors  :")
            for w in warnings:
                lines.append("    [warn] %s (%d > %d): %s"
                             % (w["detector"], w["value"], w["threshold"], w["message"]))
        else:
            lines.append("  GPU detectors  : none tripped (host-side heuristic)")
        return "\n".join(lines)

    def validate(self):
        """Structural validation of the IR. Raises ValueError on a malformed program."""
        if not self._commits:
            raise ValueError("a time Program must commit each advanced block exactly once "
                             "(no block was committed)")
        seen = set()
        for v in self._values:
            for inp in v.inputs:
                if inp.id not in seen:
                    raise ValueError("IR value '%s' used before definition" % inp.name)
            seen.add(v.id)
            if v.op == "while":
                self._validate_block(v.attrs["cond_block"], seen)
                self._validate_block(v.attrs["body_block"], seen)
            elif v.op in ("range", "if"):
                self._validate_block(v.attrs["body_block"], seen)
            elif v.op == "matrix_free_operator" and v.attrs.get("apply_block"):
                # The apply sub-block is self-contained (its in/out placeholders + scratch are defined
                # inside it); it reads nothing from the enclosing scope.
                self._validate_block(v.attrs["apply_block"], seen)
            elif v.op == "solve_local_nonlinear":
                # The residual sub-block is self-contained: the iterate / guess State placeholders are
                # defined inside it (first ops) and every op reads only the placeholders or earlier
                # sub-block ops. Validate against an EMPTY outer scope so a residual that closes over an
                # enclosing value (which the per-cell kernel cannot evaluate) fails loud here, not as a
                # codegen KeyError.
                self._validate_block(v.attrs["residual_block"], set())
        return True

    def _validate_block(self, block, outer_seen):
        """Validate a control-flow sub-block: each op may read values defined earlier in the SAME block
        or in the enclosing scope (the loop variable / anything defined before the while). @p outer_seen
        is the enclosing scope's def set (copied, not mutated -- the sub-block ops are not visible
        outside)."""
        seen = set(outer_seen)
        for v in block:
            for inp in v.inputs:
                if inp.id not in seen:
                    raise ValueError("IR value '%s' used before definition" % inp.name)
            seen.add(v.id)

    # --- serialization / hash ---
    @staticmethod
    def _serialize_node(v):
        attrs = dict(v.attrs)
        if "schedule" in attrs:  # an authoring annotation: serialize its repr so the IR hash is
            attrs["schedule"] = repr(attrs["schedule"])  # schedule-sensitive yet JSON-safe
        if "coeffs" in attrs:  # dict keys (powers) -> sorted [power, value] for stable JSON
            attrs["coeffs"] = [sorted((int(p), c) for p, c in d.items()) for d in attrs["coeffs"]]
        if "a_coeff" in attrs:  # solve_local_linear: the dt-polynomial a in (I - a*L)
            attrs["a_coeff"] = sorted((int(p), c) for p, c in attrs["a_coeff"].items())
        if "c_dt" in attrs:  # rhs_jacvec: the dt-polynomial BDF coefficient c*dt
            attrs["c_dt"] = sorted((int(p), c) for p, c in attrs["c_dt"].items())
        if v.op == "while":  # the cond/body sub-blocks are nested node lists; the results are ids
            attrs["cond_block"] = [Program._serialize_node(w) for w in attrs["cond_block"]]
            attrs["body_block"] = [Program._serialize_node(w) for w in attrs["body_block"]]
            attrs["cond"] = attrs["cond"].id
            attrs["body"] = attrs["body"].id
        elif v.op in ("range", "if"):  # body sub-block (range carries its int count in attrs too)
            attrs["body_block"] = [Program._serialize_node(w) for w in attrs["body_block"]]
            attrs["body"] = attrs["body"].id
        elif v.op == "matrix_free_operator":  # the apply sub-block is a nested node list; refs are ids
            attrs["apply_block"] = ([Program._serialize_node(w) for w in attrs["apply_block"]]
                                    if attrs.get("apply_block") else None)
            for k in ("apply_result", "apply_in", "apply_out"):
                ref = attrs.get(k)
                attrs[k] = (_affine_ids(ref) if isinstance(ref, _Affine)
                            else (ref.id if isinstance(ref, Value) else None))
        elif v.op == "solve_local_nonlinear":  # the residual sub-block is a nested node list; refs ids
            attrs["residual_block"] = [Program._serialize_node(w) for w in attrs["residual_block"]]
            for k in ("residual", "iterate", "guess"):
                attrs[k] = attrs[k].id
        return {"id": v.id, "vtype": v.vtype, "op": v.op, "block": v.block,
                "inputs": [i.id for i in v.inputs], "attrs": attrs}

    def _serialize(self):
        nodes = [Program._serialize_node(v) for v in self._values]
        commits = sorted((b, s.id) for b, s in self._commits.items())
        # NAME-based block binding (Spec 3 criterion 23, ADC-457): the block names in P.state
        # declaration order are part of the IR identity -- the .so exports them (adc_program_block_name)
        # and install_program binds System blocks to them BY NAME. Reordering P.state changes this list,
        # so two Programs differing only by block order get distinct IR hashes (and distinct .so caches).
        _order = self._block_indices()
        block_order = sorted(_order, key=_order.get)
        out = {"name": self.name, "version": 1, "nodes": nodes, "commits": commits,
               "block_order": block_order}
        # The optional dt bound (spec s18 / ADC-417) is part of the IR identity: its presence and its
        # scalar sub-program feed the hash (the compiled-problem cache key) so two Programs differing
        # only by a dt bound get distinct .so caches.
        if self._dt_bound is not None:
            sub, result = self._dt_bound
            out["dt_bound"] = {"nodes": [Program._serialize_node(w) for w in sub],
                               "result": result.id}
        return out

    def _ir_hash(self):
        """Stable SHA-256 of the IR (feeds the compiled-problem cache key in a later phase)."""
        blob = json.dumps(self._serialize(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    # --- C++ codegen (Phase 2c-ii / Phase 4b): lower the IR to a problem.so source ---
    def emit_cpp_program(self, model=None):
        """Generate the C++ source of a problem.so implementing this Program (codegen).

        Exports the stable .so ABI -- ``adc_program_abi_key`` (the ``ADC_ABI_KEY_LITERAL``
        preprocessor literal, NOT the interposable inline), ``adc_program_name``, ``adc_program_hash``,
        ``adc_install_program`` -- and installs the macro step as a closure built from `ProgramContext`
        primitives only (no MultiFab / flux / solver reimplementation). It is the source the C++ loader
        (`System::install_program`) compiles, dlopens, and runs.

        Lowers the Program by a topological walk of the SSA IR: each block's current state is its base
        (``ctx.state(idx)``); ``solve_fields()`` runs the elliptic solve; each RHS becomes a
        scratch + ``rhs_into``; each intermediate ``linear_combine`` becomes a zero scratch accumulated
        with ``axpy``; the committed combine writes the block state via ``lincomb``. Forward Euler,
        SSPRK2/SSPRK3 and RK4 all lower this way -- no per-scheme class.

        Multi-block (ADC-426): N ``P.state(\"a\")`` / ``P.state(\"b\")`` declarations + N ``P.commit``
        are lowered -- each op routes to its own block's runtime index (``_block_indices``, in the order
        the blocks are first declared via ``P.state``). The .so also exports its block NAMES in that
        order (``adc_program_block_count`` / ``adc_program_block_name``); ``System::install_program``
        binds them to the instantiated System blocks BY NAME (Spec 3 criterion 23, ADC-457), so the
        System blocks (``sim.add_equation`` / ``sim.add_block``) may be added in ANY order -- a Program
        block whose name has no instantiated System block fails loud (``Program requires block instance
        '<name>', but simulation did not instantiate it``). A block declared but never committed is a
        READ-ONLY block (allowed; e.g. a passive field whose charge couples the others through the shared
        Poisson). A commit of a block no ``P.state`` declares is rejected. A single-block Program lowers
        byte-identically (its one block is index 0; an order-matching multi-block Program too -- the
        name map is the identity).

        Phase-4b also lowers the SPLIT-SOURCE / LOCAL-LINEAR ops -- ``source`` (a named ``m.source_term``
        evaluated per cell), ``apply`` (LU for a named ``m.linear_source``) and ``solve_local_linear``
        ((I -/+ a*L) U = rhs solved cell by cell via a dense per-cell inverse) -- but ONLY when the
        physical ``model`` (the ``adc.dsl`` model whose ``source_term`` / ``linear_source`` they name)
        is provided: the codegen reads the model's symbolic coefficients to emit the per-cell kernels.
        Without ``model`` those ops raise NotImplementedError (the Program cannot be lowered in
        isolation); ``model=None`` still lowers FE / SSPRK / RK4 (no model needed). A ``rhs`` routes its
        base on its ``flux`` flag and whether ``"default"`` is among the requested ``sources`` (ADC-425 /
        ADC-430, spec criterion 17 -- flux and sources are explicit, never summed implicitly). With
        ``flux=True``: ``"default"`` present -> ``ctx.rhs_into`` (= ``-div F`` + the model's
        default/composite source, the historical path); ``"default"`` absent (incl. the empty list
        ``[]``) -> ``ctx.neg_div_flux_default_into`` (= ``-div F`` only, NO default source). With
        ``flux=False`` (SOURCE-ONLY, ADC-430): NO ``-div F`` base -- ``"default"`` present (or ``None``)
        -> ``ctx.source_default_into`` (= S only, the exact mirror); ``"default"`` absent -> the zeroed
        scratch (the named sources, if any, are the whole RHS). Each NAMED source (``sources=[...]``
        beyond ``"default"``) then lowers with a model: the same per-cell ``m.source_term`` kernel as the
        standalone ``source`` op, accumulated onto ``R`` via ``axpy``. So ``flux=True,sources=[]`` is flux
        only, ``flux=True,sources=["default"]`` is flux + default source (unchanged),
        ``flux=False,sources=["default"]`` is the default source only, ``flux=False,sources=["s"]`` is
        just ``s`` -- the named ones never double-count the default (it is folded in iff "default" was
        listed). More than one block now lowers (ADC-426): each op routes to its block's runtime index
        (``_block_indices``, in P.state declaration order) and control flow (while/range/if) inside a
        block lowers per block; a SIMULTANEOUS multi-target coupled field solve
        (``solve_fields_from_blocks([Ua, Ub])``) lowers to ``ctx.solve_fields_from_blocks`` (see below).

        Each ``solve_fields(state=...)`` op lowers to ``ctx.solve_fields_from_state(idx, <stage state>)``
        (ADC-409): the elliptic fields are re-solved -- and the shared aux re-filled -- from THAT stage's
        state, not the block's current state. So a field-coupled multi-stage scheme (Poisson feedback
        into the flux) is exact: stage k's RHS reads phi solved from stage k's own state. For the first
        stage the stage state is U^n, so this is identical to the historical ``solve_fields()``; for an
        uncoupled model the field solve is inert either way. This is already a COUPLED multi-block solve:
        the system Poisson RHS is ``Sum_s elliptic_rhs_s(U_s)`` (``assemble_poisson_rhs``), so block
        ``idx`` reads its stage state while every OTHER block contributes its LIVE state into the one
        shared phi/aux. A per-block ``P.solve_fields(state=Ub)`` therefore sees all blocks' charge. A
        SIMULTANEOUS multi-target override (several blocks at their stage states in ONE solve) lowers to
        ``ctx.solve_fields_from_blocks(<vec>)`` (Spec 3 criterion 24, ADC-457): the RHS is
        ``Sum_s elliptic_rhs_s(U_s)`` reading EVERY listed block's stage state at once
        (``assemble_poisson_rhs_from_blocks``), each slotted at its block index (nullptr = the block's
        live state) -- the coupled multi-species field solve."""
        self.validate()
        self._check_lowerable(model)
        prelude, body = self._emit_body(model)
        # Optional dt bound (spec s18 / ADC-417): emit the SECOND ABI pair -- adc_program_has_dt_bound()
        # (true iff a bound was set) and adc_program_dt_bound(ProgramContext*, cfl) (the lowered scalar
        # expression). Without a bound, has_dt_bound() returns false and the dt_bound function returns a
        # +inf sentinel (never reached: the loader stores the closure only when has_dt_bound() is true).
        has_dt_bound, dt_bound_body = self._emit_dt_bound(model)
        return _PROGRAM_CPP_TEMPLATE.format(
            name=json.dumps(self.name), hash=self._ir_hash(), prelude=prelude, body=body,
            has_dt_bound=has_dt_bound, dt_bound_body=dt_bound_body,
            module_metadata=self._emit_module_metadata(model),
            block_names=self._emit_block_names())

    def _emit_block_names(self):
        """C++ source of the NAME-based block-binding ABI the .so exports (Spec 3 criterion 23, ADC-457):
        ``adc_program_block_count()`` and ``adc_program_block_name(int)`` -- the Program's block names in
        ``_block_indices`` order (P.state declaration order, the order the step body's ``ctx.state(idx)``
        addresses). System::install_program reads them, matches each to the instantiated System block of
        that name, and stores the program-index -> system-index map (read by ProgramContext), so the
        System blocks may be added in ANY order vs the Program's P.state declarations -- a Program block
        whose name has no System block fails loud. The block names are also part of the IR identity (the
        block_order field of _serialize feeds the IR hash), so reordering P.state changes the hash."""
        order = self._block_indices()  # name -> index, declaration order
        names = sorted(order, key=order.get)
        cases = "".join('    case %d: return %s;\n' % (order[nm], json.dumps(nm)) for nm in names)
        return (
            "// NAME-based block binding (Spec 3 criterion 23, ADC-457): the Program's block names in\n"
            "// P.state declaration order. install_program matches each to a System block BY NAME (not\n"
            "// add-order) and builds the program-index -> system-index map ProgramContext resolves.\n"
            'extern "C" int adc_program_block_count() { return %d; }\n' % len(names) +
            'extern "C" const char* adc_program_block_name(int i) {\n'
            '  switch (i) {\n%s    default: return "";\n  }\n}\n' % cases)

    def _emit_module_metadata(self, model=None):
        """C++ source of the GeneratedModule metadata the .so exports (Spec 2 / ADC-442).

        A combined model+program .so carries, alongside ``GeneratedProgram`` (the step), a
        ``GeneratedModule`` descriptor: ``extern "C"`` accessors exposing the typed operator registry
        -- a count and, per integer ``OperatorId`` (the array index), the operator name / kind /
        signature / requirements -- plus the state and field space names. These are read ONCE at
        install (introspection + requirement validation, ``module_metadata.hpp``); the step body never
        calls them, so operators stay inlined and there is no string lookup in any hot kernel.
        ``model=None`` emits an empty module (count 0). The metadata is derived from the model's typed
        registry, so it does not perturb the program IR hash.
        """
        ops, states, fields = [], [], []
        if model is not None and hasattr(model, "operator_registry"):
            reg = model.operator_registry()
            ops = [reg.get(nm) for nm in reg.names()]
            if hasattr(model, "state_space"):
                states = [model.state_space().name]
            if hasattr(model, "field_space"):
                fields = [model.field_space().name]

        def table(accessor, values):
            cases = "".join('    case %d: return %s;\n' % (i, json.dumps(v))
                            for i, v in enumerate(values))
            return ('extern "C" const char* adc_module_%s(int i) {\n'
                    '  switch (i) {\n%s    default: return "";\n  }\n}\n' % (accessor, cases))

        def req_json(op):
            # The operator's own kind always wins (a requirements dict must not shadow it).
            return json.dumps({**op.requirements, "kind": op.kind})

        parts = [
            "// GeneratedModule metadata (Spec 2 / ADC-442): the typed operator registry exposed by\n"
            "// the .so for introspection + install-time validation. OperatorId = the array index.\n"
            "// NOT called from any hot kernel -- operators are inlined at codegen.\n",
            'extern "C" int adc_module_operator_count() { return %d; }\n' % len(ops),
            'extern "C" int adc_module_state_space_count() { return %d; }\n' % len(states),
            'extern "C" int adc_module_field_space_count() { return %d; }\n' % len(fields),
            table("operator_name", [op.name for op in ops]),
            table("operator_kind", [op.kind for op in ops]),
            table("operator_signature", [repr(op.signature) for op in ops]),
            table("operator_requirements", [req_json(op) for op in ops]),
            table("state_space_name", states),
            table("field_space_name", fields),
        ]
        return "".join(parts)

    def _emit_dt_bound(self, model=None):
        """Lower the optional dt bound (spec s18 / ADC-417) to ``(has_dt_bound, body)``: the bool literal
        adc_program_has_dt_bound returns and the C++ body of adc_program_dt_bound. No bound -> ("false",
        a +inf return that is never reached). The bound is a READ-ONLY scalar sub-program: it reuses the
        same per-op lowering (state -> ctx.state(idx), reductions, cfl/hmin/max_wave_speed, scalar_op) and
        returns the final scalar. ADC-426: a multi-block dt bound may read several blocks' states (e.g.
        the min over blocks of cfl*hmin/max_wave_speed), so each op resolves its OWN block index / base.
        No commit lives in a dt bound (empty committed_ids)."""
        if self._dt_bound is None:
            return "false", "    return std::numeric_limits<adc::Real>::infinity();"
        sub, result = self._dt_bound
        block_idx = self._block_indices()
        bases = {}
        for v in sub:
            if v.op == "state" and v.block not in bases:
                bases[v.block] = v
        var = {}
        lines = []
        for v in sub:
            self._emit_op(v, bases.get(v.block), frozenset(), var, model, lines, None, block_idx)
        lines.append("return %s;" % var[result.id])
        body = "\n".join("    " + ln for ln in lines)
        return "true", body

    # Ops the Phase-4b codegen lowers ONLY when a physical model is supplied (they read the model's
    # symbolic source_term / linear_source coefficients). Without a model they raise NotImplementedError.
    _MODEL_OPS = ("source", "apply", "solve_local_linear", "solve_local_nonlinear")

    def _block_indices(self):
        """Map each block name to a stable runtime block index, in the order the Program FIRST declares
        it via ``P.state(...)`` (ADC-426). Index 0 is the first declared block, 1 the second, ...; the
        single-block program keeps index 0 (byte-identical lowering). The generated ``.so`` addresses
        blocks by this index in its step body AND exports the block NAMES in this order
        (``adc_program_block_name``); ``System::install_program`` binds them to the instantiated System
        blocks BY NAME (Spec 3 criterion 23, ADC-457), so the System block add-order need NOT match the
        Program's ``P.state`` order -- the historical positional convention is the identity special case
        (names already in add-order)."""
        order = {}
        for v in self._values:
            if v.op == "state" and v.block not in order:
                order[v.block] = len(order)
        return order

    def _check_lowerable(self, model=None):
        """Raise NotImplementedError if the IR uses a construct the current codegen cannot lower yet,
        naming the offending construct (never a silent mis-lowering). @p model: the physical model that
        declares the named sources / linear sources; required for the Phase-4b ops.

        Multi-block (ADC-426): N ``P.state`` blocks + N ``P.commit`` are supported -- each op routes to
        its block's index (``_block_indices``). Validation: a block is committed AT MOST once (enforced
        at ``commit`` time); a read-only block (declared via ``P.state`` but never committed) is allowed
        (e.g. a passive field whose charge couples the others); a commit of a block that was never
        declared by ``P.state`` is rejected (an unknown-block commit cannot route to an index)."""
        blocks = self._block_indices()
        for b in self._commits:
            if b not in blocks:
                raise ValueError(
                    "commit of unknown block '%s': no P.state('%s') declares it (declared blocks: %s)"
                    % (b, b, sorted(blocks)))
        self._check_schedules_lowerable()
        for v in self._values:
            self._check_op_lowerable(v, model)
        # Per-cell dense fallback bound for the local dense solves (mat_inverse<N> uses fixed stack
        # buffers): solve_local_linear (M = I - a*L) and solve_local_nonlinear (the Newton FD Jacobian).
        dense_ops = ("solve_local_linear", "solve_local_nonlinear")
        if model is not None and any(v.op in dense_ops for v in self._all_ops()):
            impl = _model_impl(model)
            n_cons = len(getattr(impl, "cons_names", []) or [])
            if n_cons > 8:
                raise ValueError(
                    "local dense fallback currently supports n_cons <= 8 (got %d)" % n_cons)

    def _check_schedules_lowerable(self):
        """Gate the unified Program scheduler lowering (ADC-458, Spec 3 sections 17-18). EVERY kind/policy
        now lowers (``_emit_schedule_wrap``) EXCEPT the two that need a runtime primitive the compiled
        .so does not have, which still fail loud (never a silent no-op):

          - ``on_end()``: a compiled ``sim.step(dt)`` loop carries no end-of-run signal, so the .so cannot
            know which step is the last. (Use an on_end host hook instead.)
          - ``when(cond)`` whose cond is a bare Python callable, not a Program Bool predicate: a callable
            is not a Program value and cannot be lowered to C++.

        The cadence RUNTIME (the cache cadence in a stepping .so) is exercised on ROMEO; the cache
        MANAGER is unit-tested by tests/test_cache_manager.cpp."""
        for v in self._all_ops():
            sched = v.attrs.get("schedule")
            if sched is None or sched.is_always():
                continue
            if sched.kind == "on_end":
                raise NotImplementedError(
                    "schedule on_end() on node %r (op '%s') is not lowerable: a compiled sim.step(dt) "
                    "loop never sees an end-of-run signal, so the .so cannot know the last step. Use "
                    "on_start()/every()/when()/subcycle(), or an on_end host hook (ADC-458)."
                    % (v.name, v.op))
            if sched.kind == "when":
                cond = sched.params.get("cond")
                if not (isinstance(cond, Value) and cond.vtype == "bool"):
                    raise NotImplementedError(
                        "schedule when(cond) on node %r lowers only a Program Bool predicate (e.g. "
                        "P.norm2(r) < tol), not a Python callable (ADC-458)." % v.name)
            if sched.kind == "subcycle" and v.op not in Program._AUX_OUTPUT_OPS:
                # subcycle re-runs the body COUNT times in a for-loop scope. A node whose output is a
                # step-body scratch (rhs / source / linear_combine / ...) would declare that scratch
                # INSIDE the loop, leaving it out of scope for any downstream consumer -- broken C++. Only
                # an aux-output op (a field solve, which writes the persistent System aux) is well-defined
                # under sub-cycling; a scratch sub-step has no single 'result' to consume. Fail loud.
                raise NotImplementedError(
                    "schedule subcycle on node %r (op '%s') is lowerable only for a field solve (its "
                    "output is the persistent System aux); a scratch-output op sub-cycled has no single "
                    "result a downstream node can read (ADC-458). Sub-cycle the field solve, or express "
                    "the inner steps explicitly." % (v.name, v.op))

    # 'linear_source' is a pure NAME-reference SSA node (vtype 'operator'): it carries no runtime work
    # (consumed by apply / solve_local_linear, which read the model coefficients), so it lowers to
    # nothing -- always allowed, model or not. 'reduce' / 'compare' / 'while' are the ADC-404a control
    # flow / reduction ops (lowered inline via adc::dot; no model needed). 'matrix_free_operator' /
    # 'scalar_field' / 'laplacian' / 'gradient' / 'divergence' / 'solve_linear' are the ADC-405 / ADC-412
    # matrix-free Krylov ops (the operator declaration carries an apply sub-block; solve_linear lowers to
    # adc::*_solve; divergence is the centered FV divergence of a gradient field).
    _ALLOWED_OPS = frozenset({"state", "solve_fields", "solve_fields_from_blocks", "rhs",
                              "linear_combine", "linear_source",
                              "reduce", "compare", "while", "range", "if", "matrix_free_operator",
                              "scalar_field", "laplacian", "gradient", "divergence", "solve_linear",
                              "apply_in", "apply_out", "history", "store_history",
                              "fill_boundary", "project", "record_scalar",
                              "cell_compare", "where", "rhs_jacvec",
                              "schur_coeffs", "apply_laplacian_coeff", "schur_explicit_flux",
                              "schur_rhs", "schur_reconstruct", "schur_energy",
                              "coupled_rate", "coupled_rate_out"})

    # Ops NOT wrapped in a per-node profile scope (ADC-459): they bind a reference or read a cached
    # scalar and do no per-step numerical work, so timing them only adds always-zero noise to
    # sim.profile_report(). Every other op that emits a statement is wrapped (rhs / solve_fields /
    # linear_combine / source / apply / reductions / loops / Schur kernels / ...).
    _PROFILE_SKIP_OPS = frozenset({"state", "history", "hmin", "cfl"})

    def _all_ops(self):
        """Iterate over every op of the Program, descending into control-flow + apply sub-blocks (a flat
        view used by the lowerability guards: the sub-block ops are not in self._values). Nested control
        flow is disallowed, so the sub-blocks are flat (one level)."""
        for v in self._values:
            yield v
            for key in ("cond_block", "body_block", "apply_block", "residual_block"):
                blk = v.attrs.get(key)
                if isinstance(blk, list):
                    yield from blk

    def _check_op_lowerable(self, v, model):
        """Lowerability check for a single op (used for both the top-level walk and a while sub-block).
        Raises NotImplementedError / ValueError naming the offending construct (never a mis-lowering)."""
        if v.op in Program._MODEL_OPS:
            if model is None:
                raise NotImplementedError(
                    "emit_cpp_program cannot lower op '%s' (value '%s') without the physical model "
                    "that declares its named source / linear source; pass model= "
                    "(compile_problem threads it through)" % (v.op, v.name))
            if v.op == "solve_local_nonlinear":  # recurse: the residual sub-block ops must lower too
                for w in v.attrs["residual_block"]:
                    self._check_op_lowerable(w, model)
            return  # _emit_op lowers it from the model's symbolic coefficients
        if v.op not in Program._ALLOWED_OPS:
            raise NotImplementedError(
                "emit_cpp_program cannot lower op '%s' (value '%s') yet; supported ops are %s "
                "(+ %s with a model; nested control flow / Krylov are later phases)"
                % (v.op, v.name, sorted(Program._ALLOWED_OPS), sorted(Program._MODEL_OPS)))
        if v.op == "coupled_rate":
            # A coupled_rate (collisions / ionization, Spec 3 criterion 27) lowers to ONE multi-state
            # for_each_cell kernel (see _emit_coupled_rate_kernel). The lowering reaches the operator
            # body (its per-block component formulas) through the BOUND registry, and binds each input
            # state's cons names from that input's StateSpace -- so the operator must be bound and the
            # formulas must be cons-only (the MVP). Validate both here so a non-lowerable coupled_rate
            # fails loud naming ADC-457, never emits an undefined reference.
            self._coupled_rate_components(v)
            return
        if v.op == "coupled_rate_out":
            # A pure projection of one block out of the coupled bundle: it emits nothing (its var
            # aliases that block's rate scratch). Lowerable iff its producing coupled_rate is (checked
            # when that node is walked); nothing to validate here.
            return
        if v.op in ("while", "range", "if"):  # recurse: the cond / body sub-blocks must lower too
            for key in ("cond_block", "body_block"):
                for w in v.attrs.get(key, []):
                    self._check_op_lowerable(w, model)
            return
        if v.op == "matrix_free_operator":  # recurse into the apply sub-block (set by set_apply)
            if v.attrs.get("apply_block") is None:
                raise ValueError(
                    "matrix_free_operator '%s' has no apply; call P.set_apply before lowering"
                    % v.name)
            for w in v.attrs["apply_block"]:
                self._check_op_lowerable(w, model)
            return
        if v.op == "solve_fields":
            # A NAMED elliptic field (ADC-419/ADC-428) drives a SECOND elliptic solve into its own aux
            # channel. The runtime now hosts it (System::solve_fields_from_state(field, ...) via
            # ProgramContext); lowering needs the model so the field name can be validated against the
            # declared m.elliptic_field set (the codegen emits the named ctx call).
            field = v.attrs.get("field")
            if field is not None:
                if model is None:
                    raise NotImplementedError(
                        "emit_cpp_program cannot lower solve_fields with a named elliptic field "
                        "('%s') without the physical model that declares it (m.elliptic_field); pass "
                        "model= (compile_problem threads it through)" % field)
                if field not in _model_impl(model)._elliptic_fields:
                    raise ValueError(
                        "unknown elliptic_field '%s' in solve_fields '%s'; declared: %s"
                        % (field, v.name, sorted(_model_impl(model)._elliptic_fields)))
            return
        if v.op == "rhs":
            named_fluxes = _named_fluxes(v)
            # ADC-430: flux=False is SOURCE-ONLY -- no -div F base. Named fluxes (a -div of selected
            # flux_terms) contradict "no flux": reject the combination loud rather than silently picking
            # one (request flux=True for named fluxes, or flux=False for a source-only stage).
            if not v.attrs.get("flux", True) and named_fluxes is not None:
                raise ValueError(
                    "rhs '%s' sets flux=False (source-only) but also requests named fluxes %r; a "
                    "source-only stage has no flux divergence -- drop fluxes= or set flux=True"
                    % (v.name, named_fluxes))
            if named_fluxes is not None:  # NAMED fluxes (ADC-419): need the model's flux_term coeffs
                if model is None:
                    raise NotImplementedError(
                        "emit_cpp_program cannot lower rhs '%s' with named fluxes %r without the "
                        "physical model that declares them (m.flux_term); pass model= "
                        "(compile_problem threads it through)" % (v.name, named_fluxes))
                impl_f = _model_impl(model)
                ft = impl_f._flux_terms
                for f in named_fluxes:
                    if f not in ft:
                        raise ValueError(
                            "unknown flux_term '%s' in rhs '%s'; declared flux_terms: %s"
                            % (f, v.name, sorted(ft)))
                # The named-flux path emits -div(selected fluxes) only (no ctx.rhs_into), so the model's
                # DEFAULT source would be silently dropped -- reject it (it must be requested as a named
                # source_term instead). The named sources below are still axpy'd on top.
                if getattr(impl_f, "_source", None):
                    raise NotImplementedError(
                        "rhs with named fluxes %r needs a model whose default source is empty (no "
                        "m.source); rhs '%s' has a non-empty default source that the named-flux path "
                        "would drop (declare it as a source_term instead)" % (named_fluxes, v.name))
            extra = [s for s in (v.attrs.get("sources") or []) if s != "default"]
            if not extra:
                return
            # A named source in an rhs reads the model's symbolic source_term coefficients (same as the
            # standalone 'source' op): lowering needs the model.
            if model is None:
                raise NotImplementedError(
                    "emit_cpp_program cannot lower rhs '%s' with named sources %r without the "
                    "physical model that declares them (m.source_term); pass model= "
                    "(compile_problem threads it through)" % (v.name, extra))
            impl = _model_impl(model)
            # ADC-425: the named sources are axpy'd on top of an EXPLICIT base. With "default" requested
            # the base is ctx.rhs_into (flux + the model's default/composite source); without it the base
            # is ctx.neg_div_flux_default_into (flux only). Either way the default source is folded in iff
            # the caller listed "default", so adding distinct named source_terms cannot double-count it --
            # the old "model default source must be empty" rejection is gone (the routing is now exact).
            for s in extra:
                if s not in impl._source_terms:
                    raise ValueError(
                        "unknown source_term '%s' in rhs '%s'; declared source_terms: %s"
                        % (s, v.name, sorted(impl._source_terms)))

    def _coupled_rate_components(self, v):
        """Resolve a ``coupled_rate`` node @p v to its per-block component formulas (Spec 3 criterion
        27, ADC-457), validated for the cons-only MVP. Returns ``{block: [Expr, ...]}`` (one formula
        per component of that block's StateSpace).

        The component formulas live in the BOUND operator's body (``op.body`` = the ``expr=`` dict
        passed to ``Module.operator``), reachable through the registry the node's ``operator`` attr
        names; the input states' cons names come from each input value's StateSpace (set by
        ``P.state(space=...)``). Raises a clear NotImplementedError naming ADC-457 when a coupled_rate
        cannot lower in this MVP: no bound registry, no operator body, a block whose component count
        does not match its StateSpace, or a formula referencing a non-cons (prim / aux) Var."""
        from . import dsl
        op_name = v.attrs["operator"]
        if self._registry is None:
            raise NotImplementedError(
                "the coupled_rate kernel codegen (ADC-457) needs the bound operator registry to reach "
                "operator %r's component formulas; call P.bind_operators(module) before emitting "
                "(node %r)" % (op_name, v.name))
        op = self._registry.get(op_name)
        expr = op.body
        if not isinstance(expr, dict):
            raise NotImplementedError(
                "the coupled_rate kernel codegen (ADC-457) needs operator %r to carry its per-block "
                "component formulas as an expr={block: [Expr, ...]} dict (got %r); a decorator-body "
                "coupled_rate is a later phase (node %r)" % (op_name, type(expr).__name__, v.name))
        # Each coupled_rate_out block must own one input state (its rate scratch is shaped like that
        # block's state) whose StateSpace gives the component count + cons names.
        by_block = {s.block: s for s in v.inputs}
        components = {}
        for blk, comps in expr.items():
            state_in = by_block.get(blk)
            if state_in is None or getattr(state_in, "space", None) is None:
                raise NotImplementedError(
                    "the coupled_rate kernel codegen (ADC-457) needs every output block to map to an "
                    "input State declared with a StateSpace (P.state(%r, space=...)); operator %r "
                    "block %r has none (node %r)" % (blk, op_name, blk, v.name))
            ncons = len(state_in.space.components)
            if len(comps) != ncons:
                raise NotImplementedError(
                    "coupled_rate operator %r block %r emits %d component formulas but its StateSpace "
                    "has %d components; the rate must be full-rank over the block state (ADC-457, "
                    "node %r)" % (op_name, blk, len(comps), ncons, v.name))
            for e in comps:
                for node in self._walk_expr(e):
                    if isinstance(node, dsl.Var) and node.kind != "cons":
                        raise NotImplementedError(
                            "coupled_rate formulas referencing prim/aux vars are deferred (ADC-457): "
                            "operator %r block %r references %s var %r; the MVP per-cell binding is "
                            "cons-only (node %r)" % (op_name, blk, node.kind, node.name, v.name))
            components[blk] = list(comps)
        # Every cons var a formula references must be a component of SOME input state (an output block
        # OR a read-only catalyst input). A name in no input state -- a typo, or a name the author
        # forgot to add to a P.state(space=...) -- would emit an undefined C++ identifier that only
        # fails at the AOT compile, far from the authoring site; reject it loud here, like prim/aux.
        all_cons = {c for s in v.inputs if getattr(s, "space", None) is not None
                    for c in s.space.components}
        referenced = set()
        for comps in components.values():
            for e in comps:
                referenced |= e.deps()
        missing = referenced - all_cons
        if missing:
            raise NotImplementedError(
                "coupled_rate operator %r references cons var(s) %s that are a component of no input "
                "state; declare them via P.state(space=...) or fix the formula (ADC-457, node %r)"
                % (op_name, sorted(missing), v.name))
        return components

    @staticmethod
    def _walk_expr(e):
        """Yield every node of a dsl Expr tree (used to scan a coupled_rate formula for non-cons Vars)."""
        from . import dsl
        stack = [e]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(dsl._children(node))

    def _emit_body(self, model=None):
        """Generate the C++ of the install function in TWO phases (each list indented uniformly by the
        template). Assumes `_check_lowerable` has passed. @p model supplies the symbolic coefficients of
        the Phase-4b source / apply / solve_local_linear ops. Returns ``(prelude, body)``:

          - ``prelude``: INSTALL-TIME C++ (before ``ctx.install``) -- persistent scratch fields (held
            via ``std::shared_ptr`` so they outlive the install call and are reused across every step
            and every Krylov iteration) and the matrix-free apply lambdas. Captured by value into the
            step closure (shared_ptr / lambda / ctx all copy cheaply).
          - ``body``: the STEP closure body (one macro-step over dt).

        Multi-block (ADC-426): the SSA walk allocates a per-block base (``ctx.state(idx)`` for each
        declared block) and routes every op to ITS block's index via ``_block_indices`` / ``v.block``.
        Each committed block's final value is copied into that block's state (a scratch commit) or was
        written in place (a linear_combine commit). A single block reduces to the historical lowering."""
        block_idx = self._block_indices()
        # The first-declared state Value per block: the "base" any op of that block clones / commits into.
        bases = {}
        for v in self._values:
            if v.op == "state" and v.block not in bases:
                bases[v.block] = v
        # IR value id -> C++ token: a MultiFab variable name (states / RHS scratches), a scalar variable
        # name (reductions, ``s{id}``) or a parenthesized boolean expression (compares).
        var = {}
        prelude = []
        lines = []
        # Bool-predicate value id -> its C++ token, for a when(cond) schedule whose cond is a Program
        # compare value emitted earlier in the body (ADC-458). Reset per emit (tokens are body-local).
        self._when_tokens = {}
        committed_ids = {s.id for s in self._commits.values()}
        # Multistep histories (ADC-406a): register each declared history at its MAX lag FIRST (a
        # registration-only call, NOT a read -- a read before the first store fails loud), so the ring
        # depth is locked before any store. The first ctx.store_history then cold-start-fills every
        # (already-allocated) slot -- step 0 reads the same value at every lag and the scheme degenerates
        # to a one-step method. register_history is idempotent (no-op once registered).
        for name, lag in sorted(self._histories.items()):
            lines.append("ctx.register_history(%s, %d);" % (json.dumps(name), int(lag)))
        for v in self._values:
            base = bases.get(v.block)  # the block-state value of THIS op's block (None: a scalar op)
            self._emit_op(v, base, committed_ids, var, model, lines, prelude, block_idx)
        # Each committed block: a scratch commit (solve_local_linear / solve_linear / a non-base
        # linear_combine wrote a scratch) is copied into the block state; a linear_combine commit already
        # wrote ctx.state(idx) in place (var == base), so its copy is a no-op (skipped).
        for block, committed in self._commits.items():
            base = bases[block]
            if var[committed.id] != var[base.id]:
                lines.append(
                    "ctx.lincomb(%s, static_cast<adc::Real>(0), %s, static_cast<adc::Real>(1), %s);"
                    % (var[base.id], var[base.id], var[committed.id]))
        # Rotate the history rings ONCE at the very end of the step (after the commit), so the next step
        # reads lag k as the value k stores ago. Only emitted when the Program uses histories.
        if self._histories:
            lines.append("ctx.rotate_histories();")
        prelude_src = "\n".join("  " + ln for ln in prelude)
        body_src = "\n".join("    " + ln for ln in lines)
        return prelude_src, body_src

    def _emit_op(self, v, base, committed_ids, var, model, lines, prelude=None, block_idx=None):
        """Lower a SINGLE op to C++, appending to @p lines and recording its C++ token in @p var. Shared
        by the top-level walk and the while sub-blocks (a while body re-runs this per op each pass), so
        reductions / compares / linear_combine all lower identically inside the loop. @p base is the
        block-state value of THIS op's block (its C++ var is the loop variable inside a while sub-block);
        @p committed_ids is the set of committed value ids (empty inside a sub-block: a body combine is
        never a commit). @p prelude collects INSTALL-TIME lines (persistent scratch + apply lambdas) for
        the matrix-free Krylov ops; None inside a sub-block (those ops only appear at the top level for
        now). @p block_idx maps a block name to its runtime index (ADC-426); None inside a sub-block,
        where every op shares the single enclosing block, so a missing map resolves to index 0."""
        bidx = (block_idx or {}).get(v.block, 0)  # this op's runtime block index (0 single-block)
        # PER-NODE PROFILING (ADC-459, Spec 3 section 29): bracket the C++ this op emits with a
        # steady_clock pair recorded under "node:<v.name>", so sim.profile_report() shows each Program
        # node's wall time next to the coarse step / field_solve phases. A steady_clock now() + a
        # ctx.profile_record (NOT a RAII ProfileScope block) is used so the node's emitted C++
        # declarations stay at the step-body scope -- a surrounding { } would hide them from later
        # nodes (e.g. r2 / acc3 read across ops). The timing is additive and ~free when profiling is
        # off (ctx.profile_record early-returns inside Profiler::record); it changes no numerics. Ops
        # that emit no statement (a pure inline token: cfl / compare) are not wrapped (the len guard
        # below skips them). _start marks where this op's lines begin so the open line can be inserted.
        _profile_start = len(lines)
        if v.op == "state":
            var[v.id] = "u%d" % v.id
            lines.append("adc::MultiFab& %s = ctx.state(%d);" % (var[v.id], bidx))
        elif v.op == "solve_fields":
            # Per-stage field solve (ADC-409): solve from the EXPLICIT stage state recorded by
            # P.solve_fields(state=...) so a field-coupled multi-stage scheme re-solves phi from each
            # stage's own state (the shared aux is re-filled before this stage's RHS reads it). For the
            # first stage state == U^n, so this is identical to the old ctx.solve_fields(). Multi-block
            # (ADC-426): solve_fields_from_state(idx, U_stage) is a genuinely COUPLED solve -- the system
            # Poisson RHS is Sum_s elliptic_rhs_s(U_s) (assemble_poisson_rhs), so block idx reads its
            # stage state while every OTHER block contributes its live state into the shared phi/aux.
            (state_in,) = v.inputs  # solve_fields inputs = (state,)
            field = v.attrs.get("field")
            if field is not None:
                # NAMED multi-elliptic field (ADC-428): a SECOND elliptic solve into the field's OWN aux
                # channel (distinct from the shared phi/grad). Lowers to the named overload
                # ctx.solve_fields_from_state(field, block, U) -- block from block_idx (ADC-426); the
                # default (unnamed) path keeps the 2-arg overload below, byte-identical.
                solve_stmt = ('ctx.solve_fields_from_state(%s, %d, %s);'
                              % (json.dumps(field), bidx, var[state_in.id]))
            else:
                solve_stmt = "ctx.solve_fields_from_state(%d, %s);" % (bidx, var[state_in.id])
            lines.append(solve_stmt)
        elif v.op == "solve_fields_from_blocks":
            # Coupled multi-block field solve (Spec 3 criterion 24, ADC-457): a SIMULTANEOUS solve where
            # EVERY listed block reads its OWN stage state at once -- the system Poisson RHS is
            # Sum_s elliptic_rhs_s(U_s) over all coupled blocks (assemble_poisson_rhs_from_blocks), not a
            # single-target override. Lowers to ctx.solve_fields_from_blocks(u_stages), a vector indexed
            # BY BLOCK INDEX (size == ctx.n_blocks(); a nullptr entry uses the block's live state). The
            # listed states are slotted at their block index, so the runtime sees each coupled block at
            # its stage state and every other (unlisted) block at its live state -- the seam a multi-
            # species step uses (the IR commit_many guarantee: no operator observes a partial group).
            # Each input is routed to the slot of ITS OWN block index (not its position in the list), so
            # a reordered list still solves correctly; an input whose block was never declared via
            # P.state has no slot -> fail loud at emit rather than silently mis-route to index 0.
            bmap = block_idx or {}
            vec = "u_stages_%d" % v.id
            lines.append("std::vector<const adc::MultiFab*> %s(ctx.n_blocks(), nullptr);" % vec)
            for st in v.inputs:  # inputs = the N state values, slotted by their own block index
                if st.block not in bmap:
                    raise ValueError(
                        "solve_fields_from_blocks: input node %r has block %r, which is not a "
                        "declared program block %r -- cannot route it to a coupled slot"
                        % (st.id, st.block, sorted(bmap)))
                lines.append("%s[%d] = &%s;" % (vec, bmap[st.block], var[st.id]))
            lines.append("ctx.solve_fields_from_blocks(%s);" % vec)
            # solve_fields_from_blocks returns a FieldContext (the shared aux); its var aliases the first
            # listed state so a downstream rhs(state, fields) reads the refreshed shared aux like any
            # solve_fields result (the FieldContext carries no readable buffer of its own).
            var[v.id] = var[v.inputs[0].id]
        elif v.op == "coupled_rate":
            # A coupled rate (collisions / ionization, Spec 3 criterion 27, ADC-457): ONE multi-state
            # for_each_cell kernel fills the per-block rate scratch of EVERY participating block at
            # once -- the component formulas reference cons vars from MULTIPLE input states, so the
            # blocks cannot be lowered as independent single-block rates. Allocate one rate scratch per
            # block (shaped like that block's state, via rhs_scratch_like), emit the shared kernel that
            # binds each input state's Array4 + cons names and writes all block scratches, and record
            # each block's scratch name so the coupled_rate_out for that block aliases it. All input
            # states are co-located (same ba/dm as the System aux), so a single shared loop is sound
            # (the same co-distribution every aux-reading kernel relies on; see _kernel_open).
            components = self._coupled_rate_components(v)
            by_block = {s.block: s for s in v.inputs}
            scratch = {}
            for blk in components:                       # bundle / expr block order
                scratch[blk] = "cr%d_%s" % (v.id, blk)
                lines.append("adc::MultiFab %s = ctx.rhs_scratch_like(%s);"
                             % (scratch[blk], var[by_block[blk].id]))
            lines += _emit_coupled_rate_kernel(components, by_block, var, scratch)
            # Per-block scratch names keyed by (coupled node id, block) so each coupled_rate_out aliases
            # its block's scratch (the projection emits no code of its own).
            self._coupled_scratch.update({(v.id, blk): scratch[blk] for blk in scratch})
            var[v.id] = scratch[next(iter(scratch))]     # a stable alias (the bundle has no single value)
        elif v.op == "coupled_rate_out":
            # Pure projection of one block out of the coupled bundle: its var aliases that block's rate
            # scratch (filled by the coupled_rate kernel above). Emits nothing -- like the FieldContext
            # alias of solve_fields_from_blocks. The producing coupled_rate is the node's sole input.
            (coupled_in,) = v.inputs
            var[v.id] = self._coupled_scratch[(coupled_in.id, v.attrs["out_block"])]
        elif v.op == "history":
            # Read the SYSTEM-OWNED history slot (a MultiFab&, ADC-406a): lag steps back. The reference
            # is bound to a C++ name the affine combine then reads like any other state/RHS term.
            var[v.id] = "h%d" % v.id
            lines.append("adc::MultiFab& %s = ctx.history(%s, %d);"
                         % (var[v.id], json.dumps(v.attrs["history"]), int(v.attrs["lag"])))
        elif v.op == "store_history":
            # Side-effect: copy the value into the current slot of the history (the cold-start fill on
            # the first store happens System-side). store_history is a State-typed node but carries no
            # readable value -- nothing combines it. Its var maps to the stored value (a harmless alias).
            (value_in,) = v.inputs
            lines.append("ctx.store_history(%s, %s);"
                         % (json.dumps(v.attrs["history"]), var[value_in.id]))
            var[v.id] = var[value_in.id]
        elif v.op == "fill_boundary":
            # Side effect on the field's ghosts (the valid cells are untouched). The result aliases the
            # input field (any subsequent op reading it sees the same C++ MultiFab, now with filled
            # halos). Forwards to ctx.fill_boundary (the shared transport-BC ghost exchange).
            (x,) = v.inputs
            lines.append("ctx.fill_boundary(%s);" % var[x.id])
            var[v.id] = var[x.id]
        elif v.op == "project":
            # In-place positivity projection of the state (the block's own project closure). The result
            # aliases the input state. Forwards to ctx.apply_projection(idx, state) (ADC-426: the op's
            # own block, so each block runs its own projection).
            (state_in,) = v.inputs
            lines.append("ctx.apply_projection(%d, %s);" % (bidx, var[state_in.id]))
            var[v.id] = var[state_in.id]
        elif v.op == "cell_compare":
            # A PER-CELL threshold (spec op 17, ADC-418): mask(i,j,0) = field(i,j,0) <cmp> value ? 1 : 0,
            # a fresh 1-component scalar_field. Lowered to a for_each_cell select kernel (the mask the
            # `where` op selects on); no aux / model needed -- it reads component 0 of the input field.
            (field_in,) = v.inputs
            var[v.id] = "m%d" % v.id
            lines.append("adc::MultiFab %s = ctx.alloc_scalar_field(1, 1);" % var[v.id])
            lines += _emit_cell_compare_kernel(var[field_in.id], var[v.id], v.attrs["cmp"],
                                               v.attrs["value"])
        elif v.op == "where":
            # A PER-CELL conditional select (spec op 17, ADC-418): out(i,j,c) = mask ? a(i,j,c) :
            # b(i,j,c), COMPONENT-WISE. A fresh scratch the same shape as `a` (its vtype / ncomp); the
            # ternary is decided per cell inside the kernel (NOT a scalar runtime branch -- that is if_).
            mask_in, a_in, b_in = v.inputs
            var[v.id] = "w%d" % v.id
            lines.append("adc::MultiFab %s = ctx.scratch_state_like(%s);" % (var[v.id], var[a_in.id]))
            lines += _emit_where_kernel(var[mask_in.id], var[a_in.id], var[b_in.id], var[v.id])
        elif v.op == "record_scalar":
            # Store the (already-computed) Scalar into the System diagnostics map under its name. A
            # side-effecting op; its var maps to the recorded scalar (a harmless alias). The scalar input
            # is a 'reduce' result emitted earlier in the body (a const adc::Real local).
            (scalar_in,) = v.inputs
            lines.append("ctx.record_scalar(%s, %s);"
                         % (json.dumps(v.attrs["diagnostic"]), var[scalar_in.id]))
            var[v.id] = var[scalar_in.id]
        elif v.op == "rhs":
            state_in = v.inputs[0]  # rhs inputs = (state[, fields]); the state is first
            var[v.id] = "r%d" % v.id
            lines.append("adc::MultiFab %s = ctx.rhs_scratch_like(%s);"
                         % (var[v.id], var[state_in.id]))
            named_fluxes = _named_fluxes(v)
            requested = v.attrs.get("sources")
            want_flux = v.attrs.get("flux", True)
            # ADC-425 routing (spec criterion 17): the default/composite source is folded in iff the
            # caller did NOT exclude it -- i.e. sources is None (the legacy default) OR "default" is in
            # the explicit list. An EMPTY list [] (or a list of only named sources) excludes it -> flux
            # only. None and [] are recorded distinctly in the IR, so this is unambiguous.
            want_default_source = requested is None or "default" in requested
            if not want_flux:
                # SOURCE-ONLY (ADC-430): flux=False -- NO -div F base (the rhs_scratch starts at zero).
                # The default/composite source is added iff requested (the same want_default_source
                # routing as flux=True): "default" present (or None) -> ctx.source_default_into (S only,
                # the exact mirror of neg_div_flux_default_into); excluded -> R stays the zeroed scratch.
                # The named source_terms below axpy on top either way -- so flux=False,sources=["default"]
                # is the default source only; flux=False,sources=["s"] is just s; flux=False,sources=[]
                # is the zero RHS. Named fluxes are rejected upstream (no flux base to divide). This is
                # the fix: before ADC-430 a flux=False stage still emitted the -div F base (it ignored the
                # flux attr), double-adding the flux on any non-zero-flux model in a Lie/Strang split.
                if want_default_source:
                    lines.append("ctx.source_default_into(%d, %s, %s);"
                                 % (bidx, var[state_in.id], var[v.id]))
            elif named_fluxes is None:
                if want_default_source:
                    # R <- -div F + default/composite source (ctx.rhs_into) for THIS op's block (ADC-426
                    # bidx), the historical path: sources is None (legacy) or "default" is requested.
                    lines.append("ctx.rhs_into(%d, %s, %s);" % (bidx, var[state_in.id], var[v.id]))
                else:
                    # FLUX-ONLY (ADC-425): "default" is NOT among the requested sources (the empty list
                    # [] or a named-only list) -> R <- -div F(U) WITHOUT the model's default source
                    # (ctx.neg_div_flux_default_into), for THIS op's block (bidx). The named source_terms
                    # below are then axpy'd on top -- sources=[] is flux only, ["a","b"] is flux + a + b.
                    lines.append("ctx.neg_div_flux_default_into(%d, %s, %s);"
                                 % (bidx, var[state_in.id], var[v.id]))
            else:
                # NAMED fluxes (ADC-419): R <- -div(sum of selected named fluxes). Evaluate the SUM of
                # the flux expressions per direction into two n_cons scratch fields (fx / fy) by a
                # per-cell kernel, then take the negated centered FV divergence into R. Linear in the
                # named pieces -> splitting the physical flux into named pieces that sum to it gives the
                # SAME -div (to round-off). Distinct stencil from rhs_into (centered FV vs Riemann), so
                # this path is NEVER mixed with the default (guarded by _named_fluxes).
                fx = "%s_fx" % var[v.id]
                fy = "%s_fy" % var[v.id]
                lines.append("adc::MultiFab %s = ctx.rhs_scratch_like(%s);" % (fx, var[state_in.id]))
                lines.append("adc::MultiFab %s = ctx.rhs_scratch_like(%s);" % (fy, var[state_in.id]))
                lines += _emit_flux_kernel(model, named_fluxes, var[state_in.id], fx, fy)
                lines.append("ctx.neg_div_flux_into(%s, %s, %s);" % (var[v.id], fx, fy))
            named = [s for s in (v.attrs.get("sources") or []) if s != "default"]
            for s in named:
                # R += S_s(U, aux): assemble the named source into a scratch (same per-cell kernel as
                # the standalone 'source' op) and axpy it onto R.
                ssrc = "%s_%s" % (var[v.id], s)
                lines.append("adc::MultiFab %s = ctx.rhs_scratch_like(%s);"
                             % (ssrc, var[state_in.id]))
                lines += _emit_source_kernel(model, s, var[state_in.id], ssrc)
                lines.append("ctx.axpy(%s, static_cast<adc::Real>(1), %s);" % (var[v.id], ssrc))
        elif v.op == "source":
            state_in = v.inputs[0]  # source inputs = (state[, fields]); the state is first
            var[v.id] = "r%d" % v.id
            lines.append("adc::MultiFab %s = ctx.rhs_scratch_like(%s);"
                         % (var[v.id], var[state_in.id]))
            lines += _emit_source_kernel(model, v.attrs["source"], var[state_in.id], var[v.id])
        elif v.op == "apply":
            state_in = v.inputs[0]  # apply inputs = (state[, fields]); the state is first
            var[v.id] = "r%d" % v.id
            lines.append("adc::MultiFab %s = ctx.rhs_scratch_like(%s);"
                         % (var[v.id], var[state_in.id]))
            lines += _emit_apply_kernel(model, v.attrs["linear_source"], var[state_in.id], var[v.id])
        elif v.op == "solve_local_linear":
            rhs_in = v.inputs[0]  # solve inputs = (rhs_state, op_value[, fields]); rhs first
            var[v.id] = "u%d" % v.id
            lines.append("adc::MultiFab %s = ctx.scratch_state_like(%s);"
                         % (var[v.id], var[base.id]))
            lines += _emit_solve_local_linear_kernel(
                model, v.attrs["linear_source"], v.attrs["a_coeff"], var[rhs_in.id], var[v.id])
        elif v.op == "solve_local_nonlinear":
            # Per-cell Newton (spec op 10): solve residual(U) = 0 from the initial guess U0, cell by
            # cell, with an in-kernel FD Jacobian + the SAME stack dense inverse solve_local_linear
            # uses. The output is a fresh scratch state; the guess input seeds the iterate.
            guess_in = v.inputs[0]  # solve inputs = (initial_guess,)
            var[v.id] = "u%d" % v.id
            lines.append("adc::MultiFab %s = ctx.scratch_state_like(%s);"
                         % (var[v.id], var[base.id]))
            lines += _emit_solve_local_nonlinear_kernel(model, v, var[guess_in.id], var[v.id])
        elif v.op == "schur_coeffs":
            # Anisotropic condensed-Schur coefficient bundle (ADC-421): allocate the four 1-component
            # coefficient fields ONCE (persistent shared_ptr in the prelude, captured by the apply
            # lambda) and FILL them per step in the body from the live state + B_z aux. The bundle's var
            # is the 4 shared_ptr names; apply_laplacian_coeff dereferences them inside the apply.
            if prelude is None:
                raise NotImplementedError(
                    "schur_coeffs is only lowerable at the top level / step body, not inside a "
                    "control-flow (if/while/range) body")
            self._emit_schur_coeffs(v, var, lines, prelude)
        elif v.op == "scalar_field":
            # A step-body scratch scalar field (e.g. the explicit-flux buffer the RHS assembly fills):
            # a persistent shared_ptr (prelude, alloc-once) reused every step. Inside an apply sub-block
            # the scalar_field is handled by _emit_matrix_free_operator instead (this branch is the
            # top-level / step-body path -- prelude is not None there).
            if prelude is None:
                raise NotImplementedError(
                    "scalar_field is only lowerable at the top level / step body or inside a "
                    "matrix_free_operator apply sub-block, not inside a control-flow (if/while/range) body")
            sp = "sf%d" % v.id
            var[v.id] = "(*%s)" % sp
            ncomp = int(v.attrs.get("ncomp", 1))
            prelude.append("auto %s = std::make_shared<adc::MultiFab>(ctx.alloc_scalar_field(%d, 1));"
                           % (sp, ncomp))
        elif v.op == "schur_explicit_flux":
            # F = B^{-1} (mx, my) per cell into a 2-component scalar field (Fx comp 0, Fy comp 1): the
            # explicit condensed-Schur flux, a step-body one-shot (not a per-iteration apply).
            out_in, state_in = v.inputs
            lines.append("ctx.schur_explicit_flux(%s, %s, %s, %d, %d, %d);"
                         % (_deref(var[out_in.id]), var[state_in.id], _coeff_cpp(v.attrs["th_dt"]),
                            v.attrs["c_mx"], v.attrs["c_my"], v.attrs["c_bz"]))
            var[v.id] = var[out_in.id]
        elif v.op == "schur_rhs":
            # Fused condensed-Schur RHS = -Lap(phi^n) - g*div(F) into a 1-component scalar field: the
            # native assemble_rhs in one ctx call (no scalar-field affine combine at IR level).
            out_in, phi_in, state_in = v.inputs
            lines.append(
                "ctx.assemble_schur_rhs(%s, %s, %s, %s, %s, %d, %d, %d);"
                % (_deref(var[out_in.id]), _deref(var[phi_in.id]), var[state_in.id],
                   _coeff_cpp(v.attrs["th_dt"]), _coeff_cpp(v.attrs["g"]), v.attrs["c_mx"],
                   v.attrs["c_my"], v.attrs["c_bz"]))
            var[v.id] = var[out_in.id]
        elif v.op == "laplacian":
            # Step-body bare Laplacian (e.g. Lap phi^n for the condensed RHS). Inside an apply sub-block
            # this op is handled by _emit_matrix_free_operator; here it is the top-level path.
            o, i = v.inputs
            lines.append("ctx.laplacian(%s, %s);" % (_deref(var[o.id]), _deref(var[i.id])))
            var[v.id] = var[o.id]
        elif v.op == "gradient":
            o, p = v.inputs
            lines.append("ctx.gradient(%s, %s);" % (_deref(var[o.id]), _deref(var[p.id])))
            var[v.id] = var[o.id]
        elif v.op == "divergence":
            o, fx, fy = v.inputs
            lines.append("ctx.divergence(%s, %s, %s);"
                         % (_deref(var[o.id]), _deref(var[fx.id]), _deref(var[fy.id])))
            var[v.id] = var[o.id]
        elif v.op == "schur_reconstruct":
            # In-place velocity reconstruction v = B^{-1}(v^n - theta dt grad phi); mom = rho v. Result
            # aliases the input state (mx/my overwritten). phi is a scalar_field / 1-comp State token.
            state_in, phi_in = v.inputs
            lines.append("ctx.schur_reconstruct(%s, %s, %s, %d, %d, %d, %d);"
                         % (var[state_in.id], _deref(var[phi_in.id]), _coeff_cpp(v.attrs["th_dt"]),
                            v.attrs["c_rho"], v.attrs["c_mx"], v.attrs["c_my"], v.attrs["c_bz"]))
            var[v.id] = var[state_in.id]
        elif v.op == "schur_energy":
            # In-place energy increment E += (1/2) rho (|v^{n+1}|^2 - |v^n|^2) (ADC-427). Reads v^{n+1}
            # from the updated state and v^n / E^n from state_old (U^n); rho frozen (same in both).
            state_in, old_in = v.inputs
            lines.append("ctx.schur_energy(%s, %s, %d, %d, %d, %d);"
                         % (var[state_in.id], var[old_in.id], v.attrs["c_rho"], v.attrs["c_mx"],
                            v.attrs["c_my"], v.attrs["c_E"]))
            var[v.id] = var[state_in.id]
        elif v.op == "matrix_free_operator":
            # Install-time: emit the apply lambda `apply_A{id}` into the prelude. Its persistent scratch
            # (the scalar_field ops of the apply sub-block) are shared_ptr fields, captured by value so
            # they outlive the install call and are reused across every Krylov iteration (alloc-once).
            # The lambda is itself captured by the step closure ([=]) and passed to adc::*_solve. An
            # rhs_jacvec apply (ADC-431) also captures persistent jac_uk / jac_r0 scratch the lambda
            # dereferences; the step body refreshes them from the live iterate / rhs(U^k) here (@p lines).
            self._emit_matrix_free_operator(v, var, prelude, lines)
        elif v.op in ("apply_in", "apply_out", "apply_laplacian_coeff"):
            # The lambda in/out placeholders and the coefficiented apply matvec only appear INSIDE a
            # matrix_free_operator apply sub-block (lowered by _emit_matrix_free_operator); they never
            # lower standalone at the top level.
            raise NotImplementedError(
                "emit_cpp_program: op '%s' (value '%s') is only lowerable inside a matrix_free_operator "
                "apply sub-block" % (v.op, v.name))
        elif v.op == "solve_linear":
            self._emit_solve_linear(v, base, var, prelude, lines)
        elif v.op == "reduce":
            # A collective all_reduce -> a C++ scalar. norm2 = sqrt(dot(u, u)); dot(a, b) directly;
            # sum/max/min (over a component) via the matching adc reduction. All MUST run on every rank
            # (the reductions are collective all_reduce); they sit at the top of the loop body.
            var[v.id] = "s%d" % v.id
            kind = v.attrs["kind"]
            if kind == "norm2":
                (u,) = v.inputs
                lines.append("const adc::Real %s = std::sqrt(adc::dot(%s, %s));"
                             % (var[v.id], var[u.id], var[u.id]))
            elif kind == "norm_inf":
                (u,) = v.inputs
                lines.append("const adc::Real %s = adc::norm_inf(%s);" % (var[v.id], var[u.id]))
            elif kind in ("sum", "max", "min"):
                (u,) = v.inputs
                comp = int(v.attrs.get("comp", 0))
                lines.append("const adc::Real %s = adc::reduce_%s(%s, %d);"
                             % (var[v.id], kind, var[u.id], comp))
            else:  # dot
                a, b = v.inputs
                lines.append("const adc::Real %s = adc::dot(%s, %s);"
                             % (var[v.id], var[a.id], var[b.id]))
        elif v.op == "cfl":
            # The dt_bound's runtime cfl argument -- the C++ parameter of adc_program_dt_bound. It is
            # NOT a statement; its token is the bound parameter name (spec s18 / ADC-417).
            var[v.id] = "cfl"
        elif v.op == "hmin":
            # MIN physical cell size (ctx.hmin(), = the native CFL's hmin). A scalar local (spec s18).
            var[v.id] = "s%d" % v.id
            lines.append("const adc::Real %s = ctx.hmin();" % var[v.id])
        elif v.op == "max_wave_speed":
            # Max |wave speed| of the block on the state (ctx.max_wave_speed(idx, u)): the SAME per-block
            # reduction the native CFL reads, REUSED (spec s18). A collective reduction -> a scalar local.
            # ADC-426: the wave speed of the input state's OWN block (idx of u.block).
            (u,) = v.inputs
            var[v.id] = "s%d" % v.id
            lines.append("const adc::Real %s = ctx.max_wave_speed(%d, %s);"
                         % (var[v.id], (block_idx or {}).get(u.block, 0), var[u.id]))
        elif v.op == "scalar_op":
            # Scalar arithmetic (add/sub/mul/div) over scalar locals / literal constants -> a new scalar
            # local. Used by the dt_bound expression cfl * hmin / max_wave_speed (spec s18).
            var[v.id] = "s%d" % v.id
            toks = []
            for kind, val in v.attrs["operands"]:
                if kind == "v":
                    toks.append(var[v.inputs[val].id])
                else:  # a literal constant
                    toks.append("static_cast<adc::Real>(%s)" % repr(float(val)))
            cppop = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[v.attrs["fn"]]
            lines.append("const adc::Real %s = (%s %s %s);"
                         % (var[v.id], toks[0], cppop, toks[1]))
        elif v.op == "compare":
            # A predicate over scalars -> an inline boolean C++ expression (no statement of its own; the
            # while op embeds it directly in `if (!(<expr>)) break;`).
            lhs = v.inputs[0]
            if len(v.inputs) == 2:  # scalar vs scalar
                rhs_tok = var[v.inputs[1].id]
            else:  # scalar vs float tolerance
                rhs_tok = "static_cast<adc::Real>(%s)" % repr(float(v.attrs["rhs"]))
            var[v.id] = "(%s %s %s)" % (var[lhs.id], v.attrs["cmp"], rhs_tok)
            self._when_tokens[v.id] = var[v.id]  # reusable as a when(cond) due test (ADC-458)
        elif v.op == "while":
            self._emit_while(v, base, var, model, lines, block_idx)
        elif v.op == "range":
            self._emit_range(v, base, var, model, lines, block_idx)
        elif v.op == "if":
            self._emit_if(v, base, var, model, lines, block_idx)
        elif v.op == "linear_combine":
            terms = list(zip(v.inputs, v.attrs["coeffs"], strict=True))
            if v.id in committed_ids:
                # Commit: block state <- c_base * base + sum(non-base coeff * term), in place.
                c_base = {0: 0.0}
                acc = "acc%d" % v.id
                lines.append("adc::MultiFab %s = ctx.scratch_state_like(%s);" % (acc, var[base.id]))
                for inp, coeff in terms:
                    if inp.id == base.id:
                        c_base = coeff
                    else:
                        lines.append("ctx.axpy(%s, %s, %s);" % (acc, _coeff_cpp(coeff), var[inp.id]))
                lines.append("ctx.lincomb(%s, %s, %s, static_cast<adc::Real>(1), %s);"
                             % (var[base.id], _coeff_cpp(c_base), var[base.id], acc))
                var[v.id] = var[base.id]  # the commit wrote the block state in place (no final copy)
            else:
                var[v.id] = "u%d" % v.id  # an intermediate stage state (scratch, zero-initialized)
                lines.append("adc::MultiFab %s = ctx.scratch_state_like(%s);" % (var[v.id], var[base.id]))
                for inp, coeff in terms:
                    lines.append("ctx.axpy(%s, %s, %s);" % (var[v.id], _coeff_cpp(coeff), var[inp.id]))
        # UNIFIED SCHEDULER (ADC-458, Spec 3 sections 17-18): if this op carries a non-always schedule,
        # wrap the statements it just emitted (lines[_profile_start:]) in the due-test guard + policy
        # branch. Done HERE, after the op lowered itself, so EVERY schedulable node (field solve, rhs,
        # source, linear_combine, where, ...) reuses the one general mechanism -- no per-op special
        # case. The wrap nests INSIDE the per-node profiling pair below (the profiler times the guarded
        # block as the node's cost). An always() schedule (or no schedule) leaves the lines untouched.
        self._emit_schedule_wrap(v, var, lines, _profile_start)
        # PER-NODE PROFILING (ADC-459): if this op emitted at least one statement, bracket those
        # statements with the steady_clock pair (see the note at the top of _emit_op). A ProfileScope is
        # named "node:<v.name>"; profile_record(name, _pt) accumulates now() - _pt into the System
        # Profiler. Inserted only when lines grew (a pure inline-token op emits nothing and is skipped).
        # The pure reference-binding ops (state / history bind a MultiFab&; hmin reads a cached scalar)
        # do no per-step numerical work, so they are not wrapped -- the report keeps the meaningful
        # work nodes (rhs / solve_fields / linear_combine / source / apply / reductions / loops).
        if v.op not in Program._PROFILE_SKIP_OPS and len(lines) > _profile_start:
            node_name = json.dumps("node:%s" % v.name)
            pt = "_pt%d" % v.id  # unique per node id (no redefinition at body scope or in a loop pass)
            lines.insert(_profile_start,
                         "const auto %s = std::chrono::steady_clock::now();  // ProfileScope %s"
                         % (pt, node_name))
            lines.append("ctx.profile_record(%s, %s);" % (node_name, pt))

    # Ops whose schedulable OUTPUT is the System aux (phi / grad / E), not a step-body scratch: a held
    # field solve caches and restores the aux. Every other schedulable op writes a named scratch
    # MultiFab (var[v.id]) that the scratch cache holds/restores instead.
    _AUX_OUTPUT_OPS = frozenset({"solve_fields", "solve_fields_from_blocks"})

    def _schedule_due_test(self, v, sched):
        """The C++ boolean 'is this node due this step' for a non-subcycle schedule kind. Reused as the
        guard of the policy branch. Raises (naming ADC-458) for a kind that needs a runtime primitive the
        compiled .so does not have (on_end: no end-of-run signal reaches a sim.step(dt) loop)."""
        kind = sched.kind
        if kind == "every":
            # Cadence: due cold-start, then every N macro-steps (CacheManager::is_due via macro_step()).
            return "ctx.cache_should_update(%d, %d)" % (v.id, int(sched.params.get("n", 1)))
        if kind == "on_start":
            return "(ctx.macro_step() == 0)"
        if kind == "when":
            # A runtime predicate: a Program Bool value already lowered to a parenthesized C++ expr token
            # (a compare over reductions). A bare Python callable cannot lower (it is not a Program value).
            cond = sched.params.get("cond")
            if not (isinstance(cond, Value) and cond.vtype == "bool"):
                raise NotImplementedError(
                    "when(cond) lowers only a Program Bool predicate (e.g. P.norm2(r) < tol), not a "
                    "Python callable: node %r (ADC-458). Build the condition with Program compares."
                    % v.name)
            if cond.id not in self._when_tokens:
                raise ValueError(
                    "when(cond) on node %r references a Bool value not emitted before it; build the "
                    "predicate earlier in the Program (ADC-458)" % v.name)
            return self._when_tokens[cond.id]
        raise NotImplementedError(
            "schedule kind %r on node %r is not lowerable: on_end() needs an end-of-run signal that a "
            "compiled sim.step(dt) loop never sees (the .so cannot know the last step); use on_start()/"
            "every()/when()/subcycle() or an on_end host hook (ADC-458)." % (kind, v.name))

    def _emit_schedule_wrap(self, v, var, lines, start):
        """Wrap the C++ statements node @p v emitted (``lines[start:]``) in its schedule's due-test guard
        + policy branch (ADC-458, Spec 3 sections 17-18). Generic over the op: a field solve caches the
        System aux, any other node caches its named scratch (var[v.id]). An always()/absent schedule
        leaves the lines untouched (byte-identical to the unscheduled lowering)."""
        sched = v.attrs.get("schedule")
        if sched is None or sched.is_always():
            return
        body = lines[start:]
        del lines[start:]
        if sched.kind == "subcycle":
            # Structured sub-cycling of a field solve (the gate restricts subcycle to an aux-output op):
            # run the op body COUNT times over the sub-dt (macro_dt / count by default, or an explicit
            # dt), refreshing the persistent System aux each pass. A pure recompute cadence -- no cache.
            # The sub-dt is a const local exposed for a dt-scaled body; the MVP field-solve body is
            # dt-free, so it documents the cadence (the aux solve re-runs COUNT times).
            count = int(sched.params["count"])
            sub_dt = sched.params.get("dt")
            sd = "_subdt%d" % v.id
            if sub_dt is None:
                lines.append("const adc::Real %s = dt / static_cast<adc::Real>(%d);" % (sd, count))
            else:
                lines.append("const adc::Real %s = static_cast<adc::Real>(%s);"
                             % (sd, repr(float(sub_dt))))
            lines.append("(void)%s;" % sd)  # the MVP body is self-contained; sd documents the cadence
            lines.append("for (int _sub%d = 0; _sub%d < %d; ++_sub%d) {" % (v.id, v.id, count, v.id))
            lines += ["  " + ln for ln in body]
            lines.append("}")
            return
        due = self._schedule_due_test(v, sched)
        policy = sched.policy
        is_aux = v.op in Program._AUX_OUTPUT_OPS
        # The scratch node's output token (the MultiFab the policy holds / zeroes). A field solve writes
        # the System aux and sets no var[v.id], so out is read only on the scratch path.
        out = None if is_aux else var.get(v.id)
        if policy == "recompute":
            # Run only when due; on a NOT-due step do nothing (the aux / scratch keeps its last content).
            # recompute off-cadence is simply 'run when due' -- no cache, no else branch. A scratch node
            # hoists its output declaration so the buffer stays in scope when the body does not run.
            if is_aux:
                lines.append("if (%s) {" % due)
                lines += ["  " + ln for ln in body]
                lines.append("}")
            else:
                decl, rest = self._split_output_decl(body, out, v)
                lines.append(decl)
                lines.append("if (%s) {" % due)
                lines += ["  " + ln for ln in rest]
                lines.append("}")
            return
        if policy == "skip":
            # Do not run the op off-cadence: the value keeps its previous content (the cacheable contract
            # -- downstream must tolerate a stale value). A scratch node hoists its output declaration so
            # the stale buffer stays in scope across the guard (no else branch: nothing happens off-
            # cadence); a field solve writes the persistent aux, so its whole body simply guards.
            if is_aux:
                lines.append("if (%s) {  // skip: stale aux off-cadence" % due)
                lines += ["  " + ln for ln in body]
                lines.append("}")
            else:
                decl, rest = self._split_output_decl(body, out, v)
                lines.append(decl)
                lines.append("if (%s) {  // skip: stale value off-cadence" % due)
                lines += ["  " + ln for ln in rest]
                lines.append("}")
            return
        if policy == "zero":
            # Off-cadence, zero the node's output. The output must EXIST in both branches: for a scratch
            # node hoist its allocation out of the guard (the first emitted line declares var[v.id]); the
            # aux always exists (System-owned).
            if is_aux:
                lines.append("if (%s) {" % due)
                lines += ["  " + ln for ln in body]
                lines.append("} else {")
                lines.append("  ctx.aux().set_val(static_cast<adc::Real>(0));")
                lines.append("}")
            else:
                decl, rest = self._split_output_decl(body, out, v)
                lines.append(decl)
                lines.append("if (%s) {" % due)
                lines += ["  " + ln for ln in rest]
                lines.append("} else {")
                lines.append("  %s.set_val(static_cast<adc::Real>(0));" % out)
                lines.append("}")
            return
        if policy == "hold":
            # Recompute + cache when due; restore the cached value off-cadence (no recompute). The aux
            # path uses cache_store_aux/restore_aux; a scratch node hoists its allocation and uses the
            # named-scratch cache. _validate_schedule already rejected hold on a non-cacheable operator.
            if is_aux:
                lines.append("if (%s) {" % due)
                lines += ["  " + ln for ln in body]
                lines.append("  ctx.cache_store_aux(%d);" % v.id)
                lines.append("} else {")
                lines.append("  ctx.cache_restore_aux(%d);" % v.id)
                lines.append("}")
            else:
                decl, rest = self._split_output_decl(body, out, v)
                lines.append(decl)
                lines.append("if (%s) {" % due)
                lines += ["  " + ln for ln in rest]
                lines.append("  ctx.cache_store_scratch(%d, %s);" % (v.id, out))
                lines.append("} else {")
                lines.append("  ctx.cache_restore_scratch(%d, %s);" % (v.id, out))
                lines.append("}")
            return
        if policy == "accumulate_dt":
            # Off-cadence: accumulate THIS step's dt (the real skipped dt, never N*dt_current) and hold the
            # cached value. When due: read eff_dt = dt + sum(skipped) (resets the accumulator), recompute,
            # cache. eff_dt is bound so a dt-dependent recompute can read it (the MVP field solve / scratch
            # fill is dt-free, but eff_dt is exposed for a dt-scaled body). Cacheable (validated upstream).
            ed = "_effdt%d" % v.id
            if is_aux:
                lines.append("if (%s) {" % due)
                lines.append("  const adc::Real %s = ctx.cache_effective_dt(%d, dt); (void)%s;"
                             % (ed, v.id, ed))
                lines += ["  " + ln for ln in body]
                lines.append("  ctx.cache_store_aux(%d);" % v.id)
                lines.append("} else {")
                lines.append("  ctx.cache_accumulate_dt(%d, dt);" % v.id)
                lines.append("  ctx.cache_restore_aux(%d);" % v.id)
                lines.append("}")
            else:
                decl, rest = self._split_output_decl(body, out, v)
                lines.append(decl)
                lines.append("if (%s) {" % due)
                lines.append("  const adc::Real %s = ctx.cache_effective_dt(%d, dt); (void)%s;"
                             % (ed, v.id, ed))
                lines += ["  " + ln for ln in rest]
                lines.append("  ctx.cache_store_scratch(%d, %s);" % (v.id, out))
                lines.append("} else {")
                lines.append("  ctx.cache_accumulate_dt(%d, dt);" % v.id)
                lines.append("  ctx.cache_restore_scratch(%d, %s);" % (v.id, out))
                lines.append("}")
            return
        if policy == "error":
            # Guard that a stale value is never read off-cadence: run when due, else fail loud (the node
            # asserts it is only consumed on its cadence). Emitted as a runtime throw on the not-due path.
            # A scratch node hoists its output declaration so the buffer stays in scope (the throw never
            # returns, but the C++ must still be well-scoped); a field solve guards the aux body directly.
            err = ('ctx.scheduler_error(%s);'
                   % json.dumps("node '%s' (op '%s') read off its schedule cadence (policy=error)"
                                % (v.name, v.op)))
            if is_aux:
                lines.append("if (%s) {" % due)
                lines += ["  " + ln for ln in body]
                lines.append("} else {")
                lines.append("  " + err)
                lines.append("}")
            else:
                decl, rest = self._split_output_decl(body, out, v)
                lines.append(decl)
                lines.append("if (%s) {" % due)
                lines += ["  " + ln for ln in rest]
                lines.append("} else {")
                lines.append("  " + err)
                lines.append("}")
            return
        raise NotImplementedError(
            "schedule policy %r on node %r is not lowerable (ADC-458)" % (policy, v.name))

    def _split_output_decl(self, body, out, v):
        """Split a scratch node's emitted @p body into (declaration_line, rest): the OUTPUT scratch
        ``out`` must be declared OUTSIDE the policy guard so both branches see it, while the fill stays
        inside. The op declares its output as its FIRST emitted line (``adc::MultiFab <out> = ...;``);
        hoist exactly that one line. Raises if the shape is unexpected (a node whose output is not a
        freshly-declared scratch cannot use a cache/zero policy through this path)."""
        decl_prefix = "adc::MultiFab %s = " % out
        if not body or not body[0].startswith(decl_prefix):
            raise NotImplementedError(
                "schedule policy on node %r (op '%s') needs its output scratch %r declared as its first "
                "emitted line to hoist it out of the guard; got %r (ADC-458)"
                % (v.name, v.op, out, body[0] if body else None))
        return body[0], body[1:]

    def _emit_while(self, v, base, var, model, lines, block_idx=None):
        """Lower a while op to an infinite C++ loop with a break (the condition re-evaluates each pass).
        The loop variable is a single MultiFab mutated IN PLACE across iterations; the cond / body sub-
        blocks re-run the per-op lowering each pass, with the loop-variable value id seeded to the loop
        var so their references resolve to it."""
        loop_in = v.inputs[0]  # the initial loop-variable state
        x = "x%d" % v.id
        var[v.id] = x
        # Hoist + initialize the loop variable from the entry state (x <- loop_in).
        lines.append("adc::MultiFab %s = ctx.scratch_state_like(%s);" % (x, var[base.id]))
        lines.append("ctx.lincomb(%s, static_cast<adc::Real>(0), %s, static_cast<adc::Real>(1), %s);"
                     % (x, x, var[loop_in.id]))
        lines.append("for (;;) {")
        # The sub-blocks see the loop variable in place of the entry-state value id (the body / cond were
        # built reading the loop-var State; they resolve to x here). A fresh sub-var map keeps the inner
        # scratch names from leaking out, but inherits the outer bindings (the loop var, target, ...).
        sub = dict(var)
        sub[loop_in.id] = x
        body_lines = []
        for w in v.attrs["cond_block"]:
            self._emit_op(w, base, frozenset(), sub, model, body_lines, block_idx=block_idx)
        cond_expr = sub[v.attrs["cond"].id]
        body_lines.append("if (!(%s)) break;" % cond_expr)
        for w in v.attrs["body_block"]:
            self._emit_op(w, base, frozenset(), sub, model, body_lines, block_idx=block_idx)
        # Write the next state into the loop variable in place (x <- body result).
        body_lines.append("ctx.lincomb(%s, static_cast<adc::Real>(0), %s, static_cast<adc::Real>(1), %s);"
                          % (x, x, sub[v.attrs["body"].id]))
        lines += ["  " + ln for ln in body_lines]
        lines.append("}")

    def _emit_range(self, v, base, var, model, lines, block_idx=None):
        """Lower a range op to a C++ ``for`` over a fixed count. Like a while, the loop variable is one
        MultiFab mutated in place and the body sub-block is emitted ONCE inside the loop (re-run each
        pass at runtime); the loop-variable value id is seeded to the loop var for the sub-block."""
        loop_in = v.inputs[0]
        x = "x%d" % v.id
        i = "i%d" % v.id
        var[v.id] = x
        lines.append("adc::MultiFab %s = ctx.scratch_state_like(%s);" % (x, var[base.id]))
        lines.append("ctx.lincomb(%s, static_cast<adc::Real>(0), %s, static_cast<adc::Real>(1), %s);"
                     % (x, x, var[loop_in.id]))
        lines.append("for (int %s = 0; %s < %d; ++%s) {" % (i, i, int(v.attrs["count"]), i))
        sub = dict(var)
        sub[loop_in.id] = x
        body_lines = []
        for w in v.attrs["body_block"]:
            self._emit_op(w, base, frozenset(), sub, model, body_lines, block_idx=block_idx)
        body_lines.append("ctx.lincomb(%s, static_cast<adc::Real>(0), %s, static_cast<adc::Real>(1), %s);"
                          % (x, x, sub[v.attrs["body"].id]))
        lines += ["  " + ln for ln in body_lines]
        lines.append("}")

    def _emit_if(self, v, base, var, model, lines, block_idx=None):
        """Lower an if op to a C++ branch. @p cond was emitted at the top level (its boolean expression
        is var[cond.id]); the loop variable is a copy of the input state, overwritten in place only when
        the branch is taken (so the result is the input state when the condition is false at runtime)."""
        state_in, cond = v.inputs[0], v.inputs[1]
        x = "x%d" % v.id
        var[v.id] = x
        lines.append("adc::MultiFab %s = ctx.scratch_state_like(%s);" % (x, var[base.id]))
        lines.append("ctx.lincomb(%s, static_cast<adc::Real>(0), %s, static_cast<adc::Real>(1), %s);"
                     % (x, x, var[state_in.id]))
        lines.append("if (%s) {" % var[cond.id])
        sub = dict(var)
        sub[state_in.id] = x
        body_lines = []
        for w in v.attrs["body_block"]:
            self._emit_op(w, base, frozenset(), sub, model, body_lines, block_idx=block_idx)
        body_lines.append("ctx.lincomb(%s, static_cast<adc::Real>(0), %s, static_cast<adc::Real>(1), %s);"
                          % (x, x, sub[v.attrs["body"].id]))
        lines += ["  " + ln for ln in body_lines]
        lines.append("}")

    def _emit_schur_coeffs(self, v, var, lines, prelude):
        """Lower a schur_coeffs bundle (ADC-421): allocate the four 1-component coefficient fields
        (eps_x, eps_y, a_xy, a_yx) ONCE as persistent shared_ptrs (prelude, alloc-once, captured by the
        step closure and by the apply lambda that consumes them) and FILL them per step in the body from
        the live state + B_z aux via ``ctx.assemble_schur_coeffs`` (the SAME native
        detail::SchurOperatorCoeffKernel). The bundle's token is the tuple of the four shared_ptr names;
        ``apply_laplacian_coeff`` dereferences them inside the matrix-free apply."""
        (state_in,) = v.inputs
        ex = "ceps_x%d" % v.id
        ey = "ceps_y%d" % v.id
        axy = "ca_xy%d" % v.id
        ayx = "ca_yx%d" % v.id
        for sp in (ex, ey, axy, ayx):
            prelude.append(
                "auto %s = std::make_shared<adc::MultiFab>(ctx.alloc_scalar_field(1, 1));" % sp)
        var[v.id] = (ex, ey, axy, ayx)  # the bundle token: the four coefficient shared_ptr names
        lines.append(
            "ctx.assemble_schur_coeffs(*%s, *%s, *%s, *%s, %s, %s, %s, %d, %d);"
            % (ex, ey, axy, ayx, var[state_in.id], _coeff_cpp(v.attrs["c"]),
               _coeff_cpp(v.attrs["th_dt"]), v.attrs["c_rho"], v.attrs["c_bz"]))

    def _emit_matrix_free_operator(self, v, var, prelude, lines=None):
        """Lower a matrix_free_operator to an INSTALL-TIME C++ apply lambda ``apply_A{id}`` (appended to
        @p prelude). The lambda has the adc::ApplyFn signature ``(adc::MultiFab& out, const adc::MultiFab&
        in)``; its body re-emits the apply sub-block:

          - each ``scalar_field`` scratch -> a PERSISTENT shared_ptr field (declared in the prelude
            BEFORE the lambda, captured by value), reused across every Krylov iteration (alloc-once);
          - ``laplacian(o, i)`` -> ``ctx.laplacian(*o, i)`` (i const_cast when it is the lambda's ``in``,
            which is logically read-only -- the fill only writes ghosts, as in test_generic_krylov);
          - ``rhs_jacvec(out, in, iterate, r0, ...)`` (ADC-431) -> a finite-difference Jacobian-vector
            product ``out = in - (c*dt/eps)(rhs(U^k + eps*in) - rhs(U^k))`` calling ``ctx.rhs_into`` (or
            ``neg_div_flux_default_into``) on PERSISTENT jac_uk / jac_r0 scratch the lambda captures; the
            step body refreshes that scratch from the live iterate / rhs(U^k) (@p lines, see below);
          - the apply RESULT (the affine the body returned, e.g. ``in - alpha*Lap(in)``) is written into
            ``out`` via the same accumulate-then-lincomb idiom as a linear_combine commit.

        The lambda captures ``[ctx, <scratch shared_ptrs>]``; the step closure captures it by value. @p
        lines is the step-body line list (for the rhs_jacvec scratch refresh); None when the operator has
        no jacvec op (the historical matrix-free path, prelude only)."""
        apply_id = v.id
        lam = "apply_A%d" % apply_id
        var[apply_id] = lam
        in_sf = v.attrs["apply_in"]
        out_sf = v.attrs["apply_out"]
        block = v.attrs["apply_block"]
        result = v.attrs["apply_result"]
        # Sub-scope token map: the lambda params + persistent scratch. `in` is the const lambda param;
        # `out` is the (non-const) lambda param the result is written into.
        sub = {in_sf.id: "in", out_sf.id: "out"}
        # 1) Persistent scratch (the scalar_field ops): one shared_ptr per scratch, declared before the
        #    lambda so it is in scope to capture. Collected first so the capture list is known.
        scratch = [w for w in block if w.op == "scalar_field"]
        captures = ["ctx"]
        for w in scratch:
            sp = "sf%d_%d" % (apply_id, w.id)
            sub[w.id] = sp
            ncomp = int(w.attrs.get("ncomp", 1))  # >1 for a gradient buffer consumed by divergence
            prelude.append(
                "auto %s = std::make_shared<adc::MultiFab>(ctx.alloc_scalar_field(%d, 1));"
                % (sp, ncomp))
            captures.append(sp)
        # The affine result-write accumulator: one PERSISTENT shared_ptr (alloc-once, like the scratch),
        # zeroed and reused every matvec instead of allocated per call -- so the apply lambda allocates
        # NOTHING per Krylov iteration (the runtime r/p/Ap scratch in generic_krylov.hpp is likewise
        # alloc-once). _emit_field_combine writes the affine into `out` through it. It carries the
        # operator's component count so the axpy / lincomb cover ALL components (a vector / state apply).
        op_ncomp = int(v.attrs["ncomp"])
        acc_sp = "acc%d" % apply_id
        prelude.append(
            "auto %s = std::make_shared<adc::MultiFab>(ctx.alloc_scalar_field(%d, 1));"
            % (acc_sp, op_ncomp))
        captures.append(acc_sp)
        # A coefficiented apply (apply_laplacian_coeff) reads an OUTER schur_coeffs bundle (assembled in
        # the step body, before the operator): capture its four coefficient shared_ptrs (already
        # allocated in the prelude by _emit_schur_coeffs) so the lambda can dereference them.
        for w in block:
            if w.op == "apply_laplacian_coeff":
                coeffs = w.inputs[2]
                for sp in var[coeffs.id]:
                    if sp not in captures:
                        captures.append(sp)
        # An rhs_jacvec apply (ADC-431, implicit-flux BDF) needs the FROZEN Newton iterate U^k and its
        # precomputed rhs(U^k) inside the lambda. They are step-body locals that CHANGE each Newton
        # iteration, so -- like schur_coeffs -- they become PERSISTENT shared_ptr scratch (jac_uk / jac_r0)
        # captured by value (shared pointee), refreshed from the live iterate / r0 in the step body BEFORE
        # the solve. Plus a perturbed-state scratch (jac_up) and a perturbed-rhs scratch (jac_rp) the
        # lambda fills per matvec. All carry the operator's component count (= the block n_cons).
        jac_ops = [w for w in block if w.op == "rhs_jacvec"]
        if jac_ops and lines is None:
            raise NotImplementedError(
                "rhs_jacvec is only lowerable in a top-level / step-body matrix-free solve, not inside a "
                "control-flow (if/while/range) body (the Newton outer loop must be a static_range unroll)")
        jac_scratch = {}  # jacvec op id -> (uk, r0, up, rp, cdt) names
        ng_state = "ctx.state(0).n_grow()"  # the jacvec scratch needs the state's ghost count for rhs_into
        for w in jac_ops:
            uk = "jac_uk%d_%d" % (apply_id, w.id)
            r0 = "jac_r0%d_%d" % (apply_id, w.id)
            up = "jac_up%d_%d" % (apply_id, w.id)
            rp = "jac_rp%d_%d" % (apply_id, w.id)
            for sp in (uk, r0, up, rp):
                prelude.append(
                    "auto %s = std::make_shared<adc::MultiFab>(ctx.alloc_scalar_field(%d, %s));"
                    % (sp, op_ncomp, ng_state))
                captures.append(sp)
            # The BDF coefficient c*dt depends on the step's dt (the step-closure parameter), which the
            # install-time lambda cannot see; carry it through a captured shared_ptr<Real> the step body
            # sets to its dt value before the solve (the same persistent-scratch idiom as jac_uk).
            cdt = "jac_cdt%d_%d" % (apply_id, w.id)
            prelude.append("auto %s = std::make_shared<adc::Real>(static_cast<adc::Real>(0));" % cdt)
            captures.append(cdt)
            jac_scratch[w.id] = (uk, r0, up, rp, cdt)
            # Step body: refresh the FROZEN captures from this iteration's live iterate / rhs(U^k) / dt.
            iterate_in, r0_in = w.inputs[2], w.inputs[3]
            lines.append("ctx.lincomb(*%s, static_cast<adc::Real>(0), *%s, static_cast<adc::Real>(1), %s);"
                         % (uk, uk, var[iterate_in.id]))
            lines.append("ctx.lincomb(*%s, static_cast<adc::Real>(0), *%s, static_cast<adc::Real>(1), %s);"
                         % (r0, r0, var[r0_in.id]))
            lines.append("*%s = %s;" % (cdt, _coeff_cpp(w.attrs["c_dt"])))
        # 2) The lambda body: the laplacian / gradient ops + the result write into `out`.
        body = []
        for w in block:
            if w.op in ("scalar_field", "apply_in", "apply_out"):
                continue  # scratch shared_ptr / lambda params: already bound in `sub`, nothing to emit
            if w.op == "laplacian":
                o, i = w.inputs
                sub[w.id] = sub[o.id]
                body.append("ctx.laplacian(*%s, %s);" % (sub[o.id], _apply_in_arg(sub, i)))
            elif w.op == "gradient":
                o, p = w.inputs
                sub[w.id] = sub[o.id]
                body.append("ctx.gradient(*%s, %s);" % (sub[o.id], _apply_in_arg(sub, p)))
            elif w.op == "divergence":
                o, fx, fy = w.inputs
                sub[w.id] = sub[o.id]
                body.append("ctx.divergence(*%s, %s, %s);"
                            % (sub[o.id], _apply_in_arg(sub, fx), _apply_in_arg(sub, fy)))
            elif w.op == "apply_laplacian_coeff":
                # out = div(A grad in), A the schur_coeffs tensor: forwards to the native
                # apply_laplacian coefficient path. eps_x/eps_y/a_xy/a_yx are the captured coeff fields.
                o, i, coeffs = w.inputs
                ex, ey, axy, ayx = var[coeffs.id]
                sub[w.id] = sub[o.id]
                body.append("ctx.apply_laplacian_coeff(*%s, %s, *%s, *%s, *%s, *%s);"
                            % (sub[o.id], _apply_in_arg(sub, i), ex, ey, axy, ayx))
            elif w.op == "rhs_jacvec":
                # out = J(U^k) in = in - (c*dt/h)(rhs(U^k + h*in) - rhs(U^k)), the finite-difference
                # Jacobian-vector product of the implicit-flux BDF residual (ADC-431). h is a relatively
                # scaled FD step (Brown-Saad / WP: h = eps*(1+||U^k||)/||in||, eps the relative step). The
                # captured jac_uk / jac_r0 hold U^k and rhs(U^k) (refreshed in the step body); jac_up /
                # jac_rp are per-matvec scratch; jac_cdt holds c*dt. The op writes directly into `out`.
                o, i = w.inputs[0], w.inputs[1]
                uk, r0, up, rp, cdt = jac_scratch[w.id]
                in_arg = _apply_in_arg(sub, i)        # the Krylov vector v (the lambda's const `in`)
                out_tok = sub[o.id]                   # the apply out buffer (== "out")
                eps = repr(float(w.attrs["eps"]))
                sub[w.id] = out_tok
                want_default = w.attrs.get("sources")
                want_default = want_default is None or "default" in want_default
                rhs_call = ("ctx.rhs_into(0, *%s, *%s);" if (w.attrs["flux"] and want_default)
                            else "ctx.neg_div_flux_default_into(0, *%s, *%s);") % (up, rp)
                body.append("{")
                # FD step norms via krylov_dot (all components when ncomp>1, component 0 otherwise --
                # the SAME reduction the Krylov loop uses for its residual norm).
                body.append("  const adc::Real jvn = std::sqrt(adc::detail::krylov_dot(%s, %s));"
                            % (in_arg, in_arg))
                body.append("  const adc::Real jukn = std::sqrt(adc::detail::krylov_dot(*%s, *%s));"
                            % (uk, uk))
                body.append("  const adc::Real jh = jvn > adc::Real(0) ? "
                            "static_cast<adc::Real>(%s) * (adc::Real(1) + jukn) / jvn "
                            ": static_cast<adc::Real>(%s);" % (eps, eps))
                # U^k + h*v -> jac_up; rhs(U^k + h*v) -> jac_rp (one rhs per matvec, U^k / rhs(U^k) frozen).
                body.append("  ctx.lincomb(*%s, adc::Real(1), *%s, jh, %s);" % (up, uk, in_arg))
                body.append("  %s" % rhs_call)
                # out = v - (c*dt/h)(rhs(U^k + h*v) - rhs(U^k)): lincomb then axpy back the frozen rhs(U^k).
                body.append("  const adc::Real jc = *%s / jh;" % cdt)
                body.append("  ctx.lincomb(%s, adc::Real(1), %s, -jc, *%s);" % (out_tok, in_arg, rp))
                body.append("  ctx.axpy(%s, jc, *%s);" % (out_tok, r0))
                body.append("}")
            else:
                raise NotImplementedError(
                    "emit_cpp_program: op '%s' is not lowerable inside a matrix_free_operator apply "
                    "(supported: scalar_field, laplacian, gradient, divergence, apply_laplacian_coeff, "
                    "rhs_jacvec)" % w.op)
        body += _emit_field_combine(result, "out", sub, acc_sp)
        prelude.append("adc::ApplyFn %s = [%s](adc::MultiFab& out, const adc::MultiFab& in) {"
                       % (lam, ", ".join(captures)))
        prelude += ["  " + ln for ln in body]
        prelude.append("};")

    def _emit_solve_linear(self, v, base, var, prelude, lines):
        """Lower solve_linear to a call into the runtime's matrix-free Krylov loop. The solution field
        ``sf_sol{id}`` is a PERSISTENT shared_ptr (prelude, captured by the step closure); the step body
        seeds the initial guess (zero, or a copy of the supplied guess), then calls
        ``adc::cg_solve`` / ``bicgstab_solve`` / ``richardson_solve`` with the operator's apply lambda.
        The KrylovResult is kept (diagnostics) but the trip count is decided C++-side, inside the loop --
        invisible to the IR. The result token is the solution field, dereferenced for the final copy back
        into the block state at commit."""
        op_value = v.inputs[0]
        rhs_in = v.inputs[1]
        guess_in = v.inputs[2] if v.attrs["has_guess"] else None
        lam = var[op_value.id]  # the apply lambda (already emitted into the prelude)
        sol_sp = "sf_sol%d" % v.id
        # The solution carries the operator's component count: a vector / state solve writes an ncomp
        # iterate (the Krylov scratch r/p/Ap is co-allocated from it, so the whole loop is ncomp-wide).
        op_ncomp = int(v.attrs.get("ncomp", 1))
        prelude.append(
            "auto %s = std::make_shared<adc::MultiFab>(ctx.alloc_scalar_field(%d, 1));"
            % (sol_sp, op_ncomp))
        var[v.id] = "(*%s)" % sol_sp  # token: the dereferenced solution MultiFab
        # Initial guess: zero (default) or a copy of the guess field.
        if guess_in is None:
            lines.append("%s->set_val(static_cast<adc::Real>(0));" % sol_sp)
        else:
            lines.append("ctx.lincomb(*%s, static_cast<adc::Real>(0), *%s, static_cast<adc::Real>(1), "
                         "%s);" % (sol_sp, sol_sp, var[guess_in.id]))
        tol = "static_cast<adc::Real>(%s)" % repr(float(v.attrs["tol"]))
        max_iter = int(v.attrs["max_iter"])
        rhs_tok = var[rhs_in.id]
        method = v.attrs["method"]
        kr = "kr%d" % v.id
        if method == "cg":
            lines.append("adc::KrylovResult %s = adc::cg_solve(%s, *%s, %s, %s, %d);"
                         % (kr, lam, sol_sp, rhs_tok, tol, max_iter))
        elif method == "bicgstab":
            # Identity preconditioner = empty ApplyFn (unpreconditioned BiCGStab).
            lines.append("adc::KrylovResult %s = adc::bicgstab_solve(%s, adc::ApplyFn{}, *%s, %s, %s, "
                         "%d);" % (kr, lam, sol_sp, rhs_tok, tol, max_iter))
        elif method == "gmres":
            # Restarted GMRES(m): identity preconditioner = empty ApplyFn; restart = the basis size.
            restart = int(v.attrs["restart"])
            lines.append("adc::KrylovResult %s = adc::gmres_solve(%s, adc::ApplyFn{}, *%s, %s, %s, "
                         "%d, %d);" % (kr, lam, sol_sp, rhs_tok, tol, max_iter, restart))
        else:  # richardson: omega = 1 (the operator is expected to be pre-scaled / well-conditioned)
            lines.append("adc::KrylovResult %s = adc::richardson_solve(%s, *%s, %s, "
                         "static_cast<adc::Real>(1), %s, %d);"
                         % (kr, lam, sol_sp, rhs_tok, tol, max_iter))
        lines.append("(void)%s;" % kr)


def _deref(tok):
    """C++ MultiFab-lvalue argument for a top-level (step-body) field token. Every top-level token is
    already a MultiFab lvalue expression: a state / RHS scratch (``u5``, ``r5``), a history (``h5``) or
    a dereferenced scratch scalar field (``(*sf5)``). The step-body laplacian / gradient / divergence /
    schur ops take ``adc::MultiFab&`` arguments, so the token passes through unchanged (the apply-block
    counterpart is `_apply_in_arg`, which additionally const_casts the lambda's ``in`` param)."""
    return tok


def _apply_in_arg(sub, value):
    """C++ argument for the INPUT field of a laplacian / gradient inside an apply lambda. When the input
    is the lambda's ``in`` (a const&), const_cast it (ctx.laplacian / gradient take a non-const MultiFab&
    and only write the ghosts, never the valid cells -- the same contract test_generic_krylov relies on);
    a persistent scratch shared_ptr is dereferenced."""
    tok = sub[value.id]
    if tok == "in":
        return "const_cast<adc::MultiFab&>(in)"
    return "*%s" % tok


def _emit_field_combine(result, target, sub, acc):
    """Emit C++ writing the affine combination @p result into the field @p target (a C++ MultiFab token,
    e.g. ``out``). Mirrors the linear_combine commit: zero the PERSISTENT accumulator @p acc (a scratch
    shared_ptr allocated once at install time -- no per-call/per-iteration allocation), accumulate the
    non-`target` terms onto it, then ``ctx.lincomb(target, c_target, target, 1, *acc)``. A single unit
    term that already is the target is a no-op. @p sub maps IR value ids to C++ tokens (``in``/``out``/
    scratch shared_ptrs); @p acc is the install-time accumulator shared_ptr name. Zeroing via
    ``set_val(0)`` reproduces the old ``scratch_state_like`` (a zero-initialized scratch) bit-for-bit
    over the valid cells the axpy / lincomb touch."""
    aff = _to_affine(result)._merge()
    terms = [(v, c.as_dict()) for v, c in aff]
    lines = ["%s->set_val(static_cast<adc::Real>(0));" % acc]
    c_target = {0: 0.0}
    for value, coeff in terms:
        tok = sub[value.id]
        ref = "const_cast<adc::MultiFab&>(in)" if tok == "in" else (
            "*%s" % tok if tok.startswith("sf") else tok)
        if tok == target:
            c_target = coeff
        else:
            lines.append("ctx.axpy(*%s, %s, %s);" % (acc, _coeff_cpp(coeff), ref))
    lines.append("ctx.lincomb(%s, %s, %s, static_cast<adc::Real>(1), *%s);"
                 % (target, _coeff_cpp(c_target), target, acc))
    return lines


def _coeff_cpp(powers):
    """Render a dt-polynomial coefficient (``power -> float`` dict) as a C++ ``adc::Real`` expression
    in the closure's ``dt`` parameter: ``{1: 1.0}`` -> ``static_cast<adc::Real>(dt)``,
    ``{1: 0.5}`` -> ``static_cast<adc::Real>(0.5 * dt)``, ``{0: 2.0}`` ->
    ``static_cast<adc::Real>(2.0)``. Drops a unit factor and a zero polynomial collapses to 0."""
    if not powers:
        return "static_cast<adc::Real>(0)"
    terms = []
    for power, coeff in sorted(powers.items()):
        factors = ["dt"] * int(power)
        if float(coeff) != 1.0 or not factors:
            factors = [repr(float(coeff))] + factors
        terms.append(" * ".join(factors))
    return "static_cast<adc::Real>(%s)" % " + ".join(terms)


# --- Phase-4b: lower a model's split-source / local-linear ops to per-cell C++ kernels ----------
# These helpers emit the body of a for_each_cell kernel over the VALID cells of each local fab. They
# reuse the dsl Expr -> C++ machinery (Var.to_cpp returns the bare name; we bind those names to locals)
# and the existing numerics (adc::detail::mat_inverse). A device kernel must stay heap-free /
# allocation-free: only stack scalars + fixed-size arrays, no std::vector / std::function / Eigen.

def _model_impl(model):
    """The underlying HyperbolicModel carrying the symbolic coefficients: the public adc.dsl.Model
    wraps it as ``_m``; a HyperbolicModel is already itself."""
    return getattr(model, "_m", model)


def _named_fluxes(v):
    """Resolve a ``rhs`` op's ``fluxes`` attr to the list of NAMED fluxes to assemble (ADC-419), or
    ``None`` for the historical default flux path (``ctx.rhs_into`` -- byte-identical -div F). ``None``
    or ``["default"]`` -> default path; a list of named fluxes -> that list. Mixing ``"default"`` with
    named fluxes is rejected (the centered-FV named-flux stencil differs from the Riemann rhs_into
    stencil, so they cannot be summed)."""
    fluxes = v.attrs.get("fluxes")
    if not fluxes or fluxes == ["default"]:
        return None
    named = [f for f in fluxes if f != "default"]
    if len(named) != len(fluxes):
        raise ValueError(
            "rhs '%s': fluxes mixes 'default' with named fluxes %r; request either the default flux "
            "(-div F via rhs_into) or a set of named fluxes (their -div sum), not both" % (v.name, named))
    return named


def _aux_comp(impl, name):
    """Component index of an aux field @p name in the System aux channel: canonical (dsl.AUX_CANONICAL)
    or a model NAMED aux field (dsl.AUX_NAMED_BASE + position in aux_extra_names). @p impl is the
    HyperbolicModel."""
    from . import dsl
    if name in dsl.AUX_CANONICAL:
        return dsl.AUX_CANONICAL[name]
    extra = list(getattr(impl, "aux_extra_names", []) or [])
    if name in extra:
        return dsl.AUX_NAMED_BASE + extra.index(name)
    raise NotImplementedError(
        "emit_cpp_program: aux field '%s' is neither canonical (%s) nor a declared named aux field "
        "(%s); cannot map it to an aux component" % (name, sorted(dsl.AUX_CANONICAL), extra))


def _check_no_runtime_param(exprs):
    """Phase-4b kernels read coefficients from the state / aux only (const params are inlined as
    literals by the dsl Expr tree). A RUNTIME parameter would emit ``params.get(idx)``, unavailable in
    a ProgramContext kernel -> raise NotImplementedError (deferred), never a .so that fails to link."""
    from . import dsl
    stack = list(exprs)
    while stack:
        e = stack.pop()
        if isinstance(e, dsl.RuntimeParamRef):
            raise NotImplementedError(
                "emit_cpp_program: a Phase-4b source / linear source references a RUNTIME parameter "
                "(%s); only constants and aux fields are supported in the per-cell kernel yet "
                "(runtime params in compiled programs are a later phase)" % e.name)
        stack.extend(dsl._children(e))


def _cell_locals(impl, exprs, state_var, *, with_cons, with_prim):
    """C++ local declarations binding the names the @p exprs reference to per-cell values:
      - aux fields -> ``const adc::Real <name> = auxA(i, j, <comp>);`` (always, by dependency);
      - conservative vars -> ``const adc::Real <name> = <state>A(i, j, <idx>);`` (when @p with_cons);
      - primitives -> their dsl formula, in declaration order, only the LIVE ones (when @p with_prim).
    @p impl is the HyperbolicModel; @p state_var the C++ MultiFab variable (its const Array4 is
    ``<state_var>A``, the aux Array4 is ``auxA``). Raises on a runtime-param dependency (deferred)."""
    _check_no_runtime_param(exprs)
    deps = set()
    for e in exprs:
        deps |= e.deps()
    lines = []
    live = impl._live_prims(exprs) if with_prim else set()
    # A live primitive's formula (e.g. u = mx / rho) references conservative variables that the top
    # expressions may not name directly: bind those TRANSITIVE cons too, else the emitted prim line
    # references an undeclared local. (Existing source/apply kernels read cons directly, so this only
    # ADDS the cons a live prim pulls in -- it never drops one that was already bound.)
    cons_needed = set(deps)
    for p in live:
        cons_needed |= {d for d in impl.prim_defs[p].deps() if d in impl.cons_names}
    if with_cons:
        for idx, c in enumerate(impl.cons_names):
            if c in cons_needed:
                lines.append("const adc::Real %s = %sA(i, j, %d);" % (c, state_var, idx))
    if with_prim:
        for p, expr in impl.prim_defs.items():  # declaration order (a prim may use an earlier prim)
            if p in live:
                lines.append("const adc::Real %s = %s;" % (p, expr.to_cpp()))
    aux_deps = set(impl.aux_names) | set(getattr(impl, "aux_extra_names", []) or [])
    for name in sorted(deps & aux_deps):
        lines.append("const adc::Real %s = auxA(i, j, %d);" % (name, _aux_comp(impl, name)))
    return lines


def _kernel_open(out_var, state_var):
    """Open the per-fab loop + per-cell for_each_cell over the VALID cells of @p out_var, binding the
    write handle ``outA``, the read state handle ``<state_var>A`` and the aux read handle ``auxA``.

    Pairing by local fab index ``li`` is sound: the System aux is built with the SAME box array AND
    distribution map as the blocks (``aux(ba, dm, ...)`` in System::Impl), and a scratch state comes
    from ``scratch_state_like(state(0))`` which copies that ``(ba, dm)`` -- so ``out``, the input
    state and ``aux`` share one ``(ba, dm)`` and ``fab(li)`` is the same box on every rank. This is the
    same co-distribution the existing aux-reading kernels (compiled_block_abi / source bricks) rely on."""
    return [
        "adc::MultiFab& %s_aux = ctx.aux();" % out_var,
        "for (int li = 0; li < %s.local_size(); ++li) {" % out_var,
        "  const adc::Array4 outA = %s.fab(li).array();" % out_var,
        "  const adc::ConstArray4 %sA = %s.fab(li).const_array();" % (state_var, state_var),
        "  const adc::ConstArray4 auxA = %s_aux.fab(li).const_array();" % out_var,
        "  adc::for_each_cell(%s.box(li), [=] ADC_HD(int i, int j) {" % out_var,
    ]


def _kernel_close():
    return ["  });", "}"]


# --- per-cell conditional select (spec op 17, ADC-418): model-free for_each_cell kernels --------------
# `cell_compare` and `where` are pure layout ops over co-distributed MultiFabs (no aux / no model
# coefficients): they reuse the SAME for_each_cell + Array4 per-fab pattern as the source kernels, but
# bind several read handles (no auxA) and loop over the runtime component count `<out>.ncomp()`. Pairing
# by local fab index li is sound: a cell_compare mask is alloc_scalar_field (the System (ba, dm)), a
# where scratch is scratch_state_like(a) (a's (ba, dm)) and the inputs are the same co-distributed
# states / scalar_fields, so fab(li) is the same box on every rank.


def _emit_cell_compare_kernel(field_var, mask_var, cmp, value):
    """Lower ``cell_compare``: maskA(i,j,0) = fieldA(i,j,0) <cmp> value ? 1 : 0 over the valid cells of
    the 1-component mask. Reads component 0 of @p field_var; writes the 0/1 mask into @p mask_var."""
    return [
        "for (int li = 0; li < %s.local_size(); ++li) {" % mask_var,
        "  const adc::Array4 maskA = %s.fab(li).array();" % mask_var,
        "  const adc::ConstArray4 fieldA = %s.fab(li).const_array();" % field_var,
        "  adc::for_each_cell(%s.box(li), [=] ADC_HD(int i, int j) {" % mask_var,
        "    maskA(i, j, 0) = (fieldA(i, j, 0) %s static_cast<adc::Real>(%s)) "
        "? static_cast<adc::Real>(1) : static_cast<adc::Real>(0);" % (cmp, repr(float(value))),
        "  });",
        "}",
    ]


def _emit_where_kernel(mask_var, a_var, b_var, out_var):
    """Lower ``where``: outA(i,j,c) = maskA(i,j,mc) != 0 ? aA(i,j,c) : bA(i,j,c) COMPONENT-WISE over the
    valid cells of @p out_var (out's runtime ncomp). The mask component mc is 0 when the mask is
    1-component (a shared mask) and c when the mask has the SAME ncomp as a/b (a per-component mask) --
    decided per cell from the mask's own ncomp, so both layouts lower with ONE kernel."""
    return [
        "for (int li = 0; li < %s.local_size(); ++li) {" % out_var,
        "  const adc::Array4 outA = %s.fab(li).array();" % out_var,
        "  const adc::ConstArray4 maskA = %s.fab(li).const_array();" % mask_var,
        "  const adc::ConstArray4 aA = %s.fab(li).const_array();" % a_var,
        "  const adc::ConstArray4 bA = %s.fab(li).const_array();" % b_var,
        "  const int ncomp_ = %s.ncomp();" % out_var,
        "  const int mask_ncomp_ = %s.ncomp();" % mask_var,
        "  adc::for_each_cell(%s.box(li), [=] ADC_HD(int i, int j) {" % out_var,
        "    for (int c = 0; c < ncomp_; ++c) {",
        "      const int mc = (mask_ncomp_ == 1) ? 0 : c;",
        "      outA(i, j, c) = (maskA(i, j, mc) != static_cast<adc::Real>(0)) ? aA(i, j, c) "
        ": bA(i, j, c);",
        "    }",
        "  });",
        "}",
    ]


def _emit_source_kernel(model, name, state_var, out_var):
    """Lower ``source`` (a named ``m.source_term``): outA(i,j,c) = S_c(U, prims, aux) per cell."""
    impl = _model_impl(model)
    if name not in impl._source_terms:
        raise NotImplementedError(
            "emit_cpp_program: source '%s' is not declared on the model (m.source_term); declared: %s"
            % (name, sorted(impl._source_terms)))
    exprs = impl._source_terms[name]
    body = _kernel_open(out_var, state_var)
    body += ["    " + ln for ln in _cell_locals(impl, exprs, state_var, with_cons=True,
                                                 with_prim=True)]
    body += ["    outA(i, j, %d) = %s;" % (c, e.to_cpp()) for c, e in enumerate(exprs)]
    body += _kernel_close()
    return body


def _emit_coupled_rate_kernel(components, by_block, var, scratch):
    """Lower a ``coupled_rate`` (Spec 3 criterion 27, ADC-457) to ONE multi-state for_each_cell kernel
    filling every participating block's rate scratch at once.

    @p components: ``{block: [Expr, ...]}`` -- the per-block component formulas (cons-only MVP).
    @p by_block:   ``{block: state Value}`` -- each block's input state (its StateSpace gives the cons
                   names + their component indices; its C++ token gives the read Array4).
    @p var:        the id -> C++ token map (the input states are already bound to ``ctx.state(idx)``).
    @p scratch:    ``{block: scratch var name}`` -- the per-block rate scratch (alloc'd by the caller).

    The component formulas reference cons vars from MULTIPLE input states, so the blocks share ONE loop
    (they cannot be independent single-block rates). The first block drives the loop; all inputs and
    scratches are co-located (same ba/dm as the System aux), so ``fab(li)`` is the same box on every
    rank -- the co-distribution every aux-reading kernel relies on (see _kernel_open). Each input
    state binds its OWN read handle (``<state token>A``); a referenced cons var binds from its state's
    Array4 at its component index. A cons NAME shared by two states' components AND referenced by a
    formula is ambiguous (no single source) -- rejected loud, never silently bound to one state."""
    blocks = list(components)
    driver = scratch[blocks[0]]                  # the block whose box / local_size drives the loop
    # Which cons vars does any formula reference, and from which state does each come?
    referenced = set()
    for comps in components.values():
        for e in comps:
            referenced |= e.deps()
    cons_source = {}                             # cons name -> (state token, component index)
    for st in by_block.values():                 # ALL input states, incl. read-only catalysts
        if getattr(st, "space", None) is None:
            continue
        for idx, c in enumerate(st.space.components):
            if c not in referenced:
                continue
            if c in cons_source and cons_source[c] != (var[st.id], idx):
                raise NotImplementedError(
                    "coupled_rate kernel codegen (ADC-457): cons var %r is a component of more than "
                    "one input state; the cons-only MVP needs disjoint component names across the "
                    "coupled states (rename one of them)" % (c,))
            cons_source[c] = (var[st.id], idx)

    def state_handle(token):
        return "%sA" % token                     # read handle for an input state token (u0A / u1A)

    lines = ["for (int li = 0; li < %s.local_size(); ++li) {" % driver]
    # Bind a write handle per OUTPUT block scratch, then a read handle per DISTINCT input state that a
    # formula actually reads (incl. a read-only catalyst input that is not an output block), all inside
    # the per-fab loop and BEFORE for_each_cell so the device lambda captures them by value.
    for blk in blocks:
        lines.append("  const adc::Array4 %sA = %s.fab(li).array();" % (scratch[blk], scratch[blk]))
    read_tokens = {src[0] for src in cons_source.values()}
    seen_states = []
    for st in by_block.values():                 # input order (v.inputs); deterministic
        tok = var[st.id]
        if tok in read_tokens and tok not in seen_states:
            seen_states.append(tok)
            lines.append("  const adc::ConstArray4 %s = %s.fab(li).const_array();"
                         % (state_handle(tok), tok))
    lines.append("  adc::for_each_cell(%s.box(li), [=] ADC_HD(int i, int j) {" % driver)
    for c in sorted(cons_source):                # bind only the referenced cons (no unused locals)
        tok, idx = cons_source[c]
        lines.append("    const adc::Real %s = %s(i, j, %d);" % (c, state_handle(tok), idx))
    for blk in blocks:
        for comp, e in enumerate(components[blk]):
            lines.append("    %sA(i, j, %d) = %s;" % (scratch[blk], comp, e.to_cpp()))
    lines += ["  });", "}"]
    return lines


def _emit_flux_kernel(model, names, state_var, fx_var, fy_var):
    """Lower NAMED fluxes (ADC-419): fxA(i,j,c) = sum_k F^k_x[c](U, prims, aux),
    fyA(i,j,c) = sum_k F^k_y[c](U, prims, aux) over the selected named fluxes @p names. ONE kernel
    evaluates the SUM per direction into the two n_cons flux fields (the subsequent neg_div_flux_into
    takes -div). Reuses the same per-cell local machinery as the source kernel (cons/prim/aux locals)."""
    impl = _model_impl(model)
    flux_terms = impl._flux_terms
    for name in names:
        if name not in flux_terms:
            raise NotImplementedError(
                "emit_cpp_program: flux '%s' is not declared on the model (m.flux_term); declared: %s"
                % (name, sorted(flux_terms)))
    n = len(impl.cons_names)
    x_exprs = [flux_terms[names[0]]["x"][c] for c in range(n)]
    y_exprs = [flux_terms[names[0]]["y"][c] for c in range(n)]
    for name in names[1:]:  # accumulate the additional named fluxes (their SUM is one -div)
        x_exprs = [x_exprs[c] + flux_terms[name]["x"][c] for c in range(n)]
        y_exprs = [y_exprs[c] + flux_terms[name]["y"][c] for c in range(n)]
    body = _kernel_open(fx_var, state_var)
    # fx and fy share the (ba, dm) of the scratch state, so the SAME loop / handles write both: bind a
    # second write handle to fy's local fab right after _kernel_open's outA (= fxA), still INSIDE the
    # per-fab loop and BEFORE for_each_cell so the device lambda captures it.
    body.insert(3, "  const adc::Array4 fyA = %s.fab(li).array();" % fy_var)
    body += ["    " + ln for ln in _cell_locals(impl, x_exprs + y_exprs, state_var, with_cons=True,
                                                 with_prim=True)]
    body += ["    outA(i, j, %d) = %s;" % (c, e.to_cpp()) for c, e in enumerate(x_exprs)]
    body += ["    fyA(i, j, %d) = %s;" % (c, e.to_cpp()) for c, e in enumerate(y_exprs)]
    body += _kernel_close()
    return body


def _emit_apply_kernel(model, name, state_var, out_var):
    """Lower ``apply`` (a named ``m.linear_source`` L): outA(i,j,r) = sum_c L[r][c](aux) * U(i,j,c)."""
    impl = _model_impl(model)
    rows = _linear_source_rows(impl, name)
    n = len(rows)
    flat = [e for row in rows for e in row]
    body = _kernel_open(out_var, state_var)
    # L coefficients depend on aux / const only (linear_source invariant): cons/prim locals not needed.
    body += ["    " + ln for ln in _cell_locals(impl, flat, state_var, with_cons=False,
                                                 with_prim=False)]
    for r in range(n):
        terms = ["(%s) * %sA(i, j, %d)" % (rows[r][c].to_cpp(), state_var, c) for c in range(n)]
        body.append("    outA(i, j, %d) = %s;" % (r, " + ".join(terms)))
    body += _kernel_close()
    return body


def _emit_solve_local_linear_kernel(model, name, a_coeff, rhs_var, out_var):
    """Lower ``solve_local_linear``: per cell M = I - a*L (a = a_coeff(dt)), invert M (dense N x N
    via adc::detail::mat_inverse) and set outA(i,j,r) = sum_c Minv[r][c] * q(i,j,c), q = the rhs state.
    L's coefficients depend on aux / const only, so M is assembled from the aux locals + the literal a."""
    impl = _model_impl(model)
    rows = _linear_source_rows(impl, name)
    n = len(rows)
    flat = [e for row in rows for e in row]
    a_cpp = _coeff_cpp(a_coeff)
    body = _kernel_open(out_var, rhs_var)
    body += ["    " + ln for ln in _cell_locals(impl, flat, rhs_var, with_cons=False,
                                                 with_prim=False)]
    body.append("    const adc::Real a_ = %s;" % a_cpp)
    body.append("    adc::Real M_[%d][%d];" % (n, n))
    for r in range(n):
        for c in range(n):
            ident = "adc::Real(1)" if r == c else "adc::Real(0)"
            body.append("    M_[%d][%d] = %s - a_ * (%s);" % (r, c, ident, rows[r][c].to_cpp()))
    body.append("    adc::Real Minv_[%d][%d];" % (n, n))
    # mat_inverse returns false on a singular M; we do not branch in the device kernel (no throw on
    # device). I - a*L is invertible for a well-posed local source (e.g. Lorentz: det = 1 + (a*B)^2 > 0);
    # a singular user operator yields a non-finite result that surfaces downstream, not a plausible wrong one.
    body.append("    adc::detail::mat_inverse<%d>(M_, Minv_);" % n)
    for r in range(n):
        terms = ["Minv_[%d][%d] * %sA(i, j, %d)" % (r, c, rhs_var, c) for c in range(n)]
        body.append("    outA(i, j, %d) = %s;" % (r, " + ".join(terms)))
    body += _kernel_close()
    return body


def _residual_term_exprs(impl, w):
    """The per-component Expr list of one LOCAL residual sub-block op @p w, as a function of the bare
    conservative-variable names (which the Newton kernel binds to the iterate stack ``Ueval[c]``):

      - ``source`` (a named ``m.source_term``): S_c(U) -- the declared source expressions;
      - ``apply`` (a named ``m.linear_source`` L): (L U)_c = sum_k L[c][k] * <cons_k>.

    The iterate / guess State placeholders and ``linear_combine`` are handled by the affine walk in
    `_emit_residual_eval`, not here (they are not standalone-evaluable Exprs)."""
    from . import dsl
    if w.op == "source":
        name = w.attrs["source"]
        if name not in impl._source_terms:
            raise NotImplementedError(
                "emit_cpp_program: residual source '%s' is not declared on the model (m.source_term); "
                "declared: %s" % (name, sorted(impl._source_terms)))
        return list(impl._source_terms[name])
    if w.op == "apply":
        rows = _linear_source_rows(impl, w.attrs["linear_source"])
        n = len(rows)
        # (L U)_r = sum_c L[r][c] * cons_c -- a per-component Expr in the cons names + aux.
        return [sum((rows[r][c] * dsl.Var(impl.cons_names[c], "cons") for c in range(n)),
                    dsl.Const(0.0)) for r in range(n)]
    raise NotImplementedError(
        "emit_cpp_program: residual op '%s' is not a per-cell Expr term (source / apply only)" % w.op)


def _emit_residual_eval(impl, v, n):
    """Build the device residual-evaluation lambda body for ``solve_local_nonlinear``: lines computing
    ``rout[0..n-1] = r(Ueval)`` from the iterate stack ``Ueval`` (bound to the conservative names), the
    frozen guess stack ``Gval`` (the initial-guess State, read as a per-cell constant) and the captured
    aux locals. Mirrors the affine walk: each residual sub-block op is one of the iterate / guess State
    placeholders, a ``source`` / ``apply`` per-cell Expr term, or a ``linear_combine`` (an affine over
    earlier terms). The result is the affine the residual returned.

    @p v is the solve_local_nonlinear op; @p n the conservative count. Returns the lambda BODY lines
    (indented two spaces past the lambda header). The lambda captures the aux locals + ``Gval`` by ref."""
    block = v.attrs["residual_block"]
    iterate_id = v.attrs["iterate"].id
    guess_id = v.attrs["guess"].id
    # term id -> a list of n C++ expression strings (one per conservative component). The iterate is the
    # stack Ueval; the guess is the frozen Gval; source / apply lower to Exprs over the cons names.
    comps = {iterate_id: ["Ueval[%d]" % c for c in range(n)],
             guess_id: ["Gval[%d]" % c for c in range(n)]}
    lines = []
    for w in block:
        if w.op == "state":
            continue  # the iterate / guess placeholders: bound in `comps` above, nothing to emit
        if w.op in ("source", "apply"):
            exprs = _residual_term_exprs(impl, w)
            comps[w.id] = ["(%s)" % e.to_cpp() for e in exprs]
        elif w.op == "linear_combine":
            # An affine sum over earlier terms: comps[w] = sum_k coeff_k(dt) * comps[input_k].
            coeffs = w.attrs["coeffs"]  # aligned with w.inputs; each a dt-polynomial power->float dict
            for inp in w.inputs:
                if inp.id not in comps:  # an input outside the residual sub-block (validate() guards this)
                    raise NotImplementedError(
                        "emit_cpp_program: residual combine reads value '%s' which is not produced "
                        "inside the residual (only the iterate / guess and earlier residual ops are "
                        "available to a per-cell Newton kernel)" % inp.name)
            row = []
            for c in range(n):
                parts = []
                for inp, coeff in zip(w.inputs, coeffs, strict=True):
                    parts.append("%s * (%s)" % (_coeff_cpp(coeff), comps[inp.id][c]))
                row.append(" + ".join(parts) if parts else "static_cast<adc::Real>(0)")
            comps[w.id] = row
        else:  # builder guards _RESIDUAL_LOCAL_OPS; this is belt-and-suspenders
            raise NotImplementedError(
                "emit_cpp_program: residual op '%s' is not lowerable in a local Newton kernel" % w.op)
    result = comps[v.attrs["residual"].id]
    for c in range(n):
        lines.append("rout[%d] = %s;" % (c, result[c]))
    return lines


def _emit_solve_local_nonlinear_kernel(model, v, guess_var, out_var):
    """Lower ``solve_local_nonlinear`` (spec op 10) to a per-cell Newton kernel: from the initial guess
    U0 (the @p guess_var state), iterate ``J dU = -r``, ``U -= dU`` until ``max_c |r_c| < tol`` or the
    fixed budget, then write the converged U into @p out_var. Reuses ``adc::for_each_cell`` + the SAME
    stack dense inverse ``adc::detail::mat_inverse<N>`` as `solve_local_linear` -- no heap / std::function
    / Eigen in the device kernel (only stack scalars + fixed ``[N]`` / ``[N][N]`` arrays).

    The residual is evaluated by an inlined device lambda built from the residual sub-block (the
    iterate stack, the frozen guess, named ``source`` / ``apply`` per-cell Exprs, affine combines). The
    Jacobian is finite-difference: column j perturbs ``U[j] += eps`` and forms ``(r(U+eps e_j)-r(U))/eps``
    with a relative ``eps`` so it scales with the iterate magnitude."""
    impl = _model_impl(model)
    n = len(impl.cons_names)
    tol = repr(float(v.attrs["tol"]))
    max_iter = int(v.attrs["max_iter"])
    # The aux fields the residual reads: bind them once per cell (constant across the Newton iterates),
    # so the residual lambda captures them by reference. Gather the dependency set over every term Expr.
    term_exprs = []
    for w in v.attrs["residual_block"]:
        if w.op in ("source", "apply"):
            term_exprs += _residual_term_exprs(impl, w)
    body = _kernel_open(out_var, guess_var)
    # Per-cell aux + the live primitives are NOT pre-bound here: the prims depend on the ITERATE (they
    # are recomputed inside the residual lambda from Ueval). Only the aux locals are cell constants.
    body += ["    " + ln for ln in _cell_locals(impl, term_exprs, guess_var, with_cons=False,
                                                with_prim=False)]
    # The frozen initial guess as a stack vector (the residual reads it as a per-cell constant).
    body.append("    adc::Real Gval[%d];" % n)
    for c in range(n):
        body.append("    Gval[%d] = %sA(i, j, %d);" % (c, guess_var, c))
    # The residual-eval lambda r(Ueval) -> rout (device, stack-only, no std::function): captures the
    # cell-constant aux locals + the frozen guess by reference; recomputes the iterate-dependent
    # primitives inside from Ueval (bound to the conservative names).
    body.append("    auto residual_eval = [&](const adc::Real (&Ueval)[%d], adc::Real (&rout)[%d]) {"
                % (n, n))
    for c, cn in enumerate(impl.cons_names):
        body.append("      const adc::Real %s = Ueval[%d];" % (cn, c))
    # Live primitives of the residual terms, in declaration order (a prim may use an earlier prim).
    live = impl._live_prims(term_exprs) if term_exprs else set()
    for p, expr in impl.prim_defs.items():
        if p in live:
            body.append("      const adc::Real %s = %s;" % (p, expr.to_cpp()))
    body += ["      " + ln for ln in _emit_residual_eval(impl, v, n)]
    body.append("    };")
    # Newton state: the iterate U_ (seeded to the guess), the residual r_, the FD Jacobian J_ and step.
    body.append("    adc::Real U_[%d];" % n)
    for c in range(n):
        body.append("    U_[%d] = Gval[%d];" % (c, c))
    body.append("    adc::Real r_[%d];" % n)
    body.append("    for (int it_ = 0; it_ < %d; ++it_) {" % max_iter)
    body.append("      residual_eval(U_, r_);")
    # Convergence on max_c |r_c| (the per-cell residual infinity norm).
    body.append("      adc::Real rmax_ = adc::Real(0);")
    body.append("      for (int c_ = 0; c_ < %d; ++c_) rmax_ = std::fmax(rmax_, std::fabs(r_[c_]));" % n)
    body.append("      if (rmax_ < static_cast<adc::Real>(%s)) break;" % tol)
    # FD Jacobian J_[i][j] = (r_i(U + eps e_j) - r_i(U)) / eps, eps relative to |U_j| (floored).
    body.append("      adc::Real J_[%d][%d];" % (n, n))
    body.append("      adc::Real Up_[%d];" % n)
    body.append("      adc::Real rp_[%d];" % n)
    body.append("      for (int j_ = 0; j_ < %d; ++j_) {" % n)
    body.append("        for (int k_ = 0; k_ < %d; ++k_) Up_[k_] = U_[k_];" % n)
    body.append("        const adc::Real eps_ = static_cast<adc::Real>(1e-7) "
                "* std::fmax(std::fabs(U_[j_]), static_cast<adc::Real>(1));")
    body.append("        Up_[j_] += eps_;")
    body.append("        residual_eval(Up_, rp_);")
    body.append("        for (int i_ = 0; i_ < %d; ++i_) J_[i_][j_] = (rp_[i_] - r_[i_]) / eps_;" % n)
    body.append("      }")
    # Newton step J dU = -r via the SAME stack dense inverse solve_local_linear uses; U -= dU.
    body.append("      adc::Real Jinv_[%d][%d];" % (n, n))
    body.append("      adc::detail::mat_inverse<%d>(J_, Jinv_);" % n)
    body.append("      for (int i_ = 0; i_ < %d; ++i_) {" % n)
    body.append("        adc::Real du_ = adc::Real(0);")
    body.append("        for (int k_ = 0; k_ < %d; ++k_) du_ += Jinv_[i_][k_] * r_[k_];" % n)
    body.append("        U_[i_] -= du_;")
    body.append("      }")
    body.append("    }")
    for c in range(n):
        body.append("    outA(i, j, %d) = U_[%d];" % (c, c))
    body += _kernel_close()
    return body


def _linear_source_rows(impl, name):
    """The n_cons x n_cons matrix of Expr of a model linear source @p name (m.linear_source).
    @p impl is the HyperbolicModel."""
    if name not in impl._linear_sources:
        raise NotImplementedError(
            "emit_cpp_program: linear source '%s' is not declared on the model (m.linear_source); "
            "declared: %s" % (name, sorted(impl._linear_sources)))
    return impl._linear_sources[name]


# Source of a generated problem.so. The includes + adc_install_program closure match the shape
# tests/test_program_loader compiles+runs in CI; adc_program_hash is added per the spec .so ABI (a
# cache/restart key) and is not yet consumed by System::install_program. {name} is a JSON-escaped C
# string literal, {hash} the IR hash, {prelude} the INSTALL-TIME C++ (persistent scratch + matrix-free
# apply lambdas, captured into the step closure by [=]), {body} the step-closure body (both already
# indented); the literal braces of the C++ scaffold are doubled for str.format.
_PROGRAM_CPP_TEMPLATE = '''\
// GENERATED by adc.time.Program.emit_cpp_program (epic ADC-399 / ADC-401). Do not edit by hand.
// A compiled time Program installed across the stable .so ABI: it drives sim.step(dt) entirely in
// C++ via ProgramContext, reusing the adc_cpp runtime (no MultiFab / flux / solver reimplementation).
#include <adc/runtime/program/program_context.hpp>
#include <adc/runtime/dynamic/abi_key.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/mesh/storage/fab2d.hpp>          // Array4 / ConstArray4 (per-cell handles)
#include <adc/mesh/execution/for_each.hpp>     // for_each_cell (Phase-4b per-cell kernels)
#include <adc/numerics/linalg/dense_eig.hpp>   // adc::detail::mat_inverse (local dense solve)
#include <adc/numerics/elliptic/linear/generic_krylov.hpp>  // adc::cg_solve / bicgstab_solve / richardson_solve / gmres_solve (matrix-free)
#include <adc/core/foundation/types.hpp>
#include <chrono>                              // std::chrono::steady_clock (per-node profiling pair, ADC-459)
#include <cmath>                               // std::sqrt / std::fabs / std::pow in lowered formulas
#include <limits>                              // std::numeric_limits (dt_bound +inf sentinel)
#include <memory>                              // std::make_shared (persistent matrix-free scratch)
#include <vector>                              // pointer list for the coupled multi-block field-solve (ADC-457)

extern "C" const char* adc_program_abi_key() {{ return ADC_ABI_KEY_LITERAL; }}
extern "C" const char* adc_program_name() {{ return {name}; }}
extern "C" const char* adc_program_hash() {{ return "{hash}"; }}

{block_names}
{module_metadata}
extern "C" void adc_install_program(void* sys) {{
  adc::runtime::program::ProgramContext ctx(sys);
{prelude}
  ctx.install([=](double dt) {{
    (void)dt;
{body}
  }});
}}

// OPTIONAL dt bound (spec s18 / ADC-417). adc_program_has_dt_bound() is true iff the Program set one;
// adc_program_dt_bound(ctx, cfl) returns the lowered scalar bound (min'd into the native CFL by
// step_cfl). When no bound was set, has_dt_bound() is false and dt_bound returns +inf (unreached).
extern "C" bool adc_program_has_dt_bound() {{ return {has_dt_bound}; }}
extern "C" adc::Real adc_program_dt_bound(adc::runtime::program::ProgramContext* ctxp, adc::Real cfl) {{
  adc::runtime::program::ProgramContext& ctx = *ctxp;
  (void)ctx; (void)cfl;
{dt_bound_body}
}}
'''


# --- Standard library: time-stepping macros that LOWER to the Program IR (adc.time.std, ADC-407) ----
# These are NOT separate C++ steppers: each builds adc.time.Program IR via the same builder ops + the
# affine algebra over dt, so a scheme is expressed ONCE with no scheme-specific class (spec acceptance
# 25-29; RK4 has no special RK4 class). The generated problem.so (compile_problem, Phase 2c) executes
# the lowered IR. forward_euler / ssprk2 / ssprk3 reproduce adc.Explicit(method="euler"/"ssprk2"/"ssprk3").
def _stage_rhs(P, U, sources, flux):
    """Solve the elliptic fields from U and assemble its RHS for one stage. The FieldContext is
    distinct per stage (no stale global aux). flux=False builds a source-only sub-flow (e.g. Strang S)."""
    fields = P.solve_fields(U) if flux else None
    return P.rhs(state=U, fields=fields, flux=flux, sources=list(sources))


def forward_euler(P, block, *, sources=("default",), flux=True):
    """Forward Euler: U^{n+1} = U + dt * R(U)."""
    U = P.state(block)
    R = _stage_rhs(P, U, sources, flux)
    P.commit(block, P.linear_combine("fe_step", U + P.dt * R))


def ssprk2(P, block, *, sources=("default",), flux=True):
    """SSPRK2 (Heun / Shu-Osher): U1 = U0 + dt k0; U^{n+1} = 1/2 U0 + 1/2 (U1 + dt k1)."""
    U0 = P.state(block)
    k0 = _stage_rhs(P, U0, sources, flux)
    U1 = P.linear_combine("ssprk2_U1", U0 + P.dt * k0)
    k1 = _stage_rhs(P, U1, sources, flux)
    P.commit(block, P.linear_combine("ssprk2_step", 0.5 * U0 + 0.5 * (U1 + P.dt * k1)))


def ssprk3(P, block, *, sources=("default",), flux=True):
    """SSPRK3 (Shu-Osher): U1 = U0 + dt k0; U2 = 3/4 U0 + 1/4 (U1 + dt k1);
    U^{n+1} = 1/3 U0 + 2/3 (U2 + dt k2)."""
    U0 = P.state(block)
    k0 = _stage_rhs(P, U0, sources, flux)
    U1 = P.linear_combine("ssprk3_U1", U0 + P.dt * k0)
    k1 = _stage_rhs(P, U1, sources, flux)
    U2 = P.linear_combine("ssprk3_U2", 0.75 * U0 + 0.25 * (U1 + P.dt * k1))
    k2 = _stage_rhs(P, U2, sources, flux)
    P.commit(block, P.linear_combine("ssprk3_step", (1.0 / 3.0) * U0 + (2.0 / 3.0) * (U2 + P.dt * k2)))


def rk4(P, block, *, sources=("default",), flux=True):
    """Classic RK4, expressed with NO special RK4 class (spec acceptance 29):
    U^{n+1} = U0 + dt/6 (k1 + 2 k2 + 2 k3 + k4)."""
    U0 = P.state(block)
    k1 = _stage_rhs(P, U0, sources, flux)
    U1 = P.linear_combine("rk4_U1", U0 + 0.5 * P.dt * k1)
    k2 = _stage_rhs(P, U1, sources, flux)
    U2 = P.linear_combine("rk4_U2", U0 + 0.5 * P.dt * k2)
    k3 = _stage_rhs(P, U2, sources, flux)
    U3 = P.linear_combine("rk4_U3", U0 + P.dt * k3)
    k4 = _stage_rhs(P, U3, sources, flux)
    P.commit(block, P.linear_combine(
        "rk4_step", U0 + P.dt / 6.0 * k1 + P.dt / 3.0 * k2 + P.dt / 3.0 * k3 + P.dt / 6.0 * k4))


# Adams-Bashforth weights b_j on R_{n-j} (j = 0..order-1), per order (ADC-423). AB1 is Forward Euler.
_AB_WEIGHTS = {
    1: (1.0,),
    2: (1.5, -0.5),                       # 3/2, -1/2
    3: (23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0),
}


def adams_bashforth(P, block, order, *, sources=("default",), flux=True):
    """Adams-Bashforth, explicit ``order``-step, over the System-owned history ring (ADC-406a / ADC-423):

        R_n     = R(U)
        U^{n+1} = U + dt * sum_{j=0}^{order-1} b_j * R_{n-j}
        store_history(block.R, R_n)

    ``order`` selects the classic AB weights b_j:
      - **AB1** == Forward Euler (b = 1), with NO history (it never reads or stores the ring);
      - **AB2** == (3/2, -1/2) on (R_n, R_{n-1});
      - **AB3** == (23/12, -16/12, 5/12) on (R_n, R_{n-1}, R_{n-2}).

    COLD START: the store of R_n is recorded BEFORE the lag reads, and the runtime fills EVERY history
    slot on the FIRST store, so step 0 reads R_{n-j} = R_0 for all j and the recurrence degenerates to a
    single Forward-Euler step (U^1 = U^0 + dt*R_0, since sum_j b_j = 1). From step ``order-1`` on it is
    the true AB recurrence; in between it runs the same partially-filled ring the runtime exposes. This
    is deterministic and exact; an offline reference mirrors it (FE-fill cold start then AB). The history
    name is ``"<block>.R"`` (the block's previous RHS).

    AB1 keeps Forward Euler's exact IR (no history op); AB2 keeps the historical ``"ab2_step"`` combine
    so a pre-ADC-423 AB2 program's ``.so`` cache key is byte-identical."""
    if isinstance(order, bool) or not isinstance(order, int) or order not in _AB_WEIGHTS:
        raise ValueError("adams_bashforth: order must be an int in %s (got %r)"
                         % (sorted(_AB_WEIGHTS), order))
    b = _AB_WEIGHTS[order]
    if order == 1:  # AB1 == Forward Euler: no history, identical IR to forward_euler.
        forward_euler(P, block, sources=sources, flux=flux)
        return
    name = block + ".R"
    step_name = "ab2_step" if order == 2 else ("ab%d_step" % order)
    U = P.state(block)
    R_n = _stage_rhs(P, U, sources, flux)
    # Store R_n FIRST (so the first store cold-start-fills the ring), then read R_{n-j} = lag j.
    P.store_history(name, R_n)
    expr = U + (P.dt * b[0]) * R_n
    for j in range(1, order):
        expr = expr + (P.dt * b[j]) * P.history(name, lag=j)
    P.commit(block, P.linear_combine(step_name, expr))


def adams_bashforth2(P, block, *, sources=("default",), flux=True):
    """Adams-Bashforth 2, a thin back-compat alias for ``adams_bashforth(P, block, 2)`` (ADC-423).

    Kept so existing callers and the historical ``"ab2_step"`` IR are unchanged: this lowers to the
    SAME IR as before (R_n stored first, R_{n-1} read at lag 1, weights 3/2 / -1/2)."""
    adams_bashforth(P, block, 2, sources=sources, flux=flux)


def strang(P, block, half_flow, source, *, commit=True):
    """Strang splitting macro H(dt/2); S(dt); H(dt/2), the macro form of adc.Strang (lowers to the SAME
    IR, no special class). @p half_flow and @p source are IR-building callables (prog, state, frac) ->
    state that advance the hyperbolic flow and the source by a fraction @p frac of dt. Returns the final
    state (committed when @p commit)."""
    U = P.state(block)
    U1 = half_flow(P, U, 0.5)
    U2 = source(P, U1, 1.0)
    U3 = half_flow(P, U2, 0.5)
    if commit:
        P.commit(block, U3)
    return U3


def condensed_schur(P, block, *, alpha, theta=1.0, c_rho=0, c_mx=1, c_my=2, c_bz=3, c_E=None,
                    method="bicgstab", tol=1e-10, max_iter=400, commit=True):
    """Condensed-Schur implicit electrostatic-Lorentz SOURCE stage as a compiled Program (epic ADC-399,
    acceptance 32), mirroring the native ``adc.CondensedSchur`` (CondensedSchurSourceStepper) sequence:

      1. assemble the anisotropic tensor coefficient ``A = I + c*rho*B^{-1}`` (``P.schur_coeffs``,
         ``c = theta^2 dt^2 alpha``);
      2. assemble the fused RHS ``-Lap(phi^n) - theta*dt*alpha*div(B^{-1}(mx,my))`` (``P.schur_rhs``);
      3. solve ``-div(A grad phi^{n+theta}) = RHS`` matrix-free (``P.matrix_free_operator`` +
         ``P.apply_laplacian_coeff`` negated, ``P.solve_linear``), warm-started from phi^n;
      4. reconstruct ``v^{n+theta} = B^{-1}(v^n - theta*dt*grad phi)`` and write ``mom = rho*v``
         (``P.schur_reconstruct``, the closed B^{-1}); rho stays frozen;
      5. (``theta < 1``) extrapolate the theta-stage state to ``n+1`` by the native factor ``1/theta``:
         ``U^{n+1} = U^n + (1/theta)(U^{n+theta} - U^n)`` (the affine algebra, see THETA below);
      6. (``c_E`` given) update the total energy ``E^{n+1} = E^n + (1/2)rho(|v^{n+1}|^2 - |v^n|^2)``
         (``P.schur_energy``, the native kinetic-energy increment).

    phi^n is a fresh zero scalar field each step (NO persistent history -- see the cross-step carry note
    under DEFERRED below). The phi solve runs to tolerance and the velocity reconstruction reads only the
    solved phi^{n+theta}, so a single step matches the native single step taken from ``phi^n = 0`` (the
    System also initializes phi to zero). Every numerical kernel REUSES a native primitive (no stencil /
    B^{-1} / elimination reimplementation); the native ``adc.CondensedSchur`` stepper is untouched.

    THETA != 1 (ADC-427). The native stepper takes the implicit stage at ``n+theta`` and extrapolates phi
    and the MOMENTUM (not rho) to ``n+1`` by the factor ``1/theta``. This macro lowers that extrapolation
    with the EXISTING affine algebra, no component-restricted IR op: ``schur_reconstruct`` freezes rho
    (and energy), so ``rho^{n+theta} = rho^n`` and ``mom^{n+theta} = rho v^{n+theta}``,
    ``mom^n = rho v^n``. The plain STATE affine ``U^n + (1/theta)(U^{n+theta} - U^n)`` therefore leaves
    rho (and a yet-unwritten energy) untouched -- ``rho^{n+1} = (1-1/theta)rho^n + (1/theta)rho^n =
    rho^n`` -- and on the momentum it equals the native ``mom^{n+1} = mom^n + (1/theta)(mom^{n+theta} -
    mom^n) = rho(v^n + (1/theta)(v^{n+theta} - v^n))``. The phi extrapolation is a no-op here because
    phi^n = 0 (no carry) and the reconstruction already read phi^{n+theta}; phi^{n+1} would only matter
    as the NEXT step's warm start (the deferred persistent-phi carry).

    @p alpha is the electrostatic coupling constant; @p theta the theta-scheme implicitness in ``(0, 1]``;
    @p c_rho / @p c_mx / @p c_my the conserved-variable components, @p c_bz the aux component of B_z
    (canonical 3, filled by ``solve_fields``) and @p c_E the OPTIONAL energy component (None = no energy
    update, like a rho/mx/my isothermal block). @p method / @p tol / @p max_iter configure the Krylov phi
    solve.

    DEFERRED (documented partial, spec's "if too large" clause):
      - **cross-step phi^n carry**. The native stepper freezes phi^n (the previous stage's potential)
        and keeps it in the RHS (``-Lap(phi^n)``) and as the solve warm start. The System history ring
        is sized to the block's ncomp (a full state) and stores via ``adc::lincomb`` (matching ncomp), so
        a 1-component phi cannot be carried through it without a scalar-history runtime path (a new
        ncomp-aware ``register_history`` + scalar-typed history IR ops + the extrapolated-phi store/read
        dataflow) -- a runtime change too large for this slice. This macro therefore solves each step
        from ``phi^n = 0`` (a fresh zero scalar field). At theta != 1 the FIRST step still matches the
        native first step (both start from phi = 0); the cross-step difference is the warm-start /
        ``-Lap(phi^n)`` term the native stepper carries (a smoother convergence, the same fixed point).

    NEAR-MATCH to native, not bit-exact: the native solve is BiCGStab + GeometricMG preconditioner while
    the Program solve is matrix-free BiCGStab WITHOUT a preconditioner -- the SAME operator and RHS, a
    different Krylov path (both converge to the same phi at tolerance). ``python/tests/
    test_time_condensed_schur.py`` checks against an offline reference of the identical assemble / solve
    / reconstruct / extrapolate steps and documents the gap vs native (theta == 1 and theta == 0.5)."""
    if not (0.0 < float(theta) <= 1.0):
        raise ValueError("condensed_schur: theta must be in (0, 1] (got %r)" % (theta,))
    if c_E is not None and (isinstance(c_E, bool) or not isinstance(c_E, int) or c_E < 0):
        raise ValueError("condensed_schur: c_E must be None or a Python int >= 0 (got %r)" % (c_E,))
    U = P.state(block)
    P.solve_fields(U)  # fill the shared aux (B_z at c_bz) from the current state, like the native stage
    # phi^n = 0 (a fresh zero scalar field): the RHS Laplacian term -Lap(phi^n) vanishes and the solve
    # warm starts from zero. Cross-step phi^n carry is deferred (see the docstring).
    phi_n = P.scalar_field(block + ".schur_phi_n")
    c_coeff = (float(theta) * float(theta) * float(alpha)) * P.dt * P.dt  # c = theta^2 dt^2 alpha
    th_dt = float(theta) * P.dt  # theta dt
    g = (float(theta) * float(alpha)) * P.dt  # theta dt alpha (coefficient of the div(F) term)
    coeffs = P.schur_coeffs(state=U, c=c_coeff, th_dt=th_dt, c_rho=c_rho, c_bz=c_bz)
    rhs = P.scalar_field(block + ".schur_rhs")
    P.schur_rhs(rhs, phi_n, U, th_dt, g, c_mx=c_mx, c_my=c_my, c_bz=c_bz)
    A = P.matrix_free_operator(block + ".schur_op")

    def apply(P, out, x):  # out <- A(x) = -div((I + c rho B^{-1}) grad x) = -apply_laplacian_coeff(x)
        lap = P.scalar_field("schur_lap")
        P.apply_laplacian_coeff(lap, x, coeffs)
        return -1.0 * lap  # the condensed operator -div(A grad phi); the affine is the lowered result

    P.set_apply(A, apply)
    phi = P.solve_linear(operator=A, rhs=rhs, method=method, tol=tol, max_iter=max_iter)
    # The reconstruction overwrites the MOMENTUM in place. theta == 1 with no energy keeps the historical
    # IR byte-identical (reconstruct directly on U). For theta < 1 OR an energy update we need U^n
    # (mom^n / E^n) AFTER the reconstruction, so reconstruct on a fresh COPY of U^n and keep U^n intact.
    needs_un = float(theta) != 1.0 or c_E is not None
    target = P.linear_combine(block + ".schur_un_copy", 1.0 * U) if needs_un else U
    out = P.schur_reconstruct(state=target, phi=phi, th_dt=th_dt, c_rho=c_rho, c_mx=c_mx, c_my=c_my,
                              c_bz=c_bz)
    # 5) theta-stage -> n+1 extrapolation (ADC-427). theta < 1 lowers U^n + (1/theta)(U^{n+theta} - U^n)
    # with the affine algebra (out is the theta-stage on the copy, U^n is the untouched original). rho is
    # frozen by the reconstruction, so this affine leaves rho (and the not-yet-written energy) at U^n.
    if float(theta) != 1.0:
        inv_theta = 1.0 / float(theta)
        out = P.linear_combine(block + ".schur_extrap", U + inv_theta * (out - U))
    # 6) energy role (ADC-427). E^{n+1} = E^n + (1/2)rho(|v^{n+1}|^2 - |v^n|^2): the kinetic-energy
    # increment from v^n (= mom^n/rho, read from U^n) to v^{n+1} (= mom^{n+1}/rho, in `out`). Skipped
    # for an isothermal rho/mx/my block (c_E is None).
    if c_E is not None:
        out = P.schur_energy(state=out, state_old=U, c_rho=c_rho, c_mx=c_mx, c_my=c_my, c_E=c_E)
    if commit:
        P.commit(block, out)
    return out


def lie(P, block, half_flow, source, *, commit=True):
    """Lie (Godunov) splitting macro H(dt); S(dt) -- the sequential first-order sibling of `strang`
    (ADC-423). @p half_flow and @p source are the SAME IR-building callables `strang` takes
    ``(prog, state, frac) -> state`` (each advances its sub-flow by a fraction @p frac of dt); Lie
    just composes them sequentially over the FULL step (H over dt, then S over dt) with no half-steps.
    Lowers to the SAME IR primitives as `strang` (no scheme-specific class). Returns the final state
    (committed when @p commit)."""
    U = P.state(block)
    U1 = half_flow(P, U, 1.0)
    U2 = source(P, U1, 1.0)
    if commit:
        P.commit(block, U2)
    return U2


# Classic explicit Butcher tableaux (A lower-triangular, b weights, c nodes) for `rk` (ADC-423).
class ButcherTableau:
    """An explicit Butcher tableau ``(A, b, c)`` for `rk`: ``A`` is strictly lower-triangular (stage i
    depends only on stages j < i), ``b`` the final weights, ``c`` the (unused-by-the-lowering) nodes.
    Validated as explicit and consistent (``len(A) == len(b)``, row i has i entries, ``sum(b) == 1``)."""

    def __init__(self, A, b, c=None, name=None):
        self.A = [list(row) for row in A]
        self.b = list(b)
        self.c = list(c) if c is not None else [sum(row) for row in self.A]
        self.name = name
        s = len(self.b)
        if len(self.A) != s or len(self.c) != s:
            raise ValueError("ButcherTableau: A, b, c must share the stage count")
        for i, row in enumerate(self.A):
            if len(row) > i and any(row[j] != 0.0 for j in range(i, len(row))):
                raise ValueError(
                    "ButcherTableau: A must be strictly lower-triangular (stage %d reads stage >= %d); "
                    "rk lowers EXPLICIT tableaux only" % (i, i))
        if abs(sum(self.b) - 1.0) > 1e-12:
            raise ValueError("ButcherTableau: weights b must sum to 1 (got %r)" % (sum(self.b),))

    @property
    def stages(self):
        return len(self.b)


# RK4 (classic): the same tableau the rk4 macro hard-codes, written data-driven.
RK4_TABLEAU = ButcherTableau(
    A=[[],
       [0.5],
       [0.0, 0.5],
       [0.0, 0.0, 1.0]],
    b=[1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0],
    c=[0.0, 0.5, 0.5, 1.0],
    name="rk4")

# SSPRK2 (Heun) in NON-Shu-Osher Butcher form: k1 at U, k2 at U+dt*k1, U^{n+1}=U+dt(1/2 k1+1/2 k2).
SSPRK2_TABLEAU = ButcherTableau(
    A=[[],
       [1.0]],
    b=[0.5, 0.5],
    c=[0.0, 1.0],
    name="ssprk2")


def rk(P, block, tableau, *, sources=("default",), flux=True):
    """Generic explicit Runge-Kutta from a Butcher @p tableau (ADC-423), lowered to the SAME stage chain
    the hard-coded `rk4` macro emits -- ``solve_fields`` + ``rhs`` + ``linear_combine``, no RK class:

        k_i      = R( U + dt * sum_{j<i} A[i][j] * k_j )       (the i-th stage RHS)
        U^{n+1}  = U + dt * sum_i b[i] * k_i

    @p tableau is a `ButcherTableau` (or a raw ``(A, b, c)`` triple); ``A`` must be strictly
    lower-triangular (explicit). ``RK4_TABLEAU`` and ``SSPRK2_TABLEAU`` are provided as the classic
    constants: ``rk(P, blk, RK4_TABLEAU)`` builds the identical final affine combination as
    ``rk4(P, blk)`` (a permutation of the same ``U0 + dt(1/6 k1 + 1/3 k2 + 1/3 k3 + 1/6 k4)`` inputs),
    and ``rk(P, blk, SSPRK2_TABLEAU)`` matches Heun's ``U + dt(1/2 k1 + 1/2 k2)``."""
    if not isinstance(tableau, ButcherTableau):
        A, b, c = tableau if len(tableau) == 3 else (tableau[0], tableau[1], None)
        tableau = ButcherTableau(A, b, c)
    tag = (tableau.name + "_") if tableau.name else "rk_"
    U0 = P.state(block)
    ks = []
    for i in range(tableau.stages):
        if i == 0:
            Ui = U0  # the first stage reads U^n directly (no scratch combine, like rk4)
        else:
            expr = U0
            for j in range(i):
                aij = tableau.A[i][j]
                if aij != 0.0:
                    expr = expr + (P.dt * aij) * ks[j]
            Ui = P.linear_combine("%sU%d" % (tag, i), expr)
        ks.append(_stage_rhs(P, Ui, sources, flux))
    final = U0
    for i in range(tableau.stages):
        bi = tableau.b[i]
        if bi != 0.0:
            final = final + (P.dt * bi) * ks[i]
    P.commit(block, P.linear_combine("%sstep" % tag, final))


def imex_local(P, block, *, linear_source, sources=("default",), flux=True, theta=1.0):
    """IMEX with an EXPLICIT flux/source and an IMPLICIT cell-local linear source (ADC-423).

    One step of a theta-implicit splitting of ``dU/dt = R_explicit(U) + L U`` where ``L`` is a named
    model ``m.linear_source`` (e.g. a Lorentz operator) solved cell by cell:

        R   = R_explicit(U)                                     (P.rhs: -div F + the named sources)
        U^{n+1} = (I - theta*dt*L)^{-1} (U + dt*R)              (P.solve_local_linear)

    The explicit part is assembled with `P.rhs` (flux + the requested named @p sources, on the fields
    solved from U); the implicit part is the local solve of ``(I - theta*dt*L) U^{n+1} = U + dt*R``
    via `P.solve_local_linear`, exactly the predictor half of the codebase's predictor-corrector
    pattern (``test_time_local_solve``). At ``theta == 1`` this is backward Euler on the L term and
    forward Euler on R; ``theta == 0`` would drop the implicit solve (use `forward_euler` instead) and
    is rejected. @p linear_source is the name of the model ``m.linear_source``; @p theta the
    implicitness of the L term (0 < theta <= 1)."""
    if not (isinstance(linear_source, str) and linear_source):
        raise ValueError("imex_local: linear_source must be a non-empty m.linear_source name")
    if not (0.0 < float(theta) <= 1.0):
        raise ValueError(
            "imex_local: theta must be in (0, 1] (got %r); theta == 0 is fully explicit -- use "
            "forward_euler instead" % (theta,))
    U = P.state(block)
    fields = P.solve_fields(U) if flux else None
    R = P.rhs(state=U, fields=fields, flux=flux, sources=list(sources))
    rhs = P.linear_combine(block + "_imex_rhs", U + P.dt * R)
    operator = P.I - (float(theta) * P.dt) * P.linear_source(linear_source)
    out = P.solve_local_linear(name=block + "_imex_step", operator=operator, rhs=rhs, fields=fields)
    P.commit(block, out)
    return out


def _bdf_local_linear(P, block, order, linear_source, sources, flux):
    """The cell-LOCAL linear-source BDF fast path (the historical lowering): the BDF system is
    block-diagonal, so ``(c0*I - dt*L) U^{n+1} = rhs`` is solved per cell by `P.solve_local_linear`.

      - **BDF1** (backward Euler): ``(I - dt*L) U^{n+1} = U^n [+ dt R]``;
      - **BDF2**: ``(I - (2/3) dt L) U^{n+1} = (2/3)(2 U^n - 1/2 U^{n-1}) [+ dt R]`` over the System
        history ring, with a BDF1 cold start (the first store fills every slot -> U^{n-1} = U^n)."""
    U = P.state(block)
    fields = P.solve_fields(U) if flux else None
    # Optional EXPLICIT flux/source RHS folded into the BDF right-hand side (lagged at U^n).
    R = P.rhs(state=U, fields=fields, flux=flux, sources=list(sources)) if (flux or sources) else None

    def _with_explicit(expr):
        return (expr + P.dt * R) if R is not None else expr

    if order == 1:  # (I - dt*L) U^{n+1} = U^n [+ dt R]
        rhs = P.linear_combine(block + "_bdf1_rhs", _with_explicit(1.0 * U))
        operator = P.I - P.dt * P.linear_source(linear_source)
        out = P.solve_local_linear(name=block + "_bdf1_step", operator=operator, rhs=rhs, fields=fields)
        P.commit(block, out)
        return out
    # BDF2: (3/2 I - dt*L) U^{n+1} = 2 U^n - 1/2 U^{n-1} [+ dt R], over the history ring.
    name = block + ".U"
    P.store_history(name, U)                       # store U^n first (cold-start fills the ring)
    U_nm1 = P.history(name, lag=1)                 # U^{n-1} (== U^n on step 0 -> BDF1 cold start)
    rhs = P.linear_combine(block + "_bdf2_rhs", _with_explicit(2.0 * U - 0.5 * U_nm1))
    operator = P.I - (P.dt * (2.0 / 3.0)) * P.linear_source(linear_source)
    # Divide both sides by 3/2: (I - (2/3) dt L) U^{n+1} = (2/3)(2 U^n - 1/2 U^{n-1} [+ dt R]).
    rhs = P.linear_combine(block + "_bdf2_rhs_scaled", (2.0 / 3.0) * rhs)
    out = P.solve_local_linear(name=block + "_bdf2_step", operator=operator, rhs=rhs, fields=fields)
    P.commit(block, out)
    return out


def _bdf_implicit_flux(P, block, order, sources, flux, ncomp, newton_tol, newton_max, krylov_tol,
                       krylov_max, krylov_restart, eps):
    """The IMPLICIT-FLUX BDF lowering (ADC-431): a matrix-free Newton-Krylov solve of the coupled
    nonlinear system, composed PURELY from existing IR primitives (no new C++ stepper).

    The implicit BDF step solves ``F(U^{n+1}) = 0`` with::

        BDF1:  F(U) = U - U^n            - dt*rhs(U)
        BDF2:  F(U) = U - (4/3)U^n + (1/3)U^{n-1} - (2/3)*dt*rhs(U)

    (BDF2 reads ``U^{n-1}`` from the System history ring with a BDF1 cold start.) ``rhs(U) = -div F(U)
    [+ sources]`` is the SAME hyperbolic residual the explicit schemes use, so the flux couples the
    cells through its stencil and the Newton system is GLOBAL.

    Newton's method (the outer loop) is a fixed `static_range` unroll of @p newton_max iterations -- each
    iteration is independent IR (its own matrix-free operator + Krylov solve), which the codegen lowers
    at the top level (the install-time apply lambda the Krylov loop needs cannot live inside a runtime
    while/range body). Each iteration:

      1. ``R^k = rhs(U^k)`` (one rhs evaluation; also the frozen base of the matvec FD);
      2. ``F^k = U^k - U^n_terms - c*dt*R^k`` (the residual; ``c = 1`` BDF1, ``c = 2/3`` BDF2);
      3. solve ``J dU = -F^k`` with GMRES (J nonsymmetric), J applied matrix-free via `rhs_jacvec`
         (``J v = v - c*dt * d(rhs)/dU v``, a finite-difference Jacobian-vector product around U^k);
      4. ``U^{k+1} = U^k + dU``.

    The final residual norm ``||F||`` is recorded as the diagnostic ``"<block>.bdf_residual"`` (read via
    ``sim.program_diagnostic``). @p ncomp is the block component count (1 by default -- a scalar model
    like inviscid Burgers / linear advection; pass the model's n_cons for a multi-component block)."""
    c = 1.0 if order == 1 else (2.0 / 3.0)
    U0 = P.state(block)
    fields = P.solve_fields(U0) if flux else None  # frozen-Poisson coupling, solved once from U^n
    # Snapshot U^n into a scratch: the commit writes ctx.state(0) IN PLACE at the very end, so the lagged
    # term must read this frozen copy (not the live state) -- otherwise the post-commit residual
    # diagnostic would read U^{n+1} as U^n. The Newton-loop residuals (before the commit) would be correct
    # either way; the snapshot keeps every residual (loop + diagnostic) reading the true U^n.
    Un = P.linear_combine(block + "_bdf_Un", 1.0 * U0)
    if order == 2:
        name = block + ".U"
        P.store_history(name, U0)                   # store U^n (cold-start fills the ring)
        U_nm1 = P.history(name, lag=1)              # U^{n-1} (== U^n on step 0 -> BDF1 cold start)

    def _un_terms():
        # The lagged (constant-in-Newton) part of the residual: U^n for BDF1, (4/3)U^n - (1/3)U^{n-1}
        # for BDF2 (the constant-state coefficients of the BDF residual normalized to a unit U^{n+1}).
        if order == 1:
            return 1.0 * Un
        return (4.0 / 3.0) * Un - (1.0 / 3.0) * U_nm1

    src = list(sources) if sources is not None else None
    kind = "scalar" if ncomp == 1 else "state"

    def _residual(P, Uk, tag):
        # F^k = U^k - U^n_terms - c*dt*rhs(U^k); returns (F^k, R^k) so the matvec can reuse R^k.
        Rk = P.rhs(name="%s_R" % tag, state=Uk, fields=fields, flux=flux, sources=src)
        Fk = P.linear_combine("%s_F" % tag, _un_terms() * (-1.0) + 1.0 * Uk - (c * P.dt) * Rk)
        return Fk, Rk

    def _newton_step(P, Uk, k):
        tag = "%s_bdf%d_n%d" % (block, order, k)
        Fk, Rk = _residual(P, Uk, tag)
        negF = P.linear_combine("%s_negF" % tag, -1.0 * Fk)
        A = P.matrix_free_operator("%s_J" % tag, domain=kind, range_=kind,
                                   ncomp=(None if ncomp == 1 else ncomp))

        def apply(P, out, v):
            # J v = v - c*dt * d(rhs)/dU v, matrix-free FD around the frozen iterate U^k (r0 = R^k).
            return P.rhs_jacvec(out, v, iterate=Uk, r0=Rk, c_dt=(c * P.dt), eps=eps, flux=flux,
                                sources=sources)

        P.set_apply(A, apply)
        dU = P.solve_linear(name="%s_dU" % tag, operator=A, rhs=negF, method="gmres", tol=krylov_tol,
                            max_iter=krylov_max, restart=krylov_restart)
        return P.linear_combine("%s_next" % tag, 1.0 * Uk + 1.0 * dU)

    # Outer Newton loop: a fixed unroll of newton_max iterations (each independent top-level IR).
    Uk = U0
    for k in range(newton_max):
        Uk = _newton_step(P, Uk, k)
    # Record the final residual norm for diagnostics (sim.program_diagnostic("<block>.bdf_residual")).
    Ffinal, _ = _residual(P, Uk, "%s_bdf%d_final" % (block, order))
    P.record_scalar(block + ".bdf_residual", P.norm2(Ffinal))
    P.commit(block, Uk)
    return Uk


def bdf(P, block, order, *, linear_source=None, sources=("default",), flux=True, ncomp=1,
        newton_tol=1e-10, newton_max=20, krylov_tol=1e-10, krylov_max=200, krylov_restart=None,
        eps=1e-7):
    """Backward Differentiation Formula, IMPLICIT ``order``-step (ADC-423 / ADC-431).

    Two lowerings share this entry point, selected by whether an implicit @p linear_source is named:

      - **implicit FLUX** (the default, ADC-431): ``F(U^{n+1}) = 0`` for the coupled nonlinear system
        ``U - U^n - dt*rhs(U)`` (BDF1) / ``U - (4/3)U^n + (1/3)U^{n-1} - (2/3)dt*rhs(U)`` (BDF2) is
        solved by a matrix-free Newton-Krylov iteration -- ``rhs(U) = -div F [+ sources]`` couples the
        cells through the flux stencil, so the Jacobian ``J = I - c*dt*d(rhs)/dU`` is GLOBAL and applied
        matrix-free by a finite-difference Jacobian-vector product (`P.rhs_jacvec`); each Newton step
        solves ``J dU = -F`` with GMRES (J nonsymmetric). The outer Newton loop is a fixed unroll of
        @p newton_max iterations. The final ``||F||`` is recorded as ``"<block>.bdf_residual"``. This is
        a pure-macro composition of existing primitives (matrix_free_operator + solve_linear + the affine
        algebra + history) -- no new C++ runtime stepper.

      - **cell-local linear SOURCE** (the fast path, ADC-423): when @p linear_source names a model
        ``m.linear_source`` ``L``, the BDF system is block-diagonal and ``(c0*I - dt*L) U^{n+1} = rhs``
        is solved per cell by `P.solve_local_linear` (no Newton / Krylov). @p flux / @p sources then add
        an EXPLICIT flux/source RHS lagged at U^n (like `imex_local`).

    @p order is 1 (backward Euler) or 2 (BDF2, over the System history ring with a BDF1 cold start).
    @p ncomp is the block component count for the implicit-flux path (1 for a scalar model such as
    inviscid Burgers / linear advection; pass the model's n_cons for a multi-component block).
    @p newton_max / @p newton_tol bound the Newton iteration; @p krylov_tol / @p krylov_max /
    @p krylov_restart configure each GMRES inner solve; @p eps is the relative finite-difference step of
    the Jacobian-vector product."""
    if isinstance(order, bool) or not isinstance(order, int) or order not in (1, 2):
        raise ValueError("bdf: order must be the int 1 or 2 (got %r)" % (order,))
    if linear_source is not None:
        if not (isinstance(linear_source, str) and linear_source):
            raise ValueError("bdf: linear_source must be a non-empty model linear-source name or None")
        return _bdf_local_linear(P, block, order, linear_source, sources, flux)
    # The implicit-flux Newton-Krylov path (ADC-431): a flux-less BDF with no implicit term is a no-op.
    if not flux:
        raise ValueError(
            "bdf with flux=False needs a cell-local implicit linear_source (there is no implicit term to "
            "solve); pass linear_source='<name>' for the relaxation BDF, or flux=True for the "
            "implicit-flux Newton-Krylov BDF")
    if isinstance(ncomp, bool) or not isinstance(ncomp, int) or ncomp < 1:
        raise ValueError("bdf: ncomp must be a positive int (the block component count); got %r"
                         % (ncomp,))
    if isinstance(newton_max, bool) or not isinstance(newton_max, int) or newton_max < 1:
        raise ValueError("bdf: newton_max must be a positive int (got %r)" % (newton_max,))
    return _bdf_implicit_flux(P, block, order, sources, flux, ncomp, newton_tol, newton_max,
                              krylov_tol, krylov_max, krylov_restart, eps)


# adc.time.std.<scheme>(Program, block, ...) -- the spec's standard library entry point.
# --- operator-first standard macros (Spec 2) --------------------------------------------------
# These macros are MODEL-FREE: they take typed operator NAMES (not physical terms) and compose them
# with P.call against the registry bound to the Program (P.bind_operators(module)). The SAME macro
# runs against any Module that provides operators with the expected signatures. They never mention
# flux / source / poisson / lorentz / rho / mx / my.
def _op_space_arity(P, name):
    """Number of space-typed inputs (State / FieldSpace) of operator @p name in the bound registry."""
    if P._registry is None:
        raise ValueError("operator-first macro: bind a module first (P.bind_operators(module))")
    op = P._registry.get(name)
    return sum(1 for t in op.signature.inputs if getattr(t, "kind", None) in ("state", "field"))


def _opcall(P, name, *candidate_args, value_name=None):
    """Call operator @p name passing exactly as many leading args as its signature's space inputs
    (so an operator that ignores the fields is called with the state alone, and a fields-free linear
    operator with no args)."""
    arity = _op_space_arity(P, name)
    return P.call(name, *candidate_args[:arity], name=value_name)


def predictor_corrector_local_linear(P, block, *, fields_operator, explicit_rate_operator,
                                     implicit_operator, state_space="U", commit=True):
    """Generic predictor-corrector for ``dU/dt = R(U, fields) + L(fields) U`` (Spec 2, operator-first).

    Composes THREE typed operators by name -- a field operator ``fields_operator: U -> Fields``, an
    explicit rate ``explicit_rate_operator: (U, Fields) -> Rate(U)`` and a local linear operator
    ``implicit_operator: Fields -> LocalLinearOperator(U, U)`` -- into one trapezoidal step with the
    L term treated implicitly via local solves::

        U*    = (I - dt L_n)^{-1} (U^n + dt R_n)
        U^n+1 = (I - 1/2 dt L*)^{-1} (U^n + 1/2 dt R_n + 1/2 dt R* + 1/2 dt L* U*)

    It mentions no physics; ``state_space`` is informational. Requires ``P.bind_operators(module)``.
    """
    u_n = P.state(block)
    fields_n = _opcall(P, fields_operator, u_n, value_name="fields_n")
    r_n = _opcall(P, explicit_rate_operator, u_n, fields_n, value_name="R_n")
    l_n = _opcall(P, implicit_operator, fields_n, value_name="L_n")
    u_star = P.solve_local_linear("U_star", operator=P.I - P.dt * l_n,
                                  rhs=P.linear_combine("U_star_rhs", u_n + P.dt * r_n),
                                  fields=fields_n)
    fields_star = _opcall(P, fields_operator, u_star, value_name="fields_star")
    r_star = _opcall(P, explicit_rate_operator, u_star, fields_star, value_name="R_star")
    l_star = _opcall(P, implicit_operator, fields_star, value_name="L_star")
    c_star = P.apply(l_star, u_star, fields=fields_star, name="C_star")
    q = P.linear_combine("Q", u_n + 0.5 * P.dt * r_n + 0.5 * P.dt * r_star + 0.5 * P.dt * c_star)
    u_np1 = P.solve_local_linear("U_np1", operator=P.I - 0.5 * P.dt * l_star, rhs=q,
                                 fields=fields_star)
    if commit:
        P.commit(block, u_np1)
    return u_np1


def explicit_rk(P, block, *, rhs_operator, fields_operator=None, tableau=None, A=None, b=None,
                c=None, state_space="U"):
    """Generic explicit Runge-Kutta over a typed rate operator (Spec 2, operator-first).

    Each stage is ``k_i = rhs_operator(U_i[, fields_operator(U_i)])``; the tableau lowers to the same
    affine stage chain as :func:`rk`. Pass a ``ButcherTableau`` / ``(A, b, c)`` via ``tableau`` or the
    raw ``A`` / ``b`` / ``c``. ``fields_operator`` is optional (a pure-flux rate needs no fields).
    """
    if tableau is None:
        if A is None or b is None:
            raise ValueError("explicit_rk: provide a tableau or A and b")
        tableau = ButcherTableau(A, b, c)
    elif not isinstance(tableau, ButcherTableau):
        ta, tb, tc = tableau if len(tableau) == 3 else (tableau[0], tableau[1], None)
        tableau = ButcherTableau(ta, tb, tc)
    tag = (tableau.name + "_") if tableau.name else "rk_"
    u0 = P.state(block)
    ks = []
    for i in range(tableau.stages):
        if i == 0:
            u_i = u0
        else:
            expr = u0
            for j in range(i):
                aij = tableau.A[i][j]
                if aij != 0.0:
                    expr = expr + (P.dt * aij) * ks[j]
            u_i = P.linear_combine("%sU%d" % (tag, i), expr)
        if fields_operator is not None:
            f_i = _opcall(P, fields_operator, u_i)
            ks.append(_opcall(P, rhs_operator, u_i, f_i, value_name="%sk%d" % (tag, i)))
        else:
            ks.append(_opcall(P, rhs_operator, u_i, value_name="%sk%d" % (tag, i)))
    final = u0
    for i in range(tableau.stages):
        bi = tableau.b[i]
        if bi != 0.0:
            final = final + (P.dt * bi) * ks[i]
    P.commit(block, P.linear_combine("%sstep" % tag, final))


def imex_local_linear(P, block, *, explicit_operator, implicit_operator, fields_operator=None,
                      theta=1.0, state_space="U"):
    """Generic IMEX with an explicit rate and an implicit local linear operator (Spec 2).

    One theta-implicit step of ``dU/dt = R(U[, fields]) + L([fields]) U``::

        U^{n+1} = (I - theta dt L)^{-1} (U^n + dt R)

    composing the typed ``explicit_operator`` and ``implicit_operator`` (and an optional
    ``fields_operator``) by name. Requires ``P.bind_operators(module)``.
    """
    if not (0.0 < theta <= 1.0):
        raise ValueError("imex_local_linear: theta must be in (0, 1]")
    u = P.state(block)
    fields = _opcall(P, fields_operator, u, value_name="fields") if fields_operator else None
    r = _opcall(P, explicit_operator, u, fields, value_name="R")
    lin = _opcall(P, implicit_operator, fields, value_name="L")
    q = P.linear_combine("imex_rhs", u + P.dt * r)
    u1 = P.solve_local_linear("imex_step", operator=P.I - theta * P.dt * lin, rhs=q, fields=fields)
    P.commit(block, u1)
    return u1


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


std = types.SimpleNamespace(forward_euler=forward_euler, ssprk2=ssprk2, ssprk3=ssprk3, rk4=rk4,
                            rk=rk, RK4_TABLEAU=RK4_TABLEAU, SSPRK2_TABLEAU=SSPRK2_TABLEAU,
                            ButcherTableau=ButcherTableau,
                            adams_bashforth=adams_bashforth, adams_bashforth2=adams_bashforth2,
                            strang=strang, lie=lie, imex_local=imex_local, bdf=bdf,
                            condensed_schur=condensed_schur,
                            predictor_corrector_local_linear=predictor_corrector_local_linear,
                            explicit_rk=explicit_rk, imex_local_linear=imex_local_linear)


class CompiledTime:
    """Record of a compiled `Program`'s macro-step cadence (`substeps` / `stride`).

    A compiled Program OWNS the whole step body: it is installed via `sim.install_program` and driven
    by `sim.step(dt)`. Its cadence is applied to the System with `sim.set_program_cadence(substeps,
    stride)` (call it after `install_program`); a `CompiledTime` just records those values. The
    compiled program is NOT attached via `sim.add_equation(time=CompiledTime(...))` -- that path is
    rejected with an explicit error (the transport policy passed to `add_equation` is a native
    `adc.Explicit`/etc.; the compiled program is installed separately). `substeps` and
    `stride` are wired (ADC-411) as a SYSTEM-level orchestration AROUND the opaque program closure
    (`System.set_program_cadence`, mirroring the native per-block advance loop): `substeps=n` runs the
    program n times over `eff_dt/n`; `stride=M` runs the whole program once per M macro-steps with
    `eff_dt = M*dt` (GLOBAL hold-then-catch-up, the clock still ticks every macro-step).

    Two semantic limits to keep in mind (cf. system_stepper.hpp):
      - `substeps > 1` is bit-exact vs native `adc.Explicit(substeps=n)` ONLY for an UNCOUPLED /
        transport-only program: `program_step_(h)` re-runs the WHOLE program (its `solve_fields`
        included), whereas native substeps subdivides ONLY the transport (solve_fields runs once).
      - `stride` here is GLOBAL (a compiled program is one whole-system closure), so it equals native
        per-block stride only for a single-block system (or all blocks sharing the stride).

    A non-default `cfl` is still deferred (the Program receives a bare `dt`; pass an explicit `dt` to
    `sim.step(dt)`) -- it fails loud rather than being silently ignored."""

    def __init__(self, substeps=1, stride=1, cfl="default"):
        if not isinstance(substeps, int) or substeps < 1:
            raise ValueError("CompiledTime: substeps must be a positive int (got %r)" % (substeps,))
        if not isinstance(stride, int) or stride < 1:
            raise ValueError("CompiledTime: stride must be a positive int (got %r)" % (stride,))
        if cfl != "default":
            raise NotImplementedError(
                "CompiledTime: cfl != 'default' is deferred (ADC-401 Phase 2c); pass an explicit dt "
                "to sim.step(dt)")
        self.substeps = substeps
        self.stride = stride
        self.cfl = cfl
        self.kind = "compiled"

    def __repr__(self):
        return "CompiledTime(substeps=%d, stride=%d, cfl=%r)" % (self.substeps, self.stride, self.cfl)
