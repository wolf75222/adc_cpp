"""pops.time value algebra -- typed SSA handles and the affine/operator algebra.

A ``Value`` is a typed SSA node in a Program IR; field-like values support an affine algebra
(``U + dt * R``) and scalars compose into ``scalar_op`` nodes. ``_Coeff`` / ``_Affine`` /
``_Operator`` are the coefficient + linear-combination carriers; ``StageStateSet`` and
``_CoupledResult`` are multi-block grouping handles. Authoring + evaluation only: no codegen,
no _pops, no module-scope dsl import.
"""
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
    pops::MultiFab and support the affine algebra."""
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
    to an pops::MultiFab; a ``matrix_free_op`` names a matrix-free operator A whose apply sub-block is
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
        # Operator-first type tag (Spec 2): the pops.model space/operator-type this value lives over
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
        so :meth:`pops.math.Unknown.__rmatmul__` builds the solve left-hand side.
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


