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

__all__ = ["Program", "std", "CompiledTime"]


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

    def __add__(self, other):
        return self._affine() + _to_affine(other)

    __radd__ = __add__

    def __neg__(self):
        return -self._affine()

    def __sub__(self, other):
        return self._affine() - _to_affine(other)

    def __rsub__(self, other):
        return _to_affine(other) - self._affine()

    def __mul__(self, other):
        if self.vtype == "operator":  # a linear-source operator: scalar/dt * L -> an _Operator term
            if isinstance(other, (int, float)):
                other = _Coeff({0: float(other)})
            if isinstance(other, _Coeff):
                return _Operator(_Coeff({}), [(self, other)])
            return NotImplemented
        if isinstance(other, (int, float, _Coeff)):
            return self._affine() * other
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self._affine() * _Coeff({0: 1.0 / other})
        return NotImplemented

    def __repr__(self):
        return "<%s %s #%d>" % (self.vtype, self.name, self.id)


class Program:
    """A compiled time program (builder mode). Holds the SSA value list and the committed blocks.

    The Python object only BUILDS the IR; it is never executed numerically during ``sim.step``."""

    def __init__(self, name):
        self.name = name
        self._values = []
        self._next_id = 0
        self._commits = {}      # block -> State value
        self._recording = []    # stack of sub-block lists (a control-flow body); see _new / while_
        self.dt = _Coeff({1: 1.0})   # symbolic time step; participates in coefficient arithmetic

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

    def state(self, block):
        """Reference the current conservative state of ``block`` at the start of the step."""
        if not isinstance(block, str) or not block:
            raise ValueError("state: block must be a non-empty string")
        return self._new("state", "state", (), {}, block, block)

    def solve_fields(self, name=None, state=None):
        """Solve the elliptic fields from ``state`` and return a FieldContext. Accepts
        ``solve_fields(state)`` or ``solve_fields(name, state)``. Each call is a DISTINCT
        FieldContext (a stage's RHS must read the fields solved from its own state, never a stale
        global)."""
        if isinstance(name, Value) and state is None:
            name, state = None, name
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("solve_fields: a State value is required")
        return self._new("fields", "solve_fields", (state,), {}, name, state.block)

    def rhs(self, name=None, state=None, fields=None, flux=True, sources=None, fluxes=None):
        """Build R = -div F(U) + sum of the requested named ``sources``. ``fields`` is the explicit
        FieldContext any field-dependent source reads (no implicit global aux). Named sources are
        never summed implicitly: ``sources`` lists exactly the ones to include."""
        if isinstance(name, Value):
            raise ValueError("rhs: pass state=/fields= by keyword (first arg is the debug name)")
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("rhs: a State value is required (state=...)")
        if fields is not None and not (isinstance(fields, Value) and fields.vtype == "fields"):
            raise ValueError("rhs: fields must be a FieldContext from solve_fields")
        src = list(sources) if sources is not None else []
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
        for v, _ in aff:
            if v.vtype == "state":
                block = v.block
                break
        if block is None:
            block = aff[0][0].block
        inputs = tuple(v for v, _ in aff)
        coeffs = [c.as_dict() for _, c in aff]
        return self._new("state", "linear_combine", inputs, {"coeffs": coeffs}, name, block)

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

    def apply(self, operator=None, state=None, fields=None, name=None):
        """Apply a linear-source operator to a state: ``LU = L_name(aux, params) U``. ``operator`` is
        a `linear_source` value (or its name). Returns an RHS-like value."""
        lname = self._linear_source_name(operator, "apply")
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("apply: a State value is required (state=...)")
        if fields is not None and not (isinstance(fields, Value) and fields.vtype == "fields"):
            raise ValueError("apply: fields must be a FieldContext from solve_fields")
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
        lname = op_value.attrs["linear_source"]
        a = (-l_coeff).as_dict()  # operator = I - a*L, so the L term carries the coefficient -a
        inputs = (rhs, op_value, fields) if fields is not None else (rhs, op_value)
        return self._new("state", "solve_local_linear", inputs,
                         {"linear_source": lname, "a_coeff": a}, name, rhs.block)

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
    # call into the runtime's Krylov loop (adc::cg_solve / bicgstab_solve / richardson_solve): the
    # iteration is DYNAMIC and lives C++-side (inside the loop), invisible to the IR -- the Program only
    # supplies the apply (a C++ lambda) + the rhs / tolerance / iteration budget.
    _KRYLOV_METHODS = frozenset({"cg", "bicgstab", "richardson"})

    def scalar_field(self, name=None):
        """A fresh, zero-initialized scalar field (1 component): scratch the apply sub-block uses (e.g.
        the Laplacian output). Lowered to ``ctx.alloc_scalar_field()``."""
        return self._new("scalar_field", "scalar_field", (), {}, name, None)

    def matrix_free_operator(self, name, domain="scalar", range_="scalar"):
        """Declare a matrix-free operator ``A : domain -> range_`` (scalar field -> scalar field for
        now). Supply its apply via ``P.set_apply(A, body_fn)`` before using it in ``P.solve_linear``.
        domain / range_ other than ``"scalar"`` are a later phase (vector / state-valued operators)."""
        if domain != "scalar" or range_ != "scalar":
            raise NotImplementedError(
                "matrix_free_operator currently supports domain='scalar' and range_='scalar' only "
                "(vector / state-valued operators are a later phase); got domain=%r range_=%r"
                % (domain, range_))
        return self._new("matrix_free_op", "matrix_free_operator", (),
                         {"domain": domain, "range": range_, "apply_block": None, "apply_result": None,
                          "apply_in": None, "apply_out": None}, name, None)

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
        try:
            out_sf = self._new("scalar_field", "apply_out", (), {}, "apply_out", None)
            in_sf = self._new("scalar_field", "apply_in", (), {}, "apply_in", None)
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

    def solve_linear(self, name=None, operator=None, rhs=None, initial_guess=None, method="cg",
                     preconditioner="identity", tol=1e-8, max_iter=None):
        """Solve the matrix-free linear system ``operator x = rhs`` with the runtime's Krylov loop and
        return the solution as a scalar_field. The iteration is DYNAMIC (C++-side, inside the loop):
        the IR only carries the operator (its apply lambda), the rhs, the initial guess, and the
        method / tolerance / iteration budget.

          - @p operator: a ``matrix_free_operator`` value (with a ``set_apply`` body);
          - @p rhs: the right-hand side -- a scalar_field, or (MVP) a 1-component State value;
          - @p initial_guess: warm start (defaults to zero);
          - @p method: ``"cg"`` (SPD), ``"bicgstab"`` (general) or ``"richardson"``;
          - @p preconditioner: ``"identity"`` only for now;
          - @p tol: relative L2 residual stop (> 0);
          - @p max_iter: iteration budget (REQUIRED, > 0: a dynamic solver loop with no budget is a
            configuration error -- ``adc::*_solve`` itself throws on a non-positive budget)."""
        if not (isinstance(operator, Value) and operator.vtype == "matrix_free_op"):
            raise ValueError("solve_linear: operator must be a matrix_free_operator value")
        if operator.attrs["apply_block"] is None:
            raise ValueError("solve_linear: operator '%s' has no apply; call P.set_apply first"
                             % operator.name)
        if not _is_field_value(rhs):
            raise ValueError("solve_linear: rhs must be a scalar_field or State value (rhs=...)")
        if initial_guess is not None and not _is_field_value(initial_guess):
            raise ValueError("solve_linear: initial_guess must be a scalar_field or State value")
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
        inputs = (operator, rhs) if initial_guess is None else (operator, rhs, initial_guess)
        return self._new("scalar_field", "solve_linear", inputs,
                         {"method": method, "preconditioner": preconditioner, "tol": float(tol),
                          "max_iter": int(max_iter), "has_guess": initial_guess is not None},
                         name, rhs.block)

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
        if "coeffs" in attrs:  # dict keys (powers) -> sorted [power, value] for stable JSON
            attrs["coeffs"] = [sorted((int(p), c) for p, c in d.items()) for d in attrs["coeffs"]]
        if "a_coeff" in attrs:  # solve_local_linear: the dt-polynomial a in (I - a*L)
            attrs["a_coeff"] = sorted((int(p), c) for p, c in attrs["a_coeff"].items())
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
        return {"id": v.id, "vtype": v.vtype, "op": v.op, "block": v.block,
                "inputs": [i.id for i in v.inputs], "attrs": attrs}

    def _serialize(self):
        nodes = [Program._serialize_node(v) for v in self._values]
        commits = sorted((b, s.id) for b, s in self._commits.items())
        return {"name": self.name, "version": 1, "nodes": nodes, "commits": commits}

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

        Lowers a SINGLE-BLOCK Program by a topological walk of the SSA IR: the block's current state is
        the base (``ctx.state(0)``); ``solve_fields()`` runs the elliptic solve; each RHS becomes a
        scratch + ``rhs_into``; each intermediate ``linear_combine`` becomes a zero scratch accumulated
        with ``axpy``; the committed combine writes the block state via ``lincomb``. Forward Euler,
        SSPRK2/SSPRK3 and RK4 all lower this way -- no per-scheme class.

        Phase-4b also lowers the SPLIT-SOURCE / LOCAL-LINEAR ops -- ``source`` (a named ``m.source_term``
        evaluated per cell), ``apply`` (LU for a named ``m.linear_source``) and ``solve_local_linear``
        ((I -/+ a*L) U = rhs solved cell by cell via a dense per-cell inverse) -- but ONLY when the
        physical ``model`` (the ``adc.dsl`` model whose ``source_term`` / ``linear_source`` they name)
        is provided: the codegen reads the model's symbolic coefficients to emit the per-cell kernels.
        Without ``model`` those ops raise NotImplementedError (the Program cannot be lowered in
        isolation); ``model=None`` still lowers FE / SSPRK / RK4 (no model needed). A ``rhs`` with NAMED
        sources (``sources=[...]`` beyond ``"default"``) also lowers with a model: ``ctx.rhs_into`` (=
        ``-div F`` + the model's default source) plus, for each named source, the same per-cell
        ``m.source_term`` kernel as the standalone ``source`` op, accumulated onto ``R`` via ``axpy``.
        That is sound only when the model's DEFAULT source is empty (else ``rhs_into`` already folds a
        source and adding named ones double-counts), so a named-source ``rhs`` against a model with a
        non-empty ``m.source`` is refused (deferred). More than one block, control flow or Krylov remain
        later phases and raise NotImplementedError, never a silent mis-lowering.

        NOTE: ``solve_fields()`` solves from the block's CURRENT state, so a multi-stage scheme that
        needs the elliptic fields re-solved from each STAGE state (a field-coupled model) is not yet
        exact; for an uncoupled model (no Poisson feedback into the flux) the field solve is inert and
        the scheme is exact. ``solve_fields_from_state`` is a later phase."""
        self.validate()
        self._check_lowerable(model)
        prelude, body = self._emit_body(model)
        return _PROGRAM_CPP_TEMPLATE.format(name=json.dumps(self.name), hash=self._ir_hash(),
                                            prelude=prelude, body=body)

    # Ops the Phase-4b codegen lowers ONLY when a physical model is supplied (they read the model's
    # symbolic source_term / linear_source coefficients). Without a model they raise NotImplementedError.
    _MODEL_OPS = ("source", "apply", "solve_local_linear")

    def _check_lowerable(self, model=None):
        """Raise NotImplementedError if the IR uses a construct the current codegen cannot lower yet,
        naming the offending construct (never a silent mis-lowering). @p model: the physical model that
        declares the named sources / linear sources; required for the Phase-4b ops."""
        if len(self._commits) != 1:
            raise NotImplementedError(
                "emit_cpp_program currently supports a single committed block; this Program commits "
                "%d (multi-block is a later phase)" % len(self._commits))
        states = [v for v in self._values if v.op == "state"]
        if len(states) != 1:
            raise NotImplementedError(
                "emit_cpp_program currently supports a single block state (one P.state(...)); got %d"
                % len(states))
        for v in self._values:
            self._check_op_lowerable(v, model)
        # Per-cell dense fallback bound for solve_local_linear (mat_inverse<N> uses fixed stack buffers).
        if model is not None and any(v.op == "solve_local_linear" for v in self._all_ops()):
            impl = _model_impl(model)
            n_cons = len(getattr(impl, "cons_names", []) or [])
            if n_cons > 8:
                raise ValueError(
                    "solve_local_linear dense fallback currently supports n_cons <= 8 (got %d)"
                    % n_cons)

    # 'linear_source' is a pure NAME-reference SSA node (vtype 'operator'): it carries no runtime work
    # (consumed by apply / solve_local_linear, which read the model coefficients), so it lowers to
    # nothing -- always allowed, model or not. 'reduce' / 'compare' / 'while' are the ADC-404a control
    # flow / reduction ops (lowered inline via adc::dot; no model needed). 'matrix_free_operator' /
    # 'scalar_field' / 'laplacian' / 'gradient' / 'solve_linear' are the ADC-405 matrix-free Krylov ops
    # (the operator declaration carries an apply sub-block; solve_linear lowers to adc::*_solve).
    _ALLOWED_OPS = frozenset({"state", "solve_fields", "rhs", "linear_combine", "linear_source",
                              "reduce", "compare", "while", "range", "if", "matrix_free_operator",
                              "scalar_field", "laplacian", "gradient", "solve_linear", "apply_in",
                              "apply_out"})

    def _all_ops(self):
        """Iterate over every op of the Program, descending into control-flow + apply sub-blocks (a flat
        view used by the lowerability guards: the sub-block ops are not in self._values). Nested control
        flow is disallowed, so the sub-blocks are flat (one level)."""
        for v in self._values:
            yield v
            for key in ("cond_block", "body_block", "apply_block"):
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
            return  # _emit_op lowers it from the model's symbolic coefficients
        if v.op not in Program._ALLOWED_OPS:
            raise NotImplementedError(
                "emit_cpp_program cannot lower op '%s' (value '%s') yet; supported ops are %s "
                "(+ %s with a model; nested control flow / Krylov are later phases)"
                % (v.op, v.name, sorted(Program._ALLOWED_OPS), sorted(Program._MODEL_OPS)))
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
        if v.op == "rhs":
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
            # ctx.rhs_into already folds the model's DEFAULT/composite source (m.source -> _source).
            # Adding a named source on top of a non-empty default would DOUBLE-COUNT, so the named-
            # source rhs is only sound when the default source is empty (the named-source-only model the
            # predictor-corrector uses). A flux-only seam ('default' / no sources) is unaffected.
            if getattr(impl, "_source", None):
                raise NotImplementedError(
                    "rhs with named sources needs a model whose default source is empty (no "
                    "m.source), or a flux-only seam; rhs '%s' requests %r but the model has a "
                    "non-empty default source (deferred)" % (v.name, extra))
            for s in extra:
                if s not in impl._source_terms:
                    raise ValueError(
                        "unknown source_term '%s' in rhs '%s'; declared source_terms: %s"
                        % (s, v.name, sorted(impl._source_terms)))

    def _emit_body(self, model=None):
        """Generate the C++ of the install function in TWO phases (each list indented uniformly by the
        template). Assumes `_check_lowerable` has passed: one block state (the base = ctx.state(0)), one
        commit. @p model supplies the symbolic coefficients of the Phase-4b source / apply /
        solve_local_linear ops. Returns ``(prelude, body)``:

          - ``prelude``: INSTALL-TIME C++ (before ``ctx.install``) -- persistent scratch fields (held
            via ``std::shared_ptr`` so they outlive the install call and are reused across every step
            and every Krylov iteration) and the matrix-free apply lambdas. Captured by value into the
            step closure (shared_ptr / lambda / ctx all copy cheaply).
          - ``body``: the STEP closure body (one macro-step over dt)."""
        base = next(v for v in self._values if v.op == "state")
        committed = next(iter(self._commits.values()))
        # IR value id -> C++ token: a MultiFab variable name (states / RHS scratches), a scalar variable
        # name (reductions, ``s{id}``) or a parenthesized boolean expression (compares).
        var = {}
        prelude = []
        lines = []
        for v in self._values:
            self._emit_op(v, base, committed.id, var, model, lines, prelude)
        # The committed value may be an op that wrote into a SCRATCH (e.g. solve_local_linear,
        # solve_linear): copy it into the block state. A linear_combine commit already wrote ctx.state(0)
        # in place (var == base), so this is a no-op there (skipped).
        if var[committed.id] != var[base.id]:
            lines.append("ctx.lincomb(%s, static_cast<adc::Real>(0), %s, static_cast<adc::Real>(1), %s);"
                         % (var[base.id], var[base.id], var[committed.id]))
        prelude_src = "\n".join("  " + ln for ln in prelude)
        body_src = "\n".join("    " + ln for ln in lines)
        return prelude_src, body_src

    def _emit_op(self, v, base, committed_id, var, model, lines, prelude=None):
        """Lower a SINGLE op to C++, appending to @p lines and recording its C++ token in @p var. Shared
        by the top-level walk and the while sub-blocks (a while body re-runs this per op each pass), so
        reductions / compares / linear_combine all lower identically inside the loop. @p base is the
        block-state value (its C++ var is the loop variable inside a while sub-block); @p committed_id
        is the committed value's id (None inside a sub-block: a body combine is never the commit).
        @p prelude collects INSTALL-TIME lines (persistent scratch + apply lambdas) for the matrix-free
        Krylov ops; None inside a sub-block (those ops only appear at the top level for now)."""
        if v.op == "state":
            var[v.id] = "u%d" % v.id
            lines.append("adc::MultiFab& %s = ctx.state(0);" % var[v.id])
        elif v.op == "solve_fields":
            lines.append("ctx.solve_fields();")
        elif v.op == "rhs":
            state_in = v.inputs[0]  # rhs inputs = (state[, fields]); the state is first
            var[v.id] = "r%d" % v.id
            lines.append("adc::MultiFab %s = ctx.rhs_scratch_like(%s);"
                         % (var[v.id], var[state_in.id]))
            # R <- -div F + default/composite source (ctx.rhs_into). _check_lowerable guarantees the
            # model's default source is EMPTY when named sources are requested, so rhs_into here is
            # flux-only (-div F) and the named sources below are added without double-counting.
            lines.append("ctx.rhs_into(0, %s, %s);" % (var[state_in.id], var[v.id]))
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
        elif v.op == "matrix_free_operator":
            # Install-time: emit the apply lambda `apply_A{id}` into the prelude. Its persistent scratch
            # (the scalar_field ops of the apply sub-block) are shared_ptr fields, captured by value so
            # they outlive the install call and are reused across every Krylov iteration (alloc-once).
            # The lambda is itself captured by the step closure ([=]) and passed to adc::*_solve.
            self._emit_matrix_free_operator(v, var, prelude)
        elif v.op in ("scalar_field", "laplacian", "gradient", "apply_in", "apply_out"):
            # These ops only appear INSIDE an apply sub-block (lowered by _emit_matrix_free_operator) or
            # as the operands of solve_linear; they never lower standalone at the top level.
            raise NotImplementedError(
                "emit_cpp_program: op '%s' (value '%s') is only lowerable inside a matrix_free_operator "
                "apply sub-block" % (v.op, v.name))
        elif v.op == "solve_linear":
            self._emit_solve_linear(v, base, var, prelude, lines)
        elif v.op == "reduce":
            # A collective all_reduce -> a C++ scalar. norm2 = sqrt(dot(u, u)); dot(a, b) directly. Both
            # MUST run on every rank (adc::dot is collective); they sit at the top of the loop body.
            var[v.id] = "s%d" % v.id
            if v.attrs["kind"] == "norm2":
                (u,) = v.inputs
                lines.append("const adc::Real %s = std::sqrt(adc::dot(%s, %s));"
                             % (var[v.id], var[u.id], var[u.id]))
            elif v.attrs["kind"] == "norm_inf":
                (u,) = v.inputs
                lines.append("const adc::Real %s = adc::norm_inf(%s);" % (var[v.id], var[u.id]))
            else:  # dot
                a, b = v.inputs
                lines.append("const adc::Real %s = adc::dot(%s, %s);"
                             % (var[v.id], var[a.id], var[b.id]))
        elif v.op == "compare":
            # A predicate over scalars -> an inline boolean C++ expression (no statement of its own; the
            # while op embeds it directly in `if (!(<expr>)) break;`).
            lhs = v.inputs[0]
            if len(v.inputs) == 2:  # scalar vs scalar
                rhs_tok = var[v.inputs[1].id]
            else:  # scalar vs float tolerance
                rhs_tok = "static_cast<adc::Real>(%s)" % repr(float(v.attrs["rhs"]))
            var[v.id] = "(%s %s %s)" % (var[lhs.id], v.attrs["cmp"], rhs_tok)
        elif v.op == "while":
            self._emit_while(v, base, var, model, lines)
        elif v.op == "range":
            self._emit_range(v, base, var, model, lines)
        elif v.op == "if":
            self._emit_if(v, base, var, model, lines)
        elif v.op == "linear_combine":
            terms = list(zip(v.inputs, v.attrs["coeffs"], strict=True))
            if v.id == committed_id:
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

    def _emit_while(self, v, base, var, model, lines):
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
            self._emit_op(w, base, None, sub, model, body_lines)
        cond_expr = sub[v.attrs["cond"].id]
        body_lines.append("if (!(%s)) break;" % cond_expr)
        for w in v.attrs["body_block"]:
            self._emit_op(w, base, None, sub, model, body_lines)
        # Write the next state into the loop variable in place (x <- body result).
        body_lines.append("ctx.lincomb(%s, static_cast<adc::Real>(0), %s, static_cast<adc::Real>(1), %s);"
                          % (x, x, sub[v.attrs["body"].id]))
        lines += ["  " + ln for ln in body_lines]
        lines.append("}")

    def _emit_range(self, v, base, var, model, lines):
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
            self._emit_op(w, base, None, sub, model, body_lines)
        body_lines.append("ctx.lincomb(%s, static_cast<adc::Real>(0), %s, static_cast<adc::Real>(1), %s);"
                          % (x, x, sub[v.attrs["body"].id]))
        lines += ["  " + ln for ln in body_lines]
        lines.append("}")

    def _emit_if(self, v, base, var, model, lines):
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
            self._emit_op(w, base, None, sub, model, body_lines)
        body_lines.append("ctx.lincomb(%s, static_cast<adc::Real>(0), %s, static_cast<adc::Real>(1), %s);"
                          % (x, x, sub[v.attrs["body"].id]))
        lines += ["  " + ln for ln in body_lines]
        lines.append("}")

    def _emit_matrix_free_operator(self, v, var, prelude):
        """Lower a matrix_free_operator to an INSTALL-TIME C++ apply lambda ``apply_A{id}`` (appended to
        @p prelude). The lambda has the adc::ApplyFn signature ``(adc::MultiFab& out, const adc::MultiFab&
        in)``; its body re-emits the apply sub-block:

          - each ``scalar_field`` scratch -> a PERSISTENT shared_ptr field (declared in the prelude
            BEFORE the lambda, captured by value), reused across every Krylov iteration (alloc-once);
          - ``laplacian(o, i)`` -> ``ctx.laplacian(*o, i)`` (i const_cast when it is the lambda's ``in``,
            which is logically read-only -- the fill only writes ghosts, as in test_generic_krylov);
          - the apply RESULT (the affine the body returned, e.g. ``in - alpha*Lap(in)``) is written into
            ``out`` via the same accumulate-then-lincomb idiom as a linear_combine commit.

        The lambda captures ``[ctx, <scratch shared_ptrs>]``; the step closure captures it by value."""
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
            prelude.append(
                "auto %s = std::make_shared<adc::MultiFab>(ctx.alloc_scalar_field(1, 1));" % sp)
            captures.append(sp)
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
            else:
                raise NotImplementedError(
                    "emit_cpp_program: op '%s' is not lowerable inside a matrix_free_operator apply "
                    "(supported: scalar_field, laplacian, gradient)" % w.op)
        body += _emit_field_combine(result, "out", sub)
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
        prelude.append(
            "auto %s = std::make_shared<adc::MultiFab>(ctx.alloc_scalar_field(1, 1));" % sol_sp)
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
        else:  # richardson: omega = 1 (the operator is expected to be pre-scaled / well-conditioned)
            lines.append("adc::KrylovResult %s = adc::richardson_solve(%s, *%s, %s, "
                         "static_cast<adc::Real>(1), %s, %d);"
                         % (kr, lam, sol_sp, rhs_tok, tol, max_iter))
        lines.append("(void)%s;" % kr)


def _apply_in_arg(sub, value):
    """C++ argument for the INPUT field of a laplacian / gradient inside an apply lambda. When the input
    is the lambda's ``in`` (a const&), const_cast it (ctx.laplacian / gradient take a non-const MultiFab&
    and only write the ghosts, never the valid cells -- the same contract test_generic_krylov relies on);
    a persistent scratch shared_ptr is dereferenced."""
    tok = sub[value.id]
    if tok == "in":
        return "const_cast<adc::MultiFab&>(in)"
    return "*%s" % tok


def _emit_field_combine(result, target, sub):
    """Emit C++ writing the affine combination @p result into the field @p target (a C++ MultiFab token,
    e.g. ``out``). Mirrors the linear_combine commit: accumulate the non-`target` terms onto a zero
    scratch, then ``ctx.lincomb(target, c_target, target, 1, acc)``. A single unit term that already is
    the target is a no-op. @p sub maps IR value ids to C++ tokens (``in``/``out``/scratch shared_ptrs)."""
    aff = _to_affine(result)._merge()
    terms = [(v, c.as_dict()) for v, c in aff]
    lines = ["adc::MultiFab acc_%s = ctx.scratch_state_like(%s);" % (target, target)]
    c_target = {0: 0.0}
    for value, coeff in terms:
        tok = sub[value.id]
        ref = "const_cast<adc::MultiFab&>(in)" if tok == "in" else (
            "*%s" % tok if tok.startswith("sf") else tok)
        if tok == target:
            c_target = coeff
        else:
            lines.append("ctx.axpy(acc_%s, %s, %s);" % (target, _coeff_cpp(coeff), ref))
    lines.append("ctx.lincomb(%s, %s, %s, static_cast<adc::Real>(1), acc_%s);"
                 % (target, _coeff_cpp(c_target), target, target))
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
    if with_cons:
        for idx, c in enumerate(impl.cons_names):
            if c in deps:
                lines.append("const adc::Real %s = %sA(i, j, %d);" % (c, state_var, idx))
    if with_prim:
        live = impl._live_prims(exprs)  # closure over the live primitives
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
#include <adc/numerics/elliptic/linear/generic_krylov.hpp>  // adc::cg_solve / bicgstab_solve / richardson_solve (matrix-free)
#include <adc/core/foundation/types.hpp>
#include <cmath>                               // std::sqrt / std::fabs / std::pow in lowered formulas
#include <memory>                              // std::make_shared (persistent matrix-free scratch)

extern "C" const char* adc_program_abi_key() {{ return ADC_ABI_KEY_LITERAL; }}
extern "C" const char* adc_program_name() {{ return {name}; }}
extern "C" const char* adc_program_hash() {{ return "{hash}"; }}

extern "C" void adc_install_program(void* sys) {{
  adc::runtime::program::ProgramContext ctx(sys);
{prelude}
  ctx.install([=](double dt) {{
    (void)dt;
{body}
  }});
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


# adc.time.std.<scheme>(Program, block, ...) -- the spec's standard library entry point.
std = types.SimpleNamespace(forward_euler=forward_euler, ssprk2=ssprk2, ssprk3=ssprk3, rk4=rk4,
                            strang=strang)


class CompiledTime:
    """Time-stepping policy for a compiled `Program`, passed to `sim.add_equation(..., time=...)`.

    A compiled Program OWNS the whole step body (it is installed via `sim.install_program` and driven
    by `sim.step(dt)`); `CompiledTime` records the macro-step cadence around it. MVP scope: a single
    Forward-Euler step. `substeps > 1`, `stride > 1`, and a non-default `cfl` are deferred -- the
    Program currently receives a bare `dt` and `step_cfl` / substep / stride orchestration is not yet
    wired for the program path (cf. system_stepper.hpp; ADC-401 Phase 2c) -- so they fail loud here
    rather than be silently ignored."""

    def __init__(self, substeps=1, stride=1, cfl="default"):
        if substeps != 1:
            raise NotImplementedError(
                "CompiledTime: substeps > 1 is deferred (ADC-401 Phase 2c); the compiled Program "
                "receives a bare dt with no substep orchestration yet")
        if stride != 1:
            raise NotImplementedError(
                "CompiledTime: stride > 1 is deferred (ADC-401 Phase 2c); the compiled Program has "
                "no macro-step cadence yet")
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
