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

__all__ = ["Program", "std"]


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


class Value:
    """A typed SSA node in a Program IR. Field-like values (State, RHS) support affine arithmetic."""

    _FIELD = ("state", "rhs")

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
        raise TypeError("runtime Scalar cannot be used as a Python bool; use ctx.if_ or "
                        "Program control flow")

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

    def commit(self, block, state):
        """Replace the current state of ``block`` with ``state`` at the end of the step. Each block
        is committed AT MOST once; read-only blocks need no commit."""
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("commit: a State value is required")
        if state.prog is not self:
            raise ValueError("commit: the State value belongs to a different Program")
        if block in self._commits:
            raise ValueError("block '%s' committed more than once" % block)
        self._commits[block] = state

    def commits(self):
        """Map of committed block -> committed State value (copy)."""
        return dict(self._commits)

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
        return True

    # --- serialization / hash ---
    def _serialize(self):
        nodes = []
        for v in self._values:
            attrs = dict(v.attrs)
            if "coeffs" in attrs:  # dict keys (powers) -> sorted [power, value] for stable JSON
                attrs["coeffs"] = [sorted((int(p), c) for p, c in d.items()) for d in attrs["coeffs"]]
            nodes.append({"id": v.id, "vtype": v.vtype, "op": v.op, "block": v.block,
                          "inputs": [i.id for i in v.inputs], "attrs": attrs})
        commits = sorted((b, s.id) for b, s in self._commits.items())
        return {"name": self.name, "version": 1, "nodes": nodes, "commits": commits}

    def _ir_hash(self):
        """Stable SHA-256 of the IR (feeds the compiled-problem cache key in a later phase)."""
        blob = json.dumps(self._serialize(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


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
