"""Blackboard-style physics model authoring (Spec 3, layer 1).

``adc.physics.Model`` lets a user write a model the way it appears on a
blackboard -- a state, primitives, a flux, an elliptic field solve, sources and
local linear operators, tied together by equations such as
``ddt(U) == -div(F) + S`` and ``-laplacian(phi) == rho`` -- and lowers it to the
Spec 2 operator-first IR (:class:`adc.model.Module`) and the :mod:`adc.dsl`
codegen engine. It is a thin TRANSLATION layer: it owns no numerics and no
codegen of its own. ``adc.dsl.Model`` (the PDE facade) remains valid; the board
API is sugar that produces the same typed operators.

The board notation lives in :mod:`adc.math` (``ddt`` / ``div`` / ``grad`` /
``laplacian`` / ``sqrt`` / ``rate`` / ``unknown`` / ``integral``). The typed view
is reachable through :pyattr:`Model.module`; the codegen model through
:pyattr:`Model.dsl`.

Multi-species board authoring (``m.species`` for N > 1, ``RateBundle``,
``commit_many``) lowers to a multi-block runtime tracked by ADC-457; this module
implements the single-state board model and the multi-species *authoring types*
that do not need that runtime.
"""
import re

from . import math as _bm

__all__ = ["Model", "Invariant", "FluxHandle", "SourceHandle", "FieldsHandle",
           "LocalLinearOperatorExpr", "CallableOperator", "StateHandle", "VectorHandle"]


def _safe_name(name):
    """A C-identifier-safe operator name derived from a board display name."""
    s = re.sub(r"[^0-9a-zA-Z_]", "_", str(name)).strip("_")
    if not s:
        raise ValueError("operator name %r has no identifier characters" % (name,))
    if s[0].isdigit():
        s = "_" + s
    return s


# Board role vocabulary -> dsl canonical role (adc::VariableRole). The dsl roles_for() uses an
# explicit role override verbatim, so a board role must already be canonical for the native HLLC/Roe
# role lookup (which indexes "Density"/"MomentumX"/"MomentumY"/"Energy") to find it.
_BOARD_ROLE = {
    "density": "Density",
    "momentum_x": "MomentumX", "momentum_y": "MomentumY", "momentum_z": "MomentumZ",
    "energy": "Energy", "pressure": "Pressure", "temperature": "Temperature",
}


def _canon_role(role):
    """Canonicalize a board role string to a dsl role; pass through None and unknown roles."""
    if role is None:
        return None
    return _BOARD_ROLE.get(str(role).lower(), role)


def _roles_for(hyp):
    """The canonical dsl roles of a HyperbolicModel's conservative state."""
    from . import dsl as _dsl
    return _dsl.roles_for(hyp.cons_names, hyp.cons_roles)


class StateHandle:
    """A declared state: a name plus the ordered :mod:`adc.dsl` component vars.

    Unpacks into its components (``rho, mx, my = U``) and remembers its name and
    roles for the typed :class:`adc.model.StateSpace`.
    """

    def __init__(self, name, components, vars_, roles):
        self.name = str(name)
        self.components = tuple(components)
        self.vars = tuple(vars_)
        self.roles = dict(roles or {})

    def __iter__(self):
        return iter(self.vars)

    def __len__(self):
        return len(self.vars)

    def __getitem__(self, i):
        return self.vars[i]

    def __repr__(self):
        return "StateHandle(%r, %r)" % (self.name, list(self.components))


class FieldHandle:
    """A solved/auxiliary scalar field (e.g. the potential ``phi``)."""

    def __init__(self, name):
        self.name = str(name)

    def __repr__(self):
        return "FieldHandle(%r)" % (self.name,)


class VectorHandle:
    """A named vector field with ``.x`` / ``.y`` expression components."""

    def __init__(self, name, x, y):
        self.name = str(name)
        self.x = x
        self.y = y

    def __repr__(self):
        return "VectorHandle(%r)" % (self.name,)


class FluxHandle:
    """A declared physical flux (the default hyperbolic flux of a model)."""

    def __init__(self, name, is_default=True):
        self.name = str(name)
        self.is_default = bool(is_default)

    def __repr__(self):
        return "FluxHandle(%r)" % (self.name,)


class SourceHandle(_bm.RateTerm):
    """A declared local source term -- a summand of a rate equation."""

    def __init__(self, display_name, reg_name):
        self.name = str(display_name)
        self.reg_name = str(reg_name)

    def _rate_terms(self):
        return [("source", self, 1.0)]

    def __repr__(self):
        return "SourceHandle(%r)" % (self.name,)


class LocalLinearOperatorExpr:
    """A LOCAL linear operator object ``L: U -> U`` -- a MATH object, not a callable operator.

    ``m.local_linear_operator(...)`` returns this; it carries the matrix but is NOT yet a
    typed registry operator. Register it with ``m.operator(name, returns=...)`` (or
    ``@module.operator``) to obtain a callable operator. Calling the math object directly
    is an error -- it cannot resolve its field inputs without a registration.
    """

    def __init__(self, display_name, matrix, on=None):
        self.name = str(display_name)
        self.matrix = matrix
        self.on = on

    def __call__(self, *args, **kwargs):
        raise TypeError(
            "local_linear_operator object %r is not a callable operator. Register it with "
            "m.operator(%r, returns=...) or @module.operator(...) first." % (self.name, self.name))

    def __repr__(self):
        return "LocalLinearOperatorExpr(%r)" % (self.name,)


class CallableOperator:
    """A registered, typed operator usable in a time Program: ``op(U, fields, ...)``.

    Returned by ``m.rate`` / ``m.operator``. Calling it with Program values lowers to
    ``P.call(name, ...)`` on the values' Program (binding the model's operator registry on
    first use), so a board-style program can write ``explicit_rate(U_n, fields_n)`` and get
    the same IR as the explicit operator-first ``P.call("explicit_rate", U_n, fields_n)``.
    """

    def __init__(self, name, model):
        self.name = str(name)
        self.reg_name = self.name
        self._model = model     # bound to its FRESH module at call time (sees all operators)

    def __call__(self, *args, name=None):
        prog = next((a.prog for a in args if hasattr(a, "prog")), None)
        if prog is None:
            raise ValueError(
                "operator %r must be called with time-Program values (inside a Program); "
                "got %r" % (self.name, args))
        reg = getattr(prog, "_registry", None)
        # Bind (or rebind) the model's FRESH module if the program has no registry yet or
        # the bound one predates this operator -- so operators registered in any order all
        # resolve, not just those present when the program was first bound.
        if self._model is not None and (reg is None or self.name not in reg):
            prog.bind_operators(self._model.module)
        return prog.call(self.name, *args, name=name)

    def __repr__(self):
        return "CallableOperator(%r)" % (self.name,)


class FieldsHandle:
    """The result of a field-solve operator: a named bundle of solved fields."""

    def __init__(self, name, outputs, solver):
        self.name = str(name)
        self.outputs = dict(outputs or {})
        self.solver = solver

    def __repr__(self):
        return "FieldsHandle(%r)" % (self.name,)


class Invariant:
    """A generic invariant: a typed function ``StateSet -> Scalar``.

    Carries a board ``integral(...)`` value expression and the states it ranges
    over. Nothing about mass / charge / momentum / energy is built in: the value
    is whatever the user writes. Used for diagnostics and conservation checks.
    """

    def __init__(self, name, value, over=None):
        self.name = str(name)
        self.value = value
        self.over = tuple(over) if over else ()

    def __repr__(self):
        return "Invariant(%r)" % (self.name,)


class Model:
    """A blackboard-style physical model that lowers to the operator-first IR."""

    def __init__(self, name):
        from . import dsl as _dsl  # lazy: dsl pulls numpy
        self._dsl = _dsl.Model(name)
        self.name = str(name)
        self._states = {}
        self._fields = {}
        self._fluxes = {}
        self._sources = {}
        self._operators = {}
        self._operator_inputs = {}  # registered op name -> declared field-input names
        self._aliases = {}          # board operator name -> registered op name
        self._invariants = {}
        self._riemann = None        # selected Riemann descriptor (board surface)
        self._reconstruction = None
        self._riemann_hooks = {}    # capability formulas for the native-hook codegen (ADC-456)
        self._field_solvers = {}    # field-operator name -> solver descriptor

    # --- escape hatches ---
    @property
    def dsl(self):
        """The underlying :class:`adc.dsl.Model` (the codegen engine)."""
        return self._dsl

    @property
    def module(self):
        """The typed :class:`adc.model.Module` view (operator-first IR)."""
        return self._dsl.module

    # --- state / species ---
    def state(self, name="U", components=(), roles=None):
        """Declare the conservative state. Returns an unpackable :class:`StateHandle`.

        Board role strings (``density`` / ``momentum_x`` / ``momentum_y`` / ``energy`` / ...)
        are canonicalized to the dsl roles (``Density`` / ``MomentumX`` / ...) so the native
        Riemann capabilities (HLLC/Roe role lookup) recognize them.
        """
        role_list = None
        if roles:
            role_list = [_canon_role(roles.get(c)) for c in components]
        vars_ = self._dsl.conservative_vars(*components, roles=role_list)
        handle = StateHandle(name, components, vars_, roles)
        self._states[handle.name] = handle
        return handle

    def species(self, name, state=(), roles=None):
        """Declare a named species (a :class:`StateHandle`).

        For a single species this is exactly :meth:`state` under the species name.
        Multiple species lower to a multi-block runtime tracked by ADC-457; a
        second species raises a clear error rather than silently mis-lowering.
        """
        if self._states:
            raise NotImplementedError(
                "multi-species board models (a second species %r) lower to the "
                "multi-block runtime tracked by ADC-457; this build supports one "
                "species per physics.Model" % (name,))
        return self.state(name, components=state, roles=roles)

    # --- quantities ---
    def primitive(self, name, expr):
        """Define a primitive quantity by its formula; returns a usable expression."""
        return self._dsl.primitive(name, expr)

    def scalar(self, name, expr):
        """Define a named derived scalar (e.g. pressure, sound speed)."""
        return self._dsl.primitive(name, expr)

    def param(self, name, value=0.0, kind="const"):
        """Declare a named scalar parameter; returns a usable expression."""
        return self._dsl.param(name, value, kind=kind)

    def aux(self, name):
        """Declare an auxiliary field read by the model (e.g. an imposed ``B_z``)."""
        canonical = {"phi", "grad_x", "grad_y", "B_z", "T_e"}
        if name in canonical:
            return self._dsl.aux(name)
        return self._dsl.aux_field(name)

    def field(self, name):
        """Declare a solved scalar field (e.g. the potential ``phi``)."""
        h = FieldHandle(name)
        self._fields[h.name] = h
        return h

    def vector_field(self, name, x, y):
        """Define a named vector field with ``.x`` / ``.y`` expression components."""
        h = VectorHandle(name, self._to_expr(x), self._to_expr(y))
        self._fields[name] = h
        return h

    # --- operators (board equations) ---
    def flux(self, name, on=None, x=None, y=None, waves=None):
        """Declare the physical flux and (optionally) its characteristic speeds.

        ``x`` / ``y`` are the per-component flux expressions; ``waves`` gives the
        per-direction eigenvalues. Lowers to the model's default flux.
        """
        if x is None or y is None:
            raise ValueError("flux(%r) requires per-component x= and y= expressions" % (name,))
        self._dsl.flux(list(x), list(y))
        if waves is not None:
            self._dsl.eigenvalues(list(waves["x"]), list(waves["y"]))
        h = FluxHandle(name, is_default=True)
        self._fluxes[name] = h
        return h

    def source(self, name, on=None, value=None):
        """Declare a named local source term; returns a :class:`SourceHandle`."""
        if value is None:
            raise ValueError("source(%r) requires value= (one expression per component)" % (name,))
        reg = _safe_name(name)
        self._dsl.source_term(reg, [self._to_expr(e) for e in value])
        h = SourceHandle(name, reg)
        self._sources[reg] = h
        return h

    def local_linear_operator(self, name, on=None, matrix=None):
        """Build a local linear operator ``L: U -> U`` as a MATH object (not a callable
        operator). It carries the matrix; register it with :meth:`operator` (or
        ``@module.operator``) to obtain a callable operator. Calling the math object
        directly raises a clear error -- see :class:`LocalLinearOperatorExpr`."""
        if matrix is None:
            raise ValueError("local_linear_operator(%r) requires matrix=" % (name,))
        return LocalLinearOperatorExpr(name, matrix, on=on)

    def solve_field(self, name, equation=None, outputs=None, solver=None):
        """Declare an elliptic field solve ``-laplacian(phi) == rhs``.

        Lowers to the model's Poisson coupling; ``outputs`` names the produced
        fields, ``solver`` records the required elliptic solver.
        """
        if not isinstance(equation, _bm.Equation):
            raise TypeError("solve_field expects an equation '-laplacian(phi) == rhs'")
        lhs = equation.lhs
        if not isinstance(lhs, _bm.Laplacian):
            raise ValueError(
                "solve_field left-hand side must be (-)laplacian(field); got %r" % (lhs,))
        rhs = self._to_expr(equation.rhs)
        # -laplacian(phi) == rhs  ->  -Delta phi = rhs (the dsl Poisson convention).
        # laplacian(phi) == rhs  ->  -Delta phi = -rhs.
        if lhs.scale > 0:
            rhs = -rhs
        self._dsl.elliptic_rhs(rhs)
        h = FieldsHandle(name, outputs, solver)
        self._fields[name] = h
        if solver is not None:
            self._field_solvers[name] = solver
        return h

    def rate(self, name, equation):
        """Declare a rate operator from ``ddt(U) == -div(F) + sources``."""
        if not isinstance(equation, _bm.Equation):
            raise TypeError("rate expects an equation 'ddt(U) == -div(F) + sources'")
        if not isinstance(equation.lhs, _bm.TimeDerivative):
            raise ValueError("rate left-hand side must be ddt(U) / rate(U)")
        flux, sources = self._destructure_rate(equation.rhs)
        self._dsl.rate_operator(_safe_name(name), flux=flux, sources=sources)
        return CallableOperator(_safe_name(name), self)

    def finite_volume_rate(self, name, flux=None, riemann=None, reconstruction=None,
                           sources=()):
        """Declare a rate assembled by the native finite-volume machinery.

        Selects the native Riemann solver and reconstruction (by descriptor) and
        the source terms; lowers to the same rate operator a board equation does.
        The native-brick hook codegen for a custom Riemann is tracked by ADC-456.
        """
        self._reconstruction = reconstruction
        # Selecting a Riemann solver validates the model's capabilities for it and enables
        # the role-derived hooks (criterion 10); accept a string or an adc.lib descriptor.
        if riemann is not None:
            scheme = getattr(riemann, "scheme", riemann)
            self.riemann(scheme)
        src_names = [s.reg_name if isinstance(s, SourceHandle) else _safe_name(s)
                     for s in sources]
        # A finite-volume rate always assembles -div F; the flux selection is recorded
        # for the native bricks (riemann/reconstruction), not toggled off here.
        self._dsl.rate_operator(_safe_name(name), flux=True, sources=src_names)
        return name

    def operator(self, name, handle=None, *, inputs=None, returns=None):
        """Register a typed, callable operator under ``name`` from a math object.

        ``returns`` (or the positional ``handle``) is the operator body; ``inputs`` names
        its field dependencies (metadata for requirements). A
        :class:`LocalLinearOperatorExpr` registers as a ``local_linear_operator``
        ``Fields -> LocalLinearOperator(U, U)``. Returns a :class:`CallableOperator`.
        """
        obj = returns if returns is not None else handle
        if obj is None:
            raise TypeError("operator(%r) requires returns= (or a positional handle)" % (name,))
        reg = _safe_name(name)
        if isinstance(obj, LocalLinearOperatorExpr):
            self._dsl.linear_source(
                reg, [[self._to_expr(e) for e in row] for row in obj.matrix])
            self._operators[reg] = obj
            self._operator_inputs[reg] = tuple(inputs) if inputs else ()
            return CallableOperator(reg, self)
        if isinstance(obj, CallableOperator):
            # aliasing an already-registered operator under a new role name
            self._aliases[name] = obj.reg_name
            return obj
        raise TypeError(
            "operator(%r): returns= must be a local_linear_operator object or a "
            "registered operator; got %r" % (name, obj))

    def riemann(self, name, flux=None, pressure=None, velocity=None, sound_speed=None,
                wave_speeds=None, contact_speed=None, star_state=None):
        """Select a Riemann solver and validate the model's capabilities for it.

        The native solvers are C++ (``adc::RusanovFlux`` / ``HLLFlux`` / ``HLLCFlux`` /
        ``RoeFlux``). HLLC/Roe need model capabilities: a pressure primitive and the fluid
        roles Density/MomentumX/MomentumY (the dsl ``enable_hllc`` / ``enable_roe`` then
        generate the ``ADC_HD`` ``contact_speed`` / ``hllc_star_state`` / ``roe_dissipation``
        hooks from those roles). Missing capabilities are rejected here with a clear message
        (Spec 3 criterion 10).

        ADC-456: passing an explicit board formula for a capability quantity (e.g.
        ``pressure=<adc.math expr>``) overrides the role-derived hook with that formula's codegen
        (lowered via :meth:`adc.dsl.Model.set_riemann_hooks`). A capability hook DESCRIPTOR
        (``adc.lib.riemann.hllc.contact_speed.euler()``) or ``None`` keeps the role-derived default.
        A formula referencing a quantity the model cannot provide still raises the clear capability
        error at codegen.
        """
        self._riemann = name
        self._riemann_hooks = {
            "flux": flux, "pressure": pressure, "velocity": velocity,
            "sound_speed": sound_speed, "wave_speeds": wave_speeds,
            "contact_speed": contact_speed, "star_state": star_state,
        }
        kind = str(name).lower()
        self._validate_riemann_capabilities(kind, pressure, wave_speeds)
        if kind == "hllc":
            self._dsl.enable_hllc()
        elif kind == "roe":
            self._dsl.enable_roe()
        # Wire any ARBITRARY board formula through to the dsl codegen (ADC-456). Resolve board nodes
        # to dsl Exprs; the dsl method codegen's the Expr ones and ignores descriptors / None (the
        # role-derived default stands). Off the hot path for the role-derived case (no Expr -> no-op).
        self._dsl.set_riemann_hooks(
            pressure=self._to_expr(pressure) if pressure is not None else None,
            sound_speed=self._to_expr(sound_speed) if sound_speed is not None else None,
            contact_speed=self._to_expr(contact_speed) if contact_speed is not None else None,
            star_state=self._to_expr(star_state) if star_state is not None else None,
        )
        return name

    def _validate_riemann_capabilities(self, kind, pressure, wave_speeds):
        """Reject a model that lacks the capabilities the chosen Riemann solver needs
        (Spec 3 criterion 10). Rusanov needs only a max wave speed (always available from the
        flux/eigenvalues); HLL needs wave speeds; HLLC/Roe need a pressure and fluid roles."""
        hyp = self._dsl._m
        roles = set(_roles_for(hyp))
        has_pressure = ("p" in hyp.prim_defs) or (pressure is not None)
        fluid = {"Density", "MomentumX", "MomentumY"}
        if kind in ("hllc", "roe"):
            if not has_pressure:
                raise ValueError(
                    "riemann %s requires model capability 'pressure' for state %r: declare a "
                    "primitive m.primitive('p', ...) or pass m.riemann(..., pressure=...)"
                    % (kind.upper(), self._state_name()))
            missing = fluid - roles
            if missing:
                raise ValueError(
                    "riemann %s requires model capability 'hllc_star_state' for state %r: the "
                    "fluid roles %s are needed (declare m.state(..., roles={...})); missing %s"
                    % (kind.upper(), self._state_name(),
                       sorted(fluid), sorted(missing)))
        elif kind == "hll":
            if (wave_speeds is None and not hyp._eig and hyp._wave_speeds is None
                    and hyp._ws_jacobian is None):
                raise ValueError(
                    "riemann HLL requires model capability 'wave_speeds': declare m.flux(..., "
                    "waves=...) or pass m.riemann('hll', wave_speeds=...)")
        # rusanov: only max_wave_speed, always derivable -> no extra requirement.

    def _state_name(self):
        return next(iter(self._states), "U")

    def invariant(self, name, expression=None, over=None):
        """Declare a generic invariant ``StateSet -> Scalar`` from an ``integral(...)``."""
        inv = Invariant(name, expression, over=over)
        self._invariants[inv.name] = inv
        return inv

    def invariants(self):
        """The declared invariants, by name."""
        return dict(self._invariants)

    # --- validation / compile ---
    def check(self):
        """Validate that every referenced quantity is declared (delegates to dsl)."""
        return self._dsl.check()

    def compile(self, *args, **kwargs):
        """Compile the model to a ``.so`` (delegates to :meth:`adc.dsl.Model.compile`)."""
        return self._dsl.compile(*args, **kwargs)

    # --- introspection ---
    def list_operators(self):
        return self._dsl.list_operators()

    def operator_alias(self, name):
        """The registered operator name for a board role name (``operator(...)``)."""
        return self._aliases.get(name, name)

    # --- inspection / debug (Spec 3 section 33): show the lowering ---
    def dump_physics(self):
        """A board-level view of what was declared (states, params, fields, fluxes,
        sources, operators) -- the layer-1 surface."""
        lines = ["# physics.Model %s" % self.name]
        lines.append("states: %s" % {n: list(h.components) for n, h in self._states.items()})
        lines.append("params: %s" % list(self._dsl.params))
        lines.append("fields: %s" % list(self._fields))
        lines.append("fluxes: %s" % list(self._fluxes))
        lines.append("sources: %s" % list(self._sources))
        lines.append("invariants: %s" % list(self._invariants))
        lines.append("operators: %s" % self.list_operators())
        return "\n".join(lines)

    def dump_module_ir(self):
        """The operator-first :class:`adc.model.Module` this model lowers to: the typed
        spaces and operators with signatures (layer 2)."""
        mod = self.module
        reg = mod.operator_registry()
        lines = ["# adc.model.Module %s" % mod.name]
        for n, s in mod.state_spaces().items():
            lines.append("StateSpace %s: %s" % (n, list(s.components)))
        for n, f in mod.field_spaces().items():
            lines.append("FieldSpace %s: %s" % (n, list(f.components)))
        for op in mod.list_operators():
            lines.append("Operator %s [%s]: %r" % (op, reg.get(op).kind, mod.operator_signature(op)))
        return "\n".join(lines)

    def dump_capabilities(self):
        """The requirements / capabilities declared by each typed operator."""
        mod = self.module
        lines = ["# capabilities / requirements of %s" % mod.name]
        for op in mod.list_operators():
            lines.append("%s: caps=%s reqs=%s"
                         % (op, mod.operator_capabilities(op), mod.operator_requirements(op)))
        return "\n".join(lines)

    # --- internals ---
    def _destructure_rate(self, rhs):
        """Split a rate right-hand side into ``(flux, [source names])``."""
        terms = _bm._as_rate(rhs)._rate_terms()
        flux = False
        sources = []
        for kind, payload, sign in terms:
            if kind == "flux":
                if sign >= 0:
                    raise ValueError(
                        "a rate equation flux term must be -div(F) (negative); "
                        "write 'ddt(U) == -div(F) + ...'")
                flux = True
            elif kind == "source":
                if sign <= 0:
                    raise ValueError(
                        "a rate equation source term %r must be added (positive sign)"
                        % (payload.name,))
                sources.append(payload.reg_name)
            else:  # pragma: no cover - defensive
                raise ValueError("unknown rate term kind %r" % (kind,))
        return flux, sources

    def _to_expr(self, node):
        """Resolve a board node to an :mod:`adc.dsl` expression in this model's context."""
        if isinstance(node, _bm.Partial):
            field = node.field
            fname = field.name if isinstance(field, FieldHandle) else str(field)
            aux_name = self._gradient_aux(fname, node.axis)
            expr = self._dsl.aux(aux_name)
            if node.scale != 1.0:
                expr = node.scale * expr
            return expr
        if isinstance(node, _bm.Gradient):
            raise TypeError("a gradient is a vector; use grad(field).x / .y")
        if isinstance(node, _bm.Laplacian):
            raise TypeError("a laplacian only appears as a field-solve operator")
        return node  # already a dsl Expr / Var / Param / number

    @staticmethod
    def _gradient_aux(field_name, axis):
        """Canonical gradient aux name of ``field_name`` along ``axis`` (0=x, 1=y)."""
        if field_name == "phi":
            return "grad_x" if axis == 0 else "grad_y"
        # generic fields keep a <field>_grad_x / _grad_y convention
        return "%s_grad_%s" % (field_name, "x" if axis == 0 else "y")

    def __repr__(self):
        return "physics.Model(%r)" % (self.name,)
