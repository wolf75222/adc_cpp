"""Blackboard-style physics model authoring (Spec 3, layer 1).

``pops.physics.Model`` lets a user write a model the way it appears on a
blackboard -- a state, primitives, a flux, an elliptic field solve, sources and
local linear operators, tied together by equations such as
``ddt(U) == -div(F) + S`` and ``-laplacian(phi) == rho`` -- and lowers it to the
Spec 2 operator-first IR (:class:`pops.model.Module`) and the :mod:`pops.dsl`
codegen engine. It is a thin TRANSLATION layer: it owns no numerics and no
codegen of its own. ``pops.dsl.Model`` (the PDE facade) remains valid; the board
API is sugar that produces the same typed operators.

The board notation lives in :mod:`pops.math` (``ddt`` / ``div`` / ``grad`` /
``laplacian`` / ``sqrt`` / ``rate`` / ``unknown`` / ``integral``). The typed view
is reachable through :pyattr:`Model.module`; the codegen model through
:pyattr:`Model.dsl`.

Multi-species board authoring (``m.species`` for N >= 2, ``m.coupled_rate``,
``m.solve_fields_from_species``) LOWERS to the existing operator-first multi-block
IR (an :class:`pops.model.Module` with N :class:`pops.model.StateSpace`, a
``coupled_rate`` operator over a :class:`pops.model.RateBundle`, and a multi-input
field operator), not a second runtime: the board surface produces the SAME typed
operators a hand-written ``pops.model.Module`` registers (ADC-457). The single-species
path is byte-identical to the single-state board model. The compiled multi-block
``.so`` run is validated on ROMEO (Kokkos-only AOT).
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


# Board role vocabulary -> dsl canonical role (pops::VariableRole). The dsl roles_for() uses an
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
    """A declared state: a name plus the ordered :mod:`pops.dsl` component vars.

    Unpacks into its components (``rho, mx, my = U``), indexes them by position
    (``U[0]``) or by component name (``e["ne"]`` -- the board access of Spec 3
    section 12.3/16), and remembers its name and roles for the typed
    :class:`pops.model.StateSpace`. The string index returns the conservative
    :class:`pops.dsl.Var` of that component, so a board coupled-rate formula
    written as ``e["ni"] - e["ne"]`` is the same IR as the hand-written
    operator-first ``dsl.Var("ni", "cons") - dsl.Var("ne", "cons")``.
    """

    def __init__(self, name, components, vars_, roles, space=None):
        self.name = str(name)
        self.components = tuple(components)
        self.vars = tuple(vars_)
        self.roles = dict(roles or {})
        # The typed pops.model.StateSpace this species instantiates (multi-species
        # mode); None for the single-state dsl-backed path, where the space is
        # derived on demand from the dsl model.
        self.space = space

    def __iter__(self):
        return iter(self.vars)

    def __len__(self):
        return len(self.vars)

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                return self.vars[self.components.index(key)]
            except ValueError:
                raise KeyError(
                    "state %r has no component %r (have: %s)"
                    % (self.name, key, ", ".join(self.components))) from None
        return self.vars[key]

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
        # Multi-species mode (Spec 3 sections 12, 16): once a SECOND species is declared the
        # model owns a multi-block pops.model.Module directly (N StateSpaces + a coupled_rate +
        # a multi-block field operator). The single-species path stays byte-identical: it keeps
        # the dsl.Model and exposes dsl.Model.module. _multi_module is None until N > 1.
        self._multi_module = None
        self._species = {}          # species name -> StateHandle (multi-species mode)

    # --- escape hatches ---
    @property
    def dsl(self):
        """The underlying :class:`pops.dsl.Model` (the codegen engine)."""
        return self._dsl

    @property
    def module(self):
        """The typed :class:`pops.model.Module` view (operator-first IR).

        Single-species: the dsl-derived Module (one StateSpace). Multi-species: the
        multi-block Module this model assembled directly (N StateSpaces, a
        ``coupled_rate`` operator, a multi-block field operator) -- the SAME
        operator-first IR a hand-written :class:`pops.model.Module` would build.
        """
        if self._multi_module is not None:
            return self._multi_module
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
        """Declare a named species: a named block instance of its own StateSpace.

        Each species lowers to one :class:`pops.model.StateSpace` and a named block
        (Spec 3 sections 12, 16). The returned :class:`StateHandle` unpacks into its
        component vars and indexes them by name (``e["ne"]``) for a coupled-rate
        formula. Arbitrary arity: declare 2, 3, 4, ... species. The single-species
        case is byte-identical to :meth:`state` (no multi-block Module is created);
        the multi-block path engages only from the SECOND species, lowering to the
        existing operator-first multi-block IR (``pops.model.Module`` with N spaces +
        ``coupled_rate`` + ``solve_fields_from_blocks``), never a parallel runtime.
        """
        if name in self._species:
            raise ValueError(
                "species %r is already declared; each species needs a distinct name "
                "(a reused name would silently alias the StateSpace)" % name)
        if not self._species and not self._multi_module:
            # First species: keep the single-state dsl-backed path byte-identical to state().
            handle = self.state(name, components=state, roles=roles)
            self._species[handle.name] = handle
            return handle
        # Second (or later) species: promote to multi-species mode. The first species was
        # authored on the dsl model; re-realize it as a typed StateSpace on the multi-block
        # Module so all species live in the SAME operator-first IR.
        self._promote_to_multispecies()
        return self._add_species(name, components=state, roles=roles)

    def _promote_to_multispecies(self):
        """Build the multi-block :class:`pops.model.Module` and migrate the first species into it.

        The single-state dsl model authored the first species; multi-species mode realizes every
        species as a typed StateSpace on a shared Module so N >= 2 species lower to the existing
        operator-first multi-block IR (N spaces + coupled_rate + multi-block field solve), not a
        second runtime. The first species' :class:`StateHandle` is updated IN PLACE (its ``.space``
        is set) so the reference the caller already holds stays valid after promotion."""
        if self._multi_module is not None:
            return
        from . import model as _model
        self._multi_module = _model.Module(self.name)
        for nm, h in self._species.items():
            self._add_species(nm, components=h.components, roles=h.roles, handle=h)

    def _add_species(self, name, components=(), roles=None, handle=None):
        """Add one typed StateSpace to the multi-block Module and return its StateHandle.

        ``handle`` updates an existing :class:`StateHandle` in place (promotion of the first
        species); otherwise a fresh handle is created and recorded."""
        from . import dsl as _dsl
        comps = tuple(components)
        canon = {c: _canon_role(roles.get(c)) for c in comps} if roles else {}
        space = self._multi_module.state_space(str(name), comps, roles=canon)
        vars_ = tuple(_dsl.Var(c, "cons") for c in comps)
        if handle is None:
            handle = StateHandle(name, comps, vars_, roles, space=space)
        else:
            handle.vars = vars_
            handle.space = space
        self._species[handle.name] = handle
        self._states[handle.name] = handle
        return handle

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

    def coupled_rate(self, name, inputs=(), outputs=None, preserves=None, dissipates=None):
        """Declare a coupled rate over several species (collisions, ionization, radiation).

        ``inputs`` is the ordered list of participating species (:class:`StateHandle`); a species
        may appear as a READ-ONLY catalyst input without being an output block. ``outputs`` maps
        each output species to its per-component rate formulas (one expression per cons component,
        written over the input species' cons vars via ``e["ne"]``). Arbitrary arity: 2, 3, 4, ...
        inputs, no two-input limit.

        Lowers to the existing operator-first ``coupled_rate`` operator (the SAME kind #287/#300
        lower): a :class:`pops.model.RateBundle` signature over the input :class:`StateSpace` set,
        with the per-block component formulas as the operator body. ``preserves`` / ``dissipates``
        are recorded as capabilities (a generic invariant tag), not numerics. Requires multi-species
        mode (declare the species with :meth:`species`).
        """
        from . import model as _model
        if self._multi_module is None:
            raise ValueError(
                "coupled_rate(%r) needs at least two species; declare them with m.species(...)"
                % (name,))
        in_handles = self._as_species_list("coupled_rate", name, inputs)
        if not outputs:
            raise ValueError("coupled_rate(%r) requires outputs={species: [per-component exprs]}"
                             % (name,))
        in_spaces = tuple(h.space for h in in_handles)
        bundle = _model.RateBundle()
        expr = {}
        for sp, comps in outputs.items():
            h = self._species_handle("coupled_rate", name, sp)
            comp_list = [self._to_expr(c) for c in self._as_iter(comps)]
            if len(comp_list) != len(h.components):
                raise ValueError(
                    "coupled_rate(%r) output %r has %d component formula(s) but its state %r has %d"
                    % (name, h.name, len(comp_list), h.name, len(h.components)))
            bundle.add(h.name, h.space)
            expr[h.name] = comp_list
        caps = {}
        if preserves is not None:
            caps["preserves"] = preserves
        if dissipates is not None:
            caps["dissipates"] = dissipates
        reg = _safe_name(name)
        self._multi_module.operator(name=reg, kind="coupled_rate",
                                    signature=_model.Signature(in_spaces, bundle),
                                    capabilities=caps or None, expr=expr)
        return CallableOperator(reg, self)

    def solve_fields_from_species(self, name, inputs=(), equation=None, outputs=None, solver=None):
        """Declare a coupled field solve over several species (multi-block Poisson).

        ``inputs`` is the ordered list of contributing species; the field RHS reads every listed
        species' stage state at once. Lowers to a typed ``field_operator`` over the N input
        :class:`StateSpace` set, the operator-first surface of ``P.solve_fields_from_blocks``
        (the existing multi-block field solve, Spec 3 criterion 24). ``equation`` /  ``outputs`` /
        ``solver`` record the elliptic problem and the produced fields for introspection.
        """
        from . import model as _model
        if self._multi_module is None:
            raise ValueError(
                "solve_fields_from_species(%r) needs at least two species; declare them with "
                "m.species(...)" % (name,))
        in_handles = self._as_species_list("solve_fields_from_species", name, inputs)
        in_spaces = tuple(h.space for h in in_handles)
        out_comps = tuple(outputs.keys()) if outputs else ("phi",)
        fields = self._multi_module.field_space(_safe_name(name), out_comps)
        reg = _safe_name(name)
        reqs = {"solver": solver} if solver is not None else None
        self._multi_module.operator(name=reg, kind="field_operator",
                                    signature=_model.Signature(in_spaces, fields),
                                    requirements=reqs,
                                    expr={"blocks": [h.name for h in in_handles]})
        h = FieldsHandle(name, outputs, solver)
        self._fields[name] = h
        if solver is not None:
            self._field_solvers[name] = solver
        return h

    def _as_species_list(self, op, name, items):
        """Resolve a list of species handles / names to StateHandles (multi-species mode)."""
        if not items:
            raise ValueError("%s(%r) requires inputs=[species, ...]" % (op, name))
        return [self._species_handle(op, name, s) for s in self._as_iter(items)]

    def _species_handle(self, op, name, sp):
        """Resolve one species (a StateHandle or a species name) to its StateHandle."""
        if isinstance(sp, StateHandle):
            handle = self._species.get(sp.name)
        else:
            handle = self._species.get(str(sp))
        if handle is None:
            known = ", ".join(self._species) or "<none>"
            raise KeyError("%s(%r): unknown species %r (declared: %s)"
                           % (op, name, sp, known))
        return handle

    @staticmethod
    def _as_iter(x):
        """A list view of a single item or an iterable (so inputs=e and inputs=[e, i] both work)."""
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    def finite_volume_rate(self, name, flux=None, riemann=None, reconstruction=None,
                           sources=()):
        """Declare a rate assembled by the native finite-volume machinery.

        Selects the native Riemann solver and reconstruction (by descriptor) and
        the source terms; lowers to the same rate operator a board equation does.
        The native-brick hook codegen for a custom Riemann is tracked by ADC-456.
        """
        self._reconstruction = reconstruction
        # Selecting a Riemann solver validates the model's capabilities for it and enables
        # the role-derived hooks (criterion 10); accept a string or an pops.lib descriptor.
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

        The native solvers are C++ (``pops::RusanovFlux`` / ``HLLFlux`` / ``HLLCFlux`` /
        ``RoeFlux``). HLLC/Roe need model capabilities: a pressure primitive and the fluid
        roles Density/MomentumX/MomentumY (the dsl ``enable_hllc`` / ``enable_roe`` then
        generate the ``POPS_HD`` ``contact_speed`` / ``hllc_star_state`` / ``roe_dissipation``
        hooks from those roles). Missing capabilities are rejected here with a clear message
        (Spec 3 criterion 10).

        ADC-456: passing an explicit board formula for a capability quantity (e.g.
        ``pressure=<pops.math expr>``) overrides the role-derived hook with that formula's codegen
        (lowered via :meth:`pops.dsl.Model.set_riemann_hooks`). A capability hook DESCRIPTOR
        (``pops.lib.riemann.hllc.contact_speed.euler()``) or ``None`` keeps the role-derived default.
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
        """Validate that every referenced quantity is declared (single-species path).

        Multi-species models compose their blocks in a time Program and validate at emit
        (``P.emit_cpp_program`` / ``P._check_lowerable``), so a model-level ``check`` is a
        single-species notion; it is a no-op for a multi-species model."""
        if self._multi_module is not None:
            return None
        return self._dsl.check()

    def compile(self, *args, **kwargs):
        """Compile the single-species model to a ``.so`` (delegates to
        :meth:`pops.dsl.Model.compile`).

        A multi-species model is compiled as a multi-block time Program (the operator-first
        path: ``m.module`` bound to a :class:`pops.time.Program`, then ``P.compile_problem`` /
        ``emit_cpp_program``), not through this single-state shortcut, so this raises a clear
        error rather than compiling an empty single-state model."""
        if self._multi_module is not None:
            raise NotImplementedError(
                "a multi-species physics.Model compiles as a multi-block time Program: bind "
                "m.module to an pops.time.Program and emit/compile that (the operator-first "
                "multi-block path), not via m.compile() (ADC-457)")
        return self._dsl.compile(*args, **kwargs)

    # --- introspection ---
    def list_operators(self):
        if self._multi_module is not None:
            return self._multi_module.list_operators()
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
        """The operator-first :class:`pops.model.Module` this model lowers to: the typed
        spaces and operators with signatures (layer 2)."""
        mod = self.module
        reg = mod.operator_registry()
        lines = ["# pops.model.Module %s" % mod.name]
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
        """Resolve a board node to an :mod:`pops.dsl` expression in this model's context."""
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
