"""Blackboard-style physics model authoring (Spec 3, layer 1).

``pops.physics.Model`` lets a user write a model the way it appears on a
blackboard -- a state, primitives, a flux, an elliptic field solve, sources and
local linear operators, tied together by equations such as ``ddt(U) == -div(F) + S``
and ``-laplacian(phi) == rho`` -- and lowers it to the Spec 2 operator-first IR
(:class:`pops.model.Module`) and the :mod:`pops.dsl` codegen engine. It is a thin
TRANSLATION layer: it owns no numerics and no codegen of its own. ``pops.dsl.Model``
(the PDE facade) remains valid; the board API is sugar that produces the same typed
operators.

The board notation lives in :mod:`pops.math` (``ddt`` / ``div`` / ``grad`` /
``laplacian`` / ``sqrt`` / ``rate`` / ``unknown`` / ``integral``). The typed view
is reachable through :pyattr:`Model.module`; the codegen model through
:pyattr:`Model.dsl`.

The handle classes and the multi-species / inspection half live in
``board_handles`` and ``_board_multispecies`` so no file exceeds the Spec-4
500-line bound. Import-graph rule: only :mod:`pops.math` / :mod:`pops.model` /
:mod:`pops.dsl` (the last two LAZILY inside methods); codegen-free, ``_pops``-free.
"""
from .. import math as _bm
from .board_handles import (CallableOperator, FieldHandle, FieldsHandle, FluxHandle,
                            Invariant, LocalLinearOperatorExpr, SourceHandle, StateHandle,
                            VectorHandle, _canon_role, _roles_for, _safe_name)
from ._board_multispecies import _MultiSpeciesMixin



class Model(_MultiSpeciesMixin):
    """A blackboard-style physical model that lowers to the operator-first IR."""

    def __init__(self, name):
        from .facade import Model as _PdeModel  # lazy: the facade pulls numpy
        self._dsl = _PdeModel(name)
        self.name = str(name)
        self._states = {}
        self._fields = {}
        self._fluxes = {}
        self._sources = {}
        self._operators = {}
        self._operator_inputs = {}  # registered op name -> declared field-input names
        self._aliases = {}          # board operator name -> registered op name
        self._invariants = {}
        self._field_problems = {}   # name -> inert pops.fields field problem (Spec 5 sec.5.1/9.6)
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

    def field_problem(self, name, equation, outputs=None, solver=None, bcs=None,
                      coefficients=None):
        """Author an inspectable elliptic field problem (Spec 5 sec.5.1 / sec.9.6).

        The typed-object ergonomic shortcut: it CONSTRUCTS and RETURNS an inert
        :class:`pops.fields.PoissonProblem` (or a :class:`pops.fields.FieldProblem` when
        ``coefficients`` are present, e.g. a screened / anisotropic operator) describing the
        solve ``-laplacian(phi) == rhs`` directly from a :class:`pops.math.Equation`, and
        records it on the model's authoring state so :meth:`inspect` surfaces it.

        Unlike :meth:`solve_field`, this method is INERT: it lowers ONLY to an inspectable
        field-problem descriptor; it does NOT touch the dsl model, the elliptic right-hand
        side, the operator graph, codegen or the runtime. Wiring the problem into the operator
        graph (a second elliptic operator + aux channel) is the deeper lowering and stays
        DEFERRED (see :meth:`solve_field` / the multi-elliptic runtime); this entry point only
        produces the typed descriptor a user can ``validate()`` / ``inspect()`` before any run.

        Args:
            name: the field-problem name (also the unknown's display name when not derivable).
            equation: a :class:`pops.math.Equation` of the form ``-laplacian(phi) == rhs``.
            outputs: the produced fields (passed through to the descriptor's ``outputs``).
            solver: the elliptic solver descriptor (carried; ``None`` leaves it unset so the
                descriptor's own ``available`` / ``validate`` flags the missing solver).
            bcs: an iterable of field boundary-condition descriptors (``pops.fields.bcs``).
            coefficients: an optional operator coefficient; when present the descriptor is a
                general :class:`pops.fields.FieldProblem` rather than a ``PoissonProblem``.
        """
        from pops import fields as _fields  # lazy: keep the module import-graph numpy-free.

        if not isinstance(equation, _bm.Equation):
            raise TypeError(
                "field_problem(%r) expects a pops.math.Equation '-laplacian(phi) == rhs'; got %r"
                % (name, type(equation).__name__))
        unknown = equation.lhs.field if isinstance(equation.lhs, _bm.Laplacian) else name
        cls = _fields.FieldProblem if coefficients is not None else _fields.PoissonProblem
        problem = cls(name=name, unknown=unknown, equation=equation,
                      coefficients=coefficients, bcs=tuple(bcs or ()), outputs=outputs,
                      solver=solver)
        self._field_problems[name] = problem
        return problem

    def inspect(self):
        """A plain-dict, inert view of the model's authoring state (Spec 5 sec.12.1).

        Reports the declared state / field / flux / source / operator names and the inspectable
        field problems authored via :meth:`field_problem` (each as its descriptor's
        :meth:`~pops.fields.FieldProblem.inspect` dict). Read-only: it touches no numerics,
        codegen or runtime.
        """
        return {
            "name": self.name,
            "states": sorted(self._states),
            "fields": sorted(self._fields),
            "fluxes": sorted(self._fluxes),
            "sources": sorted(self._sources),
            "operators": sorted(self._operators),
            "field_problems": {nm: prob.inspect()
                               for nm, prob in self._field_problems.items()},
        }

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
        (``pops.numerics.riemann.hllc.contact_speed.euler()``) or ``None`` keeps the role-derived default.
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
