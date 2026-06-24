"""Operator-first type system (Spec 2, phase S2-1).

This module defines the abstract spaces and typed operators that a model-free
``adc.time.Program`` composes:

* ``StateSpace`` -- a conservative/primitive state space (the components of ``U``);
* ``FieldSpace`` -- an auxiliary or solved-field space (e.g. ``phi, grad_x, grad_y``);
* ``RateSpace`` / ``Rate(U)`` -- the tangent of a ``StateSpace`` (``dU/dt``);
* ``LocalLinearOperator(U, U)`` / ``MatrixFreeOperator`` -- operator-valued types;
* ``Signature`` -- a typed ``(inputs) -> output`` contract;
* ``Operator`` and ``OperatorRegistry`` -- a named, typed, integer-id'd registry.

These types are a TYPED VIEW: they carry no numerics and no array data. In phase
S2-1 the registry is DERIVED from an existing :class:`adc.dsl` model -- the PDE
shortcuts ``source_term`` / ``linear_source`` / ``elliptic_field`` / ``flux`` lower
into typed operators without changing the public PDE API. The public
``adc.model.Module`` front-end (S2-3), the typed ``P.call`` (S2-2) and the C++
codegen consumption (S2-6) build on these primitives in later phases.

The module imports only the standard library so it can be exercised without the
compiled ``_adc`` extension.
"""
import hashlib

# The ten operator kinds of Spec 2. A kind is metadata only; the Signature carries
# the actual type contract that Program validation checks.
OPERATOR_KINDS = (
    "local_rate",
    "local_source",
    "local_linear_operator",
    "field_operator",
    "grid_operator",
    "projection",
    "diagnostic",
    "matrix_free_operator",
    "local_nonlinear_residual",
    "global_residual",
    "coupled_rate",
)


class Space:
    """Base of a typed space: a kind, a name and an ordered tuple of components.

    Equality and hashing are by ``(kind, name, components, layout)`` so two spaces
    built independently from the same model compare equal (used by Program type
    checks). Subclasses may carry extra metadata (roles, storage) that does not
    participate in identity.
    """

    kind = "space"

    def __init__(self, name, components=(), layout="cell"):
        self.name = str(name)
        self.components = tuple(components)
        self.layout = str(layout)

    def _key(self):
        return (self.kind, self.name, self.components, self.layout)

    def __eq__(self, other):
        return isinstance(other, Space) and self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        return "%s(%r, components=%r)" % (
            type(self).__name__, self.name, list(self.components))

    # Operator-first signature sugar: ``U >> Fields`` and ``(U, Fields) >> Rate(U)``.
    def __rshift__(self, output):
        """``space >> output`` -- a Signature with this space as the sole input."""
        return Signature((self,), output)

    def __rrshift__(self, inputs):
        """``(a, b) >> space`` -- this space is the output, the left tuple the inputs."""
        return Signature(_as_signature_inputs(inputs), self)


def _as_signature_inputs(inputs):
    """Normalize the left side of ``>>`` to a tuple of input types."""
    if isinstance(inputs, (tuple, list)):
        return tuple(inputs)
    return (inputs,)


class StateSpace(Space):
    """A conservative state space: the components of ``U`` plus optional physical
    roles, storage kind and conserved flags. Roles are metadata for diagnostics /
    CFL / projections; a generic Program must not depend on a specific role."""

    kind = "state"

    def __init__(self, name="U", components=(), roles=None, layout="cell",
                 storage="multifab"):
        super().__init__(name, components, layout)
        self.roles = dict(roles) if roles else {}
        self.storage = str(storage)


class FieldSpace(Space):
    """An auxiliary / solved-field space (elliptic field, gradient, divergence,
    magnetic field, derived quantities). Not necessarily produced by Poisson."""

    kind = "field"


class RateSpace(Space):
    """The tangent space of a :class:`StateSpace` -- values of ``dU/dt``.

    Identity is by the BASE state name so ``Rate("U") == Rate(state_space_U)``;
    this lets introspection compare an operator's declared output type against a
    string-named expectation without rebuilding the full state space.
    """

    kind = "rate"

    def __init__(self, base):
        base_name = base.name if isinstance(base, StateSpace) else str(base)
        # Components are intentionally NOT copied from the base: identity is by the
        # base name (encoded in the space name "Rate(<base>)"), so Rate("U") built
        # from a string compares equal to Rate(U) built from the state space.
        super().__init__("Rate(%s)" % base_name, components=(), layout="cell")
        self.base_name = base_name


def Rate(base):  # noqa: N802 (type-constructor sugar, intentionally capitalized)
    """Return the :class:`RateSpace` tangent of ``base`` (a StateSpace or a name)."""
    return RateSpace(base)


class LocalLinearOperator:
    """Operator-valued type ``State -> State`` (an ``L`` such that ``L: U -> U``).

    Identity is by ``(domain_name, range_name)`` so ``LocalLinearOperator("U", "U")``
    compares equal to one built from the actual state spaces. Used to type
    ``linear_source`` operators and to check ``solve_local_linear(I - dt*L, rhs)``.
    """

    def __init__(self, domain, range_):
        self.domain_name = domain.name if isinstance(domain, Space) else str(domain)
        self.range_name = range_.name if isinstance(range_, Space) else str(range_)

    def _key(self):
        return ("local_linear_operator", self.domain_name, self.range_name)

    def __eq__(self, other):
        return (isinstance(other, LocalLinearOperator)
                and self._key() == other._key())

    def __hash__(self):
        return hash(self._key())

    def __rrshift__(self, inputs):
        """``(fields,) >> LocalLinearOperator(U, U)`` signature sugar."""
        return Signature(_as_signature_inputs(inputs), self)

    def __repr__(self):
        return "LocalLinearOperator(%r, %r)" % (self.domain_name, self.range_name)


class MatrixFreeOperator:
    """Operator-valued type ``VectorSpace -> VectorSpace`` usable by a Krylov solve
    (``solve_linear``). Identity is by ``(domain_name, range_name)``."""

    def __init__(self, domain, range_):
        self.domain_name = domain.name if isinstance(domain, Space) else str(domain)
        self.range_name = range_.name if isinstance(range_, Space) else str(range_)

    def _key(self):
        return ("matrix_free_operator", self.domain_name, self.range_name)

    def __eq__(self, other):
        return (isinstance(other, MatrixFreeOperator)
                and self._key() == other._key())

    def __hash__(self):
        return hash(self._key())

    def __rrshift__(self, inputs):
        """``(v,) >> MatrixFreeOperator(V, V)`` signature sugar."""
        return Signature(_as_signature_inputs(inputs), self)

    def __repr__(self):
        return "MatrixFreeOperator(%r, %r)" % (self.domain_name, self.range_name)


class Signature:
    """A typed contract ``(inputs) -> output``.

    ``inputs`` is a tuple of spaces / operator-types; ``output`` is a space or an
    operator-type. Equality is structural so two signatures built from the same
    model compare equal. The ``>>`` operator-first sugar (``(U, Fields) >> Rate(U)``)
    lands with the public ``adc.model.Module`` API (S2-3); here the canonical
    keyword form is used.
    """

    def __init__(self, inputs, output):
        self.inputs = tuple(inputs)
        self.output = output

    def _key(self):
        return (self.inputs, self.output)

    def __eq__(self, other):
        return isinstance(other, Signature) and self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        ins = ", ".join(repr(x) for x in self.inputs)
        return "Signature((%s) -> %r)" % (ins, self.output)


class Operator:
    """A named, typed operator: ``name``, ``kind`` (one of :data:`OPERATOR_KINDS`),
    ``signature``, plus ``capabilities`` and ``requirements`` dicts and a ``source``
    tag naming the API that created it (for debug / introspection). Carries no
    numerics; the body lives in the model / codegen."""

    def __init__(self, name, kind, signature, capabilities=None,
                 requirements=None, source=None, lowering=None, body=None):
        if kind not in OPERATOR_KINDS:
            raise ValueError("operator %r: unknown kind %r (expected one of %s)"
                             % (name, kind, ", ".join(OPERATOR_KINDS)))
        if not isinstance(signature, Signature):
            raise TypeError("operator %r: signature must be a Signature" % (name,))
        self.name = str(name)
        self.kind = str(kind)
        self.signature = signature
        self.capabilities = dict(capabilities) if capabilities else {}
        self.requirements = dict(requirements) if requirements else {}
        self.source = source
        # Codegen hint consumed by the lowering of a typed P.call (e.g. a composite
        # rate operator carries {"flux", "sources", "fluxes"}); empty for primitives.
        self.lowering = dict(lowering) if lowering else {}
        # OPTIONAL body: the callable / expression that builds the operator IR when the
        # operator is declared via Module.operator; None for a derived dsl operator.
        self.body = body

    def __repr__(self):
        return "Operator(%r, kind=%r, %r)" % (
            self.name, self.kind, self.signature)


class OperatorRegistry:
    """An ordered, name-keyed registry of :class:`Operator` with stable integer ids.

    Insertion order fixes the ``OperatorId`` (``id_of`` / ``by_id``) so the C++
    codegen (S2-6) can dispatch by integer in hot kernels while strings stay for
    debug / validation only. Re-registering an existing name raises.
    """

    def __init__(self):
        self._by_name = {}
        self._order = []

    def register(self, operator):
        """Register ``operator`` and return it; its id is its insertion index."""
        if not isinstance(operator, Operator):
            raise TypeError("register expects an Operator, got %r" % (operator,))
        if operator.name in self._by_name:
            raise ValueError("operator %r already registered" % (operator.name,))
        self._by_name[operator.name] = operator
        self._order.append(operator.name)
        return operator

    def get(self, name):
        """Return the operator named ``name`` or raise a clear KeyError."""
        try:
            return self._by_name[name]
        except KeyError:
            known = ", ".join(self._order) or "<none>"
            raise KeyError(
                "unknown operator %r (registered: %s)" % (name, known)) from None

    def names(self):
        """Operator names in registration (id) order."""
        return list(self._order)

    def operators_of_kind(self, kind):
        """Operators of the given kind, in registration order."""
        return [self._by_name[n] for n in self._order if self._by_name[n].kind == kind]

    def default_of_kind(self, kind):
        """The default operator of ``kind`` for model-free resolution.

        Picks the operator flagged ``capabilities["default"]`` if there is exactly
        one; otherwise the sole operator of that kind. Raises a clear error when none
        exists, or when several are compatible and none is privileged -- the caller
        must then disambiguate with an explicit ``P.call(name, ...)``.
        """
        candidates = self.operators_of_kind(kind)
        privileged = [op for op in candidates if op.capabilities.get("default")]
        if len(privileged) == 1:
            return privileged[0]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise KeyError("no %s operator registered" % kind)
        names = ", ".join(op.name for op in candidates)
        raise ValueError(
            "multiple %s operators are compatible (%s); call P.call(name, ...) "
            "explicitly" % (kind, names))

    def id_of(self, name):
        """Integer OperatorId of ``name`` (its registration index)."""
        return self._order.index(name)

    def by_id(self, operator_id):
        """Operator at integer id ``operator_id``."""
        return self._by_name[self._order[operator_id]]

    def __contains__(self, name):
        return name in self._by_name

    def __iter__(self):
        return (self._by_name[n] for n in self._order)

    def __len__(self):
        return len(self._order)

    def __repr__(self):
        return "OperatorRegistry(%s)" % ", ".join(self._order)


class ParameterSpace:
    """A named scalar parameter of a Module: a default value and a dtype. The Module
    holds only the declaration; the runtime value belongs to the Simulation (read in a
    generated kernel via ProgramContext / ModuleContext, never frozen at codegen)."""

    def __init__(self, name, default=0.0, dtype="real"):
        self.name = str(name)
        self.default = default
        self.dtype = str(dtype)

    def __repr__(self):
        return "ParameterSpace(%r, default=%r, dtype=%r)" % (
            self.name, self.default, self.dtype)


class AuxSpace:
    """A named auxiliary field provided or updated by the Simulation (e.g. an externally
    imposed magnetic field, a mask). Distinct from a FieldSpace, which an operator
    produces; an AuxSpace is imposed runtime data the operators may read."""

    def __init__(self, name, kind="cell_scalar"):
        self.name = str(name)
        self.kind = str(kind)

    def __repr__(self):
        return "AuxSpace(%r, kind=%r)" % (self.name, self.kind)


class Module:
    """A model as typed spaces + a registry of typed operators (Spec 2, operator-first).

    A Module owns the RULES -- state/field spaces, parameters, aux declarations and the
    typed operators a Program composes by signature. The Simulation owns the DATA
    (grid, arrays, solvers, clock). :class:`adc.dsl.Model` is the PDE convenience facade
    that populates a Module's registry (``source_term`` / ``linear_source`` /
    ``elliptic_field`` / ``flux`` register typed operators); a Module can also be built
    directly with ``state_space`` / ``field_space`` / ``parameters`` / ``aux_fields`` /
    ``operator``. A generic Program bound to ``module.operator_registry()`` runs against
    any Module that provides operators with the expected signatures.
    """

    def __init__(self, name):
        self.name = str(name)
        self._state_spaces = {}
        self._field_spaces = {}
        self._params = {}
        self._aux = {}
        self._registry = OperatorRegistry()
        # Wave speeds for the Riemann solver of a compilable Module: {"x": [Expr], "y": [Expr]}
        # eigenvalues, or None (set via eigenvalues()). Carried so a pure Module is self-contained;
        # lowered to dsl.Model.eigenvalues by compile_problem.
        self._eigenvalues = None

    # --- spaces ---
    def state_space(self, name="U", components=(), roles=None, layout="cell",
                    storage="multifab"):
        """Declare and return a :class:`StateSpace`."""
        space = StateSpace(name, components, roles, layout, storage)
        self._state_spaces[space.name] = space
        return space

    def field_space(self, name, components=(), layout="cell"):
        """Declare and return a :class:`FieldSpace`."""
        space = FieldSpace(name, components, layout)
        self._field_spaces[space.name] = space
        return space

    # --- parameters / aux ---
    def param(self, name, default=0.0, dtype="real"):
        """Declare and return one :class:`ParameterSpace`."""
        p = ParameterSpace(name, default, dtype)
        self._params[p.name] = p
        return p

    def parameters(self, **defaults):
        """Declare several parameters by keyword; return ``{name: ParameterSpace}``."""
        return {k: self.param(k, v) for k, v in defaults.items()}

    def aux_field(self, name, kind="cell_scalar"):
        """Declare and return one :class:`AuxSpace`."""
        a = AuxSpace(name, kind)
        self._aux[a.name] = a
        return a

    def aux_fields(self, **kinds):
        """Declare several aux fields by keyword; return ``{name: AuxSpace}``."""
        return {k: self.aux_field(k, v) for k, v in kinds.items()}

    # --- operators ---
    def operator(self, name=None, signature=None, kind=None, capabilities=None,
                 requirements=None, lowering=None, expr=None):
        """Register a typed operator.

        Builder mode (``expr`` given) registers the operator immediately and returns the
        :class:`Operator`. Decorator mode (no ``expr``) returns a decorator that records
        the decorated body as the operator and returns it unchanged::

            @module.operator(name="explicit_rhs",
                             signature=(U, Fields) >> Rate(U), kind="local_rate")
            def explicit_rhs(U, fields):
                ...
        """
        if name is None or signature is None or kind is None:
            raise ValueError("module.operator requires name, signature and kind")
        if not isinstance(signature, Signature):
            raise TypeError(
                "module.operator(%r): signature must be a Signature (use the >> sugar or "
                "Signature(inputs, output)); got %r" % (name, signature))

        def _register(body):
            op = Operator(name, kind, signature, capabilities=capabilities,
                          requirements=requirements, lowering=lowering, source="module",
                          body=body)
            self._registry.register(op)
            return op

        if expr is not None:
            return _register(expr)

        def decorator(func):
            _register(func)
            return func

        return decorator

    def rate_operator(self, name, state_space="U", flux=True, sources=("default",), fluxes=None):
        """Register a composite ``local_rate`` operator ``R = -div F + sum(sources)`` from named
        sub-operators (the flux and the listed source operators). Mirrors ``dsl.rate_operator``; the
        ``lowering`` carries the flux/sources/fluxes so ``P.call`` and the codegen compose it."""
        u = self._state_spaces.get(state_space) or StateSpace(state_space)
        srcs = list(sources) if sources is not None else None
        op = Operator(name, "local_rate", Signature((u,), RateSpace(u)),
                      capabilities={"local": False, "produces_rate": True, "supports_device": True},
                      lowering={"flux": bool(flux), "sources": srcs,
                                "fluxes": list(fluxes) if fluxes else None},
                      source="module")
        self._registry.register(op)
        return op

    def eigenvalues(self, x, y):
        """Declare the per-direction wave speeds (eigenvalues) the Riemann solver needs, as lists of
        IR expressions over the state. Carried so a pure Module is a self-contained, compilable model
        (lowered to ``dsl.Model.eigenvalues``)."""
        self._eigenvalues = {"x": list(x), "y": list(y)}
        return self._eigenvalues

    def adopt_registry(self, registry):
        """Use ``registry`` as this Module's operator registry (the dsl.Model facade adopts
        the derived registry of its HyperbolicModel). Returns ``self``."""
        if not isinstance(registry, OperatorRegistry):
            raise TypeError("adopt_registry expects an OperatorRegistry")
        self._registry = registry
        return self

    def operator_registry(self):
        """The Module's :class:`OperatorRegistry` (bind it to a Program with P.bind_operators)."""
        return self._registry

    def to_dsl(self):
        """Lower this Module to a :class:`adc.dsl.Model` -- the physical/codegen engine -- by mapping
        each typed operator (with its IR body) to the dsl method of its kind. Reuses the dsl backend
        (a translation, not a second codegen). ``adc.compile_problem(model=module, ...)`` does this
        implicitly; call it directly to build the block model for ``sim.add_equation``."""
        from . import dsl as _dsl  # lazy: dsl imports this module, so import only when compiling
        return _dsl._module_to_model(self)

    # --- introspection (Spec 2, S2-5) ---
    def state_spaces(self):
        return dict(self._state_spaces)

    def field_spaces(self):
        return dict(self._field_spaces)

    def params(self):
        return dict(self._params)

    def aux(self):
        return dict(self._aux)

    def list_state_spaces(self):
        """Names of the declared state spaces."""
        return list(self._state_spaces)

    def list_field_spaces(self):
        """Names of the declared field spaces."""
        return list(self._field_spaces)

    def list_operators(self):
        """Operator names in registration (id) order."""
        return self._registry.names()

    def operator_signature(self, name):
        """The :class:`Signature` of operator ``name``."""
        return self._registry.get(name).signature

    def operator_requirements(self, name):
        """The requirements dict of operator ``name`` (aux / solver / params / ...)."""
        return dict(self._registry.get(name).requirements)

    def operator_capabilities(self, name, **caps):
        """Get or set the capabilities of operator ``name``.

        Called with only a name it is a getter (returns a copy of the dict). Called with
        keyword capabilities (e.g. ``cacheable=True``, ``stale_allowed=True``,
        ``requires_fresh_inputs=True``) it UPDATES them in place and returns the new dict.
        ``cacheable`` is consumed by the Program scheduler to validate a ``hold`` schedule.
        """
        op = self._registry.get(name)
        if caps:
            op.capabilities.update(caps)
        return dict(op.capabilities)

    def module_hash(self):
        """Stable hash of the ModuleSpec for the compiled-artifact cache (Spec 2, S2-7).

        Folds the spaces, parameters, aux declarations and -- for every operator -- the name,
        kind, signature, capabilities, requirements and a body identity (the source of a callable
        body, else its repr). Sensitive to an operator body, signature, capability or space change;
        deterministic for an identical module. A spec2 tag namespaces it away from any spec1 key.
        """
        parts = ["spec2-module", self.name]
        for nm in sorted(self._state_spaces):
            s = self._state_spaces[nm]
            parts.append("state:%s:%s:%s" % (
                s.name, ",".join(s.components), sorted(s.roles.items())))
        for nm in sorted(self._field_spaces):
            f = self._field_spaces[nm]
            parts.append("field:%s:%s" % (f.name, ",".join(f.components)))
        for nm in sorted(self._params):
            p = self._params[nm]
            parts.append("param:%s:%r:%s" % (p.name, p.default, p.dtype))
        for nm in sorted(self._aux):
            a = self._aux[nm]
            parts.append("aux:%s:%s" % (a.name, a.kind))
        if self._eigenvalues is not None:
            for direction in ("x", "y"):
                parts.append("eig_%s:%s" % (
                    direction, ";".join(repr(e) for e in self._eigenvalues[direction])))
        for op in self._registry:  # registration (id) order
            parts.append("op:%s:%s:%s:caps=%s:reqs=%s:body=%s" % (
                op.name, op.kind, repr(op.signature),
                sorted(op.capabilities.items()), sorted(op.requirements.items()),
                _body_identity(op.body)))
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    def __repr__(self):
        return "Module(%r, operators=[%s])" % (self.name, ", ".join(self._registry.names()))


def _body_identity(body):
    """A stable string identifying an operator body for the module hash: the source of a callable
    (so editing it invalidates the cache), else its repr; never raises."""
    if body is None:
        return "none"
    try:
        import inspect
        return inspect.getsource(body)
    except (OSError, TypeError):
        return repr(body)


def _block_name(key):
    """The block/species name of a RateBundle key: a name string, or a space's name."""
    return key.name if isinstance(key, Space) else str(key)


class RateBundle:
    """A typed multi-output of a coupled operator: a mapping ``block -> Rate(StateSpace)``.

    A coupled rate (``collisions(e, i, n) -> RateBundle``) returns one tangent per
    participating block; ``bundle["electrons"]`` is the :class:`RateSpace` of that
    block. The arity is arbitrary (2, 3, 4, ... species). :meth:`require` enforces
    that a block's rate lives over the expected StateSpace, so adding a
    ``Rate(electron_state)`` where a ``Rate(ion_state)`` is expected is rejected.
    """

    def __init__(self, entries=None):
        self._rates = {}
        for block, rate in (entries or {}).items():
            self.add(block, rate)

    def add(self, block, rate):
        """Bind ``block`` to ``rate`` (a :class:`RateSpace`, a :class:`StateSpace`, or a name)."""
        rs = rate if isinstance(rate, RateSpace) else Rate(rate)
        self._rates[_block_name(block)] = rs
        return self

    def require(self, block, state):
        """Return the block's rate, raising if it is not ``Rate(state)`` (typed multi-output check)."""
        name = _block_name(block)
        got = self._rates.get(name)
        if got is None:
            known = ", ".join(self._rates) or "<none>"
            raise KeyError("RateBundle has no rate for block %r (have: %s)" % (name, known))
        want = Rate(state)
        if got != want:
            raise TypeError(
                "RateBundle[%r] is %r, not %r: a rate must live over its block's StateSpace"
                % (name, got, want))
        return got

    def __getitem__(self, block):
        return self._rates[_block_name(block)]

    def __contains__(self, block):
        return _block_name(block) in self._rates

    def keys(self):
        return list(self._rates)

    def items(self):
        return list(self._rates.items())

    def __len__(self):
        return len(self._rates)

    def _key(self):
        # order-independent identity so a Signature output compares structurally
        return tuple(sorted((k, repr(v)) for k, v in self._rates.items()))

    def __eq__(self, other):
        return isinstance(other, RateBundle) and self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        return "RateBundle({%s})" % ", ".join(
            "%r: %r" % (k, v) for k, v in self._rates.items())
