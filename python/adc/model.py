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
                 requirements=None, source=None, lowering=None):
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
