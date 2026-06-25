"""Typed spaces of the operator-first type system (Spec 2).

Defines the abstract spaces a model-free ``adc.time.Program`` composes:
``StateSpace`` (the components of ``U``), ``FieldSpace`` (auxiliary / solved
fields), ``RateSpace`` / ``Rate(U)`` (the tangent of a ``StateSpace``), plus the
``ParameterSpace`` and ``AuxSpace`` declarations a Module owns. These carry no
numerics and no array data; they are a TYPED VIEW only.

Imports only the standard library so it can be exercised without the compiled
``_adc`` extension.
"""


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
        from adc.model.signatures import Signature
        return Signature((self,), output)

    def __rrshift__(self, inputs):
        """``(a, b) >> space`` -- this space is the output, the left tuple the inputs."""
        from adc.model.signatures import Signature
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
