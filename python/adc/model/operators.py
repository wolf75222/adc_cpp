"""Operator-valued types and the named, typed :class:`Operator` (Spec 2).

Defines the ten :data:`OPERATOR_KINDS`, the operator-valued types
``LocalLinearOperator`` / ``MatrixFreeOperator`` (a ``Space -> Space`` map usable
by a local-linear or Krylov solve), and the named :class:`Operator` that pairs a
kind with a :class:`adc.model.signatures.Signature`. Carries no numerics; the body
lives in the model / codegen.
"""
from .signatures import Signature
from .spaces import Space, _as_signature_inputs

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
