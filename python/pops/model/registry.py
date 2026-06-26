"""The ordered, name-keyed :class:`OperatorRegistry` (Spec 2).

Insertion order fixes each operator's integer ``OperatorId`` so the C++ codegen
can dispatch by integer in hot kernels while strings stay for debug / validation.
"""
from .operators import Operator


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
