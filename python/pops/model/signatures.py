"""The typed ``(inputs) -> output`` contract of the operator-first type system.

A :class:`Signature` is structural: it holds a tuple of input types (spaces or
operator-types) and one output type. Equality is by ``(inputs, output)`` so two
signatures built from the same model compare equal. Carries no numerics.
"""


class Signature:
    """A typed contract ``(inputs) -> output``.

    ``inputs`` is a tuple of spaces / operator-types; ``output`` is a space or an
    operator-type. Equality is structural so two signatures built from the same
    model compare equal. The ``>>`` operator-first sugar (``(U, Fields) >> Rate(U)``)
    lands with the public ``pops.model.Module`` API (S2-3); here the canonical
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
