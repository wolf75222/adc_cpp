"""Invariant: a typed diagnostic function over a StateSet."""


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
