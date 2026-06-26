"""FluxHandle: a declared physical flux descriptor."""


class FluxHandle:
    """A declared physical flux (the default hyperbolic flux of a model)."""

    def __init__(self, name, is_default=True):
        self.name = str(name)
        self.is_default = bool(is_default)

    def __repr__(self):
        return "FluxHandle(%r)" % (self.name,)
