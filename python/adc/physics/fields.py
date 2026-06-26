"""FieldHandle, VectorHandle, FieldsHandle: solved/auxiliary field descriptors."""


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


class FieldsHandle:
    """The result of a field-solve operator: a named bundle of solved fields."""

    def __init__(self, name, outputs, solver):
        self.name = str(name)
        self.outputs = dict(outputs or {})
        self.solver = solver

    def __repr__(self):
        return "FieldsHandle(%r)" % (self.name,)
