"""LocalLinearOperatorExpr and CallableOperator: operator math objects and registry callables."""


class LocalLinearOperatorExpr:
    """A LOCAL linear operator object ``L: U -> U`` -- a MATH object, not a callable operator.

    ``m.local_linear_operator(...)`` returns this; it carries the matrix but is NOT yet a
    typed registry operator. Register it with ``m.operator(name, returns=...)`` (or
    ``@module.operator``) to obtain a callable operator. Calling the math object directly
    is an error -- it cannot resolve its field inputs without a registration.
    """

    def __init__(self, display_name, matrix, on=None):
        self.name = str(display_name)
        self.matrix = matrix
        self.on = on

    def __call__(self, *args, **kwargs):
        raise TypeError(
            "local_linear_operator object %r is not a callable operator. Register it with "
            "m.operator(%r, returns=...) or @module.operator(...) first." % (self.name, self.name))

    def __repr__(self):
        return "LocalLinearOperatorExpr(%r)" % (self.name,)


class CallableOperator:
    """A registered, typed operator usable in a time Program: ``op(U, fields, ...)``.

    Returned by ``m.rate`` / ``m.operator``. Calling it with Program values lowers to
    ``P.call(name, ...)`` on the values' Program (binding the model's operator registry on
    first use), so a board-style program can write ``explicit_rate(U_n, fields_n)`` and get
    the same IR as the explicit operator-first ``P.call("explicit_rate", U_n, fields_n)``.
    """

    def __init__(self, name, model):
        self.name = str(name)
        self.reg_name = self.name
        self._model = model     # bound to its FRESH module at call time (sees all operators)

    def __call__(self, *args, name=None):
        prog = next((a.prog for a in args if hasattr(a, "prog")), None)
        if prog is None:
            raise ValueError(
                "operator %r must be called with time-Program values (inside a Program); "
                "got %r" % (self.name, args))
        reg = getattr(prog, "_registry", None)
        # Bind (or rebind) the model's FRESH module if the program has no registry yet or
        # the bound one predates this operator -- so operators registered in any order all
        # resolve, not just those present when the program was first bound.
        if self._model is not None and (reg is None or self.name not in reg):
            prog.bind_operators(self._model.module)
        return prog.call(self.name, *args, name=name)

    def __repr__(self):
        return "CallableOperator(%r)" % (self.name,)
