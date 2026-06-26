"""StateHandle: a declared conservative state with component access."""


class StateHandle:
    """A declared state: a name plus the ordered :mod:`adc.dsl` component vars.

    Unpacks into its components (``rho, mx, my = U``), indexes them by position
    (``U[0]``) or by component name (``e["ne"]`` -- the board access of Spec 3
    section 12.3/16), and remembers its name and roles for the typed
    :class:`adc.model.StateSpace`. The string index returns the conservative
    :class:`adc.dsl.Var` of that component, so a board coupled-rate formula
    written as ``e["ni"] - e["ne"]`` is the same IR as the hand-written
    operator-first ``dsl.Var("ni", "cons") - dsl.Var("ne", "cons")``.
    """

    def __init__(self, name, components, vars_, roles, space=None):
        self.name = str(name)
        self.components = tuple(components)
        self.vars = tuple(vars_)
        self.roles = dict(roles or {})
        # The typed adc.model.StateSpace this species instantiates (multi-species
        # mode); None for the single-state dsl-backed path, where the space is
        # derived on demand from the dsl model.
        self.space = space

    def __iter__(self):
        return iter(self.vars)

    def __len__(self):
        return len(self.vars)

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                return self.vars[self.components.index(key)]
            except ValueError:
                raise KeyError(
                    "state %r has no component %r (have: %s)"
                    % (self.name, key, ", ".join(self.components))) from None
        return self.vars[key]

    def __repr__(self):
        return "StateHandle(%r, %r)" % (self.name, list(self.components))
