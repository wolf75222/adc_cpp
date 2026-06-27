"""Blackboard-style physics model authoring (Spec 3, layer 1).

``pops.physics.Model`` lets a user write a model the way it appears on a
blackboard -- a state, primitives, a flux, an elliptic field solve, sources and
local linear operators, tied together by equations such as
``ddt(U) == -div(F) + S`` and ``-laplacian(phi) == rho`` -- and lowers it to the
Spec 2 operator-first IR (:class:`pops.model.Module`) and the :mod:`pops.dsl`
codegen engine. It is a thin TRANSLATION layer: it owns no numerics and no
codegen of its own. ``pops.dsl.Model`` (the PDE facade) remains valid; the board
API is sugar that produces the same typed operators.

The board notation lives in :mod:`pops.math` (``ddt`` / ``div`` / ``grad`` /
``laplacian`` / ``sqrt`` / ``rate`` / ``unknown`` / ``integral``). The typed view
is reachable through :pyattr:`Model.module`; the codegen model through
:pyattr:`Model.dsl`.

Multi-species board authoring (``m.species`` for N >= 2, ``m.coupled_rate``,
``m.solve_fields_from_species``) LOWERS to the existing operator-first multi-block
IR (an :class:`pops.model.Module` with N :class:`pops.model.StateSpace`, a
``coupled_rate`` operator over a :class:`pops.model.RateBundle`, and a multi-input
field operator), not a second runtime: the board surface produces the SAME typed
operators a hand-written ``pops.model.Module`` registers (ADC-457). The single-species
path is byte-identical to the single-state board model. The compiled multi-block
``.so`` run is validated on ROMEO (Kokkos-only AOT).
"""
import re

from .. import math as _bm

__all__ = ["Invariant", "FluxHandle", "SourceHandle", "FieldsHandle", "FieldHandle",
           "LocalLinearOperatorExpr", "CallableOperator", "StateHandle", "VectorHandle",
           "_safe_name", "_canon_role", "_roles_for", "_BOARD_ROLE"]


def _safe_name(name):
    """A C-identifier-safe operator name derived from a board display name."""
    s = re.sub(r"[^0-9a-zA-Z_]", "_", str(name)).strip("_")
    if not s:
        raise ValueError("operator name %r has no identifier characters" % (name,))
    if s[0].isdigit():
        s = "_" + s
    return s


# Board role vocabulary -> dsl canonical role (pops::VariableRole). The dsl roles_for() uses an
# explicit role override verbatim, so a board role must already be canonical for the native HLLC/Roe
# role lookup (which indexes "Density"/"MomentumX"/"MomentumY"/"Energy") to find it.
_BOARD_ROLE = {
    "density": "Density",
    "momentum_x": "MomentumX", "momentum_y": "MomentumY", "momentum_z": "MomentumZ",
    "energy": "Energy", "pressure": "Pressure", "temperature": "Temperature",
}


def _canon_role(role):
    """Canonicalize a board role string to a dsl role; pass through None and unknown roles."""
    if role is None:
        return None
    return _BOARD_ROLE.get(str(role).lower(), role)


def _roles_for(hyp):
    """The canonical roles of a HyperbolicModel's conservative state."""
    from .aux import roles_for
    return roles_for(hyp.cons_names, hyp.cons_roles)


class StateHandle:
    """A declared state: a name plus the ordered :mod:`pops.dsl` component vars.

    Unpacks into its components (``rho, mx, my = U``), indexes them by position
    (``U[0]``) or by component name (``e["ne"]`` -- the board access of Spec 3
    section 12.3/16), and remembers its name and roles for the typed
    :class:`pops.model.StateSpace`. The string index returns the conservative
    :class:`pops.dsl.Var` of that component, so a board coupled-rate formula
    written as ``e["ni"] - e["ne"]`` is the same IR as the hand-written
    operator-first ``dsl.Var("ni", "cons") - dsl.Var("ne", "cons")``.
    """

    def __init__(self, name, components, vars_, roles, space=None):
        self.name = str(name)
        self.components = tuple(components)
        self.vars = tuple(vars_)
        self.roles = dict(roles or {})
        # The typed pops.model.StateSpace this species instantiates (multi-species
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


class FluxHandle:
    """A declared physical flux (the default hyperbolic flux of a model)."""

    def __init__(self, name, is_default=True):
        self.name = str(name)
        self.is_default = bool(is_default)

    def __repr__(self):
        return "FluxHandle(%r)" % (self.name,)


class SourceHandle(_bm.RateTerm):
    """A declared local source term -- a summand of a rate equation."""

    def __init__(self, display_name, reg_name):
        self.name = str(display_name)
        self.reg_name = str(reg_name)

    def _rate_terms(self):
        return [("source", self, 1.0)]

    def __repr__(self):
        return "SourceHandle(%r)" % (self.name,)


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


class FieldsHandle:
    """The result of a field-solve operator: a named bundle of solved fields."""

    def __init__(self, name, outputs, solver):
        self.name = str(name)
        self.outputs = dict(outputs or {})
        self.solver = solver

    def __repr__(self):
        return "FieldsHandle(%r)" % (self.name,)


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

