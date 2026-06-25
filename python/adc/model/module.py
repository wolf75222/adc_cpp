"""The :class:`Module` front-end of the operator-first type system (Spec 2, S2-3).

A Module owns the RULES -- state/field spaces, parameters, aux declarations and a
registry of typed operators a Program composes by signature. The Simulation owns
the DATA. ``Module.to_dsl()`` lowers a pure Module to a :class:`adc.dsl.Model`.

Imports only the standard library (plus the sibling operator-first types) so it
can be exercised without the compiled ``_adc`` extension; ``adc.dsl`` is imported
lazily inside :meth:`Module.to_dsl` to avoid an import cycle.
"""
import hashlib

from .operators import Operator
from .registry import OperatorRegistry
from .signatures import Signature
from .spaces import (
    AuxSpace,
    FieldSpace,
    ParameterSpace,
    RateSpace,
    StateSpace,
)


class Module:
    """A model as typed spaces + a registry of typed operators (Spec 2, operator-first).

    A Module owns the RULES -- state/field spaces, parameters, aux declarations and the
    typed operators a Program composes by signature. The Simulation owns the DATA
    (grid, arrays, solvers, clock). :class:`adc.dsl.Model` is the PDE convenience facade
    that populates a Module's registry (``source_term`` / ``linear_source`` /
    ``elliptic_field`` / ``flux`` register typed operators); a Module can also be built
    directly with ``state_space`` / ``field_space`` / ``parameters`` / ``aux_fields`` /
    ``operator``. A generic Program bound to ``module.operator_registry()`` runs against
    any Module that provides operators with the expected signatures.
    """

    def __init__(self, name):
        self.name = str(name)
        self._state_spaces = {}
        self._field_spaces = {}
        self._params = {}
        self._aux = {}
        self._registry = OperatorRegistry()
        # Wave speeds for the Riemann solver of a compilable Module: {"x": [Expr], "y": [Expr]}
        # eigenvalues, or None (set via eigenvalues()). Carried so a pure Module is self-contained;
        # lowered to dsl.Model.eigenvalues by compile_problem.
        self._eigenvalues = None

    # --- spaces ---
    def state_space(self, name="U", components=(), roles=None, layout="cell",
                    storage="multifab"):
        """Declare and return a :class:`StateSpace`."""
        space = StateSpace(name, components, roles, layout, storage)
        self._state_spaces[space.name] = space
        return space

    def field_space(self, name, components=(), layout="cell"):
        """Declare and return a :class:`FieldSpace`."""
        space = FieldSpace(name, components, layout)
        self._field_spaces[space.name] = space
        return space

    # --- parameters / aux ---
    def param(self, name, default=0.0, dtype="real"):
        """Declare and return one :class:`ParameterSpace`."""
        p = ParameterSpace(name, default, dtype)
        self._params[p.name] = p
        return p

    def parameters(self, **defaults):
        """Declare several parameters by keyword; return ``{name: ParameterSpace}``."""
        return {k: self.param(k, v) for k, v in defaults.items()}

    def aux_field(self, name, kind="cell_scalar"):
        """Declare and return one :class:`AuxSpace`."""
        a = AuxSpace(name, kind)
        self._aux[a.name] = a
        return a

    def aux_fields(self, **kinds):
        """Declare several aux fields by keyword; return ``{name: AuxSpace}``."""
        return {k: self.aux_field(k, v) for k, v in kinds.items()}

    # --- operators ---
    def operator(self, name=None, signature=None, kind=None, capabilities=None,
                 requirements=None, lowering=None, expr=None):
        """Register a typed operator.

        Builder mode (``expr`` given) registers the operator immediately and returns the
        :class:`Operator`. Decorator mode (no ``expr``) returns a decorator that records
        the decorated body as the operator and returns it unchanged::

            @module.operator(name="explicit_rhs",
                             signature=(U, Fields) >> Rate(U), kind="local_rate")
            def explicit_rhs(U, fields):
                ...
        """
        if name is None or signature is None or kind is None:
            raise ValueError("module.operator requires name, signature and kind")
        if not isinstance(signature, Signature):
            raise TypeError(
                "module.operator(%r): signature must be a Signature (use the >> sugar or "
                "Signature(inputs, output)); got %r" % (name, signature))

        def _register(body):
            op = Operator(name, kind, signature, capabilities=capabilities,
                          requirements=requirements, lowering=lowering, source="module",
                          body=body)
            self._registry.register(op)
            return op

        if expr is not None:
            return _register(expr)

        def decorator(func):
            _register(func)
            return func

        return decorator

    def rate_operator(self, name, state_space="U", flux=True, sources=("default",), fluxes=None):
        """Register a composite ``local_rate`` operator ``R = -div F + sum(sources)`` from named
        sub-operators (the flux and the listed source operators). Mirrors ``dsl.rate_operator``; the
        ``lowering`` carries the flux/sources/fluxes so ``P.call`` and the codegen compose it."""
        u = self._state_spaces.get(state_space) or StateSpace(state_space)
        srcs = list(sources) if sources is not None else None
        op = Operator(name, "local_rate", Signature((u,), RateSpace(u)),
                      capabilities={"local": False, "produces_rate": True, "supports_device": True},
                      lowering={"flux": bool(flux), "sources": srcs,
                                "fluxes": list(fluxes) if fluxes else None},
                      source="module")
        self._registry.register(op)
        return op

    def eigenvalues(self, x, y):
        """Declare the per-direction wave speeds (eigenvalues) the Riemann solver needs, as lists of
        IR expressions over the state. Carried so a pure Module is a self-contained, compilable model
        (lowered to ``dsl.Model.eigenvalues``)."""
        self._eigenvalues = {"x": list(x), "y": list(y)}
        return self._eigenvalues

    def adopt_registry(self, registry):
        """Use ``registry`` as this Module's operator registry (the dsl.Model facade adopts
        the derived registry of its HyperbolicModel). Returns ``self``."""
        if not isinstance(registry, OperatorRegistry):
            raise TypeError("adopt_registry expects an OperatorRegistry")
        self._registry = registry
        return self

    def operator_registry(self):
        """The Module's :class:`OperatorRegistry` (bind it to a Program with P.bind_operators)."""
        return self._registry

    def to_dsl(self):
        """Lower this Module to a :class:`adc.dsl.Model` -- the physical/codegen engine -- by mapping
        each typed operator (with its IR body) to the dsl method of its kind. Reuses the dsl backend
        (a translation, not a second codegen). ``adc.compile_problem(model=module, ...)`` does this
        implicitly; call it directly to build the block model for ``sim.add_equation``."""
        from adc import dsl as _dsl  # lazy: dsl imports this module, so import only when compiling
        return _dsl._module_to_model(self)

    # --- introspection (Spec 2, S2-5) ---
    def state_spaces(self):
        return dict(self._state_spaces)

    def field_spaces(self):
        return dict(self._field_spaces)

    def params(self):
        return dict(self._params)

    def aux(self):
        return dict(self._aux)

    def list_state_spaces(self):
        """Names of the declared state spaces."""
        return list(self._state_spaces)

    def list_field_spaces(self):
        """Names of the declared field spaces."""
        return list(self._field_spaces)

    def list_operators(self):
        """Operator names in registration (id) order."""
        return self._registry.names()

    def operator_signature(self, name):
        """The :class:`Signature` of operator ``name``."""
        return self._registry.get(name).signature

    def operator_requirements(self, name):
        """The requirements dict of operator ``name`` (aux / solver / params / ...)."""
        return dict(self._registry.get(name).requirements)

    def operator_capabilities(self, name, **caps):
        """Get or set the capabilities of operator ``name``.

        Called with only a name it is a getter (returns a copy of the dict). Called with
        keyword capabilities (e.g. ``cacheable=True``, ``stale_allowed=True``,
        ``requires_fresh_inputs=True``) it UPDATES them in place and returns the new dict.
        ``cacheable`` is consumed by the Program scheduler to validate a ``hold`` schedule.
        """
        op = self._registry.get(name)
        if caps:
            op.capabilities.update(caps)
        return dict(op.capabilities)

    def module_hash(self):
        """Stable hash of the ModuleSpec for the compiled-artifact cache (Spec 2, S2-7).

        Folds the spaces, parameters, aux declarations and -- for every operator -- the name,
        kind, signature, capabilities, requirements and a body identity (the source of a callable
        body, else its repr). Sensitive to an operator body, signature, capability or space change;
        deterministic for an identical module. A spec2 tag namespaces it away from any spec1 key.
        """
        parts = ["spec2-module", self.name]
        for nm in sorted(self._state_spaces):
            s = self._state_spaces[nm]
            parts.append("state:%s:%s:%s" % (
                s.name, ",".join(s.components), sorted(s.roles.items())))
        for nm in sorted(self._field_spaces):
            f = self._field_spaces[nm]
            parts.append("field:%s:%s" % (f.name, ",".join(f.components)))
        for nm in sorted(self._params):
            p = self._params[nm]
            parts.append("param:%s:%r:%s" % (p.name, p.default, p.dtype))
        for nm in sorted(self._aux):
            a = self._aux[nm]
            parts.append("aux:%s:%s" % (a.name, a.kind))
        if self._eigenvalues is not None:
            for direction in ("x", "y"):
                parts.append("eig_%s:%s" % (
                    direction, ";".join(repr(e) for e in self._eigenvalues[direction])))
        for op in self._registry:  # registration (id) order
            parts.append("op:%s:%s:%s:caps=%s:reqs=%s:body=%s" % (
                op.name, op.kind, repr(op.signature),
                sorted(op.capabilities.items()), sorted(op.requirements.items()),
                _body_identity(op.body)))
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    def __repr__(self):
        return "Module(%r, operators=[%s])" % (self.name, ", ".join(self._registry.names()))


def _body_identity(body):
    """A stable string identifying an operator body for the module hash: the source of a callable
    (so editing it invalidates the cache), else its repr; never raises."""
    if body is None:
        return "none"
    try:
        import inspect
        return inspect.getsource(body)
    except (OSError, TypeError):
        return repr(body)
