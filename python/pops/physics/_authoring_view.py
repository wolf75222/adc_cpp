"""Authoring mixin: operator-first typed view (Spec 2, S2-1).

A DERIVED, typed view of the model as spaces + a registry of typed operators
(:mod:`pops.model`). It carries NO numerics and does NOT touch the model hash or
the codegen. Methods only; the touched attributes are created by
``HyperbolicModel.__init__``. Imports only :mod:`pops.model` and :mod:`pops.ir`
(no codegen, no ``_pops``).
"""
from ._modelpkg import model as _model
from .aux import AUX_CANONICAL, roles_for


class _OperatorViewMixin:
    """Typed StateSpace / FieldSpace / OperatorRegistry view of the model."""

    def _aux_name_set(self):
        """Names that denote an auxiliary field read by a formula (canonical + named)."""
        return set(AUX_CANONICAL) | set(self.aux_extra_names)

    def _aux_requirements(self, exprs):
        """{'aux': [...]} of the aux fields the expressions read, or {} if none."""
        aux_set = self._aux_name_set()
        read = sorted({d for e in exprs for d in (e.deps() & aux_set)})
        return {"aux": read} if read else {}

    def state_space(self, name="U"):
        """Typed :class:`pops.model.StateSpace` view of the conservative state: its
        components and canonical physical roles. Derived; carries no data."""
        role_list = roles_for(self.cons_names, self.cons_roles)
        roles = dict(zip(self.cons_names, role_list, strict=True))
        return _model.StateSpace(name=name, components=tuple(self.cons_names),
                                 roles=roles, layout="cell")

    def field_space(self, name="fields"):
        """Typed :class:`pops.model.FieldSpace` view of the auxiliary surface the model
        reads (canonical aux + named aux fields, in read order, de-duplicated)."""
        comps = []
        for nm in list(self.aux_names) + list(self.aux_extra_names):
            if nm not in comps:
                comps.append(nm)
        return _model.FieldSpace(name=name, components=tuple(comps), layout="cell")

    def operator_registry(self, state_name="U"):
        """Typed :class:`pops.model.OperatorRegistry` derived from this model.

        Lowers the PDE shortcuts into typed operators (ids follow registration order):
        flux -> grid_operator ``(State) -> Rate(State)``; source_term -> local_source
        ``(State[, Fields]) -> Rate(State)``; linear_source -> local_linear_operator
        ``(Fields?) -> LocalLinearOperator(State, State)``; elliptic_field ->
        field_operator ``(State) -> FieldSpace``; projection -> projection
        ``(State) -> State``. The implicit defaults surface as ``flux_default`` /
        ``source_default`` / ``fields_from_state``. Pure view: no hash / codegen impact.
        """
        reg = _model.OperatorRegistry()
        state = self.state_space(state_name)
        fields = self.field_space()
        aux_set = self._aux_name_set()

        def reads_fields(exprs):
            return any(e.deps() & aux_set for e in exprs)

        # Flux divergence (grid_operator: State -> Rate(State)).
        if self._flux:
            reg.register(_model.Operator(
                "flux_default", "grid_operator",
                _model.Signature([state], _model.Rate(state)),
                capabilities={"local": False, "linear": False, "produces_rate": True,
                              "requires_ghosts": 1, "supports_device": True,
                              "default": True},
                source="dsl.flux"))
        for nm in sorted(self._flux_terms):
            reg.register(_model.Operator(
                nm, "grid_operator", _model.Signature([state], _model.Rate(state)),
                capabilities={"local": False, "linear": False, "produces_rate": True,
                              "requires_ghosts": 1, "supports_device": True},
                source="dsl.flux_term"))

        # Local sources (local_source: State[, Fields] -> Rate(State)).
        if self._source is not None:
            rf = reads_fields(self._source)
            reg.register(_model.Operator(
                "source_default", "local_source",
                _model.Signature([state, fields] if rf else [state],
                                 _model.Rate(state)),
                capabilities={"local": True, "linear": False, "requires_fields": rf,
                              "produces_rate": True, "supports_device": True,
                              "default": True},
                requirements=self._aux_requirements(self._source),
                source="dsl.source"))
        for nm in sorted(self._source_terms):
            exprs = self._source_terms[nm]
            rf = reads_fields(exprs)
            reg.register(_model.Operator(
                nm, "local_source",
                _model.Signature([state, fields] if rf else [state],
                                 _model.Rate(state)),
                capabilities={"local": True, "linear": False, "requires_fields": rf,
                              "produces_rate": True, "supports_device": True},
                requirements=self._aux_requirements(exprs),
                source="dsl.source_term"))

        # Local linear operators (local_linear_operator: Fields? -> L(State, State)).
        for nm in sorted(self._linear_sources):
            coeffs = [c for row in self._linear_sources[nm] for c in row]
            rf = reads_fields(coeffs)
            reg.register(_model.Operator(
                nm, "local_linear_operator",
                _model.Signature([fields] if rf else [],
                                 _model.LocalLinearOperator(state, state)),
                capabilities={"local": True, "linear": True, "solve_i_minus_a": True,
                              "matrix_available": True, "supports_device": True},
                requirements=self._aux_requirements(coeffs),
                source="dsl.linear_source"))

        # Field operators (field_operator: State -> FieldSpace).
        if self._elliptic is not None:
            reg.register(_model.Operator(
                "fields_from_state", "field_operator",
                # The Poisson solve PRODUCES the canonical electrostatic triple; an externally
                # imposed aux (e.g. B_z) read by sources is part of field_space() but not produced
                # here, so the produced FieldSpace is the triple, not the full read surface.
                _model.Signature([state], _model.FieldSpace(
                    "fields", components=("phi", "grad_x", "grad_y"))),
                capabilities={"requires_solver": True, "supports_device": True,
                              "default": True},
                requirements={"elliptic_operator": "poisson"},
                source="dsl.elliptic_rhs"))
        for nm in sorted(self._elliptic_fields):
            info = self._elliptic_fields[nm]
            reg.register(_model.Operator(
                nm, "field_operator",
                _model.Signature([state],
                                 _model.FieldSpace(nm, components=tuple(info["aux"]))),
                capabilities={"requires_solver": True, "supports_device": True},
                requirements={"elliptic_operator": info["operator"]},
                source="dsl.elliptic_field"))

        # Pointwise projection (projection: State -> State).
        if self._proj is not None:
            reg.register(_model.Operator(
                "projection", "projection", _model.Signature([state], state),
                capabilities={"local": True, "idempotent": True,
                              "supports_device": True},
                source="dsl.projection"))

        # Composite rate operators (local_rate: State[, Fields] -> Rate(State)); aliases
        # for ctx.rhs(flux=, sources=, fluxes=), carried as a lowering hint for P.call.
        for nm in sorted(self._rate_operators):
            cfg = self._rate_operators[nm]
            src_names = cfg["sources"] if cfg["sources"] is not None else ["default"]
            needs = False
            for s in src_names:
                if s == "default":
                    needs = needs or (self._source is not None
                                      and reads_fields(self._source))
                elif s in self._source_terms:
                    needs = needs or reads_fields(self._source_terms[s])
            reg.register(_model.Operator(
                nm, "local_rate",
                _model.Signature([state, fields] if needs else [state],
                                 _model.Rate(state)),
                capabilities={"local": False, "linear": False, "requires_fields": needs,
                              "produces_rate": True, "supports_device": True},
                lowering={"flux": cfg["flux"], "sources": cfg["sources"],
                          "fluxes": cfg["fluxes"]},
                source="dsl.rate_operator"))
        return reg

