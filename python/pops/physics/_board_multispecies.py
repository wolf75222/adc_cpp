"""Board-facade mixin: multi-species lowering and inspection.

Splits the multi-species authoring (``species`` promotion, ``coupled_rate``,
``solve_fields_from_species``) and the inspection/dump helpers out of the board
:class:`pops.physics.board.Model` so neither file exceeds the Spec-4 500-line
bound. Methods only; they operate on the board ``Model`` instance attributes
(``_multi_module`` / ``_species`` / ``_states`` / ``_dsl`` / ...). Lowers to the
operator-first multi-block IR (:mod:`pops.model`); codegen-free, ``_pops``-free.
"""
from .board_handles import (CallableOperator, FieldsHandle, StateHandle,
                            _canon_role, _safe_name)


class _MultiSpeciesMixin:
    """Multi-species promotion, coupled_rate, field-solve, and inspection dumps."""

    def _promote_to_multispecies(self):
        """Build the multi-block :class:`pops.model.Module` and migrate the first species into it.

        The single-state dsl model authored the first species; multi-species mode realizes every
        species as a typed StateSpace on a shared Module so N >= 2 species lower to the existing
        operator-first multi-block IR (N spaces + coupled_rate + multi-block field solve), not a
        second runtime. The first species' :class:`StateHandle` is updated IN PLACE (its ``.space``
        is set) so the reference the caller already holds stays valid after promotion."""
        if self._multi_module is not None:
            return
        from .. import model as _model
        self._multi_module = _model.Module(self.name)
        for nm, h in self._species.items():
            self._add_species(nm, components=h.components, roles=h.roles, handle=h)

    def _add_species(self, name, components=(), roles=None, handle=None):
        """Add one typed StateSpace to the multi-block Module and return its StateHandle.

        ``handle`` updates an existing :class:`StateHandle` in place (promotion of the first
        species); otherwise a fresh handle is created and recorded."""
        from ..ir.expr import Var
        comps = tuple(components)
        canon = {c: _canon_role(roles.get(c)) for c in comps} if roles else {}
        space = self._multi_module.state_space(str(name), comps, roles=canon)
        vars_ = tuple(Var(c, "cons") for c in comps)
        if handle is None:
            handle = StateHandle(name, comps, vars_, roles, space=space)
        else:
            handle.vars = vars_
            handle.space = space
        self._species[handle.name] = handle
        self._states[handle.name] = handle
        return handle

    # --- quantities ---

    def coupled_rate(self, name, inputs=(), outputs=None, preserves=None, dissipates=None):
        """Declare a coupled rate over several species (collisions, ionization, radiation).

        ``inputs`` is the ordered list of participating species (:class:`StateHandle`); a species
        may appear as a READ-ONLY catalyst input without being an output block. ``outputs`` maps
        each output species to its per-component rate formulas (one expression per cons component,
        written over the input species' cons vars via ``e["ne"]``). Arbitrary arity: 2, 3, 4, ...
        inputs, no two-input limit.

        Lowers to the existing operator-first ``coupled_rate`` operator (the SAME kind #287/#300
        lower): a :class:`pops.model.RateBundle` signature over the input :class:`StateSpace` set,
        with the per-block component formulas as the operator body. ``preserves`` / ``dissipates``
        are recorded as capabilities (a generic invariant tag), not numerics. Requires multi-species
        mode (declare the species with :meth:`species`).
        """
        from .. import model as _model
        if self._multi_module is None:
            raise ValueError(
                "coupled_rate(%r) needs at least two species; declare them with m.species(...)"
                % (name,))
        in_handles = self._as_species_list("coupled_rate", name, inputs)
        if not outputs:
            raise ValueError("coupled_rate(%r) requires outputs={species: [per-component exprs]}"
                             % (name,))
        in_spaces = tuple(h.space for h in in_handles)
        bundle = _model.RateBundle()
        expr = {}
        for sp, comps in outputs.items():
            h = self._species_handle("coupled_rate", name, sp)
            comp_list = [self._to_expr(c) for c in self._as_iter(comps)]
            if len(comp_list) != len(h.components):
                raise ValueError(
                    "coupled_rate(%r) output %r has %d component formula(s) but its state %r has %d"
                    % (name, h.name, len(comp_list), h.name, len(h.components)))
            bundle.add(h.name, h.space)
            expr[h.name] = comp_list
        caps = {}
        if preserves is not None:
            caps["preserves"] = preserves
        if dissipates is not None:
            caps["dissipates"] = dissipates
        reg = _safe_name(name)
        self._multi_module.operator(name=reg, kind="coupled_rate",
                                    signature=_model.Signature(in_spaces, bundle),
                                    capabilities=caps or None, expr=expr)
        return CallableOperator(reg, self)

    def solve_fields_from_species(self, name, inputs=(), equation=None, outputs=None, solver=None):
        """Declare a coupled field solve over several species (multi-block Poisson).

        ``inputs`` is the ordered list of contributing species; the field RHS reads every listed
        species' stage state at once. Lowers to a typed ``field_operator`` over the N input
        :class:`StateSpace` set, the operator-first surface of ``P.solve_fields_from_blocks``
        (the existing multi-block field solve, Spec 3 criterion 24). ``equation`` /  ``outputs`` /
        ``solver`` record the elliptic problem and the produced fields for introspection.
        """
        from .. import model as _model
        if self._multi_module is None:
            raise ValueError(
                "solve_fields_from_species(%r) needs at least two species; declare them with "
                "m.species(...)" % (name,))
        in_handles = self._as_species_list("solve_fields_from_species", name, inputs)
        in_spaces = tuple(h.space for h in in_handles)
        out_comps = tuple(outputs.keys()) if outputs else ("phi",)
        fields = self._multi_module.field_space(_safe_name(name), out_comps)
        reg = _safe_name(name)
        reqs = {"solver": solver} if solver is not None else None
        self._multi_module.operator(name=reg, kind="field_operator",
                                    signature=_model.Signature(in_spaces, fields),
                                    requirements=reqs,
                                    expr={"blocks": [h.name for h in in_handles]})
        h = FieldsHandle(name, outputs, solver)
        self._fields[name] = h
        if solver is not None:
            self._field_solvers[name] = solver
        return h

    def _as_species_list(self, op, name, items):
        """Resolve a list of species handles / names to StateHandles (multi-species mode)."""
        if not items:
            raise ValueError("%s(%r) requires inputs=[species, ...]" % (op, name))
        return [self._species_handle(op, name, s) for s in self._as_iter(items)]

    def _species_handle(self, op, name, sp):
        """Resolve one species (a StateHandle or a species name) to its StateHandle."""
        if isinstance(sp, StateHandle):
            handle = self._species.get(sp.name)
        else:
            handle = self._species.get(str(sp))
        if handle is None:
            known = ", ".join(self._species) or "<none>"
            raise KeyError("%s(%r): unknown species %r (declared: %s)"
                           % (op, name, sp, known))
        return handle

    @staticmethod
    def _as_iter(x):
        """A list view of a single item or an iterable (so inputs=e and inputs=[e, i] both work)."""
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]


    def list_operators(self):
        if self._multi_module is not None:
            return self._multi_module.list_operators()
        return self._dsl.list_operators()

    def operator_alias(self, name):
        """The registered operator name for a board role name (``operator(...)``)."""
        return self._aliases.get(name, name)

    # --- inspection / debug (Spec 3 section 33): show the lowering ---
    def dump_physics(self):
        """A board-level view of what was declared (states, params, fields, fluxes,
        sources, operators) -- the layer-1 surface."""
        lines = ["# physics.Model %s" % self.name]
        lines.append("states: %s" % {n: list(h.components) for n, h in self._states.items()})
        lines.append("params: %s" % list(self._dsl.params))
        lines.append("fields: %s" % list(self._fields))
        lines.append("fluxes: %s" % list(self._fluxes))
        lines.append("sources: %s" % list(self._sources))
        lines.append("invariants: %s" % list(self._invariants))
        lines.append("operators: %s" % self.list_operators())
        return "\n".join(lines)

    def dump_module_ir(self):
        """The operator-first :class:`pops.model.Module` this model lowers to: the typed
        spaces and operators with signatures (layer 2)."""
        mod = self.module
        reg = mod.operator_registry()
        lines = ["# pops.model.Module %s" % mod.name]
        for n, s in mod.state_spaces().items():
            lines.append("StateSpace %s: %s" % (n, list(s.components)))
        for n, f in mod.field_spaces().items():
            lines.append("FieldSpace %s: %s" % (n, list(f.components)))
        for op in mod.list_operators():
            lines.append("Operator %s [%s]: %r" % (op, reg.get(op).kind, mod.operator_signature(op)))
        return "\n".join(lines)

    def dump_capabilities(self):
        """The requirements / capabilities declared by each typed operator."""
        mod = self.module
        lines = ["# capabilities / requirements of %s" % mod.name]
        for op in mod.list_operators():
            lines.append("%s: caps=%s reqs=%s"
                         % (op, mod.operator_capabilities(op), mod.operator_requirements(op)))
        return "\n".join(lines)
