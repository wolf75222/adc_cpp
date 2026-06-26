"""Authoring mixin: runtime parameter collection and hook-form validation.

Collects ``RuntimeParamRef`` nodes across the model formulas, assigns their flat
indices, and emits the generated ``pops::RuntimeParams`` member; also validates
arbitrary Riemann hook formulas. Methods only; touched attributes come from
``HyperbolicModel.__init__``. Codegen-free and ``_pops``-free.
"""
from pops.ir import Expr, _wrap  # noqa: F401  -- _validate_hook_form isinstance checks
from pops.ir.values import RuntimeParamRef  # noqa: F401
from pops.ir.visitors import _children  # noqa: F401

from .aux import _K_MAX_RUNTIME_PARAMS


class _RuntimeParamsMixin:
    """Runtime parameter discovery, index assignment, and hook validation."""

    def _all_exprs(self):
        """All the Expr of the model (primitives, flux, eigenvalues, source, elliptic,
        cons_from). Used to discover the RuntimeParamRef nodes hidden in the tree."""
        out = list(self.prim_defs.values())
        for d in ("x", "y"):
            out += self._flux.get(d, [])
            out += self._eig.get(d, [])
        if self._wave_speeds is not None:  # explicit signed speeds: runtime params included
            for d in ("x", "y"):
                out += list(self._wave_speeds[d])
        if self._ws_jacobian is not None and self._ws_jacobian["rows"] is not None:
            for d in ("x", "y"):  # jacobian entries: runtime params included
                out += [e for row in self._ws_jacobian["rows"][d] for e in row]
        if self._source is not None:
            out += [_wrap(e) for e in self._source]
        if self.cons_from is not None:
            out += list(self.cons_from)
        if self._elliptic is not None:
            out.append(self._elliptic)
        if self._roe_rows is not None:  # Roe rows provided: discover their runtime params (via StateRef)
            out += self._roe_rows["x"] + self._roe_rows["y"]
        return out

    def runtime_param_nodes(self):
        """RuntimeParamRef nodes PRESENT in the formulas, deduplicated by name (the same param may
        appear several times but shares the SAME node object). Order SORTED by name (stable index
        = position in this list, mirror of RuntimeParams on the C++ side)."""
        seen = {}

        def walk(e):
            if isinstance(e, RuntimeParamRef):
                seen.setdefault(e.name, e)
                return
            for c in _children(e):
                walk(c)

        for e in self._all_exprs():
            walk(e)
        return [seen[k] for k in sorted(seen)]

    def assign_runtime_indices(self):
        """Assigns to each RuntimeParamRef its STABLE index (sorted order of names) and returns the
        ordered list of nodes. CALLED before any brick codegen: without this call, to_cpp() would raise
        (index -1). Idempotent (reassigns the same indices). Rejects a model exceeding the C++ bound
        kMaxRuntimeParams (otherwise the fixed-size array would overflow)."""
        nodes = self.runtime_param_nodes()
        if len(nodes) > _K_MAX_RUNTIME_PARAMS:
            raise ValueError(
                "model '%s': %d runtime parameters > kMaxRuntimeParams bound=%d "
                "(include/pops/runtime/runtime_params.hpp); reduce the number of runtime params"
                % (self.name, len(nodes), _K_MAX_RUNTIME_PARAMS))
        for k, node in enumerate(nodes):
            node.index = k
        return nodes

    def _runtime_params_member(self):
        """C++ line declaring the RuntimeParams member of a generated brick, initialized to the
        DECLARATION values (default without a runtime set call). Empty string if the model has no runtime
        param (brick strictly identical to history -> bit-identity of const params preserved)."""
        nodes = self.assign_runtime_indices()
        if not nodes:
            return ""
        vals = ", ".join(repr(node.value) for node in nodes)
        return ("  pops::RuntimeParams params{%d, {%s}};  // params RUNTIME (P7-b) : ecrasables a "
                "l'execution\n" % (len(nodes), vals))

    def has_runtime_params(self):
        """True if at least one formula reads a runtime parameter (kind='runtime')."""
        return bool(self.runtime_param_nodes())

    def _validate_hook_form(self, hook, form, allow_aux=True):
        """Reject an arbitrary-formula Riemann hook (ADC-456) that references a quantity the model
        cannot provide -- the same dependency rule as :meth:`check`, surfaced as a clear capability
        error. @p allow_aux: a single-state hook (e.g. pressure(U)) takes no Aux parameter, so an
        aux dependency is also a missing capability there."""
        known = set(self.cons_names) | set(self.prim_defs)
        if allow_aux:
            known |= set(self.aux_names) | set(self.aux_extra_names)
        missing = sorted(form.deps() - known)
        if missing:
            raise ValueError(
                "riemann hook %r references undeclared quantity %s: the formula needs model "
                "capabilities %s that are not provided (declare them, or use the role-derived "
                "default)" % (hook, missing, missing))

