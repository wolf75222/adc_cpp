"""System diagnostics mixin (Spec-4 PR-F): runtime block self-check.

``check_model``, the runtime counterpart of dsl.Model.check_model (which checks formulas before
compilation). Mixed into ``System`` via inheritance; operates on ``self._s``.
"""


class _SystemDiagnostics:
    """Runtime block verification method of System."""

    def check_model(self, block, raise_on_error=True, rtol=1e-8, atol=1e-10):
        """Generic RUNTIME verification of an installed block (audit 2026-06, work item 6): check
        on the CURRENT STATE of the block (whatever the backend: native composed, .so JIT/AOT/production):

        - finite state U;
        - finite residual -div F + S (exercises flux + source + reconstruction end to end);
        - positivity of the components with role Density (via variable_roles);
        - positivity of the primitive with role Pressure / named 'p' (via get_primitive_state);
        - round-trip cons -> prim -> cons ~= identity (model conversions consistent;
          the state is SAVED then RESTORED, the block is not modified).

        RUNTIME counterpart of dsl.Model.check_model (which checks the FORMULAS before compilation).
        @return dict {"ok", "failures", "block"}; raise_on_error=True (default) raises ValueError."""
        import numpy as np
        failures = []
        nv = self._s.n_vars(block)
        U = np.asarray(self._s.get_state(block), dtype=float)
        if not np.all(np.isfinite(U)):
            failures.append("state U not finite")
        self._s.solve_fields()  # aux up to date: the residual reads phi / grad phi
        R = np.asarray(self._s.eval_rhs(block), dtype=float)
        if not np.all(np.isfinite(R)):
            failures.append("residual -div F + S not finite (flux/source/reconstruction)")
        ncell = U.size // max(nv, 1)
        Uc = U.reshape(nv, ncell)
        roles = [r.lower() for r in self._s.variable_roles(block, "conservative")]
        names = list(self._s.variable_names(block, "conservative"))
        for i, r in enumerate(roles):
            if r == "density" and not bool(np.all(Uc[i] > 0)):
                failures.append("component '%s' (role Density) not strictly positive" % names[i])
        prim_roles = [r.lower() for r in self._s.variable_roles(block, "primitive")]
        prim_names = list(self._s.variable_names(block, "primitive"))
        try:
            P = np.asarray(self._s.get_primitive_state(block), dtype=float)
            if not np.all(np.isfinite(P)):
                failures.append("primitive state not finite (to_primitive)")
            else:
                for i, (r, nm) in enumerate(zip(prim_roles, prim_names)):
                    if (r == "pressure" or nm == "p") and not bool(np.all(P[i] > 0)):
                        failures.append("primitive '%s' (pressure) not strictly positive" % nm)
                # round-trip cons -> prim -> cons: state saved then restored (no net mutation).
                U0 = U.copy()
                self._s.set_primitive_state(block, P)
                U1 = np.asarray(self._s.get_state(block), dtype=float)
                self._s.set_state(block, U0)
                if not np.allclose(U1, U0, rtol=rtol, atol=atol):
                    err = float(np.max(np.abs(U1 - U0)))
                    failures.append("round-trip to_conservative(to_primitive(U)) != U "
                                    "(max gap %g: inconsistent model conversions)" % err)
        except RuntimeError as ex:  # block without conversions (earlier .so paths): report it
            failures.append("cons<->prim conversions unavailable on this block (%s)" % ex)
        report = {"ok": not failures, "failures": failures, "block": block}
        if failures and raise_on_error:
            raise ValueError("System.check_model('%s'): %d failure(s):\n  - %s"
                             % (block, len(failures), "\n  - ".join(failures)))
        return report
