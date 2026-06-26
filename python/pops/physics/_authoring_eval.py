"""Authoring mixin: numpy CPU interpreter and model self-check.

Host-side evaluators (``flux`` / ``max_wave_speed`` / ``wave_speeds_value`` /
``source_value`` / ``projection_value``-adjacent helpers) plus ``check`` /
``check_model``. Methods only; all touched attributes are created by
``HyperbolicModel.__init__``. Imports numpy, :mod:`pops.ir` and the aux/role
helpers; codegen-free and ``_pops``-free.
"""
import numpy as np

from pops.ir.visitors import _expr_uses_cons_or_prim  # noqa: F401

from .aux import roles_for


class _EvalMixin:
    """numpy evaluators and the numerical model self-check."""

    def _env(self, U, aux):
        """Environment: cons (from U), aux (provided), then derived primitives (insertion
        order = dependency order)."""
        env = {self.cons_names[i]: U[i] for i in range(len(self.cons_names))}
        if aux:
            env.update(aux)
        for pname, pexpr in self.prim_defs.items():
            env[pname] = pexpr.eval(env)
        return env

    def flux(self, U, aux, dir):
        """Physical flux in direction dir (0=x, 1=y). U: numpy (n_vars, ...)."""
        env = self._env(U, aux)
        comps = self._flux["x" if dir == 0 else "y"]
        return np.stack([np.broadcast_to(c.eval(env), U[0].shape) for c in comps], axis=0)

    def max_wave_speed(self, U, aux, dir):
        """max_k max_cells ``|lambda_k|``: Rusanov / CFL bound. Source: eigenvalues
        (legacy); WITHOUT set_eigenvalues, ``max(|smin|, |smax|)`` of the explicit signed speeds
        (set_wave_speeds), an exact mirror of the C++ emission."""
        env = self._env(U, aux)
        key = "x" if dir == 0 else "y"
        if not self._eig.get(key) and self._ws_jacobian is not None:
            lo, hi = self._ws_jacobian_value(U, env, key)
            return max(float(np.max(np.abs(lo))), float(np.max(np.abs(hi))))
        exprs = self._eig.get(key) or (list(self._wave_speeds[key])
                                       if self._wave_speeds is not None else None)
        if not exprs:
            raise ValueError("max_wave_speed: neither set_eigenvalues(...) nor set_wave_speeds(...) nor "
                             "set_wave_speeds_from_jacobian(...) declared on model '%s'"
                             % self.name)
        return max(float(np.max(np.abs(np.asarray(e.eval(env))))) for e in exprs)

    def _ws_jacobian_value(self, U, env, key):
        """Numpy evaluator of the jacobian path: extremes of the real parts of the eigenvalues
        of the sub-blocks, per sample (mirror of the emitted wave_speeds; np.linalg.eigvals)."""
        ws = self._ws_jacobian
        nv = self.n_vars
        nsmp = int(np.asarray(U[0]).reshape(-1).shape[0])
        if ws["eig"] == "fd":
            base = np.stack([np.broadcast_to(np.asarray(c.eval(env), dtype=float), (nsmp,))
                             if hasattr(c, "eval") else np.full((nsmp,), float(c))
                             for c in (self._flux[key])], axis=0)
            J = np.empty((nsmp, nv, nv))
            Uflat = np.stack([np.broadcast_to(np.asarray(env[c], dtype=float), (nsmp,))
                              for c in self.cons_names], axis=0)
            for k in range(nv):
                eps = 1e-6 * np.abs(Uflat[0]) + 1e-30
                Up = Uflat.copy()
                Up[k] += eps
                envp = self._env(Up, {n: env[n] for n in self.aux_names} if self.aux_names else None)
                Fp = np.stack([np.broadcast_to(np.asarray(c.eval(envp), dtype=float), (nsmp,))
                               for c in self._flux[key]], axis=0)
                J[:, :, k] = ((Fp - base) / eps).T
        else:
            rows = ws["rows"][key]
            J = np.empty((nsmp, nv, nv))
            for i in range(nv):
                for j in range(nv):
                    J[:, i, j] = np.broadcast_to(
                        np.asarray(rows[i][j].eval(env), dtype=float), (nsmp,))
        lo = np.full((nsmp,), np.inf)
        hi = np.full((nsmp,), -np.inf)
        for b in ws["blocks"][key]:
            idx = np.asarray(b)
            lam = np.linalg.eigvals(J[:, idx[:, None], idx[None, :]])
            lo = np.minimum(lo, lam.real.min(axis=1))
            hi = np.maximum(hi, lam.real.max(axis=1))
        return lo, hi

    def _flux_jacobian_spectral_radius(self, U, aux, dir):
        """Spectral radius max_cells max_k |Re(lambda_k)| of the FULL dense Jacobian A = dF_dir/dU,
        evaluated by CENTRAL finite differences on the interpreted flux. Independent of any declared
        partition (set_wave_speeds_from_jacobian blocks=...): serves as a non-circular reference bound
        against max_wave_speed. Returns None if a perturbed state leaves the domain (non-finite flux)
        -- in which case nothing can be concluded."""
        nv = self.n_vars
        U = np.asarray(U, dtype=float)
        nsmp = U.shape[1]
        J = np.empty((nsmp, nv, nv))
        for j in range(nv):
            eps = 1e-6 * np.abs(U[j]) + 1e-7
            Up = U.copy()
            Up[j] = Up[j] + eps
            Um = U.copy()
            Um[j] = Um[j] - eps
            Fp = self.flux(Up, aux, dir)
            Fm = self.flux(Um, aux, dir)
            if not (bool(np.all(np.isfinite(Fp))) and bool(np.all(np.isfinite(Fm)))):
                return None
            for i in range(nv):
                J[:, i, j] = (np.broadcast_to(Fp[i], (nsmp,))
                              - np.broadcast_to(Fm[i], (nsmp,))) / (2.0 * eps)
        lam = np.linalg.eigvals(J)
        return float(np.max(np.abs(lam.real)))

    def wave_speeds_value(self, U, aux, dir):
        """Numpy evaluator of the signed speeds (smin, smax) -- mirror of the emitted wave_speeds:
        explicit pair (set_wave_speeds) if declared, otherwise min/max of the eigenvalues (legacy
        path, which requires 'p' to be EMITTED but remains evaluable here)."""
        env = self._env(U, aux)
        key = "x" if dir == 0 else "y"
        if self._wave_speeds is not None:
            lo, hi = self._wave_speeds[key]
            return (np.asarray(lo.eval(env), dtype=float),
                    np.asarray(hi.eval(env), dtype=float))
        if self._ws_jacobian is not None:
            return self._ws_jacobian_value(U, env, key)
        eigs = [np.asarray(e.eval(env), dtype=float) for e in self._eig.get(key, [])]
        if not eigs:
            raise ValueError("wave_speeds_value: neither set_wave_speeds(...) nor set_eigenvalues(...) "
                             "declared on model '%s'" % self.name)
        eigs = list(np.broadcast_arrays(*eigs)) if len(eigs) > 1 else eigs  # mixed shapes (constant lambda)
        return (np.min(np.stack(eigs), axis=0), np.max(np.stack(eigs), axis=0))

    def source_value(self, U, aux):
        """Source term (numpy (n_vars, ...)), or zeros if not defined. A model that declares only
        NAMED sources (no m.source default) cannot answer the legacy total-source query: the named
        terms are never summed implicitly, so an old stepper asking for the total source is rejected
        (use pops.compile_problem(...) with a time Program, or define m.source(...) explicitly)."""
        if self._source is None:
            if self._source_terms:
                raise ValueError("model has multiple named sources; use pops.compile_problem(...) "
                                 "or define m.source(...) explicitly")
            return np.zeros_like(U)
        env = self._env(U, aux)
        return np.stack([np.broadcast_to(s.eval(env), U[0].shape) for s in self._source], axis=0)

    def to_python_flux(self, aux=None):
        """Produces an pops.PythonFlux (host backend) from the formulas: the model RUNS
        (interpreted on CPU). aux: dict name -> array (auxiliary fields), frozen for this flux."""
        import pops
        a = aux or {}
        return pops.PythonFlux(
            lambda U, d: self.flux(U, a, d),
            lambda U: max(self.max_wave_speed(U, a, 0), self.max_wave_speed(U, a, 1)))

    def check(self):
        """Checks that every referenced variable (primitives, flux, eigenvalues, source) is
        properly declared (cons / prim / aux). Raises ValueError otherwise (dependency check)."""
        known = (set(self.cons_names) | set(self.prim_defs) | set(self.aux_names)
                 | set(self.aux_extra_names))  # named aux fields (aux_field): ADC-70
        used = set()
        groups = [self._flux.get("x", []), self._flux.get("y", []),
                  self._eig.get("x", []), self._eig.get("y", []), self._source or [],
                  [e for e in (self._stab_speed, self._stab_dt, self._src_freq)
                   if e is not None],
                  self._proj or [],  # projection ponctuelle post-pas (ADC-177)
                  [e for row in (self._src_jac or []) for e in row]]
        if self._wave_speeds is not None:
            groups.append(list(self._wave_speeds["x"]) + list(self._wave_speeds["y"]))
        if self._ws_jacobian is not None and self._ws_jacobian["rows"] is not None:
            for d in ("x", "y"):
                groups.append([e for row in self._ws_jacobian["rows"][d] for e in row])
        if self._roe_rows is not None:
            groups.append(self._roe_rows["x"])
            groups.append(self._roe_rows["y"])
        for exprs in self._source_terms.values():  # NAMED sources (source_term)
            groups.append(exprs)
        for mat in self._linear_sources.values():  # NAMED linear operators (linear_source)
            groups.append([e for row in mat for e in row])
        for flx in self._flux_terms.values():  # NAMED fluxes (flux_term, ADC-419)
            groups.append(list(flx["x"]) + list(flx["y"]))
        for e in self.prim_defs.values():
            used |= e.deps()
        for grp in groups:
            for e in grp:
                used |= e.deps()
        if self._elliptic is not None:
            used |= self._elliptic.deps()
        for fld in self._elliptic_fields.values():  # NAMED elliptic fields (elliptic_field, ADC-419)
            used |= fld["rhs"].deps()
        missing = used - known
        if missing:
            raise ValueError("model '%s': undefined variables %s" % (self.name, sorted(missing)))
        # source_frequency is a property of the SOURCE (emitted on the generated source brick):
        # declaring it without a source would be SILENTLY lost -> explicit error.
        if self._src_freq is not None and self._source is None:
            raise ValueError("model '%s': source_frequency(...) declared without a source "
                             "(call m.source([...]) -- the frequency is emitted on the generated "
                             "source brick)" % self.name)
        if self._src_jac is not None and self._source is None:
            raise ValueError("model '%s': source_jacobian(...) declared without a source "
                             "(call m.source([...]) -- the Jacobian is emitted on the generated "
                             "source brick)" % self.name)
        # roe_dissipation and enable_roe are two providers of the SAME hook: exclusive (defensive;
        # already rejected at declaration). Structural re-check of the rows (left/right) along the way.
        if self._roe_rows is not None:
            from pops.codegen.cpp_writer import _roe_validate  # lazy: keep physics codegen-free at import
            if self._roe:
                raise ValueError("model '%s': enable_roe() and roe_dissipation(...) declared "
                                 "together -- a single provider of the roe_dissipation hook" % self.name)
            for key in ("x", "y"):
                for e in self._roe_rows[key]:
                    _roe_validate(e, False)
        # linear_source coefficients stay linear in U (defensive re-check: also caught at declaration).
        for nm, mat in self._linear_sources.items():
            for row in mat:
                for coeff in row:
                    if _expr_uses_cons_or_prim(coeff):
                        raise ValueError("linear_source '%s' coefficients must not depend on "
                                         "conservative or primitive variables" % nm)
        return True

    def check_model(self, samples=None, n_samples=64, seed=0, aux=None, rtol=1e-8, atol=1e-10,
                    raise_on_error=True, jac_rtol=1e-3, jac_atol=1e-9):
        """Generic NUMERICAL verification of the symbolic model (audit 2026-06, work item 6):
        evaluates the formulas on sample states and checks, when the piece exists:

        - finite flux (both directions);
        - finite source;
        - finite elliptic_rhs;
        - finite and real eigenvalues; finite max_wave_speed and >= 0;
        - consistency wave_speeds <-> max_wave_speed: ``max(|lambda_min|, |lambda_max|) <= mws``;
        - NON-CIRCULAR bounding of the spectrum: the spectral radius of the full dense flux Jacobian
          (central finite differences, independent of any blocks= partition) does not exceed
          max_wave_speed -- catches a set_wave_speeds_from_jacobian partition that does NOT bound the
          eigenvalues (mws underestimated, unsafe CFL) where the consistency above, derived from the
          SAME partition, still holds;
        - round-trip to_conservative(to_primitive(U)) ~= U (if prim_state + cons_from declared);
        - positivity of the Density-role components (and of the primitive 'p' if declared) on the
          samples (which are generated positive for these roles).

        @p jac_rtol, @p jac_atol: tolerances of the spectral bounding (radius <= mws*(1+jac_rtol)
        + jac_atol); relaxed to absorb the noise of the finite-difference Jacobian.

        @p samples: array (n_vars, N) of conservative states to test; None -> N = n_samples random
        states (fixed seed, reproducible): Density-role components in [0.1, 2], the others
        in [-1, 1]; an Energy-role component gets ``1 + |kinetic|`` to stay physical.
        @p aux: dict name -> value(s) of the auxiliary fields (default: zeros).
        @return dict {"ok": bool, "failures": [str], "n_samples": N}. raise_on_error=True (default)
        raises ValueError listing the failures. PRE-COMPILATION: checks the FORMULAS (the compiled .so
        emits exactly these formulas); the RUNTIME counterpart on an installed block is
        System.check_model(block)."""
        self.check()  # declared dependencies (raises if a variable does not exist)
        rng = np.random.default_rng(seed)
        nv = self.n_vars
        roles = roles_for(self.cons_names, self.cons_roles)
        if samples is None:
            U = rng.uniform(-1.0, 1.0, size=(nv, int(n_samples)))
            kinetic = np.zeros(int(n_samples))
            for i, r in enumerate(roles):
                if r == "Density":
                    U[i] = rng.uniform(0.1, 2.0, size=int(n_samples))
            for i, r in enumerate(roles):
                if r in ("MomentumX", "MomentumY"):
                    kinetic += U[i] ** 2
            for i, r in enumerate(roles):
                if r == "Energy":
                    U[i] = 1.0 + kinetic  # above the kinetic: pressure > 0 for an ideal gas
        else:
            U = np.asarray(samples, dtype=float)
            if U.ndim != 2 or U.shape[0] != nv:
                raise ValueError("check_model: samples must be (n_vars=%d, N)" % nv)
        a = {n: np.zeros(U.shape[1]) for n in (self.aux_names + self.aux_extra_names)}
        if aux:
            for k, v in aux.items():
                a[k] = np.broadcast_to(np.asarray(v, dtype=float), (U.shape[1],)).copy()
        failures = []

        def finite(x):
            return bool(np.all(np.isfinite(np.asarray(x, dtype=float))))

        for d, dn in ((0, "x"), (1, "y")):
            if not finite(self.flux(U, a, d)):
                failures.append("flux %s non-finite on the samples" % dn)
        if self._source is not None and not finite(self.source_value(U, a)):
            failures.append("source non-finite on the samples")
        if self._elliptic is not None:
            env = self._env(U, a)
            if not finite(self._elliptic.eval(env)):
                failures.append("elliptic_rhs non-finite on the samples")
        env = self._env(U, a)
        for d in ("x", "y"):
            for k, e in enumerate(self._eig.get(d, [])):
                lam = np.asarray(e.eval(env), dtype=float)
                if np.iscomplexobj(lam):
                    failures.append("eigenvalue %s[%d] complex (non-hyperbolic system?)" % (d, k))
                elif not finite(lam):
                    failures.append("eigenvalue %s[%d] non-finite" % (d, k))
        if self._wave_speeds is not None:
            for d in ("x", "y"):
                lo = np.asarray(self._wave_speeds[d][0].eval(env), dtype=float)
                hi = np.asarray(self._wave_speeds[d][1].eval(env), dtype=float)
                if not (finite(lo) and finite(hi)):
                    failures.append("wave_speeds %s (explicit) non-finite" % d)
                elif bool(np.any(lo > hi)):
                    failures.append("wave_speeds %s (explicit): smin > smax on some samples" % d)
        for d, dn in ((0, "x"), (1, "y")):
            mws = self.max_wave_speed(U, a, d)
            if not np.isfinite(mws) or mws < 0:
                failures.append("max_wave_speed %s non-finite or negative (%r)" % (dn, mws))
            else:
                # consistency wave_speeds <-> max_wave_speed: the SIGNED extremes actually emitted
                # (explicit pair if declared, otherwise eigenvalues) must be covered by the
                # Rusanov / CFL bound.
                lo, hi = self.wave_speeds_value(U, a, d)
                ext = max(float(np.max(np.abs(lo))), float(np.max(np.abs(hi))))
                if ext > mws * (1.0 + rtol) + atol:
                    failures.append("wave_speeds %s inconsistent with max_wave_speed (%g > %g)"
                                    % (dn, ext, mws))
                # NON-CIRCULAR bounding: the spectral radius of the dense flux Jacobian (central
                # FD, no partition) must be bounded by max_wave_speed. A blocks= partition that is
                # not really block-triangular yields sub-block extremes that do NOT bound the
                # spectrum -> mws too small, detected here.
                if self._flux:
                    radius = self._flux_jacobian_spectral_radius(U, a, d)
                    if radius is not None and radius > mws * (1.0 + jac_rtol) + jac_atol:
                        failures.append(
                            "partition %s: max_wave_speed (%g) does not bound the spectrum of the "
                            "flux Jacobian (spectral radius %g) -- the blocks= partition of "
                            "set_wave_speeds_from_jacobian does not bound the eigenvalues, the "
                            "Rusanov/CFL bound is underestimated" % (dn, mws, radius))
        # round-trip cons -> prim -> cons (when both directions are declared)
        if self.prim_state and self.cons_from is not None:
            penv = {nm: np.broadcast_to(np.asarray(env[nm], dtype=float), (U.shape[1],))
                    for nm in self.prim_state}
            U2 = np.stack([np.broadcast_to(np.asarray(e.eval(penv), dtype=float), (U.shape[1],))
                           for e in self.cons_from], axis=0)
            if not finite(U2):
                failures.append("to_conservative(to_primitive(U)) non-finite")
            elif not np.allclose(U2, U, rtol=rtol, atol=atol):
                err = float(np.max(np.abs(U2 - U)))
                failures.append("round-trip to_conservative(to_primitive(U)) != U (max deviation %g: "
                                "inconsistent conversions)" % err)
        # positivity: Density roles (conservative) and primitive 'p' (pressure) if declared
        for i, r in enumerate(roles):
            if r == "Density" and not bool(np.all(U[i] > 0)):
                failures.append("component '%s' (Density role) not strictly positive on the "
                                "samples" % self.cons_names[i])
        if "p" in self.prim_defs:
            p = np.asarray(env["p"], dtype=float)
            if not finite(p):
                failures.append("primitive 'p' (pressure) non-finite")
            elif not bool(np.all(p > 0)):
                failures.append("primitive 'p' (pressure) not strictly positive on physical "
                                "states (suspicious EOS)")
        report = {"ok": not failures, "failures": failures, "n_samples": int(U.shape[1])}
        if failures and raise_on_error:
            raise ValueError("check_model('%s'): %d failure(s):\n  - %s"
                             % (self.name, len(failures), "\n  - ".join(failures)))
        return report

