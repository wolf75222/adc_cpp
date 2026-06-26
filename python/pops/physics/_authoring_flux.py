"""Authoring mixin: flux, eigenvalues, wave speeds, flux Jacobian, gamma.

Methods only; the instance attributes (``_flux`` / ``_flux_terms`` / ``_eig`` /
``_wave_speeds`` / ``_ws_jacobian`` / ``gamma``) are created by
``HyperbolicModel.__init__``. Codegen-free and ``_pops``-free at module scope:
the one pure codegen helper used (``_dir_key``) is imported LAZILY inside the
method body, mirroring how the compile wrappers delegate to ``pops.codegen``.
"""
from pops.ir import _wrap, diff
from pops.ir.ops import left, right


class _FluxMixin:
    """Physical flux, eigenvalues, signed wave speeds, and the flux Jacobian."""

    def set_flux(self, x, y): self._flux = {"x": list(x), "y": list(y)}
    def set_eigenvalues(self, x, y): self._eig = {"x": list(x), "y": list(y)}

    def flux_term(self, name, x, y):
        """Declare a NAMED physical flux F_name(U, primitives, aux, params): exactly n_cons
        expressions per direction (x= the x-flux, y= the y-flux), free to depend on cons / primitives /
        aux / aux_field / params / constants -- the same dependency surface as set_flux. A named flux is
        OPT-IN: it is emitted only when a compiled time Program selects it (ctx.rhs(..., fluxes=[name,
        ...])) and is NEVER folded into the historical -div F (rhs_into). name == "default" is the
        backward-compatible alias of m.flux(...) (stored in self._flux, hash unchanged): so
        ctx.rhs(fluxes=["default"]) is byte-identical to the historical flux-only RHS. Other names must
        be valid identifiers, unique, and not collide with another named flux. A Program that requests
        several named fluxes assembles -div of their SUM (the codegen emits one kernel per name and
        sums them), so splitting the physical flux into named pieces that sum to it reproduces -div F."""
        n = self.n_vars
        if n == 0:
            raise ValueError("flux_term(%r): declare conservative_vars(...) first" % (name,))
        if not isinstance(name, str) or not name:
            raise ValueError("flux_term: name must be a non-empty string")
        x = [_wrap(e) for e in x]
        y = [_wrap(e) for e in y]
        if len(x) != n or len(y) != n:
            raise ValueError("flux_term('%s'): %d/%d expressions (x/y) for %d conservative variables"
                             % (name, len(x), len(y), n))
        if name == "default":
            self._flux = {"x": x, "y": y}   # equivalent to m.flux(...) -- the legacy default flux
            return
        if not name.isidentifier():
            raise ValueError("flux_term('%s'): name must be a valid identifier "
                             "(letters/digits/_, no leading digit)" % name)
        if name in self._flux_terms:
            raise ValueError("flux_term('%s'): already declared" % name)
        self._flux_terms[name] = {"x": x, "y": y}

    def set_wave_speeds(self, x, y):
        """Explicit SIGNED wave speeds per direction: x = (smin_x, smax_x), y = (smin_y,
        smax_y), expressions of cons / prims / aux. Emits ``wave_speeds(U, aux, dir, smin, smax)``
        on the generated brick WITHOUT requiring a primitive 'p': riemann='hll' becomes available for
        a pressureless model (moment system, isothermal...). The core only gates HLL on
        requires { m.wave_speeds(...) } (block_builder.hpp): no C++ change.

        Takes priority over the historical path (primitive 'p' -> wave_speeds = min/max of
        eigenvalues) when both exist. WITHOUT a call: strictly historical emission.
        If set_eigenvalues is NOT called, max_wave_speed (Rusanov / CFL) is derived from
        ``max(|smin|, |smax|)`` over the two expressions of the direction."""
        x, y = tuple(x), tuple(y)
        if len(x) != 2 or len(y) != 2:
            raise ValueError("set_wave_speeds : expected x=(smin, smax) and y=(smin, smax) "
                             "(got x=%d expression(s), y=%d)" % (len(x), len(y)))
        if self._ws_jacobian is not None:
            raise ValueError("set_wave_speeds : set_wave_speeds_from_jacobian already declared -- one "
                             "single wave_speeds provider")
        self._wave_speeds = {"x": (_wrap(x[0]), _wrap(x[1])),
                             "y": (_wrap(y[0]), _wrap(y[1]))}

    def set_wave_speeds_from_jacobian(self, x=None, y=None, eig="numeric", blocks=None):
        """EXACT signed wave speeds: smin/smax = extremes of the flux jacobian's eigenvalues
        A = dF/dU, computed NUMERICALLY per cell (pops::real_eig_minmax, Francis QR
        on a stack buffer, Gershgorin fallback on non-convergence = safe outer bound). Emits
        ``wave_speeds(U, aux, dir, smin, smax)`` (core HLL gate) and, without set_eigenvalues,
        ``max_wave_speed`` = ``max(|smin|, |smax|)`` over the same blocks.

        @p x, @p y : n_vars x n_vars matrices of expressions dA[i][j] = dF_dir[i]/dU[j]. None
        (default) = AUTODIFF of the declared flux via flux_jacobian(dir) (dsl.diff, primitives
        expanded by the chain rule) -- the jacobian can then not desynchronize from the
        flux. Providing explicit x/y only makes sense to bypass autodiff (hand-simplified
        forms); check_model then confronts them against the flux finite differences.

        @p eig : "numeric" (default) = jacobian entries emitted as formulas, per-block
        eigenvalues at runtime; "fd" = jacobian built BY COLUMNS from the finite differences of the
        COMPILED flux ((flux(U + eps e_k) - flux(U))/eps, ``eps = 1e-6 |U[0]| + 1e-30``, mirror of the
        flagsym != 1 branch of the reference MATLAB) -- generic bring-up/debug, never
        production (O(eps) truncation).

        @p blocks : None (default) = ONE full n_vars x n_vars block, the only unconditionally
        correct mode. Otherwise, a list of INDEX LISTS (possibly non-contiguous, e.g.
        [[0, 1, 4], [2, 3]]) applied to BOTH directions, or a dict {"x": [...], "y": [...]}
        (the block-triangular structures of dFx/dU and dFy/dU differ in general: for a
        moment system, the chains in x are contiguous and those in y are not).
        The extremes are taken over the union of the spectra of the diagonal sub-blocks A[idx][idx].
        CONTRACT: the caller ASSERTS that A is block-(lower-)triangular according to this
        partition (up to permutation) -- on an arbitrary matrix the sub-block extremes
        DO NOT BOUND the spectrum (counter-example [[0, k], [k, 0]]: spectrum +-k, 1x1 sub-blocks
        zero). Indices may be omitted (rows/columns carrying no extreme eigenvalue,
        cf. the skipped block of the reference MATLAB).

        Diagnostics: QR non-convergence silently falls back to the block's Gershgorin bound
        (WIDER, never wrong -- HLL stays stable, only more diffusive); a loss
        of hyperbolicity (complex eigenvalues) is not reported per cell -- verify it
        offline (check_model, golden type eigenvalues15_2D)."""
        if self._wave_speeds is not None:
            raise ValueError("set_wave_speeds_from_jacobian : set_wave_speeds already declared -- one "
                             "single wave_speeds provider")
        if eig not in ("numeric", "fd"):
            raise ValueError("set_wave_speeds_from_jacobian : eig 'numeric' | 'fd' (got %r)" % (eig,))
        nv = self.n_vars
        if (x is None) != (y is None):
            raise ValueError("set_wave_speeds_from_jacobian : provide x AND y, or neither (autodiff)")
        if eig == "fd" and x is not None:
            raise ValueError("set_wave_speeds_from_jacobian : eig='fd' builds the jacobian from "
                             "finite differences of the compiled flux -- x/y make no sense here")
        rows = {}
        if eig == "numeric":
            if x is None:
                if not self._flux:
                    raise ValueError("set_wave_speeds_from_jacobian : call set_flux(...) first "
                                     "(jacobian autodiff)")
                rows = {"x": self.flux_jacobian(0), "y": self.flux_jacobian(1)}
            else:
                for key, mat in (("x", x), ("y", y)):
                    if len(mat) != nv or any(len(r) != nv for r in mat):
                        raise ValueError("set_wave_speeds_from_jacobian : jacobian %s expected "
                                         "%d x %d" % (key, nv, nv))
                    rows[key] = [[_wrap(e) for e in r] for r in mat]
        def norm_blocks(blk, label):
            blk = [list(int(i) for i in b) for b in blk]
            seen = set()
            for b in blk:
                if not b:
                    raise ValueError("set_wave_speeds_from_jacobian : empty block (%s)" % label)
                local = set()
                for i in b:
                    if not (0 <= i < nv):
                        raise ValueError("set_wave_speeds_from_jacobian : index %d out of [0, %d) "
                                         "(%s)" % (i, nv, label))
                    if i in local:
                        raise ValueError("set_wave_speeds_from_jacobian : index %d present twice "
                                         "in the same block (%s)" % (i, label))
                    if i in seen:
                        raise ValueError("set_wave_speeds_from_jacobian : index %d present in "
                                         "two blocks (%s)" % (i, label))
                    local.add(i)
                    seen.add(i)
            return blk

        if blocks is None:
            per_dir = {"x": [list(range(nv))], "y": [list(range(nv))]}
        elif isinstance(blocks, dict):
            if set(blocks) != {"x", "y"}:
                raise ValueError("set_wave_speeds_from_jacobian : blocks dict expected with "
                                 "keys 'x' and 'y' (got %r)" % sorted(blocks))
            per_dir = {k: norm_blocks(blocks[k], k) for k in ("x", "y")}
        else:
            shared = norm_blocks(blocks, "x and y")
            per_dir = {"x": shared, "y": [list(b) for b in shared]}
        self._ws_jacobian = {"rows": rows or None, "eig": eig, "blocks": per_dir,
                             "explicit": x is not None}

    def flux_jacobian(self, dir):
        """Flux jacobian A = dF_dir/dU : n_vars x n_vars matrix of expressions, A[i][j] =
        d(flux_dir[i])/d(cons[j]), auto-derived from the declared fluxes (via dsl.diff with primitive
        substitution). CONSTRUCTION HELPER (the user uses it to write m.roe_dissipation):
        EMITS NOTHING by itself. @p dir : 0/'x' (x axis) or 1/'y' (y axis). REQUIRES set_flux(...).

        The primitives are expanded by their definition (chain); a non-derived primitive
        stays a symbol in the result (evaluating it numerically requires an env containing its values,
        e.g. HyperbolicModel._env)."""
        if not self._flux:
            raise ValueError("flux_jacobian : call set_flux(...) first")
        from pops.codegen.cpp_writer import _dir_key  # lazy: keep physics codegen-free at import
        key = _dir_key(dir)
        comps = self._flux.get(key, [])
        if len(comps) != self.n_vars:
            raise ValueError("flux_jacobian : flux %s expected with %d components (got %d)"
                             % (key, self.n_vars, len(comps)))
        defs = self.prim_defs
        return [[diff(comps[i], self.cons_names[j], defs) for j in range(self.n_vars)]
                for i in range(self.n_vars)]

    def left(self, expr):
        """Marks @p expr as evaluated on the LEFT state UL (sugar for dsl.left, m.roe_dissipation)."""
        return left(expr)

    def right(self, expr):
        """Marks @p expr as evaluated on the RIGHT state UR (sugar for dsl.right, m.roe_dissipation)."""
        return right(expr)

    def set_gamma(self, gamma):
        """Adiabatic index of the block (compressible EOS). Carried by the generated .so via the
        optional symbol pops_compiled_gamma, so that the System's inter-species couplings (collision,
        thermal exchange, T_e) use the RIGHT gamma instead of the historical default 1.4. Without a call,
        no gamma symbol is emitted (backward compat: the System keeps its default)."""
        self.gamma = float(gamma)

