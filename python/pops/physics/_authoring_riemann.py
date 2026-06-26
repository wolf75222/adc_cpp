"""Authoring mixin: Riemann capabilities (HLLC, Roe) and hook overrides.

Methods only; the touched attributes (``_hllc`` / ``_roe`` / ``_roe_rows`` /
``_roe_jacobian`` / ``_riemann_hook_forms``) are created by
``HyperbolicModel.__init__``. ``roe_from_jacobian`` reuses ``flux_jacobian``
(provided by the flux mixin) on ``self``. Codegen-free and ``_pops``-free at
module scope: ``_roe_validate`` (a pure marker validator) is imported LAZILY
inside ``roe_dissipation``.
"""
from pops.ir import Expr, _wrap  # noqa: F401  -- _wrap in roe_dissipation, Expr in hook checks


class _RiemannMixin:
    """HLLC / Roe capability emission and arbitrary-formula hook overrides."""

    # Riemann capability hooks whose body is a SINGLE-state formula (signature ``hook(U)``) and so
    # can be codegen'd from an arbitrary board Expr in the model's own symbols (ADC-456). The
    # two-state hooks (contact_speed / hllc_star_state, signature over UL/UR/pL/pR/sL/sR) are NOT
    # in this set: an arbitrary single-Expr override is ill-defined for them, so they keep the
    # role-derived default (selected by a capability-hook descriptor).
    _FORMULA_HOOKS = ("pressure",)

    def enable_hllc(self):
        """Emits the HLLC CAPABILITY (audit wave 3): ``contact_speed`` (Toro) + ``hllc_star_state``
        GENERATED from the block's ROLES (Density / MomentumX / MomentumY, Energy optional) and the
        primitive 'p' -- the core's contact-resolving HLLC solver (C++ trait HasHLLCStructure)
        then becomes available for THIS model, EVEN outside 4-variable Euler (3-var isothermal,
        moments with passive scalars: any component without a particular role is advected
        passively in the star state, Us[c] = fac*U[c]/rho). REQUIRES: roles Density/MomentumX/
        MomentumY declared + primitive 'p' (explicit error at emission otherwise)."""
        self._hllc = True
        return self

    def set_riemann_hooks(self, **forms):
        """Record ARBITRARY-formula overrides of the role-derived Riemann hooks (ADC-456, Spec 3
        section 11). Each keyword is a hook name; the value is an :class:`Expr` (an ``pops.math`` /
        dsl formula) that REPLACES the canonical role-derived body at codegen, or ``None`` to keep
        the default. Currently codegen'd hook: ``pressure`` (replaces the ``pressure(U)`` body,
        default = the primitive 'p' formula).

        Only :class:`Expr` values are recorded; a non-Expr (a :class:`pops.lib.BrickDescriptor`
        selecting a canonical scheme, or ``None``) is ignored and the role-derived default stands.
        An Expr passed for a two-state hook (``contact_speed`` / ``star_state`` / ``sound_speed``)
        raises: those keep the role-derived default selected via a hook descriptor. A formula
        referencing a quantity the model cannot provide still raises the clear capability error at
        emission. Without any Expr override the module hash and the emitted brick are bit-identical
        to the role-derived path."""
        for name, form in forms.items():
            if not isinstance(form, Expr):
                continue  # descriptor / None: role-derived default stands
            if name not in self._FORMULA_HOOKS:
                raise NotImplementedError(
                    "riemann hook %r does not yet accept an arbitrary board formula (its C++ "
                    "signature spans two states); pass a capability-hook descriptor "
                    "(e.g. pops.lib.riemann.hllc.contact_speed.euler()) for the role-derived "
                    "default. Formula override is available for: %s"
                    % (name, ", ".join(self._FORMULA_HOOKS)))
            self._riemann_hook_forms[name] = form
        return self

    def enable_roe(self):
        """Emits the ROE CAPABILITY (audit balance, GENERICITY_2026-06.md point 11):
        ``roe_dissipation(UL, AL, UR, AR, dir)`` = ``|A_roe| (UR - UL)`` GENERATED from the block's
        ROLES -- the core's Roe-like solver (C++ trait HasRoeDissipation, F = 1/2(FL+FR) - 1/2 d)
        becomes available for THIS model, EVEN outside 4-variable Euler:

        - roles Density/MomentumX/MomentumY + Energy: ideal-gas Roe algebra, exact
          TRANSCRIPTION of the canonical C++ path (sqrt(rho)-weighted averages, gamma-1 deduced from
          ``p/(E - 1/2 rho |v|^2)``, Harten entropy fix on the acoustic waves);
        - roles Density/MomentumX/MomentumY WITHOUT Energy (isothermal / pseudo-pressure): same
          decomposition without the energy row, LOCAL sound speed c = sqrt(p/rho) Roe-averaged
          (standard generalization outside ideal gas);
        - any component OUTSIDE the fluid roles is treated as a PASSIVE SCALAR carried by the
          entropy wave (row identical to the tangential momentum, phi = q/rho).

        REQUIRES: roles Density/MomentumX/MomentumY declared + primitive 'p' (explicit error at
        emission otherwise). Without a call: nothing emitted, riemann='roe' stays Euler-4-var-only.

        EXCLUSIVE with m.roe_dissipation: the capability from the roles and the dissipation PROVIDED by
        the user are two providers of the SAME roe_dissipation hook -- declaring both
        raises (one single provider)."""
        if self._roe_rows is not None:
            raise ValueError("enable_roe : roe_dissipation(...) already provided -- one single provider "
                             "of the roe_dissipation hook (capability from the roles OR provided)")
        if self._roe_jacobian is not None:
            raise ValueError("enable_roe : roe_from_jacobian() already declared -- one single provider "
                             "of the roe_dissipation hook")
        self._roe = True

    def roe_dissipation(self, x, y):
        """Roe dissipation PROVIDED by the user (outside the fluid-role families): n_vars
        expressions per direction (rows d_i), emitted as the C++ hook
        ``roe_dissipation(UL, AL, UR, AR, dir)`` = d (HasRoeDissipation trait; the core does
        F = 1/2(FL+FR) - 1/2 d, cf. RoeFlux). It is the 'provided' counterpart of m.enable_roe (generated
        from the ROLES): here the user writes THEIR eigenstructure -- same spirit as
        m.source_jacobian (provided, not invented). The helper m.flux_jacobian(dir) (A = dF/dU
        auto-derived by dsl.diff) assists this writing.

        TWO-STATE VOCABULARY: each variable/primitive must be wrapped by dsl.left(...) (state
        UL) or dsl.right(...) (state UR); a Roe average is therefore written explicitly
        (left(sqrt(rho))*left(u) + right(sqrt(rho))*right(u)) / (left(sqrt(rho)) + right(sqrt(rho))).
        A BARE variable (without a marker) raises at declaration (undetermined state).

        @p x, @p y : lists of n_vars expressions (rows for dir=0 and dir=1). TWO EXPLICIT sets
        (no role mapping here): at dir=0 the normal component is the x axis, at dir=1 the y axis.

        Guards: length n_vars per direction; each variable under left/right; conflict with
        enable_roe (one single provider of the hook) -> error. WITHOUT a call: nothing emitted (bit-identical).
        Requires the 'aot' or 'production' backend (the hook is emitted in the generated brick)."""
        if self._roe:
            raise ValueError("roe_dissipation : enable_roe() already called -- one single provider of the "
                             "roe_dissipation hook (capability from the roles OR provided)")
        if self._roe_jacobian is not None:
            raise ValueError("roe_dissipation : roe_from_jacobian() already declared -- one single "
                             "provider of the roe_dissipation hook")
        rx, ry = list(x), list(y)
        if len(rx) != self.n_vars or len(ry) != self.n_vars:
            raise ValueError("roe_dissipation : %d expressions expected per direction (got x=%d, "
                             "y=%d)" % (self.n_vars, len(rx), len(ry)))
        from pops.codegen.cpp_writer import _roe_validate  # lazy: keep physics codegen-free at import
        rows = {"x": [_wrap(e) for e in rx], "y": [_wrap(e) for e in ry]}
        for key in ("x", "y"):
            for e in rows[key]:
                _roe_validate(e, False)  # rejects any variable outside a left()/right() marker
        self._roe_rows = rows

    def roe_from_jacobian(self):
        """Generic moment Roe: emit the hook ``roe_dissipation(UL, AL, UR, AR, dir)`` =
        ``|A| (UR - UL)`` with ``A = dF_dir/dU`` the flux Jacobian (m.flux_jacobian, autodiff)
        evaluated at the ARITHMETIC MEAN interface state ``Uavg = 1/2 (UL + UR)``, and ``|A|`` via
        the matrix-sign kernel ``pops::roe_abs_apply`` (dense_eig.hpp): for a real-diagonalizable A
        this is ``R |Lambda| R^-1`` exactly, the dissipation of the reference flux_ROE. On a complex
        or singular spectrum the kernel returns false and the hook FALLS BACK to a spectral-radius
        (Rusanov) dissipation ``rho (UR - UL)``, ``rho = max(|lmin|, |lmax|)`` of
        ``pops::real_eig_minmax(A)`` -- so the dissipation is always well defined.

        Unlike m.enable_roe (which needs fluid roles Density/MomentumX/MomentumY + primitive 'p'),
        this path needs NEITHER -- it is the GENERIC provider for a moment hierarchy (HyQMOM), making
        riemann='roe' available with no Euler-4-var assumption. The FULL n_vars x n_vars Jacobian is
        always eigendecomposed (as the reference flux_ROE does), not a block partition.

        EXCLUSIVE with m.enable_roe and m.roe_dissipation: the three are providers of the SAME
        roe_dissipation hook (declaring more than one raises). Requires set_flux(...) and the 'aot'
        or 'production' backend (the hook is emitted in the generated brick). WITHOUT a call: nothing
        emitted (bit-identical cache key)."""
        if self._roe:
            raise ValueError("roe_from_jacobian : enable_roe() already called -- one single provider "
                             "of the roe_dissipation hook")
        if self._roe_rows is not None:
            raise ValueError("roe_from_jacobian : roe_dissipation(...) already provided -- one single "
                             "provider of the roe_dissipation hook")
        self._roe_jacobian = {"x": self.flux_jacobian(0), "y": self.flux_jacobian(1)}
