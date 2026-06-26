"""Stable PDE-model facade: ``pops.physics.facade.Model``.

``Model`` COMPOSES a private :class:`~pops.physics.model.HyperbolicModel`
(composition, not inheritance) and delegates each call: no new numerics. It is
the PDE convenience facade that ``pops.dsl.Model`` re-exports; the blackboard
board facade (also named ``Model``, re-exported as ``pops.physics.Model``) lives
in :mod:`pops.physics.board`. cf. ``docs/DSL_MODEL_DESIGN.md`` sections 1-3.

Import-graph rule (Spec 4): module-scope imports are confined to :mod:`pops.ir`
and :mod:`pops.model`; codegen is imported LAZILY inside ``Model.compile``.
"""
from pops.ir import Var  # noqa: F401  -- primitive_vars self-reference check
from pops.ir.ops import left, right  # noqa: F401  -- Model.left / Model.right sugar

from ._modelpkg import model as _model
from .aux import aux_total_n_aux, roles_for  # noqa: F401  -- used in Model.compile
from .model import HyperbolicModel, Param
from ._facade_compile import _FacadeCompileMixin


class Model(_FacadeCompileMixin):
    """STABLE facade of a DSL model (Phase A). COMPOSES a private HyperbolicModel (_m, composition and
    NOT inheritance) and delegates each call to an existing method: no new numerics.

        m = pops.dsl.Model("euler")
        rho, rhou, rhov, E = m.conservative_vars("rho", "rho_u", "rho_v", "E")
        g = m.param("gamma", 1.4)                 # NAMED constant, inlined at codegen
        u = m.primitive("u", rhou / rho)
        p = m.primitive("p", (g - 1.0) * (E - 0.5 * rho * (u*u + ...)))
        m.flux(x=[...], y=[...])                   # symbolic DECLARATOR of the physical flux
        m.eval_flux(U, aux, dir)                   # numpy EVALUATOR (debug), DISTINCT name
        m.primitive_vars(rho=rho, u=u, v=v, p=p)   # ordered Prim layout (kwargs order)
        compiled = m.compile(so_path, include, backend="aot")  # -> CompiledModel

    cf. docs/DSL_MODEL_DESIGN.md sections 1-3."""

    def __init__(self, name):
        self._m = HyperbolicModel(name)
        self.params = {}   # name -> Param (introspection / reproducibility)

    @property
    def name(self): return self._m.name

    # --- variable declaration (direct delegation to HyperbolicModel) ---
    def conservative_vars(self, *names, roles=None):
        """Declares the conservative variables. @p roles: same convention as HyperbolicModel."""
        return self._m.conservative_vars(*names, roles=roles)

    def primitive(self, name, expr):
        """Defines a primitive by its formula (as a function of the cons / preceding primitives)."""
        return self._m.primitive(name, expr)

    def primitive_vars(self, *vars, roles=None, **named):
        """Declares the primitives AND the ORDERED layout of Prim. Two forms:

        - KWARGS (target style): `primitive_vars(rho=expr, u=expr, v=expr, p=expr)`: each kwarg
          DEFINES a primitive (m.primitive(name, expr)) AND fixes the layout of Prim in the
          insertion order of the kwargs (Python 3.7+: order guaranteed). @p roles (list) optional.
        - POSITIONAL: `primitive_vars(rho, u, v, p, roles=...)`: names/Var already defined, fixes
          only the layout (delegates to set_primitive_state, like HyperbolicModel).

        The two forms are exclusive (mixing named kwargs and positional raises)."""
        if named and vars:
            raise ValueError("primitive_vars: mixing positional form and named kwargs "
                             "(choose one; kwargs define AND order the primitives)")
        if named:
            # kwargs: define each primitive, then fix the layout in insertion order.
            # A primitive is NOT (re)defined if the kwarg is the Var of the SAME name -- otherwise the codegen
            # would emit `const Real x = x;` (auto-init -> NaN). Two cases of self-reference, both
            # left to JOIN the layout without redefinition (target style primitive_vars(rho=rho, u=u, ...)):
            #  - name ALREADY CONSERVATIVE (e.g. rho=rho: the density, primitive == conservative);
            #  - PRIMITIVE Var ALREADY DEFINED of the same name (e.g. u=u when u comes from m.primitive('u', ...)).
            # Otherwise (kwarg = expression, e.g. p=cs2*rho or u=mx/rho) the primitive is defined normally.
            ordered = list(named.keys())
            for nm in ordered:
                val = named[nm]
                self_ref = isinstance(val, Var) and getattr(val, "name", None) == nm
                if nm in self._m.cons_names or self_ref:
                    continue
                self._m.primitive(nm, val)
            self._m.set_primitive_state(*ordered, roles=roles)
            return tuple(Var(nm, "prim") for nm in ordered)
        # positional form: fixes the layout from already-defined names/Var.
        self._m.set_primitive_state(*vars, roles=roles)
        return None

    def aux(self, name):
        """CANONICAL auxiliary field (must be a key of AUX_CANONICAL: phi/grad_x/grad_y/B_z/T_e)."""
        return self._m.aux(name)

    def aux_field(self, name):
        """NAMED auxiliary field (ADC-70 phase 1) provided by a block via System.set_aux_field(block, name,
        array). name is ARBITRARY (identifier); the k-th call reserves the aux channel component
        AUX_NAMED_BASE + k (read in C++ via aux.extra_field(k)). At most AUX_NAMED_MAX per model.
        Returns a Var usable in flux / source / eigenvalues. Delegates to HyperbolicModel.aux_field."""
        return self._m.aux_field(name)

    def conservative_from(self, exprs):
        """Inverse prim -> cons (the DSL cannot invert symbolically)."""
        self._m.set_conservative_from(exprs)

    # --- flux: symbolic DECLARATOR vs numpy EVALUATOR (DISTINCT names, settled decision) ---
    def flux(self, x, y):
        """Symbolic DECLARATOR of the physical flux (delegates to set_flux). x/y: lists of Expr, one
        per conservative component. DO NOT confuse with the numpy evaluator eval_flux."""
        self._m.set_flux(x, y)

    def flux_term(self, name, x, y):
        """NAMED physical flux F_name(U, primitives, aux, params): exactly n_cons expressions per
        direction (delegates to HyperbolicModel.flux_term). Opt-in -- emitted only when a compiled time
        Program selects it (ctx.rhs(..., fluxes=[name, ...])), never folded into the historical -div F.
        name='default' is the backward-compatible alias of m.flux(...): ctx.rhs(fluxes=['default']) is
        byte-identical to the historical flux-only RHS. A Program requesting several named fluxes
        assembles -div of their SUM."""
        self._m.flux_term(name, x, y)

    def eval_flux(self, U, aux, dir):
        """numpy EVALUATOR of the physical flux (debug / host proto; delegates to HyperbolicModel.flux).
        U: numpy (n_vars, ...); aux: dict name -> array; dir: 0=x, 1=y."""
        return self._m.flux(U, aux, dir)

    def eval_source(self, U, aux):
        """numpy EVALUATOR of the source term (debug / host proto; delegates to
        HyperbolicModel.source_value). U: numpy (n_vars, ...); aux: dict name -> array. Returns
        zeros when no source was declared. Lets a host test check the emitted source (e.g. a BGK
        collision) against an oracle without compiling."""
        return self._m.source_value(U, aux)

    def eigenvalues(self, x, y):
        """Eigenvalues (characteristic speeds) per direction (delegates to set_eigenvalues)."""
        self._m.set_eigenvalues(x, y)

    def wave_speeds(self, x, y):
        """Explicit SIGNED wave speeds per direction: x = (smin_x, smax_x), y = (smin_y,
        smax_y). Emits ``wave_speeds(U, aux, dir, smin, smax)`` on the brick WITHOUT requiring a
        primitive 'p': riemann='hll' becomes available for a model without pressure (moment
        system, isothermal...). Takes priority over the historical path (eigenvalues + 'p'); if
        eigenvalues is not declared, max_wave_speed (Rusanov / CFL) derives from ``max(|smin|, |smax|)``.
        Delegates to set_wave_speeds; cf. HyperbolicModel.set_wave_speeds."""
        self._m.set_wave_speeds(x, y)

    def wave_speeds_from_jacobian(self, x=None, y=None, eig="numeric", blocks=None):
        """EXACT signed wave speeds from the eigenvalues of the flux jacobian (delegates to
        set_wave_speeds_from_jacobian, see its full contract): x/y = dF/dU as Expr (None =
        AUTODIFF of the declared flux via flux_jacobian); eig = 'numeric' | 'fd' (finite differences
        of the compiled flux, debug); blocks = lists of indices of the diagonal sub-blocks (None = full
        block, the only unconditionally correct mode -- the blocks ASSERT a
        block-triangular structure)."""
        self._m.set_wave_speeds_from_jacobian(x=x, y=y, eig=eig, blocks=blocks)

    def eval_wave_speeds(self, U, aux, dir):
        """numpy EVALUATOR of the emitted signed speeds (smin, smax) (delegates to
        HyperbolicModel.wave_speeds_value): explicit pair, jacobian (numpy eig per blocks) or
        min/max of the eigenvalues."""
        return self._m.wave_speeds_value(U, aux, dir)

    def stability_speed(self, expr):
        """STABILITY speed lambda* driving the block CFL (OPTIONAL; delegates to
        HyperbolicModel.stability_speed). Fallback without a call: max(abs(eigenvalues)), strictly
        the historical behavior. Compiled like flux/source (production GPU/MPI, no per-cell callback)."""
        self._m.stability_speed(expr)

    def stability_dt(self, expr_dt):
        """Direct ADMISSIBLE step dt(U, aux) local bound of the step (OPTIONAL; delegates to
        HyperbolicModel.stability_dt). The cfl is not applied to this bound. Fallback without a call:
        no additional bound (historical step policy)."""
        self._m.stability_dt(expr_dt)

    def source(self, s):
        """Source term S(U, aux), one expression per component (optional; delegates to set_source)."""
        self._m.set_source(s)

    def source_term(self, name, exprs):
        """NAMED local source S_name(U, primitives, aux, params): exactly n_cons expressions
        (delegates to HyperbolicModel.source_term). Opt-in -- emitted only when a compiled time
        Program requests it (ctx.rhs(..., sources=[name]) / ctx.source(name)), never summed
        implicitly. name='default' is the backward-compatible alias of m.source([...])."""
        self._m.source_term(name, exprs)

    def linear_source(self, name, matrix):
        """NAMED local linear operator L_name(aux, params) U, an n_cons x n_cons matrix whose
        coefficients are independent of U / primitives (delegates to HyperbolicModel.linear_source).
        Used explicitly by a Program (ctx.linear_source / ctx.apply / ctx.solve_local_linear);
        never folded into m.source or ctx.rhs."""
        self._m.linear_source(name, matrix)

    def rate_operator(self, name, *, flux=True, sources=("default",), fluxes=None):
        """NAMED composite rate operator R_name = -div F + sum(sources) (Spec 2, operator-first):
        a Program-side alias for ctx.rhs(flux=, sources=, fluxes=) so a model-free Program can call
        P.call(name, U[, fields]) instead of P.rhs(...). Delegates to HyperbolicModel.rate_operator."""
        self._m.rate_operator(name, flux=flux, sources=sources, fluxes=fluxes)

    def source_frequency(self, expr_mu):
        """Local frequency mu(U, aux) [1/s] of the source -- the 'source' step bound from the meeting
        (dt <= cfl*substeps/(stride*max mu), without a space step). Emitted on the generated SOURCE
        brick (frequency(U, aux)), forwarded by CompositeModel, aggregated by System/AmrSystem
        step_cfl. REQUIRES m.source([...]). Delegates to HyperbolicModel.source_frequency."""
        self._m.source_frequency(expr_mu)

    def source_jacobian(self, rows):
        """ANALYTIC Jacobian dS/dU of the source (rows[r][c] = dS_r/dU_c, an n_vars x n_vars
        matrix of expressions): the implicit Newton (IMEX/SourceImplicitBE) uses it instead of
        finite differences. REQUIRES m.source. Delegates to HyperbolicModel.source_jacobian."""
        self._m.source_jacobian(rows)

    def projection(self, exprs):
        """PROJECTION PONCTUELLE post-pas U <- P(U, aux) (ADC-177, OPTIONNEL) : une expression par
        composante conservative, appliquee par le System a la FIN de chaque macro-pas ENTIER (jamais
        par etage RK) sur les cellules valides. CONTRAT : idempotente et ponctuelle ; clamps en
        max/min via abs_/sign. Backends 'aot'/'production' (System) ; 'prototype' et AMR rejetes.
        Delegue a HyperbolicModel.projection (cf. son contrat complet)."""
        self._m.projection(exprs)

    def projection_value(self, U, aux=None):
        """EVALUATEUR numpy de la projection emise (reference de test ; delegue a
        HyperbolicModel.projection_value)."""
        return self._m.projection_value(U, aux)

    def implicit_source(self, jacobian=None):
        """GROUPED declaration of the local implicit (wave 3 audit, sugar): the RESIDUAL is already
        implied by m.source (backward-Euler: F = W - U^n - dt*S(W)); @p jacobian (optional) =
        analytic dS/dU matrix (cf. source_jacobian). Without jacobian: finite differences."""
        if jacobian is not None:
            self._m.source_jacobian(jacobian)

    def enable_hllc(self):
        """Emits the HLLC capability (contact_speed + hllc_star_state generated from the ROLES +
        primitive 'p'): riemann='hllc' becomes available for this model EVEN outside 4-variable
        Euler. Delegates to HyperbolicModel.enable_hllc."""
        self._m.enable_hllc()

    def set_riemann_hooks(self, **forms):
        """Record ARBITRARY-formula overrides of the role-derived Riemann hooks (ADC-456): e.g.
        ``pressure=<Expr>`` codegen's that formula as the ``pressure(U)`` hook body. Descriptors /
        ``None`` keep the role-derived default. Delegates to HyperbolicModel.set_riemann_hooks."""
        self._m.set_riemann_hooks(**forms)
        return self

    def enable_roe(self):
        """Emits the ROE capability (roe_dissipation = ``|A_roe| dU`` generated from the ROLES +
        primitive 'p'): riemann='roe' becomes available for this model EVEN outside 4-variable
        Euler (without Energy: c = sqrt(p/rho) averaged Roe-style; components outside the fluid
        roles = passive scalars on the entropy wave). Delegates to HyperbolicModel.enable_roe."""
        self._m.enable_roe()

    def roe_dissipation(self, x, y):
        """Roe dissipation PROVIDED by the user (outside the fluid roles): n_vars expressions per
        direction (x=, y=), written with m.left(...)/m.right(...) (or dsl.left/right) of the two states,
        emitted as the C++ hook roe_dissipation(UL, AL, UR, AR, dir). During the 'provided' mode of enable_roe
        (a single provider: supplying both together raises). The helper m.flux_jacobian assists the writing.
        Delegates to HyperbolicModel.roe_dissipation (cf. its doc)."""
        self._m.roe_dissipation(x, y)

    def flux_jacobian(self, dir):
        """Flux Jacobian A = dF_dir/dU (an n_vars x n_vars matrix of Expr, A[i][j]=d(F_i)/d(U_j)),
        auto-derived from the fluxes declared via dsl.diff (expanded primitives). HELPER for building
        m.roe_dissipation, emits nothing. @p dir: 0/'x' or 1/'y'. Delegates to HyperbolicModel."""
        return self._m.flux_jacobian(dir)

    def roe_from_jacobian(self):
        """Generic moment Roe: emits roe_dissipation = ``|A| (UR-UL)`` with A the flux Jacobian at
        Uavg = 1/2(UL+UR) and |A| via pops::roe_abs_apply (matrix-sign), spectral-radius Rusanov
        fallback on a complex/singular spectrum. Roles-free (no Density/Momentum, no 'p'): makes
        riemann='roe' available for a moment hierarchy. Exclusive with enable_roe / roe_dissipation.
        Delegates to HyperbolicModel.roe_from_jacobian (cf. its doc)."""
        self._m.roe_from_jacobian()

    def left(self, expr):
        """Marks @p expr as evaluated on the LEFT state UL (m.roe_dissipation). Sugar for dsl.left."""
        return left(expr)

    def right(self, expr):
        """Marks @p expr as evaluated on the RIGHT state UR (m.roe_dissipation). Sugar for dsl.right."""
        return right(expr)

    def elliptic_rhs(self, e):
        """Contribution to the elliptic right-hand side (Poisson coupling; delegates to set_elliptic_rhs)."""
        self._m.set_elliptic_rhs(e)

    def elliptic_field(self, name, rhs, operator="poisson", aux=None):
        """NAMED elliptic field: an elliptic solve operator(field) = rhs(U) populating the named @p aux
        fields (default ['phi', 'grad_x', 'grad_y']); delegates to HyperbolicModel.elliptic_field. The
        IR + validation + hash land; the multi-field RUNTIME (a second elliptic operator + aux channel)
        is DEFERRED -- ctx.solve_fields(field=name) raises NotImplementedError on lowering."""
        self._m.elliptic_field(name, rhs, operator=operator, aux=aux)

    def gamma(self, value):
        """Adiabatic index (EOS), carried by POPS_EXPORT_BLOCK_GAMMA (delegates to set_gamma)."""
        self._m.set_gamma(value)

    def param(self, name, value, kind="const"):
        """NAMED parameter usable in the formulas. Mode (a) (`kind="const"`, default): constant
        frozen at compile time, inlined at codegen; stored in m.params (introspection /
        reproducibility). Mode (b) (`kind="runtime"`, P7-b): SUPPORTED on the "aot" backend -- the param
        emits `params.get(<index>)` (member of pops::RuntimeParams) and its value can be CHANGED at runtime
        via System.set_block_params(name, values) WITHOUT recompiling (the declaration value serves as the
        default); cf. CompiledModel.runtime_param_names. The "prototype"/"production" backends freeze a
        runtime param at its declaration value.

        gamma CASE: if name == "gamma", ALSO calls set_gamma(value) so that the ABI metadata
        stays consistent (otherwise the System falls back to 1.4)."""
        p = Param(name, value, kind=kind)  # 'runtime' -> RuntimeParamRef (P7-b), 'const' -> inline
        self.params[name] = p
        if name == "gamma":
            self._m.set_gamma(p.value)
        return p

    def check(self):
        """Checks the dependencies (referenced variables are declared). Raises ValueError otherwise."""
        return self._m.check()

    def check_model(self, samples=None, n_samples=64, seed=0, aux=None, rtol=1e-8, atol=1e-10,
                    raise_on_error=True, jac_rtol=1e-3, jac_atol=1e-9):
        """Generic NUMERICAL verification of the model (finite flux/source/elliptic, real and finite
        eigenvalues, wave_speeds/max_wave_speed consistency, non-circular bounding of the spectrum by
        the dense Jacobian, cons<->prim round-trip, positivity of Density/'p') on sample states.
        Delegates to HyperbolicModel.check_model (cf. its doc). To be called BEFORE compile(); the
        runtime counterpart of an installed block is System.check_model."""
        return self._m.check_model(samples=samples, n_samples=n_samples, seed=seed, aux=aux,
                                   rtol=rtol, atol=atol, raise_on_error=raise_on_error,
                                   jac_rtol=jac_rtol, jac_atol=jac_atol)

    # --- introspection (read-only, delegated to the backing model) ---
    @property
    def cons_names(self): return self._m.cons_names

    @property
    def prim_state(self): return self._m.prim_state

    @property
    def n_vars(self): return self._m.n_vars

    def state_space(self, name="U"):
        """Typed pops.model.StateSpace view of the conservative state (Spec 2).
        Delegates to HyperbolicModel.state_space."""
        return self._m.state_space(name)

    def field_space(self, name="fields"):
        """Typed pops.model.FieldSpace view of the auxiliary surface (Spec 2).
        Delegates to HyperbolicModel.field_space."""
        return self._m.field_space(name)

    def operator_registry(self, state_name="U"):
        """Typed pops.model.OperatorRegistry derived from this model (Spec 2): the PDE
        shortcuts (source_term / linear_source / elliptic_field / flux) lower into
        typed operators. Pure view -- no hash or codegen impact. Delegates to
        HyperbolicModel.operator_registry."""
        return self._m.operator_registry(state_name)

    @property
    def module(self):
        """The pops.model.Module view of this PDE model (Spec 2, operator-first): its typed
        StateSpace / FieldSpace and the OperatorRegistry that source_term / linear_source /
        elliptic_field / flux / rate_operator populate. dsl.Model is the PDE convenience
        facade; the Module is the model-free view a generic Program binds to (P.bind_operators).
        The Module carries no numerics; codegen still reads this Model via compile_problem."""
        mod = _model.Module(self.name)
        st = self._m.state_space()
        mod.state_space(st.name, st.components, roles=st.roles, layout=st.layout,
                        storage=st.storage)
        fs = self._m.field_space()
        mod.field_space(fs.name, fs.components, layout=fs.layout)
        mod.adopt_registry(self._m.operator_registry())
        return mod

    # --- operator introspection (Spec 2, S2-5) ---
    def list_operators(self):
        """Names of the typed operators this model exposes (registration order)."""
        return self._m.operator_registry().names()

    def list_state_spaces(self):
        """Names of the model's state spaces (one, the conservative state)."""
        return [self._m.state_space().name]

    def list_field_spaces(self):
        """Names of the model's field spaces (one, the auxiliary surface)."""
        return [self._m.field_space().name]

    def operator_signature(self, name):
        """The pops.model.Signature of operator ``name``."""
        return self._m.operator_registry().get(name).signature

    def operator_requirements(self, name):
        """The requirements dict of operator ``name``."""
        return dict(self._m.operator_registry().get(name).requirements)

    def operator_capabilities(self, name):
        """The capabilities dict of operator ``name``."""
        return dict(self._m.operator_registry().get(name).capabilities)

