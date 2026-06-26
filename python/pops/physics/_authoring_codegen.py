"""Authoring mixin: thin codegen wrappers (lazy, codegen-free at import).

Every method here delegates to a free function in :mod:`pops.codegen`; the
codegen module is imported LAZILY inside each method body so that importing
``pops.physics`` never pulls in :mod:`pops.codegen` or ``_pops`` (Spec-4
import-graph rule). This is the same delegation the historical ``dsl.py`` used.
"""
from .aux import roles_for


def _cg_compile():
    """The :mod:`pops.codegen.compile` module (lazy import; keeps physics codegen-free)."""
    from pops.codegen import compile as _cg
    return _cg


class _CodegenMixin:
    """C++ emission and compilation wrappers, all delegating lazily to pops.codegen."""

    def _codegen_exprs(self, exprs, cse, real="pops::Real", indent="    "):
        from pops.codegen import module_codegen as _cg
        return _cg._codegen_exprs(self, exprs, cse, real=real, indent=indent)

    def _live_prims(self, exprs, seed=()):
        from pops.codegen import module_codegen as _cg
        return _cg._live_prims(self, exprs, seed=seed)

    def _prim_block(self, live=None, hoist=False):
        from pops.codegen import module_codegen as _cg
        return _cg._prim_block(self, live=live, hoist=hoist)

    def _jac_entries(self):
        from pops.codegen import module_codegen as _cg
        return _cg._jac_entries(self)

    def emit_cpp(self, func=None, cse=True):
        """Generates a compilable C++ function computing the physical flux from the symbolic
        tree (each Expr node knows how to write itself in C++ via to_cpp).

        Produced signature : template <class Real> void <func>_flux(const Real* U, Real* F, int dir).
        Constants inlined ; each primitive becomes a local variable. cse=True (default) factors
        the common subexpressions (H, c...) into ``cseK_`` locals ; cse=False recomputes them inline.

        Step (2) of the DSL (see docs/ARCHITECTURE_CIBLE.md sect. 3) : HOST C++ (templatable on Real)."""
        from pops.codegen import module_codegen as _cg
        return _cg.emit_cpp(self, func=func, cse=cse)

    def emit_cpp_brick(self, name=None, namespace="pops_generated", cse=True,
                       hoist_reciprocals=False):
        """Generates a C++ BRICK satisfying the pops::HyperbolicModel concept (wrapping : step
        2bis). The produced struct uses StateVec / Aux / POPS_HD / Variables and exposes flux,
        max_wave_speed, to_primitive, to_conservative, conservative_vars, primitive_vars : it can
        therefore enter a CompositeModel and run in the compiled solver.

        Requires set_primitive_state(...) (Prim layout) and set_conservative_from([...]) (to_conservative,
        which the DSL cannot invert on its own). cse=True (default) factors the common
        subexpressions (H, c...) into ``cseK_`` locals. Still to do (see ARCHITECTURE_CIBLE.md sect. 3) :
        Kokkos/CUDA codegen, JIT."""
        from pops.codegen import module_codegen as _cg
        return _cg.emit_cpp_brick(self, name=name, namespace=namespace, cse=cse,
                                  hoist_reciprocals=hoist_reciprocals)

    def emit_cpp_source(self, name=None, namespace="pops_generated", cse=True,
                        hoist_reciprocals=False):
        """Generate a composable C++ SOURCE BRICK (in the pops sense) from self._source.

        The produced struct exposes apply(U, a) returning the source term S(U, aux), with one line per
        conservative component (S[i] = self._source[i].to_cpp()). It has the same form as the source
        bricks written by hand (NoSource, PotentialForce in pops/model/bricks.hpp) and can therefore
        enter as the Source parameter of a CompositeModel.

        CONVENTION: the auxiliary names (set via aux(...)) must be FIELDS of pops::Aux,
        because they are read directly as a.<name> (e.g. aux('grad_x') -> a.grad_x, aux('grad_y') ->
        a.grad_y). This convention is the same as that of the manual bricks, where the source reads
        the outer state only through the pops::Aux channel (potential and its gradient).

        Style identical to emit_cpp_brick (inlined constants, cons -> locals, primitives -> locals;
        plus, aux -> locals); cse=True factors the common sub-expressions. Raises ValueError if
        set_source(...) has not been called."""
        from pops.codegen import module_codegen as _cg
        return _cg.emit_cpp_source(self, name=name, namespace=namespace, cse=cse,
                                   hoist_reciprocals=hoist_reciprocals)

    def _emit_bricks(self, name=None, hoist_reciprocals=False):
        """Generate the bricks (hyperbolic + source + elliptic) and the CompositeModel<...> type
        shared by BOTH backends (JIT IModel and AOT). Source / elliptic OPTIONAL: without
        set_source -> pops::NoSource; without set_elliptic_rhs -> zero rhs (no Poisson coupling).
        @p hoist_reciprocals: codegen option propagated to the bricks (cf. emit_cpp_brick).
        Returns (nv, bricks_code, composite_type)."""
        from pops.codegen import module_codegen as _cg
        return _cg._emit_bricks(self, name=name, hoist_reciprocals=hoist_reciprocals)

    def _elliptic_field_registrations(self, nm):
        """Per named elliptic field (ADC-428): (field, brick_struct, phi_comp, gx_comp, gy_comp) for the
        native loader. The aux component of each output name is its channel index: a CANONICAL name
        (phi/grad_x/...) maps via AUX_CANONICAL; a model-named aux (aux_field) maps to
        AUX_NAMED_BASE + its position in aux_extra_names. A name the model never declared as an aux is
        rejected (the solve would write a component no source can read). gx/gy default to -1 (phi only)
        when the field lists fewer than 3 aux names."""
        from pops.codegen import module_codegen as _cg
        return _cg._elliptic_field_registrations(self, nm)

    def _emit_metadata(self, model_alias):
        """OPTIONAL metadata symbols of the .so block, read by dlsym on the System side. SHARED by both
        backends (JIT and AOT). The NAMES + ROLES are always emitted (POPS_EXPORT_BLOCK_METADATA):
        they come from the model's VariableSet (single source of truth), the System reads them instead of
        the u0.. fallback / no roles. The GAMMA is emitted (POPS_EXPORT_BLOCK_GAMMA) only if set_gamma(...)
        has been called; otherwise no gamma symbol -> the System keeps its default 1.4 (backward-compat).

        @p model_alias must be an alias WITHOUT a top-level comma (the preprocessor splits
        macro arguments on commas): callers pass a `using ... = CompositeModel<...>`."""
        from pops.codegen import module_codegen as _cg
        return _cg._emit_metadata(self, model_alias)

    def emit_cpp_so_source(self, name=None, hoist_reciprocals=False):
        """Thin wrapper: delegates to pops.codegen.compile.emit_cpp_so_source."""
        return _cg_compile().emit_cpp_so_source(self, name=name, hoist_reciprocals=hoist_reciprocals)

    def compile_so(self, so_path, include=None, name=None, cxx=None, std="c++20",
                   hoist_reciprocals=False):
        """Thin wrapper: delegates to pops.codegen.compile.compile_so."""
        return _cg_compile().compile_so(self, so_path, include=include, name=name, cxx=cxx, std=std,
                              hoist_reciprocals=hoist_reciprocals)

    def emit_cpp_aot_source(self, name=None, hoist_reciprocals=False):
        """Thin wrapper: delegates to pops.codegen.compile.emit_cpp_aot_source."""
        return _cg_compile().emit_cpp_aot_source(self, name=name, hoist_reciprocals=hoist_reciprocals)

    def compile_aot(self, so_path, include=None, name=None, cxx=None, std="c++20",
                    hoist_reciprocals=False):
        """Thin wrapper: delegates to pops.codegen.compile.compile_aot."""
        return _cg_compile().compile_aot(self, so_path, include=include, name=name, cxx=cxx, std=std,
                               hoist_reciprocals=hoist_reciprocals)

    def emit_cpp_native_loader(self, name=None, target="system", hoist_reciprocals=False):
        """Thin wrapper: delegates to pops.codegen.compile.emit_cpp_native_loader."""
        return _cg_compile().emit_cpp_native_loader(self, name=name, target=target,
                                          hoist_reciprocals=hoist_reciprocals)

    def compile_native(self, so_path, include=None, name=None, cxx=None, std="c++23", target="system",
                       hoist_reciprocals=False):
        """Thin wrapper: delegates to pops.codegen.compile.compile_native."""
        return _cg_compile().compile_native(self, so_path, include=include, name=name, cxx=cxx, std=std,
                                  target=target, hoist_reciprocals=hoist_reciprocals)

    def compile_or_jit(self, so_path, include=None, mode="jit", name=None, cxx=None, std="c++20",
                       target="system", hoist_reciprocals=False):
        """Thin wrapper: delegates to pops.codegen.compile.compile_or_jit."""
        return _cg_compile().compile_or_jit(self, so_path, include=include, mode=mode, name=name, cxx=cxx,
                                  std=std, target=target, hoist_reciprocals=hoist_reciprocals)

    # --- production facade: a single entry point per INTENTION (backend) -----------------
    # Routes the compilation backend by INTENTION rather than by implementation detail. Each
    # entry designates one of the existing engines (compile_so / compile_aot) AND the System adder to use
    # at runtime -- coupled here so that a caller does not wire an AOT .so onto add_dynamic_block (or
    # vice versa), which would load but with an inconsistent ABI/numerics.
    #   "prototype"  -> compile_so  (JIT, IModel, virtual dispatch, host first-order Rusanov; fast
    #                   iteration, to be wired via System.add_dynamic_block);
    #   "aot"        -> compile_aot (AOT, host-marshaled PRODUCTION path: assemble_rhs<Limiter,
    #                   Flux>, HLLC/Roe, second order, SSPRK2/IMEX on a LOCAL grid of the .so; numerics
    #                   identical to native but marshaled arrays, via add_compiled_block);
    #   "production" -> compile_native (NATIVE LOADER): the .so inlines add_compiled_model<ProdModel>, which
    #                   installs the generated model as a NATIVE System block (closures over the REAL
    #                   grid_context). The block runs ZERO-COPY the SAME path as add_block (no
    #                   marshaling); device-clean by construction (named functors from block_builder).
    #                   To be wired via System.add_native_block (ABI key verified). This is the path
    #                   prepared for a real production backend (Kokkos/CUDA codegen = later PR).

    def _model_hash(self, params=None):
        """Stable hash of the model; delegates to pops.codegen.compile.model_hash."""
        return _cg_compile().model_hash(self, params=params)

    def _check_require_metadata(self, require_metadata, backend):
        """require_metadata guard rails (pure-Python, deterministic on the model + backend). Factored out
        to be called BEFORE the cache (in HyperbolicModel AND Model): a cache HIT must never
        mask a metadata requirement. Without require_metadata, no-op."""
        if not require_metadata:
            return
        # backend "prototype" (add_dynamic_block, VIRTUAL dispatch, host first-order Rusanov): NOT a
        # device-clean production path -> requesting metadata on it is inconsistent (clear error).
        if backend == "prototype":
            raise ValueError(
                "compile: backend 'prototype' (JIT, host virtual dispatch) incompatible with "
                "require_metadata=True; use backend='aot' or 'production' for the "
                "device-clean path with guaranteed metadata")
        missing = []
        roles = roles_for(self.cons_names, self.cons_roles)
        if all(r == "Custom" for r in roles):
            missing.append("physical roles (conservative_vars(..., roles=[...]) or canonical names)")
        if self.gamma is None:
            missing.append("gamma (set_gamma(...))")
        if missing:
            raise ValueError(
                "compile(require_metadata=True): model '%s' does not provide %s; the .so "
                "would fall back to the System fallback (roles 'custom' / gamma 1.4)"
                % (self.name, " nor ".join(missing)))

    def compile(self, so_path=None, include=None, backend="auto", name=None, cxx=None, std=None,
                require_metadata=False, target="system", hoist_reciprocals=False):
        """Thin wrapper: delegates to pops.codegen.compile.compile_model."""
        return _cg_compile().compile_model(self, so_path=so_path, include=include, backend=backend,
                                 name=name, cxx=cxx, std=std,
                                 require_metadata=require_metadata, target=target,
                                 hoist_reciprocals=hoist_reciprocals)

    @classmethod
    def adder_for(cls, backend):
        """Name of the System method to use to wire the .so produced by compile(backend=...):
        'add_dynamic_block' (prototype/JIT), 'add_compiled_block' (aot) or 'add_native_block'
        (production/native). Delegates to pops.codegen.compile.adder_for."""
        return _cg_compile().adder_for(backend)

    def emit_cpp_elliptic(self, name=None, namespace="pops_generated", cse=True,
                          hoist_reciprocals=False):
        """Generates a composable elliptic RIGHT-HAND SIDE BRICK from self._elliptic.

        The produced struct exposes rhs(U) -> Real (charge density, background, gravity...), same shape as
        the manual bricks (ChargeDensity, BackgroundDensity in pops/model/bricks.hpp): it enters
        as the Elliptic parameter of a CompositeModel. Inlined constants, cons/primitives -> locals,
        cse=True factors out common sub-expressions. ValueError if set_elliptic_rhs(...) is missing."""
        from pops.codegen import module_codegen as _cg
        return _cg.emit_cpp_elliptic(self, name=name, namespace=namespace, cse=cse,
                                     hoist_reciprocals=hoist_reciprocals)

    def emit_cpp_elliptic_field(self, field, struct_name, namespace="pops_generated",
                                hoist_reciprocals=False, cse=True):
        """Generates a SELF-CONTAINED elliptic RHS brick for the NAMED field @p field (ADC-428).

        Unlike emit_cpp_elliptic (which emits only ``rhs(U)``, consumed by CompositeModel), this brick
        is shaped like a minimal Model so the runtime can pair it with pops::make_poisson_rhs directly:
        it declares ``n_vars`` + ``State`` (so load_state<Brick> reads the conservative state) and
        exposes ``elliptic_rhs(State)`` (what detail::PoissonRhs<Brick> calls per cell). The native
        loader builds one std::function per named field via make_poisson_rhs(Brick{}) and attaches it to
        the block (System::set_block_elliptic_field). The RHS reads ONLY the conservative state (+
        primitives), never the aux (enforced at declaration). Reuses _codegen_exprs / _prim_block so the
        formula lowers IDENTICALLY to the default elliptic brick."""
        from pops.codegen import module_codegen as _cg
        return _cg.emit_cpp_elliptic_field(self, field, struct_name, namespace=namespace,
                                           hoist_reciprocals=hoist_reciprocals, cse=cse)
