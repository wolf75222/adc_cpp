"""AmrSystem equation/aux mixin (Spec-4 PR-F).

``add_equation`` (the AMR backend dispatcher) + the named-aux resolution / set of
:class:`pops.runtime.amr_system.AmrSystem`, plus the module-level guard
``_reject_newton_amr_compiled`` used only by this path. Mixed in via inheritance; operates on
``self._s`` and ``self._aux_field_index``.
"""

from pops._bootstrap import ModelSpec
from pops.runtime.bricks import Spatial, Explicit, Split


def _reject_newton_amr_compiled(label, time):
    """REJECTS Newton options/diagnostics on the COMPILED AMR path (.so loader, flat ABI
    add_native_block / pops_install_native_amr) -- wave 3, settle. On the NATIVE side (pops.Model(...)), the
    Newton OPTIONS are now wired in single-block (coupler) AND multi-block (engine), and the
    newton_diagnostics REPORT in native multi-block ; but the flat ABI of the .so loader transports NEITHER
    the options (newton_max_iters/rel_tol/abs_tol/fd_eps/damping/fail_policy) NOR the report. Passed
    via the loader, they would be taken at their defaults SILENTLY (iters=2, no report). We
    REJECT them explicitly (same spirit as the stride/mask rejection of the AMR production path). For these
    parameters : AmrSystem.add_block (native model) or add_compiled_model(AmrSystem&) directly (C++)."""
    if (getattr(time, "newton_max_iters", 2) != 2
            or getattr(time, "newton_rel_tol", 0.0) != 0.0
            or getattr(time, "newton_abs_tol", 0.0) != 0.0
            or getattr(time, "newton_fd_eps", 1e-7) != 1e-7
            or getattr(time, "newton_damping", 1.0) != 1.0
            or getattr(time, "newton_fail_policy", "none") != "none"
            or getattr(time, "newton_diagnostics", False)):
        raise ValueError(
            "%s : the Newton options/diagnostics (newton_max_iters/rel_tol/abs_tol/fd_eps/damping/"
            "fail_policy/diagnostics) are not transported by the AMR production path (loader "
            ".so, flat ABI add_native_block : they would be taken at their defaults silently). "
            "Use AmrSystem.add_block (native model pops.Model(...)) or add_compiled_model("
            "AmrSystem&) directly (C++)." % label)


class _AmrSystemEquation:
    """add_equation + named-aux methods of AmrSystem."""

    def add_equation(self, name, model, spatial=None, time=None, substeps=None):
        """Add the SINGLE AMR equation/block by dispatching on the TYPE of @p model (DSL Phase D):

        - a ModelSpec (pops.Model(...)) -> add_block (native bricks composed on the hierarchy);
        - a CompiledModel(backend='production', target='amr_system') (m.compile(...)) -> NATIVE path
          add_native_block: the .so loader inlines add_compiled_model(AmrSystem&), so the block runs
          the SAME AMR hierarchy as add_block (conservative reflux, regrid), ZERO-COPY.

        Time handling is wired to {explicit, imex}: imex treats the stiff source IMPLICITLY
        (backward_euler_source), the remaining transport explicit and carried by the conservative reflux
        (parity with the System IMEX; the source being cell-local, the implicit split does not touch
        conservation at the coarse-fine interfaces). recon "primitive" and flux "roe"/"hllc"/weno5 are
        WIRED on AMR (parity with add_block; cf. dispatch_amr_compiled). limiter="weno5" (WENO5-Z,
        3 ghosts): the coupler allocates its levels to Limiter::n_ghost and the regrid inherits n_grow(), so
        the 5-point stencil does not read out of bounds. cf. DSL_MODEL_DESIGN.md Phase D (point 10).

        MULTIRATE CADENCE (stride) and PARTIAL IMEX MASK (implicit_vars / implicit_roles):

        - ModelSpec path (pops.Model(...)): FORWARDED to AmrSystem::add_block, which SUPPORTS and
          validates them (parity with the add_block wrapper);
        - CompiledModel production path (.so): explicitly REJECTED (ValueError). The flat ABI of the
          loader (add_native_block / pops_install_native_amr) does not transport them; they would be taken
          at their defaults SILENTLY (stride=1, full backward-Euler). For a multirate .so or one with a
          partial IMEX mask, use AmrSystem.add_block (native) or add_compiled_model(AmrSystem&) directly
          (C++), which expose stride and the mask.

        @p spatial: pops.FiniteVolume(...) / pops.Spatial(...) (default minmod+rusanov+conservative).
        @p time: pops.Explicit (default) or pops.IMEX (implicit stiff source). @p substeps: overrides
        time.substeps.
        """
        # Late imports (the codegen/physics modules import this package: avoid the cycle).
        from pops.codegen.loader import CompiledModel
        from pops.physics.aux import AUX_NAMED_BASE

        spatial = spatial if spatial is not None else Spatial()
        time = time if time is not None else Explicit()

        # positivity_floor (ADC-259) IS wired on the NATIVE AMR transport (Density-role face states +
        # C/F fine ghost means). It is threaded below on the ModelSpec (native) branch and on the
        # amr-schur transport (the recursive add_equation on time.hyperbolic). The COMPILED .so path
        # carries it too now (ADC-322): the regenerated loader marshals it (pops_install_native_amr),
        # so the CompiledModel branch below forwards it to add_native_block instead of rejecting it.

        # --- pops.Split (Lie) / pops.Strang (2nd order): amr-schur PATH (GLOBAL condensed source stage) --
        # During AMR of System.add_equation (cf. ~line 925): we first add the block with its single
        # explicit HYPERBOLIC stage (SOURCE-FREE transport; the model must carry a NoSource source
        # brick), THEN we attach the condensed SOURCE stage (set_source_stage, C++) and the splitting
        # policy (set_time_scheme: "lie" for Split, "strang" for Strang). The condensed stage is
        # GLOBAL (assembles/solves the electrostatic/Lorentz operator on the coarse grid, composing
        # the uniform stage), as opposed to the LOCAL cell-by-cell IMEX source of time=pops.IMEX.
        # PREREQUISITE: call sim.set_magnetic_field(B_z) BEFORE the 1st step (the Lorentz term reads
        # Omega = B_z); otherwise a clear error at build. MONO-BLOCK only (set_source_stage raises otherwise).
        if isinstance(time, Split):
            self.add_equation(name, model, spatial=spatial, time=time.hyperbolic, substeps=substeps)
            src = time.source
            # Settings TRANSPORTED by the amr-schur path since wave 3 (System parity):
            # coarse-solve Krylov tolerances + field descriptors (stable role or variable name,
            # resolved at build against the concrete Model). magnetic_field stays pinned to
            # the dedicated coarse B_z buffer (amr_write_coarse_bz): another aux field has no
            # AMR counterpart -> explicit rejection (no silent ignore).
            if getattr(src, "bz_aux_component", -1) >= 0:
                raise ValueError(
                    "AmrSystem.add_equation: magnetic_field != 'B_z' is not transported by the "
                    "amr-schur path (the AMR stage reads the dedicated coarse B_z buffer). Keep "
                    "magnetic_field='B_z', or use System (mono-level).")
            self._s.set_source_stage(name, src.kind, src.theta, src.alpha,
                                     getattr(src, "krylov_tol", 0.0),
                                     getattr(src, "krylov_max_iters", 0),
                                     getattr(src, "density_spec", ""),
                                     getattr(src, "momentum_x_spec", ""),
                                     getattr(src, "momentum_y_spec", ""),
                                     getattr(src, "energy_spec", ""))
            self._s.set_time_scheme(time.scheme)  # "lie" (Split) or "strang" (Strang)
            return

        nsub = substeps if substeps is not None else getattr(time, "substeps", 1)

        # --- ModelSpec: native bricks composed -> add_block (existing path) ---
        # We FORWARD stride (multirate, capstone iv) AND the partial IMEX mask implicit_vars /
        # implicit_roles (capstone vii), exactly like the AmrSystem.add_block wrapper above:
        # the C++ AmrSystem::add_block SUPPORTS and validates them (empty -> full backward-Euler; a
        # mask requested in explicit or in mono-block raises a clear error on the C++ side,
        # amr_system.cpp:325-328 / :283-287). Do NOT duplicate these guards here.
        if isinstance(model, ModelSpec):
            # NATIVE model: Newton OPTIONS wired (mono + multi) + newton_diagnostics (native multi-block,
            # rejected at C++ build in mono-block). No facade filtering: C++ AmrSystem::add_block validates.
            self._s.add_block(name, model, spatial.limiter, spatial.flux, spatial.recon, time.kind,
                              nsub, getattr(time, "stride", 1),
                              getattr(time, "implicit_vars", []), getattr(time, "implicit_roles", []),
                              getattr(time, "newton_max_iters", 2),
                              getattr(time, "newton_rel_tol", 0.0),
                              getattr(time, "newton_abs_tol", 0.0),
                              getattr(time, "newton_fd_eps", 1e-7),
                              getattr(time, "newton_damping", 1.0),
                              getattr(time, "newton_fail_policy", "none"),
                              getattr(time, "newton_diagnostics", False),
                              getattr(spatial, "positivity_floor", 0.0))  # Zhang-Shu floor (ADC-259)
            return

        if not isinstance(model, CompiledModel):
            raise TypeError("AmrSystem.add_equation: model must be an pops.Model(...) (ModelSpec) "
                            "or a CompiledModel (m.compile(...)); received %r" % type(model).__name__)

        compiled = model
        # Only the NATIVE "production" path targets AmrSystem: it inlines add_compiled_model(AmrSystem&).
        # The prototype (JIT) / aot .so have no AMR counterpart (add_dynamic_block/add_compiled_block
        # are mono-level). We therefore require backend='production' + target='amr_system'.
        if compiled.adder != "add_native_block":
            raise ValueError(
                "AmrSystem.add_equation: only a CompiledModel backend='production' (native path) "
                "is attachable on AMR; received backend=%r (the prototype/aot .so are mono-level, "
                "without AMR counterpart)" % compiled.backend)
        if getattr(compiled, "target", "system") != "amr_system":
            raise ValueError(
                "AmrSystem.add_equation: the CompiledModel was compiled for target='system'; "
                "recompile with m.compile(..., backend='production', target='amr_system') so that "
                "the loader inlines add_compiled_model(AmrSystem&) (symbol pops_install_native_amr)")

        # recon "primitive" and flux "roe"/"hllc" are WIRED on AMR via dispatch_amr_compiled: the
        # .so path passes recon_prim to AmrBuildParams (consumed by advance_amr/compute_face_fluxes)
        # and hllc/roe are instantiated under the SAME requires guard as System::make_block (compressible
        # transport with 4 variables + pressure). No more facade rejection (strict parity with
        # add_block, cf. test_amr_riemann_native).
        # Numerical flux guard (gate of System.add_equation): HLLC/Roe require a pressure; the
        # generated brick only emits pressure()/wave_speeds() if a primitive 'p' is declared. Without
        # 'p', dispatch_amr_compiled falls back on the else branch (requires not satisfied) and raises a
        # generic C++ error: we diagnose HERE, clearly, before the C++ boundary.
        if (spatial.flux in ("roe", "hllc") and "p" not in compiled.prim_names
                and not (spatial.flux == "hllc" and getattr(compiled, "has_hllc", False))
                and not (spatial.flux == "roe" and getattr(compiled, "has_roe", False))):
            raise ValueError(
                "AmrSystem.add_equation: riemann '%s' requires a pressure: declare a primitive 'p' "
                "(m.primitive('p', ...)) in the model, or emit the capability "
                "(m.enable_hllc() / m.enable_roe()); otherwise use riemann='rusanov'"
                % spatial.flux)
        # HLL: same early guard as System.add_equation (wave_speeds emitted by the explicit pair
        # m.wave_speeds(x=, y=) OR primitive 'p'; the C++ gate only triggers at first use).
        if spatial.flux == "hll" and not getattr(compiled, "has_wave_speeds", True):
            raise ValueError(
                "AmrSystem.add_equation: riemann 'hll' requires signed wave speeds: declare "
                "m.wave_speeds(x=(smin, smax), y=(smin, smax)) (without pressure), or a primitive "
                "'p' (m.primitive('p', ...)); otherwise use riemann='rusanov'")

        # The flat ABI of the .so loader (pops_install_native_amr / add_native_block) transports NEITHER the
        # multirate cadence (stride) NOR the partial IMEX mask (implicit_vars / implicit_roles):
        # add_compiled_model(AmrSystem&) exposes them only DIRECTLY (C++ path). Passed through the
        # loader, they would take their defaults (stride=1, empty mask = full backward-Euler) SILENTLY.
        # We REJECT them rather than ignore them (explicit route, same spirit as the rejection
        # of stride/mask on the compiled backends of System.add_equation, cf. ~lines 886-955).
        nstride = getattr(time, "stride", 1)
        if nstride != 1:
            raise ValueError(
                "AmrSystem.add_equation: stride=%d not transported by the production AMR path "
                "(.so loader, flat ABI add_native_block: the block would run at stride=1 silently). "
                "Use AmrSystem.add_block (native model pops.Model(...), wired cadence) or "
                "add_compiled_model(AmrSystem&) directly (C++) which exposes stride." % nstride)
        if getattr(time, "implicit_vars", []) or getattr(time, "implicit_roles", []):
            raise ValueError(
                "AmrSystem.add_equation: implicit_vars / implicit_roles (partial IMEX mask) not "
                "transported by the production AMR path (.so loader, flat ABI add_native_block: the "
                "mask would be empty = full backward-Euler silently). Use AmrSystem.add_block "
                "(native model pops.Model(...), wired mask) or add_compiled_model(AmrSystem&) "
                "directly (C++) which exposes the IMEX mask.")
        # Newton options / diagnostics: same flat ABI -> neither the options nor the report transit
        # through the .so loader. Explicit rejection (otherwise iters=2 / no report silently), parity with
        # the stride/mask rejection above and with System.add_equation (compiled backend).
        _reject_newton_amr_compiled("AmrSystem.add_equation", time)
        # positivity_floor (ADC-322): the regenerated .so loader carries the Zhang-Shu floor now
        # (pops_install_native_amr -> add_compiled_model -> set_compiled_block), so it is threaded
        # through instead of rejected. 0 (default) = inactive, bit-identical. The C++
        # add_native_block validates floor >= 0 and finite (parity with add_block).

        # PRE-DLOPEN guard at attach (covers the cache HIT, cf. System.add_equation): module
        # _pops stale vs .so compiled against the up-to-date headers -> actionable error, not a dlopen
        # 'symbol not found' cryptic message.
        from pops.codegen.abi import check_compiled_matches_module
        check_compiled_matches_module(getattr(compiled, "abi_key", ""))
        gamma = compiled.gamma if compiled.gamma is not None else 1.4
        self._s.add_native_block(name, compiled.so_path, spatial.limiter, spatial.flux,
                                 spatial.recon, time.kind, gamma, nsub,
                                 getattr(spatial, "positivity_floor", 0.0))
        # ADC-291: record the named aux fields the block declares (component of the k-th name =
        # AUX_NAMED_BASE + k), so set_aux_field(block, name, array) can resolve name -> component.
        extra = list(getattr(compiled, "aux_extra_names", []) or [])
        if extra:
            self._aux_field_index[name] = {nm: AUX_NAMED_BASE + k for k, nm in enumerate(extra)}

    def _resolve_aux_field(self, block, name):
        """Resolve (block, named aux field) -> aux channel component (ADC-291). Mirror of
        System._resolve_aux_field: a canonical name is redirected to its dedicated path; an unknown
        block or an undeclared field raises (no silent component-0 fallback)."""
        from pops.physics.aux import AUX_CANONICAL
        if name in AUX_CANONICAL:
            if name == "B_z":
                raise ValueError(
                    "set_aux_field: 'B_z' (magnetic field) is set via sim.set_magnetic_field(Bz), "
                    "NOT via set_aux_field (B_z is a canonical aux field, not a named field).")
            raise ValueError(
                "set_aux_field: '%s' is a CANONICAL aux field (derived by the solver, not settable); "
                "set_aux_field only carries the NAMED fields declared by m.aux_field(...)." % name)
        table = self._aux_field_index.get(block)
        if table is None:
            raise ValueError(
                "set_aux_field: block '%s' unknown (or added without a named aux field); add the block "
                "via add_equation(model=...) with a model declaring m.aux_field('%s')." % (block, name))
        if name not in table:
            raise ValueError(
                "set_aux_field: aux field '%s' not declared by block '%s'; known named fields: %s"
                % (name, block, sorted(table)))
        return table[name]

    def set_aux_field(self, block, name, field, halo=None):
        """Set a model-NAMED aux field of @p block (declared via m.aux_field(name)) on the AMR
        hierarchy. AMR counterpart of System.set_aux_field. @p field: 2D array (n, n) on the COARSE
        base level; it is STATIC (re-applied each step, injected to the fine levels, survives a regrid).
        Call BEFORE the first step (like set_density). Mono-rank facade.

        @p halo (ADC-369): an optional pops.AuxHalo declaring this field's own coarse-level ghost
        boundary policy (foextrap / dirichlet), applied to the non-periodic faces after the shared aux
        fill. Default None inherits the shared aux BC (bit-identical)."""
        import numpy as np
        comp = self._resolve_aux_field(block, name)
        arr = np.asarray(field, dtype=float)
        self._s.set_aux_field_component(comp, arr.reshape(-1))
        if halo is not None:
            self._s.set_aux_field_halo_component(comp, halo.bc_type, halo.value)
