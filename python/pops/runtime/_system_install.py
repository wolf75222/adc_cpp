"""System install mixin (Spec-4 PR-F): block/equation/coupling installation.

Holds the densest part of :class:`pops.runtime.system.System`: ``add_block`` /
``add_equation`` (the backend-adder dispatch + explicit-rejection guards), ``set_source_stage``,
``add_background``, ``add_elliptic_model`` and ``add_coupling``. Mixed into ``System`` via
inheritance; methods operate on ``self._s`` (the compiled facade) and ``self._aux_field_index``.
"""

from pops._bootstrap import ModelSpec
from pops.runtime.bricks import (
    Spatial, Explicit, Split, DivEpsGrad, CompositeRhs, ChargeDensitySource,
    Ionization, Collision, ThermalExchange,
)


class _SystemInstall:
    """Block/equation/coupling installation methods of System."""

    def add_block(self, name, model, spatial=None, time=None, evolve=True):
        """Installs an evolved block composed of NATIVE BRICKS on the shared system Poisson.

        Primary entry point for a model composed in Python from native bricks
        (pops.Model(...)). For a compiled DSL model (.so) or an automatic dispatch on the model type,
        use add_equation. The arguments are marshaled to the C++ facade
        (System::add_block), which validates the block (names / roles / implicit mask) against the model.

        @param name unique block name; indexes set_density(name) / mass(name) / density(name).
        @param model an pops.Model(...) (ModelSpec: state + transport + source + elliptic brick).
        @param spatial spatial discretization, an pops.Spatial(...) / pops.FiniteVolume(...) (default
            minmod + rusanov + conservative). Carries the limiter (none / minmod / vanleer / weno5 --
            weno5 is exposed ONLY by this native path), the Riemann flux (rusanov / hll / hllc /
            roe) and the reconstructed variables (conservative / primitive). positivity_floor is read
            here (Zhang-Shu positivity limiter).
        @param time temporal treatment, an pops.Explicit (default) / pops.IMEX / pops.SourceImplicit.
            Carries substeps (sub-steps per macro-step), stride (multirate hold-then-catch-up cadence),
            the implicit mask (implicit_vars / implicit_roles) and the Newton options (IMEX). All
            these parameters are forwarded as-is to the C++.
        @param evolve True (default) = block advances; False = frozen field (background) which still
            contributes to the right-hand side of the system Poisson.
        @throws TypeError if time is an pops.Split / pops.Strang (Schur-condensed source stage),
            not wired here: go through add_equation(..., time=pops.Split(...)).
        """
        spatial = spatial if spatial is not None else Spatial()
        time = time if time is not None else Explicit()
        # pops.Split (condensed source stage) is only wired by add_equation (which plugs
        # set_source_stage after adding the block): reject it HERE rather than running only the transport
        # silently (the condensed source would be lost).
        if isinstance(time, Split):
            raise TypeError(
                "System.add_block: pops.Split (Schur-condensed source stage) is only supported by "
                "add_equation (which plugs the source stage); use add_equation(..., time=pops.Split(...)).")
        # Implicit mask + Newton options carried by the temporal policy (IMEX/SourceImplicit);
        # neutral defaults on the other policies (Explicit). Resolved/validated on the C++ side
        # (System::add_block) against the block's names/roles.
        self._s.add_block(name, model, spatial.limiter, spatial.flux, spatial.recon, time.kind,
                          getattr(time, "substeps", 1), evolve, getattr(time, "stride", 1),
                          getattr(time, "implicit_vars", []), getattr(time, "implicit_roles", []),
                          getattr(time, "newton_max_iters", 2),
                          getattr(time, "newton_rel_tol", 0.0),
                          getattr(time, "newton_abs_tol", 0.0),
                          getattr(time, "newton_fd_eps", 1e-7),
                          getattr(time, "newton_diagnostics", False),
                          getattr(time, "newton_damping", 1.0),
                          getattr(time, "newton_fail_policy", "none"),
                          getattr(spatial, "positivity_floor", 0.0),
                          getattr(spatial, "wave_speed_cache", False))

    def add_equation(self, name, model, spatial=None, time=None, substeps=None, names=None,
                     evolve=True, stride=None):
        """Adds an equation/block by dispatching on the TYPE of @p model (DSL Phase A):

        - a ModelSpec (pops.Model(...)) -> add_block (composed native bricks);
        - a CompiledModel (m.compile(...)) -> the backend adder (add_dynamic_block for prototype,
          add_compiled_block for aot, add_native_block for production), with the names/roles/gamma
          carried by the .so.

        Centralizes the backend <-> adder coupling (an AOT .so must not be plugged into
        add_dynamic_block, and vice versa). cf. docs/DSL_MODEL_DESIGN.md section 3.

        @p spatial : pops.FiniteVolume(...) / pops.Spatial(...) (default minmod+rusanov+conservative).
        @p time : pops.Explicit / IMEX (default Explicit). @p substeps : overrides time.substeps.
        @p stride : overrides time.stride (1 = each macro-step, default bit-identical).
        @p names : component names (length = n_vars of the compiled model). @p evolve : block advances;
        evolve=False (frozen field) is only wired on the native path (ModelSpec -> add_block, backend
        'production' -> add_native_block). On backend 'prototype'/'aot' (the .so ABI does not carry
        evolve) an evolve=False is REJECTED explicitly -> use a native block (add_background).
        """
        # Late imports (the codegen/physics modules import this package: avoid the cycle).
        from pops.codegen.abi import check_compiled_matches_module
        from pops.codegen.loader import CompiledModel
        from pops.physics.aux import AUX_NAMED_BASE

        spatial = spatial if spatial is not None else Spatial()
        time = time if time is not None else Explicit()

        # --- pops.Split (Lie) / pops.Strang (2nd order): EXPLICIT / IMPLICIT splitting, Schur OPT-IN --
        # The block is first added with the explicit HYPERBOLIC stage (existing production path,
        # no dispatch duplication), THEN we plug the condensed SOURCE stage (set_source_stage,
        # C++). The source is run AFTER the transport at each step. The default (without Split) is unchanged.
        # The splitting POLICY (Lie / Strang) is WIRED to the system stepper via set_time_scheme:
        # pops.Split -> "lie" (default, bit-identical), pops.Strang -> "strang" (H(dt/2) S(dt) H(dt/2)).
        if isinstance(time, Split):
            self.add_equation(name, model, spatial=spatial, time=time.hyperbolic,
                              substeps=substeps, names=names, evolve=evolve, stride=stride)
            src = time.source
            self._s.set_source_stage(name, src.kind, src.theta, src.alpha,
                                     getattr(src, "krylov_tol", 0.0),
                                     getattr(src, "krylov_max_iters", 0),
                                     getattr(src, "density_spec", ""),
                                     getattr(src, "momentum_x_spec", ""),
                                     getattr(src, "momentum_y_spec", ""),
                                     getattr(src, "energy_spec", ""),
                                     getattr(src, "bz_aux_component", -1))
            self._s.set_time_scheme(time.scheme)  # "lie" (Split) or "strang" (Strang)
            return

        nsub = substeps if substeps is not None else getattr(time, "substeps", 1)
        nstride = stride if stride is not None else getattr(time, "stride", 1)

        # --- ModelSpec: composed native bricks -> add_block (existing path) ---
        # NB: we call _s.add_block DIRECTLY with nsub/nstride (not self.add_block, whose
        # signature has no substeps -> it would use time.substeps and IGNORE the overrides).
        if isinstance(model, ModelSpec):
            self._s.add_block(name, model, spatial.limiter, spatial.flux, spatial.recon, time.kind,
                              nsub, evolve, nstride,
                              getattr(time, "implicit_vars", []), getattr(time, "implicit_roles", []),
                              getattr(time, "newton_max_iters", 2),
                              getattr(time, "newton_rel_tol", 0.0),
                              getattr(time, "newton_abs_tol", 0.0),
                              getattr(time, "newton_fd_eps", 1e-7),
                              getattr(time, "newton_diagnostics", False),
                          getattr(time, "newton_damping", 1.0),
                          getattr(time, "newton_fail_policy", "none"),
                          getattr(spatial, "positivity_floor", 0.0),
                          getattr(spatial, "wave_speed_cache", False))
            return

        # Implicit mask (IMEX): only the composed native path (ModelSpec -> add_block) wires it. The
        # compiled backends (.so: dynamic/aot/production) do not expose the argument -> we REJECT a
        # non-empty mask rather than ignore it silently (cf. the stride rejection on backend 'aot').
        if getattr(time, "implicit_vars", []) or getattr(time, "implicit_roles", []):
            raise ValueError(
                "add_equation: implicit_vars / implicit_roles (per-block IMEX mask) are only supported "
                "on a composed model pops.Model(...) (-> add_block). The compiled model (.so) does not "
                "carry the mask; use a native pops.Model(...).")
        # Same rules for the Newton options/diagnostics (IMEX): not carried by the .so ABI.
        # Non-default values would be ignored SILENTLY -> explicit rejection.
        if (getattr(time, "newton_max_iters", 2) != 2
                or getattr(time, "newton_rel_tol", 0.0) != 0.0
                or getattr(time, "newton_abs_tol", 0.0) != 0.0
                or getattr(time, "newton_fd_eps", 1e-7) != 1e-7
                or getattr(time, "newton_diagnostics", False)
                or getattr(time, "newton_damping", 1.0) != 1.0
                or getattr(time, "newton_fail_policy", "none") != "none"):
            raise ValueError(
                "add_equation: the Newton options (newton_max_iters/rel_tol/abs_tol/fd_eps/"
                "diagnostics/damping/fail_policy) are only supported on a composed model "
                "pops.Model(...) (-> add_block). The compiled model (.so) ABI does not carry "
                "them; use a native pops.Model(...).")

        if not isinstance(model, CompiledModel):
            raise TypeError("add_equation: model must be an pops.Model(...) (ModelSpec) or a "
                            "CompiledModel (m.compile(...)); got %r" % type(model).__name__)

        compiled = model
        # Names guard: explicit length checked early (the C++ also raises, but we diagnose here).
        if names is not None and len(names) != compiled.n_vars:
            raise ValueError("add_equation: names= has %d names but block '%s' has %d variables"
                             % (len(names), name, compiled.n_vars))
        names_arg = list(names) if names is not None else []

        # NAMED aux fields (ADC-70 phase 1): table name -> block component, from the ORDERED names
        # of the compiled model (the k-th name = component dsl.AUX_NAMED_BASE + k, mirror of the C++ emission).
        # Consumed by set_aux_field / aux_field. add_compiled_block / add_native_block / add_dynamic_block
        # have already widened the aux channel (pops_compiled_naux -> ensure_aux_width), so the component exists.
        extra = list(getattr(compiled, "aux_extra_names", []) or [])
        self._aux_field_index[name] = {nm: AUX_NAMED_BASE + k for k, nm in enumerate(extra)}

        backend = compiled.backend
        # Numerical flux guard: HLLC/Roe require a pressure -> the generated brick emits
        # pressure()/wave_speeds() only if a primitive 'p' is declared. Without 'p', make_block does not
        # compile the flux: we diagnose it here before the C++ boundary.
        # hllc / roe: the emitted capability (m.enable_hllc -> has_hllc, m.enable_roe -> has_roe)
        # OPENS the flux even outside 4-var Euler (the C++ requires-gate accepts it); otherwise the
        # canonical path requires 'p' in the primitives.
        if (spatial.flux in ("hllc", "roe") and "p" not in compiled.prim_names
                and not (spatial.flux == "hllc" and getattr(compiled, "has_hllc", False))
                and not (spatial.flux == "roe" and getattr(compiled, "has_roe", False))):
            raise ValueError(
                "add_equation: riemann '%s' requires a pressure: declare a primitive 'p' "
                "(m.primitive('p', ...)) in the model, or emit the capability "
                "(m.enable_hllc() / m.enable_roe()); otherwise use riemann='rusanov'"
                % spatial.flux)
        # HLL: the generated brick emits wave_speeds either from the EXPLICIT pair
        # m.wave_speeds(x=, y=) (WITHOUT primitive 'p': moments, isothermal..., cf. has_wave_speeds),
        # or as soon as a primitive 'p' is DECLARED (m.primitive('p', ...)), even OUTSIDE the
        # primitive_vars layout (isothermal 3-var Hoffart case: prim_names = rho/u/v without 'p'). EARLY
        # guard here: the C++ requires-gate of make_block only triggers at the FIRST use
        # (eval_rhs / step, lazy construction of the closures) -- we diagnose at
        # installation, like hllc/roe. getattr default True = belt-and-suspenders: CompiledModel
        # ALWAYS sets has_wave_speeds (Model.compile from 'p'/pair; HybridModel from the
        # transport brick, True for a native brick = unknown) -- the default only applies to
        # a foreign object without the flag, which then falls back on the C++ gate (history).
        if spatial.flux == "hll" and not getattr(compiled, "has_wave_speeds", True):
            raise ValueError(
                "add_equation: riemann 'hll' requires signed wave speeds: declare "
                "m.wave_speeds(x=(smin, smax), y=(smin, smax)) (without pressure), or a primitive "
                "'p' (m.primitive('p', ...)); otherwise use riemann='rusanov'")

        # AUTHORITATIVE dispatch by the CompiledModel adder (fixed by the backend, cf. dsl._BACKENDS):
        # prototype -> add_dynamic_block, aot -> add_compiled_block, production -> add_native_block (#85).
        adder = compiled.adder
        if adder == "add_dynamic_block":
            # JIT, HOST Rusanov order-1 residual: takes only the MUSCL LIMITER (none/minmod/vanleer)
            # + substeps; no HLLC/Roe flux, no primitive recon. WENO5 (5-point stencil) is
            # NOT a MUSCL limiter and this path does not run assemble_rhs: we reject it HERE (the
            # aot/production paths, on the other hand, accept weno5 -- the .so grid / native block allocate
            # block_n_ghost(limiter) = 3 ghosts).
            if spatial.limiter == "weno5":
                raise ValueError(
                    "add_equation: limiter 'weno5' not supported on backend 'prototype' (JIT, host "
                    "Rusanov order-1 residual, without assemble_rhs); use backend='aot'/'production' "
                    "(WENO5 wired end to end) or add_block (composed model pops.Model(...)).")
            if spatial.flux != "rusanov":
                raise ValueError(
                    "add_equation: backend 'prototype' (JIT, host Rusanov order-1 residual) only exposes "
                    "riemann='rusanov' (got '%s'); use backend='aot'/'production' for "
                    "HLLC/Roe" % spatial.flux)
            # evolve=False (FROZEN block / fixed background) is NOT wired: the add_dynamic_block ABI does
            # not carry evolve (push_dynamic forces it to true on the C++ side) -> the block would be
            # advanced SILENTLY. We REJECT it (rejection rather than silent ignore). For a frozen field,
            # use a native/production block (add_background -> add_block(..., evolve=False)).
            if not evolve:
                raise ValueError(
                    "add_equation: evolve=False not supported on backend 'prototype' (the JIT .so ABI "
                    "does not carry evolve; the block would be advanced silently). Use a composed "
                    "native model pops.Model(...) -> add_block(..., evolve=False) (or add_background) "
                    "for a frozen field.")
            # positivity_floor (ADC-76) is NOT wired on the host JIT path (no assemble_rhs,
            # dedicated Rusanov order-1 residual): explicit rejection rather than a silently ignored floor.
            if getattr(spatial, "positivity_floor", 0.0) > 0.0:
                raise ValueError(
                    "add_equation: positivity_floor not supported on backend 'prototype' (dedicated "
                    "host residual, without high-order reconstruction); use backend='aot'/'production' "
                    "or a composed model pops.Model(...) -> add_block.")
            # NB wave_speed_cache (ADC-199): no dedicated guard here -- the cache requires riemann='hll',
            # already rejected above on 'prototype' (rusanov order 1 only) -> never silently ignored on
            # this backend.
            self._s.add_dynamic_block(name, compiled.so_path, nsub, names_arg, spatial.limiter)
            return
        if adder == "add_compiled_block":
            # AOT host-marshaled: limiter x riemann x recon, single-rank (without MPI/AMR). The extern "C"
            # ABI of the AOT .so (add_compiled_block) does NOT carry a cadence: the block would run at stride=1
            # SILENTLY. We therefore REJECT stride > 1 on this backend (explicit route) rather than
            # ignore it. The per-block stride is wired on add_block (composed native) and add_native_block
            # (backend='production'). We read time.stride AND the stride= override (nstride covers both).
            if nstride != 1:
                raise ValueError(
                    "add_equation: stride=%d not supported on backend 'aot' (the AOT .so ABI does not "
                    "carry the cadence; the block would run at stride=1 silently). Use "
                    "backend='production' (native path, cadence wired) or a composed native model "
                    "pops.Model(...) -> add_block." % nstride)
            # evolve=False (FROZEN block / fixed background) is NOT wired: the add_compiled_block ABI does
            # not carry evolve (add_compiled_block forces it to true on the C++ side) -> the block would be
            # advanced SILENTLY. We REJECT it (rejection rather than silent ignore). For a frozen field,
            # use backend='production' (add_native_block carries evolve) or a composed native model
            # pops.Model(...) -> add_block(..., evolve=False) (or add_background).
            if not evolve:
                raise ValueError(
                    "add_equation: evolve=False not supported on backend 'aot' (the AOT .so ABI does not "
                    "carry evolve; the block would be advanced silently). Use "
                    "backend='production' (native path, evolve wired) or a composed native model "
                    "pops.Model(...) -> add_block(..., evolve=False) (or add_background) for a frozen field.")
            # wave_speed_cache (ADC-199): the AOT .so ABI does not carry the wave speed cache -> it would
            # be silently ignored. Explicit rejection (the cache is only wired on the composed native
            # add_block).
            if getattr(spatial, "wave_speed_cache", False):
                raise ValueError(
                    "add_equation: wave_speed_cache not supported on backend 'aot' (the AOT .so ABI does "
                    "not carry the HLL wave speed cache; it would be silently ignored). Use a composed "
                    "native model pops.Model(...) -> add_block.")
            self._s.add_compiled_block(name, compiled.so_path, spatial.limiter, spatial.flux,
                                       spatial.recon, time.kind, nsub, names_arg,
                                       getattr(spatial, "positivity_floor", 0.0))
            return
        if adder == "add_native_block":
            # NATIVE zero-copy (#85): block installed on the REAL System CONTEXT (same path as
            # add_block). Takes a gamma, NO names= (the names/roles come from the .so metadata).
            # End-to-end device/MPI validation from Python is a later dedicated PR.
            if names is not None:
                raise ValueError(
                    "add_equation: names= not supported on the native path (production); the names and "
                    "roles are carried by the compiled model metadata (.so)")
            # PRE-DLOPEN guard at plug time: ALSO covers the cache HIT (where compile_native does not
            # run) -- a stale _pops module would otherwise give a cryptic dlopen 'symbol not found'.
            # wave_speed_cache (ADC-199): the add_native_block ABI does not (yet) carry the wave speed
            # cache -> it would be silently ignored. Explicit rejection BEFORE the C++ boundary (and
            # before the ABI check: a clear message rather than a dlopen error). The cache is wired on
            # the composed native add_block path (System.add_block).
            if getattr(spatial, "wave_speed_cache", False):
                raise ValueError(
                    "add_equation: wave_speed_cache not supported on backend 'production' (the "
                    "add_native_block ABI does not carry the HLL wave speed cache; it would be silently "
                    "ignored). Use a composed native model pops.Model(...) -> add_block.")
            check_compiled_matches_module(getattr(compiled, "abi_key", ""))
            gamma = compiled.gamma if compiled.gamma is not None else 1.4
            self._s.add_native_block(name, compiled.so_path, spatial.limiter, spatial.flux,
                                     spatial.recon, time.kind, gamma, nsub, evolve, nstride,
                                     getattr(spatial, "positivity_floor", 0.0))
            return
        raise ValueError("add_equation: adder %r unknown (backend %r)" % (adder, backend))

    def set_source_stage(self, name, kind, theta, alpha,
                         krylov_tol=0.0, krylov_max_iters=0,
                         density="", momentum_x="", momentum_y="", energy="",
                         bz_aux_component=-1):
        """Attach a Schur-condensed source stage to an already-added block (ADC-308).

        Thin public pass-through to the C++ binding (_pops.System.set_source_stage): same flat
        signature and defaults. add_equation(time=pops.Split(source=pops.CondensedSchur(...))) wires
        this internally; this method exposes the same control for a block added with a plain
        transport time scheme, so cases configure the stage without reaching into the private _s.
        @p name: block; @p kind: 'electrostatic_lorentz'; @p theta in (0, 1]; @p alpha: stage
        coupling. The krylov_* / field descriptors / bz_aux_component defaults reproduce the
        historical bit-identical behavior. Prerequisite: B_z set via set_magnetic_field beforehand.
        """
        self._s.set_source_stage(name, kind, theta, alpha, krylov_tol, krylov_max_iters,
                                 density, momentum_x, momentum_y, energy, bz_aux_component)

    def add_background(self, name, model, density, spatial=None):
        """FROZEN species (not advanced): a fixed background that contributes to the system Poisson (and,
        in the future, to coupled sources). density: n*n array. Equivalent to add_block(evolve=False)
        followed by set_density."""
        self.add_block(name, model, spatial=spatial, evolve=False)
        self.set_density(name, density)

    def add_elliptic_model(self, name, model, solver=None, bc="auto", wall="none",
                           wall_radius=0.0):
        """EPM: configures the system elliptic model (Poisson is its current instance).
        model = pops.elliptic(operator=pops.div_eps_grad(eps), rhs=pops.composite_rhs(),
        output=pops.electric_field_from_potential()). set_poisson(...) remains the equivalent shortcut.

        Operator: div(eps grad) with CONSTANT eps (eps != 1 supported: eps lap phi = f); variable
        eps(x) is plugged in via set_epsilon_field. Right-hand side: composite_rhs() = GENERIC sum
        of the elliptic bricks carried by the blocks (charge q n, background alpha (n-n0), gravity
        coupling sign 4piG (rho-rho0)); charge_density() is its usual case. Diffusion / projection (other
        operator) would require a variable-coefficient solver (refinement not available)."""
        if not isinstance(model.operator, DivEpsGrad):
            raise NotImplementedError("add_elliptic_model: only the div_eps_grad operator (Poisson) "
                                      "is supported; diffusion / projection -> refinement (solver)")
        if not isinstance(model.rhs, CompositeRhs):
            raise NotImplementedError("add_elliptic_model: rhs must be composite_rhs() (sum of the "
                                      "per-block bricks) or charge_density() (its usual case)")
        kind = solver.kind if solver is not None else "geometric_mg"
        # Honest token: "composite" for a generic right-hand side, "charge_density" (alias,
        # bit-identical) when all blocks carry a charge density. Both take the
        # SAME numerical path on the C++ side (sum of each block's elliptic bricks).
        rhs_tok = "charge_density" if type(model.rhs) is ChargeDensitySource else "composite"
        self.set_poisson(rhs=rhs_tok, solver=kind, bc=bc, wall=wall, wall_radius=wall_radius,
                         epsilon=model.operator.epsilon)

    def add_coupling(self, coupling):
        """Add an inter-species coupling (operator-split, applied after transport):

        - NAMED object pops.Ionization / Collision / ThermalExchange -> fixed formula
          (add_ionization / add_collision / add_thermal_exchange);
        - CompiledCoupledSource (pops.dsl.CoupledSource(...).compile(...)) -> GENERIC source described in
          formulas, carried as bytecode and interpreted on the C++ side (System.add_coupled_source; no
          per-cell Python callback, MPI-safe)."""
        # Late import (the multispecies module imports this package: avoid the cycle).
        from pops.physics.multispecies import CompiledCoupledSource

        if isinstance(coupling, CompiledCoupledSource):
            self._s.add_coupled_source(coupling.in_blocks, coupling.in_roles, coupling.consts,
                                       coupling.out_blocks, coupling.out_roles, coupling.prog_ops,
                                       coupling.prog_args, coupling.prog_lens,
                                       getattr(coupling, "frequency", 0.0), coupling.name,
                                       # PER-CELL frequency mu(U) (empty = constant only, cf.
                                       # CoupledSource.frequency(Expr)). Forwarded to the C++ boundary.
                                       getattr(coupling, "freq_prog_ops", []),
                                       getattr(coupling, "freq_prog_args", []))
        elif isinstance(coupling, Ionization):
            self.add_ionization(electron=coupling.electron, ion=coupling.ion,
                                neutral=coupling.neutral, rate=coupling.rate)
        elif isinstance(coupling, Collision):
            self.add_collision(coupling.a, coupling.b, coupling.rate)
        elif isinstance(coupling, ThermalExchange):
            self.add_thermal_exchange(coupling.a, coupling.b, coupling.rate)
        else:
            raise TypeError("add_coupling expects pops.Ionization / Collision / ThermalExchange or a "
                            "CompiledCoupledSource (pops.dsl.CoupledSource(...).compile(...))")
