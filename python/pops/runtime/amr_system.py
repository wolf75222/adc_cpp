"""AmrSystem : the refined runtime coupler (Spec-4 PR-F composed class).

``AmrSystem`` carries one or several blocks on an AMR hierarchy. Its ~590 lines are split into
the ``_amr_system_equation`` (add_equation + named-aux) and ``_amr_system_io`` (write /
checkpoint / restart) mixins to satisfy the <=500-line cap ; this module composes them and keeps
the constructor + the native-add_block / coupling / diagnostics glue.
"""

from pops._bootstrap import AmrSystemConfig, _AmrSystem
from pops.runtime import threading as _threading
from pops.runtime.bricks import Spatial, Explicit, Split
from pops.runtime._amr_system_equation import _AmrSystemEquation
from pops.runtime._amr_system_io import _AmrSystemIO
from pops.runtime._system_unified_install import validate_install_arguments


class AmrSystem(_AmrSystemEquation, _AmrSystemIO):
    """Refined counterpart of System : one or SEVERAL blocks carried on an AMR hierarchy.

    SINGLE-BLOCK (1 add_block) : historical AmrCouplerMP path (dynamic regrid, reflux). MULTI-BLOCK
    (>= 2 add_block) : N blocks co-located on ONE SHARED AMR hierarchy (AmrRuntime engine),
    SYSTEM Poisson with co-located SUMMED right-hand side (Sum_b q_b n_b), conservation PER BLOCK. The
    blocks may have DIFFERENT SPATIAL SCHEMES, a per-block TEMPORAL TREATMENT (explicit /
    imex), MULTIRATE (substeps / stride), COUPLED inter-species SOURCES and the multi-block production
    DSL. In multi-block the block NAME indexes set_density(name) / mass(name) / density(name).

    UNION-OF-TAGS REGRID (multi-block + regrid_every > 0) : the shared hierarchy is re-gridded from
    the UNION of the tags of all blocks. Two criteria compose (cell-by-cell OR) :

    - PER-BLOCK VARIABLE (set_refinement(threshold, variable=, role=)) : refine where the SELECTED
      variable of a block exceeds threshold. Default = component 0 (historical density), bit-identical ;
      ADC-296 lets you select it per block by name (variable=) or physical role (role=), resolved against
      the block's conserved variables (a block lacking the name/role raises, no silent component-0
      fallback). Non-default selector is multi-block only (mono-block / compiled .so : component 0 only) ;
    - ``grad phi`` (set_phi_refinement(grad_threshold)) : refine where the norm of the gradient of the
      electrostatic potential exceeds grad_threshold (diocotron ring edge). Disabled by default
      (grad_threshold <= 0). MULTI-BLOCK only.

    regrid_every == 0 -> FROZEN hierarchy (regrid never called, bit-identical).
    """

    def __init__(self, config=None, **cfg_kw):
        if config is None:
            config = AmrSystemConfig()
            for k, v in cfg_kw.items():
                setattr(config, k, v)
        # cf. System.__init__ : _AmrSystem(config) triggers the Kokkos init (lazy). set_threads
        # has no more effect after this point.
        _threading._first_system_built = True
        self._s = _AmrSystem(config)
        self._L = float(config.L)  # side of [0, L]^2 (for patch_rectangles : index -> physical)
        # Regrid cadence (checkpoint/restart ADC-65) : a BIT-IDENTICAL resume requires regrid_every == 0
        # (otherwise the post-restart regrid would re-diverge the hierarchy). Memorized for the restart guard.
        self._regrid_every = int(config.regrid_every)
        # ADC-291: block name -> {aux field name -> channel component}, filled by add_equation from a
        # CompiledModel.aux_extra_names (component of the k-th name = AUX_NAMED_BASE + k). Drives
        # set_aux_field(block, name, array). Empty for blocks without a named aux field. Mirror of
        # System._aux_field_index.
        self._aux_field_index = {}

    def patch_rectangles(self):
        """Physical rectangles (x0, y0, width, height) of the current fine patches, in [0, L]^2.

        Converts patch_boxes() (index space, inclusive corners) into physical coordinates. The level
        spacing is dx = L / (n << level) (ratio 2 per level) ; a patch [ilo..ihi] x [jlo..jhi]
        covers (ihi - ilo + 1) cells in x from x0 = ilo * dx (and likewise in y). Grid convention
        ne[j, i] -> index 0 = x (i), index 1 = y (j), consistent with density() and an imshow
        with extent [0, L, 0, L]. Convenient to plot the REAL patches (e.g. matplotlib Rectangle) without
        rebuilding a density proxy. Returns a list of (x0, y0, w, h), one per fine patch (all
        fine levels combined). Query (between steps) : triggers the lazy build like
        n_patches(), no cost on the hot path.
        """
        n, L = self._s.nx(), self._L
        rects = []
        for level, ilo, jlo, ihi, jhi in self._s.patch_boxes():
            dx = L / (n << level)
            rects.append((ilo * dx, jlo * dx, (ihi - ilo + 1) * dx, (jhi - jlo + 1) * dx))
        return rects

    def coarse_local_boxes(self):
        """Number of coarse (base) boxes owned by this MPI rank (ADC-319 diagnostic).

        The base level is a MultiFab whose boxes are spread across ranks by a DistributionMapping.
        Returns this rank's owned-fab count (level-0 local_size()). With distribute_coarse=True the base
        is split into several boxes round-robin, so each rank owns a strict subset and the coarse
        transport is distributed; a replicated or single-box base owns the full count on every rank.
        Compare with coarse_total_boxes() and pops.n_ranks() to confirm MPI strong-scaling of the base.
        Triggers the lazy build like n_patches().
        """
        return self._s.coarse_local_boxes()

    def coarse_total_boxes(self):
        """Total number of coarse (base) boxes across all ranks (ADC-319 diagnostic).

        Identical on every rank (BoxArray size, no communication). With distribute_coarse=True this is
        the number of round-robin base tiles; with a single-box or replicated base it is 1. A rank
        distributes the coarse transport when coarse_local_boxes() < coarse_total_boxes().
        Triggers the lazy build like n_patches().
        """
        return self._s.coarse_total_boxes()

    def add_block(self, name, model, spatial=None, time=None):
        """Installs an evolved block composed of NATIVE BRICKS on the shared AMR hierarchy.

        Refined counterpart of System.add_block. The 1st add_block opens the single-block path
        (AmrCouplerMP : dynamic regrid, reflux) ; each subsequent add_block co-locates one more block
        on THE SAME hierarchy (AmrRuntime engine, system Poisson with summed right-hand side).
        In multi-block the name indexes set_density(name) / mass(name) / density(name). The arguments
        are marshaled to the C++ facade (AmrSystem::add_block), which validates the block against the model.
        For a compiled DSL model (.so) or a dispatch on the model type, use add_equation.

        @param name unique name of the block.
        @param model an pops.Model(...) (ModelSpec : composed native bricks).
        @param spatial spatial discretization, an pops.Spatial(...) / pops.FiniteVolume(...) (default
            minmod + rusanov + conservative). Limiter (none / minmod / vanleer / weno5 ; weno5 = 3
            ghosts, the coupler allocates its levels at Limiter::n_ghost and the regrid inherits n_grow()),
            Riemann flux (rusanov / hll / hllc / roe) and reconstructed variables
            (conservative / primitive).
        @param time temporal treatment, an pops.Explicit (default) / pops.IMEX / pops.SourceImplicit.
            Carries substeps, stride (multirate hold-then-catch-up), the implicit mask (implicit_vars
            / implicit_roles) and the Newton options, threaded to the C++. newton_diagnostics is
            wired in native multi-block and rejected at the C++ build in single-block (the coupler does not
            aggregate a report).
        @throws TypeError if time is an pops.Split / pops.Strang (Schur-condensed source stage) :
            go through add_equation(..., time=pops.Strang(...)) (amr-schur path).
        spatial.positivity_floor > 0 (ADC-259) floors the Density-role face states AND the
        coarse-fine fine ghost means to >= floor on the AMR transport (Zhang-Shu, parity with the
        uniform System). Guarantee = face / ghost-state Density positivity only (order-1 fallback),
        NOT updated-mean nor pressure positivity. A model without a Density role rejects it at the
        first step. The COMPILED .so path carries it too now (ADC-322): a loader regenerated against
        the current headers marshals the floor (add_equation on a CompiledModel, add_native_block).
        """
        spatial = spatial if spatial is not None else Spatial()
        time = time if time is not None else Explicit()
        # pops.Split / pops.Strang (Schur-condensed source stage) is only wired by add_equation (which
        # connects set_source_stage + set_time_scheme AFTER adding the block) : we reject it HERE rather
        # than playing only the transport and SILENTLY LOSING the source (same guard as System.add_block).
        if isinstance(time, Split):
            raise TypeError(
                "AmrSystem.add_block : pops.Split / pops.Strang (Schur-condensed source stage) is "
                "supported only by add_equation (which connects the source stage) ; use "
                "add_equation(..., time=pops.Strang(hyperbolic=pops.Explicit(...), "
                "source=pops.CondensedSchur(...))).")
        # positivity_floor (ADC-259) IS now wired on the AMR transport (Density-role face states +
        # C/F fine ghost means). Threaded to AmrSystem::add_block below; the compiled .so path carries
        # it too (ADC-322, regenerated loader). The C++ side rejects it on a model without a Density role.
        # wave_speed_cache (ADC-199) is NOT wired on the AMR path (AmrSystem::add_block does not
        # transport it) : explicit rejection rather than a silently ignored cache.
        if getattr(spatial, "wave_speed_cache", False):
            raise ValueError(
                "AmrSystem.add_block : wave_speed_cache not supported on the AMR path (separate "
                "work item) ; remove wave_speed_cache or use the uniform System.")
        # We thread substeps/stride (multirate, capstone iv), the partial IMEX mask, the Newton OPTIONS
        # AND newton_diagnostics (wave 3, settle). Resolved / validated on the C++ side (AmrSystem::add_block)
        # against the block names/roles : empty -> full backward-Euler. The options are wired in single-block
        # (coupler) AND multi-block ; newton_diagnostics is wired in native MULTI-BLOCK and REJECTED at the
        # C++ build in single-block (the coupler does not aggregate a report) -- no facade-side filtering here
        # (the facade does not yet know the total number of blocks : the single/multi decision is at build).
        self._s.add_block(name, model, spatial.limiter, spatial.flux, spatial.recon, time.kind,
                          getattr(time, "substeps", 1), getattr(time, "stride", 1),
                          getattr(time, "implicit_vars", []), getattr(time, "implicit_roles", []),
                          getattr(time, "newton_max_iters", 2),
                          getattr(time, "newton_rel_tol", 0.0),
                          getattr(time, "newton_abs_tol", 0.0),
                          getattr(time, "newton_fd_eps", 1e-7),
                          getattr(time, "newton_damping", 1.0),
                          getattr(time, "newton_fail_policy", "none"),
                          getattr(time, "newton_diagnostics", False),
                          getattr(spatial, "positivity_floor", 0.0))

    def install(self, compiled=None, *, instances=None, params=None, aux=None, solvers=None,
                cadence=None):
        """Unified install on the AMR hierarchy (Spec 5 sec.10) -- signature parity with
        ``System.install``.

        Runs the SAME early bind-input validation (``validate_install_arguments``: reject -- BEFORE
        any native mutation -- an install missing a REQUIRED argument the artifact declares, with one
        clear actionable error), then lowers to the AMR layer:

          - NATIVE install (``compiled=None``): wires each instance with ``add_equation`` (native
            bricks / a CompiledModel target='amr_system'), sets the field solvers (``set_poisson``),
            the aux inputs (``set_magnetic_field`` / ``set_aux_field``) and each instance's initial
            density (``set_density``). This is the real AMR add path; a full run is Kokkos-gated.
          - COMPILED install (a ``compiled`` handle carrying a time Program): the early validation
            runs, then a clear ``NotImplementedError`` -- the AMR runtime has no ``install_program``
            seam (a compiled whole-system time Program is a single-level System concept today). Use
            the NATIVE AMR path (``compiled=None`` with a ``target='amr_system'`` CompiledModel per
            instance), or ``System`` for a compiled time Program.

        @param compiled a compiled time-Program handle, or ``None`` for a native AMR install.
        @param instances dict {name: {"initial": array, "spatial": <brick>, "model": <model>,
            "time": <policy>}}; the block is bound by the dict KEY.
        @param params runtime parameters -- NOT wired on AMR (no ``set_block_params``); a non-empty
            ``params=`` raises ``NotImplementedError`` rather than dropping them silently.
        @param aux dict {field_name: array}: "B_z" -> set_magnetic_field, "T_e" rejected (derived),
            any other -> set_aux_field on the declaring block.
        @param solvers dict {field: <solver>}: lowered to set_poisson (default Poisson field only).
        @param cadence NOT wired on AMR (no ``set_program_cadence``); a non-None value raises.
        """
        instances = instances or {}
        params = params or {}
        aux = aux or {}
        solvers = solvers or {}

        # (0) EARLY VALIDATION (shared with System.install): reject a compiled install missing a
        # required declared argument BEFORE any native mutation. Inert (reads arguments() metadata).
        validate_install_arguments(self, compiled, instances, params, aux, solvers)

        if compiled is not None:
            raise NotImplementedError(
                "AmrSystem.install: a COMPILED time Program is not installable on the AMR runtime "
                "(no install_program seam: a compiled whole-system time Program is a single-level "
                "System concept today). Use the NATIVE AMR path (compiled=None with a "
                "target='amr_system' CompiledModel per instance), or System for a compiled Program. "
                "The early bind-input validation above still ran.")
        if params:
            raise NotImplementedError(
                "AmrSystem.install: runtime params (params=%s) are not wired on AMR (no "
                "set_block_params); set them on the native model, or use System." % sorted(params))
        if cadence is not None:
            raise NotImplementedError(
                "AmrSystem.install: a program cadence is not wired on AMR (no set_program_cadence); "
                "set substeps / stride on the native time policy instead.")

        # (1) FIELD SOLVERS first (parity with System: set_poisson before adding blocks).
        for field, solver_brick in solvers.items():
            self._install_solver(field, solver_brick)

        # (2) INSTANCES: add each named block (add_equation), then set its initial density.
        for name, spec in instances.items():
            if not isinstance(spec, dict):
                raise TypeError("AmrSystem.install: instances[%r] must be a dict "
                                "(initial/spatial/time/model); got %r"
                                % (name, type(spec).__name__))
            model = spec.get("model")
            if model is None:
                raise ValueError(
                    "AmrSystem.install: instance %r has no block model -- supply "
                    "instances[%r]['model'] (an pops.Model(...) / a target='amr_system' "
                    "CompiledModel). A native AMR install carries no compiled handle to fall back "
                    "on." % (name, name))
            spatial = spec.get("spatial")
            time = spec.get("time")
            self.add_equation(name, model, spatial=spatial, time=time)

        # (3) AUX fields: B_z -> set_magnetic_field; named -> set_aux_field. After the blocks exist
        # (a named aux resolves against the block's declared aux table).
        for field_name, field in aux.items():
            self._install_aux(field_name, field)

        # (4) INITIAL state per instance (set_density on the AMR coarse base level).
        for name, spec in instances.items():
            initial = spec.get("initial")
            if initial is not None:
                self.set_density(name, initial)

    def _install_solver(self, field, solver_brick):
        """Lower a field-solver selection to set_poisson (AMR). Only the default Poisson field is
        wired; a second named elliptic field is deferred (NotImplementedError). Mirror of
        System._install_solver, minus the System-only solver options the AMR set_poisson lacks."""
        if field not in ("phi", "poisson", "charge_density", "default"):
            raise NotImplementedError(
                "AmrSystem.install: a second named elliptic field (%r) is not wired; only the "
                "default Poisson field ('phi') is supported." % (field,))
        token = solver_brick if isinstance(solver_brick, str) else (
            getattr(solver_brick, "scheme", None) or getattr(solver_brick, "name", None))
        if token is None:
            raise TypeError("AmrSystem.install: solver must be a token string or an "
                            "pops.lib.fields.<Solver>(...) descriptor; got %r"
                            % type(solver_brick).__name__)
        self.set_poisson(solver=token)

    def _install_aux(self, field_name, field):
        """Lower an aux entry on AMR: 'B_z' -> set_magnetic_field; 'T_e' rejected (derived); any
        other name -> set_aux_field on the block that declares it. Mirror of System._install_aux."""
        if field_name == "B_z":
            self.set_magnetic_field(field)
            return
        if field_name == "T_e":
            raise ValueError(
                "AmrSystem.install: aux 'T_e' is DERIVED from a fluid block, not a static aux "
                "field; use set_electron_temperature_from(block).")
        block = None
        for blk, table in self._aux_field_index.items():
            if field_name in table:
                block = blk
                break
        if block is None:
            raise ValueError(
                "AmrSystem.install: aux field %r is not declared by any installed instance; add the "
                "instance with a model declaring m.aux_field(%r)." % (field_name, field_name))
        self.set_aux_field(block, field_name, field)

    def add_coupling(self, coupling):
        """Add a generic inter-species COUPLED SOURCE (pops.dsl.CoupledSource(...).compile(...))
        on the SHARED AMR hierarchy (MULTI-BLOCK), refined counterpart of System.add_coupling. The source
        is transported as bytecode and interpreted on the C++ side (AmrSystem.add_coupled_source; no
        per-cell Python callback). The coupling frequency (CoupledSource.frequency) is honored:
        constant -> dt bound dt <= cfl/mu; Expr -> PER-CELL frequency mu(U) evaluated on the COARSE grid at
        each step_cfl (the freq_prog_* vectors are forwarded). Must be called BEFORE the first
        step (the source is frozen then injected at the lazy build of the runtime engine)."""
        # Late import (the multispecies module imports this package: avoid the cycle).
        from pops.physics.multispecies import CompiledCoupledSource

        if isinstance(coupling, CompiledCoupledSource):
            self._s.add_coupled_source(coupling.in_blocks, coupling.in_roles, coupling.consts,
                                       coupling.out_blocks, coupling.out_roles, coupling.prog_ops,
                                       coupling.prog_args, coupling.prog_lens,
                                       getattr(coupling, "frequency", 0.0), coupling.name,
                                       getattr(coupling, "freq_prog_ops", []),
                                       getattr(coupling, "freq_prog_args", []))
        else:
            raise TypeError("AmrSystem.add_coupling expects a CompiledCoupledSource "
                            "(pops.dsl.CoupledSource(...).compile(...)): the AMR coupled source is "
                            "MULTI-BLOCK and described in formulas")

    @property
    def amr(self):
        """The live AMR runtime inspection handle (Spec 5 sec.8.12), an
        :class:`pops.runtime.amr.AmrRuntimeView`.

        Bound to THIS built hierarchy: ``sim.amr.patch_table()`` /
        ``sim.amr.hierarchy_snapshot()`` / ``sim.amr.explain_regrid()`` /
        ``explain_ghosts()`` / ``explain_reflux()`` / ``explain_checkpoint()`` return short, inert
        reports of the patches that actually exist, the regrid cadence in force, and the
        ghost / reflux / checkpoint route limitations. The view READS the runtime (the box
        accessors + the retained config); it builds / allocates / steps NOTHING.

        ``System.amr`` does not exist: the inspection surface is AMR-specific (a uniform System
        carries no hierarchy). Use ``pops.inspect_amr(layout)`` for the STATIC authoring report.
        """
        from pops.runtime.amr import AmrRuntimeView  # lazy: keeps the constructor import-light.

        return AmrRuntimeView(self)

    def __str__(self):
        """Short, array-free summary: block names on the AMR hierarchy (Spec 5 sec.12.1).

        Field/patch data stays out of the summary -- it prints the block registry only.
        """
        try:
            blocks = list(self._s.block_names())
        except Exception:  # pragma: no cover - defensive: _AmrSystem not fully wired
            blocks = []
        return "AmrSystem(blocks=%s)" % (blocks,)

    def explain_bind(self, compiled):
        """A printable :class:`pops.codegen.inspect_report.BindReport` of @p compiled vs this AMR sim
        (Spec 5 sec.12.1, criterion #15). INERT parity with ``System.explain_bind``: reads the
        artifact's DECLARED bind inputs (``compiled.arguments()``) and the blocks / named aux wired on
        this AmrSystem, then reuses ADC-463 :func:`collect_missing_arguments` to report PROVIDED vs
        still-REQUIRED per group. It binds nothing and mutates nothing -- a read-only bind plan."""
        from pops.codegen.inspect_report import build_bind_report
        return build_bind_report(self, compiled)

    def __getattr__(self, attr):
        return getattr(self._s, attr)
