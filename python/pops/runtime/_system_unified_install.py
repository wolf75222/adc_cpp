"""System unified-install mixin (Spec-4 PR-F): the Spec-3 section-22 ``install`` surface.

``install`` (the single entry point that lowers to add_equation / set_poisson /
set_magnetic_field / set_aux_field / set_block_params / install_program) plus its private
lowering helpers. Mixed into ``System`` via inheritance; methods operate on ``self`` (calling the
other mixins' methods) and ``self._s``.
"""

from pops._bootstrap import ModelSpec
from pops.runtime.bricks import Spatial


def collect_missing_arguments(args, provided_blocks, provided_params, provided_aux,
                              provided_solvers):
    """Pure core of the early bind-input check (Spec 5 sec.10); no engine call -> host-testable.

    Compare an :class:`pops.codegen.inspect_compiled.Arguments` against what an install supplies and
    return one actionable line per MISSING required argument (empty list when everything required is
    met). Shared by ``System.install`` and ``AmrSystem.install`` so both enforce the SAME contract.

    Only entries whose ``required`` flag is true are enforced: an input the artifact marks optional
    (a const param, an unrequired solver -- the default Poisson field has a working default and is
    NOT flagged required by ``arguments()``) is never demanded, so a previously valid install passes
    through unchanged. ``provided_*`` are the supplied sets (block names, param names, aux names,
    solver fields); a block already added on the sim counts as provided. Each line names EXACTLY what
    is missing and the matching ``install`` keyword to supply it."""
    missing = []
    for name, spec in sorted(getattr(args, "instances", {}).items()):
        if spec.get("required") and name not in provided_blocks:
            missing.append("instance %r (a state block the program advances); add it via "
                           "install(instances={%r: {'initial': <array>, ...}})" % (name, name))
    for name, spec in sorted(getattr(args, "params", {}).items()):
        if spec.get("required") and name not in provided_params:
            missing.append("runtime param %r; pass install(params={%r: <value>})" % (name, name))
    for name, spec in sorted(getattr(args, "aux", {}).items()):
        if spec.get("required") and name not in provided_aux:
            missing.append("aux field %r; pass install(aux={%r: <array>})" % (name, name))
    for name, spec in sorted(getattr(args, "solvers", {}).items()):
        if spec.get("required") and name not in provided_solvers:
            missing.append("solver for field %r; pass install(solvers={%r: <Solver>})"
                           % (name, name))
    return missing


def validate_install_arguments(sim, compiled, instances, params, aux, solvers):
    """Early bind-input validation (Spec 5 sec.10) for a COMPILED install on @p sim (System OR
    AmrSystem): reject -- BEFORE any native mutation -- an install missing a REQUIRED argument the
    artifact declares, with one clear actionable error aggregating every missing input.

    Reads ``compiled.arguments()`` (the inert metadata the .so DECLARES) and confirms every argument
    marked ``required`` is supplied by this install call (@p instances / params / aux / solvers) OR
    already wired on the sim (an added block, a declared named aux). A NATIVE install
    (``compiled is None``) carries no declared arguments and is skipped; a handle whose
    ``arguments()`` is unavailable or unreadable is skipped too (conservative -- a missing check
    never breaks a working install)."""
    if compiled is None or not hasattr(compiled, "arguments"):
        return
    try:
        args = compiled.arguments()
    except Exception:  # noqa: BLE001 -- introspection must never break a valid install
        return
    provided_blocks = set(instances)
    try:
        provided_blocks |= set(sim.block_names())
    except Exception:  # noqa: BLE001 -- block_names is a convenience; absence is not a failure
        pass
    # Named aux already declared on the sim (B_z has no queryable trace, so it must come via aux=).
    provided_named_aux = set()
    for table in getattr(sim, "_aux_field_index", {}).values():
        provided_named_aux |= set(table)
    missing = collect_missing_arguments(
        args, provided_blocks, set(params), set(aux) | provided_named_aux, set(solvers))
    if missing:
        raise ValueError("install: the compiled artifact is missing required argument(s):\n  "
                         + "\n  ".join(missing))


class _SystemUnifiedInstall:
    """The unified ``install`` lowering surface of System."""

    def install(self, compiled=None, *, instances=None, params=None, aux=None, solvers=None, cadence=None):
        """Unified install (Spec 3 section 22): wire a compiled handle + per-instance state/spatial +
        params + aux + field solvers in ONE call, then install the compiled time Program.

        This is the clean single entry point of Spec 3. It LOWERS to the existing lower-layer calls
        (add_equation / set_poisson / set_magnetic_field / set_aux_field / set_block_params /
        install_program) -- there is NO parallel runtime (Spec section 3). The lower-layer calls stay
        available and unchanged; sim.install just sequences them in the right order so the
        install-time validation (section 24) sees a fully-configured simulation.

        install() is the ONE entry for BOTH runtime modes (Spec 4 amendment): a COMPILED-program sim
        (pass the compile_problem(...) handle as ``compiled``) and a NATIVE sim (``compiled=None``;
        each instance carries its own native model + native time policy, no compiled .so).

        @param compiled the compiled problem handle (compile_problem(...) result) carrying ``so_path``,
            installed via install_program after every instance/solver/aux is wired. Pass ``None`` for a
            NATIVE sim: no Program is installed; each instance must supply its own native ``"model"``
            and (optionally) ``"time"`` policy, and the native per-block advance loop drives stepping.
        @param instances dict {name: {"initial": array, "spatial": <brick>, "model": <pops.Model>,
            "time": <pops.Explicit/IMEX>}}. The block is bound by the dict KEY @p name (Spec criterion
            23), not a "state" field. Each entry adds the named block (add_equation), sets its
            "initial" state (if given) and lowers the "spatial" brick to the add_equation spatial args.
            The block model is the per-instance ``"model"`` if given, else ``compiled`` (single-
            instance case). ``spatial`` is an pops.FiniteVolume(...) / pops.Spatial(...) OR an
            pops.lib.spatial.FiniteVolume(...) descriptor.
        @param params dict {param_name: value} of RUNTIME parameters, routed to the instance whose
            compiled model declares the name (set_block_params). Unknown names raise.
        @param aux dict {field_name: array}: "B_z" -> set_magnetic_field, "T_e" -> rejected (it is
            DERIVED, use set_electron_temperature_from), any other -> set_aux_field on the instance
            declaring it. Set BEFORE install_program so the section-24 aux requirement check sees it.
        @param solvers dict {field: <pops.lib.fields.GeometricMG(...)/pops.GeometricMG(...)>}: lowered to
            set_poisson(solver=...). Only the default Poisson field ("phi"/"charge_density"/"poisson")
            is wired today; a second named elliptic field raises NotImplementedError (deferred).
        @param cadence optional pops.CompiledTime(substeps=, stride=): the compiled Program's macro-step
            cadence, applied with set_program_cadence AFTER install_program. A compiled Program is ONE
            whole-system closure, so its cadence is GLOBAL (one program-level value, not per-block) --
            hence a single kwarg rather than a per-instance "time". A non-default cfl is deferred.

        @throws the verbatim Spec section-24 errors at install for a missing aux / solver / block
            instance / Riemann capability. (A disallowed schedule is rejected earlier, at Program
            authoring/compile -- see pops.time._validate_schedule -- not here.)
        """
        instances = instances or {}
        params = params or {}
        aux = aux or {}
        solvers = solvers or {}

        # (0) EARLY VALIDATION (Spec 5 sec.10): in the COMPILED path, read the artifact's DECLARED
        # bind inputs (compiled.arguments()) and reject -- BEFORE any native call, so a misuse cannot
        # leave a half-configured System -- an install that does not supply a REQUIRED argument
        # (instance / runtime param / aux / solver). It enforces only arguments() 'required' flags;
        # an input the artifact marks optional (a const param, an unrequired solver) is never demanded.
        # Inert: it reads metadata and compares dicts (no compile / bind / allocation). It never
        # rejects an install that supplies everything required, so a valid install is unchanged.
        self._validate_install_arguments(compiled, instances, params, aux, solvers)

        # (1) FIELD SOLVERS first: set_poisson must run before install_program (the C++ section-24
        # solver requirement reads poisson_solver()).
        for field, solver_brick in solvers.items():
            self._install_solver(field, solver_brick)

        # (2) INSTANCES: add each named block (binds the Program block of that name, criterion 23),
        # lower its spatial brick and set its initial state. The block model is the per-instance
        # "model" if given, else the PHYSICAL model carried by the compiled handle
        # (CompiledProblem.model) -- NOT the handle itself (which is the time Program .so installed in
        # step 5). For a single-instance plasma case the carried model is the block.
        # COMPILED vs NATIVE mode. COMPILED: `compiled` is a compile_problem(...) handle carrying a
        # .so_path time Program (installed in step 5, with the section-24 validation). NATIVE:
        # `compiled is None` -- there is no compiled Program; each instance carries its OWN native model
        # + native time policy (pops.Explicit / pops.Strang), step 5 is skipped, and the native
        # per-block advance loop drives stepping. Validate the handle up front, BEFORE any System
        # mutation, so a misuse cannot leave a half-configured System.
        so_path = None
        compiled_model = None
        if compiled is not None:
            so_path = getattr(compiled, "so_path", None)
            if so_path is None:
                raise TypeError(
                    "install: compiled handle has no .so_path (got %r); pass a compile_problem(...) "
                    "result, or compiled=None for a native sim (each instance carries its own native "
                    "model)." % type(compiled).__name__)
            compiled_model = getattr(compiled, "model", None)
        resolved_models = {}  # instance name -> RESOLVED (CompiledModel), reused by the params step
        for name, spec in instances.items():
            if not isinstance(spec, dict):
                raise TypeError("install: instances[%r] must be a dict (initial/spatial/time/model); "
                                "got %r" % (name, type(spec).__name__))
            model = spec.get("model", compiled_model)
            if model is None:
                raise ValueError(
                    "install: instance %r has no block model -- supply instances[%r]['model'] "
                    "(an pops.Model(...) / CompiledModel), or pass a compiled handle that carries one "
                    "(compile_problem(model=...))." % (name, name))
            model = self._resolve_instance_model(model)
            resolved_models[name] = model
            spatial = self._lower_spatial(spec.get("spatial"))
            time = spec.get("time")
            # Capability check (section 24): the selected Riemann flux must be backed by the model.
            self._validate_riemann_capability(model, spatial)
            self.add_equation(name, model, spatial=spatial, time=time)
            initial = spec.get("initial")
            if initial is not None:
                self.set_state(name, initial)

        # (3) AUX fields: B_z -> set_magnetic_field; named -> set_aux_field. Before install_program.
        for field_name, field in aux.items():
            self._install_aux(field_name, field)

        # (4) PARAMS: route each runtime param to the instance whose RESOLVED (compiled) model declares
        # it -- the raw dsl.Model has no runtime_param_names, so we read the resolved CompiledModel.
        if params:
            self._install_params(resolved_models, params)

        # (5) COMPILED mode only: install the compiled time Program (binds blocks by name + runs the
        # section-24 .so requirement validation: aux / solver / block instance, verbatim messages). In
        # NATIVE mode (compiled=None) there is no Program -- the blocks added in step 2 are driven by
        # the native per-block advance loop, so this step is skipped.
        if so_path is not None:
            self.install_program(so_path)

        # (6) PROGRAM CADENCE (substeps / stride): a compiled Program is ONE whole-system closure, so
        # its macro-step cadence is GLOBAL (not per-block). Apply it AFTER install_program (the cadence
        # wraps the installed closure). It is a compiled-program concept; a native sim sets substeps /
        # stride on its native time policy (pops.Explicit(substeps=, stride=)) instead.
        if cadence is not None:
            if so_path is None:
                raise ValueError(
                    "install(cadence=): a cadence applies to a compiled time Program; a native sim "
                    "(compiled=None) has no Program -- set substeps / stride on the native time policy "
                    "(pops.Explicit(substeps=, stride=)) instead.")
            self._install_cadence(cadence)

    def explain_bind(self, compiled):
        """A printable :class:`pops.codegen.inspect_report.BindReport` of @p compiled vs this sim
        (Spec 5 sec.12.1, criterion #15). INERT: reads the artifact's DECLARED bind inputs
        (``compiled.arguments()``) and the blocks / named aux ALREADY wired on this System, then
        reuses the ADC-463 :func:`collect_missing_arguments` to compute, per group
        (instances / params / aux / solvers), which inputs are PROVIDED vs still REQUIRED. It binds
        nothing and mutates nothing -- the read-only counterpart of ``install``'s early validation."""
        from pops.codegen.inspect_report import build_bind_report
        return build_bind_report(self, compiled)

    def _validate_install_arguments(self, compiled, instances, params, aux, solvers):
        """Early bind-input validation (Spec 5 sec.10): reject a COMPILED install missing a REQUIRED
        argument the artifact declares, BEFORE any native mutation. Thin wrapper around the shared
        module-level :func:`validate_install_arguments` (reused by ``AmrSystem.install`` for parity)."""
        validate_install_arguments(self, compiled, instances, params, aux, solvers)

    # Host-testable alias of the pure core (mirrors _route_block_params: callable as
    # System._collect_missing_arguments without building a System).
    _collect_missing_arguments = staticmethod(collect_missing_arguments)

    def _install_cadence(self, cadence):
        """Apply a CompiledTime macro-step cadence to the installed program (set_program_cadence).

        set_program_cadence is a SYSTEM-level orchestration around the opaque program closure
        (program.py): substeps=n re-runs the whole program over eff_dt/n; stride=M runs it once per M
        macro-steps. A non-default cfl is deferred (pass an explicit dt to sim.step)."""
        from pops.time.program import CompiledTime
        if not isinstance(cadence, CompiledTime):
            raise TypeError("install(cadence=): expected a pops.CompiledTime(substeps=, stride=), "
                            "got %r" % type(cadence).__name__)
        if cadence.cfl != "default":
            raise NotImplementedError(
                "install(cadence=): a non-default cfl is deferred; pass an explicit dt to sim.step(dt)")
        self.set_program_cadence(cadence.substeps, cadence.stride)

    def _lower_spatial(self, spatial):
        """Lower a spatial selection to an pops.Spatial consumed by add_equation. Accepts an
        pops.Spatial / pops.FiniteVolume (returned as-is), an pops.lib.spatial.FiniteVolume(...)
        BrickDescriptor (read its riemann/reconstruction/positivity_floor options), or None (default
        Spatial)."""
        if spatial is None:
            return Spatial()
        if isinstance(spatial, Spatial):
            return spatial
        # A lib BrickDescriptor carries the scheme options as STRING tokens in .options. Lower them
        # to the canonical Spatial tokens directly (Spatial._from_tokens bypasses the public typed-
        # descriptor guard, which the runtime FiniteVolume now enforces -- Spec 5 sec.7).
        opts = getattr(spatial, "options", None)
        if isinstance(opts, dict):
            limiter = opts.get("reconstruction", opts.get("limiter", "minmod"))
            riemann = opts.get("riemann", opts.get("flux", "rusanov"))
            variables = opts.get("variables", opts.get("recon", "conservative"))
            return Spatial._from_tokens(
                limiter, riemann, variables,
                positivity_floor=opts.get("positivity_floor"),
                wave_speed_cache=bool(opts.get("wave_speed_cache", False)))
        raise TypeError("install: spatial must be an pops.FiniteVolume / pops.Spatial or an "
                        "pops.lib.spatial.FiniteVolume(...) descriptor; got %r"
                        % type(spatial).__name__)

    def _resolve_instance_model(self, model):
        """Resolve an instance's block model to something add_equation accepts. A ModelSpec
        (pops.Model(...)) or a dsl.CompiledModel passes through unchanged. A dsl.Model (the PDE
        builder, e.g. carried by compile_problem(model=...)) is compiled to a CompiledModel so the
        block is added on the real System context.

        Backend choice (P7-b): a dsl.Model declaring RUNTIME params is compiled via AOT, because the
        production/native backend FREEZES runtime params at their declaration value (so
        install(params=...) -> set_block_params would raise 'block ... has no runtime parameter'); a
        const-only model keeps the native production path (no .so dlopen). The AOT block gates its OWN
        time integrator to SSPRK2 + backward-Euler, but that is harmless here: a unified install runs
        the compiled time Program, which drives the step (the per-instance ``time`` is not the stepper;
        cf. compile_problem). A runtime-param instance must therefore use an AOT-compatible
        ``time`` (the default pops.Explicit() == SSPRK2 is fine; euler/ssprk3 raise at add_equation)."""
        # Late imports (the codegen/physics modules import this package: avoid the cycle).
        from pops.codegen.loader import CompiledModel
        from pops.physics.facade import Model
        if isinstance(model, (ModelSpec, CompiledModel)):
            return model
        if isinstance(model, Model):
            has_runtime = any(getattr(p, "kind", "const") == "runtime"
                              for p in model.params.values())
            return model.compile(backend="aot" if has_runtime else "production")
        return model  # unknown -> let add_equation raise its own clear error

    def _validate_riemann_capability(self, model, spatial):
        """Section 24 capability check: reject the selected Riemann flux when the model does not back
        it, with the verbatim spec message ``riemann <FLUX> requires capability '<cap>'``. Lowered
        from the model's emitted capabilities (CompiledModel.has_hllc / has_roe / has_wave_speeds);
        a composed native pops.Model(...) carries the capability in its bricks (the C++ requires-gate
        is the backstop), so we only gate the compiled (.so) path here."""
        from pops.codegen.loader import CompiledModel  # late import (codegen <-> __init__ cycle)
        flux = getattr(spatial, "flux", "rusanov")
        if not isinstance(model, CompiledModel):
            return  # native composed model: the C++ requires-gate validates at first use
        if flux == "hllc" and not (getattr(model, "has_hllc", False)
                                   or "p" in getattr(model, "prim_names", [])):
            raise RuntimeError("riemann HLLC requires capability 'hllc_star_state'")
        if flux == "roe" and not (getattr(model, "has_roe", False)
                                  or "p" in getattr(model, "prim_names", [])):
            raise RuntimeError("riemann Roe requires capability 'roe_dissipation'")
        if flux == "hll" and not getattr(model, "has_wave_speeds", True):
            raise RuntimeError("riemann HLL requires capability 'wave_speeds'")

    def _install_solver(self, field, solver_brick):
        """Lower a field-solver selection to set_poisson. Only the default Poisson field is wired
        today; a second named elliptic field is deferred (NotImplementedError)."""
        if field not in ("phi", "poisson", "charge_density", "default"):
            raise NotImplementedError(
                "install: a second named elliptic field (%r) is not wired yet; only the default "
                "Poisson field ('phi') is supported. Configure extra fields via the lower-layer "
                "register_elliptic_field path." % (field,))
        token = self._solver_token(solver_brick)
        opts = getattr(solver_brick, "options", {}) or {}
        self.set_poisson(rhs=opts.get("rhs", "charge_density"), solver=token,
                         bc=opts.get("bc", "auto"), wall=opts.get("wall", "none"),
                         wall_radius=float(opts.get("wall_radius", 0.0)),
                         epsilon=float(opts.get("epsilon", 1.0)),
                         abs_tol=float(opts.get("abs_tol", 0.0)))

    @staticmethod
    def _solver_token(solver_brick):
        """Resolve a field-solver selection to its set_poisson token. Accepts a string, or a
        descriptor carrying ``scheme`` (pops.lib.fields.GeometricMG -> 'geometric_mg')."""
        if isinstance(solver_brick, str):
            return solver_brick
        token = getattr(solver_brick, "scheme", None) or getattr(solver_brick, "name", None)
        if token is None:
            raise TypeError("install: solver must be a token string or an pops.lib.fields.<Solver>(...) "
                            "descriptor; got %r" % type(solver_brick).__name__)
        return token

    def _install_aux(self, field_name, field):
        """Lower an aux entry: 'B_z' -> set_magnetic_field; 'T_e' rejected (derived); any other name
        -> set_aux_field on the block that declares it."""
        if field_name == "B_z":
            self.set_magnetic_field(field)
            return
        if field_name == "T_e":
            raise ValueError(
                "install: aux 'T_e' is DERIVED from a fluid block via "
                "set_electron_temperature_from(block), not set as a static aux field.")
        block = self._block_declaring_aux(field_name)
        if block is None:
            raise ValueError(
                "install: aux field %r is not declared by any installed instance; add the instance "
                "with a model declaring m.aux_field(%r)." % (field_name, field_name))
        self.set_aux_field(block, field_name, field)

    def _block_declaring_aux(self, field_name):
        """The block whose named-aux table declares @p field_name, or None."""
        for block, table in self._aux_field_index.items():
            if field_name in table:
                return block
        return None

    @staticmethod
    def _route_block_params(resolved_models, params):
        """Pure routing core of _install_params (no engine call -> host-testable). Map a flat
        {param_name: value} to {block: sorted runtime-param value vector} using each RESOLVED model's
        runtime_param_names (declaration defaults for unspecified names), and return the param names
        declared by no instance. @p resolved_models maps each instance name to its RESOLVED
        CompiledModel: the raw dsl.Model has no runtime_param_names accessor, so a model passed
        UNRESOLVED here contributes no params (the bug install's resolve step prevents -- see
        install step (2)). @return (per_block, unknown), per_block only listing blocks with params."""
        consumed = set()
        per_block = {}
        for name, model in resolved_models.items():
            # runtime_param_names is a @property (list); runtime_param_values is a method.
            rt_names = list(getattr(model, "runtime_param_names", []) or [])
            if not rt_names:
                continue
            values_fn = getattr(model, "runtime_param_values", None)
            defaults = list(values_fn()) if callable(values_fn) else [None] * len(rt_names)
            values = []
            for k, pname in enumerate(rt_names):
                if pname in params:
                    values.append(float(params[pname]))
                    consumed.add(pname)
                else:
                    values.append(float(defaults[k]) if defaults[k] is not None else 0.0)
            per_block[name] = values
        unknown = sorted(set(params) - consumed)
        return per_block, unknown

    def _install_params(self, resolved_models, params):
        """Route flat {param_name: value} to set_block_params per instance: build each instance's
        sorted runtime-param vector (declaration defaults for unspecified names) and push it. A param
        name declared by no instance raises (no silent drop). @p resolved_models maps each instance
        name to its RESOLVED CompiledModel (the raw dsl.Model has no runtime_param_names accessor)."""
        per_block, unknown = self._route_block_params(resolved_models, params)
        for name, values in per_block.items():
            self._s.set_block_params(name, values)
        if unknown:
            raise ValueError("install: params %s declared by no instance's runtime parameters"
                             % (unknown,))
