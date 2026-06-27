"""System unified-install mixin (Spec-4 PR-F): the Spec-3 section-22 ``install`` surface.

``install`` (the single entry point that lowers to add_equation / set_poisson /
set_magnetic_field / set_aux_field / set_block_params / install_program) plus its private
lowering helpers. Mixed into ``System`` via inheritance; methods operate on ``self`` (calling the
other mixins' methods) and ``self._s``.
"""

from pops._bootstrap import ModelSpec
from pops.runtime.bricks import Spatial, FiniteVolume


class _SystemUnifiedInstall:
    """The unified ``install`` lowering surface of System."""

    def install(self, compiled, *, instances=None, params=None, aux=None, solvers=None):
        """Unified install (Spec 3 section 22): wire a compiled handle + per-instance state/spatial +
        params + aux + field solvers in ONE call, then install the compiled time Program.

        This is the clean single entry point of Spec 3. It LOWERS to the existing lower-layer calls
        (add_equation / set_poisson / set_magnetic_field / set_aux_field / set_block_params /
        install_program) -- there is NO parallel runtime (Spec section 3). The lower-layer calls stay
        available and unchanged; sim.install just sequences them in the right order so the
        install-time validation (section 24) sees a fully-configured simulation.

        @param compiled the compiled problem handle (compile_problem(...) result) carrying ``so_path``;
            installed via install_program after every instance/solver/aux is wired.
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

        @throws the verbatim Spec section-24 errors at install for a missing aux / solver / block
            instance / Riemann capability. (A disallowed schedule is rejected earlier, at Program
            authoring/compile -- see pops.time._validate_schedule -- not here.)
        """
        instances = instances or {}
        params = params or {}
        aux = aux or {}
        solvers = solvers or {}

        # (1) FIELD SOLVERS first: set_poisson must run before install_program (the C++ section-24
        # solver requirement reads poisson_solver()).
        for field, solver_brick in solvers.items():
            self._install_solver(field, solver_brick)

        # (2) INSTANCES: add each named block (binds the Program block of that name, criterion 23),
        # lower its spatial brick and set its initial state. The block model is the per-instance
        # "model" if given, else the PHYSICAL model carried by the compiled handle
        # (CompiledProblem.model) -- NOT the handle itself (which is the time Program .so installed in
        # step 5). For a single-instance plasma case the carried model is the block.
        # Validate the compiled handle up front, BEFORE any System mutation, so a misuse cannot leave a
        # half-configured System.
        so_path = getattr(compiled, "so_path", None)
        if so_path is None:
            raise TypeError("install: compiled handle has no .so_path (got %r); pass a "
                            "compile_problem(...) result" % type(compiled).__name__)

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

        # (5) Install the compiled time Program (binds blocks by name + runs the section-24 .so
        # requirement validation: aux / solver / block instance, verbatim messages).
        self.install_program(so_path)

    def _lower_spatial(self, spatial):
        """Lower a spatial selection to an pops.Spatial consumed by add_equation. Accepts an
        pops.Spatial / pops.FiniteVolume (returned as-is), an pops.lib.spatial.FiniteVolume(...)
        BrickDescriptor (read its riemann/reconstruction/positivity_floor options), or None (default
        Spatial)."""
        if spatial is None:
            return Spatial()
        if isinstance(spatial, Spatial):
            return spatial
        # A lib BrickDescriptor carries the scheme options in .options.
        opts = getattr(spatial, "options", None)
        if isinstance(opts, dict):
            limiter = opts.get("reconstruction", opts.get("limiter", "minmod"))
            riemann = opts.get("riemann", opts.get("flux", "rusanov"))
            variables = opts.get("variables", opts.get("recon", "conservative"))
            return FiniteVolume(limiter=limiter, riemann=riemann, variables=variables,
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
