"""pops.codegen.loader : result/wrapper classes for compiled .so artefacts.

``CompiledModel`` packages a model ``.so`` with the metadata needed to wire
it (adder, names, roles, gamma, n_aux, params, caps, abi_key, model_hash).

``CompiledProblem`` packages a program ``.so`` (a compiled time
``pops.time.Program``) plus the metadata to install + reproduce it.

Neither class imports ``pops.dsl`` or ``pops.physics`` at module level.
"""


class CompiledProblem:
    """Result of ``pops.compile_problem(...)``; a generated ``problem.so``
    (a compiled time Program) plus the metadata to install + reproduce it.
    Install it with ``sim.install_program(compiled.so_path)`` AFTER the
    physical block has been added (``sim.add_equation`` / ``sim.add_block``);
    the Program then drives ``sim.step(dt)`` entirely in C++ via
    ``ProgramContext``.

    The ``.so`` is compiled against the pops headers with the SAME Kokkos
    toolchain as the loaded _pops module (cf. ``pops_loader_build_flags``),
    so its ABI key matches and ``System::install_program`` accepts it.
    ``os.fspath(compiled)`` returns ``so_path`` (it can be passed where a
    path is expected).
    """

    def __init__(self, so_path, program, model, abi_key, cxx, std, libraries=None):
        self.so_path = so_path
        self.program = program          # the pops.time.Program that was lowered
        self.model = model              # the physical model (optional; added as a block in the MVP)
        self.program_name = getattr(program, "name", None)
        self.program_hash = program._ir_hash() if hasattr(program, "_ir_hash") else None
        self.abi_key = abi_key          # cache key: header signature | compiler | C++ standard
        self.cxx = cxx
        self.std = std
        # Validated brick libraries (Spec 3 section 21, ADC-464): the LibraryManifests read +
        # ABI-checked from libraries=[...]. Empty when none were passed. Their bricks (and their
        # generated symbols) are exposed to the problem; a compiled library .so was already
        # dlopen'd (and ABI-guarded) by read_library_manifest.
        self.libraries = list(libraries) if libraries else []

    def __fspath__(self):
        return self.so_path

    # --- operator introspection (Spec 2, S2-5): metadata read from the carried model,
    # no need to load or run the .so.
    def _intro_model(self):
        if self.model is None:
            raise ValueError("this CompiledProblem carries no model; operator introspection "
                             "is unavailable")
        return self.model

    def list_operators(self):
        """Names of the typed operators of the compiled module (registration order)."""
        return self._intro_model().operator_registry().names()

    def list_state_spaces(self):
        """Names of the compiled module's state spaces."""
        return self._intro_model().list_state_spaces()

    def list_field_spaces(self):
        """Names of the compiled module's field spaces."""
        return self._intro_model().list_field_spaces()

    def operator_signature(self, name):
        """The pops.model.Signature of operator ``name`` in the compiled module."""
        return self._intro_model().operator_registry().get(name).signature

    def operator_requirements(self, name):
        """The requirements dict of operator ``name``."""
        return dict(self._intro_model().operator_registry().get(name).requirements)

    def operator_capabilities(self, name):
        """The capabilities dict of operator ``name``."""
        return dict(self._intro_model().operator_registry().get(name).capabilities)

    def __repr__(self):
        return "<CompiledProblem %r -> %s>" % (self.program_name, self.so_path)


class CompiledModel:
    """Result of ``m.compile(...)``: packages the produced ``.so`` + EVERYTHING
    needed to wire it correctly (dispatch adder, ABI diagnostic,
    reproducibility). Replaces the historical pair (str so_path,
    adder_for(backend)) with a single object.

    The metadata is NOT re-read from the ``.so``: Python already holds
    names/roles/gamma/n_aux/params (the HyperbolicModel carries them);
    CompiledModel just exposes them for dispatch (add_equation) and
    diagnostics. cf. DSL_MODEL_DESIGN.md section 3.
    """

    def __init__(self, so_path, backend, adder, cons_names, cons_roles, prim_names, n_vars,
                 gamma, n_aux, params, caps, abi_key, model_hash, cxx, std, target="system",
                 hllc=False, roe=False, aux_extra_names=None, wave_speeds=False):
        self.has_hllc = bool(hllc)   # HLLC capability emitted (enable_hllc): hllc available beyond 4-var Euler
        self.has_roe = bool(roe)     # ROE hook emitted (enable_roe roles OR m.roe_dissipation provided): roe available beyond 4-var Euler
        self.has_wave_speeds = bool(wave_speeds)  # wave_speeds emitted (explicit pair OR 'p'): hll available
        self.so_path = so_path
        self.backend = backend       # "prototype" | "aot" | "production"
        self.target = target         # "system" | "amr_system": targeted facade (native AMR loader if amr_system)
        self.adder = adder           # method name (Amr)System: add_dynamic_block / add_compiled_block / add_native_block
        self.cons_names = list(cons_names)
        self.cons_roles = list(cons_roles)
        self.prim_names = list(prim_names)
        self.n_vars = int(n_vars)
        self.gamma = gamma           # None = historical default 1.4 on the System side
        self.n_aux = int(n_aux)
        # Names of the NAMED aux fields (aux_field, ADC-70), ORDERED: component index = position
        # AUX_NAMED_BASE + k. The System.add_equation facade builds the name -> component table per
        # block from it, consumed by System.set_aux_field / aux_field. Empty for a model without a named field.
        self.aux_extra_names = list(aux_extra_names) if aux_extra_names else []
        self.params = dict(params)   # {name: Param}
        self.caps = dict(caps)       # {cpu/mpi/amr/gpu: bool}
        self.abi_key = abi_key       # ABI key mirroring pops_header_signature + compiler/std
        self.model_hash = model_hash  # stable hash formulas+roles+n_aux+params
        self.cxx = cxx
        self.std = std

    @property
    def runtime_param_names(self):
        """Names of the model's RUNTIME parameters (kind='runtime'), SORTED: this is the ORDER of
        the indices on the C++ side (RuntimeParams) AND the order expected by
        System.set_block_params(name, values) (P7-b). Empty if the model has only const params."""
        return sorted(k for k, p in self.params.items() if getattr(p, "kind", "const") == "runtime")

    def runtime_param_values(self):
        """DECLARATION values of the runtime params, parallel to runtime_param_names (default as
        long as no set_block_params has been called)."""
        return [self.params[k].value for k in self.runtime_param_names]

    def check_runtime(self, n=16, state=None, raise_on_error=True, rtol=1e-8, atol=1e-10):
        """RUNTIME re-verification of a CompiledModel ALONE (audit balance, GENERICITY pt 9):
        without the original dsl.Model, the FORMULAS are no longer re-verifiable (symbolic
        check_model), but the .so itself is -- we install it in an EPHEMERAL System (n x n
        periodic, neutral Poisson, minmod+rusanov) and delegate to System.check_model (finite
        state, residual -div F + S finite, positivity by roles, round-trip of THE MODEL
        conversions).

        @p state: dict {conservative variable name: ndarray (n, n)} to control the tested state.
        None -> SMOKE state by ROLES (Density = 1 + gaussian bump, Momentum* = 0,
        Energy = 2.5, other components = 0.5) -- enough to exercise flux/source/conversions;
        provide state= for a precise physical regime. @return the dict from System.check_model.
        """
        import numpy as np  # lazy: only needed at check_runtime call time
        if getattr(self, "target", "system") != "system":
            raise ValueError(
                "CompiledModel.check_runtime: only target='system' is re-verifiable in an "
                "ephemeral System; a target='amr_system' loader is checked installed in its "
                "AmrSystem (AMR test invariants), not in isolation.")
        from pops import System, FiniteVolume, Explicit  # lazy: avoids a top-level runtime import
        from pops.numerics.reconstruction.limiters import Minmod
        from pops.numerics.riemann import Rusanov
        sim = System(n=int(n), L=1.0, periodic=True)
        sim.set_poisson()
        sim.add_equation("check", model=self,
                         spatial=FiniteVolume(limiter=Minmod(), riemann=Rusanov()),
                         time=Explicit())
        x = (np.arange(n) + 0.5) / float(n)
        X, Y = np.meshgrid(x, x, indexing="xy")
        bump = 1.0 + 0.3 * np.exp(-40.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))
        comps = []
        for name, role in zip(self.cons_names, self.cons_roles, strict=True):
            if state is not None and name in state:
                comps.append(np.asarray(state[name], dtype=float).reshape(n, n))
            elif role == "Density":
                comps.append(bump)
            elif role in ("MomentumX", "MomentumY"):
                comps.append(np.zeros((n, n)))
            elif role == "Energy":
                comps.append(2.5 + 0.0 * bump)
            else:
                comps.append(0.5 + 0.0 * bump)
        sim._s.set_state("check", np.stack(comps).ravel())
        return sim.check_model("check", raise_on_error=raise_on_error, rtol=rtol, atol=atol)

    def __repr__(self):
        return ("CompiledModel(backend=%r, target=%r, so_path=%r, n_vars=%d, gamma=%r, n_aux=%d, "
                "adder=%r, runtime_params=%r, abi_key=%.12s..., model_hash=%.12s...)"
                % (self.backend, self.target, self.so_path, self.n_vars, self.gamma, self.n_aux,
                   self.adder, self.runtime_param_names, self.abi_key or "", self.model_hash or ""))
