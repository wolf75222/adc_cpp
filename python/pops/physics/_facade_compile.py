"""Compile mixin for the PDE-model facade (:class:`pops.physics.facade.Model`).

Splits ``Model._model_hash`` + ``Model.compile`` out of ``facade.py`` so neither
file exceeds the Spec-4 500-line bound. The mixin operates on ``self._m`` (the
private :class:`HyperbolicModel`) and ``self.params``; the build engine is pulled
in LAZILY inside ``compile`` (toolchain/cache/abi/loader names), so importing
``pops.physics`` never loads it (Spec-4 import-graph rule).
"""
from .aux import aux_total_n_aux, roles_for
from .model import HyperbolicModel


class _FacadeCompileMixin:
    """Model-hash + compile half of the PDE facade (lazy codegen)."""

    def _model_hash(self):
        """Stable hash of the model: formulas (flux/eig/source/elliptic/primitives/cons_from) + roles +
        n_aux + NAMED params (m.params). Used to identify/reuse an already-compiled .so (cache key)
        and to trace the run. Delegates to the shared computation HyperbolicModel._model_hash, passing it
        the Param of the facade (otherwise two models differing only by a param would have the same hash)."""
        return self._m._model_hash(params=self.params)

    def compile(self, so_path=None, include=None, backend="auto", target="system", name=None,
                cxx=None, std=None, require_metadata=False, hoist_reciprocals=False):
        """Compiles the model into a CompiledModel (Phase A). Delegates the GENERATION + compilation to
        HyperbolicModel.compile (engines unchanged: compile_so / compile_aot / compile_native), then
        packages the .so with the already-known metadata (no re-reading of the .so).

        - ``backend``: "prototype" | "aot" | "production" (cf. HyperbolicModel.compile).
        - ``target``: "system" (default) | "amr_system" (DSL Phase D). "amr_system" requires
          backend="production" (the native loader inlines add_compiled_model(AmrSystem&), the only
          .so AMR path; cf. compile_or_jit) -> to be wired via AmrSystem.add_equation. Another backend
          with target="amr_system" raises ValueError (no AMR path outside native).

        NO ``device`` argument: the GPU/MPI/AMR capabilities are checked at wiring time
        (add_equation) / at execution, not frozen at compile time (DSL_MODEL_DESIGN.md point 7).

        ERGONOMICS (does not change the numerics):

        - ``include`` None -> auto-detected (pops_include()); passing include= remains possible;
        - ``so_path`` None -> .so in an out-of-source cache (pops_cache_dir()), file name keyed on
          model_hash (PARAMS INCLUDED) + abi_key (+ backend/target/name). Cache HIT (.so already present)
          -> reuse without recompilation; cache MISS (model/param/toolchain change) ->
          recompilation + storage. Passing so_path= forces that path and recompiles (backward-compat).

        Returns a CompiledModel carrying so_path, backend, target, adder, names/roles/gamma/n_aux/params,
        caps, abi_key, model_hash, cxx, std."""
        import os
        # Lazy codegen import (keeps pops.physics codegen-free at module load; Spec-4 rule):
        from pops.codegen.toolchain import (resolve_auto_backend, loader_cxx_std,
                                            _native_kokkos_compiler, _default_cxx,
                                            _native_feature_key, pops_include)
        from pops.codegen.cache import _cache_so_path, _record_so_backend
        from pops.codegen.abi import _abi_key_python
        from pops.codegen.compile import _BACKEND_CAPS
        from pops.codegen.loader import CompiledModel
        # 'auto' DEFAULT (ADC-63): production if toolchain parity is established, aot otherwise. The reason
        # is recorded on the CompiledModel (backend_auto_reason) -- never a silent choice.
        auto_reason = None
        if backend == "auto":
            backend, auto_reason = resolve_auto_backend(include)
        if backend not in HyperbolicModel._BACKENDS:
            raise ValueError("compile: unknown backend %r (expected %s + 'auto')"
                             % (backend, sorted(HyperbolicModel._BACKENDS)))
        if target not in ("system", "amr_system"):
            raise ValueError("compile: target 'system' | 'amr_system' (got %r)" % (target,))

        m = self._m
        # effective std: same per-backend default as HyperbolicModel.compile. The native one follows the
        # loader's standard (c++20 under Kokkos, c++23 otherwise, cf. loader_cxx_std); the others stay c++20.
        mode = HyperbolicModel._BACKENDS[backend][0]
        if target == "amr_system" and mode != "native":
            raise ValueError("compile: target='amr_system' only exists for backend='production' "
                             "(native AMR path); got backend=%r" % (backend,))
        eff_std = std if std is not None else (loader_cxx_std() if mode == "native" else "c++20")
        # native AND aot (mode "compile") compile the pops headers -> real Kokkos (compiler +
        # kokkos feature-key) so that the cache key MATCHES the produced .so (cf. compile_aot).
        kokkos_like = mode in ("native", "compile")
        eff_cxx = _native_kokkos_compiler(cxx) if kokkos_like else _default_cxx(cxx)
        if include is None:  # ergonomics: auto-detection of the pops headers folder
            include = pops_include()

        # Metadata guards BEFORE the cache (a HIT must not mask them; cf.
        # HyperbolicModel._check_require_metadata).
        m._check_require_metadata(require_metadata, backend)

        # PARAMS-INCLUDED model_hash (the one carried by the CompiledModel) AND the ABI key: both also
        # serve as cache keys, so we compute them here to reuse them (key/metadata consistency).
        model_hash = self._model_hash()
        abi_key = _abi_key_python(include, eff_cxx, eff_std)

        # OUT-OF-SOURCE cache when so_path is omitted: we RESOLVE the keyed path here (with the
        # params-included hash) and pass it explicitly to the engine -- the cache of HyperbolicModel.compile
        # would otherwise use the hash WITHOUT params (the Model facade adds the Param). HIT -> we skip the
        # compilation. Explicit so_path -> forced path, always recompiles (strict backward-compat).
        cache_hit = False
        if so_path is None:
            # kokkos feature-key in the key (cf. compile_native): a SERIAL .so is not reused
            # on a Kokkos module. MUST match the engine's key, otherwise repeated recompilations.
            cache_backend = (backend + ";" + _native_feature_key()) if kokkos_like else backend
            if hoist_reciprocals:  # distinct codegen -> distinct key (cf. HyperbolicModel.compile)
                cache_backend += ";hoist"
            so_path = _cache_so_path(model_hash, abi_key, cache_backend, target, name)
            cache_hit = os.path.exists(so_path)

        if cache_hit:
            out_path = so_path  # .so already compiled for this key: no recompilation
        else:
            # Compilation (engines unchanged, require_metadata/backend/target guards of
            # HyperbolicModel.compile: the loader emits pops_install_native_amr for target="amr_system").
            out_path = m.compile(so_path, include, backend=backend, name=name, cxx=cxx, std=std,
                                 require_metadata=require_metadata, target=target,
                                 hoist_reciprocals=hoist_reciprocals)
        # The keyed path (cache HIT) or the path retained by the engine carries the written backend: we
        # record it so a cross-backend reuse of the SAME path in this process is detected.
        _record_so_backend(out_path, backend)

        adder = HyperbolicModel.adder_for(backend)
        cons_roles = roles_for(m.cons_names, m.cons_roles)
        cm = CompiledModel(
            so_path=out_path, backend=backend, adder=adder, target=target,
            cons_names=m.cons_names, cons_roles=cons_roles, prim_names=m.prim_state,
            n_vars=m.n_vars, gamma=m.gamma, n_aux=aux_total_n_aux(m.aux_names, m.aux_extra_names),
            params=self.params, caps=_BACKEND_CAPS[backend],
            abi_key=abi_key, model_hash=model_hash,
            cxx=eff_cxx, std=eff_std, hllc=m._hllc,
            roe=(m._roe or getattr(m, '_roe_rows', None) is not None
                 or getattr(m, '_roe_jacobian', None) is not None),
            aux_extra_names=m.aux_extra_names,
            wave_speeds=(m._wave_speeds is not None or m._ws_jacobian is not None
                         or "p" in m.prim_defs))
        # Trace of the 'auto' policy (ADC-63): None if the backend was explicit. Diagnostic,
        # never a silent choice -- cm.backend says what was built, this says WHY.
        cm.backend_auto_reason = auto_reason
        return cm
