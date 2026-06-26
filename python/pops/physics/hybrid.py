"""Hybrid model composer: native + DSL bricks into one CompositeModel ``.so``.

:class:`HybridModel` takes three slots (transport, source, elliptic), each a
:class:`pops.physics.bricks.NativeBrick` or :class:`pops.physics.bricks.CompiledBrick`,
assembles a mixed ``pops::CompositeModel<...>`` and compiles it into ONE ``.so``
(prototype/aot/production backends).

Import-graph rule (Spec 4): module-scope imports stay within :mod:`pops.physics`;
all toolchain/cache/abi/loader names are imported LAZILY inside ``compile`` /
``_compiled_model`` (no codegen at ``pops.physics`` load).
"""
from .aux import roles_for
from .bricks import CompiledBrick, NativeBrick
from .model import HyperbolicModel


class HybridModel:
    """Composer of a HYBRID model: three slots (transport, source, elliptic), each provided by a
    NATIVE brick (NativeBrick) or a DSL brick (CompiledBrick). Assembles a mixed pops::CompositeModel<...>
    and compiles it into ONE .so (prototype: backend 'aot'). Returns a CompiledModel pluggable via
    System.add_equation (adder add_compiled_block).

    The transport (hyperbolic) slot FIXES the layout: n_vars, conservative names, primitives, gamma. A
    DSL source/elliptic brick must declare the SAME n_vars ; a templated native brick (source/
    elliptic) only needs to satisfy its min_vars (e.g. PotentialForce requires >= 3 variables)."""

    def __init__(self, transport, source, elliptic, name="hybrid"):
        self.name = name
        hyp = self._norm(transport, "hyperbolic", "NatHyp")
        src = self._norm(source, "source", "NatSrc")
        ell = self._norm(elliptic, "elliptic", "NatEll")

        nv = hyp["n_vars"]
        if nv is None:
            raise ValueError("HybridModel: the transport slot must fix n_vars (hyperbolic brick)")
        for role, slot in (("source", src), ("elliptic", ell)):
            if slot["provider"] == "dsl":
                if slot["n_vars"] != nv:
                    raise ValueError(
                        "HybridModel: the DSL brick %s declares %d variables but the transport has %d ; "
                        "align conservative_vars(...)" % (role, slot["n_vars"], nv))
            elif slot["min_vars"] > nv:
                raise ValueError(
                    "HybridModel: the native brick %s requires >= %d variables (transport=%d) ; e.g. a "
                    "fluid force makes no sense on a scalar transport"
                    % (role, slot["min_vars"], nv))

        self.n_vars = nv
        self.cons_names = list(hyp["cons_names"])
        self.cons_roles = list(hyp["cons_roles"])
        self.prim_names = list(hyp["prim_names"])
        self.gamma = hyp["gamma"]
        self.n_aux = max(hyp["n_aux"], src["n_aux"], ell["n_aux"])
        self._has_wave_speeds = bool(hyp.get("wave_speeds", True))
        self._slots = (hyp, src, ell)

    @staticmethod
    def _norm(prov, role, native_struct_name):
        """Normalize a slot (DSL CompiledBrick or NativeBrick) into a common dict."""
        if isinstance(prov, CompiledBrick):
            if prov.kind != role:
                raise ValueError("HybridModel: DSL brick of type %r placed in slot %r"
                                 % (prov.kind, role))
            d = dict(provider="dsl", struct_text=prov.struct_src, type_name=prov.type_name,
                     n_vars=prov.n_vars, min_vars=prov.n_vars, n_aux=prov.n_aux)
            if role == "hyperbolic":
                d.update(cons_names=prov.cons_names, cons_roles=prov.cons_roles,
                         prim_names=prov.prim_names, gamma=prov.gamma,
                         wave_speeds=getattr(prov, "has_wave_speeds", True))
            return d
        if isinstance(prov, NativeBrick):
            if prov.kind != role:
                raise ValueError("HybridModel: native brick of type %r placed in slot %r"
                                 % (prov.kind, role))
            d = dict(provider="native", struct_text=prov.emit(native_struct_name),
                     type_name="pops_generated::" + native_struct_name,
                     n_vars=prov.n_vars, min_vars=prov.min_vars, n_aux=prov.n_aux)
            if role == "hyperbolic":
                names = prov.var_names or []
                d.update(cons_names=list(names), cons_roles=roles_for(names),
                         prim_names=list(prov.prim_names or names), gamma=prov.gamma,
                         wave_speeds=True)  # native: unknown, the C++ requires-gate decides (historical)
            return d
        raise TypeError("HybridModel: slot %r must be a native brick (pops.* / NativeBrick) or a "
                        "compiled DSL brick (CompiledBrick) ; got %r" % (role, type(prov).__name__))

    def _emit_aot_source(self):
        """C++ source of the hybrid composite .so, behind the extern \"C\" ABI of compiled_block_abi.hpp
        (aot backend: same flat ABI as emit_cpp_aot_source). The bricks (generated DSL or native binding
        structs) are stitched together, then assembled into pops::CompositeModel<...>."""
        hyp, src, ell = self._slots
        parts = ['#include <pops/runtime/builders/compiled/compiled_block_abi.hpp>\n',
                 '#include <pops/physics/bricks/bricks.hpp>\n',   # CompositeModel + native bricks
                 '#include <pops/core/state/variables.hpp>\n']   # POPS_EXPORT_BLOCK_METADATA / _GAMMA
        for slot in self._slots:
            if slot["struct_text"]:
                parts.append(slot["struct_text"])
        parts.append('\nnamespace pops_generated { using AotModel = pops::CompositeModel<%s, %s, %s>; }\n'
                     % (hyp["type_name"], src["type_name"], ell["type_name"]))
        parts.append('POPS_DEFINE_COMPILED_BLOCK(pops_generated::AotModel)\n')
        parts.append('POPS_EXPORT_BLOCK_METADATA(pops_generated::AotModel)\n')
        if self.gamma is not None:
            parts.append('POPS_EXPORT_BLOCK_GAMMA(%r)\n' % self.gamma)
        return "".join(parts)

    def _model_hash(self):
        """Stable hash of the composite: provider + type + generated text of each slot (the text encodes
        the DSL formulas and the baked native parameters). Used as a cache key."""
        import hashlib
        parts = ["hybrid", self.name]
        for slot in self._slots:
            parts.append("%s|%s|%s" % (slot["provider"], slot["type_name"], slot.get("struct_text", "")))
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()

    def _bricks_and_composite(self):
        """C++ text of the stitched bricks (generated DSL + native binding structs) + composite type."""
        hyp, src, ell = self._slots
        bricks = "".join(s["struct_text"] for s in self._slots if s["struct_text"])
        composite = ("pops::CompositeModel<%s, %s, %s>"
                     % (hyp["type_name"], src["type_name"], ell["type_name"]))
        return bricks, composite

    def _emit_metadata(self, alias):
        """ABI metadata symbols (names/roles from conservative_vars, optional gamma), SHARED
        by the backends. @p alias: an alias WITHOUT a top-level comma (the preprocessor splits
        macro arguments on commas)."""
        out = '\nPOPS_EXPORT_BLOCK_METADATA(%s)\n' % alias
        if self.gamma is not None:
            out += 'POPS_EXPORT_BLOCK_GAMMA(%r)\n' % self.gamma
        return out

    def _emit_jit_source(self):
        """Source of the JIT library (backend 'prototype'): the hybrid composite behind an
        extern \"C\" factory (pops_make_model via pops::ModelAdapter). Host VIRTUAL dispatch (order-1
        Rusanov residual): fast iteration, to be plugged via System.add_dynamic_block. Hybrid
        counterpart of emit_cpp_so_source."""
        bricks, composite = self._bricks_and_composite()
        return ('#include <pops/runtime/dynamic/dynamic_model.hpp>\n'
                '#include <pops/physics/bricks/bricks.hpp>\n'
                '#include <pops/core/state/variables.hpp>\n'
                + bricks
                + '\nnamespace pops_generated { using JitModel = %s; }\n' % composite
                + 'extern "C" int pops_model_nvars() { return %d; }\n' % self.n_vars
                + 'extern "C" void* pops_make_model() { return new pops::ModelAdapter<pops_generated::JitModel>(); }\n'
                + 'extern "C" void pops_destroy_model(void* p) { delete static_cast<pops::IModel<%d>*>(p); }\n'
                % self.n_vars
                + self._emit_metadata("pops_generated::JitModel"))

    def _emit_native_source(self, target="system"):
        """C++ source of the NATIVE LOADER (backend 'production'): the hybrid composite as CompositeModel<...>
        behind a THIN extern \"C\" ABI. Like emit_cpp_native_loader, the .so does NOT carry the
        numerics: it INSTALLS the generated model as a native block of the facade via add_compiled_model<>,
        which builds the closures on the REAL CONTEXT of the facade -> same path as add_block,
        ZERO-COPY (MPI by construction, device-clean). pops_native_abi_key() freezes the ABI key at
        compile time, compared against the module's abi_key() by add_native_block (explicit rejection if
        headers/compiler/std diverge).

        @p target: 'system' -> pops_install_native (System&, evolve) ; 'amr_system' ->
        pops_install_native_amr (AmrSystem&, without evolve) inline add_compiled_model(AmrSystem&): the block
        runs the SAME AMR hierarchy as AmrSystem.add_block (reflux, regrid). DISTINCT symbols per
        target (a System loader is not pluggable onto AmrSystem.add_native_block, and vice versa)."""
        if target not in ("system", "amr_system"):
            raise ValueError("_emit_native_source: target 'system' | 'amr_system' (got %r)" % (target,))
        bricks, composite = self._bricks_and_composite()
        head = ('#include <pops/runtime/dynamic/abi_key.hpp>\n'        # POPS_ABI_KEY_LITERAL (key frozen at compile time)
                '#include <pops/physics/bricks/bricks.hpp>\n'         # CompositeModel + native bricks
                '#include <pops/core/state/variables.hpp>\n'
                '#include <string>\n')
        # Header template of the target (selective: do not pull the AMR machinery into a System loader).
        head += ('#include <pops/runtime/builders/compiled/dsl_block.hpp>\n' if target == "system"
                 else '#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>\n')
        # Preprocessor LITERAL, no call to abi_key_string(): an inline would be interposed
        # (ELF/RTLD_GLOBAL) toward the module's copy -> module key returned -> tautological guard.
        key = ('#if defined(_WIN32)\n'
               '#define POPS_LOADER_API extern "C" __declspec(dllexport)\n'
               '#else\n'
               '#define POPS_LOADER_API extern "C"\n'
               '#endif\n'
               'POPS_LOADER_API const char* pops_native_abi_key() {\n'
               '  return POPS_ABI_KEY_LITERAL;\n'
               '}\n')
        if target == "system":
            # pos_floor (ADC-76, Zhang-Shu positivity limiter): final flat argument, marshaled
            # down to the loader's make_block via add_compiled_model. Old signature = old .so =
            # rejected by the ABI key (the headers changed), never a wrong argument layout.
            install = ('POPS_LOADER_API void pops_install_native(void* sys, const char* name, const char* limiter,\n'
                       '                                    const char* riemann, const char* recon,\n'
                       '                                    const char* time, double gamma, int substeps,\n'
                       '                                    int evolve, int stride, double pos_floor) {\n'
                       '  pops::System* s = reinterpret_cast<pops::System*>(sys);\n'
                       '  pops::add_compiled_model<pops_generated::ProdModel>(*s, name, pops_generated::ProdModel{},\n'
                       '                                                    limiter, riemann, recon, time, gamma,\n'
                       '                                                    substeps, evolve != 0, stride,\n'
                       '                                                    pos_floor);\n'
                       '}\n')
        else:  # amr_system: AmrSystem overload (no evolve parameter, mono-block AMR)
            install = ('POPS_LOADER_API void pops_install_native_amr(void* sys, const char* name,\n'
                       '                                        const char* limiter, const char* riemann,\n'
                       '                                        const char* recon, const char* time,\n'
                       '                                        double gamma, int substeps) {\n'
                       '  pops::AmrSystem* s = reinterpret_cast<pops::AmrSystem*>(sys);\n'
                       '  pops::add_compiled_model<pops_generated::ProdModel>(*s, name, pops_generated::ProdModel{},\n'
                       '                                                    limiter, riemann, recon, time, gamma,\n'
                       '                                                    substeps);\n'
                       '}\n')
        return (head + bricks
                + '\nnamespace pops_generated { using ProdModel = %s; }\n' % composite
                + key + install + self._emit_metadata("pops_generated::ProdModel"))

    def compile(self, backend="aot", so_path=None, include=None, name=None, cxx=None, std=None,
                target="system"):
        """Compile the hybrid composite into a CompiledModel.

        ``backend`` :

        - 'prototype' -> add_dynamic_block: JIT, host VIRTUAL dispatch (order-1 Rusanov), fast
          iteration ; no MPI/AMR, no HLLC/Roe flux nor primitive recon ;
        - 'aot' -> add_compiled_block: self-sufficient .so (flat ABI, host-marshaled), mono-rank
          production path ; without MPI/AMR ;
        - 'production' -> add_native_block: zero-copy native loader that inlines add_compiled_model<>, SAME
          path as add_block (closures on the facade's real context), MPI by construction.
          The names/roles/gamma come from the .so metadata (no names=).

        ``target`` : 'system' (default) | 'amr_system'. 'amr_system' REQUIRES backend='production': the loader
        inlines add_compiled_model(AmrSystem&) (symbol pops_install_native_amr), the only AMR .so path ; to be
        plugged via AmrSystem.add_equation. The other backends have no AMR counterpart.

        so_path None -> out-of-source cache (key = model_hash + abi_key + backend + target)."""
        import os
        import sys
        import tempfile
        # Lazy codegen import (keeps pops.physics codegen-free at module load; Spec-4 rule):
        from pops.codegen.toolchain import (pops_include, loader_cxx_std, _native_kokkos_compiler,
                                            _default_cxx, _check_headers_match_module,
                                            _warn_kokkos_parity, _probe_cxx_std,
                                            _native_feature_key, _native_kokkos_flags,
                                            _native_kokkos_root, _run_compile, pops_header_signature)
        from pops.codegen.cache import (_cache_so_path, _record_so_backend,
                                        _backend_distinct_so_path, _dsl_optflags)
        from pops.codegen.abi import _abi_key_python
        if backend not in ("prototype", "aot", "production"):
            raise ValueError("HybridModel.compile: backend 'prototype' | 'aot' | 'production' (got %r)"
                             % (backend,))
        if target not in ("system", "amr_system"):
            raise ValueError("HybridModel.compile: target 'system' | 'amr_system' (got %r)" % (target,))
        if target == "amr_system" and backend != "production":
            raise ValueError("HybridModel.compile: target='amr_system' only exists for "
                             "backend='production' (native AMR path) ; got backend=%r" % (backend,))
        mode = {"prototype": "jit", "aot": "aot", "production": "native"}[backend]
        if include is None:
            include = pops_include()
        if std is None:  # the native loader shares the module ABI (std derived from the loader: c++20 under
            # Kokkos, c++23 otherwise, cf. loader_cxx_std/compile_native); jit/aot stay at c++20.
            std = loader_cxx_std() if mode == "native" else "c++20"
        # NATIVE (production) AND AOT: compiler following the Kokkos backend (g++ by default,
        # nvcc_wrapper if explicit), Kokkos flags without linking libkokkos (single runtime), feature-key
        # kokkos in the cache. KOKKOS-ONLY: the hybrid aot includes the pops headers
        # (compiled_block_abi.hpp -> multifab/for_each) which require POPS_HAS_KOKKOS, same flags as
        # compile_aot; only the jit (prototype) stays pure host (-O2, dynamic_model/bricks without
        # multifab). kokkos_like also serves the cache key.
        native = (mode == "native")
        kokkos_like = native or mode == "aot"
        if mode == "aot" and _native_kokkos_root() is None:
            raise RuntimeError(
                "HybridModel.compile: adc_cpp is Kokkos-only -- the AOT model includes the pops "
                "headers which require Kokkos. Point to an installed Kokkos via POPS_KOKKOS_ROOT (or "
                "Kokkos_ROOT), e.g. `export POPS_KOKKOS_ROOT=/path/to/kokkos` (Serial is enough "
                "on CPU). Run `python -c \"import pops; pops.doctor()\"` for a full diagnosis.")
        if native:  # pre-dlopen guard: headers != build of _pops -> clear remedy (cf. compile_native)
            _check_headers_match_module(include)
            _warn_kokkos_parity()
        eff_cxx = _native_kokkos_compiler(cxx) if kokkos_like else _default_cxx(cxx)
        if not eff_cxx:
            raise RuntimeError("HybridModel.compile: no C++ compiler found")
        std = _probe_cxx_std(eff_cxx, std)  # ACTIONABLE error if the std is not supported
        model_hash = self._model_hash()
        abi_key = _abi_key_python(include, eff_cxx, std)
        if so_path is None:
            cache_backend = (("hybrid-" + backend + ";" + _native_feature_key()) if kokkos_like
                             else "hybrid-" + backend)
            so_path = _cache_so_path(model_hash, abi_key, cache_backend, target, name)
            if os.path.exists(so_path):
                _record_so_backend(so_path, "hybrid-" + backend)
                return self._compiled_model(so_path, backend, target, abi_key, model_hash, eff_cxx, std)
        else:
            # Explicit so_path: avoid the dlopen handle cache re-serving ANOTHER backend already loaded at
            # this path in the process (cf. _backend_distinct_so_path). The hybrid backend is distinct from
            # the non-hybrid backend of the same name (different ABI) -> 'hybrid-' prefix.
            so_path = _backend_distinct_so_path(so_path, "hybrid-" + backend)

        # aot AND native run the production path -> same optimization flags (cf. _dsl_optflags);
        # only the jit/prototype stays at -O2 (Rusanov host residue, perf out of scope).
        optflags = _dsl_optflags() if kokkos_like else ["-O2"]
        flags = ["-shared", "-fPIC", "-std=" + std, *optflags]
        kokkos_link_flags = []
        if mode == "jit":
            source = self._emit_jit_source()
        elif mode == "aot":
            # Like compile_aot: Kokkos flags without linking libkokkos (the _pops module has already loaded the
            # runtime, singleton), undefined symbols resolved at load time; Apple-ld then requires
            # -undefined dynamic_lookup (on ELF/Linux -shared already allows them).
            source = self._emit_aot_source()
            kokkos_compile_flags, kokkos_link_flags = _native_kokkos_flags()
            flags += kokkos_compile_flags
            if sys.platform == "darwin":
                flags += ["-undefined", "dynamic_lookup"]
        else:  # native: header signature + Kokkos backend parity (cf. compile_native / _native_kokkos_flags)
            source = self._emit_native_source(target=target)  # undefined symbols resolved at load time (_pops module)
            flags.append('-DPOPS_HEADER_SIG="%s"' % pops_header_signature(include))
            kokkos_compile_flags, kokkos_link_flags = _native_kokkos_flags()
            flags += kokkos_compile_flags
            if sys.platform == "darwin":  # Apple-ld: explicitly allow undefined symbols (cf. compile_native)
                flags += ["-undefined", "dynamic_lookup"]
        with tempfile.TemporaryDirectory() as tmp:
            cpp = os.path.join(tmp, "model_hybrid.cpp")
            with open(cpp, "w") as f:
                f.write(source)
            _run_compile([eff_cxx, *flags, "-I", include, cpp, "-o", so_path, *kokkos_link_flags],
                         "HybridModel, backend " + backend)
        _record_so_backend(so_path, "hybrid-" + backend)
        return self._compiled_model(so_path, backend, target, abi_key, model_hash, eff_cxx, std)

    def _compiled_model(self, so_path, backend, target, abi_key, model_hash, cxx, std):
        from pops.codegen.loader import CompiledModel  # lazy: physics stays codegen-free
        from pops.codegen.compile import _BACKEND_CAPS
        return CompiledModel(
            so_path=so_path, backend=backend, adder=HyperbolicModel.adder_for(backend),
            target=target, cons_names=self.cons_names, cons_roles=self.cons_roles,
            prim_names=self.prim_names, n_vars=self.n_vars, gamma=self.gamma, n_aux=self.n_aux,
            params={}, caps=_BACKEND_CAPS[backend], abi_key=abi_key, model_hash=model_hash,
            cxx=cxx, std=std, hllc=getattr(self, "_hllc", False),
            roe=getattr(self, "_roe", False),
            wave_speeds=getattr(self, "_has_wave_speeds", True))
