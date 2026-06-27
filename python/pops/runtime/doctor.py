"""Environment doctor + official capability matrix (Spec-4 PR-F).

``doctor()`` diagnoses the links the module AND the runtime DSL compilation depend on (the class
of "build environment != execution environment" bugs). ``capabilities()`` is the single
introspectable matrix of what each facade / geometry / backend supports. Both read ``_pops`` and
the codegen / physics layers, so they live in the runtime layer.
"""

from pops.runtime import threading as _threading
from pops.runtime.threading import has_kokkos


def doctor(verbose=True):
    """Diagnose the pops environment in ONE command : python -c "import pops; pops.doctor()".

    Checks each link on which the module AND the runtime compilation of the DSL depend (the class of
    bugs "build environment != execution environment", e.g. the `which c++` of a conda env
    that rejects -std=c++23). Returns a dict {check: (ok, detail)} ; verbose=True prints it."""
    import os
    import sys
    checks = {}

    # 1. interpreter + extension (cpython-3XY ABI trap)
    from pops import _pops
    so = getattr(_pops, "__file__", "?")
    checks["interpreteur"] = (True, "%s (%d.%d) ; extension %s"
                              % (sys.executable, sys.version_info[0], sys.version_info[1], so))

    # 2. numpy (required by the codegen IR host evaluator)
    try:
        import numpy
        checks["numpy"] = (True, numpy.__version__)
    except Exception as e:
        checks["numpy"] = (False, "ABSENT from this interpreter (%s) -> compiling a model will fail. "
                                  "Install numpy in THIS python." % e)

    # 3. compiled compute backend
    hk = has_kokkos()
    checks["kokkos"] = (hk is not False,
                        {True: "Kokkos module (multi-thread possible ; pops.set_threads active)",
                         False: "SERIAL module (set_threads has no effect ; rebuild preset python-parallel)",
                         None: "undetermined (old module without __has_kokkos__)"}[hk])

    # 4. runtime DSL compiler (the link of the -std=c++23 bug)
    try:
        from pops.codegen import toolchain as _tc
        from pops.codegen import abi as _abi
    except Exception as e:
        checks["dsl"] = (False, "import pops.codegen failed (%s)" % e)
        _tc = None
    if _tc is not None:
        baked = _tc.loader_cxx_compiler()
        cc = _tc._default_cxx(None)
        if not cc:
            checks["compilateur"] = (False, "NO C++ compiler found (POPS_CXX, module, PATH). "
                                            "Install Xcode CLT (macOS) or `conda install cxx-compiler`.")
        else:
            origin = ("$POPS_CXX" if os.environ.get("POPS_CXX") == cc
                      else "baked by the _pops build" if cc == baked else "PATH (which)")
            try:
                std = _tc._probe_cxx_std(cc, _tc.loader_cxx_std())
                checks["compilateur"] = (True, "%s [%s] ; -std=%s accepted" % (cc, origin, std))
            except RuntimeError as e:
                checks["compilateur"] = (False, str(e).splitlines()[0])
            if baked and cc != baked:
                checks["compilateur_abi"] = (False, "runtime compiler (%s) != build (%s) -> risk "
                                                    "of 'incompatible ABI' rejection on production "
                                                    "backend. export POPS_CXX=%r to force the one "
                                                    "from the build." % (cc, baked, baked))

        # 5. pops headers (production DSL : the signature must match the one baked into _pops)
        try:
            inc = _tc.pops_include()
            checks["include"] = (True, inc)
            # 5b. SYNCHRONIZATION headers <-> module (real bug : module built BEFORE a git pull ->
            # the DSL loader references C++ signatures absent from the old .so -> dlopen 'symbol
            # not found' cryptic). We compare the baked signature to the one of the current tree.
            baked_sig = _abi.module_header_signature()
            if baked_sig is not None:
                cur_sig = _tc.pops_header_signature(inc)
                if cur_sig == baked_sig:
                    checks["headers_sync"] = (True, "headers == module build (sig %s...)"
                                              % baked_sig[:12])
                else:
                    checks["headers_sync"] = (False, "headers MODIFIED since the _pops build "
                                                     "(stale module) -> rebuild : cmake --build "
                                                     "build-py --target _pops (otherwise : dlopen "
                                                     "'symbol not found' on production backend)")
        except RuntimeError as e:
            checks["include"] = (False, "pops headers not found (set POPS_INCLUDE) : %s" % e)

        # 5c. Kokkos root for the DSL production/aot backend (the tutorial's "no DSL backend" blocker).
        # adc_cpp is Kokkos-only : every DSL .so that includes the pops headers MUST compile against an
        # installed Kokkos (Serial is enough on CPU), found via POPS_KOKKOS_ROOT / Kokkos_ROOT.
        kroot = _tc._native_kokkos_root()
        if kroot is None:
            checks["kokkos_root"] = (False,
                "POPS_KOKKOS_ROOT / Kokkos_ROOT not set -> DSL backend='production'/'aot' cannot compile "
                "(the tutorial dead-ends on 'no DSL backend'). Fix (conda) :\n"
                "      conda env config vars set POPS_KOKKOS_ROOT=\"$CONDA_PREFIX\"\n"
                "      conda env config vars set Kokkos_ROOT=\"$CONDA_PREFIX\"\n"
                "      conda deactivate && conda activate pops")
        else:
            checks["kokkos_root"] = (True, kroot)
            # 5d. A CUDA Kokkos on a host without nvcc breaks BOTH `pip install .` (find_package picks it
            # -> nvcc) AND the production .so. On a CPU host, install the CPU Kokkos variant instead.
            import shutil
            cuda = False
            try:
                with open(os.path.join(kroot, "include", "KokkosCore_config.h")) as _f:
                    cuda = any(line.startswith("#define KOKKOS_ENABLE_CUDA") for line in _f)
            except OSError:
                pass
            if cuda and shutil.which("nvcc") is None:
                checks["kokkos_cuda"] = (False,
                    "Kokkos at %s is a CUDA build but nvcc is not on PATH -> `pip install .` fails "
                    "'Could not find nvcc'. Fix for a CPU host, recreate the env with CPU Kokkos :\n"
                    "      CONDA_OVERRIDE_CUDA=\"\" bash scripts/setup_env.sh" % kroot)

    # 6. current threads
    checks["threads"] = (True, "OMP_NUM_THREADS=%s ; first System created=%s"
                         % (os.environ.get("OMP_NUM_THREADS", "(default)"),
                            _threading._first_system_built))

    # 7. POPS_JIT_BACKDOOR (Spec 5 sec.12.4, criterion #48): the UNSAFE debug gate must never be
    # silently honored. Surface it loudly here (in addition to compiled.inspect()) so a stray export
    # is visible at a glance. OK when unset / disabled; FAIL (loud) when enabled -- a debug-only
    # escape hatch in a healthy environment. No backdoor behavior is wired; this is the guard only.
    # Read the env directly (no codegen import, so doctor stays lightweight even without numpy); the
    # truthy convention mirrors pops.codegen.env.jit_backdoor_enabled.
    _backdoor = os.environ.get("POPS_JIT_BACKDOOR", "").strip().lower() in ("1", "on", "true",
                                                                            "yes", "y")
    if _backdoor:
        checks["jit_backdoor"] = (False, "POPS_JIT_BACKDOOR is SET -> the UNSAFE debug JIT gate is "
                                         "ENABLED. Never set this in production; unset it to disable.")
    else:
        checks["jit_backdoor"] = (True, "disabled (POPS_JIT_BACKDOOR unset -- the safe default)")

    if verbose:
        for cname, (ok, detail) in checks.items():
            print("[%s] %-16s %s" % ("OK " if ok else "FAIL", cname, detail))
        if all(ok for ok, _ in checks.values()):
            print("=> healthy environment : module importable, DSL compilable, ABI coherent.")
        else:
            print("=> fix the FAILs above before using the DSL backend='production'.")
    return checks


def capabilities():
    """OFFICIAL MATRIX of capabilities by facade / geometry / backend (audit 2026-06, wave 2).

    SINGLE source of truth consultable by scripts and docs (the audits showed that System,
    AMR, polar and the DSL backends diverged silently). The entries reflect the GATES
    actually coded (make_block / dispatch_amr_* / block_builder_polar / dsl._BACKENDS) ; the
    combinations outside the matrix raise an explicit error on the C++ side (never a silent ignore).
    """
    from pops import _pops as _pops_mod  # ADC-291: read the aux limit from the SINGLE C++ source
    from pops.physics.aux import AUX_NAMED_MAX  # fallback mirror (no second hardcoded literal)
    aux_max_extra = int(getattr(_pops_mod, "__aux_max_extra__", AUX_NAMED_MAX))
    return {
        # Spatial dimension of the core (ADC-294 / ADR-0001 Decision 1). The solver is structurally
        # 2D: a load-bearing invariant baked into the data layout (Fab2D operator()(i, j, c)), the
        # paired FaceFluxX / FaceFluxY kernels, the 2-component momentum, the 5-point Poisson and the
        # Box2D / Geometry index space -- not a naming detail. Published as an explicit, introspectable
        # structured scalar (hard limits are scalars, not prose) so scripts and the limitations doc can
        # key off it. The polar mesh is a second GEOMETRY at the SAME dimension ((r, theta) is a
        # 2-index Box2D), so this is a separate top-level key, NOT nested under "geometry". An ND core
        # (BoxND / GeometryND) is deferred to a future milestone; see
        # docs/sphinx/reference/known-limitations.md and include/pops/mesh/box2d.hpp.
        "dimension": 2,
        "riemann": {
            "system_cartesian": ["rusanov", "hll", "hllc", "roe"],
            "system_polar": ["rusanov", "hll"],
            "amr": ["rusanov", "hll", "hllc", "roe"],
            "notes": {
                "rusanov": "minimal generic (max_wave_speed only)",
                "hll": "generic with signed waves (model.wave_speeds ; DSL : m.wave_speeds(x=, y=) "
                       "explicit WITHOUT primitive 'p', or historical path eigenvalues + 'p') ; "
                       "polar : eligible for the isothermal fluid (IsothermalFluxPolar), not for "
                       "scalar ExB (no wave_speeds) -- same gate as the cartesian one",
                "hllc": "canonical 2D Euler (4 var + pressure) OR model capability "
                        "HasHLLCStructure -- emitted by the DSL via m.enable_hllc() (roles + 'p', "
                        "including 3-var non Euler, passive advected scalars)",
                "roe": "canonical 2D perfect-gas Euler OR model capability HasRoeDissipation "
                       "-- TWO DSL paths : (a) m.enable_roe() generated from the roles (roles + "
                       "'p' : with Energy = transcribed canonical algebra, without Energy = "
                       "c=sqrt(p/rho) Roe average, passive scalars on the entropy wave) ; (b) "
                       "m.roe_dissipation(x=, y=) PROVIDED by the user (own eigenstructure, "
                       "left()/right() of the two states, helper m.flux_jacobian auto-derived). Paths "
                       "exclusive (a single provider of the hook). has_roe covers both",
            },
        },
        "time": {
            "system": ["explicit (ssprk2|ssprk3)", "imex (= SourceImplicitBE)",
                       "imexrk_ars222 (IMEX-RK family, ARS(2,2,2) scheme, order 2 ; cartesian only ; "
                       "fully implicit source)",
                       "split lie|strang + CondensedSchur"],
            "amr": ["explicit (forward Euler per substep)", "ssprk3 (order 3 + reflux per stage)",
                    "imex (= SourceImplicitBE)",
                    "split lie|strang + CondensedSchur (mono-block, coarse)"],
            "system_polar": ["explicit (ssprk2|ssprk3)", "split + polar CondensedSchur"],
            "newton_options": "options (max_iters/tol/fd_eps/damping/fail_policy) : System + AMR "
                              "mono-block AND native multi-block (.so loaders : explicit rejection) ; "
                              "analytic jacobian via m.source_jacobian ; newton_diagnostics/"
                              "newton_report : System + AMR native multi-block (mono-block AMR and "
                              ".so loaders : explicit rejection)",
        },
        "stability_policy": {
            "system": ["transport (max_wave_speed | stability_speed)", "source_frequency",
                       "stability_dt", "coupled_source.frequency", "add_dt_bound (global, "
                       "all_reduce_min)", "last_dt_bound"],
            "amr": ["transport (max_wave_speed | stability_speed)", "source_frequency",
                    "stability_dt", "coupled_source.frequency (multi-block)", "add_dt_bound",
                    "last_dt_bound"],
            "system_polar": ["transport (max_wave_speed | stability_speed)", "source_frequency",
                             "stability_dt", "coupled_source.frequency", "add_dt_bound",
                             "last_dt_bound"],
        },
        "poisson": {
            "system_cartesian": ["geometric_mg (wall, eps(x), aniso, screened)",
                                 "fft (periodic, n = 2^k, constant eps, mono-box)",
                                 "fft_spectral (same as fft, continuous spectral symbol)"],
            "system_polar": ["polar direct (mono-rank, one box) -- clear UPSTREAM REJECT if theta_boxes>1"],
            "amr": ["geometric_mg only ; rhs charge_density|composite"],
        },
        "geometry": {
            "system_cartesian": "square n x n ; mono-box (multi-box = AmrSystem or MPI mono-box)",
            "system_polar": "ring (r, theta) global ; theta_boxes=1 mono-box (default) OR "
                            "theta_boxes>1 split into theta bands (divides ntheta). MATRIX "
                            "multi-box (ADC-67) : TRANSPORT (assemble_rhs_polar + fill_ghosts "
                            "collective) multi-box OK ; polar Poisson DIRECT mono-box only (upstream "
                            "reject if theta_boxes>1) ; polar tensor Schur stage multi-box. "
                            "get/set state (and eval_rhs/density) reconstruct the global ring "
                            "multi-box ; mono-rank (the direct Poisson refuses MPI).",
            "amr": "hierarchy of levels (BoxArray per level, dynamic regrid) ; "
                   "refinement_ratio = 2 only (single native AMR invariant, centralized in "
                   "include/pops/amr/refinement_ratio.hpp ; a non-2 ratio is rejected at "
                   "hierarchy construction, not silently mis-coarsened)",
        },
        "schur": {
            "system_cartesian": "complete ; configurable roles/fields (density=/momentum=/energy=/"
                                "magnetic_field=), configurable krylov_tol/max_iters",
            "system_polar": "configurable roles (density=/momentum=/energy=, wave 3) ; "
                            "magnetic_field freezes B_z ; multi-box C++ solver, facade one global box",
            "amr": "mono-block ; roles + configurable krylov_tol/max_iters (wave 3, "
                   "magnetic_field freezes coarse B_z) ; complete mono-level + composite Phase 4a "
                   "(2 levels, 1..N disjoint non-adjacent fine patches, mono-rank) ; Phase 4b "
                   "(adjacent patches/>2 levels/MPI/multi-block) to be done",
        },
        "backends_dsl": {
            "default": "auto (ADC-63) : production if toolchain parity established (module loaded + "
                       "baked compiler + matching headers), aot otherwise ; reason set on "
                       "CompiledModel.backend_auto_reason ; explicit backend = short-circuit",
            "prototype": {"adder": "add_dynamic_block", "riemann": ["rusanov"],
                          "limiter": ["none", "minmod", "vanleer"], "stride": False,
                          "evolve_false": False, "mpi": False, "amr": False},
            "aot": {"adder": "add_compiled_block", "riemann": ["rusanov", "hll", "hllc", "roe"],
                    "limiter": ["none", "minmod", "vanleer", "weno5"], "stride": False,
                    "evolve_false": False, "mpi": False, "amr": False,
                    "runtime_params": True},
            "production": {"adder": "add_native_block",
                           "riemann": ["rusanov", "hll", "hllc", "roe"],
                           "limiter": ["none", "minmod", "vanleer", "weno5"], "stride": True,
                           "evolve_false": True, "mpi": True, "amr": "target='amr_system'",
                           "stability_hooks": True},
        },
        "io": {
            "write": ["vtk (.vti cartesian)", "npz",
                      "hdf5 (h5py optional, rank-0 gather aggregation by default)",
                      "parallel hdf5 (write(parallel=True) : hyperslabs per rank via h5py mpio + "
                      "mpi4py ; opt-in, clear error if h5py without MPI ; true parallelism in "
                      "MULTI-BOX, mono-box cartesian System)",
                      "AmrSystem.write npz/vtk (coarse + patch rectangles)"],
            "checkpoint_restart": "v1 npz mono-rank/rank-0 gather (System ; states + phi + t/macro_step ; "
                                  "composition replayed by the script ; bit-identical resume ; "
                                  "checkpoint(parallel=True) raises, stays rank-0 gather npz) ; "
                                  "AMR mono-block mono-rank regrid_every=0 (ADC-65 : complete "
                                  "conservative state per level + phi warm-start + imposed hierarchy ; resume "
                                  "bit-identical) ; AMR multi-block / np>1 and parallel HDF5 "
                                  "CHECKPOINT = follow-up (docs/IO_CHECKPOINT_PLAN.md ; explicit rejections)",
        },
        "amr_layout": {
            "set_conservative_state": "mono-block AND native multi-block (wave 3 ; .so loaders : "
                                      "explicit rejection)",
        },
        "regrid": {
            # ADC-296 / ADR-0001 Decision 5. The MULTI-BLOCK AMR regrid variable is selectable PER BLOCK
            # by name or physical role (set_refinement(threshold, variable=|role=)); default = component
            # 0 (historical density), bit-identical 1e30 no-op. A block lacking the requested name/role
            # raises at build (no silent component-0 fallback). Mono-block (AmrCouplerMP) and the compiled
            # .so loader refine on component 0 ONLY (a non-default selector is rejected there).
            "variable_selector": ["component_0", "by_name", "by_role"],
            "multi_block": "component_0 | by_name (variable=) | by_role (role=)",
            "mono_block": "component_0 only (selector rejected)",
            "compiled_so": "component_0 only (selector rejected)",
            "phi_gradient": "set_phi_refinement(grad_threshold) : |grad phi|, multi-block, unioned",
        },
        "aux": {
            "canonical": "phi/grad_x/grad_y (base) + B_z (set_magnetic_field) + T_e "
                         "(set_electron_temperature_from), closed list POPS_AUX_FIELDS / AUX_CANONICAL "
                         "(C++ name table pops/core/aux_names.hpp, mirror of Python AUX_CANONICAL)",
            "named": {
                # Model-declared NAMED aux fields (ADC-70 phase 1 + ADC-291 phase 2): m.aux_field('name')
                # reserves component AUX_NAMED_BASE + k (read in C++ via aux.extra_field(k));
                # set_aux_field(block, name, array) carries the static field. STATIC + persistent.
                "backends": ["system_cartesian", "system_polar", "amr_single_block",
                             "amr_multi_block"],
                # The ONLY remaining compile-time aux limit, declarative + introspectable (= C++
                # kAuxMaxExtra, mirrored by dsl.AUX_NAMED_MAX ; test_capabilities.py pins the match).
                "limit": aux_max_extra,
                # Aux ghost width is fixed at 1 cell (the halo EXCHANGE is already component-generic, so
                # a named field participates ; a per-field CONFIGURABLE radius is a follow-up).
                "halo_radius": 1,
                "persistent": True,
                # Per-field aux HALO/BC policy (ADC-369): a named field can declare its own ghost BC via
                # pops.AuxHalo(kind, value), applied to the NON-PERIODIC faces (periodic faces -- periodic
                # domain, polar theta -- keep their wrap). Uniform over the 4 faces; per-face asymmetric
                # BC is a follow-up. Default (no halo) inherits the shared aux BC, bit-identical.
                "halo_policy": {
                    "kinds": ["inherit", "foextrap", "dirichlet"],
                    "faces": "uniform (non-periodic faces ; periodic faces keep their wrap)",
                    "backends": ["system_cartesian", "system_polar", "amr_coarse"],
                },
            },
            "followups": "per-field CONFIGURABLE aux halo radius (today fixed at 1) ; named aux on the "
                         "AMR path needs backend='production' target='amr_system', on polar a "
                         "System+AOT compiled block (the in-AMR compiled .so is mono-level) ; the "
                         "opt-in single-block composite-FAC Poisson path (set_composite_poisson, not "
                         "facade-reachable) does not yet carry named aux to the fine level",
        },
    }
