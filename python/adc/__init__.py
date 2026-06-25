"""adc : Python bindings for the adc_cpp library.

The core exposes generic compiled BRICKS (transport, source, elliptic right-hand
side) ; a MODEL is a composition of bricks, named on the application side. Python
composes the bricks (objects), the cell-by-cell computation stays in compiled C++ (no
numpy, GPU/MPI preserved).

    import adc
    sim = adc.System(n=192, periodic=False)
    sim.add_block(
        "ne",
        model=adc.Model(state=adc.Scalar(), transport=adc.ExB(B0=1.0),
                        source=adc.NoSource(), elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0)),
        spatial=adc.Spatial(minmod=True), time=adc.Explicit())
    sim.set_poisson(bc="dirichlet", wall="circle", wall_radius=0.40)
    sim.set_density("ne", ne_numpy)
    sim.step_cfl(0.4)

The scenario names (diocotron, electron_euler...) are compositions on the
application side (see adc_cases). No scenario name here.
"""

import os as _os
import sys as _sys

# The "production" DSL backend then loads a .so loader via dlopen. This loader
# resolves C++ symbols exported by the _adc extension (System::install_block,
# grid_context, ensure_aux_width, etc.). CPython normally loads extensions
# with RTLD_LOCAL on Unix/macOS ; the symbols then stay invisible to the loader and
# add_native_block fails at dlopen ("symbol not found in flat namespace"). So we
# load _adc with RTLD_GLOBAL, then restore the flags for the following imports.
# The already-loaded module keeps its global scope.
def _explain_missing_extension(exc):
    """Turn the raw ModuleNotFoundError on adc._adc into an ACTIONABLE message (recurring bug :
    the extension is pinned to the cpython-3XY ABI of the interpreter that built it ; under a
    different python, the import fails without saying why). We list the .so files present next to the package
    and compare their tag to the current interpreter."""
    import glob
    here = _os.path.dirname(__file__)
    sos = sorted(_os.path.basename(p) for p in glob.glob(_os.path.join(here, "_adc.*")))
    cur = "cpython-%d%d" % (_sys.version_info[0], _sys.version_info[1])
    if not sos:
        hint = ("no _adc.*.so extension in %s : the module is not built. Build with "
                "`cmake --preset python && cmake --build --preset python`, then PYTHONPATH=<build>/python."
                % here)
    elif not any(cur in s for s in sos):
        hint = ("extension(s) present : %s, but the current interpreter is %s (%s). Use the "
                "python that built the module (conda env `adc`), or rebuild with this interpreter "
                "(-DPython_EXECUTABLE=%s)." % (", ".join(sos), cur, _sys.executable, _sys.executable))
    else:
        hint = ("the extension %s matches the interpreter (%s) but its import fails : missing "
                "dependency or corrupt .so ; rerun the module build." % (", ".join(sos), cur))
    return ImportError("import adc._adc failed : %s\n(original cause : %s)" % (hint, exc))


if hasattr(_sys, "setdlopenflags") and hasattr(_sys, "getdlopenflags"):
    _adc_old_dlopenflags = _sys.getdlopenflags()
    _adc_global_dlopenflags = _adc_old_dlopenflags
    if hasattr(_os, "RTLD_NOW"):
        _adc_global_dlopenflags |= _os.RTLD_NOW
    if hasattr(_os, "RTLD_GLOBAL"):
        _adc_global_dlopenflags |= _os.RTLD_GLOBAL
    _sys.setdlopenflags(_adc_global_dlopenflags)
    try:
        from ._adc import (SystemConfig, ModelSpec, System as _System,
                           AmrSystemConfig, AmrSystem as _AmrSystem,
                           abi_key)  # module ABI key ("production" DSL path / diagnostic)
    except ImportError as _e:
        raise _explain_missing_extension(_e) from _e
    finally:
        _sys.setdlopenflags(_adc_old_dlopenflags)
    del _adc_old_dlopenflags, _adc_global_dlopenflags
else:
    try:
        from ._adc import (SystemConfig, ModelSpec, System as _System,
                           AmrSystemConfig, AmrSystem as _AmrSystem,
                           abi_key)  # module ABI key ("production" DSL path / diagnostic)
    except ImportError as _e:
        raise _explain_missing_extension(_e) from _e

del _os, _sys, _explain_missing_extension

# Package version = the one baked into the extension (single source : project(VERSION) CMake).
# Old module without the attribute -> degrade to "unknown" rather than breaking the import.
try:
    from ._adc import __version__
except ImportError:
    __version__ = "unknown"


# --- Parallelism : a single runtime knob --------------------------------------------------------
# The compute backend is COMPILED into _adc. Multi-threading (and the GPU) are possible ONLY if
# _adc was built with -DADC_USE_KOKKOS=ON (OpenMP device). At runtime, Kokkos initializes
# LAZILY at the creation of the 1st System/AmrSystem and reads OMP_NUM_THREADS at that exact moment.
# adc.set_threads(n) writes OMP_NUM_THREADS BEFORE this init : a single call replaces the ritual
# `OMP_NUM_THREADS=n python ...`. To be called right after `import adc`, before creating the 1st system.
_first_system_built = False


def has_kokkos():
    """True if _adc was compiled with Kokkos (multi-thread/GPU possible), False if SERIAL.

    None if the module is too old to expose the info (attribute __has_kokkos__ absent)."""
    from . import _adc
    return getattr(_adc, "__has_kokkos__", None)


def set_threads(n=None):
    """Set the number of compute threads (Kokkos OpenMP backend) in ONE line.

    Equivalent to exporting OMP_NUM_THREADS=n before launching Python, but without touching the shell. Has
    an effect only if _adc was compiled with -DADC_USE_KOKKOS=ON (preset 'python-parallel'), and MUST
    be called BEFORE the 1st System/AmrSystem (Kokkos initializes lazily at that moment and
    reads OMP_NUM_THREADS only once) :

        import adc
        adc.set_threads(8)     # 8 threads
        adc.set_threads()      # all cores (os.cpu_count())
        sim = adc.System(n=256)

    A SERIAL module or a late call are flagged by a warning (without raising an exception)."""
    import os
    import warnings
    if n is None:                       # default : all available logical cores
        n = os.cpu_count() or 1
    n = int(n)
    if n < 1:
        raise ValueError("adc.set_threads : n must be >= 1")
    # Source of truth : the REAL state of the Kokkos runtime (covers ALL lazy init paths --
    # System, AmrSystem, DSL .so, direct use of _adc). The Python flag stays the fallback for
    # an old module without the binding.
    from . import _adc
    _kokkos_started = getattr(_adc, "kokkos_is_initialized", lambda: _first_system_built)()
    if _kokkos_started or _first_system_built:
        warnings.warn(
            "adc.set_threads : called AFTER the runtime initialization (1st System/AmrSystem or "
            "1st allocation) -> NO EFFECT. Call set_threads right after `import adc`.",
            RuntimeWarning, stacklevel=2)
        return
    if has_kokkos() is False:
        warnings.warn(
            "adc.set_threads : _adc is SERIAL (compiled without -DADC_USE_KOKKOS=ON) -> the thread "
            "setting is ignored at compute time. Rebuild with -DADC_USE_KOKKOS=ON "
            "-DKokkos_ROOT=$CONDA_PREFIX for multi-threading.", RuntimeWarning, stacklevel=2)
    # We write the env even in case of doubt (harmless) : a DSL .so with backend='production' compiled with
    # Kokkos will also read OMP_NUM_THREADS at its initialization.
    # We set TWO variables to be agnostic to the backend that Kokkos was compiled with :
    #   - OMP_NUM_THREADS  : read by the OpenMP device (usual case) ;
    #   - KOKKOS_NUM_THREADS : read by Kokkos::initialize whatever the device (OpenMP OR Threads),
    #     useful if the installed Kokkos (e.g. conda-forge) uses the Threads backend and not OpenMP.
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["KOKKOS_NUM_THREADS"] = str(n)
    # OMP_PROC_BIND=false ONLY on macOS (avoids libomp warnings/oversubscription on
    # dev Macs). On Linux/cluster we impose NOTHING : disabling affinity there would degrade
    # NUMA scaling, and a SLURM job that exports OMP_PROC_BIND=close/spread stays in control (setdefault
    # would not override it anyway).
    import sys as _s
    if _s.platform == "darwin":
        os.environ.setdefault("OMP_PROC_BIND", "false")


def parallel_info():
    """Parallelism state : compiled backend, current OMP_NUM_THREADS, Kokkos init already done."""
    import os
    return {
        "has_kokkos": has_kokkos(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "first_system_built": _first_system_built,
    }


def doctor(verbose=True):
    """Diagnose the adc environment in ONE command : python -c "import adc; adc.doctor()".

    Checks each link on which the module AND the runtime compilation of the DSL depend (the class of
    bugs "build environment != execution environment", e.g. the `which c++` of a conda env
    that rejects -std=c++23). Returns a dict {check: (ok, detail)} ; verbose=True prints it."""
    import os
    import sys
    checks = {}

    # 1. interpreter + extension (cpython-3XY ABI trap)
    from . import _adc
    so = getattr(_adc, "__file__", "?")
    checks["interpreteur"] = (True, "%s (%d.%d) ; extension %s"
                              % (sys.executable, sys.version_info[0], sys.version_info[1], so))

    # 2. numpy (required at import of adc.dsl)
    try:
        import numpy
        checks["numpy"] = (True, numpy.__version__)
    except Exception as e:
        checks["numpy"] = (False, "ABSENT from this interpreter (%s) -> `import adc.dsl` will fail. "
                                  "Install numpy in THIS python." % e)

    # 3. compiled compute backend
    hk = has_kokkos()
    checks["kokkos"] = (hk is not False,
                        {True: "Kokkos module (multi-thread possible ; adc.set_threads active)",
                         False: "SERIAL module (set_threads has no effect ; rebuild preset python-parallel)",
                         None: "undetermined (old module without __has_kokkos__)"}[hk])

    # 4. runtime DSL compiler (the link of the -std=c++23 bug)
    try:
        from . import dsl as _dsl
    except Exception as e:
        checks["dsl"] = (False, "import adc.dsl failed (%s)" % e)
        _dsl = None
    if _dsl is not None:
        baked = _dsl.loader_cxx_compiler()
        cc = _dsl._default_cxx(None)
        if not cc:
            checks["compilateur"] = (False, "NO C++ compiler found (ADC_CXX, module, PATH). "
                                            "Install Xcode CLT (macOS) or `conda install cxx-compiler`.")
        else:
            origin = ("$ADC_CXX" if os.environ.get("ADC_CXX") == cc
                      else "baked by the _adc build" if cc == baked else "PATH (which)")
            try:
                std = _dsl._probe_cxx_std(cc, _dsl.loader_cxx_std())
                checks["compilateur"] = (True, "%s [%s] ; -std=%s accepted" % (cc, origin, std))
            except RuntimeError as e:
                checks["compilateur"] = (False, str(e).splitlines()[0])
            if baked and cc != baked:
                checks["compilateur_abi"] = (False, "runtime compiler (%s) != build (%s) -> risk "
                                                    "of 'incompatible ABI' rejection on production "
                                                    "backend. export ADC_CXX=%r to force the one "
                                                    "from the build." % (cc, baked, baked))

        # 5. adc headers (production DSL : the signature must match the one baked into _adc)
        try:
            inc = _dsl.adc_include()
            checks["include"] = (True, inc)
            # 5b. SYNCHRONIZATION headers <-> module (real bug : module built BEFORE a git pull ->
            # the DSL loader references C++ signatures absent from the old .so -> dlopen 'symbol
            # not found' cryptic). We compare the baked signature to the one of the current tree.
            baked_sig = _dsl.module_header_signature()
            if baked_sig is not None:
                cur_sig = _dsl.adc_header_signature(inc)
                if cur_sig == baked_sig:
                    checks["headers_sync"] = (True, "headers == module build (sig %s...)"
                                              % baked_sig[:12])
                else:
                    checks["headers_sync"] = (False, "headers MODIFIED since the _adc build "
                                                     "(stale module) -> rebuild : cmake --build "
                                                     "build-py --target _adc (otherwise : dlopen "
                                                     "'symbol not found' on production backend)")
        except RuntimeError as e:
            checks["include"] = (False, "adc headers not found (set ADC_INCLUDE) : %s" % e)

        # 5c. Kokkos root for the DSL production/aot backend (the tutorial's "no DSL backend" blocker).
        # adc_cpp is Kokkos-only : every DSL .so that includes the adc headers MUST compile against an
        # installed Kokkos (Serial is enough on CPU), found via ADC_KOKKOS_ROOT / Kokkos_ROOT.
        kroot = _dsl._native_kokkos_root()
        if kroot is None:
            checks["kokkos_root"] = (False,
                "ADC_KOKKOS_ROOT / Kokkos_ROOT not set -> DSL backend='production'/'aot' cannot compile "
                "(the tutorial dead-ends on 'no DSL backend'). Fix (conda) :\n"
                "      conda env config vars set ADC_KOKKOS_ROOT=\"$CONDA_PREFIX\"\n"
                "      conda env config vars set Kokkos_ROOT=\"$CONDA_PREFIX\"\n"
                "      conda deactivate && conda activate adc")
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
                         % (os.environ.get("OMP_NUM_THREADS", "(default)"), _first_system_built))

    if verbose:
        for cname, (ok, detail) in checks.items():
            print("[%s] %-16s %s" % ("OK " if ok else "FAIL", cname, detail))
        if all(ok for ok, _ in checks.values()):
            print("=> healthy environment : module importable, DSL compilable, ABI coherent.")
        else:
            print("=> fix the FAILs above before using the DSL backend='production'.")
    return checks


# The PUBLIC API exposes ONLY composable bricks (System, AmrSystem, Model...) : no named
# physical scenario. The two-fluid AP integrator (asymptotic-preserving scheme, not
# composable brick by brick) has left the core : it is not a generic brick but a SCENARIO,
# which now lives in adc_cases (see adc_cases/two_fluid_ap/), compiled on the fly against
# the generic headers of adc_cpp. It is therefore neither re-exported here nor present in the _adc module.
__all__ = [
    "System", "SystemConfig", "AmrSystem", "AmrSystemConfig", "Model", "CompositeModel",
    "CartesianMesh", "PolarMesh", "AuxHalo",
    "Scalar", "FluidState", "ExB", "CompressibleFlux", "IsothermalFlux",
    "NoSource", "PotentialForce", "GravityForce", "MagneticLorentzForce", "PotentialMagneticForce",
    "ChargeDensity", "BackgroundDensity", "GravityCoupling",
    "Spatial", "FiniteVolume", "Explicit", "IMEX", "IMEXRK", "SourceImplicit", "SourceImplicitBE",
    "Implicit", "Split", "Strang", "CondensedSchur", "Role", "integrate",
    "elliptic", "div_eps_grad", "charge_density", "composite_rhs",
    "electric_field_from_potential", "EllipticSolver", "EllipticModel",
    "Ionization", "Collision", "ThermalExchange",
    "PythonFlux", "dsl", "time", "model", "math", "physics", "lib", "library",
    "abi_key", "capabilities",
    "set_threads", "has_kokkos", "parallel_info", "doctor",
    "compile_problem", "CompiledProblem", "CompiledTime",
    "compile_library", "read_library_manifest", "LibraryManifest",
]


# --- Mesh / geometry ("polar grid" project, Phase 1) --------------
# The CHOICE of geometry lives in a MESH object, not in the scheme : adc.FiniteVolume stays
# reconstruction + Riemann flux + variables (no geometry argument). The mesh is passed to the
# system via adc.System(mesh=...). adc.CartesianMesh is the implicit default (square domain, numerics
# STRICTLY unchanged, bit-identical). adc.PolarMesh describes a global ring (r, theta).
class CartesianMesh:
    """CARTESIAN mesh (implicit default) : square domain [0, L]^2, n x n cells.

    This is the historical geometry : adc.System(mesh=adc.CartesianMesh(n, L, periodic)) is STRICTLY
    equivalent (bit-identical) to adc.System(n=n, L=L, periodic=periodic). Provided for symmetry with
    adc.PolarMesh (the geometry choice is explicit on both sides)."""

    def __init__(self, n=64, L=1.0, periodic=True):
        self.n = int(n)
        self.L = float(L)
        self.periodic = bool(periodic)
    def _apply(self, config):
        config.geometry = "cartesian"
        config.n = self.n
        config.L = self.L
        config.periodic = self.periodic


class PolarMesh:
    """GLOBAL ANNULAR POLAR mesh ("polar diocotron grid" workstream, Phase 1): domain
    r in [r_min, r_max] x theta in [0, 2pi), nr x ntheta cells. theta is PERIODIC, r carries a
    PHYSICAL boundary condition (wall / outlet). Axis convention: direction 0 = radial,
    direction 1 = azimuthal (cf. PolarGeometry / assemble_rhs_polar on the C++ side).

    The Phase-0 prototype quantified that the Cartesian grid diffuses the RADIAL gradient of a ring in
    azimuthal rotation (ratio 73 vs polar): carrying the radial direction on a grid axis lifts
    this structural lock of the diocotron.

    SCOPE (audit update 2026-06): the polar path is WIRED into System.step (polar transport
    assemble_rhs_polar + polar Poisson + aux drift in the local basis (e_r, e_theta)).
    adc.System(mesh=adc.PolarMesh(...)) builds a global ring and advances on it. THREE levels not
    to confuse:

    - polar transport: scalar ExB AND isothermal fluid (IsothermalFluxPolar); Riemann flux
      'rusanov' (default, all transport) AND 'hll' (isothermal fluid only -- gated on model.wave_speeds,
      identical to the Cartesian one; scalar ExB does not provide wave_speeds -> 'hll' raises a
      clear rejection). 'hllc'/'roe' remain rejected on the C++ side (Euler 4 vars, no polar brick);
    - DIRECT polar Poisson (PolarPoissonSolver): single-rank, one box covering the ring;
    - TENSORIAL polar Schur stage (PolarCondensedSchurSourceStepper, via adc.Split/CondensedSchur):
      the C++ solver is multi-rank/multi-box (theta split).

    THETA SPLIT OF THE TRANSPORT (theta_boxes, ADC-67). theta_boxes=1 (default) = single-box,
    STRICTLY bit-identical to history. theta_boxes>1 = the ring is split into theta BANDS
    (each box covers the whole radius and one azimuthal band; theta_boxes must DIVIDE ntheta and
    stay <= ntheta) and the polar TRANSPORT (assemble_rhs_polar + collective fill_ghosts) runs
    multi-box. MATRIX of multi-box capabilities:

    - polar TRANSPORT (System transport, get/set state, eval_rhs, density): multi-box OK
      (per-box assembly + collective halos; the global state is reconstructed on read);
    - DIRECT polar Poisson (PolarPoissonSolver): SINGLE-BOX ONLY. A System with theta_boxes>1 that
      solves the direct field (solve_fields / step / potential, e.g. a coupled scalar ExB block)
      raises a clear UPSTREAM error (the direct solver requires complete theta rows + r columns on
      one box): use theta_boxes=1 OR the tensorial Schur stage;
    - tensorial polar Schur stage (adc.Split + adc.CondensedSchur): multi-box (multi-box C++
      solver; the theta split is now driven by theta_boxes).

    The DIRECT polar Poisson refuses MPI (single-rank). No Cartesian<->polar coupling (global
    ring). Optional step bounds (stability_speed/stability_dt/source_frequency) ARE wired on the
    polar path (default without trait = max_wave_speed, bit-identical). Cf. docs/GENERICITY_2026-06.md
    section 3 and adc.capabilities()['geometry'] / ['stability_policy']['system_polar']."""

    def __init__(self, r_min, r_max, nr, ntheta, theta_boxes=1):
        if not (r_max > r_min >= 0.0):
            raise ValueError("PolarMesh: requires r_max > r_min >= 0 (ring)")
        # nr >= 3: the radial drift of the aux (System.solve_fields_polar) uses a 2nd-order ONE-SIDED
        # stencil at both walls on phi (without ghost); nr < 3 would read phi out of bounds. A global
        # ring always has nr >= 3. ntheta >= 1 (the azimuthal drift wraps the periodic index).
        if nr < 3:
            raise ValueError("PolarMesh: nr >= 3 (2nd-order one-sided radial stencil at the walls)")
        if ntheta < 1:
            raise ValueError("PolarMesh: ntheta >= 1")
        # theta_boxes: split of the transport into theta bands (1 = single-box, default). We validate HERE
        # (Python side, clear message) AND on the C++ side (check_geometry, for a SystemConfig built by
        # hand): 1 <= theta_boxes <= ntheta AND theta_boxes DIVIDES ntheta (equal azimuthal bands).
        tb = int(theta_boxes)
        if tb < 1:
            raise ValueError("PolarMesh: theta_boxes >= 1 (1 = single-box)")
        if tb > int(ntheta):
            raise ValueError("PolarMesh: theta_boxes <= ntheta (at least one azimuthal cell per band)")
        if int(ntheta) % tb != 0:
            raise ValueError("PolarMesh: theta_boxes must DIVIDE ntheta (equal azimuthal bands)")
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.nr = int(nr)
        self.ntheta = int(ntheta)
        self.theta_boxes = tb

    def _apply(self, config):
        config.geometry = "polar"
        config.nr = self.nr
        config.ntheta = self.ntheta
        config.r_min = self.r_min
        config.r_max = self.r_max
        config.theta_boxes = self.theta_boxes
        config.n = self.nr  # n serves as the default size for the rest of the config (diagnostics)


# --- Aux halo policy ("polar grid"/generalisation, ADC-369) ---------------
class AuxHalo:
    """Per-field aux halo/ghost boundary policy, passed to set_aux_field(..., halo=adc.AuxHalo(...)).

    A model-NAMED aux field (m.aux_field(name)) normally inherits the SHARED aux ghost behavior derived
    from the potential phi (periodic preserved, otherwise zero-gradient). adc.AuxHalo lets that ONE
    field declare its own boundary policy instead, applied AFTER the shared fill to its component only:

    - ``kind='foextrap'`` : zero-gradient (ghost = mirror interior cell);
    - ``kind='dirichlet'`` : fixed boundary value (ghost = 2*value - interior), ``value`` the imposed value.

    The policy is applied UNIFORMLY to the NON-PERIODIC faces; periodic faces (a fully periodic domain,
    the polar theta direction) keep their wrap, so a per-field policy never breaks the domain's periodic
    structure. Works on System (Cartesian + polar) and the AMR coarse level. No halo (default) -> the
    shared aux BC, strictly bit-identical. Per-face asymmetric BC is a follow-up.
    """

    # Mirrors adc::BCType on the C++ side: Periodic=0, Foextrap=1, Dirichlet=2.
    _KINDS = {"foextrap": 1, "dirichlet": 2}

    def __init__(self, kind, value=0.0):
        if kind not in self._KINDS:
            raise ValueError("AuxHalo: kind must be 'foextrap' or 'dirichlet' (got %r)" % (kind,))
        self.kind = kind
        self.bc_type = self._KINDS[kind]
        self.value = float(value)

    def __repr__(self):
        return "AuxHalo(%r, value=%g)" % (self.kind, self.value)


# --- State bricks ---------------------------------------------------------
class Scalar:
    """Scalar state (1 variable, e.g. a transported density)."""


class FluidState:
    """Fluid state. kind = "compressible" (gamma) or "isothermal" (cs2).

    vacuum_floor (isothermal only, ADC-77): quasi-vacuum density floor. When > 0 the model computes
    the velocity as u = m/max(rho, vacuum_floor), bounding the wave speed and the advective flux where
    the flow evacuates the background (rho -> ~0). It does NOT modify the conserved state (only the
    velocity estimate). 0 (default) = inactive (bit-identical). This is independent of the spatial
    positivity_floor (the Zhang-Shu reconstruction limiter): the two address different failure modes
    and must be enabled separately. Honored on the native adc.Model(...) / System / AmrSystem path;
    the compiled/DSL path (adc.CompositeModel / JIT / AOT) does not carry it yet, so set it on the
    native path.
    """

    def __init__(self, kind="compressible", gamma=1.4, cs2=0.5, vacuum_floor=0.0):
        self.kind = kind
        self.gamma = float(gamma)
        self.cs2 = float(cs2)
        if not (float(vacuum_floor) >= 0.0):
            raise ValueError("FluidState: vacuum_floor >= 0 (0 = inactive)")
        self.vacuum_floor = float(vacuum_floor)


# --- Transport bricks ---------------------------------------------------
class ExB:
    """Scalar advection by the E x B drift (magnetic field B0)."""

    def __init__(self, B0=1.0):
        self.B0 = float(B0)


class CompressibleFlux:
    """Compressible Euler flux (gamma comes from the FluidState state)."""


class IsothermalFlux:
    """Isothermal Euler flux (cs2 comes from the FluidState state)."""


# --- Source bricks ------------------------------------------------------
class NoSource:
    """No source."""


class PotentialForce:
    """Potential force (q/m) rho E on the momentum (+ work if 4 vars)."""

    def __init__(self, charge=1.0):
        self.charge = float(charge)


class GravityForce:
    """Gravitational force rho g (+ work if 4 vars)."""


class MagneticLorentzForce:
    """MAGNETIC Lorentz force q (v x B_z) on the momentum (native C++ brick
    adc::MagneticLorentzForce, exposed to the Python API by the 2026-06 audit).

    EXPLICIT regime (moderate omega_c): pointwise algebraic term, no work (F . v = 0, energy
    unchanged). Reads B_z from the aux channel (canonical component 3): call
    ``sim.set_magnetic_field(Bz)`` to populate it. Requires a fluid transport >= 3 variables (momentum
    on 2 axes); rejected on a scalar. The STIFF regime (large omega_c) goes through the condensed stage
    adc.CondensedSchur (Schur), NOT through this explicit brick.

    ``charge`` = q/m, sign included (same convention as PotentialForce)."""

    def __init__(self, charge=1.0):
        self.charge = float(charge)


class PotentialMagneticForce:
    """Electrostatic force + magnetic Lorentz SUMMED: (q/m) rho E + q (v x B_z) (native C++
    brick CompositeSource<PotentialForce, MagneticLorentzForce>, the full magnetized diocotron
    force). Same q/m for both forces (same species). Reads B_z (set_magnetic_field); requires a
    fluid transport >= 3 variables. ``charge`` = q/m, sign included."""

    def __init__(self, charge=1.0):
        self.charge = float(charge)


# --- Elliptic right-hand-side bricks ------------------------------------
class ChargeDensity:
    """Charge density f = q n."""

    def __init__(self, charge=1.0):
        self.charge = float(charge)


class BackgroundDensity:
    """Neutralizing background f = alpha (n - n0)."""

    def __init__(self, alpha=1.0, n0=0.0):
        self.alpha = float(alpha)
        self.n0 = float(n0)


class GravityCoupling:
    """Self-consistent coupling f = sign 4piG (rho - rho0). sign = +1 gravity, -1 plasma."""

    def __init__(self, sign=1.0, four_pi_G=1.0, rho0=1.0):
        self.sign = float(sign)
        self.four_pi_G = float(four_pi_G)
        self.rho0 = float(rho0)


def Model(state, transport, source, elliptic):
    """Compose a model (ModelSpec) from state, transport, source, elliptic bricks.

    Validates the state <-> transport consistency (Scalar with ExB; compressible FluidState with
    CompressibleFlux; isothermal with IsothermalFlux) and carries the parameters into the spec.
    """
    spec = ModelSpec()

    if isinstance(state, Scalar):
        if not isinstance(transport, ExB):
            raise ValueError("Scalar requires transport=ExB(...)")
    elif isinstance(state, FluidState):
        if state.kind == "compressible":
            spec.gamma = state.gamma
            if not isinstance(transport, CompressibleFlux):
                raise ValueError("FluidState(compressible) requires transport=CompressibleFlux()")
        elif state.kind == "isothermal":
            spec.cs2 = state.cs2
            spec.vacuum_floor = getattr(state, "vacuum_floor", 0.0)  # ADC-77 quasi-vacuum velocity bound
            if not isinstance(transport, IsothermalFlux):
                raise ValueError("FluidState(isothermal) requires transport=IsothermalFlux()")
        else:
            raise ValueError("FluidState.kind: 'compressible' | 'isothermal'")
    else:
        raise ValueError("state: adc.Scalar() | adc.FluidState(...)")

    if isinstance(transport, ExB):
        spec.transport = "exb"; spec.B0 = transport.B0
    elif isinstance(transport, CompressibleFlux):
        spec.transport = "compressible"
    elif isinstance(transport, IsothermalFlux):
        spec.transport = "isothermal"
    else:
        raise ValueError("transport: ExB | CompressibleFlux | IsothermalFlux")

    if isinstance(source, NoSource):
        spec.source = "none"
    elif isinstance(source, PotentialForce):
        spec.source = "potential"; spec.qom = source.charge
    elif isinstance(source, GravityForce):
        spec.source = "gravity"
    elif isinstance(source, MagneticLorentzForce):
        spec.source = "magnetic"; spec.qom = source.charge
    elif isinstance(source, PotentialMagneticForce):
        spec.source = "potential_magnetic"; spec.qom = source.charge
    else:
        raise ValueError("source: NoSource | PotentialForce | GravityForce | MagneticLorentzForce "
                         "| PotentialMagneticForce")

    if isinstance(elliptic, ChargeDensity):
        spec.elliptic = "charge"; spec.q = elliptic.charge
    elif isinstance(elliptic, BackgroundDensity):
        spec.elliptic = "background"; spec.alpha = elliptic.alpha; spec.n0 = elliptic.n0
    elif isinstance(elliptic, GravityCoupling):
        spec.elliptic = "gravity"; spec.sign = elliptic.sign
        spec.four_pi_G = elliptic.four_pi_G; spec.rho0 = elliptic.rho0
    else:
        raise ValueError("elliptic: ChargeDensity | BackgroundDensity | GravityCoupling")

    return spec


# --- HYBRID composition: native brick + DSL brick IN A model --------
# adc.Model(...) composes 100% native bricks into a ModelSpec (C++ tags); adc.dsl.Model(...)
# generates a 100% DSL model. adc.CompositeModel(...) fills the in-between: MIX, in ONE SINGLE
# model, NATIVE bricks (adc.ExB / PotentialForce / ChargeDensity ...) and PARTIAL compiled DSL
# bricks (adc.dsl.HyperbolicBrick(...).compile() / SourceBrick / EllipticBrick). The
# mix is compiled into ONE composite .so (prototype: backend 'aot'). cf. adc/dsl.py (Phase B).
def _native_to_brick(obj, role):
    """Translate a NATIVE brick (adc.* object) into a dsl.NativeBrick descriptor for the @p role slot.
    An already-compiled DSL brick (dsl.CompiledBrick) passes through unchanged (after slot check)."""
    from . import dsl
    if isinstance(obj, dsl.CompiledBrick):
        if obj.kind != role:
            raise ValueError("adc.CompositeModel: DSL brick of type %r placed in the %r slot"
                             % (obj.kind, role))
        return obj
    if role == "hyperbolic":
        if isinstance(obj, ExB):
            return dsl.NativeBrick("adc::ExBVelocity", "hyperbolic", fields={"B0": obj.B0},
                                   var_names=["n"], n_vars=1, prim_names=["n"])
        if isinstance(obj, CompressibleFlux):
            g = float(getattr(obj, "gamma", 1.4))
            return dsl.NativeBrick("adc::CompressibleFlux", "hyperbolic", fields={"gamma": g},
                                   var_names=["rho", "rho_u", "rho_v", "E"], n_vars=4,
                                   prim_names=["rho", "u", "v", "p"], gamma=g)
        if isinstance(obj, IsothermalFlux):
            cs2 = float(getattr(obj, "cs2", 1.0))
            return dsl.NativeBrick("adc::IsothermalFlux", "hyperbolic", fields={"cs2": cs2},
                                   var_names=["rho", "rho_u", "rho_v"], n_vars=3,
                                   prim_names=["rho", "u", "v"])
        raise ValueError("adc.CompositeModel transport: ExB | CompressibleFlux | IsothermalFlux "
                         "(native) or dsl.HyperbolicBrick(...).compile()")
    if role == "source":
        if isinstance(obj, NoSource):
            return dsl.NativeBrick("adc::NoSource", "source", min_vars=1)
        if isinstance(obj, PotentialForce):
            return dsl.NativeBrick("adc::PotentialForce", "source", fields={"qom": obj.charge},
                                   min_vars=3)
        if isinstance(obj, GravityForce):
            return dsl.NativeBrick("adc::GravityForce", "source", min_vars=3)
        if isinstance(obj, MagneticLorentzForce):
            # n_aux=4: the brick reads B_z (canonical aux channel 3) -> the composite sizes the aux.
            return dsl.NativeBrick("adc::MagneticLorentzForce", "source",
                                   fields={"qom": obj.charge}, min_vars=3, n_aux=4)
        if isinstance(obj, PotentialMagneticForce):
            # NESTED fields of CompositeSource (public members a / b): the NativeBrick emit
            # writes `a.qom = ...; b.qom = ...;` in the constructor of the derived struct.
            return dsl.NativeBrick(
                "adc::CompositeSource<adc::PotentialForce, adc::MagneticLorentzForce>", "source",
                fields={"a.qom": obj.charge, "b.qom": obj.charge}, min_vars=3, n_aux=4)
        raise ValueError("adc.CompositeModel source: NoSource | PotentialForce | GravityForce | "
                         "MagneticLorentzForce | PotentialMagneticForce (native) or "
                         "dsl.SourceBrick(...).compile()")
    if role == "elliptic":
        if isinstance(obj, ChargeDensity):
            return dsl.NativeBrick("adc::ChargeDensity", "elliptic", fields={"q": obj.charge},
                                   min_vars=1)
        if isinstance(obj, BackgroundDensity):
            return dsl.NativeBrick("adc::BackgroundDensity", "elliptic",
                                   fields={"alpha": obj.alpha, "n0": obj.n0}, min_vars=1)
        if isinstance(obj, GravityCoupling):
            return dsl.NativeBrick("adc::GravityCoupling", "elliptic",
                                   fields={"sign": obj.sign, "four_pi_G": obj.four_pi_G,
                                           "rho0": obj.rho0}, min_vars=1)
        raise ValueError("adc.CompositeModel elliptic: ChargeDensity | BackgroundDensity | "
                         "GravityCoupling (native) or dsl.EllipticBrick(...).compile()")
    raise ValueError("adc.CompositeModel: unknown slot %r" % (role,))


def CompositeModel(transport, source, elliptic, name="hybrid"):
    """Compose a HYBRID model mixing NATIVE bricks and PARTIAL DSL bricks in ONE model.

    Each slot (transport / source / elliptic) is EITHER a native brick (adc.ExB(...),
    adc.PotentialForce(...), adc.ChargeDensity(...) ...), OR a compiled partial DSL brick
    (adc.dsl.HyperbolicBrick(...).compile(), adc.dsl.SourceBrick(...).compile(),
    adc.dsl.EllipticBrick(...).compile()). AT LEAST one slot must be a DSL brick: a
    100% native composition is written with adc.Model(...) (ModelSpec).

        tr = adc.dsl.HyperbolicBrick("iso") ...        # DSL transport
        m  = adc.CompositeModel(transport=tr.compile(),
                                source=adc.PotentialForce(charge=-1.0),   # native source
                                elliptic=adc.ChargeDensity(charge=-1.0))  # native elliptic
        co = m.compile(backend="aot")                  # -> CompiledModel
        sim.add_equation("ions", co, spatial=adc.FiniteVolume(), names=[...])

    Returns an adc.dsl.HybridModel; call .compile(backend="aot") for a CompiledModel pluggable
    via System.add_equation. (Prototype: only the 'aot' backend is wired.)"""
    from . import dsl
    tr = _native_to_brick(transport, "hyperbolic")
    sr = _native_to_brick(source, "source")
    el = _native_to_brick(elliptic, "elliptic")
    if not any(isinstance(b, dsl.CompiledBrick) for b in (tr, sr, el)):
        raise ValueError(
            "adc.CompositeModel: all-native composition; use adc.Model(...) (ModelSpec) for "
            "a 100% native model. CompositeModel is for MIXING native + DSL in a single model.")
    return dsl.HybridModel(tr, sr, el, name=name)


# --- Elliptic model (EPM): Poisson = a composable instance ------------
# The elliptic model is not a hard-coded special case; it is an EllipticPhysicalModel
# composed of bricks (operator + right-hand side + output). Poisson is its current instance.
class DivEpsGrad:
    """Elliptic operator D = div(eps grad .). eps constant (1.0 = Poisson). Variable eps(x) and
    other operators (diffusion, projection) are refinements (they would touch the solver)."""

    def __init__(self, epsilon=1.0):
        self.epsilon = float(epsilon)


class CompositeRhs:
    """System right-hand side f = sum_s elliptic_rhs_s(u_s): the SUM of the elliptic bricks
    carried by the blocks. Each block chooses its brick (charge q n, background alpha (n-n0), gravity
    coupling sign 4piG (rho-rho0)) via Model(elliptic=...); this right-hand side assembles them. It is the
    GENERIC right-hand side of the EPM: it assumes NO particular form for the contributions."""


class ChargeDensitySource(CompositeRhs):
    """Usual case of the composite right-hand side: all blocks carry a charge density, so
    f = sum_s q_s n_s. Historical alias of CompositeRhs (the computation stays the sum of the bricks)."""


class ElectricFieldFromPotential:
    """Post-processing: E = -grad phi, reinjected into aux of the hyperbolic models."""


class EllipticModel:
    """EllipticPhysicalModel: unknown + operator + right-hand side + output."""

    def __init__(self, unknown, operator, rhs, output):
        self.unknown = unknown
        self.operator = operator
        self.rhs = rhs
        self.output = output


def div_eps_grad(epsilon=1.0):
    return DivEpsGrad(epsilon)


def charge_density():
    return ChargeDensitySource()


def composite_rhs():
    """Generic right-hand side f = sum_s elliptic_rhs_s(u_s) (sum of the per-block bricks)."""
    return CompositeRhs()


def electric_field_from_potential():
    return ElectricFieldFromPotential()


def elliptic(unknown="phi", operator=None, rhs=None, output=None):
    """Compose an EPM. Poisson = elliptic(operator=div_eps_grad(), rhs=charge_density(),
    output=electric_field_from_potential()). The right-hand side can be composite_rhs() (GENERIC
    sum of the per-block elliptic bricks: charge, background, gravity); charge_density() is
    the usual case (alias)."""
    return EllipticModel(unknown, operator or DivEpsGrad(), rhs or CompositeRhs(),
                         output or ElectricFieldFromPotential())


class EllipticSolver:
    """Elliptic solver: 'geometric_mg' (any case, wall) | 'fft' (periodic, n = 2^k, discrete
    stencil) | 'fft_spectral' (periodic, continuous symbol -(kx^2+ky^2): fidelity to spectral
    references such as poisson_fft.m, exact on sinusoids)."""

    def __init__(self, kind="geometric_mg"):
        self.kind = kind


# --- Inter-species couplings (operator-split): objects passed to sim.add_coupling ---
class Ionization:
    """Ionization n_g -> n_i + n_e (rate k n_e n_g). Mass transferred from the neutral to the ion."""

    def __init__(self, electron, ion, neutral, rate):
        self.electron = electron
        self.ion = ion
        self.neutral = neutral
        self.rate = rate


class Collision:
    """Inter-species friction: force k (u_a - u_b), momentum conserved. Fluid blocks (>= 3 var)."""

    def __init__(self, a, b, rate):
        self.a = a
        self.b = b
        self.rate = rate


class ThermalExchange:
    """Thermal exchange k (T_a - T_b), energy conserved. Euler blocks (4 var)."""

    def __init__(self, a, b, rate):
        self.a = a
        self.b = b
        self.rate = rate


# --- Spatial scheme + time treatment (per block) ------------------------
class Spatial:
    """Spatial discretization: reconstruction (limiter) + numerical Riemann flux.

    - ``limiter``: "none" | "minmod" | "vanleer" | "weno5" (shortcuts none=/minmod=/vanleer=/weno5=).
      weno5 = WENO5-Z, order 5 in smooth regions, 5-point stencil (3 ghosts), oscillation-free
      capture near a front; only the native ``add_block`` path exposes it (the compiled .so paths
      allocate 2 ghosts -> explicit rejection).
    - ``flux``: "rusanov" | "hll" | "hllc" | "roe".
      rusanov = minimal generic (requires only max_wave_speed, any model).
      hll = generic with signed waves (requires model.wave_speeds: native isothermal/compressible model,
      or a DSL model declaring a primitive 'p'); less diffusive than rusanov, without requiring a
      pressure or n_vars == 4. This is the recommended path for a NON Euler model with signed waves
      (moment system, isothermal): ``hll`` + ``minmod``.
      hllc / roe = contact-resolving (HLLC) and Roe-linearized solvers. Canonical path is 2D Euler
      (4 variables rho/rho_u/rho_v/E + ideal-gas pressure); they are also generic when the model
      supplies the hooks HasHLLCStructure / HasRoeDissipation (DSL m.enable_hllc()/m.enable_roe()),
      with EulerHLLCFlux2D / EulerRoeFlux2D naming the Euler fallback on the C++ side.
    - ``recon``: "conservative" | "primitive" (reconstructed variables; primitive more robust
      for Euler: positivity of rho and p; shortcut primitive=).
    - ``positivity_floor``: DENSITY floor of the reconstructed face states (positivity limiter
      Zhang-Shu, ADC-76): conservative scaling of the face state toward the cell mean
      so that rho_face >= floor. 0/None (default) = inactive, bit-identical path.
      Motivated by the top-hat jump of contrast 1e6 in the Hoffart diocotron, where WENO5 reconstructs a
      negative density -> NaN. Requires a model exposing the Density role.
    - ``wave_speed_cache``: flux='hll' + explicit time ONLY. Pre-computes model.wave_speeds ONCE per
      cell and direction (instead of per face) then bounds each face by min/max of the two neighbor
      cells. Net gain when wave_speeds is expensive (moment hierarchy). With limiter='none' +
      recon='conservative' it is BIT-IDENTICAL to the per-face path; with a 2nd-order+ limiter it is a
      Davis bound on the cell values (different result, opt-in assumed). False (default) = per-face path
      unchanged. Wired on the FULL cartesian advance only: refused if flux != 'hll', IMEX time, polar
      geometry, or a staircase/cutcell disc transport mode is active (set_disc_domain / set_geometry_mode).
    """

    def __init__(self, limiter="minmod", flux="rusanov", recon="conservative", *, none=False,
                 minmod=False, vanleer=False, weno5=False, primitive=False,
                 positivity_floor=None, wave_speed_cache=False):
        if none:
            limiter = "none"
        elif minmod:
            limiter = "minmod"
        elif vanleer:
            limiter = "vanleer"
        elif weno5:
            limiter = "weno5"
        if primitive:
            recon = "primitive"
        self.limiter = limiter
        self.flux = flux
        self.recon = recon
        pf = 0.0 if positivity_floor is None else float(positivity_floor)
        if not (pf >= 0.0):
            raise ValueError("Spatial: positivity_floor >= 0 (0/None = inactive; received %r)"
                             % (positivity_floor,))
        self.positivity_floor = pf
        self.wave_speed_cache = bool(wave_speed_cache)


def FiniteVolume(limiter="minmod", riemann="rusanov", variables="conservative",
                 positivity_floor=None, wave_speed_cache=False):
    """Finite-volume scheme (stable surface Phase A): remaps onto the existing Spatial object.

    The NUMERICAL Riemann flux is named ``riemann`` (NOT ``flux``, reserved for the PHYSICAL flux of the
    DSL model m.flux) so the two meanings do not collide. Argument mapping:

    - ``limiter`` -> Spatial.limiter ("none" | "minmod" | "vanleer" | "weno5")
    - ``riemann`` -> Spatial.flux ("rusanov" | "hll" | "hllc" | "roe"); "hll" is the generic
      signed-wave path (requires model.wave_speeds), "hllc"/"roe" run on the canonical Euler 2D
      layout or generically via the model hooks HasHLLCStructure / HasRoeDissipation
    - ``variables`` -> Spatial.recon ("conservative" | "primitive")

    cf. docs/DSL_MODEL_DESIGN.md section 6. Returns a Spatial (consumed as-is by add_block /
    add_equation). adc.Spatial stays available identically. ``positivity_floor`` (ADC-76):
    density floor of the face states (Zhang-Shu limiter), None/0 = inactive.
    ``wave_speed_cache``: HLL wave speed cache (riemann='hll' + explicit), cf. Spatial."""
    return Spatial(limiter=limiter, flux=riemann, recon=variables,
                   positivity_floor=positivity_floor, wave_speed_cache=wave_speed_cache)


class Explicit:
    """Explicit time treatment.

    substeps=N: the block advances N times per macro-step, each substep of length dt/N
                 (fast electrons: substeps=10). Default 1 = historical behavior.
    stride=M   : block cadence, HOLD-THEN-CATCH-UP semantics (catch-up at the END of the window).
                 The block is HELD (not advanced) while (macro_step + 1) % M != 0, then advances by an
                 effective step M*dt at the macro-step where (macro_step + 1) % M == 0, i.e. at the end of each
                 window of M macro-steps (slow block, e.g. neutrals: stride=20). It thus stays
                 temporally CONSISTENT with the fast blocks (never advanced "into the future"). Default
                 1 = every macro-step, bit-identical to the historical behavior. substeps and stride are ORTHOGONAL:
                 stride=M, substeps=N -> N substeps of M*dt/N once at the end of the window.
                 POISSON COUPLING: between two catch-ups, the held block contributes to the right-hand side of the
                 system Poisson (and to the coupled sources) with its STALE state -- its last advanced
                 density/charge, frozen until the next catch-up. step_cfl honors the cadence: the stable
                 step includes the stride factor (dt <= cfl*h*substeps / (stride*w)).
                 NB: the 'aot' backend (System.add_equation on a CompiledModel backend='aot') does NOT
                 carry the cadence and REJECTS stride > 1 (explicit path, no silent ignore);
                 add_block (native) and backend='production' support the stride.
    method     : "ssprk2" (default, Shu-Osher 2-stage order 2) | "ssprk3" (3-stage order 3,
                 less dissipative, to pair with weno5) | "euler" (ForwardEuler, order 1: fidelity
                 to first-order references, validation only). Shortcut ssprk3=True.
    """

    def __init__(self, substeps=1, method="ssprk2", stride=1, *, ssprk3=False):
        if ssprk3:
            method = "ssprk3"
        if method not in ("ssprk2", "ssprk3", "euler"):
            raise ValueError("Explicit: method 'ssprk2' | 'ssprk3' | 'euler' (received %r)" % (method,))
        if int(substeps) < 1:
            raise ValueError("Explicit: substeps >= 1 (received %r)" % (substeps,))
        if int(stride) < 1:
            raise ValueError("Explicit: stride >= 1 (received %r)" % (stride,))
        self.substeps = int(substeps)
        self.stride = int(stride)
        self.method = method
        # kind passed to the compiled facade: "explicit" (SSPRK2, bit-identical default), "ssprk3"
        # or "euler" (order 1, fidelity to first-order references -- validation, never default).
        self.kind = method if method in ("ssprk3", "euler") else "explicit"


def _role_to_stable(name):
    """Normalize a role name to the STABLE key expected by the C++ (role_from_name): lowercase
    snake_case ("momentum_x", "energy"). Tolerates the PascalCase variants of the C++ enum exposed in
    the target API (e.g. "MomentumX" -> "momentum_x", "Energy" -> "energy") by inserting a '_' before each
    internal uppercase letter before lowercasing. A name already in snake_case ("momentum_x") is unchanged."""
    s = str(name).strip()
    if not s:
        return s
    if s == s.lower():  # already snake_case / lowercase: unchanged
        return s
    out = [s[0].lower()]
    for ch in s[1:]:
        if ch.isupper():
            out.append("_")
            out.append(ch.lower())
        else:
            out.append(ch)
    return "".join(out)


def _norm_implicit(label, implicit_vars, implicit_roles):
    """Normalize the implicit-mask lists (names / physical roles) into lists of strings.

    None -> [] (default: inactive mask, model default, bit-identical). A bare string is tolerated
    (e.g. implicit_vars="rho_u" -> ["rho_u"]). The roles are reduced to the STABLE C++ key (snake_case)
    via _role_to_stable -> "MomentumX" and "momentum_x" are equivalent. The mask lives on the TEMPORAL
    POLICY / block side (and NOT the model): the SAME model is reused with distinct implicit treatments.
    The RESOLUTION of names/roles -> indices and the validation (name/role absent from the block) lives
    on the C++ side (System::add_block), the only source of truth for the block names/roles."""
    def as_list(x, what):
        if x is None:
            return []
        if isinstance(x, str):
            return [x]
        try:
            out = [str(v) for v in x]
        except TypeError:
            raise ValueError("%s: %s must be a list of strings (received %r)" % (label, what, x))
        return out
    names = as_list(implicit_vars, "implicit_vars")
    roles = [_role_to_stable(r) for r in as_list(implicit_roles, "implicit_roles")]
    return names, roles


class IMEX:
    """IMEX: explicit transport (SSPRK) + stiff implicit source (backward-Euler, local Newton).

    PARTIAL treatment: only the SOURCE is implicit (backward-Euler, local cell Newton,
    via backward_euler_source / ImplicitSourceStepper on the C++ side). The TRANSPORT stays explicit
    (advanced by the SSPRK core). This is NOT a global implicit solver (flux + source + Poisson
    solved implicitly / Newton-Krylov) -- that work is a distinct future phase.

    - ``substeps=N``: substeps per macro-step (cf. Explicit). Default 1.
    - ``stride=M``: block cadence, hold-then-catch-up semantics (cf. Explicit): the block is held
      while (macro_step + 1) % M != 0, then advances by an effective step M*dt at the end of the window. Between
      two catch-ups, its STALE state contributes to the system Poisson. Default 1 = every macro-step,
      bit-identical. Backend 'aot': stride > 1 rejected (cf. Explicit).
    - ``implicit_vars``: names of the conserved variables to treat IMPLICITLY in the source step;
      the others stay explicit (forward Euler). The mask is CARRIED BY THIS POLICY / the block,
      NOT by the model -> the SAME model is reused with different implicit treatments.
      Default [] (+ implicit_roles []) = model default (Model::is_implicit, or all implicit by
      default), BIT-IDENTICAL. Resolved on the C++ side against the block names (an absent name raises an error).
      E.g. adc.IMEX(implicit_vars=["rho_u", "rho_v"]).
    - ``implicit_roles``: same mask but by physical ROLE ("density", "momentum_x", "energy", ...)
      instead of the name (cf. System.variable_roles). Union with implicit_vars. E.g.
      adc.IMEX(implicit_roles=["MomentumX", "MomentumY", "Energy"]).
    - ``newton_max_iters``: iteration budget of the local Newton (default 2 = historical constant).
    - ``newton_rel_tol`` / ``newton_abs_tol``: per-cell stopping criterion
      ||F||_inf <= abs_tol + rel_tol*||F0||_inf (0/0 = disabled, bit-identical historical loop).
    - ``newton_fd_eps``: step of the finite-difference Jacobian (default 1e-7 = historical).
    - ``newton_diagnostics``: enables the Newton report (sim.newton_report(name) -> dict
      {enabled, converged, max_residual, max_iters_used, n_failed}), aggregated over the last advance
      of the block. OPT-IN: default False = zero extra cost.

    NOMENCLATURE (audit 2026-06): the wired scheme is exactly ForwardEuler(transport without
    source) + local backward-Euler on the source ("SourceImplicitBE"). It is NOT an
    IMEX-RK / ARK family (no choice of Butcher tableau, ``method=`` of the explicit does not apply
    to the IMEX half-step); a true IMEXRK family would be a distinct future work.
    """

    kind = "imex"
    def __init__(self, substeps=1, stride=1, implicit_vars=None, implicit_roles=None,
                 newton_max_iters=2, newton_rel_tol=0.0, newton_abs_tol=0.0,
                 newton_fd_eps=1e-7, newton_diagnostics=False, newton_damping=1.0,
                 newton_fail_policy="none"):
        if int(substeps) < 1:
            raise ValueError("IMEX: substeps >= 1 (got %r)" % (substeps,))
        if int(stride) < 1:
            raise ValueError("IMEX: stride >= 1 (got %r)" % (stride,))
        if int(newton_max_iters) < 1:
            raise ValueError("IMEX: newton_max_iters >= 1 (got %r)" % (newton_max_iters,))
        if not (0.0 < float(newton_damping) <= 1.0):
            raise ValueError("IMEX: newton_damping in (0, 1] (got %r)" % (newton_damping,))
        if newton_fail_policy not in ("none", "warn", "throw"):
            raise ValueError("IMEX: newton_fail_policy 'none'|'warn'|'throw' (got %r)"
                             % (newton_fail_policy,))
        self.substeps = int(substeps)
        self.stride = int(stride)
        self.implicit_vars, self.implicit_roles = _norm_implicit("IMEX", implicit_vars, implicit_roles)
        self.newton_max_iters = int(newton_max_iters)
        self.newton_rel_tol = float(newton_rel_tol)
        self.newton_abs_tol = float(newton_abs_tol)
        self.newton_fd_eps = float(newton_fd_eps)
        self.newton_diagnostics = bool(newton_diagnostics)
        self.newton_damping = float(newton_damping)
        self.newton_fail_policy = str(newton_fail_policy)


class SourceImplicit:
    """Implicit treatment of the STIFF SOURCE (backward-Euler, local Newton), explicit transport.

    Clear name for the source-only IMEX scheme: only the SOURCE is treated implicitly
    (backward-Euler solved by local per-cell Newton, via backward_euler_source /
    ImplicitSourceStepper on the C++ side). TRANSPORT stays EXPLICIT (advanced by the SSPRK core).

    IMPORTANT -- this is NOT a global implicit PDE solver. A global implicit solver
    (flux + source + Poisson all implicit, Newton-Krylov or global Schur) is a distinct
    future effort. SourceImplicit = source-only IMEX (strictly equivalent to IMEX/adc.Implicit,
    bit-identical numerics).

    WHEN TO USE IT (SourceImplicit LOCAL vs adc.CondensedSchur GLOBAL) -- both mechanisms
    treat a stiff source implicitly, but at different scales:

    - SourceImplicit is LOCAL: the implicit part couples only the components of A SINGLE CELL
      (backward-Euler solved by per-cell Newton), there is NO spatial coupling between
      cells. Suited to purely local stiff terms (relaxation, reactions, friction).
    - adc.CondensedSchur (via adc.Split) is GLOBAL: it assembles and solves a tensor
      elliptic operator by Schur (Krylov BiCGStab) that COUPLES the whole domain. Suited to
      non-local stiff Lorentz / electrostatic coupling (e.g. magnetized Euler-Poisson from the
      Hoffart paper, arXiv:2510.11808). A local stiff source does NOT need Schur.

    - ``substeps=N``: substeps per macro-step (cf. Explicit). Default 1.
    - ``stride=M``: block cadence, hold-then-catch-up semantics (cf. Explicit). Default 1.
    - ``implicit_vars`` / ``implicit_roles``: implicit mask by NAME or by physical ROLE of the
      conserved variables to treat implicitly in the source step (cf. IMEX). Mask CARRIED BY
      THIS POLICY / the block, not by the model. Defaults [] = model default, bit-identical.
    """

    kind = "imex"  # same C++ path as IMEX (ImplicitSourceStepper)

    def __init__(self, substeps=1, stride=1, implicit_vars=None, implicit_roles=None,
                 newton_max_iters=2, newton_rel_tol=0.0, newton_abs_tol=0.0,
                 newton_fd_eps=1e-7, newton_diagnostics=False, newton_damping=1.0,
                 newton_fail_policy="none"):
        if int(substeps) < 1:
            raise ValueError("SourceImplicit: substeps >= 1 (got %r)" % (substeps,))
        if int(stride) < 1:
            raise ValueError("SourceImplicit: stride >= 1 (got %r)" % (stride,))
        if int(newton_max_iters) < 1:
            raise ValueError("SourceImplicit: newton_max_iters >= 1 (got %r)" % (newton_max_iters,))
        if not (0.0 < float(newton_damping) <= 1.0):
            raise ValueError("SourceImplicit: newton_damping in (0, 1] (got %r)"
                             % (newton_damping,))
        if newton_fail_policy not in ("none", "warn", "throw"):
            raise ValueError("SourceImplicit: newton_fail_policy 'none'|'warn'|'throw' (got %r)"
                             % (newton_fail_policy,))
        self.substeps = int(substeps)
        self.stride = int(stride)
        self.implicit_vars, self.implicit_roles = _norm_implicit(
            "SourceImplicit", implicit_vars, implicit_roles)
        self.newton_max_iters = int(newton_max_iters)
        self.newton_rel_tol = float(newton_rel_tol)
        self.newton_abs_tol = float(newton_abs_tol)
        self.newton_fd_eps = float(newton_fd_eps)
        self.newton_diagnostics = bool(newton_diagnostics)
        self.newton_damping = float(newton_damping)
        self.newton_fail_policy = str(newton_fail_policy)


# PRECISE name of the scheme wired by IMEX / SourceImplicit (audit 2026-06): ForwardEuler transport
# without source + LOCAL backward-Euler on the source (per-cell Newton). STRICT alias of
# SourceImplicit (same object): to use when you want to name the hypothesis in a script.
SourceImplicitBE = SourceImplicit


class IMEXRK:
    """IMEX-RK family (Implicit-Explicit Runge-Kutta), ARS(2,2,2) scheme, ORDER 2.

    Ascher-Ruuth-Spiteri scheme (1997): the hyperbolic transport L = -div F is treated by the
    EXPLICIT tableau, the stiff source S by the IMPLICIT tableau (LOCAL per-cell backward-Euler,
    Newton, like adc.IMEX) -- but with coupled stages that raise the GLOBAL ORDER TO 2 (transport
    AND source), whereas adc.IMEX stays a ForwardEuler(transport) + backward-Euler(source) of order 1.

    Coefficients: gamma = 1 - 1/sqrt(2), delta = 1 - 1/(2 gamma). Tableaus (stiffly accurate):
    explicit A_E = [[0,0,0],[gamma,0,0],[delta,1-delta,0]], b_E = [delta,1-delta,0];
    implicit A_I = [[0,0,0],[0,gamma,0],[0,1-gamma,gamma]], b_I = [0,1-gamma,gamma].

    DISTINCT FAMILY from adc.IMEX (kind="imexrk_ars222" != "imex"): the adc.IMEX default (local
    backward-Euler, order 1) is UNCHANGED / bit-identical. SCOPE: CARTESIAN System only -- AMR, the
    polar grid, compiled models (.so: prototype/aot/production) and the Strang/Schur splittings
    REJECT it explicitly (use adc.IMEX / adc.Explicit on those paths).

    - ``scheme``: "ars222" (only wired scheme; another name raises an explicit error).
    - ``substeps=N``: substeps per macro-step (cf. adc.Explicit). Default 1.
    - ``stride=M``: block cadence, hold-then-catch-up semantics (cf. adc.Explicit). Default 1.
    - ``newton_*``: SAME options as adc.IMEX (max_iters/rel_tol/abs_tol/fd_eps/damping/fail_policy/
      diagnostics) -- they parametrize BOTH implicit stage solves of the scheme. Defaults =
      historical constants (max_iters=2, fd_eps=1e-7), without extra cost.

    FULLY IMPLICIT SOURCE: unlike adc.IMEX, IMEXRK does NOT expose implicit_vars /
    implicit_roles (the ARS(2,2,2) stage-consistency relation assumes a homogeneous solve). A partial
    mask is rejected on the C++ side; for a partial per-component IMEX, use adc.IMEX.
    """

    kind = "imexrk_ars222"

    def __init__(self, scheme="ars222", substeps=1, stride=1,
                 newton_max_iters=2, newton_rel_tol=0.0, newton_abs_tol=0.0,
                 newton_fd_eps=1e-7, newton_diagnostics=False, newton_damping=1.0,
                 newton_fail_policy="none"):
        if scheme != "ars222":
            raise ValueError("IMEXRK: scheme 'ars222' (only wired IMEX-RK scheme; got %r)"
                             % (scheme,))
        if int(substeps) < 1:
            raise ValueError("IMEXRK: substeps >= 1 (got %r)" % (substeps,))
        if int(stride) < 1:
            raise ValueError("IMEXRK: stride >= 1 (got %r)" % (stride,))
        if int(newton_max_iters) < 1:
            raise ValueError("IMEXRK: newton_max_iters >= 1 (got %r)" % (newton_max_iters,))
        if not (0.0 < float(newton_damping) <= 1.0):
            raise ValueError("IMEXRK: newton_damping in (0, 1] (got %r)" % (newton_damping,))
        if newton_fail_policy not in ("none", "warn", "throw"):
            raise ValueError("IMEXRK: newton_fail_policy 'none'|'warn'|'throw' (got %r)"
                             % (newton_fail_policy,))
        self.scheme = str(scheme)
        self.substeps = int(substeps)
        self.stride = int(stride)
        self.newton_max_iters = int(newton_max_iters)
        self.newton_rel_tol = float(newton_rel_tol)
        self.newton_abs_tol = float(newton_abs_tol)
        self.newton_fd_eps = float(newton_fd_eps)
        self.newton_diagnostics = bool(newton_diagnostics)
        self.newton_damping = float(newton_damping)
        self.newton_fail_policy = str(newton_fail_policy)


def Implicit(dt_ratio=1, substeps=None, stride=1):
    """DEPRECATED -- use adc.SourceImplicit(...) or adc.IMEX(...) instead.

    adc.Implicit was an alias of IMEX (implicit stiff source via backward-Euler, explicit
    transport). The name "Implicit" is MISLEADING: it suggests a global implicit PDE solver
    (flux + source + Poisson all implicit / Newton-Krylov), which is NOT the case.
    adc.SourceImplicit is the clear name of the same scheme (bit-identical numerics).

    Kept for backward compatibility; emits a DeprecationWarning. Use:
      adc.SourceImplicit(substeps=k, stride=s)  -- new clear name
      adc.IMEX(substeps=k, stride=s)            -- official acronym
    """
    import warnings
    warnings.warn(
        "adc.Implicit is deprecated: the name is misleading (it is NOT a global implicit "
        "PDE solver). Use adc.SourceImplicit(...) (implicit backward-Euler source, "
        "explicit transport) or adc.IMEX(...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return IMEX(substeps=substeps if substeps is not None else dt_ratio, stride=stride)


class Role:
    """PHYSICAL roles of a model's components (cf. VariableRole on the C++ side / variable_roles).

    Lets you address a component by its MEANING in adc.CondensedSchur(density=adc.Role.Density,
    momentum=(adc.Role.MomentumX, adc.Role.MomentumY), energy=adc.Role.Energy) rather than by a literal
    name. The values are the STABLE keys expected by the C++ (role_from_name: snake_case). The
    role -> component RESOLUTION is done on the C++ side (the block reads its own VariableRole): these
    constants serve to EXPRESS the intent in the formula and to validate that a required role is requested.
    """

    Density = "density"
    MomentumX = "momentum_x"
    MomentumY = "momentum_y"
    MomentumZ = "momentum_z"
    Energy = "energy"
    VelocityX = "velocity_x"
    VelocityY = "velocity_y"
    VelocityZ = "velocity_z"
    Pressure = "pressure"
    Temperature = "temperature"
    Scalar = "scalar"


class CondensedSchur:
    """SOURCE stage condensed by Schur (Hoffart et al., arXiv:2510.11808; cf.
    docs/SCHUR_CONDENSATION_DESIGN.md). NAMES the algorithm of the implicit source coupling potential /
    velocity / Lorentz and MAPS the fields onto the block's physical roles. This is the `source=` of an
    adc.Split temporal policy (EXPLICIT / IMPLICIT splitting).

    kind="electrostatic_lorentz" (only one for now) selects ElectrostaticLorentzCondensation:
    the stage assembles the condensed elliptic operator A = I + theta^2 dt^2 alpha rho B^{-1}, solves it
    (MG-preconditioned BiCGStab), reconstructs the velocity v = B^{-1}(v^n - theta dt grad phi) and extrapolates
    to the full step. Everything is in C++ (CondensedSchurSourceStepper, #126): NO per-cell Python callback.

    The block must expose the Density / MomentumX / MomentumY roles (Energy optional) and a B_z field
    (set_magnetic_field) -- a missing role / B_z raises an EXPLICIT error at add_equation. Works for
    a native-brick model as well as for a compiled DSL model that declares these roles (electrons).

    GEOMETRY: wired in CARTESIAN (System(mesh=adc.CartesianMesh(...))) AND in POLAR
    (System(mesh=adc.PolarMesh(...)), ring (r, theta), Track A step 2c). The choice of the condensed stepper
    (cartesian CondensedSchurSourceStepper / polar PolarCondensedSchurSourceStepper) is made on the C++ side
    according to the System geometry: the SAME adc.CondensedSchur(...) is used in both cases. The
    polar counterpart is MULTI-RANK-SAFE (correct collectives under MPI) but the facade still builds
    ONE global box (on the owner rank): correct and bit-identical to single-rank, without
    effective parallelism at this level -- the facade theta decomposition is a dedicated follow-up (update
    audit 2026-06; the old mention "n_ranks>1 raises" was stale).

    WHEN TO USE IT (CondensedSchur GLOBAL vs adc.SourceImplicit LOCAL). CondensedSchur is a
    GLOBAL implicit: it COUPLES the whole domain via the condensed tensor elliptic operator
    (solved by Krylov BiCGStab), for non-local stiff Lorentz / electrostatic coupling. If the
    stiff source is purely LOCAL (couples only the components of a single cell, without spatial
    coupling: relaxation, reactions, friction), prefer adc.SourceImplicit instead: it is cheaper
    and there is then NO elliptic solve to do.

    - ``theta``: theta-scheme in (0, 1] (0.5 = Crank-Nicolson, 1 = backward Euler).
    - ``alpha``: electrostatic coupling constant of the source subsystem
      (d_t(-Lap phi) = -alpha div(rho v)).
    - ``density`` / ``momentum`` / ``energy`` / ``magnetic_field`` / ``potential``: role / field
      descriptors. They EXPRESS the intent; the role -> component resolution is done on the C++ side
      (the block reads its own VariableRole). They accept adc.Role.* (recommended), a stable role name,
      or a variable name of the block. momentum is a pair (x, y).
    - ``krylov_tol`` / ``krylov_max_iters``: tolerance and budget of the stage's Krylov (BiCGStab)
      solve. None (defaults) = historical constants (1e-10; 400 in cartesian, 600 in polar),
      made configurable by the 2026-06 audit (explicit numerical constants).
    """

    def __init__(self, kind="electrostatic_lorentz", theta=0.5, alpha=1.0,
                 density=Role.Density, momentum=(Role.MomentumX, Role.MomentumY),
                 energy=None, magnetic_field="B_z", potential="phi",
                 krylov_tol=None, krylov_max_iters=None):
        self.krylov_tol = float(krylov_tol) if krylov_tol is not None else 0.0
        self.krylov_max_iters = int(krylov_max_iters) if krylov_max_iters is not None else 0
        if krylov_tol is not None and not (0.0 < self.krylov_tol < 1.0):
            raise ValueError("CondensedSchur: krylov_tol must be in (0, 1) (got %r)" % (krylov_tol,))
        if krylov_max_iters is not None and self.krylov_max_iters < 1:
            raise ValueError("CondensedSchur: krylov_max_iters >= 1 (got %r)" % (krylov_max_iters,))
        if kind != "electrostatic_lorentz":
            raise ValueError(
                "CondensedSchur: kind 'electrostatic_lorentz' (only one supported); got %r" % (kind,))
        if not (0.0 < float(theta) <= 1.0):
            raise ValueError("CondensedSchur: theta must be in (0, 1] (got %r)" % (theta,))
        # momentum must be a pair (role_x, role_y); a bare string (iterable of characters)
        # is rejected explicitly (otherwise tuple("xy") would give two components by accident).
        if isinstance(momentum, str):
            raise ValueError(
                "CondensedSchur: momentum must be a pair (role_x, role_y), not a string (got %r)"
                % (momentum,))
        try:
            mom = tuple(momentum)
        except TypeError:
            raise ValueError(
                "CondensedSchur: momentum must be a pair (role_x, role_y) (got %r)" % (momentum,))
        if len(mom) != 2:
            raise ValueError(
                "CondensedSchur: momentum must be a pair (role_x, role_y) (got %r)" % (momentum,))
        # Role / field descriptors CARRIED in the C++ ABI (audit wave 2): density /
        # momentum / energy accept an adc.Role.* (stable role name) OR a variable name of the
        # block; the role-or-name -> component resolution is done on the C++ side (set_source_stage,
        # explicit error if not found). The DEFAULTS (canonical roles) keep the bit-identical
        # historical behavior. magnetic_field accepts a canonical aux field name
        # (AUX_CANONICAL: "B_z", "T_e", ...) -> carried aux component. potential stays fixed
        # to "phi" (the stage uses the system Poisson potential; another field would have
        # no solver behind it -> explicit rejection, no silent ignore).
        def _spec(v):
            return "" if v is None else str(v)
        # Canonical defaults -> EMPTY strings on the ABI side (the C++ then resolves the canonical
        # roles, historical path strictly unchanged).
        self.density_spec = "" if density == Role.Density else _spec(density)
        self.momentum_x_spec = "" if mom[0] == Role.MomentumX else _spec(mom[0])
        self.momentum_y_spec = "" if mom[1] == Role.MomentumY else _spec(mom[1])
        if energy is None:
            self.energy_spec = ""
        elif energy == Role.Energy:
            self.energy_spec = ""
        else:
            self.energy_spec = _spec(energy)
        if magnetic_field == "B_z":
            self.bz_aux_component = -1  # canonical channel (default, bit-identical)
        else:
            from . import dsl as _dsl
            if magnetic_field not in _dsl.AUX_CANONICAL:
                raise ValueError(
                    "CondensedSchur: magnetic_field=%r unknown (canonical aux fields: %s)"
                    % (magnetic_field, sorted(_dsl.AUX_CANONICAL)))
            self.bz_aux_component = int(_dsl.AUX_CANONICAL[magnetic_field])
        if potential != "phi":
            raise ValueError(
                "CondensedSchur: potential=%r not configurable (the source stage solves the "
                "system Poisson potential phi; another field would have no solver "
                "behind it); leave potential='phi' (default)." % (potential,))
        self.kind = kind
        self.theta = float(theta)
        self.alpha = float(alpha)
        self.density = density
        self.momentum = mom
        self.energy = energy
        self.magnetic_field = magnetic_field
        self.potential = potential
    def _has_field_overrides(self):
        """True if a non-canonical descriptor is requested (AMR: explicit rejection, not wired)."""
        return bool(self.density_spec or self.momentum_x_spec or self.momentum_y_spec
                    or self.energy_spec or self.bz_aux_component >= 0)


class Split:
    """Temporal policy EXPLICIT / IMPLICIT SPLITTING: an EXPLICIT hyperbolic transport stage
    (adc.Explicit, SSPRK) followed by a separate SOURCE stage (cf. docs/SCHUR_CONDENSATION_DESIGN.md
    section 6). This is the OPT-IN of the Schur work: a block that does NOT use adc.Split keeps the
    default path (Explicit / IMEX / SourceImplicit), BIT-IDENTICAL.

    ::

        time=adc.Split(
            hyperbolic=adc.Explicit(ssprk3=True),
            source=adc.CondensedSchur(kind="electrostatic_lorentz", theta=0.5, ...),
        )

    - ``hyperbolic`` : adc.Explicit (the transport; SSPRK2/3, substeps, stride inherit from it).
    - ``source`` : adc.CondensedSchur (the condensed source stage, runs AFTER the transport). Only
      source backend wired for now.
    """

    # kind="explicit": the transport is run by the core explicit path (SSPRK), the condensed source
    # is plugged IN ADDITION via set_source_stage (cf. System.add_equation). The block is therefore
    # NOT IMEX (the local stiff source backward-Euler): its source is the condensed stage, apart.
    def __init__(self, hyperbolic=None, source=None):
        hyperbolic = hyperbolic if hyperbolic is not None else Explicit()
        if not isinstance(hyperbolic, Explicit):
            raise TypeError(
                "Split: hyperbolic must be an adc.Explicit (explicit SSPRK transport); got %r"
                % type(hyperbolic).__name__)
        if source is None:
            raise ValueError(
                "Split: source= is required (the separate source stage); e.g. "
                "adc.Split(hyperbolic=adc.Explicit(), source=adc.CondensedSchur(...))")
        if not isinstance(source, CondensedSchur):
            raise TypeError(
                "Split: source must be an adc.CondensedSchur(...) (only wired source stage); got %r"
                % type(source).__name__)
        self.hyperbolic = hyperbolic
        self.source = source
        # The transport takes the core explicit path: we relay the kind / substeps / stride of
        # the hyperbolic stage (SSPRK2/3). The condensed source is plugged separately (add_equation).
        self.kind = hyperbolic.kind
        self.method = hyperbolic.method
        self.substeps = hyperbolic.substeps
        self.stride = hyperbolic.stride
        # Splitting policy WIRED to the system stepper (set_time_scheme). adc.Split = "lie"
        # (Godunov, 1st order): H(dt) then S(dt) once per macro-step, BIT-IDENTICAL to the history.
        # adc.Strang overrides this attribute to "strang" (cf. below).
        self.scheme = "lie"


class Strang(Split):
    """Temporal policy STRANG SPLITTING (symmetric, 2nd order): one macro-step runs
    H(dt/2); S(dt); H(dt/2), where H is the EXPLICIT hyperbolic transport (adc.Explicit, SSPRK)
    and S the separate SOURCE stage (adc.CondensedSchur). This is the 2nd-order extension of adc.Split
    (Lie / Godunov, 1st order): same bricks (SSPRK transport + condensed source stage), only the ORDER
    and the cadence of field solves change.

    ::

        time=adc.Strang(
            hyperbolic=adc.Explicit(ssprk3=True),
            source=adc.CondensedSchur(theta=0.5, alpha=alpha),
        )

    The system stepper RE-SOLVES solve_fields BETWEEN stages (before each half-advance and before
    the source) so that the transport always reads a phi consistent with the current density (the
    SINGLE leading solve_fields, sufficient for Lie or a single transport advance to follow, does not
    suffice for the 2nd Strang half-advance). cf. docs/HOFFART_STEP_SEQUENCE.md and SystemStepper::step_strang.

    ``hyperbolic`` / ``source`` : identical to adc.Split. Wired by add_equation (which plugs the source
    stage AND calls set_time_scheme('strang') on the System)."""

    def __init__(self, hyperbolic=None, source=None):
        super().__init__(hyperbolic=hyperbolic, source=source)
        self.scheme = "strang"


class System:
    """The system/coupler: composes blocks, shares a Poisson, advances the whole.

    add_block takes a composed model (adc.Model(...)) + Spatial / Explicit / IMEX objects.
    Everything else (set_poisson, set_density, step, step_cfl, step_adaptive, diagnostics,
    primitives eval_rhs/get_state/set_state) is forwarded to the compiled facade.

    GEOMETRY: the choice lives in a MESH object passed as mesh= (adc.CartesianMesh / adc.PolarMesh),
    NOT in the scheme (adc.FiniteVolume stays reconstruction + Riemann + variables). Default (mesh=None
    or adc.CartesianMesh) = square domain, bit-identical to the history. adc.PolarMesh (global ring)
    is WIRED in System.step (Phase 2b): polar ExB transport + polar Poisson + aux in local basis
    (e_r, e_theta). Limits: scalar ExB transport, single-rank, no cart<->polar coupling."""

    def __init__(self, config=None, mesh=None, **cfg_kw):
        if config is None:
            config = SystemConfig()
            for k, v in cfg_kw.items():
                setattr(config, k, v)
        # The mesh (if provided) carries the geometry CHOICE and overrides the corresponding fields
        # of the config. Applied AFTER cfg_kw: mesh= takes precedence over the n=/L= passed as keywords.
        if mesh is not None:
            if not hasattr(mesh, "_apply"):
                raise TypeError("System: mesh must be an adc.CartesianMesh / adc.PolarMesh (got %r)"
                                % type(mesh).__name__)
            mesh._apply(config)
        # Mark the Kokkos init as imminent: _System(config) allocates Fabs -> Kokkos initializes
        # (lazy) here. After this point, adc.set_threads has no further effect (warned by set_threads).
        global _first_system_built
        _first_system_built = True
        self._s = _System(config)  # geometry == 'polar' builds a global ring (Phase 2b, cf. PolarMesh)
        # Table of NAMED aux fields per block (ADC-70 phase 1): block -> {name: canonical component}.
        # Filled by add_equation from CompiledModel.aux_extra_names (the component of the k-th name =
        # dsl.AUX_NAMED_BASE + k). The FACADE holds the names: the C++ only manipulates component
        # indices (set_aux_field_component / aux_field_component). Empty for a block without a
        # named aux field. cf. set_aux_field / aux_field.
        self._aux_field_index = {}

    def add_block(self, name, model, spatial=None, time=None, evolve=True):
        """Installs an evolved block composed of NATIVE BRICKS on the shared system Poisson.

        Primary entry point for a model composed in Python from native bricks
        (adc.Model(...)). For a compiled DSL model (.so) or an automatic dispatch on the model type,
        use add_equation. The arguments are marshaled to the C++ facade
        (System::add_block), which validates the block (names / roles / implicit mask) against the model.

        @param name unique block name; indexes set_density(name) / mass(name) / density(name).
        @param model an adc.Model(...) (ModelSpec: state + transport + source + elliptic brick).
        @param spatial spatial discretization, an adc.Spatial(...) / adc.FiniteVolume(...) (default
            minmod + rusanov + conservative). Carries the limiter (none / minmod / vanleer / weno5 --
            weno5 is exposed ONLY by this native path), the Riemann flux (rusanov / hll / hllc /
            roe) and the reconstructed variables (conservative / primitive). positivity_floor is read
            here (Zhang-Shu positivity limiter).
        @param time temporal treatment, an adc.Explicit (default) / adc.IMEX / adc.SourceImplicit.
            Carries substeps (sub-steps per macro-step), stride (multirate hold-then-catch-up cadence),
            the implicit mask (implicit_vars / implicit_roles) and the Newton options (IMEX). All
            these parameters are forwarded as-is to the C++.
        @param evolve True (default) = block advances; False = frozen field (background) which still
            contributes to the right-hand side of the system Poisson.
        @throws TypeError if time is an adc.Split / adc.Strang (Schur-condensed source stage),
            not wired here: go through add_equation(..., time=adc.Split(...)).
        """
        spatial = spatial if spatial is not None else Spatial()
        time = time if time is not None else Explicit()
        # adc.Split (condensed source stage) is only wired by add_equation (which plugs
        # set_source_stage after adding the block): reject it HERE rather than running only the transport
        # silently (the condensed source would be lost).
        if isinstance(time, Split):
            raise TypeError(
                "System.add_block: adc.Split (Schur-condensed source stage) is only supported by "
                "add_equation (which plugs the source stage); use add_equation(..., time=adc.Split(...)).")
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

        - a ModelSpec (adc.Model(...)) -> add_block (composed native bricks);
        - a CompiledModel (m.compile(...)) -> the backend adder (add_dynamic_block for prototype,
          add_compiled_block for aot, add_native_block for production), with the names/roles/gamma
          carried by the .so.

        Centralizes the backend <-> adder coupling (an AOT .so must not be plugged into
        add_dynamic_block, and vice versa). cf. docs/DSL_MODEL_DESIGN.md section 3.

        @p spatial : adc.FiniteVolume(...) / adc.Spatial(...) (default minmod+rusanov+conservative).
        @p time : adc.Explicit / IMEX (default Explicit). @p substeps : overrides time.substeps.
        @p stride : overrides time.stride (1 = each macro-step, default bit-identical).
        @p names : component names (length = n_vars of the compiled model). @p evolve : block advances;
        evolve=False (frozen field) is only wired on the native path (ModelSpec -> add_block, backend
        'production' -> add_native_block). On backend 'prototype'/'aot' (the .so ABI does not carry
        evolve) an evolve=False is REJECTED explicitly -> use a native block (add_background).
        """
        from . import dsl  # late import (dsl imports this module: avoid the import cycle)

        spatial = spatial if spatial is not None else Spatial()
        time = time if time is not None else Explicit()

        # --- adc.Split (Lie) / adc.Strang (2nd order): EXPLICIT / IMPLICIT splitting, Schur OPT-IN --
        # The block is first added with the explicit HYPERBOLIC stage (existing production path,
        # no dispatch duplication), THEN we plug the condensed SOURCE stage (set_source_stage,
        # C++). The source is run AFTER the transport at each step. The default (without Split) is unchanged.
        # The splitting POLICY (Lie / Strang) is WIRED to the system stepper via set_time_scheme:
        # adc.Split -> "lie" (default, bit-identical), adc.Strang -> "strang" (H(dt/2) S(dt) H(dt/2)).
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
                "on a composed model adc.Model(...) (-> add_block). The compiled model (.so) does not "
                "carry the mask; use a native adc.Model(...).")
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
                "adc.Model(...) (-> add_block). The compiled model (.so) ABI does not carry "
                "them; use a native adc.Model(...).")

        if not isinstance(model, dsl.CompiledModel):
            raise TypeError("add_equation: model must be an adc.Model(...) (ModelSpec) or a "
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
        # have already widened the aux channel (adc_compiled_naux -> ensure_aux_width), so the component exists.
        extra = list(getattr(compiled, "aux_extra_names", []) or [])
        self._aux_field_index[name] = {nm: dsl.AUX_NAMED_BASE + k for k, nm in enumerate(extra)}

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
                    "(WENO5 wired end to end) or add_block (composed model adc.Model(...)).")
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
                    "native model adc.Model(...) -> add_block(..., evolve=False) (or add_background) "
                    "for a frozen field.")
            # positivity_floor (ADC-76) is NOT wired on the host JIT path (no assemble_rhs,
            # dedicated Rusanov order-1 residual): explicit rejection rather than a silently ignored floor.
            if getattr(spatial, "positivity_floor", 0.0) > 0.0:
                raise ValueError(
                    "add_equation: positivity_floor not supported on backend 'prototype' (dedicated "
                    "host residual, without high-order reconstruction); use backend='aot'/'production' "
                    "or a composed model adc.Model(...) -> add_block.")
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
                    "adc.Model(...) -> add_block." % nstride)
            # evolve=False (FROZEN block / fixed background) is NOT wired: the add_compiled_block ABI does
            # not carry evolve (add_compiled_block forces it to true on the C++ side) -> the block would be
            # advanced SILENTLY. We REJECT it (rejection rather than silent ignore). For a frozen field,
            # use backend='production' (add_native_block carries evolve) or a composed native model
            # adc.Model(...) -> add_block(..., evolve=False) (or add_background).
            if not evolve:
                raise ValueError(
                    "add_equation: evolve=False not supported on backend 'aot' (the AOT .so ABI does not "
                    "carry evolve; the block would be advanced silently). Use "
                    "backend='production' (native path, evolve wired) or a composed native model "
                    "adc.Model(...) -> add_block(..., evolve=False) (or add_background) for a frozen field.")
            # wave_speed_cache (ADC-199): the AOT .so ABI does not carry the wave speed cache -> it would
            # be silently ignored. Explicit rejection (the cache is only wired on the composed native
            # add_block).
            if getattr(spatial, "wave_speed_cache", False):
                raise ValueError(
                    "add_equation: wave_speed_cache not supported on backend 'aot' (the AOT .so ABI does "
                    "not carry the HLL wave speed cache; it would be silently ignored). Use a composed "
                    "native model adc.Model(...) -> add_block.")
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
            # run) -- a stale _adc module would otherwise give a cryptic dlopen 'symbol not found'.
            # wave_speed_cache (ADC-199): the add_native_block ABI does not (yet) carry the wave speed
            # cache -> it would be silently ignored. Explicit rejection BEFORE the C++ boundary (and
            # before the ABI check: a clear message rather than a dlopen error). The cache is wired on
            # the composed native add_block path (System.add_block).
            if getattr(spatial, "wave_speed_cache", False):
                raise ValueError(
                    "add_equation: wave_speed_cache not supported on backend 'production' (the "
                    "add_native_block ABI does not carry the HLL wave speed cache; it would be silently "
                    "ignored). Use a composed native model adc.Model(...) -> add_block.")
            dsl.check_compiled_matches_module(getattr(compiled, "abi_key", ""))
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

        Thin public pass-through to the C++ binding (_adc.System.set_source_stage): same flat
        signature and defaults. add_equation(time=adc.Split(source=adc.CondensedSchur(...))) wires
        this internally; this method exposes the same control for a block added with a plain
        transport time scheme, so cases configure the stage without reaching into the private _s.
        @p name: block; @p kind: 'electrostatic_lorentz'; @p theta in (0, 1]; @p alpha: stage
        coupling. The krylov_* / field descriptors / bz_aux_component defaults reproduce the
        historical bit-identical behavior. Prerequisite: B_z set via set_magnetic_field beforehand.
        """
        self._s.set_source_stage(name, kind, theta, alpha, krylov_tol, krylov_max_iters,
                                 density, momentum_x, momentum_y, energy, bz_aux_component)

    def _resolve_aux_field(self, block, name):
        """Resolve (block, NAMED aux field name) -> canonical component of the aux channel (ADC-70 phase 1).
        Resolution rule: a CANONICAL name (phi/grad/B_z/T_e) is REJECTED here -- these fields have
        their dedicated paths (B_z -> set_magnetic_field, T_e -> set_electron_temperature_from, phi/grad
        derived by solve_fields). Otherwise look it up in the block table (filled at add_equation from
        the compiled model). Raises ValueError with an actionable message on unknown block/name."""
        from . import dsl  # late import (dsl <-> __init__ cycle)
        if name == "B_z":
            raise ValueError(
                "set_aux_field: 'B_z' (magnetic field) is set via sim.set_magnetic_field(Bz), "
                "NOT via set_aux_field (B_z is a canonical aux field, not a named field).")
        if name == "T_e":
            raise ValueError(
                "set_aux_field: 'T_e' (electron temperature) is DERIVED from a fluid block via "
                "sim.set_electron_temperature_from(block), NOT set via set_aux_field.")
        if name in dsl.AUX_CANONICAL:
            raise ValueError(
                "set_aux_field: '%s' is a CANONICAL aux field (derived by the solver, not settable); "
                "set_aux_field only carries the NAMED fields declared by m.aux_field(...)." % name)
        table = self._aux_field_index.get(block)
        if table is None:
            raise ValueError(
                "set_aux_field: block '%s' unknown (or added without a named aux field); add the block "
                "via add_equation(model=...) with a model declaring m.aux_field('%s')." % (block, name))
        if name not in table:
            known = sorted(table) if table else "(none)"
            raise ValueError(
                "set_aux_field: aux field '%s' not declared by block '%s'; known named fields: %s"
                % (name, block, known))
        return table[name]

    def set_aux_field(self, block, name, field, halo=None):
        """Set a NAMED aux field (ADC-70 phase 1) of a block: @p name must have been declared by the
        model via m.aux_field(name) (and the block added via add_equation). @p field: 2D array (ny, nx)
        or flat (n*n), row-major. The field is STATIC (user-supplied, like B_z) and PERSISTS
        from one step to the next (solve_fields never rewrites named components). For B_z / T_e,
        use their dedicated paths (set_magnetic_field / set_electron_temperature_from).

        @p halo (ADC-369): an optional adc.AuxHalo declaring this field's own ghost boundary policy
        (foextrap / dirichlet), applied to the non-periodic faces after the shared aux fill. Default
        None inherits the shared aux BC (bit-identical)."""
        import numpy as np
        comp = self._resolve_aux_field(block, name)
        arr = np.asarray(field, dtype=float)
        self._s.set_aux_field_component(comp, arr.reshape(-1))
        if halo is not None:
            self._s.set_aux_field_halo_component(comp, halo.bc_type, halo.value)

    def aux_field(self, block, name):
        """Read a NAMED aux field (ADC-70 phase 1) of a block -> 2D array (ny, nx). Equals 0 everywhere as
        long as no set_aux_field has written it (aux channel initialized to zero, never rewritten by
        solve_fields beyond the derived components). @p name: declared by m.aux_field(name)."""
        import numpy as np
        comp = self._resolve_aux_field(block, name)
        return np.asarray(self._s.aux_field_component(comp), dtype=float)

    def run(self, t_end, cfl=0.4, max_steps=1_000_000):
        """Advance up to t_end by CFL steps (sugar: `while time() < t_end: step_cfl(cfl)`).

        @p cfl: Courant number passed to step_cfl. @p max_steps: guard (avoids an infinite
        loop if dt -> 0). Returns the number of steps taken. cf. DSL_MODEL_DESIGN.md section 6."""
        steps = 0
        while self.time() < t_end and steps < max_steps:
            self.step_cfl(cfl)
            steps += 1
        return steps

    def add_background(self, name, model, density, spatial=None):
        """FROZEN species (not advanced): a fixed background that contributes to the system Poisson (and,
        in the future, to coupled sources). density: n*n array. Equivalent to add_block(evolve=False)
        followed by set_density."""
        self.add_block(name, model, spatial=spatial, evolve=False)
        self.set_density(name, density)

    def add_elliptic_model(self, name, model, solver=None, bc="auto", wall="none",
                           wall_radius=0.0):
        """EPM: configures the system elliptic model (Poisson is its current instance).
        model = adc.elliptic(operator=adc.div_eps_grad(eps), rhs=adc.composite_rhs(),
        output=adc.electric_field_from_potential()). set_poisson(...) remains the equivalent shortcut.

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

    def set_disc_domain(self, cx, cy, R, mode="none"):
        """Set the TRANSPORT DOMAIN as a DISC of center (cx, cy) and radius R, and WIRE the
        transport according to mode= (T2 / T5-PR3 work). Materializes a 0/1 cell-centered mask (cell
        active when its center is inside the disc, level set hypot(x-cx, y-cy) - R < 0, SAME convention
        as the conducting wall of Poisson). This is the finite-volume counterpart of the elliptic wall: the paper
        (Hoffart et al., arXiv:2510.11808) transports on a REAL disc whereas ADC transports on the
        full Cartesian square, the circle acting only in the Poisson wall (lock from the Cartesian ring
        edges, cf. docs/HOFFART_FIDELITY.md).

        The ``mode`` parameter wires the transport:

        - 'none' (default): the mask is materialized (queryable via disc_mask()) but the transport
          stays FULL Cartesian (assemble_rhs) -> step() BIT-IDENTICAL even with the disc set;
        - 'staircase': conservative masked transport (assemble_rhs_masked, 0/1 face gate);
        - 'cutcell': cut-cell / embedded-boundary transport (assemble_rhs_eb, apertures alpha_f +
          volume fraction kappa, smooth boundary, order 2 inside the disc).

        The mode is honored under Lie AND Strang (cf. Split / Strang). R > 0; Cartesian only (the
        polar one already bounds the ring by its radial walls -> explicit error)."""
        self._s.set_disc_domain(cx, cy, R, mode)

    def set_geometry_mode(self, mode):
        """Switch ONLY the disc transport mode ('none'|'staircase'|'cutcell') without (re)defining the
        disc. A mode != 'none' requires a disc already set (set_disc_domain) -> error otherwise. Setting
        back to 'none' restores the full Cartesian transport (bit-identical)."""
        self._s.set_geometry_mode(mode)

    def disc_mask(self):
        """0/1 cell-centered domain mask, array (ny, nx) (diagnostic / contract
        verification). All 1.0 as long as set_disc_domain has not been called (subdomain = whole
        domain, default path)."""
        return self._s.disc_mask()

    def add_coupling(self, coupling):
        """Add an inter-species coupling (operator-split, applied after transport):

        - NAMED object adc.Ionization / Collision / ThermalExchange -> fixed formula
          (add_ionization / add_collision / add_thermal_exchange);
        - CompiledCoupledSource (adc.dsl.CoupledSource(...).compile(...)) -> GENERIC source described in
          formulas, carried as bytecode and interpreted on the C++ side (System.add_coupled_source; no
          per-cell Python callback, MPI-safe)."""
        from . import dsl  # late import (dsl imports this module: avoid the import cycle)

        if isinstance(coupling, dsl.CompiledCoupledSource):
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
            raise TypeError("add_coupling expects adc.Ionization / Collision / ThermalExchange or a "
                            "CompiledCoupledSource (adc.dsl.CoupledSource(...).compile(...))")

    def block_names(self):
        """Names of the added blocks, in order (useful for a Python integrator).

        Delegates to the C++ block registry (single source), so it includes the blocks loaded via
        add_dynamic_block (.so JIT) and add_compiled_block (.so AOT), not only add_block.
        """
        return list(self._s.block_names())

    @staticmethod
    def abi_key():
        """Module ABI key (compiler, C++ standard, signature of the adc headers). Compared to
        that of a native loader by add_native_block. Also exposed as a class attribute (the
        __getattr__ delegate only covers instances), so adc.System.abi_key() works."""
        return _System.abi_key()

    def set_primitive_state(self, name, **prims):
        """Initialize a block from its PRIMITIVE variables, named (rho/u/v/p ...):

            sim.set_primitive_state("electrons", rho=rho0, u=u0, v=v0, p=p0)

        Each primitive is an (n, n) array. The expected names are those of
        variable_names(name, "primitive") (the order of the block model). The (ncomp, n, n) array is
        assembled in that order, then CONVERTED to conservative variables by the block model (on the
        C++ side: compressible E = p/(g-1) + 1/2 rho|v|^2; isothermal rho u; scalar identity) and written
        to the state. Ergonomic counterpart of set_density (which only sets the density, leaving it at rest).

        Raises a clear error if a primitive name is unknown for the block, or if one is missing."""
        import numpy as np  # local: numpy is only required for this host assembly

        names = list(self._s.variable_names(name, "primitive"))
        n = self.nx()
        unknown = [k for k in prims if k not in names]
        if unknown:
            raise ValueError(
                "set_primitive_state: unknown primitive(s) %r for block '%s'; "
                "expected primitives: %r" % (unknown, name, names))
        missing = [k for k in names if k not in prims]
        if missing:
            raise ValueError(
                "set_primitive_state: missing primitive(s) %r for block '%s'; "
                "provide all the primitives: %r" % (missing, name, names))
        # Assemble (ncomp, n, n) in the model ORDER (primitive_vars), not the kwargs order.
        prim = np.empty((len(names), n, n), dtype=np.float64)
        for c, nm in enumerate(names):
            arr = np.asarray(prims[nm], dtype=np.float64)
            if arr.shape != (n, n):
                raise ValueError(
                    "set_primitive_state: primitive '%s' of shape %r, expected (%d, %d)"
                    % (nm, tuple(arr.shape), n, n))
            prim[c] = arr
        self._s.set_primitive_state(name, prim)

    def get_primitive_state(self, name):
        """Read the conservative state of a block and return it in PRIMITIVE variables (diagnostic):

            P = sim.get_primitive_state("electrons")   # {"rho": ..., "u": ..., "v": ..., "p": ...}

        Returns a dict {primitive_name: array (n, n)} in the order of variable_names(name,
        "primitive"). Inverse of set_primitive_state (exact round-trip to machine precision, the
        model cons <-> prim conversion being consistent)."""
        names = list(self._s.variable_names(name, "primitive"))
        prim = self._s.get_primitive_state(name)  # (ncomp, n, n)
        return {nm: prim[c] for c, nm in enumerate(names)}

    def check_model(self, block, raise_on_error=True, rtol=1e-8, atol=1e-10):
        """Generic RUNTIME verification of an installed block (audit 2026-06, work item 6): check
        on the CURRENT STATE of the block (whatever the backend: native composed, .so JIT/AOT/production):

        - finite state U;
        - finite residual -div F + S (exercises flux + source + reconstruction end to end);
        - positivity of the components with role Density (via variable_roles);
        - positivity of the primitive with role Pressure / named 'p' (via get_primitive_state);
        - round-trip cons -> prim -> cons ~= identity (model conversions consistent;
          the state is SAVED then RESTORED, the block is not modified).

        RUNTIME counterpart of dsl.Model.check_model (which checks the FORMULAS before compilation).
        @return dict {"ok", "failures", "block"}; raise_on_error=True (default) raises ValueError."""
        import numpy as np
        failures = []
        nv = self._s.n_vars(block)
        U = np.asarray(self._s.get_state(block), dtype=float)
        if not np.all(np.isfinite(U)):
            failures.append("state U not finite")
        self._s.solve_fields()  # aux up to date: the residual reads phi / grad phi
        R = np.asarray(self._s.eval_rhs(block), dtype=float)
        if not np.all(np.isfinite(R)):
            failures.append("residual -div F + S not finite (flux/source/reconstruction)")
        ncell = U.size // max(nv, 1)
        Uc = U.reshape(nv, ncell)
        roles = [r.lower() for r in self._s.variable_roles(block, "conservative")]
        names = list(self._s.variable_names(block, "conservative"))
        for i, r in enumerate(roles):
            if r == "density" and not bool(np.all(Uc[i] > 0)):
                failures.append("component '%s' (role Density) not strictly positive" % names[i])
        prim_roles = [r.lower() for r in self._s.variable_roles(block, "primitive")]
        prim_names = list(self._s.variable_names(block, "primitive"))
        try:
            P = np.asarray(self._s.get_primitive_state(block), dtype=float)
            if not np.all(np.isfinite(P)):
                failures.append("primitive state not finite (to_primitive)")
            else:
                for i, (r, nm) in enumerate(zip(prim_roles, prim_names)):
                    if (r == "pressure" or nm == "p") and not bool(np.all(P[i] > 0)):
                        failures.append("primitive '%s' (pressure) not strictly positive" % nm)
                # round-trip cons -> prim -> cons: state saved then restored (no net mutation).
                U0 = U.copy()
                self._s.set_primitive_state(block, P)
                U1 = np.asarray(self._s.get_state(block), dtype=float)
                self._s.set_state(block, U0)
                if not np.allclose(U1, U0, rtol=rtol, atol=atol):
                    err = float(np.max(np.abs(U1 - U0)))
                    failures.append("round-trip to_conservative(to_primitive(U)) != U "
                                    "(max gap %g: inconsistent model conversions)" % err)
        except RuntimeError as ex:  # block without conversions (earlier .so paths): report it
            failures.append("cons<->prim conversions unavailable on this block (%s)" % ex)
        report = {"ok": not failures, "failures": failures, "block": block}
        if failures and raise_on_error:
            raise ValueError("System.check_model('%s'): %d failure(s):\n  - %s"
                             % (block, len(failures), "\n  - ".join(failures)))
        return report

    # ------------------------------------------------------------------
    # OUTPUTS / CHECKPOINT / RESTART v1 (audit 2026-06, IO; cf. docs/IO_CHECKPOINT_PLAN.md).
    # Pure Python (zero change to the C++ hot path), single-rank; HDF5 aggregated/parallel and AMR =
    # PR-IO-3. ATOMIC write (.tmp file then os.replace: a crash mid-write never
    # corrupts a previous checkpoint).
    # ------------------------------------------------------------------
    def write(self, path, format="vtk", step=None, fields=None, parallel=False):
        """VISUALIZATION OUTPUT: writes the current state to an opened file (ParaView/numpy).

        - ``format="vtk"``: ImageData .vti ASCII (Cartesian; opened by ParaView / VisIt) -- one
          CellData per conservative variable of each block + the potential phi.
        - ``format="npz"``: compressed np.savez (any backend / any geometry) -- per-block states,
          names/roles, phi, t, macro_step, grid.
        - @p step: numbered suffix (path_000123.vti); None = raw path + extension.
        - @p fields: subset of blocks to write (None = all).
        - @p parallel: PARALLEL HDF5 write by hyperslabs (opt-in, format='hdf5' ONLY). Default
          False = rank-0 gather path below, STRICTLY unchanged. True = each rank writes ITS
          boxes into a single file via h5py(mpio) -- requires h5py built with MPI + mpi4py (otherwise a
          CLEAR error with a remedy, never a silent degraded write). Cf. _write_hdf5_parallel.
        - @return the written path.

        MULTI-RANK (MPI np>1): the fields are gathered via the GLOBAL collective accessors
        (state_global / potential_global -- every rank MUST therefore call write), then ONLY rank 0
        writes the file (a single file, identical to single-rank). The System being mono-box (one
        box covering the whole domain, on rank 0), the gather is exact. The other ranks return the
        path without I/O. PARALLEL HDF5 (per-rank hyperslabs): parallel=True (cf. _write_hdf5_parallel;
        real parallelism only in MULTI-BOX, the Cartesian System being mono-box)."""
        import os
        import numpy as np
        from . import _adc
        if parallel and format != "hdf5":
            raise ValueError(
                "write: parallel=True is only supported for format='hdf5' (write by "
                "hyperslabs); format=%r goes through the rank-0 gather path (parallel=False)."
                % (format,))
        rank0 = (_adc.my_rank() == 0)
        blocks = [b for b in self._s.block_names() if fields is None or b in fields]
        suffix = ("_%06d" % int(step)) if step is not None else ""
        nxv, nyv = self._s.nx(), self._s.ny()
        if format == "npz":
            # COLLECTIVE gather (all ranks) BEFORE the rank-0 guard: state_global / potential_global
            # do an internal all_reduce and must be called by each rank.
            out = {"t": self._s.time(), "macro_step": self._s.macro_step(),
                   "nx": nxv, "ny": nyv, "blocks": np.array(blocks)}
            for b in blocks:
                nv = self._s.n_vars(b)
                out["state_" + b] = np.asarray(self._s.state_global(b), dtype=np.float64).reshape(
                    nv, nyv, nxv)
                out["names_" + b] = np.array(list(self._s.variable_names(b, "conservative")))
                out["roles_" + b] = np.array(list(self._s.variable_roles(b, "conservative")))
            out["phi"] = np.asarray(self._s.potential_global(), dtype=np.float64).reshape(nyv, nxv)
            target = path + suffix + ".npz"
            if not rank0:
                return target  # only rank 0 writes the file (gather already done collectively)
            tmp = target + ".tmp"
            with open(tmp, "wb") as f:
                np.savez_compressed(f, **out)
            os.replace(tmp, target)
            return target
        if format == "vtk":
            target = path + suffix + ".vti"
            arrays, names = [], []
            for b in blocks:
                nv = self._s.n_vars(b)
                st = np.asarray(self._s.state_global(b), dtype=np.float64).reshape(nv, nyv, nxv)
                for c, nm in enumerate(self._s.variable_names(b, "conservative")):
                    arrays.append(st[c]); names.append("%s_%s" % (b, nm))
            arrays.append(np.asarray(self._s.potential_global(), dtype=np.float64).reshape(nyv, nxv))
            names.append("phi")
            if not rank0:
                return target  # collective gather done above; only rank 0 writes
            lines = ['<?xml version="1.0"?>',
                     '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">',
                     '  <ImageData WholeExtent="0 %d 0 %d 0 0" Origin="0 0 0" '
                     'Spacing="%.17g %.17g 1">' % (nxv, nyv, 1.0 / nxv, 1.0 / nyv),
                     '    <Piece Extent="0 %d 0 %d 0 0">' % (nxv, nyv),
                     '      <CellData>']
            for nm, arr in zip(names, arrays):
                lines.append('        <DataArray type="Float64" Name="%s" format="ascii">' % nm)
                lines.append("          " + " ".join("%.17g" % v for v in arr.ravel()))
                lines.append('        </DataArray>')
            lines += ['      </CellData>', '    </Piece>', '  </ImageData>', '</VTKFile>', '']
            tmp = target + ".tmp"
            with open(tmp, "w") as f:
                f.write("\n".join(lines))
            os.replace(tmp, target)
            return target
        if format == "hdf5":
            if parallel:
                # PARALLEL HDF5 by hyperslabs (PR-IO-3, opt-in): each rank writes ITS boxes into
                # a single file (h5py mpio), no global gather. SEPARATE path -- the serial path
                # below stays STRICTLY unchanged.
                return self._write_hdf5_parallel(path + suffix + ".h5", blocks, nxv, nyv)
            # AGGREGATED HDF5 v1 (wave 3, plan PR-IO-2): a single file, one group per block,
            # attributes for the clock/grid. Multi-rank: collective gather (state_global /
            # potential_global) then rank-0 write (a single file). PARALLEL HDF5 (per-rank
            # hyperslabs) = parallel=True (branch above). h5py optional: absent -> clear error.
            # COLLECTIVE gather (all ranks) BEFORE the rank-0 guard.
            states = {b: np.asarray(self._s.state_global(b), dtype=np.float64).reshape(
                self._s.n_vars(b), nyv, nxv) for b in blocks}
            phi_g = np.asarray(self._s.potential_global(), dtype=np.float64).reshape(nyv, nxv)
            target = path + suffix + ".h5"
            if not rank0:
                return target  # only rank 0 writes the file
            try:
                import h5py
            except ImportError:
                raise RuntimeError(
                    "write(format='hdf5'): h5py missing (pip/conda install h5py); "
                    "use format='npz' (equivalent, no dependency) in the meantime.")
            tmp = target + ".tmp"
            with h5py.File(tmp, "w") as f:
                f.attrs["t"] = self._s.time()
                f.attrs["macro_step"] = self._s.macro_step()
                f.attrs["nx"] = nxv
                f.attrs["ny"] = nyv
                f.attrs["abi_key"] = abi_key()
                for b in blocks:
                    g = f.create_group(b)
                    g.create_dataset("state", data=states[b], compression="gzip")
                    g.attrs["names"] = [s.encode() for s in
                                        self._s.variable_names(b, "conservative")]
                    g.attrs["roles"] = [s.encode() for s in
                                        self._s.variable_roles(b, "conservative")]
                f.create_dataset("phi", data=phi_g, compression="gzip")
            os.replace(tmp, target)
            return target
        raise ValueError("write: format 'vtk' | 'npz' | 'hdf5' (received %r)" % (format,))

    def _write_hdf5_parallel(self, target, blocks, nxv, nyv):
        """PARALLEL HDF5 WRITE by hyperslabs (write(format='hdf5', parallel=True)) -- PR-IO-3.

        OPT-IN PATH, separate from the serial path (rank-0 gather) which stays untouched. Instead of
        gathering the whole field on rank 0, each rank WRITES ITS BOXES into a SINGLE file opened
        collectively (h5py driver='mpio'). The global datasets (ncomp, ny, nx) per block + phi (ny, nx)
        are created COLLECTIVELY, the metadata (t, macro_step, nx, ny, abi_key, names/roles) written
        collectively, then each rank writes its hyperslabs dset[:, jlo:jhi+1, ilo:ihi+1] in INDEPENDENT
        I/O (disjoint boxes ; a rank with no box writes nothing).

        TRUE PARALLELISM = MULTI-BOX only. The cartesian System is MONO-BOX (one box covering the
        domain, on rank 0) : under np>1 rank 0 writes the single box and the other ranks carry no
        box -- the hyperslab gain appears on a multi-box geometry (cf. AMR, ADC-65). The mechanics
        stay CORRECT in the general case (iteration over all local fabs). phi is solved/gathered
        COLLECTIVELY (potential_global, all_reduce) then written by rank 0 alone (full scalar field,
        contiguous dataset).

        CONTIGUOUS DATASETS (no gzip) : parallel HDF5 does not allow independent writes to
        chunk-filtered datasets. The serial path keeps gzip ; the re-read VALUES are identical field by
        field (parallel=True under np=1 == parallel=False, verified by test_hdf5_parallel).

        NEVER SILENT : h5py absent, h5py without MPI, or mpi4py absent -> RuntimeError with remedy
        (install h5py built with MPI + mpi4py, or parallel=False)."""
        import os
        import numpy as np
        from . import _adc
        # h5py FIRST, THEN the MPI support test : an h5py present but WITHOUT MPI must give
        # the targeted error (remedy), independently of whether mpi4py is present.
        try:
            import h5py
        except ImportError:
            raise RuntimeError(
                "write(format='hdf5', parallel=True) : h5py absent. Remedy : install h5py built with "
                "MPI (parallel HDF5), or parallel=False (global gather + rank-0 write).")
        if not h5py.get_config().mpi:
            raise RuntimeError(
                "write(format='hdf5', parallel=True) : h5py present but WITHOUT MPI support "
                "(h5py.get_config().mpi == False). Remedy : install h5py built with MPI (parallel "
                "HDF5), or parallel=False (global gather + rank-0 write).")
        try:
            from mpi4py import MPI
        except ImportError:
            raise RuntimeError(
                "write(format='hdf5', parallel=True) : mpi4py absent (required to open in mpio). "
                "Remedy : install mpi4py, or parallel=False (global gather + rank-0 write).")
        # Guard : _adc module built BEFORE the local accessors (build prior to ADC-66).
        if not hasattr(self._s, "local_boxes"):
            raise RuntimeError(
                "write(format='hdf5', parallel=True) : the loaded _adc module does not expose "
                "local_boxes/local_state (build prior to hyperslab writes). Remedy : "
                "rebuild adc_cpp, or parallel=False.")
        comm = MPI.COMM_WORLD
        rank0 = (_adc.my_rank() == 0)
        # phi : solved + gathered COLLECTIVELY (all ranks ; potential_global does the all_reduce),
        # then written by rank 0 alone (global scalar field, contiguous dataset).
        phi_g = np.asarray(self._s.potential_global(), dtype=np.float64).reshape(nyv, nxv)
        # Descriptors identical on all ranks (shared composition) : pre-computed for coherent
        # collective operations (create_dataset / attrs).
        ncomp = {b: self._s.n_vars(b) for b in blocks}
        names = {b: [s.encode() for s in self._s.variable_names(b, "conservative")] for b in blocks}
        roles = {b: [s.encode() for s in self._s.variable_roles(b, "conservative")] for b in blocks}
        tmp = target + ".tmp"
        # COLLECTIVE open (all ranks open the same file via mpio).
        f = h5py.File(tmp, "w", driver="mpio", comm=comm)
        try:
            # Collective metadata -- identical to the serial path.
            f.attrs["t"] = self._s.time()
            f.attrs["macro_step"] = self._s.macro_step()
            f.attrs["nx"] = nxv
            f.attrs["ny"] = nyv
            f.attrs["abi_key"] = abi_key()
            for b in blocks:
                g = f.create_group(b)  # collective
                # GLOBAL dataset (ncomp, ny, nx) CONTIGUOUS (no gzip : independent writes forbidden
                # on a chunk-filtered dataset in parallel).
                dset = g.create_dataset("state", shape=(ncomp[b], nyv, nxv), dtype="f8")  # collective
                g.attrs["names"] = names[b]
                g.attrs["roles"] = roles[b]
                # Each rank writes ITS local boxes as hyperslabs (independent I/O : disjoint
                # boxes, a rank with no box -> empty loop). local_state already returns (ncomp, bny, bnx).
                for li, (ilo, jlo, ihi, jhi) in enumerate(self._s.local_boxes(b)):
                    dset[:, jlo:jhi + 1, ilo:ihi + 1] = np.asarray(
                        self._s.local_state(b, li), dtype=np.float64)
            phi_d = f.create_dataset("phi", shape=(nyv, nxv), dtype="f8")  # collective
            if rank0:
                phi_d[...] = phi_g  # global field already gathered : written by rank 0 alone
        finally:
            f.close()  # collective
        comm.Barrier()  # all ranks have closed BEFORE the atomic rename
        if rank0:
            os.replace(tmp, target)
        comm.Barrier()  # the rename is visible (shared FS) before any return
        return target

    def checkpoint(self, path, parallel=False):
        """RESTARTABLE CHECKPOINT v1 (npz) : COMPLETE block state + clock (t, macro_step --
        MANDATORY for the stride cadence) + grid + provenance (abi_key). CONTRACT (cf.
        docs/IO_CHECKPOINT_PLAN.md) : restart does NOT rebuild the composition -- the user
        script replays its add_block/set_poisson/couplings then calls sim.restart(path), which
        VERIFIES the consistency (blocks, sizes) and raises an explicit error otherwise. @return the path.

        MULTI-RANK (MPI np>1) : the states are gathered by the collective GLOBAL accessors
        (state_global / potential_global -- all ranks MUST call checkpoint), then ONLY
        rank 0 writes the SINGLE file (identical to mono-rank). The checkpoint/restart pair stays
        bit-identical under np>1 (mono-box System : all the state lives on rank 0, exact gather).

        @p parallel : the v1 checkpoint is ALWAYS rank-0 gather (npz format, not HDF5). The hyperslab
        write (parallel=True) only applies to the visualization OUTPUT
        write(format='hdf5') : an npz checkpoint has neither HDF5 datasets nor box partitioning. Passing
        parallel=True therefore raises an EXPLICIT error (never a silently degraded write) : for a
        parallel output, use write(format='hdf5', parallel=True) ; a restartable parallel HDF5
        checkpoint is later work (PR-IO-3, cf. docs/IO_CHECKPOINT_PLAN.md)."""
        import os
        import numpy as np
        from . import _adc
        if parallel:
            raise NotImplementedError(
                "checkpoint(parallel=True) : the v1 checkpoint is a rank-0 gather npz (non "
                "HDF5 format, no box partitioning). The hyperslab write only concerns "
                "write(format='hdf5', parallel=True) (visualization output). A restartable parallel "
                "HDF5 checkpoint remains to be done (PR-IO-3, docs/IO_CHECKPOINT_PLAN.md) ; for "
                "now : checkpoint(parallel=False).")
        blocks = list(self._s.block_names())
        out = {"adc_checkpoint_version": 1,
               "t": self._s.time(), "macro_step": self._s.macro_step(),
               "nx": self._s.nx(), "ny": self._s.ny(),
               "abi_key": abi_key(), "blocks": np.array(blocks)}
        # COLLECTIVE gather (all ranks) BEFORE the rank-0 guard.
        for b in blocks:
            nv = self._s.n_vars(b)
            out["ncomp_" + b] = nv
            out["state_" + b] = np.asarray(self._s.state_global(b), dtype=np.float64)
            out["names_" + b] = np.array(list(self._s.variable_names(b, "conservative")))
        # phi : multigrid warm start (BIT-IDENTICAL restart) ; physical STATE if
        # gauss_policy="evolve" (phi is no longer re-derived from rho there).
        out["phi"] = np.asarray(self._s.potential_global(), dtype=np.float64)
        # COMPILED-PROGRAM HISTORIES (ADC-406b): a compiled time Program with multistep histories (e.g.
        # Adams-Bashforth 2) carries the System-owned ring buffers across macro-steps. To make a
        # (run, checkpoint, restart, continue) run bit-identical to a continuous run, the rings (the
        # previous RHS R_{n-1}, ...) MUST survive the checkpoint -- else AB2 cold-starts again and
        # diverges. The program HASH is recorded too: a restart against a DIFFERENT compiled Program is
        # rejected (the buffers / cadence would be meaningless). Both groups of keys are OPTIONAL: a
        # checkpoint with no installed program / no history restarts exactly as before (back-compatible).
        prog_hash = ""
        if hasattr(self._s, "installed_program_hash"):
            prog_hash = self._s.installed_program_hash()
        if prog_hash:
            out["program_hash"] = prog_hash
        hist_names = list(self._s.history_names()) if hasattr(self._s, "history_names") else []
        if hist_names:
            out["history_names"] = np.array(hist_names)
            for hname in hist_names:
                depth = int(self._s.history_depth(hname))
                out["history_depth_" + hname] = depth
                out["history_ncomp_" + hname] = int(self._s.history_ncomp(hname))
                out["history_init_" + hname] = bool(self._s.history_initialized(hname))
                # COLLECTIVE gather of every slot (all ranks call), like state_global above.
                for k in range(depth):
                    out["history_%s_%d" % (hname, k)] = np.asarray(
                        self._s.history_global(hname, k), dtype=np.float64)
        target = path if path.endswith(".npz") else path + ".npz"
        if _adc.my_rank() != 0:
            return target  # only rank 0 writes the checkpoint (gather already done)
        tmp = target + ".tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **out)
        os.replace(tmp, target)
        return target

    def restart(self, path):
        """RESUMES a v1 checkpoint : VERIFIES the composition (same blocks, same sizes -- explicit
        error otherwise, never a silently wrong resume), restores the state of each block
        then the clock (t, macro_step : the stride cadence resumes exactly). The COMPOSITION
        (add_block / set_poisson / set_magnetic_field / couplings) must have been replayed by the
        script BEFORE the call (v1 contract, cf. checkpoint).

        MULTI-RANK (MPI np>1) : all ranks read the file (shared file system) and
        call set_state / set_potential / set_clock. set_state / set_potential are MPI-safe (the
        owner rank -- rank 0, mono-box -- writes, the others are no-ops) ; set_clock sets
        the clock on each rank. The resume is therefore bit-identical under np>1."""
        import numpy as np
        target = path if path.endswith(".npz") else path + ".npz"
        d = np.load(target, allow_pickle=False)
        if int(d["adc_checkpoint_version"]) != 1:
            raise ValueError("restart : checkpoint version %r not supported (expected 1)"
                             % (d["adc_checkpoint_version"],))
        if int(d["nx"]) != self._s.nx() or int(d["ny"]) != self._s.ny():
            raise ValueError("restart : checkpoint grid (%d x %d) != system (%d x %d)"
                             % (int(d["nx"]), int(d["ny"]), self._s.nx(), self._s.ny()))
        chk_blocks = [str(b) for b in d["blocks"]]
        cur_blocks = list(self._s.block_names())
        if chk_blocks != cur_blocks:
            raise ValueError("restart : checkpoint blocks %r != current composition %r "
                             "(replay the SAME composition before restart)" % (chk_blocks, cur_blocks))
        for b in chk_blocks:
            if int(d["ncomp_" + b]) != self._s.n_vars(b):
                raise ValueError("restart : block '%s' has %d components in the checkpoint, %d here"
                                 % (b, int(d["ncomp_" + b]), self._s.n_vars(b)))
            self._s.set_state(b, np.asarray(d["state_" + b], dtype=np.float64))
        # phi BEFORE the clock : warm start of the restored solver (bit-identical restart ; physical
        # state in gauss_policy="evolve").
        if "phi" in d:
            self._s.set_potential(np.asarray(d["phi"], dtype=np.float64).ravel())
        # COMPILED-PROGRAM HASH GUARD (ADC-406b): if the checkpoint recorded an installed program hash,
        # the user must have RE-INSTALLED the SAME compiled Program before restart (the v1 replay
        # contract). A different Program (different IR hash) makes the restored histories / cadence
        # meaningless -> fail loud rather than silently continue with the wrong scheme.
        if "program_hash" in d:
            chk_hash = str(d["program_hash"])
            cur_hash = (self._s.installed_program_hash()
                        if hasattr(self._s, "installed_program_hash") else "")
            if cur_hash != chk_hash:
                raise RuntimeError("checkpoint was created with a different compiled Program hash")
        # COMPILED-PROGRAM HISTORIES (ADC-406b): restore each ring (the previous RHS, ...) so a
        # multistep scheme (AB2) resumes EXACTLY where it stopped -- continuous == (run, ckpt, restart,
        # continue) bit-for-bit. The program re-registers the rings on its first post-restart step;
        # restoring them here (before that step) seeds the slots and the initialized flag so the first
        # post-restart read sees the true R_{n-1} (no phantom cold-start re-fill).
        available = set(str(h) for h in d["history_names"]) if "history_names" in d else set()
        # MISSING-HISTORY GUARD (ADC-414, spec error 18): a history the CURRENT program already
        # registered (it stepped at least once before this restart, or was re-installed and run) but the
        # checkpoint never recorded cannot be restored -> the multistep scheme would silently cold-start.
        # Fail loud, distinct from the hash-mismatch message above. A fresh program that has not yet
        # registered any ring (the common install-then-restart flow) has no required history -> no false
        # positive; the rings it re-registers on its first post-restart step are restored below.
        if hasattr(self._s, "history_names"):
            for hname in self._s.history_names():
                if hname not in available:
                    raise RuntimeError(
                        "checkpoint does not contain required Program history '%s'" % hname)
        if available:
            for hname in (str(h) for h in d["history_names"]):
                depth = int(d["history_depth_" + hname])
                for k in range(depth):
                    self._s.restore_history(
                        hname, k, np.asarray(d["history_%s_%d" % (hname, k)], dtype=np.float64))
                self._s.set_history_initialized(hname, bool(d["history_init_" + hname]))
        self._s.set_clock(float(d["t"]), int(d["macro_step"]))

    def __getattr__(self, attr):
        return getattr(self._s, attr)


def capabilities():
    """OFFICIAL MATRIX of capabilities by facade / geometry / backend (audit 2026-06, wave 2).

    SINGLE source of truth consultable by scripts and docs (the audits showed that System,
    AMR, polar and the DSL backends diverged silently). The entries reflect the GATES
    actually coded (make_block / dispatch_amr_* / block_builder_polar / dsl._BACKENDS) ; the
    combinations outside the matrix raise an explicit error on the C++ side (never a silent ignore).
    """
    from . import _adc as _adc_mod  # ADC-291: read the aux limit from the SINGLE C++ source
    from . import dsl as _dsl_caps  # fallback mirror (no second hardcoded literal)
    aux_max_extra = int(getattr(_adc_mod, "__aux_max_extra__", _dsl_caps.AUX_NAMED_MAX))
    return {
        # Spatial dimension of the core (ADC-294 / ADR-0001 Decision 1). The solver is structurally
        # 2D: a load-bearing invariant baked into the data layout (Fab2D operator()(i, j, c)), the
        # paired FaceFluxX / FaceFluxY kernels, the 2-component momentum, the 5-point Poisson and the
        # Box2D / Geometry index space -- not a naming detail. Published as an explicit, introspectable
        # structured scalar (hard limits are scalars, not prose) so scripts and the limitations doc can
        # key off it. The polar mesh is a second GEOMETRY at the SAME dimension ((r, theta) is a
        # 2-index Box2D), so this is a separate top-level key, NOT nested under "geometry". An ND core
        # (BoxND / GeometryND) is deferred to a future milestone; see
        # docs/sphinx/reference/known-limitations.md and include/adc/mesh/box2d.hpp.
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
                   "include/adc/amr/refinement_ratio.hpp ; a non-2 ratio is rejected at "
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
                         "(set_electron_temperature_from), closed list ADC_AUX_FIELDS / AUX_CANONICAL "
                         "(C++ name table adc/core/aux_names.hpp, mirror of Python AUX_CANONICAL)",
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
                # adc.AuxHalo(kind, value), applied to the NON-PERIODIC faces (periodic faces -- periodic
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

def _reject_newton_amr_compiled(label, time):
    """REJECTS Newton options/diagnostics on the COMPILED AMR path (.so loader, flat ABI
    add_native_block / adc_install_native_amr) -- wave 3, settle. On the NATIVE side (adc.Model(...)), the
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
            "Use AmrSystem.add_block (native model adc.Model(...)) or add_compiled_model("
            "AmrSystem&) directly (C++)." % label)


class AmrSystem:
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
        global _first_system_built
        _first_system_built = True
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
        Compare with coarse_total_boxes() and adc.n_ranks() to confirm MPI strong-scaling of the base.
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
        @param model an adc.Model(...) (ModelSpec : composed native bricks).
        @param spatial spatial discretization, an adc.Spatial(...) / adc.FiniteVolume(...) (default
            minmod + rusanov + conservative). Limiter (none / minmod / vanleer / weno5 ; weno5 = 3
            ghosts, the coupler allocates its levels at Limiter::n_ghost and the regrid inherits n_grow()),
            Riemann flux (rusanov / hll / hllc / roe) and reconstructed variables
            (conservative / primitive).
        @param time temporal treatment, an adc.Explicit (default) / adc.IMEX / adc.SourceImplicit.
            Carries substeps, stride (multirate hold-then-catch-up), the implicit mask (implicit_vars
            / implicit_roles) and the Newton options, threaded to the C++. newton_diagnostics is
            wired in native multi-block and rejected at the C++ build in single-block (the coupler does not
            aggregate a report).
        @throws TypeError if time is an adc.Split / adc.Strang (Schur-condensed source stage) :
            go through add_equation(..., time=adc.Strang(...)) (amr-schur path).
        spatial.positivity_floor > 0 (ADC-259) floors the Density-role face states AND the
        coarse-fine fine ghost means to >= floor on the AMR transport (Zhang-Shu, parity with the
        uniform System). Guarantee = face / ghost-state Density positivity only (order-1 fallback),
        NOT updated-mean nor pressure positivity. A model without a Density role rejects it at the
        first step. The COMPILED .so path carries it too now (ADC-322): a loader regenerated against
        the current headers marshals the floor (add_equation on a CompiledModel, add_native_block).
        """
        spatial = spatial if spatial is not None else Spatial()
        time = time if time is not None else Explicit()
        # adc.Split / adc.Strang (Schur-condensed source stage) is only wired by add_equation (which
        # connects set_source_stage + set_time_scheme AFTER adding the block) : we reject it HERE rather
        # than playing only the transport and SILENTLY LOSING the source (same guard as System.add_block).
        if isinstance(time, Split):
            raise TypeError(
                "AmrSystem.add_block : adc.Split / adc.Strang (Schur-condensed source stage) is "
                "supported only by add_equation (which connects the source stage) ; use "
                "add_equation(..., time=adc.Strang(hyperbolic=adc.Explicit(...), "
                "source=adc.CondensedSchur(...))).")
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

    def write(self, path, format="npz", step=None):
        """AMR VISUALIZATION OUTPUT (wave 3) : COARSE fields per block + phi + footprints of the
        fine patches. format='npz' (per-block densities, phi, patch_rectangles, t) or 'vtk' (.vti of
        the COARSE : per-block density + phi -- the fine patches are provided in npz via their
        rectangles, the multi-resolution VTK = PR-IO-3). @p step : numbered suffix. @return path."""
        import os
        import numpy as np
        n = self._s.nx()
        suffix = ("_%06d" % int(step)) if step is not None else ""
        # EACH block, by its name (binding AmrSystem::block_names, parity with System) : in multi-block,
        # density() without a name would read ONLY block 0 and would lose the others SILENTLY.
        names = list(self._s.block_names())
        if not names:
            names = [""]
        if format == "npz":
            out = {"t": self._s.time(), "n": n,
                   "patch_rectangles": np.array(self.patch_rectangles(), dtype=np.float64)
                   if self.patch_rectangles() else np.zeros((0, 4))}
            for b in names:
                key = b if b else "block"
                out["density_" + key] = np.asarray(self.density(b) if b else self.density(),
                                                   dtype=np.float64)
            out["phi"] = np.asarray(self.potential(), dtype=np.float64)
            target = path + suffix + ".npz"
            tmp = target + ".tmp"
            with open(tmp, "wb") as f:
                np.savez_compressed(f, **out)
            os.replace(tmp, target)
            return target
        if format == "vtk":
            target = path + suffix + ".vti"
            arrays, labels = [], []
            for b in names:
                key = b if b else "block"
                arrays.append(np.asarray(self.density(b) if b else self.density(),
                                         dtype=np.float64).reshape(n, n))
                labels.append("%s_density" % key)
            arrays.append(np.asarray(self.potential(), dtype=np.float64).reshape(n, n))
            labels.append("phi")
            lines = ['<?xml version="1.0"?>',
                     '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">',
                     '  <ImageData WholeExtent="0 %d 0 %d 0 0" Origin="0 0 0" '
                     'Spacing="%.17g %.17g 1">' % (n, n, self._L / n, self._L / n),
                     '    <Piece Extent="0 %d 0 %d 0 0">' % (n, n),
                     '      <CellData>']
            for nm, arr in zip(labels, arrays):
                lines.append('        <DataArray type="Float64" Name="%s" format="ascii">' % nm)
                lines.append("          " + " ".join("%.17g" % v for v in arr.ravel()))
                lines.append('        </DataArray>')
            lines += ['      </CellData>', '    </Piece>', '  </ImageData>', '</VTKFile>', '']
            tmp = target + ".tmp"
            with open(tmp, "w") as f:
                f.write("\n".join(lines))
            os.replace(tmp, target)
            return target
        raise ValueError("AmrSystem.write : format 'npz' | 'vtk' (received %r)" % (format,))

    def checkpoint(self, path):
        """RESTARTABLE BIT-IDENTICAL AMR CHECKPOINT v1 (npz), SINGLE-BLOCK SINGLE-RANK (ADC-65). Writes
        the FULL CONSERVATIVE STATE of EACH level (all components ; the coarse AND the fine patches,
        valid cells), the phi of each level (level 0 = WARM-START of the multigrid,
        load-bearing for the bit-identical resume), the HIERARCHY (patch_boxes), the clock (t,
        macro_step) and the regrid cadence. CONTRACT (parity with System.checkpoint) : restart does NOT
        rebuild the composition -- the script replays its add_block/set_poisson/set_refinement/set_density
        then calls sim.restart(path), which CHECKS consistency and raises otherwise. @return the path.

        SCOPE (EXPLICIT rejections, never a silently wrong/partial checkpoint) :
          - SINGLE-BLOCK only : multi-block (AmrRuntime engine) shares layout AND aux between blocks
            and does not expose the per-level/block state -> follow-up (the C++ accessors also reject).
          - SINGLE-RANK (np == 1) : the level accessors read the LOCAL fabs without an MPI gather ;
            a per-level gather (BoxArray + DistributionMapping) is a follow-up.
          - regrid_every == 0 : a bit-identical resume requires a FROZEN hierarchy (otherwise the regrid
            would re-diverge after the restart). We reject at the checkpoint (early failure, clear message).

        Out-of-scope fallback : AmrSystem.write (visualization) or a single-level System."""
        import os
        import numpy as np
        from . import _adc
        if _adc.n_ranks() != 1:
            raise NotImplementedError(
                "AmrSystem.checkpoint : MPI np>1 not wired (ADC-65 single-rank : the per-level states "
                "are read on the LOCAL fabs, the per-level gather = follow-up). Run single-rank, or "
                "use a single-level System (bit-identical checkpoint/restart including under MPI).")
        if self._s.n_blocks() != 1:
            raise NotImplementedError(
                "AmrSystem.checkpoint : multi-block not wired (ADC-65 single-block : the AmrRuntime engine "
                "shares layout AND aux between blocks and does not expose the per-level/block state = follow-up). "
                "Use a single add_block, or a single-level System (bit-identical checkpoint/restart).")
        if self._regrid_every != 0:
            raise ValueError(
                "AmrSystem.checkpoint : bit-identical resume wired for regrid_every == 0 only "
                "(frozen hierarchy) ; this system has regrid_every=%d (the post-restart regrid would re-diverge "
                "the hierarchy). Rebuild the system with regrid_every=0." % self._regrid_every)
        nlev = int(self._s.n_levels())
        pb = self._s.patch_boxes()  # (level, ilo, jlo, ihi, jhi) inclusive, index space of the level
        out = {"adc_amr_checkpoint_version": 1,
               "t": self._s.time(), "macro_step": self._s.macro_step(),
               "n": self._s.nx(), "L": self._L, "regrid_every": self._regrid_every,
               "abi_key": abi_key(), "blocks": np.array(list(self._s.block_names())),
               "n_vars": int(self._s.n_vars()), "n_levels": nlev,
               "patch_boxes": (np.asarray(pb, dtype=np.int64) if pb
                               else np.zeros((0, 5), dtype=np.int64))}
        for k in range(nlev):
            # FULL conservative state of level k (c*nf*nf + j*nf + i) + phi (nf*nf). Fine level : only
            # the patch cells are defined (0 elsewhere) ; the restart only rewrites those cells.
            out["state_%d" % k] = np.asarray(self._s.level_state(k), dtype=np.float64)
            out["phi_%d" % k] = np.asarray(self._s.level_potential(k), dtype=np.float64)
        target = path if path.endswith(".npz") else path + ".npz"
        tmp = target + ".tmp"  # ATOMIC write (.tmp + os.replace : a crash corrupts nothing)
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **out)
        os.replace(tmp, target)
        return target

    def restart(self, path):
        """RESUMES an AMR v1 checkpoint (BIT-IDENTICAL, SINGLE-BLOCK SINGLE-RANK, ADC-65). CHECKS
        consistency (version, grid, blocks, components, regrid_every == 0) then : (1) IMPOSES the
        saved fine hierarchy (set_hierarchy, instead of Berger-Rigoutsos clustering) ; (2) restores
        the FULL conservative state of each level AS-IS (no re-prolongation) ; (3) restores the
        phi of each level (level 0 = warm-start of the multigrid -> the 1st solve post-restart
        starts from the same guess) ; (4) restores the clock (t, macro_step). The COMPOSITION (add_block /
        set_poisson / set_refinement / set_density) must have been REPLAYED by the script BEFORE the call.

        ORDER : set_hierarchy BEFORE set_level_state (imposing the layout precedes restoring the
        valid cells) ; phi and clock after. The 1st step replays update() (sync_down + warm-start solve)
        then advance -- the ghosts (coarse AND fine) are remade by the step, exactly
        like after a regrid, hence the bit-identical resume without restoring any ghosts."""
        import numpy as np
        from . import _adc
        if _adc.n_ranks() != 1:
            raise NotImplementedError(
                "AmrSystem.restart : MPI np>1 not wired (ADC-65 single-rank ; cf. checkpoint). Run "
                "single-rank, or use a single-level System.")
        if self._s.n_blocks() != 1:
            raise NotImplementedError(
                "AmrSystem.restart : multi-block not wired (ADC-65 single-block ; cf. checkpoint). Use "
                "a single add_block, or a single-level System.")
        if self._regrid_every != 0:
            raise ValueError(
                "AmrSystem.restart : requires regrid_every == 0 (frozen hierarchy ; otherwise the regrid "
                "post-restart would re-diverge the restored hierarchy). Rebuild the system with "
                "regrid_every=0 before restart. (current regrid_every = %d)" % self._regrid_every)
        target = path if path.endswith(".npz") else path + ".npz"
        d = np.load(target, allow_pickle=False)
        if int(d["adc_amr_checkpoint_version"]) != 1:
            raise ValueError("restart : AMR checkpoint version %r not supported (expected 1)"
                             % (d["adc_amr_checkpoint_version"],))
        if int(d["n"]) != self._s.nx():
            raise ValueError("restart : checkpoint grid (n=%d) != system (n=%d)"
                             % (int(d["n"]), self._s.nx()))
        if float(d["L"]) != self._L:
            raise ValueError("restart : checkpoint domain (L=%r) != system (L=%r) -- different dx"
                             % (float(d["L"]), self._L))
        if int(d["regrid_every"]) != 0:
            raise ValueError("restart : checkpoint taken with regrid_every=%d != 0 (bit-identical "
                             "resume impossible)" % int(d["regrid_every"]))
        chk_blocks = [str(b) for b in d["blocks"]]
        cur_blocks = list(self._s.block_names())
        if chk_blocks != cur_blocks:
            raise ValueError("restart : checkpoint blocks %r != current composition %r "
                             "(replay the SAME composition before restart)" % (chk_blocks, cur_blocks))
        if int(d["n_vars"]) != int(self._s.n_vars()):
            raise ValueError("restart : %d components in the checkpoint, %d here"
                             % (int(d["n_vars"]), int(self._s.n_vars())))
        nlev = int(d["n_levels"])
        if nlev != int(self._s.n_levels()):
            raise ValueError("restart : %d levels in the checkpoint, %d here (does the composition / the "
                             "refinement differ ?)" % (nlev, int(self._s.n_levels())))
        # (1) IMPOSE the saved fine hierarchy (the coupler filters level 1), except a
        # SINGLE-LEVEL hierarchy (n_levels == 1, e.g. amr-schur path with no fine patch) : nothing to impose then.
        boxes = [tuple(int(x) for x in row) for row in np.asarray(d["patch_boxes"], dtype=np.int64)]
        if nlev >= 2:
            if not any(b[0] == 1 for b in boxes):
                raise ValueError("restart : %d-level hierarchy but no fine patch (level 1) "
                                 "in the checkpoint (inconsistent)." % nlev)
            self._s.set_hierarchy(boxes)
        # (2) restore the FULL conservative state of each level AS-IS (no re-prolongation) ;
        # set_level_state flattens the array and only writes the valid cells (the patches).
        for k in range(nlev):
            self._s.set_level_state(k, np.asarray(d["state_%d" % k], dtype=np.float64))
        # (3) restore the phi (level 0 = warm-start of the multigrid : bit-identical resume).
        for k in range(nlev):
            self._s.set_level_potential(k, np.asarray(d["phi_%d" % k], dtype=np.float64).ravel())
        # (4) restore the clock AFTER the state (parity with System ; macro_step advances the cadence phase).
        self._s.set_clock(float(d["t"]), int(d["macro_step"]))
    def add_equation(self, name, model, spatial=None, time=None, substeps=None):
        """Add the SINGLE AMR equation/block by dispatching on the TYPE of @p model (DSL Phase D):

        - a ModelSpec (adc.Model(...)) -> add_block (native bricks composed on the hierarchy);
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

        - ModelSpec path (adc.Model(...)): FORWARDED to AmrSystem::add_block, which SUPPORTS and
          validates them (parity with the add_block wrapper);
        - CompiledModel production path (.so): explicitly REJECTED (ValueError). The flat ABI of the
          loader (add_native_block / adc_install_native_amr) does not transport them; they would be taken
          at their defaults SILENTLY (stride=1, full backward-Euler). For a multirate .so or one with a
          partial IMEX mask, use AmrSystem.add_block (native) or add_compiled_model(AmrSystem&) directly
          (C++), which expose stride and the mask.

        @p spatial: adc.FiniteVolume(...) / adc.Spatial(...) (default minmod+rusanov+conservative).
        @p time: adc.Explicit (default) or adc.IMEX (implicit stiff source). @p substeps: overrides
        time.substeps.
        """
        from . import dsl  # late import (dsl imports this module: avoid the import cycle)

        spatial = spatial if spatial is not None else Spatial()
        time = time if time is not None else Explicit()

        # positivity_floor (ADC-259) IS wired on the NATIVE AMR transport (Density-role face states +
        # C/F fine ghost means). It is threaded below on the ModelSpec (native) branch and on the
        # amr-schur transport (the recursive add_equation on time.hyperbolic). The COMPILED .so path
        # carries it too now (ADC-322): the regenerated loader marshals it (adc_install_native_amr),
        # so the CompiledModel branch below forwards it to add_native_block instead of rejecting it.

        # --- adc.Split (Lie) / adc.Strang (2nd order): amr-schur PATH (GLOBAL condensed source stage) --
        # During AMR of System.add_equation (cf. ~line 925): we first add the block with its single
        # explicit HYPERBOLIC stage (SOURCE-FREE transport; the model must carry a NoSource source
        # brick), THEN we attach the condensed SOURCE stage (set_source_stage, C++) and the splitting
        # policy (set_time_scheme: "lie" for Split, "strang" for Strang). The condensed stage is
        # GLOBAL (assembles/solves the electrostatic/Lorentz operator on the coarse grid, composing
        # the uniform stage), as opposed to the LOCAL cell-by-cell IMEX source of time=adc.IMEX.
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

        if not isinstance(model, dsl.CompiledModel):
            raise TypeError("AmrSystem.add_equation: model must be an adc.Model(...) (ModelSpec) "
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
                "the loader inlines add_compiled_model(AmrSystem&) (symbol adc_install_native_amr)")

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

        # The flat ABI of the .so loader (adc_install_native_amr / add_native_block) transports NEITHER the
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
                "Use AmrSystem.add_block (native model adc.Model(...), wired cadence) or "
                "add_compiled_model(AmrSystem&) directly (C++) which exposes stride." % nstride)
        if getattr(time, "implicit_vars", []) or getattr(time, "implicit_roles", []):
            raise ValueError(
                "AmrSystem.add_equation: implicit_vars / implicit_roles (partial IMEX mask) not "
                "transported by the production AMR path (.so loader, flat ABI add_native_block: the "
                "mask would be empty = full backward-Euler silently). Use AmrSystem.add_block "
                "(native model adc.Model(...), wired mask) or add_compiled_model(AmrSystem&) "
                "directly (C++) which exposes the IMEX mask.")
        # Newton options / diagnostics: same flat ABI -> neither the options nor the report transit
        # through the .so loader. Explicit rejection (otherwise iters=2 / no report silently), parity with
        # the stride/mask rejection above and with System.add_equation (compiled backend).
        _reject_newton_amr_compiled("AmrSystem.add_equation", time)
        # positivity_floor (ADC-322): the regenerated .so loader carries the Zhang-Shu floor now
        # (adc_install_native_amr -> add_compiled_model -> set_compiled_block), so it is threaded
        # through instead of rejected. 0 (default) = inactive, bit-identical. The C++
        # add_native_block validates floor >= 0 and finite (parity with add_block).

        # PRE-DLOPEN guard at attach (covers the cache HIT, cf. System.add_equation): module
        # _adc stale vs .so compiled against the up-to-date headers -> actionable error, not a dlopen
        # 'symbol not found' cryptic message.
        from . import dsl as _dsl_guard
        _dsl_guard.check_compiled_matches_module(getattr(compiled, "abi_key", ""))
        gamma = compiled.gamma if compiled.gamma is not None else 1.4
        self._s.add_native_block(name, compiled.so_path, spatial.limiter, spatial.flux,
                                 spatial.recon, time.kind, gamma, nsub,
                                 getattr(spatial, "positivity_floor", 0.0))
        # ADC-291: record the named aux fields the block declares (component of the k-th name =
        # AUX_NAMED_BASE + k), so set_aux_field(block, name, array) can resolve name -> component.
        extra = list(getattr(compiled, "aux_extra_names", []) or [])
        if extra:
            self._aux_field_index[name] = {nm: dsl.AUX_NAMED_BASE + k for k, nm in enumerate(extra)}

    def _resolve_aux_field(self, block, name):
        """Resolve (block, named aux field) -> aux channel component (ADC-291). Mirror of
        System._resolve_aux_field: a canonical name is redirected to its dedicated path; an unknown
        block or an undeclared field raises (no silent component-0 fallback)."""
        from . import dsl
        if name in dsl.AUX_CANONICAL:
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

        @p halo (ADC-369): an optional adc.AuxHalo declaring this field's own coarse-level ghost
        boundary policy (foextrap / dirichlet), applied to the non-periodic faces after the shared aux
        fill. Default None inherits the shared aux BC (bit-identical)."""
        import numpy as np
        comp = self._resolve_aux_field(block, name)
        arr = np.asarray(field, dtype=float)
        self._s.set_aux_field_component(comp, arr.reshape(-1))
        if halo is not None:
            self._s.set_aux_field_halo_component(comp, halo.bc_type, halo.value)

    def add_coupling(self, coupling):
        """Add a generic inter-species COUPLED SOURCE (adc.dsl.CoupledSource(...).compile(...))
        on the SHARED AMR hierarchy (MULTI-BLOCK), refined counterpart of System.add_coupling. The source
        is transported as bytecode and interpreted on the C++ side (AmrSystem.add_coupled_source; no
        per-cell Python callback). The coupling frequency (CoupledSource.frequency) is honored:
        constant -> dt bound dt <= cfl/mu; Expr -> PER-CELL frequency mu(U) evaluated on the COARSE grid at
        each step_cfl (the freq_prog_* vectors are forwarded). Must be called BEFORE the first
        step (the source is frozen then injected at the lazy build of the runtime engine)."""
        from . import dsl  # late import (dsl imports this module: avoid the import cycle)

        if isinstance(coupling, dsl.CompiledCoupledSource):
            self._s.add_coupled_source(coupling.in_blocks, coupling.in_roles, coupling.consts,
                                       coupling.out_blocks, coupling.out_roles, coupling.prog_ops,
                                       coupling.prog_args, coupling.prog_lens,
                                       getattr(coupling, "frequency", 0.0), coupling.name,
                                       getattr(coupling, "freq_prog_ops", []),
                                       getattr(coupling, "freq_prog_args", []))
        else:
            raise TypeError("AmrSystem.add_coupling expects a CompiledCoupledSource "
                            "(adc.dsl.CoupledSource(...).compile(...)): the AMR coupled source is "
                            "MULTI-BLOCK and described in formulas")

    def __getattr__(self, attr):
        return getattr(self._s, attr)


class PythonFlux:
    """PROTOTYPING backend (host, numpy) for the Flux interface: the user provides the physical
    flux and the wave speed in Python, and PythonFlux assembles the residual -div(F*) by finite
    volumes (Rusanov, order 1, periodic domain) over the whole array at once.

    OUT of the GPU/MPI hot path: this is a pure HOST path (numpy), it NEVER goes through a Kokkos
    kernel. For production (GPU/MPI), compose a COMPILED flux (adc.CompressibleFlux brick,
    adc.ExB...). PythonFlux formalizes the pattern of the custom_scheme case: iterate quickly on a
    novel flux without recompiling (adc.System serving as Poisson oracle if needed).

    flux(U, dir) -> F: U and F are numpy (ncomp, n, n); dir = 0 (x) or 1 (y).
    max_wave_speed(U) -> float: bound for the Rusanov flux and the CFL.
    """

    def __init__(self, flux, max_wave_speed):
        self.flux = flux
        self.max_wave_speed = max_wave_speed

    def residual(self, U, dx, dy=None):
        """-div(F*) by Rusanov flux (order 1, periodic). U numpy (ncomp, n, n); returns dU/dt."""
        import numpy as np
        dy = dx if dy is None else dy
        a = float(self.max_wave_speed(U))
        res = np.zeros_like(U)
        for axis, h, d in ((2, dx, 0), (1, dy, 1)):  # x = axis 2, y = axis 1
            F = self.flux(U, d)
            UR = np.roll(U, -1, axis=axis)
            FR = np.roll(F, -1, axis=axis)
            face = 0.5 * (F + FR) - 0.5 * a * (UR - U)       # flux at the +d face of each cell
            res -= (face - np.roll(face, 1, axis=axis)) / h  # -div: (F_{i+1/2} - F_{i-1/2}) / h
        return res

    def cfl_dt(self, U, h, cfl=0.4):
        """Stable time step: dt = cfl * h / max_wave_speed(U)."""
        return cfl * h / max(float(self.max_wave_speed(U)), 1e-30)


from . import integrate  # noqa: E402  (after the definition of System; without numpy dependency)
from . import time  # noqa: E402  (adc.time.Program IR; pure stdlib, no numpy/_adc dependency)
from . import model  # noqa: E402  (adc.model operator-first type system; pure stdlib, Spec 2)
from . import math  # noqa: E402  (adc.math board operators; pure stdlib, Spec 3, dsl lazy)
from . import lib  # noqa: E402  (adc.lib typed-brick descriptor catalog; pure stdlib, Spec 3)
from . import physics  # noqa: E402  (adc.physics board model authoring; numpy-free import, Spec 3)
from . import library  # noqa: E402  (adc.compile_library brick-manifest layer; numpy-free, Spec 3)
from .library import (  # noqa: E402,F401  (re-export: brick-library manifest API, Spec 3 section 21)
    LibraryManifest, compile_library, read_library_manifest)
from .time import CompiledTime  # noqa: E402,F401  (re-export: compiled-Program time policy)


# LAZY adc.dsl (PEP 562): dsl.py does `import numpy` at module level (host evaluator of the
# prototype). The eager import made numpy mandatory for the ENTIRE `import adc`, whereas the
# native path (System/add_block) and the production backend do not need it. With this
# __getattr__, `adc.dsl.Model(...)` and `from adc import dsl` work identically, but numpy
# is required ONLY AT the first use of the DSL -- and its absence gives a targeted message (doctor too).
def __getattr__(name):
    if name == "dsl":
        import importlib
        try:
            return importlib.import_module(".dsl", __name__)
        except ImportError as exc:
            raise ImportError(
                "adc.dsl requires numpy in this interpreter (host evaluator of the DSL): "
                "`pip install numpy` / `conda install numpy`. The rest of adc (System, add_block) "
                "works without it. Cause: %s" % exc) from exc
    # adc.compile_problem / adc.CompiledProblem live in adc.dsl (which imports numpy); expose them at
    # the top level LAZILY so `import adc` stays numpy-free until the DSL/compile path is first used.
    if name in ("compile_problem", "CompiledProblem"):
        return getattr(__getattr__("dsl"), name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
