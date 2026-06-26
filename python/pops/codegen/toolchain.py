"""pops.codegen.toolchain : BUILD-INFRA / toolchain helpers for DSL compilation.

Extracted verbatim from pops.dsl (bodies byte-for-byte); only import lines adjusted.
Public API re-exported from pops.codegen.__init__.
"""
import os
import shutil
import sys


# --- _pops access (mirrors dsl.py: try top-level then relative) ---
def _pops_module():
    """The _pops extension module if it is loadable, otherwise None (dsl.py stays usable alone)."""
    try:
        import _pops
        return _pops
    except Exception:
        try:
            from pops import _pops  # noqa: PLC0415  # SPEC4-TODO: lazy import
            return _pops
        except Exception:
            return None


# --- Signature of the core header tree (ABI key of the "production" path) -------------
# The "production" backend (compile_native) emits a .so loader that inlines the header template
# pops::add_compiled_model and calls off-line methods of the already-loaded _pops module. Loader and
# module MUST share the same C++ ABI (same headers, compiler, standard). We materialize the
# "header signature" in the ABI key (pops/runtime/abi_key.hpp, token POPS_HEADER_SIG) ; the
# module build bakes it (CMake) and compile_native re-bakes it (-D flag) by computing it IDENTICALLY.
# The computation MUST be bit-for-bit identical on the CMake side (python/CMakeLists.txt) and here : sha256 of the
# sorted concatenation "<relpath>\n<sha256(content)>\n" of each .hpp/.h under include/. cf. abi_key.hpp.
def pops_header_signature(include):
    """Stable signature of the pops header tree under @p include : sha256 of the sorted concatenation
    "<relative path>\\n<sha256 of content>\\n" of each .hpp/.h. EXACT MIRROR of the CMake computation
    (python/CMakeLists.txt) : if a header changes, the signature changes on both sides, so the ABI
    key diverges and add_native_block raises an explicit error (never silent UB)."""
    import hashlib
    import os
    entries = []
    for root, _dirs, files in os.walk(include):
        for fn in files:
            if fn.endswith((".hpp", ".h")):
                p = os.path.join(root, fn)
                rel = os.path.relpath(p, include).replace(os.sep, "/")  # CMake writes paths with '/' (even on Windows)
                with open(p, "rb") as f:
                    digest = hashlib.sha256(f.read()).hexdigest()
                entries.append("%s\n%s\n" % (rel, digest))
    blob = "".join(sorted(entries)).encode()
    return hashlib.sha256(blob).hexdigest()


# --- Auto-detection of the pops include directory -----------------------------------
# To make m.compile(...) ergonomic, the pops headers directory is deduced automatically
# when the caller does not pass it. MIRROR of adc_cases/common/native.py::pops_include : we try
# $POPS_INCLUDE (explicit override), then we climb from the installed `pops` package (build-py/python/
# pops/ -> ../../../include), then the neighboring repo ../adc_cpp/include. Validity criterion : the
# canonical file pops/mesh/multifab.hpp exists. No hard import of pops here (the dsl module may be loaded
# outside the package) : we resolve `pops.__file__` lazily.
def pops_include():
    """include/ directory of adc_cpp (header-only headers of the core), auto-detected.

    Priority : $POPS_INCLUDE (override), otherwise from the installed `pops` package
    (.../pops -> ../../../include), otherwise the neighboring repo (.../adc_cpp/include from this module).
    Requires that pops/mesh/storage/multifab.hpp exists. Raises RuntimeError if not found (diagnostic listing the
    candidates), so as to NEVER compile against a silently wrong include."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))           # .../python/pops/codegen
    candidates = []
    env = os.environ.get("POPS_INCLUDE")
    if env:
        candidates.append(env)
    try:
        import pops as _pops_pkg
        pkg = os.path.dirname(os.path.abspath(_pops_pkg.__file__))   # .../pops
        candidates.append(os.path.normpath(os.path.join(pkg, "..", "..", "..", "include")))
    except Exception:
        pass
    # from this file (python/pops/codegen/toolchain.py) : python/pops/codegen -> python/pops -> python -> repo root -> include
    candidates.append(os.path.normpath(os.path.join(here, "..", "..", "..", "include")))
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "pops", "mesh", "storage", "multifab.hpp")):
            return c
    raise RuntimeError(
        "pops headers not found (looking for pops/mesh/storage/multifab.hpp). "
        "Pass include=<adc_cpp>/include or set POPS_INCLUDE. Candidates tried : "
        + ", ".join(repr(c) for c in candidates))


# --- C++ standard of the native loader (ABI boundary of the "production" path) ----------
# The "production" backend generates a .so loader that inlines add_compiled_model<> and calls off-line
# methods of the ALREADY-loaded _pops module. The ABI key (pops/runtime/abi_key.hpp) encodes __cplusplus :
# the loader and the module must therefore share the SAME C++ standard, otherwise add_native_block rejects
# ("incompatible ABI"). The module bakes its real standard (POPS_CXX_STD : 20 under Kokkos because CUDA 12.x
# has no -std=c++23, 23 otherwise) and exposes it as _pops.__cxx_std__. We derive the expected -std flag of the
# native model from it INSTEAD OF freezing c++23 (which broke the native path under Kokkos/GH200, where the module is
# in c++20). Direct MIRROR of the build, so never a silent gap between loader and model.
def loader_cxx_std():
    """Flag '-std=c++NN' that the native model (backend="production") MUST use to share the ABI
    of the loaded _pops module. Source of truth : _pops.__cxx_std__ (integer 20/23 baked by the build, =
    POPS_CXX_STD : 20 under Kokkos, 23 otherwise). Graceful fallbacks if the attribute is missing (old module) :
    we parse __cplusplus from _pops.abi_key() (>202002L -> c++23, otherwise c++20) ; failing all that,
    we fall back to the historical default c++23 (non-Kokkos host case, unchanged)."""
    try:
        import _pops
    except Exception:
        try:
            from pops import _pops  # noqa: PLC0415  # SPEC4-TODO: lazy import
        except Exception:
            _pops = None
    std = _pops_cxx_std_from_module(_pops) if _pops is not None else None
    return std or "c++23"


def _pops_cxx_std_from_module(mod):
    """C++ standard of the module @p mod as 'c++NN', or None if undeterminable. Priority to the integer
    __cxx_std__ (baked by the build) ; otherwise we extract std=<__cplusplus> from the ABI key."""
    n = getattr(mod, "__cxx_std__", None)
    if isinstance(n, int) and n in (20, 23):
        return "c++%d" % n
    # Fallback : parse "...;std=<__cplusplus>;..." from the ABI key (old module without __cxx_std__).
    abi_key = getattr(mod, "abi_key", None)
    if callable(abi_key):
        try:
            key = abi_key()
        except Exception:
            return None
        for tok in str(key).split(";"):
            if tok.startswith("std="):
                val = tok[len("std="):].rstrip("Ll")
                if val.isdigit():
                    return "c++23" if int(val) > 202002 else "c++20"
    return None


# --- Compiler of the DSL .so files (ABI boundary, counterpart of loader_cxx_std) -------------------------
# REAL BUG fixed here : in an active conda env, `which c++` often points to ANOTHER compiler
# than the one that built _pops (old gcc/clang from the conda PATH). Symptom : the runtime compilation
# of the production DSL loader fails with the raw compiler error ("error: invalid value 'c++23'
# in '-std=c++23'") ; and even if it passed, the ABI key (which encodes __VERSION__ of the compiler,
# cf. abi_key.hpp) would reject the .so ("incompatible ABI"). The ONLY guaranteed-compatible compiler
# is the one from the _pops build : CMake bakes it (POPS_CXX_COMPILER -> _pops.__cxx_compiler__) and we
# prefer it here over the PATH. $POPS_CXX remains the conscious override (chosen conda toolchain, wrapper...).
def loader_cxx_compiler():
    """Path of the compiler that BUILT the _pops module (baked by CMake as __cxx_compiler__),
    or None if it is unknown (old module, manual build) or absent from this machine.

    macOS : CMake often bakes the INTERNAL c++ of the Xcode / CommandLineTools toolchain
    (.../XcodeDefault.xctoolchain/usr/bin/c++), which invokes clang WITHOUT an SDK sysroot -> every DSL
    .so fails on \"'string' file not found\". The /usr/bin/c++ shim (xcrun) runs THE SAME
    clang while resolving the SDK : same __VERSION__, hence same ABI key -- so we prefer the shim
    (pitfall and remedy identical to compile_loader of the native C++ tests)."""
    import sys
    mod = _pops_module()
    cc = getattr(mod, "__cxx_compiler__", "") if mod is not None else ""
    if not (cc and os.path.isfile(cc) and os.access(cc, os.X_OK)):
        return None
    if sys.platform == "darwin" and (".xctoolchain/" in cc or "/CommandLineTools/" in cc) \
            and os.path.isfile("/usr/bin/c++"):
        return "/usr/bin/c++"
    return cc


def _check_headers_match_module(include):
    """PRE-DLOPEN GUARD of the native path (real bug) : if the headers under @p include have changed since
    the build of _pops (recent pull, another clone...), the loader compiled against them references
    C++ signatures that the OLD module does not export -> the dlopen of add_native_block fails BEFORE the
    ABI guard, with a cryptic error ("symbol not found in flat namespace '__ZN3adc6System13
    install_block...'"). So we compare HERE, before any compilation, the header signature baked
    into the module with that of the @p include tree, and we fail with a clear remedy. No-op if the
    module is not loadable or has no signature (manual build : historical degradation)."""
    from .abi import module_header_signature  # intra-package; avoids circular at module level
    baked = module_header_signature()
    current = pops_header_signature(include)
    if baked is not None and current != baked:
        mod = _pops_module()
        so = getattr(mod, "__file__", "(unknown)")
        raise RuntimeError(
            "pops.dsl : the pops headers of %r DO NOT MATCH those with which the _pops module "
            "was built (%s).\n"
            "  current header signature : %s\n"
            "  signature baked in _pops  : %s\n"
            "Typical cause : `git pull` / headers edited AFTER the module build -> the DSL loader "
            "would reference C++ signatures absent from the module (dlopen : 'symbol not found').\n"
            "Remedy : REBUILD the module with these headers :\n"
            "  cmake --preset python && cmake --build --preset python   (or the usual build-py)\n"
            "or point POPS_INCLUDE at the headers of the build that produced this module."
            % (include, so, current[:16], baked[:16]))
    return current  # signature of the @p include tree, reusable (avoids a 2nd walk+sha256)


def resolve_auto_backend(include=None):
    """DEFAULT backend policy (backend='auto', decision recorded -- ADC-63).

    'production' (zero-copy native loader, strict add_block parity) AS SOON AS the
    toolchain parity with the _pops module is established : module loadable + known baked compiler +
    header signature of @p include == the one baked into the module. OTHERWISE 'aot' (historical
    default : host-marshaled, works without module or parity). Never silent : returns
    (backend, reason) and the facades set the reason on CompiledModel.backend_auto_reason.
    An EXPLICIT backend passed by the caller short-circuits this policy (unchanged)."""
    from .abi import module_header_signature  # intra-package; avoids circular at module level
    mod = _pops_module()
    if mod is None:
        return "aot", "_pops module not loadable (the production path requires the module)"
    if not loader_cxx_compiler():
        return "aot", "module compiler unknown (old module or manual build)"
    baked = module_header_signature()
    if not baked:
        return "aot", "header signature absent from the module (manual build)"
    try:
        inc = include if include is not None else pops_include()
        sig = pops_header_signature(inc)
    except Exception as e:  # headers not found / unreadable -> fall back on default
        return "aot", "pops headers not found for parity (%s)" % e
    if sig != baked:
        return "aot", ("headers != module (rebuild the module or point at the build headers ; "
                       "production would refuse, cf. _check_headers_match_module)")
    return "production", "toolchain parity established (module + baked compiler + matching headers)"


def _default_cxx(cxx=None):
    """CENTRALIZED resolution of the DSL .so compiler (all backends). Priority :
      1. explicit cxx (caller argument) ;
      2. $POPS_CXX (conscious environment override) ;
      3. the compiler that built _pops (the only one guaranteed ABI-compatible, cf. above) ;
      4. c++ / g++ / clang++ from the PATH (historical behavior, last resort)."""
    return (cxx or os.environ.get("POPS_CXX") or loader_cxx_compiler()
            or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++"))


# Historical spellings of the same language levels : clang < 17 / gcc < 11 know
# only 'c++2b'/'c++2a'. Same level requested ; on an OLD compiler __cplusplus may differ from the
# module -> if applicable, explicit ABI rejection downstream (never silent UB).
_STD_ALIAS = {"c++23": "c++2b", "c++20": "c++2a"}


def _run_compile(cmd, what):
    """Run the compilation command @p cmd CAPTURING stderr : on failure, raises a
    SELF-CONTAINED RuntimeError (command + compiler output + remedies) instead of the raw
    CalledProcessError whose message contains only the command line (real bug : the user sees
    only a 'returned non-zero exit status 1' drowned in the traceback)."""
    import subprocess
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        err = (r.stderr or b"").decode(errors="replace").strip()
        out = (r.stdout or b"").decode(errors="replace").strip()  # MSVC cl writes errors on STDOUT
        err = (err + "\n" + out).strip() if out else err
        raise RuntimeError(
            "pops.dsl: compiling the .so (%s) failed (exit %d).\n"
            "Command: %s\n"
            "Compiler output:\n%s\n"
            "Hints: `python -c \"import pops; pops.doctor()\"` diagnoses the environment "
            "(compiler/standard/headers); POPS_CXX forces a specific compiler."
            % (what, r.returncode, " ".join(cmd), err[:4000] or "(empty)"))


_probe_cache = {}  # (cc, std) -> effective std: avoids re-probing repeatedly (N compiled models)


def _probe_cxx_std(cc, std):
    """Checks BEFORE compilation that @p cc accepts -std=@p std (probe -fsyntax-only on empty source).

    Returns the EFFECTIVE std: @p std if it passes, otherwise its historical alias (c++23 -> c++2b) if
    it passes, otherwise raises an ACTIONABLE RuntimeError (compiler used, build compiler,
    solutions) instead of the raw compiler error. Skipped for nvcc_wrapper (different -x
    semantics; explicit GPU path, already gated by POPS_KOKKOS_CXX/POPS_KOKKOS_USE_NVCC_WRAPPER).
    Result memoized per (cc, std): a single probe even if N models are compiled in a row."""
    import subprocess
    if "nvcc" in os.path.basename(cc or ""):
        return std
    if sys.platform == "win32":
        return std  # cl/clang-cl: -fsyntax-only probe inapplicable; std translated to /std: at compile
    cached = _probe_cache.get((cc, std))
    if cached is not None:
        return cached

    def accepts(s):
        try:
            r = subprocess.run([cc, "-x", "c++", "-std=" + s, "-fsyntax-only", "-"],
                               input=b"", capture_output=True, timeout=60)
            return r.returncode == 0, (r.stderr or b"").decode(errors="replace")
        except Exception as exc:  # not found, not executable, timeout: same actionable diagnostic
            return False, str(exc)

    good, err = accepts(std)
    if good:
        _probe_cache[(cc, std)] = std
        return std
    alias = _STD_ALIAS.get(std)
    if alias:
        good_alias, _ = accepts(alias)
        if good_alias:
            _probe_cache[(cc, std)] = alias
            return alias
    baked = loader_cxx_compiler()
    raise RuntimeError(
        "pops.dsl: the compiler %r does not support -std=%s (standard required to share the ABI of "
        "the _pops module).\nCompiler output:\n%s\n"
        "Compiler of the _pops build: %s\n"
        "Solutions:\n"
        "  - use the build compiler: export POPS_CXX=%r (or cxx=... in m.compile);\n"
        "  - macOS: update Xcode / the Command Line Tools (recent AppleClang);\n"
        "  - conda: `conda install -c conda-forge cxx-compiler` (gcc>=13 / clang>=17) then "
        "export POPS_CXX=$CONDA_PREFIX/bin/clang++ (macOS) or $CONDA_PREFIX/bin/g++ (Linux).\n"
        "NB: a compiler DIFFERENT from the build one may compile but then be rejected "
        "('incompatible ABI': the ABI key encodes the compiler version); prefer the build one."
        % (cc, std, (err or "").strip()[:800],
           baked or "(unknown: module without __cxx_compiler__, rebuild _pops to bake it)",
           baked or "<path/to/build/compiler>"))


def _native_kokkos_root():
    """Kokkos root to compile the DSL loaders with the SAME backend as the _pops module.

    adc_cpp is KOKKOS-ONLY: every DSL .so that includes the pops headers (aot, native) MUST be compiled
    with Kokkos (for_each.hpp #error otherwise). The root is read from POPS_KOKKOS_ROOT / Kokkos_ROOT /
    KOKKOS_ROOT; None if not found (the caller then raises an explicit error)."""
    for key in ("POPS_KOKKOS_ROOT", "Kokkos_ROOT", "KOKKOS_ROOT"):
        root = os.environ.get(key)
        if root and os.path.isfile(os.path.join(root, "include", "Kokkos_Core.hpp")):
            return root
    return None


def _libomp_prefix():
    """Homebrew libomp prefix on macOS (for -Xpreprocessor -fopenmp), or None. AppleClang does not
    handle `-fopenmp` alone: it needs -Xpreprocessor -fopenmp + the libomp include/lib (cf. CMakeLists)."""
    if sys.platform != "darwin":
        return None
    import subprocess
    try:
        p = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True, text=True)
        prefix = p.stdout.strip()
        if prefix and os.path.isdir(os.path.join(prefix, "lib")):
            return prefix
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _native_feature_key():
    """Traits that change the inline code of the native loader and must therefore enter the cache (else
    a cached SERIAL .so would be reused on a Kokkos module -> silent serial fallback).

    Beyond on/off, the key fingerprints KokkosCore_config.h of the targeted install: this generated
    header encodes the Kokkos VERSION AND its active backends. Without it, changing Kokkos between two
    runs (update, switch Serial->OpenMP at the same prefix) would not invalidate the cache -> reuse
    of a .so compiled against the old Kokkos."""
    root = _native_kokkos_root()
    if not root:
        kk = "kokkos=off"
    else:
        import hashlib
        cfg = os.path.join(root, "include", "KokkosCore_config.h")
        try:
            with open(cfg, "rb") as f:
                tag = hashlib.sha256(f.read()).hexdigest()[:12]
        except OSError:
            tag = "unknown"
        kk = "kokkos=on;kcfg=%s" % tag
    # ADC-319: the MPI seam of the loader (POPS_HAS_MPI on/off, cf. _native_mpi_flags) changes the
    # compiled code (real comm vs serial stubs n_ranks()=1/my_rank()=0) -> it MUST enter the cache,
    # else a SERIAL-stub .so would be reused on an MPI module and any distributed layout built inside
    # the loader (e.g. AmrSystem(distribute_coarse=True)) would replicate on every rank (no scaling).
    mod = _pops_module()
    mpi = "mpi=on" if (mod is not None and getattr(mod, "__has_mpi__", False)) else "mpi=off"
    return "%s;%s" % (kk, mpi)


def _warn_kokkos_parity():
    """Warns (without blocking) when the BACKEND of the native loader would diverge from that of the _pops
    module:
    - Kokkos module + serial loader (POPS_KOKKOS_ROOT missing) -> the DSL block falls back to the
      SERIAL path silently: zero-copy but DOES NOT SCALE with threads/GPU (ROMEO measurement: DSL warm
      invariant threads=1/4/8). This is a silent PERF degradation, not a crash -> explicit warning.
    - serial module + POPS_KOKKOS_ROOT defined -> the loader would instantiate -DPOPS_HAS_KOKKOS against a
      module that is not (divergent allocator/type layouts, not covered by the ABI key).
    Source of truth: _pops.__has_kokkos__ (baked by the build) vs _native_kokkos_root() (env)."""
    import warnings
    mod = _pops_module()
    has = getattr(mod, "__has_kokkos__", None) if mod is not None else None
    root = _native_kokkos_root()
    if has is True and root is None:
        warnings.warn(
            "pops.dsl: the _pops module is compiled WITH Kokkos but POPS_KOKKOS_ROOT is not defined "
            "-> the 'production' DSL block will be compiled SERIAL (it will run, but will not scale "
            "with threads/GPU). Set POPS_KOKKOS_ROOT=<build Kokkos install> for parity.",
            RuntimeWarning, stacklevel=3)
    elif has is False and root is not None:
        warnings.warn(
            "pops.dsl: POPS_KOKKOS_ROOT is defined but the _pops module is SERIAL (compiled without "
            "-DPOPS_USE_KOKKOS=ON) -> the loader would be compiled with Kokkos against a module that is "
            "not (divergent memory layouts, not covered by the ABI key). Remove "
            "POPS_KOKKOS_ROOT or rebuild _pops with Kokkos (preset python-parallel).",
            RuntimeWarning, stacklevel=3)


def _env_truthy(value):
    return str(value or "").strip().lower() in ("1", "on", "true", "yes", "y")


def _native_kokkos_compiler(cxx):
    """Effective compiler of the native loader. CUDA must be EXPLICIT (POPS_KOKKOS_CXX=<nvcc_wrapper>
    or POPS_KOKKOS_USE_NVCC_WRAPPER=1): many Kokkos OpenMP installs also provide a
    nvcc_wrapper, choosing it by default would break CPU jobs without nvcc. For Kokkos OpenMP, the host suffices."""
    if cxx:
        return cxx
    env = os.environ.get("POPS_KOKKOS_CXX")
    if env:
        return env
    root = _native_kokkos_root()
    wrapper = os.path.join(root, "bin", "nvcc_wrapper") if root else ""
    if wrapper and os.path.exists(wrapper) and _env_truthy(os.environ.get("POPS_KOKKOS_USE_NVCC_WRAPPER")):
        return wrapper
    # Centralized fallback: $POPS_CXX, then the _pops build compiler (the only ABI-compatible one
    # guaranteed for a native loader), then the PATH (historical). cf. _default_cxx.
    return _default_cxx(None)


def _pops_import_lib():
    """(Windows, ADC-100) Path of the import library _pops.lib (System POPS_EXPORT symbols) against
    which to link the DSL .dll. Searched next to the _pops module. None if absent."""
    mod = _pops_module()
    if mod is None:
        return None
    d = os.path.dirname(getattr(mod, "__file__", "") or "")
    cand = os.path.join(d, "_pops.lib")
    return cand if os.path.exists(cand) else None


def _native_kokkos_flags():
    """Compile/link flags so the production DSL loader instantiates add_compiled_model WITH Kokkos.

    The loader contains the header-only templates (make_block / assemble_rhs / for_each_cell). Compiled
    without POPS_HAS_KOKKOS while _pops is built WITH Kokkos, the DSL block stays zero-copy but its kernels
    are instantiated on the SERIAL fallback (does not scale threads/GPU). POPS_KOKKOS_ROOT forces parity."""
    root = _native_kokkos_root()
    if not root:
        return [], []
    inc = os.path.join(root, "include")
    if sys.platform == "win32":
        # MSVC/clang-cl: Kokkos as a SHARED DLL -> link the import lib kokkoscore.lib (ONE single runtime;
        # _pops loads the same kokkoscore.dll). cl accepts -D/-I. No -fopenmp/-ldl/-pthread (POSIX).
        return (["-DPOPS_HAS_KOKKOS", "-DKOKKOS_DEPENDENCE", "-I", inc],
                [os.path.join(root, "lib", "kokkoscore.lib")])
    compile_flags = ["-DPOPS_HAS_KOKKOS", "-DKOKKOS_DEPENDENCE", "-I", inc]
    # Do NOT link libkokkos* INTO the .so: the _pops module has already loaded the Kokkos runtime, a
    # SINGLETON (global registry of execution spaces), and add_native_block promotes it to global
    # scope (RTLD_GLOBAL). Linking a 2nd copy of Kokkos into the loader gives two runtimes: the
    # computation runs, but on exit Kokkos::finalize() aborts "Execution space instance to be removed
    # couldn't be found!" (SIGABRT, atexit). So we leave the Kokkos symbols UNDEFINED in the .so
    # (resolved at load time against the module, like install_block/grid_context). Only -fopenmp (the
    # OpenMP backend of the generated kernel) + -ldl/-pthread remain. ROMEO validation: DSL warm scales and
    # clean exit (exit 0), ratio ~0.96x the bricks.
    link_flags = ["-ldl", "-pthread"]
    if "nvcc_wrapper" not in os.path.basename(_native_kokkos_compiler(None) or ""):
        # OpenMP required for the Kokkos OpenMP exec space (harmless/ignored under Kokkos Serial).
        if sys.platform == "darwin":
            # macOS / AppleClang: `-fopenmp` alone is rejected -> -Xpreprocessor -fopenmp (+ include
            # Homebrew libomp if present). We do NOT link libomp (-lomp) into the .so: a 2nd copy of
            # libomp gives TWO OpenMP runtimes ("mutex lock failed: Invalid argument" at runtime).
            # The omp_*/__kmpc_* symbols resolve at load time (flat namespace, -undefined
            # dynamic_lookup set by compile_aot/compile_native) against the libomp already loaded by _pops.
            libomp = _libomp_prefix()
            compile_flags += ["-Xpreprocessor", "-fopenmp"]
            if libomp is not None:
                compile_flags += ["-I", os.path.join(libomp, "include")]
        else:
            # ELF/Linux: libgomp is shared by soname (no double-runtime), -fopenmp on both sides.
            compile_flags.append("-fopenmp")
            link_flags.append("-fopenmp")
    return compile_flags, link_flags


def pops_loader_build_flags(cxx=None):
    """Flags to compile OUTSIDE CMake a .so that INCLUDES the pops headers and will be loaded into the
    _pops module (DSL loaders, ABI tests). adc_cpp being Kokkos-only, the .so MUST be compiled with
    Kokkos (for_each.hpp #error otherwise). Returns (compiler, compile_flags, link_flags): Kokkos +
    (macOS) -undefined dynamic_lookup. The Kokkos symbols stay UNDEFINED, resolved at load time
    against the Kokkos runtime already loaded by _pops (no 2nd copy). Raises if no installed Kokkos is
    visible via POPS_KOKKOS_ROOT / Kokkos_ROOT (Serial suffices on CPU)."""
    if _native_kokkos_root() is None:
        raise RuntimeError(
            "pops_loader_build_flags: adc_cpp is Kokkos-only -- point to an installed Kokkos via "
            "POPS_KOKKOS_ROOT (or Kokkos_ROOT), e.g. `export POPS_KOKKOS_ROOT=/path/to/kokkos`.")
    cc = _native_kokkos_compiler(cxx)
    cflags, lflags = _native_kokkos_flags()
    if sys.platform == "darwin":
        cflags = list(cflags) + ["-undefined", "dynamic_lookup"]
    return cc, cflags, lflags
