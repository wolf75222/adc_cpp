"""adc.codegen.toolchain : toolchain / compile-infrastructure helpers.

Verbatim extraction of the toolchain block of adc.dsl (lines 57-735).
Covers: header-signature, include auto-detection, C++ standard / compiler
resolution, cache-path helpers, Kokkos/MPI flag builders, and the
module-level aux-channel constants.

Do NOT import this module directly from outside the package.  The canonical
entry point is adc.dsl which still owns the full source.
"""
import os
import shutil
import sys


# --- Signature of the core header tree (ABI key of the "production" path) -------------
# The "production" backend (compile_native) emits a .so loader that inlines the header template
# adc::add_compiled_model and calls off-line methods of the already-loaded _adc module. Loader and
# module MUST share the same C++ ABI (same headers, compiler, standard). We materialize the
# "header signature" in the ABI key (adc/runtime/abi_key.hpp, token ADC_HEADER_SIG) ; the
# module build bakes it (CMake) and compile_native re-bakes it (-D flag) by computing it IDENTICALLY.
# The computation MUST be bit-for-bit identical on the CMake side (python/CMakeLists.txt) and here : sha256 of the
# sorted concatenation "<relpath>\n<sha256(content)>\n" of each .hpp/.h under include/. cf. abi_key.hpp.
def adc_header_signature(include):
    """Stable signature of the adc header tree under @p include : sha256 of the sorted concatenation
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


# --- Auto-detection of the adc include directory -----------------------------------
# To make m.compile(...) ergonomic, the adc headers directory is deduced automatically
# when the caller does not pass it. MIRROR of adc_cases/common/native.py::adc_include : we try
# $ADC_INCLUDE (explicit override), then we climb from the installed `adc` package (build-py/python/
# adc/ -> ../../../include), then the neighboring repo ../adc_cpp/include. Validity criterion : the
# canonical file adc/mesh/multifab.hpp exists. No hard import of adc here (the dsl module may be loaded
# outside the package) : we resolve `adc.__file__` lazily.
def adc_include():
    """include/ directory of adc_cpp (header-only headers of the core), auto-detected.

    Priority : $ADC_INCLUDE (override), otherwise from the installed `adc` package
    (.../adc -> ../../../include), otherwise the neighboring repo (.../adc_cpp/include from this module).
    Requires that adc/mesh/storage/multifab.hpp exists. Raises RuntimeError if not found (diagnostic listing the
    candidates), so as to NEVER compile against a silently wrong include."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))           # .../python/adc/codegen
    candidates = []
    env = os.environ.get("ADC_INCLUDE")
    if env:
        candidates.append(env)
    try:
        import adc as _adc_pkg
        pkg = os.path.dirname(os.path.abspath(_adc_pkg.__file__))   # .../adc
        candidates.append(os.path.normpath(os.path.join(pkg, "..", "..", "..", "include")))
    except Exception:
        pass
    # from this file (python/adc/codegen/toolchain.py) : codegen -> adc -> python -> repo root -> include
    candidates.append(os.path.normpath(os.path.join(here, "..", "..", "..", "include")))
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "adc", "mesh", "storage", "multifab.hpp")):
            return c
    raise RuntimeError(
        "adc headers not found (looking for adc/mesh/storage/multifab.hpp). "
        "Pass include=<adc_cpp>/include or set ADC_INCLUDE. Candidates tried : "
        + ", ".join(repr(c) for c in candidates))


# --- C++ standard of the native loader (ABI boundary of the "production" path) ----------
# The "production" backend generates a .so loader that inlines add_compiled_model<> and calls off-line
# methods of the ALREADY-loaded _adc module. The ABI key (adc/runtime/abi_key.hpp) encodes __cplusplus :
# the loader and the module must therefore share the SAME C++ standard, otherwise add_native_block rejects
# ("incompatible ABI"). The module bakes its real standard (ADC_CXX_STD : 20 under Kokkos because CUDA 12.x
# has no -std=c++23, 23 otherwise) and exposes it as _adc.__cxx_std__. We derive the expected -std flag of the
# native model from it INSTEAD OF freezing c++23 (which broke the native path under Kokkos/GH200, where the module is
# in c++20). Direct MIRROR of the build, so never a silent gap between loader and model.
def loader_cxx_std():
    """Flag '-std=c++NN' that the native model (backend="production") MUST use to share the ABI
    of the loaded _adc module. Source of truth : _adc.__cxx_std__ (integer 20/23 baked by the build, =
    ADC_CXX_STD : 20 under Kokkos, 23 otherwise). Graceful fallbacks if the attribute is missing (old module) :
    we parse __cplusplus from _adc.abi_key() (>202002L -> c++23, otherwise c++20) ; failing all that,
    we fall back to the historical default c++23 (non-Kokkos host case, unchanged)."""
    try:
        import _adc
    except Exception:
        try:
            from adc import _adc  # adc package : the extension is a submodule
        except Exception:
            _adc = None
    std = _adc_cxx_std_from_module(_adc) if _adc is not None else None
    return std or "c++23"


def _adc_cxx_std_from_module(mod):
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
# than the one that built _adc (old gcc/clang from the conda PATH). Symptom : the runtime compilation
# of the production DSL loader fails with the raw compiler error ("error: invalid value 'c++23'
# in '-std=c++23'") ; and even if it passed, the ABI key (which encodes __VERSION__ of the compiler,
# cf. abi_key.hpp) would reject the .so ("incompatible ABI"). The ONLY guaranteed-compatible compiler
# is the one from the _adc build : CMake bakes it (ADC_CXX_COMPILER -> _adc.__cxx_compiler__) and we
# prefer it here over the PATH. $ADC_CXX remains the conscious override (chosen conda toolchain, wrapper...).
def _adc_module():
    """The _adc extension module if it is loadable, otherwise None (dsl.py stays usable alone)."""
    try:
        import _adc
        return _adc
    except Exception:
        try:
            from adc import _adc
            return _adc
        except Exception:
            return None


def loader_cxx_compiler():
    """Path of the compiler that BUILT the _adc module (baked by CMake as __cxx_compiler__),
    or None if it is unknown (old module, manual build) or absent from this machine.

    macOS : CMake often bakes the INTERNAL c++ of the Xcode / CommandLineTools toolchain
    (.../XcodeDefault.xctoolchain/usr/bin/c++), which invokes clang WITHOUT an SDK sysroot -> every DSL
    .so fails on \"'string' file not found\". The /usr/bin/c++ shim (xcrun) runs THE SAME
    clang while resolving the SDK : same __VERSION__, hence same ABI key -- so we prefer the shim
    (pitfall and remedy identical to compile_loader of the native C++ tests)."""
    import sys
    mod = _adc_module()
    cc = getattr(mod, "__cxx_compiler__", "") if mod is not None else ""
    if not (cc and os.path.isfile(cc) and os.access(cc, os.X_OK)):
        return None
    if sys.platform == "darwin" and (".xctoolchain/" in cc or "/CommandLineTools/" in cc) \
            and os.path.isfile("/usr/bin/c++"):
        return "/usr/bin/c++"
    return cc


def module_header_signature():
    """Header signature BAKED into the loaded _adc module (token headers= of abi_key()), or None
    if the module is not loadable / the key is absent ("unknown" from a manual build -> None too)."""
    mod = _adc_module()
    abi = getattr(mod, "abi_key", None) if mod is not None else None
    if not callable(abi):
        return None
    try:
        key = abi()
    except Exception:
        return None
    for tok in str(key).split(";"):
        if tok.startswith("headers="):
            sig = tok[len("headers="):]
            return sig if sig and sig != "unknown" else None
    return None


def resolve_auto_backend(include=None):
    """DEFAULT backend policy (backend='auto', decision recorded -- ADC-63).

    'production' (zero-copy native loader, strict add_block parity) AS SOON AS the
    toolchain parity with the _adc module is established : module loadable + known baked compiler +
    header signature of @p include == the one baked into the module. OTHERWISE 'aot' (historical
    default : host-marshaled, works without module or parity). Never silent : returns
    (backend, reason) and the facades set the reason on CompiledModel.backend_auto_reason.
    An EXPLICIT backend passed by the caller short-circuits this policy (unchanged)."""
    mod = _adc_module()
    if mod is None:
        return "aot", "_adc module not loadable (the production path requires the module)"
    if not loader_cxx_compiler():
        return "aot", "module compiler unknown (old module or manual build)"
    baked = module_header_signature()
    if not baked:
        return "aot", "header signature absent from the module (manual build)"
    try:
        inc = include if include is not None else adc_include()
        sig = adc_header_signature(inc)
    except Exception as e:  # headers not found / unreadable -> fall back on default
        return "aot", "adc headers not found for parity (%s)" % e
    if sig != baked:
        return "aot", ("headers != module (rebuild the module or point at the build headers ; "
                       "production would refuse, cf. _check_headers_match_module)")
    return "production", "toolchain parity established (module + baked compiler + matching headers)"


def _check_headers_match_module(include):
    """PRE-DLOPEN GUARD of the native path (real bug) : if the headers under @p include have changed since
    the build of _adc (recent pull, another clone...), the loader compiled against them references
    C++ signatures that the OLD module does not export -> the dlopen of add_native_block fails BEFORE the
    ABI guard, with a cryptic error ("symbol not found in flat namespace '__ZN3adc6System13
    install_block...'"). So we compare HERE, before any compilation, the header signature baked
    into the module with that of the @p include tree, and we fail with a clear remedy. No-op if the
    module is not loadable or has no signature (manual build : historical degradation)."""
    baked = module_header_signature()
    current = adc_header_signature(include)
    if baked is not None and current != baked:
        mod = _adc_module()
        so = getattr(mod, "__file__", "(unknown)")
        raise RuntimeError(
            "adc.dsl : the adc headers of %r DO NOT MATCH those with which the _adc module "
            "was built (%s).\n"
            "  current header signature : %s\n"
            "  signature baked in _adc  : %s\n"
            "Typical cause : `git pull` / headers edited AFTER the module build -> the DSL loader "
            "would reference C++ signatures absent from the module (dlopen : 'symbol not found').\n"
            "Remedy : REBUILD the module with these headers :\n"
            "  cmake --preset python && cmake --build --preset python   (or the usual build-py)\n"
            "or point ADC_INCLUDE at the headers of the build that produced this module."
            % (include, so, current[:16], baked[:16]))
    return current  # signature of the @p include tree, reusable (avoids a 2nd walk+sha256)


def check_compiled_matches_module(abi_key):
    """PRE-DLOPEN guard at the WIRING of a CompiledModel (add_equation -> add_native_block).

    INDISPENSABLE COMPLEMENT of _check_headers_match_module : on a cache HIT, compile_native does not
    run (the .so comes out of the cache) and the compilation guard therefore does not protect this path --
    a stale _adc module would dlopen the .so and fail on the cryptic 'symbol not found'.
    Here we compare the header signature EMBEDDED in the Python key of the .so ('<sig>|<cxx>|<std>',
    cf. _abi_key_python, recomputed on the CURRENT include tree at each compile()) with the one baked
    into the module. Pure STRING comparison (no re-hash). No-op if either of the two is missing
    (old module, CompiledModel built by hand)."""
    baked = module_header_signature()
    if baked is None or not abi_key:
        return
    so_sig = str(abi_key).split("|", 1)[0]
    if so_sig and so_sig != baked:
        raise RuntimeError(
            "adc : the compiled model (.so) was produced against adc headers DIFFERENT from those "
            "of the loaded _adc module (signature %s... vs %s... baked in the module).\n"
            "Typical cause : stale _adc module (built before a `git pull`) while the .so has just "
            "been (re)compiled on the up-to-date headers -- the dlopen would fail with a cryptic "
            "'symbol not found'.\n"
            "Remedy : REBUILD the module then re-run :\n"
            "  cmake --preset python && cmake --build --preset python\n"
            "Full diagnostic : python -c \"import adc; adc.doctor()\"."
            % (so_sig[:12], baked[:12]))


def _default_cxx(cxx=None):
    """CENTRALIZED resolution of the DSL .so compiler (all backends). Priority :
      1. explicit cxx (caller argument) ;
      2. $ADC_CXX (conscious environment override) ;
      3. the compiler that built _adc (the only one guaranteed ABI-compatible, cf. above) ;
      4. c++ / g++ / clang++ from the PATH (historical behavior, last resort)."""
    return (cxx or os.environ.get("ADC_CXX") or loader_cxx_compiler()
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
            "adc.dsl: compiling the .so (%s) failed (exit %d).\n"
            "Command: %s\n"
            "Compiler output:\n%s\n"
            "Hints: `python -c \"import adc; adc.doctor()\"` diagnoses the environment "
            "(compiler/standard/headers); ADC_CXX forces a specific compiler."
            % (what, r.returncode, " ".join(cmd), err[:4000] or "(empty)"))


_probe_cache = {}  # (cc, std) -> effective std: avoids re-probing repeatedly (N compiled models)


def _probe_cxx_std(cc, std):
    """Checks BEFORE compilation that @p cc accepts -std=@p std (probe -fsyntax-only on empty source).

    Returns the EFFECTIVE std: @p std if it passes, otherwise its historical alias (c++23 -> c++2b) if
    it passes, otherwise raises an ACTIONABLE RuntimeError (compiler used, build compiler,
    solutions) instead of the raw compiler error. Skipped for nvcc_wrapper (different -x
    semantics; explicit GPU path, already gated by ADC_KOKKOS_CXX/ADC_KOKKOS_USE_NVCC_WRAPPER).
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
        "adc.dsl: the compiler %r does not support -std=%s (standard required to share the ABI of "
        "the _adc module).\nCompiler output:\n%s\n"
        "Compiler of the _adc build: %s\n"
        "Solutions:\n"
        "  - use the build compiler: export ADC_CXX=%r (or cxx=... in m.compile);\n"
        "  - macOS: update Xcode / the Command Line Tools (recent AppleClang);\n"
        "  - conda: `conda install -c conda-forge cxx-compiler` (gcc>=13 / clang>=17) then "
        "export ADC_CXX=$CONDA_PREFIX/bin/clang++ (macOS) or $CONDA_PREFIX/bin/g++ (Linux).\n"
        "NB: a compiler DIFFERENT from the build one may compile but then be rejected "
        "('incompatible ABI': the ABI key encodes the compiler version); prefer the build one."
        % (cc, std, (err or "").strip()[:800],
           baked or "(unknown: module without __cxx_compiler__, rebuild _adc to bake it)",
           baked or "<path/to/build/compiler>"))


# --- Out-of-source build cache -----------------------------------------------
# When the caller does not provide so_path, m.compile(...) writes the .so into a SHARED out-of-source
# cache (never next to the temporary .cpp), indexed by a stable model key: model_hash (formulas
# + roles + n_aux + params) AND abi_key (header signature + compiler + std). Two compilations
# of the SAME model (same key) reuse the cached .so (cache HIT, no recompilation); changing the
# model OR a parameter OR the toolchain changes the key -> new .so (cache MISS, recompilation). The
# file name carries the key, so several variants coexist without collision. cf. the same idea
# in adc_cases/common/native.py (ABI key = compiler + flags + header signature).
def adc_cache_dir():
    """Cache directory for the .so files generated by m.compile() without an explicit so_path.

    $ADC_CACHE_DIR (override), else $XDG_CACHE_HOME/adc/dsl, else ~/.cache/adc/dsl. Created as needed.
    Out-of-source by construction (never inside the repo tree), so nothing to ignore on the git side."""
    import os
    base = os.environ.get("ADC_CACHE_DIR")
    if not base:
        xdg = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        base = os.path.join(xdg, "adc", "dsl")
    os.makedirs(base, exist_ok=True)
    return base


def _cache_so_path(model_hash, abi_key, backend, target, name):
    """Cached .so path for this (model_hash, abi_key, backend, target, name).

    The cache key combines model_hash (the WHAT: formulas/roles/params) and abi_key (the HOW:
    headers + compiler + std), plus backend/target/name which change the emitted code (native loader
    vs AOT vs JIT, System vs AmrSystem). The file name is <model_hash[:16]>-<sha(rest)[:16]>.so:
    readable (prefix = model identity) and collision-free (suffix = rest of the key)."""
    import hashlib
    import os
    # _platform_cache_key: the CPU arch + the optflags enter the key (a .so x86_64 or
    # -march=native reused on another machine via a shared cache = silent SIGILL).
    parts = [abi_key or "", backend or "", target or "", name or "", _platform_cache_key()]
    # AOT schema marker: the aot .so built before aligning the flags on native were compiled at
    # a hardcoded -O2 while the key already advertised the native optflags -> set them apart so a
    # shared cache does not serve a stale -O2 binary. Native/jit, whose key already reflects their
    # binary, keep an UNCHANGED file name (marker added for aot only).
    if (backend or "").split(";", 1)[0] in ("aot", "hybrid-aot"):
        parts.append("aot-optflags")
    rest = "|".join(parts).encode()
    tag = hashlib.sha256(rest).hexdigest()[:16]
    fname = "%s-%s.so" % ((model_hash or "nohash")[:16], tag)
    return os.path.join(adc_cache_dir(), fname)


# In-process registry of the backend already written to each resolved .so path (key = absolute path,
# value = backend). It models a cache that neither the explicit so_path nor the out-of-source cache key
# represented: the HANDLE cache of the dynamic loader (dlopen / dyld), keyed BY PATH and BLIND to the
# file content. On macOS notably, dlopen('/x/m.so') already loaded returns the SAME handle even if the
# file was recompiled in between: recompiling a 'production' .so ON a path where an 'aot' .so was already
# loaded re-serves the stale aot handle (add_native_block -> 'adc_native_abi_key missing'). The
# out-of-source cache key already includes the backend (distinct path per backend, so no collision); the
# EXPLICIT so_path, however, is pinned by the caller -> two backends overwrite each other there. We avoid
# it by redirecting to a per-backend DISTINCT sibling as soon as another backend already occupies that
# path in the process. Reset every process (the dlopen state is too); a new process re-reads the current
# file, so recompiling at the same path across two processes stays safe.
_process_so_backend = {}


def _backend_distinct_so_path(so_path, backend):
    """Return a .so path safe for @p backend: so_path unchanged when no OTHER backend already occupies it
    in this process, otherwise a distinct sibling (inserts '.<backend>' before the extension) so dlopen
    reloads a fresh handle instead of re-serving the stale one (cf. _process_so_backend). Touches neither
    the disk nor the registry; the caller records the backend of the RETAINED path after compilation."""
    import os
    prev = _process_so_backend.get(os.path.abspath(so_path))
    if prev is not None and prev != backend:
        root, ext = os.path.splitext(so_path)
        so_path = "%s.%s%s" % (root, backend, ext or ".so")
    return so_path


def _record_so_backend(so_path, backend):
    """Record (in process) the backend written to @p so_path: lets the next path resolution detect a
    cross-backend reuse of the SAME path (cf. _backend_distinct_so_path)."""
    import os
    _process_so_backend[os.path.abspath(so_path)] = backend


def _native_kokkos_root():
    """Kokkos root to compile the DSL loaders with the SAME backend as the _adc module.

    adc_cpp is KOKKOS-ONLY: every DSL .so that includes the adc headers (aot, native) MUST be compiled
    with Kokkos (for_each.hpp #error otherwise). The root is read from ADC_KOKKOS_ROOT / Kokkos_ROOT /
    KOKKOS_ROOT; None if not found (the caller then raises an explicit error)."""
    for key in ("ADC_KOKKOS_ROOT", "Kokkos_ROOT", "KOKKOS_ROOT"):
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
    # ADC-319: the MPI seam of the loader (ADC_HAS_MPI on/off, cf. _native_mpi_flags) changes the
    # compiled code (real comm vs serial stubs n_ranks()=1/my_rank()=0) -> it MUST enter the cache,
    # else a SERIAL-stub .so would be reused on an MPI module and any distributed layout built inside
    # the loader (e.g. AmrSystem(distribute_coarse=True)) would replicate on every rank (no scaling).
    mod = _adc_module()
    mpi = "mpi=on" if (mod is not None and getattr(mod, "__has_mpi__", False)) else "mpi=off"
    return "%s;%s" % (kk, mpi)


# Optimization flags shared by the DSL .so that run the PRODUCTION path (aot and native).
# Default -O3 -DNDEBUG: hot-loop asserts disarmed + full vectorization -> parity with a native block (at
# -O2 without -DNDEBUG the generated kernel is ~1.48x). $ADC_DSL_OPTFLAGS overrides; affects NEITHER the
# ABI NOR portability. The only .so that stays at -O2 is the JIT/prototype (Rusanov host residue, perf
# out of scope).
_DSL_OPTFLAGS_DEFAULT = "-O3 -DNDEBUG"


def _dsl_optflags():
    """Optimization flags list for the production DSL .so (cf. _DSL_OPTFLAGS_DEFAULT)."""
    return os.environ.get("ADC_DSL_OPTFLAGS", _DSL_OPTFLAGS_DEFAULT).split()


def _platform_cache_key():
    """MACHINE traits that change the binary code of the .so without changing the C++ ABI key (__VERSION__
    is identical cross-arch): CPU architecture + optimization flags. Without them in the cache
    key, a .so x86_64 (Rosetta) or -march=native would be reused on another machine/arch via
    a shared cache (NFS, synchronized home) -> SIGILL (illegal instruction) or cryptic dlopen."""
    import platform
    return "arch=%s;optflags=%s" % (platform.machine(),
                                    os.environ.get("ADC_DSL_OPTFLAGS", _DSL_OPTFLAGS_DEFAULT))


def _warn_kokkos_parity():
    """Warns (without blocking) when the BACKEND of the native loader would diverge from that of the _adc
    module:
    - Kokkos module + serial loader (ADC_KOKKOS_ROOT missing) -> the DSL block falls back to the
      SERIAL path silently: zero-copy but DOES NOT SCALE with threads/GPU (ROMEO measurement: DSL warm
      invariant threads=1/4/8). This is a silent PERF degradation, not a crash -> explicit warning.
    - serial module + ADC_KOKKOS_ROOT defined -> the loader would instantiate -DADC_HAS_KOKKOS against a
      module that is not (divergent allocator/type layouts, not covered by the ABI key).
    Source of truth: _adc.__has_kokkos__ (baked by the build) vs _native_kokkos_root() (env)."""
    import warnings
    mod = _adc_module()
    has = getattr(mod, "__has_kokkos__", None) if mod is not None else None
    root = _native_kokkos_root()
    if has is True and root is None:
        warnings.warn(
            "adc.dsl: the _adc module is compiled WITH Kokkos but ADC_KOKKOS_ROOT is not defined "
            "-> the 'production' DSL block will be compiled SERIAL (it will run, but will not scale "
            "with threads/GPU). Set ADC_KOKKOS_ROOT=<build Kokkos install> for parity.",
            RuntimeWarning, stacklevel=3)
    elif has is False and root is not None:
        warnings.warn(
            "adc.dsl: ADC_KOKKOS_ROOT is defined but the _adc module is SERIAL (compiled without "
            "-DADC_USE_KOKKOS=ON) -> the loader would be compiled with Kokkos against a module that is "
            "not (divergent memory layouts, not covered by the ABI key). Remove "
            "ADC_KOKKOS_ROOT or rebuild _adc with Kokkos (preset python-parallel).",
            RuntimeWarning, stacklevel=3)


def _env_truthy(value):
    return str(value or "").strip().lower() in ("1", "on", "true", "yes", "y")


def _native_kokkos_compiler(cxx):
    """Effective compiler of the native loader. CUDA must be EXPLICIT (ADC_KOKKOS_CXX=<nvcc_wrapper>
    or ADC_KOKKOS_USE_NVCC_WRAPPER=1): many Kokkos OpenMP installs also provide a
    nvcc_wrapper, choosing it by default would break CPU jobs without nvcc. For Kokkos OpenMP, the host suffices."""
    if cxx:
        return cxx
    env = os.environ.get("ADC_KOKKOS_CXX")
    if env:
        return env
    root = _native_kokkos_root()
    wrapper = os.path.join(root, "bin", "nvcc_wrapper") if root else ""
    if wrapper and os.path.exists(wrapper) and _env_truthy(os.environ.get("ADC_KOKKOS_USE_NVCC_WRAPPER")):
        return wrapper
    # Centralized fallback: $ADC_CXX, then the _adc build compiler (the only ABI-compatible one
    # guaranteed for a native loader), then the PATH (historical). cf. _default_cxx.
    return _default_cxx(None)


def _adc_import_lib():
    """(Windows, ADC-100) Path of the import library _adc.lib (System ADC_EXPORT symbols) against
    which to link the DSL .dll. Searched next to the _adc module. None if absent."""
    mod = _adc_module()
    if mod is None:
        return None
    d = os.path.dirname(getattr(mod, "__file__", "") or "")
    cand = os.path.join(d, "_adc.lib")
    return cand if os.path.exists(cand) else None


def _native_kokkos_flags():
    """Compile/link flags so the production DSL loader instantiates add_compiled_model WITH Kokkos.

    The loader contains the header-only templates (make_block / assemble_rhs / for_each_cell). Compiled
    without ADC_HAS_KOKKOS while _adc is built WITH Kokkos, the DSL block stays zero-copy but its kernels
    are instantiated on the SERIAL fallback (does not scale threads/GPU). ADC_KOKKOS_ROOT forces parity."""
    root = _native_kokkos_root()
    if not root:
        return [], []
    inc = os.path.join(root, "include")
    if sys.platform == "win32":
        # MSVC/clang-cl: Kokkos as a SHARED DLL -> link the import lib kokkoscore.lib (ONE single runtime;
        # _adc loads the same kokkoscore.dll). cl accepts -D/-I. No -fopenmp/-ldl/-pthread (POSIX).
        return (["-DADC_HAS_KOKKOS", "-DKOKKOS_DEPENDENCE", "-I", inc],
                [os.path.join(root, "lib", "kokkoscore.lib")])
    compile_flags = ["-DADC_HAS_KOKKOS", "-DKOKKOS_DEPENDENCE", "-I", inc]
    # Do NOT link libkokkos* INTO the .so: the _adc module has already loaded the Kokkos runtime, a
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
            # dynamic_lookup set by compile_aot/compile_native) against the libomp already loaded by _adc.
            libomp = _libomp_prefix()
            compile_flags += ["-Xpreprocessor", "-fopenmp"]
            if libomp is not None:
                compile_flags += ["-I", os.path.join(libomp, "include")]
        else:
            # ELF/Linux: libgomp is shared by soname (no double-runtime), -fopenmp on both sides.
            compile_flags.append("-fopenmp")
            link_flags.append("-fopenmp")
    return compile_flags, link_flags


def _native_mpi_flags():
    """Compile flags so the production/AOT DSL loader uses comm.hpp's REAL MPI seam (ADC-319).

    The loader inlines the runtime templates (System / AmrSystem coupler), which call adc::n_ranks() /
    my_rank() from comm.hpp to lay out the distributed grid (DistributionMapping over n_ranks()).
    Compiled WITHOUT ADC_HAS_MPI while _adc is built WITH MPI, comm.hpp falls back to its SERIAL stubs
    (n_ranks()=1, my_rank()=0): any distributed layout built INSIDE the loader then collapses to a
    single owner on EVERY rank -- e.g. AmrSystem(distribute_coarse=True) replicates the coarse
    transport on all ranks (no MPI strong-scaling, ADC-319). We compile WITH -DADC_HAS_MPI + the SAME
    MPI include dir as _adc (baked as __mpi_include__) and leave the MPI symbols UNDEFINED, resolved at
    load time against the libmpi already loaded by _adc / mpi4py (RTLD_GLOBAL) -- no 2nd libmpi linked,
    exactly like the Kokkos runtime. Empty (no flag) when _adc is a serial build (__has_mpi__ False),
    so the serial loader path stays bit-identical."""
    mod = _adc_module()
    if mod is None or not getattr(mod, "__has_mpi__", False):
        return []
    flags = ["-DADC_HAS_MPI"]
    inc = getattr(mod, "__mpi_include__", "") or ""
    for d in inc.split("|"):  # CMake bakes the include dirs joined by '|' (paths may contain ';')
        if d:
            flags += ["-I", d]
    return flags


def adc_loader_build_flags(cxx=None):
    """Flags to compile OUTSIDE CMake a .so that INCLUDES the adc headers and will be loaded into the
    _adc module (DSL loaders, ABI tests). adc_cpp being Kokkos-only, the .so MUST be compiled with
    Kokkos (for_each.hpp #error otherwise). Returns (compiler, compile_flags, link_flags): Kokkos +
    (macOS) -undefined dynamic_lookup. The Kokkos symbols stay UNDEFINED, resolved at load time
    against the Kokkos runtime already loaded by _adc (no 2nd copy). Raises if no installed Kokkos is
    visible via ADC_KOKKOS_ROOT / Kokkos_ROOT (Serial suffices on CPU)."""
    if _native_kokkos_root() is None:
        raise RuntimeError(
            "adc_loader_build_flags: adc_cpp is Kokkos-only -- point to an installed Kokkos via "
            "ADC_KOKKOS_ROOT (or Kokkos_ROOT), e.g. `export ADC_KOKKOS_ROOT=/path/to/kokkos`.")
    cc = _native_kokkos_compiler(cxx)
    cflags, lflags = _native_kokkos_flags()
    if sys.platform == "darwin":
        cflags = list(cflags) + ["-undefined", "dynamic_lookup"]
    return cc, cflags, lflags


# --- Aux channel: canonical layout ------------------------------------------
# The named auxiliary fields (aux('...')) are FIXED-index COMPONENTS of the aux channel
# (cf. adc::Aux / kAuxBaseComps on the C++ side). phi/grad_x/grad_y = BASE contract (3 components);
# the following ones (B_z, ...) WIDEN the channel -> the generated brick then declares n_aux so that
# the system sizes and populates the shared channel (cf. CompositeModel::n_aux, ensure_aux_width).
#
# INHERENT C++ <-> Python DUPLICATION: the table below MUST stay the MIRROR of the single C++
# source ADC_AUX_FIELDS (include/adc/core/state.hpp), from which load_aux (device read)
# and the host marshaling (python/system.cpp) are generated. Python does not read the C++ headers, so
# we cannot generate it: adding an extra aux field = 1 line here AND 1 line in ADC_AUX_FIELDS,
# with the SAME {name, index}. This is the only remaining duplication; the 3 C++ sites are now unified.
AUX_CANONICAL = {"phi": 0, "grad_x": 1, "grad_y": 2, "B_z": 3, "T_e": 4}
AUX_BASE_COMPS = 3

# Aux fields NAMED by the model (ADC-70 phase 1): m.aux_field("name"). Components starting from
# AUX_NAMED_BASE (= 5, just after T_e=4) -- the k-th declared name is component AUX_NAMED_BASE + k,
# read in C++ via aux.extra_field(k). MIRRORS of kAuxNamedBase / kAuxMaxExtra (include/adc/core/state.hpp,
# single C++ source). Decouples the user names from the canonical channel: B_z / T_e keep their indices
# 3 / 4 and their dedicated paths (set_magnetic_field / set_electron_temperature_from).
AUX_NAMED_BASE = 5
AUX_NAMED_MAX = 4  # maximum number of named aux fields per model (= kAuxMaxExtra on the C++ side)

# Bound on the number of RUNTIME parameters per block (P7-b). MIRROR of kMaxRuntimeParams
# (include/adc/runtime/runtime_params.hpp): the C++ carrier RuntimeParams has an array of this FIXED
# size (device-copiable without allocation), so a model exceeding the bound is rejected at codegen.
_K_MAX_RUNTIME_PARAMS = 32
