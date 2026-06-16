"""adc.dsl : SYMBOLIC mini-DSL for physical models.

Python WRITES the formulas (named variables, expressions), not a function called per cell.
The operations (+, -, *, /, **, adc.dsl.sqrt) build an expression TREE. A
HyperbolicModel declares its conservative variables, its primitives (defined by formulas),
its flux, its eigenvalues, its source and its elliptic contribution.

    e = adc.dsl.HyperbolicModel("euler")
    rho, rhou, rhov, E = e.conservative_vars("rho", "rho_u", "rho_v", "E")
    u = e.primitive("u", rhou / rho)
    p = e.primitive("p", (gamma - 1.0) * (E - 0.5 * rho * (u*u + v*v)))
    e.set_flux(x=[rhou, rhou*u + p, ...], y=[...])
    e.set_eigenvalues(x=[u - c, u, u + c], y=[...])

Backends available via m.compile(backend=..., target=...) :
  - "prototype" : NumPy/host JIT evaluator, first-order Rusanov, TEST only (host alone) ;
  - "aot"       : generates a flat-ABI .so (debug ABI), ahead-of-time compilation ;
  - "production": zero-copy native loader (add_native_block), RECOMMENDED by default ; device-clean
                  path (named functors, validated GH200, #97/#93) and MPI/AMR-ready.

Target (target=) :
  - "System"    : single-level system ;
  - "AmrSystem" : AMR-wired system (#92/#105), only with backend "production".

Ergonomics :
  - m.compile() auto-detects the adc includes and caches the .so by (model_hash, abi_key) (#103).
  - The physical roles (gamma, n_aux, B_z, T_e) are preserved and passed through to the C++.

Inter-species coupling (#131, #167) :
  - CoupledSource describes an exchange between blocks via DSL formulas (arbitrary cross source).
  - CoupledSource.add_pair(block_a, block_b, role, expr) guarantees conservation by
    construction (+expr on A, -expr on B, SAME subtree) ; DSL equivalent of
    add_collision / add_thermal_exchange on the C++ side. compile(verify_conservation=True) checks
    the property symbolically over all the terms (add and add_pair).
"""
import os
import shutil
import sys

import numpy as np


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
    Requires that adc/mesh/multifab.hpp exists. Raises RuntimeError if not found (diagnostic listing the
    candidates), so as to NEVER compile against a silently wrong include."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))           # .../python/adc
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
    # from this file (python/adc/dsl.py) : python/adc -> python -> repo root -> include
    candidates.append(os.path.normpath(os.path.join(here, "..", "..", "include")))
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "adc", "mesh", "multifab.hpp")):
            return c
    raise RuntimeError(
        "adc headers not found (looking for adc/mesh/multifab.hpp). "
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
            from . import _adc  # adc package : the extension is a submodule
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
            from . import _adc
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
    rest = "|".join((abi_key or "", backend or "", target or "", name or "",
                     _platform_cache_key())).encode()
    tag = hashlib.sha256(rest).hexdigest()[:16]
    fname = "%s-%s.so" % ((model_hash or "nohash")[:16], tag)
    return os.path.join(adc_cache_dir(), fname)


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
        return "kokkos=off"
    import hashlib
    cfg = os.path.join(root, "include", "KokkosCore_config.h")
    try:
        with open(cfg, "rb") as f:
            tag = hashlib.sha256(f.read()).hexdigest()[:12]
    except OSError:
        tag = "unknown"
    return "kokkos=on;kcfg=%s" % tag


def _platform_cache_key():
    """MACHINE traits that change the binary code of the .so without changing the C++ ABI key (__VERSION__
    is identical cross-arch): CPU architecture + optimization flags. Without them in the cache
    key, a .so x86_64 (Rosetta) or -march=native would be reused on another machine/arch via
    a shared cache (NFS, synchronized home) -> SIGILL (illegal instruction) or cryptic dlopen."""
    import platform
    return "arch=%s;optflags=%s" % (platform.machine(),
                                    os.environ.get("ADC_DSL_OPTFLAGS", "-O3 -DNDEBUG"))


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


def aux_n_aux(aux_names):
    """Aux channel width required by these CANONICAL fields: max(3, largest index + 1).
    Raises ValueError on an unknown name (a canonical aux field MUST be a component of adc::Aux)."""
    w = AUX_BASE_COMPS
    for nm in aux_names:
        if nm not in AUX_CANONICAL:
            raise ValueError("unknown aux field '%s': expected %s (components of adc::Aux)"
                             % (nm, sorted(AUX_CANONICAL)))
        w = max(w, AUX_CANONICAL[nm] + 1)
    return w


def aux_total_n_aux(aux_names, aux_extra_names):
    """TOTAL width of the aux channel: max of the canonical width (aux_n_aux) and, if NAMED fields
    (aux_field) are declared, AUX_NAMED_BASE + number of names (the last name = component
    AUX_NAMED_BASE + len-1). Without a named field -> aux_n_aux (historical path, bit-identical)."""
    w = aux_n_aux(aux_names)
    if aux_extra_names:
        w = max(w, AUX_NAMED_BASE + len(aux_extra_names))
    return w


# --- Physical roles: variable name -> VariableRole -------------------------
# CANONICAL mapping name -> physical role (cf. adc::VariableRole / role_name on the C++ side). Lets a
# generated brick DECLARE the MEANING of its components (density, momentum, energy...) instead of
# empty roles, so that inter-species couplings (System::add_collision / add_thermal_exchange)
# resolve via index_of(role) rather than via a literal index. The usual names of fluid models
# (rho, rho_u, u, p, E, n...) are recognized; an unknown name stays 'Custom'. A model can impose
# its roles explicitly (conservative_vars(..., roles=[...]) / set_primitive_state(..., roles=[...]))
# for a non-standard layout. Key = EXACT variable name, value = member of adc::VariableRole.
CANONICAL_ROLES = {
    "rho": "Density", "n": "Density", "density": "Density",
    "rho_u": "MomentumX", "rhou": "MomentumX", "mom_x": "MomentumX", "mx": "MomentumX",
    "rho_v": "MomentumY", "rhov": "MomentumY", "mom_y": "MomentumY", "my": "MomentumY",
    "rho_w": "MomentumZ", "rhow": "MomentumZ", "mom_z": "MomentumZ", "mz": "MomentumZ",
    "E": "Energy", "rho_E": "Energy", "ener": "Energy", "energy": "Energy",
    "u": "VelocityX", "v": "VelocityY", "w": "VelocityZ",
    "vx": "VelocityX", "vy": "VelocityY", "vz": "VelocityZ",
    "p": "Pressure", "pressure": "Pressure",
    "T": "Temperature", "temperature": "Temperature",
}


def role_of(name):
    """CANONICAL physical role of name @p name (member of adc::VariableRole), 'Custom' if unknown."""
    return CANONICAL_ROLES.get(name, "Custom")


def roles_for(names, override=None):
    """List of roles (adc::VariableRole members) parallel to @p names. @p override (optional):
    list of the same length explicitly fixing the roles (string 'Density'... or None to fall back
    on the canonical mapping of the name). Used for non-standard layouts where names are not enough."""
    if override is None:
        return [role_of(nm) for nm in names]
    if len(override) != len(names):
        raise ValueError("roles: %d roles for %d variables" % (len(override), len(names)))
    return [(r if r is not None else role_of(nm)) for nm, r in zip(names, override)]


# --- Expression tree -------------------------------------------------------
class Expr:
    """Symbolic expression node. Operators build the tree; eval(env) applies it to
    numpy arrays (env: name -> array or scalar)."""

    def __add__(self, o): return Add(self, _wrap(o))
    def __radd__(self, o): return Add(_wrap(o), self)
    def __sub__(self, o): return Sub(self, _wrap(o))
    def __rsub__(self, o): return Sub(_wrap(o), self)
    def __mul__(self, o): return Mul(self, _wrap(o))
    def __rmul__(self, o): return Mul(_wrap(o), self)
    def __truediv__(self, o): return Div(self, _wrap(o))
    def __rtruediv__(self, o): return Div(_wrap(o), self)
    def __neg__(self): return Neg(self)
    def __pos__(self): return self  # +expr = identity (the CoupledSource API writes +k*ne*ng)
    def __abs__(self): return Abs(self)  # abs(expr) -> |expr| (absolute value, e.g. |lambda| of Roe)
    def __pow__(self, o): return Pow(self, _wrap(o))

    def eval(self, env): raise NotImplementedError
    def deps(self): return set()
    def __repr__(self): return self._str()
    def _str(self): return "?"


def _wrap(o):
    if isinstance(o, Expr):
        return o
    # A Param exposes its internal tree NODE (_node: Const for 'const', RuntimeParamRef for
    # 'runtime'). We promote it via that node, NOT via float(o): otherwise sqrt(param_runtime) /
    # dsl.sqrt(param) would inline the declaration value (Const) instead of emitting params.get(...).
    node = getattr(o, "_node", None)
    if isinstance(node, Expr):
        return node
    return Const(float(o))


class Const(Expr):
    def __init__(self, value): self.value = float(value)
    def eval(self, env): return self.value
    def to_cpp(self): return repr(self.value)
    def _str(self): return repr(self.value)


class Var(Expr):
    """Named variable: conservative, primitive, auxiliary (field) or constant."""

    def __init__(self, name, kind): self.name = name; self.kind = kind
    def eval(self, env):
        if self.name not in env:
            raise KeyError("variable '%s' (%s) missing from the environment" % (self.name, self.kind))
        return env[self.name]
    def deps(self): return {self.name}
    def to_cpp(self): return self.name
    def _str(self): return self.name


class _Bin(Expr):
    op = "?"
    def __init__(self, a, b): self.a = a; self.b = b
    def deps(self): return self.a.deps() | self.b.deps()
    def to_cpp(self): return "(%s %s %s)" % (self.a.to_cpp(), self.op, self.b.to_cpp())
    def _str(self): return "(%s %s %s)" % (self.a, self.op, self.b)


class Add(_Bin):
    op = "+"
    def eval(self, env): return self.a.eval(env) + self.b.eval(env)


class Sub(_Bin):
    op = "-"
    def eval(self, env): return self.a.eval(env) - self.b.eval(env)


class Mul(_Bin):
    op = "*"
    def eval(self, env): return self.a.eval(env) * self.b.eval(env)


class Div(_Bin):
    op = "/"
    def eval(self, env): return self.a.eval(env) / self.b.eval(env)


class Pow(_Bin):
    op = "**"
    def eval(self, env): return self.a.eval(env) ** self.b.eval(env)
    def to_cpp(self): return "std::pow(%s, %s)" % (self.a.to_cpp(), self.b.to_cpp())


class Neg(Expr):
    def __init__(self, a): self.a = a
    def eval(self, env): return -self.a.eval(env)
    def deps(self): return self.a.deps()
    def to_cpp(self): return "(-%s)" % self.a.to_cpp()
    def _str(self): return "(-%s)" % self.a


class Sqrt(Expr):
    def __init__(self, a): self.a = a
    def eval(self, env): return np.sqrt(self.a.eval(env))
    def deps(self): return self.a.deps()
    def to_cpp(self): return "std::sqrt(%s)" % self.a.to_cpp()
    def _str(self): return "sqrt(%s)" % self.a


def sqrt(x):
    """Symbolic square root."""
    return Sqrt(_wrap(x))


class Abs(Expr):
    """Absolute value ``|a|`` (e.g. ``|lambda_k|`` of a Roe dissipation). Emitted as std::fabs at codegen
    (equal to the ternary a<0?-a:a outside -0.0). Not differentiable by dsl.diff (no sign node)."""
    def __init__(self, a): self.a = a
    def eval(self, env): return np.abs(self.a.eval(env))
    def deps(self): return self.a.deps()
    def to_cpp(self): return "std::fabs(%s)" % self.a.to_cpp()
    def _str(self): return "abs(%s)" % self.a


def abs_(x):
    """Symbolic absolute value (equivalent of abs(expr); suffixed name so as not to shadow abs)."""
    return Abs(_wrap(x))


class Sign(Expr):
    """Signe de a : -1, 0 ou 1 (np.sign cote interprete). Emis au codegen comme le ternaire SANS
    branche (a > 0) - (a < 0) (exact en adc::Real). Sert aux selections par masques des branches par
    cellule (ADC-177 : clamps de projection en max/min via abs/sign, sans if). Derivee nulle presque
    partout (saut en 0, mesure nulle), cf. dsl.diff."""
    def __init__(self, a): self.a = a
    def eval(self, env): return np.sign(self.a.eval(env))
    def deps(self): return self.a.deps()
    def to_cpp(self):
        s = self.a.to_cpp()
        return "(adc::Real(%s > 0) - adc::Real(%s < 0))" % (s, s)
    def _str(self): return "sign(%s)" % self.a


def sign(x):
    """Signe symbolique (-1 / 0 / 1) : selections par masques sans branche par cellule."""
    return Sign(_wrap(x))


# Champs scalaires d'adc::EigBounds exposables comme une valeur DSL (cf. dense_eig.hpp).
_EIG_FIELDS = {
    "max_im": "max_im",  # plus grande |Im(lambda)| -> temoin de VP complexes (0 = spectre reel)
    "lmin": "lmin",      # plus petite partie reelle du spectre
    "lmax": "lmax",      # plus grande partie reelle du spectre
}


class EigWitness(Expr):
    """Valeur scalaire issue du spectre d'une PETITE matrice dense construite a partir d'expressions
    (ADC-289). La matrice @c rows (liste de @c k lignes de @c k Expr, ordre ligne-major) est diagonalisee
    par ``adc::real_eig_minmax`` (dense_eig.hpp, ADC_HD, repli Gershgorin sur non-convergence, cap QR
    releve par ADC-195) ; @c field choisit le champ rendu de @c adc::EigBounds : ``max_im`` (temoin de
    VP complexes : 0 = spectre reel donc hyperbolique), ``lmin`` / ``lmax`` (extremes des parties
    reelles). Sert la logique branchless de m.projection : ``si max_im > tol alors corriger`` s'ecrit
    en masque max/min/sign sur cette valeur, sans branche dynamique.

    Codegen device-clean : l'emission est un FONCTEUR NOMME (methode statique ADC_HD de la brique
    generee) qui remplit un ``adc::Real M[k][k]`` puis appelle ``real_eig_minmax`` -- jamais de lambda
    etendue cross-TU (casse nvcc). to_cpp() rend l'APPEL de ce foncteur en passant les entrees comme
    arguments scalaires (chacune evaluee une seule fois cote appelant : compatible CSE). La brique
    declare le foncteur une fois par couple (field, k) rencontre (cf. _eig_witness_helpers).

    eval(env) : miroir hote via numpy.linalg.eigvals (reference de test) ; field 'max_im' = max des
    |Im| (0 si toutes reelles), 'lmin'/'lmax' = extremes des parties reelles. ATTENTION : le repli
    Gershgorin du chemin C++ (non-convergence d'un bloc >= 3 sous le cap QR) n'est PAS reproduit ici --
    sur des matrices saines (cas vise) les deux chemins coincident a la tolerance QR (cf. dense_eig)."""

    def __init__(self, rows, field):
        if field not in _EIG_FIELDS:
            raise ValueError("EigWitness : field '%s' inconnu (attendu : %s)"
                             % (field, ", ".join(sorted(_EIG_FIELDS))))
        rows = [list(r) for r in rows]
        k = len(rows)
        if k < 1:
            raise ValueError("EigWitness : matrice vide (au moins 1 ligne)")
        if k > 16:
            raise ValueError("EigWitness : matrice %dx%d > 16x16 (limite de real_eig_minmax, "
                             "tampon pile O(N^2) par thread device)" % (k, k))
        for r in rows:
            if len(r) != k:
                raise ValueError("EigWitness : matrice non carree (%d lignes, ligne de %d entrees)"
                                 % (k, len(r)))
        self.rows = [[_wrap(e) for e in r] for r in rows]
        self.k = k
        self.field = field

    def entries(self):
        """Entrees de la matrice a plat (ordre ligne-major), une par enfant Expr."""
        return [e for row in self.rows for e in row]

    def helper_name(self):
        """Nom du foncteur nomme emis dans la brique pour ce couple (field, taille)."""
        return "adc_eig_%s_%dx%d" % (self.field, self.k, self.k)

    def eval(self, env):
        # Miroir hote (reference de test / prototypage) : empile la matrice par cellule puis numpy.
        # Les entrees sont diffusees a une forme commune ; eigvals s'applique sur le dernier axe 2x2.
        vals = [e.eval(env) for e in self.entries()]
        bshape = np.broadcast(*[np.asarray(v) for v in vals]).shape if vals else ()
        k = self.k
        M = np.empty(bshape + (k, k), dtype=float)
        for idx, v in enumerate(vals):
            M[..., idx // k, idx % k] = np.broadcast_to(np.asarray(v, dtype=float), bshape)
        ev = np.linalg.eigvals(M)  # (..., k) complexe
        if self.field == "max_im":
            out = np.max(np.abs(ev.imag), axis=-1)
        elif self.field == "lmin":
            out = np.min(ev.real, axis=-1)
        else:  # lmax
            out = np.max(ev.real, axis=-1)
        return out if bshape else float(out)

    def deps(self):
        d = set()
        for e in self.entries():
            d |= e.deps()
        return d

    def to_cpp(self):
        return "%s(%s)" % (self.helper_name(), ", ".join(e.to_cpp() for e in self.entries()))

    def _str(self):
        return "eig_%s([%s])" % (self.field, ", ".join(str(e) for e in self.entries()))


def eig_max_im(rows):
    """Temoin de VALEURS PROPRES COMPLEXES d'une petite matrice dense @p rows (liste de @c k lignes de
    @c k Expr) : valeur scalaire = max des |Im(lambda)| (0 = spectre reel, donc hyperbolique), via
    ``adc::real_eig_minmax`` (ADC-289). Sert les selections branchless de m.projection (p.ex. ``si
    max_im > tol, corriger``, ecrit en max/min/sign). Le caller assemble la matrice (bloc 3x3 du
    jacobien, compagnon...) a partir d'expressions de moments ; aucune physique n'est dans le coeur."""
    return EigWitness(rows, "max_im")


def eig_lmin(rows):
    """Plus petite PARTIE REELLE du spectre de la matrice dense @p rows (cf. eig_max_im), via
    ``adc::real_eig_minmax`` -- valeur scalaire DSL (extreme de borne de vitesse / spectre reel)."""
    return EigWitness(rows, "lmin")


def eig_lmax(rows):
    """Plus grande PARTIE REELLE du spectre de la matrice dense @p rows (cf. eig_max_im), via
    ``adc::real_eig_minmax`` -- valeur scalaire DSL (extreme de borne de vitesse / spectre reel)."""
    return EigWitness(rows, "lmax")


class StateRef(Expr):
    """STATE marker for the TWO-state Roe dissipation (m.roe_dissipation): the
    enclosed sub-expression evaluates on the LEFT state UL (side='L', dsl.left) or RIGHT state UR
    (side='R', dsl.right). At codegen of the hook roe_dissipation(UL, AL, UR, AR, dir), left(e) emits e
    with the locals computed from UL, right(e) from UR. Has meaning ONLY in the lines
    given to m.roe_dissipation: the numpy interpreter does not handle it (the two-state dissipation is
    compiled into C++, not evaluated on the host). deps() = deps of the sub-expression (dependency
    checking)."""

    def __init__(self, side, expr):
        if side not in ("L", "R"):
            raise ValueError("StateRef: side must be 'L' (UL) or 'R' (UR), got %r" % (side,))
        self.side = side
        self.expr = _wrap(expr)

    def deps(self):
        return self.expr.deps()

    def eval(self, env):
        raise NotImplementedError(
            "StateRef (dsl.left / dsl.right) is not evaluated by the numpy interpreter: the two-state "
            "Roe dissipation is EMITTED in C++ (m.roe_dissipation), not interpreted on the host.")

    def _str(self):
        return "%s(%s)" % ("left" if self.side == "L" else "right", self.expr)


def left(expr):
    """Marks @p expr as evaluated on the LEFT state UL (Roe dissipation, m.roe_dissipation)."""
    return StateRef("L", expr)


def right(expr):
    """Marks @p expr as evaluated on the RIGHT state UR (Roe dissipation, m.roe_dissipation)."""
    return StateRef("R", expr)


class RuntimeParamRef(Expr):
    """Reference to a RUNTIME parameter (P7-b) in the expression tree. Unlike a Const
    (const param inlined HARD), this node emits `params.get(<index>)` at codegen: the generated brick
    READS the value from its adc::RuntimeParams member instead of having it baked in. The value can thus
    be CHANGED at runtime without recompiling the .so (cf. include/adc/runtime/runtime_params.hpp).

    @c index: STABLE index of the parameter in the RuntimeParams block (assigned by the model at
    compilation, sorted order of names); -1 as long as it is not assigned. @c value: DECLARATION
    value (used by the numpy eval interpreter and as the default of the generated member -> without a
    set call at runtime, the block behaves as with a const param of this value).

    Structural CSE key (cf. _key): the NAME (two refs to the same runtime param share the same
    CSE local); the declaration value does not enter the key (it is runtime, not structural)."""

    def __init__(self, name, value, index=-1):
        self.name = name
        self.value = float(value)
        self.index = index

    def eval(self, env):
        # Numpy interpreter (host proto / debug): the declaration value stands in for the current value
        # (the numpy path does not go through RuntimeParams; it serves prototyping, not production).
        return self.value

    def deps(self):
        # A runtime parameter is NOT an environment variable (cons/prim/aux): it comes from the
        # RuntimeParams channel, so nothing to check in check() (like a Const).
        return set()

    def to_cpp(self):
        if self.index < 0:
            raise RuntimeError(
                "RuntimeParamRef('%s'): index not assigned at codegen (call the compilation via "
                "dsl.Model which assigns the runtime indices)" % self.name)
        return "params.get(%d)" % self.index

    def _str(self):
        return "rparam(%s)" % self.name


# --- Common subexpression elimination (CSE) --------------------------------
# The codegen inlines each sub-expression at every occurrence (H, c... recomputed). CSE detects
# the COMPOUND (non-leaf) sub-expressions appearing multiple times and hoists them into local
# variables 'cseK_', in dependency order (the smallest first). Relies on a STRUCTURAL key
# per node: two identical subtrees have the same key, hence the same local.
def _children(e):
    if isinstance(e, _Bin):
        return (e.a, e.b)
    if isinstance(e, (Neg, Sqrt, Abs, Sign)):
        return (e.a,)
    if isinstance(e, EigWitness):
        return tuple(e.entries())  # entrees de la matrice : enfants pour CSE / decouverte deps
    if isinstance(e, StateRef):
        return (e.expr,)  # left/right marker: a single child (discovery of runtime params, etc.)
    return ()


def _key(e):
    if isinstance(e, Const):
        return ("const", e.value)
    if isinstance(e, RuntimeParamRef):
        return ("rparam", e.name)  # key = name: two refs to the same runtime param share the CSE local
    if isinstance(e, Var):
        return ("var", e.name)
    if isinstance(e, Neg):
        return ("neg", _key(e.a))
    if isinstance(e, Sqrt):
        return ("sqrt", _key(e.a))
    if isinstance(e, Abs):
        return ("abs", _key(e.a))
    if isinstance(e, Sign):
        return ("sign", _key(e.a))
    if isinstance(e, EigWitness):
        # cle = (field, taille, cles des entrees) : deux temoins de la MEME matrice partagent une locale
        return ("eig", e.field, e.k, tuple(_key(c) for c in e.entries()))
    if isinstance(e, StateRef):
        return ("state", e.side, _key(e.expr))  # defensive: the Roe lines do not go through CSE
    return (e.op, tuple(_key(c) for c in _children(e)))  # _Bin (Add/Sub/Mul/Div/Pow)


def _cpp_expand(e, cse_map):
    """C++ of node e expanding ITS level; the children go through _cpp_cse (-> CSE locals)."""
    if isinstance(e, Const):
        return repr(e.value)
    if isinstance(e, RuntimeParamRef):
        return e.to_cpp()  # params.get(<index>): reads the brick's RuntimeParams member
    if isinstance(e, Var):
        return e.name
    if isinstance(e, Neg):
        return "(-%s)" % _cpp_cse(e.a, cse_map)
    if isinstance(e, Sqrt):
        return "std::sqrt(%s)" % _cpp_cse(e.a, cse_map)
    if isinstance(e, Abs):
        return "std::fabs(%s)" % _cpp_cse(e.a, cse_map)
    if isinstance(e, Sign):
        s = _cpp_cse(e.a, cse_map)  # l'enfant peut etre une locale CSE : evalue UNE fois
        return "(adc::Real(%s > 0) - adc::Real(%s < 0))" % (s, s)
    if isinstance(e, EigWitness):
        # appel du foncteur nomme (declare dans la brique) : chaque entree passee en argument scalaire
        # (via _cpp_cse -> partage les locales CSE, evaluee une seule fois cote appelant).
        args = ", ".join(_cpp_cse(c, cse_map) for c in e.entries())
        return "%s(%s)" % (e.helper_name(), args)
    if isinstance(e, Pow):
        return "std::pow(%s, %s)" % (_cpp_cse(e.a, cse_map), _cpp_cse(e.b, cse_map))
    if isinstance(e, _Bin):
        return "(%s %s %s)" % (_cpp_cse(e.a, cse_map), e.op, _cpp_cse(e.b, cse_map))
    raise TypeError("expression not handled by the codegen: %r" % (e,))


def _cpp_cse(e, cse_map):
    """C++ of e; if e matches an already-defined CSE local, returns its name."""
    k = _key(e)
    if k in cse_map:
        return cse_map[k]
    return _cpp_expand(e, cse_map)


def _cse_emit(roots, real, indent):
    """Return (local_declaration_lines, [C++ per root]). Compound subexpressions seen >= 2 times
    become ``cseK_`` locals. roots: list of Expr.

    Memo by id(): a shared Expr OBJECT (DAG, e.g. intermediates reused from a large model)
    is traversed only once; its later occurrences re-credit the counter of its
    subtree without re-walking. Without the memo the traversal costs O(number of root-to-leaf PATHS),
    exponential on a deep DAG (polynomial flux of polynomials). Strictly equivalent to the
    historical re-walk: same counts, same sizes, same key INSERTION ORDER
    (post-order of the first visit) -> emitted C++ is bit-identical."""
    counts, rep, size = {}, {}, {}
    memo = {}  # id(e) -> (size, {key: occurrences} of the subtree, in post-order of insertion)

    def visit(e):
        if isinstance(e, (Const, Var)):
            return 1, None
        sub = memo.get(id(e))
        if sub is None:
            k = _key(e)
            s, cnt = 1, {}
            for c in _children(e):
                cs, ccnt = visit(c)
                s += cs
                if ccnt:
                    for ck, cc in ccnt.items():
                        cnt[ck] = cnt.get(ck, 0) + cc
            cnt[k] = cnt.get(k, 0) + 1  # post-order: children first, like the historical re-walk
            rep.setdefault(k, e)
            size[k] = s
            sub = (s, cnt)
            memo[id(e)] = sub
        return sub

    for r in roots:
        _, cnt = visit(r)
        if cnt:
            for k, c in cnt.items():
                counts[k] = counts.get(k, 0) + c
    cand = sorted((k for k, c in counts.items() if c >= 2), key=lambda k: size[k])
    cse_map, lines = {}, []
    for i, k in enumerate(cand):
        name = "cse%d_" % i
        lines.append("%sconst %s %s = %s;" % (indent, real, name, _cpp_expand(rep[k], cse_map)))
        cse_map[k] = name
    return lines, [_cpp_cse(r, cse_map) for r in roots]


# --- Foncteurs nommes des temoins de valeurs propres (EigWitness, ADC-289) ---
# Le codegen d'EigWitness emet un APPEL (to_cpp) vers une methode statique ADC_HD de la brique qui
# remplit un adc::Real M[k][k] puis appelle adc::real_eig_minmax. Cette methode est un FONCTEUR NOMME
# (device-clean : pas de lambda etendue cross-TU qui casse nvcc) declare une fois par couple (field, k)
# rencontre dans les formules de la brique. Ces deux fonctions decouvrent les couples et emettent les
# declarations.
def _collect_eig_witnesses(exprs):
    """Couples (field, k) des EigWitness presents dans @p exprs, dedupliques, en ordre stable
    (tri par (k, field)). Vide si aucun temoin -> aucune declaration, brique bit-identique a l'histoire."""
    seen = set()
    memo = set()  # id() : DAG -> chaque objet Expr visite une fois

    def walk(e):
        if id(e) in memo:
            return
        memo.add(id(e))
        if isinstance(e, EigWitness):
            seen.add((e.field, e.k))
        for c in _children(e):
            walk(c)

    for e in exprs:
        walk(_wrap(e))
    return sorted(seen, key=lambda fk: (fk[1], fk[0]))


def _eig_witness_helpers(pairs, indent="  "):
    """Lignes C++ des foncteurs nommes (methodes statiques ADC_HD) pour les couples (field, k) de
    @p pairs (cf. _collect_eig_witnesses). Chaque foncteur prend les k*k entrees de la matrice en
    arguments scalaires (ordre ligne-major), remplit adc::Real M[k][k] et renvoie le champ @c field de
    ``adc::real_eig_minmax(M)``. Aucun argument -> aucune ligne."""
    L = []
    for field, k in pairs:
        params = ", ".join("adc::Real m%d" % i for i in range(k * k))
        L.append("%sstatic ADC_HD adc::Real adc_eig_%s_%dx%d(%s) {"
                 % (indent, field, k, k, params))
        L.append("%s  adc::Real M[%d][%d];" % (indent, k, k))
        for r in range(k):
            sets = " ".join("M[%d][%d] = m%d;" % (r, c, r * k + c) for c in range(k))
            L.append("%s  %s" % (indent, sets))
        L.append("%s  return adc::real_eig_minmax(M).%s;" % (indent, _EIG_FIELDS[field]))
        L.append("%s}" % indent)
    return L


# --- Reciprocal hoist (OPT-IN) -----------------------------------------------
# Without -ffast-math, the compiler cannot replace N divisions by the same denominator
# with 1 reciprocal + N multiplications (rounding would change). We do it at codegen, opt-in:
# for a recurring conservative denominator (>= 2 uses in the live primitives of a
# method), emit once inv_<name> = 1 / <name> and replace its divisions by products.
# Restricted to CONSERVATIVE VARIABLE denominators: computable right after the cons locals,
# before the primitives. OPT-IN because rounding changes (not bit-identical to the default output).
def _count_cons_denoms(e, cons_set, counts):
    """Collects the Var-conservative denominators of @p e into @p counts (name -> occurrences)."""
    if isinstance(e, Div) and isinstance(e.b, Var) and e.b.name in cons_set:
        counts[e.b.name] = counts.get(e.b.name, 0) + 1
    for c in _children(e):
        _count_cons_denoms(c, cons_set, counts)


def _recip_rewrite(e, inv_set):
    """Rewrites @p e: any division by a conservative Var of @p inv_set becomes a product by
    its hoisted reciprocal inv_<name>. Rebuilds NEW nodes (does not mutate the model)."""
    if isinstance(e, Div):
        a = _recip_rewrite(e.a, inv_set)
        if isinstance(e.b, Var) and e.b.name in inv_set:
            return Mul(a, Var("inv_" + e.b.name, "hoist"))
        return Div(a, _recip_rewrite(e.b, inv_set))
    if isinstance(e, _Bin):
        return type(e)(_recip_rewrite(e.a, inv_set), _recip_rewrite(e.b, inv_set))
    if isinstance(e, Neg):
        return Neg(_recip_rewrite(e.a, inv_set))
    if isinstance(e, Sqrt):
        return Sqrt(_recip_rewrite(e.a, inv_set))
    if isinstance(e, Abs):
        return Abs(_recip_rewrite(e.a, inv_set))
    if isinstance(e, StateRef):
        return StateRef(e.side, _recip_rewrite(e.expr, inv_set))
    return e


# --- Symbolic differentiation (autodiff of the Expr tree) -------------------
# dsl.diff(expr, var) differentiates the tree node by node: linearity (+, -), product (a*b)' = a'b + ab',
# quotient (a/b)' = (a'b - ab')/b^2, power (a^n)' = n a^(n-1) a' (constant exponent), root
# sqrt(a)' = a'/(2 sqrt(a)), negation. Used to build the flux Jacobian A = dF/dU (flux_jacobian)
# that the user employs to write its Roe dissipation (m.roe_dissipation). A DEFINED primitive
# is differentiated BY ITS DEFINITION (chain rule); a NON differentiated occurrence stays a
# symbol (readable emission), only the DERIVATIVE descends to the conservatives. Unknown node ->
# NotImplementedError (never a silent zero). Minimal simplifications (0*x, 1*x, x+0).
def _is_const(e, val=None):
    """True if e is a numeric constant (Const); if @p val is given, equal to val."""
    return isinstance(e, Const) and (val is None or e.value == val)


def _s_add(a, b):
    if _is_const(a, 0.0):
        return b
    if _is_const(b, 0.0):
        return a
    if isinstance(a, Const) and isinstance(b, Const):
        return Const(a.value + b.value)
    return Add(a, b)


def _s_neg(a):
    if _is_const(a, 0.0):
        return Const(0.0)
    if isinstance(a, Const):
        return Const(-a.value)
    if isinstance(a, Neg):
        return a.a
    return Neg(a)


def _s_sub(a, b):
    if _is_const(b, 0.0):
        return a
    if _is_const(a, 0.0):
        return _s_neg(b)
    if isinstance(a, Const) and isinstance(b, Const):
        return Const(a.value - b.value)
    return Sub(a, b)


def _s_mul(a, b):
    if _is_const(a, 0.0) or _is_const(b, 0.0):
        return Const(0.0)
    if _is_const(a, 1.0):
        return b
    if _is_const(b, 1.0):
        return a
    if isinstance(a, Const) and isinstance(b, Const):
        return Const(a.value * b.value)
    return Mul(a, b)


def _s_div(a, b):
    if _is_const(a, 0.0):
        return Const(0.0)
    if _is_const(b, 1.0):
        return a
    return Div(a, b)


def _s_pow(a, b):
    # b: exponent (Expr), here assumed INDEPENDENT of the differentiation variable.
    if _is_const(b, 0.0):
        return Const(1.0)
    if _is_const(b, 1.0):
        return a
    return Pow(a, b)


def diff(expr, var, defs=None):
    """Symbolic derivative of @p expr with respect to @p var (variable name or Var).

    @p defs (optional): dictionary {primitive name: definition Expr}. When the differentiation
    meets a DEFINED primitive, it differentiates its DEFINITION (chain rule) -- the primitives
    are expanded down to the conservatives without manual substitution. A primitive with the same name
    as @p var is treated as the independent variable (derivative 1). Without defs, any variable
    other than @p var is independent (derivative 0).

    @return an Expr minimally simplified (0*x, 1*x, x+0, ... removed for a readable emission).
    Raises NotImplementedError on a non differentiable node (naming its type) or a power whose
    exponent depends on @p var (would need a logarithm, a node absent from the DSL)."""
    target = var.name if isinstance(var, Var) else str(var)
    d = defs or {}

    def go(e):
        if isinstance(e, Const):
            return Const(0.0)
        if isinstance(e, RuntimeParamRef):
            return Const(0.0)  # runtime parameter: constant with respect to the conservative state
        if isinstance(e, Var):
            if e.name == target:
                return Const(1.0)
            if e.name in d:
                return go(d[e.name])  # defined primitive -> derivative of its definition (chain)
            return Const(0.0)         # another variable, independent of var
        if isinstance(e, Add):
            return _s_add(go(e.a), go(e.b))
        if isinstance(e, Sub):
            return _s_sub(go(e.a), go(e.b))
        if isinstance(e, Mul):
            return _s_add(_s_mul(go(e.a), e.b), _s_mul(e.a, go(e.b)))
        if isinstance(e, Div):
            num = _s_sub(_s_mul(go(e.a), e.b), _s_mul(e.a, go(e.b)))
            return _s_div(num, _s_mul(e.b, e.b))
        if isinstance(e, Neg):
            return _s_neg(go(e.a))
        if isinstance(e, Sqrt):
            return _s_div(go(e.a), _s_mul(Const(2.0), Sqrt(e.a)))
        if isinstance(e, Abs):
            # d|u| = (u / |u|) u' -- exact derivative away from the fold u = 0 (the smooth floors
            # max(x, eps) = ((x+eps) + |x-eps|)/2 of the 'robust' models give there exactly
            # the expected indicator); AT the fold, u/|u| is NaN: a zero-measure singularity,
            # documented (like the division of quotients).
            return _s_mul(_s_div(e.a, Abs(e.a)), go(e.a))
        if isinstance(e, Sign):
            # d sign(u) = 0 presque partout (saut en u = 0, mesure nulle -- meme convention que le
            # pli de Abs : singularite documentee, jamais rencontree par les clamps sur un ouvert).
            return Const(0.0)
        if isinstance(e, Pow):
            if not _is_const(go(e.b), 0.0):
                raise NotImplementedError(
                    "dsl.diff: derivative of a**b with exponent depending on '%s' (needs a "
                    "logarithm, a node absent from the DSL)" % target)
            # constant exponent with respect to var: (a^b)' = b a^(b-1) a'
            return _s_mul(_s_mul(e.b, _s_pow(e.a, _s_sub(e.b, Const(1.0)))), go(e.a))
        raise NotImplementedError("dsl.diff: non differentiable node %s (%r)" % (type(e).__name__, e))

    return go(_wrap(expr))


def _dir_key(direction):
    """Normalize a direction into 'x' / 'y' (accepts 0/'x'/'X' and 1/'y'/'Y'). Raises otherwise."""
    if direction in (0, "x", "X"):
        return "x"
    if direction in (1, "y", "Y"):
        return "y"
    raise ValueError("invalid direction %r (expected 0/'x' or 1/'y')" % (direction,))


# --- Roe dissipation PROVIDED by the user (m.roe_dissipation) ---------
# The lines d_i are written in terms of left(...)/right(...) of variables/primitives + constants.
# _roe_validate checks that no variable appears OUTSIDE a marker (undetermined state) and that no
# marker is nested; _cpp_roe renders the C++ by resolving left/right through a local prefix
# (L_ for UL, R_ for UR).
def _roe_validate(e, in_marker):
    """Structural check of a roe_dissipation line. Raises ValueError if a variable is outside a
    left()/right() marker (undetermined state) or if a marker is nested. Const / runtime
    parameter: allowed everywhere (without state). Evaluates nothing (usable before the assignment of
    the runtime indices)."""
    if isinstance(e, StateRef):
        if in_marker:
            raise ValueError("m.roe_dissipation: nested left()/right() marker forbidden "
                             "(a subexpression belongs to a single state)")
        _roe_validate(e.expr, True)
        return
    if isinstance(e, Var):
        if not in_marker:
            raise ValueError(
                "m.roe_dissipation: variable '%s' outside marker; wrap each variable or "
                "primitive with dsl.left(...) (state UL) or dsl.right(...) (state UR)" % e.name)
        return
    if isinstance(e, (Const, RuntimeParamRef)):
        return
    for c in _children(e):
        _roe_validate(c, in_marker)


def _cpp_roe(e, prefix):
    """C++ of a roe_dissipation line expression. @p prefix: None at the root level (no active
    state -> a bare variable is an error), 'L_' / 'R_' inside a marker (the variables take
    that local prefix). ALSO used to render a primitive definition with a state prefix
    (e then contains no StateRef). Assumes _roe_validate already passed (defensive errors)."""
    if isinstance(e, StateRef):
        if prefix is not None:
            raise ValueError("m.roe_dissipation: nested left()/right() marker forbidden")
        return _cpp_roe(e.expr, e.side + "_")
    if isinstance(e, Const):
        return repr(e.value)
    if isinstance(e, RuntimeParamRef):
        return e.to_cpp()
    if isinstance(e, Var):
        if prefix is None:
            raise ValueError("m.roe_dissipation: variable '%s' outside left()/right() marker"
                             % e.name)
        return prefix + e.name
    if isinstance(e, Neg):
        return "(-%s)" % _cpp_roe(e.a, prefix)
    if isinstance(e, Sqrt):
        return "std::sqrt(%s)" % _cpp_roe(e.a, prefix)
    if isinstance(e, Abs):
        return "std::fabs(%s)" % _cpp_roe(e.a, prefix)
    if isinstance(e, Sign):
        s = _cpp_roe(e.a, prefix)
        return "(adc::Real(%s > 0) - adc::Real(%s < 0))" % (s, s)
    if isinstance(e, Pow):
        return "std::pow(%s, %s)" % (_cpp_roe(e.a, prefix), _cpp_roe(e.b, prefix))
    if isinstance(e, _Bin):
        return "(%s %s %s)" % (_cpp_roe(e.a, prefix), e.op, _cpp_roe(e.b, prefix))
    raise TypeError("m.roe_dissipation: expression not handled by the codegen: %r" % (e,))


# --- Declarative hyperbolic model -----------------------------------------
class HyperbolicModel:
    """Hyperbolic model written as FORMULAS: conservative variables, primitives (defined by
    expressions), flux, eigenvalues, source, elliptic contribution. cf. module docstring."""

    def __init__(self, name):
        self.name = name
        self.cons_names = []
        self.prim_defs = {}     # name -> Expr (in terms of the cons / previous prims / aux)
        self.aux_names = []      # CANONICAL aux fields read (phi/grad/B_z/T_e), cf. AUX_CANONICAL
        self.aux_extra_names = []  # NAMED aux fields (aux_field): order = index AUX_NAMED_BASE + k
        self._flux = {}         # "x" / "y" -> list of Expr (one per conservative component)
        self._eig = {}          # "x" / "y" -> list of Expr (eigenvalues)
        self._wave_speeds = None  # {"x"/"y": (smin Expr, smax Expr)}: explicit SIGNED speeds
                                  # (set_wave_speeds); None = derived from the eigenvalues if 'p' (historical)
        self._ws_jacobian = None  # {"x"/"y": [[Expr]]} + meta (eig, blocks): EXACT signed speeds
                                  # from the eigenvalues of the flux jacobian (set_wave_speeds_from_jacobian)
        self._source = None     # list of Expr (one per component) or None
        self._elliptic = None   # Expr (contribution to the elliptic right-hand side) or None
        self._stab_speed = None  # Expr: STABILITY speed lambda* (None = fallback eigenvalues)
        self._stab_dt = None     # Expr: direct ADMISSIBLE step dt(U, aux) (None = no bound)
        self._src_freq = None    # Expr: frequency mu(U, aux) of the SOURCE (None = no bound)
        self._proj = None        # [Expr]: PROJECTION ponctuelle post-pas U <- P(U, aux) (ADC-177)
        self._src_jac = None     # [[Expr]] n x n: ANALYTIC Jacobian dS/dU (None = finite differences)
        self._hllc = False       # True: emit the HLLC capability (contact_speed + star state)
        self._roe = False        # True: emit the ROE capability (roe_dissipation from the roles)
        self._roe_rows = None    # {"x": [Expr], "y": [Expr]}: roe_dissipation PROVIDED (outside roles)
        self.prim_state = []    # ordered names of the primitive state (Prim layout); for the codegen
        self.cons_from = None   # list of Expr: conservative in terms of the primitives (to_conservative)
        self.cons_roles = None  # explicit override of the conservative roles (otherwise canonical mapping)
        self.prim_roles = None  # explicit override of the primitive roles (otherwise canonical mapping)
        self.gamma = None       # adiabatic index of the block (EOS), read by the inter-species couplings
                                # on the System side. None -> symbol adc_compiled_gamma not emitted (the System
                                # then falls back to its historical default 1.4, strict backward compatibility).

    def cons(self, name):
        self.cons_names.append(name)
        return Var(name, "cons")

    def conservative_vars(self, *names, roles=None):
        """Declare the conservative variables. @p roles (optional): list of the same length explicitly
        setting the physical role of each component (string 'Density'/'MomentumX'... or None
        to fall back on the canonical mapping of the name); useful for a non-standard layout where the names
        do not suffice to deduce the meaning. Without roles, the canonical name -> role mapping applies."""
        if roles is not None and len(roles) != len(names):
            raise ValueError("conservative_vars: %d roles for %d variables" % (len(roles), len(names)))
        self.cons_roles = list(roles) if roles is not None else None
        return tuple(self.cons(n) for n in names)

    def primitive(self, name, expr):
        """Define a primitive by its formula (in terms of the cons / previous primitives)."""
        self.prim_defs[name] = _wrap(expr)
        return Var(name, "prim")

    def aux(self, name):
        """CANONICAL auxiliary field (e.g. grad_x, grad_y, B_z, T_e) provided at execution. The name
        MUST be a key of AUX_CANONICAL. For an arbitrary NAMED field, see aux_field."""
        self.aux_names.append(name)
        return Var(name, "aux")

    def aux_field(self, name):
        """NAMED auxiliary field (ADC-70 phase 1) provided at execution per block via
        System.set_aux_field(bloc, name, array). Unlike aux(...) (CANONICAL components
        phi/grad/B_z/T_e), name is ARBITRARY: the k-th call reserves component
        AUX_NAMED_BASE + k of the aux channel (read in C++ via aux.extra_field(k)). Returns a Var
        usable in flux / source / eigenvalues / elliptic_rhs like any other aux variable.

        At most AUX_NAMED_MAX named fields per model (FIXED bound on the C++ side, Aux POD). A name already
        canonical (B_z, T_e, phi...) is REJECTED: those fields have their dedicated paths (aux('B_z') +
        set_magnetic_field, etc.); a duplicate named name is also rejected."""
        # The name becomes a C++ LOCAL in the generated formula (cf. _aux_locals_lines) AND the key of the
        # facade table: it must be a valid C++ identifier (letters/digits/_, not a
        # leading digit). Explicit rejection rather than a .so that does not compile.
        if not (isinstance(name, str) and name.isidentifier()):
            raise ValueError("aux_field(%r): invalid name (C++ identifier expected: "
                             "letters/digits/_, without a leading digit)" % (name,))
        if name in AUX_CANONICAL:
            raise ValueError(
                "aux_field('%s') : '%s' is a CANONICAL aux field; use aux('%s') (and the "
                "dedicated path, e.g. set_magnetic_field for B_z, set_electron_temperature_from "
                "for T_e)" % (name, name, name))
        if name in self.aux_extra_names:
            raise ValueError("aux_field('%s') : field already declared" % name)
        if len(self.aux_extra_names) >= AUX_NAMED_MAX:
            raise ValueError("aux_field('%s') : at most %d named aux fields per model "
                             "(kAuxMaxExtra bound on the C++ side)" % (name, AUX_NAMED_MAX))
        self.aux_extra_names.append(name)
        return Var(name, "aux")

    def _aux_locals_lines(self):
        """C++ locals for the aux fields read in a formula: canonical '<n>' <- a.<n> ;
        named '<n>' <- a.extra_field(k) (k = position in aux_extra_names). The local name is
        IDENTICAL to the one the Expr emits (Var.to_cpp), so the formula references it directly."""
        lines = ["    const adc::Real %s = a.%s;" % (n, n) for n in self.aux_names]
        lines += ["    const adc::Real %s = a.extra_field(%d);" % (n, k)
                  for k, n in enumerate(self.aux_extra_names)]
        return lines

    def _reads_aux(self):
        """True if a formula reads an aux field (canonical or named): drives the naming of the Aux
        parameter ('a' vs anonymous) so as not to trigger an unused-parameter warning."""
        return bool(self.aux_names) or bool(self.aux_extra_names)

    def _total_n_aux(self):
        """TOTAL width of the model's aux channel (canonical + named fields)."""
        return aux_total_n_aux(self.aux_names, self.aux_extra_names)

    def set_flux(self, x, y): self._flux = {"x": list(x), "y": list(y)}
    def set_eigenvalues(self, x, y): self._eig = {"x": list(x), "y": list(y)}

    def set_wave_speeds(self, x, y):
        """Explicit SIGNED wave speeds per direction: x = (smin_x, smax_x), y = (smin_y,
        smax_y), expressions of cons / prims / aux. Emits ``wave_speeds(U, aux, dir, smin, smax)``
        on the generated brick WITHOUT requiring a primitive 'p': riemann='hll' becomes available for
        a pressureless model (moment system, isothermal...). The core only gates HLL on
        requires { m.wave_speeds(...) } (block_builder.hpp): no C++ change.

        Takes priority over the historical path (primitive 'p' -> wave_speeds = min/max of
        eigenvalues) when both exist. WITHOUT a call: strictly historical emission.
        If set_eigenvalues is NOT called, max_wave_speed (Rusanov / CFL) is derived from
        ``max(|smin|, |smax|)`` over the two expressions of the direction."""
        x, y = tuple(x), tuple(y)
        if len(x) != 2 or len(y) != 2:
            raise ValueError("set_wave_speeds : expected x=(smin, smax) and y=(smin, smax) "
                             "(got x=%d expression(s), y=%d)" % (len(x), len(y)))
        if self._ws_jacobian is not None:
            raise ValueError("set_wave_speeds : set_wave_speeds_from_jacobian already declared -- one "
                             "single wave_speeds provider")
        self._wave_speeds = {"x": (_wrap(x[0]), _wrap(x[1])),
                             "y": (_wrap(y[0]), _wrap(y[1]))}

    def set_wave_speeds_from_jacobian(self, x=None, y=None, eig="numeric", blocks=None):
        """EXACT signed wave speeds: smin/smax = extremes of the flux jacobian's eigenvalues
        A = dF/dU, computed NUMERICALLY per cell (adc::real_eig_minmax, Francis QR
        on a stack buffer, Gershgorin fallback on non-convergence = safe outer bound). Emits
        ``wave_speeds(U, aux, dir, smin, smax)`` (core HLL gate) and, without set_eigenvalues,
        ``max_wave_speed`` = ``max(|smin|, |smax|)`` over the same blocks.

        @p x, @p y : n_vars x n_vars matrices of expressions dA[i][j] = dF_dir[i]/dU[j]. None
        (default) = AUTODIFF of the declared flux via flux_jacobian(dir) (dsl.diff, primitives
        expanded by the chain rule) -- the jacobian can then not desynchronize from the
        flux. Providing explicit x/y only makes sense to bypass autodiff (hand-simplified
        forms); check_model then confronts them against the flux finite differences.

        @p eig : "numeric" (default) = jacobian entries emitted as formulas, per-block
        eigenvalues at runtime; "fd" = jacobian built BY COLUMNS from the finite differences of the
        COMPILED flux ((flux(U + eps e_k) - flux(U))/eps, ``eps = 1e-6 |U[0]| + 1e-30``, mirror of the
        flagsym != 1 branch of the reference MATLAB) -- generic bring-up/debug, never
        production (O(eps) truncation).

        @p blocks : None (default) = ONE full n_vars x n_vars block, the only unconditionally
        correct mode. Otherwise, a list of INDEX LISTS (possibly non-contiguous, e.g.
        [[0, 1, 4], [2, 3]]) applied to BOTH directions, or a dict {"x": [...], "y": [...]}
        (the block-triangular structures of dFx/dU and dFy/dU differ in general: for a
        moment system, the chains in x are contiguous and those in y are not).
        The extremes are taken over the union of the spectra of the diagonal sub-blocks A[idx][idx].
        CONTRACT: the caller ASSERTS that A is block-(lower-)triangular according to this
        partition (up to permutation) -- on an arbitrary matrix the sub-block extremes
        DO NOT BOUND the spectrum (counter-example [[0, k], [k, 0]]: spectrum +-k, 1x1 sub-blocks
        zero). Indices may be omitted (rows/columns carrying no extreme eigenvalue,
        cf. the skipped block of the reference MATLAB).

        Diagnostics: QR non-convergence silently falls back to the block's Gershgorin bound
        (WIDER, never wrong -- HLL stays stable, only more diffusive); a loss
        of hyperbolicity (complex eigenvalues) is not reported per cell -- verify it
        offline (check_model, golden type eigenvalues15_2D)."""
        if self._wave_speeds is not None:
            raise ValueError("set_wave_speeds_from_jacobian : set_wave_speeds already declared -- one "
                             "single wave_speeds provider")
        if eig not in ("numeric", "fd"):
            raise ValueError("set_wave_speeds_from_jacobian : eig 'numeric' | 'fd' (got %r)" % (eig,))
        nv = self.n_vars
        if (x is None) != (y is None):
            raise ValueError("set_wave_speeds_from_jacobian : provide x AND y, or neither (autodiff)")
        if eig == "fd" and x is not None:
            raise ValueError("set_wave_speeds_from_jacobian : eig='fd' builds the jacobian from "
                             "finite differences of the compiled flux -- x/y make no sense here")
        rows = {}
        if eig == "numeric":
            if x is None:
                if not self._flux:
                    raise ValueError("set_wave_speeds_from_jacobian : call set_flux(...) first "
                                     "(jacobian autodiff)")
                rows = {"x": self.flux_jacobian(0), "y": self.flux_jacobian(1)}
            else:
                for key, mat in (("x", x), ("y", y)):
                    if len(mat) != nv or any(len(r) != nv for r in mat):
                        raise ValueError("set_wave_speeds_from_jacobian : jacobian %s expected "
                                         "%d x %d" % (key, nv, nv))
                    rows[key] = [[_wrap(e) for e in r] for r in mat]
        def norm_blocks(blk, label):
            blk = [list(int(i) for i in b) for b in blk]
            seen = set()
            for b in blk:
                if not b:
                    raise ValueError("set_wave_speeds_from_jacobian : empty block (%s)" % label)
                local = set()
                for i in b:
                    if not (0 <= i < nv):
                        raise ValueError("set_wave_speeds_from_jacobian : index %d out of [0, %d) "
                                         "(%s)" % (i, nv, label))
                    if i in local:
                        raise ValueError("set_wave_speeds_from_jacobian : index %d present twice "
                                         "in the same block (%s)" % (i, label))
                    if i in seen:
                        raise ValueError("set_wave_speeds_from_jacobian : index %d present in "
                                         "two blocks (%s)" % (i, label))
                    local.add(i)
                    seen.add(i)
            return blk

        if blocks is None:
            per_dir = {"x": [list(range(nv))], "y": [list(range(nv))]}
        elif isinstance(blocks, dict):
            if set(blocks) != {"x", "y"}:
                raise ValueError("set_wave_speeds_from_jacobian : blocks dict expected with "
                                 "keys 'x' and 'y' (got %r)" % sorted(blocks))
            per_dir = {k: norm_blocks(blocks[k], k) for k in ("x", "y")}
        else:
            shared = norm_blocks(blocks, "x and y")
            per_dir = {"x": shared, "y": [list(b) for b in shared]}
        self._ws_jacobian = {"rows": rows or None, "eig": eig, "blocks": per_dir,
                             "explicit": x is not None}
    def set_source(self, s): self._source = [_wrap(e) for e in s]
    def set_elliptic_rhs(self, e): self._elliptic = _wrap(e)

    def stability_speed(self, expr):
        """STABILITY speed lambda* (expression of cons / prims / aux): drives the block CFL
        instead of ``max(|eigenvalues|)``. Emitted as ``stability_speed(U, aux, dir)`` (C++ trait
        ``HasStabilitySpeed``): System::step_cfl then uses it for the transport bound
        dt <= cfl*h/lambda*, while the Riemann solvers keep reading max_wave_speed
        (stability != accuracy). WITHOUT a call, the FALLBACK is strictly historical:
        max(abs(eigenvalues)) via max_wave_speed. Compiled like flux/source (no per-cell Python
        callback: compatible with GPU/MPI production). Wired into System AND AmrSystem (mono and
        multi-block; on the AMR side the reduction is evaluated on the COARSE level, where the CFL lives)."""
        self._stab_speed = _wrap(expr)

    def stability_dt(self, expr_dt):
        """Direct ADMISSIBLE step dt(U, aux) (expression > 0, in time units): local step
        bound, emitted as ``stability_dt(U, aux)`` (C++ trait ``HasStabilityDt``). System::step_cfl
        imposes dt <= min_cells(stability_dt) * substeps / stride (the cfl is NOT applied: the
        model already declares an admissible step). The most general form (stiff source, local coupling,
        non-reducible transport+source formula). WITHOUT a call, no additional bound (historical
        step policy). Compiled like flux/source (GPU/MPI production). Wired into System AND
        AmrSystem (mono and multi-block; on the AMR side evaluated on the COARSE level)."""
        self._stab_dt = _wrap(expr_dt)

    def source_frequency(self, expr_mu):
        """Local FREQUENCY mu(U, aux) [1/time] of the SOURCE (relaxation, collision, reaction):
        the 'second CFL' of the meeting -- bound dt <= cfl * substeps / (stride * max_cells(mu)),
        WITHOUT a space step (a source bounded in 1/time). Emitted as ``frequency(U, aux)`` on the
        generated SOURCE BRICK (C++ contract of source bricks, cf. physics/source.hpp);
        CompositeModel forwards it (HasSourceFrequency trait) and System/AmrSystem::step_cfl
        aggregate it. REQUIRES set_source/m.source (the frequency is a property of the source).
        WITHOUT a call, the source does not constrain the step (historical). Compiled (GPU/MPI production)."""
        self._src_freq = _wrap(expr_mu)

    def projection(self, exprs):
        """PROJECTION PONCTUELLE post-pas (ADC-177) : U <- P(U, aux), une expression par composante
        conservative (en fonction des cons / prims / aux). Emise comme ``project(U, aux)`` sur la
        brique hyperbolique generee (trait C++ ``HasPointwiseProjection``) ; le System l'applique sur
        les cellules VALIDES de chaque bloc a la FIN de chaque macro-pas ENTIER (apres transport +
        etage source + couplages ; jamais par etage RK -- semantique POST-PAS). CONTRAT : P doit etre
        une PROJECTION (idempotente : P(P(U)) == P(U)) et PONCTUELLE (aucune lecture de voisin). Les
        formules de realisabilite restent cote cas ; les clamps s'ecrivent SANS branche, en max/min
        via abs_ / sign : p.ex. positivite q >= 0 : (q + abs_(q)) / 2. Compilee comme flux/source
        (CSE comprise, production GPU/MPI -- remplace le callback Python par cellule). Backends
        'aot' (add_compiled_block) et 'production' System (add_native_block) ; le backend 'prototype'
        et target='amr_system' la REJETTENT explicitement (jamais d'ignore silencieux). SANS appel :
        aucun hook emis, chemin bit-identique."""
        exprs = [_wrap(e) for e in exprs]
        if len(exprs) != self.n_vars:
            raise ValueError("projection : %d expressions attendues (une par composante "
                             "conservative), recu %d" % (self.n_vars, len(exprs)))
        self._proj = exprs

    def projection_value(self, U, aux=None):
        """EVALUATEUR numpy de la projection ponctuelle emise (miroir exact du project(U, aux) C++) :
        U (n_vars, ...) -> U projete. Reference de test / prototypage hote. ValueError si
        projection([...]) n'a pas ete appelee."""
        if self._proj is None:
            raise ValueError("projection_value : appeler projection([...]) d'abord")
        env = self._env(U, aux)
        shape = np.asarray(U[0]).shape
        return np.stack([np.broadcast_to(e.eval(env), shape) for e in self._proj], axis=0)

    def source_jacobian(self, rows):
        """ANALYTICAL JACOBIAN of the source: dS/dU, n_vars x n_vars matrix of expressions
        (rows[r][c] = dS_r/dU_c, as a function of cons / prims / aux). Emitted as
        ``jacobian(U, aux, J)`` on the generated SOURCE brick, forwarded by CompositeModel
        (C++ trait ``HasSourceJacobian``): the Newton of the implicit source (IMEX /
        SourceImplicitBE) uses it INSTEAD of finite differences -- exactness (no more
        fd_eps noise) and saved source evaluations. REQUIRES m.source. WITHOUT a call: historical
        finite differences, bit-identical."""
        self._src_jac = [[_wrap(e) for e in row] for row in rows]

    def enable_hllc(self):
        """Emits the HLLC CAPABILITY (audit wave 3): ``contact_speed`` (Toro) + ``hllc_star_state``
        GENERATED from the block's ROLES (Density / MomentumX / MomentumY, Energy optional) and the
        primitive 'p' -- the core's contact-resolving HLLC solver (C++ trait HasHLLCStructure)
        then becomes available for THIS model, EVEN outside 4-variable Euler (3-var isothermal,
        moments with passive scalars: any component without a particular role is advected
        passively in the star state, Us[c] = fac*U[c]/rho). REQUIRES: roles Density/MomentumX/
        MomentumY declared + primitive 'p' (explicit error at emission otherwise)."""
        self._hllc = True
        return self

    def enable_roe(self):
        """Emits the ROE CAPABILITY (audit balance, GENERICITY_2026-06.md point 11):
        ``roe_dissipation(UL, AL, UR, AR, dir)`` = ``|A_roe| (UR - UL)`` GENERATED from the block's
        ROLES -- the core's Roe-like solver (C++ trait HasRoeDissipation, F = 1/2(FL+FR) - 1/2 d)
        becomes available for THIS model, EVEN outside 4-variable Euler:

        - roles Density/MomentumX/MomentumY + Energy: ideal-gas Roe algebra, exact
          TRANSCRIPTION of the canonical C++ path (sqrt(rho)-weighted averages, gamma-1 deduced from
          ``p/(E - 1/2 rho |v|^2)``, Harten entropy fix on the acoustic waves);
        - roles Density/MomentumX/MomentumY WITHOUT Energy (isothermal / pseudo-pressure): same
          decomposition without the energy row, LOCAL sound speed c = sqrt(p/rho) Roe-averaged
          (standard generalization outside ideal gas);
        - any component OUTSIDE the fluid roles is treated as a PASSIVE SCALAR carried by the
          entropy wave (row identical to the tangential momentum, phi = q/rho).

        REQUIRES: roles Density/MomentumX/MomentumY declared + primitive 'p' (explicit error at
        emission otherwise). Without a call: nothing emitted, riemann='roe' stays Euler-4-var-only.

        EXCLUSIVE with m.roe_dissipation: the capability from the roles and the dissipation PROVIDED by
        the user are two providers of the SAME roe_dissipation hook -- declaring both
        raises (one single provider)."""
        if self._roe_rows is not None:
            raise ValueError("enable_roe : roe_dissipation(...) already provided -- one single provider "
                             "of the roe_dissipation hook (capability from the roles OR provided)")
        self._roe = True

    def roe_dissipation(self, x, y):
        """Roe dissipation PROVIDED by the user (outside the fluid-role families): n_vars
        expressions per direction (rows d_i), emitted as the C++ hook
        ``roe_dissipation(UL, AL, UR, AR, dir)`` = d (HasRoeDissipation trait; the core does
        F = 1/2(FL+FR) - 1/2 d, cf. RoeFlux). It is the 'provided' counterpart of m.enable_roe (generated
        from the ROLES): here the user writes THEIR eigenstructure -- same spirit as
        m.source_jacobian (provided, not invented). The helper m.flux_jacobian(dir) (A = dF/dU
        auto-derived by dsl.diff) assists this writing.

        TWO-STATE VOCABULARY: each variable/primitive must be wrapped by dsl.left(...) (state
        UL) or dsl.right(...) (state UR); a Roe average is therefore written explicitly
        (left(sqrt(rho))*left(u) + right(sqrt(rho))*right(u)) / (left(sqrt(rho)) + right(sqrt(rho))).
        A BARE variable (without a marker) raises at declaration (undetermined state).

        @p x, @p y : lists of n_vars expressions (rows for dir=0 and dir=1). TWO EXPLICIT sets
        (no role mapping here): at dir=0 the normal component is the x axis, at dir=1 the y axis.

        Guards: length n_vars per direction; each variable under left/right; conflict with
        enable_roe (one single provider of the hook) -> error. WITHOUT a call: nothing emitted (bit-identical).
        Requires the 'aot' or 'production' backend (the hook is emitted in the generated brick)."""
        if self._roe:
            raise ValueError("roe_dissipation : enable_roe() already called -- one single provider of the "
                             "roe_dissipation hook (capability from the roles OR provided)")
        rx, ry = list(x), list(y)
        if len(rx) != self.n_vars or len(ry) != self.n_vars:
            raise ValueError("roe_dissipation : %d expressions expected per direction (got x=%d, "
                             "y=%d)" % (self.n_vars, len(rx), len(ry)))
        rows = {"x": [_wrap(e) for e in rx], "y": [_wrap(e) for e in ry]}
        for key in ("x", "y"):
            for e in rows[key]:
                _roe_validate(e, False)  # rejects any variable outside a left()/right() marker
        self._roe_rows = rows

    def flux_jacobian(self, dir):
        """Flux jacobian A = dF_dir/dU : n_vars x n_vars matrix of expressions, A[i][j] =
        d(flux_dir[i])/d(cons[j]), auto-derived from the declared fluxes (via dsl.diff with primitive
        substitution). CONSTRUCTION HELPER (the user uses it to write m.roe_dissipation):
        EMITS NOTHING by itself. @p dir : 0/'x' (x axis) or 1/'y' (y axis). REQUIRES set_flux(...).

        The primitives are expanded by their definition (chain); a non-derived primitive
        stays a symbol in the result (evaluating it numerically requires an env containing its values,
        e.g. HyperbolicModel._env)."""
        if not self._flux:
            raise ValueError("flux_jacobian : call set_flux(...) first")
        key = _dir_key(dir)
        comps = self._flux.get(key, [])
        if len(comps) != self.n_vars:
            raise ValueError("flux_jacobian : flux %s expected with %d components (got %d)"
                             % (key, self.n_vars, len(comps)))
        defs = self.prim_defs
        return [[diff(comps[i], self.cons_names[j], defs) for j in range(self.n_vars)]
                for i in range(self.n_vars)]

    def left(self, expr):
        """Marks @p expr as evaluated on the LEFT state UL (sugar for dsl.left, m.roe_dissipation)."""
        return left(expr)

    def right(self, expr):
        """Marks @p expr as evaluated on the RIGHT state UR (sugar for dsl.right, m.roe_dissipation)."""
        return right(expr)

    def set_gamma(self, gamma):
        """Adiabatic index of the block (compressible EOS). Carried by the generated .so via the
        optional symbol adc_compiled_gamma, so that the System's inter-species couplings (collision,
        thermal exchange, T_e) use the RIGHT gamma instead of the historical default 1.4. Without a call,
        no gamma symbol is emitted (backward compat: the System keeps its default)."""
        self.gamma = float(gamma)

    def set_primitive_state(self, *vars_or_names, roles=None):
        """Declares the ORDERED layout of the primitive state (Prim): component names, in order.
        Necessary for the brick codegen (to_primitive fills Prim in this order). Each name must
        be a conservative variable or an already-defined primitive. @p roles (optional): same
        convention as conservative_vars (explicit per-component override, None = canonical mapping)."""
        self.prim_state = [v.name if isinstance(v, Var) else str(v) for v in vars_or_names]
        if roles is not None and len(roles) != len(self.prim_state):
            raise ValueError("set_primitive_state : %d roles for %d variables"
                             % (len(roles), len(self.prim_state)))
        self.prim_roles = list(roles) if roles is not None else None

    def set_conservative_from(self, exprs):
        """Formulas of the conservative state as a function of the primitives (one per conservative
        variable, in conservative_vars order). Used to generate to_conservative: the DSL cannot invert
        the primitives symbolically, so the user provides the inverse explicitly."""
        self.cons_from = [_wrap(e) for e in exprs]

    @property
    def n_vars(self): return len(self.cons_names)

    # --- evaluation (CPU interpreter, numpy) ---
    def _env(self, U, aux):
        """Environment: cons (from U), aux (provided), then derived primitives (insertion
        order = dependency order)."""
        env = {self.cons_names[i]: U[i] for i in range(len(self.cons_names))}
        if aux:
            env.update(aux)
        for pname, pexpr in self.prim_defs.items():
            env[pname] = pexpr.eval(env)
        return env

    def flux(self, U, aux, dir):
        """Physical flux in direction dir (0=x, 1=y). U: numpy (n_vars, ...)."""
        env = self._env(U, aux)
        comps = self._flux["x" if dir == 0 else "y"]
        return np.stack([np.broadcast_to(c.eval(env), U[0].shape) for c in comps], axis=0)

    def max_wave_speed(self, U, aux, dir):
        """max_k max_cells ``|lambda_k|``: Rusanov / CFL bound. Source: eigenvalues
        (legacy); WITHOUT set_eigenvalues, ``max(|smin|, |smax|)`` of the explicit signed speeds
        (set_wave_speeds), an exact mirror of the C++ emission."""
        env = self._env(U, aux)
        key = "x" if dir == 0 else "y"
        if not self._eig.get(key) and self._ws_jacobian is not None:
            lo, hi = self._ws_jacobian_value(U, env, key)
            return max(float(np.max(np.abs(lo))), float(np.max(np.abs(hi))))
        exprs = self._eig.get(key) or (list(self._wave_speeds[key])
                                       if self._wave_speeds is not None else None)
        if not exprs:
            raise ValueError("max_wave_speed: neither set_eigenvalues(...) nor set_wave_speeds(...) nor "
                             "set_wave_speeds_from_jacobian(...) declared on model '%s'"
                             % self.name)
        return max(float(np.max(np.abs(np.asarray(e.eval(env))))) for e in exprs)

    def _ws_jacobian_value(self, U, env, key):
        """Numpy evaluator of the jacobian path: extremes of the real parts of the eigenvalues
        of the sub-blocks, per sample (mirror of the emitted wave_speeds; np.linalg.eigvals)."""
        ws = self._ws_jacobian
        nv = self.n_vars
        nsmp = int(np.asarray(U[0]).reshape(-1).shape[0])
        if ws["eig"] == "fd":
            base = np.stack([np.broadcast_to(np.asarray(c.eval(env), dtype=float), (nsmp,))
                             if hasattr(c, "eval") else np.full((nsmp,), float(c))
                             for c in (self._flux[key])], axis=0)
            J = np.empty((nsmp, nv, nv))
            Uflat = np.stack([np.broadcast_to(np.asarray(env[c], dtype=float), (nsmp,))
                              for c in self.cons_names], axis=0)
            for k in range(nv):
                eps = 1e-6 * np.abs(Uflat[0]) + 1e-30
                Up = Uflat.copy()
                Up[k] += eps
                envp = self._env(Up, {n: env[n] for n in self.aux_names} if self.aux_names else None)
                Fp = np.stack([np.broadcast_to(np.asarray(c.eval(envp), dtype=float), (nsmp,))
                               for c in self._flux[key]], axis=0)
                J[:, :, k] = ((Fp - base) / eps).T
        else:
            rows = ws["rows"][key]
            J = np.empty((nsmp, nv, nv))
            for i in range(nv):
                for j in range(nv):
                    J[:, i, j] = np.broadcast_to(
                        np.asarray(rows[i][j].eval(env), dtype=float), (nsmp,))
        lo = np.full((nsmp,), np.inf)
        hi = np.full((nsmp,), -np.inf)
        for b in ws["blocks"][key]:
            idx = np.asarray(b)
            lam = np.linalg.eigvals(J[:, idx[:, None], idx[None, :]])
            lo = np.minimum(lo, lam.real.min(axis=1))
            hi = np.maximum(hi, lam.real.max(axis=1))
        return lo, hi

    def _flux_jacobian_spectral_radius(self, U, aux, dir):
        """Spectral radius max_cells max_k |Re(lambda_k)| of the FULL dense Jacobian A = dF_dir/dU,
        evaluated by CENTRAL finite differences on the interpreted flux. Independent of any declared
        partition (set_wave_speeds_from_jacobian blocks=...): serves as a non-circular reference bound
        against max_wave_speed. Returns None if a perturbed state leaves the domain (non-finite flux)
        -- in which case nothing can be concluded."""
        nv = self.n_vars
        U = np.asarray(U, dtype=float)
        nsmp = U.shape[1]
        J = np.empty((nsmp, nv, nv))
        for j in range(nv):
            eps = 1e-6 * np.abs(U[j]) + 1e-7
            Up = U.copy(); Up[j] = Up[j] + eps
            Um = U.copy(); Um[j] = Um[j] - eps
            Fp = self.flux(Up, aux, dir)
            Fm = self.flux(Um, aux, dir)
            if not (bool(np.all(np.isfinite(Fp))) and bool(np.all(np.isfinite(Fm)))):
                return None
            for i in range(nv):
                J[:, i, j] = (np.broadcast_to(Fp[i], (nsmp,))
                              - np.broadcast_to(Fm[i], (nsmp,))) / (2.0 * eps)
        lam = np.linalg.eigvals(J)
        return float(np.max(np.abs(lam.real)))

    def wave_speeds_value(self, U, aux, dir):
        """Numpy evaluator of the signed speeds (smin, smax) -- mirror of the emitted wave_speeds:
        explicit pair (set_wave_speeds) if declared, otherwise min/max of the eigenvalues (legacy
        path, which requires 'p' to be EMITTED but remains evaluable here)."""
        env = self._env(U, aux)
        key = "x" if dir == 0 else "y"
        if self._wave_speeds is not None:
            lo, hi = self._wave_speeds[key]
            return (np.asarray(lo.eval(env), dtype=float),
                    np.asarray(hi.eval(env), dtype=float))
        if self._ws_jacobian is not None:
            return self._ws_jacobian_value(U, env, key)
        eigs = [np.asarray(e.eval(env), dtype=float) for e in self._eig.get(key, [])]
        if not eigs:
            raise ValueError("wave_speeds_value: neither set_wave_speeds(...) nor set_eigenvalues(...) "
                             "declared on model '%s'" % self.name)
        eigs = list(np.broadcast_arrays(*eigs)) if len(eigs) > 1 else eigs  # mixed shapes (constant lambda)
        return (np.min(np.stack(eigs), axis=0), np.max(np.stack(eigs), axis=0))

    def source_value(self, U, aux):
        """Source term (numpy (n_vars, ...)), or zeros if not defined."""
        if self._source is None:
            return np.zeros_like(U)
        env = self._env(U, aux)
        return np.stack([np.broadcast_to(s.eval(env), U[0].shape) for s in self._source], axis=0)

    def to_python_flux(self, aux=None):
        """Produces an adc.PythonFlux (host backend) from the formulas: the model RUNS
        (interpreted on CPU). aux: dict name -> array (auxiliary fields), frozen for this flux."""
        import adc
        a = aux or {}
        return adc.PythonFlux(
            lambda U, d: self.flux(U, a, d),
            lambda U: max(self.max_wave_speed(U, a, 0), self.max_wave_speed(U, a, 1)))

    def check(self):
        """Checks that every referenced variable (primitives, flux, eigenvalues, source) is
        properly declared (cons / prim / aux). Raises ValueError otherwise (dependency check)."""
        known = (set(self.cons_names) | set(self.prim_defs) | set(self.aux_names)
                 | set(self.aux_extra_names))  # named aux fields (aux_field): ADC-70
        used = set()
        groups = [self._flux.get("x", []), self._flux.get("y", []),
                  self._eig.get("x", []), self._eig.get("y", []), self._source or [],
                  [e for e in (self._stab_speed, self._stab_dt, self._src_freq)
                   if e is not None],
                  self._proj or [],  # projection ponctuelle post-pas (ADC-177)
                  [e for row in (self._src_jac or []) for e in row]]
        if self._wave_speeds is not None:
            groups.append(list(self._wave_speeds["x"]) + list(self._wave_speeds["y"]))
        if self._ws_jacobian is not None and self._ws_jacobian["rows"] is not None:
            for d in ("x", "y"):
                groups.append([e for row in self._ws_jacobian["rows"][d] for e in row])
        if self._roe_rows is not None:
            groups.append(self._roe_rows["x"])
            groups.append(self._roe_rows["y"])
        for e in self.prim_defs.values():
            used |= e.deps()
        for grp in groups:
            for e in grp:
                used |= e.deps()
        if self._elliptic is not None:
            used |= self._elliptic.deps()
        missing = used - known
        if missing:
            raise ValueError("model '%s': undefined variables %s" % (self.name, sorted(missing)))
        # source_frequency is a property of the SOURCE (emitted on the generated source brick):
        # declaring it without a source would be SILENTLY lost -> explicit error.
        if self._src_freq is not None and self._source is None:
            raise ValueError("model '%s': source_frequency(...) declared without a source "
                             "(call m.source([...]) -- the frequency is emitted on the generated "
                             "source brick)" % self.name)
        if self._src_jac is not None and self._source is None:
            raise ValueError("model '%s': source_jacobian(...) declared without a source "
                             "(call m.source([...]) -- the Jacobian is emitted on the generated "
                             "source brick)" % self.name)
        # roe_dissipation and enable_roe are two providers of the SAME hook: exclusive (defensive;
        # already rejected at declaration). Structural re-check of the rows (left/right) along the way.
        if self._roe_rows is not None:
            if self._roe:
                raise ValueError("model '%s': enable_roe() and roe_dissipation(...) declared "
                                 "together -- a single provider of the roe_dissipation hook" % self.name)
            for key in ("x", "y"):
                for e in self._roe_rows[key]:
                    _roe_validate(e, False)
        return True

    def check_model(self, samples=None, n_samples=64, seed=0, aux=None, rtol=1e-8, atol=1e-10,
                    raise_on_error=True, jac_rtol=1e-3, jac_atol=1e-9):
        """Generic NUMERICAL verification of the symbolic model (audit 2026-06, work item 6):
        evaluates the formulas on sample states and checks, when the piece exists:

        - finite flux (both directions);
        - finite source;
        - finite elliptic_rhs;
        - finite and real eigenvalues; finite max_wave_speed and >= 0;
        - consistency wave_speeds <-> max_wave_speed: ``max(|lambda_min|, |lambda_max|) <= mws``;
        - NON-CIRCULAR bounding of the spectrum: the spectral radius of the full dense flux Jacobian
          (central finite differences, independent of any blocks= partition) does not exceed
          max_wave_speed -- catches a set_wave_speeds_from_jacobian partition that does NOT bound the
          eigenvalues (mws underestimated, unsafe CFL) where the consistency above, derived from the
          SAME partition, still holds;
        - round-trip to_conservative(to_primitive(U)) ~= U (if prim_state + cons_from declared);
        - positivity of the Density-role components (and of the primitive 'p' if declared) on the
          samples (which are generated positive for these roles).

        @p jac_rtol, @p jac_atol: tolerances of the spectral bounding (radius <= mws*(1+jac_rtol)
        + jac_atol); relaxed to absorb the noise of the finite-difference Jacobian.

        @p samples: array (n_vars, N) of conservative states to test; None -> N = n_samples random
        states (fixed seed, reproducible): Density-role components in [0.1, 2], the others
        in [-1, 1]; an Energy-role component gets ``1 + |kinetic|`` to stay physical.
        @p aux: dict name -> value(s) of the auxiliary fields (default: zeros).
        @return dict {"ok": bool, "failures": [str], "n_samples": N}. raise_on_error=True (default)
        raises ValueError listing the failures. PRE-COMPILATION: checks the FORMULAS (the compiled .so
        emits exactly these formulas); the RUNTIME counterpart on an installed block is
        System.check_model(block)."""
        self.check()  # declared dependencies (raises if a variable does not exist)
        rng = np.random.default_rng(seed)
        nv = self.n_vars
        roles = roles_for(self.cons_names, self.cons_roles)
        if samples is None:
            U = rng.uniform(-1.0, 1.0, size=(nv, int(n_samples)))
            kinetic = np.zeros(int(n_samples))
            for i, r in enumerate(roles):
                if r == "Density":
                    U[i] = rng.uniform(0.1, 2.0, size=int(n_samples))
            for i, r in enumerate(roles):
                if r in ("MomentumX", "MomentumY"):
                    kinetic += U[i] ** 2
            for i, r in enumerate(roles):
                if r == "Energy":
                    U[i] = 1.0 + kinetic  # above the kinetic: pressure > 0 for an ideal gas
        else:
            U = np.asarray(samples, dtype=float)
            if U.ndim != 2 or U.shape[0] != nv:
                raise ValueError("check_model: samples must be (n_vars=%d, N)" % nv)
        a = {n: np.zeros(U.shape[1]) for n in (self.aux_names + self.aux_extra_names)}
        if aux:
            for k, v in aux.items():
                a[k] = np.broadcast_to(np.asarray(v, dtype=float), (U.shape[1],)).copy()
        failures = []

        def finite(x):
            return bool(np.all(np.isfinite(np.asarray(x, dtype=float))))

        for d, dn in ((0, "x"), (1, "y")):
            if not finite(self.flux(U, a, d)):
                failures.append("flux %s non-finite on the samples" % dn)
        if self._source is not None and not finite(self.source_value(U, a)):
            failures.append("source non-finite on the samples")
        if self._elliptic is not None:
            env = self._env(U, a)
            if not finite(self._elliptic.eval(env)):
                failures.append("elliptic_rhs non-finite on the samples")
        env = self._env(U, a)
        for d in ("x", "y"):
            for k, e in enumerate(self._eig.get(d, [])):
                lam = np.asarray(e.eval(env), dtype=float)
                if np.iscomplexobj(lam):
                    failures.append("eigenvalue %s[%d] complex (non-hyperbolic system?)" % (d, k))
                elif not finite(lam):
                    failures.append("eigenvalue %s[%d] non-finite" % (d, k))
        if self._wave_speeds is not None:
            for d in ("x", "y"):
                lo = np.asarray(self._wave_speeds[d][0].eval(env), dtype=float)
                hi = np.asarray(self._wave_speeds[d][1].eval(env), dtype=float)
                if not (finite(lo) and finite(hi)):
                    failures.append("wave_speeds %s (explicit) non-finite" % d)
                elif bool(np.any(lo > hi)):
                    failures.append("wave_speeds %s (explicit): smin > smax on some samples" % d)
        for d, dn in ((0, "x"), (1, "y")):
            mws = self.max_wave_speed(U, a, d)
            if not np.isfinite(mws) or mws < 0:
                failures.append("max_wave_speed %s non-finite or negative (%r)" % (dn, mws))
            else:
                # consistency wave_speeds <-> max_wave_speed: the SIGNED extremes actually emitted
                # (explicit pair if declared, otherwise eigenvalues) must be covered by the
                # Rusanov / CFL bound.
                lo, hi = self.wave_speeds_value(U, a, d)
                ext = max(float(np.max(np.abs(lo))), float(np.max(np.abs(hi))))
                if ext > mws * (1.0 + rtol) + atol:
                    failures.append("wave_speeds %s inconsistent with max_wave_speed (%g > %g)"
                                    % (dn, ext, mws))
                # NON-CIRCULAR bounding: the spectral radius of the dense flux Jacobian (central
                # FD, no partition) must be bounded by max_wave_speed. A blocks= partition that is
                # not really block-triangular yields sub-block extremes that do NOT bound the
                # spectrum -> mws too small, detected here.
                if self._flux:
                    radius = self._flux_jacobian_spectral_radius(U, a, d)
                    if radius is not None and radius > mws * (1.0 + jac_rtol) + jac_atol:
                        failures.append(
                            "partition %s: max_wave_speed (%g) does not bound the spectrum of the "
                            "flux Jacobian (spectral radius %g) -- the blocks= partition of "
                            "set_wave_speeds_from_jacobian does not bound the eigenvalues, the "
                            "Rusanov/CFL bound is underestimated" % (dn, mws, radius))
        # round-trip cons -> prim -> cons (when both directions are declared)
        if self.prim_state and self.cons_from is not None:
            penv = {nm: np.broadcast_to(np.asarray(env[nm], dtype=float), (U.shape[1],))
                    for nm in self.prim_state}
            U2 = np.stack([np.broadcast_to(np.asarray(e.eval(penv), dtype=float), (U.shape[1],))
                           for e in self.cons_from], axis=0)
            if not finite(U2):
                failures.append("to_conservative(to_primitive(U)) non-finite")
            elif not np.allclose(U2, U, rtol=rtol, atol=atol):
                err = float(np.max(np.abs(U2 - U)))
                failures.append("round-trip to_conservative(to_primitive(U)) != U (max deviation %g: "
                                "inconsistent conversions)" % err)
        # positivity: Density roles (conservative) and primitive 'p' (pressure) if declared
        for i, r in enumerate(roles):
            if r == "Density" and not bool(np.all(U[i] > 0)):
                failures.append("component '%s' (Density role) not strictly positive on the "
                                "samples" % self.cons_names[i])
        if "p" in self.prim_defs:
            p = np.asarray(env["p"], dtype=float)
            if not finite(p):
                failures.append("primitive 'p' (pressure) non-finite")
            elif not bool(np.all(p > 0)):
                failures.append("primitive 'p' (pressure) not strictly positive on physical "
                                "states (suspicious EOS)")
        report = {"ok": not failures, "failures": failures, "n_samples": int(U.shape[1])}
        if failures and raise_on_error:
            raise ValueError("check_model('%s'): %d failure(s):\n  - %s"
                             % (self.name, len(failures), "\n  - ".join(failures)))
        return report

    # --- RUNTIME parameters (P7-b): collection + index assignment + generated member -------------
    def _all_exprs(self):
        """All the Expr of the model (primitives, flux, eigenvalues, source, elliptic,
        cons_from). Used to discover the RuntimeParamRef nodes hidden in the tree."""
        out = list(self.prim_defs.values())
        for d in ("x", "y"):
            out += self._flux.get(d, [])
            out += self._eig.get(d, [])
        if self._wave_speeds is not None:  # explicit signed speeds: runtime params included
            for d in ("x", "y"):
                out += list(self._wave_speeds[d])
        if self._ws_jacobian is not None and self._ws_jacobian["rows"] is not None:
            for d in ("x", "y"):  # jacobian entries: runtime params included
                out += [e for row in self._ws_jacobian["rows"][d] for e in row]
        if self._source is not None:
            out += [_wrap(e) for e in self._source]
        if self.cons_from is not None:
            out += list(self.cons_from)
        if self._elliptic is not None:
            out.append(self._elliptic)
        if self._roe_rows is not None:  # Roe rows provided: discover their runtime params (via StateRef)
            out += self._roe_rows["x"] + self._roe_rows["y"]
        return out

    def runtime_param_nodes(self):
        """RuntimeParamRef nodes PRESENT in the formulas, deduplicated by name (the same param may
        appear several times but shares the SAME node object). Order SORTED by name (stable index
        = position in this list, mirror of RuntimeParams on the C++ side)."""
        seen = {}

        def walk(e):
            if isinstance(e, RuntimeParamRef):
                seen.setdefault(e.name, e)
                return
            for c in _children(e):
                walk(c)

        for e in self._all_exprs():
            walk(e)
        return [seen[k] for k in sorted(seen)]

    def assign_runtime_indices(self):
        """Assigns to each RuntimeParamRef its STABLE index (sorted order of names) and returns the
        ordered list of nodes. CALLED before any brick codegen: without this call, to_cpp() would raise
        (index -1). Idempotent (reassigns the same indices). Rejects a model exceeding the C++ bound
        kMaxRuntimeParams (otherwise the fixed-size array would overflow)."""
        nodes = self.runtime_param_nodes()
        if len(nodes) > _K_MAX_RUNTIME_PARAMS:
            raise ValueError(
                "model '%s': %d runtime parameters > kMaxRuntimeParams bound=%d "
                "(include/adc/runtime/runtime_params.hpp); reduce the number of runtime params"
                % (self.name, len(nodes), _K_MAX_RUNTIME_PARAMS))
        for k, node in enumerate(nodes):
            node.index = k
        return nodes

    def _runtime_params_member(self):
        """C++ line declaring the RuntimeParams member of a generated brick, initialized to the
        DECLARATION values (default without a runtime set call). Empty string if the model has no runtime
        param (brick strictly identical to history -> bit-identity of const params preserved)."""
        nodes = self.assign_runtime_indices()
        if not nodes:
            return ""
        vals = ", ".join(repr(node.value) for node in nodes)
        return ("  adc::RuntimeParams params{%d, {%s}};  // params RUNTIME (P7-b) : ecrasables a "
                "l'execution\n" % (len(nodes), vals))

    def has_runtime_params(self):
        """True if at least one formula reads a runtime parameter (kind='runtime')."""
        return bool(self.runtime_param_nodes())

    # --- codegen (step 2 : symbolic tree -> compilable C++) ---
    def _codegen_exprs(self, exprs, cse, real="adc::Real", indent="    "):
        """(CSE local lines, [C++ per expr]). If cse, factor the common subexpressions
        (H, c...) into ``cseK_`` locals ; otherwise inline each expression via to_cpp."""
        if cse:
            return _cse_emit(list(exprs), real, indent)
        return [], [e.to_cpp() for e in exprs]

    def _live_prims(self, exprs, seed=()):
        """Names of the primitives transitively referenced by @p exprs (and the @p seed names).
        Closure over prim_defs: a live primitive pulls in its own primitive dependencies.
        Used to emit in a method only the primitives actually used (dead-code elimination):
        the live expressions stay identical, so the values are bit-identical."""
        prim = self.prim_defs
        live = set()
        stack = [n for n in seed if n in prim]
        for e in exprs:
            stack.extend(d for d in e.deps() if d in prim)
        while stack:
            nm = stack.pop()
            if nm in live:
                continue
            live.add(nm)
            stack.extend(d for d in prim[nm].deps() if d in prim)
        return live

    def _prim_block(self, live=None, hoist=False):
        """``const adc::Real <prim> = ...;`` lines of a method. @p live (default None = all):
        declares only the live primitives. @p hoist: hoists at the top the reciprocal of the
        recurring conservative denominators (>= 2 uses) and replaces those divisions by
        products (OPT-IN, changes the rounding). Without @p hoist and with live=None, historical output."""
        items = [(p, e) for p, e in self.prim_defs.items() if live is None or p in live]
        if not hoist:
            return ["    const adc::Real %s = %s;" % (p, e.to_cpp()) for p, e in items]
        cons_set = set(self.cons_names)
        counts = {}
        for _, e in items:
            _count_cons_denoms(e, cons_set, counts)
        inv = [n for n in self.cons_names if counts.get(n, 0) >= 2]  # stable cons order
        inv_set = set(inv)
        lines = ["    const adc::Real inv_%s = adc::Real(1) / %s;" % (n, n) for n in inv]
        lines += ["    const adc::Real %s = %s;" % (p, _recip_rewrite(e, inv_set).to_cpp())
                  for p, e in items]
        return lines

    def _jac_entries(self):
        """Entries (Expr) of the Jacobian sub-blocks of both directions (wave_speeds 'numeric'
        path). Drives the dead-code elimination of max_wave_speed / wave_speeds."""
        ws = self._ws_jacobian
        out = []
        for key in ("x", "y"):
            rows = ws["rows"][key]
            for b in ws["blocks"][key]:
                for gi in b:
                    for gj in b:
                        out.append(rows[gi][gj])
        return out

    def emit_cpp(self, func=None, cse=True):
        """Generates a compilable C++ function computing the physical flux from the symbolic
        tree (each Expr node knows how to write itself in C++ via to_cpp).

        Produced signature : template <class Real> void <func>_flux(const Real* U, Real* F, int dir).
        Constants inlined ; each primitive becomes a local variable. cse=True (default) factors
        the common subexpressions (H, c...) into ``cseK_`` locals ; cse=False recomputes them inline.

        Step (2) of the DSL (see docs/ARCHITECTURE_CIBLE.md sect. 3) : HOST C++ (templatable on Real)."""
        name = func or self.name
        if not self._flux:
            raise ValueError("emit_cpp : call set_flux(...) first")
        if len(self._flux.get("x", [])) != self.n_vars or len(self._flux.get("y", [])) != self.n_vars:
            raise ValueError("emit_cpp : flux expected with %d components per direction" % self.n_vars)
        nc = self.n_vars
        out = [
            "// genere depuis le modele symbolique '%s' (adc.dsl.emit_cpp)" % self.name,
            "// flux physique F = flux(U, dir) ; dir 0=x, 1=y ; U et F de taille %d." % nc,
            "#include <cmath>",
            "template <class Real>",
            "inline void %s_flux(const Real* U, Real* F, int dir) {" % name,
        ]
        out += ["  const Real %s = U[%d];" % (c, i) for i, c in enumerate(self.cons_names)]
        out += ["  const Real %s = %s;" % (p, e.to_cpp()) for p, e in self.prim_defs.items()]
        tl, cpps = self._codegen_exprs(self._flux["x"] + self._flux["y"], cse, real="Real", indent="  ")
        out += tl
        out.append("  if (dir == 0) {")
        out += ["    F[%d] = %s;" % (i, cpps[i]) for i in range(nc)]
        out.append("  } else {")
        out += ["    F[%d] = %s;" % (i, cpps[nc + i]) for i in range(nc)]
        out += ["  }", "}"]
        return "\n".join(out) + "\n"

    def emit_cpp_brick(self, name=None, namespace="adc_generated", cse=True,
                       hoist_reciprocals=False):
        """Generates a C++ BRICK satisfying the adc::HyperbolicModel concept (wrapping : step
        2bis). The produced struct uses StateVec / Aux / ADC_HD / Variables and exposes flux,
        max_wave_speed, to_primitive, to_conservative, conservative_vars, primitive_vars : it can
        therefore enter a CompositeModel and run in the compiled solver.

        Requires set_primitive_state(...) (Prim layout) and set_conservative_from([...]) (to_conservative,
        which the DSL cannot invert on its own). cse=True (default) factors the common
        subexpressions (H, c...) into ``cseK_`` locals. Still to do (see ARCHITECTURE_CIBLE.md sect. 3) :
        Kokkos/CUDA codegen, JIT."""
        if not self.prim_state:
            raise ValueError("emit_cpp_brick : call set_primitive_state(...) first")
        if self.cons_from is None or len(self.cons_from) != self.n_vars:
            raise ValueError("emit_cpp_brick : set_conservative_from([...]) expected (%d expressions)"
                             % self.n_vars)
        if not self._flux:
            raise ValueError("emit_cpp_brick : call set_flux(...) first")
        if len(self._flux.get("x", [])) != self.n_vars or len(self._flux.get("y", [])) != self.n_vars:
            raise ValueError("emit_cpp_brick : flux expected with %d components per direction"
                             % self.n_vars)
        if not self._eig and self._wave_speeds is None and self._ws_jacobian is None:
            raise ValueError("emit_cpp_brick : call set_eigenvalues(...), set_wave_speeds(...) "
                             "or set_wave_speeds_from_jacobian(...) first (source of "
                             "max_wave_speed / CFL)")
        nm = name or (self.name.capitalize() + "Gen")
        nc, npr = self.n_vars, len(self.prim_state)

        def cons_locals():
            return ["    const adc::Real %s = U[%d];" % (c, i) for i, c in enumerate(self.cons_names)]

        def prim_locals(live=None):
            # FILTER on the live primitives (live) + OPT-IN hoist; without live, full output.
            return self._prim_block(live, hoist_reciprocals)

        def aux_locals():
            return self._aux_locals_lines()  # canonical (a.<n>) + named (a.extra_field(k)), ADC-70

        # Aux parameter named 'a' only if a formula reads an auxiliary field (canonical OR
        # named ; otherwise anonymous, so as not to trigger an unused-parameter warning).
        aux_param = "const Aux& a" if self._reads_aux() else "const Aux&"

        def eig_reduce(cpps, ind):
            # cpps : C++ already generated (possibly CSE) for the eigenvalues. Internal names suffixed
            # '_' : they shadow neither a user variable nor the Aux parameter 'a' (see adversarial review).
            lines = ["%sconst adc::Real lam%d_ = %s;" % (ind, k, c) for k, c in enumerate(cpps)]
            lines.append("%sadc::Real mws_ = lam0_ < 0 ? -lam0_ : lam0_;" % ind)
            for k in range(1, len(cpps)):
                lines.append("%s{ const adc::Real cand_ = lam%d_ < 0 ? -lam%d_ : lam%d_;"
                             " if (cand_ > mws_) mws_ = cand_; }" % (ind, k, k, k))
            lines.append("%sreturn mws_;" % ind)
            return lines

        def eig_minmax(cpps, ind):
            # signed wave speeds : smin = smallest, smax = largest eigenvalue (for
            # HLLC / Roe). Same internal names suffixed '_' as eig_reduce.
            lines = ["%sconst adc::Real lam%d_ = %s;" % (ind, k, c) for k, c in enumerate(cpps)]
            lines.append("%ssmin = lam0_; smax = lam0_;" % ind)
            for k in range(1, len(cpps)):
                lines.append("%sif (lam%d_ < smin) smin = lam%d_;" % (ind, k, k))
                lines.append("%sif (lam%d_ > smax) smax = lam%d_;" % (ind, k, k))
            return lines

        def ws_jac_pieces(key):
            # 'numeric' jacobian path : CSE of the NON-ZERO entries of the sub-blocks of
            # direction @p key ; the structural zeros (10 identity rows of a moment system,
            # arbitrary sparsity) are emitted as literals without going through the CSE.
            ws = self._ws_jacobian
            rows = ws["rows"][key]
            entries, zeros = [], []
            for bi, b in enumerate(ws["blocks"][key]):
                for r, gi in enumerate(b):
                    for c, gj in enumerate(b):
                        e = rows[gi][gj]
                        if isinstance(e, Const) and e.value == 0.0:
                            zeros.append((bi, r, c))
                        else:
                            entries.append((bi, r, c, e))
            tl, cpps = self._codegen_exprs([e for (_, _, _, e) in entries], cse)
            fill = {}
            for (bi, r, c, _), cpp in zip(entries, cpps):
                fill.setdefault(bi, []).append((r, c, cpp))
            for (bi, r, c) in zeros:
                fill.setdefault(bi, []).append((r, c, "adc::Real(0)"))
            return tl, fill

        def ws_jac_body(ind, lo, hi, key="x", fill=None):
            # body of the jacobian computation -> extremes (@p lo/@p hi : destination names).
            # eig='fd' : column-wise jacobian by finite differences of the COMPILED flux ;
            # eig='numeric' : fill of the sub-blocks from @p fill. @p key : direction
            # (chooses the block partition).
            ws = self._ws_jacobian
            nv = self.n_vars
            L = []
            if ws["eig"] == "fd":
                L.append("%sconst State F0_ = flux(U, a, dir);" % ind)
                L.append("%sadc::Real Jf_[%d][%d];" % (ind, nv, nv))
                L.append("%sconst adc::Real eps_ = adc::Real(1e-6) * (U[0] < 0 ? -U[0] : U[0])"
                         " + adc::Real(1e-30);" % ind)
                L.append("%sfor (int k_ = 0; k_ < %d; ++k_) {" % (ind, nv))
                L.append("%s  State Up_ = U;" % ind)
                L.append("%s  Up_[k_] += eps_;" % ind)
                L.append("%s  const State Fk_ = flux(Up_, a, dir);" % ind)
                L.append("%s  for (int i_ = 0; i_ < %d; ++i_) Jf_[i_][k_] = (Fk_[i_] - F0_[i_])"
                         " / eps_;" % (ind, nv))
                L.append("%s}" % ind)
            for bi, b in enumerate(ws["blocks"][key]):
                nb = len(b)
                L.append("%s{" % ind)
                L.append("%s  adc::Real Jb_[%d][%d];" % (ind, nb, nb))
                if ws["eig"] == "fd":
                    for r, gi in enumerate(b):
                        for c, gj in enumerate(b):
                            L.append("%s  Jb_[%d][%d] = Jf_[%d][%d];" % (ind, r, c, gi, gj))
                else:
                    for (r, c, cpp) in sorted(fill.get(bi, [])):
                        L.append("%s  Jb_[%d][%d] = %s;" % (ind, r, c, cpp))
                L.append("%s  const adc::EigBounds eb_ = adc::real_eig_minmax(Jb_);" % ind)
                if bi == 0:
                    L.append("%s  %s = eb_.lmin; %s = eb_.lmax;" % (ind, lo, hi))
                else:
                    L.append("%s  if (eb_.lmin < %s) %s = eb_.lmin;" % (ind, lo, lo))
                    L.append("%s  if (eb_.lmax > %s) %s = eb_.lmax;" % (ind, hi, hi))
                L.append("%s}" % ind)
            return L

        cnames = ", ".join('"%s"' % c for c in self.cons_names)
        pnames = ", ".join('"%s"' % p for p in self.prim_state)
        # Physical roles parallel to the names : C++ initializer of adc::VariableSet::roles. Emitted IF at
        # least one component has a recognized role (otherwise empty roles -> brick identical to history,
        # couplings fall back on the fallback indices). The roles let System
        # resolve inter-species couplings by index_of(role) instead of a literal index.
        def roles_init(roles):
            if all(r == "Custom" for r in roles):
                return None  # no useful role : we do not emit the 4th field (strict back-compat)
            return ", ".join("adc::VariableRole::%s" % r for r in roles)

        croles = roles_init(roles_for(self.cons_names, self.cons_roles))
        proles = roles_init(roles_for(self.prim_state, self.prim_roles))
        # P7-b : assign the runtime indices BEFORE any to_cpp() (a RuntimeParamRef raises otherwise).
        rt_member = self._runtime_params_member()
        S = [
            "#include <cmath>",  # std::sqrt / std::pow : self-sufficient brick (g++ does not pull cmath)
            "// brique HYPERBOLIQUE generee depuis le modele symbolique '%s' (adc.dsl.emit_cpp_brick)."
            % self.name,
            "// Satisfait adc::HyperbolicModel : flux + max_wave_speed + conversions + descripteurs.",
        ]
        if rt_member:  # RuntimeParams header only if a formula reads a runtime param
            S.append("#include <adc/runtime/runtime_params.hpp>")
        # dense_eig.hpp : eigenvalues of dense blocks (exact wave_speeds) OU temoin de VP dans la
        # projection (m.projection + dsl.eig_max_im, ADC-289). Sans l'un ou l'autre : non inclus.
        eig_pairs = _collect_eig_witnesses(self._proj or [])
        if self._ws_jacobian is not None or eig_pairs:
            S.append("#include <adc/numerics/dense_eig.hpp>")
        S += [
            "namespace %s {" % namespace,
            "struct %s {" % nm,
            "  using State = adc::StateVec<%d>;" % nc,
            "  using Prim  = adc::StateVec<%d>;" % npr,
            "  using Aux   = adc::Aux;",
            "  static constexpr int n_vars = %d;" % nc,
        ]
        if rt_member:  # member adc::RuntimeParams params{count, {defaults}} (P7-b)
            S.append(rt_member.rstrip("\n"))
        # Foncteurs nommes des temoins de VP (EigWitness) : methodes statiques ADC_HD remplissant
        # M[k][k] + real_eig_minmax, declarees une fois par couple (field, k). Device-clean (ADC-289).
        S += _eig_witness_helpers(eig_pairs)
        # n_aux if a formula (flux / eigenvalues) reads an extra aux field : canonical
        # (B_z...) OR named (aux_field -> kAuxNamedBase + k). Without an extra field -> no n_aux emitted,
        # brick strictly bit-identical to history.
        if self._total_n_aux() > AUX_BASE_COMPS:
            S.append("  static constexpr int n_aux = %d;" % self._total_n_aux())
        S += [
            "",
            "  ADC_HD State flux(const State& U, %s, int dir) const {" % aux_param,
        ]
        S += cons_locals() + prim_locals(self._live_prims(self._flux["x"] + self._flux["y"])) \
            + aux_locals()
        ftl, fcpps = self._codegen_exprs(self._flux["x"] + self._flux["y"], cse)
        S += ftl
        S.append("    State F{};")
        S.append("    if (dir == 0) {")
        S += ["      F[%d] = %s;" % (i, fcpps[i]) for i in range(nc)]
        S.append("    } else {")
        S += ["      F[%d] = %s;" % (i, fcpps[nc + i]) for i in range(nc)]
        S += ["    }", "    return F;", "  }", ""]

        # in 'fd' jacobian mode WITHOUT eigenvalues, max_wave_speed calls flux(U, a, dir) : the
        # Aux parameter must be named even if no formula reads an aux.
        jac_fd = self._ws_jacobian is not None and self._ws_jacobian["eig"] == "fd"
        mws_aux_param = "const Aux& a" if (jac_fd and not self._eig) else aux_param
        S.append("  ADC_HD adc::Real max_wave_speed(const State& U, %s, int dir) const {"
                 % mws_aux_param)
        if self._eig:
            mws_drv = self._eig["x"] + self._eig["y"]
        elif self._wave_speeds is not None:
            mws_drv = list(self._wave_speeds["x"]) + list(self._wave_speeds["y"])
        elif self._ws_jacobian["eig"] == "fd":
            mws_drv = []  # fd path: max_wave_speed calls flux(), no direct primitive
        else:
            mws_drv = self._jac_entries()
        S += cons_locals() + prim_locals(self._live_prims(mws_drv)) + aux_locals()
        if self._eig:
            # historical source : max(|eigenvalues|), bit-identical.
            nx = len(self._eig["x"])
            etl, ecpps = self._codegen_exprs(self._eig["x"] + self._eig["y"], cse)
            S += etl
            S.append("    if (dir == 0) {")
            S += eig_reduce(ecpps[:nx], "      ")
            S.append("    } else {")
            S += eig_reduce(ecpps[nx:], "      ")
            S += ["    }", "  }", ""]
        elif self._wave_speeds is not None:
            # WITHOUT eigenvalues : Rusanov / CFL bound derived from the explicit SIGNED wave speeds,
            # max(|smin|, |smax|) -- the pair bounds the spectrum by set_wave_speeds contract.
            ws = self._wave_speeds
            wtl, wcpps = self._codegen_exprs(list(ws["x"]) + list(ws["y"]), cse)
            S += wtl
            S.append("    if (dir == 0) {")
            S += eig_reduce(wcpps[:2], "      ")
            S.append("    } else {")
            S += eig_reduce(wcpps[2:], "      ")
            S += ["    }", "  }", ""]
        else:
            # WITHOUT eigenvalues : Rusanov / CFL bound = max(|smin|, |smax|) of the jacobian
            # spectrum extremes (same blocks as wave_speeds : Rusanov and HLL share the
            # same truth).
            S.append("    adc::Real lo_ = adc::Real(0), hi_ = adc::Real(0);")
            jac_same_blocks = self._ws_jacobian["blocks"]["x"] == self._ws_jacobian["blocks"]["y"]
            if self._ws_jacobian["eig"] == "fd" and jac_same_blocks:
                S += ws_jac_body("    ", "lo_", "hi_")
            elif self._ws_jacobian["eig"] == "fd":
                S.append("    if (dir == 0) {")
                S += ws_jac_body("      ", "lo_", "hi_", "x")
                S.append("    } else {")
                S += ws_jac_body("      ", "lo_", "hi_", "y")
                S.append("    }")
            else:
                ptx, pty = ws_jac_pieces("x"), ws_jac_pieces("y")
                S.append("    if (dir == 0) {")
                S += ptx[0]
                S += ws_jac_body("      ", "lo_", "hi_", "x", ptx[1])
                S.append("    } else {")
                S += pty[0]
                S += ws_jac_body("      ", "lo_", "hi_", "y", pty[1])
                S.append("    }")
            S.append("    const adc::Real alo_ = lo_ < 0 ? -lo_ : lo_;")
            S.append("    const adc::Real ahi_ = hi_ < 0 ? -hi_ : hi_;")
            S += ["    return alo_ > ahi_ ? alo_ : ahi_;", "  }", ""]

        # pressure : emitted IF a primitive 'p' (pressure) is declared (compressible convention) ;
        # required by the canonical HLLC / Roe fluxes (make_block : requires { m.pressure(s); }).
        if "p" in self.prim_defs:
            S.append("  ADC_HD adc::Real pressure(const State& U) const {")
            S += cons_locals() + prim_locals(self._live_prims([], seed=["p"]))
            S += ["    return p;", "  }", ""]

        # SIGNED wave speeds wave_speeds(U, aux, dir, smin, smax) : HLL gate of the core
        # (block_builder.hpp requires { m.wave_speeds(...) }). Two sources, by priority :
        #   1. EXPLICIT pair set_wave_speeds (smin, smax per direction) -- INDEPENDENT of 'p' :
        #      a model without pressure (moments, isothermal...) gets access to riemann='hll' ;
        #   2. historical : min/max of the eigenvalues, emitted ONLY if 'p' is declared
        #      (compressible HLLC / Roe convention, bit-identical to the existing one).
        # Without either of the two (e.g. ExB scalar transport) : nothing emitted, Rusanov alone, unchanged.
        if self._wave_speeds is not None:
            ws = self._wave_speeds
            S.append("  ADC_HD void wave_speeds(const State& U, %s, int dir, adc::Real& smin, "
                     "adc::Real& smax) const {" % aux_param)
            S += cons_locals() \
                + prim_locals(self._live_prims(list(ws["x"]) + list(ws["y"]))) + aux_locals()
            wtl, wcpps = self._codegen_exprs(list(ws["x"]) + list(ws["y"]), cse)
            S += wtl
            S.append("    if (dir == 0) {")
            S.append("      smin = %s; smax = %s;" % (wcpps[0], wcpps[1]))
            S.append("    } else {")
            S.append("      smin = %s; smax = %s;" % (wcpps[2], wcpps[3]))
            S += ["    }", "  }", ""]
        elif self._ws_jacobian is not None:
            # EXACT speeds via jacobian eigenvalues (see set_wave_speeds_from_jacobian :
            # 'numeric' = entries as formulas, 'fd' = columns by finite differences of the compiled
            # flux ; extremes per sub-block via adc::real_eig_minmax, safe Gershgorin fallback).
            ws_aux = aux_param if self._ws_jacobian["eig"] != "fd" else "const Aux& a"
            S.append("  ADC_HD void wave_speeds(const State& U, %s, int dir, adc::Real& smin, "
                     "adc::Real& smax) const {" % ws_aux)
            ws_drv = [] if self._ws_jacobian["eig"] == "fd" else self._jac_entries()
            S += cons_locals() + prim_locals(self._live_prims(ws_drv)) + aux_locals()
            ws_same_blocks = self._ws_jacobian["blocks"]["x"] == self._ws_jacobian["blocks"]["y"]
            if self._ws_jacobian["eig"] == "fd" and ws_same_blocks:
                S += ws_jac_body("    ", "smin", "smax")
            elif self._ws_jacobian["eig"] == "fd":
                S.append("    if (dir == 0) {")
                S += ws_jac_body("      ", "smin", "smax", "x")
                S.append("    } else {")
                S += ws_jac_body("      ", "smin", "smax", "y")
                S.append("    }")
            else:
                ptx, pty = ws_jac_pieces("x"), ws_jac_pieces("y")
                S.append("    if (dir == 0) {")
                S += ptx[0]
                S += ws_jac_body("      ", "smin", "smax", "x", ptx[1])
                S.append("    } else {")
                S += pty[0]
                S += ws_jac_body("      ", "smin", "smax", "y", pty[1])
                S.append("    }")
            S += ["  }", ""]
        elif "p" in self.prim_defs:
            nx = len(self._eig["x"])
            S.append("  ADC_HD void wave_speeds(const State& U, %s, int dir, adc::Real& smin, "
                     "adc::Real& smax) const {" % aux_param)
            S += cons_locals() \
                + prim_locals(self._live_prims(self._eig["x"] + self._eig["y"])) + aux_locals()
            wtl, wcpps = self._codegen_exprs(self._eig["x"] + self._eig["y"], cse)
            S += wtl
            S.append("    if (dir == 0) {")
            S += eig_minmax(wcpps[:nx], "      ")
            S.append("    } else {")
            S += eig_minmax(wcpps[nx:], "      ")
            S += ["    }", "  }", ""]

        # CAPABILITY HLLC (m.enable_hllc, audit wave 3): contact_speed (Toro) + hllc_star_state
        # GENERATED from the block ROLES (no literal index: Density/MomentumX/MomentumY
        # resolved, Energy optional, any other component advected passively Us[c]=fac*U[c]/r).
        # The core (HasHLLCStructure) then applies the generic HLLC algorithm -- including on a
        # NON Euler model (isothermal 3-var, moments + passive scalars). Without a call: nothing emitted.
        if self._hllc:
            roles_l = roles_for(self.cons_names, self.cons_roles)
            if "p" not in self.prim_defs:
                raise ValueError("enable_hllc: the primitive 'p' (pressure) must be declared "
                                 "(m.primitive('p', ...)) -- contact_speed/star state depend on it")
            try:
                iD = roles_l.index("Density")
                iX = roles_l.index("MomentumX")
                iY = roles_l.index("MomentumY")
            except ValueError:
                raise ValueError("enable_hllc: roles Density / MomentumX / MomentumY required "
                                 "(declare conservative_vars(..., roles=[...])); current roles %r"
                                 % (roles_l,))
            iE = roles_l.index("Energy") if "Energy" in roles_l else -1
            S.append("  // CAPABILITY HLLC generee depuis les ROLES (enable_hllc) : algorithme")
            S.append("  // contact-resolving generique du coeur (HasHLLCStructure), aucun layout fige.")
            S.append("  ADC_HD adc::Real contact_speed(const State& UL, const State& UR, "
                     "adc::Real pL, adc::Real pR, adc::Real sL, adc::Real sR, int dir) const {")
            S.append("    const int in_ = dir == 0 ? %d : %d;" % (iX, iY))
            S.append("    const adc::Real rL = UL[%d], rR = UR[%d];" % (iD, iD))
            S.append("    const adc::Real unL = UL[in_] / rL, unR = UR[in_] / rR;")
            S.append("    return (pR - pL + rL * unL * (sL - unL) - rR * unR * (sR - unR)) /")
            S.append("           (rL * (sL - unL) - rR * (sR - unR));")
            S += ["  }", ""]
            S.append("  ADC_HD State hllc_star_state(const State& U, adc::Real p, adc::Real s, "
                     "adc::Real sStar, int dir) const {")
            S.append("    const int in_ = dir == 0 ? %d : %d;" % (iX, iY))
            S.append("    const adc::Real r = U[%d];" % iD)
            S.append("    const adc::Real un = U[in_] / r;")
            S.append("    const adc::Real fac = r * (s - un) / (s - sStar);")
            S.append("    State Us{};")
            S.append("    for (int c = 0; c < %d; ++c) Us[c] = fac * (U[c] / r);  "
                     "// defaut : advection passive" % nc)
            S.append("    Us[%d] = fac;" % iD)
            S.append("    Us[in_] = fac * sStar;")
            if iE >= 0:
                S.append("    Us[%d] = fac * (U[%d] / r + (sStar - un) * (sStar + p / "
                         "(r * (s - un))));" % (iE, iE))
            S += ["    return Us;", "  }", ""]

        # CAPABILITY ROE (m.enable_roe, audit balance): roe_dissipation = |A_roe| (UR - UL)
        # GENERATED from the ROLES. With Energy: exact TRANSCRIPTION of the core canonical Euler
        # algebra (numerical_flux.hpp), gamma-1 deduced from p/(E - 1/2 rho |v|^2) -- numerical
        # parity expected with the historical path. Without Energy: same decomposition without the
        # E line, c = sqrt(p/rho) per side then Roe average (standard generalization). The
        # components OUTSIDE the fluid roles are passive scalars carried by the entropy wave
        # (tangential line, phi = q/rho). The core (HasRoeDissipation) does F = 1/2(FL+FR) - d/2.
        if self._roe:
            roles_l = roles_for(self.cons_names, self.cons_roles)
            if "p" not in self.prim_defs:
                raise ValueError("enable_roe: the primitive 'p' (pressure) must be declared "
                                 "(m.primitive('p', ...)) -- the Roe linearization depends on it")
            try:
                iD = roles_l.index("Density")
                iX = roles_l.index("MomentumX")
                iY = roles_l.index("MomentumY")
            except ValueError:
                raise ValueError("enable_roe: roles Density / MomentumX / MomentumY required "
                                 "(declare conservative_vars(..., roles=[...])); current roles %r"
                                 % (roles_l,))
            iE = roles_l.index("Energy") if "Energy" in roles_l else -1
            passives = [c for c in range(nc) if c not in (iD, iX, iY, iE)]
            S.append("  // CAPABILITY ROE generee depuis les ROLES (enable_roe) : dissipation")
            S.append("  // |A_roe| dU du coeur generique (HasRoeDissipation), aucun layout fige.")
            S.append("  ADC_HD State roe_dissipation(const State& UL, const adc::Aux&, "
                     "const State& UR, const adc::Aux&, int dir) const {")
            S.append("    const int in_ = dir == 0 ? %d : %d;" % (iX, iY))
            S.append("    const int it_ = dir == 0 ? %d : %d;" % (iY, iX))
            S.append("    const adc::Real rL = UL[%d], rR = UR[%d];" % (iD, iD))
            S.append("    const adc::Real unL = UL[in_] / rL, unR = UR[in_] / rR;")
            S.append("    const adc::Real utL = UL[it_] / rL, utR = UR[it_] / rR;")
            S.append("    const adc::Real pL = pressure(UL), pR = pressure(UR);")
            S.append("    const adc::Real sqL = std::sqrt(rL), sqR = std::sqrt(rR), "
                     "den = sqL + sqR;")
            S.append("    const adc::Real un = (sqL * unL + sqR * unR) / den;")
            S.append("    const adc::Real ut = (sqL * utL + sqR * utR) / den;")
            S.append("    const adc::Real rho = sqL * sqR;")
            if iE >= 0:
                S.append("    // gaz parfait : H de Roe + gamma-1 deduit (algebre canonique du coeur)")
                S.append("    const adc::Real HL = (UL[%d] + pL) / rL, HR = (UR[%d] + pR) / rR;"
                         % (iE, iE))
                S.append("    const adc::Real H = (sqL * HL + sqR * HR) / den;")
                S.append("    const adc::Real q2 = un * un + ut * ut;")
                S.append("    const adc::Real gm1 = pL / (UL[%d] - adc::Real(0.5) * rL * "
                         "(unL * unL + utL * utL));" % iE)
                S.append("    const adc::Real c2 = gm1 * (H - adc::Real(0.5) * q2);")
                S.append("    const adc::Real c = std::sqrt(c2);")
            else:
                S.append("    // sans Energy : c LOCAL = sqrt(p/rho) par cote, moyenne de Roe")
                S.append("    const adc::Real c = (sqL * std::sqrt(pL / rL) + sqR * "
                         "std::sqrt(pR / rR)) / den;")
                S.append("    const adc::Real c2 = c * c;")
            S.append("    const adc::Real dr = rR - rL, dp = pR - pL, dun = unR - unL, "
                     "dut = utR - utL;")
            S.append("    const adc::Real a1 = (dp - rho * c * dun) / (adc::Real(2) * c2);")
            S.append("    const adc::Real a2 = dr - dp / c2;")
            S.append("    const adc::Real a3 = rho * dut;")
            S.append("    const adc::Real a5 = (dp + rho * c * dun) / (adc::Real(2) * c2);")
            S.append("    // correction d'entropie de Harten, MEME politique que le chemin")
            S.append("    // canonique : eps = adc::kRoeEntropyFixFraction * c (0.1, Euler/Roe).")
            S.append("    const adc::Real eps = adc::Real(0.1) * c;")
            S.append("    const adc::Real l1r = un - c, l5r = un + c;")
            S.append("    const adc::Real al1 = (l1r < 0 ? -l1r : l1r) < eps ? "
                     "adc::Real(0.5) * (l1r * l1r / eps + eps) : (l1r < 0 ? -l1r : l1r);")
            S.append("    const adc::Real al2 = un < 0 ? -un : un;")
            S.append("    const adc::Real al5 = (l5r < 0 ? -l5r : l5r) < eps ? "
                     "adc::Real(0.5) * (l5r * l5r / eps + eps) : (l5r < 0 ? -l5r : l5r);")
            S.append("    State d{};")
            S.append("    d[%d] = al1 * a1 + al2 * a2 + al5 * a5;" % iD)
            S.append("    d[in_] = al1 * a1 * (un - c) + al2 * a2 * un + al5 * a5 * (un + c);")
            S.append("    d[it_] = al1 * a1 * ut + al2 * (a2 * ut + a3) + al5 * a5 * ut;")
            if iE >= 0:
                S.append("    d[%d] = al1 * a1 * (H - un * c) + al2 * (a2 * adc::Real(0.5) * q2 "
                         "+ a3 * ut) + al5 * a5 * (H + un * c);" % iE)
            for cpa in passives:
                S.append("    {  // scalaire passif [%d] : porte par l'onde entropique (phi = q/rho)"
                         % cpa)
                S.append("      const adc::Real fL = UL[%d] / rL, fR = UR[%d] / rR;" % (cpa, cpa))
                S.append("      const adc::Real ft = (sqL * fL + sqR * fR) / den;")
                S.append("      d[%d] = al1 * a1 * ft + al2 * (a2 * ft + rho * (fR - fL)) "
                         "+ al5 * a5 * ft;" % cpa)
                S.append("    }")
            S += ["    return d;", "  }", ""]

        # CAPABILITY ROE PROVIDED (m.roe_dissipation): 'user' counterpart of enable_roe. The d_i
        # rows come from the user (their eigenstructure), written with left()/right() of both states.
        # We emit the SAME hook roe_dissipation(UL, AL, UR, AR, dir) as the roles path (trait
        # HasRoeDissipation, the core does F = 1/2(FL+FR) - 1/2 d). left(e) -> e on the L_ locals
        # (computed from UL), right(e) -> R_ locals (from UR). _roe_rows and _roe are exclusive
        # (guard at declaration and in check()).
        if self._roe_rows is not None:
            has_aux = bool(self.aux_names)  # Aux parameters named aL/aR only if some aux exist
            aL = "const adc::Aux& aL" if has_aux else "const adc::Aux&"
            aR = "const adc::Aux& aR" if has_aux else "const adc::Aux&"
            S.append("  // CAPABILITY ROE FOURNIE (m.roe_dissipation) : dissipation d ecrite par")
            S.append("  // l'utilisateur via left()/right() des deux etats ; hook HasRoeDissipation.")
            S.append("  ADC_HD State roe_dissipation(const State& UL, %s, const State& UR, %s, "
                     "int dir) const {" % (aL, aR))
            # locals of BOTH states: conservatives, primitives (def with prefix), then aux read.
            for side, U, av in (("L_", "UL", "aL"), ("R_", "UR", "aR")):
                S += ["    const adc::Real %s%s = %s[%d];" % (side, c, U, i)
                      for i, c in enumerate(self.cons_names)]
                S += ["    const adc::Real %s%s = %s;" % (side, p, _cpp_roe(e, side))
                      for p, e in self.prim_defs.items()]
                if has_aux:
                    S += ["    const adc::Real %s%s = %s.%s;" % (side, n, av, n)
                          for n in self.aux_names]
            S.append("    State d{};")
            S.append("    if (dir == 0) {")
            S += ["      d[%d] = %s;" % (i, _cpp_roe(self._roe_rows["x"][i], None)) for i in range(nc)]
            S.append("    } else {")
            S += ["      d[%d] = %s;" % (i, _cpp_roe(self._roe_rows["y"][i], None)) for i in range(nc)]
            S += ["    }", "    return d;", "  }", ""]

        # OPTIONAL step bounds (m.stability_speed / m.stability_dt): emitted like the C++
        # traits HasStabilitySpeed / HasStabilityDt (cf. adc/core/physical_model.hpp). A single
        # expression (isotropic): dir is ignored. WITHOUT a call, nothing emitted -> strict fallback
        # max_wave_speed (historical step policy).
        if self._stab_speed is not None:
            S.append("  ADC_HD adc::Real stability_speed(const State& U, %s, int dir) const {"
                     % aux_param)
            S.append("    (void)dir;  // borne isotrope : une seule expression pour les deux directions")
            S += cons_locals() + prim_locals(self._live_prims([self._stab_speed])) + aux_locals()
            stl, scpps = self._codegen_exprs([self._stab_speed], cse)
            S += stl
            S += ["    return %s;" % scpps[0], "  }", ""]
        if self._stab_dt is not None:
            S.append("  ADC_HD adc::Real stability_dt(const State& U, %s) const {" % aux_param)
            S += cons_locals() + prim_locals(self._live_prims([self._stab_dt])) + aux_locals()
            dtl, dcpps = self._codegen_exprs([self._stab_dt], cse)
            S += dtl
            S += ["    return %s;" % dcpps[0], "  }", ""]

        # PROJECTION PONCTUELLE post-pas (m.projection, ADC-177) : emise comme le trait C++
        # HasPointwiseProjection (project(U, aux) -> State), appliquee par le stepper a la FIN de
        # chaque macro-pas entier. SANS appel, rien d'emis -> aucun hook (chemin bit-identique).
        if self._proj is not None:
            S.append("  ADC_HD State project(const State& U, %s) const {" % aux_param)
            S += cons_locals() + prim_locals() + aux_locals()
            ptl, pcpps = self._codegen_exprs(self._proj, cse)
            S += ptl
            S.append("    State Up{};")
            S += ["    Up[%d] = %s;" % (i, c) for i, c in enumerate(pcpps)]
            S += ["    return Up;", "  }", ""]

        S.append("  ADC_HD Prim to_primitive(const State& U) const {")
        S += cons_locals() + prim_locals(self._live_prims([], seed=self.prim_state))
        S.append("    Prim P{};")
        S += ["    P[%d] = %s;" % (i, p) for i, p in enumerate(self.prim_state)]
        S += ["    return P;", "  }", ""]

        S.append("  ADC_HD State to_conservative(const Prim& P) const {")
        S += ["    const adc::Real %s = P[%d];" % (p, i) for i, p in enumerate(self.prim_state)]
        ctl, ccpps = self._codegen_exprs(self.cons_from, cse)
        S += ctl
        S.append("    State U{};")
        S += ["    U[%d] = %s;" % (i, c) for i, c in enumerate(ccpps)]
        S += ["    return U;", "  }", ""]

        cons_set = "{adc::VariableKind::Conservative, {%s}, %d%s}" % (
            cnames, nc, (", {%s}" % croles) if croles is not None else "")
        prim_set = "{adc::VariableKind::Primitive, {%s}, %d%s}" % (
            pnames, npr, (", {%s}" % proles) if proles is not None else "")
        S.append('  static adc::VariableSet conservative_vars() { return %s; }' % cons_set)
        S.append('  static adc::VariableSet primitive_vars() { return %s; }' % prim_set)
        S += ["};", "}  // namespace %s" % namespace]
        return "\n".join(S) + "\n"

    def emit_cpp_source(self, name=None, namespace="adc_generated", cse=True,
                        hoist_reciprocals=False):
        """Generate a composable C++ SOURCE BRICK (in the adc sense) from self._source.

        The produced struct exposes apply(U, a) returning the source term S(U, aux), with one line per
        conservative component (S[i] = self._source[i].to_cpp()). It has the same form as the source
        bricks written by hand (NoSource, PotentialForce in adc/model/bricks.hpp) and can therefore
        enter as the Source parameter of a CompositeModel.

        CONVENTION: the auxiliary names (set via aux(...)) must be FIELDS of adc::Aux,
        because they are read directly as a.<name> (e.g. aux('grad_x') -> a.grad_x, aux('grad_y') ->
        a.grad_y). This convention is the same as that of the manual bricks, where the source reads
        the outer state only through the adc::Aux channel (potential and its gradient).

        Style identical to emit_cpp_brick (inlined constants, cons -> locals, primitives -> locals;
        plus, aux -> locals); cse=True factors the common sub-expressions. Raises ValueError if
        set_source(...) has not been called."""
        if self._source is None:
            raise ValueError("emit_cpp_source: call set_source([...]) first")
        nm = name or (self.name.capitalize() + "Source")
        nc = self.n_vars

        def cons_locals():
            return ["    const adc::Real %s = U[%d];" % (c, i) for i, c in enumerate(self.cons_names)]

        def prim_locals(live=None):
            # FILTER on the live primitives (live) + OPT-IN hoist; without live, full output.
            return self._prim_block(live, hoist_reciprocals)

        def aux_locals():
            return self._aux_locals_lines()  # canonical (a.<n>) + named (a.extra_field(k)), ADC-70

        na = self._total_n_aux()  # required aux width (B_z / T_e / named fields -> > 3)
        rt_member = self._runtime_params_member()  # P7-b: runtime indices BEFORE any to_cpp()
        S = [
            "#include <cmath>",  # self-sufficient for std::sqrt / std::pow
            "// brique de SOURCE generee depuis le modele symbolique '%s' (adc.dsl.emit_cpp_source)."
            % self.name,
            "// apply(U, a) -> terme source S(U, aux) ; noms aux = champs de adc::Aux (grad_x, grad_y).",
        ]
        if rt_member:  # RuntimeParams header only if a formula reads a runtime param
            S.append("#include <adc/runtime/runtime_params.hpp>")
        if self._ws_jacobian is not None:  # dense-block eigenvalues (exact wave_speeds)
            S.append("#include <adc/numerics/dense_eig.hpp>")
        S += [
            "namespace %s {" % namespace,
            "struct %s {" % nm,
        ]
        if rt_member:  # adc::RuntimeParams params{count, {defaults}} member (P7-b)
            S.append(rt_member.rstrip("\n"))
        # If a formula reads an EXTRA aux field (B_z...), declare n_aux: CompositeModel
        # propagates it (max over the bricks) and the system sizes/populates the shared aux channel.
        # Without an extra field -> no n_aux emitted -> brick strictly identical to the historical one.
        if na > AUX_BASE_COMPS:
            S.append("  static constexpr int n_aux = %d;" % na)
        S.append("  ADC_HD adc::StateVec<%d> apply(const adc::StateVec<%d>& U, const adc::Aux& a) const {"
                 % (nc, nc))
        src_exprs = [_wrap(e) for e in self._source]
        S += cons_locals() + prim_locals(self._live_prims(src_exprs)) + aux_locals()
        # _wrap: a component may be a Python literal (e.g. 0.0), promoted to Const.
        stl, scpps = self._codegen_exprs(src_exprs, cse)
        S += stl
        S.append("    adc::StateVec<%d> S{};" % nc)
        S += ["    S[%d] = %s;" % (i, c) for i, c in enumerate(scpps)]
        S += ["    return S;", "  }"]
        # Source FREQUENCY (m.source_frequency, audit wave 2): emitted as frequency(U, a)
        # -- the OPTIONAL contract of the source bricks (cf. physics/source.hpp), forwarded by
        # CompositeModel (HasSourceFrequency) and aggregated by step_cfl. Without a call: nothing emitted.
        if self._src_freq is not None:
            S.append("")
            S.append("  ADC_HD adc::Real frequency(const adc::StateVec<%d>& U, const adc::Aux& a) "
                     "const {" % nc)
            S += cons_locals() + prim_locals(self._live_prims([self._src_freq])) + aux_locals()
            ftl, fcpps = self._codegen_exprs([self._src_freq], cse)
            S += ftl
            S += ["    return %s;" % fcpps[0], "  }"]
        # ANALYTIC JACOBIAN (m.source_jacobian, audit wave 3): emitted as jacobian(U, a, J),
        # forwarded by CompositeModel (HasSourceJacobian) -> the implicit Newton replaces the
        # finite differences. Without a call: nothing emitted (historical FD, bit-identical).
        if self._src_jac is not None:
            if len(self._src_jac) != nc or any(len(r) != nc for r in self._src_jac):
                raise ValueError("source_jacobian: expected %dx%d matrix (dS_r/dU_c)" % (nc, nc))
            S.append("")
            S.append("  ADC_HD void jacobian(const adc::StateVec<%d>& U, const adc::Aux& a, "
                     "adc::Real (&J)[%d][%d]) const {" % (nc, nc, nc))
            flat = [e for row in self._src_jac for e in row]
            S += cons_locals() + prim_locals(self._live_prims(flat)) + aux_locals()
            jtl, jcpps = self._codegen_exprs(flat, cse)
            S += jtl
            for r in range(nc):
                for c in range(nc):
                    S.append("    J[%d][%d] = %s;" % (r, c, jcpps[r * nc + c]))
            S += ["  }"]
        S += ["};", "}  // namespace %s" % namespace]
        return "\n".join(S) + "\n"

    def _emit_bricks(self, name=None, hoist_reciprocals=False):
        """Generate the bricks (hyperbolic + source + elliptic) and the CompositeModel<...> type
        shared by BOTH backends (JIT IModel and AOT). Source / elliptic OPTIONAL: without
        set_source -> adc::NoSource; without set_elliptic_rhs -> zero rhs (no Poisson coupling).
        @p hoist_reciprocals: codegen option propagated to the bricks (cf. emit_cpp_brick).
        Returns (nv, bricks_code, composite_type)."""
        nm = name or (self.name.capitalize() + "Gen")
        nv = self.n_vars
        # CODEGEN guard (not only check(), which compile() does not call): a source
        # frequency or jacobian without m.source(...) would be silently PURGED by the
        # NoSource branch below -- explicit rejection (rule: never an ignored option).
        if self._source is None:
            if self._src_freq is not None:
                raise ValueError("source_frequency(...) declared without source: call "
                                 "m.source([...]) (the frequency is emitted on the source brick)")
            if self._src_jac is not None:
                raise ValueError("source_jacobian(...) declared without source: call "
                                 "m.source([...]) (the jacobian is emitted on the source brick)")
        parts = [self.emit_cpp_brick(name=nm + "Hyp", hoist_reciprocals=hoist_reciprocals)]
        if self._source is not None:  # source brick generated, otherwise NoSource
            parts.append(self.emit_cpp_source(name=nm + "Src", hoist_reciprocals=hoist_reciprocals))
            src_type = "adc_generated::%sSrc" % nm
        else:
            src_type = "adc::NoSource"
        if self._elliptic is not None:  # elliptic brick generated, otherwise zero rhs (no coupling)
            parts.append(self.emit_cpp_elliptic(name=nm + "Ell", hoist_reciprocals=hoist_reciprocals))
        else:
            parts.append(
                "namespace adc_generated { struct %sEll {\n"
                "  template <class State> ADC_HD adc::Real rhs(const State&) const { return adc::Real(0); }\n"
                "}; }\n" % nm)
        composite = ("adc::CompositeModel<adc_generated::%sHyp, %s, adc_generated::%sEll>"
                     % (nm, src_type, nm))
        return nv, "".join(parts), composite

    def _emit_metadata(self, model_alias):
        """OPTIONAL metadata symbols of the .so block, read by dlsym on the System side. SHARED by both
        backends (JIT and AOT). The NAMES + ROLES are always emitted (ADC_EXPORT_BLOCK_METADATA):
        they come from the model's VariableSet (single source of truth), the System reads them instead of
        the u0.. fallback / no roles. The GAMMA is emitted (ADC_EXPORT_BLOCK_GAMMA) only if set_gamma(...)
        has been called; otherwise no gamma symbol -> the System keeps its default 1.4 (backward-compat).

        @p model_alias must be an alias WITHOUT a top-level comma (the preprocessor splits
        macro arguments on commas): callers pass a `using ... = CompositeModel<...>`."""
        out = "\nADC_EXPORT_BLOCK_METADATA(%s)\n" % model_alias
        if self.gamma is not None:
            out += "ADC_EXPORT_BLOCK_GAMMA(%r)\n" % self.gamma
        # Table of NAMED aux names (aux_field, ADC-70), ordered CSV (order = AUX_NAMED_BASE +
        # k index). OPTIONAL symbol, names/roles pattern: makes the .so SELF-DESCRIBING (a C++ loader
        # could resolve name -> component; on the Python side the table already lives in CompiledModel).
        # Emitted ONLY if the model declares named fields -> backward-compatible (.so without a named
        # field unchanged, symbol absent).
        if self.aux_extra_names:
            # Names = valid C++ identifiers (validated in aux_field) -> CSV without quotes, safe C
            # literal (only [A-Za-z0-9_,]).
            out += ('extern "C" const char* adc_compiled_aux_extra_names() { return "%s"; }\n'
                    % ",".join(self.aux_extra_names))
        return out

    def emit_cpp_so_source(self, name=None, hoist_reciprocals=False):
        """Source of the JIT library (backend "jit"): the FULL MODEL as CompositeModel<GenHyp,
        GenSrc, GenEll> behind an extern "C" factory (adc_model_nvars / adc_make_model /
        adc_destroy_model via adc::ModelAdapter). This is what compile_so compiles and what
        System.add_dynamic_block loads as a coupled block with VIRTUAL DISPATCH (host prototyping)."""
        # PROJECTION ponctuelle (ADC-177) : le chemin JIT (IModel, dispatch virtuel) ne la transporte
        # pas -- elle serait IGNOREE en silence (le bloc dynamique n'a pas de hook post-pas). Rejet
        # explicite (regle : jamais d'option ignoree) ; backends 'aot' / 'production' la portent.
        if self._proj is not None:
            raise ValueError("backend 'prototype' (JIT, IModel) : projection ponctuelle "
                             "(m.projection) non transportee par ce chemin ; utiliser "
                             "backend='aot' ou 'production'")
        nv, bricks, composite = self._emit_bricks(name, hoist_reciprocals=hoist_reciprocals)
        return ('#include <adc/runtime/dynamic_model.hpp>\n'
                '#include <adc/physics/bricks.hpp>\n'  # CompositeModel + NoSource + bricks
                '#include <adc/core/variables.hpp>\n'
                + bricks
                + '\nnamespace adc_generated { using JitModel = %s; }\n' % composite  # comma-free alias (metadata macro)
                + 'extern "C" int adc_model_nvars() { return %d; }\n' % nv
                + 'extern "C" void* adc_make_model() { return new adc::ModelAdapter<adc_generated::JitModel>(); }\n'
                + 'extern "C" void adc_destroy_model(void* p) { delete static_cast<adc::IModel<%d>*>(p); }\n' % nv
                + self._emit_metadata("adc_generated::JitModel"))

    def compile_so(self, so_path, include=None, name=None, cxx=None, std="c++20",
                   hoist_reciprocals=False):
        """JIT: generate the FULL MODEL (emit_cpp_so_source) and compile a shared library
        loadable by System.add_dynamic_block (dlopen). The .so exposes a CompositeModel<hyperbolic,
        source, elliptic>: the dynamic block applies the flux AND the source, and contributes to the
        system Poisson via elliptic_rhs (a real coupled block, no longer just transport). include = adc
        headers directory (None -> auto-detected via adc_include()); cxx = compiler (default
        c++/g++/clang++). Returns so_path. Requires set_primitive_state(...) and
        set_conservative_from([...]) (like emit_cpp_brick)."""
        import os
        import shutil
        import subprocess
        import tempfile

        if include is None:
            include = adc_include()
        src = self.emit_cpp_so_source(name=name, hoist_reciprocals=hoist_reciprocals)
        cc = _default_cxx(cxx)
        if not cc:
            raise RuntimeError("compile_so: no C++ compiler found")
        std = _probe_cxx_std(cc, std)  # ACTIONABLE error if the std is not supported (vs raw error)
        with tempfile.TemporaryDirectory() as tmp:
            cpp = os.path.join(tmp, "model.cpp")
            with open(cpp, "w") as f:
                f.write(src)
            _run_compile([cc, "-shared", "-fPIC", "-std=" + std, "-O2", "-I", include, cpp,
                          "-o", so_path], "backend jit, compile_so")
        return so_path

    def emit_cpp_aot_source(self, name=None, hoist_reciprocals=False):
        """Source of the AOT library (backend "compile"): the FULL MODEL as CompositeModel<...>
        behind the extern "C" ABI of compiled_block_abi.hpp. The .so RUNS the PRODUCTION path
        (assemble_rhs<Limiter, Flux>, the core's SSPRK2/IMEX) on the generated model: inlined numerics,
        identical to a native add_block block. As opposed to the "jit" backend (IModel, virtual dispatch)."""
        nv, bricks, composite = self._emit_bricks(name, hoist_reciprocals=hoist_reciprocals)
        return ('#include <adc/runtime/compiled_block_abi.hpp>\n'
                '#include <adc/physics/bricks.hpp>\n'  # CompositeModel + NoSource + bricks
                '#include <adc/core/variables.hpp>\n'
                + bricks
                + '\nnamespace adc_generated { using AotModel = %s; }\n' % composite
                + 'ADC_DEFINE_COMPILED_BLOCK(adc_generated::AotModel)\n'
                + self._emit_metadata("adc_generated::AotModel"))  # comma-free alias (metadata macro)

    def compile_aot(self, so_path, include=None, name=None, cxx=None, std="c++20",
                    hoist_reciprocals=False):
        """Backend "compile" (AOT): generate the FULL MODEL (emit_cpp_aot_source) and compile a .so
        loadable by System.add_compiled_block. Unlike the "jit" backend (compile_so: IModel,
        virtual dispatch, host Rusanov), the block here runs the PRODUCTION path (HLLC/Roe flux at
        will, order 2, SSPRK2/IMEX) on the generated model -- numerics identical to a native block.
        include = adc headers directory (None -> auto-detected via adc_include()); cxx = compiler.
        Returns so_path.

        KOKKOS-ONLY: the AOT model includes the adc headers (multifab/for_each), which do NOT compile
        without ADC_HAS_KOKKOS. So we compile the .so WITH Kokkos (same flags as the native loader), which
        also aligns its ABI with the _adc module (also Kokkos). An installed Kokkos must be visible
        via ADC_KOKKOS_ROOT / Kokkos_ROOT (Serial is enough on CPU)."""
        import os
        import subprocess
        import tempfile

        if include is None:
            include = adc_include()
        src = self.emit_cpp_aot_source(name=name, hoist_reciprocals=hoist_reciprocals)
        if _native_kokkos_root() is None:
            raise RuntimeError(
                "compile_aot: adc_cpp is Kokkos-only -- the AOT model includes the adc headers which "
                "require Kokkos. Point at an installed Kokkos via ADC_KOKKOS_ROOT (or Kokkos_ROOT), e.g. "
                "`export ADC_KOKKOS_ROOT=/path/to/kokkos` (Serial is enough on CPU).")
        cc = _native_kokkos_compiler(cxx)
        if not cc:
            raise RuntimeError("compile_aot: no C++ compiler found")
        std = _probe_cxx_std(cc, std)  # ACTIONABLE error if the std is not supported (vs raw error)
        kokkos_compile_flags, kokkos_link_flags = _native_kokkos_flags()
        # Like the native loader, the AOT .so leaves the Kokkos symbols UNDEFINED (resolved at load
        # against the Kokkos runtime already loaded by _adc -- no 2nd copy). macOS/Apple-ld then requires
        # -undefined dynamic_lookup (on ELF/Linux -shared already allows it; the option is NOT GNU ld's).
        link_extra = ["-undefined", "dynamic_lookup"] if sys.platform == "darwin" else []
        with tempfile.TemporaryDirectory() as tmp:
            cpp = os.path.join(tmp, "model_aot.cpp")
            with open(cpp, "w") as f:
                f.write(src)
            _run_compile([cc, "-shared", "-fPIC", "-std=" + std, "-O2", "-I", include]
                         + kokkos_compile_flags + link_extra + [cpp, "-o", so_path] + kokkos_link_flags,
                         "backend aot, compile_aot")
        return so_path

    def emit_cpp_native_loader(self, name=None, target="system", hoist_reciprocals=False):
        """Source of the NATIVE LOADER (backend "production"): the FULL MODEL as CompositeModel<...>
        behind a THIN extern "C" ABI.

        Unlike the "aot" backend (emit_cpp_aot_source: flat array ABI, where the .so
        recomputes everything on a local grid and marshals the arrays), the native loader does NOT carry
        the numerics: it merely INSTALLS the generated model as a NATIVE block of the already-built
        facade, via the header template adc::add_compiled_model<ProdModel>. That template builds the
        closures on the facade's REAL CONTEXT -> the block then runs the SAME path as
        add_block, ZERO-COPY, device-clean (named functors).

        @p target: "system" (default) | "amr_system". Selects the targeted facade and thus the
        add_compiled_model OVERLOAD called:

        - "system": adc::System -> add_compiled_model(System&, ..., evolve); flat single-level
          block (closures on grid_context, add_block production path).
        - "amr_system": adc::AmrSystem -> add_compiled_model(AmrSystem&, ...); single block carried
          over the AMR hierarchy (conservative reflux, regrid). NO evolve parameter (single-block AMR).

        Emitted extern "C" symbols:

        - adc_native_abi_key(): ABI key frozen at the LOADER's compilation, emitted as a preprocessor
          LITERAL (ADC_ABI_KEY_LITERAL) and NOT via the inline function abi_key_string(): under
          ELF/RTLD_GLOBAL, an inline (weak linkage) would be interposed toward the module's copy and
          the loader would return the MODULE's key (tautological guard, never a rejection -- a real CI
          bug when gcc stops inlining). add_native_block compares it to the module's abi_key() -> explicit
          error if headers / compiler / standard diverge (no silent UB).
          Common to both targets.
        - adc_install_native (target="system") OR adc_install_native_amr (target="amr_system"):
          reinterpret_cast<adc::System*|adc::AmrSystem*>(sys) then add_compiled_model<ProdModel>(...).
          The scheme passes through flat arguments (strings + double + int); no C++ object
          crosses the ABI in THIS direction (only the facade* is taken by reference on the loader side, hence
          the requirement of an identical ABI verified by the key). DISTINCT symbol per target: a System
          loader cannot be wired onto AmrSystem.add_native_block, and vice versa."""
        if target not in ("system", "amr_system"):
            raise ValueError("emit_cpp_native_loader: target 'system' | 'amr_system' (got %r)"
                             % (target,))
        nv, bricks, composite = self._emit_bricks(name, hoist_reciprocals=hoist_reciprocals)
        # std headers FIRST (before any namespace). MSVC: a #include <std> while an adc namespace
        # is open makes std seen as adc::std (<vector> errors); g++ tolerates it because already included via
        # guard. Hoisting them here makes the brick-internal #include std harmless (no-op guard).
        head = ('#include <cmath>\n'
                '#include <vector>\n'
                '#include <array>\n'
                '#include <cstddef>\n'
                '#include <string>\n'
                '#include <adc/runtime/abi_key.hpp>\n'         # ADC_ABI_KEY_LITERAL (key frozen at compile)
                '#include <adc/physics/bricks.hpp>\n'          # CompositeModel + NoSource + bricks
                '#include <adc/core/variables.hpp>\n')
        # Header template of the target: dsl_block.hpp (System) or amr_dsl_block.hpp (AmrSystem). Included
        # selectively so as not to pull the AMR machinery into a System loader (and vice versa).
        head += ('#include <adc/runtime/dsl_block.hpp>\n' if target == "system"
                 else '#include <adc/runtime/amr_dsl_block.hpp>\n')
        # preprocessor LITERAL, no call to abi_key_string(): an inline would be interposed
        # (ELF/RTLD_GLOBAL) toward the module's copy -> module's key returned -> tautological guard.
        key = ('#if defined(_WIN32)\n'
               '#define ADC_LOADER_API extern "C" __declspec(dllexport)\n'
               '#else\n'
               '#define ADC_LOADER_API extern "C"\n'
               '#endif\n'
               'ADC_LOADER_API const char* adc_native_abi_key() {\n'
               '  return ADC_ABI_KEY_LITERAL;\n'
               '}\n')
        if target == "system":
            # pos_floor (ADC-76, Zhang-Shu positivity limiter): final flat argument, marshaled
            # down to the loader's make_block via add_compiled_model. Old signature = old .so =
            # rejected by the ABI key (the headers changed), never a wrong argument layout.
            install = ('ADC_LOADER_API void adc_install_native(void* sys, const char* name, const char* limiter,\n'
                       '                                    const char* riemann, const char* recon,\n'
                       '                                    const char* time, double gamma, int substeps,\n'
                       '                                    int evolve, int stride, double pos_floor) {\n'
                       '  adc::System* s = reinterpret_cast<adc::System*>(sys);\n'
                       '  adc::add_compiled_model<adc_generated::ProdModel>(*s, name, adc_generated::ProdModel{},\n'
                       '                                                    limiter, riemann, recon, time, gamma,\n'
                       '                                                    substeps, evolve != 0, stride,\n'
                       '                                                    pos_floor);\n'
                       '}\n')
        else:  # amr_system: AmrSystem overload (no evolve parameter, single-block AMR)
            install = ('ADC_LOADER_API void adc_install_native_amr(void* sys, const char* name,\n'
                       '                                        const char* limiter, const char* riemann,\n'
                       '                                        const char* recon, const char* time,\n'
                       '                                        double gamma, int substeps) {\n'
                       '  adc::AmrSystem* s = reinterpret_cast<adc::AmrSystem*>(sys);\n'
                       '  adc::add_compiled_model<adc_generated::ProdModel>(*s, name, adc_generated::ProdModel{},\n'
                       '                                                    limiter, riemann, recon, time, gamma,\n'
                       '                                                    substeps);\n'
                       '}\n')
        return (head
                + bricks
                + '\nnamespace adc_generated { using ProdModel = %s; }\n' % composite  # comma-free alias
                + key
                + install
                + self._emit_metadata("adc_generated::ProdModel"))  # names/roles/gamma (diagnostic, like AOT/JIT)

    def compile_native(self, so_path, include=None, name=None, cxx=None, std="c++23", target="system",
                       hoist_reciprocals=False):
        """Backend "production": generate the NATIVE LOADER (emit_cpp_native_loader) and compile it into a
        .so loadable by System.add_native_block (target="system") or AmrSystem.add_native_block
        (target="amr_system"). The .so inlines add_compiled_model<ProdModel>: the block runs the
        NATIVE zero-copy path (strict parity with add_block / add_compiled_model<>).

        @p target: "system" (default) | "amr_system" (cf. emit_cpp_native_loader). Selects the
        targeted facade and thus the header template + the installation symbol emitted.

        The loader calls out-of-line methods of the _adc module (install_block / grid_context /
        ensure_aux_width on the System side; set_compiled_block on the AmrSystem side) DEFINED elsewhere: so we
        compile with '-undefined dynamic_lookup' (macOS) to allow these undefined ones (resolved at
        runtime against the already-loaded module; cf. add_native_block). We also bake
        -DADC_HEADER_SIG=<signature> IDENTICAL to the module's so that the ABI keys match when
        the headers match. std: a std different from the module would change __cplusplus hence the key ->
        explicit rejection; the callers (Model.compile/HybridModel.compile) therefore default to the
        loader's standard (loader_cxx_std: c++20 under Kokkos, c++23 otherwise) and not c++23 hard-coded.
        include = adc headers directory (None -> auto-detected via adc_include()); cxx = compiler.
        Returns so_path."""
        import os
        import shutil
        import subprocess
        import sys
        import tempfile

        if include is None:
            include = adc_include()
        # PRE-DLOPEN GUARD: headers != those of the _adc build -> clear error HERE ("rebuild the
        # module") instead of a cryptic dlopen 'symbol not found' in add_native_block. Returns the
        # computed signature (reused for -DADC_HEADER_SIG: a single walk+sha256, not two).
        sig = _check_headers_match_module(include)
        _warn_kokkos_parity()  # Kokkos module + serial loader (or the reverse) -> warn, do not block
        src = self.emit_cpp_native_loader(name=name, target=target,
                                          hoist_reciprocals=hoist_reciprocals)
        cc = _native_kokkos_compiler(cxx)
        if not cc:
            raise RuntimeError("compile_native: no C++ compiler found")
        # Probe BEFORE compilation: if the compiler does not support the standard (real case: old
        # gcc/clang of a conda env picked from the PATH), an actionable error instead of the raw error
        # "invalid value 'c++23'". May fall back to the c++2b spelling (same level).
        std = _probe_cxx_std(cc, std)
        # -DADC_HEADER_SIG: SAME signature as the module build (ABI key concordance).
        #
        # (1) BACKEND PARITY (most important for scaling): if _adc is compiled with Kokkos
        # (OpenMP/CUDA), the loader MUST be too. The header-only templates (assemble_rhs /
        # for_each_cell) compiled WITHOUT -DADC_HAS_KOKKOS instantiate on the SERIAL fallback: the
        # DSL block stays zero-copy but does NOT scale with threads/GPU (ROMEO measurement: DSL warm ~341 ms
        # invariant for threads=1/4/8 whereas the native scales 292->239->177). _native_kokkos_flags() adds
        # -DADC_HAS_KOKKOS + Kokkos includes/libs + -fopenmp when ADC_KOKKOS_ROOT (or Kokkos_ROOT)
        # points at an install; otherwise the historical serial behavior. The compiler follows the backend:
        # g++ (OpenMP) by default, nvcc_wrapper ONLY if explicit (CUDA).
        #
        # (2) OPTIMIZATION PARITY: at -O2 without -DNDEBUG the generated kernel is ~1.48x the native
        # (hot-loop asserts + weak vectorization); -O3 -DNDEBUG => parity (ROMEO measurement CV<1%, ratio 1.04x). These
        # flags affect NEITHER the ABI NOR portability. $ADC_DSL_OPTFLAGS overrides (e.g. add
        # -march=native: the .so being JIT-compiled on the machine -> ~0.88x the generic native; not
        # default because a shared .so cache reused on a different micro-arch = illegal-instr risk).
        kokkos_compile_flags, kokkos_link_flags = _native_kokkos_flags()
        with tempfile.TemporaryDirectory() as tmp:
            cpp = os.path.join(tmp, "model_native.cpp")
            # Windows: bake ADC_HEADER_SIG via #define AT THE TOP of the source (quoting an inline
            # macro string on the cl command line is unmanageable); POSIX keeps the historical -D below.
            src_eff = ('#define ADC_HEADER_SIG "%s"\n' % sig + src) if sys.platform == "win32" else src
            with open(cpp, "w") as f:
                f.write(src_eff)
            if sys.platform == "win32":
                # MSVC/clang-cl (ADC-100): .dll linked against kokkoscore.lib (Kokkos SHARED) + _adc.lib
                # (System ADC_EXPORT symbols). cl accepts -D/-I; output /Fe; libs after /link. No
                # RTLD_GLOBAL: undefined symbols are resolved at the .dll LINK step (import libraries).
                adc_lib = _adc_import_lib()
                if not adc_lib:
                    raise RuntimeError(
                        "compile_native: _adc.lib not found next to the _adc module (required to "
                        "link the DSL .dll; rebuild _adc with ADC_EXPORT_BUILDING_MODULE).")
                # /DNOMINMAX: windows.h (pulled by dynlib.hpp) must not define min/max (breaks the STL).
                # /bigobj: large template TU. NO /Zc:__cplusplus: keep __cplusplus aligned with the
                # module build (otherwise the ABI key diverges).
                cl_flags = ["/nologo", "/LD", "/std:" + std, "/O2", "/DNDEBUG", "/EHsc",
                            "/permissive-", "/Zc:preprocessor", "/DNOMINMAX", "/bigobj"] + kokkos_compile_flags
                cmd = ([cc] + cl_flags + ["-I", include, cpp,
                        "/Fe:" + so_path, "/Fo" + tmp + os.sep,
                        "/link"] + kokkos_link_flags + [adc_lib])
            else:
                optflags = os.environ.get("ADC_DSL_OPTFLAGS", "-O3 -DNDEBUG").split()
                flags = ["-shared", "-fPIC", "-std=" + std, *optflags,
                         "-DADC_HEADER_SIG=\"%s\"" % sig, *kokkos_compile_flags]
                # macOS/Apple-ld: explicitly allow undefined symbols (resolved at runtime).
                if sys.platform == "darwin":
                    flags += ["-undefined", "dynamic_lookup"]
                cmd = [cc, *flags, "-I", include, cpp, "-o", so_path, *kokkos_link_flags]
            _run_compile(cmd, "backend production, compile_native")
        return so_path

    def compile_or_jit(self, so_path, include=None, mode="jit", name=None, cxx=None, std="c++20",
                       target="system", hoist_reciprocals=False):
        """Unified API (facade of the ideal m.compile_or_jit()) selecting the backend:

        - mode="jit" -> compile_so (IModel, virtual dispatch: host prototyping, to be wired via
          System.add_dynamic_block);
        - mode="compile" -> compile_aot (AOT production path, numerically identical to native: to be
          wired via System.add_compiled_block);
        - mode="native" -> compile_native (native zero-copy loader: add_compiled_model<> via
          System.add_native_block or AmrSystem.add_native_block; "production" path).

        @p target: "system" (default) | "amr_system". ONLY consumed by mode="native" (choice of
        the target facade, cf. compile_native). The other modes (jit/compile) target only System;
        a target="amr_system" there is rejected (the AMR .so path exists only for the native backend)."""
        if mode == "jit":
            if target != "system":
                raise ValueError("compile_or_jit: target='amr_system' not supported in mode 'jit' "
                                 "(the AMR path exists only for mode='native')")
            return self.compile_so(so_path, include, name=name, cxx=cxx, std=std,
                                   hoist_reciprocals=hoist_reciprocals)
        if mode == "compile":
            if target != "system":
                raise ValueError("compile_or_jit: target='amr_system' not supported in mode 'compile' "
                                 "(the AMR path exists only for mode='native')")
            return self.compile_aot(so_path, include, name=name, cxx=cxx, std=std,
                                    hoist_reciprocals=hoist_reciprocals)
        if mode == "native":
            return self.compile_native(so_path, include, name=name, cxx=cxx, std=std, target=target,
                                       hoist_reciprocals=hoist_reciprocals)
        raise ValueError("compile_or_jit: mode 'jit' | 'compile' | 'native' (received %r)" % mode)

    # --- production facade: a single entry point per INTENTION (backend) -----------------
    # Routes the compilation backend by INTENTION rather than by implementation detail. Each
    # entry designates one of the existing engines (compile_so / compile_aot) AND the System adder to use
    # at runtime -- coupled here so that a caller does not wire an AOT .so onto add_dynamic_block (or
    # vice versa), which would load but with an inconsistent ABI/numerics.
    #   "prototype"  -> compile_so  (JIT, IModel, virtual dispatch, host first-order Rusanov; fast
    #                   iteration, to be wired via System.add_dynamic_block);
    #   "aot"        -> compile_aot (AOT, host-marshaled PRODUCTION path: assemble_rhs<Limiter,
    #                   Flux>, HLLC/Roe, second order, SSPRK2/IMEX on a LOCAL grid of the .so; numerics
    #                   identical to native but marshaled arrays, via add_compiled_block);
    #   "production" -> compile_native (NATIVE LOADER): the .so inlines add_compiled_model<ProdModel>, which
    #                   installs the generated model as a NATIVE System block (closures over the REAL
    #                   grid_context). The block runs ZERO-COPY the SAME path as add_block (no
    #                   marshaling); device-clean by construction (named functors from block_builder).
    #                   To be wired via System.add_native_block (ABI key verified). This is the path
    #                   prepared for a real production backend (Kokkos/CUDA codegen = later PR).
    _BACKENDS = {
        "prototype": ("jit", "add_dynamic_block"),
        "aot": ("compile", "add_compiled_block"),
        "production": ("native", "add_native_block"),
    }

    def _model_hash(self, params=None):
        """Stable hash of the model: formulas (flux/eig/source/elliptic/primitives/cons_from) + roles +
        n_aux + gamma (+ any NAMED params). Single source of the hash, reused by Model._model_hash
        (which passes its Param). Serves to identify/reuse an already compiled .so (cache key) and to trace
        the run. Relies on repr(Expr) (stable, structural); insensitive to dict ordering (sorted)."""
        import hashlib
        m = self
        parts = []
        parts.append("name=%s" % m.name)
        parts.append("cons=%s" % ",".join(m.cons_names))
        parts.append("croles=%s" % ",".join(roles_for(m.cons_names, m.cons_roles)))
        parts.append("prim_state=%s" % ",".join(m.prim_state))
        parts.append("proles=%s" % ",".join(roles_for(m.prim_state, m.prim_roles)))
        parts.append("prim=%s" % ";".join("%s=%r" % (k, m.prim_defs[k]) for k in m.prim_defs))
        for d in ("x", "y"):
            parts.append("flux_%s=%s" % (d, ";".join(repr(e) for e in m._flux.get(d, []))))
            parts.append("eig_%s=%s" % (d, ";".join(repr(e) for e in m._eig.get(d, []))))
        parts.append("source=%s" % (";".join(repr(e) for e in m._source) if m._source else ""))
        parts.append("cons_from=%s" % (";".join(repr(e) for e in m.cons_from) if m.cons_from else ""))
        parts.append("elliptic=%s" % (repr(m._elliptic) if m._elliptic is not None else ""))
        parts.append("stab_speed=%s" % (repr(m._stab_speed) if m._stab_speed is not None else ""))
        parts.append("stab_dt=%s" % (repr(m._stab_dt) if m._stab_dt is not None else ""))
        parts.append("src_freq=%s" % (repr(m._src_freq) if m._src_freq is not None else ""))
        parts.append("src_jac=%s" % (";".join(repr(e) for row in m._src_jac for e in row)
                                     if m._src_jac is not None else ""))
        # Projection ponctuelle post-pas (ADC-177) : ajoutee au hash UNIQUEMENT si declaree (sans
        # appel, hash strictement identique a l'historique -> cle de cache .so preservee).
        if getattr(m, "_proj", None) is not None:
            parts.append("proj=%s" % ";".join(repr(e) for e in m._proj))
        parts.append("hllc=%d" % (1 if m._hllc else 0))
        parts.append("roe=%d" % (1 if getattr(m, "_roe", False) else 0))
        # roe_dissipation PROVIDED: added to the hash ONLY if present (without a call, hash unchanged
        # -> bit-identity of the cache key for existing models preserved).
        if getattr(m, "_roe_rows", None) is not None:
            parts.append("roe_rows=%s" % ";".join(repr(e) for k in ("x", "y")
                                                  for e in m._roe_rows[k]))
        # EXPLICIT signed wave speeds (set_wave_speeds): same conditional policy (without a call,
        # hash strictly identical to the historical one -> .so cache of existing models preserved).
        if getattr(m, "_wave_speeds", None) is not None:
            parts.append("wave_speeds=%s" % ";".join(repr(e) for k in ("x", "y")
                                                     for e in m._wave_speeds[k]))
        if getattr(m, "_ws_jacobian", None) is not None:
            ws = m._ws_jacobian
            parts.append("ws_jac=%s|%s|%s" % (
                ws["eig"],
                "//".join(";".join(",".join(str(i) for i in b) for b in ws["blocks"][k])
                          for k in ("x", "y")),
                ";".join(repr(e) for k in ("x", "y") for row in ws["rows"][k] for e in row)
                if ws["rows"] is not None else ""))

        parts.append("n_aux=%d" % aux_total_n_aux(m.aux_names, m.aux_extra_names))
        # NAMED aux fields (aux_field, ADC-70): their ORDER fixes the index (AUX_NAMED_BASE + k) -> they
        # enter the hash (two models differing only by a named-aux name/order are distinct). Adds the key
        # ONLY if named fields exist: a model without aux_field thus keeps a STRICTLY identical hash to
        # the historical one (.so cache + traceability preserved).
        if m.aux_extra_names:
            parts.append("aux_extra=%s" % ",".join(m.aux_extra_names))
        parts.append("gamma=%r" % m.gamma)
        # Params enter the hash via (name, DECLARATION value, kind). The value of a RUNTIME param
        # (P7-b) appears there because it SEEDS the default of the generated RuntimeParams member (so two
        # .so with a different default are distinct); the "no recompilation" of P7-b applies to
        # set_block_params at RUNTIME, not to a new compile() call with a changed declaration value.
        params = params or {}
        parts.append("params=%s" % ";".join("%s=%r:%s" % (k, params[k].value, params[k].kind)
                                             for k in sorted(params)))
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()

    def _check_require_metadata(self, require_metadata, backend):
        """require_metadata guard rails (pure-Python, deterministic on the model + backend). Factored out
        to be called BEFORE the cache (in HyperbolicModel AND Model): a cache HIT must never
        mask a metadata requirement. Without require_metadata, no-op."""
        if not require_metadata:
            return
        # backend "prototype" (add_dynamic_block, VIRTUAL dispatch, host first-order Rusanov): NOT a
        # device-clean production path -> requesting metadata on it is inconsistent (clear error).
        if backend == "prototype":
            raise ValueError(
                "compile: backend 'prototype' (JIT, host virtual dispatch) incompatible with "
                "require_metadata=True; use backend='aot' or 'production' for the "
                "device-clean path with guaranteed metadata")
        missing = []
        roles = roles_for(self.cons_names, self.cons_roles)
        if all(r == "Custom" for r in roles):
            missing.append("physical roles (conservative_vars(..., roles=[...]) or canonical names)")
        if self.gamma is None:
            missing.append("gamma (set_gamma(...))")
        if missing:
            raise ValueError(
                "compile(require_metadata=True): model '%s' does not provide %s; the .so "
                "would fall back to the System fallback (roles 'custom' / gamma 1.4)"
                % (self.name, " nor ".join(missing)))

    def compile(self, so_path=None, include=None, backend="auto", name=None, cxx=None, std=None,
                require_metadata=False, target="system", hoist_reciprocals=False):
        """Compilation facade by INTENTION: compiles the model into a .so via the engine designated
        by @p backend and returns its path. Wraps the existing engines (compile_so / compile_aot /
        compile_native) WITHOUT changing the numerics; preserves end-to-end names, VariableRole, gamma,
        n_aux, B_z and T_e (the same bricks + ABI metadata as compile_or_jit).

        ERGONOMICS (does not change the numerics):
          - @p include None -> auto-detected (adc_include(): $ADC_INCLUDE, installed adc package, neighbor
            repository); passing include= remains possible (back-compat);
          - @p so_path None -> compiles into an out-of-source cache (adc_cache_dir()), with a file name
            keyed on model_hash + abi_key (+ backend/target/name). On a cache HIT (.so already
            present for this key), NO recompilation: the cached .so is reused as is.
            On a cache MISS (model/parameter/toolchain change -> different key), recompilation then
            storage. Passing so_path= forces this path and always compiles (strict back-compat).

        @p backend:
          "prototype"  -> JIT (compile_so): fast iteration, host virtual dispatch (first-order Rusanov),
                          to be wired on the System side via add_dynamic_block;
          "aot"        -> AOT (compile_aot): host-marshaled production path, numerics identical to the
                          native block, to be wired via add_compiled_block;
          "production" -> NATIVE (compile_native): .so loader inlining add_compiled_model<ProdModel>, native
                          zero-copy block (strict parity add_block / add_compiled_model<>), to be wired
                          via add_native_block (ABI key verified). Device-clean path prepared.

        @p target: "system" (default) | "amr_system". Only the "production" backend targets AmrSystem
        (System.add_native_block vs AmrSystem.add_native_block); a target="amr_system" on the other
        backends is rejected (no AMR .so path outside native, cf. compile_or_jit).

        @p std: C++ standard. Default None -> THE LOADER's standard for "production" (loader_cxx_std:
        c++20 under Kokkos because CUDA 12.x has no -std=c++23, c++23 otherwise; the native loader shares
        the module's ABI, a different std would change __cplusplus hence the ABI key -> explicit rejection
        by add_native_block), "c++20" for the others (unchanged).

        @p require_metadata (default False): if True, requires that the .so carry useful physical roles
        AND an explicit gamma (set_gamma), failing which the System would fall back to the fallback
        (roles 'custom' / gamma 1.4) -- silent regression of inter-species couplings. Serves a
        production pipeline that wants an EXPLICIT error rather than a silent fallback.

        Raises ValueError on an unknown backend or a feature incompatible with the requested backend
        (rather than an obscure failure at runtime). Returns so_path.

        To know which System adder to use: see adder_for(backend)."""
        import os
        import shutil
        # DEFAULT 'auto' (ADC-63): production if toolchain parity with the module is established,
        # aot otherwise (historical default). An explicit backend short-circuits (unchanged).
        if backend == "auto":
            backend, _auto_reason = resolve_auto_backend(include)
        if backend not in self._BACKENDS:
            raise ValueError("compile: backend %r unknown (expected %s + 'auto')"
                             % (backend, sorted(self._BACKENDS)))
        if target not in ("system", "amr_system"):
            raise ValueError("compile: target 'system' | 'amr_system' (received %r)" % (target,))
        mode, adder = self._BACKENDS[backend]
        if target == "amr_system" and mode != "native":
            raise ValueError("compile: target='amr_system' exists only for backend='production' "
                             "(native AMR path); received backend=%r" % (backend,))
        if std is None:  # default per backend: native shares the module's ABI (c++20 under Kokkos,
            # c++23 otherwise -- derived from the loader, cf. loader_cxx_std), the others stay on c++20.
            std = loader_cxx_std() if mode == "native" else "c++20"
        if include is None:  # ergonomics: auto-detection of the adc headers directory
            include = adc_include()

        # Metadata guard rails (before any cache: they depend only on the model + backend, and a
        # cache HIT must not mask them).
        self._check_require_metadata(require_metadata, backend)

        # Out-of-source CACHE when so_path is omitted: file name keyed on model_hash + abi_key
        # (+ backend/target/name). Cache HIT (.so already present for this key) -> reuse without
        # recompilation. Cache MISS -> compilation in the keyed path (thus stored for next
        # time). Explicit so_path -> forced path, always recompiled (strict back-compat).
        if so_path is None:
            # The backends that compile the adc headers (native production and aot) follow the real Kokkos
            # (compiler + kokkos feature-key in the cache key): under Kokkos-only, their .so is
            # always compiled WITH Kokkos (cf. compile_aot / compile_native), the key must reflect it.
            kokkos_like = backend in ("production", "aot")
            eff_cxx = _native_kokkos_compiler(cxx) if kokkos_like else _default_cxx(cxx)
            abi_key = _abi_key_python(include, eff_cxx, std)
            cache_backend = (backend + ";" + _native_feature_key()) if kokkos_like else backend
            if hoist_reciprocals:  # distinct codegen -> distinct key (no collision with the default output)
                cache_backend += ";hoist"
            so_path = _cache_so_path(self._model_hash(), abi_key, cache_backend, target, name)
            if os.path.exists(so_path):
                return so_path  # cache HIT: .so already compiled for this key, reused as is

        return self.compile_or_jit(so_path, include, mode=mode, name=name, cxx=cxx, std=std,
                                  target=target, hoist_reciprocals=hoist_reciprocals)

    @classmethod
    def adder_for(cls, backend):
        """Name of the System method to use to wire the .so produced by compile(backend=...):
        'add_dynamic_block' (prototype/JIT), 'add_compiled_block' (aot) or 'add_native_block'
        (production/native). Couples the compilation backend to its adder to avoid an inconsistent
        ABI boundary. ValueError if unknown."""
        if backend not in cls._BACKENDS:
            raise ValueError("adder_for: backend %r unknown (expected %s)"
                             % (backend, sorted(cls._BACKENDS)))
        return cls._BACKENDS[backend][1]

    def emit_cpp_elliptic(self, name=None, namespace="adc_generated", cse=True,
                          hoist_reciprocals=False):
        """Generates a composable elliptic RIGHT-HAND SIDE BRICK from self._elliptic.

        The produced struct exposes rhs(U) -> Real (charge density, background, gravity...), same shape as
        the manual bricks (ChargeDensity, BackgroundDensity in adc/model/bricks.hpp): it enters
        as the Elliptic parameter of a CompositeModel. Inlined constants, cons/primitives -> locals,
        cse=True factors out common sub-expressions. ValueError if set_elliptic_rhs(...) is missing."""
        if self._elliptic is None:
            raise ValueError("emit_cpp_elliptic: call set_elliptic_rhs(...) first")
        nm = name or (self.name.capitalize() + "Elliptic")
        rt_member = self._runtime_params_member()  # P7-b: runtime indices BEFORE any to_cpp()
        out = [
            "#include <cmath>",  # self-sufficient for std::sqrt / std::pow
            "// brique de SECOND MEMBRE elliptique generee depuis '%s' (adc.dsl.emit_cpp_elliptic)."
            % self.name,
            "// rhs(U) -> Real : second membre f(U) de l'operateur elliptique (p.ex. densite de charge).",
        ]
        if rt_member:  # RuntimeParams header only if a formula reads a runtime param
            out.append("#include <adc/runtime/runtime_params.hpp>")
        out += [
            "namespace %s {" % namespace,
            "struct %s {" % nm,
        ]
        if rt_member:  # member adc::RuntimeParams params{count, {defaults}} (P7-b)
            out.append(rt_member.rstrip("\n"))
        out += [
            "  template <class State>",
            "  ADC_HD adc::Real rhs(const State& U) const {",
        ]
        out += ["    const adc::Real %s = U[%d];" % (c, i) for i, c in enumerate(self.cons_names)]
        out += self._prim_block(self._live_prims([self._elliptic]), hoist_reciprocals)
        tl, cpps = self._codegen_exprs([self._elliptic], cse)
        out += tl
        out += ["    return %s;" % cpps[0], "  }", "};", "}  // namespace %s" % namespace]
        return "\n".join(out) + "\n"


# === Phase A: pure-Python user facade =================================
# The STABLE surface the user writes (dsl.Model / Param / CompiledModel). Pure sugar:
# no new numerics, no engine change. dsl.Model COMPOSES a private HyperbolicModel
# (_m) and delegates each call to an existing method; Param is a NAMED constant that inlines
# at codegen; CompiledModel packages the .so + the metadata already known on the Python side (no
# re-reading of the .so). cf. docs/DSL_MODEL_DESIGN.md (Phase A).

# HONEST characteristics per backend (cf. DSL_MODEL_DESIGN.md section 5). Serves diagnostics and
# the device/MPI/AMR guard rails (checked at wiring/execution, not frozen at compilation).
_BACKEND_CAPS = {
    # backend: (cpu, mpi, amr, gpu)  -- True/False according to what the path SUPPORTS today
    "prototype": {"cpu": True, "mpi": False, "amr": False, "gpu": False},
    "aot": {"cpu": True, "mpi": False, "amr": False, "gpu": False},
    # production = NATIVE path (add_native_block, #85): same engine as add_block, hence MPI-capable
    # by construction (halos fill_boundary). amr=True: the native loader now has an AMR counterpart
    # (m.compile(backend='production', target='amr_system') -> AmrSystem.add_native_block, DSL Phase D)
    # which inlines add_compiled_model(AmrSystem&) -> SAME AMR hierarchy as AmrSystem.add_block (reflux,
    # regrid). gpu=False out of CAUTION: the native path is device-clean in C++ (GH200) but the
    # end-to-end validation from Python (add_native_block on device) is a dedicated PR (DSL sect. 5).
    "production": {"cpu": True, "mpi": True, "amr": True, "gpu": False},
}


class Param:
    """NAMED parameter of a DSL model, usable like an Expr in formulas.

    Mode (a), constant fixed at compilation: `kind="const"` (default). The codegen INLINES every
    constant (Const.to_cpp -> repr(value)), so the param inlines as Const(value) at codegen (value
    written HARD-CODED in the .so) while keeping its IDENTITY (name/value/kind) for introspection
    (m.params), diagnostics and reproducibility. UNCHANGED by P7-b: bit-identical to the history.

    Mode (b), RUNTIME parameter (modifiable WITHOUT recompiling): `kind="runtime"` (P7-b). At codegen, the
    param emits `params.get(<index>)` (read of an adc::RuntimeParams member of the brick) instead
    of a constant; its value is carried at runtime by the AOT .so ABI and can be CHANGED
    (block.set_param / System.set_block_params) without recompiling. The value passed at declaration serves
    as the DEFAULT (without a set call, the block behaves as with a const param of that value).
    SUPPORTED by the "aot" backend (add_compiled_block). The "prototype" (JIT) and "production"
    (native) backends compile a runtime param as its declaration value (fixed): a set_param there has no
    effect, the API reports it (cf. CompiledModel.runtime_param_names / System.set_block_params).

    Param DOES NOT INHERIT from Expr (see NB below): it EXPOSES the same tree hooks
    (`eval`/`to_cpp`/`deps`) and operators by DELEGATING to an internal NODE (Const for 'const',
    RuntimeParamRef for 'runtime'), so `g * (E - ...)` builds the expected tree directly. The
    value is not an environment variable -> no dependency to check in check()."""

    # NB: Param DOES NOT INHERIT from Expr to avoid embedding its state (name/kind) in the CSE
    # structural key; it EXPOSES the tree hooks instead by delegating to an internal node.
    def __init__(self, name, value, kind="const"):
        if kind not in ("const", "runtime"):
            raise ValueError("Param: kind 'const' | 'runtime' (got %r)" % (kind,))
        self.name = name
        self.value = float(value)
        self.kind = kind
        if kind == "runtime":
            # SHARED RUNTIME node: all occurrences of the param in formulas point to this same
            # object, so setting its .index at compilation (Model._assign_runtime_indices) is enough to
            # route all its reads to params.get(<index>). index=-1 while not assigned.
            self._node = RuntimeParamRef(self.name, self.value)
        else:
            self._node = Const(self.value)  # inlines at codegen: value written HARD-CODED in the .so

    # --- tree hooks (delegated to the internal node): Param usable like an Expr ---
    def eval(self, env): return self._node.eval(env)
    def to_cpp(self): return self._node.to_cpp()
    def deps(self): return set()  # neither const nor runtime has a dependency (nothing to check in check())

    # --- operators: Param combines like an Expr (promotion via _wrap of the internal node) ---
    def __add__(self, o): return Add(self._node, _wrap(o))
    def __radd__(self, o): return Add(_wrap(o), self._node)
    def __sub__(self, o): return Sub(self._node, _wrap(o))
    def __rsub__(self, o): return Sub(_wrap(o), self._node)
    def __mul__(self, o): return Mul(self._node, _wrap(o))
    def __rmul__(self, o): return Mul(_wrap(o), self._node)
    def __truediv__(self, o): return Div(self._node, _wrap(o))
    def __rtruediv__(self, o): return Div(_wrap(o), self._node)
    def __neg__(self): return Neg(self._node)
    def __pos__(self): return self._node  # +param = identity (Expr), for +k*ne*ng
    def __pow__(self, o): return Pow(self._node, _wrap(o))

    def __float__(self): return self.value
    def __repr__(self): return "Param(%r, %r, kind=%r)" % (self.name, self.value, self.kind)


def RuntimeParam(name, value):
    """Sugar: a RUNTIME parameter (modifiable without recompiling). Equivalent to Param(name, value,
    kind='runtime'). cf. Param mode (b) and include/adc/runtime/runtime_params.hpp (P7-b)."""
    return Param(name, value, kind="runtime")


class CompiledModel:
    """Result of `m.compile(...)`: packages the produced `.so` + EVERYTHING needed to wire it
    correctly (dispatch adder, ABI diagnostic, reproducibility). Replaces the historical pair
    (str so_path, adder_for(backend)) with a single object.

    The metadata is NOT re-read from the `.so`: Python already holds names/roles/gamma/n_aux/params
    (the HyperbolicModel carries them); CompiledModel just exposes them for dispatch (add_equation)
    and diagnostics. cf. DSL_MODEL_DESIGN.md section 3."""

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
        self.abi_key = abi_key       # ABI key mirroring adc_header_signature + compiler/std
        self.model_hash = model_hash  # stable hash formulas+roles+n_aux+params
        self.cxx = cxx
        self.std = std

    @property
    def runtime_param_names(self):
        """Names of the model's RUNTIME parameters (kind='runtime'), SORTED: this is the ORDER of the
        indices on the C++ side (RuntimeParams) AND the order expected by System.set_block_params(name, values) (P7-b).
        Empty if the model has only const params."""
        return sorted(k for k, p in self.params.items() if getattr(p, "kind", "const") == "runtime")

    def runtime_param_values(self):
        """DECLARATION values of the runtime params, parallel to runtime_param_names (default as long
        as no set_block_params has been called)."""
        return [self.params[k].value for k in self.runtime_param_names]

    def check_runtime(self, n=16, state=None, raise_on_error=True, rtol=1e-8, atol=1e-10):
        """RUNTIME re-verification of a CompiledModel ALONE (audit balance, GENERICITY pt 9): without the
        original dsl.Model, the FORMULAS are no longer re-verifiable (symbolic check_model), but
        the .so itself is -- we install it in an EPHEMERAL System (n x n periodic, neutral
        Poisson, minmod+rusanov) and delegate to System.check_model (finite state, residual -div F + S
        finite, positivity by roles, round-trip of THE MODEL conversions).

        @p state: dict {conservative variable name: ndarray (n, n)} to control the tested state.
        None -> SMOKE state by ROLES (Density = 1 + gaussian bump, Momentum* = 0,
        Energy = 2.5, other components = 0.5) -- enough to exercise flux/source/conversions;
        provide state= for a precise physical regime. @return the dict from System.check_model."""
        import numpy as np
        if getattr(self, "target", "system") != "system":
            raise ValueError(
                "CompiledModel.check_runtime: only target='system' is re-verifiable in an "
                "ephemeral System; a target='amr_system' loader is checked installed in its "
                "AmrSystem (AMR test invariants), not in isolation.")
        from . import System, FiniteVolume, Explicit
        sim = System(n=int(n), L=1.0, periodic=True)
        sim.set_poisson()
        sim.add_equation("check", model=self,
                         spatial=FiniteVolume(limiter="minmod", riemann="rusanov"),
                         time=Explicit())
        x = (np.arange(n) + 0.5) / float(n)
        X, Y = np.meshgrid(x, x, indexing="xy")
        bump = 1.0 + 0.3 * np.exp(-40.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))
        comps = []
        for name, role in zip(self.cons_names, self.cons_roles):
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


def _abi_key_python(include, cxx, std):
    """ABI key on the Python side, MIRROR of adc::detail::abi_key_string (compiler + standard +
    header signature). Makes the verification + diagnostic available on the Python side BEFORE
    loading the .so (the native path compares its own on the C++ side). Stable and readable form:
    "<header sig>|<cxx>|<std>". include absent -> empty signature (degraded diagnostic, no UB)."""
    import os
    sig = adc_header_signature(include) if include and os.path.isdir(include) else ""
    return "%s|%s|%s" % (sig, cxx or "", std or "")


class Model:
    """STABLE facade of a DSL model (Phase A). COMPOSES a private HyperbolicModel (_m, composition and
    NOT inheritance) and delegates each call to an existing method: no new numerics.

        m = adc.dsl.Model("euler")
        rho, rhou, rhov, E = m.conservative_vars("rho", "rho_u", "rho_v", "E")
        g = m.param("gamma", 1.4)                 # NAMED constant, inlined at codegen
        u = m.primitive("u", rhou / rho)
        p = m.primitive("p", (g - 1.0) * (E - 0.5 * rho * (u*u + ...)))
        m.flux(x=[...], y=[...])                   # symbolic DECLARATOR of the physical flux
        m.eval_flux(U, aux, dir)                   # numpy EVALUATOR (debug), DISTINCT name
        m.primitive_vars(rho=rho, u=u, v=v, p=p)   # ordered Prim layout (kwargs order)
        compiled = m.compile(so_path, include, backend="aot")  # -> CompiledModel

    cf. docs/DSL_MODEL_DESIGN.md sections 1-3."""

    def __init__(self, name):
        self._m = HyperbolicModel(name)
        self.params = {}   # name -> Param (introspection / reproducibility)

    @property
    def name(self): return self._m.name

    # --- variable declaration (direct delegation to HyperbolicModel) ---
    def conservative_vars(self, *names, roles=None):
        """Declares the conservative variables. @p roles: same convention as HyperbolicModel."""
        return self._m.conservative_vars(*names, roles=roles)

    def primitive(self, name, expr):
        """Defines a primitive by its formula (as a function of the cons / preceding primitives)."""
        return self._m.primitive(name, expr)

    def primitive_vars(self, *vars, roles=None, **named):
        """Declares the primitives AND the ORDERED layout of Prim. Two forms:

        - KWARGS (target style): `primitive_vars(rho=expr, u=expr, v=expr, p=expr)`: each kwarg
          DEFINES a primitive (m.primitive(name, expr)) AND fixes the layout of Prim in the
          insertion order of the kwargs (Python 3.7+: order guaranteed). @p roles (list) optional.
        - POSITIONAL: `primitive_vars(rho, u, v, p, roles=...)`: names/Var already defined, fixes
          only the layout (delegates to set_primitive_state, like HyperbolicModel).

        The two forms are exclusive (mixing named kwargs and positional raises)."""
        if named and vars:
            raise ValueError("primitive_vars: mixing positional form and named kwargs "
                             "(choose one; kwargs define AND order the primitives)")
        if named:
            # kwargs: define each primitive, then fix the layout in insertion order.
            # A primitive is NOT (re)defined if the kwarg is the Var of the SAME name -- otherwise the codegen
            # would emit `const Real x = x;` (auto-init -> NaN). Two cases of self-reference, both
            # left to JOIN the layout without redefinition (target style primitive_vars(rho=rho, u=u, ...)):
            #  - name ALREADY CONSERVATIVE (e.g. rho=rho: the density, primitive == conservative);
            #  - PRIMITIVE Var ALREADY DEFINED of the same name (e.g. u=u when u comes from m.primitive('u', ...)).
            # Otherwise (kwarg = expression, e.g. p=cs2*rho or u=mx/rho) the primitive is defined normally.
            ordered = list(named.keys())
            for nm in ordered:
                val = named[nm]
                self_ref = isinstance(val, Var) and getattr(val, "name", None) == nm
                if nm in self._m.cons_names or self_ref:
                    continue
                self._m.primitive(nm, val)
            self._m.set_primitive_state(*ordered, roles=roles)
            return tuple(Var(nm, "prim") for nm in ordered)
        # positional form: fixes the layout from already-defined names/Var.
        self._m.set_primitive_state(*vars, roles=roles)
        return None

    def aux(self, name):
        """CANONICAL auxiliary field (must be a key of AUX_CANONICAL: phi/grad_x/grad_y/B_z/T_e)."""
        return self._m.aux(name)

    def aux_field(self, name):
        """NAMED auxiliary field (ADC-70 phase 1) provided by a block via System.set_aux_field(block, name,
        array). name is ARBITRARY (identifier); the k-th call reserves the aux channel component
        AUX_NAMED_BASE + k (read in C++ via aux.extra_field(k)). At most AUX_NAMED_MAX per model.
        Returns a Var usable in flux / source / eigenvalues. Delegates to HyperbolicModel.aux_field."""
        return self._m.aux_field(name)

    def conservative_from(self, exprs):
        """Inverse prim -> cons (the DSL cannot invert symbolically)."""
        self._m.set_conservative_from(exprs)

    # --- flux: symbolic DECLARATOR vs numpy EVALUATOR (DISTINCT names, settled decision) ---
    def flux(self, x, y):
        """Symbolic DECLARATOR of the physical flux (delegates to set_flux). x/y: lists of Expr, one
        per conservative component. DO NOT confuse with the numpy evaluator eval_flux."""
        self._m.set_flux(x, y)

    def eval_flux(self, U, aux, dir):
        """numpy EVALUATOR of the physical flux (debug / host proto; delegates to HyperbolicModel.flux).
        U: numpy (n_vars, ...); aux: dict name -> array; dir: 0=x, 1=y."""
        return self._m.flux(U, aux, dir)

    def eigenvalues(self, x, y):
        """Eigenvalues (characteristic speeds) per direction (delegates to set_eigenvalues)."""
        self._m.set_eigenvalues(x, y)

    def wave_speeds(self, x, y):
        """Explicit SIGNED wave speeds per direction: x = (smin_x, smax_x), y = (smin_y,
        smax_y). Emits ``wave_speeds(U, aux, dir, smin, smax)`` on the brick WITHOUT requiring a
        primitive 'p': riemann='hll' becomes available for a model without pressure (moment
        system, isothermal...). Takes priority over the historical path (eigenvalues + 'p'); if
        eigenvalues is not declared, max_wave_speed (Rusanov / CFL) derives from ``max(|smin|, |smax|)``.
        Delegates to set_wave_speeds; cf. HyperbolicModel.set_wave_speeds."""
        self._m.set_wave_speeds(x, y)

    def wave_speeds_from_jacobian(self, x=None, y=None, eig="numeric", blocks=None):
        """EXACT signed wave speeds from the eigenvalues of the flux jacobian (delegates to
        set_wave_speeds_from_jacobian, see its full contract): x/y = dF/dU as Expr (None =
        AUTODIFF of the declared flux via flux_jacobian); eig = 'numeric' | 'fd' (finite differences
        of the compiled flux, debug); blocks = lists of indices of the diagonal sub-blocks (None = full
        block, the only unconditionally correct mode -- the blocks ASSERT a
        block-triangular structure)."""
        self._m.set_wave_speeds_from_jacobian(x=x, y=y, eig=eig, blocks=blocks)

    def eval_wave_speeds(self, U, aux, dir):
        """numpy EVALUATOR of the emitted signed speeds (smin, smax) (delegates to
        HyperbolicModel.wave_speeds_value): explicit pair, jacobian (numpy eig per blocks) or
        min/max of the eigenvalues."""
        return self._m.wave_speeds_value(U, aux, dir)

    def stability_speed(self, expr):
        """STABILITY speed lambda* driving the block CFL (OPTIONAL; delegates to
        HyperbolicModel.stability_speed). Fallback without a call: max(abs(eigenvalues)), strictly
        the historical behavior. Compiled like flux/source (production GPU/MPI, no per-cell callback)."""
        self._m.stability_speed(expr)

    def stability_dt(self, expr_dt):
        """Direct ADMISSIBLE step dt(U, aux) local bound of the step (OPTIONAL; delegates to
        HyperbolicModel.stability_dt). The cfl is not applied to this bound. Fallback without a call:
        no additional bound (historical step policy)."""
        self._m.stability_dt(expr_dt)

    def source(self, s):
        """Source term S(U, aux), one expression per component (optional; delegates to set_source)."""
        self._m.set_source(s)

    def source_frequency(self, expr_mu):
        """Local frequency mu(U, aux) [1/s] of the source -- the 'source' step bound from the meeting
        (dt <= cfl*substeps/(stride*max mu), without a space step). Emitted on the generated SOURCE
        brick (frequency(U, aux)), forwarded by CompositeModel, aggregated by System/AmrSystem
        step_cfl. REQUIRES m.source([...]). Delegates to HyperbolicModel.source_frequency."""
        self._m.source_frequency(expr_mu)

    def source_jacobian(self, rows):
        """ANALYTIC Jacobian dS/dU of the source (rows[r][c] = dS_r/dU_c, an n_vars x n_vars
        matrix of expressions): the implicit Newton (IMEX/SourceImplicitBE) uses it instead of
        finite differences. REQUIRES m.source. Delegates to HyperbolicModel.source_jacobian."""
        self._m.source_jacobian(rows)

    def projection(self, exprs):
        """PROJECTION PONCTUELLE post-pas U <- P(U, aux) (ADC-177, OPTIONNEL) : une expression par
        composante conservative, appliquee par le System a la FIN de chaque macro-pas ENTIER (jamais
        par etage RK) sur les cellules valides. CONTRAT : idempotente et ponctuelle ; clamps en
        max/min via abs_/sign. Backends 'aot'/'production' (System) ; 'prototype' et AMR rejetes.
        Delegue a HyperbolicModel.projection (cf. son contrat complet)."""
        self._m.projection(exprs)

    def projection_value(self, U, aux=None):
        """EVALUATEUR numpy de la projection emise (reference de test ; delegue a
        HyperbolicModel.projection_value)."""
        return self._m.projection_value(U, aux)

    def implicit_source(self, jacobian=None):
        """GROUPED declaration of the local implicit (wave 3 audit, sugar): the RESIDUAL is already
        implied by m.source (backward-Euler: F = W - U^n - dt*S(W)); @p jacobian (optional) =
        analytic dS/dU matrix (cf. source_jacobian). Without jacobian: finite differences."""
        if jacobian is not None:
            self._m.source_jacobian(jacobian)

    def enable_hllc(self):
        """Emits the HLLC capability (contact_speed + hllc_star_state generated from the ROLES +
        primitive 'p'): riemann='hllc' becomes available for this model EVEN outside 4-variable
        Euler. Delegates to HyperbolicModel.enable_hllc."""
        self._m.enable_hllc()

    def enable_roe(self):
        """Emits the ROE capability (roe_dissipation = ``|A_roe| dU`` generated from the ROLES +
        primitive 'p'): riemann='roe' becomes available for this model EVEN outside 4-variable
        Euler (without Energy: c = sqrt(p/rho) averaged Roe-style; components outside the fluid
        roles = passive scalars on the entropy wave). Delegates to HyperbolicModel.enable_roe."""
        self._m.enable_roe()

    def roe_dissipation(self, x, y):
        """Roe dissipation PROVIDED by the user (outside the fluid roles): n_vars expressions per
        direction (x=, y=), written with m.left(...)/m.right(...) (or dsl.left/right) of the two states,
        emitted as the C++ hook roe_dissipation(UL, AL, UR, AR, dir). During the 'provided' mode of enable_roe
        (a single provider: supplying both together raises). The helper m.flux_jacobian assists the writing.
        Delegates to HyperbolicModel.roe_dissipation (cf. its doc)."""
        self._m.roe_dissipation(x, y)

    def flux_jacobian(self, dir):
        """Flux Jacobian A = dF_dir/dU (an n_vars x n_vars matrix of Expr, A[i][j]=d(F_i)/d(U_j)),
        auto-derived from the fluxes declared via dsl.diff (expanded primitives). HELPER for building
        m.roe_dissipation, emits nothing. @p dir: 0/'x' or 1/'y'. Delegates to HyperbolicModel."""
        return self._m.flux_jacobian(dir)

    def left(self, expr):
        """Marks @p expr as evaluated on the LEFT state UL (m.roe_dissipation). Sugar for dsl.left."""
        return left(expr)

    def right(self, expr):
        """Marks @p expr as evaluated on the RIGHT state UR (m.roe_dissipation). Sugar for dsl.right."""
        return right(expr)

    def elliptic_rhs(self, e):
        """Contribution to the elliptic right-hand side (Poisson coupling; delegates to set_elliptic_rhs)."""
        self._m.set_elliptic_rhs(e)

    def gamma(self, value):
        """Adiabatic index (EOS), carried by ADC_EXPORT_BLOCK_GAMMA (delegates to set_gamma)."""
        self._m.set_gamma(value)

    def param(self, name, value, kind="const"):
        """NAMED parameter usable in the formulas. Mode (a) (`kind="const"`, default): constant
        frozen at compile time, inlined at codegen; stored in m.params (introspection /
        reproducibility). Mode (b) (`kind="runtime"`, P7-b): SUPPORTED on the "aot" backend -- the param
        emits `params.get(<index>)` (member of adc::RuntimeParams) and its value can be CHANGED at runtime
        via System.set_block_params(name, values) WITHOUT recompiling (the declaration value serves as the
        default); cf. CompiledModel.runtime_param_names. The "prototype"/"production" backends freeze a
        runtime param at its declaration value.

        gamma CASE: if name == "gamma", ALSO calls set_gamma(value) so that the ABI metadata
        stays consistent (otherwise the System falls back to 1.4)."""
        p = Param(name, value, kind=kind)  # 'runtime' -> RuntimeParamRef (P7-b), 'const' -> inline
        self.params[name] = p
        if name == "gamma":
            self._m.set_gamma(p.value)
        return p

    def check(self):
        """Checks the dependencies (referenced variables are declared). Raises ValueError otherwise."""
        return self._m.check()

    def check_model(self, samples=None, n_samples=64, seed=0, aux=None, rtol=1e-8, atol=1e-10,
                    raise_on_error=True, jac_rtol=1e-3, jac_atol=1e-9):
        """Generic NUMERICAL verification of the model (finite flux/source/elliptic, real and finite
        eigenvalues, wave_speeds/max_wave_speed consistency, non-circular bounding of the spectrum by
        the dense Jacobian, cons<->prim round-trip, positivity of Density/'p') on sample states.
        Delegates to HyperbolicModel.check_model (cf. its doc). To be called BEFORE compile(); the
        runtime counterpart of an installed block is System.check_model."""
        return self._m.check_model(samples=samples, n_samples=n_samples, seed=seed, aux=aux,
                                   rtol=rtol, atol=atol, raise_on_error=raise_on_error,
                                   jac_rtol=jac_rtol, jac_atol=jac_atol)

    # --- introspection (read-only, delegated to the backing model) ---
    @property
    def cons_names(self): return self._m.cons_names

    @property
    def prim_state(self): return self._m.prim_state

    @property
    def n_vars(self): return self._m.n_vars

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

        - ``include`` None -> auto-detected (adc_include()); passing include= remains possible;
        - ``so_path`` None -> .so in an out-of-source cache (adc_cache_dir()), file name keyed on
          model_hash (PARAMS INCLUDED) + abi_key (+ backend/target/name). Cache HIT (.so already present)
          -> reuse without recompilation; cache MISS (model/param/toolchain change) ->
          recompilation + storage. Passing so_path= forces that path and recompiles (backward-compat).

        Returns a CompiledModel carrying so_path, backend, target, adder, names/roles/gamma/n_aux/params,
        caps, abi_key, model_hash, cxx, std."""
        import os
        import shutil
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
        # native AND aot (mode "compile") compile the adc headers -> real Kokkos (compiler +
        # kokkos feature-key) so that the cache key MATCHES the produced .so (cf. compile_aot).
        kokkos_like = mode in ("native", "compile")
        eff_cxx = _native_kokkos_compiler(cxx) if kokkos_like else _default_cxx(cxx)
        if include is None:  # ergonomics: auto-detection of the adc headers folder
            include = adc_include()

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
            # HyperbolicModel.compile: the loader emits adc_install_native_amr for target="amr_system").
            out_path = m.compile(so_path, include, backend=backend, name=name, cxx=cxx, std=std,
                                 require_metadata=require_metadata, target=target,
                                 hoist_reciprocals=hoist_reciprocals)

        adder = HyperbolicModel.adder_for(backend)
        cons_roles = roles_for(m.cons_names, m.cons_roles)
        cm = CompiledModel(
            so_path=out_path, backend=backend, adder=adder, target=target,
            cons_names=m.cons_names, cons_roles=cons_roles, prim_names=m.prim_state,
            n_vars=m.n_vars, gamma=m.gamma, n_aux=aux_total_n_aux(m.aux_names, m.aux_extra_names),
            params=self.params, caps=_BACKEND_CAPS[backend],
            abi_key=abi_key, model_hash=model_hash,
            cxx=eff_cxx, std=eff_std, hllc=m._hllc,
            roe=(m._roe or getattr(m, '_roe_rows', None) is not None),
            aux_extra_names=m.aux_extra_names,
            wave_speeds=(m._wave_speeds is not None or m._ws_jacobian is not None
                         or "p" in m.prim_defs))
        # Trace of the 'auto' policy (ADC-63): None if the backend was explicit. Diagnostic,
        # never a silent choice -- cm.backend says what was built, this says WHY.
        cm.backend_auto_reason = auto_reason
        return cm


# === Phase B (prototype): HYBRID composition of a native brick + a DSL brick in ONE model ========
# Until now, mixing native and DSL was done at the SYSTEM level (a native add_block block + a DSL
# add_equation block). One could not mix a native brick and a DSL brick INSIDE A SINGLE
# model. This prototype allows it, in BOTH directions (DSL transport + native source/elliptic, AND
# native transport + DSL source/elliptic).
#
# ARCHITECTURE (option B): the C++ core already COMPOSES heterogeneous brick types effortlessly --
# adc::CompositeModel<Hyperbolic, Source, Elliptic> accepts any type conforming to its slot,
# native (include/adc/physics/) or generated by the DSL, and physics/bricks.hpp is already included by all
# the backends. The mix is therefore generated at the compilation of the final COMPOSITE: a single .so, on the
# SAME path as full DSL models (inlined numerics, identical to a native block). No .so
# per brick nor a partial virtual ABI (that would be option A: host virtual dispatch, without GPU
# inlining -- discarded).
#
# The only subtlety: the backends carry only the model TYPE (default-constructed), so
# the PARAMETERS of a native brick (qom, q, cs2...) must be BAKED into the type. For this we emit
# a small derived struct that fixes the public fields in its constructor (host):
#   namespace adc_generated { struct NatSrc : adc::PotentialForce { NatSrc() { qom = adc::Real(-1.0); } }; }
# It inherits EXACTLY the native numerics (true native path, zero re-derivation) and satisfies the
# slot contract. Without a parameter -> a simple `using` alias.
#
# PROTOTYPE state: "aot" backend (add_compiled_block, self-sufficient .so, production host-marshaled
# path). The hybrid "production" (native zero-copy) and "prototype" (JIT) backends, the
# fine propagation of roles/gamma/n_aux and the amr_system target arrive in the following PRs.


class NativeBrick:
    """Descriptor of a NATIVE brick (include/adc/physics/) for hybrid composition.

    Carries the C++ type of the brick, the PARAMETERS to bake into the type (public field -> value) and,
    for a hyperbolic brick, the variable layout (conservative names, n_vars, primitives,
    gamma). emit(struct_name) returns the C++ text to sew into the composite .so: a derived struct that
    fixes the parameters (or a simple `using` alias if the brick has no parameter).

    - ``kind``: 'hyperbolic' | 'source' | 'elliptic' (target slot).
    - ``fields``: dict {public C++ field name -> value}; ORDER preserved (insertion).
    - ``var_names`` / ``n_vars`` / ``prim_names`` / ``gamma``: layout metadata (hyperbolic
      slot only).
    - ``min_vars``: minimal number of variables that a TEMPLATED brick (source/elliptic) requires;
      e.g. PotentialForce indexes s[1]/s[2] so it requires >= 3 variables. Checked by HybridModel.
    - ``n_aux``: width of the aux channel that the brick READS (>= 3 if it reads B_z/T_e)."""

    def __init__(self, cpp_type, kind, fields=None, var_names=None, n_vars=None, prim_names=None,
                 gamma=None, min_vars=1, n_aux=AUX_BASE_COMPS):
        if kind not in ("hyperbolic", "source", "elliptic"):
            raise ValueError("NativeBrick: kind 'hyperbolic' | 'source' | 'elliptic' (got %r)" % (kind,))
        self.cpp_type = cpp_type
        self.kind = kind
        self.fields = dict(fields or {})
        self.var_names = list(var_names) if var_names else None
        self.n_vars = n_vars
        self.prim_names = list(prim_names) if prim_names else (list(var_names) if var_names else None)
        self.gamma = gamma
        self.min_vars = min_vars
        self.n_aux = n_aux

    def emit(self, struct_name, namespace="adc_generated"):
        """C++ text of the brick sewn into the composite .so. Without a parameter -> `using` alias
        (zero cost); with parameters -> a derived struct that fixes them in its host constructor
        (the values are WRITTEN HARD, like an inlined DSL constant)."""
        if not self.fields:
            return "namespace %s { using %s = %s; }\n" % (namespace, struct_name, self.cpp_type)
        sets = " ".join("%s = adc::Real(%s);" % (k, repr(float(v))) for k, v in self.fields.items())
        return ("namespace %s { struct %s : %s { %s() { %s } }; }\n"
                % (namespace, struct_name, self.cpp_type, struct_name, sets))


class CompiledBrick:
    """Result of <partial DSL brick>.compile(): the C++ of ONE brick (the generated struct) + its
    metadata, ready to be sewn into a hybrid CompositeModel. The MACHINE compilation happens at the
    level of the composite (a single .so); this object carries the brick already GENERATED and frozen."""

    def __init__(self, kind, struct_src, type_name, n_vars=None, n_aux=AUX_BASE_COMPS,
                 cons_names=None, cons_roles=None, prim_names=None, gamma=None, hash_part="",
                 wave_speeds=True):
        self.kind = kind                 # 'hyperbolic' | 'source' | 'elliptic'
        self.struct_src = struct_src     # C++ text of the struct (namespace adc_generated { struct ... })
        self.type_name = type_name       # qualified type to place in CompositeModel<...>
        self.n_vars = n_vars             # layout (hyperbolic) or declared number of variables (src/ell)
        self.n_aux = n_aux
        self.cons_names = list(cons_names) if cons_names else []
        self.cons_roles = list(cons_roles) if cons_roles else []
        self.prim_names = list(prim_names) if prim_names else []
        self.gamma = gamma
        self.hash_part = hash_part       # stable hash slice (formulas) for the composite cache key
        # wave_speeds emitted by the struct (DSL hyperbolic brick: 'p' OR explicit pair); True by
        # default = unknown (native brick): we let the C++ requires-gate decide (historical).
        self.has_wave_speeds = bool(wave_speeds)

    def __repr__(self):
        return "CompiledBrick(kind=%r, type=%r, n_vars=%r)" % (self.kind, self.type_name, self.n_vars)


class CompiledHyperbolicBrick(CompiledBrick):
    """Compiled DSL hyperbolic brick (vars/flux/eigenvalues/conversions)."""
    def __init__(self, **kw): super().__init__("hyperbolic", **kw)


class CompiledSourceBrick(CompiledBrick):
    """Compiled DSL source brick (apply(U, aux))."""
    def __init__(self, **kw): super().__init__("source", **kw)


class CompiledEllipticBrick(CompiledBrick):
    """Compiled DSL elliptic right-hand side brick (rhs(U))."""
    def __init__(self, **kw): super().__init__("elliptic", **kw)


class HyperbolicBrick:
    """PARTIAL hyperbolic DSL brick (variables/flux/eigenvalues/conversions), composable with
    native or DSL bricks for the source and the elliptic. Same surface as dsl.Model but limited
    to the hyperbolic slot. compile() -> CompiledHyperbolicBrick."""

    def __init__(self, name):
        self._m = HyperbolicModel(name)

    @property
    def name(self): return self._m.name

    def conservative_vars(self, *names, roles=None):
        return self._m.conservative_vars(*names, roles=roles)

    def primitive(self, name, expr):
        return self._m.primitive(name, expr)

    def primitive_vars(self, *vars, roles=None):
        """ORDERED layout of Prim (positional form, names/Var already defined)."""
        self._m.set_primitive_state(*vars, roles=roles)

    def aux(self, name): return self._m.aux(name)
    def flux(self, x, y): self._m.set_flux(x, y)
    def eigenvalues(self, x, y): self._m.set_eigenvalues(x, y)

    def wave_speeds(self, x, y):
        """Explicit SIGNED wave speeds (smin, smax) per direction, WITHOUT requiring 'p' --
        same contract as Model.wave_speeds (the brick struct goes through emit_cpp_brick, which
        emits wave_speeds from the pair ; the hybrid CompositeModel forwards it to the HLL gate)."""
        self._m.set_wave_speeds(x, y)

    def conservative_from(self, exprs): self._m.set_conservative_from(exprs)
    def gamma(self, value): self._m.set_gamma(value)
    def check(self): return self._m.check()

    def compile(self):
        """Validate + emit the hyperbolic C++ struct (emit_cpp_brick) -> CompiledHyperbolicBrick."""
        self._m.check()
        struct_name = "Hyp" + self._m.name.capitalize()
        struct_src = self._m.emit_cpp_brick(name=struct_name)
        return CompiledHyperbolicBrick(
            struct_src=struct_src, type_name="adc_generated::" + struct_name,
            n_vars=self._m.n_vars, cons_names=list(self._m.cons_names),
            cons_roles=roles_for(self._m.cons_names, self._m.cons_roles),
            prim_names=list(self._m.prim_state), gamma=self._m.gamma,
            n_aux=aux_n_aux(self._m.aux_names), hash_part=self._m._model_hash(),
            wave_speeds=("p" in self._m.prim_defs or self._m._wave_speeds is not None))


class SourceBrick:
    """PARTIAL DSL brick for a source S(U, aux), composable with a native or DSL transport and
    elliptic. Declares its conservatives (the layout must match the transport) + its aux fields
    + the source formula. compile() -> CompiledSourceBrick."""

    def __init__(self, name):
        self._m = HyperbolicModel(name)

    def conservative_vars(self, *names, roles=None):
        return self._m.conservative_vars(*names, roles=roles)

    def primitive(self, name, expr): return self._m.primitive(name, expr)
    def aux(self, name): return self._m.aux(name)
    def source(self, s): self._m.set_source(s)

    def compile(self):
        """Validate + emit the source C++ struct (emit_cpp_source) -> CompiledSourceBrick."""
        if self._m._source is None:
            raise ValueError("SourceBrick.compile: call source([...]) first")
        struct_name = "Src" + self._m.name.capitalize()
        struct_src = self._m.emit_cpp_source(name=struct_name)
        return CompiledSourceBrick(
            struct_src=struct_src, type_name="adc_generated::" + struct_name,
            n_vars=self._m.n_vars, n_aux=aux_n_aux(self._m.aux_names),
            hash_part=self._m._model_hash())


class EllipticBrick:
    """PARTIAL DSL brick for an elliptic right-hand side rhs(U), composable with a native or DSL
    transport and source. Declares its conservatives (layout) + the right-hand side formula.
    compile() -> CompiledEllipticBrick."""

    def __init__(self, name):
        self._m = HyperbolicModel(name)

    def conservative_vars(self, *names, roles=None):
        return self._m.conservative_vars(*names, roles=roles)

    def primitive(self, name, expr): return self._m.primitive(name, expr)
    def elliptic_rhs(self, e): self._m.set_elliptic_rhs(e)

    def compile(self):
        """Validate + emit the elliptic C++ struct (emit_cpp_elliptic) -> CompiledEllipticBrick."""
        if self._m._elliptic is None:
            raise ValueError("EllipticBrick.compile: call elliptic_rhs(...) first")
        struct_name = "Ell" + self._m.name.capitalize()
        struct_src = self._m.emit_cpp_elliptic(name=struct_name)
        return CompiledEllipticBrick(
            struct_src=struct_src, type_name="adc_generated::" + struct_name,
            n_vars=self._m.n_vars, n_aux=AUX_BASE_COMPS, hash_part=self._m._model_hash())


class HybridModel:
    """Composer of a HYBRID model: three slots (transport, source, elliptic), each provided by a
    NATIVE brick (NativeBrick) or a DSL brick (CompiledBrick). Assembles a mixed adc::CompositeModel<...>
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
                     type_name="adc_generated::" + native_struct_name,
                     n_vars=prov.n_vars, min_vars=prov.min_vars, n_aux=prov.n_aux)
            if role == "hyperbolic":
                names = prov.var_names or []
                d.update(cons_names=list(names), cons_roles=roles_for(names),
                         prim_names=list(prov.prim_names or names), gamma=prov.gamma,
                         wave_speeds=True)  # native: unknown, the C++ requires-gate decides (historical)
            return d
        raise TypeError("HybridModel: slot %r must be a native brick (adc.* / NativeBrick) or a "
                        "compiled DSL brick (CompiledBrick) ; got %r" % (role, type(prov).__name__))

    def _emit_aot_source(self):
        """C++ source of the hybrid composite .so, behind the extern \"C\" ABI of compiled_block_abi.hpp
        (aot backend: same flat ABI as emit_cpp_aot_source). The bricks (generated DSL or native binding
        structs) are stitched together, then assembled into adc::CompositeModel<...>."""
        hyp, src, ell = self._slots
        parts = ['#include <adc/runtime/compiled_block_abi.hpp>\n',
                 '#include <adc/physics/bricks.hpp>\n',   # CompositeModel + native bricks
                 '#include <adc/core/variables.hpp>\n']   # ADC_EXPORT_BLOCK_METADATA / _GAMMA
        for slot in self._slots:
            if slot["struct_text"]:
                parts.append(slot["struct_text"])
        parts.append('\nnamespace adc_generated { using AotModel = adc::CompositeModel<%s, %s, %s>; }\n'
                     % (hyp["type_name"], src["type_name"], ell["type_name"]))
        parts.append('ADC_DEFINE_COMPILED_BLOCK(adc_generated::AotModel)\n')
        parts.append('ADC_EXPORT_BLOCK_METADATA(adc_generated::AotModel)\n')
        if self.gamma is not None:
            parts.append('ADC_EXPORT_BLOCK_GAMMA(%r)\n' % self.gamma)
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
        composite = ("adc::CompositeModel<%s, %s, %s>"
                     % (hyp["type_name"], src["type_name"], ell["type_name"]))
        return bricks, composite

    def _emit_metadata(self, alias):
        """ABI metadata symbols (names/roles from conservative_vars, optional gamma), SHARED
        by the backends. @p alias: an alias WITHOUT a top-level comma (the preprocessor splits
        macro arguments on commas)."""
        out = '\nADC_EXPORT_BLOCK_METADATA(%s)\n' % alias
        if self.gamma is not None:
            out += 'ADC_EXPORT_BLOCK_GAMMA(%r)\n' % self.gamma
        return out

    def _emit_jit_source(self):
        """Source of the JIT library (backend 'prototype'): the hybrid composite behind an
        extern \"C\" factory (adc_make_model via adc::ModelAdapter). Host VIRTUAL dispatch (order-1
        Rusanov residual): fast iteration, to be plugged via System.add_dynamic_block. Hybrid
        counterpart of emit_cpp_so_source."""
        bricks, composite = self._bricks_and_composite()
        return ('#include <adc/runtime/dynamic_model.hpp>\n'
                '#include <adc/physics/bricks.hpp>\n'
                '#include <adc/core/variables.hpp>\n'
                + bricks
                + '\nnamespace adc_generated { using JitModel = %s; }\n' % composite
                + 'extern "C" int adc_model_nvars() { return %d; }\n' % self.n_vars
                + 'extern "C" void* adc_make_model() { return new adc::ModelAdapter<adc_generated::JitModel>(); }\n'
                + 'extern "C" void adc_destroy_model(void* p) { delete static_cast<adc::IModel<%d>*>(p); }\n'
                % self.n_vars
                + self._emit_metadata("adc_generated::JitModel"))

    def _emit_native_source(self, target="system"):
        """C++ source of the NATIVE LOADER (backend 'production'): the hybrid composite as CompositeModel<...>
        behind a THIN extern \"C\" ABI. Like emit_cpp_native_loader, the .so does NOT carry the
        numerics: it INSTALLS the generated model as a native block of the facade via add_compiled_model<>,
        which builds the closures on the REAL CONTEXT of the facade -> same path as add_block,
        ZERO-COPY (MPI by construction, device-clean). adc_native_abi_key() freezes the ABI key at
        compile time, compared against the module's abi_key() by add_native_block (explicit rejection if
        headers/compiler/std diverge).

        @p target: 'system' -> adc_install_native (System&, evolve) ; 'amr_system' ->
        adc_install_native_amr (AmrSystem&, without evolve) inline add_compiled_model(AmrSystem&): the block
        runs the SAME AMR hierarchy as AmrSystem.add_block (reflux, regrid). DISTINCT symbols per
        target (a System loader is not pluggable onto AmrSystem.add_native_block, and vice versa)."""
        if target not in ("system", "amr_system"):
            raise ValueError("_emit_native_source: target 'system' | 'amr_system' (got %r)" % (target,))
        bricks, composite = self._bricks_and_composite()
        head = ('#include <adc/runtime/abi_key.hpp>\n'        # ADC_ABI_KEY_LITERAL (key frozen at compile time)
                '#include <adc/physics/bricks.hpp>\n'         # CompositeModel + native bricks
                '#include <adc/core/variables.hpp>\n'
                '#include <string>\n')
        # Header template of the target (selective: do not pull the AMR machinery into a System loader).
        head += ('#include <adc/runtime/dsl_block.hpp>\n' if target == "system"
                 else '#include <adc/runtime/amr_dsl_block.hpp>\n')
        # Preprocessor LITERAL, no call to abi_key_string(): an inline would be interposed
        # (ELF/RTLD_GLOBAL) toward the module's copy -> module key returned -> tautological guard.
        key = ('#if defined(_WIN32)\n'
               '#define ADC_LOADER_API extern "C" __declspec(dllexport)\n'
               '#else\n'
               '#define ADC_LOADER_API extern "C"\n'
               '#endif\n'
               'ADC_LOADER_API const char* adc_native_abi_key() {\n'
               '  return ADC_ABI_KEY_LITERAL;\n'
               '}\n')
        if target == "system":
            # pos_floor (ADC-76, Zhang-Shu positivity limiter): final flat argument, marshaled
            # down to the loader's make_block via add_compiled_model. Old signature = old .so =
            # rejected by the ABI key (the headers changed), never a wrong argument layout.
            install = ('ADC_LOADER_API void adc_install_native(void* sys, const char* name, const char* limiter,\n'
                       '                                    const char* riemann, const char* recon,\n'
                       '                                    const char* time, double gamma, int substeps,\n'
                       '                                    int evolve, int stride, double pos_floor) {\n'
                       '  adc::System* s = reinterpret_cast<adc::System*>(sys);\n'
                       '  adc::add_compiled_model<adc_generated::ProdModel>(*s, name, adc_generated::ProdModel{},\n'
                       '                                                    limiter, riemann, recon, time, gamma,\n'
                       '                                                    substeps, evolve != 0, stride,\n'
                       '                                                    pos_floor);\n'
                       '}\n')
        else:  # amr_system: AmrSystem overload (no evolve parameter, mono-block AMR)
            install = ('ADC_LOADER_API void adc_install_native_amr(void* sys, const char* name,\n'
                       '                                        const char* limiter, const char* riemann,\n'
                       '                                        const char* recon, const char* time,\n'
                       '                                        double gamma, int substeps) {\n'
                       '  adc::AmrSystem* s = reinterpret_cast<adc::AmrSystem*>(sys);\n'
                       '  adc::add_compiled_model<adc_generated::ProdModel>(*s, name, adc_generated::ProdModel{},\n'
                       '                                                    limiter, riemann, recon, time, gamma,\n'
                       '                                                    substeps);\n'
                       '}\n')
        return (head + bricks
                + '\nnamespace adc_generated { using ProdModel = %s; }\n' % composite
                + key + install + self._emit_metadata("adc_generated::ProdModel"))

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
        inlines add_compiled_model(AmrSystem&) (symbol adc_install_native_amr), the only AMR .so path ; to be
        plugged via AmrSystem.add_equation. The other backends have no AMR counterpart.

        so_path None -> out-of-source cache (key = model_hash + abi_key + backend + target)."""
        import os
        import shutil
        import subprocess
        import sys
        import tempfile
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
            include = adc_include()
        if std is None:  # the native loader shares the module ABI (std derived from the loader: c++20 under
            # Kokkos, c++23 otherwise, cf. loader_cxx_std/compile_native); jit/aot stay at c++20.
            std = loader_cxx_std() if mode == "native" else "c++20"
        # NATIVE (production) AND AOT: compiler following the Kokkos backend (g++ by default,
        # nvcc_wrapper if explicit), Kokkos flags without linking libkokkos (single runtime), feature-key
        # kokkos in the cache. KOKKOS-ONLY: the hybrid aot includes the adc headers
        # (compiled_block_abi.hpp -> multifab/for_each) which require ADC_HAS_KOKKOS, same flags as
        # compile_aot; only the jit (prototype) stays pure host (-O2, dynamic_model/bricks without
        # multifab). kokkos_like also serves the cache key.
        native = (mode == "native")
        kokkos_like = native or mode == "aot"
        if mode == "aot" and _native_kokkos_root() is None:
            raise RuntimeError(
                "HybridModel.compile: adc_cpp is Kokkos-only -- the AOT model includes the adc "
                "headers which require Kokkos. Point to an installed Kokkos via ADC_KOKKOS_ROOT (or "
                "Kokkos_ROOT), e.g. `export ADC_KOKKOS_ROOT=/path/to/kokkos` (Serial is enough "
                "on CPU).")
        if native:  # pre-dlopen guard: headers != build of _adc -> clear remedy (cf. compile_native)
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
                return self._compiled_model(so_path, backend, target, abi_key, model_hash, eff_cxx, std)

        optflags = os.environ.get("ADC_DSL_OPTFLAGS", "-O3 -DNDEBUG").split() if native else ["-O2"]
        flags = ["-shared", "-fPIC", "-std=" + std, *optflags]
        kokkos_link_flags = []
        if mode == "jit":
            source = self._emit_jit_source()
        elif mode == "aot":
            # Like compile_aot: Kokkos flags without linking libkokkos (the _adc module has already loaded the
            # runtime, singleton), undefined symbols resolved at load time; Apple-ld then requires
            # -undefined dynamic_lookup (on ELF/Linux -shared already allows them).
            source = self._emit_aot_source()
            kokkos_compile_flags, kokkos_link_flags = _native_kokkos_flags()
            flags += kokkos_compile_flags
            if sys.platform == "darwin":
                flags += ["-undefined", "dynamic_lookup"]
        else:  # native: header signature + Kokkos backend parity (cf. compile_native / _native_kokkos_flags)
            source = self._emit_native_source(target=target)  # undefined symbols resolved at load time (_adc module)
            flags.append('-DADC_HEADER_SIG="%s"' % adc_header_signature(include))
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
        return self._compiled_model(so_path, backend, target, abi_key, model_hash, eff_cxx, std)

    def _compiled_model(self, so_path, backend, target, abi_key, model_hash, cxx, std):
        return CompiledModel(
            so_path=so_path, backend=backend, adder=HyperbolicModel.adder_for(backend),
            target=target, cons_names=self.cons_names, cons_roles=self.cons_roles,
            prim_names=self.prim_names, n_vars=self.n_vars, gamma=self.gamma, n_aux=self.n_aux,
            params={}, caps=_BACKEND_CAPS[backend], abi_key=abi_key, model_hash=model_hash,
            cxx=cxx, std=std, hllc=getattr(self, "_hllc", False),
            roe=getattr(self, "_roe", False),
            wave_speeds=getattr(self, "_has_wave_speeds", True))


# --- Generic COUPLED inter-species source (P5 phase 1, EXPLICIT splitting) -----------------------
#
# adc.dsl.CoupledSource describes an ARBITRARY coupling between species as FORMULAS, beyond the named
# couplings (Ionization / Collision / ThermalExchange) which freeze a formula. We read fields (block,
# role) as INPUT and WRITE source terms (block, role) given by symbolic expressions
# (same Expr as the models DSL: +, *, -, /, **, sqrt, params). compile(backend) compiles each
# expr into postfix BYTECODE (stack machine) that C++ evaluates in the same for_each_cell device as the
# named couplings -- NO .so nor Python callback per cell. Applied AFTER transport (split).

# Inverse of adc::role_name: DSL role name (CamelCase, cf. CANONICAL_ROLES) -> canonical lowercase
# name expected by adc::role_from_name (C++ boundary). Single source of the correspondence.
_ROLE_TO_CANONICAL = {
    "Density": "density",
    "MomentumX": "momentum_x", "MomentumY": "momentum_y", "MomentumZ": "momentum_z",
    "Energy": "energy",
    "VelocityX": "velocity_x", "VelocityY": "velocity_y", "VelocityZ": "velocity_z",
    "Pressure": "pressure", "Temperature": "temperature", "Scalar": "scalar",
}


def _role_canonical(role):
    """Canonical role name (lowercase, C++ boundary) for a DSL role. Accepts already-canonical."""
    if role in _ROLE_TO_CANONICAL:
        return _ROLE_TO_CANONICAL[role]
    if role in _ROLE_TO_CANONICAL.values():
        return role
    raise ValueError("CoupledSource: unknown role %r (roles: %s)"
                     % (role, ", ".join(sorted(_ROLE_TO_CANONICAL))))


# Stack machine opcodes: MIRROR of adc::CsOp (coupled_source_program.hpp). FROZEN values
# (transported as-is by the Python -> C++ ABI).
_CS_PUSHREG = 0
_CS_ADD = 1
_CS_SUB = 2
_CS_MUL = 3
_CS_DIV = 4
_CS_NEG = 5
_CS_POW = 6
_CS_SQRT = 7

# FROZEN capacities, mirror of coupled_source_program.hpp (kCsMaxReg / kCsMaxTerms / kCsMaxProg). We
# diagnose on the Python side (clear error) before reaching the C++ boundary.
_CS_MAX_REG = 32
_CS_MAX_TERMS = 16
_CS_MAX_PROG = 256


class _CsField(Var):
    """Symbolic handle of a (block, role): it is a Var (hence a full Expr) whose environment NAME
    '<block>::<role>' indexes both the numpy eval (env) and the input register at the
    bytecode codegen. Subclassing Var directly gives operators / _wrap / to_cpp / eval / deps, so
    `+k * ne * ng` builds the expected Expr tree without delegation."""

    def __init__(self, block, role):
        super().__init__("%s::%s" % (block, role), "coupled_field")
        self.block = block
        self.role = role

    def __repr__(self): return "_CsField(%r, %r)" % (self.block, self.role)


class _CsBlock:
    """Construction helper: `src.block("electrons").role("density")` -> _CsField. Records the
    (block, role) requested on the source to fix the order of the input registers."""

    def __init__(self, src, name):
        self._src = src
        self.name = name

    def role(self, role):
        return self._src._field(self.name, role)


class CompiledCoupledSource:
    """Result of CoupledSource.compile(...): packages the FLAT ABI (bytecode) ready for
    System.add_coupled_source, + the REFERENCE numpy evaluator (same Expr) for tests / a Python
    integrator. No .so: the coupling is interpreted on the C++ side (device stack machine)."""

    def __init__(self, name, backend, in_blocks, in_roles, consts, out_blocks, out_roles,
                 prog_ops, prog_args, prog_lens, terms, reg_order, frequency=0.0,
                 freq_prog_ops=None, freq_prog_args=None, frequency_expr=None):
        self.name = name
        self.backend = backend
        self.frequency = float(frequency)  # mu [1/s] declared CONSTANT (0 = no constant bound)
        self.in_blocks = list(in_blocks)
        self.in_roles = list(in_roles)        # canonical (lowercase, C++ boundary)
        self.consts = list(consts)
        self.out_blocks = list(out_blocks)
        self.out_roles = list(out_roles)      # canonical
        self.prog_ops = list(prog_ops)
        self.prog_args = list(prog_args)
        self.prog_lens = list(prog_lens)
        # PER-CELL frequency mu(U): bytecode (same inputs/constants as the source). EMPTY =
        # constant frequency only. Transported to System/AmrSystem.add_coupled_source.
        self.freq_prog_ops = list(freq_prog_ops) if freq_prog_ops else []
        self.freq_prog_args = list(freq_prog_args) if freq_prog_args else []
        self._frequency_expr = frequency_expr  # reference Expr (numpy eval for tests); None if constant
        self._terms = list(terms)             # [(block, role_canonical, Expr)]: numpy reference
        self._reg_order = list(reg_order)     # env names '<block>::<role>' in register order

    def __repr__(self):
        return ("CompiledCoupledSource(name=%r, backend=%r, n_in=%d, n_const=%d, n_terms=%d)"
                % (self.name, self.backend, len(self.in_blocks), len(self.consts),
                   len(self.out_blocks)))

    def reference_terms(self, fields):
        """Evaluates the source terms on numpy arrays (REFERENCE for tests). @p fields:
        dict (block, role_canonical) -> array; returns [(block, role_canonical, dS)] with dS = S
        (the evaluated symbolic term), BEFORE multiplication by dt. Same Expr as the C++ codegen."""
        env = {}
        for (block, role), arr in fields.items():
            env["%s::%s" % (block, _role_canonical(role))] = arr
        return [(b, r, e.eval(env)) for (b, r, e) in self._terms]

    def reference_frequency(self, fields):
        """Evaluates the PER-CELL frequency mu(U) on numpy arrays (REFERENCE for tests):
        same Expr / register table as the C++ bytecode. @p fields: dict (block, role_canonical) ->
        array; returns the mu array (same formulas). Returns None if the frequency is CONSTANT
        (no Expr) -- use .frequency in that case."""
        if self._frequency_expr is None:
            return None
        env = {}
        for (block, role), arr in fields.items():
            env["%s::%s" % (block, _role_canonical(role))] = arr
        return self._frequency_expr.eval(env)


class CoupledSource:
    """Generic COUPLED inter-species source (adc.dsl), Phase 1 (EXPLICIT splitting). Reuses the
    Expr of the models DSL for the source formulas; no coupling is hard-coded.

        src = adc.dsl.CoupledSource("ionization")
        ne = src.block("electrons").role("density")
        ng = src.block("neutrals").role("density")
        k  = src.param("Kiz", 1.0)
        src.add("electrons", role="density", expr=+k * ne * ng)
        src.add("neutrals",  role="density", expr=-k * ne * ng)
        sim.add_coupling(src.compile(backend="production"))

    compile(backend) -> CompiledCoupledSource: flat ABI (bytecode) consumed by
    System.add_coupled_source (C++ side: stack machine evaluated in a for_each_cell device, MPI-safe,
    named functor). The backend (production / prototype) does NOT change the numerics: the bytecode is
    interpreted on the C++ side in both cases (no .so per coupling); it is kept for introspection
    and API parity with the models DSL."""

    def __init__(self, name="coupled_source"):
        self.name = name
        self._fields = {}    # '<block>::<role>' -> _CsField (single input register per (block, role))
        self._reg_order = []  # order of appearance of the input fields (-> register order)
        self._params = {}    # name -> Param
        self._terms = []     # [(block, role_canonical, Expr)]
        # Indices (in self._terms) of the terms EMITTED BY add_pair, by pair: [(idx_gain, idx_loss)].
        # add_pair guarantees that the two terms carry EXACTLY the same evaluated Expr, one +expr,
        # the other -expr (Neg) -> conservative exchange by construction. verify_conservation=True
        # revisits them at compile time to CHECK the property (and detect a breach on the manual add side).
        self._pairs = []
        self._frequency = 0.0  # mu [1/s] declared CONSTANT (coupling step bound; 0 = no bound)
        # optional PER-CELL mu(U): an Expr (same vocabulary as the terms: block().role() +
        # param()) emitted into bytecode at compile() against the SAME register table. None = constant.
        self._frequency_expr = None

    def frequency(self, mu):
        """Declared coupling FREQUENCY mu [1/s] (vague 3 audit): step bound dt <= cfl / mu
        aggregated by System/AmrSystem::step_cfl (reason 'coupled_source:<name>'). Couplings are
        applied ONCE per MACRO-step (splitting, apply_couplings(dt)): the bound applies to the
        macro-dt, WITHOUT a substeps/stride factor. mu <= 0 = no bound (historical).

        @p mu accepts TWO forms:
          - a number (float / int) -> CONSTANT frequency (historical path, bit-identical);
          - an Expr of the SAME vocabulary as the terms (fields block().role() + param()) ->
            PER-CELL frequency mu(U), emitted into bytecode at compile() and evaluated per cell on the
            C++ side (MAX + all_reduce_max -> dt <= cfl / max(mu)). The referenced fields MUST be
            declared via .block(...).role(...) (as for the terms); otherwise compile() raises an
            EXPLICIT ValueError (field used without .block(...).role(...)).

        Returns self (chainable)."""
        # An Expr/_CsField/Param -> per-cell frequency (bytecode); a scalar -> constant.
        if isinstance(mu, (int, float)) and not isinstance(mu, bool):
            self._frequency = float(mu)
            self._frequency_expr = None
        else:
            self._frequency_expr = _wrap(mu)  # Expr / _CsField / Param -> tree node (cf. _wrap)
            self._frequency = 0.0             # the bytecode carries the bound; no duplicate constant
        return self

    # --- symbolic construction ----------------------------------------------------------------
    def block(self, name):
        """Handle of a block: .role(role) derives a symbolic field (block, role) from it."""
        return _CsBlock(self, name)

    def _field(self, block, role):
        canon = _role_canonical(role)
        key = "%s::%s" % (block, canon)
        if key not in self._fields:
            f = _CsField(block, canon)
            self._fields[key] = f
            self._reg_order.append(key)
        return self._fields[key]

    def param(self, name, value):
        """NAMED constant parameter, usable like an Expr (inlines as a real in the bytecode)."""
        p = Param(name, value, kind="const")
        self._params[name] = p
        return p

    def add(self, block, role=None, expr=None):
        """Adds a source TERM: d_t (block.role) += expr. @p expr is an Expr / _CsField / Param /
        number. Several adds on the same (block, role) ADD UP (sum of source terms)."""
        if role is None:
            raise ValueError("CoupledSource.add: role= required")
        if expr is None:
            raise ValueError("CoupledSource.add: expr= required")
        e = expr if isinstance(expr, Expr) else _wrap(expr)  # _CsField / Var are already Expr; Param/scalar -> _wrap
        self._terms.append((block, _role_canonical(role), e))
        return self

    def add_pair(self, block_a, block_b, role=None, expr=None):
        """Adds a CONSERVATIVE EXCHANGE of the quantity @p role between @p block_a and @p block_b, described
        by a SINGLE expression @p expr (Expr / _CsField / Param / number).

        Sign convention (to remember): @p block_a GAINS +expr, @p block_b LOSES -expr, on the SAME
        evaluated value of @p expr. In other words:

            d_t (block_a.role) += +expr
            d_t (block_b.role) += -expr

        It is the DSL equivalent of the NAMED C++ couplings (add_collision / add_thermal_exchange), which
        compute ONE value and apply it with two opposite signs: the sum over the two blocks of the
        exchanged term is zero at each cell and at each step, so the total quantity sum(role) over
        (block_a, block_b) is CONSERVED by construction -- independently of the chosen formula, the dt
        and the state. Choose @p expr >= 0 for a transfer from B to A (A gains, B loses); a negative
        sign of expr simply reverses the transfer direction (conservation holds in all cases).

        Contrast with two hand-written .add(...) (+expr on A, -expr on B): add_pair guarantees that
        the TWO legs carry the SAME Expr (the second is exactly Neg of the first), whereas by hand
        nothing prevents writing two slightly different formulas by mistake -> conservation broken
        silently. add_pair removes this risk; compile(verify_conservation=True) also checks it
        for hand-written couplings.

        @p block_a and @p block_b must be distinct. add_pair is purely ADDITIVE on top of .add:
        the manual API stays available and unchanged. Returns self (chainable)."""
        if role is None:
            raise ValueError("CoupledSource.add_pair: role= required")
        if expr is None:
            raise ValueError("CoupledSource.add_pair: expr= required")
        if block_a == block_b:
            raise ValueError("CoupledSource.add_pair: block_a and block_b must be distinct "
                             "(received %r for both)" % (block_a,))
        canon = _role_canonical(role)
        gain = expr if isinstance(expr, Expr) else _wrap(expr)  # +expr (gaining leg)
        loss = Neg(gain)                                        # -expr: SAME subtree, opposite sign
        idx_gain = len(self._terms)
        self._terms.append((block_a, canon, gain))
        idx_loss = len(self._terms)
        self._terms.append((block_b, canon, loss))
        self._pairs.append((idx_gain, idx_loss))
        return self

    # --- bytecode codegen -----------------------------------------------------------------------
    def _emit_program(self, expr, reg_index):
        """Compile @p expr (Expr tree) into postfix bytecode (parallel ops/args lists) against the
        register table @p reg_index (env name '<bloc>::<role>' OR constant value -> register index).
        Constants (Const / inline Param) become a dedicated constant register. Postfix traversal
        (recursion over the tree structure): a Var pushes its register, a binary emits its two
        subtrees then the opcode, etc. -- exactly the semantics of CsProgram::eval on the C++ side."""
        ops, args = [], []

        def emit(node):
            if isinstance(node, Var):
                if node.name not in reg_index:
                    raise ValueError("CoupledSource: field %r used without .block(...).role(...)"
                                     % node.name)
                ops.append(_CS_PUSHREG); args.append(reg_index[node.name])
            elif isinstance(node, Const):
                ops.append(_CS_PUSHREG); args.append(self._const_reg(node.value, reg_index))
            elif isinstance(node, Neg):
                emit(node.a); ops.append(_CS_NEG); args.append(0)
            elif isinstance(node, Sqrt):
                emit(node.a); ops.append(_CS_SQRT); args.append(0)
            elif isinstance(node, Add):
                emit(node.a); emit(node.b); ops.append(_CS_ADD); args.append(0)
            elif isinstance(node, Sub):
                emit(node.a); emit(node.b); ops.append(_CS_SUB); args.append(0)
            elif isinstance(node, Mul):
                emit(node.a); emit(node.b); ops.append(_CS_MUL); args.append(0)
            elif isinstance(node, Div):
                emit(node.a); emit(node.b); ops.append(_CS_DIV); args.append(0)
            elif isinstance(node, Pow):
                emit(node.a); emit(node.b); ops.append(_CS_POW); args.append(0)
            else:
                raise TypeError("CoupledSource: expression node not supported in Phase 1: %r "
                                "(supported: +, -, *, /, **, unary -, sqrt, field, constant)"
                                % type(node).__name__)

        emit(expr)
        return ops, args

    def _const_reg(self, value, reg_index):
        """Register index of a constant @p value (deduplicated). Constants occupy the
        registers AFTER the input fields (cf. CoupledSourceKernel: r[n_in + c] = consts[c])."""
        key = ("const", float(value))
        if key not in reg_index:
            reg_index[key] = len(self._reg_order) + len(self._consts)
            self._consts.append(float(value))
        return reg_index[key]

    @staticmethod
    def _signed_key(expr):
        """SIGNED STRUCTURAL key of @p expr: (sign, body_key), where sign is +1 / -1 and
        body_key is the structural key (_key) of the expression stripped of ALL its leading Neg
        (the sign folds in each peeled Neg). Two expressions are structurally OPPOSITE iff they have
        the SAME body_key and opposite signs (e.g. E and Neg(E), or -E and Neg(-E)=E). Peeling
        ALL leading Neg makes the key robust to the +expr / -expr pair from add_pair EVEN when expr is
        already a Neg (add_pair sets loss = Neg(gain), hence one more Neg). We do NOT normalize the
        internal algebra (k*ne vs ne*k): at worst a false 'non-conservative' on differently written
        forms, NEVER a false 'conservative' (the check stays conservative, hence sound)."""
        sign = 1
        while isinstance(expr, Neg):
            sign = -sign
            expr = expr.a
        return (sign, _key(expr))

    def _verify_conservation(self):
        """Verify that, role by role, the sum of the source terms CANCELS structurally: each
        contribution +E on one block is compensated by a contribution -E (same structural body) on
        another block. Raises an EXPLICIT ValueError otherwise. This is exactly the property add_pair
        guarantees by construction; this check extends it to hand-written couplings (two .add) and detects
        a break (slightly different formulas, forgotten sign, orphan term). Purely symbolic
        (no numerical evaluation): same structural key as the codegen CSE."""
        from collections import Counter
        per_role = {}
        for (_block, role, expr) in self._terms:
            sign, body = self._signed_key(expr)
            # Signed counter per structural body: +1 for +E, -1 for -E. Everything cancels => conservative.
            c = per_role.setdefault(role, Counter())
            c[body] += sign
        offenders = []
        for role in sorted(per_role):
            for body, net in per_role[role].items():
                if net != 0:
                    offenders.append((role, body, net))
        if offenders:
            details = "; ".join(
                "role '%s': term %r not compensated (net=%+d)" % (role, body, net)
                for (role, body, net) in offenders)
            raise ValueError(
                "CoupledSource.compile(verify_conservation=True): NON-conservative coupling. "
                "Each contribution +E on one block must be compensated by -E (same expression) "
                "on another block (use add_pair to guarantee it). Uncompensated terms: "
                + details)

    def compile(self, backend="production", verify_conservation=False):
        """Compile the source into a CompiledCoupledSource (flat bytecode ABI). @p backend documents
        the intent (API parity with the model DSL); the numerics are identical (C++ interpreter).

        @p verify_conservation (opt-in, default False): SYMBOLIC check that the coupling conserves
        each quantity (role) -- the sum of the source terms of a same role cancels structurally
        (each +E compensated by a -E on another block). add_pair satisfies this property by
        construction; this mode extends it to hand-written couplings (two .add) and raises an EXPLICIT
        ValueError if a term is not compensated (divergent formula, forgotten sign, orphan term).
        Off by default: a deliberately NON-conservative coupling (net creation/destruction, e.g.
        ionization creating an e/i pair) stays legal without passing the flag."""
        if not self._terms:
            raise ValueError("CoupledSource.compile: no term (.add(...) required)")
        if verify_conservation:
            self._verify_conservation()
        # Register table: input fields first (order of appearance), constants next.
        reg_index = {key: i for i, key in enumerate(self._reg_order)}
        self._consts = []
        prog_ops, prog_args, prog_lens = [], [], []
        out_blocks, out_roles = [], []
        for (block, role, expr) in self._terms:
            ops, args = self._emit_program(expr, reg_index)
            if len(ops) > _CS_MAX_PROG:
                raise ValueError("CoupledSource: program of term (%s.%s) too long (%d > %d)"
                                 % (block, role, len(ops), _CS_MAX_PROG))
            prog_ops += ops
            prog_args += args
            prog_lens.append(len(ops))
            out_blocks.append(block)
            out_roles.append(role)
        # Optional PER-CELL FREQUENCY: its program is emitted AFTER the terms, against the SAME
        # register table (reg_index) and the SAME constant list (self._consts) -- the referenced fields
        # must be declared via .block().role() (otherwise _emit_program raises: field used
        # without .block(...).role(...)). The frequency's own constants are appended after those of the
        # terms; on the C++ side they occupy the same registers r[n_in ..] (CoupledFreqKernel loads them
        # like the source). Constant frequency (or none) -> empty program (historical path).
        freq_prog_ops, freq_prog_args = [], []
        if self._frequency_expr is not None:
            freq_prog_ops, freq_prog_args = self._emit_program(self._frequency_expr, reg_index)
            if len(freq_prog_ops) > _CS_MAX_PROG:
                raise ValueError("CoupledSource: frequency program too long (%d > %d)"
                                 % (len(freq_prog_ops), _CS_MAX_PROG))
        n_reg = len(self._reg_order) + len(self._consts)
        if n_reg > _CS_MAX_REG:
            raise ValueError("CoupledSource: too many registers (inputs + constants = %d > %d)"
                             % (n_reg, _CS_MAX_REG))
        if len(out_blocks) > _CS_MAX_TERMS:
            raise ValueError("CoupledSource: too many source terms (%d > %d)"
                             % (len(out_blocks), _CS_MAX_TERMS))
        in_blocks = [self._fields[key].block for key in self._reg_order]
        in_roles = [self._fields[key].role for key in self._reg_order]
        return CompiledCoupledSource(
            name=self.name, backend=backend, in_blocks=in_blocks, in_roles=in_roles,
            consts=list(self._consts), out_blocks=out_blocks, out_roles=out_roles,
            prog_ops=prog_ops, prog_args=prog_args, prog_lens=prog_lens,
            terms=self._terms, reg_order=self._reg_order, frequency=self._frequency,
            freq_prog_ops=freq_prog_ops, freq_prog_args=freq_prog_args,
            frequency_expr=self._frequency_expr)
