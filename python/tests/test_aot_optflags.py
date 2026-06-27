"""Optimization flags of the AOT backend (compile_aot): the AOT .so runs the SAME production path
as native, so it must be compiled with the SAME flags ($POPS_DSL_OPTFLAGS, default -O3 -DNDEBUG)
and not a hardcoded -O2 (at -O2 without -DNDEBUG the marshaled kernel is ~1.48x).

Three parts:
(a) COMMAND LINE (hermetic, no compiler nor Kokkos required): we intercept the compile command of
    compile_aot and check it carries the expected flags -- default -O3 -DNDEBUG, and a custom
    $POPS_DSL_OPTFLAGS honored (tracer define).
(b) CACHE KEY (hermetic): the flags enter the key (a stale -O2 .so is not served), and the key of
    the native/jit backends stays UNCHANGED (no collateral invalidation).
(c) NUMERIC PARITY (auto-skip without compiler / Kokkos): the same model compiled via aot and via
    production yields the SAME state after a few steps (same production bricks, same flags).
"""
from pops.numerics.riemann import HLLC
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.variables import Primitive
import hashlib
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import pops  # noqa: E402  (the .so paths require the native module, like the neighboring AOT tests)
from pops.codegen.cache import _cache_so_path, _platform_cache_key, pops_cache_dir
from pops.codegen.toolchain import _native_kokkos_root
from pops.codegen import compile_drivers as _cg_compile  # noqa: E402  (compile_aot + its toolchain helpers live here, after the Spec-4 codegen split)
from test_dsl_phase_a import INCLUDE, build_euler, initial_state  # noqa: E402


class _Captured(Exception):
    """Sentinel: carries the intercepted compile command (we cut before the real compiler)."""

    def __init__(self, cmd):
        self.cmd = list(cmd)


def _capture_compile_aot_cmd(optflags_env):
    """Returns the argument list compile_aot would pass to the compiler, WITHOUT compiling anything.

    We neutralize the toolchain probes (Kokkos / std / compiler) and replace _run_compile with a
    capture: the test stays valid even without a compiler or Kokkos installed."""
    saved_env = os.environ.get("POPS_DSL_OPTFLAGS")
    # compile_aot is now pops.codegen.compile.compile_aot and calls these helpers as its own
    # module globals (imported there from codegen.toolchain). Patching dsl no longer intercepts;
    # patch the names where compile_aot actually resolves them.
    saved = {nm: getattr(_cg_compile, nm) for nm in (
        "_run_compile", "_native_kokkos_root", "_native_kokkos_compiler",
        "_native_kokkos_flags", "_probe_cxx_std")}
    if optflags_env is None:
        os.environ.pop("POPS_DSL_OPTFLAGS", None)
    else:
        os.environ["POPS_DSL_OPTFLAGS"] = optflags_env
    try:
        def _grab(cmd, what):
            raise _Captured(cmd)

        _cg_compile._run_compile = _grab
        _cg_compile._native_kokkos_root = lambda: "/dummy/kokkos"     # Kokkos-only guard cleared
        _cg_compile._native_kokkos_compiler = lambda cxx=None: "c++"  # no real which()
        _cg_compile._native_kokkos_flags = lambda: ([], [])           # no Kokkos includes/libs
        _cg_compile._probe_cxx_std = lambda cc, std: std              # no -fsyntax-only probe
        m = build_euler("euler_optflags")
        try:
            m._m.compile_aot(os.path.join(tempfile.gettempdir(), "unused_optflags.so"), INCLUDE)
        except _Captured as c:
            return c.cmd
        raise AssertionError("compile_aot did not reach _run_compile (capture missed)")
    finally:
        for nm, fn in saved.items():
            setattr(_cg_compile, nm, fn)
        if saved_env is None:
            os.environ.pop("POPS_DSL_OPTFLAGS", None)
        else:
            os.environ["POPS_DSL_OPTFLAGS"] = saved_env


def check_default_flags():
    """Without $POPS_DSL_OPTFLAGS, compile_aot must build at -O3 -DNDEBUG (native parity), never hardcoded -O2."""
    cmd = _capture_compile_aot_cmd(None)
    assert "-O3" in cmd, "compile_aot default: -O3 missing (got %r)" % (cmd,)
    assert "-DNDEBUG" in cmd, "compile_aot default: -DNDEBUG missing (got %r)" % (cmd,)
    assert "-O2" not in cmd, "compile_aot default: hardcoded -O2 persists (got %r)" % (cmd,)
    print("OK  compile_aot default -> -O3 -DNDEBUG (no more hardcoded -O2)")


def check_env_override_honored():
    """$POPS_DSL_OPTFLAGS must be honored by compile_aot (same variable as the native path)."""
    cmd = _capture_compile_aot_cmd("-O2 -DPOPS_TEST_FLAG")
    assert "-O2" in cmd, "POPS_DSL_OPTFLAGS=-O2 ... not honored (got %r)" % (cmd,)
    assert "-DPOPS_TEST_FLAG" in cmd, "POPS_DSL_OPTFLAGS tracer define not forwarded (got %r)" % (cmd,)
    assert "-O3" not in cmd and "-DNDEBUG" not in cmd, \
        "the default -O3 -DNDEBUG leaks despite the override (got %r)" % (cmd,)
    print("OK  compile_aot honors $POPS_DSL_OPTFLAGS (-O2 -DPOPS_TEST_FLAG, tracer define forwarded)")


def _old_cache_path(model_hash, abi_key, backend, target, name):
    """Rebuilds the .so file name BEFORE the aot schema marker (5-component key)."""
    rest = "|".join((abi_key or "", backend or "", target or "", name or "",
                     _platform_cache_key())).encode()
    tag = hashlib.sha256(rest).hexdigest()[:16]
    return os.path.join(pops_cache_dir(), "%s-%s.so" % ((model_hash or "nohash")[:16], tag))


def check_cache_key():
    """The flags enter the cache key of the aot artifact (a stale -O2 is not served); native/jit
    keep an unchanged name (no collateral invalidation)."""
    saved_env = os.environ.get("POPS_DSL_OPTFLAGS")
    aot_be = "aot;kokkos=on;kcfg=deadbeef"
    prod_be = "production;kokkos=on;kcfg=deadbeef"
    try:
        # (1) the optflags change the aot .so name -> a binary built with other flags is distinct
        os.environ.pop("POPS_DSL_OPTFLAGS", None)
        p_o3 = _cache_so_path("mh", "abi", aot_be, "system", None)
        os.environ["POPS_DSL_OPTFLAGS"] = "-O2"
        p_o2 = _cache_so_path("mh", "abi", aot_be, "system", None)
        assert p_o3 != p_o2, "aot cache key insensitive to optflags (%s == %s)" % (p_o3, p_o2)
        # (2) an aot .so built before aligning the flags (5-component key) no longer collides
        os.environ.pop("POPS_DSL_OPTFLAGS", None)
        assert _cache_so_path("mh", "abi", aot_be, "system", None) \
            != _old_cache_path("mh", "abi", aot_be, "system", None), \
            "the pre-fix -O2 aot .so would still be served (key not invalidated)"
        # (3) native (key already faithful to its binary) keeps its file name: no invalidation
        assert _cache_so_path("mh", "abi", prod_be, "system", None) \
            == _old_cache_path("mh", "abi", prod_be, "system", None), \
            "the native backend key changed (collateral invalidation)"
    finally:
        if saved_env is None:
            os.environ.pop("POPS_DSL_OPTFLAGS", None)
        else:
            os.environ["POPS_DSL_OPTFLAGS"] = saved_env
    print("OK  cache key: optflags folded into the aot .so, stale -O2 set apart, native unchanged")


def _is_local_env_limitation(err):
    """True if the failure stems from the local ENVIRONMENT, not from a code regression:
    - AOT .so not loadable (Kokkos symbol absent from the flat namespace: serial _pops / macOS two-level);
    - worktree headers != build of the _pops module (header signature: _pops built elsewhere).
    Neither case happens in CI (same headers as _pops, Kokkos runtime loaded). We do NOT mask a
    compilation failure ('the compilation of the .so ... failed') nor a parity gap (AssertionError)."""
    msg = str(err)
    if "dlopen" in msg and "Kokkos" in msg:
        return True
    return "DO NOT MATCH" in msg or "header signature" in msg


def check_numeric_parity():
    """aot (host-marshaled, now -O3 -DNDEBUG) and production (native) run the same production
    bricks -> same state after a few steps. We first compile the AOT .so at the current flags (LOUD
    proof that the flags are accepted by the compiler), then attempt end-to-end parity; this
    second part depends on the environment (native: headers == module; aot: .so loadable) and SKIPS
    cleanly if the local env does not allow it. Auto-skip too without compiler / Kokkos."""
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE) or _native_kokkos_root() is None:
        print("skip  numeric parity (compiler, pops headers or Kokkos absent)")
        return
    n = 32
    tmp = tempfile.mkdtemp()
    finals = {}
    try:
        # (1) REAL compile_aot at the current flags: a failure here (invalid flags) MUST be loud.
        cm_aot = build_euler("euler_optflags_aot").compile(
            os.path.join(tmp, "m_aot.so"), INCLUDE, backend="aot")
        print("OK  compile_aot produces a .so at the current flags (accepted by the compiler)")
        # (2) end-to-end parity aot vs production (same bricks, same flags). Depends on the local env.
        try:
            cm_prod = build_euler("euler_optflags_production").compile(
                os.path.join(tmp, "m_prod.so"), INCLUDE, backend="production")
            for backend, cm in (("aot", cm_aot), ("production", cm_prod)):
                s = pops.System(n=n, periodic=True)
                s.add_equation("gas", cm, spatial=pops.FiniteVolume(limiter=Minmod(), riemann=HLLC(),
                                                                   variables=Primitive()))
                s.set_poisson(rhs="charge_density", solver="geometric_mg")
                s.set_state("gas", initial_state(n))
                nsteps = s.run(t_end=0.02, cfl=0.4)
                assert nsteps > 0, "%s: run did not advance" % backend
                finals[backend] = np.array(s.get_state("gas"))
                assert np.all(np.isfinite(finals[backend])), "%s: non-finite state" % backend
        except RuntimeError as e:
            if _is_local_env_limitation(e):
                print("skip  end-to-end parity: local env (%s); AOT compilation already validated"
                      % str(e).splitlines()[0])
                return
            raise
        da = float(np.max(np.abs(finals["aot"] - finals["production"])))
        # same threshold as test_dsl_phase_a (aot==production parity): absolute dmax < 1e-10
        assert da < 1e-10, "aot != production after flag alignment (dmax=%.3e)" % da
        print("OK  parity aot == production (same flags, same bricks, dmax=%.3e)" % da)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    check_default_flags()
    check_env_override_honored()
    check_cache_key()
    check_numeric_parity()
    print("test_aot_optflags: all green")


if __name__ == "__main__":
    main()
