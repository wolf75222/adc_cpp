#!/usr/bin/env python3
"""m.roe_from_jacobian() (ADC-368): the GENERIC moment Roe. The DSL emits the
roe_dissipation(UL, AL, UR, AR, dir) hook = |A| (UR - UL) with A = dF_dir/dU the flux Jacobian at
the arithmetic-mean interface state Uavg = 1/2(UL + UR), |A| via the matrix-sign kernel
pops::roe_abs_apply (dense_eig.hpp), spectral-radius (Rusanov) fallback on a complex/singular
spectrum. Roles-free: riemann='roe' becomes available for a moment hierarchy (HyQMOM) with no
Density/Momentum roles and no primitive 'p' (unlike m.enable_roe).

  (A) wiring (no compiler): roe=True sets the capability (CompiledModel-side has_roe) and emits the
      hook + pops::roe_abs_apply + the dense_eig.hpp include; roe=False is bit-identical (no hook);
      the three Roe providers (enable_roe / roe_dissipation / roe_from_jacobian) are mutually
      exclusive; the facade re-exports roe_from_jacobian.
  (B) [compiler] compile AOT + System riemann='roe': 10 steps finite, mass (M00) conserved; a
      roe=False moment model REJECTS riemann='roe' (no capability -> ValueError).

Invariants by assert; prints "OK test_dsl_roe_from_jacobian" on success.
"""
import os
import shutil
import sys
import tempfile

import numpy as np

import pops
from pops import dsl
from pops.moments import build_moment_model, gaussian_closure

fails = 0
INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def err_msg(fn):
    try:
        fn()
        return None
    except (ValueError, RuntimeError) as e:
        return str(e)


def gaussian(n, amp=0.4):
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="xy")
    return 1.0 + amp * np.exp(-60.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))


# --- (A) wiring / codegen, no compiler needed --------------------------------
print("== (A) wiring + codegen of m.roe_from_jacobian ==")
mr = build_moment_model("roe_src", 2, gaussian_closure(2), roe=True)
src = mr._m.emit_cpp_brick()
chk(mr._m._roe_jacobian is not None, "roe=True: _roe_jacobian set on the engine")
chk("State roe_dissipation(" in src, "roe=True: roe_dissipation hook emitted")
chk("pops::roe_abs_apply(" in src, "roe=True: hook calls pops::roe_abs_apply")
chk("#include <pops/numerics/linalg/dense_eig.hpp>" in src, "roe=True: dense_eig.hpp included")
chk("pops::real_eig_minmax(" in src, "roe=True: spectral-radius (Rusanov) fallback emitted")

m0 = build_moment_model("noroe_src", 2, gaussian_closure(2), roe=False)
src0 = m0._m.emit_cpp_brick()
chk(m0._m._roe_jacobian is None, "roe=False: _roe_jacobian is None")
chk("roe_dissipation" not in src0, "roe=False: NO roe_dissipation hook (bit-identical history)")

# mutual exclusivity (three providers of the same hook), both directions
chk("provider" in (err_msg(lambda: build_moment_model("x", 2, gaussian_closure(2), roe=True)
                                   ._m.enable_roe()) or ""),
    "exclusivity: enable_roe() after roe_from_jacobian -> error")
me = build_moment_model("y", 2, gaussian_closure(2), roe=False)
me._m._roe = True
chk("provider" in (err_msg(lambda: me._m.roe_from_jacobian()) or ""),
    "exclusivity: roe_from_jacobian() after enable_roe -> error")
chk(hasattr(mr, "roe_from_jacobian") and callable(mr.roe_from_jacobian),
    "facade Model.roe_from_jacobian re-exported")

# --- (B) compile AOT + System riemann='roe' (compiler-gated) ------------------
cxx = dsl._default_cxx(None)
if not cxx or not os.path.isdir(INCLUDE):
    print("== (B) saute : compilateur C++ ou en-tetes pops absents ==")
    print("FAILS =", fails)
    sys.exit(1 if fails else 0)

print("== (B) compile AOT + System riemann='roe' ==")
tmp = tempfile.mkdtemp(prefix="pops_roe_jac_")
try:
    n = 24
    compiled = build_moment_model("g2roe", 2, gaussian_closure(2), roe=True).compile(
        os.path.join(tmp, "g2roe.so"), INCLUDE, backend="aot")
    chk(getattr(compiled, "has_roe", False), "CompiledModel.has_roe = True")

    # realizable Maxwellian moments (u = v = 0, T = 1) modulated by a smooth density bump
    base = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    U0 = base[:, None, None] * gaussian(n)[None, :, :]

    sim = pops.System(n=n, L=1.0, periodic=True)
    sim.add_equation("mom", model=compiled,
                     spatial=pops.FiniteVolume(limiter="none", riemann="roe"),
                     time=pops.Explicit())
    sim.set_state("mom", U0)
    for _ in range(10):
        sim.step(5e-4)
    out = np.asarray(sim.get_state("mom"))
    chk(np.isfinite(out).all(), "10 pas ROE : etat fini")
    dm = abs(out[0].sum() - U0[0].sum()) / abs(U0[0].sum())
    chk(dm < 1e-12, f"10 pas ROE : masse M00 conservee ({dm:.2e})")

    # a roe=False moment model must REJECT riemann='roe' (no capability)
    cm_no = build_moment_model("g2noroe", 2, gaussian_closure(2), roe=False).compile(
        os.path.join(tmp, "g2noroe.so"), INCLUDE, backend="aot")
    s2 = pops.System(n=16, L=1.0, periodic=True)
    msg = err_msg(lambda: s2.add_equation(
        "mom", model=cm_no, spatial=pops.FiniteVolume(limiter="none", riemann="roe"),
        time=pops.Explicit()))
    chk(msg is not None, f"roe=False: riemann='roe' rejete ({(msg or '')[:48]}...)")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("OK test_dsl_roe_from_jacobian" if not fails else f"FAILS = {fails}")
sys.exit(1 if fails else 0)
