#!/usr/bin/env python3
"""adc.time RUNTIME parameters in compiled kernels -- ctx.param, no compile-time freeze (ADC-435).

A model param declared dsl.Param(..., kind='runtime') used by a compiled time Program lowers to a HOST
read ``const adc::Real <name> = ctx.param("<name>");`` ONCE before the per-cell loop (captured by value)
instead of a baked literal. The value is then read from the System param store (sim.set_param) and can
CHANGE at runtime WITHOUT recompiling the .so -- only the param NAME is in the generated source, so the
.so cache key (sha256 of the source) is invariant under a runtime-param value change. A FROZEN const
param (kind='const', the default) stays inline / baked (bit-identical to the historical path).

(A) Codegen / IR (pure Python, always runs):
    - a runtime-param source kernel emits ctx.param("k") (host scalar) + uses the bare name in the cell,
      and the declaration value (a literal) is ABSENT from the source;
    - the generated source is IDENTICAL across two declaration VALUES of the same runtime param
      (cache-key invariance: a value change does not change the .so);
    - a frozen-param source kernel bakes the literal (no ctx.param) and the source DIFFERS across two
      values (a frozen-param change requires a recompile -- correct).

(B) End-to-end probe (skips unless the full toolchain is present): a ZERO-FLUX scalar model with a
    NAMED source S = k*rho (k runtime). Compile ONCE; install; sim.set_param("k", K1); step -> out ==
    rho0*(1 + dt*K1). Then sim.set_param("k", K2) on the SAME installed .so; re-seed; step -> out ==
    rho0*(1 + dt*K2). The kernel uses the NEW k with NO recompile. A frozen-k twin is bit-identical to
    an offline baked literal. Recompiling the runtime model with a DIFFERENT declaration value reuses
    the cached .so (same so_path: the cache key excludes runtime-param values). Self-skips if _adc lacks
    the param binding, numpy/_adc is absent, or no compiler/Kokkos is visible -- never faking the engine.

Run with python3 (PYTHONPATH = built adc package).
"""
import sys


def _skip(msg):
    print("skip test_time_runtime_param (%s)" % msg)
    sys.exit(0)


try:
    import numpy as np

    import adc
    from adc import dsl
    from adc import time as adctime
except Exception as exc:  # noqa: BLE001  -- numpy or _adc unavailable in this interpreter
    _skip("adc/numpy unavailable: %s" % exc)

fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


def decay_model(name, k, kind):
    """Scalar 'rho' with NO flux and a NAMED source 'decay' = k*rho, k a param of @p kind
    ('runtime' | 'const'). Zero flux so -div F == 0 exactly: the only dynamics is the named source,
    which isolates the param effect bit-exactly. A complete, compilable production block."""
    m = dsl.Model(name)
    (rho,) = m.conservative_vars("rho")
    zero = 0.0 * rho  # null expr (flux / eig do not wrap a bare float)
    m.flux(x=[zero], y=[zero])
    m.eigenvalues(x=[zero], y=[zero])
    m.primitive_vars(rho=rho)
    m.conservative_from([rho])
    kparam = m.param("k", k, kind=kind)
    m.source_term("decay", [kparam * rho])  # NAMED source S = k*rho (lowered by the program codegen)
    return m


def one_step_source_program(name):
    """U^{n+1} = U + dt * S_decay(U), committed on block 'plasma' (a single named-source step)."""
    P = adctime.Program(name)
    U = P.state("plasma")
    S = P.source("decay", state=U)
    P.commit("plasma", P.linear_combine("%s_step" % name, U + P.dt * S))
    return P


# ---- (A) codegen / IR: pure Python, always runs ----
print("== (A) runtime vs frozen param: codegen split ==")

src_rt = one_step_source_program("p").emit_cpp_program(model=decay_model("rt", 0.7, "runtime"))
src_rt2 = one_step_source_program("p").emit_cpp_program(model=decay_model("rt", 1.9, "runtime"))
src_fr = one_step_source_program("p").emit_cpp_program(model=decay_model("fr", 0.7, "const"))
src_fr2 = one_step_source_program("p").emit_cpp_program(model=decay_model("fr", 1.9, "const"))

chk('const adc::Real k = ctx.param("k");' in src_rt,
    "runtime param emits a HOST ctx.param read (const adc::Real k = ctx.param(\"k\"))")
chk("(k * rho)" in src_rt,
    "runtime param uses the BARE name in the cell (k * rho), not params.get / a literal")
chk("0.7" not in src_rt and "params.get" not in src_rt,
    "runtime param: the declaration value (0.7) is ABSENT and no params.get(idx) leaks in")
chk(src_rt == src_rt2,
    "runtime param: source IDENTICAL across declaration values (.so cache key value-invariant)")

chk("ctx.param" not in src_fr,
    "frozen param: NO ctx.param read (baked literal path, bit-identical default)")
chk("0.7" in src_fr,
    "frozen param: the value 0.7 is baked as a literal in the source")
chk(src_fr != src_fr2,
    "frozen param: a value change CHANGES the source (a recompile is required -- correct)")


# EigWitness recursion: a runtime param nested inside an eig-witness matrix entry is a legal scalar
# coefficient. runtime_param_names_in discovers it (via _children -> entries()), so the host ctx.param
# read MUST be emitted AND freeze_runtime_params_as_vars must rewrite the entry's tree -- else the
# un-rewritten RuntimeParamRef raises "index not assigned" at codegen.
def eig_runtime_model(name, k):
    m = dsl.Model(name)
    (rho,) = m.conservative_vars("rho")
    zero = 0.0 * rho
    m.flux(x=[zero], y=[zero])
    m.eigenvalues(x=[zero], y=[zero])
    m.primitive_vars(rho=rho)
    m.conservative_from([rho])
    kparam = m.param("k", k, kind="runtime")
    m.source_term("decay", [dsl.eig_max_im([[kparam * rho]]) * rho])  # runtime param nested in eig
    return m


src_eig = one_step_source_program("p").emit_cpp_program(model=eig_runtime_model("eig", 0.5))
chk('const adc::Real k = ctx.param("k");' in src_eig,
    "runtime param nested in eig_max_im still emits the host ctx.param read")
chk("params.get" not in src_eig,
    "runtime param in eig_max_im: the entry's RuntimeParamRef is rewritten (no params.get, no index error)")


# ---- (B) end-to-end probe: skips unless the full toolchain is present ----
_probe = adc.System(n=8, L=1.0, periodic=True)
if not (hasattr(_probe, "install_program") and hasattr(_probe, "set_param")):
    print("-- (B) skipped: _adc lacks install_program / set_param (rebuild _adc) --")
    print("%s test_time_runtime_param (A only)" % ("FAIL" if fails else "PASS"))
    sys.exit(1 if fails else 0)

print("== (B) end-to-end: ctx.param varies the kernel without a recompile ==")

DT = 0.05
N = 16
K1 = 0.7
K2 = 2.3


def _rho0():
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)


def make_sim(block_model):
    """Fresh System with @p block_model added as a normal block (the program drives the step)."""
    sim = adc.System(n=N, L=1.0, periodic=True)
    try:
        compiled_model = block_model.compile(backend="production")
    except RuntimeError as exc:  # no compiler / no Kokkos visible
        _skip("model compile could not build the .so: %s" % str(exc)[:160])
    sim.add_equation("plasma", compiled_model,
                     spatial=adc.FiniteVolume(limiter="none", riemann="rusanov"),
                     time=adc.Explicit(method="euler"))
    return sim


# Compile the runtime-param program ONCE.
try:
    compiled_rt = adc.compile_problem(model=decay_model("rt_block", K1, "runtime"),
                                      time=one_step_source_program("rt_step"))
except RuntimeError as exc:  # no compiler / no Kokkos / .so compile failed
    _skip("compile_problem (runtime) could not build the .so: %s" % str(exc)[:160])

rho0 = _rho0()
sim = make_sim(decay_model("rt_block", K1, "runtime"))
sim.install_program(compiled_rt.so_path)

# Step with k = K1.
sim.set_param("k", K1)
sim.set_state("plasma", np.stack([rho0]))
sim.step(DT)
out1 = np.array(sim.get_state("plasma"))[0]
ref1 = rho0 * (1.0 + DT * K1)
d1 = float(np.abs(out1 - ref1).max())
print("  k=K1=%.2f  max|out - rho0*(1+dt*K1)| = %.3e" % (K1, d1))
chk(d1 < 1e-12, "compiled kernel with k=K1 reproduces the offline reference (out == rho0*(1+dt*K1))")

# Change k to K2 on the SAME installed .so (no recompile) and step from the same rho0.
sim.set_param("k", K2)
sim.set_state("plasma", np.stack([rho0]))
sim.step(DT)
out2 = np.array(sim.get_state("plasma"))[0]
ref2 = rho0 * (1.0 + DT * K2)
d2 = float(np.abs(out2 - ref2).max())
print("  k=K2=%.2f  max|out - rho0*(1+dt*K2)| = %.3e" % (K2, d2))
chk(d2 < 1e-12, "set_param(k=K2) changes the kernel at RUNTIME (out == rho0*(1+dt*K2), no recompile)")
chk(float(np.abs(out2 - out1).max()) > 1e-6, "K1 vs K2 actually produce DIFFERENT states")
chk(sim.param("k") == K2, "sim.param('k') reflects the last set_param value")

# Frozen-param twin: a kind='const' model bakes K1 -> bit-identical to the offline baked literal, and
# set_param has no effect on it (the value is compiled in). Compile a SEPARATE frozen program/model.
try:
    compiled_fr = adc.compile_problem(model=decay_model("fr_block", K1, "const"),
                                      time=one_step_source_program("fr_step"))
except RuntimeError as exc:
    _skip("compile_problem (frozen) could not build the .so: %s" % str(exc)[:160])
sim_fr = make_sim(decay_model("fr_block", K1, "const"))
sim_fr.install_program(compiled_fr.so_path)
sim_fr.set_param("k", K2)  # MUST be ignored: the frozen param is baked at K1
sim_fr.set_state("plasma", np.stack([rho0]))
sim_fr.step(DT)
out_fr = np.array(sim_fr.get_state("plasma"))[0]
d_fr = float(np.abs(out_fr - rho0 * (1.0 + DT * K1)).max())
print("  frozen k baked=K1 (set_param ignored)  max|out - rho0*(1+dt*K1)| = %.3e" % d_fr)
chk(d_fr < 1e-12, "frozen param stays baked at K1 (set_param ignored) -- bit-identical baked literal")

# Cache-key invariance: recompiling the runtime model with a DIFFERENT declaration value reuses the
# SAME cached .so (the runtime-param VALUE is not in the cache key, only its NAME / the program source).
try:
    compiled_rt_b = adc.compile_problem(model=decay_model("rt_block", 5.5, "runtime"),
                                        time=one_step_source_program("rt_step"))
except RuntimeError as exc:
    _skip("compile_problem (runtime b) could not build the .so: %s" % str(exc)[:160])
chk(compiled_rt_b.so_path == compiled_rt.so_path,
    "runtime-param value change reuses the SAME cached .so (cache key excludes the value)")

print("%s test_time_runtime_param" % ("FAIL (%d)" % fails if fails else "PASS"))
sys.exit(1 if fails else 0)
