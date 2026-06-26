#!/usr/bin/env python3
"""Compiled time-program macro-step cadence: substeps + stride (epic ADC-399 / ADC-411).

`pops.CompiledTime(substeps=, stride=)` records a SYSTEM-level cadence applied AROUND the opaque
compiled-program closure (`System.set_program_cadence`, mirroring the native per-block advance loop):
  - substeps=n  -> program_step_(eff_dt/n) called n times (subdivides the effective step);
  - stride=M    -> the whole program runs ONCE per M macro-steps with eff_dt = M*dt (GLOBAL
                   hold-then-catch-up), and the clock t STILL ticks every macro-step.

Bit-exactness vs the native path holds ONLY on the simple cases the tests deliberately use:
  - substeps: a compiled Forward-Euler program over an UNCOUPLED / transport-only model matches
    native pops.Explicit(method="euler", substeps=n) bit-for-bit. (program_step_(h) re-runs the WHOLE
    program -- its solve_fields included -- so n>1 equals native substeps only when solve_fields is
    inert, i.e. no Poisson feedback into the flux.)
  - stride: a SINGLE-block system. The compiled program is one whole-system closure, so its stride is
    GLOBAL; it equals native per-block stride only when there is a single block (or all blocks share M).

Section (A) (pure Python) always runs: the substeps/stride guards are gone, the values are stored, and
negative/zero raise. Section (B) (parity) needs _pops + a compiler + a visible Kokkos (POPS_KOKKOS_ROOT)
and self-skips cleanly otherwise -- it never fakes the engine.
"""
import sys


def _skip(msg):
    print("skip test_time_substeps_stride (%s)" % msg)
    sys.exit(0)


try:
    import numpy as np

    import pops
    from pops import time as adctime
except Exception as exc:  # noqa: BLE001
    _skip("pops/numpy unavailable: %s" % exc)

fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


def raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


# ---- (A) validation: pure Python, always runs ----
print("== (A) CompiledTime substeps/stride validation (ADC-411) ==")
ct2 = pops.CompiledTime(substeps=2)
chk(ct2.substeps == 2 and ct2.stride == 1, "CompiledTime(substeps=2) stores substeps (no raise)")
cts = pops.CompiledTime(stride=2)
chk(cts.stride == 2 and cts.substeps == 1, "CompiledTime(stride=2) stores stride (no raise)")
chk(pops.CompiledTime(substeps=3, stride=4).substeps == 3
    and pops.CompiledTime(substeps=3, stride=4).stride == 4,
    "CompiledTime(substeps=3, stride=4) stores both")
chk(raises(ValueError, lambda: pops.CompiledTime(substeps=0)), "substeps=0 rejected (ValueError)")
chk(raises(ValueError, lambda: pops.CompiledTime(substeps=-1)), "substeps<0 rejected (ValueError)")
chk(raises(ValueError, lambda: pops.CompiledTime(stride=0)), "stride=0 rejected (ValueError)")
chk(raises(ValueError, lambda: pops.CompiledTime(stride=-2)), "stride<0 rejected (ValueError)")
chk(raises(NotImplementedError, lambda: pops.CompiledTime(cfl="program")),
    "cfl != 'default' still deferred (NotImplementedError)")

# ---- (B) end-to-end parity: skips unless the full toolchain is present ----
print("== (B) compiled cadence vs native (parity) ==")


def transport_model():
    # Pure transport (isothermal, NoSource); BackgroundDensity(n0=0) keeps solve_fields well-defined
    # but INERT (no Poisson feedback into the flux), so re-running solve_fields per substep / per
    # program call changes nothing -> the compiled cadence is bit-exact vs the native cadence.
    return pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                     transport=pops.IsothermalFlux(),
                     source=pops.NoSource(),
                     elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0))


N = 24


def make_sim(time):
    sim = pops.System(n=N, L=1.0, periodic=True)
    sim.add_block("ions", transport_model(),
                  spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"), time=time)
    sim.set_poisson("charge_density", "geometric_mg")
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("ions", np.stack([rho, 0.4 * rho, -0.2 * rho]))
    return sim


def fe_program(name="fe_cadence"):
    P = adctime.Program(name)
    U = P.state("ions")
    f = P.solve_fields(U)
    R = P.rhs(state=U, fields=f, flux=True, sources=["default"])
    P.commit("ions", P.linear_combine("U1", U + P.dt * R))
    return P


# Skip cleanly without the install_program / set_program_cadence binding (rebuild _pops).
probe = pops.System(n=8, L=1.0, periodic=True)
if not hasattr(probe, "install_program") or not hasattr(probe, "set_program_cadence"):
    _skip("_pops lacks install_program / set_program_cadence (rebuild _pops) (A passed)")

try:
    compiled = pops.compile_problem(model=transport_model(), time=fe_program())
except RuntimeError as exc:  # no compiler / no Kokkos visible / compile failed
    _skip("compile_problem could not build the .so: %s" % str(exc)[:160])

dt = 2e-3


def run_compiled(cadence, n_steps, dt_step=dt):
    """Install the FE program, set the cadence, step n_steps macro-steps; returns (state, t)."""
    sim = make_sim(pops.Explicit(method="euler"))
    sim.install_program(compiled.so_path)
    sim.set_program_cadence(cadence.substeps, cadence.stride)  # CompiledTime -> set_program_cadence
    for _ in range(n_steps):
        sim.step(dt_step)
    return np.array(sim.get_state("ions")), float(sim.time())


# ----- SUBSTEPS: compiled CompiledTime(substeps=2) vs native pops.Explicit(euler, substeps=2) -----
print("-- substeps --")
sub2, _ = run_compiled(pops.CompiledTime(substeps=2), 1)

nat = make_sim(pops.Explicit(method="euler", substeps=2))
nat.step(dt)
sub2_native = np.array(nat.get_state("ions"))
e_sub = float(np.abs(sub2 - sub2_native).max())
chk(e_sub < 1e-12, "compiled substeps=2 == native pops.Explicit(euler, substeps=2) (max|d|=%.2e)"
    % e_sub)

# Consistency: substeps=2 over dt == two compiled substeps=1 steps of dt/2 (the same FE sub-iteration
# sequence; both re-run the inert solve_fields each call).
half_twice, _ = run_compiled(pops.CompiledTime(substeps=1), 2, dt_step=dt / 2.0)
e_half = float(np.abs(sub2 - half_twice).max())
chk(e_half < 1e-12, "compiled substeps=2 (dt) == two compiled substeps=1 (dt/2) (max|d|=%.2e)"
    % e_half)

# Non-degenerate: substeps=2 must DIFFER from substeps=1 (otherwise the orchestration is a no-op).
sub1, _ = run_compiled(pops.CompiledTime(substeps=1), 1)
d_sub = float(np.abs(sub2 - sub1).max())
chk(d_sub > 1e-9, "substeps=2 differs from substeps=1 (non-degenerate, max|d|=%.2e)" % d_sub)

# ----- STRIDE: single-block system, compiled stride=2 vs native pops.Explicit(stride=2) -----
print("-- stride --")
K = 4  # macro-steps (even -> ends on a catch-up window boundary)
str2, t_str = run_compiled(pops.CompiledTime(stride=2), K)

nat_s = make_sim(pops.Explicit(method="euler", stride=2))
for _ in range(K):
    nat_s.step(dt)
str2_native = np.array(nat_s.get_state("ions"))
e_str = float(np.abs(str2 - str2_native).max())
chk(e_str < 1e-12, "compiled stride=2 == native pops.Explicit(euler, stride=2) over %d steps "
    "(max|d|=%.2e)" % (K, e_str))

# The clock ticks EVERY macro-step (held steps included): t = K*dt after K steps.
chk(abs(t_str - K * dt) < 1e-14, "clock advanced every macro-step: t=%.6g == %d*dt=%.6g"
    % (t_str, K, K * dt))

# Non-degenerate: stride=2 must DIFFER from stride=1 over the same K steps.
str1, _ = run_compiled(pops.CompiledTime(stride=1), K)
d_str = float(np.abs(str2 - str1).max())
chk(d_str > 1e-9, "stride=2 differs from stride=1 (non-degenerate, max|d|=%.2e)" % d_str)

print("%s test_time_substeps_stride" % ("FAIL (%d)" % fails if fails else "PASS"))
sys.exit(1 if fails else 0)
