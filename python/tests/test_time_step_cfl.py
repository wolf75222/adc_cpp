#!/usr/bin/env python3
"""step_cfl routes through an installed compiled time Program (epic ADC-399 criterion 7 / spec 19).

`System.step_cfl(cfl)` must CONSERVE the existing CFL logic (the per-block CFL dt is still computed in
adc_cpp on the native state) AND, when a compiled program is installed, drive the macro-step at that dt
through the SAME cadence helper as `step()` (run_program_cadence). Before ADC-413, step_cfl drove ONLY
the native per-block path and silently ignored an installed program (the state stayed frozen / advanced
natively instead of running the program).

Semantics (matched to step()'s program branch): step_cfl computes dt from the CFL bounds, then runs the
program cadence + ticks the clock, with NO implicit couplings / projections (the Program expresses them
itself). Equivalently: step_cfl(cfl) == step(dt_cfl) for the dt_cfl step_cfl chose -- bit-exact.

Section (A) (pure Python) is minimal -- this is runtime behavior; the substantive checks are Section (B)
(end-to-end), which needs _pops + a compiler + a visible Kokkos (POPS_KOKKOS_ROOT) and self-skips cleanly
otherwise. It never fakes the engine.
"""
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
import sys


def _skip(msg):
    print("skip test_time_step_cfl (%s)" % msg)
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


# ---- (A) sanity: the System API exposes step_cfl (always runs) ----
print("== (A) step_cfl API present ==")
probe = pops.System(n=8, L=1.0, periodic=True)
chk(hasattr(probe, "step_cfl"), "System exposes step_cfl")


# ---- (B) end-to-end: step_cfl drives the installed program ----
print("== (B) step_cfl routes through the compiled program ==")


def transport_model():
    # Pure transport (isothermal, NoSource); BackgroundDensity(n0=0) keeps solve_fields well-defined
    # but INERT (no Poisson feedback into the flux), so re-running solve_fields per program call changes
    # nothing -> the compiled cadence is bit-exact vs the native cadence (same trick as substeps_stride).
    return pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                     transport=pops.IsothermalFlux(),
                     source=pops.NoSource(),
                     elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0))


N = 24


def make_sim(time):
    sim = pops.System(n=N, L=1.0, periodic=True)
    sim.add_block("ions", transport_model(),
                  spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()), time=time)
    sim.set_poisson("charge_density", "geometric_mg")
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("ions", np.stack([rho, 0.4 * rho, -0.2 * rho]))
    return sim


def fe_program(name="fe_stepcfl"):
    P = adctime.Program(name)
    U = P.state("ions")
    f = P.solve_fields(U)
    R = P.rhs(state=U, fields=f, flux=True, sources=["default"])
    P.commit("ions", P.linear_combine("U1", U + P.dt * R))
    return P


if not hasattr(probe, "install_program") or not hasattr(probe, "set_program_cadence"):
    _skip("_pops lacks install_program / set_program_cadence (rebuild _pops) (A passed)")

try:
    compiled = pops.compile_problem(model=transport_model(), time=fe_program())
except RuntimeError as exc:  # no compiler / no Kokkos visible / compile failed
    _skip("compile_problem could not build the .so: %s" % str(exc)[:160])

CFL = 0.4


def make_compiled(cadence=None):
    """Install the FE program (optionally setting a cadence) and return the sim."""
    sim = make_sim(pops.Explicit(method="euler"))
    sim.install_program(compiled.so_path)
    if cadence is not None:
        sim.set_program_cadence(cadence.substeps, cadence.stride)
    return sim


# ----- (a) step_cfl ADVANCES the state (program ran; not frozen) -----
print("-- (a) state advanced --")
sim_a = make_compiled()
u0 = np.array(sim_a.get_state("ions"))
dt_cfl = sim_a.step_cfl(CFL)
u1 = np.array(sim_a.get_state("ions"))
d_adv = float(np.abs(u1 - u0).max())
chk(dt_cfl > 0.0 and np.isfinite(dt_cfl), "step_cfl returned a finite positive dt (%.6g)" % dt_cfl)
chk(d_adv > 1e-9, "step_cfl advanced the state via the program (max|du|=%.2e)" % d_adv)

# ----- (b) the dt chosen == the native CFL dt AND step_cfl(cfl) == step(dt_cfl) bit-exact -----
print("-- (b) dt == native CFL dt, and step_cfl == step(dt_cfl) bit-exact --")
# A NATIVE System (no program) on the SAME state computes the SAME CFL dt: step_cfl keeps the CFL
# logic in adc_cpp (per-block bounds on the native state), only the advance is routed to the program.
sim_native = make_sim(pops.Explicit(method="euler"))
dt_native = sim_native.step_cfl(CFL)
chk(abs(dt_cfl - dt_native) < 1e-14,
    "step_cfl dt (program) == native step_cfl dt (%.10g vs %.10g)" % (dt_cfl, dt_native))

# step(dt_cfl) on a fresh compiled sim must reproduce the step_cfl result EXACTLY: step_cfl just
# computes dt then calls the same cadence helper that step() uses.
sim_b = make_compiled()
sim_b.step(dt_cfl)
u_step = np.array(sim_b.get_state("ions"))
e_bx = float(np.abs(u1 - u_step).max())
chk(e_bx == 0.0, "step_cfl(cfl) == step(dt_cfl) bit-exact (max|d|=%.2e)" % e_bx)

# ----- (c) substeps=2 / stride=2 honored UNDER step_cfl (CompiledTime cadence applied) -----
print("-- (c) substeps / stride honored under step_cfl --")
# substeps=2: step_cfl computes the SAME dt (CFL is on the state, cadence-independent at step 1),
# then runs the cadence -> 2x program_step_(dt/2). Must match step(dt) with substeps=2 set, and DIFFER
# from substeps=1.
sim_c1 = make_compiled(pops.CompiledTime(substeps=1))
dt_c1 = sim_c1.step_cfl(CFL)
u_c1 = np.array(sim_c1.get_state("ions"))

sim_c2 = make_compiled(pops.CompiledTime(substeps=2))
dt_c2 = sim_c2.step_cfl(CFL)
u_c2 = np.array(sim_c2.get_state("ions"))
chk(abs(dt_c2 - dt_c1) < 1e-14, "step_cfl dt is cadence-independent at step 1 (%.10g vs %.10g)"
    % (dt_c2, dt_c1))

sim_c2_ref = make_compiled(pops.CompiledTime(substeps=2))
sim_c2_ref.step(dt_c2)
u_c2_ref = np.array(sim_c2_ref.get_state("ions"))
e_c2 = float(np.abs(u_c2 - u_c2_ref).max())
chk(e_c2 == 0.0, "step_cfl substeps=2 == step(dt) substeps=2 bit-exact (max|d|=%.2e)" % e_c2)
d_c = float(np.abs(u_c2 - u_c1).max())
chk(d_c > 1e-9, "substeps=2 differs from substeps=1 under step_cfl (non-degenerate, max|d|=%.2e)"
    % d_c)

# stride=2: over an even number of macro-steps (ends on a catch-up boundary) a stride=2 program run
# under step_cfl matches the native-cadence stride=2 reference (compiled step(dt) with stride=2), with
# the clock ticking every macro-step. Use a FIXED dt for the reference so both see identical steps:
# capture the first step_cfl dt, then drive the reference with step(that_dt).
print("-- (c') stride honored under step_cfl --")
K = 4
sim_s = make_compiled(pops.CompiledTime(stride=2))
dts = []
for _ in range(K):
    dts.append(sim_s.step_cfl(CFL))
u_s = np.array(sim_s.get_state("ions"))
t_s = float(sim_s.time())

sim_s_ref = make_compiled(pops.CompiledTime(stride=2))
for dt_k in dts:
    sim_s_ref.step(dt_k)
u_s_ref = np.array(sim_s_ref.get_state("ions"))
e_s = float(np.abs(u_s - u_s_ref).max())
chk(e_s == 0.0, "step_cfl stride=2 == step(dt_k) stride=2 over %d steps bit-exact (max|d|=%.2e)"
    % (K, e_s))

sim_s1 = make_compiled(pops.CompiledTime(stride=1))
for dt_k in dts:
    sim_s1.step(dt_k)
u_s1 = np.array(sim_s1.get_state("ions"))
d_s = float(np.abs(u_s - u_s1).max())
chk(d_s > 1e-9, "stride=2 differs from stride=1 under step_cfl (non-degenerate, max|d|=%.2e)" % d_s)

# ----- (d) time / macro_step coherent after step_cfl -----
print("-- (d) clock coherent --")
t_expect = float(np.sum(dts))
chk(abs(t_s - t_expect) < 1e-12, "step_cfl ticks the clock every macro-step: t=%.10g == sum(dt)=%.10g"
    % (t_s, t_expect))
chk(sim_s.macro_step() == K, "macro_step incremented once per step_cfl: %d == %d"
    % (sim_s.macro_step(), K))

print("%s test_time_step_cfl" % ("FAIL (%d)" % fails if fails else "PASS"))
sys.exit(1 if fails else 0)
