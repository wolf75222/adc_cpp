#!/usr/bin/env python3
"""Optional per-Program dt bound (epic ADC-399 / ADC-417, spec section 18).

A compiled time Program may OPTIONALLY provide a dt bound via ``@P.dt_bound`` (decorator) or
``P.set_dt_bound(expr_or_fn)``. The bound builds an IR scalar sub-program (reading the live state +
reductions + the geometry hmin + the per-block max wave speed); it is NOT run in Python during
``sim.step_cfl``. ``step_cfl`` then uses ``min(native CFL dt, program dt bound)``: a program bound
SMALLER than the native CFL wins; a LARGER bound loses (native CFL wins); and a Program WITHOUT a dt
bound leaves the native CFL UNCHANGED.

The generated .so exports a SECOND ABI pair alongside the macro step: ``pops_program_has_dt_bound()``
(true iff a bound was set) and ``pops_program_dt_bound(ProgramContext*, cfl)`` (the lowered scalar).

Section (A) (pure Python) pins the IR + codegen: the bound is recorded, the two ABI functions are
emitted, and a Program WITHOUT a dt bound emits ``has_dt_bound() -> false``. Section (B) is end-to-end
(needs _pops + a compiler + a visible Kokkos via POPS_KOKKOS_ROOT) and self-skips cleanly otherwise; it
never fakes the engine.
"""
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
import sys


def _skip(msg):
    print("skip test_time_dt_bound (%s)" % msg)
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


# ====================================================================================================
# Section (A): pure Python -- the dt bound is recorded in the IR and lowered to the two ABI functions.
# ====================================================================================================
print("== (A) IR + codegen ==")


def _fe(name="fe_dtbound"):
    P = adctime.Program(name)
    U = P.state("ions")
    f = P.solve_fields(U)
    R = P.rhs(state=U, fields=f, flux=True, sources=["default"])
    P.commit("ions", P.linear_combine("U1", U + P.dt * R))
    return P


# (A1) a Program WITHOUT a dt bound emits has_dt_bound() -> false; the dt_bound function returns +inf.
P_no = _fe("fe_no_bound")
chk(not P_no.has_dt_bound(), "a fresh Program has no dt bound")
src_no = P_no.emit_cpp_program()
chk("bool pops_program_has_dt_bound()" in src_no, "has_dt_bound ABI function emitted")
chk("pops::Real pops_program_dt_bound(" in src_no, "dt_bound ABI function emitted")
chk("return false;" in src_no, "no-bound Program: has_dt_bound() returns false")
chk("std::numeric_limits<pops::Real>::infinity()" in src_no, "no-bound dt_bound returns +inf sentinel")

# (A2) @P.dt_bound records a scalar sub-program (cfl * hmin / max_wave_speed); the codegen emits the
# bound expression reading ctx.hmin() / ctx.max_wave_speed.
P_dec = _fe("fe_decorator")


@P_dec.dt_bound
def _dt_bound(P, cfl):
    U = P.state("ions")
    w = P.max_wave_speed(U)
    return cfl * P.hmin() / w


chk(P_dec.has_dt_bound(), "@P.dt_bound records the bound")
src_dec = P_dec.emit_cpp_program()
chk("return true;" in src_dec, "Program with a bound: has_dt_bound() returns true")
chk("ctx.hmin()" in src_dec, "dt_bound lowers P.hmin() -> ctx.hmin()")
chk("ctx.max_wave_speed(0, " in src_dec, "dt_bound lowers P.max_wave_speed -> ctx.max_wave_speed(0, .)")
chk("cfl" in src_dec.split("pops_program_dt_bound", 1)[1], "the cfl argument is used in the bound body")

# (A3) P.set_dt_bound(expr) (the non-decorator form) records the same way; a different bound -> a
# different IR hash (the bound is part of the IR identity / cache key).
P_set = _fe("fe_setter")
Ub = P_set.state("ions")
P_set.set_dt_bound(0.5 * P_set.hmin() / P_set.max_wave_speed(Ub))
chk(P_set.has_dt_bound(), "P.set_dt_bound(expr) records the bound")
chk(P_no._ir_hash() != P_set._ir_hash(), "a dt bound changes the IR hash (distinct cache key)")

# (A4) fail-loud: the body must return a Scalar, set at most once, and read only (no commit).
P_bad = _fe("fe_bad")
try:
    P_bad.set_dt_bound(lambda P, cfl: P.state("ions"))  # returns a State, not a Scalar
except ValueError as exc:
    chk("Scalar" in str(exc), "non-Scalar dt bound body rejected")
else:
    chk(False, "non-Scalar dt bound body should be rejected")

P_twice = _fe("fe_twice")
P_twice.set_dt_bound(P_twice.hmin())
try:
    P_twice.set_dt_bound(P_twice.hmin())
except ValueError as exc:
    chk("already set" in str(exc), "a second set_dt_bound is rejected")
else:
    chk(False, "a second set_dt_bound should be rejected")

# A runtime Scalar must never collapse to a Python bool / index (it is unknown until the step runs).
try:
    bool(P_dec.hmin())
except TypeError:
    chk(True, "a Scalar cannot be used as a Python bool")
else:
    chk(False, "a Scalar used as a Python bool should raise")

if fails:
    print("FAIL test_time_dt_bound (Section A): %d failure(s)" % fails)
    sys.exit(1)
print("PASS test_time_dt_bound Section A")


# ====================================================================================================
# Section (B): end-to-end -- step_cfl uses min(native CFL, program dt bound).
# ====================================================================================================
print("== (B) step_cfl applies min(native CFL, program dt bound) ==")

probe = pops.System(n=8, L=1.0, periodic=True)
if not hasattr(probe, "install_program") or not hasattr(probe, "set_program_cadence"):
    _skip("_pops lacks install_program (rebuild _pops) (A passed)")


def transport_model():
    # Pure transport (isothermal, NoSource); BackgroundDensity(n0=0) keeps solve_fields well-defined
    # but INERT (no Poisson feedback into the flux), so the compiled cadence is bit-exact vs native.
    return pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                     transport=pops.IsothermalFlux(),
                     source=pops.NoSource(),
                     elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0))


N = 24
CFL = 0.4


def make_sim():
    sim = pops.System(n=N, L=1.0, periodic=True)
    sim.add_block("ions", transport_model(),
                  spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                  time=pops.Explicit(method="euler"))
    sim.set_poisson("charge_density", "geometric_mg")
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("ions", np.stack([rho, 0.4 * rho, -0.2 * rho]))
    return sim


def fe_program(name, *, factor=None):
    """Forward Euler; with factor set, attach a dt bound factor * cfl * hmin / max_wave_speed (a
    multiple of the native single-block CFL dt = cfl * h / w): factor < 1 tightens, factor > 1 loosens."""
    P = adctime.Program(name)
    U = P.state("ions")
    f = P.solve_fields(U)
    R = P.rhs(state=U, fields=f, flux=True, sources=["default"])
    P.commit("ions", P.linear_combine("U1", U + P.dt * R))
    if factor is not None:
        @P.dt_bound
        def _b(Pr, cfl, _f=factor):
            Us = Pr.state("ions")
            w = Pr.max_wave_speed(Us)
            return _f * cfl * Pr.hmin() / w
    return P


try:
    prog_none = pops.compile_problem(model=transport_model(), time=fe_program("fe_none"))
    prog_tight = pops.compile_problem(model=transport_model(),
                                     time=fe_program("fe_tight", factor=0.5))
    prog_loose = pops.compile_problem(model=transport_model(),
                                     time=fe_program("fe_loose", factor=2.0))
except RuntimeError as exc:  # no compiler / no Kokkos visible / compile failed
    _skip("compile_problem could not build the .so: %s" % str(exc)[:160])


def install(prog):
    sim = make_sim()
    sim.install_program(prog.so_path)
    return sim


# Baseline: a NATIVE System (no program) computes the native CFL dt on the same state.
dt_native = make_sim().step_cfl(CFL)
chk(dt_native > 0 and np.isfinite(dt_native), "native step_cfl dt finite (%.6g)" % dt_native)

# (B1) a Program WITHOUT a dt bound: step_cfl uses the native CFL dt UNCHANGED.
dt_no = install(prog_none).step_cfl(CFL)
chk(abs(dt_no - dt_native) < 1e-14,
    "no dt bound -> step_cfl uses the native CFL dt (%.10g vs %.10g)" % (dt_no, dt_native))

# (B2) a Program WITH a dt bound SMALLER than native (factor 0.5): step_cfl uses the program bound.
dt_tight = install(prog_tight).step_cfl(CFL)
chk(dt_tight < dt_native,
    "tighter dt bound applied: step_cfl dt < native (%.10g < %.10g)" % (dt_tight, dt_native))
chk(abs(dt_tight - 0.5 * dt_native) < 1e-9 * dt_native,
    "the achieved dt == the program bound (0.5 * native) (%.10g vs %.10g)"
    % (dt_tight, 0.5 * dt_native))

# (B3) a Program WITH a dt bound LARGER than native (factor 2.0): native wins (min semantics).
dt_loose = install(prog_loose).step_cfl(CFL)
chk(abs(dt_loose - dt_native) < 1e-14,
    "looser dt bound ignored: native CFL wins (%.10g vs %.10g)" % (dt_loose, dt_native))

# (B4) the tighter-bound program actually advanced the state, and step_cfl(cfl) == step(dt_tight)
# bit-exact (step_cfl just computes dt -- now the program bound -- then runs the same cadence as step).
sim_t = install(prog_tight)
u0 = np.array(sim_t.get_state("ions"))
dt_t = sim_t.step_cfl(CFL)
u1 = np.array(sim_t.get_state("ions"))
chk(float(np.abs(u1 - u0).max()) > 1e-9, "tight-bound program advanced the state")
chk(abs(dt_t - dt_tight) < 1e-12, "tight-bound dt reproducible (%.10g vs %.10g)" % (dt_t, dt_tight))

sim_ref = install(prog_tight)
sim_ref.step(dt_t)  # drive the reference at the dt step_cfl chose
u_ref = np.array(sim_ref.get_state("ions"))
chk(float(np.abs(u1 - u_ref).max()) == 0.0,
    "step_cfl(cfl) == step(dt_tight) bit-exact (max|d|=%.2e)" % float(np.abs(u1 - u_ref).max()))

if fails:
    print("FAIL test_time_dt_bound: %d failure(s)" % fails)
    sys.exit(1)
print("PASS test_time_dt_bound (Sections A + B)")
