#!/usr/bin/env python3
"""Multi-stage compiled time Programs: SSPRK2 + RK4 parity (epic ADC-399 / ADC-407).

`Program.emit_cpp_program` lowers a multi-stage scheme (intermediate scratch states + a `lincomb`
commit) to a problem.so. This test builds SSPRK2 and RK4 Programs, compiles + installs + runs each,
and checks parity against an OFFLINE stage-by-stage reference computed from the same runtime
primitives (`set_state` + `solve_fields` + `eval_rhs`) -- the exact stages the compiled program
drives -- so the match is to machine precision. SSPRK2 is additionally checked against the native
`pops.Explicit("ssprk2")` step (spec test 32).

Uses a pure-transport (isothermal, no field coupling) model so the per-stage `solve_fields` is inert
and identical along both paths (the compiled codegen now lowers each solve_fields to a per-stage
solve_fields_from_state, ADC-409; with no Poisson feedback into the flux the field solve changes
nothing, so re-solving from a stage state vs the current state is bit-identical here -- the
field-coupled case is exercised by test_time_solve_fields_from_state). Skips cleanly (exit 0)
without the install_program binding / numpy / a compiler / a visible Kokkos -- never fakes the engine.
"""
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
import sys


def _skip(msg):
    print("skip test_time_multistage (%s)" % msg)
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


def transport_model():
    return pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                     transport=pops.IsothermalFlux(),
                     source=pops.NoSource(),
                     elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0))


N = 24


def make_sim(method="euler"):
    sim = pops.System(n=N, L=1.0, periodic=True)
    sim.add_block("ions", transport_model(),
                  spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                  time=pops.Explicit(method=method))
    sim.set_poisson("charge_density", "geometric_mg")
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("ions", np.stack([rho, 0.4 * rho, -0.2 * rho]))
    return sim


def initial_state():
    return np.array(make_sim().get_state("ions"))


def offline_rhs(ref, U):
    """The semi-discrete RHS at state U, via the runtime primitives the compiled program also uses."""
    ref.set_state("ions", U)
    ref.solve_fields()
    return np.array(ref.eval_rhs("ions"))


def run_compiled(P, dt):
    """compile_problem(P) -> install -> one step; returns the advanced state (or None to skip)."""
    try:
        compiled = pops.compile_problem(model=transport_model(), time=P)
    except RuntimeError as exc:  # no compiler / no Kokkos visible / compile failed
        _skip("compile_problem could not build the .so: %s" % str(exc)[:160])
    sim = make_sim()
    sim.install_program(compiled.so_path)
    sim.step(dt)
    return np.array(sim.get_state("ions"))


def ssprk2_program():
    P = adctime.Program("ssprk2_parity")
    dt = P.dt
    U0 = P.state("ions")
    f0 = P.solve_fields(U0)
    k0 = P.rhs(state=U0, fields=f0, flux=True, sources=["default"])
    U1 = P.linear_combine("U1", U0 + dt * k0)
    f1 = P.solve_fields(U1)
    k1 = P.rhs(state=U1, fields=f1, flux=True, sources=["default"])
    P.commit("ions", P.linear_combine("U2", 0.5 * U0 + 0.5 * (U1 + dt * k1)))
    return P


def rk4_program():
    P = adctime.Program("rk4_parity")
    dt = P.dt
    U0 = P.state("ions")
    k1 = P.rhs(state=U0, fields=P.solve_fields(U0), flux=True, sources=["default"])
    U1 = P.linear_combine("U1", U0 + 0.5 * dt * k1)
    k2 = P.rhs(state=U1, fields=P.solve_fields(U1), flux=True, sources=["default"])
    U2 = P.linear_combine("U2", U0 + 0.5 * dt * k2)
    k3 = P.rhs(state=U2, fields=P.solve_fields(U2), flux=True, sources=["default"])
    U3 = P.linear_combine("U3", U0 + dt * k3)
    k4 = P.rhs(state=U3, fields=P.solve_fields(U3), flux=True, sources=["default"])
    P.commit("ions", P.linear_combine(
        "Unp1", U0 + dt / 6.0 * k1 + dt / 3.0 * k2 + dt / 3.0 * k3 + dt / 6.0 * k4))
    return P


if not hasattr(pops.System(n=8, L=1.0, periodic=True), "install_program"):
    _skip("_pops lacks the install_program binding (rebuild _pops)")

dt = 2e-3
U0 = initial_state()
ref = make_sim()

print("== SSPRK2 ==")
k0 = offline_rhs(ref, U0)
U1 = U0 + dt * k0
k1 = offline_rhs(ref, U1)
ssprk2_ref = 0.5 * U0 + 0.5 * (U1 + dt * k1)
ssprk2_prog = run_compiled(ssprk2_program(), dt)
e = float(np.abs(ssprk2_prog - ssprk2_ref).max())
chk(e < 1e-12, "compiled SSPRK2 == offline stage reference (max|d| = %.2e)" % e)

# Native cross-check: the compiled SSPRK2 reproduces pops.Explicit("ssprk2") (spec test 32).
nat = make_sim("ssprk2")
nat.step(dt)
en = float(np.abs(ssprk2_prog - np.array(nat.get_state("ions"))).max())
chk(en < 1e-12, "compiled SSPRK2 == native pops.Explicit('ssprk2') (max|d| = %.2e)" % en)

print("== RK4 ==")
k1 = offline_rhs(ref, U0)
k2 = offline_rhs(ref, U0 + 0.5 * dt * k1)
k3 = offline_rhs(ref, U0 + 0.5 * dt * k2)
k4 = offline_rhs(ref, U0 + dt * k3)
rk4_ref = U0 + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
rk4_prog = run_compiled(rk4_program(), dt)
e = float(np.abs(rk4_prog - rk4_ref).max())
chk(e < 1e-12, "compiled RK4 == offline stage reference (max|d| = %.2e)" % e)
chk(float(np.abs(rk4_prog - U0).max()) > 1e-9, "RK4 actually advanced the state")

print("%s test_time_multistage" % ("FAIL (%d)" % fails if fails else "PASS"))
sys.exit(1 if fails else 0)
