#!/usr/bin/env python3
"""Spec 2 (S2-9 / ADC-445): perf of a compiled time Program vs the native stepper.

The HPC contract: a generated Program is NOT an alternative runtime, it is a specialized scheduler
that lowers every heavy operation to the same adc_cpp primitives (flux, fill_boundary, reductions,
...). So a compiled SSPRK3 Program should step at a rate comparable to the native SSPRK3 stepper --
no silent CPU fallback, no per-step Python.

This benchmark times the SSPRK3 step loop for the SAME 2D Euler model two ways:
  1. native     -- ``pops.Explicit(method="ssprk3")`` driving ``sim.step``;
  2. program    -- ``pops.lib.time.ssprk3`` -> ``compile_problem`` -> ``sim.install_program``,
                   which lowers to the same three Shu-Osher stages.
It reports ms/step for each and the generated/native ratio. Needs a compiler + Kokkos
(``POPS_KOKKOS_ROOT``); prints a skip notice and exits 0 otherwise (run it on ROMEO).

The model is pure Euler with NO elliptic coupling, so ``solve_fields`` is a no-op and the two
paths do the same work per step. (On a field-coupled model the comparison would be apples-to-
oranges: the generated Program re-solves the elliptic field at each SSPRK3 stage -- the correct,
stage-consistent semantics -- versus one step-level solve in the native path; pick a transport-only
model for a clean scheduler-overhead measurement.)
"""
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
import sys
import time

try:
    import numpy as np

    import pops
    from pops.ir.ops import sqrt
    from pops.physics.facade import Model
    from pops import time as adctime
    import pops.lib.time as libtime  # ready schemes live in pops.lib.time (Spec 4)
except Exception as exc:  # noqa: BLE001
    print("skip operator_first_perf (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)

GAMMA = 1.4
N = 256
STEPS = 50
WARMUP = 5
DT = 1.0e-4


def euler_model(name):
    m = Model(name)
    rho, rhou, rhov, e = m.conservative_vars(
        "rho", "rho_u", "rho_v", "E",
        roles=["Density", "MomentumX", "MomentumY", "Energy"])
    g = m.param("gamma", GAMMA)
    u = m.primitive("u", rhou / rho)
    v = m.primitive("v", rhov / rho)
    p = m.primitive("p", (g - 1.0) * (e - 0.5 * rho * (u * u + v * v)))
    h = (e + p) / rho
    c = sqrt(g * p / rho)
    m.flux(x=[rhou, rhou * u + p, rhou * v, rho * h * u],
           y=[rhov, rhov * u, rhov * v + p, rho * h * v])
    m.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    m.primitive_vars(rho=rho, u=u, v=v, p=p)
    m.conservative_from([rho, rho * u, rho * v,
                         p / (g - 1.0) + 0.5 * rho * (u * u + v * v)])
    return m


def initial_state(n):
    x = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.1 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
    u = 0.2 + 0.05 * np.sin(2 * np.pi * yy)
    v = -0.1 + 0.05 * np.cos(2 * np.pi * xx)
    p = np.ones_like(rho)
    state = np.zeros((4, n, n))
    state[0] = rho
    state[1] = rho * u
    state[2] = rho * v
    state[3] = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)
    return state


def native_sim():
    sim = pops.System(n=N, L=1.0, periodic=True)
    sim.add_equation("gas", euler_model("perf_native").compile(backend="production"),
                     spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                     time=pops.Explicit(method="ssprk3"))
    sim.set_state("gas", initial_state(N).reshape(-1))
    return sim


def program_sim():
    m = euler_model("perf_program")
    prog = adctime.Program("ssprk3_perf")
    libtime.ssprk3(prog, "gas", sources=[], flux=True)
    compiled = pops.compile_problem(model=m, time=prog)
    sim = pops.System(n=N, L=1.0, periodic=True)
    sim.add_equation("gas", m.compile(backend="production"),
                     spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                     time=pops.Explicit(method="euler"))  # block; the installed Program drives step
    sim.set_state("gas", initial_state(N).reshape(-1))
    sim.install_program(compiled.so_path)
    return sim


def ms_per_step(sim):
    for _ in range(WARMUP):
        sim.step(DT)
    t0 = time.perf_counter()
    for _ in range(STEPS):
        sim.step(DT)
    return (time.perf_counter() - t0) * 1000.0 / STEPS


def main():
    if not hasattr(pops.System(n=8, L=1.0, periodic=True), "install_program"):
        print("skip operator_first_perf (_pops lacks install_program; rebuild _pops)")
        return 0
    try:
        native = native_sim()
        program = program_sim()
    except RuntimeError as exc:
        print("skip operator_first_perf (compile needs Kokkos: %s)" % str(exc)[:140])
        return 0

    native_ms = ms_per_step(native)
    program_ms = ms_per_step(program)
    ratio = program_ms / native_ms if native_ms > 0 else float("inf")
    print("grid %dx%d, SSPRK3, %d steps (+%d warmup)" % (N, N, STEPS, WARMUP))
    print("native  stepper : %.4f ms/step" % native_ms)
    print("program (.so)   : %.4f ms/step" % program_ms)
    print("ratio (program/native) = %.3f" % ratio)
    # The contract is "comparable", not "identical": a small scheduler overhead is acceptable.
    print("OK  generated comparable to native" if ratio <= 1.5
          else "WARN  generated slower than native (ratio %.2f)" % ratio)
    return 0


if __name__ == "__main__":
    sys.exit(main())
