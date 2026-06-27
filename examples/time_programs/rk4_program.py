#!/usr/bin/env python3
"""Classic RK4 as a compiled multi-stage time Program (epic ADC-399 / ADC-407).

Writes the four-stage RK4 scheme with ``pops.time.Program`` (three intermediate stage states + a
linear-combination commit), compiles it to a ``problem.so``, installs it, advances one step
C++-side, and checks it against an offline stage-by-stage reference built from the same runtime
primitives. There is NO special RK4 C++ class -- the scheme is just IR lowered by the codegen.

Run::

    python examples/time_programs/rk4_program.py

Requires a compiler + a visible Kokkos (``POPS_KOKKOS_ROOT``); prints a skip notice and exits 0
otherwise. cf. docs/sphinx/reference/time-program.md.
"""
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
import sys

try:
    import numpy as np

    import pops
    from pops import time as adctime
except Exception as exc:  # noqa: BLE001
    print("skip rk4_program (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)


def gas_model():
    return pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                     transport=pops.IsothermalFlux(),
                     source=pops.NoSource(),
                     elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0))


N = 48


def initial_state():
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    return np.stack([rho, 0.4 * rho, -0.2 * rho])


def build_system():
    """The native reference System (lower-level add_block path), evaluated one RHS stage at a time."""
    sim = pops.System(n=N, L=1.0, periodic=True)
    sim.add_block("plasma", gas_model(),
                  spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                  time=pops.Explicit(method="euler"))
    sim.set_poisson("charge_density", "geometric_mg")
    sim.set_state("plasma", initial_state())
    return sim


def rk4_program():
    P = adctime.Program("rk4_example")
    dt = P.dt
    U0 = P.state("plasma")
    k1 = P.rhs(state=U0, fields=P.solve_fields(U0), flux=True, sources=["default"])
    U1 = P.linear_combine("U1", U0 + 0.5 * dt * k1)
    k2 = P.rhs(state=U1, fields=P.solve_fields(U1), flux=True, sources=["default"])
    U2 = P.linear_combine("U2", U0 + 0.5 * dt * k2)
    k3 = P.rhs(state=U2, fields=P.solve_fields(U2), flux=True, sources=["default"])
    U3 = P.linear_combine("U3", U0 + dt * k3)
    k4 = P.rhs(state=U3, fields=P.solve_fields(U3), flux=True, sources=["default"])
    P.commit("plasma", P.linear_combine(
        "Unp1", U0 + dt / 6.0 * k1 + dt / 3.0 * k2 + dt / 3.0 * k3 + dt / 6.0 * k4))
    return P


def offline_rhs(ref, U):
    ref.set_state("plasma", U)
    ref.solve_fields()
    return np.array(ref.eval_rhs("plasma"))


def main():
    if not hasattr(pops.System(n=8, L=1.0, periodic=True), "install_program"):
        print("skip rk4_program (_pops lacks install_program; rebuild _pops)")
        return 0
    dt = 2e-3
    try:
        compiled = pops.compile_problem(model=gas_model(), time=rk4_program())
    except RuntimeError as exc:
        print("skip rk4_program (compile_problem could not build the .so: %s)" % str(exc)[:160])
        return 0

    # Compiled path via the unified headline entry: install() wires the block instance, its initial
    # state and the Poisson solver, then installs the compiled time Program -- in one call.
    sim = pops.System(n=N, L=1.0, periodic=True)
    sim.install(compiled,
                instances={"plasma": {"model": gas_model(),
                                      "spatial": pops.FiniteVolume(limiter=FirstOrder(),
                                                                   riemann=Rusanov()),
                                      "initial": initial_state()}},
                solvers={"phi": pops.lib.fields.GeometricMG()})
    U0 = np.array(sim.get_state("plasma"))
    sim.step(dt)
    U_prog = np.array(sim.get_state("plasma"))

    ref = build_system()
    k1 = offline_rhs(ref, U0)
    k2 = offline_rhs(ref, U0 + 0.5 * dt * k1)
    k3 = offline_rhs(ref, U0 + 0.5 * dt * k2)
    k4 = offline_rhs(ref, U0 + dt * k3)
    U_ref = U0 + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    err = float(np.abs(U_prog - U_ref).max())
    print("compiled RK4 Program vs offline stage reference: max|d| = %.2e" % err)
    ok = err < 1e-12
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
