#!/usr/bin/env python3
"""Forward Euler as a compiled time Program (epic ADC-399).

Writes a one-step Forward-Euler algorithm with ``pops.time.Program``, compiles it to a ``problem.so``
with ``pops.compile_problem``, installs it on a System, and advances one step entirely C++-side via
``sim.step`` -- then checks it reproduces the native ``pops.Explicit("euler")`` one-step
``U0 + dt * rhs`` to machine precision.

Run::

    python examples/time_programs/forward_euler_program.py

Requires a C++ compiler and a visible Kokkos (``POPS_KOKKOS_ROOT``); prints a skip notice and exits 0
otherwise (so it is safe in a docs/CI smoke run). cf. docs/sphinx/reference/time-program.md.
"""
import sys

try:
    import numpy as np

    import pops
    from pops import time as adctime
except Exception as exc:  # noqa: BLE001  -- pops/numpy unavailable in this interpreter
    print("skip forward_euler_program (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)


def euler_model():
    """A pure-transport isothermal block (no source); the inert elliptic + set_poisson make the
    program's solve_fields well-defined."""
    return pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                     transport=pops.IsothermalFlux(),
                     source=pops.NoSource(),
                     elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0))


def forward_euler_program():
    """U^{n+1} = U^n + dt * R(U^n), built as a typed IR (no arrays computed here)."""
    P = adctime.Program("forward_euler_example")
    dt = P.dt
    U = P.state("plasma")
    fields = P.solve_fields(U)
    R = P.rhs(state=U, fields=fields, flux=True, sources=["default"])
    P.commit("plasma", P.linear_combine("U1", U + dt * R))
    return P


def build_system():
    n = 48
    sim = pops.System(n=n, L=1.0, periodic=True)
    sim.add_block("plasma", euler_model(),
                  spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                  time=pops.Explicit(method="euler"))
    sim.set_poisson("charge_density", "geometric_mg")
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("plasma", np.stack([rho, 0.4 * rho, -0.2 * rho]))
    return sim


def main():
    if not hasattr(pops.System(n=8, L=1.0, periodic=True), "install_program"):
        print("skip forward_euler_program (_pops lacks the install_program binding; rebuild _pops)")
        return 0

    dt = 2e-3

    # Reference: the native one-step Forward Euler via the same primitives the program drives.
    ref = build_system()
    U0 = np.array(ref.get_state("plasma"))
    ref.solve_fields()
    U_ref = U0 + dt * np.array(ref.eval_rhs("plasma"))

    # Compiled time-program path.
    try:
        compiled = pops.compile_problem(model=euler_model(), time=forward_euler_program())
    except RuntimeError as exc:  # no compiler / no Kokkos visible / compile failed
        print("skip forward_euler_program (compile_problem could not build the .so: %s)"
              % str(exc)[:160])
        return 0

    sim = build_system()
    sim.install_program(compiled.so_path)
    sim.step(dt)
    U_prog = np.array(sim.get_state("plasma"))

    err = float(np.abs(U_prog - U_ref).max())
    print("compiled Forward-Euler Program vs native one-step: max|d| = %.2e" % err)
    print("problem.so: %s" % compiled.so_path)
    ok = err < 1e-12
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
