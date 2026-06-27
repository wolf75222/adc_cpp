#!/usr/bin/env python3
"""Strang splitting H(dt/2); S(dt); H(dt/2) as a compiled time Program (epic ADC-399 / ADC-410).

Builds the Strang composition ONCE as Program IR via the ``pops.lib.time.std.strang`` combinator (no
scheme-specific C++ stepper), compiles it to a ``problem.so``, installs it, advances N steps
C++-side, and checks it reproduces the native engine Strang macro-step
(``pops.Strang`` via ``set_time_scheme("strang")``) BIT-for-bit. Mirrors
python/tests/test_time_strang_parity.py, which proves max|d| = 0.00e+00.

The simple case where the compiled half-flow EXACTLY mirrors native H: an UNCOUPLED isothermal
fluid (no Poisson feedback into the flux), NO source brick, Forward Euler. There native H(dt/2) is a
single Euler transport step U + (dt/2)*(-div F); native S(dt) (run_source_stage) is a genuine no-op
(no Schur / source stage is installed); the per-stage solve_fields fences are inert. The compiled
program runs the matching half_flow (flux-only U + frac*dt*R, the model's default source is empty)
around a no-op source, so the two paths coincide to the last bit.

Run::

    python examples/time_programs/strang_program.py

Requires a compiler + a visible Kokkos (``POPS_KOKKOS_ROOT``); prints a skip notice and exits 0
otherwise. cf. docs/sphinx/reference/time-program.md.
"""
import sys

try:
    import numpy as np

    import pops
    from pops import time as adctime
    import pops.lib.time as libtime  # ready schemes live in pops.lib.time (Spec 4)
except Exception as exc:  # noqa: BLE001
    print("skip strang_program (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)


def transport_model():
    """Uncoupled isothermal fluid (no field coupling into the flux), NO source brick: native H is a
    pure Euler transport step and native S (run_source_stage) is a no-op."""
    return pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                     transport=pops.IsothermalFlux(),
                     source=pops.NoSource(),
                     elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0))


def half_flow(prog, U, frac):
    """One Forward-Euler hyperbolic half-flow U + frac*dt*R, R = -div F (the default source is empty
    on the uncoupled NoSource model). The per-stage solve_fields is inert (no field feedback)."""
    R = prog.rhs(state=U, fields=prog.solve_fields(U), flux=True, sources=["default"])
    return prog.linear_combine(None, U + (frac * prog.dt) * R)


def no_op_source(prog, U, frac):  # noqa: ARG001  -- frac unused: S is the identity on a NoSource model
    """The Strang source stage S(dt). With no source brick the native run_source_stage is a no-op, so
    the compiled source returns U unchanged (no extra IR stage)."""
    return U


def strang_program(name="strang_example", block="ions"):
    """The compiled Strang program H(dt/2); S(dt); H(dt/2) built via pops.lib.time.std.strang."""
    P = adctime.Program(name)
    libtime.std.strang(P, block, half_flow, no_op_source)
    return P


def make_sim():
    n = 24
    sim = pops.System(n=n, L=1.0, periodic=True)
    sim.add_block("ions", transport_model(),
                  spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                  time=pops.Explicit(method="euler"))
    sim.set_poisson("charge_density", "geometric_mg")  # inert: BackgroundDensity n0=0, flux reads no phi
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("ions", np.stack([rho, 0.4 * rho, -0.2 * rho]))
    return sim


def main():
    if not hasattr(pops.System(n=8, L=1.0, periodic=True), "install_program"):
        print("skip strang_program (_pops lacks install_program; rebuild _pops)")
        return 0
    try:
        compiled = pops.compile_problem(model=transport_model(), time=strang_program())
    except RuntimeError as exc:
        print("skip strang_program (compile_problem could not build the .so: %s)" % str(exc)[:160])
        return 0

    dt = 2e-3
    nstep = 4

    # Native engine Strang: set_time_scheme("strang") drives SystemStepper::step_strang. WITHOUT a
    # Schur / source stage, run_source_stage is a no-op -- exactly H(dt/2); no-op; H(dt/2).
    native = make_sim()
    native._s.set_time_scheme("strang")

    # Compiled Strang program: installed, driven by sim.step(dt).
    prog = make_sim()
    prog.install_program(compiled.so_path)

    for _ in range(nstep):
        native.step(dt)
        prog.step(dt)

    U_native = np.array(native.get_state("ions"))
    U_prog = np.array(prog.get_state("ions"))
    err = float(np.abs(U_native - U_prog).max())
    print("compiled std.strang vs native pops.Strang over %d steps: max|d| = %.2e" % (nstep, err))
    ok = np.array_equal(U_native, U_prog)
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
