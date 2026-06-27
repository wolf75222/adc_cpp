#!/usr/bin/env python3
"""SSPRK2 as a compiled multi-stage time Program (epic ADC-399 / ADC-407).

Writes the two-stage SSPRK2 scheme with ``pops.time.Program`` (an intermediate stage state + a
linear-combination commit), compiles it to a ``problem.so`` with ``pops.compile_problem``, installs
it, advances one step C++-side, and checks it reproduces the native ``pops.Explicit("ssprk2")`` step
bit-for-bit. There is NO special SSPRK2 C++ class -- the scheme is just IR lowered by the codegen.

Run::

    python examples/time_programs/ssprk2_program.py

Requires a compiler + a visible Kokkos (``POPS_KOKKOS_ROOT``); prints a skip notice and exits 0
otherwise. cf. docs/sphinx/reference/time-program.md.
"""
import sys

try:
    import numpy as np

    import pops
    from pops import time as adctime
except Exception as exc:  # noqa: BLE001
    print("skip ssprk2_program (pops/numpy unavailable: %s)" % exc)
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


def build_system(method="ssprk2"):
    """The native reference System (lower-level add_block path), stepped natively for comparison."""
    sim = pops.System(n=N, L=1.0, periodic=True)
    sim.add_block("plasma", gas_model(),
                  spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                  time=pops.Explicit(method=method))
    sim.set_poisson("charge_density", "geometric_mg")
    sim.set_state("plasma", initial_state())
    return sim


def ssprk2_program():
    """U^{n+1} = 1/2 U^n + 1/2 (U1 + dt R(U1)), U1 = U^n + dt R(U^n) -- built as typed IR."""
    P = adctime.Program("ssprk2_example")
    dt = P.dt
    U0 = P.state("plasma")
    k0 = P.rhs(state=U0, fields=P.solve_fields(U0), flux=True, sources=["default"])
    U1 = P.linear_combine("U1", U0 + dt * k0)
    k1 = P.rhs(state=U1, fields=P.solve_fields(U1), flux=True, sources=["default"])
    P.commit("plasma", P.linear_combine("U2", 0.5 * U0 + 0.5 * (U1 + dt * k1)))
    return P


def main():
    if not hasattr(pops.System(n=8, L=1.0, periodic=True), "install_program"):
        print("skip ssprk2_program (_pops lacks install_program; rebuild _pops)")
        return 0
    dt = 2e-3
    try:
        compiled = pops.compile_problem(model=gas_model(), time=ssprk2_program())
    except RuntimeError as exc:
        print("skip ssprk2_program (compile_problem could not build the .so: %s)" % str(exc)[:160])
        return 0

    # Compiled path via the unified headline entry: install() wires the block instance, its initial
    # state and the Poisson solver, then installs the compiled time Program -- in one call.
    sim = pops.System(n=N, L=1.0, periodic=True)
    sim.install(compiled,
                instances={"plasma": {"model": gas_model(),
                                      "spatial": pops.FiniteVolume(limiter="none",
                                                                   riemann="rusanov"),
                                      "initial": initial_state()}},
                solvers={"phi": pops.lib.fields.GeometricMG()})
    sim.step(dt)
    U_prog = np.array(sim.get_state("plasma"))

    native = build_system("ssprk2")
    native.step(dt)
    err = float(np.abs(U_prog - np.array(native.get_state("plasma"))).max())
    print("compiled SSPRK2 Program vs native pops.Explicit('ssprk2'): max|d| = %.2e" % err)
    ok = err < 1e-12
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
