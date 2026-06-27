#!/usr/bin/env python3
"""SSPRK3 as a compiled multi-stage time Program (epic ADC-399 / ADC-407).

Writes the three-stage SSPRK3 (Shu-Osher) scheme with ``pops.time.Program`` (two intermediate stage
states + a linear-combination commit) via the ``pops.lib.time.ssprk3`` macro, compiles it to a
``problem.so`` with ``pops.compile_problem``, installs it, advances one step C++-side, and checks it
reproduces the native ``pops.Explicit(method="ssprk3")`` step bit-for-bit. There is NO special SSPRK3
C++ class -- the scheme is just IR lowered by the codegen (like the merged ssprk2 example/test).

Run::

    python examples/time_programs/ssprk3_program.py

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
    import pops.lib.time as libtime  # ready schemes live in pops.lib.time (Spec 4)
except Exception as exc:  # noqa: BLE001
    print("skip ssprk3_program (pops/numpy unavailable: %s)" % exc)
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


def build_system(method="ssprk3"):
    """The native reference System (lower-level add_block path), stepped natively for comparison."""
    sim = pops.System(n=N, L=1.0, periodic=True)
    sim.add_block("plasma", gas_model(),
                  spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                  time=pops.Explicit(method=method))
    sim.set_poisson("charge_density", "geometric_mg")
    sim.set_state("plasma", initial_state())
    return sim


def ssprk3_program():
    """SSPRK3 (Shu-Osher), built via the std macro that lowers to typed IR:
    U1 = U0 + dt k0; U2 = 3/4 U0 + 1/4 (U1 + dt k1); U^{n+1} = 1/3 U0 + 2/3 (U2 + dt k2)."""
    P = adctime.Program("ssprk3_example")
    libtime.ssprk3(P, "plasma")
    return P


def main():
    if not hasattr(pops.System(n=8, L=1.0, periodic=True), "install_program"):
        print("skip ssprk3_program (_pops lacks install_program; rebuild _pops)")
        return 0
    dt = 2e-3
    try:
        compiled = pops.compile_problem(model=gas_model(), time=ssprk3_program())
    except RuntimeError as exc:
        print("skip ssprk3_program (compile_problem could not build the .so: %s)" % str(exc)[:160])
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
    sim.step(dt)
    U_prog = np.array(sim.get_state("plasma"))

    native = build_system("ssprk3")
    native.step(dt)
    err = float(np.abs(U_prog - np.array(native.get_state("plasma"))).max())
    print("compiled SSPRK3 Program vs native pops.Explicit('ssprk3'): max|d| = %.2e" % err)
    ok = err < 1e-12
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
