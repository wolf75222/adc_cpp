#!/usr/bin/env python3
"""Adams-Bashforth 2 over a System-owned history as a compiled time Program (epic ADC-399 / ADC-406a).

A compiled Program can declare / read / write a SYSTEM-OWNED history field carried across macro-steps
(a HistoryManager in System::Impl, not a closure capture), which enables the explicit 2-step AB2
recurrence ``U^{n+1} = U + dt*(3/2 R_n - 1/2 R_{n-1})`` then ``store_history(block.R, R_n)``. The
``pops.lib.time.std.adams_bashforth2`` macro builds this with ``P.history(name, lag=1)`` / ``store_history``
(the codegen appends ``ctx.rotate_histories()`` at the end of the step body).

COLD START (step 0): the runtime fills EVERY history slot on the FIRST store, so step 0 reads
R_{n-1} = R_0 and AB2 degenerates to one Forward-Euler step. The offline reference mirrors this
exactly (FE step 0, AB2 thereafter), so the comparison is to machine precision. Mirrors
python/tests/test_time_history.py, which matches to ~1.11e-16.

The model is a 1-variable block (rho) with ZERO flux and a manufactured LINEAR source S(rho) = c*rho
(so R = c*rho changes every step), stepped N macro-steps. Run::

    python examples/time_programs/adams_bashforth2_program.py

Requires a compiler + a visible Kokkos (``POPS_KOKKOS_ROOT``); prints a skip notice and exits 0
otherwise. cf. docs/sphinx/reference/time-program.md.
"""
import sys

try:
    import numpy as np

    import pops
    from pops.physics.facade import Model
    from pops import time as adctime
    import pops.lib.time as libtime  # ready schemes live in pops.lib.time (Spec 4)
except Exception as exc:  # noqa: BLE001
    print("skip adams_bashforth2_program (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)

C = 0.75  # source coefficient: S(rho) = C * rho (a linear ODE rho' = c rho; R changes every step)


def source_model(name):
    """A 1-variable model (rho), ZERO flux, default LINEAR source S(rho) = C*rho (so R = c*rho changes
    every step). A complete compilable block (flux + primitive + eigenvalue + source)."""
    m = Model(name)
    (rho,) = m.conservative_vars("rho")
    u = m.primitive("u", 0.0 * rho)
    m.primitive_vars(rho=rho, u=u)
    m.conservative_from([rho])
    m.flux(x=[0.0 * rho], y=[0.0 * rho])
    m.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
    m.source([C * rho])  # default source folded by ctx.rhs_into
    return m


def ab2_program():
    """The AB2 step built via the std macro: R_n = R(U); U^{n+1} = U + dt*(3/2 R_n - 1/2 R_{n-1});
    store_history(blk.R, R_n). The lag-1 read R_{n-1} = P.history("blk.R", lag=1)."""
    P = adctime.Program("adams_bashforth2_example")
    libtime.std.adams_bashforth2(P, "blk")
    return P


def offline_ab2(rho0, dt, nsteps):
    """The IDENTICAL AB2 recurrence, cell by cell, with the same FE cold start the runtime uses:
        R_n = C * rho_n
        rho_{n+1} = rho_n + dt*(3/2 R_n - 1/2 R_{n-1})     (R_{-1} := R_0 -> step 0 is FE)
    Returns the final rho after @p nsteps macro-steps."""
    rho = rho0.copy()
    r_prev = C * rho  # cold start: R_{-1} = R_0 (first store fills all slots) -> step 0 is FE
    for _ in range(nsteps):
        r_n = C * rho
        rho = rho + dt * (1.5 * r_n - 0.5 * r_prev)
        r_prev = r_n
    return rho


def main():
    n = 16
    if not hasattr(pops.System(n=8, L=1.0, periodic=True), "install_program"):
        print("skip adams_bashforth2_program (_pops lacks install_program; rebuild _pops)")
        return 0
    try:
        compiled = pops.compile_problem(model=source_model("ab2_prog"), time=ab2_program())
        block_model = source_model("ab2_block").compile(backend="production")
    except RuntimeError as exc:
        print("skip adams_bashforth2_program (compile_problem could not build the .so: %s)"
              % str(exc)[:160])
        return 0

    sim = pops.System(n=n, L=1.0, periodic=True)
    sim.add_equation("blk", block_model,
                     spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                     time=pops.Explicit(method="euler"))
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("blk", np.stack([rho0]))

    sim.install_program(compiled.so_path)
    dt = 0.01
    nsteps = 5
    for _ in range(nsteps):
        sim.step(dt)
    U_prog = np.array(sim.get_state("blk"))[0]

    rho_ref = offline_ab2(rho0, dt, nsteps)
    err = float(np.abs(U_prog - rho_ref).max())
    print("compiled AB2 Program vs offline AB2 reference over %d steps: max|d| = %.2e" % (nsteps, err))
    ok = err < 1e-12
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
