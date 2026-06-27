#!/usr/bin/env python3
"""Schur-like matrix-free solve via the divergence primitive (epic ADC-399 / ADC-412, acceptance 32).

Builds a matrix-free operator ``A(phi) = phi - alpha*div(grad phi)`` whose apply is an IR sub-block
(``P.set_apply``) written with ``P.gradient`` (into a 2-component buffer) chained into ``P.divergence``,
then solves ``A phi = b`` with ``P.solve_linear`` -- which lowers to the runtime's matrix-free Krylov
loop (``pops::bicgstab_solve``). This exercises ``ctx.divergence`` (the centered finite-volume divergence
``pops::apply_divergence``, the ``div(flux)`` structure of the condensed-Schur operator) inside a real
matrix-free solve.

The centered ``div(grad)`` is the WIDE-stencil Laplacian ``(x(i+2) - 2 x(i) + x(i-2))/(4 h^2)`` (not the
compact 5-point ``apply_laplacian``), so ``A == (I - alpha*div(grad))`` is a well-posed SPD Helmholtz
operator on that wide stencil; the compiled result is checked against a plain offline NumPy CG on the
SAME discrete operator. The single conservative variable doubles as the scalar field: the Program solves
``(I - alpha*div(grad)) phi = U`` and commits ``phi`` back into the block.

This is the scalar divergence + Krylov building block of the condensed-Schur Program (acceptance 32);
the FULL anisotropic ``pops.lib.time.condensed_schur`` macro (ADC-421) is in
``condensed_schur_program.py`` and ``python/tests/test_time_condensed_schur.py``. The native
``pops.CondensedSchur`` source stepper remains supported. Run::

    python examples/time_programs/divergence_solve.py

Requires a C++ compiler and a visible Kokkos (``POPS_KOKKOS_ROOT``); prints a skip notice and exits 0
otherwise. cf. docs/sphinx/reference/time-program.md.
"""
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
import sys

try:
    import numpy as np

    import pops
    from pops.physics.facade import Model
    from pops import time as adctime
except Exception as exc:  # noqa: BLE001  -- pops/numpy unavailable in this interpreter
    print("skip divergence_solve (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)

ALPHA = 0.1  # Helmholtz coefficient: A = I - alpha*div(grad) = I - alpha*Lap is SPD (no null space)


def passive_model(name):
    """A 1-variable block with no flux and no Poisson coupling: the Program runs neither a flux RHS nor
    solve_fields, so the single conservative variable is just the scalar field the solve writes."""
    m = Model(name)
    (rho,) = m.conservative_vars("rho")
    u = m.primitive("u", 0.0 * rho)
    m.primitive_vars(rho=rho, u=u)
    m.conservative_from([rho])
    m.flux(x=[0.0 * rho], y=[0.0 * rho])
    m.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
    return m


def solve_program(method="bicgstab", tol=1e-10, max_iter=200):
    """phi = solve (I - alpha*div(grad)) phi = U, matrix-free, committed back into the block."""
    P = adctime.Program("divergence_solve_example")
    U = P.state("blk")
    A = P.matrix_free_operator("A")

    def apply(P, out, x):  # out <- A(x) = x - alpha*div(grad(x)); the affine is the lowered result
        g = P.scalar_field("g", ncomp=2)  # 2-component gradient buffer (d/dx, d/dy)
        P.gradient(g, x)
        d = P.scalar_field("d")
        P.divergence(d, g, g)  # div(grad x) == Lap x (fy reads component 1 of the same buffer)
        return x - ALPHA * d

    P.set_apply(A, apply)
    phi = P.solve_linear(operator=A, rhs=U, method=method, tol=tol, max_iter=max_iter)
    P.commit("blk", phi)
    return P


def discrete_divgrad_helmholtz(n, alpha):
    """A x = x - alpha*div(grad x), the discrete periodic matvec matching the compiled centered
    gradient->divergence chain (the WIDE-stencil Laplacian (x(i+2) - 2 x(i) + x(i-2))/(4 h^2))."""
    h = 1.0 / n

    def apply(x):
        gx = (np.roll(x, -1, 0) - np.roll(x, 1, 0)) / (2 * h)
        gy = (np.roll(x, -1, 1) - np.roll(x, 1, 1)) / (2 * h)
        div = (np.roll(gx, -1, 0) - np.roll(gx, 1, 0)) / (2 * h) + \
              (np.roll(gy, -1, 1) - np.roll(gy, 1, 1)) / (2 * h)
        return x - alpha * div

    return apply


def offline_cg(apply, b, tol=1e-10, max_iter=2000):
    """Plain NumPy CG solving A x = b from x = 0 -- the reference for the compiled matrix-free solve."""
    x = np.zeros_like(b)
    r = b - apply(x)
    p = r.copy()
    rs_old = float(np.sum(r * r))
    bnorm = float(np.sqrt(np.sum(b * b))) or 1.0
    iters = 0
    for _ in range(max_iter):
        ap = apply(p)
        pap = float(np.sum(p * ap))
        if abs(pap) < 1e-300:
            break
        a = rs_old / pap
        x = x + a * p
        r = r - a * ap
        rs_new = float(np.sum(r * r))
        iters += 1
        if np.sqrt(rs_new) <= tol * bnorm:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x, iters


def main():
    n = 16
    if not hasattr(pops.System(n=8, L=1.0, periodic=True), "install_program"):
        print("skip divergence_solve (_pops lacks the install_program binding; rebuild _pops)")
        return 0

    try:
        compiled = pops.compile_problem(model=passive_model("div_prog"), time=solve_program())
    except RuntimeError as exc:  # no compiler / no Kokkos visible / compile failed
        print("skip divergence_solve (compile_problem could not build the .so: %s)" % str(exc)[:160])
        return 0

    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    b = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)  # the right-hand side (= U)

    # Compiled path via the unified headline entry: install() pre-resolves the board Model (compiling
    # it to the block), wires its initial state, then installs the compiled time Program -- in one call.
    # The passive block carries no Poisson coupling, so no solvers= is needed.
    sim = pops.System(n=n, L=1.0, periodic=True)
    sim.install(compiled,
                instances={"blk": {"model": passive_model("div_blk"),
                                   "spatial": pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                                   "time": pops.Explicit(method="euler"),
                                   "initial": np.stack([b])}})
    sim.step(0.01)  # dt is irrelevant -- the program is a pure solve
    phi_prog = np.array(sim.get_state("blk"))[0]

    phi_ref, iters = offline_cg(discrete_divgrad_helmholtz(n, ALPHA), b)
    err = float(np.abs(phi_prog - phi_ref).max())
    print("compiled div(grad) matrix-free solve vs offline NumPy CG: max|d| = %.2e  (offline iters = %d)"
          % (err, iters))
    print("problem.so: %s" % compiled.so_path)
    ok = err < 1e-9
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
