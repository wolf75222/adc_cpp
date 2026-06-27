#!/usr/bin/env python3
"""Matrix-free global linear solve as a compiled time Program (epic ADC-399 / ADC-405, spec example 6).

Builds a matrix-free operator ``A x = x - alpha*Lap(x)`` (a symmetric-positive-definite Helmholtz
operator) whose ``apply`` is an IR sub-block (``P.set_apply``, written with ``P.laplacian`` + the
affine algebra), then solves ``A phi = b`` with ``P.solve_linear`` -- which lowers to the runtime's
matrix-free Krylov loop (``pops::cg_solve``). The dynamic iteration runs entirely C++-side; the IR
carries only the apply, the right-hand side, and the solver options. Nothing re-enters Python.

The block has a single conservative variable, so its state doubles as the scalar field: the Program
solves ``(I - alpha*Lap) phi = U`` and commits ``phi`` back into the block. The result is checked
against a plain offline NumPy CG on the identical discrete periodic 5-point system.

``method=`` selects the runtime solver (``cg`` for SPD here; ``bicgstab`` and ``richardson`` are also
available). Run::

    python examples/time_programs/matrix_free_solve.py

Requires a C++ compiler and a visible Kokkos (``POPS_KOKKOS_ROOT``); prints a skip notice and exits 0
otherwise. cf. docs/sphinx/reference/time-program.md.
"""
import sys

try:
    import numpy as np

    import pops
    from pops.physics.facade import Model
    from pops import time as adctime
except Exception as exc:  # noqa: BLE001  -- pops/numpy unavailable in this interpreter
    print("skip matrix_free_solve (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)

ALPHA = 0.1  # Helmholtz coefficient: A = I - alpha*Lap is SPD positive-definite (no null space)


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


def solve_program(method="cg", tol=1e-10, max_iter=200):
    """phi = solve (I - alpha*Lap) phi = U, matrix-free, committed back into the block."""
    P = adctime.Program("matrix_free_solve_example")
    U = P.state("blk")
    A = P.matrix_free_operator("A")

    def apply(P, out, x):  # out <- A(x) = x - alpha*Lap(x); the affine is the lowered result
        lap = P.scalar_field("lap")
        P.laplacian(lap, x)
        return x - ALPHA * lap

    P.set_apply(A, apply)
    phi = P.solve_linear(operator=A, rhs=U, method=method, tol=tol, max_iter=max_iter)
    P.commit("blk", phi)
    return P


def discrete_helmholtz(n, alpha):
    """A x = x - alpha*Lap(x), the discrete periodic 5-point matvec matching pops::apply_laplacian."""
    h2 = (1.0 / n) ** 2

    def apply(x):
        lap = (np.roll(x, -1, 0) + np.roll(x, 1, 0) - 2 * x) / h2 + \
              (np.roll(x, -1, 1) + np.roll(x, 1, 1) - 2 * x) / h2
        return x - alpha * lap

    return apply


def offline_cg(apply, b, tol=1e-10, max_iter=2000):
    """Plain NumPy CG solving A x = b from x = 0 -- the reference for the compiled matrix-free CG."""
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
        print("skip matrix_free_solve (_pops lacks the install_program binding; rebuild _pops)")
        return 0

    try:
        compiled = pops.compile_problem(model=passive_model("mf_prog"), time=solve_program())
        block_model = passive_model("mf_blk").compile(backend="production")
    except RuntimeError as exc:  # no compiler / no Kokkos visible / compile failed
        print("skip matrix_free_solve (compile_problem could not build the .so: %s)" % str(exc)[:160])
        return 0

    sim = pops.System(n=n, L=1.0, periodic=True)
    sim.add_equation("blk", block_model,
                     spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                     time=pops.Explicit(method="euler"))
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    b = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)  # the right-hand side (= U)
    sim.set_state("blk", np.stack([b]))

    sim.install_program(compiled.so_path)
    sim.step(0.01)  # dt is irrelevant -- the program is a pure solve
    phi_prog = np.array(sim.get_state("blk"))[0]

    phi_ref, iters = offline_cg(discrete_helmholtz(n, ALPHA), b)
    err = float(np.abs(phi_prog - phi_ref).max())
    print("compiled matrix-free CG vs offline NumPy CG: max|d| = %.2e  (offline iters = %d)"
          % (err, iters))
    print("problem.so: %s" % compiled.so_path)
    ok = err < 1e-9
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
