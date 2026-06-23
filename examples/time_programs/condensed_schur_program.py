#!/usr/bin/env python3
"""Condensed-Schur: available primitives + the documented macro gap (epic ADC-399 / acceptance 32).

The full ``adc.time.std.condensed_schur`` macro is a DOCUMENTED STUB: a Program rewrite of the native
``adc.CondensedSchur`` global matrix-free Schur solve is blocked on two deep IR features out of scope
for this epic -- (A) multi-component ``solve_linear`` (``P.matrix_free_operator`` / ``P.solve_linear``
are scalar-field only) and (B) anisotropic position-dependent operator-coefficient assembly (the Schur
operator -div((I + c*rho*B^-1) grad phi) has a per-cell tensor coefficient). This example is therefore
a "primitives + documented gap" showcase, NOT a fake macro:

  (a) it RUNS the primitives a hand-rolled condensed-Schur stage already has -- ctx.divergence +
      ctx.gradient chained into a matrix-free operator solved with P.solve_linear(bicgstab). The
      operator is the SCALAR Schur-like flux operator A(phi) = phi - alpha*div(grad phi) (the
      div(flux) structure of the condensed Poisson operator on the WIDE-stencil Laplacian), checked
      against an offline NumPy CG on the SAME discrete operator (like divergence_solve.py);
  (b) it shows ``adc.time.std.condensed_schur(P)`` RAISING NotImplementedError with its message
      (caught + printed), pointing to the still-supported native ``adc.CondensedSchur`` source stepper.

The wide-stencil centered div(grad) is the Laplacian (x(i+2) - 2 x(i) + x(i-2))/(4 h^2); A is a
well-posed SPD Helmholtz operator on it. Run::

    python examples/time_programs/condensed_schur_program.py

Requires a compiler + a visible Kokkos (``ADC_KOKKOS_ROOT``); prints a skip notice and exits 0
otherwise. cf. docs/sphinx/reference/time-program.md.
"""
import sys

try:
    import numpy as np

    import adc
    from adc import dsl
    from adc import time as adctime
except Exception as exc:  # noqa: BLE001
    print("skip condensed_schur_program (adc/numpy unavailable: %s)" % exc)
    sys.exit(0)

ALPHA = 0.1  # Helmholtz coefficient: A = I - alpha*div(grad) is SPD (no null space)


def passive_model(name):
    """A 1-variable block with no flux and no Poisson coupling: the Program runs neither a flux RHS nor
    solve_fields, so the single conservative variable is just the scalar field the solve writes."""
    m = dsl.Model(name)
    (rho,) = m.conservative_vars("rho")
    u = m.primitive("u", 0.0 * rho)
    m.primitive_vars(rho=rho, u=u)
    m.conservative_from([rho])
    m.flux(x=[0.0 * rho], y=[0.0 * rho])
    m.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
    return m


def schur_like_program(method="bicgstab", tol=1e-10, max_iter=200):
    """phi = solve (I - alpha*div(grad)) phi = U, matrix-free via ctx.gradient -> ctx.divergence, then
    P.solve_linear -- the AVAILABLE primitives a hand-rolled condensed-Schur stage would use."""
    P = adctime.Program("condensed_schur_primitives_example")
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


def show_documented_gap():
    """(b) std.condensed_schur is a documented NotImplementedError stub: show it raising, print the
    message (it points to the native adc.CondensedSchur + the available primitives)."""
    P = adctime.Program("condensed_schur_gap")
    try:
        adctime.std.condensed_schur(P)
    except NotImplementedError as exc:
        print("std.condensed_schur is a documented stub -- it raises NotImplementedError:")
        print("  %s" % str(exc)[:200])
        return True
    print("UNEXPECTED: std.condensed_schur did not raise NotImplementedError")
    return False


def main():
    n = 16
    if not hasattr(adc.System(n=8, L=1.0, periodic=True), "install_program"):
        print("skip condensed_schur_program (_adc lacks the install_program binding; rebuild _adc)")
        return 0

    # (b) the documented gap is pure Python -- it always runs (no toolchain needed).
    gap_ok = show_documented_gap()

    # (a) the available primitives: a matrix-free Schur-like solve.
    try:
        compiled = adc.compile_problem(model=passive_model("cs_prog"), time=schur_like_program())
        block_model = passive_model("cs_blk").compile(backend="production")
    except RuntimeError as exc:  # no compiler / no Kokkos visible / compile failed
        print("skip condensed_schur_program (compile_problem could not build the .so: %s)"
              % str(exc)[:160])
        return 0

    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.add_equation("blk", block_model,
                     spatial=adc.FiniteVolume(limiter="none", riemann="rusanov"),
                     time=adc.Explicit(method="euler"))
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    b = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)  # the right-hand side (= U)
    sim.set_state("blk", np.stack([b]))

    sim.install_program(compiled.so_path)
    sim.step(0.01)  # dt is irrelevant -- the program is a pure solve
    phi_prog = np.array(sim.get_state("blk"))[0]

    phi_ref, iters = offline_cg(discrete_divgrad_helmholtz(n, ALPHA), b)
    err = float(np.abs(phi_prog - phi_ref).max())
    print("compiled div(grad) matrix-free Schur-like solve vs offline NumPy CG: max|d| = %.2e  "
          "(offline iters = %d)" % (err, iters))
    ok = gap_ok and err < 1e-9
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
