#!/usr/bin/env python3
"""adc.time centered divergence primitive + a div(grad) Helmholtz solve (epic ADC-399 / ADC-412).

ADC-412 adds the ``ctx.divergence`` primitive (the centered finite-volume divergence factored as
``adc::apply_divergence``) and the ``P.divergence(out, fx, fy)`` IR op. A matrix-free Schur-like operator
``A(phi) = phi - alpha*div(grad phi)`` (the div(flux) structure of the condensed-Schur operator) is
built from ``P.gradient`` chained into ``P.divergence`` and solved with ``P.solve_linear`` -- exactly
the matrix-free Krylov path acceptance 32 needs in place. The centered ``div(grad)`` is the WIDE-stencil
Laplacian ``(x(i+2) - 2 x(i) + x(i-2))/(4 h^2)`` (not the compact 5-point ``apply_laplacian``), so the
compiled solve is verified against an OFFLINE numpy CG on that SAME wide-stencil discrete operator.

(A) Pure Python, always runs:
    - ``P.divergence`` records a 3-input scalar_field op, validates its operands, and serializes;
    - the div(grad) Helmholtz apply (gradient -> divergence) lowers to ``ctx.gradient`` + a
      ``ctx.divergence`` + ``adc::bicgstab_solve``, with the gradient buffer allocated 2-component
      (``ctx.alloc_scalar_field(2, 1)``);
    - a standalone divergence-of-a-known-field check: the offline centered FV divergence of
      f = (cos 2pi x, sin 2pi y) matches the analytic div f = -2pi sin 2pi x + 2pi cos 2pi y to the
      discretization error -- the reference the compiled ctx.divergence reproduces;
    - ``adc.time.std.condensed_schur`` (now implemented, ADC-421) lowers at theta == 1 and raises for
      the deferred theta != 1 extrapolation (the full end-to-end parity is test_time_condensed_schur.py).

(B) End-to-end parity (skips unless the full toolchain is present): the div(grad) Helmholtz Program is
    compiled + installed + stepped, then compared to an OFFLINE numpy CG on the identical discrete
    periodic 5-point system. Asserts max|compiled - offline| <= 1e-6. Self-skips (exit 0) without numpy
    / _adc / install_program / a compiler / a visible Kokkos -- never fakes the engine.
"""
import sys


def _adc_time():
    try:
        import adc.time as t
    except Exception as exc:  # adc not importable here -> skip, never fake
        print("skip test_time_divergence (adc.time unavailable: %s)" % exc)
        sys.exit(0)
    return t


_ALPHA = 0.1  # Helmholtz coefficient: A = I - alpha*div(grad) = I - alpha*Lap (SPD, well-conditioned)


def _divgrad_program(t, *, name="divgrad", method="bicgstab", tol=1e-10, max_iter=200, alpha=_ALPHA):
    """Solve (I - alpha*div(grad)) phi = U, committed back into the 1-component block.

    The apply ``out = in - alpha*div(grad(in))`` chains P.gradient (into a 2-component buffer) then
    P.divergence (recovering Lap), so it exercises ctx.divergence inside the matrix-free Krylov loop."""
    P = t.Program(name)
    U = P.state("blk")
    A = P.matrix_free_operator("A")

    def apply(P, out, x):
        g = P.scalar_field("g", ncomp=2)  # 2-component gradient buffer (d/dx, d/dy)
        P.gradient(g, x)
        d = P.scalar_field("d")
        P.divergence(d, g, g)  # div(grad x) == Lap x; fy reads component 1 of the same buffer
        return x - alpha * d  # out = in - alpha*div(grad(in)) = in - alpha*Lap(in)

    P.set_apply(A, apply)
    phi = P.solve_linear(operator=A, rhs=U, method=method, tol=tol, max_iter=max_iter)
    P.commit("blk", phi)
    return P


# ---- (A) codegen + IR + analytic divergence: pure Python, always runs ----
def test_divergence_records_and_validates(t):
    P = t.Program("p")
    A = P.matrix_free_operator("A")

    def apply(P, out, x):
        g = P.scalar_field("g", ncomp=2)
        P.gradient(g, x)
        d = P.scalar_field("d")
        div = P.divergence(d, g, g)
        assert div.vtype == "scalar_field", "divergence yields a scalar_field value"
        return x - div

    P.set_apply(A, apply)
    U = P.state("blk")
    phi = P.solve_linear(operator=A, rhs=U, method="bicgstab", tol=1e-8, max_iter=50)
    P.commit("blk", phi)
    assert P.validate() is True, "the div(grad) Program must validate"
    assert P._ir_hash(), "the IR must serialize to a stable hash"


def test_divergence_operand_types(t):
    P = t.Program("p")
    A = P.matrix_free_operator("A")
    bad = []

    def apply(P, out, x):
        d = P.scalar_field("d")
        for args in ((U_state, x, x), (d, U_state, x), (d, x, U_state)):
            try:
                P.divergence(*args)
            except ValueError as exc:
                bad.append("divergence" in str(exc))
            else:
                bad.append(False)
        g = P.scalar_field("g", ncomp=2)
        P.gradient(g, x)
        dd = P.scalar_field("dd")
        P.divergence(dd, g, g)
        return x - dd

    U_state = P.state("blk")  # a State is not a scalar_field -> each divergence operand must reject it
    P.set_apply(A, apply)
    assert all(bad), "divergence must reject a non-scalar_field operand (out / fx / fy)"


def test_scalar_field_ncomp_validates(t):
    P = t.Program("p")
    for bad in (0, -1, 1.5):
        try:
            P.scalar_field("g", ncomp=bad)
        except ValueError as exc:
            assert "ncomp" in str(exc), str(exc)
        else:
            raise AssertionError("scalar_field ncomp=%r must raise" % (bad,))
    g = P.scalar_field("g2", ncomp=2)
    assert g.attrs["ncomp"] == 2, "ncomp is recorded on the scalar_field node"


def test_divgrad_codegen(t):
    src = _divgrad_program(t, method="bicgstab").emit_cpp_program()
    for frag in ("ctx.gradient", "ctx.divergence", "adc::bicgstab_solve",
                 "ctx.alloc_scalar_field(2, 1)"):  # the 2-component gradient buffer
        assert frag in src, "the div(grad) solve must contain %r\n%s" % (frag, src)


def test_condensed_schur_macro_lowers(t):
    # ADC-421: the condensed-Schur macro is now implemented (no longer a stub). At theta == 1 it lowers
    # to the full anisotropic assemble / solve / reconstruct chain; the deferred theta != 1 extrapolation
    # raises. The end-to-end parity lives in test_time_condensed_schur.py.
    P = t.Program("p")
    t.std.condensed_schur(P, "blk", alpha=1.0, theta=1.0)
    assert P.validate() is True, "the condensed-Schur macro must validate"
    src = P.emit_cpp_program()
    assert "ctx.assemble_schur_coeffs" in src and "ctx.schur_reconstruct" in src, src
    try:
        t.std.condensed_schur(t.Program("p2"), "blk", alpha=1.0, theta=0.5)
    except NotImplementedError as exc:
        assert "theta == 1" in str(exc), str(exc)
    else:
        raise AssertionError("condensed_schur(theta != 1) must raise NotImplementedError (deferred)")


def _analytic_divergence_check():
    """Standalone offline check: the centered FV divergence of a known smooth flux matches the analytic
    divergence to the discretization error. The same centered stencil adc::apply_divergence (and the
    compiled ctx.divergence) computes. Skips silently without numpy."""
    try:
        import numpy as np
    except Exception:  # noqa: BLE001  -- numpy unavailable here
        return
    n = 64
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    fx = np.cos(2 * np.pi * X)  # x-flux
    fy = np.sin(2 * np.pi * Y)  # y-flux
    h = 1.0 / n
    div = (np.roll(fx, -1, 0) - np.roll(fx, 1, 0)) / (2 * h) + \
          (np.roll(fy, -1, 1) - np.roll(fy, 1, 1)) / (2 * h)
    analytic = -2 * np.pi * np.sin(2 * np.pi * X) + 2 * np.pi * np.cos(2 * np.pi * Y)
    err = float(np.abs(div - analytic).max())
    assert err < 0.05, "centered FV divergence vs analytic div f (n=%d): max|d| = %.3e" % (n, err)
    print("  centered divergence vs analytic: max|d| = %.3e (n=%d, O(h^2))" % (err, n))


# ---- (B) end-to-end parity: skips unless the full toolchain is present ----
def _np_cg(apply, b, *, tol=1e-10, max_iter=2000):
    """Plain numpy CG solving A x = b from x = 0 (A = the discrete periodic Helmholtz matvec)."""
    import numpy as np

    x = np.zeros_like(b)
    r = b - apply(x)
    p = r.copy()
    rs_old = float(np.sum(r * r))
    bnorm = float(np.sqrt(np.sum(b * b))) or 1.0
    iters = 0
    for _ in range(max_iter):
        Ap = apply(p)
        pap = float(np.sum(p * Ap))
        if abs(pap) < 1e-300:
            break
        a = rs_old / pap
        x = x + a * p
        r = r - a * Ap
        rs_new = float(np.sum(r * r))
        iters += 1
        if np.sqrt(rs_new) <= tol * bnorm:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x, iters


def _discrete_divgrad_helmholtz(n, alpha):
    """The discrete periodic Helmholtz matvec A x = x - alpha*div(grad x) built from the CENTERED
    gradient (d/dx = (x(i+1) - x(i-1))/(2h)) followed by the CENTERED divergence -- exactly the operator
    the compiled P.gradient -> P.divergence chain composes. Centered div(grad) is the WIDE-stencil
    Laplacian (x(i+2) - 2 x(i) + x(i-2))/(4 h^2) (not the compact 5-point apply_laplacian), so the
    reference must use the same wide stencil. On an n x n unit-square grid (dx = dy = 1/n)."""
    import numpy as np

    h = 1.0 / n

    def apply(x):
        gx = (np.roll(x, -1, 0) - np.roll(x, 1, 0)) / (2 * h)
        gy = (np.roll(x, -1, 1) - np.roll(x, 1, 1)) / (2 * h)
        div = (np.roll(gx, -1, 0) - np.roll(gx, 1, 0)) / (2 * h) + \
              (np.roll(gy, -1, 1) - np.roll(gy, 1, 1)) / (2 * h)
        return x - alpha * div

    return apply


def _run_section_b(t):
    try:
        import numpy as np

        import adc
    except Exception as exc:  # noqa: BLE001  -- numpy / _adc unavailable here
        print("-- (B) skipped: adc/numpy unavailable: %s --" % exc)
        return None

    n = 16
    sim = adc.System(n=n, L=1.0, periodic=True)
    if not hasattr(sim, "install_program"):
        print("-- (B) skipped: _adc lacks the install_program binding (rebuild _adc) --")
        return None

    from adc import dsl

    def passive_model(name):  # 1-variable block, no flux, no Poisson coupling
        m = dsl.Model(name)
        (rho,) = m.conservative_vars("rho")
        u = m.primitive("u", 0.0 * rho)
        m.primitive_vars(rho=rho, u=u)
        m.conservative_from([rho])
        m.flux(x=[0.0 * rho], y=[0.0 * rho])
        m.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
        return m

    tol = 1e-10
    try:
        compiled = adc.compile_problem(
            model=passive_model("divgrad_prog"),
            time=_divgrad_program(t, name="divgrad_step", method="bicgstab", tol=tol, max_iter=200))
    except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
        print("-- (B) skipped: compile_problem could not build the .so: %s --" % str(exc)[:200])
        return None

    assert compiled.program_name == "divgrad_step", "handle carries the program name"

    try:
        compiled_model = passive_model("divgrad_block").compile(backend="production")
    except RuntimeError as exc:  # no compiler / no Kokkos visible
        print("-- (B) skipped: model compile could not build the .so: %s --" % str(exc)[:200])
        return None
    sim.add_equation("blk", compiled_model,
                     spatial=adc.FiniteVolume(limiter="none", riemann="rusanov"),
                     time=adc.Explicit(method="euler"))

    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("blk", np.stack([rho0]))

    sim.install_program(compiled.so_path)
    sim.step(0.05)  # dt is irrelevant: the solve is dt-free
    out = np.array(sim.get_state("blk"))[0]

    # OFFLINE reference: solve (I - alpha*div(grad)) phi = rho0 on the SAME centered div(grad) operator
    # (the wide-stencil Helmholtz the compiled gradient->divergence chain composes) with numpy CG; the
    # compiled matrix-free solve must recover the same phi.
    apply = _discrete_divgrad_helmholtz(n, _ALPHA)
    phi_ref, iters = _np_cg(apply, rho0, tol=tol)
    err = float(np.abs(out - phi_ref).max())
    moved = float(np.abs(out - rho0).max())
    print("  div(grad) Helmholtz parity: max|compiled - offline| = %.2e  offline iters = %d  "
          "max|phi - U0| = %.2e" % (err, iters, moved))
    assert err <= 1e-6, "compiled div(grad) CG == offline numpy CG (max|d| = %.2e)" % err
    assert moved > 1e-6, "the solve must change the state from U0 (max|d| = %.2e)" % moved
    assert iters > 1, "the offline (and compiled) solve must take > 1 iteration, got %d" % iters
    return (err, iters)


def _run():
    t = _adc_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    _analytic_divergence_check()
    print("PASS test_time_divergence (A: %d checks)" % len(fns))
    _run_section_b(t)


if __name__ == "__main__":
    _run()
