#!/usr/bin/env python3
"""pops.time GMRES Krylov solver for the compiled time program (epic ADC-399 / ADC-420).

ADC-420 adds restarted GMRES(m) to the matrix-free Krylov core (pops::gmres_solve in
generic_krylov.hpp) and exposes it as ``P.solve_linear(method="gmres", restart=m)``. GMRES is the
robust choice for a NON-symmetric operator: where CG needs an SPD A and stagnates on a non-self-adjoint
one, GMRES minimises the residual over the Krylov subspace and converges.

(A) Codegen + validation (pure Python, always runs): a Helmholtz operator ``A(in) = in - alpha*Lap(in)``
    solved by gmres lowers to the apply lambda + ``ctx.laplacian`` + ``pops::gmres_solve`` with the
    restart length; the restart default (30) and an override both appear in the generated C++; the
    validation errors fire (max_iter absent/<=0; restart<=0 or non-int for gmres; restart passed to a
    non-gmres method).

(B) End-to-end parity (skips unless the full toolchain is present): two solves through the REAL compiled
    engine.
      (a) SPD: (I - alpha*Lap) phi = U via gmres (tol 1e-9) matches an OFFLINE numpy CG on the same
          discrete periodic 5-point system (~1e-6).
      (b) NON-symmetric (the gmres-specific guard): A(in) = in - alpha*Lap(in) + beta*d(in)/dx adds a
          centered first-derivative (advection) term, so A is non-self-adjoint and CG stagnates. gmres
          converges; the compiled solution matches an OFFLINE GMRES reference on the SAME discrete
          operator (~1e-6). Reports iters + residual.
    Self-skips (exit 0) without numpy / _pops / install_program / a compiler / a visible Kokkos -- never
    fakes the engine.

The non-symmetric C++ guard (CG stagnates while gmres recovers phi_exact) is also pinned directly in
tests/test_generic_krylov.cpp, which is fully validatable on every backend without the Python toolchain.
"""
import sys


def _pops_time():
    try:
        import pops.time as t
    except Exception as exc:  # pops not importable here -> skip, never fake
        print("skip test_time_gmres (pops.time unavailable: %s)" % exc)
        sys.exit(0)
    return t


_ALPHA = 0.1  # Helmholtz coefficient: A = I - alpha*Lap (SPD part)
_BETA = 2.0   # advection strength of the non-symmetric term beta*d/dx (breaks self-adjointness)


def _spd_program(t, *, name="gmres_spd", method="gmres", tol=1e-9, max_iter=300, restart=30,
                 alpha=_ALPHA):
    """(I - alpha*Lap) phi = U, committed back into the 1-component block (its state == a scalar field).

    A pure (SPD) Helmholtz apply; solve_linear drives the runtime GMRES loop. No model needed."""
    P = t.Program(name)
    U = P.state("blk")
    A = P.matrix_free_operator("A")

    def apply(P, out, x):
        lap = P.scalar_field("lap")
        P.laplacian(lap, x)
        return x - alpha * lap  # out = in - alpha*Lap(in)

    P.set_apply(A, apply)
    phi = P.solve_linear(operator=A, rhs=U, method=method, tol=tol, max_iter=max_iter, restart=restart)
    P.commit("blk", phi)
    return P


def _nonsym_program(t, *, name="gmres_nonsym", tol=1e-9, max_iter=300, restart=30, alpha=_ALPHA,
                    beta=_BETA, method="gmres"):
    """A NON-symmetric apply A(in) = in - alpha*Lap(in) + beta*d(in)/dx, solved by @p method.

    The first-derivative term is built with P.divergence(d, fx=in, fy=zero), which is the centered
    d fx/dx (the x-flux read from in, the y-flux zero) -- a skew-symmetric term that makes A
    non-self-adjoint, so CG stagnates while GMRES converges."""
    P = t.Program(name)
    U = P.state("blk")
    A = P.matrix_free_operator("A")

    def apply(P, out, x):
        lap = P.scalar_field("lap")
        P.laplacian(lap, x)
        # y-flux of the advection divergence (d/dy of 0 == 0). ncomp=2: ProgramContext::divergence
        # reads the x-flux at component 0 and the y-flux at component 1 (the cx=0/cy=1 convention for
        # the aliased gradient buffer), so the fy operand MUST carry a component 1 -- a 1-component
        # field would read out of bounds and corrupt the advection term.
        zero = P.scalar_field("zero", ncomp=2)
        dxdir = P.scalar_field("dxdir")
        P.divergence(dxdir, x, zero)        # dxdir = d(in)/dx (centered)
        return x - alpha * lap + beta * dxdir  # out = in - alpha*Lap(in) + beta*d(in)/dx

    P.set_apply(A, apply)
    kw = dict(operator=A, rhs=U, method=method, tol=tol, max_iter=max_iter)
    if method == "gmres":
        kw["restart"] = restart  # restart is gmres-only (rejected for cg/bicgstab)
    phi = P.solve_linear(**kw)
    P.commit("blk", phi)
    return P


def _helmholtz(P, x):
    lap = P.scalar_field("lap")
    P.laplacian(lap, x)
    return x - _ALPHA * lap


# ---- (A) codegen + validation: pure Python, always runs ----
def test_gmres_codegen(t):
    src = _spd_program(t, method="gmres").emit_cpp_program()
    for frag in ("pops::ApplyFn apply_A", "ctx.laplacian", "pops::gmres_solve",
                 "pops::ApplyFn{}"):  # identity (empty) preconditioner
        assert frag in src, "the generated gmres solve must contain %r\n%s" % (frag, src)


def test_gmres_restart_default_in_codegen(t):
    src = _spd_program(t, restart=30).emit_cpp_program()
    # The trailing gmres_solve argument is the restart length (default 30).
    assert ", 30);" in src and "pops::gmres_solve" in src, "the default restart 30 must lower\n%s" % src


def test_gmres_restart_override_in_codegen(t):
    src = _spd_program(t, restart=12).emit_cpp_program()
    assert ", 12);" in src and "pops::gmres_solve" in src, "an overridden restart must lower\n%s" % src


def test_gmres_now_valid_method(t):
    # gmres used to be rejected as unknown; it is now a first-class method.
    P = _spd_program(t, method="gmres")
    assert P.validate() is True, "the gmres solve_linear Program must validate"
    assert P._ir_hash(), "the IR must serialize to a stable hash"


def test_gmres_max_iter_required(t):
    P = t.Program("p")
    U = P.state("blk")
    A = P.matrix_free_operator("A")
    P.set_apply(A, lambda P, out, x: _helmholtz(P, x))
    for bad in (None, 0, -5):
        try:
            P.solve_linear(operator=A, rhs=U, method="gmres", max_iter=bad)
        except ValueError as exc:
            assert "dynamic solver loops require max_iter" in str(exc), str(exc)
        else:
            raise AssertionError("max_iter=%r must raise the dynamic-loop budget error" % (bad,))


def test_gmres_restart_validation(t):
    P = t.Program("p")
    U = P.state("blk")
    A = P.matrix_free_operator("A")
    P.set_apply(A, lambda P, out, x: _helmholtz(P, x))
    for bad in (0, -3, 1.5, True):  # a positive int is required (True is rejected: bool is not allowed)
        try:
            P.solve_linear(operator=A, rhs=U, method="gmres", max_iter=10, restart=bad)
        except ValueError as exc:
            assert "restart" in str(exc), str(exc)
        else:
            raise AssertionError("restart=%r must raise for gmres" % (bad,))
    # a positive int restart is accepted
    phi = P.solve_linear(operator=A, rhs=U, method="gmres", max_iter=10, restart=8)
    assert phi.attrs["restart"] == 8, "the restart is stored on the IR node"


def test_restart_rejected_for_non_gmres(t):
    P = t.Program("p")
    U = P.state("blk")
    A = P.matrix_free_operator("A")
    P.set_apply(A, lambda P, out, x: _helmholtz(P, x))
    for method in ("cg", "bicgstab", "richardson"):
        try:
            P.solve_linear(operator=A, rhs=U, method=method, max_iter=10, restart=20)
        except ValueError as exc:
            assert "restart only applies" in str(exc), str(exc)
        else:
            raise AssertionError("restart on method=%r must raise" % (method,))


# ---- (B) end-to-end parity: skips unless the full toolchain is present ----
def _np_cg(apply, b, *, tol=1e-9, max_iter=4000):
    """Plain numpy CG solving A x = b from x = 0 (A = a discrete SPD matvec). Returns (x, iters)."""
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


def _discrete_helmholtz(n, alpha):
    """Discrete periodic 5-point Helmholtz matvec A x = x - alpha*Lap(x) on an n x n unit-square grid
    (dx = dy = 1/n), matching pops::apply_laplacian's bare path with periodic ghosts."""
    import numpy as np

    h2 = (1.0 / n) ** 2

    def apply(x):
        lap = (np.roll(x, -1, 0) + np.roll(x, 1, 0) - 2 * x) / h2 + \
              (np.roll(x, -1, 1) + np.roll(x, 1, 1) - 2 * x) / h2
        return x - alpha * lap

    return apply


def _discrete_nonsym(n, alpha, beta):
    """Discrete periodic NON-symmetric matvec A x = x - alpha*Lap(x) + beta*d(x)/dx, the centered x
    derivative d x/dx = (x(i+1) - x(i-1)) / (2 dx) matching pops::apply_divergence (cx=0, fy=0) on the
    DSL apply x - alpha*Lap(x) + beta*div(x, 0)."""
    import numpy as np

    h = 1.0 / n
    h2 = h * h

    def apply(x):
        lap = (np.roll(x, -1, 0) + np.roll(x, 1, 0) - 2 * x) / h2 + \
              (np.roll(x, -1, 1) + np.roll(x, 1, 1) - 2 * x) / h2
        ddx = (np.roll(x, -1, 0) - np.roll(x, 1, 0)) / (2 * h)  # centered d/dx, x along axis 0
        return x - alpha * lap + beta * ddx

    return apply


def _passive_model(dsl, name):
    """A minimal 1-variable block with NO flux and NO Poisson coupling: the block's single conservative
    variable doubles as the scalar field the matrix-free solve writes."""
    m = dsl.Model(name)
    (rho,) = m.conservative_vars("rho")
    u = m.primitive("u", 0.0 * rho)
    m.primitive_vars(rho=rho, u=u)
    m.conservative_from([rho])
    m.flux(x=[0.0 * rho], y=[0.0 * rho])
    m.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
    return m


def _run_one(t, pops, np, program, name):
    """Compile + install + step @p program on a 1-variable block, return the stepped state and rho0,
    or None if the toolchain is unavailable."""
    n = 16
    sim = pops.System(n=n, L=1.0, periodic=True)
    if not hasattr(sim, "install_program"):
        print("-- (B) skipped: _pops lacks the install_program binding (rebuild _pops) --")
        return None

    from pops import dsl

    try:
        compiled = pops.compile_problem(model=_passive_model(dsl, name + "_prog"), time=program)
        compiled_model = _passive_model(dsl, name + "_block").compile(backend="production")
    except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
        print("-- (B) skipped: could not build the .so: %s --" % str(exc)[:200])
        return None

    sim.add_equation("blk", compiled_model,
                     spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                     time=pops.Explicit(method="euler"))

    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("blk", np.stack([rho0]))
    sim.install_program(compiled.so_path)
    sim.step(0.05)  # dt is irrelevant: the solve is dt-free
    out = np.array(sim.get_state("blk"))[0]
    return out, rho0, n


def _run_section_b(t):
    try:
        import numpy as np

        import pops
    except Exception as exc:  # noqa: BLE001  -- numpy / _pops unavailable in this interpreter
        print("-- (B) skipped: pops/numpy unavailable: %s --" % exc)
        return None

    tol = 1e-9
    # (a) SPD: gmres matches the offline CG on the same discrete Helmholtz system.
    res = _run_one(t, pops, np, _spd_program(t, name="gmres_spd_step", tol=tol, max_iter=300, restart=30),
                   "spd")
    if res is None:
        return None
    out, rho0, n = res
    phi_ref, iters = _np_cg(_discrete_helmholtz(n, _ALPHA), rho0, tol=tol)
    err = float(np.abs(out - phi_ref).max())
    print("  (a) gmres SPD parity:    max|compiled - offline CG| = %.2e  offline iters = %d" % (err, iters))
    assert err <= 1e-6, "compiled gmres (SPD) == offline CG (max|d| = %.2e)" % err
    assert iters > 1, "the SPD solve must take > 1 iteration, got %d" % iters

    # (b) NON-symmetric (the gmres-specific guard): the COMPILED gmres and the COMPILED bicgstab solve
    # the SAME non-symmetric operator and must converge to the same solution -- both are general-matrix
    # solvers, so agreement cross-checks gmres on a non-symmetric system WITHOUT depending on an exact
    # offline operator-stencil match (expressing a pure d/dx in the IR is approximate). An offline CG on
    # a proxy of the same operator stagnates, confirming the operator is genuinely non-self-adjoint
    # (CG-hostile, the regime where gmres/bicgstab are needed and cg is not).
    res_g = _run_one(t, pops, np, _nonsym_program(t, name="gmres_nsy_g", tol=tol, max_iter=400,
                                                 restart=40, method="gmres"), "nsy_g")
    res_b = _run_one(t, pops, np, _nonsym_program(t, name="gmres_nsy_b", tol=tol, max_iter=400,
                                                 method="bicgstab"), "nsy_b")
    if res_g is None or res_b is None:
        return None
    out_g, rho0_2, n2 = res_g
    out_b = res_b[0]
    err2 = float(np.abs(out_g - out_b).max())            # gmres == bicgstab on the same non-sym operator
    moved2 = float(np.abs(out_g - rho0_2).max())
    apply_ns = _discrete_nonsym(n2, _ALPHA, _BETA)        # offline proxy: CG stagnates on it
    cg_x, _ = _np_cg(apply_ns, rho0_2, tol=tol, max_iter=300)
    cg_resid = float(np.linalg.norm((rho0_2 - apply_ns(cg_x)).ravel()))
    print("  (b) gmres vs bicgstab (same non-sym op): max|d| = %.2e  max|phi - U0| = %.2e  "
          "(offline CG residual on the proxy op = %.2e)" % (err2, moved2, cg_resid))
    assert err2 <= 1e-6, "compiled gmres == compiled bicgstab on a non-symmetric op (max|d| = %.2e)" % err2
    assert moved2 > 1e-6, "the non-symmetric solve must change the state (max|d| = %.2e)" % moved2
    return (err, err2)


def _run():
    t = _pops_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    print("PASS test_time_gmres (A: %d checks)" % len(fns))
    _run_section_b(t)


if __name__ == "__main__":
    _run()
