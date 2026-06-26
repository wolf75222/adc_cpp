#!/usr/bin/env python3
"""pops.time IMPLICIT-FLUX BDF via matrix-free Newton-Krylov (epic ADC-399 / ADC-431).

ADC-431 completes ``std.bdf``: today it lowers only a cell-local linear source (a block-diagonal
``solve_local_linear``); the IMPLICIT-FLUX case (the globally coupled ``-div F`` stencil) is the new
general path. It is a pure-macro composition of EXISTING primitives -- no new C++ runtime stepper:

    BDF1:  F(U) = U - U^n            - dt*rhs(U) = 0
    BDF2:  F(U) = U - (4/3)U^n + (1/3)U^{n-1} - (2/3)*dt*rhs(U) = 0   (U^{n-1} via P.history)

solved by Newton's method (a fixed ``static_range`` unroll). Each Newton step solves ``J dU = -F`` with
GMRES (J = I - c*dt*d(rhs)/dU is nonsymmetric), J applied MATRIX-FREE by a finite-difference
Jacobian-vector product (``P.rhs_jacvec``, the codegen enabler: it calls ``rhs`` INSIDE the
matrix-free apply sub-block, perturbing the frozen Newton iterate).

(A) Pure Python, always runs: the implicit-flux BDF1/BDF2 IR builds, validates and lowers to C++ with a
    GMRES solve per Newton iteration, the rhs-inside-apply scratch (jac_uk / jac_r0 / jac_cdt), and the
    ``<block>.bdf_residual`` diagnostic; the cell-local linear-source FAST PATH IR is unchanged (no
    jacvec / gmres); the BDF2 history ops appear; the argument guards fire.

(B) End-to-end parity (skips unless the full toolchain is present): an isothermal-Euler block (a genuine
    NONLINEAR flux ``f = (mx, mx^2/rho + cs2 rho, mx my/rho)``, so the Jacobian is genuinely
    state-dependent and Newton takes > 1 iteration). The Program steps one implicit BDF step through the
    REAL compiled engine; the result is
    compared to an OFFLINE numpy Newton-Krylov on the IDENTICAL residual, using the engine's own
    ``eval_rhs`` as the black-box discrete ``rhs`` oracle (so the offline residual is byte-identical to
    the compiled one). Asserts: the compiled step satisfies ``||F(U^{n+1})|| ~ 0``; the compiled
    ``U^{n+1}`` matches the offline root ~1e-8; the offline Newton takes > 1 iteration; the inner GMRES
    runs (> 1 iteration). BDF2 is checked the same way (a cold-start history). Self-skips (exit 0)
    without numpy / _pops / install_program / a compiler / a visible Kokkos -- never fakes the engine.
"""
import sys


def _pops_time():
    try:
        import pops.time as t
    except Exception as exc:  # pops not importable here -> skip, never fake
        print("skip test_time_bdf (pops.time unavailable: %s)" % exc)
        sys.exit(0)
    return t


# ---- (A) codegen + IR validation: pure Python, always runs ----
def test_implicit_flux_bdf1_codegen(t):
    P = t.Program("bdf1")
    t.std.bdf(P, "blk", 1, sources=["default"], newton_max=3, krylov_max=50)
    assert P.validate() is True, "the implicit-flux BDF1 Program must validate"
    assert P._ir_hash(), "the IR must serialize to a stable hash"
    src = P.emit_cpp_program(model=None)  # no model needed: the jacvec reuses the block rhs closure
    assert src.count("pops::gmres_solve") == 3, ("one GMRES per Newton iteration\n%s" % src)
    # the rhs-inside-apply enabler: rhs_into called inside the matrix-free apply lambda + the scratch
    assert "ctx.rhs_into(0" in src, "rhs(U^k + eps*v) inside the matrix-free apply\n%s" % src
    assert "jac_uk" in src and "jac_r0" in src and "jac_cdt" in src, "the FD-Jacobian scratch\n%s" % src
    assert 'ctx.record_scalar("blk.bdf_residual"' in src, "the residual diagnostic\n%s" % src


def test_implicit_flux_bdf2_codegen(t):
    P = t.Program("bdf2")
    t.std.bdf(P, "blk", 2, sources=["default"], newton_max=2, krylov_max=40)
    assert P.validate() is True
    src = P.emit_cpp_program(model=None)
    assert src.count("pops::gmres_solve") == 2
    assert 'ctx.register_history("blk.U"' in src and 'ctx.store_history("blk.U"' in src, \
        "BDF2 reads U^{n-1} from the history ring\n%s" % src
    # the BDF2 implicit coefficient is c*dt = (2/3) dt
    assert any("0.6666" in ln for ln in src.splitlines() if "jac_cdt" in ln), \
        "the BDF2 Jacobian coefficient c = 2/3\n%s" % src


def test_flux_only_uses_source_free_rhs(t):
    # sources=[] (flux only) -> the apply calls neg_div_flux_default_into (no default source).
    P = t.Program("bdf1_fo")
    t.std.bdf(P, "blk", 1, sources=[], newton_max=1, krylov_max=10)
    src = P.emit_cpp_program(model=None)
    assert "ctx.neg_div_flux_default_into(0" in src, "flux-only rhs inside the apply\n%s" % src
    assert "ctx.rhs_into(0" not in src, "no default source in the flux-only jacvec\n%s" % src


def test_multicomponent_operator(t):
    # ncomp=2 (a multi-component block) -> a state-domain operator + 2-component jacvec scratch.
    P = t.Program("bdf1_mc")
    t.std.bdf(P, "blk", 1, sources=["default"], ncomp=2, newton_max=1, krylov_max=10)
    assert P.validate() is True
    src = P.emit_cpp_program(model=None)
    assert "ctx.alloc_scalar_field(2, ctx.state(0).n_grow())" in src, "2-component jacvec scratch\n%s" % src
    assert "ctx.alloc_scalar_field(2, 1)" in src, "2-component apply buffers\n%s" % src


def test_cell_local_fast_path_unchanged(t):
    # The cell-local linear-source path stays the block-diagonal solve_local_linear: NO Newton/Krylov.
    for order in (1, 2):
        P = t.Program("bdfL%d" % order)
        t.std.bdf(P, "blk", order, linear_source="lorentz")
        assert P.validate() is True
        ops = {v.op for v in P._values}
        assert "solve_local_linear" in ops, "the cell-local fast path uses solve_local_linear"
        assert "rhs_jacvec" not in ops and "matrix_free_operator" not in ops and \
            "solve_linear" not in ops, "the cell-local fast path must NOT Newton/Krylov (order %d)" % order


def test_argument_guards(t):
    for bad in (0, 3, True, 1.5, "1"):
        try:
            t.std.bdf(t.Program("x"), "b", bad)
        except ValueError:
            pass
        else:
            raise AssertionError("bdf order %r must raise" % (bad,))
    # flux=False with no linear_source has no implicit term -> rejected (not a silent no-op).
    try:
        t.std.bdf(t.Program("x"), "b", 1, flux=False)
    except ValueError as exc:
        assert "linear_source" in str(exc), str(exc)
    else:
        raise AssertionError("bdf(flux=False) with no linear_source must raise")
    # ncomp must be a positive int.
    for bad in (0, -1, True, 1.5):
        try:
            t.std.bdf(t.Program("x"), "b", 1, ncomp=bad)
        except ValueError as exc:
            assert "ncomp" in str(exc), str(exc)
        else:
            raise AssertionError("bdf ncomp=%r must raise" % (bad,))
    # rhs_jacvec is only recordable inside a set_apply body.
    P = t.Program("x")
    in_sf = P.scalar_field("in")
    out_sf = P.scalar_field("out")
    U = P.state("b")
    R = P.rhs(state=U, flux=True, sources=["default"])
    try:
        P.rhs_jacvec(out_sf, in_sf, iterate=U, r0=R, c_dt=P.dt)  # not inside set_apply
    except ValueError as exc:
        assert "set_apply" in str(exc) or "apply" in str(exc), str(exc)
    else:
        raise AssertionError("rhs_jacvec outside a set_apply body must raise")


# ---- (B) end-to-end parity: skips unless the full toolchain is present ----
_NCOMP = 3  # isothermal Euler (rho, mx, my): a genuine nonlinear flux f = (mx, mx^2/rho + cs2 rho, ...)


def _nonlinear_flux_model():
    """An isothermal Euler block: a genuine NONLINEAR flux (f_x = (mx, mx^2/rho + cs2 rho, mx my/rho)),
    so the implicit Jacobian I - c*dt*d(rhs)/dU is genuinely state-dependent and Newton takes more than
    one iteration. BackgroundDensity(n0=0) keeps solve_fields well-defined but INERT (no Poisson
    feedback into the flux), so the frozen-Poisson implicit step is exact -- the same trick the other
    compiled-program tests use. A real composed-brick model, never a fake engine."""
    import pops

    return pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                     transport=pops.IsothermalFlux(),
                     source=pops.NoSource(),
                     elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0))


def _bdf_program(t, order, *, name, newton_max, krylov_max, tol):
    P = t.Program(name)
    t.std.bdf(P, "blk", order, sources=["default"], ncomp=_NCOMP, newton_max=newton_max,
              krylov_tol=tol, krylov_max=krylov_max)
    return P


def _offline_newton_krylov(rhs_of, u_lag, dt, *, c, un_terms, tol, newton_max, eps=1e-7):
    """Offline Newton-Krylov on the IDENTICAL BDF residual, using @p rhs_of as the black-box discrete rhs
    (the engine's own eval_rhs). F(U) = un_terms(U^n[, U^{n-1}]) ... = U - un - c*dt*rhs(U). Each Newton
    step solves J dU = -F with a hand-rolled GMRES on the SAME FD Jacobian-vector product the compiled
    macro uses: J v = v - c*dt*(rhs(U + h*v) - rhs(U))/h. Returns (U, newton_iters, total_gmres_iters)."""
    import numpy as np

    U = u_lag[0].copy()  # warm start from U^n
    rU = rhs_of(U)
    newton_iters = 0
    total_gmres = 0
    for _ in range(newton_max):
        F = U - un_terms - c * dt * rU
        nF = float(np.sqrt(np.sum(F * F)))
        if nF <= tol:
            break

        def jacvec(v, _U=U, _rU=rU):
            nv = float(np.sqrt(np.sum(v * v)))
            nuk = float(np.sqrt(np.sum(_U * _U)))
            h = eps * (1.0 + nuk) / nv if nv > 0 else eps
            return v - (c * dt / h) * (rhs_of(_U + h * v) - _rU)

        dU, git = _gmres(jacvec, -F, tol=tol, max_iter=200)
        total_gmres += git
        U = U + dU
        rU = rhs_of(U)
        newton_iters += 1
    return U, newton_iters, total_gmres


def _gmres(matvec, b, *, tol, max_iter, restart=30):
    """A minimal restarted GMRES(m) for a black-box matvec (numpy). Returns (x, iters)."""
    import numpy as np

    x = np.zeros_like(b)
    bnorm = float(np.sqrt(np.sum(b * b))) or 1.0
    total = 0
    for _ in range((max_iter + restart - 1) // restart):
        r = b - matvec(x)
        beta = float(np.sqrt(np.sum(r * r)))
        if beta <= tol * bnorm:
            break
        m = restart
        Q = [r / beta]
        H = np.zeros((m + 1, m))
        g = np.zeros(m + 1)
        g[0] = beta
        k = 0
        cs = np.zeros(m)
        sn = np.zeros(m)
        for k in range(m):
            w = matvec(Q[k])
            for i in range(k + 1):
                H[i, k] = float(np.sum(w * Q[i]))
                w = w - H[i, k] * Q[i]
            H[k + 1, k] = float(np.sqrt(np.sum(w * w)))
            if H[k + 1, k] > 1e-300:
                Q.append(w / H[k + 1, k])
            else:
                Q.append(np.zeros_like(w))
            for i in range(k):  # apply previous Givens rotations
                tmp = cs[i] * H[i, k] + sn[i] * H[i + 1, k]
                H[i + 1, k] = -sn[i] * H[i, k] + cs[i] * H[i + 1, k]
                H[i, k] = tmp
            denom = float(np.hypot(H[k, k], H[k + 1, k]))
            if denom == 0.0:
                cs[k], sn[k] = 1.0, 0.0
            else:
                cs[k], sn[k] = H[k, k] / denom, H[k + 1, k] / denom
            H[k, k] = cs[k] * H[k, k] + sn[k] * H[k + 1, k]
            H[k + 1, k] = 0.0
            g[k + 1] = -sn[k] * g[k]
            g[k] = cs[k] * g[k]
            total += 1
            if abs(g[k + 1]) <= tol * bnorm:
                k += 1
                break
        else:
            k = m
        if k > 0:  # back-substitution over the k x k upper-triangular H
            y = np.zeros(k)
            for i in range(k - 1, -1, -1):
                y[i] = (g[i] - float(np.dot(H[i, i + 1:k], y[i + 1:k]))) / H[i, i]
            for i in range(k):
                x = x + y[i] * Q[i]
    return x, total


def _engine_rhs_oracle(sim, name):
    """Return rhs_of(U) -> the engine's discrete rhs at state U (set_state; solve_fields; eval_rhs). U is
    (ncomp, n, n); eval_rhs returns the same shape. This is the BLACK-BOX oracle the offline Newton-Krylov
    perturbs, so the offline residual is byte-identical to the compiled rhs (the SAME flux stencil). The
    solve_fields mirrors the compiled program's per-step elliptic solve (inert under BackgroundDensity)."""
    import numpy as np

    def rhs_of(U):
        sim.set_state(name, np.ascontiguousarray(U))
        sim.solve_fields()
        return np.array(sim.eval_rhs(name))

    return rhs_of


def _run_section_b(t):
    try:
        import numpy as np

        import pops
    except Exception as exc:  # noqa: BLE001
        print("-- (B) skipped: pops/numpy unavailable: %s --" % exc)
        return None

    probe = pops.System(n=8, L=1.0, periodic=True)
    if not hasattr(probe, "install_program") or not hasattr(probe, "eval_rhs"):
        print("-- (B) skipped: _pops lacks install_program / eval_rhs (rebuild _pops) --")
        return None

    n = 16
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)  # smooth isothermal-Euler state
    init = np.stack([rho, 0.4 * rho, -0.2 * rho])  # (3, n, n): rho, mx, my -> a clear nonlinear flux
    dt = 0.02
    tol = 1e-10
    newton_max = 20

    def _build(order):
        try:
            compiled = pops.compile_problem(
                model=_nonlinear_flux_model(),
                time=_bdf_program(t, order, name="bdf_iso_step%d" % order, newton_max=newton_max,
                                  krylov_max=200, tol=tol))
        except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
            print("-- (B) skipped: could not build the .so: %s --" % str(exc)[:200])
            return None
        return compiled

    def _make_sim():
        sim = pops.System(n=n, L=1.0, periodic=True)
        sim.add_equation("blk", _nonlinear_flux_model(),
                         spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                         time=pops.Explicit(method="euler"))
        sim.set_poisson("charge_density", "geometric_mg")
        sim.set_state("blk", init)
        return sim

    def _step_one(order, compiled):
        sim = _make_sim()
        sim.install_program(compiled.so_path)
        sim.step(dt)
        out = np.array(sim.get_state("blk"))
        try:  # the recorded ||F|| diagnostic (best-effort: optional if the binding is older)
            diag = sim.program_diagnostic("blk.bdf_residual")
            print("    recorded blk.bdf_residual = %.2e" % diag)
        except Exception:  # noqa: BLE001
            pass
        # A fresh sim for the offline oracle (set_state/eval_rhs mutate it), kept inert under solve_fields.
        return _make_sim(), out

    # ---- BDF1 ----
    compiled1 = _build(1)
    if compiled1 is None:
        return None
    sim, out1 = _step_one(1, compiled1)
    oracle = _engine_rhs_oracle(sim, "blk")  # a fresh inert sim; un_terms = U^n for BDF1.
    ref1, ni1, gi1 = _offline_newton_krylov(oracle, [init], dt, c=1.0, un_terms=init, tol=tol,
                                            newton_max=newton_max)
    err1 = float(np.abs(out1 - ref1).max())
    # The compiled step must satisfy the residual F(U^{n+1}) = U^{n+1} - U^n - dt*rhs(U^{n+1}) ~ 0.
    res1 = float(np.abs(out1 - init - dt * oracle(out1)).max())
    moved1 = float(np.abs(out1 - init).max())
    print("  BDF1 isothermal: max|compiled - offline| = %.2e  residual||inf = %.2e  offline newton = %d  "
          "gmres = %d  max|U1 - U0| = %.2e" % (err1, res1, ni1, gi1, moved1))
    assert err1 <= 1e-8, "compiled BDF1 == offline Newton-Krylov root (max|d| = %.2e)" % err1
    assert res1 <= 1e-7, "the compiled BDF1 step satisfies F(U^{n+1}) ~ 0 (||F|| = %.2e)" % res1
    assert moved1 > 1e-6, "the implicit step must change the state (max|d| = %.2e)" % moved1
    assert ni1 > 1, "the nonlinear-flux Newton must take > 1 iteration (got %d)" % ni1
    assert gi1 > 1, "the inner GMRES must run > 1 iteration (got %d)" % gi1

    # ---- BDF2 (cold start: U^{n-1} == U^n on the first step, so un_terms = (4/3 - 1/3) U^n = U^n) ----
    compiled2 = _build(2)
    if compiled2 is None:
        return (err1,)
    sim2, out2 = _step_one(2, compiled2)
    oracle2 = _engine_rhs_oracle(sim2, "blk")
    un_terms2 = (4.0 / 3.0 - 1.0 / 3.0) * init  # cold start U^{n-1} = U^n
    ref2, ni2, gi2 = _offline_newton_krylov(oracle2, [init], dt, c=2.0 / 3.0, un_terms=un_terms2,
                                            tol=tol, newton_max=newton_max)
    err2 = float(np.abs(out2 - ref2).max())
    res2 = float(np.abs(out2 - un_terms2 - (2.0 / 3.0) * dt * oracle2(out2)).max())
    print("  BDF2 isothermal (cold start): max|compiled - offline| = %.2e  residual||inf = %.2e  "
          "offline newton = %d  gmres = %d" % (err2, res2, ni2, gi2))
    assert err2 <= 1e-8, "compiled BDF2 == offline Newton-Krylov root (max|d| = %.2e)" % err2
    assert res2 <= 1e-7, "the compiled BDF2 step satisfies its residual ~ 0 (||F|| = %.2e)" % res2
    assert ni2 > 1, "the nonlinear-flux BDF2 Newton must take > 1 iteration (got %d)" % ni2
    return (err1, err2)


def _run():
    t = _pops_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    print("PASS test_time_bdf (A: %d checks)" % len(fns))
    _run_section_b(t)


if __name__ == "__main__":
    _run()
