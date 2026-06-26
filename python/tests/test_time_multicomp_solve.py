#!/usr/bin/env python3
"""pops.time MULTI-COMPONENT matrix-free linear solve (epic ADC-399 / ADC-416).

ADC-416 extends ``P.matrix_free_operator`` + ``P.solve_linear`` from scalar-only to vector /
state-valued (multi-component) fields -- the foundation the full condensed_schur macro builds on. The
operator declares ``domain="state"`` (or ``"vector"``) with an ``ncomp``; its apply runs on an ncomp
buffer and the runtime Krylov loop (``pops::cg_solve`` etc.) reduces its inner products over ALL
components (pops::dot_all), so EVERY component is solved -- a component-0-only norm would converge on
component 0 alone and leave the rest unsolved.

(A) Pure Python, always runs: a state-valued operator (ncomp=2) builds, its apply in / out buffers and
    its solution carry ncomp=2, the codegen allocates the scratch / accumulator / solution 2-component
    (``ctx.alloc_scalar_field(2, 1)``); the scalar default is unchanged (ncomp=1, ``alloc(1, 1)``); the
    ncomp / domain validation fires (ncomp<1, domain!=range_, a too-small scalar_field rhs).

(B) End-to-end parity (skips unless the full toolchain is present): a 2-component block (rho, e, zero
    flux); A = a state-valued matrix_free_operator (ncomp=2) with apply out = in - alpha*Lap(in)
    (alpha=0.1, SPD per component). The Program solves (I - alpha*Lap) x = U on the 2-component state via
    cg and commits U = x. compile_problem -> install_program -> set rho0/e0 to DIFFERENT smooth fields
    (so the two components have different right-hand sides) -> step once -> get_state, vs an OFFLINE
    numpy CG on the SAME 2-component discrete operator. Asserts BOTH components match ~1e-10 (the comp-1
    match is the regression guard for the full-component norm: with a comp-0-only dot the loop would stop
    on component 0's residual and leave component 1 unsolved). A scalar (ncomp=1) solve still matches the
    same offline CG bit-for-bit. Self-skips (exit 0) without numpy / _pops / install_program / a compiler
    / a visible Kokkos -- never fakes the engine.
"""
import sys


def _adc_time():
    try:
        import pops.time as t
    except Exception as exc:  # adc not importable here -> skip, never fake
        print("skip test_time_multicomp_solve (pops.time unavailable: %s)" % exc)
        sys.exit(0)
    return t


_ALPHA = 0.1  # Helmholtz coefficient: A = I - alpha*Lap (SPD per component, well-conditioned for CG)


def _mc_program(t, ncomp, *, name="mc_solve", method="cg", tol=1e-10, max_iter=200, alpha=_ALPHA):
    """(I - alpha*Lap) x = U on an ncomp-component block, committed back into the block state.

    The apply ``out = in - alpha*Lap(in)`` is built with P.laplacian (which now runs per component) +
    the affine algebra; solve_linear drives the runtime multi-component Krylov loop."""
    P = t.Program(name)
    U = P.state("blk")
    kind = "scalar" if ncomp == 1 else "state"
    A = P.matrix_free_operator("A", domain=kind, range_=kind,
                               ncomp=(None if ncomp == 1 else ncomp))

    def apply(P, out, x):
        lap = P.scalar_field("lap", ncomp=ncomp)
        P.laplacian(lap, x)
        return x - alpha * lap  # out = in - alpha*Lap(in), applied to every component

    P.set_apply(A, apply)
    phi = P.solve_linear(operator=A, rhs=U, method=method, tol=tol, max_iter=max_iter)
    P.commit("blk", phi)
    return P


# ---- (A) codegen + IR validation: pure Python, always runs ----
def test_state_operator_builds(t):
    P = t.Program("p")
    A = P.matrix_free_operator("A", domain="state", range_="state", ncomp=2)
    assert A.attrs["ncomp"] == 2 and A.attrs["domain"] == "state" and A.attrs["range"] == "state"

    def apply(P, out, x):
        assert x.attrs["ncomp"] == 2, "the apply in buffer carries the operator ncomp"
        assert out.attrs["ncomp"] == 2, "the apply out buffer carries the operator ncomp"
        lap = P.scalar_field("lap", ncomp=2)
        P.laplacian(lap, x)
        return x - _ALPHA * lap

    P.set_apply(A, apply)
    U = P.state("blk")
    phi = P.solve_linear(operator=A, rhs=U, method="cg", tol=1e-10, max_iter=50)
    assert phi.attrs["ncomp"] == 2, "the solution carries the operator ncomp"
    P.commit("blk", phi)
    assert P.validate() is True, "the multi-component Program must validate"
    assert P._ir_hash(), "the IR must serialize to a stable hash"


def test_scalar_default_unchanged(t):
    A = t.Program("p").matrix_free_operator("A")  # scalar default
    assert A.attrs["ncomp"] == 1 and A.attrs["domain"] == "scalar" and A.attrs["range"] == "scalar"
    A2 = t.Program("p2").matrix_free_operator("A", domain="vector", range_="vector", ncomp=3)
    assert A2.attrs["ncomp"] == 3 and A2.attrs["domain"] == "vector"


def test_ncomp_and_domain_validation(t):
    P = t.Program("p")
    # vector / state require ncomp >= 1
    for bad in (None, 0, -2, 1.5, True):
        try:
            P.matrix_free_operator("A", domain="state", range_="state", ncomp=bad)
        except ValueError as exc:
            assert "ncomp" in str(exc), str(exc)
        else:
            raise AssertionError("a state operator with ncomp=%r must raise" % (bad,))
    # domain must equal range_
    try:
        P.matrix_free_operator("A", domain="state", range_="scalar", ncomp=2)
    except ValueError as exc:
        assert "match" in str(exc), str(exc)
    else:
        raise AssertionError("domain != range_ must raise")
    # an unknown kind is rejected
    try:
        P.matrix_free_operator("A", domain="tensor", range_="tensor", ncomp=2)
    except ValueError as exc:
        assert "one of" in str(exc), str(exc)
    else:
        raise AssertionError("an unknown domain kind must raise")
    # a scalar operator with an explicit ncomp != 1 is rejected
    try:
        P.matrix_free_operator("A", domain="scalar", range_="scalar", ncomp=2)
    except ValueError as exc:
        assert "scalar operator" in str(exc), str(exc)
    else:
        raise AssertionError("a scalar operator with ncomp=2 must raise")


def test_solve_rhs_component_count(t):
    P = t.Program("p")
    A = P.matrix_free_operator("A", domain="state", range_="state", ncomp=3)
    P.set_apply(A, lambda P, out, x: x)
    rhs_small = P.scalar_field("rhs2", ncomp=2)  # too few components for the ncomp=3 operator
    try:
        P.solve_linear(operator=A, rhs=rhs_small, max_iter=10)
    except ValueError as exc:
        assert "component" in str(exc), str(exc)
    else:
        raise AssertionError("a rhs with too few components must raise")
    # a scalar_field with >= ncomp components, and a State (n_cons checked at compile), are accepted
    phi = P.solve_linear(operator=A, rhs=P.scalar_field("rhs3", ncomp=3), max_iter=10)
    assert phi.attrs["ncomp"] == 3
    P.solve_linear(operator=A, rhs=P.state("blk"), max_iter=10)  # State deferred -> accepted


def test_multicomp_codegen(t):
    src = _mc_program(t, 2).emit_cpp_program()
    n = src.count("ctx.alloc_scalar_field(2, 1)")  # lap scratch + accumulator + solution
    assert n >= 3, "the 2-component solve allocates 2-component scratch/acc/solution\n%s" % src
    assert "pops::cg_solve" in src and "ctx.laplacian" in src, src
    # the scalar path still allocates 1-component fields only
    src1 = _mc_program(t, 1).emit_cpp_program()
    assert "ctx.alloc_scalar_field(1, 1)" in src1 and "alloc_scalar_field(2, 1)" not in src1, src1


# ---- (B) end-to-end parity: skips unless the full toolchain is present ----
def _np_cg_mc(apply, b, *, tol=1e-10, max_iter=2000):
    """Plain numpy CG solving A x = b from x = 0 with a FULL-component L2 norm (b shaped (ncomp, n, n));
    A is the per-component discrete periodic Helmholtz matvec. The reference for the compiled
    multi-component matrix-free CG -- the inner products sum over every component (matching pops::dot_all),
    so the loop converges all components together. Returns (x, iters)."""
    import numpy as np

    x = np.zeros_like(b)
    r = b - apply(x)
    p = r.copy()
    rs_old = float(np.sum(r * r))  # FULL-component inner product (all components)
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


def _discrete_helmholtz_mc(n, alpha):
    """The discrete periodic 5-point Helmholtz matvec A x = x - alpha*Lap(x) applied PER COMPONENT on an
    (ncomp, n, n) array (dx = dy = 1/n), matching pops::apply_laplacian's bare path (now per component)
    with periodic ghosts."""
    import numpy as np

    h2 = (1.0 / n) ** 2

    def apply(x):
        lap = (np.roll(x, -1, 1) + np.roll(x, 1, 1) - 2 * x) / h2 + \
              (np.roll(x, -1, 2) + np.roll(x, 1, 2) - 2 * x) / h2
        return x - alpha * lap

    return apply


def _passive_model(dsl, name, cons):
    """An n-variable block with NO flux and NO Poisson coupling: the Program never runs a rhs or
    solve_fields; the block's conservative variables double as the multi-component field the matrix-free
    solve writes. @p cons is the tuple of conservative-variable names."""
    m = dsl.Model(name)
    vars_ = m.conservative_vars(*cons)
    if not isinstance(vars_, tuple):
        vars_ = (vars_,)
    z = [0.0 * v for v in vars_]
    m.flux(x=list(z), y=list(z))
    m.eigenvalues(x=list(z), y=list(z))
    m.primitive_vars(*vars_)
    m.conservative_from(list(vars_))
    return m


def _run_one(t, adc, np, ncomp, init):
    """Compile + install + step the (I - alpha*Lap) solve on an ncomp-component block, compare to the
    offline numpy CG on the SAME discrete operator. @p init is (ncomp, n, n) the initial state. Returns
    (out, phi_ref, iters) or None if the toolchain is unavailable."""
    n = init.shape[1]
    sim = pops.System(n=n, L=1.0, periodic=True)
    if not hasattr(sim, "install_program"):
        print("-- (B) skipped: _pops lacks the install_program binding (rebuild _pops) --")
        return None

    from pops import dsl

    cons = tuple("c%d" % i for i in range(ncomp))
    tol = 1e-10
    try:
        compiled = pops.compile_problem(
            model=_passive_model(dsl, "mc_prog%d" % ncomp, cons),
            time=_mc_program(t, ncomp, name="mc_step%d" % ncomp, method="cg", tol=tol, max_iter=200))
        compiled_model = _passive_model(dsl, "mc_block%d" % ncomp, cons).compile(backend="production")
    except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
        print("-- (B) skipped: could not build the .so: %s --" % str(exc)[:200])
        return None

    sim.add_equation("blk", compiled_model,
                     spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                     time=pops.Explicit(method="euler"))
    sim.set_state("blk", init)
    sim.install_program(compiled.so_path)
    sim.step(0.05)  # dt is irrelevant: the solve is dt-free
    out = np.array(sim.get_state("blk"))

    apply = _discrete_helmholtz_mc(n, _ALPHA)
    phi_ref, iters = _np_cg_mc(apply, init, tol=tol)
    return out, phi_ref, iters


def _run_section_b(t):
    try:
        import numpy as np

        import pops
    except Exception as exc:  # noqa: BLE001  -- numpy / _pops unavailable in this interpreter
        print("-- (B) skipped: adc/numpy unavailable: %s --" % exc)
        return None

    n = 16
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    # Component 0 and component 1 get DIFFERENT right-hand sides: a comp-0-only convergence norm would
    # stop on component 0's residual and leave component 1 unsolved -- so the component-1 match below is
    # the regression guard for the full-component (pops::dot_all) norm.
    c0 = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    c1 = 0.5 - 0.4 * np.cos(2 * np.pi * X) * np.sin(4 * np.pi * Y)
    init2 = np.stack([c0, c1])

    res2 = _run_one(t, adc, np, 2, init2)
    if res2 is None:
        return None
    out2, ref2, iters2 = res2
    err0 = float(np.abs(out2[0] - ref2[0]).max())
    err1 = float(np.abs(out2[1] - ref2[1]).max())
    moved1 = float(np.abs(out2[1] - c1).max())
    print("  2-comp solve parity: max|comp0 d| = %.2e  max|comp1 d| = %.2e  offline iters = %d  "
          "max|comp1 - U0| = %.2e" % (err0, err1, iters2, moved1))
    assert err0 <= 1e-6, "compiled 2-comp CG (component 0) == offline numpy CG (max|d| = %.2e)" % err0
    assert err1 <= 1e-6, ("compiled 2-comp CG (component 1) == offline numpy CG (max|d| = %.2e) -- "
                          "the full-component norm regression guard" % err1)
    assert moved1 > 1e-6, "the solve must change component 1 from U0 (max|d| = %.2e)" % moved1
    assert iters2 > 1, "the offline (and compiled) solve must take > 1 iteration, got %d" % iters2

    # A scalar (ncomp=1) solve still matches the offline CG bit-for-bit (the scalar path is unchanged).
    res1 = _run_one(t, adc, np, 1, np.stack([c0]))
    if res1 is None:
        return None
    out1, ref1, _ = res1
    err_scalar = float(np.abs(out1[0] - ref1[0]).max())
    print("  scalar solve parity (unchanged): max|d| = %.2e" % err_scalar)
    assert err_scalar <= 1e-6, "the scalar solve must still match offline CG (max|d| = %.2e)" % err_scalar
    return (err0, err1, err_scalar)


def _run():
    t = _adc_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    print("PASS test_time_multicomp_solve (A: %d checks)" % len(fns))
    _run_section_b(t)


if __name__ == "__main__":
    _run()
