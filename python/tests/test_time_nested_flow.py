#!/usr/bin/env python3
"""adc.time NESTED control flow + runtime-count range codegen (epic ADC-399 / ADC-433).

Beyond-MVP completeness on top of ADC-404a/b (P.while_ / P.range / P.if_): a control-flow body may
now itself open another while_ / range / if_ (nesting), and P.range accepts a RUNTIME integer count
built with P.to_int (e.g. a count derived from a reduction). Both lower with the existing per-op
codegen -- the emitters recurse through nested body sub-blocks, scratch names stay unique per level
(each is keyed on the globally-unique SSA value id), and the block index (ADC-426) threads through
every level.

(A) Codegen (pure Python, always runs): a while_ containing an if_ / a range nests cleanly (one outer
    `for (;;)` with an inner `if (` / `for (int i`); a runtime-count range emits `static_cast<int>`
    for the bound and reads the count once before the loop; the nesting depth is bounded (a runaway
    raises); a raw float Scalar count is rejected (must wrap in P.to_int); flat flow is unchanged.

(B) End-to-end parity (skips unless the full toolchain is present): a dt-free OUTER contraction loop
    x <- 0.5*x + 0.5*target with an INNER conditional correction (if the residual is still large,
    apply one more half-step) compiles, installs, steps, and matches the OFFLINE reference that runs
    the SAME nested loop in numpy; a nested while+range and a runtime-count range likewise match
    offline. Self-skips without numpy / _adc / a compiler / Kokkos / install_program (never faking).
"""
import sys


def _adc_time():
    try:
        import adc.time as t
    except Exception as exc:  # adc not importable here -> skip, never fake
        print("skip test_time_nested_flow (adc.time unavailable: %s)" % exc)
        sys.exit(0)
    return t


# ---- Program builders (shared by codegen + parity) ----
def _nested_while_if_program(t, *, name="nested_while_if", omega=0.5, tol=1e-10, corr_thresh=0.5):
    """Outer convergence loop x <- (1-omega)*x + omega*target, looping while ||target - x|| > tol; the
    body, AFTER the half-step, applies one INNER conditional correction (a second identical half-step)
    iff the remaining residual norm_inf still exceeds @p corr_thresh. target = 2*U0."""
    P = t.Program(name)
    U0 = P.state("blk")
    target = P.linear_combine("target", 2.0 * U0)

    def cond(P, x):
        return P.norm2(P.linear_combine("diff", target - x)) > tol

    def body(P, x):
        x1 = P.linear_combine("half", (1.0 - omega) * x + omega * target)
        far = P.norm_inf(P.linear_combine("resid", target - x1)) > corr_thresh

        def correct(P, y):
            return P.linear_combine("corr", (1.0 - omega) * y + omega * target)

        return P.if_(x1, far, correct)

    P.commit("blk", P.while_(U0, cond, body))
    return P


def _nested_while_range_program(t, *, name="nested_while_range", inner=2, tol=1e-10, omega=0.5):
    """Outer convergence loop while ||target - x|| > tol whose body runs an INNER C++ range of @p inner
    contraction passes x <- (1-omega)*x + omega*target. target = 2*U0."""
    P = t.Program(name)
    U0 = P.state("blk")
    target = P.linear_combine("target", 2.0 * U0)

    def cond(P, x):
        return P.norm2(P.linear_combine("diff", target - x)) > tol

    def body(P, x):
        def step(P, y):
            return P.linear_combine("inner", (1.0 - omega) * y + omega * target)

        return P.range(x, inner, step)

    P.commit("blk", P.while_(U0, cond, body))
    return P


def _runtime_count_program(t, *, name="runtime_count", omega=0.5):
    """A range whose count is a RUNTIME integer Scalar: count = to_int(sum(U0)) (a cell sum truncated to
    an int), applying x <- (1-omega)*x + omega*target that many passes. target = 2*U0."""
    P = t.Program(name)
    U0 = P.state("blk")
    target = P.linear_combine("target", 2.0 * U0)
    count = P.to_int(P.sum(U0))  # a runtime int Scalar (sum over component 0, truncated)

    def step(P, x):
        return P.linear_combine("step", (1.0 - omega) * x + omega * target)

    P.commit("blk", P.range(U0, count, step))
    return P


# ---- (A) codegen: pure Python, always runs ----
def test_nested_while_if_codegen(t):
    P = _nested_while_if_program(t)
    assert P.validate() is True, "the nested while/if Program must validate"
    src = P.emit_cpp_program()
    assert src.count("for (;;)") == 1, "exactly one outer while loop\n%s" % src
    assert "if (" in src, "the inner if_ branch must be emitted\n%s" % src
    assert P._ir_hash(), "the nested IR must serialize to a stable hash"


def test_nested_while_range_codegen(t):
    P = _nested_while_range_program(t, inner=2)
    assert P.validate() is True, "the nested while/range Program must validate"
    src = P.emit_cpp_program()
    assert "for (;;)" in src, "outer while loop\n%s" % src
    assert "for (int i" in src, "inner C++ range loop\n%s" % src


def test_runtime_count_range_codegen(t):
    P = _runtime_count_program(t)
    assert P.validate() is True, "the runtime-count range Program must validate"
    src = P.emit_cpp_program()
    assert "static_cast<int>" in src, "the runtime count must lower to static_cast<int>\n%s" % src
    assert "for (int i" in src, "a runtime-count range still emits a C++ for loop\n%s" % src
    # The runtime count is read ONCE (the int local) before the loop, not re-evaluated per pass.
    assert src.count("static_cast<int>(s") == 1, "the runtime count is computed once\n%s" % src


def test_runtime_count_changes_hash_vs_static(t):
    h_runtime = _runtime_count_program(t)._ir_hash()
    P = t.Program("runtime_count")
    U0 = P.state("blk")
    target = P.linear_combine("target", 2.0 * U0)
    P.commit("blk", P.range(U0, 3, lambda P, x: P.linear_combine("step", 0.5 * x + 0.5 * target)))
    assert h_runtime != P._ir_hash(), "a runtime count must hash differently from a static int count"


def test_to_int_requires_scalar(t):
    P = t.Program("p")
    U = P.state("blk")
    try:
        P.to_int(U)  # a State is not a Scalar
    except ValueError as exc:
        assert "Scalar" in str(exc), str(exc)
    else:
        raise AssertionError("to_int must reject a non-Scalar value")


def test_range_raw_float_scalar_rejected(t):
    P = t.Program("p")
    U = P.state("blk")
    try:
        P.range(U, P.sum(U), lambda P, x: P.linear_combine(0.5 * x))  # a raw float Scalar
    except TypeError as exc:
        assert "INTEGER Scalar" in str(exc) and "to_int" in str(exc), str(exc)
    else:
        raise AssertionError("range with a raw float Scalar count must raise TypeError")


def test_unbounded_nesting_rejected(t):
    P = t.Program("toodeep")
    U = P.state("blk")
    target = P.linear_combine("target", 2.0 * U)

    def make(depth):
        def body(P, x):
            if depth <= 0:
                return P.linear_combine("leaf", 0.5 * x + 0.5 * target)
            return P.range(x, 2, make(depth - 1))
        return body

    try:
        P.range(U, 2, make(32))  # nest far past the supported depth
    except RuntimeError as exc:
        assert "nesting" in str(exc).lower(), str(exc)
    else:
        raise AssertionError("an unbounded nesting must raise RuntimeError")


def test_flat_flow_unchanged(t):
    # No-regression: a flat range lowers byte-for-byte as before (the runtime-count path only triggers
    # for a Value count) -- the static-count for-bound is the plain integer literal.
    P = t.Program("rg")
    U = P.state("blk")
    P.commit("blk", P.range(U, 3, lambda P, x: P.linear_combine("s", 0.5 * x)))
    src = P.emit_cpp_program()
    assert "for (int i" in src and "< 3;" in src, "a static count emits the literal bound\n%s" % src
    assert "static_cast<int>" not in src, "a static count must NOT cast a runtime scalar\n%s" % src


# ---- (B) end-to-end parity: skips unless the full toolchain is present ----
def _passive_model(dsl, name):
    """A 1-variable zero-flux model with no Poisson coupling: the dt-free Program body needs no rhs /
    solve_fields, so this just provides a compilable block to install the program onto."""
    m = dsl.Model(name)
    (rho,) = m.conservative_vars("rho")
    u = m.primitive("u", 0.0 * rho)
    m.primitive_vars(rho=rho, u=u)
    m.conservative_from([rho])
    m.flux(x=[0.0 * rho], y=[0.0 * rho])
    m.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
    return m


def _run_section_b(t):
    try:
        import numpy as np

        import adc
    except Exception as exc:  # noqa: BLE001  -- numpy / _adc unavailable in this interpreter
        print("-- (B) skipped: adc/numpy unavailable: %s --" % exc)
        return None

    n = 8
    probe = adc.System(n=n, L=1.0, periodic=True)
    if not hasattr(probe, "install_program"):
        print("-- (B) skipped: _adc lacks install_program (rebuild _adc) --")
        return None

    from adc import dsl

    omega, tol, corr_thresh, inner = 0.5, 1e-10, 0.5, 2
    try:
        compiled_wi = adc.compile_problem(
            model=_passive_model(dsl, "nf_wi"),
            time=_nested_while_if_program(t, name="nf_wi", omega=omega, tol=tol,
                                          corr_thresh=corr_thresh))
        compiled_wr = adc.compile_problem(
            model=_passive_model(dsl, "nf_wr"),
            time=_nested_while_range_program(t, name="nf_wr", inner=inner, tol=tol, omega=omega))
        compiled_rc = adc.compile_problem(
            model=_passive_model(dsl, "nf_rc"),
            time=_runtime_count_program(t, name="nf_rc", omega=omega))
    except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
        print("-- (B) skipped: compile_problem could not build a .so: %s --" % str(exc)[:160])
        return None

    def run(handle):
        s = adc.System(n=n, L=1.0, periodic=True)
        try:
            cm = _passive_model(dsl, "blk_" + handle.program_name).compile(backend="production")
        except RuntimeError as exc:
            print("-- (B) skipped: model compile failed: %s --" % str(exc)[:140])
            return None
        s.add_equation("blk", cm, spatial=adc.FiniteVolume(limiter="none", riemann="rusanov"),
                       time=adc.Explicit(method="euler"))
        x = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(x, x, indexing="ij")
        rho0 = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        s.set_state("blk", np.stack([rho0]))
        s.install_program(handle.so_path)
        s.step(0.05)  # dt irrelevant: every body is dt-free
        return rho0, np.array(s.get_state("blk"))[0]

    # --- nested while + if: the OFFLINE reference runs the SAME nested loop in numpy. ---
    res = run(compiled_wi)
    if res is None:
        return None
    rho0, out_wi = res
    target = 2.0 * rho0

    def half(x):
        return (1.0 - omega) * x + omega * target

    xk, iters = rho0.copy(), 0
    while np.sqrt(float(np.sum((target - xk) ** 2))) > tol:
        x1 = half(xk)
        if float(np.abs(target - x1).max()) > corr_thresh:  # inner conditional correction
            x1 = half(x1)
        xk, iters = x1, iters + 1
    err_wi = float(np.abs(out_wi - xk).max())

    # --- nested while + range: OUTER loop, INNER fixed range of `inner` half-steps. ---
    _, out_wr = run(compiled_wr)
    yk, iters_wr = rho0.copy(), 0
    while np.sqrt(float(np.sum((target - yk) ** 2))) > tol:
        for _ in range(inner):
            yk = half(yk)
        iters_wr += 1
    err_wr = float(np.abs(out_wr - yk).max())

    # --- runtime-count range: count = int(sum(rho0)), `count` half-steps. ---
    _, out_rc = run(compiled_rc)
    count = int(float(np.sum(rho0)))  # the SAME truncation the to_int(sum) lowering applies
    zk = rho0.copy()
    for _ in range(count):
        zk = half(zk)
    err_rc = float(np.abs(out_rc - zk).max())

    print("  nested while/if = %.2e (iters %d) | while/range = %.2e (iters %d) | runtime range = %.2e "
          "(count %d)" % (err_wi, iters, err_wr, iters_wr, err_rc, count))
    assert err_wi <= 1e-12, "nested while/if == offline reference (max|d| = %.2e)" % err_wi
    assert err_wr <= 1e-12, "nested while/range == offline reference (max|d| = %.2e)" % err_wr
    assert err_rc <= 1e-12, "runtime-count range == offline reference (max|d| = %.2e)" % err_rc
    assert iters > 1, "the outer while/if loop must iterate (>1 pass), got %d" % iters
    assert count > 1, "the runtime count must be > 1 (got %d)" % count
    assert float(np.abs(out_rc - rho0).max()) > 1e-6, "the runtime-count loop must change the state"
    return (err_wi, err_wr, err_rc)


def _run():
    t = _adc_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    print("PASS test_time_nested_flow (A: %d checks)" % len(fns))
    _run_section_b(t)


if __name__ == "__main__":
    _run()
