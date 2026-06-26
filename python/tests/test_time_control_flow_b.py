#!/usr/bin/env python3
"""pops.time structured for-loops + if + norm_inf codegen (epic ADC-399 / ADC-404b).

Builds on ADC-404a (Scalar/Bool IR, P.norm2/P.dot, P.while_). This slice adds:
  - ``P.norm_inf(state)`` -> a Scalar (``pops::norm_inf``);
  - ``P.static_range(state, count, body)`` -- a COMPILE-TIME unrolled loop (count copies of the body
    inline, NO C++ loop);
  - ``P.range(state, count, body)`` -- a C++ ``for`` over a fixed count (body emitted ONCE, re-run each
    pass);
  - ``P.if_(state, cond, body)`` -- a C++ ``if`` branch on a runtime Bool.

(A) Codegen (pure Python, always runs): static_range unrolls (body N times, no ``for``), range emits a
    C++ ``for (int``, if_ emits ``if (``, norm_inf is a Scalar emitting ``pops::norm_inf``; the IR hash
    distinguishes loop counts and unrolled bodies; runtime-count guards fire.

(B) End-to-end parity (skips unless the full toolchain is present): a dt-free contraction
    x <- 0.5*x + 0.5*target (target = 2*U0); range(3) and static_range(3) both compile, install, step,
    and match the offline x_N = target + 0.5^N (x0 - target); if_ applies the body iff the runtime
    condition holds. Self-skips without numpy / _pops / a compiler / Kokkos (never faking the engine).
"""
import sys


def _adc_time():
    try:
        import pops.time as t
    except Exception as exc:  # adc not importable here -> skip, never fake
        print("skip test_time_control_flow_b (pops.time unavailable: %s)" % exc)
        sys.exit(0)
    return t


def _fe_body():
    """A Forward-Euler body x -> x + dt*(-div F): rhs(sources=['default']) lowers with NO model, so it
    serves the codegen asserts (one ``ctx.rhs_into`` per emitted copy of the body)."""
    def body(_P, x):
        return _P.linear_combine(x + _P.dt * _P.rhs(state=x, sources=["default"]))
    return body


def _contraction_body(target):
    """A dt-free contraction x -> 0.5*x + 0.5*target (uses only linear_combine, so it runs in the
    compiled .so with no flux / solve_fields and has a closed-form offline reference)."""
    def body(_P, x):
        return _P.linear_combine(0.5 * x + 0.5 * target)
    return body


# ---- (A) codegen: pure Python, always runs ----
def test_norm_inf_is_scalar(t):
    P = t.Program("p")
    U = P.state("blk")
    s = P.norm_inf(U)
    assert s.vtype == "scalar", "P.norm_inf returns a Scalar value (got %r)" % s.vtype


def test_static_range_unrolls(t):
    P = t.Program("sr")
    U = P.state("blk")
    Uf = P.static_range(U, 3, _fe_body())
    P.commit("blk", Uf)
    src = P.emit_cpp_program()
    assert src.count("ctx.rhs_into") == 3, "static_range(3) unrolls the body 3 times\n%s" % src
    assert "for (" not in src, "static_range must NOT emit a C++ loop (it is unrolled)\n%s" % src


def test_range_emits_for(t):
    P = t.Program("rg")
    U = P.state("blk")
    Uf = P.range(U, 3, _fe_body())
    P.commit("blk", Uf)
    src = P.emit_cpp_program()
    assert "for (int i" in src, "range must emit a C++ for loop\n%s" % src
    assert src.count("ctx.rhs_into") == 1, "range emits the body ONCE (inside the loop)\n%s" % src


def test_if_emits_branch(t):
    P = t.Program("if")
    U = P.state("blk")
    cond = P.norm_inf(U) > 0.0
    Uf = P.if_(U, cond, _fe_body())
    P.commit("blk", Uf)
    src = P.emit_cpp_program()
    assert "pops::norm_inf" in src, "norm_inf must lower to pops::norm_inf\n%s" % src
    assert "if (" in src, "if_ must emit a C++ if branch\n%s" % src
    assert "for (" not in src, "if_ alone emits no loop\n%s" % src


def test_static_range_scalar_count_rejected(t):
    P = t.Program("p")
    U = P.state("blk")
    try:
        P.static_range(U, P.norm2(U), _fe_body())  # a runtime Scalar is not a compile-time count
    except TypeError as exc:
        assert "Python int" in str(exc), str(exc)
    else:
        raise AssertionError("static_range with a Scalar count must raise TypeError")


def test_range_scalar_count_rejected(t):
    P = t.Program("p")
    U = P.state("blk")
    try:
        P.range(U, P.norm2(U), _fe_body())
    except NotImplementedError as exc:
        assert "runtime Scalar count" in str(exc), str(exc)
    else:
        raise AssertionError("range with a Scalar count must raise NotImplementedError")


def test_range_count_changes_hash(t):
    def prog(count):
        P = t.Program("rg")
        U = P.state("blk")
        P.commit("blk", P.range(U, count, _fe_body()))
        return P._ir_hash()
    assert prog(3) != prog(4), "a different range count must change the IR hash (cache key)"


def test_static_range_body_changes_hash(t):
    def prog(c):
        P = t.Program("sr")
        U = P.state("blk")
        target = P.linear_combine("target", 2.0 * U)
        P.commit("blk", P.static_range(U, 2, lambda _P, x: _P.linear_combine(c * x + 0.5 * target)))
        return P._ir_hash()
    assert prog(0.5) != prog(0.25), "a different unrolled body must change the IR hash"


# ---- (B) end-to-end parity: skips unless the full toolchain is present ----
def _contraction_program(t, kind, count, *, name):
    """target = 2*U0; loop x -> 0.5*x + 0.5*target  `count` times via `kind` in {'range','static'}."""
    P = t.Program(name)
    U0 = P.state("blk")
    target = P.linear_combine("target", 2.0 * U0)
    body = _contraction_body(target)
    xf = P.range(U0, count, body) if kind == "range" else P.static_range(U0, count, body)
    P.commit("blk", xf)
    return P


def _if_program(t, *, name, threshold):
    """if norm_inf(target - U0) > threshold: U <- 0.5*U0 + 0.5*target  (else U unchanged)."""
    P = t.Program(name)
    U0 = P.state("blk")
    target = P.linear_combine("target", 2.0 * U0)
    diff = P.linear_combine("diff", target - U0)
    cond = P.norm_inf(diff) > threshold
    xf = P.if_(U0, cond, _contraction_body(target))
    P.commit("blk", xf)
    return P


def _passive_model(dsl, name):
    """A 1-variable model with zero flux + no Poisson coupling: the dt-free Program body needs no rhs /
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

        import pops
    except Exception as exc:  # noqa: BLE001  -- numpy / _pops unavailable in this interpreter
        print("-- (B) skipped: adc/numpy unavailable: %s --" % exc)
        return None

    n = 8
    sim = pops.System(n=n, L=1.0, periodic=True)
    if not hasattr(sim, "install_program"):
        print("-- (B) skipped: _pops lacks install_program (rebuild _pops) --")
        return None

    from pops import dsl

    count = 3
    try:
        compiled = pops.compile_problem(model=_passive_model(dsl, "cf_rg"),
                                       time=_contraction_program(t, "range", count, name="cf_range"))
        compiled_sr = pops.compile_problem(model=_passive_model(dsl, "cf_sr"),
                                          time=_contraction_program(t, "static", count, name="cf_sr"))
        compiled_if = pops.compile_problem(model=_passive_model(dsl, "cf_if"),
                                          time=_if_program(t, name="cf_if", threshold=1e3))  # cond FALSE
        compiled_if_t = pops.compile_problem(model=_passive_model(dsl, "cf_ift"),
                                            time=_if_program(t, name="cf_ift", threshold=0.0))  # TRUE
    except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
        print("-- (B) skipped: compile_problem could not build a .so: %s --" % str(exc)[:160])
        return None

    def run(handle):
        s = pops.System(n=n, L=1.0, periodic=True)
        try:
            cm = _passive_model(dsl, "blk_" + handle.program_name).compile(backend="production")
        except RuntimeError as exc:
            print("-- (B) skipped: model compile failed: %s --" % str(exc)[:140])
            return None
        s.add_equation("blk", cm, spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                       time=pops.Explicit(method="euler"))
        x = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(x, x, indexing="ij")
        rho0 = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        s.set_state("blk", np.stack([rho0]))
        s.install_program(handle.so_path)
        s.step(0.05)  # dt irrelevant: the body is dt-free
        return rho0, np.array(s.get_state("blk"))[0]

    # OFFLINE reference for the contraction: x_{k+1} = 0.5 x_k + 0.5 target, target = 2 rho0.
    res = run(compiled)
    if res is None:
        return None
    rho0, out_rg = res
    target = 2.0 * rho0
    xk = rho0.copy()
    for _ in range(count):
        xk = 0.5 * xk + 0.5 * target
    err_rg = float(np.abs(out_rg - xk).max())
    _, out_sr = run(compiled_sr)
    err_sr = float(np.abs(out_sr - xk).max())
    _, out_if_false = run(compiled_if)        # cond false -> unchanged
    _, out_if_true = run(compiled_if_t)       # cond true  -> one body application
    x_one = 0.5 * rho0 + 0.5 * target
    err_if_false = float(np.abs(out_if_false - rho0).max())
    err_if_true = float(np.abs(out_if_true - x_one).max())
    print("  range(%d) parity = %.2e | static_range = %.2e | if(false) = %.2e | if(true) = %.2e"
          % (count, err_rg, err_sr, err_if_false, err_if_true))
    assert err_rg <= 1e-12, "range(%d) == offline contraction (max|d| = %.2e)" % (count, err_rg)
    assert err_sr <= 1e-12, "static_range(%d) == offline contraction (max|d| = %.2e)" % (count, err_sr)
    assert float(np.abs(out_rg - out_sr).max()) <= 1e-14, "range == static_range bit-for-bit"
    assert err_if_false <= 1e-14, "if_ with a false condition leaves the state unchanged (%.2e)" \
        % err_if_false
    assert err_if_true <= 1e-12, "if_ with a true condition applies the body once (%.2e)" % err_if_true
    assert float(np.abs(out_rg - rho0).max()) > 1e-6, "the loop must change the state"
    return (err_rg, err_sr)


def _run():
    t = _adc_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    print("PASS test_time_control_flow_b (A: %d checks)" % len(fns))
    _run_section_b(t)


if __name__ == "__main__":
    _run()
