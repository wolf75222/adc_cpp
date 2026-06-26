#!/usr/bin/env python3
"""pops.time control flow + reductions codegen (epic ADC-399 / ADC-404a).

`emit_cpp_program` now lowers a CONVERGENCE LOOP: scalar reductions (``P.norm2`` / ``P.dot`` ->
``pops::dot`` collective all_reduce), scalar comparisons (``P.norm2(diff) > tol`` -> a Bool value), and
``P.while_(state, cond_fn, body_fn)`` (an infinite C++ loop with a break that RE-EVALUATES the
condition each pass, mutating the loop-variable state in place). The cond / body ops are recorded in a
SEPARATE sub-block (a recording scope), not the flat SSA list, so the body re-runs each iteration.

(A) Codegen (pure Python, always runs): a Program with a while_ loop lowers to ``pops::dot`` +
    ``std::sqrt`` (norm2), ``for (;;)`` + ``if (!(`` + ``break;`` (the loop), and the scalar / bool
    value types + the loud Python guards (a Scalar/Bool must not silently collapse to a Python bool /
    index).

(B) End-to-end parity (skips unless the full toolchain is present): a convergent fixed-point iteration
    x <- x + omega*(target - x) (omega = 0.5) looping while ``norm2(target - x) > tol``; compile_problem
    -> problem.so, install_program, step once, and compare the final state to the OFFLINE geometric
    reference x_k = target + (1-omega)^k (x0 - target). Self-skips without numpy / _pops / a compiler /
    Kokkos / install_program (never faking the engine).
"""
import sys


def _pops_time():
    try:
        import pops.time as t
    except Exception as exc:  # pops not importable here -> skip, never fake
        print("skip test_time_control_flow (pops.time unavailable: %s)" % exc)
        sys.exit(0)
    return t


def _convergence_program(t, *, name="fixed_point", omega=0.5, tol=1e-10):
    """x <- x + omega*(target - x), looping while ||target - x|| > tol (target = 2*U0).

    The body and condition use only linear_combine + norm2, so the Program lowers with NO model
    (solve_fields is inert / absent); the loop drives the fixed point entirely in C++."""
    P = t.Program(name)
    U0 = P.state("blk")
    target = P.linear_combine("target", 2.0 * U0)  # a fixed target state == 2*U0

    def cond(P, x):
        diff = P.linear_combine("diff", target - x)
        return P.norm2(diff) > tol

    def body(P, x):
        # x_next = x + omega*(target - x) = (1-omega)*x + omega*target
        return P.linear_combine("x_next", (1.0 - omega) * x + omega * target)

    x_final = P.while_(U0, cond, body)
    P.commit("blk", x_final)
    return P


# ---- (A) codegen: pure Python, always runs ----
def test_norm2_is_scalar(t):
    P = t.Program("p")
    U = P.state("blk")
    s = P.norm2(U)
    assert s.vtype == "scalar", "P.norm2 returns a Scalar value (got %r)" % s.vtype


def test_dot_is_scalar(t):
    P = t.Program("p")
    U = P.state("blk")
    s = P.dot(U, U)
    assert s.vtype == "scalar", "P.dot returns a Scalar value (got %r)" % s.vtype


def test_compare_is_bool(t):
    P = t.Program("p")
    U = P.state("blk")
    b = P.norm2(U) > 1e-8
    assert b.vtype == "bool", "a scalar comparison returns a Bool value (got %r)" % b.vtype


def test_scalar_not_python_bool(t):
    P = t.Program("p")
    U = P.state("blk")
    s = P.norm2(U)
    try:
        bool(s)
    except TypeError as exc:
        assert "cannot be used as a Python bool" in str(exc), str(exc)
    else:
        raise AssertionError("a Scalar must not silently collapse to a Python bool")


def test_bool_not_python_bool(t):
    P = t.Program("p")
    U = P.state("blk")
    b = P.norm2(U) > 1e-8
    try:
        bool(b)
    except TypeError as exc:
        assert "cannot be used as a Python bool" in str(exc), str(exc)
    else:
        raise AssertionError("a Bool must not silently collapse to a Python bool")


def test_scalar_not_python_index(t):
    P = t.Program("p")
    U = P.state("blk")
    s = P.norm2(U)
    try:
        range(s)  # __index__ must fire, never a silent integer
    except TypeError as exc:
        assert "index" in str(exc).lower() or "bool" in str(exc).lower(), str(exc)
    else:
        raise AssertionError("a Scalar must not be usable as a Python index")


def test_state_bool_still_loud(t):
    # The existing field-value guard must stay intact (only the scalar/bool branch is ADDED).
    P = t.Program("p")
    U = P.state("blk")
    try:
        bool(U)
    except TypeError:
        pass
    else:
        raise AssertionError("a State value must still refuse to be a Python bool")


def test_while_codegen(t):
    P = _convergence_program(t)
    src = P.emit_cpp_program()
    for frag in ("pops::dot", "std::sqrt", "for (;;)", "if (!(", "break;"):
        assert frag in src, "the generated while loop must contain %r\n%s" % (frag, src)


def test_while_validates(t):
    P = _convergence_program(t)
    assert P.validate() is True, "the while Program must validate"
    assert P._ir_hash(), "the IR must serialize to a stable hash"


# ---- (B) end-to-end parity: skips unless the full toolchain is present ----
def _run_section_b(t):
    try:
        import numpy as np

        import pops
    except Exception as exc:  # noqa: BLE001  -- numpy / _pops unavailable in this interpreter
        print("-- (B) skipped: pops/numpy unavailable: %s --" % exc)
        return None

    n = 8
    sim = pops.System(n=n, L=1.0, periodic=True)
    if not hasattr(sim, "install_program"):
        print("-- (B) skipped: _pops lacks the install_program binding (rebuild _pops) --")
        return None

    from pops import dsl

    # A minimal 1-variable model with NO Poisson coupling: solve_fields is inert and the while body
    # needs no fields. A complete compilable block (flux + primitive + eigenvalue).
    def passive_model(name):
        m = dsl.Model(name)
        (rho,) = m.conservative_vars("rho")
        u = m.primitive("u", 0.0 * rho)  # passive advection at speed 0 (the Program never runs a rhs)
        m.primitive_vars(rho=rho, u=u)
        m.conservative_from([rho])
        m.flux(x=[0.0 * rho], y=[0.0 * rho])
        m.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
        return m

    omega, tol = 0.5, 1e-10
    try:
        compiled = pops.compile_problem(
            model=passive_model("cflow_prog"),
            time=_convergence_program(t, name="cflow_step", omega=omega, tol=tol))
    except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
        print("-- (B) skipped: compile_problem could not build the .so: %s --" % str(exc)[:160])
        return None

    assert compiled.program_name == "cflow_step", "handle carries the program name"

    try:
        compiled_model = passive_model("cflow_block").compile(backend="production")
    except RuntimeError as exc:  # no compiler / no Kokkos visible
        print("-- (B) skipped: model compile could not build the .so: %s --" % str(exc)[:160])
        return None
    sim.add_equation("blk", compiled_model,
                     spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                     time=pops.Explicit(method="euler"))
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("blk", np.stack([rho0]))

    sim.install_program(compiled.so_path)
    sim.step(0.05)  # dt is irrelevant: the while body is dt-free
    out = np.array(sim.get_state("blk"))[0]

    # OFFLINE geometric reference: target = 2*U0; x_{k+1} = x_k + omega*(target - x_k); iterate while
    # ||target - x_k|| > tol (the same loop the compiled program runs).
    target = 2.0 * rho0
    xk = rho0.copy()
    iters = 0
    while np.sqrt(float(np.sum((target - xk) ** 2))) > tol:
        xk = xk + omega * (target - xk)
        iters += 1
    err = float(np.abs(out - xk).max())
    moved = float(np.abs(out - rho0).max())
    print("  while parity: max|compiled - offline| = %.2e  iters = %d  max|x - U0| = %.2e"
          % (err, iters, moved))
    assert err <= 1e-12, "compiled while loop == offline geometric reference (max|d| = %.2e)" % err
    assert iters > 1, "the loop must actually iterate (>1 pass), got %d" % iters
    assert moved > 1e-6, "the loop must change the state from U0 (max|d| = %.2e)" % moved
    return (err, iters)


def _run():
    t = _pops_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    print("PASS test_time_control_flow (A: %d checks)" % len(fns))
    _run_section_b(t)


if __name__ == "__main__":
    _run()
