"""adc.time Phase 4 IR ops (epic ADC-399 / ADC-403): named sources + local linear operators.

Pins the builder surface for split sources and cell-local implicit solves -- `P.source`,
`P.linear_source`, `P.apply`, `P.solve_local_linear`, and the operator algebra `P.I - a*L`. These
build typed IR (validated structurally); the codegen that LOWERS them is a later PR, so
`emit_cpp_program` still refuses a Program that uses them with a clear NotImplementedError (never a
mis-lowering). Pure Python (no compile / no _adc runtime); skips if adc is unavailable.
"""
import sys


def _adc_time():
    try:
        import adc.time as t
    except Exception as exc:  # adc not importable here -> skip, never fake
        print("skip test_time_local_solve (adc.time unavailable: %s)" % exc)
        sys.exit(0)
    return t


def _predictor_corrector(t):
    """Spec example 5: predictor-corrector Poisson/Lorentz (electric source + Lorentz local solve)."""
    P = t.Program("predictor_corrector_poisson_lorentz")
    dt = P.dt
    U_n = P.state("plasma")
    f_n = P.solve_fields("fields_n", U_n)
    R_n = P.rhs(name="R_n", state=U_n, fields=f_n, flux=True, sources=["electric"])
    U_star_rhs = P.linear_combine("U_star_rhs", U_n + dt * R_n)
    U_star = P.solve_local_linear(name="U_star", operator=P.I - dt * P.linear_source("lorentz"),
                                  rhs=U_star_rhs, fields=f_n)
    f_star = P.solve_fields("fields_star", U_star)
    R_star = P.rhs(name="R_star", state=U_star, fields=f_star, flux=True, sources=["electric"])
    C_star = P.apply(operator=P.linear_source("lorentz"), state=U_star, fields=f_star, name="C_star")
    Q = P.linear_combine("Q", U_n + 0.5 * dt * R_n + 0.5 * dt * R_star + 0.5 * dt * C_star)
    U_np1 = P.solve_local_linear(name="U_np1", operator=P.I - 0.5 * dt * P.linear_source("lorentz"),
                                 rhs=Q, fields=f_star)
    P.solve_fields("fields_np1", U_np1)
    P.commit("plasma", U_np1)
    return P


def test_predictor_corrector_builds(t):
    P = _predictor_corrector(t)
    assert P.validate() is True, "the predictor-corrector IR must validate"
    assert P._ir_hash(), "the IR must serialize to a stable hash"


def test_a_coeff_recorded_and_hashed(t):
    # operator I - a*dt*L records a as a dt-polynomial in attrs; a changes the IR hash.
    def prog(a):
        P = t.Program("scl")
        U = P.state("plasma")
        Q = P.linear_combine("Q", 1.0 * U)
        op = P.I - a * P.dt * P.linear_source("lorentz")
        P.commit("plasma", P.solve_local_linear(name="W", operator=op, rhs=Q))
        return P
    assert prog(1.0)._ir_hash() != prog(0.5)._ir_hash(), "a different solve coefficient must rehash"


def test_solve_local_linear_rejects_non_operator(t):
    P = t.Program("bad")
    U = P.state("plasma")
    Q = P.linear_combine("Q", 1.0 * U)
    try:  # a plain State is not a local linear operator
        P.solve_local_linear(name="W", operator=U, rhs=Q)
    except ValueError as exc:
        assert "local linear operators only" in str(exc)
    else:
        raise AssertionError("expected ValueError for a non-operator")


def test_solve_local_linear_requires_identity(t):
    P = t.Program("bad2")
    U = P.state("plasma")
    Q = P.linear_combine("Q", 1.0 * U)
    try:  # a*L without the identity I is not the I +/- a*L form
        P.solve_local_linear(name="W", operator=P.dt * P.linear_source("lorentz"), rhs=Q)
    except ValueError as exc:
        assert "local linear operators only" in str(exc)
    else:
        raise AssertionError("expected ValueError for an operator without identity")


def test_source_and_apply_are_rhs_like(t):
    P = t.Program("sa")
    U = P.state("plasma")
    f = P.solve_fields(U)
    S = P.source("electric", state=U, fields=f)
    LU = P.apply("lorentz", state=U, fields=f)
    assert S.vtype == "rhs" and LU.vtype == "rhs", "source/apply are dU/dt-like (RHS) values"
    P.commit("plasma", P.linear_combine("Un", U + P.dt * S + P.dt * LU))
    assert P.validate() is True


def test_codegen_refuses_phase4_ops(t):
    # The codegen does not lower Phase-4 ops yet -> NotImplementedError, never a silent mis-lowering.
    P = _predictor_corrector(t)
    try:
        P.emit_cpp_program()
    except NotImplementedError as exc:
        msg = str(exc).lower()
        assert "op" in msg or "source" in msg or "solve_local_linear" in msg
    else:
        raise AssertionError("expected NotImplementedError (Phase-4 codegen is a later PR)")


def _run():
    t = _adc_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    print("PASS test_time_local_solve (%d checks)" % len(fns))


if __name__ == "__main__":
    _run()
