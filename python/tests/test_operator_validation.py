"""Spec 2 (S2-12 / ADC-448): operator-first Program type diagnostics.

When states are tagged with their adc.model.StateSpace (P.state(block, space=U)) and rates/operators
flow from P.call, the Program type-checks the composition: a value over one StateSpace cannot feed an
operator typed for another, a Rate(U) cannot be combined with a State(V), and an L: U -> U cannot
drive a solve over State(V). Untagged (legacy) programs skip the checks, so Spec 1 is unaffected.
Pure Python; skips if adc is not importable.
"""
import sys

try:
    from adc import dsl, model
    from adc import time as adctime
except Exception as exc:  # adc not importable here -> skip, never fake
    print("skip test_operator_validation (adc unavailable: %s)" % exc)
    sys.exit(0)


def _model():
    m = dsl.Model("ep")
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    bz = m.aux("B_z")
    m.flux(x=[mx, mx * mx / rho, mx * my / rho], y=[my, mx * my / rho, my * my / rho])
    m.source_term("electric", [dsl.Const(0.0), -rho * gx, -rho * gy])
    m.linear_source("lorentz", [[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]])
    m.elliptic_rhs(rho - 1.0)
    m.rate_operator("explicit_rhs", flux=True, sources=["electric"])
    return m


_OTHER = model.StateSpace("V", ("a", "b", "c"))


def test_well_typed_program_passes():
    m = _model()
    u = m.state_space("U")
    P = adctime.Program("ok").bind_operators(m)
    u_n = P.state("plasma", space=u)
    fields = P.call("fields_from_state", u_n)
    rate = P.call("explicit_rhs", u_n, fields)
    lin = P.call("lorentz", fields)
    rhs = P.linear_combine("rhs", u_n + P.dt * rate)
    P.solve_local_linear("ustar", operator=P.I - P.dt * lin, rhs=rhs, fields=fields)
    print("OK  a well-typed operator-first program passes")


def test_call_input_space_mismatch():
    m = _model()
    u = m.state_space("U")
    P = adctime.Program("p").bind_operators(m)
    fields = P.call("fields_from_state", P.state("plasma", space=u))
    wrong = P.state("other", space=_OTHER)
    try:
        P.call("explicit_rhs", wrong, fields)  # explicit_rhs expects state 'U', got 'V'
        raise AssertionError("expected a state-space mismatch error")
    except ValueError as exc:
        assert "expects state 'U'" in str(exc) and "over 'V'" in str(exc), str(exc)
    print("OK  P.call rejects an argument over the wrong StateSpace")


def test_combine_space_mismatch():
    m = _model()
    u = m.state_space("U")
    P = adctime.Program("p").bind_operators(m)
    u_n = P.state("plasma", space=u)
    rate = P.call("explicit_rhs", u_n, P.call("fields_from_state", u_n))  # Rate(U)
    wrong = P.state("other", space=_OTHER)
    try:
        P.linear_combine("bad", u_n + P.dt * rate + wrong)  # mixes U and V
        raise AssertionError("expected a state-space combination error")
    except ValueError as exc:
        assert "different state spaces" in str(exc), str(exc)
    print("OK  linear_combine rejects mixing two StateSpaces")


def test_solve_local_linear_domain_mismatch():
    m = _model()
    u = m.state_space("U")
    P = adctime.Program("p").bind_operators(m)
    fields = P.call("fields_from_state", P.state("plasma", space=u))
    lin = P.call("lorentz", fields)  # LocalLinearOperator(U, U)
    rhs_v = P.state("other", space=_OTHER)
    try:
        P.solve_local_linear("bad", operator=P.I - P.dt * lin, rhs=rhs_v, fields=fields)
        raise AssertionError("expected an operator/state domain error")
    except ValueError as exc:
        assert "maps U -> U" in str(exc) and "State over 'V'" in str(exc), str(exc)
    print("OK  solve_local_linear rejects L: U->U on a State(V)")


def test_legacy_untagged_unaffected():
    # No space= tags -> the checks are skipped; a plain Spec-1-style program still builds.
    m = _model()
    P = adctime.Program("legacy").bind_operators(m)
    u = P.state("plasma")
    fields = P.solve_fields(u)
    r = P.rhs(state=u, fields=fields, sources=["electric"])
    P.commit("plasma", P.linear_combine("u1", u + P.dt * r))
    P.validate()
    print("OK  untagged (legacy) programs skip the space checks")


def main():
    test_well_typed_program_passes()
    test_call_input_space_mismatch()
    test_combine_space_mismatch()
    test_solve_local_linear_domain_mismatch()
    test_legacy_untagged_unaffected()
    print("OK  test_operator_validation")


if __name__ == "__main__":
    main()
