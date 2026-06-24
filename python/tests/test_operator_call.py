"""Spec 2 (S2-2): typed P.call and m.rate_operator.

P.call resolves an operator name against the model's typed registry, type-checks the
arguments against its Signature, and lowers to the matching PDE shortcut so the
generated C++ is IDENTICAL to the Spec 1 path. m.rate_operator names a composite
-div F + sources rate as a Program-side alias. Pure Python (emit_cpp_program returns
the .so source text without compiling); skips cleanly if adc is not importable.
"""
import sys

try:
    from adc import dsl
    from adc import time as adctime
except Exception as exc:  # adc not importable here -> skip, never fake
    print("skip test_operator_call (adc unavailable: %s)" % exc)
    sys.exit(0)


def build_model():
    m = dsl.Model("euler_poisson_lorentz")
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    m.aux("phi")
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    bz = m.aux("B_z")
    m.flux(x=[mx, mx * mx / rho, mx * my / rho],
           y=[my, mx * my / rho, my * my / rho])
    m.source_term("electric", [dsl.Const(0.0), rho * (-gx), rho * (-gy)])
    m.linear_source("lorentz", [[0.0, 0.0, 0.0],
                                [0.0, 0.0, bz],
                                [0.0, -bz, 0.0]])
    m.elliptic_rhs(rho - 1.0)
    m.rate_operator("explicit_rhs", flux=True, sources=["electric"])
    return m


def _emit(build, m, name="prog"):
    P = adctime.Program(name)
    build(P, m)
    return P.emit_cpp_program(model=m)


def test_call_matches_shortcut_predictor():
    """A predictor step written with P.call emits byte-identically to the PDE shortcut."""
    m = build_model()

    def shortcut(P, _m):
        U = P.state("plasma")
        f = P.solve_fields(U)
        R = P.rhs(state=U, fields=f, flux=True, sources=["electric"])
        P.commit("plasma", P.linear_combine("u1", U + P.dt * R))

    def opfirst(P, _m):
        P.bind_operators(_m)
        U = P.state("plasma")
        f = P.call("fields_from_state", U)
        R = P.call("explicit_rhs", U, f)
        P.commit("plasma", P.linear_combine("u1", U + P.dt * R))

    assert _emit(shortcut, m) == _emit(opfirst, m)
    print("OK  P.call(fields_from_state)+P.call(explicit_rhs) == solve_fields + rhs")


def test_call_matches_source_and_flux():
    m = build_model()

    def shortcut(P, _m):
        U = P.state("plasma")
        f = P.solve_fields(U)
        s = P.source("electric", state=U, fields=f)
        flux = P.rhs(state=U, flux=True, sources=[])
        P.commit("plasma", P.linear_combine("u1", U + P.dt * s + P.dt * flux))

    def opfirst(P, _m):
        P.bind_operators(_m)
        U = P.state("plasma")
        f = P.call("fields_from_state", U)
        s = P.call("electric", U, f)
        flux = P.call("flux_default", U)
        P.commit("plasma", P.linear_combine("u1", U + P.dt * s + P.dt * flux))

    assert _emit(shortcut, m) == _emit(opfirst, m)
    print("OK  P.call(electric)/P.call(flux_default) == source / flux-only rhs")


def test_call_linear_operator_matches_solve_local_linear():
    m = build_model()

    def shortcut(P, _m):
        U = P.state("plasma")
        f = P.solve_fields(U)
        L = P.linear_source("lorentz")
        U1 = P.solve_local_linear("u1", operator=P.I - P.dt * L, rhs=U, fields=f)
        P.commit("plasma", U1)

    def opfirst(P, _m):
        P.bind_operators(_m)
        U = P.state("plasma")
        f = P.call("fields_from_state", U)
        L = P.call("lorentz", f)
        U1 = P.solve_local_linear("u1", operator=P.I - P.dt * L, rhs=U, fields=f)
        P.commit("plasma", U1)

    assert _emit(shortcut, m) == _emit(opfirst, m)
    print("OK  P.call(lorentz) operator drives solve_local_linear identically")


def test_call_typing_errors():
    m = build_model()
    P = adctime.Program("p").bind_operators(m)
    U = P.state("plasma")
    f = P.call("fields_from_state", U)

    # No bind -> clear error.
    P2 = adctime.Program("p2")
    try:
        P2.call("electric", P2.state("plasma"))
        raise AssertionError("expected an error calling without bound operators")
    except ValueError as exc:
        assert "no operators bound" in str(exc)

    # Unknown operator -> clear KeyError.
    try:
        P.call("does_not_exist", U)
        raise AssertionError("expected KeyError for an unknown operator")
    except KeyError as exc:
        assert "unknown operator" in str(exc)

    # Arity mismatch -> electric needs (state, fields).
    try:
        P.call("electric", U)
        raise AssertionError("expected arity error")
    except ValueError as exc:
        assert "expects 2 argument" in str(exc)

    # vtype mismatch -> a fields value where a state is expected.
    try:
        P.call("electric", f, f)
        raise AssertionError("expected a vtype error")
    except ValueError as exc:
        assert "expects a state value" in str(exc)
    print("OK  P.call typing: no-bind / unknown / arity / vtype errors are clear")


def test_default_resolution_and_ambiguity():
    m = build_model()
    reg = m.operator_registry()
    # The privileged defaults resolve uniquely.
    assert reg.default_of_kind("field_operator").name == "fields_from_state"
    assert reg.default_of_kind("grid_operator").name == "flux_default"
    # Add a SECOND, non-privileged field operator; the privileged default still wins.
    m.elliptic_field("psi", rhs=dsl.Var("rho", "cons"), aux=["psi_x"])
    reg2 = m.operator_registry()
    assert len(reg2.operators_of_kind("field_operator")) == 2
    assert reg2.default_of_kind("field_operator").name == "fields_from_state"
    print("OK  default_of_kind resolves privileged defaults; second field op is explicit")


def test_rate_operator_alias_not_in_hash():
    m = dsl.Model("m")
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    m.flux(x=[mx, mx, mx], y=[my, my, my])
    m.source_term("relax", [dsl.Const(0.0), -mx, -my])
    h0 = m._model_hash()
    m.rate_operator("explicit_rhs", flux=True, sources=["relax"])
    assert m._model_hash() == h0, "a rate_operator alias must not change the model hash"
    assert "explicit_rhs" in m.operator_registry()
    print("OK  m.rate_operator is a pure alias (no model-hash impact)")


def main():
    test_call_matches_shortcut_predictor()
    test_call_matches_source_and_flux()
    test_call_linear_operator_matches_solve_local_linear()
    test_call_typing_errors()
    test_default_resolution_and_ambiguity()
    test_rate_operator_alias_not_in_hash()
    print("OK  test_operator_call")


if __name__ == "__main__":
    main()
