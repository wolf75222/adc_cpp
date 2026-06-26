"""Spec 2 (S2-4): operator-first standard macros.

pops.time.std.predictor_corrector_local_linear / explicit_rk / imex_local_linear take typed
operator NAMES (not physical terms) and compose them with P.call against the Module bound to
the Program. The macros are model-free (their source mentions no physics) and reusable across
any Module with matching signatures. Pure Python (emit only); skips if adc is not importable.
"""
import inspect
import sys

try:
    from pops import dsl
    from pops import time as adctime
except Exception as exc:  # adc not importable here -> skip, never fake
    print("skip test_operator_macros (adc unavailable: %s)" % exc)
    sys.exit(0)

_PHYSICS_TOKENS = ("electric", "lorentz", "poisson", "rho", "grad_x", "grad_y", "B_z")


def _model(name, gain=1.0):
    m = dsl.Model(name)
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    bz = m.aux("B_z")
    m.flux(x=[mx, mx * mx / rho, mx * my / rho],
           y=[my, mx * my / rho, my * my / rho])
    m.source_term("electric", [dsl.Const(0.0), rho * (-gx) * gain, rho * (-gy) * gain])
    m.linear_source("lorentz", [[0.0, 0.0, 0.0],
                                [0.0, 0.0, bz],
                                [0.0, -bz, 0.0]])
    m.elliptic_rhs(rho - 1.0)
    m.rate_operator("explicit_rhs", flux=True, sources=["electric"])
    return m


def test_macros_are_model_free():
    for macro in (adctime.std.predictor_corrector_local_linear,
                  adctime.std.explicit_rk,
                  adctime.std.imex_local_linear):
        src = inspect.getsource(macro)
        for tok in _PHYSICS_TOKENS:
            assert tok not in src, "%s must not mention %r" % (macro.__name__, tok)
    print("OK  the operator-first macros mention no physics term")


def test_predictor_corrector_macro():
    m = _model("ep")
    P = adctime.Program("pc").bind_operators(m)
    adctime.std.predictor_corrector_local_linear(
        P, "plasma", fields_operator="fields_from_state",
        explicit_rate_operator="explicit_rhs", implicit_operator="lorentz")
    P.validate()
    src = P.emit_cpp_program(model=m)
    assert "pops_install_program" in src
    print("OK  predictor_corrector_local_linear composes 3 typed operators -> .so source")


def test_explicit_rk_macro():
    m = _model("rk")
    P = adctime.Program("rk").bind_operators(m)
    adctime.std.explicit_rk(P, "plasma", rhs_operator="explicit_rhs",
                            fields_operator="fields_from_state",
                            tableau=adctime.std.SSPRK2_TABLEAU)
    P.validate()
    assert "pops_install_program" in P.emit_cpp_program(model=m)
    print("OK  explicit_rk over a typed rate operator (SSPRK2 tableau)")


def test_imex_local_linear_macro():
    m = _model("imex")
    P = adctime.Program("imex").bind_operators(m)
    adctime.std.imex_local_linear(P, "plasma", explicit_operator="explicit_rhs",
                                  implicit_operator="lorentz",
                                  fields_operator="fields_from_state", theta=1.0)
    P.validate()
    assert "pops_install_program" in P.emit_cpp_program(model=m)
    print("OK  imex_local_linear (theta-implicit local linear solve)")


def test_macro_reused_across_modules():
    def build(m):
        P = adctime.Program("pc").bind_operators(m)
        adctime.std.predictor_corrector_local_linear(
            P, "plasma", fields_operator="fields_from_state",
            explicit_rate_operator="explicit_rhs", implicit_operator="lorentz")
        return P.emit_cpp_program(model=m)

    src_a = build(_model("A", 1.0))
    src_b = build(_model("B", 2.0))
    assert "pops_install_program" in src_a and src_a != src_b
    print("OK  the same predictor-corrector macro is reused across two modules")


def main():
    test_macros_are_model_free()
    test_predictor_corrector_macro()
    test_explicit_rk_macro()
    test_imex_local_linear_macro()
    test_macro_reused_across_modules()
    print("OK  test_operator_macros")


if __name__ == "__main__":
    main()
