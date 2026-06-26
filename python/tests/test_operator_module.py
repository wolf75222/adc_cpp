"""Spec 2 (S2-3): the public pops.model.Module API and dsl.Model as its PDE facade.

A Module is the model-free view: typed spaces + a registry of typed operators. dsl.Model
encapsulates a Module (its source_term / linear_source / elliptic_field / flux register
typed operators). A generic Program -- written only with operator names and signatures --
runs against any Module that provides the expected signatures. Pure Python; skips if adc
is not importable.
"""
import sys

try:
    from pops import dsl, model
    from pops import time as adctime
except Exception as exc:  # adc not importable here -> skip, never fake
    print("skip test_operator_module (adc unavailable: %s)" % exc)
    sys.exit(0)


def test_signature_sugar():
    u = model.StateSpace("U", ("rho", "mx", "my"))
    f = model.FieldSpace("fields", ("phi", "grad_x", "grad_y"))
    assert ((u, f) >> model.Rate(u)) == model.Signature((u, f), model.Rate("U"))
    assert (u >> f) == model.Signature((u,), f)
    assert ((f,) >> model.LocalLinearOperator(u, u)) \
        == model.Signature((f,), model.LocalLinearOperator("U", "U"))
    assert (() >> model.LocalLinearOperator(u, u)) \
        == model.Signature((), model.LocalLinearOperator("U", "U"))
    print("OK  >> signature sugar (single, tuple, empty inputs)")


def test_module_builder_and_decorator():
    mod = model.Module("euler_poisson_lorentz")
    u = mod.state_space("U", ("rho", "mx", "my"), roles={"rho": "Density"})
    f = mod.field_space("fields", ("phi", "grad_x", "grad_y"))
    params = mod.parameters(alpha=1.0, cs2=0.0)
    aux = mod.aux_fields(B_z="cell_scalar")
    assert params["alpha"].default == 1.0 and aux["B_z"].kind == "cell_scalar"
    assert "U" in mod.state_spaces() and "fields" in mod.field_spaces()

    # Builder mode: expr given -> registers now, returns the Operator.
    op = mod.operator(name="fields_from_state", signature=(u,) >> f,
                      kind="field_operator", expr="<ir>")
    assert isinstance(op, model.Operator) and op.kind == "field_operator"
    assert op.signature.output == f and op.body == "<ir>"

    # Decorator mode: no expr -> registers the decorated body, returns the function.
    @mod.operator(name="explicit_rhs", signature=(u, f) >> model.Rate(u),
                  kind="local_rate")
    def explicit_rhs(state, fields):
        return ("flux", "electric")

    @mod.operator(name="lorentz", signature=(f,) >> model.LocalLinearOperator(u, u),
                  kind="local_linear_operator")
    def lorentz(fields):
        return "L"

    assert explicit_rhs.__name__ == "explicit_rhs"  # decorator returns the function
    reg = mod.operator_registry()
    assert reg.names() == ["fields_from_state", "explicit_rhs", "lorentz"]
    assert reg.get("explicit_rhs").signature.output == model.Rate("U")
    assert reg.get("lorentz").body.__name__ == "lorentz"

    try:
        mod.operator(name="x", signature=(u,) >> f)  # missing kind
        raise AssertionError("expected ValueError without a kind")
    except ValueError:
        pass
    try:
        mod.operator(name="x", signature="nope", kind="field_operator", expr=1)
        raise AssertionError("expected TypeError for a non-Signature")
    except TypeError:
        pass
    print("OK  Module builder + decorator operators, parameters, aux fields")


def test_dsl_model_is_a_module_facade():
    m = dsl.Model("ep")
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    m.flux(x=[mx, mx, mx], y=[my, my, my])
    m.source_term("electric", [dsl.Const(0.0), rho * (-gx), rho * (-gy)])
    m.elliptic_rhs(rho - 1.0)
    mod = m.module
    assert isinstance(mod, model.Module)
    names = mod.operator_registry().names()
    assert "electric" in names and "fields_from_state" in names and "flux_default" in names
    assert "U" in mod.state_spaces() and "fields" in mod.field_spaces()
    print("OK  dsl.Model.module exposes the derived registry + spaces")


def _build_predictor(P, mdl):
    """A GENERIC predictor step: no physics names, only typed operator calls."""
    P.bind_operators(mdl)
    u = P.state("plasma")
    fields = P.call("fields_from_state", u)
    rate = P.call("explicit_rhs", u, fields)
    lin = P.call("lorentz", fields)
    rhs = P.linear_combine("rhs", u + P.dt * rate)
    ustar = P.solve_local_linear("ustar", operator=P.I - P.dt * lin, rhs=rhs, fields=fields)
    P.commit("plasma", ustar)


def _physics_model(name, gain):
    """Two of these differ in physics but share the operator signatures."""
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


def test_same_program_two_modules():
    ma = _physics_model("A", 1.0)
    mb = _physics_model("B", 2.0)  # same signatures, different physics
    pa = adctime.Program("pc")
    _build_predictor(pa, ma)
    src_a = pa.emit_cpp_program(model=ma)
    pb = adctime.Program("pc")
    _build_predictor(pb, mb)
    src_b = pb.emit_cpp_program(model=mb)
    assert "pops_install_program" in src_a and "pops_install_program" in src_b
    # The SAME generic function produced a valid, distinct program for each module
    # (the electric gain differs), proving reuse without mentioning any physics.
    assert src_a != src_b
    print("OK  one generic operator-first Program reused across two modules")


def main():
    test_signature_sugar()
    test_module_builder_and_decorator()
    test_dsl_model_is_a_module_facade()
    test_same_program_two_modules()
    print("OK  test_operator_module")


if __name__ == "__main__":
    main()
