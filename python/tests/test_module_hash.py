"""Spec 2 (S2-7): Module.module_hash covers the ModuleSpec (spaces + typed operators).

module_hash folds the spaces, parameters, aux and -- per operator -- name, kind, signature,
capabilities, requirements and a body identity. It is deterministic for an identical module and
invalidated by an operator body / signature / capability / space change, so a compiled artifact
keyed on it is rebuilt when the operator spec changes. The dsl codegen sensitivity to a formula
change stays with the existing dsl.Model._model_hash; module_hash adds the operator-spec layer.
Pure Python; skips if adc is not importable.
"""
import sys

try:
    from adc import model
    from adc import dsl
except Exception as exc:  # adc not importable here -> skip, never fake
    print("skip test_module_hash (adc unavailable: %s)" % exc)
    sys.exit(0)


def test_deterministic():
    def build():
        mod = model.Module("m")
        u = mod.state_space("U", ("rho", "mx", "my"), roles={"rho": "Density"})
        f = mod.field_space("fields", ("phi", "grad_x", "grad_y"))
        mod.parameters(alpha=1.0)
        mod.aux_fields(B_z="cell_scalar")
        mod.operator(name="fields_from_state", signature=(u,) >> f,
                     kind="field_operator", expr="POISSON")
        return mod

    assert build().module_hash() == build().module_hash()
    print("OK  module_hash is deterministic for an identical module")


def test_signature_change_invalidates():
    m1 = model.Module("m")
    u1 = m1.state_space("U", ("rho", "mx"))
    m1.field_space("fields", ("phi",))
    m1.operator(name="op", signature=(u1,) >> model.Rate(u1), kind="local_rate", expr="E")
    m2 = model.Module("m")
    u2 = m2.state_space("U", ("rho", "mx"))
    f2 = m2.field_space("fields", ("phi",))
    m2.operator(name="op", signature=(u2, f2) >> model.Rate(u2), kind="local_rate", expr="E")
    assert m1.module_hash() != m2.module_hash()
    print("OK  a signature change invalidates module_hash")


def test_expr_body_change_invalidates():
    def build(expr):
        mod = model.Module("m")
        u = mod.state_space("U", ("rho",))
        f = mod.field_space("fields", ("phi",))
        mod.operator(name="op", signature=(u, f) >> model.Rate(u),
                     kind="local_rate", expr=expr)
        return mod

    assert build("BODY_A").module_hash() != build("BODY_B").module_hash()
    print("OK  an operator body change invalidates module_hash")


def test_callable_body_change_invalidates():
    def mod_a():
        mod = model.Module("m")
        u = mod.state_space("U", ("rho",))
        f = mod.field_space("fields", ("phi",))

        @mod.operator(name="op", signature=(u, f) >> model.Rate(u), kind="local_rate")
        def op(state, fields):
            return "alpha"

        return mod

    def mod_b():
        mod = model.Module("m")
        u = mod.state_space("U", ("rho",))
        f = mod.field_space("fields", ("phi",))

        @mod.operator(name="op", signature=(u, f) >> model.Rate(u), kind="local_rate")
        def op(state, fields):
            return "beta"

        return mod

    assert mod_a().module_hash() != mod_b().module_hash()
    print("OK  a decorated-body source change invalidates module_hash")


def test_capability_and_space_change_invalidate():
    u = model.StateSpace("U", ("rho", "mx"))
    base = model.Module("m")
    base.state_space("U", ("rho", "mx"))
    base.field_space("fields", ("phi",))
    base.operator(name="op", signature=(u,) >> model.Rate(u), kind="local_rate",
                  capabilities={"produces_rate": True}, expr="E")
    other_caps = model.Module("m")
    other_caps.state_space("U", ("rho", "mx"))
    other_caps.field_space("fields", ("phi",))
    other_caps.operator(name="op", signature=(u,) >> model.Rate(u), kind="local_rate",
                        capabilities={"produces_rate": False}, expr="E")
    assert base.module_hash() != other_caps.module_hash()

    other_space = model.Module("m")
    other_space.state_space("U", ("rho", "mx", "my"))  # one more component
    other_space.field_space("fields", ("phi",))
    other_space.operator(name="op", signature=(u,) >> model.Rate(u), kind="local_rate",
                         capabilities={"produces_rate": True}, expr="E")
    assert base.module_hash() != other_space.module_hash()
    print("OK  a capability or a state-space change invalidates module_hash")


def test_requirements_change_invalidates():
    u = model.StateSpace("U", ("rho",))
    f = model.FieldSpace("fields", ("phi",))

    def build(reqs):
        mod = model.Module("m")
        mod.state_space("U", ("rho",))
        mod.field_space("fields", ("phi",))
        mod.operator(name="op", signature=(f,) >> model.LocalLinearOperator(u, u),
                     kind="local_linear_operator", requirements=reqs, expr="E")
        return mod

    assert build({"aux": ["B_z"]}).module_hash() != build({"aux": ["E_x"]}).module_hash()
    print("OK  a requirements change invalidates module_hash")


def test_eigenvalues_change_invalidates():
    def build(speed):
        mod = model.Module("m")
        mod.state_space("U", ("rho", "mx"))
        mod.field_space("fields", ("phi",))
        rho, mx = dsl.Var("rho", "cons"), dsl.Var("mx", "cons")
        mod.eigenvalues(x=[mx / rho - speed, mx / rho + speed],
                        y=[mx / rho - speed, mx / rho + speed])
        return mod

    assert build(dsl.sqrt(0.5)).module_hash() != build(dsl.sqrt(0.7)).module_hash()
    print("OK  an eigenvalue (wave-speed) change invalidates module_hash")


def test_dsl_backed_module_hashes():
    m = dsl.Model("ep")
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    m.flux(x=[mx, mx, mx], y=[my, my, my])
    m.source_term("electric", [dsl.Const(0.0), rho, rho])
    h = m.module.module_hash()
    assert isinstance(h, str) and len(h) == 64
    print("OK  a dsl-backed Module produces a module_hash")


def main():
    test_deterministic()
    test_signature_change_invalidates()
    test_expr_body_change_invalidates()
    test_callable_body_change_invalidates()
    test_capability_and_space_change_invalidate()
    test_requirements_change_invalidates()
    test_eigenvalues_change_invalidates()
    test_dsl_backed_module_hashes()
    print("OK  test_module_hash")


if __name__ == "__main__":
    main()
