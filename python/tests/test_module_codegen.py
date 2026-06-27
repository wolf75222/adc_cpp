"""Spec 2 (S2-6): the GeneratedModule metadata block emitted into the problem.so source.

A combined model+program .so carries GeneratedProgram (pops_install_program, the step) AND a
GeneratedModule descriptor: extern "C" pops_module_* accessors exposing the typed operator registry
by integer OperatorId. The descriptor is read once at install (introspection + requirement
validation, module_metadata.hpp); it must NOT appear in the step body, so operators stay inlined and
there is no string lookup in a hot kernel. Pure-Python codegen-text check; skips if pops is absent.
"""
import sys

try:
    from pops.ir.expr import Const
    from pops.physics.facade import Model
    from pops import time as adctime
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_module_codegen (pops unavailable: %s)" % exc)
    sys.exit(0)


def _model():
    m = Model("ep")
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    bz = m.aux("B_z")
    m.flux(x=[mx, mx * mx / rho, mx * my / rho], y=[my, mx * my / rho, my * my / rho])
    m.source_term("electric", [Const(0.0), -rho * gx, -rho * gy])
    m.linear_source("lorentz", [[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]])
    m.elliptic_rhs(rho - 1.0)
    m.rate_operator("explicit_rhs", flux=True, sources=["electric"])
    return m


def test_metadata_block_emitted():
    m = _model()
    P = adctime.Program("pc").bind_operators(m)
    adctime.std.predictor_corrector_local_linear(
        P, "plasma", fields_operator="fields_from_state",
        explicit_rate_operator="explicit_rhs", implicit_operator="lorentz")
    src = P.emit_cpp_program(model=m)
    # GeneratedModule descriptor + GeneratedProgram coexist in the one .so.
    assert "pops_install_program" in src
    assert "pops_module_operator_count() { return" in src
    assert "pops_module_operator_kind(int i)" in src
    assert "pops_module_operator_signature(int i)" in src
    assert "pops_module_operator_requirements(int i)" in src
    assert "pops_module_state_space_name(int i)" in src
    assert "pops_module_field_space_name(int i)" in src
    # The count equals the registry size (flux_default, electric, lorentz, fields_from_state,
    # explicit_rhs) and every operator name + its kind is emitted.
    reg = m.operator_registry()
    assert "pops_module_operator_count() { return %d; }" % len(reg) in src
    for op in ("flux_default", "electric", "lorentz", "fields_from_state", "explicit_rhs"):
        assert '"%s"' % op in src, "operator %r missing from the module metadata" % op
    for kind in ("grid_operator", "local_source", "local_linear_operator", "field_operator",
                 "local_rate"):
        assert '"%s"' % kind in src, "operator kind %r missing from the module metadata" % kind
    assert '"U"' in src and '"fields"' in src
    print("OK  GeneratedModule metadata block emitted alongside the program")


def test_metadata_not_in_step_body():
    m = _model()
    P = adctime.Program("pc").bind_operators(m)
    adctime.std.predictor_corrector_local_linear(
        P, "plasma", fields_operator="fields_from_state",
        explicit_rate_operator="explicit_rhs", implicit_operator="lorentz")
    src = P.emit_cpp_program(model=m)
    body = src.split("pops_install_program", 1)[1]
    assert "pops_module_" not in body, \
        "the GeneratedModule metadata must not be referenced in the step body (no hot-path lookup)"
    print("OK  module metadata is install-time only (no hot-path string lookup)")


def test_no_model_empty_module():
    P = adctime.Program("fe")
    u = P.state("plasma")
    P.commit("plasma", P.linear_combine("u1", u + P.dt * P.rhs(state=u, fields=P.solve_fields(u))))
    src = P.emit_cpp_program(model=None)
    assert "pops_module_operator_count() { return 0; }" in src
    print("OK  model=None emits an empty GeneratedModule (count 0)")


def main():
    test_metadata_block_emitted()
    test_metadata_not_in_step_body()
    test_no_model_empty_module()
    print("OK  test_module_codegen")


if __name__ == "__main__":
    main()
