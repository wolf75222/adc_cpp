"""Spec 2 (S2-11 / ADC-447): a pure adc.model.Module compiles via the dsl codegen engine.

A Module authored directly -- typed spaces + operators with IR (dsl.Expr) bodies + eigenvalues --
is a self-contained, compilable model. ``Module.to_dsl`` lowers it to a dsl.Model (reusing the dsl
backend, not a second codegen), and ``compile_problem(model=module, time=P)`` accepts it. This test
validates the translation + the emitted .so source (codegen-text); the full Kokkos/AOT compile+run
is on ROMEO. Pure Python; skips if adc is not importable.
"""
import sys

try:
    from adc import dsl, model
    from adc import time as adctime
except Exception as exc:  # adc not importable here -> skip, never fake
    print("skip test_module_compile (adc unavailable: %s)" % exc)
    sys.exit(0)


def pure_module():
    mod = model.Module("euler_poisson_lorentz_operator_first")
    u = mod.state_space("U", ("rho", "mx", "my"),
                        roles={"rho": "density", "mx": "momentum_x", "my": "momentum_y"})
    fields = mod.field_space("fields", ("phi", "grad_x", "grad_y"))
    mod.aux_fields(B_z="cell_scalar")
    # Operator bodies are plain dsl.Expr over the state/field names (evaluated at codegen only).
    rho, mx, my = dsl.Var("rho", "cons"), dsl.Var("mx", "cons"), dsl.Var("my", "cons")
    gx, gy = dsl.Var("grad_x", "aux"), dsl.Var("grad_y", "aux")
    bz = dsl.Var("B_z", "aux")
    cs = dsl.sqrt(0.5)  # isothermal sound speed (cs2 = 0.5)
    mod.operator(name="fields_from_state", signature=(u,) >> fields,
                 kind="field_operator", expr=rho)
    mod.operator(name="flux", signature=(u,) >> model.Rate(u), kind="grid_operator",
                 expr={"x": [mx, mx * mx / rho + 0.5 * rho, mx * my / rho],
                       "y": [my, mx * my / rho, my * my / rho + 0.5 * rho]})
    mod.eigenvalues(x=[mx / rho - cs, mx / rho, mx / rho + cs],
                    y=[my / rho - cs, my / rho, my / rho + cs])
    mod.operator(name="electric", signature=(u, fields) >> model.Rate(u),
                 kind="local_source", expr=[dsl.Const(0.0), -rho * gx, -rho * gy])
    mod.operator(name="lorentz", signature=(fields,) >> model.LocalLinearOperator(u, u),
                 kind="local_linear_operator",
                 expr=[[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]])
    mod.rate_operator("explicit_rhs", flux=True, sources=["electric"])
    return mod


def test_module_lowers_to_dsl():
    mod = pure_module()
    m = mod.to_dsl()
    assert m._m.cons_names == ["rho", "mx", "my"]
    assert "electric" in m._m._source_terms
    assert "lorentz" in m._m._linear_sources
    assert m._m._elliptic is not None
    assert m._m._flux and m._m._eig
    assert "explicit_rhs" in m._m._rate_operators
    print("OK  Module.to_dsl maps every typed operator to its dsl method")


def test_pure_module_program_emits():
    mod = pure_module()
    P = adctime.Program("pc").bind_operators(mod)
    adctime.std.predictor_corrector_local_linear(
        P, "plasma", fields_operator="fields_from_state",
        explicit_rate_operator="explicit_rhs", implicit_operator="lorentz")
    # compile_problem(model=Module) lowers the Module internally; emit the .so source (no compile).
    src = P.emit_cpp_program(model=mod.to_dsl())
    assert "adc_install_program" in src
    # the GeneratedModule descriptor reflects the pure Module's operators
    assert "adc_module_operator_count() { return" in src
    for op in ("electric", "lorentz", "fields_from_state", "explicit_rhs"):
        assert '"%s"' % op in src, op
    print("OK  a pure operator-first Module + generic macro emits a combined .so source")


def test_module_requires_one_state_space():
    mod = model.Module("two_states")
    mod.state_space("U", ("rho",))
    mod.state_space("V", ("n",))
    try:
        mod.to_dsl()
        raise AssertionError("expected a single-StateSpace requirement error")
    except ValueError as exc:
        assert "exactly one StateSpace" in str(exc)
    print("OK  a Module to compile must declare exactly one StateSpace")


def main():
    test_module_lowers_to_dsl()
    test_pure_module_program_emits()
    test_module_requires_one_state_space()
    print("OK  test_module_compile")


if __name__ == "__main__":
    main()
