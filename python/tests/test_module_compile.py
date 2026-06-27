"""Spec 2 (S2-11 / ADC-447): a pure pops.model.Module compiles via the dsl codegen engine.

A Module authored directly -- typed spaces + operators with IR (Expr) bodies + eigenvalues --
is a self-contained, compilable model. ``Module.to_dsl`` lowers it to a Model (reusing the dsl
backend, not a second codegen), and ``compile_problem(model=module, time=P)`` accepts it. This test
validates the translation + the emitted .so source (codegen-text); the full Kokkos/AOT compile+run
is on ROMEO. Pure Python; skips if pops is not importable.
"""
import sys

try:
    from pops import model
    from pops.ir.expr import Const, Expr, Var
    from pops.ir.ops import sqrt
    from pops.physics.facade import Model
    from pops import time as adctime
    import pops.lib.time as libtime  # ready schemes live in pops.lib.time (Spec 4)
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_module_compile (pops unavailable: %s)" % exc)
    sys.exit(0)


def pure_module():
    mod = model.Module("euler_poisson_lorentz_operator_first")
    u = mod.state_space("U", ("rho", "mx", "my"),
                        roles={"rho": "density", "mx": "momentum_x", "my": "momentum_y"})
    fields = mod.field_space("fields", ("phi", "grad_x", "grad_y"))
    mod.aux_fields(B_z="cell_scalar")
    # Operator bodies are plain Expr over the state/field names (evaluated at codegen only).
    rho, mx, my = Var("rho", "cons"), Var("mx", "cons"), Var("my", "cons")
    gx, gy = Var("grad_x", "aux"), Var("grad_y", "aux")
    bz = Var("B_z", "aux")
    cs = sqrt(0.5)  # isothermal sound speed (cs2 = 0.5)
    mod.operator(name="fields_from_state", signature=(u,) >> fields,
                 kind="field_operator", expr=rho)
    mod.operator(name="flux", signature=(u,) >> model.Rate(u), kind="grid_operator",
                 expr={"x": [mx, mx * mx / rho + 0.5 * rho, mx * my / rho],
                       "y": [my, mx * my / rho, my * my / rho + 0.5 * rho]})
    mod.eigenvalues(x=[mx / rho - cs, mx / rho, mx / rho + cs],
                    y=[my / rho - cs, my / rho, my / rho + cs])
    mod.operator(name="electric", signature=(u, fields) >> model.Rate(u),
                 kind="local_source", expr=[Const(0.0), -rho * gx, -rho * gy])
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
    # The brick codegen (to_primitive / to_conservative) must succeed: a pure Module declares no
    # primitives, so to_dsl supplies the identity primitive-state layout (regression: without it
    # emit_cpp_brick raises "call set_primitive_state(...) first").
    brick = m._m.emit_cpp_brick(name="ModuleCheck")
    assert "struct ModuleCheck" in brick
    print("OK  Module.to_dsl maps every operator + yields a compilable brick")


def test_pure_module_program_emits():
    mod = pure_module()
    P = adctime.Program("pc").bind_operators(mod)
    libtime.predictor_corrector_local_linear(
        P, "plasma", fields_operator="fields_from_state",
        explicit_rate_operator="explicit_rhs", implicit_operator="lorentz")
    # compile_problem(model=Module) lowers the Module internally; emit the .so source (no compile).
    src = P.emit_cpp_program(model=mod.to_dsl())
    assert "pops_install_program" in src
    # the GeneratedModule descriptor reflects the pure Module's operators
    assert "pops_module_operator_count() { return" in src
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


def test_decorator_body_rejected():
    mod = model.Module("deco")
    u = mod.state_space("U", ("rho",))
    fields = mod.field_space("fields", ("phi",))

    @mod.operator(name="electric", signature=(u, fields) >> model.Rate(u), kind="local_source")
    def electric(state, flds):  # a callable body, not an IR expression
        return None

    try:
        mod.to_dsl()
        raise AssertionError("expected a no-IR-body error for a decorator-authored operator")
    except ValueError as exc:
        assert "no IR body" in str(exc)
    print("OK  a decorator/callable operator body is rejected at compile")


def test_multiple_field_operators_rejected():
    mod = model.Module("twofields")
    u = mod.state_space("U", ("rho",))
    f1 = mod.field_space("fields", ("phi",))
    rho = Var("rho", "cons")
    mod.operator(name="fields_from_state", signature=(u,) >> f1, kind="field_operator", expr=rho)
    mod.operator(name="psi", signature=(u,) >> f1, kind="field_operator", expr=rho)
    try:
        mod.to_dsl()
        raise AssertionError("expected a single-field_operator error")
    except ValueError as exc:
        assert "one field_operator" in str(exc)
    print("OK  multiple field_operators are rejected (single elliptic solve)")


def test_explicit_roles_honored():
    # A non-canonical layout: the StateSpace's explicit roles must reach the dsl model, not be lost.
    mod = model.Module("custom")
    u = mod.state_space("U", ("n", "px", "py"),
                        roles={"n": "density", "px": "momentum_x", "py": "momentum_y"})
    n, px, py = Var("n", "cons"), Var("px", "cons"), Var("py", "cons")
    mod.operator(name="flux", signature=(u,) >> model.Rate(u), kind="grid_operator",
                 expr={"x": [px, px * px / n, px * py / n], "y": [py, px * py / n, py * py / n]})
    m = mod.to_dsl()
    assert m._m.cons_roles == ["Density", "MomentumX", "MomentumY"], m._m.cons_roles
    print("OK  explicit StateSpace roles are mapped through to the dsl model")


def main():
    test_explicit_roles_honored()
    test_module_lowers_to_dsl()
    test_pure_module_program_emits()
    test_module_requires_one_state_space()
    test_decorator_body_rejected()
    test_multiple_field_operators_rejected()
    print("OK  test_module_compile")


if __name__ == "__main__":
    main()
