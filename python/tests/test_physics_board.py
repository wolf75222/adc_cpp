"""Spec 3 board-like physics DSL (adc.physics.Model + adc.math).

These tests exercise the LOWERING of a blackboard-style model to the Spec 2
operator-first IR (adc.model.Module) and to the adc.dsl codegen engine. They are
pure-Python: only adc.physics / adc.math / adc.model / adc.dsl are needed; no
compiled time-program run, so they pass without a freshly built _adc beyond what
``import adc`` requires.
"""
import pytest

from adc import model as _model

physics = pytest.importorskip("adc.physics")
amath = pytest.importorskip("adc.math")


def _euler_poisson_lorentz():
    """The canonical Spec 3 board model: 2D isothermal Euler + Poisson + Lorentz."""
    from adc.math import sqrt, grad, laplacian, div, ddt

    m = physics.Model("euler_poisson_lorentz")

    U = m.state("U", components=["rho", "mx", "my"],
                roles={"rho": "density", "mx": "momentum_x", "my": "momentum_y"})
    rho, mx, my = U

    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)

    cs2 = m.param("cs2", 1.0)
    alpha = m.param("alpha", 1.0)
    rho_ref = m.param("rho_ref", 1.0)

    p = m.scalar("p", cs2 * rho)
    c = m.scalar("c", sqrt(cs2))

    F = m.flux("F", on=U,
               x=[mx, mx * u + p, mx * v],
               y=[my, my * u, my * v + p],
               waves={"x": [u - c, u, u + c], "y": [v - c, v, v + c]})

    phi = m.field("phi")
    m.solve_field(
        "fields_from_state",
        equation=(-laplacian(phi) == alpha * (rho - rho_ref)),
        outputs={"phi": phi, "grad_x": grad(phi).x, "grad_y": grad(phi).y},
        solver="geometric_mg",
    )

    E = m.vector_field("E", x=-grad(phi).x, y=-grad(phi).y)

    A_E_U = m.source("electric", on=U, value=[0.0 * rho, rho * E.x, rho * E.y])

    Bz = m.aux("B_z")
    C_B = m.local_linear_operator(
        "lorentz", on=U,
        matrix=[[0.0, 0.0, 0.0],
                [0.0, 0.0, Bz],
                [0.0, -Bz, 0.0]])

    m.rate("explicit_rate", ddt(U) == -div(F) + A_E_U)
    m.operator("implicit_operator", C_B)
    return m


def test_state_lowers_to_state_space():
    m = physics.Model("euler")
    m.state("U", components=["rho", "mx", "my"], roles={"rho": "density"})
    mod = m.module
    assert isinstance(mod, _model.Module)
    st = mod.state_spaces()["U"]
    assert st.components == ("rho", "mx", "my")
    # board roles are canonicalized to the dsl roles (density -> Density) so the native
    # Riemann capability lookup recognizes them (ADC-456).
    assert st.roles.get("rho") == "Density"


def test_state_is_unpackable_into_components():
    m = physics.Model("euler")
    U = m.state("U", components=["rho", "mx", "my"])
    rho, mx, my = U
    # components are usable as expression operands (dsl Var-like)
    expr = mx / rho
    assert expr is not None


def test_board_model_lowers_to_operator_first_ir():
    m = _euler_poisson_lorentz()
    mod = m.module
    assert isinstance(mod, _model.Module)
    # state + field spaces
    assert mod.state_spaces()["U"].components == ("rho", "mx", "my")
    assert "fields" in mod.field_spaces()
    # the operators the board declared are present in the typed registry
    ops = set(mod.list_operators())
    assert "explicit_rate" in ops          # local_rate (flux + electric source)
    assert "electric" in ops               # local_source
    assert "implicit_operator" in ops      # local_linear_operator (registered via m.operator)


def test_explicit_rate_is_a_local_rate_operator():
    m = _euler_poisson_lorentz()
    sig = m.module.operator_signature("explicit_rate")
    op = m.module.operator_registry().get("explicit_rate")
    assert op.kind == "local_rate"
    # signature output is the tangent (Rate) of the state U
    assert sig.output == _model.Rate("U")


def test_implicit_operator_is_a_local_linear_operator():
    m = _euler_poisson_lorentz()
    op = m.module.operator_registry().get("implicit_operator")
    assert op.kind == "local_linear_operator"
    assert op.signature.output == _model.LocalLinearOperator("U", "U")


def test_local_linear_operator_object_is_not_callable():
    # Spec 3 amendment: m.local_linear_operator builds a MATH object, not a callable
    # operator; calling it directly is a clear error pointing at m.operator(...).
    m = physics.Model("plasma")
    U = m.state("U", components=["rho", "mx", "my"])
    bz = m.aux("B_z")
    c_b = m.local_linear_operator("C(B)", on=U,
                                  matrix=[[0.0, 0.0, 0.0],
                                          [0.0, 0.0, bz],
                                          [0.0, -bz, 0.0]])
    with pytest.raises(TypeError, match="is not a callable operator"):
        c_b(object())
    # registering it yields a callable operator
    impl = m.operator("implicit_operator", returns=c_b, inputs=["fields"])
    assert "implicit_operator" in m.module.list_operators()
    assert callable(impl)


def test_rate_and_operator_return_callables_usable_in_a_program():
    # Spec 3 amendment: m.rate / m.operator return callable operators so a board program
    # can write explicit_rate(U_n, fields_n) and get the same IR as P.call(...).
    from adc.math import sqrt, grad, div, laplacian, ddt
    from adc.time import Program
    m = physics.Model("ep")
    U = m.state("U", components=["rho", "mx", "my"])
    rho, mx, my = U
    u, v = m.primitive("u", mx / rho), m.primitive("v", my / rho)
    cs2 = m.param("cs2", 1.0)
    p, c = m.scalar("p", cs2 * rho), m.scalar("c", sqrt(cs2))
    flux = m.flux("F", on=U, x=[mx, mx * u + p, mx * v], y=[my, my * u, my * v + p],
                  waves={"x": [u - c, u, u + c], "y": [v - c, v, v + c]})
    phi = m.field("phi")
    m.solve_field("fields_from_state", equation=(-laplacian(phi) == rho),
                  outputs={"phi": phi, "grad_x": grad(phi).x, "grad_y": grad(phi).y},
                  solver="geometric_mg")
    e_field = m.vector_field("E", x=-grad(phi).x, y=-grad(phi).y)
    a_src = m.source("electric", on=U, value=[0.0 * rho, rho * e_field.x, rho * e_field.y])
    bz = m.aux("B_z")
    c_b = m.local_linear_operator("C(B)", on=U,
                                  matrix=[[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]])
    explicit_rate = m.rate("explicit_rate", ddt(U) == -div(flux) + a_src)
    implicit_operator = m.operator("implicit_operator", returns=c_b, inputs=["fields"])
    assert callable(explicit_rate) and callable(implicit_operator)

    P = Program("board_calls")
    U_n = P.state("plasma")
    f_n = P.solve_fields("f_n", U_n)
    R = explicit_rate(U_n, f_n)         # -> P.call("explicit_rate", U_n, f_n)
    L = implicit_operator(f_n)          # -> P.call("implicit_operator", f_n)
    assert R.vtype == "rhs"
    assert L.vtype == "operator"


def test_board_model_check_passes():
    m = _euler_poisson_lorentz()
    m.check()  # must not raise: all referenced vars are declared


def test_board_module_is_consumable_by_operator_first_program():
    # Spec 2 retention: the board model lowers to a real operator-first Module that the
    # explicit P.bind_operators / P.call layer drives unchanged (board never replaces it).
    from adc.time import Program
    m = _euler_poisson_lorentz()
    P = Program("operator_first")
    P.bind_operators(m.module)
    U = P.state("plasma")
    fields = P.call("fields_from_state", U)            # field_operator -> solve_fields
    R = P.call("explicit_rate", U, fields)             # local_rate (U, fields) -> Rate(U)
    assert fields.vtype == "fields"
    assert R.vtype == "rhs"
    assert "explicit_rate" in m.module.list_operators()
