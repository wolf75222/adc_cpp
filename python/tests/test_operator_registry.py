"""Spec 2 (S2-1): the internal, typed OperatorRegistry derived from a dsl model.

Pure-Python: builds a model with the PDE shortcuts and asserts that flux / source /
source_term / linear_source / elliptic_field / projection lower into typed operators
with the right kinds and signatures, that the typed view does NOT perturb the model
hash, and that the pops.model type system (StateSpace, FieldSpace, Rate, LocalLinear
Operator, Signature, OperatorRegistry) behaves as specified. No compilation, no _pops
numerics; skips cleanly if the pops package is not importable.
"""
import sys

try:
    from pops import dsl, model
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_operator_registry (pops unavailable: %s)" % exc)
    sys.exit(0)


def build_model():
    """A small electrostatic-magnetized fluid exercising every operator kind."""
    m = dsl.Model("euler_poisson_lorentz")
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    # Auxiliary surface (read order fixes the FieldSpace components).
    m.aux("phi")
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    bz = m.aux("B_z")
    # Default flux F(U) -> grid_operator.
    m.flux(x=[mx, mx * mx / rho, mx * my / rho],
           y=[my, mx * my / rho, my * my / rho])
    # Named source reading the solved field gradients -> local_source (needs Fields).
    m.source_term("electric", [dsl.Const(0.0), rho * (-gx), rho * (-gy)])
    # Default source independent of the fields -> local_source (state only).
    m.source_term("default", [dsl.Const(0.0), dsl.Const(0.0), -rho * dsl.Const(0.1)])
    # Lorentz rotation, coefficients in B_z only -> local_linear_operator (Fields).
    m.linear_source("lorentz", [[0.0, 0.0, 0.0],
                                [0.0, 0.0, bz],
                                [0.0, -bz, 0.0]])
    # Default elliptic field U -> phi -> field_operator (fields_from_state).
    m.elliptic_rhs(rho - 1.0)
    # A second, NAMED elliptic field -> field_operator.
    m.elliptic_field("psi", rhs=mx, aux=["psi_x", "psi_y"])
    # Pointwise positivity projection -> projection.
    m.projection([dsl.abs_(rho), mx, my])
    return m


def test_spaces():
    m = build_model()
    state = m.state_space()
    assert isinstance(state, model.StateSpace)
    assert state.name == "U"
    assert state.components == ("rho", "mx", "my")
    assert state.roles["rho"] == "Density"
    assert state.roles["mx"] == "MomentumX"
    fields = m.field_space()
    assert isinstance(fields, model.FieldSpace)
    assert fields.components == ("phi", "grad_x", "grad_y", "B_z")
    # Rate identity is by base name: Rate("U") == Rate(state space U).
    assert model.Rate("U") == model.Rate(state)
    assert model.Rate("V") != model.Rate("U")
    # LocalLinearOperator identity is by (domain, range) name.
    assert model.LocalLinearOperator("U", "U") == model.LocalLinearOperator(state, state)
    assert model.LocalLinearOperator("U", "U") != model.LocalLinearOperator("U", "V")
    print("OK  spaces: StateSpace / FieldSpace / Rate / LocalLinearOperator")


def test_registry_signatures():
    m = build_model()
    reg = m.operator_registry()
    state = m.state_space()
    fields = m.field_space()

    flux = reg.get("flux_default")
    assert flux.kind == "grid_operator"
    assert flux.signature.inputs == (state,)
    assert flux.signature.output == model.Rate("U")
    assert flux.capabilities["requires_ghosts"] == 1

    electric = reg.get("electric")
    assert electric.kind == "local_source"
    assert electric.signature.inputs == (state, fields)  # reads grad -> Fields input
    assert electric.signature.output == model.Rate(state)
    assert electric.capabilities["requires_fields"] is True
    assert "grad_x" in electric.requirements["aux"]
    assert "grad_y" in electric.requirements["aux"]

    src_default = reg.get("source_default")
    assert src_default.kind == "local_source"
    assert src_default.signature.inputs == (state,)  # state only, no Fields
    assert src_default.capabilities["requires_fields"] is False

    lorentz = reg.get("lorentz")
    assert lorentz.kind == "local_linear_operator"
    assert lorentz.signature.inputs == (fields,)
    assert lorentz.signature.output == model.LocalLinearOperator("U", "U")
    assert lorentz.capabilities["linear"] is True
    assert lorentz.capabilities["solve_i_minus_a"] is True
    assert lorentz.requirements["aux"] == ["B_z"]

    fields_op = reg.get("fields_from_state")
    assert fields_op.kind == "field_operator"
    assert fields_op.signature.inputs == (state,)
    assert isinstance(fields_op.signature.output, model.FieldSpace)
    assert fields_op.requirements["elliptic_operator"] == "poisson"

    psi = reg.get("psi")
    assert psi.kind == "field_operator"
    assert psi.signature.output.components == ("psi_x", "psi_y")

    projection = reg.get("projection")
    assert projection.kind == "projection"
    assert projection.signature.inputs == (state,)
    assert projection.signature.output == state
    print("OK  registry: every PDE shortcut lowered to a typed operator")


def test_registry_ids_and_errors():
    m = build_model()
    reg = m.operator_registry()
    names = reg.names()
    assert len(reg) == len(names)
    assert reg.id_of(names[0]) == 0
    assert reg.by_id(0).name == names[0]
    assert "lorentz" in reg and "nope" not in reg
    # ids are stable across rebuilds (registration order is deterministic).
    assert m.operator_registry().names() == names

    try:
        reg.get("does_not_exist")
        raise AssertionError("expected KeyError for an unknown operator")
    except KeyError as exc:
        assert "unknown operator" in str(exc)

    state = m.state_space()
    reg2 = model.OperatorRegistry()
    op = model.Operator("a", "local_source", model.Signature([state], model.Rate(state)))
    reg2.register(op)
    try:
        reg2.register(op)
        raise AssertionError("expected ValueError on duplicate registration")
    except ValueError:
        pass
    try:
        model.Operator("x", "not_a_kind", model.Signature([state], state))
        raise AssertionError("expected ValueError on an unknown operator kind")
    except ValueError:
        pass
    print("OK  registry ids stable; unknown/duplicate/bad-kind raise clearly")


def test_view_is_non_mutating():
    m = build_model()
    before = m._model_hash()
    m.operator_registry()
    m.state_space()
    m.field_space()
    assert m._model_hash() == before, "the typed view must not perturb the model hash"
    # The HyperbolicModel underneath exposes the same API as the facade.
    hm = m._m
    assert hm.operator_registry().names() == m.operator_registry().names()
    print("OK  typed view is non-mutating; facade and HyperbolicModel agree")


def main():
    test_spaces()
    test_registry_signatures()
    test_registry_ids_and_errors()
    test_view_is_non_mutating()
    print("OK  test_operator_registry")


if __name__ == "__main__":
    main()
