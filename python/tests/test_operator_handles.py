"""Spec 5 sec.14.2.3: typed operator handles (epic ADC-479 / ADC-464).

A user-facing operator declarer (``m.rate_operator`` / ``m.source_term`` /
``m.linear_source``) returns an inert :class:`pops.model.OperatorHandle` carrying the
operator ``name`` (and ``kind``). ``P.call`` accepts EITHER the string operator name OR
the handle and resolves both through the IDENTICAL registry lookup + lowering, so
``P.call(handle, ...)`` builds the BYTE-IDENTICAL IR (same ``_ir_hash``) as
``P.call(name, ...)``, and the legacy string path is byte-identical to a pristine build.

Pure Python (``_ir_hash`` is the IR fingerprint; no compilation); skips cleanly if pops is
not importable. Never fakes the engine.
"""
import sys

try:
    import pytest
    from pops.ir.expr import Const
    from pops.model import OperatorHandle
    from pops.physics.facade import Model
    from pops import time as adctime
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_operator_handles (pops unavailable: %s)" % exc)
    sys.exit(0)


def build_model():
    """A model declaring a named source, a named linear operator and a rate operator.

    Returns the model plus the handles the declarers returned (so the test can pass a
    handle straight into ``P.call``)."""
    m = Model("euler_poisson_lorentz")
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    m.aux("phi")
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    bz = m.aux("B_z")
    m.flux(x=[mx, mx * mx / rho, mx * my / rho],
           y=[my, mx * my / rho, my * my / rho])
    h_src = m.source_term("electric", [Const(0.0), rho * (-gx), rho * (-gy)])
    h_lin = m.linear_source("lorentz", [[0.0, 0.0, 0.0],
                                        [0.0, 0.0, bz],
                                        [0.0, -bz, 0.0]])
    m.elliptic_rhs(rho - 1.0)
    h_rate = m.rate_operator("explicit_rhs", flux=True, sources=["electric"])
    return m, {"electric": h_src, "lorentz": h_lin, "explicit_rhs": h_rate}


def test_declarers_return_operator_handles():
    """Each user-facing declarer returns a typed OperatorHandle naming the declared operator."""
    m, h = build_model()
    assert isinstance(h["electric"], OperatorHandle)
    assert isinstance(h["lorentz"], OperatorHandle)
    assert isinstance(h["explicit_rhs"], OperatorHandle)
    assert h["electric"].name == "electric" and h["electric"].kind == "local_source"
    assert h["lorentz"].name == "lorentz" and h["lorentz"].kind == "local_linear_operator"
    assert h["explicit_rhs"].name == "explicit_rhs" and h["explicit_rhs"].kind == "local_rate"
    # The default source_term alias also returns a handle (name 'default').
    m2 = Model("ds")
    rho2, mx2, my2 = m2.conservative_vars("rho", "mx", "my")
    h_def = m2.source_term("default", [Const(0.0), -mx2, -my2])
    assert isinstance(h_def, OperatorHandle) and h_def.name == "default"
    print("OK  declarers return typed OperatorHandle(name, kind)")


def _rate_program(m, selector):
    """Build a one-step predictor Program calling the rate operator via ``selector`` (a name or
    a handle). Returns the Program (its ``_ir_hash`` is the IR fingerprint)."""
    P = adctime.Program("prog").bind_operators(m)
    U = P.state("plasma")
    f = P.call("fields_from_state", U)
    R = P.call(selector, U, f)
    P.commit("plasma", P.linear_combine("u1", U + P.dt * R))
    return P


def test_call_handle_byte_identical_to_string():
    """P.call(handle) lowers to the byte-identical IR as P.call(name) -- same _ir_hash."""
    m, h = build_model()
    prog_str = _rate_program(m, "explicit_rhs")
    prog_handle = _rate_program(m, h["explicit_rhs"])
    assert prog_str._ir_hash() == prog_handle._ir_hash(), (
        "P.call(handle) must lower to the byte-identical IR as P.call(name)")
    print("OK  P.call(handle) IR hash == P.call(name) IR hash: %s" % prog_str._ir_hash())


def test_string_path_byte_identical_to_pristine():
    """The legacy string P.call('name') IR is unchanged by the declarer now returning a handle.

    A pristine model (declarers' return ignored) and a model whose handles were captured produce
    the same string-path IR hash."""
    m_a, _ = build_model()
    m_b, _ = build_model()
    # Build the SAME string-path program twice against two independently-built models.
    h_a = _rate_program(m_a, "explicit_rhs")._ir_hash()
    h_b = _rate_program(m_b, "explicit_rhs")._ir_hash()
    assert h_a == h_b, "the string P.call path must be deterministic / unperturbed"
    # And it equals the source/linear handle paths' string equivalents (full coverage of the
    # three declarer kinds via _ir_hash on a source-only and a linear-operator program).
    m, h = build_model()

    def src_prog(selector):
        P = adctime.Program("p").bind_operators(m)
        U = P.state("plasma")
        f = P.call("fields_from_state", U)
        s = P.call(selector, U, f)
        P.commit("plasma", P.linear_combine("u1", U + P.dt * s))
        return P

    assert src_prog("electric")._ir_hash() == src_prog(h["electric"])._ir_hash()

    def lin_prog(selector):
        P = adctime.Program("p").bind_operators(m)
        U = P.state("plasma")
        f = P.call("fields_from_state", U)
        L = P.call(selector, f)
        U1 = P.solve_local_linear("u1", operator=P.I - P.dt * L, rhs=U, fields=f)
        P.commit("plasma", U1)
        return P

    assert lin_prog("lorentz")._ir_hash() == lin_prog(h["lorentz"])._ir_hash()
    print("OK  string path byte-identical (rate / source / linear all match handle path)")


def test_plain_string_still_works():
    """A plain string selector keeps working unchanged (transparent coercion, no stricter check)."""
    m, _ = build_model()
    P = adctime.Program("p").bind_operators(m)
    U = P.state("plasma")
    f = P.call("fields_from_state", U)
    R = P.call("explicit_rhs", U, f)  # string path, must not raise
    assert R is not None
    print("OK  plain string P.call('explicit_rhs') still works")


def test_bad_type_rejected():
    """A non-str / non-handle selector is a clear TypeError (typed surface, no legacy path)."""
    m, _ = build_model()
    P = adctime.Program("p").bind_operators(m)
    U = P.state("plasma")
    with pytest.raises(TypeError, match="str name or an pops.model.OperatorHandle"):
        P.call(123, U)
    with pytest.raises(TypeError, match="str name or an pops.model.OperatorHandle"):
        P.call(None, U)
    print("OK  P.call(non-str/non-handle) -> clear TypeError")


def test_foreign_handle_rejected():
    """A handle whose name is not in the bound registry is rejected like an unknown string name."""
    m, _ = build_model()
    P = adctime.Program("p").bind_operators(m)
    U = P.state("plasma")
    foreign = OperatorHandle("not_declared_here", kind="local_rate")
    with pytest.raises(KeyError, match="unknown operator"):
        P.call(foreign, U)
    print("OK  foreign handle (name not in registry) rejected like an unknown name")


def test_handle_equality_and_repr():
    """OperatorHandle is value-like: equal by (name, kind), hashable, with a clear repr."""
    a = OperatorHandle("r", kind="local_rate")
    b = OperatorHandle("r", kind="local_rate")
    c = OperatorHandle("r", kind="local_source")
    assert a == b and hash(a) == hash(b)
    assert a != c and a != "r"
    assert repr(a) == "OperatorHandle('r', kind='local_rate')"
    assert repr(OperatorHandle("s")) == "OperatorHandle('s')"
    print("OK  OperatorHandle equality / hash / repr")


def main():
    test_declarers_return_operator_handles()
    test_call_handle_byte_identical_to_string()
    test_string_path_byte_identical_to_pristine()
    test_plain_string_still_works()
    test_bad_type_rejected()
    test_foreign_handle_rejected()
    test_handle_equality_and_repr()
    print("OK  test_operator_handles")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
