"""Spec 3 arbitrary-formula Riemann hooks (ADC-456, section 11, criterion 17/18).

``m.riemann("hllc", pressure=<board formula>, ...)`` codegen's THAT formula into the
``pressure(U)`` capability hook of the generated brick, overriding the role-derived default.
Capability-hook descriptors (``pops.lib.riemann.hllc.contact_speed.euler()``) and ``None`` keep
the canonical role-derived hook. A formula that references a quantity the model cannot provide
still raises the clear capability error.

EMIT-LEVEL: these assert the EMITTED C++ string + the Python wiring. The brick ``.so`` compile and
run need POPS_KOKKOS_ROOT (ROMEO) and are out of scope here.
"""
import re

import pytest

physics = pytest.importorskip("pops.physics")
lib = pytest.importorskip("pops.lib")


def _euler(custom_pressure=None):
    """A complete, emittable board Euler model. @p custom_pressure: a board formula passed to
    ``m.riemann(pressure=...)`` (ADC-456), or None for the role-derived default."""
    from pops.math import sqrt
    m = physics.Model("euler")
    U = m.state("U", components=["rho", "mx", "my", "E"],
                roles={"rho": "density", "mx": "momentum_x",
                       "my": "momentum_y", "E": "energy"})
    rho, mx, my, E = U
    g = m.param("gamma", 1.4)
    p = m.primitive("p", (g - 1.0) * (E - 0.5 * (mx * mx + my * my) / rho))
    d = m.dsl
    u, v = mx / rho, my / rho
    d.conservative_from([rho, rho * u, rho * v, p / (g - 1.0) + 0.5 * rho * (u * u + v * v)])
    d._m.set_primitive_state(rho, u, v, p)
    d.flux([mx, mx * u + p, mx * v, (E + p) * u],
           [my, my * u, my * v + p, (E + p) * v])
    c = sqrt(g * p / rho)
    d.eigenvalues([u - c, u, u, u + c], [v - c, v, v, v + c])
    kw = {}
    if custom_pressure is not None:
        kw["pressure"] = custom_pressure(m, U)
    m.riemann("hllc",
              contact_speed=lib.riemann.hllc.contact_speed.euler(),
              star_state=lib.riemann.hllc.star_state.euler(), **kw)
    return m


def _pressure_body(cpp):
    """The body of the emitted ``pressure(const State& U)`` hook (text between its braces)."""
    mt = re.search(r"pops::Real pressure\(const State& U\) const \{(.*?)\n  \}", cpp, re.S)
    assert mt is not None, "no pressure(U) hook emitted"
    return mt.group(1)


def _custom_cs2_pressure(m, U):
    """A custom cs2-based pressure formula, distinct from the role-derived primitive 'p'."""
    rho, mx, my, E = U
    gamma = 1.4
    cs2 = m.scalar("cs2", gamma * (gamma - 1.0)
                   * (E / rho - 0.5 * (mx * mx + my * my) / (rho * rho)))
    return rho * cs2 / gamma


def test_pressure_formula_overrides_role_derived_hook():
    # role-derived default emits the primitive 'p' formula verbatim.
    default = _euler().dsl._m.emit_cpp_brick(name="EulerGen")
    body0 = _pressure_body(default)
    assert "return p;" in body0
    assert "cs2" not in body0

    # the arbitrary cs2-based formula is codegen'd into the hook body instead.
    override = _euler(custom_pressure=_custom_cs2_pressure).dsl._m.emit_cpp_brick(name="EulerGen")
    body1 = _pressure_body(override)
    assert "ARBITRARY board formula" in override        # the override marker comment
    assert "const pops::Real cs2 = " in body1            # the custom scalar appears as a local
    assert "return ((rho * cs2) / 1.4);" in body1       # the formula-derived return
    assert "return p;" not in body1                     # distinct from the role-derived default
    assert body0 != body1


def test_pressure_override_changes_the_module_hash():
    # a distinct hook body must key a distinct compiled-brick cache entry...
    m0 = _euler()
    m1 = _euler(custom_pressure=_custom_cs2_pressure)
    assert m0.dsl._m._model_hash() != m1.dsl._m._model_hash()
    # ...while a role-derived-only model records NO override (historical hash bit-identity).
    assert m0.dsl._m._riemann_hook_forms == {}


def test_descriptor_hooks_keep_the_role_derived_default():
    # contact_speed / star_state given as canonical descriptors (.euler()) are NOT formula
    # overrides: the role-derived HLLC capability is emitted unchanged.
    m = _euler()
    assert m.dsl._m._riemann_hook_forms == {}            # no Expr recorded from the descriptors
    cpp = m.dsl._m.emit_cpp_brick(name="EulerGen")
    assert "pops::Real contact_speed(const State& UL, const State& UR" in cpp
    assert "State hllc_star_state(const State& U" in cpp


def test_pressure_formula_referencing_a_missing_capability_raises():
    # a pressure formula that reads an undeclared aux (B_z) is a missing capability -> clear error
    # at codegen (the pressure(U) hook has no Aux parameter).
    from pops.dsl import Var
    m = physics.Model("euler")
    U = m.state("U", components=["rho", "mx", "my", "E"],
                roles={"rho": "density", "mx": "momentum_x",
                       "my": "momentum_y", "E": "energy"})
    rho, mx, my, E = U
    g = m.param("gamma", 1.4)
    p = m.primitive("p", (g - 1.0) * (E - 0.5 * (mx * mx + my * my) / rho))
    d = m.dsl
    u, v = mx / rho, my / rho
    d.conservative_from([rho, rho * u, rho * v, E])
    d._m.set_primitive_state(rho, u, v, p)
    d.flux([mx, mx, mx, mx], [my, my, my, my])
    d.eigenvalues([u, u, u, u], [v, v, v, v])
    m.riemann("hllc", pressure=rho + Var("B_z", "aux"))   # references an undeclared capability
    with pytest.raises(ValueError, match="undeclared quantity"):
        d._m.emit_cpp_brick(name="EulerGen")


def test_arbitrary_formula_for_a_two_state_hook_is_rejected():
    # contact_speed / star_state span two states (UL/UR/pL/pR/sL/sR); an arbitrary single-Expr
    # override is ill-defined, so they accept only a capability-hook descriptor.
    m = physics.Model("euler")
    U = m.state("U", components=["rho", "mx", "my", "E"],
                roles={"rho": "density", "mx": "momentum_x",
                       "my": "momentum_y", "E": "energy"})
    rho, _, _, _ = U
    m.primitive("p", rho)
    with pytest.raises(NotImplementedError, match="contact_speed"):
        m.riemann("hllc", contact_speed=rho * 2.0)


def test_missing_pressure_capability_still_rejected():
    # the pre-existing capability gate is unchanged: HLLC on a model without pressure raises.
    m = physics.Model("euler")
    m.state("U", components=["rho", "mx", "my", "E"],
            roles={"rho": "density", "mx": "momentum_x",
                   "my": "momentum_y", "E": "energy"})
    with pytest.raises(ValueError, match="requires model capability 'pressure'"):
        m.riemann("hllc")
