"""Spec 3 native Riemann capabilities (ADC-456): board m.riemann selects a native C++
solver and validates the model's capabilities for it (criterion 10), and board roles are
canonicalized so the role-derived HLLC/Roe hook generation finds them.

Pure-Python: validation + role canonicalization + descriptors. The hook C++ itself is
emitted by the dsl backend (enable_hllc/enable_roe/roe_dissipation); generating hooks from
ARBITRARY board formulas and the end-to-end compile remain ADC-456 follow-ups.
"""
import pytest

physics = pytest.importorskip("pops.physics")
lib = pytest.importorskip("pops.lib")


def _euler(with_pressure=True, with_roles=True):
    from pops.math import sqrt
    m = physics.Model("euler")
    roles = ({"rho": "density", "mx": "momentum_x", "my": "momentum_y", "E": "energy"}
             if with_roles else None)
    U = m.state("U", components=["rho", "mx", "my", "E"], roles=roles)
    rho, mx, my, E = U
    m.primitive("u", mx / rho)
    m.primitive("v", my / rho)
    g = m.param("gamma", 1.4)
    if with_pressure:
        m.primitive("p", (g - 1.0) * (E - 0.5 * (mx * mx + my * my) / rho))
    m.scalar("c", sqrt(g * (g - 1.0)))  # a sound-speed-ish scalar (value irrelevant here)
    return m, U


def test_board_roles_canonicalize_to_dsl_roles():
    m, _ = _euler()
    roles = physics._roles_for(m._dsl._m)
    assert roles == ["Density", "MomentumX", "MomentumY", "Energy"]


def test_hllc_accepts_role_tagged_model_with_pressure():
    m, _ = _euler(with_pressure=True, with_roles=True)
    m.riemann("hllc")                      # must not raise
    assert m._dsl._m._hllc is True         # enable_hllc was driven -> hooks will be generated


def test_hllc_rejects_model_without_pressure():
    m, _ = _euler(with_pressure=False, with_roles=True)
    with pytest.raises(ValueError, match="requires model capability 'pressure'"):
        m.riemann("hllc")


def test_hllc_rejects_model_without_fluid_roles():
    # a moment system: non-canonical names (q0/q1/q2) -> no fluid roles, even with pressure.
    m = physics.Model("moments")
    U = m.state("U", components=["q0", "q1", "q2"])
    m.primitive("p", U[0])
    assert physics._roles_for(m._dsl._m) == ["Custom", "Custom", "Custom"]
    with pytest.raises(ValueError, match="requires model capability 'hllc_star_state'"):
        m.riemann("hllc")


def test_roe_rejects_model_without_pressure():
    m, _ = _euler(with_pressure=False, with_roles=True)
    with pytest.raises(ValueError, match="requires model capability 'pressure'"):
        m.riemann("roe")


def test_rusanov_needs_no_pressure_or_roles():
    m, _ = _euler(with_pressure=False, with_roles=False)
    m.riemann("rusanov")                   # only max_wave_speed -> always OK
    assert m._riemann == "rusanov"


def test_hll_requires_wave_speeds():
    m, _ = _euler(with_pressure=False, with_roles=False)
    with pytest.raises(ValueError, match="requires model capability 'wave_speeds'"):
        m.riemann("hll")
    # declaring the flux waves provides them:
    from pops.math import sqrt
    m2, U = _euler(with_pressure=True, with_roles=True)
    rho, mx, my, E = U
    u, v = mx / rho, my / rho
    c = sqrt(1.4)
    m2.flux("F", on=U, x=[mx, mx * u, mx * v, E], y=[my, my * u, my * v, E],
            waves={"x": [u - c, u, u, u + c], "y": [v - c, v, v, v + c]})
    m2.riemann("hll")                      # waves declared -> OK


def test_hll_accepts_jacobian_derived_wave_speeds():
    # wave speeds can come from the flux Jacobian (_ws_jacobian), not only _eig/_wave_speeds;
    # HLL must accept that source too (mirrors the dsl max_wave_speed gate).
    m, _ = _euler(with_pressure=True, with_roles=True)
    m._dsl._m._ws_jacobian = {"x": [], "y": []}   # marker the jacobian path sets
    m.riemann("hll")                               # must not raise
    assert m._riemann == "hll"


def test_finite_volume_rate_validates_riemann():
    from pops.math import sqrt
    m, U = _euler(with_pressure=True, with_roles=True)
    rho, mx, my, E = U
    u, v = mx / rho, my / rho
    c = sqrt(1.4)
    m.flux("F", on=U, x=[mx, mx * u, mx * v, E], y=[my, my * u, my * v, E],
           waves={"x": [u - c, u, u, u + c], "y": [v - c, v, v, v + c]})
    m.finite_volume_rate("explicit_rate", flux="F",
                         riemann=lib.riemann.HLLC(),
                         reconstruction=lib.reconstruction.WENO5Z())
    assert "explicit_rate" in m.list_operators()
    assert m._dsl._m._hllc is True

    m2, U2 = _euler(with_pressure=False, with_roles=True)  # no pressure -> HLLC rejected
    with pytest.raises(ValueError, match="requires model capability 'pressure'"):
        m2.finite_volume_rate("r", riemann="hllc")


def test_lib_riemann_capability_hook_descriptors_compute_nothing():
    for d in (lib.riemann.speeds.einfeldt(),
              lib.riemann.hllc.contact_speed.euler(),
              lib.riemann.hllc.star_state.euler()):
        assert d.brick_type == "macro"     # a hook selector, not a Python computation
        assert not hasattr(d, "eval")
    assert lib.riemann.hllc.contact_speed.euler().scheme == "euler"
    assert lib.riemann.speeds.einfeldt().scheme == "einfeldt"
