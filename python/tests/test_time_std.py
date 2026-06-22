"""adc.time.std -- standard library of time-stepping macros that LOWER to the Program IR (ADC-407).

These are Python functions that BUILD adc.time.Program IR (not separate C++ steppers): Forward Euler,
SSPRK2, SSPRK3, RK4 and a Strang-splitting combinator. They reuse the merged Phase 2a builder ops and
the affine algebra over dt, so a scheme is expressed once, without any scheme-specific class (spec
acceptance: "RK4 is expressed without a special RK4 class"). This test exercises only IR CONSTRUCTION
(no codegen, no compilation): it asserts each macro produces the expected per-input coefficient
polynomials in dt on the committed state. Parity vs the old C++ steppers needs compile_problem (Phase 2c)
and is deferred.

Run with python3 (PYTHONPATH = built adc package).
"""
from adc import time as adctime


def _coeff(node, value):
    for v, c in zip(node.inputs, node.attrs["coeffs"], strict=True):
        if v is value:
            return c
    raise AssertionError("value %r not an input of %r" % (value, node))


def _committed(P, block):
    P.validate()
    node = P.commits()[block]
    assert node.op == "linear_combine", node.op
    states = [v for v in node.inputs if v.vtype == "state"]
    rhss = [v for v in node.inputs if v.vtype == "rhs"]
    return node, states, rhss


def _approx(d, power, val):
    return power in d and abs(d[power] - val) < 1e-15 and len(d) == 1


def test_forward_euler():
    P = adctime.Program("fe")
    adctime.std.forward_euler(P, "plasma")
    node, states, rhss = _committed(P, "plasma")
    assert len(states) == 1 and len(rhss) == 1
    assert _coeff(node, states[0]) == {0: 1.0}     # U
    assert _coeff(node, rhss[0]) == {1: 1.0}       # dt * R
    print("OK  forward_euler -> U + dt*R")


def test_ssprk2():
    P = adctime.Program("ssprk2")
    adctime.std.ssprk2(P, "plasma")
    node, states, rhss = _committed(P, "plasma")
    # U2 = 0.5*U0 + 0.5*U1 + 0.5*dt*k1
    assert len(states) == 2 and len(rhss) == 1
    for s in states:
        assert _approx(_coeff(node, s), 0, 0.5)
    assert _approx(_coeff(node, rhss[0]), 1, 0.5)
    print("OK  ssprk2 -> 0.5 U0 + 0.5 U1 + 0.5 dt k1")


def test_ssprk3():
    P = adctime.Program("ssprk3")
    adctime.std.ssprk3(P, "plasma")
    node, states, rhss = _committed(P, "plasma")
    # Shu-Osher final stage: U^{n+1} = 1/3 U0 + 2/3 U2 + 2/3 dt k2
    assert len(states) == 2 and len(rhss) == 1
    cs = sorted(_coeff(node, s)[0] for s in states)
    assert abs(cs[0] - 1.0 / 3.0) < 1e-15 and abs(cs[1] - 2.0 / 3.0) < 1e-15
    assert _approx(_coeff(node, rhss[0]), 1, 2.0 / 3.0)
    print("OK  ssprk3 -> 1/3 U0 + 2/3 U2 + 2/3 dt k2")


def test_rk4_no_special_class():
    P = adctime.Program("rk4")
    adctime.std.rk4(P, "plasma")
    node, states, rhss = _committed(P, "plasma")
    # Unp1 = U0 + dt/6 k1 + dt/3 k2 + dt/3 k3 + dt/6 k4
    assert len(states) == 1 and len(rhss) == 4
    assert _coeff(node, states[0]) == {0: 1.0}
    kcoeffs = sorted(round(_coeff(node, r)[1], 12) for r in rhss)
    assert kcoeffs == sorted([round(1 / 6, 12), round(1 / 3, 12), round(1 / 3, 12), round(1 / 6, 12)])
    print("OK  rk4 (no special RK4 class) -> U0 + dt(1/6 k1 + 1/3 k2 + 1/3 k3 + 1/6 k4)")


def test_strang_combinator():
    # Strang splitting H(dt/2); S(dt); H(dt/2) as IR-building callables. Here H and S are trivial
    # affine updates so we can check the macro chains three stages and commits the last.
    P = adctime.Program("strang")

    def half_flow(prog, U, frac):
        R = prog.rhs(state=U, fields=prog.solve_fields(U), flux=True, sources=["default"])
        return prog.linear_combine(None, U + (frac * prog.dt) * R)

    def source(prog, U, frac):
        S = prog.rhs(state=U, fields=None, flux=False, sources=["default"])
        return prog.linear_combine(None, U + (frac * prog.dt) * S)

    out = adctime.std.strang(P, "plasma", half_flow, source)
    P.validate()
    assert P.commits()["plasma"] is out and out.vtype == "state"
    # three linear_combine stages were built (two half flows + one source)
    n_lc = sum(1 for v in P._values if v.op == "linear_combine")
    assert n_lc == 3, n_lc
    print("OK  strang combinator chains H(dt/2); S(dt); H(dt/2)")


def main():
    test_forward_euler()
    test_ssprk2()
    test_ssprk3()
    test_rk4_no_special_class()
    test_strang_combinator()
    print("test_time_std : tout est vert")


if __name__ == "__main__":
    main()
