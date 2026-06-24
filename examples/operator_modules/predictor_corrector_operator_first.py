#!/usr/bin/env python3
"""Operator-first predictor-corrector: the model-free way to write the spec-1 Example 5.

Spec 2 (operator-first, epic ADC-436). Same physics as
``examples/time_programs/predictor_corrector_poisson_lorentz.py`` -- a 2D isothermal fluid
``U = (rho, mx, my)`` with an electric source ``-rho grad(phi)`` and a Lorentz local linear
operator -- but the time algorithm is built WITHOUT naming any physics. The model exposes three
TYPED operators:

  * ``fields_from_state`` : ``U -> Fields``            (the Poisson solve, from m.elliptic_rhs)
  * ``explicit_rhs``      : ``(U, Fields) -> Rate(U)`` (a named rate operator, m.rate_operator)
  * ``lorentz``           : ``Fields -> LocalLinearOperator(U, U)`` (m.linear_source)

and the program is the GENERIC macro ``adc.time.std.predictor_corrector_local_linear`` keyed on
those three operator names. The macro mentions no flux / source / poisson / lorentz: the SAME macro
runs against any Module that provides these signatures. The whole step still runs C++-side.

The result is checked against an OFFLINE replay of the identical stages built from the runtime
primitives (a default-source twin for ``-div F + electric``) plus the analytic Lorentz solve / apply,
exactly as the spec-1 example does -- demonstrating that the operator-first macro reproduces the
hand-written predictor-corrector.

Run::

    python examples/operator_modules/predictor_corrector_operator_first.py

Requires a compiler + a visible Kokkos (``ADC_KOKKOS_ROOT``); prints a skip notice and exits 0
otherwise. cf. docs/sphinx/reference/operator-modules.md and reference/time-program.md.
"""
import sys

try:
    import numpy as np

    import adc
    from adc import dsl
    from adc import time as adctime
except Exception as exc:  # noqa: BLE001
    print("skip predictor_corrector_operator_first (adc/numpy unavailable: %s)" % exc)
    sys.exit(0)

N = 16
BZ = 3.0
DT = 0.02


def _base_block(m):
    """Shared isothermal 2D fluid block (flux + primitives + eigenvalues + Poisson + B_z aux)."""
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    cs2 = m.param("cs2", 0.5)
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    p = m.primitive("p", cs2 * rho)
    m.primitive_vars(rho=rho, u=u, v=v, p=p)
    m.conservative_from([rho, rho * u, rho * v])
    m.flux(x=[mx, mx * u + p, my * u], y=[my, mx * v, my * v + p])
    cs = dsl.sqrt(cs2)
    m.eigenvalues(x=[u - cs, u, u + cs], y=[v - cs, v, v + cs])
    m.elliptic_rhs(rho)
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    bz = m.aux("B_z")
    return rho, mx, my, gx, gy, bz


def operator_module(name="opfirst_named"):
    """The operator-first model: the elliptic solve, the electric force and the Lorentz rotation are
    exposed as TYPED operators (fields_from_state / explicit_rhs / lorentz). ``m.rate_operator`` names
    the composite ``-div F + electric`` rate the Program calls by name."""
    m = dsl.Model(name)
    rho, mx, my, gx, gy, bz = _base_block(m)
    m.source_term("electric", [0.0, -rho * gx, -rho * gy])
    m.linear_source("lorentz", [[0.0, 0.0, 0.0],
                                [0.0, 0.0, bz],
                                [0.0, -bz, 0.0]])
    m.rate_operator("explicit_rhs", flux=True, sources=["electric"])
    return m


def default_source_model(name="opfirst_default"):
    """Same physics with the electric force as the DEFAULT source: eval_rhs returns -div F + electric
    directly, used to build the offline reference."""
    m = dsl.Model(name)
    rho, mx, my, gx, gy, bz = _base_block(m)
    m.source([0.0, -rho * gx, -rho * gy])
    m.linear_source("lorentz", [[0.0, 0.0, 0.0],
                                [0.0, 0.0, bz],
                                [0.0, -bz, 0.0]])
    return m


def operator_first_program(model, name="predictor_corrector_operator_first"):
    """The GENERIC model-free macro, keyed on the three typed operator names. No physics here."""
    P = adctime.Program(name).bind_operators(model)
    adctime.std.predictor_corrector_local_linear(
        P, "plasma",
        fields_operator="fields_from_state",
        explicit_rate_operator="explicit_rhs",
        implicit_operator="lorentz")
    return P


def make_sim(model):
    sim = adc.System(n=N, L=1.0, periodic=True)
    compiled = model.compile(backend="production")
    sim.add_equation("plasma", compiled,
                     spatial=adc.FiniteVolume(limiter="none", riemann="rusanov"),
                     time=adc.Explicit(method="euler"))
    sim.set_poisson("charge_density", "geometric_mg")
    sim.set_magnetic_field(BZ * np.ones(N * N))
    x = (np.arange(N) + 0.5) / N
    xx, yy = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
    u0 = np.stack([rho, 0.4 * rho, -0.2 * rho])
    sim.set_state("plasma", u0)
    return sim, u0


def offline_rhs_with_electric(ref, u):
    ref.set_state("plasma", u)
    ref.solve_fields()
    return np.array(ref.eval_rhs("plasma"))


def analytic_lorentz_solve(u, a):
    k = a * BZ
    den = 1.0 + k * k
    rho, mx, my = u[0], u[1], u[2]
    return np.stack([rho, (mx + k * my) / den, (-k * mx + my) / den])


def analytic_lorentz_apply(u):
    rho, mx, my = u[0], u[1], u[2]
    return np.stack([np.zeros_like(rho), BZ * my, -BZ * mx])


def main():
    if not hasattr(adc.System(n=8, L=1.0, periodic=True), "install_program"):
        print("skip predictor_corrector_operator_first (_adc lacks install_program; rebuild _adc)")
        return 0
    try:
        model = operator_module("opfirst_prog")
        compiled = adc.compile_problem(model=model, time=operator_first_program(model))
        sim, u0 = make_sim(operator_module("opfirst_block"))
        ref = make_sim(default_source_model("opfirst_ref_block"))[0]
    except RuntimeError as exc:
        print("skip predictor_corrector_operator_first (compile_problem could not build the .so: %s)"
              % str(exc)[:160])
        return 0

    sim.install_program(compiled.so_path)
    sim.step(DT)
    u_pc = np.array(sim.get_state("plasma"))

    r_n = offline_rhs_with_electric(ref, u0)
    u_star = analytic_lorentz_solve(u0 + DT * r_n, DT)
    r_star = offline_rhs_with_electric(ref, u_star)
    c_star = analytic_lorentz_apply(u_star)
    q = u0 + 0.5 * DT * r_n + 0.5 * DT * r_star + 0.5 * DT * c_star
    u_ref = analytic_lorentz_solve(q, 0.5 * DT)

    err = float(np.abs(u_pc - u_ref).max())
    print("operator-first predictor-corrector vs offline staged reference: max|d| = %.2e" % err)
    ok = err < 1e-10
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
