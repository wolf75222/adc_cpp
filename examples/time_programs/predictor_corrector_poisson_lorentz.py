#!/usr/bin/env python3
"""Predictor-corrector Poisson/Lorentz step as a compiled time Program (epic ADC-399 / ADC-403).

This is the spec's Example 5: a 2D isothermal fluid U = (rho, mx, my) driven by a NAMED electric
source ``m.source_term("electric", [0, -rho*grad_x, -rho*grad_y])`` and a NAMED Lorentz local linear
source ``m.linear_source("lorentz", [[0,0,0],[0,0,Bz],[0,-Bz,0]])``. The predictor-corrector Program
chains: solve_fields; rhs(sources=["electric"]); an implicit Lorentz local solve (I - dt*L); a second
solve_fields + rhs; a Lorentz apply; the corrector combine; a half-step implicit Lorentz solve
(I - 0.5*dt*L); commit. The whole step runs C++-side -- no numerical stage re-enters Python.

It is checked against an OFFLINE replay of the EXACT same stages built from the runtime primitives
(set_state + solve_fields + eval_rhs for ``-div F + electric``, from a second model that folds the
SAME physics as its DEFAULT source) plus the analytic Lorentz solve / apply. BOTH RHS evaluations
re-solve the elliptic fields from their OWN stage state (solve_fields(state) lowers to
ctx.solve_fields_from_state, ADC-409), so the predictor stage reads grad(U_star). Mirrors
python/tests/test_predictor_corrector.py, which matches to ~2.22e-16.

Run::

    python examples/time_programs/predictor_corrector_poisson_lorentz.py

Requires a compiler + a visible Kokkos (``POPS_KOKKOS_ROOT``); prints a skip notice and exits 0
otherwise. cf. docs/sphinx/reference/time-program.md.
"""
import sys

try:
    import numpy as np

    import pops
    from pops.ir.ops import sqrt
    from pops.physics.facade import Model
    from pops import time as adctime
except Exception as exc:  # noqa: BLE001
    print("skip predictor_corrector_poisson_lorentz (pops/numpy unavailable: %s)" % exc)
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
    cs = sqrt(cs2)
    m.eigenvalues(x=[u - cs, u, u + cs], y=[v - cs, v, v + cs])
    m.elliptic_rhs(rho)  # Poisson rhs f = rho (so solve_fields populates a non-trivial grad)
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    bz = m.aux("B_z")
    return rho, mx, my, gx, gy, bz


def named_source_model(name="pc_named"):
    """Default source EMPTY (NoSource); the electric force is a NAMED source_term and the Lorentz
    operator a NAMED linear_source (both opt-in). This is the model the Program drives."""
    m = Model(name)
    rho, mx, my, gx, gy, bz = _base_block(m)
    m.source_term("electric", [0.0, -rho * gx, -rho * gy])
    m.linear_source("lorentz", [[0.0, 0.0, 0.0],
                                [0.0, 0.0, bz],
                                [0.0, -bz, 0.0]])
    return m


def default_source_model(name="pc_default"):
    """Same physics, but the electric force is the model's DEFAULT source (m.source): eval_rhs then
    returns -div F + electric directly -- used to build the offline reference."""
    m = Model(name)
    rho, mx, my, gx, gy, bz = _base_block(m)
    m.source([0.0, -rho * gx, -rho * gy])
    m.linear_source("lorentz", [[0.0, 0.0, 0.0],
                                [0.0, 0.0, bz],
                                [0.0, -bz, 0.0]])
    return m


def predictor_corrector_program(name="predictor_corrector_poisson_lorentz"):
    """The spec Example 5 program: two electric-source RHS evaluations, two implicit Lorentz local
    solves I -/+ a*dt*L, one Lorentz apply, the corrector combine."""
    P = adctime.Program(name)
    dt = P.dt
    U_n = P.state("plasma")
    f_n = P.solve_fields("fields_n", U_n)
    R_n = P.rhs(name="R_n", state=U_n, fields=f_n, flux=True, sources=["electric"])
    U_star_rhs = P.linear_combine("U_star_rhs", U_n + dt * R_n)
    U_star = P.solve_local_linear(name="U_star", operator=P.I - dt * P.linear_source("lorentz"),
                                  rhs=U_star_rhs, fields=f_n)
    f_star = P.solve_fields("fields_star", U_star)
    R_star = P.rhs(name="R_star", state=U_star, fields=f_star, flux=True, sources=["electric"])
    C_star = P.apply(operator=P.linear_source("lorentz"), state=U_star, fields=f_star, name="C_star")
    Q = P.linear_combine("Q", U_n + 0.5 * dt * R_n + 0.5 * dt * R_star + 0.5 * dt * C_star)
    U_np1 = P.solve_local_linear(name="U_np1", operator=P.I - 0.5 * dt * P.linear_source("lorentz"),
                                 rhs=Q, fields=f_star)
    P.commit("plasma", U_np1)
    return P


def make_sim(model):
    """A System carrying ONE block (the given DSL model, production backend) + shared Poisson + B_z.
    Returns (sim, U0) with U0 the initial conservative state (n_vars, N, N)."""
    sim = pops.System(n=N, L=1.0, periodic=True)
    compiled = model.compile(backend="production")
    sim.add_equation("plasma", compiled,
                     spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                     time=pops.Explicit(method="euler"))
    sim.set_poisson("charge_density", "geometric_mg")
    sim.set_magnetic_field(BZ * np.ones(N * N))  # constant B_z over the grid
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    U0 = np.stack([rho, 0.4 * rho, -0.2 * rho])
    sim.set_state("plasma", U0)
    return sim, U0


def offline_rhs_with_electric(ref, U):
    """The semi-discrete RHS -div F + electric at state U, elliptic fields RE-SOLVED from U. ``ref``
    carries the DEFAULT-source model, so eval_rhs(plasma) already returns -div F + electric."""
    ref.set_state("plasma", U)
    ref.solve_fields()
    return np.array(ref.eval_rhs("plasma"))


def analytic_lorentz_solve(U, a):
    """(I - a*L) U' = U with L = [[0,0,0],[0,0,B],[0,-B,0]]: rho unchanged, (mx, my) rotated.
    k = a*B, den = 1 + k^2, mx' = (mx + k*my)/den, my' = (-k*mx + my)/den."""
    k = a * BZ
    den = 1.0 + k * k
    rho, mx, my = U[0], U[1], U[2]
    return np.stack([rho, (mx + k * my) / den, (-k * mx + my) / den])


def analytic_lorentz_apply(U):
    """L U with L = [[0,0,0],[0,0,B],[0,-B,0]]: row 0 = 0, row 1 = B*my, row 2 = -B*mx."""
    rho, mx, my = U[0], U[1], U[2]
    return np.stack([np.zeros_like(rho), BZ * my, -BZ * mx])


def main():
    if not hasattr(pops.System(n=8, L=1.0, periodic=True), "install_program"):
        print("skip predictor_corrector_poisson_lorentz (_pops lacks install_program; rebuild _pops)")
        return 0
    try:
        compiled = pops.compile_problem(model=named_source_model("pc_prog"),
                                       time=predictor_corrector_program())
        sim, U0 = make_sim(named_source_model("pc_block"))
        ref = make_sim(default_source_model("pc_ref_block"))[0]
    except RuntimeError as exc:
        print("skip predictor_corrector_poisson_lorentz (compile_problem could not build the .so: %s)"
              % str(exc)[:160])
        return 0

    sim.install_program(compiled.so_path)
    sim.step(DT)
    U_pc = np.array(sim.get_state("plasma"))

    # Offline replay of the EXACT same stages with the default-source reference model.
    R_n = offline_rhs_with_electric(ref, U0)            # R_n = -div F(U_n) + electric(U_n; grad U_n)
    U_star_rhs = U0 + DT * R_n
    U_star = analytic_lorentz_solve(U_star_rhs, DT)     # (I - dt*L) U_star = U_star_rhs
    R_star = offline_rhs_with_electric(ref, U_star)     # -div F(U_star) + electric(U_star; grad U_star)
    C_star = analytic_lorentz_apply(U_star)             # C_star = L U_star
    Q = U0 + 0.5 * DT * R_n + 0.5 * DT * R_star + 0.5 * DT * C_star
    U_ref = analytic_lorentz_solve(Q, 0.5 * DT)         # (I - 0.5*dt*L) U_np1 = Q

    err = float(np.abs(U_pc - U_ref).max())
    print("compiled predictor-corrector Program vs offline staged reference: max|d| = %.2e" % err)
    ok = err < 1e-10
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
