"""Spec 2 (ADC-446, criterion 24): install-time operator-requirement validation.

A compiled problem.so carries, per operator, the aux fields its body reads (the GeneratedModule
descriptor). System.install_program reads that descriptor and rejects, BEFORE installing the program,
a simulation that did not provide a required field -- here B_z, normally supplied by
set_magnetic_field -- with a spec-style message ("operator 'lorentz' requires aux field 'B_z', but
simulation did not provide it") instead of a cryptic failure mid-step. The negative and positive
cases both need a compiler + a visible Kokkos (POPS_KOKKOS_ROOT) to build the .so; the test prints a
skip notice and exits 0 otherwise (run it on ROMEO). cf. docs/sphinx/reference/operator-modules.md.
"""
import sys

try:
    import numpy as np

    import pops
    from pops import dsl
    from pops import time as adctime
except Exception as exc:  # noqa: BLE001
    print("skip test_install_requirement_validation (adc/numpy unavailable: %s)" % exc)
    sys.exit(0)

N = 16


def lorentz_model(name="adc446_model"):
    """An isothermal fluid whose Lorentz linear source reads the aux field B_z (a hard requirement)."""
    m = dsl.Model(name)
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    cs = dsl.sqrt(0.5)
    m.flux(x=[mx, mx * mx / rho + 0.5 * rho, mx * my / rho],
           y=[my, mx * my / rho, my * my / rho + 0.5 * rho])
    m.eigenvalues(x=[mx / rho - cs, mx / rho, mx / rho + cs],
                  y=[my / rho - cs, my / rho, my / rho + cs])
    m.primitive_vars(rho, mx, my)
    m.conservative_from([rho, mx, my])
    bz = m.aux("B_z")
    m.linear_source("lorentz", [[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]])
    m.elliptic_rhs(rho)
    m.rate_operator("explicit_rhs", flux=True)
    return m


def lie_program(name="adc446_prog"):
    P = adctime.Program(name)
    u = P.state("plasma")
    fields = P.solve_fields(u)
    r = P.rhs(state=u, fields=fields)
    P.commit("plasma", P.linear_combine("u1", u + P.dt * r))
    return P


def make_sim(block_model, with_bz):
    sim = pops.System(n=N, L=1.0, periodic=True)
    sim.add_equation("plasma", block_model.compile(backend="production"),
                     spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                     time=pops.Explicit(method="euler"))
    sim.set_poisson("charge_density", "geometric_mg")
    if with_bz:
        sim.set_magnetic_field(3.0 * np.ones(N * N))
    x = (np.arange(N) + 0.5) / N
    xx, yy = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
    sim.set_state("plasma", np.stack([rho, 0.4 * rho, -0.2 * rho]))
    return sim


def main():
    if not hasattr(pops.System(n=8, L=1.0, periodic=True), "install_program"):
        print("skip test_install_requirement_validation (_pops lacks install_program; rebuild _pops)")
        return 0
    m = lorentz_model()
    try:
        compiled = pops.compile_problem(model=m, time=lie_program())
    except RuntimeError as exc:
        print("skip test_install_requirement_validation (no Kokkos to build the .so: %s)"
              % str(exc)[:120])
        return 0

    # (1) Negative: a simulation WITHOUT set_magnetic_field must be rejected at install with the
    # spec-style message naming the operator and the missing aux field.
    sim_missing = make_sim(m, with_bz=False)
    try:
        sim_missing.install_program(compiled.so_path)
        print("MISMATCH: install accepted a simulation missing B_z")
        return 1
    except RuntimeError as exc:
        msg = str(exc)
        ok = "lorentz" in msg and "B_z" in msg and "did not provide" in msg
        print("OK  install rejects a missing required aux: %s" % msg if ok
              else "MISMATCH: unexpected error: %s" % msg)
        if not ok:
            return 1

    # (2) Positive: providing B_z (set_magnetic_field) lets the same program install cleanly.
    sim_ok = make_sim(m, with_bz=True)
    sim_ok.install_program(compiled.so_path)
    print("OK  install accepts the simulation once B_z is provided")
    return 0


if __name__ == "__main__":
    sys.exit(main())
