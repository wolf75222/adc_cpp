#!/usr/bin/env python3
"""Spec 4 (34): an adc_cases-style driver skeleton over the Spec 4 surface.

This shows how a scenario in ``adc_cases`` wires a Spec 4 model and time program onto a
``pops.System`` -- WITHOUT putting any named-scenario physics or initial condition in
``adc_cpp``. The reusable half (model authoring, time program, compile, install
sequencing) lives here; the scenario-specific half (grid choice, the actual initial
condition, diagnostics) is a clearly-marked stub that BELONGS IN adc_cases.

    model    = pops.physics.Model(...)           # reusable physics
    program  = pops.lib.time.forward_euler(...)  # reusable scheme (library macro)
    compiled = pops.compile_problem(model, program)
    sim.install(compiled, instances={...}, solvers={...})   # Spec section 22 entry point

The model build and program build are pure Python and always run. Compile + install need
a C++ compiler and a visible Kokkos and are guarded; the script exits 0 with a skip notice
otherwise.

Run::

    python3 examples/spec4/adc_cases_style_driver.py
"""
import sys

from pops.lib.time import forward_euler
from pops.math import ddt, div, grad, laplacian, sqrt
from pops.physics import Model
from pops.time import Program


def build_model():
    """Reusable physics: 2D isothermal Euler with an electrostatic (Poisson) source."""
    m = Model("euler_poisson")
    state = m.state(
        "U",
        components=["rho", "mx", "my"],
        roles={"rho": "density", "mx": "momentum_x", "my": "momentum_y"},
    )
    rho, mx, my = state
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    cs2 = m.param("cs2", 1.0)
    pressure = m.scalar("p", cs2 * rho)
    sound = m.scalar("c", sqrt(cs2))
    flux = m.flux(
        "F",
        on=state,
        x=[mx, mx * u + pressure, mx * v],
        y=[my, my * u, my * v + pressure],
        waves={"x": [u - sound, u, u + sound], "y": [v - sound, v, v + sound]},
    )
    phi = m.field("phi")
    m.solve_field(
        "fields_from_state",
        equation=(-laplacian(phi) == rho),
        outputs={"phi": phi, "grad_x": grad(phi).x, "grad_y": grad(phi).y},
        solver="geometric_mg",
    )
    e_field = m.vector_field("E", x=-grad(phi).x, y=-grad(phi).y)
    electric = m.source("default", on=state, value=[0.0 * rho, rho * e_field.x, rho * e_field.y])
    m.rate("explicit_rate", ddt(state) == -div(flux) + electric)
    m.check()
    return m


def build_program(module):
    """Reusable scheme: one Forward-Euler step from the time-scheme library.

    Spec 4 (s6 / s14) homes the ready schemes in ``pops.lib.time``: a scheme is a library
    macro that BUILDS ``pops.time.Program`` IR, so we import ``forward_euler`` from
    ``pops.lib.time`` and the Program (the time language) from ``pops.time``.
    """
    program = Program("forward_euler_driver")
    program.bind_operators(module)
    forward_euler(program, "plasma", sources=("default",), flux=True)
    return program


# --------------------------------------------------------------------------------------
# STUB -- belongs in adc_cases. The grid, the initial condition and the diagnostics are
# SCENARIO-specific and must NOT live in adc_cpp. A real adc_cases driver fills these in.
# --------------------------------------------------------------------------------------
def scenario_initial_condition(n):  # pragma: no cover -- scenario stub
    """Return the initial state array for the block. STUB: belongs in adc_cases.

    A real scenario builds an (n_components, n, n) array here (e.g. a perturbed density
    with zero momentum). We raise so it is obvious this half is intentionally absent.
    """
    raise NotImplementedError(
        "initial condition is scenario-specific and belongs in adc_cases, not adc_cpp"
    )


def main():
    model = build_model()
    module = model.module
    program = build_program(module)
    print("model:", module.name)
    print("operators:", module.list_operators())
    print("program nodes:", len(program._values))

    import pops

    try:
        compiled = pops.compile_problem(model=model, time=program)
    except RuntimeError as exc:  # no compiler / no Kokkos visible / compile failed
        print("skip compile+install (compile_problem could not build the .so: %s)" % str(exc)[:160])
        return 0

    # Install skeleton (Spec section 22 single entry point). The "initial" array is the
    # scenario stub; in adc_cpp we only demonstrate the WIRING, so we guard the actual run.
    n = 48
    sim = pops.System(n=n, L=1.0, periodic=True)
    try:
        initial = scenario_initial_condition(n)
    except NotImplementedError as exc:
        print("skip install/run (scenario IC is an adc_cases stub: %s)" % exc)
        print("problem.so:", compiled.so_path)
        print("OK: model + program compiled; install/run deferred to a real adc_cases driver.")
        return 0

    sim.install(
        compiled,
        instances={
            "plasma": {
                "initial": initial,
                "spatial": pops.FiniteVolume(limiter="none", riemann="rusanov"),
                "time": pops.Explicit(method="euler"),
            }
        },
        solvers={"phi": "geometric_mg"},
    )
    sim.step(2e-3)
    print("OK: stepped one Forward-Euler step via the Spec 4 install path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
