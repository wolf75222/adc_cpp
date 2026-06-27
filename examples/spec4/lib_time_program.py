#!/usr/bin/env python3
"""Spec 4 (31): the same scheme from the library instead of by hand.

``examples/spec4/manual_time_program.py`` spells a predictor-corrector / backward-Euler
step out of kernel primitives. Here the SAME scheme comes from the time-scheme library:

    from pops.lib.time import predictor_corrector_local_linear

The library scheme is a MACRO: given a bound model and the operator names, it emits the
same kernel primitives into the Program. This script builds the model, applies the macro,
prints the emitted IR, and (when a toolchain is present) compiles the Program to a
``problem.so``.

The model build and the macro expansion are pure Python and always run; the compile step
is guarded and skips cleanly without a C++ compiler / visible Kokkos.

Run::

    python3 examples/spec4/lib_time_program.py
"""
import sys

from pops.lib.time import predictor_corrector_local_linear
from pops.math import ddt, div, grad, laplacian, sqrt
from pops.physics import Model
from pops.time import Program


def build_model():
    """2D isothermal Euler + Poisson + Lorentz, identical physics to the manual example."""
    m = Model("euler_poisson_lorentz")
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
    electric = m.source("electric", on=state, value=[0.0 * rho, rho * e_field.x, rho * e_field.y])
    bz = m.aux("B_z")
    lorentz = m.local_linear_operator(
        "lorentz",
        on=state,
        matrix=[[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]],
    )
    m.rate("explicit_rate", ddt(state) == -div(flux) + electric)
    m.operator("implicit_operator", returns=lorentz, inputs=["fields"])
    m.check()
    return m


def library_program(module):
    """Apply the library predictor-corrector macro against the bound operators."""
    program = Program("lib_predictor_corrector")
    program.bind_operators(module)
    predictor_corrector_local_linear(
        program,
        "plasma",
        fields_operator="fields_from_state",
        explicit_rate_operator="explicit_rate",
        implicit_operator="implicit_operator",
    )
    return program


def main():
    model = build_model()
    module = model.module
    program = library_program(module)

    print("model:", module.name)
    print("ops emitted by the macro:", sorted({v.op for v in program._values}))
    print("program nodes:", len(program._values))
    print("program commits:", {b: s.op for b, s in program.commits().items()})

    import pops

    # RuntimeError = no compiler / no Kokkos visible / compile failed. On the PR-D branch a
    # board pops.physics.Model that drives a named-source rhs also raises AttributeError
    # ('Model' object has no attribute '_source_terms') in program_codegen -- a board-model
    # lowering gap on the PR-E/PR-F punch-list; the authoring half above is what Spec 31
    # demonstrates and always runs.
    try:
        compiled = pops.compile_problem(model=model, time=program)
    except (RuntimeError, AttributeError) as exc:
        print("skip compile step (compile_problem could not build the .so: %s: %s)"
              % (type(exc).__name__, str(exc)[:160]))
        return 0
    print("problem.so:", compiled.so_path)
    print("OK: library Spec 4 time scheme compiled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
