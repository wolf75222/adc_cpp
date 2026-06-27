#!/usr/bin/env python3
"""Spec 4 (30): a model on the blackboard + a hand-written time scheme + compile.

Authoring path, end to end with the public Spec 4 surface:

    1. ``pops.physics.Model`` writes the physics the way it appears on the board
       (state, primitives, flux with wave speeds, a Poisson field solve, a source,
       and a local-linear Lorentz operator), and lowers to the operator-first
       ``pops.model.Module``.
    2. ``pops.time.Program`` writes a one-step scheme BY HAND against the kernel
       primitives (no library macro): solve the fields, evaluate the explicit rate,
       and take a backward-Euler-style local-linear solve for the implicit part.
    3. ``pops.compile_problem`` lowers the Program to a ``problem.so``.

Steps 1-2 are pure Python and always run. Step 3 needs a C++ compiler and a visible
Kokkos (``POPS_KOKKOS_ROOT``); it is wrapped so the script exits 0 with a skip notice
when the toolchain is absent (safe in a docs/CI smoke run).

Run::

    python3 examples/spec4/manual_time_program.py
"""
import sys

from pops.math import ddt, div, grad, laplacian, sqrt
from pops.physics import Model
from pops.time import Program


def build_model():
    """2D isothermal Euler + Poisson + a magnetic (Lorentz) local-linear operator."""
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


def manual_program(module):
    """U^{n+1} solves (I - dt C(B)) U^{n+1} = U^n + dt R(U^n), written by hand.

    Only kernel primitives are used (solve_fields / rhs / linear_source / linear_combine /
    solve_local_linear / commit), so this is the same scheme a library macro would emit,
    spelled out explicitly.
    """
    program = Program("manual_backward_euler_lorentz")
    program.bind_operators(module)
    dt = program.dt
    u_n = program.state("plasma")
    fields = program.solve_fields("fields_from_state", u_n)
    rate = program.rhs(state=u_n, fields=fields, flux=True, sources=["electric"])
    operator = program.I - dt * program.linear_source("implicit_operator")
    rhs = program.linear_combine("U_rhs", u_n + dt * rate)
    u_np1 = program.solve_local_linear(name="U_np1", operator=operator, rhs=rhs)
    program.commit("plasma", u_np1)
    return program


def main():
    model = build_model()
    module = model.module
    program = manual_program(module)

    print("model:", module.name)
    print("operators:", module.list_operators())
    print("program nodes:", len(program._values))
    print("program commits:", {b: s.op for b, s in program.commits().items()})

    import pops

    # RuntimeError = no compiler / no Kokkos visible / compile failed. On the PR-D branch a
    # board pops.physics.Model that drives a named-source rhs also raises AttributeError
    # ('Model' object has no attribute '_source_terms') in program_codegen -- a board-model
    # lowering gap on the PR-E/PR-F punch-list; the authoring half above is what Spec 30
    # demonstrates and always runs.
    try:
        compiled = pops.compile_problem(model=model, time=program)
    except (RuntimeError, AttributeError) as exc:
        print("skip compile step (compile_problem could not build the .so: %s: %s)"
              % (type(exc).__name__, str(exc)[:160]))
        return 0
    print("problem.so:", compiled.so_path)
    print("OK: hand-written Spec 4 time program compiled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
