"""Spec 3 board-like model authoring: 2D isothermal Euler + Poisson + Lorentz.

Writes the model the way it appears on the blackboard and prints the operator-first
IR (pops.model.Module) it lowers to. No compilation: this demonstrates layer 1
(authoring) and the lowering to layer 2 (the typed Module).

Run: python3 examples/spec3/board_euler_poisson_lorentz.py
"""
from pops.physics import Model
from pops.math import sqrt, grad, div, laplacian, ddt


def build():
    m = Model("euler_poisson_lorentz")

    U = m.state("U", components=["rho", "mx", "my"],
                roles={"rho": "density", "mx": "momentum_x", "my": "momentum_y"})
    rho, mx, my = U

    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)

    cs2 = m.param("cs2", 1.0)
    alpha = m.param("alpha", 1.0)
    rho_ref = m.param("rho_ref", 1.0)

    p = m.scalar("p", cs2 * rho)
    c = m.scalar("c", sqrt(cs2))

    flux = m.flux("F", on=U,
                  x=[mx, mx * u + p, mx * v],
                  y=[my, my * u, my * v + p],
                  waves={"x": [u - c, u, u + c], "y": [v - c, v, v + c]})

    phi = m.field("phi")
    m.solve_field("fields_from_state",
                  equation=(-laplacian(phi) == alpha * (rho - rho_ref)),
                  outputs={"phi": phi, "grad_x": grad(phi).x, "grad_y": grad(phi).y},
                  solver="geometric_mg")

    e_field = m.vector_field("E", x=-grad(phi).x, y=-grad(phi).y)
    a_src = m.source("electric", on=U, value=[0.0 * rho, rho * e_field.x, rho * e_field.y])

    bz = m.aux("B_z")
    c_b = m.local_linear_operator("lorentz", on=U,
                                  matrix=[[0.0, 0.0, 0.0],
                                          [0.0, 0.0, bz],
                                          [0.0, -bz, 0.0]])

    m.rate("explicit_rate", ddt(U) == -div(flux) + a_src)
    m.operator("implicit_operator", c_b)
    m.check()
    return m


if __name__ == "__main__":
    m = build()
    mod = m.module
    print("model:", mod.name)
    print("state spaces:", {n: s.components for n, s in mod.state_spaces().items()})
    print("field spaces:", list(mod.field_spaces()))
    print("operators:", mod.list_operators())
    print("explicit_rate signature:    ", mod.operator_signature("explicit_rate"))
    print("implicit_operator signature:", mod.operator_signature("implicit_operator"))
