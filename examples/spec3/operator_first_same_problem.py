"""Spec 3: the SAME problem driven purely by the operator-first kernel (Spec 2 retention).

The model is authored board-style for brevity, but it lowers to an ordinary
``pops.model.Module``; the time program here uses ONLY the explicit operator-first
builder (P.bind_operators / P.call / P.linear_combine / P.solve_local_linear /
P.commit) -- no board sugar. This is the kernel a library macro or an advanced user
writes against; the board facade produces the identical IR.

Run: python3 examples/spec3/operator_first_same_problem.py
"""
from pops.math import sqrt, grad, div, laplacian, ddt
from pops.physics import Model
from pops.time import Program


def build_model():
    m = Model("euler_poisson_lorentz")
    U = m.state("U", components=["rho", "mx", "my"],
                roles={"rho": "density", "mx": "momentum_x", "my": "momentum_y"})
    rho, mx, my = U
    u, v = m.primitive("u", mx / rho), m.primitive("v", my / rho)
    cs2 = m.param("cs2", 1.0)
    p, c = m.scalar("p", cs2 * rho), m.scalar("c", sqrt(cs2))
    flux = m.flux("F", on=U, x=[mx, mx * u + p, mx * v], y=[my, my * u, my * v + p],
                  waves={"x": [u - c, u, u + c], "y": [v - c, v, v + c]})
    phi = m.field("phi")
    m.solve_field("fields_from_state", equation=(-laplacian(phi) == rho),
                  outputs={"phi": phi, "grad_x": grad(phi).x, "grad_y": grad(phi).y},
                  solver="geometric_mg")
    e_field = m.vector_field("E", x=-grad(phi).x, y=-grad(phi).y)
    a_src = m.source("electric", on=U, value=[0.0 * rho, rho * e_field.x, rho * e_field.y])
    bz = m.aux("B_z")
    c_b = m.local_linear_operator("C(B)", on=U,
                                  matrix=[[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]])
    m.rate("explicit_rate", ddt(U) == -div(flux) + a_src)
    m.operator("implicit_operator", returns=c_b, inputs=["fields"])
    return m


def operator_first_step(m):
    """A backward-Euler-ish implicit step written purely operator-first."""
    P = Program("operator_first_step")
    P.bind_operators(m.module)
    dt = P.dt
    U_n = P.state("plasma")
    fields_n = P.call("fields_from_state", U_n, name="fields_n")
    R_n = P.call("explicit_rate", U_n, fields_n, name="R_n")
    L_n = P.call("implicit_operator", fields_n, name="L_n")
    rhs = P.linear_combine("U_star_rhs", U_n + dt * R_n)
    U_star = P.solve_local_linear("U_star", operator=P.I - dt * L_n, rhs=rhs)
    P.commit("plasma", U_star)
    return P


if __name__ == "__main__":
    m = build_model()
    print(m.dump_module_ir())
    print()
    P = operator_first_step(m)
    print(P.dump_operator_ir())
