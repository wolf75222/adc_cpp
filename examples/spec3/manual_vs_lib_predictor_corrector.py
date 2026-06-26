"""Spec 3: a library time scheme is a MACRO that builds operator-first IR, not a stepper.

`pops.lib.time.*` and `pops.time.std.*` are MacroBricks: they construct a Program from
operator names using only the kernel primitives (P.call / linear_combine /
solve_local_linear / commit). This example builds a predictor-corrector with the library
macro and shows (a) `pops.lib.time` is a thin forwarder to `pops.time.std` (identical IR),
and (b) every node the macro emits is a kernel primitive op -- there is no parallel
runtime stepper. So a user can always rewrite the same scheme by hand against the kernel.

Run: python3 examples/spec3/manual_vs_lib_predictor_corrector.py
"""
from pops.math import sqrt, grad, div, laplacian, ddt
from pops.physics import Model
import pops.lib as lib
import pops.time as adctime


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


def _ir(P):
    idx = {id(x): k for k, x in enumerate(P._values)}
    nodes = [(x.vtype, x.op, tuple(idx[id(i)] for i in x.inputs)) for x in P._values]
    return (nodes, sorted((b, idx[id(s)]) for b, s in P.commits().items()))


KERNEL_OPS = {"state", "solve_fields", "rhs", "source", "apply", "linear_source",
              "linear_combine", "solve_local_linear", "solve_fields_from_blocks"}


def build(via_lib):
    m = build_model()
    P = adctime.Program("pc_lib" if via_lib else "pc_std")
    P.bind_operators(m.module)
    macro = lib.time.predictor_corrector if via_lib else adctime.std.predictor_corrector_local_linear
    macro(P, "plasma", fields_operator="fields_from_state",
          explicit_rate_operator="explicit_rate", implicit_operator="implicit_operator")
    return P


def main():
    lib_prog, std_prog = build(True), build(False)

    # (a) pops.lib.time is a thin forwarder to pops.time.std: identical IR.
    assert _ir(lib_prog) == _ir(std_prog), "pops.lib.time must forward to pops.time.std"

    # (b) the macro emits only kernel primitive ops -- no parallel stepper.
    emitted = {v.op for v in lib_prog._values}
    assert emitted <= KERNEL_OPS, "unexpected non-kernel op(s): %s" % (emitted - KERNEL_OPS)

    print("ops emitted by the macro:", sorted(emitted))
    print("pops.lib.time IR == pops.time.std IR:", _ir(lib_prog) == _ir(std_prog))
    print("\nOK: the library scheme is a macro over the operator-first kernel.")


if __name__ == "__main__":
    main()
