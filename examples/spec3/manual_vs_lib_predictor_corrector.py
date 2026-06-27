"""Spec 3: a library time scheme is a MACRO that builds operator-first IR, not a stepper.

The `pops.lib.time` schemes are MacroBricks: they construct a Program from operator names
using only the kernel primitives (P.call / linear_combine / solve_local_linear / commit).
This example builds a predictor-corrector two ways and shows (a) the Spec-3 macro-catalog
entry `pops.lib.time.macros.time.predictor_corrector` forwards to the same
`pops.lib.time.predictor_corrector_local_linear` (identical IR), and (b) every node
the macro emits is a kernel primitive op -- there is no parallel runtime stepper. So a
user can always rewrite the same scheme by hand against the kernel.

(Spec 4 s6 / s14: the ready schemes live in `pops.lib.time` (by their explicit names, no `std`
bundle -- s7), not in the time-language module `pops.time`; the catalog forwards to them.)

Run: python3 examples/spec3/manual_vs_lib_predictor_corrector.py
"""
from pops.math import sqrt, grad, div, laplacian, ddt
from pops.physics import Model
import pops.lib.time as libtime
import pops.lib.time.macros as libmacros
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


def build(via_catalog):
    m = build_model()
    P = adctime.Program("pc_catalog" if via_catalog else "pc_std")
    P.bind_operators(m.module)
    macro = (libmacros.time.predictor_corrector if via_catalog
             else libtime.predictor_corrector_local_linear)
    macro(P, "plasma", fields_operator="fields_from_state",
          explicit_rate_operator="explicit_rate", implicit_operator="implicit_operator")
    return P


def main():
    catalog_prog, std_prog = build(True), build(False)

    # (a) the Spec-3 catalog macro forwards to pops.lib.time: identical IR.
    assert _ir(catalog_prog) == _ir(std_prog), "the catalog macro must forward to pops.lib.time"

    # (b) the macro emits only kernel primitive ops -- no parallel stepper.
    emitted = {v.op for v in catalog_prog._values}
    assert emitted <= KERNEL_OPS, "unexpected non-kernel op(s): %s" % (emitted - KERNEL_OPS)

    print("ops emitted by the macro:", sorted(emitted))
    print("catalog macro IR == pops.lib.time IR:", _ir(catalog_prog) == _ir(std_prog))
    print("\nOK: the library scheme is a macro over the operator-first kernel.")


if __name__ == "__main__":
    main()
