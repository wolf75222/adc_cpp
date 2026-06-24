"""Spec 3: native HLLC needs model capabilities (criterion 10).

The Riemann solvers are native C++ (`adc::HLLCFlux` etc.). HLLC is generic but needs the
model to provide capabilities: a pressure and the fluid roles Density/MomentumX/MomentumY.
`m.riemann("hllc")` validates this and drives the dsl `enable_hllc()` that GENERATES the
`ADC_HD` `contact_speed` / `hllc_star_state` hooks from those roles. A model that lacks a
capability is rejected with a clear message; Rusanov needs only a max wave speed.

Run: python3 examples/spec3/hllc_capabilities_euler.py
"""
import adc.physics as physics
from adc.physics import Model


def euler(with_pressure=True):
    m = Model("euler")
    U = m.state("U", components=["rho", "mx", "my", "E"],
                roles={"rho": "density", "mx": "momentum_x", "my": "momentum_y", "E": "energy"})
    rho, mx, my, E = U
    m.primitive("u", mx / rho)
    m.primitive("v", my / rho)
    g = m.param("gamma", 1.4)
    if with_pressure:
        m.primitive("p", (g - 1.0) * (E - 0.5 * (mx * mx + my * my) / rho))
    return m


def main():
    # 1) a role-tagged Euler model with pressure: HLLC is accepted, hooks get generated.
    m = euler(with_pressure=True)
    m.riemann("hllc")
    print("HLLC accepted; enable_hllc driven:", m._dsl._m._hllc)
    print("roles:", physics._roles_for(m._dsl._m))

    # 2) Rusanov needs only a max wave speed -- no pressure / no roles required.
    bare = Model("bare")
    bare.state("U", components=["q0", "q1"])  # no fluid roles
    bare.riemann("rusanov")
    print("Rusanov accepted on a role-free model:", bare._riemann)

    # 3) a model WITHOUT pressure is rejected for HLLC, with a clear message.
    try:
        euler(with_pressure=False).riemann("hllc")
    except ValueError as exc:
        print("rejected (no pressure):", exc)

    print("\nOK: HLLC capability validation + role-derived hook generation.")


if __name__ == "__main__":
    main()
