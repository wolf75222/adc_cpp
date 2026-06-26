"""Spec 3 generic invariants: nothing about mass / charge / energy is hardcoded.

An invariant is a typed function StateSet -> Scalar built from a board
``integral(...)`` expression. The framework stores whatever the user writes; the
name carries no special meaning.

Run: python3 examples/spec3/invariant_generic.py
"""
from pops.physics import Model
from pops.math import integral


def build():
    m = Model("two_species")
    e = m.state("U", components=["ne", "mex", "mey"],
                roles={"ne": "density"})
    ne, mex, mey = e

    qe = m.param("qe", 1.0)
    # a generic conserved quantity: a weighted integral of a state component
    charge = m.invariant("total_charge", expression=integral(-qe * ne), over=[e])
    momentum = m.invariant("total_momentum_x", expression=integral(mex), over=[e])
    return m, (charge, momentum)


if __name__ == "__main__":
    m, (charge, momentum) = build()
    print("declared invariants:", list(m.invariants()))
    print("  total_charge value:", charge.value)
    print("  total_momentum_x value:", momentum.value)
    print("note: the framework hardcodes no mass/charge/energy; names are arbitrary")
