#!/usr/bin/env python3
"""Spec 4 (32): build a moment model with a user-supplied closure.

The moment-model kit is the Spec 4 replacement for the banned ``custom.py`` escape hatch:
a closure is a decorated callable, and a model is a recorded specification that builds a
``pops.physics`` model on demand.

    from pops.lib.moments import CartesianVelocityMoments, MomentModel
    from pops.lib.moments.closures import closure

``@closure(order=N)`` validates that the wrapped callable returns EXACTLY the standardized
moments of order ``N+1`` (the keys ``S{p}{q}`` with ``p + q == N + 1``); a wrong key set
raises a clear ``TypeError`` at evaluation time. ``CartesianVelocityMoments`` records the
order, the closure and the numerics; chainable ``add_*`` methods record transport, the
Poisson coupling and sources; ``.build(name)`` is the only call that touches the engine
and returns a ``physics`` model.

Pure Python, no compiler needed -- the example builds the model object and prints it.

Run::

    python3 examples/spec4/custom_moment_model.py
"""
import sys

from pops.lib.moments import CartesianVelocityMoments, MomentModel
from pops.lib.moments.closures import closure


@closure(order=2)
def zero_heat_flux(S):
    """A toy order-2 closure: vanishing standardized order-3 moments.

    For order N == 2 the closure must supply the order-3 standardized moments
    ``S30, S21, S12, S03``. Returning zeros is the simplest realizable choice (no skew /
    no heat flux); the decorator checks the key set so a typo fails loudly here, not deep
    in the build.
    """
    return {"S30": 0.0, "S21": 0.0, "S12": 0.0, "S03": 0.0}


def build():
    """Record an order-2 Cartesian-velocity moment model with the custom closure."""
    spec = (
        CartesianVelocityMoments(order=2, closure=zero_heat_flux)
        .add_transport()
        .add_poisson_coupling(phi="phi", eps=1.0)
        .add_numerics(robust=True, exact_speeds=True)
    )
    assert isinstance(spec, MomentModel)
    return spec


def main():
    spec = build()
    print("recorded spec:", type(spec).__name__)
    model = spec.build("custom_moment_model")
    module = model.module
    print("built model:", module.name)
    state_spaces = module.state_spaces()
    for name, space in state_spaces.items():
        print("  state %r has %d components" % (name, len(space.components)))
    print("operators:", module.list_operators())
    print("OK: custom-closure moment model built (no compiler needed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
