#!/usr/bin/env python3
"""Spec 4 (33): the ready-made HyQMOM15 model, authored only (no run).

HyQMOM15 is a 15-moment (order-4) hyperbolic-quadrature closure shipped under
``pops.lib.models.moments``. It is a model PROVIDER: a factory that returns a
``pops.physics`` model wired for the Vlasov-Poisson-magnetic problem.

    from pops.lib.models.moments import HyQMOM15
    model = HyQMOM15.vlasov_poisson_magnetic(order=4)

This example builds the model object and inspects its operator-first IR. It authors only:
no case, no grid, no time integration -- those belong to a driver (see
``examples/spec4/adc_cases_style_driver.py``) and to ``adc_cases``.

Pure Python, no compiler needed.

Run::

    python3 examples/spec4/hyqmom15_model_only.py
"""
import sys

from pops.lib.models.moments import HyQMOM15


def build():
    """Author the standard HyQMOM15 Vlasov-Poisson-magnetic model (order 4 -> 15 moments)."""
    return HyQMOM15.vlasov_poisson_magnetic(order=4)


def main():
    model = build()
    module = model.module
    print("built model:", module.name)

    state_spaces = module.state_spaces()
    for name, space in state_spaces.items():
        print("  state %r has %d moments" % (name, len(space.components)))
        print("  components:", space.components)

    field_spaces = list(module.field_spaces())
    print("fields:", field_spaces)
    print("operators:", module.list_operators())
    print("OK: HyQMOM15 model authored (no run; the case belongs in a driver / adc_cases).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
