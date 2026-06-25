"""Closures for generic 2D moment models (the only physics of the hierarchy).

A closure is a callable `S -> dict 'S{p}{q}'` returning the standardized moments of order
order+1. `gaussian_closure(order)` (Levermore: Gaussian cumulants, generic in the order) is
provided.
"""
from adc.lib.moments.closures.gaussian import gaussian_closure

__all__ = ["gaussian_closure"]
