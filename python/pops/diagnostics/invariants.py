"""pops.diagnostics.invariants -- the conservation-invariant brick catalog (Spec 3 / Spec 5).

An ``invariant`` descriptor names a conserved quantity (an optional board-value
expression kept off the identity key); ``conservation_check`` names a tolerance
check. Both are macro descriptors -- they compute nothing.

The ``conservation_check`` factory borrows its native reduction scheme label from the typed
:class:`pops.diagnostics.measures.ConservationCheck`, so the label lives in ONE place. The
factory still returns the historical ``BrickDescriptor`` (``category="invariant"``) consumers
expect; the typed class is the canonical authoring form.
"""
from types import SimpleNamespace

from pops.descriptors import BrickDescriptor
from .measures import ConservationCheck as _ConservationCheck


def _invariant(name, expression=None, over=None):
    """A catalog invariant descriptor; the value ``expression`` is kept off the
    identity key (it may be an unhashable board node) as a plain attribute."""
    return BrickDescriptor(name, "macro", category="invariant", scheme="invariant",
                           options={"over": tuple(over) if over else ()},
                           expression=expression)


invariants = SimpleNamespace(
    invariant=_invariant,
    conservation_check=lambda name, tolerance=1e-10, **o: BrickDescriptor(
        name, "macro", category="invariant", scheme=_ConservationCheck.scheme,
        options={"tolerance": tolerance, **o}),
)

__all__ = ["invariants"]
