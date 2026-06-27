"""pops.diagnostics -- the diagnostic brick catalog (Spec 3 / Spec 5).

Scalar reductions (integral / norm / mass / momentum / energy / ...) as macro
descriptors. The conservation-invariant descriptors are catalogued separately in
:mod:`pops.diagnostics.invariants`.

Spec 5 (sec.4 / sec.5.13) homes diagnostics in the top-level ``pops.diagnostics``
package (formerly ``pops.lib.diagnostics``). The reduction macros stay inert
descriptors; nothing here computes in Python.

Spec 5 sec.5.13 / 14.2.7 also names a diagnostic with a TYPED object (a
:class:`~pops.diagnostics.measures.Norm` / :class:`~pops.diagnostics.measures.Integral` /
:class:`~pops.diagnostics.measures.MinMax` / :class:`~pops.diagnostics.measures.ConservationCheck`
descriptor) rather than ``diagnostics.norm(kind="l2")``. Those typed measures live in
:mod:`pops.diagnostics.measures` and lower to the SAME native reduction schemes the factory
functions here already name. The legacy factories stay for now.
"""
from types import SimpleNamespace

from pops.descriptors import BrickDescriptor
from .invariants import invariants
from .measures import ConservationCheck, Integral, MinMax, Norm


def _diag(_dname, **o):
    return BrickDescriptor(_dname, "macro", category="diagnostic", scheme=_dname,
                           options=o or None)


diagnostics = SimpleNamespace(
    integral=lambda expr=None, **o: _diag("integral", expr=expr, **o),
    norm=lambda kind="l2", **o: _diag("norm", kind=kind, **o),
    mass=lambda **o: _diag("mass", **o),
    momentum=lambda **o: _diag("momentum", **o),
    energy=lambda **o: _diag("energy", **o),
    invariant_error=lambda name=None, **o: _diag("invariant_error", name=name, **o),
    residual=lambda **o: _diag("residual", **o),
)

# Spec 5: expose the reductions at module scope (``from pops.diagnostics import norm``).
integral = diagnostics.integral
norm = diagnostics.norm
mass = diagnostics.mass
momentum = diagnostics.momentum
energy = diagnostics.energy
invariant_error = diagnostics.invariant_error
residual = diagnostics.residual

__all__ = ["diagnostics", "invariants", "integral", "norm", "mass", "momentum",
           "energy", "invariant_error", "residual",
           # Spec 5 typed measure descriptors (pops.diagnostics.measures).
           "Norm", "Integral", "MinMax", "ConservationCheck"]
