"""pops.solvers.schur -- the Schur-complement solver catalog (Spec 5 sec.5.7).

The Schur-condensation solver eliminates a coupled (e.g. source) block and solves the reduced
system; the native symbol is ``pops::SchurCondensationOperator``. :func:`Schur` returns the
inert :class:`pops.descriptors.BrickDescriptor` naming it; :func:`CondensedSchur` is an alias
naming the SAME native operator under the condensed-Schur name. Both are inert -- the C++
runtime applies the operator. This is the Spec 5 home of the ``solvers.Schur`` entry formerly
under ``pops.lib.solvers``.

NOTE: this :func:`CondensedSchur` SOLVER descriptor is DISTINCT from
:class:`pops.CondensedSchur` (``pops.time``), which is the time-integration SPLITTING POLICY
for an implicit source. They share no namespace and are not the same object.
"""
from pops.descriptors import _native


def Schur(**options):
    """The Schur-condensation solver descriptor (``pops::SchurCondensationOperator``).

    Scheme token ``"schur"``; inert (the C++ runtime applies the condensation operator).
    """
    return _native("schur", "pops::SchurCondensationOperator", "schur",
                   category="solver", **options)


def CondensedSchur(**options):
    """Alias of :func:`Schur` naming the same ``pops::SchurCondensationOperator`` (inert).

    Distinct from the ``pops.time`` ``CondensedSchur`` time-splitting policy -- this is the
    LINEAR-SOLVER descriptor, not the time-integration role.
    """
    return Schur(**options)


__all__ = ["Schur", "CondensedSchur"]
