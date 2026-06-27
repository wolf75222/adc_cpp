"""pops.fields -- typed elliptic field-problem authoring (Spec 5 sec.5.5 / sec.9).

The ``pops.fields`` package describes a self-consistent FIELD solve: an unknown computed
by inverting an elliptic operator each step (the Poisson coupling of a plasma, a pressure
projection, ...). The central object is :class:`FieldProblem` (with the Poisson-family
shortcuts :class:`PoissonProblem` / :class:`ScreenedPoissonProblem` /
:class:`AnisotropicPoissonProblem`); the supporting submodules declare the typed pieces:

* :mod:`pops.fields.bcs` -- field-value boundary conditions + face selectors;
* :mod:`pops.fields.rhs` -- typed right-hand-side sources (``ChargeDensity``);
* :mod:`pops.fields.coefficients` -- scalar / reaction operator coefficients;
* :mod:`pops.fields.nullspace` -- ``ConstantNullspace`` for singular operators;
* :mod:`pops.fields.aux` -- static / derived aux fields + the re-exported ``AuxHalo``.

Everything is inert; the runtime materialises and solves after validation. This is the
top-level authoring package and is DISTINCT from the ``pops.lib.fields`` preset catalog.
"""
from .problem import FieldProblem
from .poisson import (PoissonProblem, ScreenedPoissonProblem,
                      AnisotropicPoissonProblem)
from . import bcs, rhs, coefficients, nullspace, aux

__all__ = [
    "FieldProblem", "PoissonProblem", "ScreenedPoissonProblem",
    "AnisotropicPoissonProblem",
    "bcs", "rhs", "coefficients", "nullspace", "aux",
]
