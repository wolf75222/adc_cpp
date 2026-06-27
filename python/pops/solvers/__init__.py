"""pops.solvers -- the linear / nonlinear / elliptic solver descriptor catalog (Spec 5 sec.5.7).

Spec 5 (sec.4 / 5.7 / 13.11.1) homes the solver catalog in this top-level central package
(alongside :mod:`pops.numerics` / :mod:`pops.linalg` / :mod:`pops.fields`), moving it out of the
transitional ``pops.lib.solvers``. Every entry is an inert descriptor: codegen / the runtime
consume it, nothing here computes in Python (the C++ solvers execute).

Sub-packages:

* :mod:`pops.solvers.krylov` -- matrix-free Krylov solvers (CG / BiCGStab / GMRES / Richardson);
* :mod:`pops.solvers.nonlinear` -- Newton / FixedPoint (planned: no native solver type yet);
* :mod:`pops.solvers.schur` -- the Schur-condensation solver (``pops::SchurCondensationOperator``);
* :mod:`pops.solvers.elliptic` -- the RICH GeometricMG (typed smoother / coarse / tolerance /
  max_cycles + capabilities) and the planned FFT spectral Poisson solver;
* :mod:`pops.solvers.preconditioners` -- Identity / Jacobi / BlockJacobi / GeometricMG / User;
* :mod:`pops.solvers.options` / :mod:`pops.solvers.tolerances` -- the typed smoother / coarse /
  tolerance sub-descriptors the elliptic solver takes;
* :mod:`pops.solvers.requirements` -- the solver capability vocabulary.

The ``solvers`` :class:`types.SimpleNamespace` gathers the Krylov + nonlinear + Schur factories
under one attribute surface (``solvers.CG()`` / ``solvers.GMRES()`` / ``solvers.Newton()`` /
``solvers.Schur()``); the custom-solver AUTHORING DSL (``@pops.lib.solver`` / ``SolverContext`` /
``build_solver_ir`` / ``generate_solver_cpp``) lives in :mod:`pops.lib.solvers.dsl` (it imports
the heavy ``pops.time`` lazily and is codegen-free). The ``pops.lib.solvers`` module is now a thin
re-export shim onto this package, so the in-flight install path (``pops.lib.solvers.GMRES()``,
``solvers.custom`` / ``solvers.registered``) keeps working.
"""
from types import SimpleNamespace

from . import elliptic, krylov, nonlinear, options, requirements, schur, tolerances
from .elliptic import FFT, GeometricMG
from .krylov import CG, BiCGStab, GMRES, Richardson
from .nonlinear import FixedPoint, Newton
from .preconditioners import preconditioners
from .schur import CondensedSchur, Schur

# The flat solver factory surface (``solvers.CG()`` ... ``solvers.Schur()``), mirroring the
# legacy ``pops.lib.solvers.solvers`` namespace. The custom-solver registry hooks
# (``solvers.custom`` / ``solvers.registered``) are attached by the ``pops.lib.solvers`` shim,
# which owns the authoring DSL -- keeping ``pops.solvers`` free of any ``pops.lib`` import.
solvers = SimpleNamespace(
    CG=CG, BiCGStab=BiCGStab, GMRES=GMRES, Richardson=Richardson,
    Newton=Newton, FixedPoint=FixedPoint, Schur=Schur,
)

__all__ = [
    "elliptic", "krylov", "nonlinear", "schur", "options", "tolerances",
    "preconditioners", "requirements",
    "GeometricMG", "FFT",
    "CG", "BiCGStab", "GMRES", "Richardson",
    "Newton", "FixedPoint", "Schur", "CondensedSchur",
    "solvers",
]
