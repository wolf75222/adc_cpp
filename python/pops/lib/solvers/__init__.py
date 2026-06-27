"""pops.lib.solvers -- BACK-COMPAT shim onto :mod:`pops.solvers` (Spec 5 sec.5.7).

Spec 5 moved the linear / nonlinear / Schur / elliptic / preconditioner solver catalog OUT of
``pops.lib`` into the top-level :mod:`pops.solvers` central package (criterion 4 / sec.5.7 /
13.11.1). This module is now a thin re-export so the in-flight install path and existing code
(``pops.lib.solvers.GMRES()``, ``pops.lib.solvers.solvers``, ``pops.lib.solvers.preconditioners``)
keep resolving. New code should import from :mod:`pops.solvers`.

The custom-solver AUTHORING DSL (``@solver`` / ``SolverContext`` / ``build_solver_ir``) and its
C++ lowering (``generate_solver_cpp``) stay PHYSICALLY here -- they author over the matrix-free
Krylov primitives, import the heavy :mod:`pops.time` lazily, and ``solver_cpp`` is deliberately
codegen-free (the ``lib`` acyclic invariant). The ``solvers`` namespace's registry hooks
(``solvers.custom`` / ``solvers.registered``) are wired here, where the DSL lives, so
:mod:`pops.solvers` itself stays free of any ``pops.lib`` import.
"""
from pops.solvers import (CG, BiCGStab, FixedPoint, GMRES, GeometricMG, Newton,
                          Richardson, Schur, solvers)
from pops.solvers.preconditioners import preconditioners
from .dsl import (solver, build_solver_ir, SolverContext, SolverIR,
                  _custom_solver, _registered_solvers, _as_descriptor)
from .solver_cpp import generate_solver_cpp

# The custom-solver registry hooks (``@pops.lib.solver`` -> solvers.custom / .registered). They
# live in the lib DSL, so they are attached here rather than in pops.solvers (which must not
# import pops.lib). Idempotent: re-importing the shim just re-binds the same callables.
solvers.custom = _custom_solver
solvers.registered = _registered_solvers

__all__ = ["solvers", "solver", "build_solver_ir", "generate_solver_cpp",
           "SolverContext", "SolverIR", "preconditioners",
           "CG", "BiCGStab", "GMRES", "Richardson", "Newton", "FixedPoint", "Schur",
           "GeometricMG"]
