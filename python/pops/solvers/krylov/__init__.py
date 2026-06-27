"""pops.solvers.krylov -- the matrix-free Krylov solver catalog (Spec 5 sec.5.7).

The Krylov solvers are FREE FUNCTIONS in the C++ ``namespace pops`` (generic_krylov.hpp);
each factory here returns an inert :class:`pops.descriptors.BrickDescriptor` naming the real
C++ symbol and the runtime ``scheme`` token. They compute nothing; codegen / the runtime
consume the descriptor. This is the Spec 5 home of the catalog formerly parked under
``pops.lib.solvers`` -- the old names re-export from here via the ``pops.lib.solvers`` shim.

* :func:`CG` -- conjugate gradient (SPD systems);
* :func:`BiCGStab` -- stabilised bi-conjugate gradient (nonsymmetric);
* :func:`GMRES` -- generalised minimal residual (nonsymmetric);
* :func:`Richardson` -- preconditioned Richardson iteration.
"""
from pops.descriptors import _native


def _solver(name, native_id, **options):
    """A native Krylov-solver descriptor in the ``solver`` category (scheme == @p name)."""
    return _native(name, native_id, name, category="solver", **options)


def CG(**options):
    """The conjugate-gradient Krylov solver (``pops::cg_solve``; scheme ``"cg"``). Inert."""
    return _solver("cg", "pops::cg_solve", **options)


def BiCGStab(**options):
    """The stabilised bi-CG Krylov solver (``pops::bicgstab_solve``; scheme ``"bicgstab"``)."""
    return _solver("bicgstab", "pops::bicgstab_solve", **options)


def GMRES(**options):
    """The GMRES Krylov solver (``pops::gmres_solve``; scheme ``"gmres"``). Inert."""
    return _solver("gmres", "pops::gmres_solve", **options)


def Richardson(**options):
    """The Richardson iteration (``pops::richardson_solve``; scheme ``"richardson"``). Inert."""
    return _solver("richardson", "pops::richardson_solve", **options)


__all__ = ["CG", "BiCGStab", "GMRES", "Richardson"]
