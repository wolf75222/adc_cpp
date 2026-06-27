"""pops.solvers.preconditioners -- the preconditioner brick catalog (Spec 5 sec.5.7).

Only the geometric-multigrid preconditioner has a native type (``pops::GeometricMG``);
identity / jacobi / block-jacobi have none yet (the polar solver has its own PolarPrecond
enum), so they are catalogued as PLANNED descriptors. :func:`User` surfaces a loaded external
preconditioner brick. This is the Spec 5 home of the catalog formerly parked under
``pops.lib.solvers.preconditioners`` -- the old name re-exports from here via the shim.
"""
from types import SimpleNamespace

from pops.descriptors import _external_descriptor, _native, _planned

preconditioners = SimpleNamespace(
    Identity=lambda: _planned("identity", "identity", category="preconditioner"),
    Jacobi=lambda: _planned("jacobi", "jacobi", category="preconditioner"),
    BlockJacobi=lambda: _planned("block_jacobi", "block_jacobi",
                                 category="preconditioner"),
    GeometricMG=lambda **o: _native("geometric_mg", "pops::GeometricMG", "geometric_mg",
                                    category="preconditioner", **o),
    User=lambda brick_id: _external_descriptor(brick_id, expect_category="preconditioner"),
)

__all__ = ["preconditioners"]
