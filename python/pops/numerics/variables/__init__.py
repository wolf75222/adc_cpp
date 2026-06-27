"""pops.numerics.variables -- the reconstructed-variable-set catalog (Spec 5 sec.5.4 / sec.7).

The finite-volume scheme reconstructs face states in either the CONSERVATIVE variables
(rho, rho_u, rho_v, E) or the PRIMITIVE variables (rho, u, v, p; more robust for Euler --
positivity of rho and p). Spec 5 sec.7 forbids naming that choice with a bare string, so the
two states are typed descriptors here, mirroring :mod:`pops.numerics.riemann` and
:mod:`pops.numerics.reconstruction`.

``Conservative()`` / ``Primitive()`` are inert :class:`pops.descriptors.BrickDescriptor`
records; ``Spatial`` / ``FiniteVolume`` read their ``.scheme`` ("conservative" / "primitive")
when lowering to the C++ ABI.
"""
from types import SimpleNamespace

from pops.descriptors import BrickDescriptor


def _variables(scheme):
    """A reconstructed-variable-set descriptor (no native C++ symbol: a per-block flag)."""
    return BrickDescriptor(scheme, "native", category="variables", native_id="",
                           scheme=scheme)


variables = SimpleNamespace(
    Conservative=lambda: _variables("conservative"),
    Primitive=lambda: _variables("primitive"),
)

# Spec 5: expose the variable sets at module scope (``from pops.numerics.variables import
# Conservative``); the namespace stays for ``variables.Conservative``.
Conservative = variables.Conservative
Primitive = variables.Primitive

__all__ = ["variables", "Conservative", "Primitive"]
