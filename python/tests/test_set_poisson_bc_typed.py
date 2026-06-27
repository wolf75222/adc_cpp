"""Spec 5 sec.14.2.6: set_poisson(bc=) accepts a typed native boundary brick.

The typed boundary bricks pops.Dirichlet() / Neumann() / Periodic() (ADC-504) lower to the native
bc tokens 'dirichlet' / 'neumann' / 'periodic'; set_poisson(bc=...) accepts them OR the legacy
string with an IDENTICAL effect (byte-identical Poisson potential). The string path stays
byte-identical: any string passes straight through to the native set_poisson, which validates an
unknown token as before -- the typed coercion adds no stricter string rejection of its own (cf. the
lower_backend(None) regression).
"""
import sys

import numpy as np
import pytest

import pops
from pops.runtime._system_install import _lower_bc

try:
    import pops._pops  # noqa: F401
    _HAVE_ENGINE = True
except Exception:  # pragma: no cover - exercised only without a built extension
    _HAVE_ENGINE = False
requires_engine = pytest.mark.skipif(
    not _HAVE_ENGINE, reason="compiled _pops extension not importable")


def test_bc_lowers_to_legacy_tokens():
    assert _lower_bc(pops.Dirichlet()) == "dirichlet"
    assert _lower_bc(pops.Neumann()) == "neumann"
    assert _lower_bc(pops.Periodic()) == "periodic"


def test_bc_string_passes_through_and_bad_type_rejected():
    # Any string passes straight through (the coercion adds no stricter string rejection of its own
    # -- an unknown token stays the native's responsibility, like lower_wall / lower_backend).
    for s in ("auto", "dirichlet", "neumann", "periodic", "bogus"):
        assert _lower_bc(s) == s
    # A non-string, non-boundary value is a genuine user error (new typed surface, no legacy path).
    with pytest.raises(TypeError):
        _lower_bc(12345)


def _poisson_system(bc):
    from pops.numerics.reconstruction.limiters import Minmod
    from pops.numerics.riemann import Rusanov
    from pops.numerics.variables import Conservative
    s = pops.System(n=32, L=1.0, periodic=False)
    s.set_poisson(rhs="charge_density", solver="geometric_mg", bc=bc,
                  wall="circle", wall_radius=0.4)
    model = pops.Model(state=pops.FluidState(kind="isothermal", cs2=1.0),
                       transport=pops.IsothermalFlux(), source=pops.NoSource(),
                       elliptic=pops.ChargeDensity())
    s.add_equation("e", model=model,
                   spatial=pops.FiniteVolume(limiter=Minmod(), riemann=Rusanov(),
                                             variables=Conservative()),
                   time=pops.Explicit())
    n = 32
    x = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.exp(-(((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.02))
    s.set_primitive_state("e", rho=rho, u=0.0 * rho, v=0.0 * rho)
    return s


@requires_engine
def test_set_poisson_typed_bc_matches_string():
    s_str = _poisson_system("dirichlet")
    s_str.solve_fields()
    s_typed = _poisson_system(pops.Dirichlet())
    s_typed.solve_fields()
    assert np.array_equal(np.array(s_str.potential()), np.array(s_typed.potential()))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
