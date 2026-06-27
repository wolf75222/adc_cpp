#!/usr/bin/env python3
"""Spec 5 sec.7 (ADC-497): FiniteVolume / Spatial choose the scheme with TYPED descriptors.

The runtime spatial builder no longer names a flux / reconstruction / limiter / variable-set with a
bare string. Every selector is a typed ``pops.numerics`` descriptor; a string raises a clear
``TypeError`` (``reject_string_selector``) that points at the typed object. These checks pin both
sides of the contract:

  - NEGATIVE: a bare-string limiter / riemann / flux / recon / variables raises ``TypeError`` whose
    message names the rejected value and the typed alternative;
  - POSITIVE: the typed ``pops.numerics`` descriptors are accepted and lowered to the canonical C++
    token stored on ``Spatial.limiter`` / ``.flux`` / ``.recon``;
  - the descriptor CATEGORY gates the slot (a Riemann flux in the limiter slot is rejected), an
    unknown descriptor scheme is rejected, and the boolean shortcuts (none=/minmod=/.../primitive=)
    stay valid sugar (they are typed flags, not strings).

Pure Python: it imports the inert authoring packages only (``import pops`` loads ``_pops`` as a side
effect, but no model is built or run).
"""
import sys

import pytest

pops = pytest.importorskip("pops")

from pops.numerics.riemann import Rusanov, HLL, HLLC, Roe  # noqa: E402
from pops.numerics.reconstruction import FirstOrder, MUSCL, WENO5, WENO5Z  # noqa: E402
from pops.numerics.reconstruction.limiters import Minmod, VanLeer  # noqa: E402
from pops.numerics.variables import Conservative, Primitive  # noqa: E402


# --- NEGATIVE: bare strings are rejected, pointing at the typed object -----------------------
def test_finitevolume_rejects_string_limiter():
    with pytest.raises(TypeError) as exc:
        pops.FiniteVolume(limiter="minmod")
    msg = str(exc.value)
    assert "limiter='minmod'" in msg
    assert "pops.numerics.reconstruction" in msg


def test_finitevolume_rejects_string_riemann():
    with pytest.raises(TypeError) as exc:
        pops.FiniteVolume(riemann="rusanov")
    msg = str(exc.value)
    # The message names the FiniteVolume parameter, not the internal Spatial slot.
    assert "riemann='rusanov'" in msg
    assert "pops.numerics.riemann" in msg


def test_finitevolume_rejects_string_variables():
    with pytest.raises(TypeError) as exc:
        pops.FiniteVolume(variables="conservative")
    msg = str(exc.value)
    assert "variables='conservative'" in msg
    assert "pops.numerics.variables" in msg


def test_spatial_rejects_string_flux_and_recon():
    with pytest.raises(TypeError) as exc:
        pops.Spatial(flux="hll")
    assert "flux='hll'" in str(exc.value)
    with pytest.raises(TypeError) as exc:
        pops.Spatial(recon="primitive")
    assert "recon='primitive'" in str(exc.value)


def test_spatial_rejects_string_limiter():
    with pytest.raises(TypeError) as exc:
        pops.Spatial(limiter="weno5")
    assert "limiter='weno5'" in str(exc.value)


# --- NEGATIVE: a descriptor of the wrong category is rejected --------------------------------
def test_wrong_category_descriptor_rejected():
    # A Riemann flux is not a reconstruction/limiter.
    with pytest.raises(TypeError) as exc:
        pops.FiniteVolume(limiter=Rusanov())
    assert "riemann" in str(exc.value)
    # A reconstruction descriptor is not a Riemann flux.
    with pytest.raises(TypeError) as exc:
        pops.Spatial(flux=Minmod())
    assert "reconstruction" in str(exc.value) or "limiter" in str(exc.value)


# --- POSITIVE: typed descriptors are accepted and lowered to canonical tokens ----------------
def test_typed_flux_descriptors_lower():
    for desc, token in ((Rusanov(), "rusanov"), (HLL(), "hll"),
                        (HLLC(), "hllc"), (Roe(), "roe")):
        s = pops.FiniteVolume(riemann=desc)
        assert s.flux == token, (desc, s.flux)


def test_typed_limiter_descriptors_lower():
    cases = ((FirstOrder(), "none"), (Minmod(), "minmod"), (VanLeer(), "vanleer"),
             (WENO5(), "weno5"), (WENO5Z(), "weno5"),
             (MUSCL(limiter="minmod"), "minmod"), (MUSCL(limiter="vanleer"), "vanleer"))
    for desc, token in cases:
        s = pops.FiniteVolume(limiter=desc)
        assert s.limiter == token, (desc, s.limiter)


def test_typed_variable_descriptors_lower():
    assert pops.FiniteVolume(variables=Conservative()).recon == "conservative"
    assert pops.FiniteVolume(variables=Primitive()).recon == "primitive"
    assert pops.Spatial(recon=Conservative()).recon == "conservative"
    assert pops.Spatial(recon=Primitive()).recon == "primitive"


def test_combined_typed_spatial():
    s = pops.Spatial(limiter=VanLeer(), flux=HLLC(), recon=Primitive())
    assert (s.limiter, s.flux, s.recon) == ("vanleer", "hllc", "primitive")


def test_defaults_are_canonical():
    s = pops.Spatial()
    assert (s.limiter, s.flux, s.recon) == ("minmod", "rusanov", "conservative")
    fv = pops.FiniteVolume()
    assert (fv.limiter, fv.flux, fv.recon) == ("minmod", "rusanov", "conservative")


def test_boolean_shortcuts_stay_valid():
    # The boolean shortcuts are typed flags, not string selectors -- they keep working.
    assert pops.Spatial(none=True).limiter == "none"
    assert pops.Spatial(minmod=True).limiter == "minmod"
    assert pops.Spatial(vanleer=True).limiter == "vanleer"
    assert pops.Spatial(weno5=True).limiter == "weno5"
    assert pops.Spatial(primitive=True).recon == "primitive"
    # A shortcut and an explicit typed flux compose.
    s = pops.Spatial(minmod=True, flux=Roe(), primitive=True)
    assert (s.limiter, s.flux, s.recon) == ("minmod", "roe", "primitive")


def test_finitevolume_forwards_boolean_shortcuts():
    # The boolean shortcuts of Spatial are forwarded THROUGH FiniteVolume identically, so the
    # docstring claim "primitive= is a FiniteVolume shortcut" is true.
    assert pops.FiniteVolume(none=True).limiter == "none"
    assert pops.FiniteVolume(minmod=True).limiter == "minmod"
    assert pops.FiniteVolume(vanleer=True).limiter == "vanleer"
    assert pops.FiniteVolume(weno5=True).limiter == "weno5"
    assert pops.FiniteVolume(primitive=True).recon == "primitive"
    # A forwarded shortcut and an explicit typed flux compose, like on Spatial.
    fv = pops.FiniteVolume(minmod=True, riemann=Roe(), primitive=True)
    assert (fv.limiter, fv.flux, fv.recon) == ("minmod", "roe", "primitive")


def test_finitevolume_reconstruction_alias():
    # Spec 5 sec.14.1 writes FiniteVolume(reconstruction=WENO5()); it is an alias for limiter=.
    alias = pops.FiniteVolume(reconstruction=WENO5())
    explicit = pops.FiniteVolume(limiter=WENO5())
    assert alias.limiter == explicit.limiter == "weno5"
    # The alias also works on Spatial.
    assert pops.Spatial(reconstruction=WENO5()).limiter == "weno5"
    # Passing both limiter= and reconstruction= is a clear error, not a silent pick.
    with pytest.raises(TypeError):
        pops.FiniteVolume(limiter=WENO5(), reconstruction=WENO5())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
