"""Spec 5 (sec.5.9-5.11 / sec.8): the pops.mesh typed descriptor surface.

These exercise the inert mesh / layout / AMR descriptors: their options/capabilities,
the explainable AMR route limits (max_levels / ratio), the typed refinement criteria,
the back-compat that pops.CartesianMesh is pops.mesh.CartesianMesh, and the short
printable summaries. Pure Python; needs only `import pops` (the compiled _pops loads but
nothing here computes on a grid).
"""
import sys

import pytest

pops = pytest.importorskip("pops")

from pops.mesh import CartesianMesh, PolarMesh, AuxHalo, PatchBox, BoxLayout  # noqa: E402
from pops.mesh.layouts import Uniform, AMR  # noqa: E402
from pops.mesh.amr import (  # noqa: E402
    Refine, TagUnion, RegridEvery, FrozenRegrid, PatchLayout, ProperNesting,
    CheckpointPolicy, AMROutput, AllLevels, NATIVE_MAX_LEVELS)
from pops.mesh.geometry import Disc, EmbeddedBoundary  # noqa: E402
from pops.mesh.masks import CutCell, NoMask, Staircase  # noqa: E402
from pops.mesh.boundaries import Periodic, Physical, FaceBC, XMin  # noqa: E402


def test_back_compat_and_package_export():
    assert pops.CartesianMesh is CartesianMesh
    assert pops.PolarMesh is PolarMesh
    assert "mesh" in pops.__all__


def test_cartesian_options_and_caps():
    m = CartesianMesh(n=128, L=2.0, periodic=False)
    assert m.options() == {"n": 128, "L": 2.0, "periodic": False}
    assert m.capabilities()["geometry"] == "cartesian"


def test_polar_validation():
    PolarMesh(0.1, 1.0, 8, 16, theta_boxes=4)  # valid
    with pytest.raises(ValueError):
        PolarMesh(1.0, 0.5, 8, 16)  # r_max <= r_min
    with pytest.raises(ValueError):
        PolarMesh(0.1, 1.0, 2, 16)  # nr < 3
    with pytest.raises(ValueError):
        PolarMesh(0.1, 1.0, 8, 16, theta_boxes=5)  # 5 does not divide 16


def test_patch_box_and_layout():
    b = PatchBox(lo=(0, 0), hi=(3, 7))
    assert b.shape == (4, 8)
    with pytest.raises(ValueError):
        PatchBox(lo=(0, 0), hi=(-1, 2))
    layout = BoxLayout([b, PatchBox((4, 0), (7, 7))])
    assert len(layout) == 2


def test_uniform_layout():
    u = Uniform(CartesianMesh(), embedded_boundary=EmbeddedBoundary(Disc(), CutCell()))
    assert u.capabilities()["supports_amr"] is False
    assert "embedded_boundary" in u.options()


def test_amr_route_limits_are_explainable():
    m = CartesianMesh(n=128)
    ok = AMR(base=m, max_levels=NATIVE_MAX_LEVELS, ratio=2, regrid=RegridEvery(20),
             patches=PatchLayout(distribute_coarse=True, coarse_max_grid=32),
             nesting=ProperNesting(buffer=1), checkpoint=CheckpointPolicy(restartable=True))
    assert ok.available().ok
    ok.validate()
    bad = AMR(base=m, max_levels=4)
    av = bad.available()
    assert not av.ok and "max_levels" in av.reason and av.alternatives
    with pytest.raises(ValueError):
        bad.validate()
    with pytest.raises(ValueError):
        AMR(base=m, ratio=3).validate()


def test_typed_refinement_criteria():
    c = Refine.on("rho").above(0.05)
    assert c.options()["predicate"] == "above" and c.threshold == 0.05
    c.validate()
    with pytest.raises(ValueError):
        Refine.on("rho").validate()  # incomplete: no predicate/threshold
    TagUnion(Refine.on("rho").above(0.05),
             Refine.on("phi").gradient_above(0.5)).validate()
    with pytest.raises(TypeError):
        TagUnion("not-a-criterion")


def test_boundaries_and_masks():
    assert Periodic().capabilities()["periodic"] is True
    assert Physical("wall").options()["kind"] == "wall"
    with pytest.raises(ValueError):
        Physical("nope")
    FaceBC(XMin(), Periodic())
    with pytest.raises(TypeError):
        FaceBC("x", Periodic())
    assert CutCell().capabilities()["conservative"] is True
    assert NoMask().capabilities()["masked_transport"] is False
    assert Staircase().capabilities()["conservative"] is False


def test_amr_policies():
    assert FrozenRegrid().options()["frozen"] is True
    assert RegridEvery(20).options()["steps"] == 20
    with pytest.raises(ValueError):
        RegridEvery(0)
    out = AMROutput(fields=["phi"], levels=AllLevels(), include_patch_boxes=True)
    assert out.options()["levels"] == "all"
    assert out.options()["include_patch_boxes"] is True


def test_printable_summaries_are_short_and_stable():
    s = str(AMR(base=CartesianMesh()))
    assert s.startswith("AMR") and len(s) < 200
    assert "CartesianMesh" in repr(CartesianMesh())
    assert str(Refine.on("rho").above(0.05)).startswith("Refine")
    assert str(AuxHalo("foextrap")) == "AuxHalo('foextrap', value=0)"


# The CI python runner invokes each test file as `python3 <file>`; run pytest on this
# module so the assertions execute (a bare import would only define the test functions).
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
