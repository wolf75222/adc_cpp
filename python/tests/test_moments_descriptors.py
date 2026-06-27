#!/usr/bin/env python3
"""Spec 5 sec.6 (ADC-498): the route-choosing pops.moments objects are typed descriptors.

The moment toolkit exposes a mix of objects. A handful CHOOSE a math algorithm -- the
wave-speed strategy (:class:`ExactSpeeds`), the realizability-floor strategy
(:class:`RealizabilityProjection`), the magnetic-source binding
(:class:`MagneticMomentSource`) and the closure variant (:class:`HyQMOM15Closure`). Spec 5
sec.6 requires every such route chooser to be an inert, inspectable
:class:`pops.descriptors.Descriptor` that declares its options / capabilities and answers
``available(context)`` with an explainable status.

The rest only construct or hold structure (``MomentModel`` / ``MomentBasis`` / the binomial
transforms / ``MomentOrdering`` / ``VlasovSources`` / ``MomentHierarchy``); they make no
algorithm choice, so they stay lightweight handles and are NOT descriptors. ``MomentOrdering``
in particular has a single forced layout, so it is a handle, not a route chooser.

Pure Python: only ``import pops`` (the compiled _pops loads as a side effect, but nothing
here builds or runs a moment model).
"""
import sys

import pytest

pops = pytest.importorskip("pops")

from pops import moments  # noqa: E402
from pops.descriptors import Availability, Descriptor, DescriptorProtocol  # noqa: E402


def test_exact_speeds_descriptor_contract():
    speeds = moments.ExactSpeeds(moments.ExactSpeeds.ROE_DISSIPATION)
    assert isinstance(speeds, (Descriptor, DescriptorProtocol))
    assert speeds.name == "ExactSpeeds"
    assert speeds.category == "wave_speed"
    assert speeds.options()["kind"] == moments.ExactSpeeds.ROE_DISSIPATION
    caps = speeds.capabilities()
    assert caps["exact_speeds"] is True and caps["roe"] is True
    assert speeds.available().ok
    assert speeds.validate() is True
    # The BOUNDED strategy turns the engine flags off (still a valid, available route).
    bounded = moments.ExactSpeeds(moments.ExactSpeeds.BOUNDED)
    assert bounded.capabilities() == {"exact_speeds": False, "roe": False}
    # from_flags round-trips to the same descriptor kind.
    assert moments.ExactSpeeds.from_flags(True, True).options()["kind"] == \
        moments.ExactSpeeds.ROE_DISSIPATION
    with pytest.raises(ValueError):
        moments.ExactSpeeds("nope")


def test_realizability_projection_descriptor_contract():
    proj = moments.RealizabilityProjection(eps_m00=1e-10, eps_cov=1e-9, robust=False)
    assert isinstance(proj, (Descriptor, DescriptorProtocol))
    assert proj.name == "RealizabilityProjection"
    assert proj.category == "realizability"
    opts = proj.options()
    assert opts["eps_m00"] == 1e-10 and opts["eps_cov"] == 1e-9 and opts["robust"] is False
    assert proj.capabilities()["guard_level"] == "bare"
    assert moments.RealizabilityProjection().capabilities()["guard_level"] == "smooth"
    assert proj.validate() is True
    # The .none() preset is the bare guard-free route.
    assert moments.RealizabilityProjection.none().options()["robust"] is False


def test_magnetic_moment_source_descriptor_contract():
    src = moments.MagneticMomentSource(q_over_m="my_q", b_field="my_b")
    assert isinstance(src, (Descriptor, DescriptorProtocol))
    assert src.name == "MagneticMomentSource"
    assert src.category == "moment_source"
    assert src.options() == {"q_over_m": "my_q", "b_field": "my_b"}
    assert src.capabilities()["provides"] == "magnetic_lorentz"
    assert src.validate() is True
    # The builder side stays: as_sources() returns a (m, M) -> list callable.
    assert callable(src.as_sources(2.0))


def test_hyqmom15_closure_descriptor_contract():
    closure = moments.HyQMOM15Closure(variant="levermore")
    assert isinstance(closure, (Descriptor, DescriptorProtocol))
    assert closure.name == "HyQMOM15Closure"
    assert closure.category == "closure"
    assert closure.order == 4
    opts = closure.options()
    assert opts["variant"] == "levermore" and opts["order"] == 4
    assert closure.capabilities()["provides"] == "order_4_standardized_moments"
    assert closure.validate() is True
    # The descriptor is still the closure callable (Spec 5 sec.6 does not change its role).
    standardized = {"S11": 0.1, "S20": 1.0, "S02": 1.0, "S30": 0.0, "S21": 0.0,
                    "S12": 0.0, "S03": 0.0, "S40": 3.0, "S31": 0.0, "S22": 1.0,
                    "S13": 0.0, "S04": 3.0}
    out = closure(standardized)
    assert set(out) == {"S%d%d" % (p, 5 - p) for p in range(6)}
    # The unvalidated custom variant is gated (ship-authoring / gate-runtime pattern).
    with pytest.raises(NotImplementedError):
        moments.HyQMOM15Closure(variant="custom")


def test_route_choosers_available_is_explainable():
    # Every moments descriptor answers available() with an Availability, never a bare bool.
    for descriptor in (moments.ExactSpeeds(), moments.RealizabilityProjection(),
                       moments.MagneticMomentSource(), moments.HyQMOM15Closure()):
        status = descriptor.available()
        assert isinstance(status, Availability)
        assert not isinstance(status, bool)
        assert status.ok is True


def test_handles_are_not_descriptors():
    # The builders / handles construct or hold; they choose no route, so they are not
    # descriptors. MomentOrdering is a single forced layout -> a handle, not a route chooser.
    handles = (
        moments.MomentOrdering(),
        moments.MomentBasis(order=2),
        moments.CenteredTransform(order=2),
        moments.StandardizedTransform(order=2),
        moments.CartesianVelocityMoments(order=2),
        moments.CartesianVelocityMoments(order=2).hierarchy(),
        moments.VlasovSources,
    )
    for handle in handles:
        name = getattr(handle, "__name__", type(handle).__name__)
        assert not isinstance(handle, Descriptor), (
            "%s is a builder/handle and must not be a Descriptor" % name)


def test_hierarchy_snapshot_exposes_inspectable_descriptors():
    # The MomentHierarchy snapshot carries the speeds / projection descriptors; they remain
    # inspectable route choosers even when reached through the snapshot.
    model = (moments.CartesianVelocityMoments(order=2)
             .add_numerics(roe=True)
             .set_realizability(moments.RealizabilityProjection.none()))
    snapshot = model.hierarchy()
    assert isinstance(snapshot.speeds, Descriptor)
    assert snapshot.speeds.options()["kind"] == moments.ExactSpeeds.ROE_DISSIPATION
    assert isinstance(snapshot.projection, Descriptor)
    assert snapshot.projection.capabilities()["guard_level"] == "bare"


# The CI python runner invokes each test file as `python3 <file>`; run pytest on this module
# so the assertions execute (a bare import would only define the test functions).
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
