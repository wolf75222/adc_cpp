"""Spec 5 Phase D (ADC-483): the formal DescriptorProtocol + capability matrix + manifest reader.

These checks pin the Spec 5 sec.6 / sec.7 descriptor contract:

  - the two descriptor families (the Descriptor base / its mesh subclass, and the older
    attribute-based BrickDescriptor) both satisfy the documented DescriptorProtocol;
  - available() on the Descriptor family returns an explainable Availability, never a bare bool;
  - lower() returns an INERT metadata dict and never raises for a valid descriptor (it must not
    run a numeric loop or touch the runtime);
  - inspect_capabilities() returns a descriptor-sourced matrix whose rows carry the required keys;
  - reject_string_selector() raises a clear TypeError with the spec message;
  - read_manifest() reads a brick manifest's metadata WITHOUT registering it;
  - Availability is now the SAME class in pops.descriptors and pops.mesh._descriptor (the
    duplicate the reviewers flagged is gone).

Pure Python: it only imports the inert authoring packages (the compiled _pops loads as a side
effect of ``import pops`` but no model is built or run).
"""
import json
import os
import sys
import tempfile

import pytest

pops = pytest.importorskip("pops")

from pops.descriptors import (  # noqa: E402
    Availability, BrickDescriptor, Descriptor, DescriptorProtocol, reject_string_selector)
from pops.numerics.riemann import HLL  # noqa: E402
from pops.mesh import CartesianMesh  # noqa: E402
import pops.mesh._descriptor as mesh_descriptor  # noqa: E402
from pops import moments  # noqa: E402

# The semantic members the DescriptorProtocol documents (Spec 5 sec.6).
PROTOCOL_MEMBERS = (
    "name", "category", "native_id",
    "requirements", "capabilities", "options",
    "available", "validate", "lower", "inspect",
)

# Spec 5 sec.6 (ADC-498): the route-CHOOSING objects of pops.moments are descriptors -- they
# pick a math algorithm (wave-speed strategy / realizability floor / magnetic source binding /
# closure variant). Each must honour the same DescriptorProtocol. Paired with its category so
# the loop also pins the declared category string.
MOMENTS_ROUTE_CHOOSERS = (
    (moments.ExactSpeeds(moments.ExactSpeeds.ROE_DISSIPATION), "wave_speed"),
    (moments.RealizabilityProjection(eps_m00=1e-10, robust=False), "realizability"),
    (moments.MagneticMomentSource(q_over_m="my_q", b_field="my_b"), "moment_source"),
    (moments.HyQMOM15Closure(variant="levermore"), "closure"),
)

# The builders / handles of pops.moments do NOT choose a route, so they stay lightweight and
# must NOT be descriptors (Spec 5 sec.6: only algorithm-choosing objects are descriptors).
# MomentOrdering has a single forced layout (no choice), so it is a handle, not a descriptor.
MOMENTS_HANDLES = (
    moments.MomentOrdering(),
    moments.MomentBasis(order=2),
    moments.CenteredTransform(order=2),
    moments.StandardizedTransform(order=2),
    moments.CartesianVelocityMoments(order=2),
    moments.CartesianVelocityMoments(order=2).hierarchy(),
    moments.VlasovSources,
)


def test_descriptor_family_satisfies_protocol():
    # The Descriptor base (via a concrete mesh subclass) honours the full protocol surface.
    mesh = CartesianMesh(n=8)
    for member in PROTOCOL_MEMBERS:
        assert hasattr(mesh, member), "Descriptor family missing protocol member %r" % member
    assert isinstance(mesh, DescriptorProtocol)


def test_brick_descriptor_satisfies_protocol():
    # The older attribute-based BrickDescriptor also satisfies the protocol (additive Phase D).
    brick = HLL()
    assert isinstance(brick, BrickDescriptor)
    for member in PROTOCOL_MEMBERS:
        assert hasattr(brick, member), "BrickDescriptor missing protocol member %r" % member
    assert isinstance(brick, DescriptorProtocol)


def test_moments_route_choosers_satisfy_protocol():
    # ADC-498: every route-choosing pops.moments object honours the full protocol surface,
    # exposes its declared category, and returns an inert inspect() dict that validate()s.
    for descriptor, category in MOMENTS_ROUTE_CHOOSERS:
        assert isinstance(descriptor, Descriptor), (
            "%s should be a pops.descriptors.Descriptor" % type(descriptor).__name__)
        assert isinstance(descriptor, DescriptorProtocol), (
            "%s does not satisfy DescriptorProtocol" % type(descriptor).__name__)
        for member in PROTOCOL_MEMBERS:
            assert hasattr(descriptor, member), (
                "%s missing protocol member %r" % (type(descriptor).__name__, member))
        assert descriptor.category == category
        record = descriptor.inspect()
        assert isinstance(record, dict)
        assert record["name"] == descriptor.name
        assert record["category"] == category
        assert isinstance(record["options"], dict) and record["options"]
        assert isinstance(record["requirements"], dict)
        assert isinstance(record["capabilities"], dict)
        # available() is an explainable Availability and validate() does not raise.
        assert isinstance(descriptor.available(), Availability)
        assert descriptor.validate() is True
        # lower() is inert metadata, never a numeric loop.
        assert descriptor.lower()["category"] == category


def test_moments_handles_are_not_descriptors():
    # ADC-498: the builders / handles (MomentModel, MomentBasis, transforms, ordering, ...)
    # are NOT route choosers, so they stay lightweight and must not be descriptors.
    for handle in MOMENTS_HANDLES:
        name = getattr(handle, "__name__", type(handle).__name__)
        assert not isinstance(handle, Descriptor), (
            "%s is a builder/handle, not a route chooser; it must not be a Descriptor" % name)


def test_hyqmom15_closure_is_still_a_callable_closure():
    # ADC-498: making HyQMOM15Closure a descriptor must not break its closure-callable role.
    closure = moments.HyQMOM15Closure()
    standardized = {"S11": 0.25, "S20": 1.0, "S02": 1.0, "S30": 0.0, "S21": 0.0,
                    "S12": 0.0, "S03": 0.0, "S40": 3.0, "S31": 0.0, "S22": 1.0,
                    "S13": 0.0, "S04": 3.0}
    out = closure(standardized)
    assert set(out) == {"S%d%d" % (p, 5 - p) for p in range(6)}


def test_available_returns_availability_not_bool():
    # On the Descriptor family, available() is an explainable Availability, never a bare bool.
    status = CartesianMesh(n=8).available()
    assert isinstance(status, Availability)
    assert not isinstance(status, bool)
    assert status.status in ("yes", "no", "partial")
    assert status.ok is True  # a plain mesh is unconditionally available.


def test_lower_is_inert_dict_and_never_raises():
    # lower() returns a metadata dict for a valid descriptor and never raises (no numeric loop).
    for descriptor in (CartesianMesh(n=8), HLL()):
        record = descriptor.lower()
        assert isinstance(record, dict)
        assert record["name"] == descriptor.name
        assert record["category"] == descriptor.category
        assert "native_id" in record
        assert "options" in record


def test_brick_descriptor_native_id_carried_in_lowering():
    # A native brick lowers with its real C++ symbol; a planned brick lowers with no symbol.
    assert HLL().lower()["native_id"] == "pops::HLLFlux"
    from pops.numerics.reconstruction.limiters import MC  # planned, no native type yet.
    assert MC().lower()["native_id"] in (None, "")


def test_inspect_capabilities_rows_have_required_keys():
    matrix = pops.inspect_capabilities()
    assert len(matrix) > 0
    seen_categories = set()
    for entry in matrix:
        row = entry.to_dict()
        for key in ("name", "category", "native_id", "available", "requirements"):
            assert key in row, "capability row missing key %r: %r" % (key, row)
        seen_categories.add(row["category"])
    # The descriptor-sourced matrix covers the discretisation + mesh + lib catalogs.
    assert {"riemann", "reconstruction", "limiter", "projection", "layout"} <= seen_categories
    # It is printable and JSON-serialisable (inert metadata only).
    assert "capability matrix" in str(matrix)
    assert isinstance(matrix.to_dict(), dict)


def test_inspect_capabilities_is_descriptor_sourced():
    # The matrix reports the native bricks as available with their real symbols.
    matrix = pops.inspect_capabilities()
    riemann = {e.name: e for e in matrix.by_category("riemann")}
    assert riemann["hll"].native_id == "pops::HLLFlux"
    assert riemann["hll"].available == "yes"


def test_reject_string_selector_raises():
    with pytest.raises(TypeError) as excinfo:
        reject_string_selector("hll", "riemann", "pops.numerics.riemann.HLL()")
    message = str(excinfo.value)
    assert "String algorithm selector rejected" in message
    assert "riemann='hll'" in message
    assert "pops.numerics.riemann.HLL()" in message


def test_availability_is_unified_single_class():
    # Spec 5 Phase D: the parallel mesh Availability duplicate is gone; it is now re-exported.
    assert mesh_descriptor.Availability is Availability
    # MeshDescriptor is a subclass of the shared Descriptor base.
    assert issubclass(mesh_descriptor.MeshDescriptor, Descriptor)
    assert isinstance(CartesianMesh(n=8), Descriptor)


def test_read_manifest_reads_without_registering():
    from pops.external import read_manifest, CompiledManifest
    from pops import descriptors as desc
    manifest = {
        "abi_key": "pops-test-abi",
        "bricks": [
            {"id": "my_flux", "category": "riemann",
             "requirements": "physical_flux,wave_speeds", "capabilities": "provides_x"},
        ],
    }
    desc._clear_external_catalog()
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        result = read_manifest(path)
        assert isinstance(result, CompiledManifest)
        assert result.ids == ["my_flux"]
        assert result.abi_key == "pops-test-abi"
        assert result.categories == ["riemann"]
        assert result.bricks[0]["requirements"] == ["physical_flux", "wave_speeds"]
        assert result.bricks[0]["capabilities"] == ["provides_x"]
        # read_manifest is INSPECTION ONLY: it did NOT register the brick in the catalog.
        with pytest.raises(LookupError):
            desc.external("my_flux")
        assert isinstance(result.to_dict(), dict)
    finally:
        os.remove(path)
        desc._clear_external_catalog()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
