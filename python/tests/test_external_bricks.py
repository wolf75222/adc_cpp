"""Spec 3 external C++ bricks (ADC-463, criterion 20).

A Spec 3 brick is native / generated / macro / external-C++. These tests cover
the last category: ``adc.lib.load_cpp_library(path)`` dlopens a user ``.so``,
reads its JSON manifest (over the C++ ``BrickRegistry``), and registers the ids
in an in-process catalog; ``adc.lib.riemann.User(id)`` / ``adc.lib.external(id)``
then surface an ``external_cpp`` descriptor carrying the manifest's requirements.
An id that was never loaded raises a CLEAR error.

The manifest-parsing seam (``_register_manifest``) is exercised directly so the
test needs no compiled ``.so``; ``load_cpp_library`` is the real ``.so`` path on
top of it. The real ``adc.lib`` functions are used -- adc is never faked.
"""
import json

import pytest

lib = pytest.importorskip("adc.lib")


@pytest.fixture(autouse=True)
def _clean_catalog():
    """Reset the in-process external-brick catalog around each test (no leakage)."""
    lib._clear_external_catalog()
    yield
    lib._clear_external_catalog()


def _manifest(*entries):
    return json.dumps({"bricks": list(entries)})


def test_unknown_external_id_raises_clear_error():
    # Not loaded -> a clear, actionable error naming the id and load_cpp_library.
    with pytest.raises(LookupError) as exc:
        lib.riemann.User("my_hllc")
    msg = str(exc.value)
    assert "my_hllc" in msg
    assert "not loaded" in msg
    assert "load_cpp_library" in msg


def test_generic_external_unknown_id_raises():
    with pytest.raises(LookupError) as exc:
        lib.external("nope")
    assert "nope" in str(exc.value)
    assert "load_cpp_library" in str(exc.value)


def test_register_manifest_then_user_surfaces_external_descriptor():
    n = lib._register_manifest(_manifest(
        {"id": "my_hllc", "category": "riemann",
         "requirements": "pressure,wave_speeds", "capabilities": "physical_flux"}))
    assert n == 1
    d = lib.riemann.User("my_hllc")
    assert d.brick_type == "external_cpp"
    assert d.category == "riemann"
    assert d.native_id == "my_hllc"
    # The CSV requirements/capabilities become list metadata on the descriptor.
    assert d.requirements == {"capabilities": ["pressure", "wave_speeds"]}
    assert d.capabilities == {"provides": ["physical_flux"]}


def test_generic_external_surfaces_descriptor_with_its_category():
    lib._register_manifest(_manifest(
        {"id": "my_precond", "category": "preconditioner", "requirements": ""}))
    d = lib.external("my_precond")
    assert d.brick_type == "external_cpp"
    assert d.category == "preconditioner"
    assert d.native_id == "my_precond"
    # No requirements -> empty metadata, never a fabricated capability.
    assert d.requirements == {}
    assert d.capabilities == {}


def test_user_category_must_match_when_registered_elsewhere():
    # Registered as a preconditioner; selecting it via riemann.User is a loud mismatch.
    lib._register_manifest(_manifest(
        {"id": "x", "category": "preconditioner", "requirements": ""}))
    with pytest.raises(ValueError) as exc:
        lib.riemann.User("x")
    assert "preconditioner" in str(exc.value)
    assert "riemann" in str(exc.value)


def test_manifest_must_be_well_formed():
    with pytest.raises(ValueError):
        lib._register_manifest("not json")
    with pytest.raises(ValueError):
        # An entry missing its id is rejected (no silently-dropped brick).
        lib._register_manifest(_manifest({"category": "riemann"}))


def test_load_cpp_library_rejects_a_missing_path():
    with pytest.raises((OSError, ValueError)):
        lib.load_cpp_library("/no/such/brick.so")
