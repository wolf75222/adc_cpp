"""Spec 5 sec.8.15 / criterion 22: the typed compile-backend + execution-platform descriptors.

These checks pin the typed backend/platform surface added under epic ADC-479:

  - the backend descriptors (Production / AOT / JIT) lower to the legacy backend string
    ("production" / "aot" / "prototype") the compile drivers already key on, and expose the same
    token via ``.scheme``;
  - ``lower_backend`` is ADDITIVE and TRANSPARENT: a typed descriptor lowers to its string, while a
    plain string / None / any other value passes through unchanged so the compile driver's existing
    ``backend not in _BACKENDS`` guard stays the single source of the unknown-backend ValueError;
  - the consumer (``compile_problem`` / ``compile_model``) accepts BOTH a string and a typed
    backend -- a typed AOT() hits the SAME production-only guard as the string "aot", proving the
    lowering runs before the guard;
  - ``Production(platform=KokkosOpenMP())`` records the platform (inert) and refuses a string;
  - the platform descriptors (KokkosSerial / KokkosOpenMP / KokkosCuda / KokkosHIP / MPI) declare
    host/gpu/mpi capabilities and answer ``available()`` with an EXPLAINABLE Availability that names
    a missing build flag, never a bare bool.

Pure Python: it imports the inert authoring/codegen packages (the compiled _pops loads as a side
effect of ``import pops`` -- platform availability reads its build flags -- but no model is built
or run, and no compiler is invoked).
"""
import sys

import pytest

pops = pytest.importorskip("pops")

from pops.codegen import AOT, JIT, Production, lower_backend  # noqa: E402
from pops.codegen.backends import BACKEND_DESCRIPTORS, _Backend  # noqa: E402
from pops.descriptors import Availability, Descriptor  # noqa: E402
from pops.runtime.platforms import (  # noqa: E402
    KokkosCuda, KokkosHIP, KokkosOpenMP, KokkosSerial, MPI)


# --- backend descriptors lower to the legacy string -------------------------------------------
def test_backend_descriptors_lower_to_legacy_string():
    assert Production().lower() == "production"
    assert AOT().lower() == "aot"
    assert JIT().lower() == "prototype"


def test_backend_scheme_matches_lower():
    for cls in (Production, AOT, JIT):
        desc = cls()
        assert desc.scheme == desc.lower()


def test_backend_descriptor_is_inert_typed_descriptor():
    prod = Production()
    assert isinstance(prod, Descriptor)
    assert isinstance(prod, _Backend)
    assert prod.category == "backend"
    # capabilities come from the honest native backend table (cpu/mpi/amr/gpu).
    caps = prod.capabilities()
    assert caps["cpu"] is True and caps["mpi"] is True and caps["amr"] is True
    assert JIT().capabilities()["mpi"] is False
    # inspect is a plain dict carrying the platform slot.
    record = prod.inspect()
    assert record["category"] == "backend"
    assert record["options"]["backend"] == "production"
    assert record["platform"] is None


def test_backend_registry_maps_token_to_class():
    assert BACKEND_DESCRIPTORS["production"] is Production
    assert BACKEND_DESCRIPTORS["aot"] is AOT
    assert BACKEND_DESCRIPTORS["prototype"] is JIT


# --- lower_backend is additive (string AND typed) ---------------------------------------------
def test_lower_backend_passes_string_through():
    for token in ("production", "aot", "prototype", "auto"):
        assert lower_backend(token) == token


def test_lower_backend_lowers_typed():
    assert lower_backend(Production()) == "production"
    assert lower_backend(AOT()) == "aot"
    assert lower_backend(JIT()) == "prototype"


def test_lower_backend_passes_non_descriptor_through():
    # lower_backend is a TRANSPARENT coercion: it lowers a typed descriptor and returns anything
    # else (None, a wrong type, an unknown string) UNCHANGED, so the compile entry point's existing
    # `backend not in _BACKENDS` guard stays the single source of the unknown-backend ValueError.
    # A guardrail such as test_dsl_compile_facade passes backend=None expecting that ValueError, so
    # lower_backend must NOT pre-empt it with a TypeError of its own.
    assert lower_backend(None) is None
    assert lower_backend(123) == 123
    assert lower_backend("nope") == "nope"


# --- the consumer accepts BOTH a string and a typed backend -----------------------------------
def _guard_error(backend):
    """Run compile_problem far enough to hit the backend guard; return the ValueError text."""
    from pops.codegen.compile_drivers import compile_problem
    try:
        compile_problem(model=None, time=None, backend=backend)
    except ValueError as err:
        return str(err)
    raise AssertionError("compile_problem(backend=%r) did not raise" % (backend,))


def test_compile_problem_accepts_string_and_typed_production():
    # Both reach PAST the production-only guard and fail identically at the time= check.
    string_msg = _guard_error("production")
    typed_msg = _guard_error(Production())
    assert "time must be" in string_msg
    assert string_msg == typed_msg


def test_compile_problem_typed_non_production_hits_same_guard_as_string():
    # A typed AOT() lowers to "aot" BEFORE the production-only guard -> the SAME message the
    # string "aot" produces (proving the additive lowering runs first).
    string_msg = _guard_error("aot")
    typed_msg = _guard_error(AOT())
    assert string_msg == "compiled time programs require backend='production'"
    assert typed_msg == string_msg


def test_compile_problem_accepts_typed_backend_with_platform():
    # Recording a platform must not change the lowered backend string.
    msg = _guard_error(Production(platform=KokkosOpenMP()))
    assert "time must be" in msg


def test_compile_model_lowers_typed_backend_past_unknown_guard():
    # compile_model's unknown-backend guard sees the LOWERED string, so a typed JIT() is not
    # rejected as unknown; it proceeds and trips later on the fake model (no .name attribute).
    from pops.codegen.compile_drivers import compile_model

    class _FakeModel:
        def _check_require_metadata(self, *a, **k):
            pass

    with pytest.raises(AttributeError):
        compile_model(_FakeModel(), backend=JIT())
    # A genuinely unknown string still raises the unknown-backend ValueError (additive, not lossy).
    with pytest.raises(ValueError, match="backend 'nope' unknown"):
        compile_model(_FakeModel(), backend="nope")


# --- Production(platform=...) records the platform / refuses a string --------------------------
def test_production_records_platform():
    prod = Production(platform=KokkosOpenMP())
    assert prod.platform is not None
    assert prod.options()["platform"] == "KokkosOpenMP"
    assert prod.inspect()["platform"]["options"]["device"] == "openmp"
    # The platform never changes the backend token.
    assert prod.lower() == "production"


def test_backend_refuses_string_platform():
    with pytest.raises(TypeError, match="platform must be a typed"):
        Production(platform="openmp")


# --- platform descriptors: capabilities + explainable availability ----------------------------
def test_platform_descriptors_declare_capabilities():
    assert KokkosSerial().capabilities() == {"host": True, "gpu": False, "mpi": False}
    assert KokkosOpenMP().capabilities()["host"] is True
    assert KokkosCuda().capabilities()["gpu"] is True
    assert KokkosHIP().capabilities()["gpu"] is True
    assert MPI().capabilities()["mpi"] is True
    for cls in (KokkosSerial, KokkosOpenMP, KokkosCuda, KokkosHIP, MPI):
        desc = cls()
        assert desc.category == "platform"
        assert isinstance(desc, Descriptor)
        assert desc.options()["device"]


def test_platform_available_is_explainable():
    # available() always returns an Availability (never a bare bool), with a reason.
    for cls in (KokkosSerial, KokkosOpenMP, KokkosCuda, KokkosHIP, MPI):
        status = cls().available()
        assert isinstance(status, Availability)
        assert status.status in ("yes", "no", "partial")
        assert status.reason


def test_serial_platform_always_available():
    assert KokkosSerial().available().ok


def test_unavailable_platform_explains_missing_build_flag():
    # On a build that lacks a flag, available() is "no"/"partial" and NAMES the missing flag +
    # an alternative. The exact verdict depends on the loaded _pops build, so assert the contract,
    # not a fixed verdict: a non-yes status must carry a reason and either missing or alternatives.
    has_mpi = getattr(pops._pops, "__has_mpi__", None)
    mpi_status = MPI().available()
    if has_mpi is False:
        assert mpi_status.status == "no"
        assert "MPI" in mpi_status.reason
        assert mpi_status.missing  # names the build flag
        assert mpi_status.alternatives
    # A GPU device on a non-GPU build is never a false "yes".
    for cls in (KokkosCuda, KokkosHIP):
        status = cls().available()
        if not status.ok:
            assert status.reason
            assert status.missing or status.alternatives


def test_platform_lower_is_inert_metadata():
    record = MPI().lower()
    assert record["category"] == "platform"
    assert record["device"] == "mpi"
    assert record["capabilities"]["mpi"] is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
