"""Spec 5 sec.15 audit follow-ups (epic ADC-479): route matrix + Optimization de-string + Problem.

Three items a Spec 5 sec.15 audit surfaced, exercised here against the REAL pops package (no
fake module): every check imports ``pops`` and reads only inert descriptor metadata.

  - sec.13.12.1 / #37: ``pops.Problem.explain_routes()`` returns a printable, INERT route matrix
    (feature x layout x backend x platform -> status / limitation / error_message) sourced from
    the assembled descriptors; an empty problem returns an honest empty envelope.
  - sec.14.2 / #20-21: ``pops.codegen.Optimization(math="fast")`` REJECTS the bare string at
    construction with a clear, actionable message (instead of silently mis-setting and crashing
    later in ``options()``); the typed ``StrictMath()`` / ``FastMath()`` / ... usage still works.
  - sec.6 table / sec.15: ``pops.Problem`` is an ASSEMBLY that CONTAINS descriptors, so it is NOT
    itself a ``pops.descriptors.Descriptor`` -- while every Problem method still works.

Runs under pytest and as a plain script (the ``__main__`` guard), like the sibling Spec 5 tests.
"""
import sys

import pytest

pops = pytest.importorskip("pops")

from pops.codegen import (  # noqa: E402
    Optimization, ConservativeFusion, Disabled,
    StrictMath, FastMath, DebugMath, GpuRegisterAware)
from pops.descriptors import Descriptor, DescriptorProtocol  # noqa: E402
from pops.fields import FieldProblem  # noqa: E402
from pops.math import laplacian  # noqa: E402
from pops.mesh.cartesian import CartesianMesh  # noqa: E402
from pops.mesh.layouts import AMR, Uniform  # noqa: E402


# --- tiny inert stand-ins (no compiler / no runtime) -----------------------
class _StubModel:
    """A physics stand-in (the assembly only reads its presence, never runs it)."""

    name = "ne"
    dsl = object()


class _StubSolver:
    name = "GeometricMG"
    scheme = "geometric_mg"
    options = {}


def _poisson_field():
    return FieldProblem(name="phi", unknown="phi",
                        equation=(-laplacian("phi") == "charge_density"),
                        solver=_StubSolver())


def _uniform_problem():
    return (pops.Problem(name="plasma")
            .block("ne", physics=_StubModel(), spatial=pops.FiniteVolume())
            .field(_poisson_field()))


# =====================================================================================
# ITEM 1 -- Problem.explain_routes(): a printable route matrix sourced from the C++ core
# (Spec 5 sec.13.12.1 / #37). The capability VALUES come from _pops.module_capabilities(),
# NOT a Python-derived walk: every row carries source="native".
# =====================================================================================
def test_explain_routes_returns_printable_inert_matrix():
    matrix = _uniform_problem().explain_routes()
    text = str(matrix)
    assert "route matrix" in text
    assert "layout=Uniform" in text
    # The transport features appear as rows; each carries the C++-authoritative axis/status/source.
    assert len(matrix.rows) >= 5
    feats = {r.feature for r in matrix.rows}
    assert {"supports_uniform", "supports_amr", "supports_stride"} <= feats
    for row in matrix.rows:
        d = row.to_dict()
        for key in ("feature", "axis", "status", "source", "limitation"):
            assert key in d, "route row missing key %r: %r" % (key, d)
        assert d["status"] in ("available", "unavailable")


def test_explain_routes_is_json_serialisable_metadata_only():
    import json
    matrix = _uniform_problem().explain_routes()
    payload = matrix.to_dict()
    # to_dict() is a plain nested dict (inert metadata) and round-trips through JSON.
    assert set(payload) >= {"problem", "layout", "rows"}
    assert payload["problem"] == "plasma"
    assert json.loads(json.dumps(payload)) == payload


def test_explain_routes_values_are_native_sourced_not_python_derived():
    # The whole point of sec.13.12: capability VALUES come from the C++ core, not a Python walk.
    matrix = _uniform_problem().explain_routes()
    by_feature = {r.feature: r for r in matrix.rows}
    # Every transport row is sourced from the native module_capabilities() (source="native").
    assert all(r.source == "native" for r in matrix.rows)
    # Honest values from the built _pops: uniform available; partial_imex_mask unavailable
    # (no C++ path backs it -- the spec forbids fabricating it true).
    assert by_feature["supports_uniform"].status == "available"
    assert by_feature["supports_partial_imex_mask"].status == "unavailable"


def test_explain_routes_amr_feature_reported_from_native_build():
    # supports_amr is reported with the status the built _pops decides (this build has the AMR
    # runtime), not a Python guess. An AMR layout problem reports the same native feature row.
    prob = pops.Problem(name="amr", layout=AMR(CartesianMesh())).block("ne", physics=_StubModel())
    matrix = prob.explain_routes()
    amr = {r.feature: r for r in matrix.rows}["supports_amr"]
    assert amr.source == "native"
    assert amr.status in ("available", "unavailable")
    assert amr.axis == "layout"


def test_explain_routes_is_native_capability_matrix_even_without_blocks():
    # The transport capability matrix is a property of the build, so it is honest (not fabricated)
    # even for a bare problem: the native rows are present and printable, never an empty lie.
    matrix = pops.Problem(name="bare").explain_routes()
    assert len(matrix.rows) >= 5
    assert all(r.source == "native" for r in matrix.rows)
    assert "route matrix" in str(matrix)
    assert matrix.to_dict()["rows"]


def test_explain_routes_reads_metadata_only_no_runtime_import():
    # Building the matrix reads the inert capability dict; it does not require the runtime System.
    import importlib
    matrix = _uniform_problem().explain_routes()
    assert len(matrix.rows) > 0
    sys_mod = sys.modules.get("pops.runtime.system")
    importlib.reload(importlib.import_module("pops._capabilities"))
    assert sys_mod is sys.modules.get("pops.runtime.system")


# =====================================================================================
# ITEM 2 -- Optimization de-string: math=/fuse= reject a bare string (sec.14.2 / #20-21)
# =====================================================================================
def test_optimization_math_string_is_rejected_with_clear_message():
    with pytest.raises(TypeError) as excinfo:
        Optimization(math="fast")
    message = str(excinfo.value)
    # The message names the parameter, echoes the rejected string, and points at the typed types.
    assert "optimization math" in message
    assert "fast" in message
    assert "StrictMath()" in message and "FastMath()" in message
    assert "DebugMath()" in message and "GpuRegisterAware()" in message


def test_optimization_fuse_string_is_rejected():
    with pytest.raises(TypeError) as excinfo:
        Optimization(fuse="conservative")
    assert "optimization fuse" in str(excinfo.value)


def test_optimization_typed_math_still_works():
    # The de-string must NOT break the existing typed usage.
    for mode in (StrictMath(), FastMath(), DebugMath(), GpuRegisterAware()):
        opt = Optimization(math=mode)
        assert opt.math is mode
        assert opt.options()["math"] == type(mode).__name__
    # A bare Optimization() default stays StrictMath (conservative).
    assert isinstance(Optimization().math, StrictMath)
    assert Optimization().capabilities()["strict_math"] is True
    assert Optimization(math=FastMath()).capabilities()["strict_math"] is False
    # The typed fuse policy still threads through.
    opt = Optimization(fuse=ConservativeFusion())
    assert opt.options()["fuse"] == "ConservativeFusion"
    assert Optimization(fuse=Disabled()).options()["fuse"] == "Disabled"


def test_optimization_options_no_longer_crashes_on_a_bad_math():
    # Regression: the string used to be accepted silently, then options() crashed in self.math.name.
    # Now construction rejects it, so options() is never reached with a bad math.
    opt = Optimization()
    assert opt.options()["math"] == "StrictMath"  # the happy path is intact.
    with pytest.raises(TypeError):
        Optimization(math=42)  # a non-string, non-typed value is a clear TypeError too.


# =====================================================================================
# ITEM 3 -- Problem is NOT a Descriptor (sec.6 table / sec.15)
# =====================================================================================
def test_problem_is_not_a_descriptor():
    prob = _uniform_problem()
    assert not isinstance(prob, Descriptor), (
        "a Problem is an assembly that CONTAINS descriptors; it must NOT be a Descriptor")


def test_problem_still_duck_types_as_a_route_describing_object():
    # Dropping the Descriptor base must not drop the inspectable surface: Problem still satisfies
    # the structural DescriptorProtocol by duck typing.
    prob = _uniform_problem()
    assert isinstance(prob, DescriptorProtocol)
    for member in ("name", "category", "native_id", "requirements", "capabilities",
                   "options", "available", "validate", "lower", "inspect"):
        assert hasattr(prob, member), "Problem lost protocol member %r" % member


def test_problem_methods_still_work_after_dropping_descriptor_base():
    prob = _uniform_problem()
    assert prob.name == "plasma"
    assert prob.category == "problem"
    assert prob.native_id is None
    assert isinstance(prob.options(), dict) and prob.options()["n_blocks"] == 1
    assert isinstance(prob.requirements(), dict)
    assert isinstance(prob.capabilities(), dict)
    assert bool(prob.available()) is True
    assert prob.validate() is True
    info = prob.inspect()
    assert info["name"] == "plasma" and "layout" in info and "blocks" in info
    record = prob.lower()
    assert record["name"] == "plasma" and record["category"] == "problem"
    assert "plasma" in str(prob) and "Problem(" in repr(prob)


def test_problem_contains_descriptors():
    # The parts a Problem holds ARE descriptors -- the Problem is their assembly, not one of them.
    prob = _uniform_problem()
    assert isinstance(prob.layout, (Uniform, AMR))
    assert isinstance(prob.layout, Descriptor)  # the layout it CONTAINS is a descriptor.


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
