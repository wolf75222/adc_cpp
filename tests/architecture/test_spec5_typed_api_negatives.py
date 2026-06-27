"""Spec 5 (sec.15): the typed-object API rejects strings and stays inspectable.

Spec 5 ("Python describes with typed objects, C++ executes; no YAML disguised as
Python") makes a set of architecture promises about the central descriptor packages:

* a free-string algorithm selector is rejected, not silently accepted
  (``pops.descriptors.reject_string_selector``);
* a typed descriptor constructor does NOT take a ``kind="..."`` string -- the type IS
  the kind, so ``HLL(kind=...)`` / ``RuntimeParam(..., kind=...)`` is a ``TypeError``;
* every catalog descriptor is inspectable (``.inspect()`` -> dict) and self-validating
  (``.validate()`` does not raise / returns truthy);
* the compiled-brick load path refuses a brick without a real manifest with a clear error.

These are negative / contract tests for the typed surface. They IMPORT ``pops`` and so
need the compiled ``_pops`` extension; if it cannot be loaded the module is skipped (like
``test_public_imports.py``), so the source-only architecture checks still run bare.
"""
import pytest

# Skip the whole module if the native extension cannot be loaded in this interpreter.
# importorskip is too strict here (pops/_bootstrap raises a custom ImportError whose .name
# does not match "pops._pops"), so catch any import failure and skip at module level.
try:
    import pops._pops  # noqa: F401
except Exception as _exc:  # pragma: no cover - exercised only without a built extension
    pytest.skip("compiled _pops extension not importable: %s" % _exc,
                allow_module_level=True)


def test_reject_string_selector_raises_and_is_actionable():
    # A free string for an algorithm selector is a TypeError that names the param, echoes
    # the rejected value, and points at the typed alternative.
    import pops.descriptors as descriptors

    with pytest.raises(TypeError) as excinfo:
        descriptors.reject_string_selector("hll", "riemann", suggestion="HLL()")
    message = str(excinfo.value)
    assert "riemann" in message            # names the rejected parameter
    assert "hll" in message                # echoes the rejected value
    assert "HLL()" in message              # points at the typed alternative


def test_reject_string_selector_does_not_touch_a_real_typed_object():
    # The guard only fires on the string branch: passing the typed object as the suggestion
    # (a real descriptor instance) still raises -- the helper ALWAYS raises by design -- but
    # the real typed object itself is untouched and remains a usable descriptor.
    import pops.descriptors as descriptors
    from pops.numerics.riemann import HLL

    typed = HLL()
    # The real typed object is not rejected on its own: it is a valid descriptor.
    assert typed.validate()
    assert typed.inspect()["native_id"] == typed.native_id
    # Used as the suggestion it is rendered, not mutated, and the guard still raises.
    with pytest.raises(TypeError):
        descriptors.reject_string_selector("hll", "riemann", suggestion=typed)
    assert typed.native_id == "pops::HLLFlux"  # untouched


def test_typed_descriptor_constructors_reject_a_kind_string():
    # The TYPE is the kind: a typed descriptor constructor must not accept a kind="..."
    # selector string. CPython raises TypeError("unexpected keyword argument 'kind'").
    from pops.numerics.riemann import HLL
    from pops.params import RuntimeParam

    with pytest.raises(TypeError):
        HLL(kind="x")
    with pytest.raises(TypeError):
        RuntimeParam("a", kind="runtime")


def test_catalog_descriptors_are_inspectable_and_validate():
    # Every catalogued numerics/diagnostics descriptor that constructs with no required
    # args exposes .inspect() -> dict and a .validate() that does not raise / returns truthy.
    import pops.diagnostics as diagnostics
    import pops.numerics.reconstruction as reconstruction
    import pops.numerics.riemann as riemann
    import pops.numerics.variables as variables

    checked = 0
    for module in (riemann, reconstruction, variables, diagnostics):
        for name in getattr(module, "__all__", ()):
            obj = getattr(module, name)
            if not callable(obj):
                continue
            try:
                instance = obj()
            except TypeError:
                # Constructors that need arguments (e.g. User(...)) are out of scope here.
                continue
            if not (hasattr(instance, "inspect") and hasattr(instance, "validate")):
                continue
            view = instance.inspect()
            assert isinstance(view, dict), "%s.%s().inspect() must return a dict" % (
                module.__name__, name)
            assert view, "%s.%s().inspect() returned an empty dict" % (module.__name__, name)
            assert instance.validate(), "%s.%s().validate() must be truthy" % (
                module.__name__, name)
            checked += 1
    assert checked >= 8, ("expected to inspect the riemann/reconstruction/variables/diagnostics "
                          "catalogs")


def test_problem_is_not_a_descriptor():
    # Spec 5 sec.6 table / sec.15: a Problem is an ASSEMBLY that CONTAINS descriptors (its layout,
    # the blocks' physics, the field problems). It is NOT itself a Descriptor. The architecture
    # promise: isinstance(Problem(...), Descriptor) is False, yet every Problem method still works
    # (it duck-types the inspectable surface), and the parts it holds ARE descriptors.
    import pops
    from pops.descriptors import Descriptor, DescriptorProtocol

    prob = pops.Problem(name="arch").block("ne", physics=type("M", (), {"name": "m"})())
    assert not isinstance(prob, Descriptor), (
        "Spec 5 sec.6: a Problem must NOT be a pops.descriptors.Descriptor (it is an assembly "
        "that contains descriptors, not one itself)")
    # The inspectable surface survives the de-Descriptor change (structural duck typing).
    assert isinstance(prob, DescriptorProtocol)
    assert prob.validate.__self__ is prob  # validate() is implemented directly on Problem.
    assert prob.inspect()["category"] == "problem"
    assert prob.lower()["name"] == "arch"
    # The layout it CONTAINS is still a descriptor (the assembly holds descriptors).
    assert isinstance(prob.layout, Descriptor)


def test_optimization_math_rejects_a_bare_string():
    # Spec 5 sec.14.2 / #20-21: the codegen Optimization math= / fuse= selectors are TYPED objects;
    # a bare string is rejected at construction (not silently mis-set and crashed later), while the
    # typed StrictMath() / FastMath() / ... usage keeps working.
    from pops.codegen import Optimization, FastMath, StrictMath

    with pytest.raises(TypeError) as excinfo:
        Optimization(math="fast")
    message = str(excinfo.value)
    assert "optimization math" in message and "fast" in message
    assert "StrictMath()" in message and "FastMath()" in message
    with pytest.raises(TypeError):
        Optimization(fuse="conservative")
    # Typed usage is intact and the default stays StrictMath.
    assert isinstance(Optimization().math, StrictMath)
    assert Optimization(math=FastMath()).options()["math"] == "FastMath"


def test_compiled_brick_without_a_manifest_is_a_clear_error():
    # The compiled-brick load path refuses a brick whose manifest does not exist. read_manifest
    # on a missing .json raises FileNotFoundError (an OSError); resolving a CompiledBrickRef whose
    # manifest is absent raises an explainable error before any runtime install.
    from pops.external import CompiledBrickRef, read_manifest

    with pytest.raises(OSError):  # FileNotFoundError is an OSError
        read_manifest("/nonexistent_pops_brick_manifest.json")

    ref = CompiledBrickRef(manifest="/nonexistent_pops_brick_manifest.json",
                           native_id="missing_brick")
    with pytest.raises((OSError, ValueError)):
        ref.resolve()
    # validate() turns the unresolvable manifest into an explainable ValueError, not a crash.
    with pytest.raises(ValueError) as excinfo:
        ref.validate()
    assert "missing_brick" in str(excinfo.value)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
