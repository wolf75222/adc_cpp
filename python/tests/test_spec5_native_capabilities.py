"""Spec 5 sec.13.12 / sec.13.12.1 (criteria #36/#37): the authoritative native capability matrix.

These checks pin the STATIC tier of Spec 5 Item A (epic ADC-479): the capability VALUES come from
the C++ core (``_pops.module_capabilities()``), not a Python-derived descriptor walk -- closing the
sec.13.12 "Python-derived, not authoritative" gap. They assert:

  - ``_pops.module_capabilities()`` is present and returns the documented flag dict with an integer
    ``abi_version`` matching ``_pops.__abi_version__``;
  - HONESTY: ``supports_partial_imex_mask`` is ``False`` (no C++ path backs it -- a tree-wide grep
    confirms it), ``supports_named_fields`` is ``True`` (the named-aux transport exists), and the
    route-dependent ``supports_stride`` is ``True`` only for the production route, ``False`` for aot;
  - ``pops.inspect_capabilities()`` cross-checks its descriptor walk against the native source: it
    appends ``source="native"`` rows and never reports a layout descriptor available that the C++
    source reports unavailable (a disagreement raises ``CapabilityMismatchError``);
  - ``Problem.explain_routes()`` prints a route matrix sourced from the native facts (criterion #37).

The STATIC tier is LOCALLY validatable once ``_pops`` is rebuilt with this header change; the tests
SKIP cleanly on an ``_pops`` that predates ``module_capabilities`` (pre-rebuild). The PER-ARTIFACT
tier (``pops_compiled_manifest()`` / ``load_compiled_manifest``) needs a compiled ``.so`` and is
ROMEO-gated (POPS_KOKKOS_ROOT is unset locally, so an AOT ``.so`` cannot be compiled / run here);
the pure-Python overlay logic ``apply_native_manifest`` is covered below WITHOUT a real ``.so``.
"""
import sys

import pytest

pops = pytest.importorskip("pops")
_pops = pytest.importorskip("pops._pops")

_EXPECTED_FLAGS = ("supports_uniform", "supports_amr", "supports_mpi", "supports_gpu",
                   "supports_stride", "supports_named_fields", "supports_partial_imex_mask")


def _module_caps(target="module"):
    """The native capability dict for @p target, or skip if _pops predates module_capabilities."""
    fn = getattr(_pops, "module_capabilities", None)
    if fn is None:
        pytest.skip("_pops predates module_capabilities (rebuild _pops to run this tier)")
    return fn(target)


# --- STATIC tier: module_capabilities + abi_version --------------------------------------
def test_module_capabilities_present_and_shaped():
    caps = _module_caps()
    assert isinstance(caps, dict)
    for flag in _EXPECTED_FLAGS:
        assert flag in caps, "module_capabilities missing %r" % flag
        assert isinstance(caps[flag], bool), "%s must be bool, got %r" % (flag, caps[flag])
    assert isinstance(caps["abi_version"], int)


def test_abi_version_matches_module_attr():
    caps = _module_caps()
    assert isinstance(getattr(_pops, "__abi_version__", None), int)
    assert caps["abi_version"] == _pops.__abi_version__


def test_partial_imex_mask_is_honestly_false():
    # No C++ path backs a partial IMEX mask (tree-wide grep is empty); claiming True would be a lie.
    assert _module_caps()["supports_partial_imex_mask"] is False
    assert _module_caps("production")["supports_partial_imex_mask"] is False
    assert _module_caps("aot")["supports_partial_imex_mask"] is False


def test_named_fields_and_layouts_are_true():
    caps = _module_caps()
    assert caps["supports_named_fields"] is True
    assert caps["supports_uniform"] is True
    assert caps["supports_amr"] is True


def test_stride_is_route_dependent():
    # The production / native route carries a cell stride; the aot / prototype path hardcodes stride=1.
    assert _module_caps("production")["supports_stride"] is True
    assert _module_caps("aot")["supports_stride"] is False
    # The route-agnostic module query is conservative (false): it does not promise the production stride.
    assert _module_caps("module")["supports_stride"] is False


def test_unknown_target_rejected():
    fn = getattr(_pops, "module_capabilities", None)
    if fn is None:
        pytest.skip("_pops predates module_capabilities")
    # The C++ binding throws std::invalid_argument -> pybind surfaces it as ValueError.
    with pytest.raises(ValueError):
        fn("not-a-route")


# --- STATIC tier: inspect_capabilities cross-check ---------------------------------------
def test_inspect_capabilities_appends_native_rows_and_source():
    _module_caps()  # skip early if the native source is absent
    matrix = pops.inspect_capabilities()
    by_source = {}
    for entry in matrix:
        assert entry.source in ("descriptor", "native")
        by_source.setdefault(entry.source, []).append(entry)
    # The cross-check appended the authoritative native transport rows.
    assert by_source.get("native"), "inspect_capabilities should append source='native' rows"
    native_names = {e.name for e in by_source["native"]}
    assert "supports_partial_imex_mask" in native_names
    # The native partial-imex row reports unavailable (honest).
    pim = next(e for e in by_source["native"] if e.name == "supports_partial_imex_mask")
    assert pim.available == "no"


def test_inspect_capabilities_prints():
    _module_caps()
    text = str(pops.inspect_capabilities())
    assert "capability matrix" in text
    assert "source=" in text


def test_capability_mismatch_error_is_exported():
    from pops._capabilities import CapabilityMismatchError
    assert issubclass(CapabilityMismatchError, RuntimeError)


# --- STATIC tier: Problem.explain_routes (criterion #37) ---------------------------------
def test_explain_routes_sourced_from_native_facts():
    _module_caps()
    prob = pops.Problem(name="cap-demo").block("ne", physics=object())
    matrix = prob.explain_routes()
    rows = {row.feature: row for row in matrix}
    for flag in _EXPECTED_FLAGS:
        assert flag in rows, "explain_routes missing feature %r" % flag
        assert rows[flag].source == "native"
        assert rows[flag].status in ("available", "unavailable")
    # partial_imex_mask is unavailable and says so (honest).
    assert rows["supports_partial_imex_mask"].status == "unavailable"


def test_explain_routes_prints():
    _module_caps()
    prob = pops.Problem(name="cap-demo").block("ne", physics=object())
    text = str(prob.explain_routes())
    assert "route matrix" in text
    assert "supports_uniform" in text


# --- PER-ARTIFACT tier: pure-Python overlay (no .so; the real .so read is ROMEO-gated) ---
def test_apply_native_manifest_overlays_authoritative_fields():
    # ROMEO note: load_compiled_manifest(path) reads a real .so's pops_compiled_manifest(); building
    # an AOT .so needs POPS_KOKKOS_ROOT (unset locally), so the round-trip is validated on ROMEO. Here
    # we exercise the pure overlay logic with a synthetic native dict (the JSON shape the macro emits).
    from pops.external import CompiledArtifactManifest, apply_native_manifest
    manifest = CompiledArtifactManifest(model_name="m", supports_stride=None,
                                        supports_partial_imex_mask=None, supports_named_fields=None)
    native = {"abi_version": 1, "n_vars": 4, "n_aux": 3, "n_params": 0, "ghost_depth": 3,
              "supports_stride": False, "supports_partial_imex_mask": False,
              "supports_named_fields": True, "roles": ["density", "momentum_x"],
              "native_entrypoints": ["pops_model_nvars", "pops_compiled_manifest"]}
    apply_native_manifest(manifest, native)
    assert manifest.abi_version == 1
    assert manifest.ghost_depth == 3
    assert manifest.supports_partial_imex_mask is False  # the .so adjudicates, not a fabricated None
    assert manifest.supports_named_fields is True
    assert manifest.roles == ["density", "momentum_x"]
    assert "pops_compiled_manifest" in manifest.native_entrypoints
    # supports_partial_imex_mask is now KNOWN-False, so it leaves the needs-followup list.
    assert "supports_partial_imex_mask" not in manifest.needs_cpp_followup()


def test_apply_native_manifest_none_is_graceful_noop():
    # An old .so without the symbol -> load returns None -> the manifest keeps its honest-None set.
    from pops.external import CompiledArtifactManifest, apply_native_manifest
    manifest = CompiledArtifactManifest(model_name="m", supports_stride=None)
    apply_native_manifest(manifest, None)
    assert manifest.supports_stride is None  # untouched: graceful fallback, never fabricated


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
