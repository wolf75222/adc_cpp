#!/usr/bin/env python3
"""CompiledManifest richness: the rich compiled-artifact manifest (Spec 5 sec.13.12, ADC-479 #36).

The compiled-artifact manifest widens from the thin brick-id / category list
:class:`pops.external.CompiledManifest` carries into the FULL self-description Spec 5 sec.13.12
requires. These checks build a SYNTHETIC ``CompiledProblem`` (a real in-memory ``pops.time.Program``
+ a real ``CompiledModel`` metadata carrier -- NO Kokkos compile, NO ``.so`` on disk) and assert:

  - ``compiled.manifest()`` returns a :class:`pops.external.CompiledArtifactManifest` whose RICH
    fields are populated from real metadata (model_name / blocks / variables / roles / aux_required
    / params_const / params_runtime / ghost_depth / field_outputs / abi_key / required_headers_sig);
  - the ``supports_*`` capability flags the C++ codegen does NOT emit are honestly ``None``
    (UNKNOWN), never a fabricated ``True`` / ``False``, and are listed by ``needs_cpp_followup()``;
  - the caps-sourced ``supports_*`` flags (uniform / amr / mpi / gpu) are read from the backend caps
    when present and are ``None`` when the carried model records no caps;
  - ``to_dict`` round-trips through JSON and ``str()`` / ``print`` work;
  - ``check_layout_supported`` raises on a KNOWN-false flag (the sec.13.12 error shape) but does NOT
    hard-reject on an unknown (``None``) flag -- the no-false-positive discipline.

Pure Python: it imports only the inert authoring packages (the compiled ``_pops`` loads as a side
effect of ``import pops`` but no model is built or run). Pytest + ``__main__`` guard.
"""
import json
import sys

import pytest

pops = pytest.importorskip("pops")

from pops.codegen.loader import CompiledModel, CompiledProblem  # noqa: E402
from pops.external import (  # noqa: E402
    CompiledArtifactManifest, build_compiled_manifest, check_layout_supported)
from pops.descriptors import Availability  # noqa: E402
from pops.physics.model import Param  # noqa: E402
from pops import time as adctime  # noqa: E402


def _program(name="manifest_demo"):
    """A real in-memory Program: a state, an elliptic field solve, a Forward-Euler commit."""
    P = adctime.Program(name)
    dt = P.dt
    U = P.state("plasma")
    f = P.solve_fields("phi", U)
    R = P.rhs(state=U, fields=f, flux=True, sources=["default"])
    P.commit("plasma", P.linear_combine("U1", U + dt * R))
    return P


def _model(*, params=None, caps=None, with_roles=True):
    """A real CompiledModel metadata carrier (no .so) -- the engine class, carrying only metadata."""
    roles = ["Density", "MomentumX", "MomentumY"] if with_roles else []
    return CompiledModel(
        so_path="/nonexistent/problem.so", backend="production", adder="add_native_block",
        cons_names=["rho", "mx", "my"], cons_roles=roles, prim_names=["rho", "mx", "my"],
        n_vars=3, gamma=1.4, n_aux=1, params=params or {},
        caps=caps if caps is not None else {"cpu": True, "mpi": True, "amr": True, "gpu": False},
        abi_key="HEADERSIG|c++|c++23", model_hash="modelhash", cxx="c++", std="c++23",
        aux_extra_names=["B_z"])


def _compiled(*, params=None, caps=None, with_roles=True):
    """A SYNTHETIC CompiledProblem: a real lowered Program + a real CompiledModel, no compile."""
    P = _program()
    m = _model(params=params, caps=caps, with_roles=with_roles)
    return CompiledProblem("/tmp/pops-cache/problem.so", P, m, "HEADERSIG|c++|c++23", "c++",
                           "c++23", problem_hash="deadbeefcafe", cache_key="0badc0de")


def _default_params():
    return {"cs2": Param("cs2", 1.0, kind="runtime"),
            "gamma_const": Param("gamma_const", 1.4, kind="const")}


# ---------------------------------------------------------------------------
# rich fields populated from real metadata
# ---------------------------------------------------------------------------

def test_manifest_accessor_returns_rich_manifest():
    """compiled.manifest() returns a CompiledArtifactManifest (the loader accessor exists)."""
    cp = _compiled(params=_default_params())
    manifest = cp.manifest()
    assert isinstance(manifest, CompiledArtifactManifest)
    # build_compiled_manifest(compiled) is the same builder, usable directly.
    assert isinstance(build_compiled_manifest(cp), CompiledArtifactManifest)


def test_rich_fields_populated_from_real_metadata():
    """The rich fields come from the carried Program + model + abi_key (not fabricated)."""
    cp = _compiled(params=_default_params())
    m = cp.manifest()
    # Identity from abi_key.
    assert m.model_name == "manifest_demo"
    assert m.abi_key == "HEADERSIG|c++|c++23"
    assert m.required_headers_sig == "HEADERSIG"  # the header token of the abi key
    # Blocks / variables / roles from the committed block + the model.
    assert m.blocks == ["plasma"]
    assert m.variables == ["rho", "mx", "my"]
    assert m.roles == ["Density", "MomentumX", "MomentumY"]
    # Aux + params split by kind.
    assert m.aux_required == ["B_z"]
    assert m.params_const == ["gamma_const"]
    assert m.params_runtime == ["cs2"]
    # Ghost depth + field outputs from the bind table / IR.
    assert m.ghost_depth == 2
    assert "phi" in m.field_outputs


def test_roles_unknown_when_model_records_none():
    """roles is honestly None (not []) when the carried model records no roles -- not fabricated."""
    cp = _compiled(with_roles=False)
    m = cp.manifest()
    assert m.roles is None
    # Variables are still populated (the names are always recorded).
    assert m.variables == ["rho", "mx", "my"]


# ---------------------------------------------------------------------------
# supports_* flags: known from caps vs honestly UNKNOWN (None)
# ---------------------------------------------------------------------------

def test_caps_sourced_flags_read_from_backend_caps():
    """supports_uniform/amr/mpi/gpu are read from the backend caps when present (real, not faked)."""
    cp = _compiled(caps={"cpu": True, "mpi": True, "amr": True, "gpu": False})
    m = cp.manifest()
    assert m.supports_uniform is True   # cpu -> uniform
    assert m.supports_mpi is True
    assert m.supports_amr is True
    assert m.supports_gpu is False      # a genuine False, honestly reported


def test_unknown_flags_are_none_not_fabricated():
    """The C++ does NOT emit stride / partial-IMEX / named-fields flags: they are None, never True."""
    cp = _compiled()
    m = cp.manifest()
    for flag in ("supports_stride", "supports_partial_imex_mask", "supports_named_fields"):
        assert getattr(m, flag) is None, "%s must be honestly None (C++ does not emit it)" % flag
    # native_entrypoints is empty until the C++ emits it.
    assert m.native_entrypoints == []
    # abi_version has no discrete C++ source yet -> honestly None.
    assert m.abi_version is None
    # needs_cpp_followup names exactly the unknown flags + native_entrypoints.
    pending = m.needs_cpp_followup()
    assert "supports_stride" in pending
    assert "supports_partial_imex_mask" in pending
    assert "supports_named_fields" in pending
    assert "native_entrypoints" in pending
    # The caps-sourced flags are KNOWN here, so they are NOT in the follow-up list.
    assert "supports_amr" not in pending


def test_caps_flags_unknown_when_no_caps():
    """With no backend caps, the caps-sourced flags are None (unknown), not fabricated False."""
    cp = _compiled(caps={})
    m = cp.manifest()
    for flag in ("supports_uniform", "supports_amr", "supports_mpi", "supports_gpu"):
        assert getattr(m, flag) is None, "%s must be None when the model records no caps" % flag
        assert flag in m.needs_cpp_followup()


# ---------------------------------------------------------------------------
# to_dict / print
# ---------------------------------------------------------------------------

def test_to_dict_round_trips_and_print_works():
    """to_dict() is JSON round-trippable (None flags stay None); str()/print render."""
    cp = _compiled(params=_default_params())
    m = cp.manifest()
    d = m.to_dict()
    assert d["model_name"] == "manifest_demo"
    assert d["blocks"] == ["plasma"]
    assert d["supports_amr"] is True
    assert d["supports_stride"] is None  # unknown stays None through serialisation
    assert json.loads(json.dumps(d)) == d, "to_dict is JSON round-trippable"
    text = str(m)
    assert "compiled-artifact manifest" in text
    assert "supports_amr" in text and "yes" in text
    assert "unknown" in text  # the unknown flags render as 'unknown', not 'no'
    assert "needs C++ follow-up" in text


# ---------------------------------------------------------------------------
# check_layout_supported: rejects KNOWN-false, never hard-rejects UNKNOWN
# ---------------------------------------------------------------------------

def test_validation_rejects_known_false_flag():
    """A KNOWN-false layout flag raises the sec.13.12 error shape."""
    # An aot artifact does NOT support AMR (caps.amr is False) -> a genuine rejection.
    cp = _compiled(caps={"cpu": True, "mpi": False, "amr": False, "gpu": False})
    m = cp.manifest()
    assert m.supports_amr is False
    with pytest.raises(ValueError) as excinfo:
        check_layout_supported(m, "AMR")
    msg = str(excinfo.value)
    assert "Compiled artifact cannot be used with layout=AMR" in msg
    assert "supports_amr=false" in msg


def test_validation_does_not_hard_reject_unknown_flag():
    """An UNKNOWN (None) flag must NOT hard-reject -- a false rejection breaks a working route."""
    # A model with no caps -> supports_amr is None (unknown), not False.
    cp = _compiled(caps={})
    m = cp.manifest()
    assert m.supports_amr is None
    status = check_layout_supported(m, "AMR")  # must NOT raise
    assert isinstance(status, Availability)
    assert status.status == "partial"
    assert "unknown" in status.reason


def test_validation_accepts_known_true_flag():
    """A KNOWN-true layout flag is accepted (yes status)."""
    cp = _compiled(caps={"cpu": True, "mpi": True, "amr": True, "gpu": False})
    m = cp.manifest()
    status = check_layout_supported(m, "AMR")
    assert status.ok is True
    # Uniform is supported too (cpu -> supports_uniform).
    assert check_layout_supported(m, "uniform").ok is True


def test_validation_is_inert_no_so_needed():
    """check_layout_supported reads flags only -- no .so on disk, no bind."""
    import os
    cp = _compiled()
    assert not os.path.exists(cp.so_path), "the synthetic .so path does not exist"
    # Building + validating the manifest must not raise despite the absent .so.
    m = cp.manifest()
    check_layout_supported(m, "uniform")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
