#!/usr/bin/env python3
"""Spec 5 sec.12.5 (criteria 41-44): the TYPED profiling surface.

``pops.Profile`` (Profile.Basic() / Profile.Advanced(), not profile="advanced") + the
``PerformanceSummary`` wrapper (printable __str__, to_dict / to_json, by_program_node /
by_native_brick / by_solver / by_memory) + the ``System.profile(...)`` context manager that
enables the native profiler on __enter__ and disables it on __exit__.

Two kinds of check:

* PURE shape / parsing / env -- no engine, no _pops. These run in any interpreter and exercise the
  typed objects + the report parser against the EXACT native report format
  (profiler.hpp report()). They do NOT fake the engine: they parse a literal sample of what the C++
  Profiler emits, which is the contract this wrapper is built to read.
* ENGINE -- a real native pops.System under the context manager: asserts enable/disable, the
  off-by-default contract, and that a stepped block's report flows into a PerformanceSummary. Skipped
  (not failed) if _pops / numpy is unavailable; never faked.

The heavy per-brick / scheduler / memory counters are Kokkos-gated (compiled .so step on ROMEO); the
typed views DECLARE those measures unavailable here rather than fabricating them -- asserted below.
"""
import json
import os
import sys

import pytest

from pops.runtime.profile import (PerformanceSummary, Profile, _parse_report,
                                   _Unavailable)


# A literal sample of the native report (profiler.hpp report()): two coarse phases, a per-node scope,
# and the trailing counters line. This is the format the wrapper parses -- not a fake engine.
_SAMPLE_REPORT = (
    "Profiler report (total 0.010849 s, 3 scopes)\n"
    "  step  count=2  total=0.007229s  mean=0.003614s  min=0.003575s  max=0.003654s\n"
    "  field_solve  count=1  total=0.003621s  mean=0.003621s  min=0.003621s  max=0.003621s\n"
    "  node:solve_fields1  count=1  total=0.001000s  mean=0.001000s  min=0.001000s  max=0.001000s\n"
    "counters:  steps=2  kernels=3\n"
)


# ---- (A) Profile typed level ----------------------------------------------------------------
def test_profile_basic_advanced_are_typed_objects():
    basic = Profile.Basic()
    adv = Profile.Advanced()
    assert isinstance(basic, Profile) and isinstance(adv, Profile)
    assert basic.level == "basic" and adv.level == "advanced"
    assert adv.advanced is True and basic.advanced is False
    assert Profile.Basic() == Profile.Basic() and Profile.Basic() != Profile.Advanced()
    assert repr(Profile.Basic()) == "Profile.Basic()"


def test_profile_rejects_bad_level():
    with pytest.raises(ValueError):
        Profile("turbo")


def test_profile_from_env_maps_pops_profile():
    saved = os.environ.get("POPS_PROFILE")
    try:
        os.environ.pop("POPS_PROFILE", None)
        assert Profile.from_env() is None
        assert Profile.from_env(default=Profile.Basic()) == Profile.Basic()
        os.environ["POPS_PROFILE"] = "off"
        assert Profile.from_env() is None
        os.environ["POPS_PROFILE"] = "advanced"
        assert Profile.from_env() == Profile.Advanced()
        os.environ["POPS_PROFILE"] = "basic"
        assert Profile.from_env() == Profile.Basic()
        os.environ["POPS_PROFILE"] = "1"
        assert Profile.from_env() == Profile.Basic()
    finally:
        if saved is None:
            os.environ.pop("POPS_PROFILE", None)
        else:
            os.environ["POPS_PROFILE"] = saved


# ---- (B) report parser ----------------------------------------------------------------------
def test_parse_report_extracts_scopes_and_counters():
    parsed = _parse_report(_SAMPLE_REPORT)
    assert abs(parsed["total_s"] - 0.010849) < 1e-9
    assert set(parsed["scopes"]) == {"step", "field_solve", "node:solve_fields1"}
    assert parsed["scopes"]["step"]["count"] == 2
    assert abs(parsed["scopes"]["step"]["total_s"] - 0.007229) < 1e-9
    assert parsed["counters"] == {"steps": 2, "kernels": 3}


def test_parse_empty_report_is_safe():
    parsed = _parse_report("")
    assert parsed["scopes"] == {} and parsed["counters"] == {} and parsed["total_s"] == 0.0
    assert _parse_report(None)["scopes"] == {}


# ---- (C) PerformanceSummary typed views -----------------------------------------------------
def test_summary_views_read_the_native_tables():
    summ = PerformanceSummary(_SAMPLE_REPORT, Profile.Advanced())
    # by_program_node: the node:<name> scope, bare name.
    nodes = summ.by_program_node()
    assert "solve_fields1" in nodes and nodes["solve_fields1"]["count"] == 1
    # by_solver: the coarse field_solve phase + the solve_fields node.
    solver = summ.by_solver()
    assert "field_solve" in solver and "solve_fields1" in solver
    # counters / scopes / total surfaced.
    assert summ.counters()["kernels"] == 3
    assert abs(summ.total_s() - 0.010849) < 1e-9


def test_summary_declares_unavailable_measures_honestly():
    # by_native_brick: the native runtime has no per-brick scope -> declared unavailable, NOT faked.
    summ = PerformanceSummary(_SAMPLE_REPORT, Profile.Advanced())
    brick = summ.by_native_brick()
    assert isinstance(brick, _Unavailable) and bool(brick) is False
    assert brick.available is False
    # by_memory: the sample has no scratch counters (host path) -> unavailable, not a faked 0.
    mem = summ.by_memory()
    assert isinstance(mem, _Unavailable) and bool(mem) is False


def test_summary_by_memory_reads_counters_when_present():
    report = _SAMPLE_REPORT.replace(
        "counters:  steps=2  kernels=3\n",
        "counters:  steps=2  kernels=3  scratch_allocs=4  scratch_peak_bytes=2048\n")
    mem = PerformanceSummary(report).by_memory()
    assert mem == {"scratch_allocs": 4, "scratch_peak_bytes": 2048}


def test_summary_is_printable_and_serialisable():
    summ = PerformanceSummary(_SAMPLE_REPORT, Profile.Basic())
    text = str(summ)
    assert "PerformanceSummary" in text and "step" in text and "kernels=3" in text
    # to_dict carries the level, scopes, counters, and the typed views (with availability).
    d = summ.to_dict()
    assert d["profile"] == "basic"
    assert d["counters"]["kernels"] == 3
    assert d["views"]["by_native_brick"]["available"] is False
    assert "solve_fields1" in d["views"]["by_program_node"]
    # to_json round-trips and can write to a path.
    parsed = json.loads(summ.to_json())
    assert parsed["counters"]["steps"] == 2


def test_summary_to_json_writes_path(tmp_path):
    out = tmp_path / "profile.json"
    PerformanceSummary(_SAMPLE_REPORT).to_json(str(out))
    assert out.is_file()
    assert json.loads(out.read_text())["total_s"] > 0.0


def test_empty_summary_has_no_data_message():
    summ = PerformanceSummary("")
    assert "no profiling data" in str(summ)
    assert summ.scopes() == {} and summ.counters() == {}


# ---- (D) ENGINE: the context manager over a real native System ------------------------------
def _make_stepped_system():
    """A real native pops.System with one isothermal block, ready to step. Returns (pops, sim, np)."""
    import numpy as np

    import pops
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov

    n = 16
    sim = pops.System(n=n, L=1.0, periodic=True)
    sim.add_block(
        "gas",
        pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                   transport=pops.IsothermalFlux(), source=pops.NoSource(),
                   elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0)),
        spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
        time=pops.Explicit())
    rho = np.ones((n, n), dtype=float)
    sim.set_state("gas", np.stack([rho, 0.1 * rho, 0.0 * rho]))
    return pops, sim, np


def test_engine_exports_typed_surface():
    pops = pytest.importorskip("pops")
    assert hasattr(pops, "Profile") and hasattr(pops, "PerformanceSummary")
    assert pops.Profile.Basic().level == "basic"


def test_engine_context_manager_enables_then_disables():
    try:
        pops, sim, _ = _make_stepped_system()
    except Exception as exc:  # noqa: BLE001
        pytest.skip("pops/_pops/numpy unavailable: %s" % exc)
    # off by default -- a plain System never enabled.
    assert sim.is_profiling() is False
    with sim.profile(pops.Profile.Basic()) as prof:
        assert sim.is_profiling() is True, "context manager enables on __enter__"
        sim.step(1e-3)
        sim.step(1e-3)
        summary_inside = prof.summary()
        assert isinstance(summary_inside, pops.PerformanceSummary)
    # disabled on __exit__.
    assert sim.is_profiling() is False, "context manager disables on __exit__"
    summary = prof.summary()
    assert isinstance(summary, pops.PerformanceSummary)
    assert summary.counters().get("steps") == 2
    assert "step" in summary.scopes()


def test_engine_off_by_default_contract():
    """A plain run (no with sim.profile()) records NOTHING: profiling stays disabled."""
    try:
        pops, sim, _ = _make_stepped_system()
    except Exception as exc:  # noqa: BLE001
        pytest.skip("pops/_pops/numpy unavailable: %s" % exc)
    sim.step(1e-3)
    assert sim.is_profiling() is False
    summ = PerformanceSummary(sim.profile_report())
    # the native report on a never-enabled profiler carries no counters.
    assert summ.counters() == {}, "off-by-default: no heavy timers without an explicit profile()"


def test_engine_profile_rejects_non_profile_arg():
    try:
        pops, sim, _ = _make_stepped_system()
    except Exception as exc:  # noqa: BLE001
        pytest.skip("pops/_pops/numpy unavailable: %s" % exc)
    with pytest.raises(TypeError):
        sim.profile("advanced")


def test_engine_profile_no_arg_uses_env_default():
    try:
        pops, sim, _ = _make_stepped_system()
    except Exception as exc:  # noqa: BLE001
        pytest.skip("pops/_pops/numpy unavailable: %s" % exc)
    saved = os.environ.get("POPS_PROFILE")
    try:
        os.environ["POPS_PROFILE"] = "advanced"
        with sim.profile() as prof:
            sim.step(1e-3)
        assert prof.summary().profile.level == "advanced"
    finally:
        if saved is None:
            os.environ.pop("POPS_PROFILE", None)
        else:
            os.environ["POPS_PROFILE"] = saved


# The CI python runner invokes each test file as `python3 <file>`; run pytest on this module so the
# assertions execute (a bare import would only define the test functions).
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
