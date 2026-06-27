"""Spec 5 (sec.8.12 / sec.8.4, criterion #34): the AMR runtime inspection surface.

Exercises ``AmrSystem.amr`` -- the live, INERT inspection handle
(:class:`pops.runtime.amr.AmrRuntimeView`) -- and its reports: ``patch_table()`` /
``hierarchy_snapshot()`` / ``explain_regrid()`` / ``explain_ghosts()`` / ``explain_reflux()`` /
``explain_checkpoint()``. The host-runnable parts build a SMALL real ``AmrSystem`` (Kokkos-Serial
on this Mac), add one native block, refine on a density bump and take a few steps so a real fine
patch forms, then read the reports off the LIVE runtime. A full multi-step regrid campaign / MPI
distribution / GPU run is Kokkos/ROMEO-gated and not asserted here.

Honesty: a measure the native build cannot answer (per-level ghost depth, per-stage reflux timing)
is asserted to be reported as UNAVAILABLE, never a fabricated number. The reports are deterministic
and array-free (sec.12.1).
"""
import operator
import sys

import numpy as np
import pytest

pops = pytest.importorskip("pops")

from pops.runtime.amr import (  # noqa: E402
    AmrRuntimeView, PatchReport, RegridReport, GhostReport, RefluxReport, CheckpointReport,
    HierarchySnapshot)


def _model():
    """A minimal single-scalar ExB block model (no DSL compile; native bricks)."""
    return pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                      source=pops.NoSource(), elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0))


def _built_amr(regrid_every=2, n=32):
    """A small built AmrSystem with one refined patch (density bump + a few steps)."""
    sim = pops.AmrSystem(n=n, L=1.0, periodic=True, regrid_every=regrid_every, coarse_max_grid=16)
    sim.add_block("ne", model=_model(), spatial=pops.Spatial(minmod=True), time=pops.Explicit())
    sim.set_refinement(threshold=0.5)
    ne = np.ones((n, n))
    ne[n // 3:2 * n // 3, n // 3:2 * n // 3] = 5.0
    sim.set_density("ne", ne)
    for _ in range(3):
        sim.step_cfl(0.4)
    return sim


# --- the handle ----------------------------------------------------------------
def test_amr_handle_is_an_inert_runtime_view():
    sim = pops.AmrSystem(n=16, L=1.0, periodic=True)
    view = sim.amr
    assert isinstance(view, AmrRuntimeView)
    # A fresh view every access (handle, not cached state); both bound to the same system.
    assert isinstance(sim.amr, AmrRuntimeView)
    # The str is short and array-free (sec.12.1).
    text = str(view)
    assert "AmrRuntimeView" in text and "array(" not in text and len(text) < 200


def test_system_has_no_amr_handle_with_a_clear_error():
    sim = pops.System(n=16, L=1.0, periodic=True)
    assert not hasattr(sim, "amr")
    with pytest.raises(AttributeError, match="AmrSystem.*inspect_amr"):
        operator.attrgetter("amr")(sim)


# --- patch_table ---------------------------------------------------------------
def test_patch_table_before_build_reports_unbuilt():
    sim = pops.AmrSystem(n=16, L=1.0, periodic=True)
    rep = sim.amr.patch_table()
    assert isinstance(rep, PatchReport)
    assert rep.built is False
    assert rep.n_patches == 0
    assert "not built" in str(rep)


def test_patch_table_on_built_hierarchy_reports_live_patches():
    sim = _built_amr()
    rep = sim.amr.patch_table()
    assert rep.built is True
    assert rep.n_levels == 2
    assert rep.base_n == 32
    # A real fine patch formed on the bump (live runtime, not config).
    assert rep.n_patches >= 1
    levels = {lvl["level"]: lvl for lvl in rep.per_level}
    assert 0 in levels and levels[0]["level"] == 0          # base box reported
    assert levels[1]["n_patches"] == rep.n_patches          # the fine patches live on level 1
    assert levels[1]["cells"] > 0
    # Coarse box distribution comes from the live MPI diagnostic accessors.
    assert rep.coarse_local_boxes == 1 and rep.coarse_total_boxes == 1
    assert rep.coarse_is_distributed is False
    # Printable, deterministic, array-free.
    text = str(rep)
    assert text.startswith("AMR patch table") and "array(" not in text
    assert str(sim.amr.patch_table()) == text
    # to_dict round-trips the same numbers.
    d = rep.to_dict()
    assert d["n_patches"] == rep.n_patches and d["n_levels"] == 2


# --- hierarchy_snapshot --------------------------------------------------------
def test_hierarchy_snapshot_composes_config_envelope_and_live_patches():
    sim = _built_amr(regrid_every=2)
    snap = sim.amr.hierarchy_snapshot()
    assert isinstance(snap, HierarchySnapshot)
    # Config envelope reused from inspect_amr (the native max_levels / ratio).
    assert snap.max_levels == 2 and snap.ratio == 2
    assert snap.config_available == "yes"
    assert any("max_levels" in note for note in snap.limitations)
    # Live parts: the block registry + the patch table.
    assert snap.blocks == ["ne"]
    assert snap.frozen is False and snap.regrid_every == 2
    assert snap.patch_table.built is True and snap.patch_table.n_patches >= 1
    text = str(snap)
    assert text.startswith("AMR hierarchy snapshot") and "array(" not in text
    assert str(sim.amr.hierarchy_snapshot()) == text


# --- explain_regrid ------------------------------------------------------------
def test_explain_regrid_dynamic_vs_frozen():
    dyn = pops.AmrSystem(n=16, L=1.0, periodic=True, regrid_every=4).amr.explain_regrid()
    assert isinstance(dyn, RegridReport)
    assert dyn.frozen is False and dyn.regrid_every == 4
    # The union-of-tags criteria are named (config-sourced shape, not a fabricated threshold).
    blob = " ".join(dyn.criteria)
    assert "set_refinement" in blob and "grad phi" in blob

    frozen = pops.AmrSystem(n=16, L=1.0, periodic=True, regrid_every=0).amr.explain_regrid()
    assert frozen.frozen is True and frozen.regrid_every == 0
    assert any("frozen" in n for n in frozen.notes)


# --- explain_ghosts (honest deferral) -----------------------------------------
def test_explain_ghosts_defers_per_level_depth_honestly():
    rep = pops.AmrSystem(n=16, L=1.0, periodic=True).amr.explain_ghosts()
    assert isinstance(rep, GhostReport)
    # Per-level ghost depth is NOT fabricated: it is None and rendered as unavailable.
    assert rep.per_level_depth is None
    assert "unavailable" in str(rep)
    # The requirement shape (stencil -> ghost depth) is still explained.
    assert "weno5" in rep.requirement_note


# --- explain_reflux ------------------------------------------------------------
def test_explain_reflux_reports_route_requirement():
    rep = pops.AmrSystem(n=16, L=1.0, periodic=True).amr.explain_reflux()
    assert isinstance(rep, RefluxReport)
    assert rep.enabled is True
    # The per-stage timing is honestly unavailable (route property, not a counter).
    assert rep.per_stage is None
    assert "unavailable" in str(rep)


# --- explain_checkpoint --------------------------------------------------------
def test_explain_checkpoint_restartable_for_frozen_single_block():
    sim = pops.AmrSystem(n=16, L=1.0, periodic=True, regrid_every=0)
    sim.add_block("ne", model=_model())
    rep = sim.amr.explain_checkpoint()
    assert isinstance(rep, CheckpointReport)
    assert rep.restartable is True and rep.violations == []
    assert "single block" in str(rep)


def test_explain_checkpoint_flags_dynamic_regrid_violation():
    sim = pops.AmrSystem(n=16, L=1.0, periodic=True, regrid_every=3)
    sim.add_block("ne", model=_model())
    rep = sim.amr.explain_checkpoint()
    assert rep.restartable is False
    assert any("regrid_every=3" in v for v in rep.violations)
    # The composite multi-level Poisson restart is declared unavailable, not faked.
    assert any("composite multi-level Poisson" in n for n in rep.notes)


# --- compiled static delegation ------------------------------------------------
def test_compiled_inspect_amr_delegates_to_top_level():
    # A CompiledModel/Problem carries no AMR layout; its inspect_amr delegates to pops.inspect_amr.
    # Build a tiny stub CompiledModel (no .so dlopen needed for the inert delegation path).
    from pops.codegen.loader import CompiledModel
    cm = CompiledModel(
        so_path="<stub>", backend="aot", adder="add_native_block", cons_names=["rho"],
        cons_roles=["Density"], prim_names=["rho"], n_vars=1, gamma=None, n_aux=0, params={},
        caps={}, abi_key="k", model_hash="h", cxx="c++", std="23", target="amr_system")
    rep = cm.inspect_amr()
    # Default (no layout) -> the native envelope report (never a fabricated hierarchy).
    assert rep.to_dict()["layout"] == "native-envelope"
    # An explicit AMR layout is reported through the same top-level inspector.
    from pops.mesh import CartesianMesh
    from pops.mesh.layouts import AMR
    rep2 = cm.inspect_amr(AMR(base=CartesianMesh(n=64), max_levels=2, ratio=2))
    assert rep2.to_dict()["layout"] == "amr" and rep2.to_dict()["max_levels"] == 2


# The CI python runner invokes each test file as `python3 <file>`; run pytest on this
# module so the assertions execute (a bare import would only define the test functions).
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
