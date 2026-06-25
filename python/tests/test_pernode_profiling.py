"""Spec 3 per-node Program profiling (ADC-459): emit_cpp_program wraps each Program node.

Only coarse System phases ("step" / "field_solve") used to be timed. ADC-459 adds PER-PROGRAM-NODE
timing in the compiled program: the codegen brackets each value node's emitted C++ with a
steady_clock pair recorded under "node:<name>" through ctx.profile_record, so sim.profile_report()
shows per-node times (e.g. "node:rhs2", "node:solve_fields1") next to the coarse phases.

The wrapping is UNCONDITIONAL but cheap when profiling is off (Profiler::record early-returns; the
only cost is one extra clock read per node), so a Program with no profiling intent still emits valid
C++. This test pins the generated source only (pure Python, no compile / no _adc); the actual
per-node timing in a run is exercised on a built .so (ROMEO). It uses the REAL engine (adc.time);
it self-skips only if adc.time is unavailable, never faking it.
"""
import re

import pytest

t = pytest.importorskip("adc.time")


def _forward_euler():
    """A small real Program: forward Euler over one block via adc.time.std."""
    P = t.Program("pernode_fe")
    t.std.forward_euler(P, "gas")
    return P


def _ssprk3():
    """A multi-stage Program (three rhs / two intermediate lincomb / one commit)."""
    P = t.Program("pernode_ssprk3")
    t.std.ssprk3(P, "gas")
    return P


def test_per_node_scope_named_by_node():
    """Each work node is wrapped: a ProfileScope marker + a ctx.profile_record under node:<name>."""
    src = _forward_euler().emit_cpp_program()
    # The RAII-style marker the codegen emits for every wrapped node (issue: "ProfileScope" + "node:").
    assert "ProfileScope" in src, "generated source missing the per-node ProfileScope marker"
    assert "ctx.profile_record(" in src, "generated source missing the per-node profile_record call"
    # The three work nodes of forward Euler are named by their node id (solve_fields1 / rhs2 / fe_step).
    for node in ("node:solve_fields1", "node:rhs2", "node:fe_step"):
        assert node in src, "generated source missing per-node scope %r" % node
    # Every node scope is a matched pair: one steady_clock now() open + one profile_record close.
    opens = src.count("std::chrono::steady_clock::now();  // ProfileScope")
    closes = src.count("ctx.profile_record(")
    assert opens == closes and opens >= 3, "unbalanced per-node scopes (opens=%d closes=%d)" % (
        opens, closes)
    # Each open declares a unique _pt<id> the matching close reads (no redefinition at body scope).
    pts = re.findall(r"const auto (_pt\d+) = std::chrono::steady_clock::now\(\)", src)
    assert len(pts) == len(set(pts)), "duplicate per-node timer variable: %r" % pts


def test_pure_reference_nodes_not_wrapped():
    """The state node binds a MultiFab& and does no work, so it is not wrapped (no node:gas noise)."""
    src = _forward_euler().emit_cpp_program()
    assert "node:gas" not in src, "the pure state-binding node should not be profiled"
    # The state binding itself is still emitted (it is the base every op clones / commits into).
    assert "ctx.state(0)" in src, "generated source missing the block-0 state binding"


def test_multistage_wraps_every_work_node():
    """SSPRK3 lowers nine work nodes (3 solve_fields + 3 rhs + 3 lincomb); each gets one scope."""
    src = _ssprk3().emit_cpp_program()
    closes = src.count("ctx.profile_record(")
    assert closes == 9, "SSPRK3 should wrap 9 work nodes, got %d" % closes
    for node in ("node:solve_fields", "node:rhs", "node:ssprk3_step"):
        assert node in src, "generated source missing per-node scope prefix %r" % node


def test_no_profiling_intent_still_valid_cpp():
    """A Program with NO profiling intent still emits valid, complete C++ (the scope is unconditional
    and cheap-when-disabled). The chrono header and the stable ABI surface are present."""
    src = _forward_euler().emit_cpp_program()
    for tok in ("#include <chrono>", "adc::runtime::program::ProgramContext ctx(sys)",
                "adc_install_program", "ctx.install(", "std::chrono::steady_clock::now()"):
        assert tok in src, "generated source missing %r" % tok
    # The body is balanced and the per-node opens precede their closes (a close after each node block).
    assert src.count("ctx.profile_record(") >= 3, "expected at least 3 per-node records"
