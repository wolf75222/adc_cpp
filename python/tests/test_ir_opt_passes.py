#!/usr/bin/env python3
"""pops.time IR optimization passes -- CSE / redundant field-solve / liveness / estimate / GPU
detectors (ADC-465, Spec 3 s28).

The dead-node pass and its parity guards live in ``test_ir_passes.py``; this file pins the REST of
the section-28 pipeline. Every TRANSFORM pass (CSE, redundant-solve elim) is OPT-IN and PROVEN to
preserve the emitted numerics, so the headline guard is byte-identity:

  - CSE collapses a duplicated PURE sub-IR to a single computation and emits C++ byte-identical to the
    same program written with that value computed once (the producer runs once; the consumer's axpy
    structure -- hence the floating-point operation sequence -- is preserved exactly);
  - CSE NEVER collapses a side-effecting / buffer-writing op (a solve_fields, a schur_rhs, a reduce);
  - redundant-solve elim removes a provably-redundant second solve_fields over the same state and KEEPS
    it when a state/aux mutation (project / fill_boundary / a commit) intervenes;
  - the liveness / buffer-reuse / cost-estimate reports return sane, internally-consistent numbers;
  - a GPU detector flags a pathological IR (a long chain of tiny per-cell kernels) and stays quiet on a
    well-behaved one;
  - CRITICAL byte-identity: a Program with NO optimizable structure emits IDENTICAL C++ with the whole
    pipeline (``P.optimize()``) on vs off -- optimization must not change results.

Pure Python: no compilation, no .so. ``model=None`` still lowers FE / SSPRK / RK4, so the parity
checks run on real emitted C++ without a model. Runs as a script (``python3 test_ir_opt_passes.py``,
the CI invocation) AND under pytest; the script form drives the same checks and exits non-zero on the
first failure.
"""
import sys

import pytest

adctime = pytest.importorskip("pops.time")
libtime = pytest.importorskip("pops.lib.time")  # ready schemes (Spec 4)


# --------------------------------------------------------------------------- fixtures

def _euler_clean():
    """Forward Euler, no optimizable structure (the byte-identity reference)."""
    P = adctime.Program("forward_euler")
    dt = P.dt
    U = P.state("plasma")
    fields = P.solve_fields(U)
    R = P.rhs("R", state=U, fields=fields, flux=True, sources=["default"])
    P.commit("plasma", P.linear_combine("U1", U + dt * R))
    return P


def _cse_dup_program():
    """Two identical PURE masks feeding DISTINCT consumers (the wheres differ by a/b order), so CSE
    collapses ONLY the duplicated mask -- the wheres stay distinct and the final combine references two
    distinct values (no further collapse, no affine-term merge). Explicit names so the CSE survivor's
    name matches the hand-deduped reference exactly (CSE keeps the FIRST node's name)."""
    P = adctime.Program("cse")
    U = P.state("plasma")
    a = P.linear_combine("a", 1.0 * U)
    b = P.linear_combine("b", 2.0 * U)
    m1 = P.cell_compare(U, 0.0, ">", name="mask")     # representative
    m2 = P.cell_compare(U, 0.0, ">", name="mask_dup")  # exact pure duplicate of m1
    w1 = P.where(m1, a, b, name="w1")
    w2 = P.where(m2, b, a, name="w2")                  # distinct consumer (a/b swapped)
    P.commit("plasma", P.linear_combine("U1", 0.5 * w1 + 0.5 * w2))
    return P


def _cse_handwritten():
    """The SAME program with the mask computed ONCE -- the CSE byte-identity reference."""
    P = adctime.Program("cse")
    U = P.state("plasma")
    a = P.linear_combine("a", 1.0 * U)
    b = P.linear_combine("b", 2.0 * U)
    m = P.cell_compare(U, 0.0, ">", name="mask")
    w1 = P.where(m, a, b, name="w1")
    w2 = P.where(m, b, a, name="w2")
    P.commit("plasma", P.linear_combine("U1", 0.5 * w1 + 0.5 * w2))
    return P


# --------------------------------------------------------------------------- CSE

def test_cse_collapses_duplicate_pure_subir():
    """CSE drops exactly the duplicated PURE mask and rewires its consumer onto the representative."""
    P = _cse_dup_program()
    assert sum(1 for v in P._values if v.op == "cell_compare") == 2, "fixture lost a mask"
    Q = adctime.eliminate_common_subexpressions(P)
    assert Q is not P
    # The original is untouched (the pass optimizes a copy).
    assert sum(1 for v in P._values if v.op == "cell_compare") == 2, "original mutated"
    # Exactly one mask survives; both wheres survive (they are distinct).
    assert sum(1 for v in Q._values if v.op == "cell_compare") == 1, "duplicate mask not collapsed"
    assert sum(1 for v in Q._values if v.op == "where") == 2, "a distinct where was wrongly dropped"


def test_cse_byte_identical_to_handwritten():
    """After CSE, the duplicated program emits C++ byte-identical to the hand-deduped reference, AND
    the IR hash matches -- the deduplicated value is computed once, the consumers are unchanged."""
    Q = adctime.eliminate_common_subexpressions(_cse_dup_program())
    H = _cse_handwritten()
    assert Q._ir_hash() == H._ir_hash(), "CSE IR hash != hand-deduped"
    assert Q.emit_cpp_program() == H.emit_cpp_program(), "CSE C++ != hand-deduped"
    # And the duplicated program DIFFERED before the pass (otherwise the test proves nothing).
    assert _cse_dup_program()._ir_hash() != H._ir_hash()


def test_cse_noop_byte_identical_when_nothing_duplicated():
    """A program with no duplicated pure sub-IR: CSE is a byte-for-byte no-op."""
    P = _euler_clean()
    Q = adctime.eliminate_common_subexpressions(P)
    assert Q._ir_hash() == P._ir_hash(), "no-op CSE changed the IR hash"
    assert Q.emit_cpp_program() == P.emit_cpp_program(), "no-op CSE changed the emitted C++"


def test_cse_never_collapses_side_effecting_solve_fields():
    """Two solve_fields over the same state are NOT pure (side-effecting: they fill the shared aux).
    CSE must NEVER collapse them -- only the explicit redundant-solve pass may, and only when sound."""
    P = adctime.Program("two_solves")
    dt = P.dt
    U = P.state("plasma")
    f1 = P.solve_fields(U)
    f2 = P.solve_fields(U)
    # Use both field contexts so neither is dead (keeps the test about CSE, not dead-node elim).
    R1 = P.rhs("R1", state=U, fields=f1, flux=True, sources=["default"])
    R2 = P.rhs("R2", state=U, fields=f2, flux=True, sources=["default"])
    P.commit("plasma", P.linear_combine("U1", U + 0.5 * dt * R1 + 0.5 * dt * R2))
    Q = adctime.eliminate_common_subexpressions(P)
    assert sum(1 for v in Q._values if v.op == "solve_fields") == 2, "CSE collapsed a solve_fields"
    assert Q._ir_hash() == P._ir_hash(), "CSE touched the side-effecting program"


def test_cse_never_collapses_reduce_or_buffer_writer():
    """A reduce (collective communication) and a buffer-writer (schur_rhs / laplacian) are NOT on the
    pure allow-list, so CSE never deduplicates them even if two are textually identical."""
    # Two identical reductions: NOT collapsed (a reduce is a global communication, off the allow-list).
    P = adctime.Program("dup_reduce")
    dt = P.dt
    U = P.state("plasma")
    f = P.solve_fields(U)
    R = P.rhs("R", state=U, fields=f, flux=True, sources=["default"])
    P.record_scalar("n1", P.norm2(R))
    P.record_scalar("n2", P.norm2(R))   # identical reduce, but a reduce is never CSE'd
    P.commit("plasma", P.linear_combine("U1", U + dt * R))
    Q = adctime.eliminate_common_subexpressions(P)
    assert sum(1 for v in Q._values if v.op == "reduce") == 2, "CSE collapsed a reduce (unsound)"
    assert Q._ir_hash() == P._ir_hash()


def test_cse_condensed_schur_buffer_writers_untouched():
    """REGRESSION: condensed_schur assembles its RHS with buffer-writer ops (schur_rhs, etc.) whose
    result is discarded but whose buffer a later solve reads by identity. CSE must be a no-op here --
    none of those ops is pure, so the emitted C++ is byte-identical (still has assemble_schur_rhs)."""
    P = adctime.Program("cs")
    libtime.condensed_schur(P, "blk", alpha=1.0, theta=1.0)
    before = P.emit_cpp_program()
    assert "assemble_schur_rhs" in before, "fixture lost its schur RHS assembly"
    Q = adctime.eliminate_common_subexpressions(P)
    assert "assemble_schur_rhs" in Q.emit_cpp_program(), "CSE dropped a buffer-writer (unsound)"
    assert Q.emit_cpp_program() == before, "CSE corrupted the condensed_schur C++"
    assert Q._ir_hash() == P._ir_hash()


def test_cse_does_not_collapse_aux_reading_rhs_across_a_solve():
    """BLOCKER regression (adversarial review): rhs/source/apply read the SHARED System aux by buffer
    identity (not via a dataflow input), and solve_fields mutates that aux in place. Two rhs with the
    SAME (state, fields) dataflow inputs that STRADDLE a second solve_fields compute DIFFERENT values
    (the second reads the freshly solved aux). They are excluded from _PURE_OPS, so CSE -- which keys
    only on dataflow inputs -- must KEEP both, never collapse the second onto the stale-aux first."""
    P = adctime.Program("aux_rhs_cse")
    dt = P.dt
    U = P.state("plasma")
    f = P.solve_fields(U)
    R1 = P.rhs("R1", state=U, fields=f, flux=True, sources=["default"])
    P.solve_fields(U)  # re-fills the shared aux IN PLACE between the two rhs reads
    R2 = P.rhs("R2", state=U, fields=f, flux=True, sources=["default"])
    P.commit("plasma", P.linear_combine("U1", U + dt * R1 + dt * R2))
    n_before = sum(1 for v in P._values if v.op == "rhs")
    assert n_before == 2, "fixture lost an rhs"
    Q = P.optimize()
    assert sum(1 for v in Q._values if v.op == "rhs") == 2, \
        "CSE collapsed two aux-reading rhs across a solve_fields (the second would read stale aux)"


# --------------------------------------------------------------------------- redundant solve_fields

def test_redundant_solve_removed_when_no_mutation():
    """Two solve_fields over U^n with NO intervening state mutation: the second is provably redundant
    and is removed; its consumer is rewired onto the first solve. The committed outputs are unchanged."""
    P = adctime.Program("rs")
    dt = P.dt
    U = P.state("plasma")
    P.solve_fields(U)          # first solve (kept)
    f2 = P.solve_fields(U)     # redundant (no mutation since the first)
    R = P.rhs("R", state=U, fields=f2, flux=True, sources=["default"])
    P.commit("plasma", P.linear_combine("U1", U + dt * R))
    Q = adctime.eliminate_redundant_field_solves(P)
    assert sum(1 for v in Q._values if v.op == "solve_fields") == 1, "redundant solve not removed"
    assert set(Q.commits()) == {"plasma"}
    assert Q._ir_hash() != P._ir_hash(), "redundant-solve removal left the hash unchanged"


def test_redundant_solve_kept_when_state_mutated():
    """A project (in-place state mutation) between the two solves makes the second NOT redundant -- it
    re-solves from a changed state. The pass must keep both."""
    P = adctime.Program("rs_project")
    dt = P.dt
    U = P.state("plasma")
    P.solve_fields(U)
    P.project(U)               # in-place state mutation: a state/aux barrier
    f2 = P.solve_fields(U)
    R = P.rhs("R", state=U, fields=f2, flux=True, sources=["default"])
    P.commit("plasma", P.linear_combine("U1", U + dt * R))
    Q = adctime.eliminate_redundant_field_solves(P)
    assert sum(1 for v in Q._values if v.op == "solve_fields") == 2, "solve wrongly removed past a project"
    assert Q._ir_hash() == P._ir_hash(), "pass touched a program with no removable solve"


def test_redundant_solve_kept_when_fill_boundary_intervenes():
    """fill_boundary changes the ghosts the next elliptic solve reads -> the second solve is not
    provably redundant. Both kept."""
    P = adctime.Program("rs_fb")
    dt = P.dt
    U = P.state("plasma")
    P.solve_fields(U)
    P.fill_boundary(U)
    f2 = P.solve_fields(U)
    R = P.rhs("R", state=U, fields=f2, flux=True, sources=["default"])
    P.commit("plasma", P.linear_combine("U1", U + dt * R))
    Q = adctime.eliminate_redundant_field_solves(P)
    assert sum(1 for v in Q._values if v.op == "solve_fields") == 2, "solve removed past a fill_boundary"


def test_redundant_solve_noop_byte_identical():
    """A program with a single solve_fields: the pass is a byte-for-byte no-op."""
    P = _euler_clean()
    Q = adctime.eliminate_redundant_field_solves(P)
    assert Q._ir_hash() == P._ir_hash()
    assert Q.emit_cpp_program() == P.emit_cpp_program()


# --------------------------------------------------------------------------- liveness / reuse / estimate

def _rk4():
    P = adctime.Program("rk4")
    libtime.rk4(P, "plasma")
    return P


def test_scratch_liveness_sane():
    """Liveness returns one entry per scratch-allocating node with a non-negative live span, def index
    before last-use index, and the indices within the flat-list bounds."""
    P = _rk4()
    rows = P.scratch_liveness()
    assert rows, "no scratch live ranges reported for RK4"
    n = len(P._values)
    for r in rows:
        assert 0 <= r["def_index"] <= r["last_use_index"], r
        assert r["live_span"] == r["last_use_index"] - r["def_index"] >= 0, r
        assert r["def_index"] < n, r
    # RK4 allocates a fresh rhs scratch + a fresh accumulator per stage.
    assert sum(1 for r in rows if r["op"] == "rhs") == 4, rows


def test_buffer_reuse_consistent_and_saves():
    """Buffer reuse never assigns two simultaneously-live scratches to one buffer, and on RK4 (whose
    stage accumulators have disjoint live ranges) it saves at least one buffer vs the naive count."""
    P = _rk4()
    rep = P.buffer_reuse_report()
    assert rep["buffer_count"] <= rep["scratch_count"], rep
    assert rep["reused"] == rep["scratch_count"] - rep["buffer_count"], rep
    assert rep["buffer_count"] >= 1
    # Soundness of the assignment: two scratches sharing a buffer have DISJOINT live ranges.
    ranges = {r["name"]: r for r in P.scratch_liveness()}
    by_buf = {}
    for name, buf in rep["assignment"].items():
        by_buf.setdefault(buf, []).append(name)
    for buf, names in by_buf.items():
        names.sort(key=lambda nm: ranges[nm]["def_index"])
        for earlier, later in zip(names, names[1:], strict=False):
            assert ranges[earlier]["last_use_index"] < ranges[later]["def_index"], (buf, earlier, later)
    # RK4's accumulators reuse a buffer -> at least one saved.
    assert rep["scratch_count"] - rep["buffer_count"] >= 1, rep


def test_estimate_internally_consistent():
    """The cost estimate is internally consistent: kernels = small + (non-small per-cell) + heavy;
    traffic = reads + writes; the buffer figures match the reuse report."""
    P = _rk4()
    est = P.estimate()
    assert est["kernel_count"] >= est["small_kernels"] + est["heavy_kernels"]
    assert est["traffic_fields"] == est["field_reads"] + est["field_writes"]
    assert est["buffers_saved"] == est["scratch_count"] - est["buffer_count"] >= 0
    assert est["heavy_kernels"] == 4, "RK4 has 4 per-stage field solves"
    assert est["kernel_count"] > 0 and est["traffic_fields"] > 0
    # The textual report mentions each section.
    rep = P.estimate_report()
    for token in ("kernels", "scratch buffers", "memory traffic", "GPU detectors"):
        assert token in rep, rep


# --------------------------------------------------------------------------- GPU detectors

def _pathological():
    """A long chain of tiny per-cell kernels + a buffer explosion (trips every GPU detector)."""
    P = adctime.Program("patho")
    U = P.state("plasma")
    a = P.linear_combine("a", 1.0 * U)
    b = P.linear_combine("b", 2.0 * U)
    acc = 1.0 * U
    for i in range(20):
        m = P.cell_compare(U, float(i), ">", name="m%d" % i)
        w = P.where(m, a, b, name="w%d" % i)
        acc = acc + 0.01 * w
    P.commit("plasma", P.linear_combine("U1", acc))
    return P


def test_gpu_detectors_flag_pathological_ir():
    """The pathological IR trips the small-kernel, scratch-count and traffic detectors -- and the
    report NEVER raises (it is a warning, not a hard error)."""
    P = _pathological()
    warns = P.gpu_detectors()
    names = {w["detector"] for w in warns}
    assert "too_many_small_kernels" in names, names
    assert "too_many_scratches" in names, names
    assert "excessive_memory_traffic" in names, names
    for w in warns:                                   # each warning is value > threshold, with a message
        assert w["value"] > w["threshold"]
        assert w["message"]


def test_gpu_detectors_quiet_on_well_behaved_ir():
    """A small Forward Euler trips no host-side GPU heuristic."""
    assert libtime.forward_euler and _euler_clean().gpu_detectors() == []


# --------------------------------------------------------------------------- pipeline / byte-identity

def test_optimize_byte_identical_when_nothing_optimizable():
    """CRITICAL (the spec's hard requirement): a Program with NO optimizable structure emits IDENTICAL
    C++ with the whole pipeline on (``P.optimize()``) vs off -- optimization must not change results."""
    for build in (_euler_clean, _rk4):
        P = build()
        Q = P.optimize()
        assert Q._ir_hash() == P._ir_hash(), "optimize changed the hash of an optimal Program (%s)" % P.name
        assert Q.emit_cpp_program() == P.emit_cpp_program(), "optimize changed the C++ (%s)" % P.name


def test_optimize_runs_all_proven_safe_passes():
    """``optimize`` chains dead-node + CSE + redundant-solve; a program exercising all three collapses
    to a smaller, still-valid IR whose commit outputs are unchanged."""
    P = adctime.Program("all")
    dt = P.dt
    U = P.state("plasma")
    P.solve_fields(U)                 # first solve
    f2 = P.solve_fields(U)            # redundant solve (no mutation) -> removed
    R = P.rhs("R", state=U, fields=f2, flux=True, sources=["default"])
    P.rhs("dead", state=U, fields=f2, flux=True, sources=["default"])  # dead -> removed
    P.commit("plasma", P.linear_combine("U1", U + dt * R))
    Q = P.optimize()
    assert sum(1 for v in Q._values if v.op == "solve_fields") == 1
    assert "dead" not in {v.name for v in Q._values}
    assert set(Q.commits()) == {"plasma"}
    assert Q.validate()


def test_dump_passes_traces_pipeline():
    """``dump_passes`` runs the pipeline on a copy and names each pass + its node-count delta; the
    original is never mutated."""
    P = _euler_clean()
    before = [(v.op, v.name) for v in P._values]
    trace = P.dump_passes()
    assert "dead-node elimination" in trace
    assert "common-subexpression elimination" in trace
    assert "redundant field-solve elimination" in trace
    assert [(v.op, v.name) for v in P._values] == before, "dump_passes mutated the Program"


def test_method_and_free_function_forms_agree():
    """The free functions mirror the methods (the house pattern from eliminate_dead_nodes)."""
    P = _cse_dup_program()
    assert adctime.eliminate_common_subexpressions(P)._ir_hash() == P.eliminate_common_subexpressions()._ir_hash()
    R = _euler_clean()
    assert adctime.optimize(R)._ir_hash() == R.optimize()._ir_hash()
    S = adctime.Program("rs")
    dt = S.dt
    U = S.state("plasma")
    S.solve_fields(U)
    f2 = S.solve_fields(U)
    Rr = S.rhs("R", state=U, fields=f2, flux=True, sources=["default"])
    S.commit("plasma", S.linear_combine("U1", U + dt * Rr))
    assert (adctime.eliminate_redundant_field_solves(S)._ir_hash()
            == S.eliminate_redundant_field_solves()._ir_hash())


# --------------------------------------------------------------------------- script entry point

def _run_as_script():
    """Run every ``test_*`` in this module, print a pass/fail line each, and return the failure count
    (the CI ``python3 test_ir_opt_passes.py`` form). Mirrors test_codegen_dead_prims.py's harness."""
    fails = 0
    tests = sorted((name, fn) for name, fn in globals().items()
                   if name.startswith("test_") and callable(fn))
    for name, fn in tests:
        try:
            fn()
            print("  [OK ] %s" % name)
        except Exception as exc:  # noqa: BLE001 -- a test harness reports every failure, not the first
            fails += 1
            print("  [XX ] %s -- %s" % (name, exc))
    print("FAILS =", fails)
    return fails


if __name__ == "__main__":
    sys.exit(1 if _run_as_script() else 0)
