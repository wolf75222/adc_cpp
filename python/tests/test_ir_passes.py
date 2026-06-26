"""pops.time IR optimization passes -- dead-node elimination (ADC-465, Spec 3 s28).

``eliminate_dead_nodes`` is an OPT-IN pass: it returns a NEW Program whose flat SSA list has the
dead nodes removed. It is SAFE-BY-DEFAULT: a node is removable ONLY if its op is on an explicit
allow-list of ops proven to allocate a FRESH result scratch and have no other side effect (rhs,
source, apply, linear_combine, linear_source, solve_local_linear, cell_compare, where, reduce,
scalar_op, compare) AND no live op consumes its result. EVERY other op -- the buffer-writers that
alias a caller-allocated input buffer (schur_rhs, laplacian, ...), the side-effecting ops, solve_linear
and the sub-block ops -- is kept even with an unconsumed result, so an unknown/new op is never wrongly
dropped. It NEVER runs on the default ``emit_cpp_program`` path, so it cannot change an existing
compiled program.

The contract this test pins:

  - a genuinely unused ``P.rhs`` / ``linear_combine`` node is removed, and ONLY it;
  - PARITY: the pass is a byte-for-byte no-op on the emitted C++ when nothing is dead, and a program
    with a dead node emits -- after the pass -- C++ byte-identical to the same program written WITHOUT
    that node;
  - side-effecting nodes (fill_boundary / record_scalar / solve_fields) are NEVER removed even with an
    unused result;
  - BUFFER-WRITERS (schur_rhs and the generic laplacian/gradient/divergence) whose result is discarded
    but whose buffer a later op reads by identity are NEVER removed (the safe-by-default whitelist);
  - the ``_ir_hash`` genuinely changes (the IR changed) yet the committed outputs are unchanged.

Pure Python: no compilation, no .so. ``model=None`` still lowers FE / SSPRK, so the parity checks run
on real emitted C++ without a model. Run with python3 (PYTHONPATH = built adc package).
"""
import pytest

adctime = pytest.importorskip("pops.time")


def _commit_signature(prog):
    """A stable, id-independent fingerprint of the committed outputs: per block, the committed
    State's op + name + the affine coefficient polynomials it combines (the actual numerics), so two
    programs that commit the same scheme match even if their node ids differ."""
    out = {}
    for block, state in prog.commits().items():
        coeffs = state.attrs.get("coeffs")
        out[block] = (state.op, state.name, repr(coeffs))
    return out


def _euler_with_dead_rhs():
    """Forward Euler whose committed combine never reads ``dead`` (a genuinely unused rhs)."""
    P = adctime.Program("forward_euler")
    dt = P.dt
    U = P.state("plasma")
    fields = P.solve_fields(U)
    R = P.rhs("R", state=U, fields=fields, flux=True, sources=["default"])
    P.rhs("dead", state=U, fields=fields, flux=True, sources=["default"])  # never consumed
    P.commit("plasma", P.linear_combine("U1", U + dt * R))
    return P


def _euler_no_dead():
    """The SAME forward Euler, written without the dead rhs (the byte-identity reference)."""
    P = adctime.Program("forward_euler")
    dt = P.dt
    U = P.state("plasma")
    fields = P.solve_fields(U)
    R = P.rhs("R", state=U, fields=fields, flux=True, sources=["default"])
    P.commit("plasma", P.linear_combine("U1", U + dt * R))
    return P


def test_removes_exactly_the_dead_node():
    P = _euler_with_dead_rhs()
    before_ops = [(v.op, v.name) for v in P._values]
    assert ("rhs", "dead") in before_ops

    Q = adctime.eliminate_dead_nodes(P)

    # The pass returns a NEW Program (the original is untouched).
    assert Q is not P
    assert [(v.op, v.name) for v in P._values] == before_ops, "original mutated"

    after_ops = [(v.op, v.name) for v in Q._values]
    assert ("rhs", "dead") not in after_ops, "dead rhs not removed"
    # Exactly one node removed; every other (op, name) kept, in order.
    assert after_ops == [op for op in before_ops if op != ("rhs", "dead")]
    # The commit target and its inputs survive unchanged.
    assert set(Q.commits()) == {"plasma"}
    assert _commit_signature(Q) == _commit_signature(P)


def test_parity_noop_when_nothing_dead():
    """A program with NO dead nodes: the pass emits byte-identical C++ (a pure no-op)."""
    P = _euler_no_dead()
    Q = adctime.eliminate_dead_nodes(P)
    assert Q._ir_hash() == P._ir_hash(), "no-op pass changed the IR hash"
    assert Q.emit_cpp_program() == P.emit_cpp_program(), "no-op pass changed the emitted C++"


def test_parity_dead_node_matches_handwritten():
    """A program WITH a dead node, after the pass, is byte-identical to the same program WRITTEN
    without that node -- both the IR hash and the full emitted C++."""
    P_dead = _euler_with_dead_rhs()
    P_clean = _euler_no_dead()
    Q = adctime.eliminate_dead_nodes(P_dead)
    assert Q._ir_hash() == P_clean._ir_hash(), "optimized hash != hand-written-clean hash"
    assert Q.emit_cpp_program() == P_clean.emit_cpp_program(), "optimized C++ != hand-written-clean"
    # And the dead program differed BEFORE the pass (otherwise the test proves nothing).
    assert P_dead._ir_hash() != P_clean._ir_hash()


def test_method_form_matches_free_function():
    P = _euler_with_dead_rhs()
    Q_fn = adctime.eliminate_dead_nodes(P)
    Q_method = P.eliminate_dead_nodes()
    assert Q_method._ir_hash() == Q_fn._ir_hash()
    assert Q_method.emit_cpp_program() == Q_fn.emit_cpp_program()


def test_side_effecting_nodes_never_removed():
    """fill_boundary / record_scalar / solve_fields are side-effecting: kept even with an unused
    result. Here the committed combine reads only U + dt*R, so none of the three feeds the commit."""
    P = adctime.Program("side_effects")
    dt = P.dt
    U = P.state("plasma")
    fields = P.solve_fields(U)            # side-effecting (fills ghosts/aux), result unused downstream
    R = P.rhs("R", state=U, fields=fields, flux=True, sources=["default"])
    P.fill_boundary(U)                    # side-effecting, result unused
    P.record_scalar("mass", P.norm2(R))  # side-effecting diagnostic, result unused
    P.commit("plasma", P.linear_combine("U1", U + dt * R))

    Q = adctime.eliminate_dead_nodes(P)
    kept = {v.op for v in Q._values}
    for op in ("solve_fields", "fill_boundary", "record_scalar"):
        assert op in kept, "%s wrongly removed (it is side-effecting)" % op
    # The norm2 feeding record_scalar is kept too (a live input of a side-effecting node).
    assert "reduce" in kept
    # Nothing was actually dead here -> hash unchanged (the pass is conservative).
    assert Q._ir_hash() == P._ir_hash()


def test_hash_changes_but_outputs_same():
    """Removing a dead node genuinely changes the IR hash, but the committed outputs are identical."""
    P = _euler_with_dead_rhs()
    Q = adctime.eliminate_dead_nodes(P)
    assert Q._ir_hash() != P._ir_hash(), "dead-node removal left the IR hash unchanged"
    assert _commit_signature(Q) == _commit_signature(P), "committed outputs changed"


def test_chained_dead_nodes_removed():
    """A dead node feeding only another dead node: BOTH go (reverse-reachability, not one level)."""
    P = adctime.Program("chain")
    dt = P.dt
    U = P.state("plasma")
    fields = P.solve_fields(U)
    R = P.rhs("R", state=U, fields=fields, flux=True, sources=["default"])
    dead0 = P.rhs("dead0", state=U, fields=fields, flux=True, sources=["default"])
    P.linear_combine("dead1", U + dt * dead0)  # consumes dead0 but is itself unused
    P.commit("plasma", P.linear_combine("U1", U + dt * R))

    Q = adctime.eliminate_dead_nodes(P)
    names = {v.name for v in Q._values}
    assert "dead0" not in names and "dead1" not in names
    assert {"R", "U1"} <= names


_ALPHA = 1.0


def test_condensed_schur_buffer_writers_never_removed():
    """REGRESSION (the safe-by-default whitelist). ``pops.time.condensed_schur`` assembles its RHS with
    ``P.schur_rhs(rhs, phi_n, U, ...)`` -- a top-level op whose RESULT is DISCARDED. Its real effect is
    filling the caller-allocated ``rhs`` scalar_field buffer, which ``P.solve_linear(rhs=rhs)`` then
    reads BY BUFFER IDENTITY, not via a dataflow input edge. A blacklist marks ``schur_rhs`` dead and
    drops it -> the emitted C++ loses ``ctx.assemble_schur_rhs`` and the Schur solve runs on a zero RHS
    (silent corruption, ``validate()`` stays True). Under the allow-list ``schur_rhs`` (and every other
    buffer-writer) is NOT removable, so the pass is a no-op: the emitted C++ is byte-identical and still
    contains ``assemble_schur_rhs``. Covers theta == 1 (historical IR) and theta < 1 + energy (the
    extra linear_combine copy / extrapolation / schur_energy buffer-writers)."""
    for theta, c_E in ((1.0, None), (0.5, None), (0.5, 3), (1.0, 3)):
        P = adctime.Program("cs")
        adctime.std.condensed_schur(P, "blk", alpha=_ALPHA, theta=theta, c_E=c_E)
        before = P.emit_cpp_program()
        assert "assemble_schur_rhs" in before, "fixture lost its schur RHS assembly"

        Q = adctime.eliminate_dead_nodes(P)

        after = Q.emit_cpp_program()
        msg = "theta=%r c_E=%r" % (theta, c_E)
        assert "assemble_schur_rhs" in after, "schur_rhs wrongly dropped (%s)" % msg
        # Nothing is dead: a pure no-op, byte-for-byte (the buffer-writers + solve are all live).
        assert after == before, "pass corrupted the condensed_schur C++ (%s)" % msg
        assert Q._ir_hash() == P._ir_hash(), "no-op pass changed the IR hash (%s)" % msg


def test_buffer_writing_op_with_discarded_result_kept():
    """GENERIC buffer-writer: a top-level ``P.laplacian(buf, buf)`` whose RESULT is discarded fills the
    caller-allocated ``buf`` that ``P.solve_linear`` then reads by BUFFER IDENTITY. The op is absent
    from the allow-list, so the safe-by-default pass keeps it -- a buffer-writer that aliases an input
    is never wrongly dropped, even with an unconsumed result."""
    P = adctime.Program("buf_writer")
    U = P.state("plasma")
    buf = P.scalar_field("buf")
    P.laplacian(buf, buf)  # buffer-writer: writes buf in place, RESULT DISCARDED
    A = P.matrix_free_operator("op")

    def apply(p, out, x):
        lap = p.scalar_field("lap")
        p.laplacian(lap, x)
        return -1.0 * lap

    P.set_apply(A, apply)
    P.solve_linear(operator=A, rhs=buf, method="cg", max_iter=10)  # reads buf by BUFFER IDENTITY
    P.commit("plasma", P.linear_combine("U1", 1.0 * U))

    before = P.emit_cpp_program()
    Q = adctime.eliminate_dead_nodes(P)
    after_ops = [(v.op, v.name) for v in Q._values]
    assert ("laplacian", "buf") in after_ops, "top-level buffer-writing laplacian wrongly removed"
    # No node is dead here -> exact no-op.
    assert Q.emit_cpp_program() == before
    assert Q._ir_hash() == P._ir_hash()


def test_control_flow_input_kept():
    """A value consumed only inside a while sub-block is LIVE (the while op lists it as an input);
    v1 never descends into sub-blocks, so anything feeding one is conservatively kept."""
    P = adctime.Program("cf")
    U = P.state("plasma")

    def cond(p, x):
        return p.norm2(x) > 1e-10

    def body(p, x):
        return p.linear_combine("it", 0.5 * x)

    Ufinal = P.while_(U, cond, body)
    P.commit("plasma", Ufinal)

    Q = adctime.eliminate_dead_nodes(P)
    ops = [v.op for v in Q._values]
    assert "while" in ops and "state" in ops
    # No node was dead (state feeds the while, while is committed) -> exact no-op.
    assert Q._ir_hash() == P._ir_hash()
    assert Q.emit_cpp_program() == P.emit_cpp_program()
