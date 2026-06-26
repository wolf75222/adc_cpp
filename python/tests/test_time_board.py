"""Spec 3 board-like time programs (pops.time board sugar).

T.fields / T.define / T.solve / T.commit_many are blackboard notation that lowers
to the SAME Program IR as the primitive solve_fields / linear_combine /
solve_local_linear / commit calls. These tests assert that IR identity, plus the
RateBundle / StageStateSet / atomic commit_many behaviour.
"""
import pytest

from pops import model as _model
from pops.time import Program
from pops.math import rate, unknown


def _ir(P):
    """Structural IR of a Program: per value (vtype, op, input positions, attrs, block)."""
    idx = {id(v): k for k, v in enumerate(P._values)}
    out = []
    for v in P._values:
        ins = tuple(idx[id(i)] for i in v.inputs)
        out.append((v.vtype, v.op, ins, repr(sorted(v.attrs.items())), v.block))
    return out


def test_fields_define_commit_match_primitive_ir():
    def build(board):
        P = Program("fe")
        dt = P.dt
        u = P.state("plasma")
        f = P.fields("f", from_state=u) if board else P.solve_fields("f", u)
        r = P.rhs(name="R", state=u, fields=f, flux=True, sources=["electric"])
        if board:
            u1 = P.define("U1", u + dt * r)
        else:
            u1 = P.linear_combine("U1", u + dt * r)
        P.commit("plasma", u1)
        return P

    assert _ir(build(True)) == _ir(build(False))


def test_solve_matches_linear_combine_plus_solve_local_linear():
    def build(board):
        P = Program("imp")
        dt = P.dt
        u = P.state("plasma")
        r = P.rhs(name="R", state=u, flux=True, sources=["electric"])
        if board:
            u1 = P.solve(
                "U1",
                (P.I - dt * P.linear_source("lorentz")) @ unknown("U1") == u + dt * r,
            )
        else:
            op = P.I - dt * P.linear_source("lorentz")  # build the operator first (board order)
            rhs = P.linear_combine("U1_rhs", u + dt * r)
            u1 = P.solve_local_linear(name="U1", operator=op, rhs=rhs)
        P.commit("plasma", u1)
        return P

    assert _ir(build(True)) == _ir(build(False))


def test_apply_operator_to_state_via_matmul():
    P = Program("apply")
    u = P.state("plasma")
    lu_board = P.linear_source("lorentz") @ u
    lu_manual = P.apply(operator=P.linear_source("lorentz"), state=u)
    assert lu_board.op == "apply" and lu_board.attrs["linear_source"] == "lorentz"
    assert lu_manual.op == "apply"


def test_define_equation_keeps_and_renames_rhs():
    P = Program("def")
    u = P.state("plasma")
    raw = P.rhs(name="tmp", state=u, flux=True, sources=["electric"])
    r = P.define("R^n", rate(u) == raw)
    assert r is raw            # same IR node
    assert r.name == "R^n"     # renamed to the board label


def test_commit_many_is_atomic():
    P = Program("ms")
    e = P.state("electrons")
    i = P.state("ions")
    e1 = P.linear_combine("e1", 2.0 * e)
    i1 = P.linear_combine("i1", 2.0 * i)
    P.commit_many({"electrons": e1, "ions": i1})
    assert set(P.commits()) == {"electrons", "ions"}


def test_commit_many_rejects_double_commit_without_partial():
    P = Program("ms")
    e = P.state("electrons")
    i = P.state("ions")
    e1 = P.linear_combine("e1", 2.0 * e)
    i1 = P.linear_combine("i1", 2.0 * i)
    P.commit("electrons", e1)
    with pytest.raises(ValueError, match="committed more than once"):
        P.commit_many({"electrons": e1, "ions": i1})
    # atomic: 'ions' must NOT have been committed because validation failed first
    assert "ions" not in P.commits()


def test_commit_many_rejects_non_state():
    P = Program("ms")
    e = P.state("electrons")
    scalar = P.norm2(e)
    with pytest.raises(ValueError, match="needs a State value"):
        P.commit_many({"electrons": scalar})


def test_state_set_drives_a_multi_block_field_solve():
    P = Program("ss")
    e = P.state("electrons")
    i = P.state("ions")
    n = P.state("neutrals")
    star = P.state_set("star", {"electrons": e, "ions": i, "neutrals": n})
    assert len(star) == 3
    f = P.fields("fstar", from_state_set=star)
    assert f.vtype == "fields" and f.op == "solve_fields_from_blocks"
    assert len(f.inputs) == 3


def test_rate_bundle_typed_multi_output():
    e = _model.StateSpace("electron_state", ["ne", "mex", "mey"])
    i = _model.StateSpace("ion_state", ["ni", "mix", "miy"])
    rb = _model.RateBundle({"electrons": _model.Rate(e), "ions": _model.Rate(i)})
    assert rb["electrons"] == _model.Rate("electron_state")
    rb.require("electrons", e)  # correct StateSpace -> ok
    with pytest.raises(TypeError):
        rb.require("electrons", i)  # wrong Rate on wrong StateSpace -> rejected


def test_record_and_check_invariant_lower_to_record_scalar():
    P = Program("inv")
    e = P.state("electrons")
    before = P.sum(e)                      # a Program scalar (reduction)
    P.record("mass", before)               # board diagnostic
    e1 = P.linear_combine("e1", 2.0 * e)
    after = P.sum(e1)
    out = P.check_invariant("mass", before=before, after=after, tolerance=1e-9)
    assert out.vtype == "scalar"
    assert out.attrs.get("tolerance") == 1e-9
    assert [v for v in P._values if v.op == "record_scalar"]  # both recorded


def test_record_rejects_non_scalar():
    P = Program("inv")
    e = P.state("electrons")
    with pytest.raises(ValueError, match="must be a Program scalar"):
        P.record("bad", e)  # a State, not a scalar


def test_rate_bundle_arbitrary_arity():
    spaces = {name: _model.StateSpace(name + "_state", ["n", "mx", "my"])
              for name in ("a", "b", "c", "d")}
    rb = _model.RateBundle({k: _model.Rate(v) for k, v in spaces.items()})
    assert len(rb) == 4  # no 2-input limit
    for k, v in spaces.items():
        rb.require(k, v)
