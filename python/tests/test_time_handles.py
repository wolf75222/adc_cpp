#!/usr/bin/env python3
"""pops.time typed temporal-version handles (Spec 5 sec.5.3.1, ADC-485).

The handle layer (``P.state("U", block="plasma")`` -> a :class:`TimeState` with ``.n`` /
``.stage`` / ``.next`` / ``.prev``, plus ``T.define`` / ``T.commit`` / ``T.keep_history``)
is SUGAR over the existing SSA IR: it lowers to the SAME ``state`` / ``linear_combine`` /
``commit`` / ``history`` / ``store_history`` ops the positional ``P.state`` style builds.

These checks are pure Python (no compilation): they exercise the handle algebra, the SSA /
read-only / history-policy guards, the CRUCIAL byte-identical ``_ir_hash`` proof (SSPRK3 via
handles == SSPRK3 via the legacy positional style), and that the handles carry no ndarray.

Run with python3 (PYTHONPATH = built pops package); falls back to pytest from the runner.
"""
import sys

import pytest

from pops import time as adctime


def _expect_value_error(fn, needle):
    """Call ``fn`` and assert it raises a ValueError whose message contains ``needle``."""
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), "wrong message: %r (wanted %r)" % (str(exc), needle)
    else:
        raise AssertionError("expected ValueError containing %r" % (needle,))


def test_current_state_is_read_only():
    P = adctime.Program("ro")
    U = P.state("U", block="plasma")
    _expect_value_error(lambda: P.define(U.n, U.n + P.dt * U.n),
                        "current state is read-only in Program")


def test_define_prev_rejected():
    P = adctime.Program("prev_def")
    U = P.state("U", block="plasma")
    _expect_value_error(lambda: P.define(U.prev, U.n),
                        "history is produced by the history policy")


def test_use_before_define_raises():
    P = adctime.Program("ubd")
    U = P.state("U", block="plasma")
    s1 = U.stage(1)
    _expect_value_error(lambda: s1 + P.dt * s1,
                        "stage 1 is undefined (define it with T.define first)")
    _expect_value_error(lambda: P.commit(s1),
                        "stage 1 is undefined")


def test_double_define_rejected():
    P = adctime.Program("dd")
    U = P.state("U", block="plasma")
    k0 = P.rhs(state=U.n, fields=P.solve_fields(U.n), sources=["default"])
    P.define(U.stage(1), U.n + P.dt * k0)
    _expect_value_error(lambda: P.define(U.stage(1), U.n + P.dt * k0),
                        "SSA version already defined")


def test_prev_without_keep_history_raises():
    P = adctime.Program("noh")
    U = P.state("U", block="plasma")
    _expect_value_error(lambda: U.prev(1), "requires keep_history first")
    _expect_value_error(lambda: U.n + P.dt * U.prev, "requires keep_history first")


def test_keep_history_then_prev_reads_history():
    P = adctime.Program("hist")
    U = P.state("U", block="plasma")
    P.keep_history(U, depth=2)
    p1 = U.prev(1)
    assert p1.vtype == "state" and p1.op == "history" and p1.attrs["lag"] == 1
    assert p1.attrs["history"] == "plasma.U"
    p2 = U.prev(2)
    assert p2.attrs["lag"] == 2
    # bare U.prev behaves as lag 1: the affine proxy reads the lag-1 history Value
    bare = U.n + P.dt * U.prev  # forces the lag-1 affine proxy
    hist_terms = [v for v, _ in bare.terms if v.op == "history"]
    assert hist_terms and hist_terms[0].attrs["lag"] == 1
    _expect_value_error(lambda: U.prev(3), "exceeds the kept history depth")


def _ssprk3_legacy(P, block):
    """SSPRK3 built with the legacy positional P.state / linear_combine / commit style."""
    U0 = P.state(block)
    f0 = P.solve_fields(U0)
    k0 = P.rhs(state=U0, fields=f0, flux=True, sources=["default"])
    U1 = P.linear_combine("ssprk3_U1", U0 + P.dt * k0)
    f1 = P.solve_fields(U1)
    k1 = P.rhs(state=U1, fields=f1, flux=True, sources=["default"])
    U2 = P.linear_combine("ssprk3_U2", 0.75 * U0 + 0.25 * (U1 + P.dt * k1))
    f2 = P.solve_fields(U2)
    k2 = P.rhs(state=U2, fields=f2, flux=True, sources=["default"])
    P.commit(block, P.linear_combine(
        "ssprk3_step", (1.0 / 3.0) * U0 + (2.0 / 3.0) * (U2 + P.dt * k2)))


def _ssprk3_handles(P, block):
    """The SAME SSPRK3, written with the typed temporal-version handles."""
    U = P.state("U", block=block)
    f0 = P.solve_fields(U.n)
    k0 = P.rhs(state=U.n, fields=f0, flux=True, sources=["default"])
    P.define(U.stage(1), U.n + P.dt * k0)
    f1 = P.solve_fields(U.stage(1))
    k1 = P.rhs(state=U.stage(1), fields=f1, flux=True, sources=["default"])
    P.define(U.stage(2), 0.75 * U.n + 0.25 * (U.stage(1) + P.dt * k1))
    f2 = P.solve_fields(U.stage(2))
    k2 = P.rhs(state=U.stage(2), fields=f2, flux=True, sources=["default"])
    P.define(U.next, (1.0 / 3.0) * U.n + (2.0 / 3.0) * (U.stage(2) + P.dt * k2))
    P.commit(U.next)


def test_ssprk3_handles_ir_byte_identical_to_legacy():
    legacy = adctime.Program("ssprk3")
    _ssprk3_legacy(legacy, "plasma")
    legacy.validate()
    handles = adctime.Program("ssprk3")
    _ssprk3_handles(handles, "plasma")
    handles.validate()
    assert legacy._ir_hash() == handles._ir_hash(), (
        "SSPRK3 via handles must produce a byte-identical IR hash to the legacy style\n"
        "  legacy : %s\n  handles: %s" % (legacy._ir_hash(), handles._ir_hash()))


def test_handles_carry_no_ndarray():
    P = adctime.Program("nodata")
    U = P.state("U", block="plasma")
    s1 = U.stage(1)
    nxt = U.next
    prev = U.prev
    # no handle (TimeState / _Version / _Prev) owns a numpy array in any attribute
    for handle in (U, s1, nxt, prev):
        for attr, val in vars(handle).items():
            assert type(val).__module__ != "numpy", (
                "%r.%s is a numpy object (%r); handles must carry no runtime data"
                % (handle, attr, type(val)))
            assert not (hasattr(val, "shape") and hasattr(val, "dtype")), (
                "%r.%s looks like an ndarray; handles must carry no runtime data"
                % (handle, attr))


def main():
    test_current_state_is_read_only()
    test_define_prev_rejected()
    test_use_before_define_raises()
    test_double_define_rejected()
    test_prev_without_keep_history_raises()
    test_keep_history_then_prev_reads_history()
    test_ssprk3_handles_ir_byte_identical_to_legacy()
    test_handles_carry_no_ndarray()
    print("test_time_handles : tout est vert")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
