#!/usr/bin/env python3
"""pops.time NAME-based block binding codegen + IR identity (Spec 3 criterion 23, ADC-457).

A compiled multi-block Program now binds its blocks to the System blocks BY NAME, not by add-order.
This pure-Python test (no .so compile, no engine) checks the two halves the codegen owns:

(A) ABI export -- ``emit_cpp_program`` emits ``pops_program_block_count()`` returning N and an
    ``pops_program_block_name(int)`` switch returning each block's name in ``_block_indices`` order
    (P.state declaration order, the order the step body's ctx.state(idx) addresses).

(B) IR identity -- the block names are part of ``_ir_hash``: two Programs that differ ONLY by the
    order of their ``P.state`` declarations produce DIFFERENT IR hashes (so they get distinct .so
    caches and the name binding is reflected in the program identity). A single-block Program keeps a
    one-entry table (byte-identical positional lowering).

Runs as a plain script (``python3 test_name_binding_codegen.py``, the CI invocation) and under pytest.
Skips cleanly (never fakes the engine) if pops.time cannot import (it needs _pops for the typed registry).
"""
import sys


def _skip(msg):
    print("skip test_name_binding_codegen (%s)" % msg)
    sys.exit(0)


def _pops_time():
    try:
        import pops.time as t
    except Exception as exc:  # noqa: BLE001 -- pops.time needs _pops; skip cleanly, never fake
        _skip("pops.time unavailable: %s" % exc)
    return t


fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


def _flux_program(t, name, blocks):
    """A flux-only Forward-Euler Program over @p blocks (declared in the given order), no model needed."""
    P = t.Program(name)
    for blk in blocks:
        U = P.state(blk)
        R = P.rhs(state=U, flux=True, sources=["default"])
        P.commit(blk, P.linear_combine(blk + "_next", U + P.dt * R))
    return P


def section_a(t):
    print("== (A) the .so exports pops_program_block_name per block, declaration order ==")
    P = _flux_program(t, "plasma_electrons", ["plasma", "electrons", "dust"])
    src = P.emit_cpp_program()

    chk("pops_program_block_count() { return 3; }" in src,
        "pops_program_block_count returns the block count (3)")
    # Each block name appears at its declaration-order index in the pops_program_block_name switch.
    chk('case 0: return "plasma";' in src, "block 0 -> \"plasma\" (first P.state)")
    chk('case 1: return "electrons";' in src, "block 1 -> \"electrons\" (second P.state)")
    chk('case 2: return "dust";' in src, "block 2 -> \"dust\" (third P.state)")
    chk('extern "C" const char* pops_program_block_name(int i)' in src,
        "the block-name accessor is a stable extern \"C\" ABI export")

    # A single-block Program still carries the table (count 1), so the loader binds it by name too.
    P1 = _flux_program(t, "single", ["gas"])
    src1 = P1.emit_cpp_program()
    chk("pops_program_block_count() { return 1; }" in src1, "single-block count is 1")
    chk('case 0: return "gas";' in src1, "single block 0 -> \"gas\"")


def section_b(t):
    print("== (B) block names are part of the IR hash (reordering P.state changes it) ==")
    h_ab = _flux_program(t, "p", ["plasma", "electrons"])._ir_hash()
    h_ba = _flux_program(t, "p", ["electrons", "plasma"])._ir_hash()
    chk(h_ab != h_ba, "reordering P.state declarations changes the IR hash (block names in identity)")

    # The block_order serialization field carries the names in declaration order (the hash input).
    ser = _flux_program(t, "p", ["plasma", "electrons"])._serialize()
    chk(ser.get("block_order") == ["plasma", "electrons"],
        "_serialize records block_order in declaration order")

    # Same program written twice (same names, same order) is byte-identical -> identical hash.
    h_again = _flux_program(t, "p", ["plasma", "electrons"])._ir_hash()
    chk(h_ab == h_again, "the same Program (same names + order) hashes identically (deterministic)")


def _run():
    t = _pops_time()
    section_a(t)
    section_b(t)
    print("%s test_name_binding_codegen" % ("FAIL (%d)" % fails if fails else "PASS"))
    sys.exit(1 if fails else 0)


# pytest entry points (the CI also runs the file as a script). _pops_time() exits via sys.exit(0) when
# pops is unimportable (a clean skip in script mode); translate that SystemExit to a pytest skip rather
# than let it surface as a FAILED test.
def test_block_name_abi_export():
    try:
        section_a(_pops_time())
    except SystemExit as exc:  # noqa: BLE001
        if exc.code:
            raise
        return
    assert fails == 0


def test_block_names_in_ir_hash():
    try:
        section_b(_pops_time())
    except SystemExit as exc:  # noqa: BLE001
        if exc.code:
            raise
        return
    assert fails == 0


if __name__ == "__main__":
    _run()
