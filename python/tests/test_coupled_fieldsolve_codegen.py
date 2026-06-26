#!/usr/bin/env python3
"""Coupled multi-block field-solve CODEGEN (Spec 3 section 12.3, criterion 24; ADC-457).

``P.solve_fields_from_blocks([U0, U1, ...])`` is a SIMULTANEOUS coupled Poisson where EVERY listed
block contributes its own stage state at once. Its IR already builds (ADC-426); this test exercises
the LOWERING ADC-457 adds: ``_check_lowerable`` no longer refuses it, and ``emit_cpp_program``
produces ``ctx.solve_fields_from_blocks(<vec>)`` with each listed block slotted by index. The vector
is sized to ``ctx.n_blocks()`` (a nullptr entry uses the block's live state), so the runtime sees
each coupled block at its stage state into the one shared phi/aux.

Pure-Python codegen check (always runs when pops.time imports; skips cleanly if _pops is absent). The
.so that runs the coupled solve is validated on ROMEO (Kokkos-only AOT, not buildable host-only)."""

import sys


def _skip(msg):
    print("skip test_coupled_fieldsolve_codegen (%s)" % msg)
    sys.exit(0)


def _adc_time():
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


def raises(exc_types, fn):
    try:
        fn()
    except exc_types:
        return True
    except Exception:  # noqa: BLE001 -- the wrong exception type is a failure, not a pass
        return False
    return False


def coupled_program(t, name, blocks):
    """An N-block Forward-Euler program with ONE simultaneous coupled field solve over @p blocks: the
    shared aux is re-filled from all blocks' stage states at once, then each block advances
    U_blk1 = U_blk + dt*rhs(U_blk, fields) with the default/composite source (no model needed)."""
    P = t.Program(name)
    dt = P.dt
    states = [P.state(b) for b in blocks]
    f = P.solve_fields_from_blocks(states)  # SIMULTANEOUS coupled solve over every block at once
    for b, U in zip(blocks, states, strict=True):
        R = P.rhs(state=U, fields=f, sources=["default"])
        P.commit(b, P.linear_combine(b + "_next", U + dt * R))
    return P


def main():
    t = _adc_time()
    print("== coupled multi-block field-solve codegen (ADC-457) ==")

    # (1) _check_lowerable no longer raises for solve_fields_from_blocks (3 blocks, N>2).
    blocks = ("e_n", "i_n", "n_n")
    P = coupled_program(t, "coupled_three", blocks)
    chk(not raises(NotImplementedError, lambda: P._check_lowerable(None)),
        "_check_lowerable(None) accepts solve_fields_from_blocks (no longer deferred)")

    # (2) emit_cpp_program lowers the coupled solve to ctx.solve_fields_from_blocks(<vec>).
    src = P.emit_cpp_program()
    chk("ctx.solve_fields_from_blocks(" in src,
        "emit contains the coupled multi-block solve call")
    chk("std::vector<const pops::MultiFab*>" in src,
        "emit builds a per-block MultiFab pointer vector")
    chk("ctx.n_blocks()" in src,
        "the pointer vector is sized to ctx.n_blocks() (nullptr = the block's live state)")

    # (3) each listed block slots its stage state by index (one `[k] = &state` push per block).
    chk(src.count("] = &") >= len(blocks),
        "every listed block (%d) slots its stage state by index" % len(blocks))
    # The three blocks bind ctx.state(0/1/2) and the coupled solve precedes the per-block RHS.
    for k in range(len(blocks)):
        chk("ctx.state(%d)" % k in src, "block index %d binds ctx.state(%d)" % (k, k))
    chk(src.index("ctx.solve_fields_from_blocks(") < src.index("ctx.rhs_into("),
        "the coupled field solve is emitted before the per-block RHS reads the shared aux")

    # (4) a 2-block coupled solve also lowers (parity with the per-block solve_fields path).
    P2 = coupled_program(t, "coupled_two", ("a", "b"))
    src2 = P2.emit_cpp_program()
    chk("ctx.solve_fields_from_blocks(" in src2 and src2.count("] = &") >= 2,
        "a 2-block coupled solve lowers with both blocks slotted")

    print("%s test_coupled_fieldsolve_codegen" % ("FAIL (%d)" % fails if fails else "PASS"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
