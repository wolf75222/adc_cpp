"""adc.time codegen (epic ADC-399 / ADC-401, ADC-407): Program.emit_cpp_program.

`emit_cpp_program` lowers the Program IR to the C++ source of a problem.so by a topological SSA walk.
This test pins the generated source: the stable .so ABI (adc_program_abi_key via the
ADC_ABI_KEY_LITERAL preprocessor literal -- never the interposable inline -- plus adc_program_name /
adc_program_hash / adc_install_program), the Forward-Euler body, and that a multi-stage scheme
(SSPRK2) now lowers (a scratch state + a second rhs + a lincomb commit). Constructs the codegen
cannot lower yet -- more than one block, named sources beyond 'default' -- must be REFUSED with a
clear NotImplementedError, never silently mis-lowered. Pure Python (no compile); skips if adc is
unavailable.
"""
import sys


def _adc_time():
    try:
        import adc.time as t
    except Exception as exc:  # adc not importable in this environment -> skip, never fake
        print("skip test_time_codegen (adc.time unavailable: %s)" % exc)
        sys.exit(0)
    return t


def _forward_euler(t):
    P = t.Program("forward_euler_program")
    dt = P.dt
    U = P.state("plasma")
    f = P.solve_fields(U)
    R = P.rhs(state=U, fields=f, flux=True, sources=["default"])
    P.commit("plasma", P.linear_combine("U1", U + dt * R))
    return P


def _ssprk2(t):
    P = t.Program("ssprk2_program")
    dt = P.dt
    U0 = P.state("plasma")
    f0 = P.solve_fields(U0)
    k0 = P.rhs(state=U0, fields=f0, flux=True, sources=["default"])
    U1 = P.linear_combine("U1", U0 + dt * k0)
    f1 = P.solve_fields(U1)
    k1 = P.rhs(state=U1, fields=f1, flux=True, sources=["default"])
    P.commit("plasma", P.linear_combine("U2", 0.5 * U0 + 0.5 * (U1 + dt * k1)))
    return P


def test_forward_euler_abi(t):
    P = _forward_euler(t)
    src = P.emit_cpp_program()
    for tok in ('extern "C"', "ADC_ABI_KEY_LITERAL", "adc_program_abi_key", "adc_program_name",
                "adc_program_hash", "adc_install_program",
                "adc::runtime::program::ProgramContext ctx(sys)"):
        assert tok in src, "generated source missing %r" % tok
    assert '"forward_euler_program"' in src, "program name not embedded"
    assert P._ir_hash() in src, "IR hash not embedded (cache/restart key)"


def test_forward_euler_algorithm(t):
    # FE: base = ctx.state(0); solve_fields; R = rhs_into(0, base); acc += dt*R; commit via lincomb.
    src = _forward_euler(t).emit_cpp_program()
    for frag in ("ctx.solve_fields();",
                 "= ctx.state(0);",
                 "ctx.rhs_scratch_like(",
                 "ctx.rhs_into(0, ",
                 "ctx.scratch_state_like(",
                 "static_cast<adc::Real>(dt)",
                 "ctx.axpy(",
                 "ctx.lincomb("):
        assert frag in src, "generated FE body missing %r" % frag
    assert "ctx.n_blocks()" not in src, "single-block codegen should target ctx.state(0), not a loop"


def test_multistage_lowers(t):
    # SSPRK2 now LOWERS (multi-stage codegen): a scratch state, two rhs_into, a lincomb commit, the 0.5
    # weights. (It previously raised NotImplementedError; that restriction is lifted.)
    src = _ssprk2(t).emit_cpp_program()
    assert src.count("ctx.rhs_into(") >= 2, "SSPRK2 should evaluate the RHS at two stages"
    assert "ctx.scratch_state_like(" in src, "SSPRK2 needs an intermediate scratch state"
    assert "ctx.lincomb(" in src, "the committed stage writes the block state via lincomb"
    assert "0.5" in src, "SSPRK2 weights (0.5) should appear in the generated source"


def test_includes_present(t):
    src = _forward_euler(t).emit_cpp_program()
    for inc in ("adc/runtime/program/program_context.hpp",
                "adc/runtime/dynamic/abi_key.hpp",
                "adc/mesh/storage/multifab.hpp"):
        assert ("#include <%s>" % inc) in src, "missing #include <%s>" % inc


def test_named_source_refused(t):
    # A non-default named source needs a source mask (Phase 4) -> refuse, never mis-lower.
    P = t.Program("electric_program")
    dt = P.dt
    U = P.state("plasma")
    f = P.solve_fields(U)
    R = P.rhs(state=U, fields=f, flux=True, sources=["electric"])
    P.commit("plasma", P.linear_combine("U1", U + dt * R))
    try:
        P.emit_cpp_program()
    except NotImplementedError as exc:
        assert "source" in str(exc).lower()
    else:
        raise AssertionError("expected NotImplementedError for a non-default named source")


def test_multiblock_refused(t):
    # Two committed blocks -> multi-block is a later phase; refuse with a clear message.
    P = t.Program("two_block")
    dt = P.dt
    for blk in ("a", "b"):
        U = P.state(blk)
        f = P.solve_fields(U)
        R = P.rhs(state=U, fields=f, flux=True, sources=["default"])
        P.commit(blk, P.linear_combine(blk + "_next", U + dt * R))
    try:
        P.emit_cpp_program()
    except NotImplementedError as exc:
        assert "block" in str(exc).lower()
    else:
        raise AssertionError("expected NotImplementedError for a multi-block Program")


def test_uncommitted_refused(t):
    # An empty Program (no commit) must fail validation, not emit garbage.
    P = t.Program("empty")
    try:
        P.emit_cpp_program()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an uncommitted Program")


def _run():
    t = _adc_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    print("PASS test_time_codegen (%d checks)" % len(fns))


if __name__ == "__main__":
    _run()
