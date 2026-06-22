"""adc.time codegen (epic ADC-399 / ADC-401 Phase 2c-ii): Program.emit_cpp_program.

`emit_cpp_program` lowers the Program IR to the C++ source of a problem.so. This test pins the
Forward-Euler lowering: the generated source must carry the stable .so ABI (adc_program_abi_key via
the ADC_ABI_KEY_LITERAL preprocessor literal -- never the interposable inline -- plus
adc_program_name / adc_program_hash / adc_install_program) and must express the FE algorithm purely
through ProgramContext primitives (solve_fields + rhs_into + axpy with the dt coefficient). This is
the SAME source shape tests/test_program_loader.cpp compiles, loads, and runs to bit-parity in CI,
so structural identity here means the generated .so compiles and runs there.

Schemes the MVP codegen cannot yet lower (multi-stage, which needs scratch states; named sources
beyond 'default', which need source masks) must be REFUSED with NotImplementedError, never silently
mis-lowered. Pure Python (no compile); skips cleanly if the adc module is unavailable.
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
    U1 = P.linear_combine("U1", U + dt * R)
    P.commit("plasma", U1)
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
    src = _forward_euler(t).emit_cpp_program()
    for line in ("ctx.solve_fields();",
                 "for (int b = 0; b < ctx.n_blocks(); ++b)",
                 "adc::MultiFab& U = ctx.state(b);",
                 "adc::MultiFab R = ctx.rhs_scratch_like(U);",
                 "ctx.rhs_into(b, U, R);",
                 "ctx.axpy(U, static_cast<adc::Real>(dt), R);"):
        assert line in src, "generated FE body missing %r" % line


def test_includes_present(t):
    src = _forward_euler(t).emit_cpp_program()
    for inc in ("adc/runtime/program/program_context.hpp",
                "adc/runtime/dynamic/abi_key.hpp",
                "adc/mesh/storage/multifab.hpp"):
        assert ("#include <%s>" % inc) in src, "missing #include <%s>" % inc


def test_multistage_refused(t):
    # SSPRK2 lowers to two linear_combine states (a scratch stage) -> not yet supported by the MVP.
    P = t.Program("ssprk2_program")
    dt = P.dt
    U0 = P.state("plasma")
    f0 = P.solve_fields(U0)
    k0 = P.rhs(state=U0, fields=f0, flux=True, sources=["default"])
    U1 = P.linear_combine("U1", U0 + dt * k0)
    f1 = P.solve_fields(U1)
    k1 = P.rhs(state=U1, fields=f1, flux=True, sources=["default"])
    U2 = P.linear_combine("U2", 0.5 * U0 + 0.5 * (U1 + dt * k1))
    P.commit("plasma", U2)
    try:
        P.emit_cpp_program()
    except NotImplementedError as exc:
        assert "single" in str(exc).lower() or "scratch" in str(exc).lower()
    else:
        raise AssertionError("expected NotImplementedError for a multi-stage Program")


def test_named_source_refused(t):
    # Single stage, but a non-default named source needs a source mask (Phase 4) -> refuse.
    P = t.Program("electric_program")
    dt = P.dt
    U = P.state("plasma")
    f = P.solve_fields(U)
    R = P.rhs(state=U, fields=f, flux=True, sources=["electric"])
    U1 = P.linear_combine("U1", U + dt * R)
    P.commit("plasma", U1)
    try:
        P.emit_cpp_program()
    except NotImplementedError as exc:
        assert "source" in str(exc).lower()
    else:
        raise AssertionError("expected NotImplementedError for a non-default named source")


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
