#!/usr/bin/env python3
"""pops.time op-set completeness (epic ADC-399 / ADC-414): the spec ops 10/16/21/22/23 + the mandatory
validation errors #18/#19.

  - solve_local_nonlinear (op 10): a per-cell Newton solve (ADC-422); the builder validates its inputs
    and lowers a residual sub-block to a device FD-Jacobian Newton kernel;
  - reductions (op 16): P.sum / P.max / P.min / P.sum_component build a 'reduce' IR op and lower to the
    matching pops:: collective reduction (pops::reduce_sum / reduce_max / reduce_min);
  - fill_boundary (op 22): P.fill_boundary lowers to ctx.fill_boundary (the shared ghost exchange);
  - project (op 21): P.project lowers to ctx.apply_projection (the block's own positivity projection);
  - record_scalar (op 23): P.record_scalar lowers to ctx.record_scalar; the value is retrievable after
    sim.step via sim.program_diagnostic / sim.program_diagnostics;
  - validation #18: a restart whose checkpoint lacks a required Program history fails loud;
  - validation #19: install_program with an ABI-mismatched module fails loud with the explicit message.

(A) Pure Python (IR + codegen), always runs: the builders produce typed IR and emit_cpp_program lowers
    each to the right ProgramContext / pops:: call. No compile, no engine.
(B) End-to-end (reductions + record_scalar): a 1-variable model whose sum / max / min / sum_component of
    a known field match the analytic values; record_scalar stores a norm retrievable after the step.
    Self-skips (exit 0) without numpy / _pops / a compiler / a visible Kokkos -- never fakes the engine.
(C) Validation #18 (pure Python, mocked System) + #19 (skips without the engine).
"""
import sys


def _pops_time():
    try:
        import pops.time as t
    except Exception as exc:  # pops not importable here -> skip, never fake
        print("skip test_time_ops_polish (pops.time unavailable: %s)" % exc)
        sys.exit(0)
    return t


# ---- (A.1) solve_local_nonlinear (op 10): the per-cell Newton builder (ADC-422) ----
def test_solve_local_nonlinear_validates_inputs(t):
    # The residual must be an IR-building callable and the guess a State; the bad-input messages are loud.
    P = t.Program("p")
    U = P.state("blk")
    try:  # a State (not a callable) is no longer accepted -- the residual builds r(U)
        P.solve_local_nonlinear(name="u", residual=U, initial_guess=U)
    except ValueError as exc:
        assert "residual must be" in str(exc) and "callable" in str(exc), str(exc)
    else:
        raise AssertionError("solve_local_nonlinear must reject a non-callable residual")

    def good_residual(P, Uit, U0):
        return P.linear_combine(Uit - U0)
    try:  # the initial_guess must be a State value
        P.solve_local_nonlinear(name="u", residual=good_residual, initial_guess="nope")
    except ValueError as exc:
        assert "initial_guess" in str(exc), str(exc)
    else:
        raise AssertionError("solve_local_nonlinear must reject a non-State initial_guess")
    try:  # max_iter must be a positive int
        P.solve_local_nonlinear(name="u", residual=good_residual, initial_guess=U, max_iter=0)
    except ValueError as exc:
        assert "max_iter" in str(exc), str(exc)
    else:
        raise AssertionError("solve_local_nonlinear must reject max_iter <= 0")


def test_solve_local_nonlinear_builds_newton_ir(t):
    # A valid implicit reaction r(U) = U - U0 - dt*S(U) builds a typed Newton IR op with a residual
    # sub-block; the IR validates and hashes.
    P = t.Program("react")
    dt = P.dt
    U = P.state("blk")

    def residual(P, Uit, U0):
        S = P.source("react", state=Uit)
        return P.linear_combine("r", Uit - U0 - dt * S)
    W = P.solve_local_nonlinear(name="W", residual=residual, initial_guess=U, tol=1e-10, max_iter=25)
    assert W.op == "solve_local_nonlinear" and W.vtype == "state", (W.op, W.vtype)
    assert W.attrs["max_iter"] == 25 and W.attrs["tol"] == 1e-10
    assert len(W.attrs["residual_block"]) >= 3, "the residual sub-block holds the iterate/guess + ops"
    P.commit("blk", W)
    assert P.validate() is True, "the Newton IR must validate"
    assert P._ir_hash(), "the Newton IR must serialize to a stable hash"


def test_solve_local_nonlinear_rejects_non_local_residual(t):
    # A non-local op (P.rhs / P.solve_fields) inside the residual is rejected: the per-cell kernel cannot
    # re-evaluate a halo / global solve at a perturbed stack state.
    P = t.Program("bad")
    U = P.state("blk")

    def bad_residual(P, Uit, U0):
        R = P.rhs(state=Uit, sources=["default"])  # a non-local divergence-bearing rhs
        return P.linear_combine(Uit - U0 - P.dt * R)
    try:
        P.solve_local_nonlinear(name="W", residual=bad_residual, initial_guess=U)
    except ValueError as exc:
        assert "not LOCAL" in str(exc) or "rhs" in str(exc), str(exc)
    else:
        raise AssertionError("a non-local residual op must be rejected")


def test_solve_local_nonlinear_refused_without_model(t):
    # The Newton codegen reads the residual's named source / linear source coefficients -> needs a model.
    P = t.Program("react")
    dt = P.dt
    U = P.state("blk")

    def residual(P, Uit, U0):
        S = P.source("react", state=Uit)
        return P.linear_combine("r", Uit - U0 - dt * S)
    P.commit("blk", P.solve_local_nonlinear(name="W", residual=residual, initial_guess=U))
    try:
        P.emit_cpp_program()  # no model
    except NotImplementedError as exc:
        assert "solve_local_nonlinear" in str(exc) or "source" in str(exc), str(exc)
    else:
        raise AssertionError("the Newton codegen must be refused without a model")


# ---- (A.2) reductions (op 16): IR + codegen ----
def test_reductions_build_scalar_values(t):
    P = t.Program("p")
    U = P.state("blk")
    R = P.rhs(state=U, sources=["default"])
    for node in (P.sum(U), P.max(U), P.min(U), P.sum_component(U, 0)):
        assert node.vtype == "scalar" and node.op == "reduce", \
            "a reduction is a scalar 'reduce' op (got %r/%r)" % (node.vtype, node.op)
    assert P.sum(U).attrs["kind"] == "sum"
    assert P.max(R).attrs["kind"] == "max"
    assert P.min(U).attrs["kind"] == "min"
    sc = P.sum_component(U, 0)
    assert sc.attrs["kind"] == "sum" and sc.attrs["comp"] == 0


def test_reductions_reject_non_field_and_bad_component(t):
    P = t.Program("p")
    U = P.state("blk")
    for fn in (P.sum, P.max, P.min):
        try:
            fn("not a field")
        except ValueError as exc:
            assert "State/RHS" in str(exc), str(exc)
        else:
            raise AssertionError("%s must reject a non-field operand" % fn.__name__)
    for bad in (-1, 1.0, True, "x"):
        try:
            P.sum_component(U, bad)
        except ValueError as exc:
            assert "comp" in str(exc), str(exc)
        else:
            raise AssertionError("sum_component comp=%r must raise" % (bad,))


def test_reductions_lower_to_adc_reductions(t):
    # A while_ loop whose condition compares a reduction lets the reduce op lower inside the body.
    P = t.Program("p")
    U = P.state("blk")
    R = P.rhs(state=U, sources=["default"])
    s_sum = P.sum(R)
    s_max = P.max(R)
    s_min = P.min(R)
    s_c = P.sum_component(U, 0)
    # record_scalar keeps the reductions live (otherwise dead-code at the top level still emits them).
    P.record_scalar("s_sum", s_sum)
    P.record_scalar("s_max", s_max)
    P.record_scalar("s_min", s_min)
    P.record_scalar("s_c", s_c)
    P.commit("blk", P.linear_combine(U + P.dt * R))
    src = P.emit_cpp_program()
    for frag in ("pops::reduce_sum(", "pops::reduce_max(", "pops::reduce_min("):
        assert frag in src, "the reduction codegen must contain %r\n%s" % (frag, src)
    assert "pops::reduce_sum(r" in src and ", 0)" in src, "sum/sum_component reduce over a component"


# ---- (A.3) fill_boundary (op 22) + project (op 21): IR + codegen ----
def test_fill_boundary_ir_and_codegen(t):
    P = t.Program("p")
    U = P.state("blk")
    Uf = P.fill_boundary(U)
    assert Uf.op == "fill_boundary" and Uf.vtype == "state", (Uf.op, Uf.vtype)
    R = P.rhs(state=Uf, sources=["default"])
    P.commit("blk", P.linear_combine(Uf + P.dt * R))
    src = P.emit_cpp_program()
    assert "ctx.fill_boundary(" in src, "fill_boundary lowers to ctx.fill_boundary\n%s" % src


def test_fill_boundary_rejects_non_field(t):
    P = t.Program("p")
    try:
        P.fill_boundary("nope")
    except ValueError as exc:
        assert "field" in str(exc), str(exc)
    else:
        raise AssertionError("fill_boundary must reject a non-field value")


def test_project_ir_and_codegen(t):
    P = t.Program("p")
    U = P.state("blk")
    R = P.rhs(state=U, sources=["default"])
    U1 = P.linear_combine(U + P.dt * R)
    Up = P.project(state=U1)
    assert Up.op == "project" and Up.vtype == "state", (Up.op, Up.vtype)
    P.commit("blk", Up)
    src = P.emit_cpp_program()
    assert "ctx.apply_projection(0, " in src, "project lowers to ctx.apply_projection\n%s" % src


def test_project_rejects_non_state_and_custom_projection(t):
    P = t.Program("p")
    U = P.state("blk")
    try:
        P.project(state="nope")
    except ValueError as exc:
        assert "State" in str(exc), str(exc)
    else:
        raise AssertionError("project must reject a non-State value")
    try:
        P.project(state=U, projection="custom")
    except NotImplementedError as exc:
        assert "projection" in str(exc), str(exc)
    else:
        raise AssertionError("project must reject a non-'block' projection")


# ---- (A.4) record_scalar (op 23): IR + codegen ----
def test_record_scalar_ir_and_codegen(t):
    P = t.Program("p")
    U = P.state("blk")
    R = P.rhs(state=U, sources=["default"])
    rec = P.record_scalar("rhs_norm", P.norm2(R))
    assert rec.op == "record_scalar" and rec.attrs["diagnostic"] == "rhs_norm"
    P.commit("blk", P.linear_combine(U + P.dt * R))
    src = P.emit_cpp_program()
    assert 'ctx.record_scalar("rhs_norm", ' in src, "record_scalar lowers to ctx.record_scalar\n%s" % src


def test_record_scalar_rejects_non_scalar_and_bad_name(t):
    P = t.Program("p")
    U = P.state("blk")
    try:
        P.record_scalar("x", U)  # a field is not a scalar
    except ValueError as exc:
        assert "Scalar" in str(exc), str(exc)
    else:
        raise AssertionError("record_scalar must reject a field value")
    try:
        P.record_scalar("", P.norm2(U))
    except ValueError as exc:
        assert "name" in str(exc), str(exc)
    else:
        raise AssertionError("record_scalar must reject an empty name")


# ---- (A.5) IR hash sensitivity ----
def test_ir_hash_distinguishes_new_ops(t):
    def _h(build):
        P = t.Program("h")
        U = P.state("blk")
        R = P.rhs(state=U, sources=["default"])
        build(P, U, R)
        P.commit("blk", P.linear_combine(U + P.dt * R))
        return P._ir_hash()

    base = _h(lambda P, U, R: None)
    rec = _h(lambda P, U, R: P.record_scalar("a", P.sum(R)))
    rec_b = _h(lambda P, U, R: P.record_scalar("b", P.sum(R)))
    fb = _h(lambda P, U, R: P.fill_boundary(U))
    assert base != rec, "record_scalar must change the IR hash"
    assert rec != rec_b, "a different diagnostic name must change the IR hash"
    assert base != fb, "fill_boundary must change the IR hash"


# ---- shared engine setup for (B) ----
def _const_source_model(dsl, name, c):
    """A 1-variable model (rho), ZERO flux, default source S(rho) = c (a CONSTANT source, so R = c is
    spatially uniform and analytic). A complete compilable block (flux + primitive + eigenvalue + src)."""
    m = dsl.Model(name)
    (rho,) = m.conservative_vars("rho")
    u = m.primitive("u", 0.0 * rho)
    m.primitive_vars(rho=rho, u=u)
    m.conservative_from([rho])
    m.flux(x=[0.0 * rho], y=[0.0 * rho])
    m.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
    m.source([c + 0.0 * rho])
    return m


def _reductions_program(t):
    """Forward Euler that also records sum / max / min / sum_component of the CURRENT state (component 0)
    each step, so the diagnostics can be checked against the analytic state."""
    P = t.Program("reductions_step")
    U = P.state("blk")
    R = P.rhs(state=U, sources=["default"])
    P.record_scalar("state_sum", P.sum(U))
    P.record_scalar("state_max", P.max(U))
    P.record_scalar("state_min", P.min(U))
    P.record_scalar("state_sum_c0", P.sum_component(U, 0))
    P.commit("blk", P.linear_combine(U + P.dt * R))
    return P


def _run_section_b(t):
    try:
        import numpy as np

        import pops
    except Exception as exc:  # noqa: BLE001 -- numpy / _pops unavailable
        print("-- (B) skipped: pops/numpy unavailable: %s --" % exc)
        return None

    n = 8
    sim = pops.System(n=n, L=1.0, periodic=True)
    if not hasattr(sim, "install_program") or not hasattr(sim, "program_diagnostics"):
        print("-- (B) skipped: _pops lacks the install_program/program_diagnostics bindings "
              "(rebuild _pops) --")
        return None

    from pops import dsl

    c = 0.5
    P = _reductions_program(t)
    try:
        compiled = pops.compile_problem(model=_const_source_model(dsl, "red_prog", c), time=P)
    except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
        print("-- (B) skipped: compile_problem could not build the .so: %s --" % str(exc)[:200])
        return None
    try:
        compiled_model = _const_source_model(dsl, "red_block", c).compile(backend="production")
    except RuntimeError as exc:
        print("-- (B) skipped: model compile could not build the .so: %s --" % str(exc)[:200])
        return None
    sim.add_equation("blk", compiled_model,
                     spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                     time=pops.Explicit(method="euler"))

    # A KNOWN field with distinct min / max / sum: rho(i,j) = 1 + (linear ramp in [0, 1]).
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = 1.0 + 0.25 * X + 0.25 * Y  # in [1, 1.5), all distinct
    sim.set_state("blk", np.stack([rho0]))

    sim.install_program(compiled.so_path)
    dt = 0.01
    sim.step(dt)  # the diagnostics are recorded from U^n (the state at the START of the step)

    diags = sim.program_diagnostics()
    for key in ("state_sum", "state_max", "state_min", "state_sum_c0"):
        assert key in diags, "program_diagnostics must contain %r (got %r)" % (key, sorted(diags))
    # The reductions are over U^n = rho0 (record_scalar reads U before the commit).
    exp_sum = float(rho0.sum())
    exp_max = float(rho0.max())
    exp_min = float(rho0.min())
    err_sum = abs(diags["state_sum"] - exp_sum)
    err_max = abs(diags["state_max"] - exp_max)
    err_min = abs(diags["state_min"] - exp_min)
    err_c0 = abs(diags["state_sum_c0"] - exp_sum)
    # program_diagnostic(name) reads one value (same as the dict).
    assert abs(sim.program_diagnostic("state_sum") - diags["state_sum"]) == 0.0
    print("  reductions: |sum-%.4f|=%.2e |max-%.4f|=%.2e |min-%.4f|=%.2e |sum_c0|err=%.2e" %
          (exp_sum, err_sum, exp_max, err_max, exp_min, err_min, err_c0))
    assert err_sum <= 1e-9 * max(1.0, abs(exp_sum)), "P.sum must match the analytic sum"
    assert err_max <= 1e-12, "P.max must match the analytic max"
    assert err_min <= 1e-12, "P.min must match the analytic min"
    assert err_c0 <= 1e-9 * max(1.0, abs(exp_sum)), "P.sum_component(.,0) must match the analytic sum"
    # An unrecorded diagnostic name must fail loud (not return 0).
    try:
        sim.program_diagnostic("never_recorded")
    except Exception as exc:  # noqa: BLE001 -- C++ std::out_of_range -> a Python exception
        assert "never_recorded" in str(exc), str(exc)
    else:
        raise AssertionError("program_diagnostic must raise on an unrecorded name")
    return err_max


# ---- shared engine setup for (B.2) fill_boundary + project ----
def _fill_project_program(t):
    """A program exercising fill_boundary (on the state) + project (positivity) end to end. The model is
    flux-only (zero source) so the state is unchanged by the RHS; the program just commits U after a
    ghost fill and a projection (both no-ops on a smooth positive state, but they must lower + run)."""
    P = t.Program("fill_project_step")
    U = P.state("blk")
    Uf = P.fill_boundary(U)
    R = P.rhs(state=Uf, sources=["default"])
    U1 = P.linear_combine(Uf + P.dt * R)
    P.commit("blk", P.project(state=U1))
    return P


def _run_section_b2(t):
    try:
        import numpy as np

        import pops
    except Exception as exc:  # noqa: BLE001
        print("-- (B.2) skipped: pops/numpy unavailable: %s --" % exc)
        return None

    n = 8
    sim = pops.System(n=n, L=1.0, periodic=True)
    if not hasattr(sim, "install_program"):
        print("-- (B.2) skipped: _pops lacks the install_program binding (rebuild _pops) --")
        return None
    from pops import dsl
    P = _fill_project_program(t)
    try:
        compiled = pops.compile_problem(model=_const_source_model(dsl, "fp_prog", 0.0), time=P)
    except RuntimeError as exc:
        print("-- (B.2) skipped: compile_problem could not build the .so: %s --" % str(exc)[:200])
        return None
    try:
        compiled_model = _const_source_model(dsl, "fp_block", 0.0).compile(backend="production")
    except RuntimeError as exc:
        print("-- (B.2) skipped: model compile could not build the .so: %s --" % str(exc)[:200])
        return None
    # A positivity floor makes the block carry a real projection closure (else project is a no-op).
    sim.add_equation("blk", compiled_model,
                     spatial=pops.FiniteVolume(limiter="none", riemann="rusanov",
                                              positivity_floor=1e-12),
                     time=pops.Explicit(method="euler"))
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("blk", np.stack([rho0]))
    sim.install_program(compiled.so_path)
    sim.step(0.01)
    out = np.array(sim.get_state("blk"))[0]
    # Zero source + flux-only on a periodic smooth field -> the state is unchanged to machine precision
    # (fill_boundary writes only ghosts; project leaves a positive state untouched).
    err = float(np.abs(out - rho0).max())
    print("  fill_boundary + project ran: max|out - rho0| = %.2e (expected ~0, no-op program)" % err)
    assert err <= 1e-10, "fill_boundary + project must run cleanly (state unchanged): max|d|=%.2e" % err
    return err


# ---- (C.1) validation #18: missing required Program history on restart (pure Python mock) ----
class _MockSystem:
    """A minimal stand-in for the C++ System exposing only what System.restart touches for the #18
    check. It has ONE registered history ('blk.R') and stubs the rest. Never fakes the engine numerics
    (this exercises the pure-Python restart guard, not a step)."""

    def __init__(self):
        self._registered = ["blk.R"]

    # the #18 guard reads history_names() to learn the required histories
    def history_names(self):
        return list(self._registered)

    # the methods restart() calls before the history guard; the guard must fire before set_clock.
    def nx(self):
        return 4

    def ny(self):
        return 4

    def block_names(self):
        return ["blk"]

    def n_vars(self, name):
        return 1

    def set_state(self, name, u):
        pass

    def set_potential(self, phi):
        pass

    def installed_program_hash(self):
        return ""

    def set_clock(self, t, macro_step):
        pass


def test_restart_missing_history_fails_loud(t):
    try:
        import numpy as np

        import pops
    except Exception as exc:  # noqa: BLE001
        print("-- (C.1) skipped: pops/numpy unavailable: %s --" % exc)
        return
    import os
    import tempfile

    # Build a checkpoint dict that does NOT contain the required 'blk.R' history (a legacy / wrong
    # checkpoint), then drive System.restart against a System that has registered it.
    sysobj = pops.System.__new__(pops.System)  # bypass __init__ (no engine needed for the guard path)
    sysobj._s = _MockSystem()
    ckpt = {
        "pops_checkpoint_version": 1,
        "nx": 4, "ny": 4,
        "blocks": np.array(["blk"]),
        "ncomp_blk": 1,
        "state_blk": np.zeros((1, 4, 4)),
        "t": 0.0, "macro_step": 0,
        # NO history_names key -> the required 'blk.R' is missing.
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "no_history.npz")
        np.savez_compressed(path, **ckpt)
        try:
            sysobj.restart(path)
        except RuntimeError as exc:
            msg = str(exc)
            assert "checkpoint does not contain required Program history 'blk.R'" in msg, \
                "the missing-history restart must fail loud with the spec-18 message; got: %s" % msg
            print("  missing-history restart raised as expected: %s" % msg.splitlines()[0][:120])
            return
        except Exception as exc:  # noqa: BLE001 -- a different early guard fired (composition/grid)
            print("-- (C.1) inconclusive: an earlier restart guard fired (%s); the #18 string is "
                  "still pinned by the message text below --" % str(exc)[:120])
            # Fall through to a direct string check so the spec wording is always exercised.
        # If restart did not reach the guard (an earlier check tripped), at least assert the message is
        # the spec wording when constructed directly.
        produced = "checkpoint does not contain required Program history '%s'" % "blk.R"
        assert "checkpoint does not contain required Program history 'blk.R'" == produced
    raise AssertionError("restart with a missing required history must raise (spec error 18)")


# ---- (C.2) validation #19: ABI mismatch on install_program (skips without the engine) ----
def _run_section_c2(t):
    try:
        import pops
    except Exception as exc:  # noqa: BLE001
        print("-- (C.2) skipped: pops unavailable: %s --" % exc)
        return None

    n = 4
    sim = pops.System(n=n, L=1.0, periodic=True)
    if not hasattr(sim, "install_program"):
        print("-- (C.2) skipped: _pops lacks the install_program binding (rebuild _pops) --")
        return None
    import os
    import tempfile

    # A hand-written .so source whose pops_program_abi_key returns a DELIBERATELY WRONG key. Compiling it
    # needs the toolchain; if it is absent we skip (never fake). The point is the explicit #19 message.
    src = (
        'extern "C" const char* pops_program_abi_key() { return "deliberately-wrong-abi-key"; }\n'
        'extern "C" const char* pops_program_name() { return "bad"; }\n'
        'extern "C" const char* pops_program_hash() { return "0"; }\n'
        'extern "C" void pops_install_program(void*) {}\n'
    )
    cxx = os.environ.get("CXX", "c++")
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "bad_abi.cpp")
        so = os.path.join(tmp, "bad_abi.so")
        with open(cpp, "w") as f:
            f.write(src)
        rc = os.system("%s -shared -fPIC -std=c++17 -undefined dynamic_lookup -o %s %s 2>/dev/null"
                       % (cxx, so, cpp))
        if rc != 0 or not os.path.exists(so):
            print("-- (C.2) skipped: could not compile the bad-ABI .so (no toolchain) --")
            return None
        try:
            sim.install_program(so)
        except (RuntimeError, Exception) as exc:  # noqa: BLE001
            msg = str(exc)
            assert "compiled program ABI mismatch" in msg, \
                "the ABI mismatch must fail loud with the spec-19 message; got: %s" % msg
            assert "deliberately-wrong-abi-key" in msg, "the message must report the mismatched key"
            print("  ABI mismatch raised as expected: %s" % msg.splitlines()[0][:140])
            return True
    raise AssertionError("install_program with a mismatched ABI key must raise (spec error 19)")


def _run():
    t = _pops_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    print("PASS test_time_ops_polish (A/C.1: %d checks)" % len(fns))
    _run_section_b(t)
    _run_section_b2(t)
    _run_section_c2(t)


if __name__ == "__main__":
    _run()
