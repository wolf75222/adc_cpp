#!/usr/bin/env python3
"""pops.time multistep histories + Adams-Bashforth 2, end to end (epic ADC-399 / ADC-406a).

A compiled `Program` can declare / read / write a SYSTEM-OWNED history field carried across macro-steps
(a HistoryManager in System::Impl, NOT a closure capture, so a later checkpoint slice can serialize it).
This enables Adams-Bashforth 2: ``U^{n+1} = U + dt*(3/2 R_n - 1/2 R_{n-1})`` then
``store_history(block.R, R_n)``.

  - ``P.history(name, lag=1)`` -> a State-typed value (the value @p lag macro-steps back);
  - ``P.store_history(name, value)`` -> a side-effecting op (copy the value into the current slot);
  - ``pops.time.std.adams_bashforth2(P, block)`` -> the AB2 IR (store-then-read, cold start = FE step 0).

The codegen lowers ``history`` -> ``ctx.history(...)``, ``store_history`` -> ``ctx.store_history(...)``,
and appends ``ctx.rotate_histories()`` at the END of the step body when any history is used.

COLD START (step 0): the runtime fills EVERY history slot on the FIRST store, so step 0 reads
R_{n-1} = R_0 and AB2 degenerates to one Forward-Euler step (U^1 = U^0 + dt R_0). The offline reference
mirrors this exactly (FE step 0, AB2 thereafter), so the comparison is to machine precision.

(A) Codegen / IR (pure Python, always runs): P.history / P.store_history build valid IR; the AB2 macro
    lowers; emit_cpp_program contains ctx.history / ctx.store_history / ctx.rotate_histories; the IR hash
    distinguishes history names and lags; the validation guards fire.

(B) End-to-end AB2 parity (skips unless the full toolchain is present): a 1-variable model (rho) with
    ZERO flux and a manufactured LINEAR source S(rho) = c*rho (so R = c*rho CHANGES every step), stepped
    N macro-steps; compare the final state to an OFFLINE reference running the IDENTICAL AB2 recurrence
    with the same FE cold start, cell by cell, to machine precision (spec test 37). Self-skips (exit 0)
    without numpy / _pops / install_program / a compiler / a visible Kokkos -- never fakes the engine.

(C) Absent-history rejection (spec test 38): a Program that reads P.history("missing.R", lag=1) and is
    stepped WITHOUT ever storing it -> sim.step surfaces a RuntimeError containing
    "history 'missing.R' with lag=1 was requested but not initialized".
"""
import sys


def _pops_time():
    try:
        import pops.time as t
    except Exception as exc:  # pops not importable here -> skip, never fake
        print("skip test_time_history (pops.time unavailable: %s)" % exc)
        sys.exit(0)
    return t


_C = 0.75  # source coefficient: S(rho) = _C * rho (a linear ODE rho' = c rho; R changes every step)


# ---- (A) codegen / IR: pure Python, always runs ----
def test_history_builds_state_value(t):
    P = t.Program("p")
    U = P.state("blk")
    R = P.rhs(state=U, sources=["default"])
    P.store_history("blk.R", R)
    Rp = P.history("blk.R", lag=1)
    assert Rp.vtype == "state", "P.history returns a State-typed value (got %r)" % Rp.vtype
    assert Rp.is_field(), "a history value is a grid field (affine algebra applies)"
    P.commit("blk", P.linear_combine(U + P.dt * (R - Rp)))
    assert P.validate() is True, "the history Program must validate"


def test_store_history_requires_a_field(t):
    P = t.Program("p")
    for bad in (5, "x", None):
        try:
            P.store_history("blk.R", bad)
        except ValueError as exc:
            assert "field" in str(exc), str(exc)
        else:
            raise AssertionError("store_history must reject a non-field value %r" % (bad,))


def test_history_lag_must_be_positive_int(t):
    P = t.Program("p")
    for bad in (0, -1, 1.0, True):
        try:
            P.history("blk.R", lag=bad)
        except ValueError as exc:
            assert "lag" in str(exc), str(exc)
        else:
            raise AssertionError("history lag=%r must raise (a positive int is required)" % (bad,))


def test_ab2_macro_lowers(t):
    P = t.Program("ab2")
    t.std.adams_bashforth2(P, "plasma")
    assert P.validate() is True, "the AB2 macro must validate"
    src = P.emit_cpp_program()
    for frag in ('ctx.history("plasma.R", 1)', 'ctx.store_history("plasma.R"',
                 "ctx.rotate_histories();"):
        assert frag in src, "the AB2 codegen must contain %r\n%s" % (frag, src)
    # The AB2 coefficients: +3/2 dt on R_n, -1/2 dt on R_{n-1}.
    assert "1.5 * dt" in src and "-0.5 * dt" in src, "AB2 weights 3/2, -1/2 on dt\n%s" % src


def test_store_before_read_in_body(t):
    """The store is emitted BEFORE the lag-1 READ (the cold-start fill makes step 0 valid). The read is
    the history line bound to a MultiFab& (``pops::MultiFab& ... = ctx.history(...)``); the bare
    ``ctx.history(...)`` at the top is only the depth-locking registration."""
    P = t.Program("ab2")
    t.std.adams_bashforth2(P, "plasma")
    src = P.emit_cpp_program()
    body = src[src.index("ctx.install"):]
    read = body.index("= ctx.history(\"plasma.R\", 1);")  # the bound read, not the bare registration
    assert body.index("ctx.store_history") < read, \
        "store_history must precede the lag-1 read in the step body\n%s" % body
    assert read < body.index("ctx.rotate_histories"), \
        "rotate_histories is the LAST history op of the step body\n%s" % body


def test_non_history_schemes_emit_no_rotate(t):
    for sched in ("forward_euler", "ssprk2", "ssprk3", "rk4"):
        P = t.Program(sched)
        getattr(t.std, sched)(P, "blk")
        src = P.emit_cpp_program()
        assert "ctx.rotate_histories" not in src, "%s must not rotate (no history)" % sched
        assert "ctx.history(" not in src, "%s must not read a history" % sched


def _hist_program(t, name, lag):
    P = t.Program("h")
    U = P.state("blk")
    R = P.rhs(state=U, sources=["default"])
    P.store_history(name, R)
    Rp = P.history(name, lag=lag)
    P.commit("blk", P.linear_combine(U + P.dt * (R - Rp)))
    return P


def test_ir_hash_distinguishes_name_and_lag(t):
    h_a1 = _hist_program(t, "a.R", 1)._ir_hash()
    h_b1 = _hist_program(t, "b.R", 1)._ir_hash()
    h_a2 = _hist_program(t, "a.R", 2)._ir_hash()
    assert h_a1 != h_b1, "a different history NAME must change the IR hash"
    assert h_a1 != h_a2, "a different history LAG must change the IR hash"


def test_absent_history_program_lowers(t):
    """A Program that reads a never-stored history still BUILDS and LOWERS (the failure is at runtime,
    spec test 38). The store is absent; the read still emits ctx.history."""
    P = t.Program("miss")
    U = P.state("blk")
    Rp = P.history("missing.R", lag=1)
    R = P.rhs(state=U, sources=["default"])
    P.commit("blk", P.linear_combine(U + P.dt * (R - Rp)))
    assert P.validate() is True
    src = P.emit_cpp_program()
    assert 'ctx.history("missing.R", 1)' in src, src
    assert "ctx.store_history" not in src, "the absent-history program never stores"


# ---- (B) end-to-end AB2 parity: skips unless the full toolchain is present ----
def _passive_source_model(name):
    """A 1-variable model (rho), ZERO flux, default LINEAR source S(rho) = _C*rho (so R = c*rho changes
    every step). A complete compilable block (flux + primitive + eigenvalue + source)."""
    from pops.physics.facade import Model
    m = Model(name)
    (rho,) = m.conservative_vars("rho")
    u = m.primitive("u", 0.0 * rho)
    m.primitive_vars(rho=rho, u=u)
    m.conservative_from([rho])
    m.flux(x=[0.0 * rho], y=[0.0 * rho])
    m.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
    m.source([_C * rho])  # default source folded by ctx.rhs_into
    return m


def _offline_ab2(rho0, dt, nsteps):
    """The IDENTICAL AB2 recurrence, cell by cell, with the same FE cold start the runtime uses:
        R_n = _C * rho_n
        rho_{n+1} = rho_n + dt*(3/2 R_n - 1/2 R_{n-1})     (R_{-1} := R_0 -> step 0 is FE)
    Returns the final rho after @p nsteps macro-steps."""
    rho = rho0.copy()
    r_prev = _C * rho  # cold start: R_{-1} = R_0 (first store fills all slots) -> step 0 is FE
    for _ in range(nsteps):
        r_n = _C * rho
        rho = rho + dt * (1.5 * r_n - 0.5 * r_prev)
        r_prev = r_n
    return rho


def _run_section_b(t):
    try:
        import numpy as np

        import pops
    except Exception as exc:  # noqa: BLE001  -- numpy / _pops unavailable in this interpreter
        print("-- (B) skipped: pops/numpy unavailable: %s --" % exc)
        return None

    n = 16
    sim = pops.System(n=n, L=1.0, periodic=True)
    if not hasattr(sim, "install_program"):
        print("-- (B) skipped: _pops lacks the install_program binding (rebuild _pops) --")
        return None

    from pops.physics.facade import Model

    P = t.Program("ab2_step")
    t.std.adams_bashforth2(P, "blk")
    try:
        compiled = pops.compile_problem(model=_passive_source_model("ab2_prog"), time=P)
    except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
        print("-- (B) skipped: compile_problem could not build the .so: %s --" % str(exc)[:200])
        return None

    assert compiled.program_name == "ab2_step", "handle carries the program name"

    try:
        compiled_model = _passive_source_model("ab2_block").compile(backend="production")
    except RuntimeError as exc:  # no compiler / no Kokkos visible
        print("-- (B) skipped: model compile could not build the .so: %s --" % str(exc)[:200])
        return None
    sim.add_equation("blk", compiled_model,
                     spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                     time=pops.Explicit(method="euler"))

    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("blk", np.stack([rho0]))

    sim.install_program(compiled.so_path)
    dt = 0.01
    nsteps = 5
    for _ in range(nsteps):
        sim.step(dt)
    out = np.array(sim.get_state("blk"))[0]

    ref = _offline_ab2(rho0, dt, nsteps)
    err = float(np.abs(out - ref).max())
    moved = float(np.abs(out - rho0).max())
    # The two-step recurrence must differ from a single-step run (so we know AB2, not FE, ran past step 0).
    fe = rho0.copy()
    for _ in range(nsteps):
        fe = fe + dt * (_C * fe)
    ab2_vs_fe = float(np.abs(ref - fe).max())
    print("  AB2 parity: max|compiled - offline| = %.2e  max|rho - rho0| = %.2e  "
          "max|AB2 - FE| = %.2e (nsteps=%d)" % (err, moved, ab2_vs_fe, nsteps))
    assert err <= 1e-12, "compiled AB2 == offline AB2 to machine precision (max|d| = %.2e)" % err
    assert moved > 1e-6, "the AB2 stepping must change the state from rho0 (max|d| = %.2e)" % moved
    assert ab2_vs_fe > 1e-9, "AB2 must differ from plain FE past step 0 (max|d| = %.2e)" % ab2_vs_fe
    return err


# ---- (C) absent-history rejection (spec test 38): skips unless the full toolchain is present ----
def _run_section_c(t):
    try:
        import numpy as np

        import pops
    except Exception as exc:  # noqa: BLE001
        print("-- (C) skipped: pops/numpy unavailable: %s --" % exc)
        return None

    n = 8
    sim = pops.System(n=n, L=1.0, periodic=True)
    if not hasattr(sim, "install_program"):
        print("-- (C) skipped: _pops lacks the install_program binding (rebuild _pops) --")
        return None

    from pops.physics.facade import Model

    # A Program that READS missing.R but NEVER stores it -> the runtime read must fail loud.
    P = t.Program("miss_step")
    U = P.state("blk")
    Rp = P.history("missing.R", lag=1)
    R = P.rhs(state=U, sources=["default"])
    P.commit("blk", P.linear_combine(U + P.dt * (R - Rp)))

    try:
        compiled = pops.compile_problem(model=_passive_source_model("miss_prog"), time=P)
    except RuntimeError as exc:
        print("-- (C) skipped: compile_problem could not build the .so: %s --" % str(exc)[:200])
        return None
    try:
        compiled_model = _passive_source_model("miss_block").compile(backend="production")
    except RuntimeError as exc:
        print("-- (C) skipped: model compile could not build the .so: %s --" % str(exc)[:200])
        return None
    sim.add_equation("blk", compiled_model,
                     spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                     time=pops.Explicit(method="euler"))
    sim.set_state("blk", np.stack([np.ones((n, n))]))
    sim.install_program(compiled.so_path)
    try:
        sim.step(0.01)
    except RuntimeError as exc:
        msg = str(exc)
        assert "history 'missing.R' with lag=1 was requested but not initialized" in msg, \
            "the uninitialized-history read must fail loud with the spec message; got: %s" % msg
        print("  absent-history read raised as expected: %s" % msg.splitlines()[0][:120])
        return True
    raise AssertionError("reading a never-stored history must raise at sim.step (spec test 38)")


def _run():
    t = _pops_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    print("PASS test_time_history (A: %d checks)" % len(fns))
    _run_section_b(t)
    _run_section_c(t)


if __name__ == "__main__":
    _run()
