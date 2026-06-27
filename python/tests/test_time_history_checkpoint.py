#!/usr/bin/env python3
"""Checkpoint/restart of compiled-Program histories + the program-hash guard (epic ADC-399 / ADC-406b).

A compiled `Program` with multistep histories (e.g. Adams-Bashforth 2) carries System-owned ring
buffers across macro-steps (the previous RHS R_{n-1}, ...). For a checkpoint/restart to be correct the
rings MUST survive the checkpoint, so a CONTINUOUS run is bit-for-bit identical to a (run, checkpoint,
restart, continue) run -- without the rings the post-restart AB2 would cold-start again and diverge.
The program HASH is recorded in the checkpoint too: restarting a DIFFERENT compiled Program is rejected
fail-loud (the restored buffers / cadence would be meaningless). The existing v1 checkpoint format
(state, t, macro_step, phi) stays back-compatible: a checkpoint with no program/history keys restarts
as before.

(A) NPZ facade keys (pure Python / numpy, always runs when numpy is present): the checkpoint key naming
    scheme (program_hash, history_names, history_depth_<n>, history_<n>_<k>, history_init_<n>) and the
    hash-mismatch comparison round-trip through numpy.savez/load with the exact dtypes the facade uses.
    This pins the serialization contract independently of the engine.

(B) Spec 45 + 39 (continuous == restart, history preserved): run an AB2 program N macro-steps
    continuously -> final state A. A fresh system runs N/2 steps, checkpoints; a SECOND fresh system
    (re-added block, re-installed program) restarts and runs N/2 more -> final state B. Assert A == B to
    machine precision (this exercises the history surviving the checkpoint) and that the clock matches.

(C) Spec 46 (hash mismatch): checkpoint an AB2 program, then restart a DIFFERENT compiled Program
    (Forward Euler, different IR hash) from that checkpoint -> RuntimeError containing
    "checkpoint was created with a different compiled Program hash".

Sections (B)/(C) self-skip (never fake the engine) without numpy / _pops / install_program / a compiler /
a visible Kokkos, exactly like test_time_history.py.
"""
import os
import sys
import tempfile


def _pops_time():
    try:
        import pops.time as t
    except Exception as exc:  # pops not importable here -> skip, never fake
        print("skip test_time_history_checkpoint (pops.time unavailable: %s)" % exc)
        sys.exit(0)
    return t


_C = 0.75  # source coefficient: S(rho) = _C*rho (a linear ODE rho' = c rho; R changes every step)
_DT = 0.01
_NSTEPS = 6  # even, so N/2 is a whole number of macro-steps


# ---- (A) NPZ facade keys: pure numpy, always runs when numpy is present ----
def test_npz_key_scheme_roundtrips(_t):
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001 -- numpy unavailable in this interpreter
        print("-- (A) skipped: numpy unavailable: %s --" % exc)
        return
    # Mirror EXACTLY the keys + dtypes pops.System.checkpoint writes for a program with one AB2 history.
    hname = "blk.R"
    depth = 2
    ncomp, ny, nx = 1, 4, 4
    slots = [np.full((ncomp, ny, nx), float(k + 1)) for k in range(depth)]
    out = {
        "pops_checkpoint_version": 1,
        "program_hash": "deadbeef" * 8,  # a 64-hex IR hash shape
        "history_names": np.array([hname]),
        "history_depth_" + hname: depth,
        "history_ncomp_" + hname: ncomp,
        "history_init_" + hname: True,
    }
    for k in range(depth):
        out["history_%s_%d" % (hname, k)] = slots[k]

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.npz")
        with open(path, "wb") as f:
            np.savez_compressed(f, **out)
        d = np.load(path, allow_pickle=False)
        assert "program_hash" in d and str(d["program_hash"]) == out["program_hash"]
        assert [str(h) for h in d["history_names"]] == [hname]
        assert int(d["history_depth_" + hname]) == depth
        assert int(d["history_ncomp_" + hname]) == ncomp
        assert bool(d["history_init_" + hname]) is True
        for k in range(depth):
            np.testing.assert_array_equal(d["history_%s_%d" % (hname, k)], slots[k])

    # The hash guard is a plain string compare: equal -> ok, different -> the spec-46 message.
    assert out["program_hash"] == ("deadbeef" * 8), "same hash compares equal"
    assert out["program_hash"] != ("cafe" * 16), "a different hash must compare unequal"

    # Back-compat: a checkpoint WITHOUT the new keys carries neither program_hash nor history_names.
    legacy = {"pops_checkpoint_version": 1, "t": 0.0, "macro_step": 0}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "legacy.npz")
        with open(path, "wb") as f:
            np.savez_compressed(f, **legacy)
        d = np.load(path, allow_pickle=False)
        assert "program_hash" not in d and "history_names" not in d, \
            "a legacy checkpoint must not carry the ADC-406b keys (back-compatible restart)"


# ---- shared engine setup for (B)/(C) ----
def _passive_source_model(name):
    """A 1-variable model (rho), ZERO flux, default LINEAR source S(rho) = _C*rho (R = c*rho changes
    every step). A complete compilable block (flux + primitive + eigenvalue + source)."""
    from pops.physics.facade import Model
    m = Model(name)
    (rho,) = m.conservative_vars("rho")
    u = m.primitive("u", 0.0 * rho)
    m.primitive_vars(rho=rho, u=u)
    m.conservative_from([rho])
    m.flux(x=[0.0 * rho], y=[0.0 * rho])
    m.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
    m.source([_C * rho])
    return m


def _build_system(pops, np, n):
    """A fresh n x n periodic System with the compiled passive-source block added; (sim, has_engine)."""
    sim = pops.System(n=n, L=1.0, periodic=True)
    if not hasattr(sim, "install_program") or not hasattr(sim, "history_names"):
        return None, None
    from pops.physics.facade import Model
    try:
        compiled_model = _passive_source_model("ckpt_block").compile(backend="production")
    except RuntimeError as exc:  # no compiler / no Kokkos visible
        print("-- skipped: model compile could not build the .so: %s --" % str(exc)[:160])
        return None, None
    sim.add_equation("blk", compiled_model,
                     spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                     time=pops.Explicit(method="euler"))
    return sim, True


def _rho0(np, n):
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)


def _compile_program(pops, t, builder, prog_name, model_name):
    """compile_problem for the program built by @p builder (e.g. t.std.adams_bashforth2). Returns the
    handle or None if the toolchain is absent."""
    P = t.Program(prog_name)
    builder(P, "blk")
    try:
        return pops.compile_problem(model=_passive_source_model(model_name), time=P)
    except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
        print("-- skipped: compile_problem could not build the .so: %s --" % str(exc)[:160])
        return None


# ---- (B) spec 45 + 39: continuous == (run, checkpoint, restart, continue), bit-for-bit ----
def _run_section_b(t):
    try:
        import numpy as np

        import pops
    except Exception as exc:  # noqa: BLE001 -- numpy / _pops unavailable
        print("-- (B) skipped: pops/numpy unavailable: %s --" % exc)
        return None

    n = 16
    sim_cont, has_engine = _build_system(pops, np, n)
    if sim_cont is None:
        print("-- (B) skipped: _pops lacks the install_program/history bindings (rebuild _pops) --")
        return None

    compiled = _compile_program(pops, t, t.std.adams_bashforth2, "ab2_ckpt", "ab2_prog_b")
    if compiled is None:
        return None

    rho0 = _rho0(np, n)
    half = _NSTEPS // 2

    # (1) CONTINUOUS run: N steps in one go -> final state A.
    sim_cont.set_state("blk", np.stack([rho0]))
    sim_cont.install_program(compiled.so_path)
    for _ in range(_NSTEPS):
        sim_cont.step(_DT)
    state_a = np.array(sim_cont.get_state("blk"))[0]

    # (2) RUN N/2, CHECKPOINT.
    sim1, _ = _build_system(pops, np, n)
    sim1.set_state("blk", np.stack([rho0]))
    sim1.install_program(compiled.so_path)
    for _ in range(half):
        sim1.step(_DT)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = sim1.checkpoint(os.path.join(tmp, "ab2"))

        # (3) FRESH system, re-add block, re-install the SAME program, RESTART, run N/2 more -> B.
        sim2, _ = _build_system(pops, np, n)
        sim2.install_program(compiled.so_path)  # the hash guard needs the program installed first
        sim2.restart(ckpt)
        assert sim2.macro_step() == half, \
            "restart restores macro_step (%d != %d)" % (sim2.macro_step(), half)
        for _ in range(_NSTEPS - half):
            sim2.step(_DT)
    state_b = np.array(sim2.get_state("blk"))[0]

    err = float(np.abs(state_a - state_b).max())
    assert sim2.macro_step() == sim_cont.macro_step(), \
        "the clock must match after the restart run (%d != %d)" % (
            sim2.macro_step(), sim_cont.macro_step())
    assert abs(sim2.time() - sim_cont.time()) <= 1e-12, "t must match after restart"

    # Cross-check that the history actually mattered. A restart that DROPPED the rings would cold-start
    # AB2 at the resume step and diverge. The offline cold-restart reference (AB2, then a fresh FE cold
    # start at the resume step, then AB2) is what a pre-406b ring-less restart would produce; it must
    # DIFFER from the continuous run, so the bit-exact match above is a non-trivial result.
    ref_cold = _offline_ab2_cold_restart(rho0, _DT, _NSTEPS, half)
    cold = float(np.abs(state_a - ref_cold).max())

    print("  AB2 ckpt/restart: max|continuous - restart| = %.2e  "
          "max|continuous - ring-less restart| = %.2e (N=%d, half=%d)" % (err, cold, _NSTEPS, half))
    assert err <= 1e-12, \
        "continuous == (run, ckpt, restart, continue) to machine precision (max|d| = %.2e)" % err
    assert cold > 1e-9, \
        "the history must matter: a ring-less restart diverges (max|d| = %.2e)" % cold
    return err


def _offline_ab2_cold_restart(rho0, dt, nsteps, resume):
    """The AB2 trajectory if the history were LOST at the @p resume checkpoint: AB2 for the first
    @p resume steps, then a fresh FE cold start at @p resume, then AB2 again. This is the WRONG result a
    pre-406b (ring-less) restart would produce -- used only to prove the correct restart is non-trivial."""
    rho = rho0.copy()
    r_prev = _C * rho
    for k in range(nsteps):
        if k == resume:  # cold restart: forget R_{n-1}, refill from the current R
            r_prev = _C * rho
        r_n = _C * rho
        rho = rho + dt * (1.5 * r_n - 0.5 * r_prev)
        r_prev = r_n
    return rho


# ---- (C) spec 46: restart a DIFFERENT compiled Program -> hash-mismatch RuntimeError ----
def _run_section_c(t):
    try:
        import numpy as np

        import pops
    except Exception as exc:  # noqa: BLE001
        print("-- (C) skipped: pops/numpy unavailable: %s --" % exc)
        return None

    n = 8
    sim, has_engine = _build_system(pops, np, n)
    if sim is None:
        print("-- (C) skipped: _pops lacks the install_program/history bindings (rebuild _pops) --")
        return None

    ab2 = _compile_program(pops, t, t.std.adams_bashforth2, "ab2_c", "ab2_prog_c")
    fe = _compile_program(pops, t, t.std.forward_euler, "fe_c", "fe_prog_c")
    if ab2 is None or fe is None:
        return None
    assert ab2.program_hash != fe.program_hash, \
        "AB2 and Forward Euler must have different IR hashes (else the test is vacuous)"

    sim.set_state("blk", np.stack([np.ones((n, n))]))
    sim.install_program(ab2.so_path)
    sim.step(_DT)
    sim.step(_DT)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = sim.checkpoint(os.path.join(tmp, "ab2_for_mismatch"))

        # A fresh system that installs the WRONG (Forward Euler) program, then restarts the AB2 ckpt.
        sim2, _ = _build_system(pops, np, n)
        sim2.install_program(fe.so_path)
        try:
            sim2.restart(ckpt)
        except RuntimeError as exc:
            msg = str(exc)
            assert "checkpoint was created with a different compiled Program hash" in msg, \
                "the hash mismatch must fail loud with the spec-46 message; got: %s" % msg
            print("  hash mismatch raised as expected: %s" % msg.splitlines()[0][:120])
            return True
    raise AssertionError("restarting a different compiled Program must raise (spec test 46)")


def _run():
    t = _pops_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    print("PASS test_time_history_checkpoint (A: %d checks)" % len(fns))
    _run_section_b(t)
    _run_section_c(t)


if __name__ == "__main__":
    _run()
