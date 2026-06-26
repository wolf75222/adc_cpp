#!/usr/bin/env python3
"""pops.time NAME-based block binding -- end-to-end runtime (Spec 3 criterion 23, ADC-457).

A compiled multi-block Program binds its blocks to the System blocks BY NAME, not by add-order. This
test PROVES the runtime binding by COMPILING a 2-block .so and installing it on Systems whose blocks
were added in DIFFERENT orders:

(1) blocks added in P.state order (a, b) -> the in-order baseline;
(2) blocks added REVERSED (b, a) -> must produce the SAME per-block result as (1): the .so addresses
    "a" / "b" by NAME, so reversing the System add-order must NOT cross-wire the states;
(3) a System missing block "b" -> install_program must FAIL LOUD with the spec message
    ``Program requires block instance 'b', but simulation did not instantiate it``.

This exercises the AOT .so path: it compiles a problem.so (production model + compiled Program) and
dlopens it via sim.install_program. That path needs a C++ compiler + Kokkos, so it CANNOT run on a
host-only Mac -- it SKIPS cleanly there (never faking the engine) and is validated on CI-Kokkos
(gate-python rebuilds _pops, ci-kokkos-python) and on ROMEO. The codegen ABI export + IR identity are
covered host-side by test_name_binding_codegen.py.

Runs as a plain script (``python3 test_name_binding_runtime.py``, the CI invocation) and under pytest.
"""
import sys


def _skip(msg):
    print("skip test_name_binding_runtime (%s)" % msg)
    sys.exit(0)


fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


def raises_with(fn, needle):
    """True iff @p fn raises and the exception text contains @p needle (the verbatim spec message)."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 -- ANY raise carrying the spec message is the pass
        return needle in str(exc)
    return False


def passive_model(dsl, name):
    """A 1-variable PASSIVE-transport scalar (rho): linear advection flux + a named linear sink, EMPTY
    default source (mirrors test_time_multiblock; avoids the default-source path so the step is exact)."""
    m = dsl.Model(name)
    (rho,) = m.conservative_vars("rho")
    a = 0.7
    u = m.primitive("u", a + 0.0 * rho)
    v = m.primitive("v", a + 0.0 * rho)
    m.primitive_vars(rho=rho, u=u, v=v)
    m.conservative_from([rho])
    m.flux(x=[a * rho], y=[a * rho])
    m.eigenvalues(x=[a + 0.0 * rho], y=[a + 0.0 * rho])
    m.source_term("decay", [-0.3 * rho])
    return m


def two_block_program(t, name="nb_two_block"):
    """Forward-Euler passive transport of blocks "a" then "b" (P.state order a=0, b=1)."""
    P = t.Program(name)
    for blk in ("a", "b"):
        U = P.state(blk)
        R = P.rhs(name="R_" + blk, state=U, flux=True, sources=["decay"])
        P.commit(blk, P.linear_combine(blk + "_next", U + P.dt * R))
    return P


def _run():
    try:
        import numpy as np

        import pops
        import pops.time as t
        from pops import dsl
    except Exception as exc:  # noqa: BLE001 -- numpy / _pops / pops.time unavailable
        _skip("pops / pops.time / numpy unavailable: %s" % exc)

    if not hasattr(pops.System(n=8, L=1.0, periodic=True), "install_program"):
        _skip("_pops lacks the install_program binding (rebuild _pops)")

    print("== NAME-based block binding: reversed System add-order matches in-order ==")
    n = 16
    dt = 0.02

    def make_ic(seed):
        x = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(x, x, indexing="ij")
        return 1.0 + 0.4 * np.sin(2 * np.pi * (X + seed)) * np.cos(2 * np.pi * Y)

    ic = {"a": make_ic(0.0), "b": make_ic(0.37)}  # DIFFERENT IC per block: cross-wiring would show

    # Compile the 2-block .so ONCE (production model + compiled Program). Needs compiler + Kokkos.
    try:
        comp = pops.compile_problem(model=passive_model(dsl, "nb_model"), time=two_block_program(t))
    except (RuntimeError, ValueError) as exc:  # no compiler / no Kokkos / .so compile failed
        _skip("compile_problem could not build the .so: %s" % str(exc)[:160])

    def make_sim(add_order):
        """A System with the two blocks added in @p add_order, each set to its OWN-named IC."""
        sim = pops.System(n=n, L=1.0, periodic=True)
        for blk in add_order:
            try:
                cm = passive_model(dsl, "nb_blk_" + blk).compile(backend="production")
            except RuntimeError as exc:  # no compiler / no Kokkos
                _skip("model compile could not build the .so: %s" % str(exc)[:160])
            sim.add_equation(blk, cm, spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                             time=pops.Explicit(method="euler"))
        for blk in add_order:
            sim.set_state(blk, ic[blk][None, :, :])
        return sim

    # (1) Baseline: blocks added in P.state order (a, b).
    sim_inorder = make_sim(["a", "b"])
    sim_inorder.install_program(comp.so_path)
    sim_inorder.step(dt)
    ref = {blk: np.array(sim_inorder.get_state(blk)) for blk in ("a", "b")}

    # (2) Reversed: blocks added (b, a). The .so binds by NAME, so each block must match the baseline.
    sim_rev = make_sim(["b", "a"])
    sim_rev.install_program(comp.so_path)
    sim_rev.step(dt)
    got = {blk: np.array(sim_rev.get_state(blk)) for blk in ("a", "b")}

    for blk in ("a", "b"):
        e = float(np.abs(got[blk] - ref[blk]).max())
        print("  reversed vs in-order: max|d(%s)| = %.2e" % (blk, e))
        chk(e < 1e-13, "block %s name-binds (reversed add-order matches in-order, max|d| = %.2e)"
            % (blk, e))
    # A real advance happened, and the blocks are distinct (so a no-op / cross-wire would have failed).
    chk(float(np.abs(got["a"] - ic["a"][None, :, :]).max()) > 1e-6, "block a actually advanced")
    chk(float(np.abs(got["a"] - got["b"]).max()) > 1e-6, "the two blocks hold distinct states")

    print("== a missing block instance fails loud with the spec message ==")
    sim_missing = make_sim(["a"])  # block "b" the Program requires is NOT instantiated
    chk(raises_with(lambda: sim_missing.install_program(comp.so_path),
                    "Program requires block instance 'b', but simulation did not instantiate it"),
        "installing a Program needing 'b' on a System without 'b' raises the verbatim spec error")

    print("%s test_name_binding_runtime" % ("FAIL (%d)" % fails if fails else "PASS"))
    sys.exit(1 if fails else 0)


def test_name_binding_runtime():
    # _run() / _skip() exit via sys.exit (script semantics, the CI runner reads the rc). Under pytest a
    # raw SystemExit reports as FAILED, so translate it: rc 0 (pass or clean Kokkos-absent skip) -> pass,
    # non-zero -> a real assertion failure.
    try:
        _run()
    except SystemExit as exc:  # noqa: BLE001 -- rc carries the verdict
        if exc.code:
            raise AssertionError("test_name_binding_runtime reported failures") from None


if __name__ == "__main__":
    _run()
