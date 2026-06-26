#!/usr/bin/env python3
"""pops.time MULTI-BLOCK compiled Programs (epic ADC-399 / ADC-426, spec "Multi-blocs").

`emit_cpp_program` now lowers N ``P.state`` / N ``P.commit``: each op routes to its block's runtime
index (``_block_indices``, declaration order), so an N-block transport program compiles and steps all
blocks in one macro-step. The block index is positional -- the System blocks (``sim.add_equation``)
MUST be added in the SAME order the Program declares them via ``P.state``.

(A) Validation + codegen (pure Python, always runs when pops.time imports): a 2-block program lowers
    with per-block ctx.state / rhs_into indices; a read-only block (declared but never committed) is
    allowed; a double commit and a commit of an undeclared block are rejected; the SIMULTANEOUS
    multi-target solve_fields_from_blocks lowers to ctx.solve_fields_from_blocks (Spec 3 crit 24).

(B) End-to-end parity (skips unless the full toolchain is present): a 2-block passive-transport model
    (a scalar with a non-trivial flux + a NAMED source_term, EMPTY default source -- avoids the
    sources=[] default-source path) is stepped two ways: (1) one 2-block System driven by the
    multi-block compiled Program; (2) two INDEPENDENT single-block Systems each driven by the
    single-block compiled Program (the offline per-block reference -- the blocks are uncoupled, no
    elliptic, so the multi-block step must equal independent single-block steps). The two states must
    match to round-off, and each block must have actually advanced. Runs in CI (gate-python rebuilds
    _pops) and locally once _pops is rebuilt; skips if _pops lacks install_program, numpy/_pops is absent,
    no compiler/Kokkos is visible, or the .so compile fails -- never faking the engine.
"""
import sys


def _skip(msg):
    print("skip test_time_multiblock (%s)" % msg)
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


def raises(exc_types, fn):
    try:
        fn()
    except exc_types:
        return True
    except Exception:  # noqa: BLE001 -- the wrong exception type is a failure, not a pass
        return False
    return False


def passive_model(dsl, name):
    """A 1-variable PASSIVE-transport scalar (rho) with a constant advection velocity baked into the
    flux and a NAMED source ``decay`` = -k*rho (a linear sink), EMPTY default source. A complete,
    compilable production block (flux + primitives + eigenvalues + named source_term)."""
    m = dsl.Model(name)
    (rho,) = m.conservative_vars("rho")
    a = 0.7  # constant advection speed (x and y)
    u = m.primitive("u", a + 0.0 * rho)
    v = m.primitive("v", a + 0.0 * rho)
    m.primitive_vars(rho=rho, u=u, v=v)
    m.conservative_from([rho])
    m.flux(x=[a * rho], y=[a * rho])           # F = a*rho (linear advection)
    m.eigenvalues(x=[a + 0.0 * rho], y=[a + 0.0 * rho])
    m.source_term("decay", [-0.3 * rho])       # S(rho) = -0.3*rho (a named linear sink)
    return m


def single_block_program(t, name, block):
    """Forward-Euler passive transport of ONE block: U1 = U + dt*(-div F + S_decay)."""
    P = t.Program(name)
    dt = P.dt
    U = P.state(block)
    R = P.rhs(name="R_" + block, state=U, flux=True, sources=["decay"])
    P.commit(block, P.linear_combine(block + "_next", U + dt * R))
    return P


def two_block_program(t, name="two_block_passive"):
    """Forward-Euler passive transport of TWO blocks ("a", "b") in one program: each block advances
    independently U_blk1 = U_blk + dt*(-div F + S_decay), committed once. The blocks are declared a
    then b -> runtime indices a=0, b=1."""
    P = t.Program(name)
    dt = P.dt
    for blk in ("a", "b"):
        U = P.state(blk)
        R = P.rhs(name="R_" + blk, state=U, flux=True, sources=["decay"])
        P.commit(blk, P.linear_combine(blk + "_next", U + dt * R))
    return P


# ============================ (A) validation + codegen (pure Python) ============================
def section_a(t):
    print("== (A) multi-block validation + codegen ==")

    # The 2-block program lowers (no model needed for FE + named-source codegen is refused without a
    # model, so test the index routing on the default-source flux-only program here).
    P = t.Program("two_block_flux")
    dt = P.dt
    for blk in ("a", "b"):
        U = P.state(blk)
        R = P.rhs(state=U, flux=True, sources=["default"])
        P.commit(blk, P.linear_combine(blk + "_next", U + dt * R))
    src = P.emit_cpp_program()
    chk("ctx.state(0)" in src and "ctx.state(1)" in src, "two blocks bind ctx.state(0) and state(1)")
    chk("ctx.rhs_into(0, " in src and "ctx.rhs_into(1, " in src, "RHS routed per block index")

    # A read-only block (declared via P.state, never committed) is allowed: only block 'a' commits.
    Pro = t.Program("readonly_b")
    Ua = Pro.state("a")
    Ub = Pro.state("b")  # noqa: F841 -- declared, read by the coupled charge but never committed
    fa = Pro.solve_fields(Ua)
    Ra = Pro.rhs(state=Ua, fields=fa, sources=["default"])
    Pro.commit("a", Pro.linear_combine("a1", Ua + Pro.dt * Ra))
    chk(Pro.validate() is True, "a read-only (uncommitted) block validates")
    src_ro = Pro.emit_cpp_program()
    chk("ctx.state(1)" in src_ro, "the read-only block still binds its index (ctx.state(1))")

    # A double commit is rejected at build time.
    Pd = t.Program("double")
    Uad = Pd.state("a")
    Pd.commit("a", Pd.linear_combine("x", 1.0 * Uad))
    chk(raises(ValueError, lambda: Pd.commit("a", Pd.linear_combine("y", 1.0 * Uad))),
        "a double commit of the same block is rejected")

    # A commit of a block no P.state declares cannot route to an index -> rejected.
    Pu = t.Program("unknown")
    Uau = Pu.state("a")
    Pu.commit("ghost", Pu.linear_combine("g", 1.0 * Uau))
    chk(raises(ValueError, lambda: Pu.emit_cpp_program()),
        "a commit of an undeclared block is rejected")

    # The SIMULTANEOUS multi-target coupled field solve LOWERS (Spec 3 criterion 24, ADC-457): every
    # listed block contributes its stage state at once into the shared phi/aux.
    Pc = t.Program("coupled")
    Uac = Pc.state("a")
    Ubc = Pc.state("b")
    Pc.solve_fields_from_blocks([Uac, Ubc])
    Pc.commit("a", Pc.linear_combine("a1", Uac + Pc.dt * Pc.rhs(state=Uac, sources=["default"])))
    Pc.commit("b", Pc.linear_combine("b1", Ubc + Pc.dt * Pc.rhs(state=Ubc, sources=["default"])))
    src_c = Pc.emit_cpp_program()
    chk("ctx.solve_fields_from_blocks(" in src_c,
        "solve_fields_from_blocks lowers to the coupled multi-block solve")
    chk("std::vector<const pops::MultiFab*>" in src_c,
        "the coupled solve builds a per-block MultiFab pointer vector")
    chk(src_c.count("] = &") >= 2, "each listed block slots its stage state by index")

    # The builder rejects malformed solve_fields_from_blocks arguments.
    Pb = t.Program("b")
    Uab = Pb.state("a")
    chk(raises(ValueError, lambda: Pb.solve_fields_from_blocks([])),
        "solve_fields_from_blocks([]) is rejected")
    chk(raises(ValueError, lambda: Pb.solve_fields_from_blocks([Uab, Uab])),
        "solve_fields_from_blocks with a block listed twice is rejected")

    # Control flow (range/while/if) inside a NON-index-0 block must route its body ops to THAT block's
    # runtime index, not silently to 0 (_emit_while/_emit_range/_emit_if forward block_idx). Block b
    # (index 1) updates inside a range loop -> the in-loop RHS must lower to ctx.rhs_into(1, ...).
    Pcf = t.Program("cf_block_routing")
    Uacf = Pcf.state("a")
    Ubcf = Pcf.state("b")
    Pcf.commit("a", Pcf.linear_combine(
        "a_n", Uacf + Pcf.dt * Pcf.rhs(state=Uacf, flux=True, sources=["default"])))

    def _cf_body(prog, x):
        return prog.linear_combine(None, x + prog.dt * prog.rhs(state=x, flux=True, sources=["default"]))

    Pcf.commit("b", Pcf.range(Ubcf, 2, _cf_body))
    src_cf = Pcf.emit_cpp_program()
    chk("ctx.rhs_into(1, " in src_cf,
        "control flow inside block b routes its body RHS to index 1 (not silently 0)")


# ============================ (B) end-to-end parity (skips without the toolchain) ===============
def section_b(t):
    try:
        import numpy as np

        import pops
        from pops import dsl
    except Exception as exc:  # noqa: BLE001 -- numpy or _pops unavailable
        print("-- (B) skipped: pops/numpy unavailable (%s) --" % exc)
        return

    if not hasattr(pops.System(n=8, L=1.0, periodic=True), "install_program"):
        print("-- (B) skipped: _pops lacks the install_program binding (rebuild _pops) --")
        return

    print("== (B) end-to-end: 2-block program vs two independent single-block runs ==")

    n = 16
    dt = 0.02

    def make_ic(seed):
        x = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(x, x, indexing="ij")
        return 1.0 + 0.4 * np.sin(2 * np.pi * (X + seed)) * np.cos(2 * np.pi * Y)

    ic_a = make_ic(0.0)
    ic_b = make_ic(0.37)  # a DIFFERENT IC per block, so a routing bug (b reads a's state) shows up

    # Compile the single-block reference programs (one per block name) and the 2-block program.
    try:
        comp_a = pops.compile_problem(model=passive_model(dsl, "pa_ref"),
                                     time=single_block_program(t, "fe_a", "a"))
        comp_b = pops.compile_problem(model=passive_model(dsl, "pb_ref"),
                                     time=single_block_program(t, "fe_b", "b"))
        comp_ab = pops.compile_problem(model=passive_model(dsl, "pab"), time=two_block_program(t))
    except (RuntimeError, ValueError) as exc:  # no compiler / no Kokkos / .so compile failed
        _skip("compile_problem could not build the .so: %s" % str(exc)[:160])

    chk(comp_ab.program_name == "two_block_passive", "the 2-block handle carries the program name")

    def make_sim(blocks):
        sim = pops.System(n=n, L=1.0, periodic=True)
        for blk in blocks:
            try:
                cm = passive_model(dsl, "blk_" + blk).compile(backend="production")
            except RuntimeError as exc:  # no compiler / no Kokkos
                _skip("model compile could not build the .so: %s" % str(exc)[:160])
            sim.add_equation(blk, cm,
                             spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                             time=pops.Explicit(method="euler"))
        return sim

    # Reference: two INDEPENDENT single-block systems.
    sim_a = make_sim(["a"])
    sim_a.set_state("a", ic_a[None, :, :])
    sim_a.install_program(comp_a.so_path)
    sim_a.step(dt)
    ref_a = np.array(sim_a.get_state("a"))

    sim_b = make_sim(["b"])
    sim_b.set_state("b", ic_b[None, :, :])
    sim_b.install_program(comp_b.so_path)
    sim_b.step(dt)
    ref_b = np.array(sim_b.get_state("b"))

    # The multi-block system: blocks added in the SAME order the Program declares them (a then b).
    sim_ab = make_sim(["a", "b"])
    sim_ab.set_state("a", ic_a[None, :, :])
    sim_ab.set_state("b", ic_b[None, :, :])
    sim_ab.install_program(comp_ab.so_path)
    sim_ab.step(dt)
    got_a = np.array(sim_ab.get_state("a"))
    got_b = np.array(sim_ab.get_state("b"))

    e_a = float(np.abs(got_a - ref_a).max())
    e_b = float(np.abs(got_b - ref_b).max())
    print("  parity: max|d(a)| = %.2e  max|d(b)| = %.2e" % (e_a, e_b))
    chk(e_a < 1e-13, "block a matches the single-block reference (max|d| = %.2e)" % e_a)
    chk(e_b < 1e-13, "block b matches the single-block reference (max|d| = %.2e)" % e_b)

    # Each block actually advanced, and the two blocks differ (no cross-block aliasing).
    chk(float(np.abs(got_a - ic_a[None, :, :]).max()) > 1e-6, "block a actually advanced")
    chk(float(np.abs(got_b - ic_b[None, :, :]).max()) > 1e-6, "block b actually advanced")
    chk(float(np.abs(got_a - got_b).max()) > 1e-6, "the two blocks hold distinct states")


def _run():
    t = _pops_time()
    section_a(t)
    section_b(t)
    print("%s test_time_multiblock" % ("FAIL (%d)" % fails if fails else "PASS"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    _run()
