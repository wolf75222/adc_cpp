#!/usr/bin/env python3
"""pops.lib.time.imex_local / lie / adams_bashforth(order) -- ADC-423 beyond-MVP macros.

All three LOWER to the existing Program IR (no new C++ stepper):

  - ``std.imex_local`` = explicit flux/source (P.rhs) + IMPLICIT cell-local linear source
    ((I - theta*dt*L)^{-1} via P.solve_local_linear), the predictor half of the codebase's
    predictor-corrector pattern;
  - ``std.lie`` = sequential Lie splitting H(dt); S(dt) (the `strang` sibling with no half-steps);
  - ``std.adams_bashforth(order)`` = the explicit AB1/AB2/AB3 recurrence over the System history ring.

(A) IR construction + codegen, pure Python (always runs): each macro builds valid IR; imex_local /
    bdf lower to the per-cell solve kernel with a model; AB3 lowers with two history reads + rotate;
    the guards (imex theta range, AB order) fire.

(B) Offline parity (skips cleanly without numpy / _pops / install_program / a compiler / a visible
    Kokkos): the compiled program is stepped and compared to an INDEPENDENT offline numpy reference of
    the identical recurrence, to machine precision. Self-skips, never fakes the engine.
"""
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
import sys


def _pops_time():
    global lt  # ready schemes live in pops.lib.time (Spec 4)
    try:
        import pops.time as t
        import pops.lib.time as lt  # ready schemes live in pops.lib.time (Spec 4)
    except Exception as exc:  # pops not importable here -> skip, never fake
        print("skip test_time_std_imex_lie_ab (pops.time unavailable: %s)" % exc)
        sys.exit(0)
    return t


_C = 0.75  # AB linear-source coefficient: S(rho) = _C*rho


# ---------- shared DSL models (compiled only in section B) ----------
def _passive_source_model(name):
    """1-variable rho, ZERO flux, default LINEAR source S = _C*rho (R changes every step) -- for AB."""
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


def _reaction_term_model(name):
    """1-variable rho, ZERO flux, a NAMED source_term S = _C*rho and NO default source -- for the Lie
    split. The hyperbolic stage H requests sources=[] (flux only): with an empty default source that is
    inert (zero flux, no source), so H is a true no-op and the source stage applies only the named
    "reaction". (A default m.source would be wrongly included by sources=[] on the flux-only path; that
    flux-only-excludes-default-source gap is tracked separately and does not affect this named-source
    model.)"""
    from pops.physics.facade import Model
    m = Model(name)
    (rho,) = m.conservative_vars("rho")
    u = m.primitive("u", 0.0 * rho)
    m.primitive_vars(rho=rho, u=u)
    m.conservative_from([rho])
    m.flux(x=[0.0 * rho], y=[0.0 * rho])
    m.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
    m.source_term("reaction", [_C * rho])
    return m


def _lorentz_model(name):
    """Isothermal 2D fluid (rho, mx, my) with a Lorentz linear source L(B_z) -- for imex_local.

    ZERO flux so the explicit RHS is the default source only (here: none), isolating the implicit
    Lorentz solve; B_z is read off the System aux. A complete compilable production block."""
    from pops.physics.facade import Model
    m = Model(name)
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    cs2 = m.param("cs2", 0.5)
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    p = m.primitive("p", cs2 * rho)
    m.primitive_vars(rho=rho, u=u, v=v, p=p)
    m.conservative_from([rho, rho * u, rho * v])
    m.flux(x=[0.0 * rho, 0.0 * rho, 0.0 * rho], y=[0.0 * rho, 0.0 * rho, 0.0 * rho])
    m.eigenvalues(x=[0.0 * rho, 0.0 * rho, 0.0 * rho], y=[0.0 * rho, 0.0 * rho, 0.0 * rho])
    bz = m.aux("B_z")
    m.linear_source("lorentz", [[0.0, 0.0, 0.0],
                                [0.0, 0.0, bz],
                                [0.0, -bz, 0.0]])
    return m


# ============================ (A) IR + codegen: pure Python =============================
def test_imex_local_builds_and_lowers(t):
    P = t.Program("imex")
    out = lt.imex_local(P, "plasma", linear_source="lorentz")
    assert P.validate() is True and P.commits()["plasma"] is out
    try:
        from pops.physics.facade import Model
    except Exception as exc:  # noqa: BLE001
        print("  (imex codegen skipped: pops.dsl unavailable: %s)" % exc)
        return
    src = P.emit_cpp_program(model=_lorentz_model("imex_m"))
    assert "pops::detail::mat_inverse<3>(" in src, "imex implicit term is a per-cell dense solve"
    assert "rhs_into" in src, "imex explicit term assembles an RHS"


def test_imex_local_theta_guard(t):
    for bad in (0.0, -0.5, 1.5):
        try:
            lt.imex_local(t.Program("x"), "plasma", linear_source="lorentz", theta=bad)
        except ValueError as exc:
            assert "theta" in str(exc)
        else:
            raise AssertionError("imex_local theta=%r must raise (0 < theta <= 1)" % (bad,))


def test_imex_local_rejects_empty_linear_source(t):
    try:
        lt.imex_local(t.Program("x"), "plasma", linear_source="")
    except ValueError as exc:
        assert "linear_source" in str(exc)
    else:
        raise AssertionError("imex_local must reject an empty linear_source name")


def test_lie_chains_two_stages(t):
    P = t.Program("lie")

    def half_flow(prog, U, frac):
        R = prog.rhs(state=U, fields=prog.solve_fields(U), flux=True, sources=["default"])
        return prog.linear_combine(None, U + (frac * prog.dt) * R)

    def source(prog, U, frac):
        S = prog.rhs(state=U, fields=None, flux=False, sources=["default"])
        return prog.linear_combine(None, U + (frac * prog.dt) * S)

    out = lt.lie(P, "plasma", half_flow, source)
    P.validate()
    assert P.commits()["plasma"] is out
    n_lc = sum(1 for v in P._values if v.op == "linear_combine")
    assert n_lc == 2, "Lie composes exactly two stages H(dt); S(dt) (got %d)" % n_lc
    # Lie advances each sub-flow over the FULL dt -> frac 1.0 on both; Strang would be 0.5/1.0/0.5.
    strang = t.Program("lie")  # same name to compare IR shape, not value
    lt.strang(strang, "plasma", half_flow, source)
    assert P._ir_hash() != strang._ir_hash(), "Lie (2 stages, full dt) differs from Strang (3 stages)"


def test_adams_bashforth_orders_build(t):
    for order in (1, 2, 3):
        P = t.Program("ab%d" % order)
        lt.adams_bashforth(P, "plasma", order)
        assert P.validate() is True, "AB%d must validate" % order
    # AB1 == forward_euler IR (no history op).
    ab1 = t.Program("p")
    lt.adams_bashforth(ab1, "plasma", 1)
    fe = t.Program("p")
    lt.forward_euler(fe, "plasma")
    assert ab1._ir_hash() == fe._ir_hash(), "AB1 must be byte-identical to forward_euler"
    # AB2 alias == adams_bashforth(2).
    a2 = t.Program("p")
    lt.adams_bashforth2(a2, "plasma")
    g2 = t.Program("p")
    lt.adams_bashforth(g2, "plasma", 2)
    assert a2._ir_hash() == g2._ir_hash(), "adams_bashforth2 is a thin alias for adams_bashforth(2)"


def test_adams_bashforth_bad_order(t):
    for bad in (0, 4, 2.0):
        try:
            lt.adams_bashforth(t.Program("x"), "plasma", bad)
        except ValueError as exc:
            assert "order" in str(exc)
        else:
            raise AssertionError("AB order=%r must raise" % (bad,))


def test_ab3_lowers_with_two_history_reads(t):
    P = t.Program("ab3")
    lt.adams_bashforth(P, "plasma", 3)
    try:
        src = P.emit_cpp_program()  # AB3 has no Phase-4 ops -> lowers without a model
    except Exception as exc:  # noqa: BLE001
        raise AssertionError("AB3 must lower without a model (no Phase-4 ops): %s" % exc) from exc
    assert 'ctx.history("plasma.R", 1)' in src and 'ctx.history("plasma.R", 2)' in src, \
        "AB3 reads R_{n-1} (lag 1) AND R_{n-2} (lag 2)"
    assert "ctx.store_history" in src and "ctx.rotate_histories();" in src, "AB3 stores + rotates"
    # AB3 weights 23/12, -16/12, 5/12 on dt.
    for w in ("1.9166", "-1.3333", "0.41666"):
        assert w in src.replace("e+00", ""), "AB3 weight %s must appear in the lowered combine" % w


# ============================ (B) offline parity: skips without the toolchain ===================
def _offline_ab(rho0, dt, nsteps, order):
    """The identical AB recurrence cell by cell, FE cold start (R_{-j} := R_0 -> step 0 is FE)."""
    import numpy as np
    b = {1: [1.0], 2: [1.5, -0.5], 3: [23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0]}[order]
    rho = rho0.copy()
    hist = [(_C * rho).copy() for _ in range(order)]  # R_0 in every slot (cold start)
    for _ in range(nsteps):
        r_n = _C * rho
        hist = [r_n.copy()] + hist[:order - 1]  # rotate: slot j is R_{n-j}
        rho = rho + dt * sum(b[j] * np.asarray(hist[j]) for j in range(order))
    return rho


def _run_ab3(t):
    try:
        import numpy as np

        import pops
        from pops.physics.facade import Model
    except Exception as exc:  # noqa: BLE001
        print("-- (B AB3) skipped: pops/numpy unavailable: %s --" % exc)
        return
    sim = pops.System(n=16, L=1.0, periodic=True)
    if not hasattr(sim, "install_program"):
        print("-- (B AB3) skipped: _pops lacks install_program (rebuild _pops) --")
        return
    P = t.Program("ab3_step")
    lt.adams_bashforth(P, "blk", 3)
    try:
        compiled = pops.compile_problem(model=_passive_source_model("ab3_prog"), time=P)
        cm = _passive_source_model("ab3_block").compile(backend="production")
    except RuntimeError as exc:
        print("-- (B AB3) skipped: compile could not build the .so: %s --" % str(exc)[:160])
        return
    n = 16
    sim.add_equation("blk", cm, spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                     time=pops.Explicit(method="euler"))
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("blk", np.stack([rho0]))
    sim.install_program(compiled.so_path)
    dt, nsteps = 0.01, 6
    for _ in range(nsteps):
        sim.step(dt)
    out = np.array(sim.get_state("blk"))[0]
    ref = _offline_ab(rho0, dt, nsteps, 3)
    ref2 = _offline_ab(rho0, dt, nsteps, 2)
    err = float(np.abs(out - ref).max())
    ab3_vs_ab2 = float(np.abs(ref - ref2).max())
    print("  AB3 parity: max|compiled - offline| = %.2e  max|AB3 - AB2| = %.2e" % (err, ab3_vs_ab2))
    assert err <= 1e-12, "compiled AB3 == offline AB3 to machine precision (%.2e)" % err
    assert ab3_vs_ab2 > 1e-9, "AB3 must differ from AB2 past the cold start (%.2e)" % ab3_vs_ab2


def _run_imex(t):
    try:
        import numpy as np

        import pops
        from pops.physics.facade import Model
    except Exception as exc:  # noqa: BLE001
        print("-- (B imex) skipped: pops/numpy unavailable: %s --" % exc)
        return
    sim = pops.System(n=16, L=1.0, periodic=True)
    if not hasattr(sim, "install_program"):
        print("-- (B imex) skipped: _pops lacks install_program (rebuild _pops) --")
        return
    P = t.Program("imex_step")
    lt.imex_local(P, "plasma", linear_source="lorentz", flux=True, sources=["default"], theta=1.0)
    try:
        compiled = pops.compile_problem(model=_lorentz_model("imex_prog"), time=P)
        cm = _lorentz_model("imex_block").compile(backend="production")
    except RuntimeError as exc:
        print("-- (B imex) skipped: compile could not build the .so: %s --" % str(exc)[:160])
        return
    n = 16
    sim.add_equation("plasma", cm, spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                     time=pops.Explicit(method="euler"))
    bz = 3.0
    sim.set_magnetic_field(bz * np.ones(n * n))
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    mx = 0.5 * rho
    my = -0.2 * rho
    sim.set_state("plasma", np.stack([rho, mx, my]))
    sim.install_program(compiled.so_path)
    dt = 0.05
    sim.step(dt)
    U = np.array(sim.get_state("plasma"))
    # Offline IMEX (theta=1, ZERO flux so the explicit RHS is the default source = none):
    #   U* = U + dt*R = U  (R == 0 here)
    #   U^{n+1} = (I - dt*L)^{-1} U*   -> the implicit Lorentz rotation of (mx, my); rho unchanged.
    k = dt * bz
    den = 1.0 + k * k
    mx_ref = (mx + k * my) / den
    my_ref = (-k * mx + my) / den
    e_rho = float(np.abs(U[0] - rho).max())
    e_mom = max(float(np.abs(U[1] - mx_ref).max()), float(np.abs(U[2] - my_ref).max()))
    print("  IMEX parity: max|d(rho)| = %.2e  max|d(mx,my)| = %.2e" % (e_rho, e_mom))
    assert e_rho < 1e-13, "rho unchanged by the implicit Lorentz term (%.2e)" % e_rho
    assert e_mom < 1e-12, "stepped (mx,my) == offline implicit IMEX rotation (%.2e)" % e_mom
    assert float(np.abs(U[1] - mx).max()) > 1e-6, "the IMEX step rotated the momentum"


def _run_lie(t):
    try:
        import numpy as np

        import pops
        from pops.physics.facade import Model
    except Exception as exc:  # noqa: BLE001
        print("-- (B lie) skipped: pops/numpy unavailable: %s --" % exc)
        return
    sim = pops.System(n=16, L=1.0, periodic=True)
    if not hasattr(sim, "install_program"):
        print("-- (B lie) skipped: _pops lacks install_program (rebuild _pops) --")
        return

    # Lie H(dt); S(dt) where H is flux-only transport (here zero flux -> no-op) and S the default
    # source S = _C*rho over the full dt. Both sub-flows are forward-Euler affine updates, so the
    # composition is exactly rho * (1 + dt*_C) per step (H is inert), which an offline ref mirrors.
    def half_flow(prog, U, frac):
        R = prog.rhs(state=U, fields=prog.solve_fields(U), flux=True, sources=[])
        return prog.linear_combine(None, U + (frac * prog.dt) * R)

    def source(prog, U, frac):
        S = prog.rhs(state=U, fields=None, flux=False, sources=["reaction"])
        return prog.linear_combine(None, U + (frac * prog.dt) * S)

    P = t.Program("lie_step")
    lt.lie(P, "blk", half_flow, source)
    try:
        compiled = pops.compile_problem(model=_reaction_term_model("lie_prog"), time=P)
        cm = _reaction_term_model("lie_block").compile(backend="production")
    except RuntimeError as exc:
        print("-- (B lie) skipped: compile could not build the .so: %s --" % str(exc)[:160])
        return
    n = 16
    sim.add_equation("blk", cm, spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                     time=pops.Explicit(method="euler"))
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("blk", np.stack([rho0]))
    sim.install_program(compiled.so_path)
    dt, nsteps = 0.01, 5
    for _ in range(nsteps):
        sim.step(dt)
    out = np.array(sim.get_state("blk"))[0]
    # Offline sequential split: H(dt) inert, then S(dt): rho <- rho + dt*_C*rho each step.
    ref = rho0.copy()
    for _ in range(nsteps):
        ref = ref + dt * (_C * ref)  # H no-op, then S over full dt
    err = float(np.abs(out - ref).max())
    print("  Lie parity: max|compiled - offline sequential split| = %.2e" % err)
    assert err <= 1e-12, "compiled Lie == offline sequential split to machine precision (%.2e)" % err


def _run():
    t = _pops_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    print("PASS test_time_std_imex_lie_ab (A: %d checks)" % len(fns))
    _run_imex(t)
    _run_lie(t)
    _run_ab3(t)


if __name__ == "__main__":
    _run()
