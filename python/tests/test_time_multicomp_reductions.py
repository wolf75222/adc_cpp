#!/usr/bin/env python3
"""adc.time FULL multi-component IR reductions (epic ADC-399 / ADC-432).

The Program reductions P.norm2 / P.dot / P.sum / P.max / P.min / P.norm_inf reduce over ALL
components of a State -- the true state norm / inner product / sum / extrema -- not component 0 only
(the documented ADC-404 "later phase"). They lower to the full-component C++ helpers added by ADC-416
(adc::dot_all) and ADC-432 (adc::reduce_sum_all / reduce_max_all / reduce_min_all / norm_inf_all). For a
single-component (ncomp==1) State each is BIT-IDENTICAL to the old component-0 reduction (the _all helper
loops the one component), so the scalar path / Richardson / GMRES tests do not regress. P.sum_component
keeps reducing one explicitly named component.

(A) Pure Python (IR + codegen), always runs: P.norm2 / P.dot lower to adc::dot_all; P.norm_inf to
    adc::norm_inf_all; P.sum / P.max / P.min to the _all reductions; P.sum_component to the per-component
    adc::reduce_sum(u, comp). The scalar (ncomp-agnostic) codegen is the SAME emitted call for ncomp==1
    (the runtime ncomp branch is inside the C++ helper, so ncomp==1 stays bit-identical).

(B) End-to-end parity (skips unless the full toolchain is present): a 3-component passive block with
    DISTINCT per-component fields; record_scalar(norm2 / dot / sum / max / min / norm_inf /
    sum_component) -> compare to an OFFLINE numpy FULL-component reduction. The multi-component norm2 /
    dot / sum / max / min / norm_inf must match the full reduction (a component-0-only reduction would
    miss the other components). A scalar (ncomp==1) run matches the offline SINGLE-component reduction
    bit-for-bit (the no-regression guard).

(C) A 2-component Richardson convergence loop (skips unless the toolchain is present): x <- x +
    omega*(target - x) looping while norm2(target - x) > tol, on a 2-component state whose two
    components converge at DIFFERENT rates. The full-component norm2 stops only once BOTH components are
    converged; a component-0-only norm would stop early and leave component 1 unconverged -- so the
    component-1 match is the regression guard. Self-skips (exit 0) without numpy / _adc / install_program
    / a compiler / a visible Kokkos -- never fakes the engine.
"""
import sys


def _adc_time():
    try:
        import adc.time as t
    except Exception as exc:  # adc not importable here -> skip, never fake
        print("skip test_time_multicomp_reductions (adc.time unavailable: %s)" % exc)
        sys.exit(0)
    return t


# ---- (A) codegen: pure Python, always runs ----
def _diag_program(t):
    """Forward Euler that records every full-state reduction of U^n (plus sum_component) each step."""
    P = t.Program("reductions_step")
    U = P.state("blk")
    R = P.rhs(state=U, sources=["default"])
    P.record_scalar("state_norm2", P.norm2(U))
    P.record_scalar("state_dot", P.dot(U, U))
    P.record_scalar("state_sum", P.sum(U))
    P.record_scalar("state_max", P.max(U))
    P.record_scalar("state_min", P.min(U))
    P.record_scalar("state_norm_inf", P.norm_inf(U))
    P.record_scalar("state_sum_c0", P.sum_component(U, 0))
    P.commit("blk", P.linear_combine(U + P.dt * R))
    return P


def test_full_state_reductions_lower_to_all_helpers(t):
    import re

    src = _diag_program(t).emit_cpp_program()
    # norm2 / dot -> the full-component inner product (adc::dot_all); norm_inf -> adc::norm_inf_all.
    assert "std::sqrt(adc::dot_all(" in src, "norm2 must lower to sqrt(adc::dot_all(...))\n%s" % src
    assert "adc::dot_all(" in src, "dot must lower to adc::dot_all\n%s" % src
    assert "adc::norm_inf_all(" in src, "norm_inf must lower to adc::norm_inf_all\n%s" % src
    # sum / max / min -> the all-component reductions.
    for frag in ("adc::reduce_sum_all(", "adc::reduce_max_all(", "adc::reduce_min_all("):
        assert frag in src, "the full-state reduction codegen must contain %r\n%s" % (frag, src)
    # sum_component still pins one named component (the per-component reduce_sum(<state>, comp)) -- the
    # call takes a SECOND comma argument (the component), unlike the no-arg reduce_sum_all(<state>).
    assert re.search(r"adc::reduce_sum\([^,()]+, 0\)", src), \
        "sum_component must lower to the per-component adc::reduce_sum(<state>, comp)\n%s" % src
    # the old component-0-only full-state lowerings must be GONE: no bare reduce_max(/reduce_min( (the
    # full-state P.max/P.min now use the _all variants; there is no max_component/min_component op).
    assert "adc::reduce_max(" not in src, "P.max must no longer lower to the comp-0 reduce_max\n%s" % src
    assert "adc::reduce_min(" not in src, "P.min must no longer lower to the comp-0 reduce_min\n%s" % src


def test_reduction_codegen_is_ncomp_agnostic(t):
    # The emitted call is the SAME regardless of component count: the ncomp branch lives in the C++ _all
    # helper (ncomp==1 -> bit-identical to the comp-0 path). So no ncomp leaks into the Python codegen.
    src = _diag_program(t).emit_cpp_program()
    assert "adc::dot_all(" in src and "adc::reduce_sum_all(" in src, src


# ---- (B) end-to-end full-state reductions vs offline numpy ----
def _passive_model(dsl, name, cons):
    """An n-variable block with NO flux and NO Poisson coupling: the conservative variables double as the
    multi-component field the reductions read. @p cons is the tuple of conservative-variable names."""
    m = dsl.Model(name)
    vars_ = m.conservative_vars(*cons)
    if not isinstance(vars_, tuple):
        vars_ = (vars_,)
    z = [0.0 * v for v in vars_]
    m.flux(x=list(z), y=list(z))
    m.eigenvalues(x=list(z), y=list(z))
    m.primitive_vars(*vars_)
    m.conservative_from(list(vars_))
    return m


def _offline_reductions(np, state):
    """The FULL-component reductions of an (ncomp, n, n) array, matching the adc:: full-state helpers."""
    return {
        "norm2": float(np.sqrt(np.sum(state * state))),
        "dot": float(np.sum(state * state)),
        "sum": float(np.sum(state)),
        "max": float(np.max(state)),
        "min": float(np.min(state)),
        "norm_inf": float(np.max(np.abs(state))),
        "sum_c0": float(np.sum(state[0])),
    }


def _run_diag(t, adc, np, ncomp, init):
    """Compile + install + step the diagnostic Program on an ncomp-component block; return the recorded
    diagnostics dict, or None if the toolchain is unavailable. @p init is (ncomp, n, n)."""
    n = init.shape[1]
    sim = adc.System(n=n, L=1.0, periodic=True)
    if not hasattr(sim, "install_program") or not hasattr(sim, "program_diagnostics"):
        print("-- (B) skipped: _adc lacks install_program/program_diagnostics (rebuild _adc) --")
        return None
    from adc import dsl
    cons = tuple("c%d" % i for i in range(ncomp))
    P = _diag_program(t)
    try:
        compiled = adc.compile_problem(model=_passive_model(dsl, "red_prog%d" % ncomp, cons), time=P)
        compiled_model = _passive_model(dsl, "red_block%d" % ncomp, cons).compile(backend="production")
    except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
        print("-- (B) skipped: could not build the .so: %s --" % str(exc)[:200])
        return None
    sim.add_equation("blk", compiled_model,
                     spatial=adc.FiniteVolume(limiter="none", riemann="rusanov"),
                     time=adc.Explicit(method="euler"))
    sim.set_state("blk", init)
    sim.install_program(compiled.so_path)
    sim.step(0.01)  # the diagnostics are recorded from U^n (the state at the START of the step)
    return sim.program_diagnostics()


def _run_section_b(t):
    try:
        import numpy as np

        import adc
    except Exception as exc:  # noqa: BLE001 -- numpy / _adc unavailable in this interpreter
        print("-- (B) skipped: adc/numpy unavailable: %s --" % exc)
        return None

    n = 8
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    # Three DISTINCT components with distinct ranges -- a comp-0-only reduction would miss comp 1 and 2.
    c0 = 1.0 + 0.25 * X + 0.25 * Y           # in [1, 1.5)
    c1 = -2.0 + 0.5 * np.sin(2 * np.pi * X)  # spans negatives (drives min, |.| of norm_inf)
    c2 = 3.0 + 0.5 * np.cos(2 * np.pi * Y)   # the largest (drives max)
    init3 = np.stack([c0, c1, c2])

    diags = _run_diag(t, adc, np, 3, init3)
    if diags is None:
        return None
    for key in ("state_norm2", "state_dot", "state_sum", "state_max", "state_min",
                "state_norm_inf", "state_sum_c0"):
        assert key in diags, "program_diagnostics must contain %r (got %r)" % (key, sorted(diags))
    exp = _offline_reductions(np, init3)
    # FULL-component parity: every reduction must cover all 3 components (a comp-0-only value would be
    # the c0-only reduction and MISS the c1 min / c2 max / the c1,c2 contributions to norm2 / sum / dot).
    rel = 1e-9
    err_norm2 = abs(diags["state_norm2"] - exp["norm2"])
    err_dot = abs(diags["state_dot"] - exp["dot"])
    err_sum = abs(diags["state_sum"] - exp["sum"])
    err_max = abs(diags["state_max"] - exp["max"])
    err_min = abs(diags["state_min"] - exp["min"])
    err_ninf = abs(diags["state_norm_inf"] - exp["norm_inf"])
    err_c0 = abs(diags["state_sum_c0"] - exp["sum_c0"])
    print("  3-comp reductions vs offline full-norm: |norm2|=%.2e |dot|=%.2e |sum|=%.2e |max|=%.2e "
          "|min|=%.2e |norm_inf|=%.2e |sum_c0|=%.2e"
          % (err_norm2, err_dot, err_sum, err_max, err_min, err_ninf, err_c0))
    assert err_norm2 <= rel * max(1.0, abs(exp["norm2"])), "P.norm2 must equal the full-state L2 norm"
    assert err_dot <= rel * max(1.0, abs(exp["dot"])), "P.dot must equal the full-state inner product"
    assert err_sum <= rel * max(1.0, abs(exp["sum"])), "P.sum must equal the all-component sum"
    assert err_max <= 1e-12, "P.max must equal the all-component max (driven by component 2)"
    assert err_min <= 1e-12, "P.min must equal the all-component min (driven by component 1)"
    assert err_ninf <= 1e-12, "P.norm_inf must equal the all-component max|.| (driven by component 2)"
    assert err_c0 <= rel * max(1.0, abs(exp["sum_c0"])), "P.sum_component(.,0) must be component 0 only"
    # The full-state reductions must DIFFER from the component-0-only values (proving they cover c1, c2).
    assert abs(exp["sum"] - exp["sum_c0"]) > 1e-3, "the test fixture must make the all-comp sum != c0 sum"
    assert exp["max"] != float(np.max(init3[0])), "the all-comp max must differ from the c0 max"
    assert exp["min"] != float(np.min(init3[0])), "the all-comp min must differ from the c0 min"

    # A scalar (ncomp==1) run matches the offline SINGLE-component reduction bit-for-bit (no regression).
    diags1 = _run_diag(t, adc, np, 1, np.stack([c0]))
    if diags1 is None:
        return None
    exp1 = _offline_reductions(np, np.stack([c0]))
    err1 = max(abs(diags1["state_%s" % k] - exp1[k])
               for k in ("norm2", "sum", "max", "min", "norm_inf"))
    print("  scalar (ncomp==1) reductions vs offline single-comp: max|d| = %.2e" % err1)
    # ncomp==1 -> the _all helpers loop the one component: bit-identical to the comp-0 reductions. max /
    # min / norm_inf are exact (no rounding); sum / norm2 within the per-tile Kokkos::Sum FP tolerance.
    assert abs(diags1["state_max"] - exp1["max"]) == 0.0, "ncomp==1 P.max must be bit-identical"
    assert abs(diags1["state_min"] - exp1["min"]) == 0.0, "ncomp==1 P.min must be bit-identical"
    assert abs(diags1["state_norm_inf"] - exp1["norm_inf"]) == 0.0, \
        "ncomp==1 P.norm_inf must be bit-identical"
    assert abs(diags1["state_sum"] - exp1["sum"]) <= 1e-9 * max(1.0, abs(exp1["sum"])), \
        "ncomp==1 P.sum must match the comp-0 sum"
    assert abs(diags1["state_norm2"] - exp1["norm2"]) <= 1e-9 * max(1.0, abs(exp1["norm2"])), \
        "ncomp==1 P.norm2 must match the comp-0 norm"
    return (err_norm2, err1)


# ---- (C) a 2-component Richardson convergence loop ----
def _richardson_program(t, *, name="richardson", omega=0.5, tol=1e-10, max_iter=400):
    """x <- x + omega*(target - x), looping while ||target - x||_2 > tol (target = 2*U0), on a state.

    The body / condition use only linear_combine + norm2 (the FULL-state norm), so the loop converges
    EVERY component. On a 2-component state whose two components have different magnitudes the loop must
    keep iterating until BOTH are converged; a component-0-only norm would stop on component 0 alone."""
    P = t.Program(name)
    U0 = P.state("blk")
    target = P.linear_combine("target", 2.0 * U0)

    def cond(P, x):
        diff = P.linear_combine("diff", target - x)
        return P.norm2(diff) > tol  # FULL-state norm -> every component must converge

    def body(P, x):
        return P.linear_combine("x_next", (1.0 - omega) * x + omega * target)

    x_final = P.while_(U0, cond, body)
    P.commit("blk", x_final)
    return P


def _run_section_c(t):
    try:
        import numpy as np

        import adc
    except Exception as exc:  # noqa: BLE001
        print("-- (C) skipped: adc/numpy unavailable: %s --" % exc)
        return None

    n = 8
    sim = adc.System(n=n, L=1.0, periodic=True)
    if not hasattr(sim, "install_program"):
        print("-- (C) skipped: _adc lacks the install_program binding (rebuild _adc) --")
        return None
    from adc import dsl
    omega, tol = 0.5, 1e-10
    P = _richardson_program(t, omega=omega, tol=tol)
    try:
        compiled = adc.compile_problem(model=_passive_model(dsl, "rich_prog", ("c0", "c1")), time=P)
        compiled_model = _passive_model(dsl, "rich_block", ("c0", "c1")).compile(backend="production")
    except RuntimeError as exc:
        print("-- (C) skipped: could not build the .so: %s --" % str(exc)[:200])
        return None
    sim.add_equation("blk", compiled_model,
                     spatial=adc.FiniteVolume(limiter="none", riemann="rusanov"),
                     time=adc.Explicit(method="euler"))

    x = (np.arange(n) + 0.5) / n
    X, _ = np.meshgrid(x, x, indexing="ij")
    # Component 0 is small (converges in a few iters under a comp-0 norm); component 1 is much larger, so
    # ||target - x|| stays > tol until component 1 also converges. The full-state norm catches this.
    c0 = 0.01 + 0.0 * X
    c1 = 100.0 + 10.0 * X
    init2 = np.stack([c0, c1])
    sim.set_state("blk", init2)
    sim.install_program(compiled.so_path)
    sim.step(0.01)
    out2 = np.array(sim.get_state("blk"))

    # The fixed point is x = target = 2*U0 (omega-independent); convergence drives x -> 2*init2.
    target = 2.0 * init2
    err = float(np.abs(out2 - target).max())
    err1 = float(np.abs(out2[1] - target[1]).max())  # the large component -- the regression guard
    print("  2-comp Richardson: max|x - 2*U0| = %.2e  max|comp1 - target1| = %.2e" % (err, err1))
    # A converged loop reaches the fixed point on BOTH components. A comp-0-only norm would stop once the
    # tiny component 0 converged and leave component 1 ~ its start (off by O(100)).
    assert err <= 1e-6, "the full-state Richardson loop must converge ALL components (max|d| = %.2e)" % err
    assert err1 <= 1e-6, ("component 1 must converge too (max|d| = %.2e) -- the full-state norm "
                          "regression guard" % err1)
    return err


def _run():
    t = _adc_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    print("PASS test_time_multicomp_reductions (A: %d checks)" % len(fns))
    _run_section_b(t)
    _run_section_c(t)


if __name__ == "__main__":
    _run()
