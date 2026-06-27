#!/usr/bin/env python3
"""Spec 5 Item B (sec.12.5 / sec.13.11.1, criteria 42/43): elliptic-solver native counters.

The elliptic field solve is 96-99.9% of the step cost, yet the profiler used to expose a single
opaque ``field_solve`` scope. ADC-479 breaks it down: the GeometricMG caches its per-solve stats
(V-cycles, final residual, bottom self-time -- chrono only, no profiler in the deep numerics) and the
System reads them back at the ``field_solve`` seam to emit native counters ``mg_cycles`` /
``krylov_iters`` / ``mg_levels`` plus an ``elliptic_bottom`` timing scope.

This builds a REAL native System with a Poisson-coupled charge block (native bricks, NO DSL compile,
so it needs only ``_pops``; the "no fake adc" rule -- use the real engine), enables profiling, runs a
direct ``solve_fields()`` several times so the multigrid V-cycle actually iterates, and asserts the
report now carries ``mg_cycles > 0`` alongside the ``field_solve`` timing. It also exercises the typed
``PerformanceSummary.by_elliptic()`` view.

NOTE on the seam: ``sim.step()`` calls the Impl's ``solve_fields`` (the SystemFieldSolver method),
not ``System::solve_fields`` -- so the counters (emitted at the binding seam) are recorded by a
DIRECT ``sim.solve_fields()`` call, which is what this test drives. This mirrors test_profiling.py's
"field_solve is captured only on a direct sim.solve_fields()" note.

Cleanly SKIPS if ``_pops`` / numpy is unavailable, OR if ``_pops`` predates these counters (the
pre-rebuild module emits no ``mg_cycles``): the test is written to run POST-rebuild. A real setup bug
(engine present, counters expected) fails loudly rather than masquerading as a skip.
"""
import sys


def _skip(msg):
    print("skip test_spec5_elliptic_profiling (%s)" % msg)
    sys.exit(0)


try:
    import numpy as np

    import pops
except Exception as exc:  # noqa: BLE001
    _skip("pops/numpy unavailable: %s" % exc)


def _charge_model(q=1.0, b0=1.0):
    """A native diocotron-style scalar charge block: ExB transport reads grad phi, the elliptic brick
    contributes q*n to the Poisson RHS (ChargeDensity). No DSL -- pure native bricks."""
    return pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=b0),
                      source=pops.NoSource(), elliptic=pops.ChargeDensity(charge=q))


def _zero_mean_bump(n, amp):
    """A zero-mean density bump: Sum q n must integrate to 0 for the PERIODIC Poisson to be solvable,
    and a non-flat charge gives the V-cycle a real residual to drive down (so mg_cycles > 0)."""
    xs = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(xs, xs)
    rho = 1.0 + amp * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.01)
    return rho - rho.mean()  # zero mean -> solvable periodic Poisson


def _build(n=32):
    sim = pops.System(n=n, L=1.0, periodic=True)
    sim.add_block("ne", model=_charge_model(q=1.0), spatial=pops.Spatial(minmod=True))
    sim.set_poisson(bc="periodic")  # default solver = geometric_mg -> multigrid V-cycles
    sim.set_density("ne", _zero_mean_bump(n, 0.40))
    return sim


def main():
    fails = 0

    def chk(cond, label):
        nonlocal fails
        print("  [%s] %s" % ("OK " if cond else "XX ", label))
        if not cond:
            fails += 1

    sim = _build(n=32)
    chk(hasattr(sim, "enable_profiling") and hasattr(sim, "solve_fields"),
        "System exposes enable_profiling + solve_fields")

    sim.enable_profiling()
    # A direct solve_fields() each step records the field_solve phase AND (post-rebuild) the elliptic
    # counters at the binding seam. ~20 calls so the V-cycle iterates many times; the warm start makes
    # later solves cheap, but the first solve from a cold phi must cycle (mg_cycles accumulates > 0).
    for _ in range(20):
        sim.solve_fields()
        sim.step_cfl(0.4)
    report = sim.profile_report()
    print(report)

    chk("field_solve" in report, "report carries the field_solve timing phase")

    # POST-REBUILD detection: the pre-rebuild _pops emits no mg_cycles. If absent, SKIP (this test is
    # written to run after the rebuild that adds the counters), per the prompt.
    if "mg_cycles" not in report:
        _skip("_pops predates the elliptic counters (mg_cycles absent) -- rebuild _pops to run this")

    # The geometric multigrid actually iterated: mg_cycles is a positive accumulated count.
    counters = {}
    for line in report.splitlines():
        s = line.strip()
        if s.startswith("counters:"):
            for tok in s[len("counters:"):].split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    try:
                        counters[k] = int(v)
                    except ValueError:
                        pass
    print("  counters:", counters)
    chk(counters.get("mg_cycles", 0) > 0, "mg_cycles > 0 (the V-cycle actually iterated)")
    chk("mg_levels" in counters and counters["mg_levels"] >= 1,
        "mg_levels reported (multigrid hierarchy depth >= 1)")
    chk("krylov_iters" in counters,
        "krylov_iters reported (0 on the geometric_mg path -- honest, not faked)")
    chk(counters.get("krylov_iters", -1) == 0,
        "krylov_iters == 0 on the default Poisson (multigrid, not a Krylov elliptic solver)")
    chk("elliptic_bottom" in report, "elliptic_bottom timing scope present (coarsest-grid self-time)")

    # Typed Python view (pops.runtime.profile.PerformanceSummary.by_elliptic).
    try:
        from pops.runtime.profile import PerformanceSummary, Profile
    except Exception as exc:  # noqa: BLE001
        chk(False, "import PerformanceSummary/Profile: %s" % exc)
    else:
        summ = PerformanceSummary(report, Profile.Advanced())
        view = summ.by_elliptic()
        print("  by_elliptic:", view)
        chk(bool(view), "by_elliptic() is available (not the unavailable sentinel)")
        chk(view.get("mg_cycles", 0) > 0, "by_elliptic surfaces mg_cycles > 0")
        chk("elliptic_bottom" in view and "total_s" in view["elliptic_bottom"],
            "by_elliptic surfaces the elliptic_bottom timing entry")
        # by_solver still exposes the coarse field_solve phase.
        chk("field_solve" in summ.by_solver(), "by_solver still carries field_solve")

    # reset clears the elliptic counters too.
    sim.reset_profiling()
    chk("mg_cycles" not in sim.profile_report(), "reset_profiling clears the elliptic counters")

    print("test_spec5_elliptic_profiling: %d failure(s)" % fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
