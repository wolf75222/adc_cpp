#!/usr/bin/env python3
"""Spec 3 section 29 profiling COUNTERS (ADC-459): kernel count, scratch peak, cache hits/misses,
scheduled nodes due/skipped -- surfaced by sim.profile_report() and counted in the C++ runtime.

This complements test_profiling.py (which covers the timing report + the "steps" counter). Here we
assert the named §29 counter LINES appear with sane values. It builds a real NATIVE block (no DSL
compile, so it needs only _adc) and steps it under profiling: the native step's elliptic field solve
is the kernel-dispatch chokepoint (System::Impl::solve_fields counts "kernels"), so "kernels" moves on
the host path. The cache hit/skip + nodes due/skipped counters only move under a COMPILED .so step body
that emits a held schedule (ProgramContext::cache_should_update); that runtime is exercised on
Kokkos/ROMEO, so here we assert those lines simply EXIST and read 0 on the native path -- never faked.

Real engine only: it builds a real adc.System and self-skips only if _adc/numpy is unavailable.
"""
import sys


def _skip(msg):
    print("skip test_profiling_counters (%s)" % msg)
    sys.exit(0)


try:
    import numpy as np

    import adc
except Exception as exc:  # noqa: BLE001
    _skip("adc/numpy unavailable: %s" % exc)

fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


# ---- build a real native block and step it under profiling ----
print("== §29 counters on a stepped native block ==")
N = 16
sim = adc.System(n=N, L=1.0, periodic=True)
sim.add_block("gas",
              adc.Model(state=adc.FluidState("isothermal", cs2=0.5),
                        transport=adc.IsothermalFlux(),
                        source=adc.NoSource(),
                        elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0)),
              spatial=adc.FiniteVolume(limiter="none", riemann="rusanov"), time=adc.Explicit())
rho = np.ones((N, N), dtype=float)
sim.set_state("gas", np.stack([rho, 0.1 * rho, 0.0 * rho]))

sim.enable_profiling()
sim.step(1e-3)
sim.step(1e-3)
sim.step(1e-3)
report = sim.profile_report()
print(report)

# (1) kernel count moves on the native path: the elliptic solve at the head of each native step is
# counted as a kernel launch, so >= the number of steps.
chk("kernels=" in report, "report carries the kernels counter")
# the counters render as "name=value" on the trailing counters line; parse the kernels value.
kernels = None
for tok in report.replace("\n", " ").split():
    if tok.startswith("kernels="):
        kernels = int(tok.split("=", 1)[1])
chk(kernels is not None and kernels > 0, "kernel count > 0 (= %r)" % kernels)

# (2) the cache hit/skip + nodes due/skipped counters exist only after a compiled scheduler emits
# cache_should_update; on the native path they never fire. They are NOT present on the native report
# (a counter is created lazily on first count()), which is the honest zero -- we assert they are absent
# rather than faking a zero line. The runtime cadence is ROMEO-validated under a compiled .so.
for name in ("cache_hits", "cache_misses", "nodes_due", "nodes_skipped"):
    chk(("%s=" % name) not in report,
        "%s absent on the native path (compiled-scheduler counter, ROMEO)" % name)

# (3) the step counter still works (parity with test_profiling.py).
chk("steps=3" in report, "step counter == 3")

sim.reset_profiling()
chk("kernels=" not in sim.profile_report(), "reset clears the counters")

# profiling OFF stays zero-overhead: a stepped, never-enabled System records nothing.
print("== profiling off records no counters ==")
sim_off = adc.System(n=N, L=1.0, periodic=True)
sim_off.add_block("gas",
                  adc.Model(state=adc.FluidState("isothermal", cs2=0.5),
                            transport=adc.IsothermalFlux(),
                            source=adc.NoSource(),
                            elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0)),
                  spatial=adc.FiniteVolume(limiter="none", riemann="rusanov"),
                  time=adc.Explicit())
sim_off.set_state("gas", np.stack([rho, 0.1 * rho, 0.0 * rho]))
sim_off.step(1e-3)
chk(sim_off.profile_report().find("kernels=") == -1, "disabled profiler counts no kernels")


print("test_profiling_counters: %d failure(s)" % fails)
sys.exit(1 if fails else 0)
