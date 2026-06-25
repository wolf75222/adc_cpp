#!/usr/bin/env python3
"""Spec 3 profiling (ADC-459): sim.enable_profiling() / profile_report() over a real System.

Section (A) checks the binding surface and the enable/disable/report state machine on a bare
System. Section (B) builds a real NATIVE block (no DSL compile, so it needs only the _adc module),
steps it under profiling, and asserts the report carries the timed "step" phase + the step counter.
It never fakes the engine -- it builds a real adc.System; it self-skips only if _adc/numpy is
unavailable, and otherwise fails loudly (a setup bug must not masquerade as a skip).
"""
import sys


def _skip(msg):
    print("skip test_profiling (%s)" % msg)
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


# ---- (A) binding surface + enable/disable state machine (no compile needed) ----
print("== (A) profiling API present + toggles ==")
sim = adc.System(n=8, L=1.0, periodic=True)
for name in ("enable_profiling", "disable_profiling", "is_profiling", "reset_profiling",
             "profile_report"):
    chk(hasattr(sim, name), "System exposes %s" % name)

chk(sim.is_profiling() is False, "disabled by default")
sim.enable_profiling()
chk(sim.is_profiling() is True, "enable_profiling -> on")
sim.disable_profiling()
chk(sim.is_profiling() is False, "disable_profiling -> off")
chk(isinstance(sim.profile_report(), str), "profile_report returns a str")


# ---- (B) end-to-end: a stepped NATIVE block records its "step" phase + counter ----
# Native block (no install_program) -> needs only _adc, no compiler/Kokkos-root. So this runs
# whenever section A ran; a failure here is a real bug, not a skip.
print("== (B) profile_report carries the timed step phase ==")
N = 16
sim2 = adc.System(n=N, L=1.0, periodic=True)
sim2.add_block("gas",
               adc.Model(state=adc.FluidState("isothermal", cs2=0.5),
                         transport=adc.IsothermalFlux(),
                         source=adc.NoSource(),
                         elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0)),
               spatial=adc.FiniteVolume(limiter="none", riemann="rusanov"), time=adc.Explicit())
rho = np.ones((N, N), dtype=float)
sim2.set_state("gas", np.stack([rho, 0.1 * rho, 0.0 * rho]))

# during step() only the "step" phase + "steps" counter are recorded (the stepper calls the Impl's
# solve_fields, not System::solve_fields -- "field_solve" is captured only on a direct sim.solve_fields()).
sim2.enable_profiling()
sim2.step(1e-3)
sim2.step(1e-3)
report = sim2.profile_report()
print(report)
chk("step" in report, "report mentions the step phase")
chk("steps=2" in report, "step counter == 2")
chk(report.count("\n") >= 1, "report is multi-line")

# a direct solve_fields() is what records the "field_solve" phase
sim2.solve_fields()
chk("field_solve" in sim2.profile_report(), "direct solve_fields records its phase")

sim2.reset_profiling()
chk(sim2.profile_report().find("steps=") == -1, "reset clears the counters")


print("test_profiling: %d failure(s)" % fails)
sys.exit(1 if fails else 0)
