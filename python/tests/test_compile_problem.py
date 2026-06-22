#!/usr/bin/env python3
"""adc.compile_problem + sim.install_program + CompiledTime, end to end (epic ADC-399 / ADC-401).

(A) Validation (pure Python, always runs): compile_problem rejects backend != 'production' and
    target != 'system' and a missing Program; a multi-stage Program is refused by the codegen;
    CompiledTime rejects substeps/stride > 1 and a non-default cfl (all deferred, fail-loud).

(B) End-to-end parity (skips cleanly unless the full toolchain is present): build a transport gas
    block + a Forward-Euler Program, compile_problem -> problem.so (compiled WITH Kokkos, so its ABI
    key matches _adc and it loads in-process), sim.install_program, sim.step(dt), and check parity
    against the reference one-step U0 + dt * eval_rhs (after solve_fields) -- the SAME primitives the
    Program drives, so bit/near parity. Runs in CI (gate-python rebuilds _adc with the install_program
    binding) and locally once _adc is rebuilt; skips if _adc lacks install_program, numpy/_adc is
    absent, no compiler/Kokkos is visible, or the .so compile fails -- never faking the engine.
"""
import sys


def _skip(msg):
    print("skip test_compile_problem (%s)" % msg)
    sys.exit(0)


try:
    import numpy as np

    import adc
    from adc import time as adctime
except Exception as exc:  # noqa: BLE001  -- numpy or _adc unavailable in this interpreter
    _skip("adc/numpy unavailable: %s" % exc)

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
    except Exception:  # noqa: BLE001  -- wrong exception type is a failure, not a pass
        return False
    return False


def _fe_program(name="forward_euler_parity"):
    P = adctime.Program(name)
    dt = P.dt
    U = P.state("ions")
    f = P.solve_fields(U)
    R = P.rhs(state=U, fields=f, flux=True, sources=["default"])
    P.commit("ions", P.linear_combine("U1", U + dt * R))
    return P


def _multistage_program():
    P = adctime.Program("ssprk2_parity")
    dt = P.dt
    U0 = P.state("ions")
    f0 = P.solve_fields(U0)
    k0 = P.rhs(state=U0, fields=f0, flux=True, sources=["default"])
    U1 = P.linear_combine("U1", U0 + dt * k0)
    f1 = P.solve_fields(U1)
    k1 = P.rhs(state=U1, fields=f1, flux=True, sources=["default"])
    P.commit("ions", P.linear_combine("U2", 0.5 * U0 + 0.5 * (U1 + dt * k1)))
    return P


# ---- (A) validation: pure Python, always runs ----
print("== (A) compile_problem / CompiledTime validation ==")
chk(raises(ValueError, lambda: adc.compile_problem(time=_fe_program(), backend="aot")),
    "compile_problem backend != 'production' rejected")
chk(raises(ValueError, lambda: adc.compile_problem(time=_fe_program(), target="amr_system")),
    "compile_problem target != 'system' rejected")
chk(raises(ValueError, lambda: adc.compile_problem(time=None)),
    "compile_problem without a Program rejected")
chk(raises(NotImplementedError, lambda: adc.compile_problem(time=_multistage_program())),
    "compile_problem refuses a multi-stage Program (codegen MVP)")
chk(raises(NotImplementedError, lambda: adc.CompiledTime(substeps=2)),
    "CompiledTime substeps>1 rejected (deferred)")
chk(raises(NotImplementedError, lambda: adc.CompiledTime(stride=2)),
    "CompiledTime stride>1 rejected (deferred)")
chk(raises(NotImplementedError, lambda: adc.CompiledTime(cfl="program")),
    "CompiledTime cfl!='default' rejected (deferred)")
chk(adc.CompiledTime().kind == "compiled", "CompiledTime() default ok (kind 'compiled')")

# ---- (B) end-to-end parity: skips unless the full toolchain is present ----
# install_program is forwarded by the System facade (__getattr__ -> self._s), so probe an instance.
if not hasattr(adc.System(n=8, L=1.0, periodic=True), "install_program"):
    print("-- (B) skipped: _adc lacks the install_program binding (rebuild _adc) --")
    print("%s test_compile_problem (A only)" % ("FAIL" if fails else "PASS"))
    sys.exit(1 if fails else 0)

print("== (B) end-to-end: compiled Program vs reference one-step ==")


def transport_model():
    # Pure transport (isothermal, NoSource); the inert elliptic + set_poisson make solve_fields
    # well-defined and identical in both the reference and the compiled-Program path.
    return adc.Model(state=adc.FluidState("isothermal", cs2=0.5),
                     transport=adc.IsothermalFlux(),
                     source=adc.NoSource(),
                     elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0))


def make_sim():
    n = 24
    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.add_block("ions", transport_model(),
                  spatial=adc.FiniteVolume(limiter="none", riemann="rusanov"),
                  time=adc.Explicit(method="euler"))
    sim.set_poisson("charge_density", "geometric_mg")
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("ions", np.stack([rho, 0.4 * rho, -0.2 * rho]))
    return sim


dt = 2e-3

# Reference: one Forward-Euler step via the SAME primitives the Program drives.
ref = make_sim()
U0 = np.array(ref.get_state("ions"))
ref.solve_fields()
R0 = np.array(ref.eval_rhs("ions"))
U_ref = U0 + dt * R0

# Compiled-Program path: lower the FE IR -> problem.so, install it, step once.
try:
    compiled = adc.compile_problem(model=transport_model(), time=_fe_program())
except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
    _skip("compile_problem could not build the .so: %s" % str(exc)[:160])

chk(compiled.program_name == "forward_euler_parity", "handle carries the program name")
chk(bool(compiled.program_hash), "handle carries the IR hash")

prog = make_sim()
prog.install_program(compiled.so_path)  # dlopen + ABI-key check + adc_install_program(this)
step0 = prog.macro_step()
prog.step(dt)  # SystemStepper dispatches to the installed Program
U_prog = np.array(prog.get_state("ions"))

emax = float(np.abs(U_prog - U_ref).max())
change = float(np.abs(U_prog - U0).max())
chk(emax < 1e-12, "compiled FE Program == reference one-step (max|d| = %.2e)" % emax)
chk(prog.macro_step() == step0 + 1, "macro_step advanced (%d -> %d)" % (step0, prog.macro_step()))
chk(change > 1e-9, "the step actually changed the state (change = %.2e)" % change)

print("%s test_compile_problem" % ("FAIL (%d)" % fails if fails else "PASS"))
sys.exit(1 if fails else 0)
