#!/usr/bin/env python3
"""Compiled pops.time.std.strang reproduces native pops.Strang on a simple case (ADC-410).

The Strang splitting macro H(dt/2); S(dt); H(dt/2) is expressed ONCE as Program IR via the
``pops.time.std.strang`` combinator (no scheme-specific C++ stepper). This test demonstrates that the
COMPILED Strang composition runs end to end C++-side and reproduces the native engine Strang macro-step
(SystemStepper::step_strang) to BIT precision on a simple, faithfully replicable case.

The simple case (the one where the compiled half-flow can EXACTLY mirror the native hyperbolic stage):
an UNCOUPLED isothermal fluid (no Poisson feedback into the flux) with NO source brick, advanced with
Forward Euler. There:
  - native H(dt/2) = advance_transport_half = one Euler transport step over dt/2 = U + (dt/2)*(-div F);
  - native S(dt)   = run_source_stage is a genuine NO-OP (the block carries no Schur / source stage:
                     we drive the Strang scheme via set_time_scheme("strang") WITHOUT set_source_stage,
                     so s.schur == nullptr and run_source_stage returns immediately);
  - the three solve_fields fences in step_strang are INERT (BackgroundDensity n0=0 elliptic; the
    isothermal flux reads no field) so they change nothing between the half-advances.
The matching compiled program: half_flow(P, U, frac) = U + frac*dt*rhs(flux=True, sources=["default"])
(flux-only transport; the model's default source is NoSource, so rhs is -div F) and source(P, U, frac)
= U (the no-op S). Both paths run U + (dt/2)*(-div F) twice around an inert middle, so they coincide.

(A) Pure Python (always runs): the std.strang macro lowers to the expected IR -- two half-flow stages
    around the (no-op) source, each a U + (dt/2)*R affine combination -- and emit_cpp_program produces
    the matching C++ (two ctx.rhs_into flux assemblies, two half-step lincombs).

(B) Native bit-parity (skips cleanly without _pops.install_program / numpy / a compiler / a visible
    Kokkos): set up two identical Systems on the uncoupled model with the same IC; one runs the native
    pops.Strang scheme (set_time_scheme("strang")), the other installs the compiled std.strang program;
    step BOTH N steps and assert max|native - compiled| is bit-exact (array_equal). A SECOND, fully
    independent reference replays the identical H(dt/2); no-op; H(dt/2) sub-steps offline (set_state +
    eval_rhs) and must match the compiled program to machine precision; a single full Euler step is
    shown to DIFFER (the composition genuinely runs two half-steps, not one).

RESULT (this case): native bit-parity IS achieved -- the compiled std.strang program reproduces the
engine's step_strang to the last bit (array_equal, max|d| = 0). It is the simplest scheme where the
compiled half-flow can mirror native H exactly: Forward Euler over dt/2 is a single affine update
U + (dt/2)*(-div F), identical to the compiled half-flow stage; an RK scheme (e.g. ssprk2) would need
the half-flow to reproduce the multi-stage native advance, which it can but only the offline-same-steps
reference would witness it -- Euler keeps the native cross-check exact.

Run with python3 (PYTHONPATH = built adc package).
"""
import sys


def _skip(msg):
    print("skip test_time_strang_parity (%s)" % msg)
    sys.exit(0)


from pops import time as adctime  # noqa: E402  -- IR construction is pure Python, always available


# ============================ (A) IR construction + codegen: pure Python =======================
def half_flow(prog, U, frac):
    """One Forward-Euler hyperbolic half-flow: U + frac*dt*R, R = -div F (+ default source). On the
    uncoupled NoSource model the default source is empty, so R is flux-only; the per-stage solve_fields
    is inert (no field feedback) -- kept so the program mirrors a real, field-coupled-ready Strang."""
    R = prog.rhs(state=U, fields=prog.solve_fields(U), flux=True, sources=["default"])
    return prog.linear_combine(None, U + (frac * prog.dt) * R)


def no_op_source(prog, U, frac):  # noqa: ARG001  -- frac unused: S is the identity on a NoSource model
    """The Strang source stage S(dt). On a model with no source brick the native run_source_stage is a
    no-op, so the compiled source returns U unchanged (no extra IR stage)."""
    return U


def strang_program(name="strang_parity", block="ions"):
    """The compiled Strang program H(dt/2); S(dt); H(dt/2) built via pops.time.std.strang (no special
    Strang class -- the same combinator + affine algebra over dt)."""
    P = adctime.Program(name)
    adctime.std.strang(P, block, half_flow, no_op_source)
    return P


def _coeff(node, value):
    for v, c in zip(node.inputs, node.attrs["coeffs"], strict=True):
        if v is value:
            return c
    raise AssertionError("value %r not an input of %r" % (value, node))


fails_a = 0


def chk_a(cond, label):
    global fails_a
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails_a += 1


print("== (A) std.strang lowers to two half-flow stages around a no-op source ==")
P = strang_program()
P.validate()
out = P.commits()["ions"]
chk_a(out.op == "linear_combine" and out.vtype == "state",
      "the committed value is the final half-flow state")

# Two half-flow linear_combine stages (the no-op source builds NONE), each a U + (dt/2)*R update.
lcs = [v for v in P._values if v.op == "linear_combine"]
rhss = [v for v in P._values if v.op == "rhs"]
chk_a(len(lcs) == 2, "exactly two half-flow stages (the no-op source adds no stage); got %d" % len(lcs))
chk_a(len(rhss) == 2, "one flux RHS per half-flow; got %d" % len(rhss))
for lc in lcs:
    states = [v for v in lc.inputs if v.vtype == "state"]
    rs = [v for v in lc.inputs if v.vtype == "rhs"]
    chk_a(len(states) == 1 and _coeff(lc, states[0]) == {0: 1.0},
          "half-flow keeps U with coefficient 1")
    chk_a(len(rs) == 1 and _coeff(lc, rs[0]) == {1: 0.5},
          "half-flow adds (dt/2)*R (coefficient 0.5 on dt^1)")

# emit_cpp_program produces the matching C++: two flux RHS assemblies + two half-step lincombs.
try:
    import numpy as np  # noqa: F401  -- pops.Model construction pulls in numpy

    import pops
except Exception:  # noqa: BLE001  -- numpy / _pops unavailable: (A) codegen is still checked below
    np = None
    adc = None

if adc is not None:
    def transport_model():
        """Uncoupled isothermal fluid (no field coupling into the flux), NO source brick: native H is a
        pure Euler transport step and native S (run_source_stage) is a no-op."""
        return pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                         transport=pops.IsothermalFlux(),
                         source=pops.NoSource(),
                         elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0))

    src = strang_program().emit_cpp_program(model=transport_model())
    chk_a(src.count("ctx.rhs_into(") == 2,
          "two flux RHS assemblies lowered (one per half-flow)")
    chk_a(src.count("ctx.lincomb(") >= 1 and src.count("ctx.axpy(") >= 2,
          "the half-step updates lower to axpy + a commit lincomb")
else:
    print("  -- emit_cpp_program codegen check skipped (numpy/_pops unavailable) --")

if fails_a:
    print("FAIL (%d) test_time_strang_parity (A)" % fails_a)
    sys.exit(1)
print("  (A) PASS")


# ============================ (B) native bit-parity: skip without the toolchain ================
if adc is None:
    _skip("pops/numpy unavailable (A passed)")
if not hasattr(pops.System(n=8, L=1.0, periodic=True), "install_program"):
    _skip("_pops lacks the install_program binding (rebuild _pops) (A passed)")

N = 24
DT = 2e-3
NSTEP = 4

fails_b = 0


def chk_b(cond, label):
    global fails_b
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails_b += 1


def make_sim():
    """A System with ONE uncoupled isothermal block (Forward Euler hyperbolic stage) + inert Poisson."""
    sim = pops.System(n=N, L=1.0, periodic=True)
    sim.add_block("ions", transport_model(),
                  spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                  time=pops.Explicit(method="euler"))
    sim.set_poisson("charge_density", "geometric_mg")  # inert: BackgroundDensity n0=0, flux reads no phi
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("ions", np.stack([rho, 0.4 * rho, -0.2 * rho]))
    return sim


print("== (B) compiled std.strang == native pops.Strang (bit-exact) ==")

# Compile the std.strang program (skips cleanly without a compiler / visible Kokkos).
try:
    compiled = pops.compile_problem(model=transport_model(), time=strang_program("strang_prog"))
except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
    _skip("compile_problem could not build the .so: %s (A passed)" % str(exc)[:160])

U0 = np.array(make_sim().get_state("ions"))

# Native engine Strang: set_time_scheme("strang") drives SystemStepper::step_strang. WITHOUT a Schur /
# source stage, run_source_stage is a no-op -- exactly the H(dt/2); no-op; H(dt/2) the program mirrors.
sim_native = make_sim()
sim_native._s.set_time_scheme("strang")

# Compiled Strang program: installed, driven by sim.step(dt).
sim_compiled = make_sim()
sim_compiled.install_program(compiled.so_path)

for _ in range(NSTEP):
    sim_native.step(DT)
    sim_compiled.step(DT)

U_native = np.array(sim_native.get_state("ions"))
U_compiled = np.array(sim_compiled.get_state("ions"))
e_native = float(np.abs(U_native - U_compiled).max())
print("  native parity: max|native - compiled| = %.2e over %d steps" % (e_native, NSTEP))
chk_b(np.array_equal(U_native, U_compiled),
      "compiled std.strang == native pops.Strang BIT-EXACTLY (max|d| = %.2e)" % e_native)

# Independent OFFLINE reference: replay the identical H(dt/2); no-op; H(dt/2) sub-steps via the runtime
# primitives the program drives (set_state + eval_rhs). Matches the compiled program to machine eps (the
# only gap is numpy's U + (dt/2)*R ordering vs the C++ axpy/lincomb ordering).
ref = make_sim()


def euler_half(U):
    ref.set_state("ions", U)
    ref.solve_fields()  # inert (no field feedback); mirrors the program's per-stage solve_fields
    return U + 0.5 * DT * np.array(ref.eval_rhs("ions"))  # one Euler transport step over dt/2


U_offline = U0.copy()
for _ in range(NSTEP):
    U_offline = euler_half(euler_half(U_offline))  # H(dt/2); S no-op; H(dt/2)
e_offline = float(np.abs(U_compiled - U_offline).max())
print("  offline same-steps reference: max|compiled - offline| = %.2e" % e_offline)
chk_b(e_offline < 1e-12,
      "compiled std.strang == offline H(dt/2); no-op; H(dt/2) reference (max|d| = %.2e)" % e_offline)

# The Strang composition genuinely runs TWO half-steps: a single full Euler step (the default Lie path
# with method='euler') gives a DIFFERENT result, so the parity above is not a degenerate single step.
sim_lie = make_sim()  # default scheme = Lie -> one full Euler transport step per macro-step
for _ in range(NSTEP):
    sim_lie.step(DT)
U_lie = np.array(sim_lie.get_state("ions"))
d_compose = float(np.abs(U_native - U_lie).max())
print("  Strang vs single full-step Euler: max|d| = %.2e" % d_compose)
chk_b(d_compose > 1e-9,
      "Strang runs two half-steps (differs from a single full Euler step by %.2e)" % d_compose)

# Sanity: the step actually advanced the state (the parity is not 0 == 0 on a frozen field).
chk_b(float(np.abs(U_compiled - U0).max()) > 1e-6, "the Strang step actually advanced the state")

if fails_b:
    print("FAIL (%d) test_time_strang_parity" % fails_b)
    sys.exit(1)
print("PASS test_time_strang_parity (native bit-parity achieved: max|d| = %.2e)" % e_native)
sys.exit(0)
