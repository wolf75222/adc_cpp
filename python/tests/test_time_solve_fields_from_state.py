#!/usr/bin/env python3
"""Per-stage elliptic field solve in the compiled time Program (ADC-409).

Each ``P.solve_fields(state=U_stage)`` op now lowers to ``ctx.solve_fields_from_state(0, <U_stage>)``:
the elliptic fields are re-solved -- and the shared aux re-filled -- from THAT stage's state, not the
block's current state. So a field-COUPLED multi-stage scheme (Poisson feedback into the RHS) is exact:
stage k's RHS reads phi solved from stage k's own state. The compiled Program runs the stages
sequentially, so stage k's solve overwrites the shared aux before stage k's RHS reads it -- no distinct
per-stage FieldContext buffer is needed.

(A) Codegen (pure Python, always runs): ``P.solve_fields(state=U_stage)`` lowers to
    ``ctx.solve_fields_from_state(0, <U_stage var>)`` in the generated C++ (and the bare
    ``ctx.solve_fields();`` no longer appears); the first stage solves from the base state, a later
    stage solves from the intermediate scratch state -- a DISTINCT C++ variable.

(B) Field-coupled parity (skips unless the full toolchain is present): a 2-stage Heun (RK2) scheme on a
    model whose RHS reads grad phi (a named ``electric`` source = -rho*grad phi, with
    m.elliptic_rhs(rho) so phi depends on rho). The compiled program does:
        stage 1: solve phi(U0); R0 = rhs(U0);  U1 = U0 + dt*R0
        stage 2: solve phi(U1) [via solve_fields_from_state -- the new path]; R1 = rhs(U1)
        commit:  U_np1 = U0 + 0.5*dt*(R0 + R1)
    It is compared to an OFFLINE reference that re-solves phi PER STAGE (set_state(U1)+solve_fields+
    eval_rhs for R1). They must match to ~1e-12. The CROSS-CHECK proves the per-stage solve mattered:
    re-using phi(U0) for stage 2 (the OLD frozen-aux behavior) gives a DIFFERENT (worse) result, so the
    compiled program is genuinely on the per-stage path, not the frozen one.

Skips cleanly (exit 0) without the install_program binding / numpy / a compiler / a visible Kokkos --
never fakes the engine.
"""
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
import sys


def _skip(msg):
    print("skip test_time_solve_fields_from_state (%s)" % msg)
    sys.exit(0)


try:
    import numpy as np

    import pops
    from pops.ir.ops import sqrt
    from pops.physics.facade import Model
    from pops import time as adctime
except Exception as exc:  # noqa: BLE001  -- numpy or _pops unavailable in this interpreter
    _skip("pops/numpy unavailable: %s" % exc)

fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


# --- a FIELD-COUPLED isothermal fluid block: the electric force reads grad phi (Poisson feedback) ----
def _base_block(m):
    """Shared isothermal 2D fluid block (flux + primitives + eigenvalues + Poisson rhs = rho)."""
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    cs2 = m.param("cs2", 0.5)
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    p = m.primitive("p", cs2 * rho)
    m.primitive_vars(rho=rho, u=u, v=v, p=p)
    m.conservative_from([rho, rho * u, rho * v])
    m.flux(x=[mx, mx * u + p, my * u], y=[my, mx * v, my * v + p])
    cs = sqrt(cs2)
    m.eigenvalues(x=[u - cs, u, u + cs], y=[v - cs, v, v + cs])
    m.elliptic_rhs(rho)  # phi depends on rho -> a stage change re-solves a DIFFERENT phi
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    return rho, mx, my, gx, gy


def named_source_model(name="sffs_named"):
    """Default source EMPTY; the electric force (-rho*grad phi) is a NAMED source the Program requests.
    This is the model the compiled Program drives."""
    m = Model(name)
    rho, mx, my, gx, gy = _base_block(m)
    m.source_term("electric", [0.0, -rho * gx, -rho * gy])
    return m


def default_source_model(name="sffs_default"):
    """Same physics, but the electric force is the model's DEFAULT source, so eval_rhs returns
    -div F + electric directly -- the offline reference reads it that way."""
    m = Model(name)
    rho, mx, my, gx, gy = _base_block(m)
    m.source([0.0, -rho * gx, -rho * gy])
    return m


# --- the 2-stage (Heun / RK2) field-coupled Program ---
def heun_program(name="sffs_heun"):
    """U1 = U0 + dt*R0 ; U_np1 = U0 + 0.5*dt*(R0 + R1) with R0 = rhs(U0; phi(U0)) and
    R1 = rhs(U1; phi(U1)) -- the second field solve is from U1 (solve_fields_from_state)."""
    P = adctime.Program(name)
    dt = P.dt
    U0 = P.state("plasma")
    f0 = P.solve_fields("fields_0", U0)
    R0 = P.rhs(name="R0", state=U0, fields=f0, flux=True, sources=["electric"])
    U1 = P.linear_combine("U1", U0 + dt * R0)
    f1 = P.solve_fields("fields_1", U1)            # <-- solved from U1, not U0 (ADC-409)
    R1 = P.rhs(name="R1", state=U1, fields=f1, flux=True, sources=["electric"])
    P.commit("plasma", P.linear_combine("U_np1", U0 + 0.5 * dt * R0 + 0.5 * dt * R1))
    return P


# ============================ (A) codegen: pure Python, always runs ============================
print("== (A) solve_fields(state) lowers to solve_fields_from_state ==")
src = heun_program().emit_cpp_program(model=named_source_model())

# Both stages lower to the per-stage solve; the bare current-state form is gone.
chk(src.count("ctx.solve_fields_from_state(0, ") == 2,
    "both field solves lower to ctx.solve_fields_from_state(0, <stage state>)")
chk("ctx.solve_fields();" not in src,
    "the bare current-state ctx.solve_fields() no longer appears (per-stage solve)")

# The first solve reads the base state var (ctx.state(0)); the second reads the intermediate scratch
# state var. They must be DISTINCT C++ variables (stage 2 solves from U1, not U0).
import re  # noqa: E402  -- local to the codegen assertions

solve_args = re.findall(r"ctx\.solve_fields_from_state\(0, (\w+)\);", src)
chk(len(solve_args) == 2 and solve_args[0] != solve_args[1],
    "the two field solves read DISTINCT stage-state variables (%r)" % solve_args)

# The base state var (= ctx.state(0)) is the first solve's argument.
base_decl = re.search(r"pops::MultiFab& (\w+) = ctx\.state\(0\);", src)
chk(base_decl is not None and solve_args and solve_args[0] == base_decl.group(1),
    "the first field solve reads the base state (ctx.state(0))")

# The OLD form -- a single solve_fields(state=U) on the current state -- still lowers + validates
# (Forward Euler: one stage, the base state).
P_fe = adctime.Program("sffs_fe")
U = P_fe.state("plasma")
f = P_fe.solve_fields(U)
R = P_fe.rhs(name="R", state=U, fields=f, flux=True, sources=["electric"])
P_fe.commit("plasma", P_fe.linear_combine("U1", U + P_fe.dt * R))
src_fe = P_fe.emit_cpp_program(model=named_source_model("sffs_fe_named"))
chk(src_fe.count("ctx.solve_fields_from_state(0, ") == 1,
    "the single-stage solve_fields(U) on the current state still lowers (per-stage form)")


# ============================ (B) field-coupled parity: skip without the toolchain ===========
if not hasattr(pops.System(n=8, L=1.0, periodic=True), "install_program"):
    print("-- (B) skipped: _pops lacks the install_program binding (rebuild _pops) --")
    print("%s test_time_solve_fields_from_state (A only)" % ("FAIL" if fails else "PASS"))
    sys.exit(1 if fails else 0)

N = 16
DT = 0.02


def make_sim(model):
    """A System with ONE field-coupled block (production backend) + shared Poisson, charged with a
    non-uniform rho so a stage change shifts phi. Returns (sim, U0)."""
    sim = pops.System(n=N, L=1.0, periodic=True)
    try:
        compiled = model.compile(backend="production")
    except RuntimeError as exc:  # no compiler / no Kokkos visible
        _skip("model compile could not build the .so: %s" % str(exc)[:160])
    sim.add_equation("plasma", compiled,
                     spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                     time=pops.Explicit(method="euler"))
    sim.set_poisson("charge_density", "geometric_mg")
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    mx = 0.4 * rho
    my = -0.2 * rho
    U0 = np.stack([rho, mx, my])
    sim.set_state("plasma", U0)
    return sim, U0


def offline_rhs(ref, U):
    """-div F + electric at U, the fields RE-SOLVED from U. ``ref`` carries the default-source model,
    so eval_rhs already returns -div F + electric."""
    ref.set_state("plasma", U)
    ref.solve_fields()
    return np.array(ref.eval_rhs("plasma"))


def offline_rhs_frozen(ref, U, U_fields):
    """-div F + electric at U with the fields (grad phi) FROZEN at U_fields -- the OLD current-state
    behavior, used only for the cross-check (it should NOT match the compiled program)."""
    ref.set_state("plasma", U_fields)
    ref.solve_fields()           # aux <- grad(phi(U_fields))
    ref.set_state("plasma", U)   # state <- U without re-solving (Poisson frozen)
    return np.array(ref.eval_rhs("plasma"))


print("== (B) field-coupled 2-stage parity (per-stage field solve) ==")
try:
    compiled = pops.compile_problem(model=named_source_model("sffs_prog"), time=heun_program())
except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
    _skip("compile_problem could not build the .so: %s" % str(exc)[:160])

sim, U0 = make_sim(named_source_model("sffs_block"))
sim.install_program(compiled.so_path)
sim.step(DT)
U_prog = np.array(sim.get_state("plasma"))

# Offline replay: BOTH stages re-solve phi from their own state (the ADC-409 per-stage path).
ref = make_sim(default_source_model("sffs_ref_block"))[0]
R0 = offline_rhs(ref, U0)                 # R0 = -div F(U0) + electric(U0; phi(U0))
U1 = U0 + DT * R0
R1 = offline_rhs(ref, U1)                 # R1 = -div F(U1) + electric(U1; phi(U1))  <-- per stage
U_ref_perstage = U0 + 0.5 * DT * R0 + 0.5 * DT * R1

e_perstage = float(np.abs(U_prog - U_ref_perstage).max())
print("  per-stage parity: max|d| = %.2e" % e_perstage)
chk(e_perstage < 1e-12,
    "compiled 2-stage == offline reference re-solving phi PER STAGE (max|d| = %.2e)" % e_perstage)

# CROSS-CHECK: a reference that re-uses phi(U0) for stage 2 (the OLD frozen-aux behavior) gives a
# DIFFERENT result. If the compiled program were still on the current-state path it would match THIS
# instead. The per-stage match above + this mismatch together prove the per-stage solve mattered.
R1_frozen = offline_rhs_frozen(ref, U1, U0)  # R1 with grad phi frozen at U0
U_ref_frozen = U0 + 0.5 * DT * R0 + 0.5 * DT * R1_frozen
e_frozen = float(np.abs(U_prog - U_ref_frozen).max())
d_stage2 = float(np.abs(R1 - R1_frozen).max())
print("  vs frozen-aux reference: max|d| = %.2e (stage-2 RHS differs by %.2e)" % (e_frozen, d_stage2))
chk(d_stage2 > 1e-9,
    "the model is genuinely field-coupled: phi(U1) != phi(U0) shifts the stage-2 RHS (d = %.2e)"
    % d_stage2)
chk(e_frozen > 1e3 * e_perstage and e_frozen > 1e-9,
    "the compiled program is NOT on the frozen-aux path (frozen ref is %.2e, worse than %.2e)"
    % (e_frozen, e_perstage))

# Sanity: the program ran, conserved mass (periodic; the electric source is momentum-only), and moved.
mass0, mass1 = float(U0[0].sum()), float(U_prog[0].sum())
chk(abs(mass1 - mass0) < 1e-9, "mass (sum rho) conserved over the step (|d| = %.2e)"
                               % abs(mass1 - mass0))
chk(float(np.abs(U_prog - U0).max()) > 1e-6, "the 2-stage step actually changed the state")

print("%s test_time_solve_fields_from_state" % ("FAIL (%d)" % fails if fails else "PASS"))
sys.exit(1 if fails else 0)
