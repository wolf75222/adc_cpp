#!/usr/bin/env python3
"""Named-source RHS codegen, end to end: the predictor-corrector Poisson/Lorentz step (ADC-403).

`emit_cpp_program` LOWERS a ``rhs`` with NAMED sources (``sources=[name, ...]`` beyond ``"default"``):
its base is the flux assembly routed on whether ``"default"`` is requested (ADC-425) -- ``ctx.rhs_into``
(= ``-div F`` + the model's default source) when ``"default"`` is in the list, else
``ctx.neg_div_flux_default_into`` (= ``-div F`` only, NO default source) -- and each named source is the
SAME per-cell ``m.source_term`` kernel as the standalone ``source`` op, accumulated onto R via
``ctx.axpy``. ``sources=["electric"]`` (no ``"default"``) therefore lowers to the flux-only base + the
electric source, so the default source is never double-counted (an unknown source name is rejected).

(A) Codegen (pure Python, always runs): a ``rhs(sources=["electric"])`` EMITS (no longer raises) and
    the generated body contains ``ctx.neg_div_flux_default_into`` (flux only, "default" not requested) +
    the electric source kernel (-rho*grad_x/-rho*grad_y) + an ``axpy`` onto R; an UNKNOWN source raises
    ValueError; a named-source rhs WITHOUT a model raises NotImplementedError (no coefficients to read).

(B) Focused parity (skips unless the full toolchain is present): a single Forward-Euler step driven by
    a compiled ``rhs(state=U, sources=["electric"])`` equals the offline one-step reference
    ``U + dt*(-div F + electric)`` computed with the runtime primitives -- where the electric source is
    obtained from a SECOND model that folds the SAME physics as its DEFAULT source (m.source), so
    ``eval_rhs`` returns ``-div F + electric`` directly. This proves the named-source path == the
    default-source path for one step.

(C) Full predictor-corrector parity (same skip gate): the spec example-5 program (two electric-source
    RHS evaluations, two implicit Lorentz local solves I -/+ a*dt*L, one Lorentz apply) compiled +
    installed + stepped, vs an OFFLINE replay of the EXACT same stages built from the runtime
    primitives (set_state + solve_fields + eval_rhs for -div F + electric) plus the analytic Lorentz
    solve / apply. The replay mirrors the compiled program faithfully: BOTH R_n and R_star re-solve the
    fields from their OWN stage state (U_n resp. U_star) -- each solve_fields(state) op lowers to
    ctx.solve_fields_from_state(0, stage state) (ADC-409), so the predictor stage reads grad(U_star).
    With that per-stage solve the match is to round-off. Also asserts the program runs, conserves mass
    (sum rho), changed the state.

Skips cleanly (exit 0) without the install_program binding / numpy / a compiler / a visible Kokkos --
never fakes the engine.
"""
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
import sys


def _skip(msg):
    print("skip test_predictor_corrector (%s)" % msg)
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


def raises(exc_types, fn):
    try:
        fn()
    except exc_types:
        return True
    except Exception:  # noqa: BLE001  -- wrong exception type is a failure, not a pass
        return False
    return False


# --- the predictor-corrector model: rho, mx, my; named electric source + Lorentz linear source -----
def _base_block(m):
    """Shared isothermal 2D fluid block (flux + primitives + eigenvalues + Poisson + B_z aux)."""
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    cs2 = m.param("cs2", 0.5)  # const sound speed^2
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    p = m.primitive("p", cs2 * rho)
    m.primitive_vars(rho=rho, u=u, v=v, p=p)
    m.conservative_from([rho, rho * u, rho * v])
    m.flux(x=[mx, mx * u + p, my * u], y=[my, mx * v, my * v + p])
    cs = sqrt(cs2)
    m.eigenvalues(x=[u - cs, u, u + cs], y=[v - cs, v, v + cs])
    m.elliptic_rhs(rho)  # Poisson rhs f = rho (so solve_fields populates a non-trivial grad)
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    bz = m.aux("B_z")
    return rho, mx, my, gx, gy, bz


def named_source_model(name="pc_named"):
    """Default source EMPTY (NoSource); the electric force is a NAMED source_term (opt-in). The
    Lorentz operator is a named linear_source. This is the model the Program drives."""
    m = Model(name)
    rho, mx, my, gx, gy, bz = _base_block(m)
    m.source_term("electric", [0.0, -rho * gx, -rho * gy])
    m.linear_source("lorentz", [[0.0, 0.0, 0.0],
                                [0.0, 0.0, bz],
                                [0.0, -bz, 0.0]])
    return m


def default_source_model(name="pc_default"):
    """Same physics, but the electric force is the model's DEFAULT source (m.source): eval_rhs then
    returns -div F + electric directly. Used to build the offline reference for the named-source path
    (named-source path == default-source path)."""
    m = Model(name)
    rho, mx, my, gx, gy, bz = _base_block(m)
    m.source([0.0, -rho * gx, -rho * gy])
    m.linear_source("lorentz", [[0.0, 0.0, 0.0],
                                [0.0, 0.0, bz],
                                [0.0, -bz, 0.0]])
    return m


# ---- (A) codegen: pure Python, always runs ----
print("== (A) named-source rhs codegen ==")
m_named = named_source_model()


def _electric_fe_program(name="electric_fe"):
    """One Forward-Euler step from a single rhs(sources=['electric'])."""
    P = adctime.Program(name)
    U = P.state("plasma")
    f = P.solve_fields(U)
    R = P.rhs(name="R", state=U, fields=f, flux=True, sources=["electric"])
    P.commit("plasma", P.linear_combine("U1", U + P.dt * R))
    return P


def _unknown_source_program(name="unknown_src"):
    """One Forward-Euler step naming a source the model never declared."""
    P = adctime.Program(name)
    U = P.state("plasma")
    f = P.solve_fields(U)
    R = P.rhs(name="R", state=U, fields=f, flux=True, sources=["does_not_exist"])
    P.commit("plasma", P.linear_combine("U1", U + P.dt * R))
    return P


src = _electric_fe_program().emit_cpp_program(model=m_named)
# sources=["electric"] excludes "default" (ADC-425) -> the flux base is the flux-only primitive, NOT
# ctx.rhs_into (which would fold the default source); the named electric source is axpy'd on top.
chk("ctx.neg_div_flux_default_into(0, " in src and "ctx.rhs_into(" not in src,
    "rhs(sources=['electric']) lowers the flux via ctx.neg_div_flux_default_into (no default source)")
chk("pops::for_each_cell(" in src, "the named electric source is a per-cell kernel")
chk("((-rho) * grad_x)" in src and "((-rho) * grad_y)" in src,
    "the electric source kernel reads -rho*grad_x / -rho*grad_y")
chk("auxA(i, j, 1)" in src and "auxA(i, j, 2)" in src,
    "grad_x / grad_y read off the canonical aux components 1 / 2")
chk("ctx.axpy(" in src, "the named source is accumulated onto R via axpy (R += S_electric)")

# Unknown source name -> clear ValueError (spec error 1).
chk(raises(ValueError, lambda: _unknown_source_program().emit_cpp_program(model=named_source_model())),
    "an unknown source_term name in rhs raises ValueError")

# ADC-425: a named-source rhs on a model WITH a non-empty DEFAULT source now LOWERS (the old
# double-count rejection is gone). A model carrying BOTH a default source and a named "extra"
# source_term: sources=["extra"] (no "default") routes the flux base to ctx.neg_div_flux_default_into
# (default source NOT folded) + the extra axpy -- no double-count; sources=["default","extra"] folds the
# default via ctx.rhs_into + the extra axpy. Both are sound and emit, not refused.
def _both_source_model(name="pc_both"):
    m = Model(name)
    rho, mx, my, gx, gy, bz = _base_block(m)
    m.source([0.0, -rho * gx, -rho * gy])          # non-empty DEFAULT source
    m.source_term("extra", [0.0, 0.5 * rho, 0.0])  # a distinct NAMED source
    return m


def _extra_fe_program(srcs, name="extra_fe"):
    P = adctime.Program(name)
    U = P.state("plasma")
    f = P.solve_fields(U)
    R = P.rhs(name="R", state=U, fields=f, flux=True, sources=srcs)
    P.commit("plasma", P.linear_combine("U1", U + P.dt * R))
    return P


src_extra_only = _extra_fe_program(["extra"]).emit_cpp_program(model=_both_source_model())
chk("ctx.neg_div_flux_default_into(0, " in src_extra_only and "ctx.rhs_into(" not in src_extra_only,
    "rhs(sources=['extra']) on a default-source model uses the flux-only base (no double-count)")
src_extra_default = _extra_fe_program(["default", "extra"]).emit_cpp_program(model=_both_source_model())
chk("ctx.rhs_into(0, " in src_extra_default,
    "rhs(sources=['default','extra']) folds the default via rhs_into + the extra source axpy'd")

# A named-source rhs without a model cannot read the coefficients -> NotImplementedError.
chk(raises(NotImplementedError, lambda: _electric_fe_program().emit_cpp_program()),
    "rhs with named sources is refused without a model")


# ---- the spec example-5 predictor-corrector Program ----
def predictor_corrector_program(name="predictor_corrector_poisson_lorentz"):
    P = adctime.Program(name)
    dt = P.dt
    U_n = P.state("plasma")
    f_n = P.solve_fields("fields_n", U_n)
    R_n = P.rhs(name="R_n", state=U_n, fields=f_n, flux=True, sources=["electric"])
    U_star_rhs = P.linear_combine("U_star_rhs", U_n + dt * R_n)
    U_star = P.solve_local_linear(name="U_star", operator=P.I - dt * P.linear_source("lorentz"),
                                  rhs=U_star_rhs, fields=f_n)
    f_star = P.solve_fields("fields_star", U_star)
    R_star = P.rhs(name="R_star", state=U_star, fields=f_star, flux=True, sources=["electric"])
    C_star = P.apply(operator=P.linear_source("lorentz"), state=U_star, fields=f_star, name="C_star")
    Q = P.linear_combine("Q", U_n + 0.5 * dt * R_n + 0.5 * dt * R_star + 0.5 * dt * C_star)
    U_np1 = P.solve_local_linear(name="U_np1", operator=P.I - 0.5 * dt * P.linear_source("lorentz"),
                                 rhs=Q, fields=f_star)
    P.commit("plasma", U_np1)
    return P


# The predictor-corrector emits (with a model) -- it uses named sources + local solves.
chk(bool(predictor_corrector_program().emit_cpp_program(model=named_source_model())),
    "the full predictor-corrector Program emits C++ (named sources + Lorentz local solves)")

# ---- (B)/(C) end-to-end: skip unless the install_program binding is present ----
if not hasattr(pops.System(n=8, L=1.0, periodic=True), "install_program"):
    print("-- (B)/(C) skipped: _pops lacks the install_program binding (rebuild _pops) --")
    print("%s test_predictor_corrector (A only)" % ("FAIL" if fails else "PASS"))
    sys.exit(1 if fails else 0)

N = 16
BZ = 3.0
DT = 0.02


def make_sim(model):
    """A System carrying ONE block (the given DSL model, production backend) + shared Poisson + B_z.
    The block is added as a normal equation; a compiled Program (or the offline primitives) drives the
    step. Returns (sim, U0) with U0 the initial conservative state (n_vars, N, N)."""
    sim = pops.System(n=N, L=1.0, periodic=True)
    try:
        compiled = model.compile(backend="production")
    except RuntimeError as exc:  # no compiler / no Kokkos visible
        _skip("model compile could not build the .so: %s" % str(exc)[:160])
    sim.add_equation("plasma", compiled,
                     spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                     time=pops.Explicit(method="euler"))
    sim.set_poisson("charge_density", "geometric_mg")
    sim.set_magnetic_field(BZ * np.ones(N * N))  # constant B_z over the grid
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    mx = 0.4 * rho
    my = -0.2 * rho
    U0 = np.stack([rho, mx, my])
    sim.set_state("plasma", U0)
    return sim, U0


def offline_rhs_with_electric(ref, U):
    """The semi-discrete RHS -div F + electric at state U, the elliptic fields RE-SOLVED from U. ``ref``
    carries the DEFAULT-source model, so eval_rhs(plasma) already returns -div F + electric."""
    ref.set_state("plasma", U)
    ref.solve_fields()
    return np.array(ref.eval_rhs("plasma"))


def analytic_lorentz_solve(U, a):
    """(I - a*L) U' = U with L = [[0,0,0],[0,0,B],[0,-B,0]], a a scalar: rho unchanged, (mx, my)
    rotated. k = a*B, den = 1 + k^2, mx' = (mx + k*my)/den, my' = (-k*mx + my)/den."""
    k = a * BZ
    den = 1.0 + k * k
    rho, mx, my = U[0], U[1], U[2]
    return np.stack([rho, (mx + k * my) / den, (-k * mx + my) / den])


def analytic_lorentz_apply(U):
    """L U with L = [[0,0,0],[0,0,B],[0,-B,0]]: row 0 = 0, row 1 = B*my, row 2 = -B*mx."""
    rho, mx, my = U[0], U[1], U[2]
    return np.stack([np.zeros_like(rho), BZ * my, -BZ * mx])


# ---- (B) focused: one FE step, named-source rhs == default-source eval_rhs ----
print("== (B) focused: rhs(sources=['electric']) == -div F + electric (one FE step) ==")
try:
    compiled_fe = pops.compile_problem(model=named_source_model("electric_fe_prog"),
                                      time=_electric_fe_program())
except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
    _skip("compile_problem could not build the .so: %s" % str(exc)[:160])

sim_fe, U0 = make_sim(named_source_model("electric_fe_block"))
sim_fe.install_program(compiled_fe.so_path)
sim_fe.step(DT)
U_fe = np.array(sim_fe.get_state("plasma"))

ref = make_sim(default_source_model("electric_ref_block"))[0]
k0 = offline_rhs_with_electric(ref, U0)
U_fe_ref = U0 + DT * k0
e_fe = float(np.abs(U_fe - U_fe_ref).max())
print("  focused FE parity: max|d| = %.2e" % e_fe)
chk(e_fe < 1e-10, "compiled rhs(sources=['electric']) FE step == offline -div F + electric "
                  "(max|d| = %.2e)" % e_fe)
chk(float(np.abs(U_fe - U0).max()) > 1e-6, "the electric source actually moved the state")

# ---- (C) full predictor-corrector parity ----
print("== (C) full predictor-corrector parity ==")
try:
    compiled_pc = pops.compile_problem(model=named_source_model("pc_prog"),
                                      time=predictor_corrector_program())
except RuntimeError as exc:
    _skip("compile_problem could not build the .so: %s" % str(exc)[:160])

chk(compiled_pc.program_name == "predictor_corrector_poisson_lorentz",
    "handle carries the predictor-corrector program name")

sim_pc, U0 = make_sim(named_source_model("pc_block"))
sim_pc.install_program(compiled_pc.so_path)
sim_pc.step(DT)
U_pc = np.array(sim_pc.get_state("plasma"))

# Offline replay of the EXACT same stages (a fresh reference System with the default-source model).
# BOTH R_n and R_star re-solve the fields from their OWN stage state: each solve_fields(state) op lowers
# to ctx.solve_fields_from_state(0, stage state) (ADC-409), so R_star reads grad(U_star) not grad(U_n).
# The implicit Lorentz solve and the Lorentz apply are local and have a closed form. This replays the
# compiled step bit-for-bit.
refc = make_sim(default_source_model("pc_ref_block"))[0]
R_n = offline_rhs_with_electric(refc, U0)            # R_n = -div F(U_n) + electric(U_n; grad U_n)
U_star_rhs = U0 + DT * R_n
U_star = analytic_lorentz_solve(U_star_rhs, DT)      # (I - dt*L) U_star = U_star_rhs
R_star = offline_rhs_with_electric(refc, U_star)     # -div F(U_star) + electric(U_star; grad U_star)
C_star = analytic_lorentz_apply(U_star)              # C_star = L U_star
Q = U0 + 0.5 * DT * R_n + 0.5 * DT * R_star + 0.5 * DT * C_star
U_np1_ref = analytic_lorentz_solve(Q, 0.5 * DT)      # (I - 0.5*dt*L) U_np1 = Q

e_pc = float(np.abs(U_pc - U_np1_ref).max())
print("  predictor-corrector parity: max|d| = %.2e" % e_pc)
chk(e_pc < 1e-10, "compiled predictor-corrector == offline staged reference (max|d| = %.2e)" % e_pc)

# Sanity: the program ran, conserved mass (periodic; Lorentz + electric are momentum-only), and moved.
mass0 = float(U0[0].sum())
mass1 = float(U_pc[0].sum())
chk(abs(mass1 - mass0) < 1e-9, "mass (sum rho) conserved over the step (|d| = %.2e)"
                               % abs(mass1 - mass0))
chk(float(np.abs(U_pc - U0).max()) > 1e-6, "the predictor-corrector step actually changed the state")

print("%s test_predictor_corrector" % ("FAIL (%d)" % fails if fails else "PASS"))
sys.exit(1 if fails else 0)
